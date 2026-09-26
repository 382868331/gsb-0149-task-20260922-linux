"""Reusable moving-garbage-collector kernel with weak references.

Model
-----
* Two semi-spaces, each ``capacity`` slots wide.  Exactly one is active;
  ``allocate`` carves objects out of its bump pointer, ``collect`` evacuates
  the live graph into the other semi-space and flips the roles.
* Every object occupies ``2 + n_strong + n_weak`` consecutive slots::

      [ header_tag, payload, strong_slot_0, ..., strong_slot_(s-1),
        weak_slot_0, ..., weak_slot_(w-1) ]

  - ``header_tag`` is a reserved slot (always ``None`` in this toy model);
    the forwarding table is kept as a side dictionary during collection,
    which keeps the slot format uniform.
  - Remaining slot values are integers: the payload or an object address.
  - Edge slots additionally accept ``None`` to denote a null edge.
* Handles pair an address with an ``epoch``.  The epoch is the number of
  completed collections: a handle is valid iff its epoch equals the heap's
  current epoch *and* an object header currently sits at its address.

Collection order
-----------------
Copy order is fully deterministic: roots in the order they were registered,
each root's object copied immediately, then strong edges are processed in
field order by a Cheney scan.  Weak edges never keep a target alive.  After
the strong closure is evacuated, weak edges are rewritten: surviving
targets point at the new address, dead targets are cleared to ``None``.

The module only implements the finite contract described in README.md; it
is not a drop-in replacement for any production runtime GC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isinf, isnan

from .errors import FragmentationError, GCError, HeapCapacityError, StaleHandleError

__all__ = [
    "Heap",
    "Handle",
    "PinHandle",
    "CollectionResult",
    "GCError",
    "StaleHandleError",
    "HeapCapacityError",
    "FragmentationError",
    "MAX_CAPACITY",
]

MAX_CAPACITY = 4096
_HEADER = 1  # slots used by the header tag
_PAYLOAD = 1  # slots used by the integer payload


def _check_int(value, what: str) -> None:
    """Validate an integer-typed parameter.

    Contract: every numeric parameter of this API is an integer.

    * ``bool`` (an ``int`` subclass in Python) -> :class:`TypeError`
    * float ``NaN`` / ``+Inf`` / ``-Inf`` -> :class:`ValueError`
      (non-finite numbers are rejected explicitly, not by accident)
    * any other float or non-numeric type -> :class:`TypeError`
    """
    if isinstance(value, bool):
        raise TypeError(f"{what} must be a plain int, got bool")
    if isinstance(value, float):
        if isnan(value):
            raise ValueError(f"{what} must be a plain int, got NaN")
        if isinf(value):
            raise ValueError(f"{what} must be a plain int, got infinity")
        raise TypeError(f"{what} must be a plain int, got non-integral float")
    if not isinstance(value, int):
        raise TypeError(f"{what} must be a plain int, got {type(value).__name__}")


@dataclass(frozen=True)
class Handle:
    """Opaque external reference to one heap object.

    ``address`` is the 0-based slot index of the object header in the
    semi-space that was active when the handle was minted.  ``epoch`` is
    the heap epoch at mint time.  Handles are intentionally immutable;
    obtain fresh handles (e.g. via :meth:`Heap.handle`) after every
    collection rather than mutating stale ones.
    """

    address: int
    epoch: int


@dataclass(frozen=True)
class PinHandle:
    """Stable external reference to a pinned object (pinning mode only).

    A pin handle simulates external code holding a stable *logical* slot
    address across collections: while at least one pin on an object is
    outstanding, the object is a strong root and is never relocated.
    This is not a promise about real Python memory addresses.

    ``generation`` is an independent per-pin serial number (unrelated to
    the heap epoch).  Repeated pins of the same object are counted
    separately; releasing one handle does not lift the others.  Once a
    pin is released its generation is forgotten, so a stale pin handle
    can never operate on a different object that later reuses the slot.
    """

    address: int
    generation: int


@dataclass(frozen=True)
class CollectionResult:
    """Outcome of one :meth:`Heap.collect` call.

    Attributes:
        relocated: mapping ``old_address -> new_address`` for every object
            that survived and was copied (including objects whose copy
            happened to land at the same slot index — epoch still changed).
        live_slots: number of slots occupied by surviving objects in the
            new active semi-space (equals the new bump pointer).
        cleared_weak_edges: ``(holder_address, field_index)`` pairs, using
            *post-collection* holder addresses, for weak fields whose
            target was unreachable and were set to ``None``.
        roots_preserved: number of registered roots that were non-null and
            updated to point at their object's copy.
        pinned_addresses: addresses kept fixed during this collection
            (pinning mode only; empty tuple otherwise).  In pinning mode
            ``relocated`` only contains objects that actually moved.
    """

    relocated: dict[int, int]
    live_slots: int
    cleared_weak_edges: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    roots_preserved: int = 0
    pinned_addresses: tuple[int, ...] = ()


class Heap:
    """Semi-space heap with strong/weak edges and explicit roots.

    Args:
        capacity: slots per semi-space; ``1 <= capacity <= MAX_CAPACITY``.

    Raises:
        TypeError: ``capacity`` is not a plain integer (booleans rejected).
        ValueError: ``capacity`` is out of range.
    """

    def __init__(self, capacity: int, *, pinning: bool = False) -> None:
        _check_int(capacity, "capacity")
        if not (1 <= capacity <= MAX_CAPACITY):
            raise ValueError(f"capacity must be in [1, {MAX_CAPACITY}], got {capacity}")
        if not isinstance(pinning, bool):
            raise TypeError(f"pinning must be a bool, got {type(pinning).__name__}")
        self._capacity: int = capacity
        self._pinning: bool = pinning
        # Two semi-spaces: lists of slot values.  Free slots hold None;
        # object header slots are reserved (None); data slots hold an int
        # payload/address, or None for a null edge.
        self._spaces: list[list] = [
            [None for _ in range(capacity)],
            [None for _ in range(capacity)],
        ]
        self._active: int = 0
        self._alloc_ptr: int = 0  # first free slot in the active space
        self._epoch: int = 0
        # Object metadata, keyed by address in whichever space is active.
        # Rebuilt during collection.  Values: (n_strong, n_weak).
        self._layout: dict[int, tuple[int, int]] = {}
        # Registered explicit roots: name -> address or None.
        self._roots: dict[str, int | None] = {}
        # Root insertion order (Python dict preserves it, but be explicit).
        self._root_order: list[str] = []
        # Pinning mode: address -> list of live pin generations (one entry
        # per outstanding pin; repeated pins are counted separately), plus
        # the monotonic serial that mints independent generations.
        self._pins: dict[int, list[int]] = {}
        self._pin_serial: int = 0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def capacity(self) -> int:
        """Slots available in each semi-space."""
        return self._capacity

    @property
    def epoch(self) -> int:
        """Current heap epoch; incremented after every successful collection."""
        return self._epoch

    @property
    def used_slots(self) -> int:
        """Slots currently occupied in the active semi-space.

        In the default Cheney mode this equals the bump pointer; in
        pinning mode (where holes may exist between pinned objects) it is
        the sum of live object sizes.
        """
        if self._pinning:
            return sum(2 + s + w for s, w in self._layout.values())
        return self._alloc_ptr

    @property
    def free_slots(self) -> int:
        """Free slots in the active semi-space."""
        return self._capacity - self.used_slots

    def objects(self) -> tuple[int, ...]:
        """Addresses (header slots) of all objects currently in the heap."""
        return tuple(sorted(self._layout))

    # ------------------------------------------------------------------ #
    # Allocation
    # ------------------------------------------------------------------ #
    def allocate(
        self,
        payload: int = 0,
        *,
        n_strong: int = 0,
        n_weak: int = 0,
    ) -> Handle:
        """Allocate one object and return its fresh :class:`Handle`.

        New edge slots start ``None`` (null).  Whenever the active
        semi-space cannot satisfy the request, exactly **one** collection
        is performed first (using the registered roots); if the request
        still does not fit afterwards -- including an object larger than a
        whole semi-space, per the "collect once, then report insufficient
        capacity" contract -- :class:`HeapCapacityError` is raised and *no*
        partial object is left behind.

        Args:
            payload: integer payload stored with the object.
            n_strong: number of strong edge slots (>= 0).
            n_weak: number of weak edge slots (>= 0).
        """
        _check_int(payload, "payload")
        _check_int(n_strong, "n_strong")
        _check_int(n_weak, "n_weak")
        if n_strong < 0:
            raise ValueError(f"n_strong must be >= 0, got {n_strong}")
        if n_weak < 0:
            raise ValueError(f"n_weak must be >= 0, got {n_weak}")
        size = _HEADER + _PAYLOAD + n_strong + n_weak
        if self._pinning:
            return self._allocate_pinning(payload, n_strong, n_weak, size)
        if self._alloc_ptr + size > self._capacity:
            # Mandatory single emergency collection (state only changes if
            # it succeeds), then re-check against the compacted semi-space.
            self.collect()
            if self._alloc_ptr + size > self._capacity:
                if size > self._capacity:
                    reason = (
                        f"object size {size} exceeds semi-space capacity "
                        f"{self._capacity}"
                    )
                else:
                    reason = (
                        f"need {size} slot(s), "
                        f"{self._capacity - self._alloc_ptr} free after collection"
                    )
                raise HeapCapacityError(f"out of capacity: {reason}")
        return self._write_new_object(payload, n_strong, n_weak, size)

    # ------------------------------------------------------------------ #
    # Pinning mode: handles, gap allocation
    # ------------------------------------------------------------------ #
    @property
    def pinning(self) -> bool:
        """Whether this heap runs in pinning (mark-compact) mode."""
        return self._pinning

    def pin(self, target: Handle) -> PinHandle:
        """Pin the object behind ``target`` and return a :class:`PinHandle`.

        Each call is counted separately: pinning the same object twice
        requires two releases before it can move again.  Only available
        on heaps created with pinning=True.
        """
        if not self._pinning:
            raise GCError("pin() requires a heap created with pinning=True")
        addr = self._resolve(target)
        generation = self._pin_serial
        self._pin_serial += 1
        self._pins.setdefault(addr, []).append(generation)
        return PinHandle(addr, generation)

    def release(self, pin_handle: PinHandle) -> None:
        """Release one outstanding pin.

        Releasing one handle never lifts the remaining pins on the same
        object.  A handle whose generation was already released (or never
        issued) is rejected, so a stale pin handle cannot operate on a
        new object that later reuses the address.
        """
        if not isinstance(pin_handle, PinHandle):
            raise TypeError(f"expected PinHandle, got {type(pin_handle).__name__}")
        gens = self._pins.get(pin_handle.address)
        if not gens or pin_handle.generation not in gens:
            raise StaleHandleError(
                f"stale pin handle: generation {pin_handle.generation} not live "
                f"at address {pin_handle.address}"
            )
        gens.remove(pin_handle.generation)
        if not gens:
            del self._pins[pin_handle.address]

    def pin_count(self, address: int) -> int:
        """Number of outstanding pins on the object at ``address``."""
        _check_int(address, "address")
        return len(self._pins.get(address, ()))

    def pinned_addresses(self) -> tuple[int, ...]:
        """Sorted addresses of all currently pinned objects."""
        return tuple(sorted(self._pins))

    def _free_gaps(self) -> list[tuple[int, int]]:
        """Sorted (start, length) free runs in the active semi-space."""
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for addr in sorted(self._layout):
            n_strong, n_weak = self._layout[addr]
            if addr > cursor:
                gaps.append((cursor, addr - cursor))
            cursor = addr + _HEADER + _PAYLOAD + n_strong + n_weak
        if cursor < self._capacity:
            gaps.append((cursor, self._capacity - cursor))
        return gaps

    def _allocate_pinning(self, payload: int, n_strong: int, n_weak: int, size: int) -> Handle:
        addr = self._first_fit(size)
        if addr is None:
            # Same contract as Cheney mode: exactly one emergency
            # collection, then re-check against the compacted heap.
            self.collect()
            addr = self._first_fit(size)
            if addr is None:
                if size > self._capacity:
                    raise HeapCapacityError(
                        f"out of capacity: object size {size} exceeds semi-space "
                        f"capacity {self._capacity}"
                    )
                free = self._capacity - self.used_slots
                if free >= size:
                    raise FragmentationError(
                        f"fragmented heap: need {size} contiguous slot(s), "
                        f"{free} free in total but no contiguous run fits"
                    )
                raise HeapCapacityError(
                    f"out of capacity: need {size} slot(s), "
                    f"{free} free after collection"
                )
        return self._write_new_object_at(addr, payload, n_strong, n_weak, size)

    def _first_fit(self, size: int) -> int | None:
        for start, length in self._free_gaps():
            if length >= size:
                return start
        return None

    def _write_new_object_at(
        self, addr: int, payload: int, n_strong: int, n_weak: int, size: int
    ) -> Handle:
        space = self._spaces[self._active]
        space[addr] = None  # header: not forwarded
        space[addr + 1] = payload
        for i in range(2, size):
            space[addr + i] = None
        self._layout[addr] = (n_strong, n_weak)
        return Handle(addr, self._epoch)

    def _write_new_object(self, payload: int, n_strong: int, n_weak: int, size: int) -> Handle:
        space = self._spaces[self._active]
        addr = self._alloc_ptr
        space[addr] = None  # header: not forwarded
        space[addr + 1] = payload
        # Edge slots are already None from initialization/previous reuse,
        # but set them explicitly so reused to-space slots are well-defined.
        for i in range(2, size):
            space[addr + i] = None
        self._alloc_ptr = addr + size
        self._layout[addr] = (n_strong, n_weak)
        return Handle(addr, self._epoch)

    # ------------------------------------------------------------------ #
    # Roots
    # ------------------------------------------------------------------ #
    def add_root(self, name: str, target: Handle) -> None:
        """Register (or replace) an explicitly named root.

        ``target`` must be a live current-epoch handle.  Replacing an
        existing root only happens after the handle is validated, so a bad
        handle never clobbers a previously registered root.
        """
        if not isinstance(name, str) or not name:
            raise TypeError("root name must be a non-empty str")
        addr = self._resolve(target)
        if name not in self._roots:
            self._root_order.append(name)
        self._roots[name] = addr

    def clear_root(self, name: str) -> None:
        """Set a registered root to null without unregistering its name."""
        if name not in self._roots:
            raise KeyError(name)
        self._roots[name] = None

    def remove_root(self, name: str) -> None:
        """Unregister a root entirely; raises ``KeyError`` if unknown."""
        if name not in self._roots:
            raise KeyError(name)
        del self._roots[name]
        self._root_order.remove(name)

    def root_handle(self, name: str) -> Handle | None:
        """Return a current handle for a root, or ``None`` if it is null."""
        if name not in self._roots:
            raise KeyError(name)
        addr = self._roots[name]
        return None if addr is None else Handle(addr, self._epoch)

    def roots(self) -> dict[str, Handle | None]:
        """Snapshot ``name -> Handle/None`` of all registered roots."""
        return {
            name: (None if addr is None else Handle(addr, self._epoch))
            for name, addr in self._roots.items()
        }

    # ------------------------------------------------------------------ #
    # Field access
    # ------------------------------------------------------------------ #
    def handle(self, address: int) -> Handle:
        """Mint a handle for the object currently living at ``address``."""
        _check_int(address, "address")
        if address not in self._layout:
            raise StaleHandleError(f"no live object at address {address}")
        return Handle(address, self._epoch)

    def _resolve(self, target: Handle) -> int:
        """Validate a handle (or pin handle) and return its address."""
        if isinstance(target, PinHandle):
            gens = self._pins.get(target.address)
            if not gens or target.generation not in gens:
                raise StaleHandleError(
                    f"stale pin handle: generation {target.generation} not live "
                    f"at address {target.address}"
                )
            if target.address not in self._layout:
                raise StaleHandleError(
                    f"no live object at address {target.address}"
                )
            return target.address
        if not isinstance(target, Handle):
            raise TypeError(f"expected Handle, got {type(target).__name__}")
        if target.epoch != self._epoch:
            raise StaleHandleError(
                f"stale handle: epoch {target.epoch} != current epoch {self._epoch} "
                f"(address {target.address})"
            )
        if target.address not in self._layout:
            raise StaleHandleError(f"no live object at address {target.address}")
        return target.address

    def payload(self, target: Handle) -> int:
        """Return the integer payload of ``target``."""
        addr = self._resolve(target)
        return self._spaces[self._active][addr + 1]

    def set_payload(self, target: Handle, value: int) -> None:
        """Replace the integer payload of ``target``."""
        _check_int(value, "value")
        addr = self._resolve(target)
        self._spaces[self._active][addr + 1] = value

    def get_strong(self, target: Handle, index: int) -> Handle | None:
        """Read strong edge ``index``; ``None`` means a null edge."""
        addr = self._resolve(target)
        n_strong, _ = self._layout[addr]
        self._check_index(index, n_strong, "strong")
        slot = self._spaces[self._active][addr + 2 + index]
        return None if slot is None else Handle(slot, self._epoch)

    def set_strong(self, target: Handle, index: int, ref: Handle | None) -> None:
        """Write strong edge ``index``; ``None`` clears it."""
        addr = self._resolve(target)
        n_strong, _ = self._layout[addr]
        self._check_index(index, n_strong, "strong")
        new_val = None if ref is None else self._resolve(ref)
        self._spaces[self._active][addr + 2 + index] = new_val

    def get_weak(self, target: Handle, index: int) -> Handle | None:
        """Read weak edge ``index``.

        ``None`` is returned both for an explicitly null edge and for an
        edge whose target was reclaimed by the most recent collection;
        callers cannot distinguish the two (standard weak-reference
        semantics).
        """
        addr = self._resolve(target)
        n_strong, n_weak = self._layout[addr]
        self._check_index(index, n_weak, "weak")
        slot = self._spaces[self._active][addr + 2 + n_strong + index]
        return None if slot is None else Handle(slot, self._epoch)

    def set_weak(self, target: Handle, index: int, ref: Handle | None) -> None:
        """Write weak edge ``index``; ``None`` clears it.

        Weak edges do not protect their target from collection.
        """
        addr = self._resolve(target)
        n_strong, n_weak = self._layout[addr]
        self._check_index(index, n_weak, "weak")
        new_val = None if ref is None else self._resolve(ref)
        self._spaces[self._active][addr + 2 + n_strong + index] = new_val

    @staticmethod
    def _check_index(index: int, length: int, kind: str) -> None:
        _check_int(index, "index")
        if not 0 <= index < length:
            raise IndexError(
                f"{kind} edge index {index} out of range for object with {length} {kind} edge(s)"
            )

    def object_shape(self, target: Handle) -> tuple[int, int]:
        """Return ``(n_strong, n_weak)`` for ``target``."""
        return self._layout[self._resolve(target)]

    # ------------------------------------------------------------------ #
    # Collection
    # ------------------------------------------------------------------ #
    def collect(self) -> CollectionResult:
        """Run one stop-the-world collection.

        Default heaps use Cheney semi-space copying; heaps created with
        pinning=True use in-place sliding compaction around pinned
        objects.  See the mode-specific docstrings for the contracts.
        """
        if self._pinning:
            return self._collect_compact()
        return self._collect_cheney()

    def _collect_cheney(self) -> CollectionResult:
        """Run one stop-the-world semi-space collection.

        Reachability is computed solely from registered roots through
        strong edges.  Surviving objects are copied into the inactive
        semi-space in root order / strong-field order (Cheney scanning).
        Roots, strong edges and the weak edges that name survivors are
        forwarded; weak edges naming dead objects are cleared.

        On success the heap flips semi-spaces and its epoch advances by
        one.  The operation is all-or-nothing: if the live set does not
        fit in a semi-space (which cannot occur when the source itself
        fits, but is defended anyway), :class:`HeapCapacityError` is
        raised with the heap left in its pre-collection state.
        """
        src_idx = self._active
        dst_idx = 1 - src_idx
        src = self._spaces[src_idx]
        dst = self._spaces[dst_idx]
        old_layout = self._layout
        capacity = self._capacity

        # Forwarding table: old address -> new address.  Doubles as the
        # mark set (an object is copied iff it appears here).
        fwd: dict[int, int] = {}
        new_layout: dict[int, tuple[int, int]] = {}
        scan = 0  # Cheney scan cursor over *destination* addresses
        free = 0  # bump pointer in destination
        relocated: dict[int, int] = {}
        new_roots: dict[str, int | None] = {}
        roots_preserved = 0

        def copy_obj(old: int) -> int:
            nonlocal free
            if old in fwd:  # NB: new address 0 is legitimate; never use .get()
                return fwd[old]
            n_strong, n_weak = old_layout[old]
            size = _HEADER + _PAYLOAD + n_strong + n_weak
            if free + size > capacity:
                raise HeapCapacityError(
                    f"live set does not fit: needed {size} slot(s) at offset {free}, "
                    f"capacity {capacity}"
                )
            new = free
            free += size
            # Blind bulk copy of the slot window, then fix up fields below.
            dst[new : new + size] = src[old : old + size]
            fwd[old] = new
            new_layout[new] = (n_strong, n_weak)
            relocated[old] = new
            return new

        # Phase 1: evacuate roots in registration order.
        try:
            for name in self._root_order:
                old_root = self._roots[name]
                if old_root is None:
                    new_roots[name] = None
                else:
                    new_roots[name] = copy_obj(old_root)
                    roots_preserved += 1

            # Phase 2: Cheney scan of the strong closure, field order preserved.
            while scan < free:
                new = scan
                n_strong, n_weak = new_layout[new]
                strong_base = new + 2
                for j in range(n_strong):
                    old_target = dst[strong_base + j]
                    if old_target is not None:
                        dst[strong_base + j] = copy_obj(old_target)
                scan = new + _HEADER + _PAYLOAD + n_strong + n_weak

            live_slots = free

            # Phase 3: fix weak edges of survivors.  Weak targets never became
            # part of the closure, so they are either in fwd (alive elsewhere)
            # or unreachable -> clear.
            cleared: list[tuple[int, int]] = []
            for new in sorted(new_layout):
                n_strong, n_weak = new_layout[new]
                weak_base = new + 2 + n_strong
                for j in range(n_weak):
                    old_target = dst[weak_base + j]
                    if old_target is None:
                        continue
                    if old_target in fwd:
                        dst[weak_base + j] = fwd[old_target]
                    else:
                        dst[weak_base + j] = None
                        cleared.append((new, j))
        except HeapCapacityError:
            # All-or-nothing: erase the partial evacuation so no half-built
            # state is observable (the heap still serves from src).  The
            # inactive space must be blank before the next attempt anyway.
            for i in range(capacity):
                dst[i] = None
            raise

        # All structural work succeeded: commit the flip atomically.
        # Rebuild the source space as a blank semi-space (its contents are
        # dead; clearing avoids leaking stale ints into a future to-space).
        for i in range(capacity):
            src[i] = None
        self._active = dst_idx
        self._alloc_ptr = live_slots
        self._layout = new_layout
        self._roots = new_roots
        self._epoch += 1

        return CollectionResult(
            relocated=relocated,
            live_slots=live_slots,
            cleared_weak_edges=tuple(cleared),
            roots_preserved=roots_preserved,
        )

    def _collect_compact(self) -> CollectionResult:
        """Pinning-mode collection: sliding compaction around pinned objects.

        Reachability starts from explicit roots *and* pinned addresses
        (pins are strong roots) and follows strong edges only.  Pinned
        survivors keep their slots; movable survivors slide, in address
        order, into the lowest free slots not covered by a pinned object.
        Roots, strong edges and weak edges naming survivors are
        forwarded; weak edges naming dead objects are cleared.  The heap
        epoch still advances by one, so ordinary Handles minted before
        the collection are rejected afterwards, exactly as in Cheney
        mode; PinHandles stay valid because their object never moved.
        """
        space = self._spaces[self._active]
        old_layout = self._layout
        capacity = self._capacity

        # Phase 1: reachability from roots and pins, strong edges only.
        reachable: set[int] = set()
        stack = [a for a in self._roots.values() if a is not None]
        stack.extend(addr for addr in self._pins if addr in old_layout)
        while stack:
            addr = stack.pop()
            if addr in reachable:
                continue
            reachable.add(addr)
            n_strong, _ = old_layout[addr]
            for j in range(n_strong):
                t = space[addr + 2 + j]
                if t is not None and t not in reachable:
                    stack.append(t)

        # Phase 2: compute new addresses.  Pinned objects stay; movable
        # objects are placed by an order-preserving first fit that skips
        # pinned intervals.
        pinned_intervals = sorted(
            (addr, addr + _HEADER + _PAYLOAD + s + w)
            for addr, (s, w) in old_layout.items()
            if addr in self._pins
        )
        fwd: dict[int, int] = {}
        free = 0
        pin_idx = 0
        for old in sorted(reachable):
            n_strong, n_weak = old_layout[old]
            size = _HEADER + _PAYLOAD + n_strong + n_weak
            if old in self._pins:
                fwd[old] = old
                continue
            while True:
                while pin_idx < len(pinned_intervals) and pinned_intervals[pin_idx][1] <= free:
                    pin_idx += 1
                if pin_idx < len(pinned_intervals) and pinned_intervals[pin_idx][0] < free + size:
                    free = pinned_intervals[pin_idx][1]
                    pin_idx += 1
                    continue
                break
            if free + size > capacity:
                # Cannot happen when the source layout was legal (the
                # live set is a subset sliding left); defended anyway.
                raise HeapCapacityError(
                    f"live set does not fit around pinned objects: needed {size} "
                    f"slot(s) at offset {free}, capacity {capacity}"
                )
            fwd[old] = free
            free += size

        # Phase 3: rewrite the space from a snapshot so overlapping
        # moves cannot clobber objects that have not been placed yet.
        snapshot = list(space)
        for i in range(capacity):
            space[i] = None
        new_layout: dict[int, tuple[int, int]] = {}
        relocated: dict[int, int] = {}
        cleared: list[tuple[int, int]] = []
        for old in sorted(reachable):
            new = fwd[old]
            n_strong, n_weak = old_layout[old]
            size = _HEADER + _PAYLOAD + n_strong + n_weak
            space[new : new + size] = snapshot[old : old + size]
            for j in range(n_strong):
                t = snapshot[old + 2 + j]
                if t is not None:
                    space[new + 2 + j] = fwd[t]
            weak_base_old = old + 2 + n_strong
            weak_base_new = new + 2 + n_strong
            for j in range(n_weak):
                t = snapshot[weak_base_old + j]
                if t is None:
                    continue
                if t in fwd:
                    space[weak_base_new + j] = fwd[t]
                else:
                    space[weak_base_new + j] = None
                    cleared.append((new, j))
            new_layout[new] = (n_strong, n_weak)
            if new != old:
                relocated[old] = new

        new_roots: dict[str, int | None] = {}
        roots_preserved = 0
        for name in self._root_order:
            old_root = self._roots[name]
            if old_root is None:
                new_roots[name] = None
            else:
                new_roots[name] = fwd[old_root]
                roots_preserved += 1

        self._layout = new_layout
        self._roots = new_roots
        self._alloc_ptr = 0  # unused in pinning mode; see used_slots
        self._epoch += 1

        return CollectionResult(
            relocated=relocated,
            live_slots=sum(2 + s + w for s, w in new_layout.values()),
            cleared_weak_edges=tuple(cleared),
            roots_preserved=roots_preserved,
            pinned_addresses=tuple(sorted(self._pins)),
        )

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def reachable_addresses(self) -> frozenset[int]:
        """Addresses reachable from roots via strong edges *right now*.

        This performs an independent read-only traversal (no copying) and
        is mainly useful for tests and diagnostics.
        """
        seen: set[int] = set()
        stack = [a for a in self._roots.values() if a is not None]
        if self._pinning:
            stack.extend(addr for addr in self._pins if addr in self._layout)
        space = self._spaces[self._active]
        while stack:
            addr = stack.pop()
            if addr in seen:
                continue
            seen.add(addr)
            n_strong, _ = self._layout[addr]
            for j in range(n_strong):
                t = space[addr + 2 + j]
                if t is not None and t not in seen:
                    stack.append(t)
        return frozenset(seen)
