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
field order by a Cheney scan.  Weak edges never keep a target alive.

Weak-key associations are a third edge discipline, kept in side metadata
rather than in object slots (so a table does *not* strongly reference its
keys or values).  An association ``table[key] = value`` makes ``value``
strong only while **both** the table object and ``key`` are independently
alive.  After the ordinary strong closure is evacuated, the collector runs
a fixpoint over the surviving tables: each pass activates (copies) the
values of entries whose key has already survived, then drains the Cheney
scan so values can strongly reach yet more keys; passes repeat until no new
entry activates.  Entries of dead tables and entries whose key never
becomes alive are deleted; surviving entries are forwarded.  A key kept
reachable only by a value that its own entry would activate can never
bootstrap itself.  Finally the plain weak edges are rewritten: surviving
targets point at the new address, dead targets are cleared to ``None``.

The module only implements the finite contract described in README.md; it
is not a drop-in replacement for any production runtime GC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isinf, isnan

from .errors import HeapCapacityError, StaleHandleError

__all__ = [
    "Heap",
    "Handle",
    "CollectionResult",
    "GCError",
    "StaleHandleError",
    "HeapCapacityError",
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
        surviving_associations: ``(table_address, key_address,
            value_address)`` triples, using *post-collection* addresses, for
            weak-key entries that stayed live (table and key independently
            alive, value evacuated); ordered by (table, key) address.
        cleared_associations: ``(table_address, key_address)`` pairs, using
            *pre-collection* addresses (a cleared key has no post-collection
            address, so each pair stays in one address namespace; translate
            the table via ``relocated``), for entries deleted because their
            key was not alive.  Entries of a dead table vanish with the
            table and are not reported here; ordered like
            ``surviving_associations``.
    """

    relocated: dict[int, int]
    live_slots: int
    cleared_weak_edges: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    roots_preserved: int = 0
    surviving_associations: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    cleared_associations: tuple[tuple[int, int], ...] = field(default_factory=tuple)


class Heap:
    """Semi-space heap with strong/weak edges and explicit roots.

    Args:
        capacity: slots per semi-space; ``1 <= capacity <= MAX_CAPACITY``.

    Raises:
        TypeError: ``capacity`` is not a plain integer (booleans rejected).
        ValueError: ``capacity`` is out of range.
    """

    def __init__(self, capacity: int) -> None:
        _check_int(capacity, "capacity")
        if not (1 <= capacity <= MAX_CAPACITY):
            raise ValueError(f"capacity must be in [1, {MAX_CAPACITY}], got {capacity}")
        self._capacity: int = capacity
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
        # Weak-key associations in side metadata (NOT object slots, so they
        # are neither strong nor weak edges): table address ->
        # {key address: value address}.  An entry keeps its value alive
        # during a collection iff its table *and* key are independently
        # alive; see the fixpoint activation in collect().
        self._associations: dict[int, dict[int, int]] = {}

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
        """Slots currently occupied in the active semi-space."""
        return self._alloc_ptr

    @property
    def free_slots(self) -> int:
        """Free slots in the active semi-space."""
        return self._capacity - self._alloc_ptr

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
        """Validate a handle and return its current-space address."""
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
    # Weak-key associations (side metadata, never a slot edge)
    # ------------------------------------------------------------------ #
    def association_put(
        self, table: Handle, key: Handle, value: Handle | None
    ) -> None:
        """Insert, replace or delete one weak-key association.

        The association lives in side metadata on the *table* object; it is
        not an object slot, so the table neither strongly nor weakly
        references ``key``/``value``.  During a collection the entry keeps
        ``value`` alive only while **both** ``table`` and ``key`` are
        independently alive (see :meth:`collect`).

        Passing ``value=None`` deletes the entry for ``key``.  All three
        handles are validated before any mutation, so a stale handle never
        creates a half entry or removes an existing one.  ``key`` must be
        non-null (unlike ordinary edges, associations have no null key);
        ``None`` is accepted for ``value`` only as the explicit deletion
        marker above.
        """
        table_addr = self._resolve(table)
        key_addr = self._resolve(key)
        value_addr = None if value is None else self._resolve(value)
        table_entries = self._associations.setdefault(table_addr, {})
        if value_addr is None:
            table_entries.pop(key_addr, None)
        else:
            table_entries[key_addr] = value_addr

    def association_get(self, table: Handle, key: Handle) -> Handle | None:
        """Return the current value handle for ``table[key]``, else ``None``.

        ``None`` means no live entry for that key: either it was never
        inserted, it was explicitly deleted, or a previous collection
        removed it because the table or its key died.  This is an explicit
        metadata lookup, not a weak edge; a stale *handle* passed in as an
        argument is still rejected.
        """
        table_addr = self._resolve(table)
        key_addr = self._resolve(key)
        value_addr = self._associations.get(table_addr, {}).get(key_addr)
        if value_addr is None:
            return None
        return Handle(value_addr, self._epoch)

    def association_remove(self, table: Handle, key: Handle) -> None:
        """Delete ``table[key]`` if present; absent keys are a no-op."""
        table_addr = self._resolve(table)
        key_addr = self._resolve(key)
        table_entries = self._associations.get(table_addr)
        if table_entries is not None:
            table_entries.pop(key_addr, None)

    def association_entries(self, table: Handle) -> tuple[tuple[Handle, Handle], ...]:
        """Snapshot ``((key_handle, value_handle), ...)`` for a live table.

        Ordered by current key address.  Only currently live entries are
        listed; a collection has already removed entries whose table or key
        died.  The returned handles are fresh for the current epoch.
        """
        table_addr = self._resolve(table)
        table_entries = self._associations.get(table_addr, {})
        return tuple(
            (Handle(key_addr, self._epoch), Handle(value_addr, self._epoch))
            for key_addr, value_addr in sorted(table_entries.items())
        )

    # ------------------------------------------------------------------ #
    # Collection
    # ------------------------------------------------------------------ #
    def collect(self) -> CollectionResult:
        """Run one stop-the-world semi-space collection.

        Reachability is computed in two levels from registered roots.
        **Level 1** is the ordinary strong closure: roots are evacuated in
        registration order and strong edges are copied by a Cheney scan in
        field order.  **Level 2** is the weak-key association fixpoint:
        only entries of tables that survived level 1 participate; repeated
        passes activate entries whose key has already been evacuated,
        copying their values as additional strong roots and draining the
        Cheney queue after each activation, until a pass adds nothing.
        Thus a value can make another entry's key alive (the reason one
        pass is insufficient), but a key reachable only through its own
        entry's not-yet-copied value never activates -- the activation test
        precedes the value copy, so the least fixpoint excludes bootstrap
        cycles.

        Surviving objects are copied into the inactive semi-space.  Roots,
        strong edges, surviving weak-key associations and the weak edges
        that name survivors are forwarded; weak edges naming dead objects
        are cleared, and associations of dead tables or with dead keys are
        deleted.

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

            # Phase 3: weak-key association activation, iterated to a least
            # fixpoint.  A table participates iff it has evacuated by the
            # time a pass starts -- this includes tables that an earlier
            # pass's activated value dragged in, not just level-1 tables; a
            # table that never survives has no participating entries and
            # its metadata vanishes unexamined in phase 4.  In each pass
            # every participating table is visited in evacuation order
            # (destination address ascending) and its entries in key-address
            # order.  An entry whose key is *already* evacuated activates:
            # its value is copied (treated as one strong root for this GC)
            # and the Cheney queue is drained immediately, so a freshly
            # copied value can strongly reach keys -- and further tables --
            # that later passes activate.  Passes repeat until a pass
            # activates no new entry.  Because the liveness test reads fwd
            # membership *before* the entry's own value is copied, a key
            # reachable only from that value can never bootstrap itself
            # into liveness.
            old_associations = self._associations
            activated: set[tuple[int, int]] = set()

            def drain_cheney() -> None:
                nonlocal scan
                while scan < free:
                    cur = scan
                    n_strong, n_weak = new_layout[cur]
                    strong_base = cur + 2
                    for j in range(n_strong):
                        old_target = dst[strong_base + j]
                        if old_target is not None:
                            dst[strong_base + j] = copy_obj(old_target)
                    scan = cur + _HEADER + _PAYLOAD + n_strong + n_weak

            while True:
                # Recomputed every pass: fwd (and thus the new addresses) is
                # fixed for already-copied tables, and activated tables are
                # appended at their bump addresses, so the order stays total
                # and deterministic.
                table_order = sorted(
                    (tbl for tbl in old_associations if tbl in fwd),
                    key=lambda tbl: fwd[tbl],
                )
                progressed = False
                for tbl_old in table_order:
                    for key_old, value_old in sorted(old_associations[tbl_old].items()):
                        entry = (tbl_old, key_old)
                        if entry in activated:
                            continue
                        if key_old in fwd:
                            copy_obj(value_old)
                            activated.add(entry)
                            progressed = True
                            drain_cheney()
                if not progressed:
                    break

            live_slots = free

            # Phase 4: rebuild association metadata against forwarded
            # addresses.  Entries of live tables whose key died are deleted
            # and reported; entries of dead tables disappear with the table
            # and are not part of the report.
            new_associations: dict[int, dict[int, int]] = {}
            surviving_entries: list[tuple[int, int, int]] = []
            cleared_entries: list[tuple[int, int]] = []
            for tbl_old in sorted(old_associations):
                if tbl_old not in fwd:
                    continue
                tbl_new = fwd[tbl_old]
                new_map: dict[int, int] = {}
                for key_old, value_old in old_associations[tbl_old].items():
                    if key_old in fwd:
                        # Activation guarantees the value was evacuated too.
                        new_map[fwd[key_old]] = fwd[value_old]
                        surviving_entries.append(
                            (tbl_new, fwd[key_old], fwd[value_old])
                        )
                    else:
                        # Pre-collection addresses for both: a dead key has
                        # no forwarded address, so the pair stays in one
                        # address namespace (translate table via fwd/relocated).
                        cleared_entries.append((tbl_old, key_old))
                if new_map:
                    new_associations[tbl_new] = new_map
            surviving_entries.sort()
            cleared_entries.sort()

            # Phase 5: fix plain weak edges of survivors.  Weak targets never
            # became part of either closure, so they are either in fwd (alive
            # elsewhere, including via an activated association value) or
            # unreachable -> clear.
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
        self._associations = new_associations
        self._epoch += 1

        return CollectionResult(
            relocated=relocated,
            live_slots=live_slots,
            cleared_weak_edges=tuple(cleared),
            roots_preserved=roots_preserved,
            surviving_associations=tuple(surviving_entries),
            cleared_associations=tuple(cleared_entries),
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
