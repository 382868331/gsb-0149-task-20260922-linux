"""Moving (semi-space copying) garbage-collection kernel with weak references.

Simulates a semi-space heap whose capacity is counted in slots (one object per
slot).  Each object holds a fixed number of strong edges, a fixed number of
weak edges and an integer payload.  External references are ``Handle`` values
pairing a slot address with an epoch; handles from a previous epoch (i.e.
minted before the last collection) are stale and rejected.

Collection (``Heap.collect``) traces only from explicit roots along strong
edges, copies live objects in root order then field order (Cheney breadth-first
evacuation), and uses the forwarding table to fix roots, strong edges, and
weak edges whose targets survived.  Weak edges to dead targets are cleared to
``None``.  A failed allocation triggers exactly one collection before raising
``HeapFullError``; no half-initialised object is ever left behind.  There are
no finalizers.

Standard library only.  Python 3.14.7.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_SLOTS = 4096
"""Hard upper bound on heap capacity, in slots."""

__all__ = [
    "MAX_SLOTS",
    "Handle",
    "CollectionResult",
    "Heap",
    "GCError",
    "InvalidArgumentError",
    "InvalidHandleError",
    "StaleHandleError",
    "HeapFullError",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GCError(Exception):
    """Base class for every error raised by gc_kernel."""


class InvalidArgumentError(GCError, ValueError):
    """A configuration value, payload, or field index is invalid."""


class InvalidHandleError(GCError, ValueError):
    """A handle is malformed, out of range, or names no live slot."""


class StaleHandleError(InvalidHandleError):
    """A handle carries an epoch older than the heap's current epoch."""


class HeapFullError(GCError):
    """Allocation failed even after one full collection was performed."""


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Handle:
    """External reference to an object: slot address plus epoch."""

    address: int
    epoch: int


@dataclass(frozen=True)
class CollectionResult:
    """Outcome of one collection.

    ``moved`` maps each surviving object's old address to its new address
    (dead objects are absent).  ``live_slots`` is the number of occupied
    slots after the flip; ``freed_slots`` the number reclaimed.  ``roots``
    holds the relocated registered roots, in registration order.  ``epoch``
    is the heap's epoch after the flip.
    """

    moved: dict[int, int]
    live_slots: int
    freed_slots: int
    roots: tuple[Handle, ...]
    epoch: int


class _Object:
    """One heap slot: fixed strong/weak edge arrays plus an integer payload."""

    __slots__ = ("payload", "strong", "weak")

    def __init__(self, payload: int, strong: list, weak: list) -> None:
        self.payload = payload
        self.strong = strong
        self.weak = weak


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_int(name: str, value, minimum=None, maximum=None) -> int:
    # bool is a subclass of int; the contract rejects it explicitly.  Any
    # non-int (including float NaN/Infinity) is rejected with a locatable
    # error rather than silently coerced.
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentError(
            f"{name} must be an int (bool and non-integral values are "
            f"rejected), got {type(value).__name__}: {value!r}"
        )
    if minimum is not None and value < minimum:
        raise InvalidArgumentError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise InvalidArgumentError(f"{name} must be <= {maximum}, got {value}")
    return value


def _check_handle_shape(name: str, handle) -> Handle:
    if not isinstance(handle, Handle):
        raise InvalidHandleError(
            f"{name} must be a gc_kernel.Handle, got "
            f"{type(handle).__name__}: {handle!r}"
        )
    for field_name, value in (("address", handle.address), ("epoch", handle.epoch)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidHandleError(
                f"{name}.{field_name} must be a non-negative int (bool and "
                f"non-integral values are rejected), got "
                f"{type(value).__name__}: {value!r}"
            )
    return handle


# ---------------------------------------------------------------------------
# Heap
# ---------------------------------------------------------------------------


class Heap:
    """A semi-space heap of ``capacity`` slots.

    ``strong_arity`` and ``weak_arity`` fix how many strong and weak edges
    every object carries.  Roots are registered explicitly with
    :meth:`add_root`; collection traces only from those roots along strong
    edges.
    """

    def __init__(self, capacity: int, strong_arity: int = 1, weak_arity: int = 1) -> None:
        self._capacity = _check_int("capacity", capacity, minimum=1, maximum=MAX_SLOTS)
        self._strong_arity = _check_int("strong_arity", strong_arity, minimum=0)
        self._weak_arity = _check_int("weak_arity", weak_arity, minimum=0)
        self._space: list[_Object] = []  # active semi-space; index == address
        self._epoch = 0
        self._roots: list[Handle] = []

    # -- introspection ----------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def strong_arity(self) -> int:
        return self._strong_arity

    @property
    def weak_arity(self) -> int:
        return self._weak_arity

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def live_slots(self) -> int:
        """Number of occupied slots in the active semi-space."""
        return len(self._space)

    def roots(self) -> tuple[Handle, ...]:
        """Currently valid handles of the registered roots, in order."""
        return tuple(self._roots)

    # -- roots --------------------------------------------------------------

    def add_root(self, handle: Handle) -> Handle:
        """Register ``handle`` as a root; returns the handle unchanged."""
        self._validate(handle, "handle")
        self._roots.append(handle)
        return handle

    def clear_roots(self) -> None:
        self._roots.clear()

    # -- handle validation --------------------------------------------------

    def _validate(self, handle: Handle, name: str = "handle") -> int:
        """Return the slot address of a live handle or raise."""
        _check_handle_shape(name, handle)
        if handle.epoch != self._epoch:
            raise StaleHandleError(
                f"{name} is stale: epoch {handle.epoch}, current epoch is "
                f"{self._epoch} (handle minted before the last collection): "
                f"{handle!r}"
            )
        if handle.address >= len(self._space):
            raise InvalidHandleError(
                f"{name} names slot {handle.address} but only "
                f"{len(self._space)} slot(s) are occupied: {handle!r}"
            )
        return handle.address

    def _check_edges(self, name: str, edges, arity: int) -> list:
        if edges is None:
            return [None] * arity
        if isinstance(edges, (str, bytes)) or not hasattr(edges, "__len__"):
            raise InvalidArgumentError(
                f"{name} must be a sequence of Handle|None of length {arity}, "
                f"got {type(edges).__name__}: {edges!r}"
            )
        edges = list(edges)
        if len(edges) != arity:
            raise InvalidArgumentError(
                f"{name} must have exactly {arity} entrie(s), got {len(edges)}"
            )
        for i, h in enumerate(edges):
            if h is not None:
                self._validate(h, f"{name}[{i}]")
        return edges

    def _check_index(self, name: str, index: int, arity: int) -> int:
        _check_int(name, index, minimum=0)
        if index >= arity:
            raise InvalidArgumentError(
                f"{name} must be < {arity} (the fixed arity), got {index}"
            )
        return index

    # -- allocation -----------------------------------------------------------

    def alloc(self, payload: int = 0, strong=None, weak=None) -> Handle:
        """Allocate one object; returns its handle.

        ``strong``/``weak`` are optional sequences of ``Handle|None`` whose
        lengths must equal the fixed arities.  If the active semi-space is
        full, exactly one collection is performed first (registered roots
        plus the strong-edge targets being installed act as roots; weak-edge
        targets are *not* kept alive).  If the heap is still full,
        ``HeapFullError`` is raised and no object is created.
        """
        _check_int("payload", payload)
        strong = self._check_edges("strong", strong, self._strong_arity)
        weak = self._check_edges("weak", weak, self._weak_arity)
        if len(self._space) >= self._capacity:
            keep = [h for h in strong if h is not None]
            result = self._collect(extra_roots=keep)
            strong = [self._remap_after_collect(h, result, strong_edge=True) for h in strong]
            weak = [self._remap_after_collect(h, result, strong_edge=False) for h in weak]
            if len(self._space) >= self._capacity:
                raise HeapFullError(
                    f"heap is full: {len(self._space)}/{self._capacity} slots "
                    f"still occupied after one collection; allocation refused, "
                    f"no object created"
                )
        address = len(self._space)
        self._space.append(_Object(payload, strong, weak))
        return Handle(address, self._epoch)

    @staticmethod
    def _remap_after_collect(handle, result: CollectionResult, strong_edge: bool):
        if handle is None:
            return None
        new_address = result.moved.get(handle.address)
        if new_address is None:
            if strong_edge:
                # Strong targets were collection roots; they must survive.
                raise GCError(
                    f"internal error: strong-edge target {handle!r} lost "
                    f"during allocation-triggered collection"
                )
            return None  # weak target died; edge is cleared
        return Handle(new_address, result.epoch)

    # -- collection -----------------------------------------------------------

    def collect(self) -> CollectionResult:
        """Trace from the registered roots and compact survivors.

        Copies live objects in root order, then breadth-first in strong-field
        order; rewrites roots, strong edges, and weak edges to surviving
        targets through the forwarding table; clears weak edges to dead
        targets.  Returns a :class:`CollectionResult` with the relocation
        map and the surviving slot count.
        """
        return self._collect(extra_roots=())

    def _collect(self, extra_roots) -> CollectionResult:
        old_space = self._space
        old_epoch = self._epoch
        new_epoch = old_epoch + 1

        root_handles = list(self._roots) + list(extra_roots)
        n_registered = len(self._roots)
        root_addresses = [
            self._validate(h, f"roots[{i}]") for i, h in enumerate(root_handles)
        ]

        forwarding: dict[int, int] = {}
        to_space: list[_Object] = []

        def evacuate(address: int) -> int:
            target = forwarding.get(address)
            if target is None:
                target = len(to_space)
                forwarding[address] = target
                to_space.append(old_space[address])
            return target

        # Roots first, in registration order.
        for address in root_addresses:
            evacuate(address)
        # Then breadth-first, scanning strong fields in field order.
        scan = 0
        while scan < len(to_space):
            obj = to_space[scan]
            for i, edge in enumerate(obj.strong):
                if edge is not None:
                    # Invariant: stored edges always carry the current epoch,
                    # because every write path validates handles.
                    obj.strong[i] = Handle(evacuate(edge.address), new_epoch)
            scan += 1
        # Weak edges never keep targets alive; fix or clear them now.
        for obj in to_space:
            for i, edge in enumerate(obj.weak):
                if edge is not None:
                    target = forwarding.get(edge.address)
                    obj.weak[i] = (
                        Handle(target, new_epoch) if target is not None else None
                    )

        self._space = to_space
        self._epoch = new_epoch
        self._roots = [
            Handle(forwarding[address], new_epoch)
            for address in root_addresses[:n_registered]
        ]
        return CollectionResult(
            moved=dict(forwarding),
            live_slots=len(to_space),
            freed_slots=len(old_space) - len(to_space),
            roots=tuple(self._roots),
            epoch=new_epoch,
        )

    # -- field access ---------------------------------------------------------

    def read_payload(self, handle: Handle) -> int:
        return self._space[self._validate(handle)].payload

    def write_payload(self, handle: Handle, payload: int) -> None:
        _check_int("payload", payload)
        self._space[self._validate(handle)].payload = payload

    def read_strong(self, handle: Handle, index: int):
        self._check_index("index", index, self._strong_arity)
        return self._space[self._validate(handle)].strong[index]

    def write_strong(self, handle: Handle, index: int, target) -> None:
        self._check_index("index", index, self._strong_arity)
        if target is not None:
            self._validate(target, "target")
        self._space[self._validate(handle)].strong[index] = target

    def read_weak(self, handle: Handle, index: int):
        self._check_index("index", index, self._weak_arity)
        return self._space[self._validate(handle)].weak[index]

    def write_weak(self, handle: Handle, index: int, target) -> None:
        self._check_index("index", index, self._weak_arity)
        if target is not None:
            self._validate(target, "target")
        self._space[self._validate(handle)].weak[index] = target

    def describe(self, handle: Handle) -> dict:
        """Snapshot of one object: payload plus strong/weak edge addresses."""
        obj = self._space[self._validate(handle)]
        return {
            "address": handle.address,
            "epoch": handle.epoch,
            "payload": obj.payload,
            "strong": [h.address if h is not None else None for h in obj.strong],
            "weak": [h.address if h is not None else None for h in obj.weak],
        }
