"""Generational, incremental garbage collector.

This module adds a two-generation heap alongside the original flat
semi-space kernel (:mod:`moving_gc.core`, whose :class:`~moving_gc.Heap`
contract is unchanged):

* **young generation** -- semi-space copying minor collector;
* **old generation** -- mark-sweep major collector (addresses never move);
* **incremental update (IU) write barrier** -- when a black object gains a
  strong edge to a white object the target is greyed immediately, so
  mutator action between finite collection steps cannot lose a reference;
* **remembered set** -- old -> young strong edges are logged on every
  cross-generation write, so a minor never scans the whole old generation
  and never misses such an edge;
* **ephemeron table** -- ``(table, key, value)`` triples whose value joins
  the closure only while both table and key are reachable; the table is
  re-scanned to a fixpoint before ordinary weak edges are cleared;
* **explicit finalizers** -- callbacks delivered as explicit events at
  major collections, at most once per object; a callback may allocate and
  resurrect its object by re-rooting it.

Every collection is driven through :meth:`GenHeap.collect_step` with an
explicit *scan budget*.  One work unit is one examination: a root slot, an
object edge slot, an ephemeron entry, an old object during finalizer
candidate discovery/sweep, or one survivor evacuation.  Each step performs
exactly ``min(budget, remaining)`` examinations (reported exactly in
``StepResult.scanned``; ``overscan`` is always 0) plus O(1) per-step
dispatch.  A step therefore never scans the whole heap internally; the
caller controls interleaving.  The only non-budgeted work is the final
commit (pointer/table swaps and one address-ordering sort); it never
examines edges and is documented in README.md.

Address convention: young object addresses are raw young slots in
``[0, young_capacity)``; old object addresses are
``young_capacity + raw_old_slot`` in
``[young_capacity, young_capacity + old_capacity)``.

Teaching model, not a production runtime.
"""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field

from .core import MAX_CAPACITY, Handle, _check_int
from .errors import (
    CollectionInProgressError,
    FinalizerError,
    HeapCapacityError,
    SealedForReclamationError,
    StaleHandleError,
)

__all__ = [
    "GenHeap",
    "StepResult",
    "YOUNG",
    "OLD",
]

YOUNG = 0
OLD = 1

_WHITE = 0
_GRAY = 1
_BLACK = 2

# Phases in which strong writes can still enlarge the closure.
_OPEN_PHASES = ("mark", "fdiscover", "ephemeron", "plan", "finalize")


@dataclass(frozen=True)
class StepResult:
    """Outcome of one :meth:`GenHeap.collect_step` call.

    Attributes:
        kind: ``"minor"`` / ``"major"``.
        phase: phase after this step; ``"done"`` iff ``completed``.
        completed: true exactly on the step that commits the cycle.
        budget: granted scan budget.
        scanned: examinations consumed this step (``<= budget``).
        overscan: always 0; the constant-overrun claim is a field, not a
            promise in prose.
        gray_remaining: grey objects still queued.
        steps_in_cycle: 1-based count of steps in this cycle.
        barrier_records: cumulative IU barrier firings in this cycle.
        relocated: minor only, ``young_addr -> new_young_addr`` for objects
            that stayed young.
        promoted: minor only, ``young_addr -> old_addr``.
        live_young_slots / live_old_slots: occupancy after commit.
        cleared_weak_edges: cleared this cycle,
            ``(holder_addr, field_index)`` sorted, on the completed step.
        cleared_ephemerons: ephemeron ids removed this cycle.
        finalized_now: ``(generation, address, token, resurrected)`` events
            delivered this step.
        finalized: same shape, cumulative for the cycle.
    """

    kind: str
    phase: str
    completed: bool
    budget: int
    scanned: int
    overscan: int
    gray_remaining: int
    steps_in_cycle: int
    barrier_records: int = 0
    relocated: dict[int, int] = field(default_factory=dict)
    promoted: dict[int, int] = field(default_factory=dict)
    live_young_slots: int = 0
    live_old_slots: int = 0
    cleared_weak_edges: tuple[tuple[int, int], ...] = ()
    cleared_ephemerons: tuple[int, ...] = ()
    finalized_now: tuple[tuple[int, int, int, bool], ...] = ()
    finalized: tuple[tuple[int, int, int, bool], ...] = ()


class _Budget:
    __slots__ = ("remaining", "used")

    def __init__(self, budget: int) -> None:
        self.remaining = budget
        self.used = 0

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        self.used += 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0


class GenHeap:
    """Two-generation heap with incremental budgeted collection.

    Args:
        young_capacity: slots per young semi-space (>= 1).
        old_capacity: slots of the old generation (>= 1).
        promote_age: survived-minor count at which a young object promotes
            (>= 1; default 2: born at age 0, survives one minor, promotes
            at the next).  Forced promotions ignore the age.
    """

    def __init__(
        self,
        young_capacity: int,
        old_capacity: int,
        *,
        promote_age: int = 2,
    ) -> None:
        _check_int(young_capacity, "young_capacity")
        _check_int(old_capacity, "old_capacity")
        _check_int(promote_age, "promote_age")
        if young_capacity < 1:
            raise ValueError(f"young_capacity must be >= 1, got {young_capacity}")
        if old_capacity < 1:
            raise ValueError(f"old_capacity must be >= 1, got {old_capacity}")
        if young_capacity + old_capacity > MAX_CAPACITY:
            raise ValueError(
                f"young_capacity + old_capacity must be <= {MAX_CAPACITY}, "
                f"got {young_capacity + old_capacity}"
            )
        if promote_age < 1:
            raise ValueError(f"promote_age must be >= 1, got {promote_age}")
        self._yc = young_capacity
        self._oc = old_capacity
        self._promote_age = promote_age
        # Two young semi-spaces plus one old mark-sweep space.
        self._young: list[list] = [
            [None for _ in range(young_capacity)],
            [None for _ in range(young_capacity)],
        ]
        self._old: list = [None for _ in range(old_capacity)]
        self._yidx = 0
        self._yspace: list = self._young[0]
        self._ylay: dict[int, tuple[int, int]] = {}
        self._olay: dict[int, tuple[int, int]] = {}
        self._yorder: list[int] = []        # sorted young raws
        self._oorder: list[int] = []        # sorted old raws
        self._yalloc = 0
        self._ohwm = 0                      # old bump high-water
        self._oocc = 0                      # occupied old slots
        self._old_free: list[tuple[int, int]] = []  # sorted (raw, size)
        self._epoch = 0
        self._roots: dict[str, tuple[int, int] | None] = {}
        self._root_order: list[str] = []
        self._age: dict[int, int] = {}
        self._force_promote: set[int] = set()
        self._rs: set[int] = set()          # old holders with a young strong edge
        self._wrs: set[int] = set()         # ditto, weak (fix-up only)
        # id -> (tg, ta, kg, ka, vg, va|None)
        self._eph: dict[int, tuple[int, int, int, int, int, int | None]] = {}
        self._eph_order: list[int] = []
        self._eph_next = 0
        self._fin: dict[tuple[int, int], tuple[object, int]] = {}
        self._fin_ran: set[tuple[int, int]] = set()
        self._fin_next = 0
        self._cycle: dict | None = None
        self._in_callback = False

    # ================================================================== #
    # Address helpers
    # ================================================================== #
    def _split(self, logical: int) -> tuple[int, int]:
        if 0 <= logical < self._yc:
            return YOUNG, logical
        if self._yc <= logical < self._yc + self._oc:
            return OLD, logical - self._yc
        raise StaleHandleError(f"address {logical} is outside the heap")

    def _join(self, gen: int, raw: int) -> int:
        return raw if gen == YOUNG else self._yc + raw

    def _lay(self, gen: int) -> dict[int, tuple[int, int]]:
        return self._ylay if gen == YOUNG else self._olay

    # ================================================================== #
    # Introspection
    # ================================================================== #
    @property
    def capacity(self) -> int:
        return self._yc + self._oc

    @property
    def young_capacity(self) -> int:
        return self._yc

    @property
    def old_capacity(self) -> int:
        return self._oc

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def used_slots(self) -> int:
        return self._yalloc + self._oocc

    @property
    def young_used_slots(self) -> int:
        return self._yalloc

    @property
    def old_used_slots(self) -> int:
        return self._oocc

    @property
    def free_slots(self) -> int:
        return (self._yc - self._yalloc) + (
            sum(s for _, s in self._old_free) + (self._oc - self._ohwm)
        )

    @property
    def phase(self) -> str | None:
        return None if self._cycle is None else self._cycle["phase"]

    def objects(self, generation: int | None = None) -> tuple[int, ...]:
        """Logical addresses of live objects, optionally one generation."""
        if generation is None:
            return tuple(self._yorder) + tuple(self._yc + r for r in self._oorder)
        _check_int(generation, "generation")
        if generation == YOUNG:
            return tuple(self._yorder)
        if generation == OLD:
            return tuple(self._yc + r for r in self._oorder)
        raise ValueError(f"generation must be 0 (young) or 1 (old), got {generation}")

    def generation_of(self, target: Handle) -> int:
        g, _ = self._resolve(target)
        return g

    def age_of(self, target: Handle) -> int:
        """Survived-minor count (0 for old objects)."""
        g, raw = self._resolve(target)
        if g == OLD:
            return 0
        c = self._cycle
        if c is not None and c["kind"] == "minor" and c["phase"] == "evac":
            ng, nr = c["fwd"].get(raw, (g, raw))
            if ng == OLD:
                return 0
            return c["new_age"].get(nr, self._age.get(raw, 0))
        return self._age.get(raw, 0)

    def remembered_set(self) -> tuple[tuple[int, int], ...]:
        """Strong remembered-set entries ``(holder_addr, field_index)``.

        Every old holder whose strong fields currently name a young object,
        sorted by holder and field.  Between cycles the set may contain
        conservative holder entries; holders are proven/pruned during a
        minor's remembered-set fix stage.
        """
        out: list[tuple[int, int]] = []
        for raw in sorted(self._rs):
            if raw not in self._olay:
                continue
            ns, _ = self._olay[raw]
            for j in range(ns):
                v = self._old[raw + 2 + j]
                if v is not None and 0 <= v < self._yc:
                    out.append((self._yc + raw, j))
        return tuple(out)

    def gray_pending(self) -> tuple[int, ...]:
        c = self._cycle
        if c is None:
            return ()
        return tuple(self._join(g, r) for g, r in c["queue"])

    def marked_addresses(self) -> frozenset[int]:
        c = self._cycle
        if c is None:
            return frozenset()
        return frozenset(self._join(g, r) for (g, r) in c["color"])

    def pending_finalizers(self) -> tuple[int, ...]:
        c = self._cycle
        if c is None or "pending_fin" not in c:
            return ()
        return tuple(self._join(OLD, r) for r in c["pending_fin"])

    def ephemeron_ids(self) -> tuple[int, ...]:
        return tuple(self._eph_order)

    def promotion_pending(self) -> tuple[int, ...]:
        return tuple(sorted(self._force_promote))

    # ================================================================== #
    # Allocation
    # ================================================================== #
    def allocate(
        self,
        payload: int = 0,
        *,
        n_strong: int = 0,
        n_weak: int = 0,
    ) -> Handle:
        """Allocate one object in the young generation.

        Edge slots start null.  If the young space is full while idle, one
        minor runs on the caller's behalf (legacy emergency rule; it only
        touches young survivors and the remembered set).  Between steps of
        an open cycle (mark/ephemeron/plan/finalize) the new object is born
        grey and joins the current closure; during a minor's ``evac`` it is
        carved from the survivor-space tail; during a major's closed
        phases (weak clearing/sweep) it is a plain young allocation that
        cannot affect the sealed old death set.  A full young space
        mid-cycle raises :class:`CollectionInProgressError`.
        """
        _check_int(payload, "payload")
        _check_int(n_strong, "n_strong")
        _check_int(n_weak, "n_weak")
        if n_strong < 0:
            raise ValueError(f"n_strong must be >= 0, got {n_strong}")
        if n_weak < 0:
            raise ValueError(f"n_weak must be >= 0, got {n_weak}")
        size = 2 + n_strong + n_weak
        if size > self._yc:
            if self._cycle is None:
                self.run_minor()
            raise HeapCapacityError(
                f"out of capacity: object size {size} exceeds young capacity {self._yc}"
            )
        c = self._cycle
        if c is not None and c["kind"] == "minor" and c["phase"] == "evac":
            return self._alloc_evac_tail(payload, n_strong, n_weak, size)
        if self._yalloc + size > self._yc:
            if c is not None:
                raise CollectionInProgressError(
                    "young space is full; finish the in-progress collection before allocating"
                )
            self.run_minor()
            if self._yalloc + size > self._yc:
                raise HeapCapacityError(
                    f"out of capacity: need {size} slot(s), "
                    f"{self._yc - self._yalloc} young free after minor collection"
                )
        raw = self._yalloc
        self._write_object(self._yspace, raw, payload, n_strong, n_weak)
        self._yalloc = raw + size
        self._ylay[raw] = (n_strong, n_weak)
        self._yorder.append(raw)
        self._age[raw] = 0
        if c is not None and c["phase"] in _OPEN_PHASES:
            # Born grey, born rooted for this cycle: nursery allocations
            # survive the in-flight collection (standard nursery rule).
            oid = (YOUNG, raw)
            c["color"][oid] = _GRAY
            c["queue"].append(oid)
            if c["kind"] == "minor":
                self._seed_add(c, raw)
        return Handle(raw, self._epoch)

    @staticmethod
    def _write_object(space, raw, payload, ns, nw) -> None:
        size = 2 + ns + nw
        space[raw] = None
        space[raw + 1] = payload
        for i in range(2, size):
            space[raw + i] = None

    def _insert_old_hole(self, raw: int, size: int) -> None:
        insort(self._old_free, (raw, size))
        # Coalesce with neighbours in one pass.
        merged: list[tuple[int, int]] = []
        for hraw, hsize in self._old_free:
            if merged and merged[-1][0] + merged[-1][1] == hraw:
                merged[-1] = (merged[-1][0], merged[-1][1] + hsize)
            else:
                merged.append((hraw, hsize))
        self._old_free = merged

    def _reserve_old(self, size: int) -> int:
        for i, (raw, hole) in enumerate(self._old_free):
            if hole >= size:
                del self._old_free[i]
                if hole > size:
                    self._insert_old_hole(raw + size, hole - size)
                self._oocc += size
                return raw
        raw = self._ohwm
        if raw + size > self._oc:
            raise HeapCapacityError(
                f"old generation full: need {size} slot(s), {self._oc - raw} past "
                f"high-water and {sum(s for _, s in self._old_free)} in holes"
            )
        self._ohwm = raw + size
        self._oocc += size
        return raw

    def _alloc_evac_tail(self, payload, ns, nw, size) -> Handle:
        c = self._cycle
        if c["tail"] + size > self._yc:
            raise HeapCapacityError(
                f"young space full during evacuation: need {size}, tail at {c['tail']}"
            )
        raw = c["tail"]
        self._write_object(c["to"], raw, payload, ns, nw)
        c["tail"] = raw + size
        # Tail allocations are post-copy objects from birth: register them
        # in the published survivor layout directly so reads/writes during
        # the rest of evac resolve to to-space.  Their edges start null and
        # therefore need no forwarding fix-up (they are excluded from
        # young_new, which drives the survivor edge-fix stages).
        c["new_lay"][raw] = (ns, nw)
        c["new_age"][raw] = 0
        insort(c["new_order"], raw)
        return Handle(raw, self._epoch)

    # ================================================================== #
    # Roots
    # ================================================================== #
    def add_root(self, name: str, target: Handle) -> None:
        """Register/replace a named root (the handle is validated first)."""
        if not isinstance(name, str) or not name:
            raise TypeError("root name must be a non-empty str")
        oid = self._resolve(target)
        eff = self._check_mutation_target(oid)
        if eff is not None:
            oid = eff
        if name not in self._roots:
            self._root_order.append(name)
        self._roots[name] = oid
        c = self._cycle
        if c is not None and c["phase"] in _OPEN_PHASES:
            if c["kind"] == "major":
                self._gray(c, oid)
            elif oid[0] == YOUNG and self._gray(c, oid):
                self._seed_add(c, oid[1])

    def clear_root(self, name: str) -> None:
        if name not in self._roots:
            raise KeyError(name)
        self._roots[name] = None

    def remove_root(self, name: str) -> None:
        if name not in self._roots:
            raise KeyError(name)
        del self._roots[name]
        self._root_order.remove(name)

    def root_handle(self, name: str) -> Handle | None:
        if name not in self._roots:
            raise KeyError(name)
        oid = self._roots[name]
        return None if oid is None else Handle(self._join(*oid), self._epoch)

    def roots(self) -> dict[str, Handle | None]:
        return {
            name: (None if oid is None else Handle(self._join(*oid), self._epoch))
            for name, oid in self._roots.items()
        }

    # ================================================================== #
    # Ephemerons / finalizers
    # ================================================================== #
    def add_ephemeron(self, table: Handle, key: Handle, value: Handle | None) -> int:
        """Register ``(table, key, value)``; returns the ephemeron id.

        Keeps neither table nor key alive.  The value joins the closure
        only while both table and key are reachable (fixpoint).  A null
        value traces nothing on activation.
        """
        t = self._resolve(table)
        k = self._resolve(key)
        v = None if value is None else self._resolve(value)
        eid = self._eph_next
        self._eph_next += 1
        self._eph[eid] = (
            t[0], t[1], k[0], k[1],
            v[0] if v is not None else 0,
            v[1] if v is not None else None,
        )
        self._eph_order.append(eid)
        return eid

    def remove_ephemeron(self, ephemeron_id: int) -> None:
        _check_int(ephemeron_id, "ephemeron_id")
        if ephemeron_id not in self._eph:
            raise KeyError(ephemeron_id)
        del self._eph[ephemeron_id]
        self._eph_order.remove(ephemeron_id)

    def ephemeron(self, ephemeron_id: int) -> tuple[Handle, Handle, Handle | None]:
        _check_int(ephemeron_id, "ephemeron_id")
        if ephemeron_id not in self._eph:
            raise KeyError(ephemeron_id)
        tg, ta, kg, ka, vg, va = self._eph[ephemeron_id]
        return (
            Handle(self._join(tg, ta), self._epoch),
            Handle(self._join(kg, ka), self._epoch),
            None if va is None else Handle(self._join(vg, va), self._epoch),
        )

    def set_finalizer(self, target: Handle, callback: object) -> int:
        """Attach ``callback(heap, handle)`` to an object; returns a token.

        Explicit major-collection events, at most once per object.  Young
        objects with an unrun finalizer are minor roots (they survive until
        promoted; the event itself is delivered by a major cycle).
        Registration mid-cycle raises :class:`CollectionInProgressError`;
        re-registration after delivery raises :class:`FinalizerError`.
        """
        if not callable(callback):
            raise TypeError("finalizer callback must be callable")
        oid = self._resolve(target)
        if oid in self._fin_ran:
            raise FinalizerError(
                f"object {self._join(*oid)} already had its one finalizer run"
            )
        if self._cycle is not None:
            raise CollectionInProgressError(
                "register finalizers only between collection cycles"
            )
        token = self._fin_next
        self._fin_next += 1
        self._fin[oid] = (callback, token)
        return token

    def finalizer_ran(self, target: Handle) -> bool:
        return self._resolve(target) in self._fin_ran

    def request_promotion(self, target: Handle) -> None:
        """Force promotion at the next minor (only survivors move)."""
        g, raw = self._resolve(target)
        c = self._cycle
        if c is not None and not (c["kind"] == "minor" and c["phase"] in _OPEN_PHASES):
            raise CollectionInProgressError(
                "promotion requests are accepted when idle or during a minor's open phases"
            )
        if g == YOUNG:
            self._force_promote.add(raw)

    # ================================================================== #
    # Handles / fields
    # ================================================================== #
    def handle(self, address: int) -> Handle:
        _check_int(address, "address")
        self._locate(address)
        return Handle(address, self._epoch)

    def _locate(self, logical: int) -> tuple[int, int]:
        g, raw = self._split(logical)
        c = self._cycle
        if g == YOUNG and c is not None and c["kind"] == "minor" and c["phase"] == "evac":
            if raw in c["fwd"]:
                return c["fwd"][raw]
            if raw in c["new_lay"]:
                return YOUNG, raw
        if raw in self._lay(g):
            return g, raw
        if g == YOUNG and c is not None and raw in c.get("fwd", {}):
            return c["fwd"][raw]
        raise StaleHandleError(f"no live object at address {logical}")

    def _resolve(self, target: Handle) -> tuple[int, int]:
        if not isinstance(target, Handle):
            raise TypeError(f"expected Handle, got {type(target).__name__}")
        if target.epoch != self._epoch:
            raise StaleHandleError(
                f"stale handle: epoch {target.epoch} != current epoch {self._epoch} "
                f"(address {target.address})"
            )
        return self._locate(target.address)

    def _space(self, gen: int, raw: int) -> list:
        c = self._cycle
        if (
            gen == YOUNG and c is not None and c["kind"] == "minor"
            and c["phase"] == "evac" and raw in c["new_lay"]
        ):
            return c["to"]
        return self._yspace if gen == YOUNG else self._old

    def _shape(self, gen: int, raw: int) -> tuple[int, int]:
        c = self._cycle
        if (
            gen == YOUNG and c is not None and c["kind"] == "minor"
            and c["phase"] == "evac" and raw in c["new_lay"]
        ):
            return c["new_lay"][raw]
        return self._lay(gen)[raw]

    def _read_edge(self, v):
        if v is None:
            return None
        g, r = self._locate(v)
        return Handle(self._join(g, r), self._epoch)

    def payload(self, target: Handle) -> int:
        g, raw = self._resolve(target)
        return self._space(g, raw)[raw + 1]

    def set_payload(self, target: Handle, value: int) -> None:
        _check_int(value, "value")
        g, raw = self._resolve(target)
        eff = self._check_mutation_target((g, raw))
        if eff is not None:
            g, raw = eff
        self._space(g, raw)[raw + 1] = value

    def get_strong(self, target: Handle, index: int) -> Handle | None:
        g, raw = self._resolve(target)
        ns, _ = self._shape(g, raw)
        self._check_index(index, ns, "strong")
        return self._read_edge(self._space(g, raw)[raw + 2 + index])

    def set_strong(self, target: Handle, index: int, ref: Handle | None) -> None:
        g, raw = self._resolve(target)
        ns, _ = self._shape(g, raw)
        self._check_index(index, ns, "strong")
        eff_h = self._check_mutation_target((g, raw))
        if eff_h is not None:
            g, raw = eff_h
        if ref is None:
            oid = None
        else:
            oid = self._resolve(ref)
            eff = self._check_mutation_target(oid)
            if eff is not None:
                oid = eff
        self._space(g, raw)[raw + 2 + index] = None if oid is None else self._join(*oid)
        if oid is not None:
            self._post_strong_write(g, raw, index, oid)

    def get_weak(self, target: Handle, index: int) -> Handle | None:
        g, raw = self._resolve(target)
        ns, nw = self._shape(g, raw)
        self._check_index(index, nw, "weak")
        return self._read_edge(self._space(g, raw)[raw + 2 + ns + index])

    def set_weak(self, target: Handle, index: int, ref: Handle | None) -> None:
        g, raw = self._resolve(target)
        ns, nw = self._shape(g, raw)
        self._check_index(index, nw, "weak")
        eff_h = self._check_mutation_target((g, raw))
        if eff_h is not None:
            g, raw = eff_h
        if ref is None:
            oid = None
        else:
            oid = self._resolve(ref)
            eff = self._check_mutation_target(oid)
            if eff is not None:
                oid = eff
        self._space(g, raw)[raw + 2 + ns + index] = (
            None if oid is None else self._join(*oid)
        )
        if g == OLD and oid is not None and oid[0] == YOUNG:
            self._wrs.add(raw)

    def _evac_ensure(self, c: dict, oid: tuple[int, int]) -> tuple[int, int]:
        """Evacuation copy barrier for one named object.

        During a minor's evac phase a mutator write/root may name a young
        survivor that has not been physically copied yet.  Copy it now
        (the survivor set itself is already fixed), so the slot stores a
        post-collection address.  Naming a young object that is *not* a
        survivor is rejected: it is sealed dead for this cycle.
        """
        g, raw = oid
        if g != YOUNG:
            return oid
        if raw in c["fwd"]:
            return c["fwd"][raw]
        if raw in c["new_lay"]:
            return YOUNG, raw
        if (YOUNG, raw) not in c["color"]:
            raise SealedForReclamationError(
                f"young object {raw} is not a survivor of the in-progress minor"
            )
        # Mutator-triggered copy of one survivor; bounded by the object
        # size and charged to the mutation, not to a step budget.
        self._publish_copy(c, raw, budget=None)
        return c["fwd"][raw]

    def _check_mutation_target(self, oid: tuple[int, int]) -> tuple[int, int] | None:
        """Validate a write target against the active cycle's sealed set.

        Returns the effective (possibly post-copy) object id during a
        minor evacuation; ``None`` when no cycle is active.
        """
        c = self._cycle
        if c is None:
            return None
        if c["kind"] == "minor" and c["phase"] == "evac":
            return self._evac_ensure(c, oid)
        if c["kind"] == "major" and c["phase"] in ("weakclear", "sweep"):
            if oid[0] == OLD and oid not in c["color"]:
                raise SealedForReclamationError(
                    f"object {self._join(*oid)} is sealed for reclamation"
                )
        return oid

    def _post_strong_write(self, g: int, raw: int, index: int,
                           oid: tuple[int, int]) -> None:
        # Cross-generation write -> remembered set (strong roots for minor).
        if g == OLD and oid[0] == YOUNG:
            self._rs.add(raw)
        c = self._cycle
        if c is None or c["phase"] not in _OPEN_PHASES:
            return
        # Minor: old holders are never coloured; every old -> young write
        # during an open minor must keep the young target in the closure.
        if c["kind"] == "minor" and g == OLD and oid[0] == YOUNG:
            self._barrier_grey(c, oid)
            return
        holder = (g, raw)
        hcol = c["color"].get(holder, _WHITE)
        if hcol == _WHITE or oid in c["color"]:
            # A white holder is scanned in full later; an already-coloured
            # target needs no protection.
            return
        if hcol == _BLACK:
            # Every strong field was already scanned: protect the target.
            self._barrier_grey(c, oid)
        elif index < c["cursor"].get(holder, 0):
            # Grey holder whose written field was already scanned.
            self._barrier_grey(c, oid)

    def _barrier_grey(self, c: dict, oid: tuple[int, int]) -> None:
        """IU barrier: keep ``oid`` in the closure (minor seeds it too)."""
        if self._gray(c, oid):
            c["barrier"] += 1
            if c["kind"] == "minor" and oid[0] == YOUNG:
                self._seed_add(c, oid[1])

    @staticmethod
    def _check_index(index: int, length: int, kind: str) -> None:
        _check_int(index, "index")
        if not 0 <= index < length:
            raise IndexError(
                f"{kind} edge index {index} out of range for object with {length} {kind} edge(s)"
            )

    def object_shape(self, target: Handle) -> tuple[int, int]:
        g, raw = self._resolve(target)
        return self._shape(g, raw)

    def reachable_addresses(self) -> frozenset[int]:
        """Independent read-only strong reachability from roots, right now."""
        seen: set[tuple[int, int]] = set()
        stack = [oid for oid in self._roots.values() if oid is not None]
        while stack:
            oid = stack.pop()
            if oid in seen:
                continue
            seen.add(oid)
            g, raw = oid
            c = self._cycle
            if g == YOUNG and c is not None and c["kind"] == "minor" and c["phase"] == "evac":
                if raw in c.get("fwd", {}):
                    g, raw = c["fwd"][raw]
            ns, _ = self._shape(g, raw)
            space = self._space(g, raw)
            for j in range(ns):
                v = space[raw + 2 + j]
                if v is None:
                    continue
                oid2 = self._split(v)
                c = self._cycle
                if oid2[0] == YOUNG and c is not None and c["kind"] == "minor" and c["phase"] == "evac":
                    if oid2[1] in c.get("fwd", {}):
                        oid2 = c["fwd"][oid2[1]]
                if oid2 not in seen:
                    stack.append(oid2)
        return frozenset(self._join(*o) for o in seen)

    # ================================================================== #
    # Incremental driver
    # ================================================================== #
    def collect_step(self, scan_budget: int, kind: str | None = None) -> StepResult:
        """Advance a cycle by at most ``scan_budget`` examinations.

        Start with ``kind="minor"``/``"major"``; resume with ``kind``
        omitted (or equal).  Mutation between steps is legal subject to the
        per-phase rules; finalizer callbacks may allocate and re-root.
        """
        _check_int(scan_budget, "scan_budget")
        if scan_budget < 1:
            raise ValueError(f"scan_budget must be >= 1, got {scan_budget}")
        if self._in_callback:
            raise CollectionInProgressError(
                "collect_step() must not be called from inside a finalizer"
            )
        c = self._cycle
        if c is None:
            if kind not in ("minor", "major"):
                raise ValueError(f"kind must be 'minor' or 'major', got {kind!r}")
            c = self._start_cycle(kind)
        elif kind is not None and kind != c["kind"]:
            raise CollectionInProgressError(
                f"{c['kind']} cycle in progress; cannot start a {kind} cycle"
            )
        c["steps"] += 1
        budget = _Budget(scan_budget)
        fin_now: list[tuple[int, int, int, bool]] = []
        completed = False
        try:
            while not budget.exhausted:
                phase = c["phase"]
                if phase == "mark":
                    self._step_mark(c, budget)
                elif phase == "fdiscover":
                    self._step_fdiscover(c, budget)
                elif phase == "plan":
                    self._step_plan(c, budget)
                elif phase == "ephemeron":
                    self._step_ephemeron(c, budget)
                elif phase == "finalize":
                    self._step_finalize(c, budget, fin_now)
                elif phase == "weakclear":
                    self._step_weakclear(c, budget)
                elif phase == "sweep":
                    completed = self._step_sweep(c, budget)
                elif phase == "evac":
                    completed = self._step_evac(c, budget)
                else:  # pragma: no cover
                    raise AssertionError(f"unknown phase {phase!r}")
                if completed:
                    break
        except HeapCapacityError:
            # Promotion planning fails before anything moves: abort the
            # minor cleanly so the caller can run a major and retry.
            if c["kind"] == "minor" and c["phase"] in ("plan", "ephemeron"):
                self._cycle = None
            raise
        if completed:
            summary = self._commit(c, fin_now, scan_budget, budget.used)
            self._cycle = None
            return summary
        return StepResult(
            kind=c["kind"],
            phase=c["phase"],
            completed=False,
            budget=scan_budget,
            scanned=budget.used,
            overscan=0,
            gray_remaining=len(c["queue"]),
            steps_in_cycle=c["steps"],
            barrier_records=c["barrier"],
            finalized_now=tuple(fin_now),
            finalized=tuple(c["finalized"]),
        )

    def run_minor(self, scan_budget: int = 256) -> StepResult:
        """Run a complete minor cycle, pumping budgeted steps internally."""
        return self._run_to_completion("minor", scan_budget)

    def run_major(self, scan_budget: int = 256) -> StepResult:
        """Run a complete major cycle, pumping budgeted steps internally."""
        return self._run_to_completion("major", scan_budget)

    def _run_to_completion(self, kind: str, scan_budget: int) -> StepResult:
        _check_int(scan_budget, "scan_budget")
        if scan_budget < 1:
            raise ValueError(f"scan_budget must be >= 1, got {scan_budget}")
        result = self.collect_step(scan_budget, kind)
        while not result.completed:
            result = self.collect_step(scan_budget)
        return result

    # ------------------------------------------------------------------ #
    # Cycle state
    # ------------------------------------------------------------------ #
    def _start_cycle(self, kind: str) -> dict:
        c: dict = {
            "kind": kind,
            "phase": "mark",
            "steps": 0,
            "color": {},
            "queue": [],
            "cursor": {},
            "barrier": 0,
            "finalized": [],
            "cleared_weak": [],
            "cleared_eph": [],
            "seeds": [],
            "seed_set": set(),
            # root enumeration cursors
            "root_i": 0,
            "rs_i": 0, "rs_j": 0,
            "fr_i": 0,
        }
        if kind == "minor":
            c["rs_snap"] = sorted(self._rs)
            c["fin_roots"] = sorted(
                r for (g, r) in self._fin if g == YOUNG and (YOUNG, r) not in self._fin_ran
            )
        else:
            c["rs_snap"] = []
            c["fin_roots"] = []
        self._cycle = c
        return c

    def _gray(self, c: dict, oid: tuple[int, int]) -> bool:
        """Grey an object. Returns True iff it newly joined the closure."""
        if oid in c["color"]:
            return False
        gen, raw = oid
        if raw not in self._lay(gen):
            return False
        c["color"][oid] = _GRAY
        c["queue"].append(oid)
        return True

    def _seed_add(self, c: dict, raw: int) -> None:
        if raw not in c["seed_set"]:
            c["seed_set"].add(raw)
            c["seeds"].append(raw)

    # ------------------------------------------------------------------ #
    # Mark
    # ------------------------------------------------------------------ #
    def _enum_one(self, c: dict, budget: _Budget) -> bool:
        """Enumerate the next root-zone examination. False if out of budget."""
        # Explicit roots (1 unit / root).
        if c["root_i"] < len(self._root_order):
            if not budget.take():
                return False
            name = self._root_order[c["root_i"]]
            c["root_i"] += 1
            oid = self._roots[name]
            if oid is None:
                return True
            if c["kind"] == "major":
                self._gray(c, oid)
            elif oid[0] == YOUNG:
                self._gray(c, oid)
                self._seed_add(c, oid[1])
            return True
        if c["kind"] != "minor":
            return True
        # Remembered set: every strong field of every logged old holder
        # (1 unit / field).
        rs = c["rs_snap"]
        if c["rs_i"] < len(rs):
            holder = rs[c["rs_i"]]
            if holder not in self._olay:
                c["rs_i"] += 1
                c["rs_j"] = 0
                return True
            ns, _ = self._olay[holder]
            if c["rs_j"] < ns:
                if not budget.take():
                    return False
                j = c["rs_j"]
                c["rs_j"] = j + 1
                v = self._old[holder + 2 + j]
                if v is not None and 0 <= v < self._yc:
                    self._gray(c, (YOUNG, v))
                    self._seed_add(c, v)
                return True
            c["rs_i"] += 1
            c["rs_j"] = 0
            return True
        # Young objects carrying an unrun finalizer (1 unit / object).
        fr = c["fin_roots"]
        if c["fr_i"] < len(fr):
            if not budget.take():
                return False
            raw = fr[c["fr_i"]]
            c["fr_i"] += 1
            self._gray(c, (YOUNG, raw))
            self._seed_add(c, raw)
        return True

    def _enum_done(self, c: dict) -> bool:
        if c["root_i"] < len(self._root_order):
            return False
        if c["kind"] == "minor":
            if c["rs_i"] < len(c["rs_snap"]) or c["fr_i"] < len(c["fin_roots"]):
                return False
        return True

    def _scan_edge(self, c: dict, budget: _Budget) -> bool:
        """Scan one strong edge of the head grey object. False = no budget.

        An object with no strong edges consumes exactly one unit (its
        visit) and immediately turns black; it must never read past the
        strong-field window into weak slots.
        """
        oid = c["queue"][0]
        gen, raw = oid
        ns, _ = self._lay(gen)[raw]
        j = c["cursor"].get(oid, 0)
        if not budget.take():
            return False
        if j < ns:
            v = self._space(gen, raw)[raw + 2 + j]
            if v is not None:
                t = self._split(v)
                if c["kind"] == "major" or t[0] == YOUNG:
                    self._gray(c, t)
            j += 1
            c["cursor"][oid] = j
        if j >= ns:
            c["color"][oid] = _BLACK
            c["queue"].pop(0)
            c["cursor"].pop(oid, None)
        return True

    def _step_mark(self, c: dict, budget: _Budget) -> None:
        while not budget.exhausted:
            if not self._enum_done(c):
                if not self._enum_one(c, budget):
                    return
                continue
            if c["queue"]:
                if not self._scan_edge(c, budget):
                    return
                continue
            self._finish_mark(c)
            return

    def _finish_mark(self, c: dict) -> None:
        if c["kind"] == "minor":
            self._eph_reset_pass(c, progressed=True)
            c["phase"] = "ephemeron"
        else:
            # Discover finalizer candidates lazily (one old object per
            # unit), in address order, at a marking fixpoint.  The snapshot
            # is only valid once a full discovery pass completes with the
            # grey queue empty (a barrier firing mid-pass invalidates it;
            # see _step_fdiscover).
            c["phase"] = "fdiscover"
            c["fdi"] = 0
            c["pending_fin"] = []

    # ------------------------------------------------------------------ #
    # Major: finalizer candidate discovery (budgeted)
    # ------------------------------------------------------------------ #
    def _step_fdiscover(self, c: dict, budget: _Budget) -> None:
        if c["queue"]:
            # Mutator action between steps added grey work: hand control
            # back to the mark phase so all edge scanning happens in one
            # place.  When the queue empties, mark re-enters discovery and
            # the pass restarts, so no classification is taken against a
            # non-fixpoint.
            c["fdi"] = 0
            c["pending_fin"] = []
            c["phase"] = "mark"
            return
        order = self._oorder
        while not budget.exhausted and c["fdi"] < len(order):
            if not budget.take():
                return
            raw = order[c["fdi"]]
            c["fdi"] += 1
            oid = (OLD, raw)
            if oid not in c["color"] and oid in self._fin and oid not in self._fin_ran:
                c["pending_fin"].append(raw)
        if c["fdi"] < len(order):
            return
        # Full address-order pass completed at a fixpoint: the candidate
        # snapshot is sealed.
        c["phase"] = "finalize"
        c["fin_i"] = 0
        c["fsub"] = "fixpoint"
        c["await"] = None
        self._eph_reset_pass(c, progressed=False)

    # ------------------------------------------------------------------ #
    # Ephemeron fixpoint (minor and major)
    # ------------------------------------------------------------------ #
    def _eph_reset_pass(self, c: dict, *, progressed: bool) -> None:
        c["eph_i"] = 0
        c["eph_pass_progressed"] = progressed

    def _entry_alive(self, c: dict, gen: int, raw: int) -> bool:
        if c["kind"] == "minor" and gen == OLD:
            return raw in self._olay
        return (gen, raw) in c["color"]

    def _eph_examine(self, c: dict, budget: _Budget) -> bool:
        """Process one ephemeron entry. False = keep going / no budget.

        Returns True only when a full sequence of passes has reached a
        stable fixpoint.  A new pass is forced exactly when examining an
        entry traces a *previously white* value (closure grew): a newly
        reachable object may be the table/key of an entry seen earlier in
        the same pass.  Termination is guaranteed because every forced
        pass adds at least one object to the closure.
        """
        i = c["eph_i"]
        if i >= len(self._eph_order):
            stable = not c["eph_pass_progressed"]
            if stable:
                return True
            self._eph_reset_pass(c, progressed=False)
            return False
        if not budget.take():
            return False
        eid = self._eph_order[i]
        c["eph_i"] = i + 1
        tg, ta, kg, ka, vg, va = self._eph[eid]
        if not self._entry_alive(c, tg, ta) or not self._entry_alive(c, kg, ka):
            return False
        # Both table and key reachable: the entry is active.  Only a newly
        # traced value can require another pass; an active entry with no
        # value, an old value during a minor, or an already-reachable value
        # changes nothing about the fixpoint.
        if va is None:
            return False
        if c["kind"] == "minor" and vg == OLD:
            return False
        val = (vg, va)
        if val in c["color"]:
            return False
        if self._gray(c, val):
            c["eph_pass_progressed"] = True
            if c["kind"] == "minor" and vg == YOUNG:
                self._seed_add(c, va)
        return False

    def _step_ephemeron(self, c: dict, budget: _Budget) -> None:
        while not budget.exhausted:
            if c["queue"]:
                if not self._scan_edge(c, budget):
                    return
                continue
            if self._eph_examine(c, budget):
                self._after_ephemeron(c)
                return

    def _after_ephemeron(self, c: dict) -> None:
        if c["kind"] == "minor":
            c["phase"] = "plan"
            c["plan_i"] = 0
            c["promote_set"] = set()
            c["res"] = {}             # young raw -> reserved destination raw
            c["plan_list"] = []       # young-address-ordered survivors
            c["promoted_new"] = []    # ascending old destination raws
            c["young_new"] = []       # ascending young destination raws
            c["young_run"] = 0
            c["survivor_slots"] = 0
            c["sim_holes"] = list(self._old_free)
            c["sim_hwm"] = self._ohwm
        else:
            self._enter_weakclear(c)

    # ------------------------------------------------------------------ #
    # Major: finalizers with resurrection x ephemeron intersection
    # ------------------------------------------------------------------ #
    def _step_finalize(self, c: dict, budget: _Budget, events: list) -> None:
        while not budget.exhausted:
            if c["queue"]:
                if not self._scan_edge(c, budget):
                    return
                continue
            if c["fsub"] == "fire":
                if c["fin_i"] >= len(c["pending_fin"]):
                    # One last fixpoint was guaranteed before entering
                    # "fire" with an empty queue: closure is final.
                    self._enter_weakclear(c)
                    return
                raw = c["pending_fin"][c["fin_i"]]
                c["fin_i"] += 1
                oid = (OLD, raw)
                if oid in c["color"]:
                    # Became reachable through an earlier resurrection /
                    # ephemeron cascade: defer, without consuming the event.
                    continue
                if not budget.take():
                    c["fin_i"] -= 1
                    return
                cb, token = self._fin[oid]
                self._fin_ran.add(oid)
                self._in_callback = True
                try:
                    cb(self, Handle(self._join(OLD, raw), self._epoch))
                finally:
                    self._in_callback = False
                c["await"] = (oid, token)
                c["fsub"] = "fixpoint"
                self._eph_reset_pass(c, progressed=True)
                continue
            # fsub == "fixpoint": ephemeron passes interleaved with draining.
            if self._eph_examine(c, budget):
                awaited = c.get("await")
                if awaited is not None:
                    oid, token = awaited
                    c["await"] = None
                    events.append((OLD, self._join(*oid), token, oid in c["color"]))
                    c["finalized"].append((OLD, self._join(*oid), token, oid in c["color"]))
                c["fsub"] = "fire"

    def _enter_weakclear(self, c: dict) -> None:
        c["phase"] = "weakclear"
        c["wh_src"] = 0   # 0 young holders, 1 old holders
        c["wh_i"] = 0
        c["wh_part"] = 0  # within a holder: 0 weak part, 1 strong part
        c["wh_j"] = 0
        c["wh_visited"] = False
        c["ephprune_i"] = 0

    def _is_dead_old(self, c: dict, gen: int, raw: int) -> bool:
        return gen == OLD and (OLD, raw) not in c["color"]

    def _step_weakclear(self, c: dict, budget: _Budget) -> None:
        # Per holder, one visit unit plus one unit per examined edge slot,
        # resumable across steps:
        #   * young holders: weak edges to dead old targets are cleared and
        #     reported; strong edges to dead old targets (which can only
        #     occur on unreachable young garbage) are nulled silently so no
        #     published slot ever names a reclaimed address;
        #   * live old holders: weak edges to dead old targets cleared and
        #     reported (a live old holder cannot strongly name a dead one);
        #   * dead old holders: skipped (their slots vanish in sweep).
        while c["wh_src"] < 2:
            gen = YOUNG if c["wh_src"] == 0 else OLD
            order = self._yorder if gen == YOUNG else self._oorder
            space = self._yspace if gen == YOUNG else self._old
            i = c["wh_i"]
            if i >= len(order):
                c["wh_src"] += 1
                c["wh_i"] = 0
                c["wh_part"] = 0
                c["wh_j"] = 0
                c["wh_visited"] = False
                continue
            if not c["wh_visited"]:
                if not budget.take():
                    return
                c["wh_visited"] = True
            raw = order[i]
            holder_alive = (gen == YOUNG) or ((OLD, raw) in c["color"])
            ns, nw = self._lay(gen)[raw]
            done = not holder_alive
            if not done:
                # part 0: weak fields; part 1 (young only): strong fields.
                part = c["wh_part"]
                j = c["wh_j"]
                n_fields = nw if part == 0 else ns
                while j < n_fields:
                    if not budget.take():
                        c["wh_i"], c["wh_part"], c["wh_j"] = i, part, j
                        return
                    if part == 0:
                        slot = raw + 2 + ns + j
                        v = space[slot]
                        j += 1
                        if v is not None:
                            tg, traw = self._split(v)
                            if self._is_dead_old(c, tg, traw):
                                space[slot] = None
                                c["cleared_weak"].append((self._join(gen, raw), j - 1))
                    else:
                        slot = raw + 2 + j
                        v = space[slot]
                        j += 1
                        if v is not None:
                            tg, traw = self._split(v)
                            if tg == OLD and (OLD, traw) not in c["color"]:
                                space[slot] = None
                if part == 0 and gen == YOUNG:
                    c["wh_part"], c["wh_j"] = 1, 0
                    continue
            c["wh_i"] = i + 1
            c["wh_part"] = 0
            c["wh_j"] = 0
            c["wh_visited"] = False
        # Prune ephemeron entries (1 unit / entry, registration order).
        i = c["ephprune_i"]
        while i < len(self._eph_order):
            if not budget.take():
                c["ephprune_i"] = i
                return
            eid = self._eph_order[i]
            i += 1
            tg, ta, kg, ka, vg, va = self._eph[eid]

            def missing(gg: int, rr: int) -> bool:
                # Reclaimed by this major sweep, or a stale address that
                # names no current object in either generation.
                if self._is_dead_old(c, gg, rr):
                    return True
                return rr not in self._lay(gg)

            if missing(tg, ta) or missing(kg, ka):
                c["cleared_eph"].append(eid)
                del self._eph[eid]
                self._eph_order.remove(eid)
                i -= 1
            elif va is not None and missing(vg, va):
                self._eph[eid] = (tg, ta, kg, ka, 0, None)
        c["ephprune_i"] = i
        # Only advance once every entry was examined within budget.
        c["phase"] = "sweep"
        c["sw_i"] = 0
        c["sw_survivors"] = []

    def _step_sweep(self, c: dict, budget: _Budget) -> bool:
        order = self._oorder
        i = c["sw_i"]
        survivors = c["sw_survivors"]
        while not budget.exhausted and i < len(order):
            if not budget.take():
                c["sw_i"] = i
                return False
            raw = order[i]
            i += 1
            if (OLD, raw) in c["color"]:
                survivors.append(raw)  # order is ascending
                continue
            ns, nw = self._olay.pop(raw)
            size = 2 + ns + nw
            self._insert_old_hole(raw, size)
            while self._old_free and self._old_free[-1][0] + self._old_free[-1][1] == self._ohwm:
                r, s = self._old_free.pop()
                self._ohwm = r
            self._oocc -= size
            self._fin.pop((OLD, raw), None)
            self._fin_ran.discard((OLD, raw))
            self._rs.discard(raw)
            self._wrs.discard(raw)
        c["sw_i"] = i
        if i < len(order):
            return False
        # Built incrementally in ascending order during the budgeted loop;
        # no final O(n) re-filter is needed.
        self._oorder = survivors
        return True

    # ------------------------------------------------------------------ #
    # Minor: promotion planning (budgeted, moves nothing)
    # ------------------------------------------------------------------ #
    def _step_plan(self, c: dict, budget: _Budget) -> None:
        # Closure growth during planning sends the cycle back to marking
        # (IU barrier greys targets); mark/ephemeron then re-run and plan
        # starts afresh, so classifications are never taken mid-growth.
        if c["queue"]:
            c["phase"] = "mark"
            return
        order = self._yorder
        i = c["plan_i"]
        while not budget.exhausted and i < len(order):
            if not budget.take():
                c["plan_i"] = i
                return
            raw = order[i]
            i += 1
            if (YOUNG, raw) not in c["color"]:
                continue
            ns, nw = self._ylay[raw]
            size = 2 + ns + nw
            if raw in self._force_promote or self._age.get(raw, 0) + 1 >= self._promote_age:
                nr = self._plan_reserve(c, raw, size)
                c["promote_set"].add(raw)
                c["promoted_new"].append(nr)
            else:
                nr = c["young_run"]
                c["young_run"] = nr + size
                c["survivor_slots"] += size
                c["young_new"].append(nr)
            c["res"][raw] = nr
            c["plan_list"].append(raw)
        c["plan_i"] = i
        if i < len(order):
            return
        self._init_evac(c)

    def _plan_reserve(self, c: dict, raw: int, size: int) -> int:
        """First-fit old placement simulated against a scratch free list."""
        holes = c["sim_holes"]
        for k, (hr, hs) in enumerate(holes):
            if hs >= size:
                del holes[k]
                if hs > size:
                    insort(holes, (hr + size, hs - size))
                return hr
        hr = c["sim_hwm"]
        if hr + size > self._oc:
            raise HeapCapacityError(
                f"minor promotion needs {size} slot(s) at old offset {hr} but the "
                "old generation cannot fit the promotion set; run a major first"
            )
        c["sim_hwm"] = hr + size
        return hr

    # ------------------------------------------------------------------ #
    # Minor: evacuation (per-slot charged copy, then edge-fix stages)
    # ------------------------------------------------------------------ #
    def _init_evac(self, c: dict) -> None:
        c["phase"] = "evac"
        c["to"] = self._young[1 - self._yidx]
        c["fwd"] = {}                 # young raw -> (gen, new raw)
        c["copied"] = set()           # young raws physically copied+published
        c["new_lay"] = {}
        c["new_age"] = {}
        c["new_order"] = []           # ascending young destination raws
        c["tail"] = c["survivor_slots"]
        c["relocated"] = {}
        c["promoted_map"] = {}
        # Remembered-set membership of promoted holders discovered while
        # their survivor edges are forwarded (old -> young after the copy).
        c["prom_rs"] = set()
        c["prom_wrs"] = set()
        # Promotions land first-fit, so destination order differs from the
        # source order used in planning: process them by destination address.
        c["promoted_new"].sort()
        c["stage"] = "copy"
        c["ev_i"] = 0

    def _publish_copy(self, c: dict, raw: int, *, budget: _Budget | None) -> bool:
        """Copy one survivor into its reserved window, atomically.

        One evacuation is a single work unit: the object's fixed-size
        slot window moves with it (bounded by that one object's declared
        layout, never a heap traversal).  All *edge* examination happens
        separately, one edge per unit, in the later fix stages, which is
        what the scan budget bounds.  ``budget is None`` is the mutator
        copy-barrier path.  Returns False only when the step has no unit
        left; every budget >= 1 therefore makes progress.
        """
        if raw in c["copied"]:
            return True
        if budget is not None and not budget.take():
            return False
        ns, nw = self._ylay[raw]
        size = 2 + ns + nw
        nr = c["res"][raw]
        if raw in c["promote_set"]:
            gen = OLD
            dst = self._old
            self._commit_old_reservation(nr, size)
            self._oocc += size
            self._olay[nr] = (ns, nw)
            insort(self._oorder, nr)
            c["promoted_map"][raw] = self._yc + nr
            self._carry_finalizer((YOUNG, raw), (OLD, nr))
        else:
            gen = YOUNG
            dst = c["to"]
            c["new_lay"][nr] = (ns, nw)
            c["new_age"][nr] = self._age.get(raw, 0) + 1
            c["new_order"].append(nr)
            c["relocated"][raw] = nr
            self._carry_finalizer((YOUNG, raw), (YOUNG, nr))
        dst[nr:nr + size] = self._yspace[raw:raw + size]
        c["fwd"][raw] = (gen, nr)
        c["copied"].add(raw)
        return True

    def _commit_old_reservation(self, nr: int, size: int) -> None:
        """Mirror the plan's simulated reservation on the real free list."""
        for k, (hr, hs) in enumerate(self._old_free):
            if hr == nr:
                del self._old_free[k]
                if hs > size:
                    self._insert_old_hole(nr + size, hs - size)
                return
        if nr == self._ohwm:
            self._ohwm = nr + size
            return
        raise HeapCapacityError(  # pragma: no cover - plan/reservation invariant
            f"internal error: promotion placement {nr} not available"
        )

    def _carry_finalizer(self, old_id, new_id) -> None:
        if old_id in self._fin:
            self._fin[new_id] = self._fin.pop(old_id)
        if old_id in self._fin_ran:
            self._fin_ran.discard(old_id)
            self._fin_ran.add(new_id)

    def _step_evac(self, c: dict, budget: _Budget) -> bool:
        if c["stage"] == "copy":
            plan_list = c["plan_list"]
            while c["ev_i"] < len(plan_list):
                raw = plan_list[c["ev_i"]]
                if raw in c["copied"]:
                    c["ev_i"] += 1
                    continue
                if not self._publish_copy(c, raw, budget=budget):
                    return False
                c["ev_i"] += 1
                if budget.exhausted:
                    return False
            c["stage"] = "strong"
            c["fx_src"] = 0
            c["fx_i"] = c["fx_j"] = 0
        if c["stage"] == "strong":
            if not self._fix_survivor_edges(c, budget, weak=False):
                return False
            c["stage"] = "weak"
            c["fx_src"] = 0
            c["fx_i"] = c["fx_j"] = 0
        if c["stage"] == "weak":
            if not self._fix_survivor_edges(c, budget, weak=True):
                return False
            c["stage"] = "ephfix"
            c["ef_i"] = 0
        if c["stage"] == "ephfix":
            i = c["ef_i"]
            while i < len(self._eph_order):
                if not budget.take():
                    c["ef_i"] = i
                    return False
                eid = self._eph_order[i]
                before = len(self._eph_order)
                self._minor_fix_ephemeron(c, eid)
                # A removal shrinks the order and shifts the next entry to
                # the same index; only advance when nothing was removed.
                if len(self._eph_order) == before:
                    i += 1
            c["ef_i"] = i
            c["stage"] = "rootsfix"
            c["rf_i"] = 0
        if c["stage"] == "rootsfix":
            i = c["rf_i"]
            while not budget.exhausted and i < len(self._root_order):
                if not budget.take():
                    c["rf_i"] = i
                    return False
                name = self._root_order[i]
                i += 1
                oid = self._roots[name]
                if oid is not None and oid[0] == YOUNG and oid[1] in c["fwd"]:
                    self._roots[name] = c["fwd"][oid[1]]
            c["rf_i"] = i
            if i < len(self._root_order):
                return False
            c["stage"] = "rsfix"
            self._init_crossfix(c, weak=False)
        if c["stage"] == "rsfix":
            if not self._fix_cross_edges(c, budget, weak=False):
                return False
            c["stage"] = "wrsfix"
            self._init_crossfix(c, weak=True)
        if c["stage"] == "wrsfix":
            if not self._fix_cross_edges(c, budget, weak=True):
                return False
        return True

    def _fix_survivor_edges(self, c: dict, budget: _Budget, *, weak: bool) -> bool:
        while c["fx_src"] < 2:
            gen = YOUNG if c["fx_src"] == 0 else OLD
            raws = c["young_new"] if gen == YOUNG else c["promoted_new"]
            space = c["to"] if gen == YOUNG else self._old
            i, j = c["fx_i"], c["fx_j"]
            while not budget.exhausted and i < len(raws):
                raw = raws[i]
                ns, nw = self._olay[raw] if gen == OLD else c["new_lay"][raw]
                n_fields = nw if weak else ns
                if j >= n_fields:
                    i += 1
                    j = 0
                    c["fx_i"], c["fx_j"] = i, j
                    continue
                if not budget.take():
                    c["fx_i"], c["fx_j"] = i, j
                    return False
                base = raw + 2 + (ns if weak else 0)
                slot = base + j
                j += 1
                v = space[slot]
                if v is None:
                    pass
                else:
                    tg, traw = self._split(v)
                    if tg == YOUNG:
                        if traw in c["fwd"]:
                            fg, fr = c["fwd"][traw]
                            space[slot] = self._join(fg, fr)
                            if gen == OLD and fg == YOUNG:
                                # Promoted old holder still points young.
                                (c["prom_wrs"] if weak else c["prom_rs"]).add(raw)
                        elif weak:
                            space[slot] = None
                            c["cleared_weak"].append((self._join(gen, raw), j - 1))
                        # Strong edge to a non-survivor cannot occur: the
                        # target would have been in the minor closure.
                c["fx_i"], c["fx_j"] = i, j
            c["fx_i"], c["fx_j"] = i, j
            if i < len(raws):
                return False
            c["fx_src"] += 1
            c["fx_i"] = c["fx_j"] = 0
        return True

    def _minor_fix_ephemeron(self, c: dict, eid: int) -> None:
        tg, ta, kg, ka, vg, va = self._eph[eid]

        def exists(gg: int, rr: int) -> bool:
            if gg == YOUNG:
                return rr in c["fwd"]
            return rr in self._olay

        # Drop the entry if its table or key names a non-existent object
        # (a young non-survivor, or a stale address from an earlier cycle).
        if not exists(tg, ta) or not exists(kg, ka):
            c["cleared_eph"].append(eid)
            del self._eph[eid]
            self._eph_order.remove(eid)
            return
        if tg == YOUNG:
            tg, ta = c["fwd"][ta]
        if kg == YOUNG:
            kg, ka = c["fwd"][ka]
        if va is not None:
            if not exists(vg, va):
                vg, va = 0, None
            elif vg == YOUNG:
                vg, va = c["fwd"][va]
        self._eph[eid] = (tg, ta, kg, ka, vg, va)

    def _init_crossfix(self, c: dict, *, weak: bool) -> None:
        # Forward edges of old holders that existed *before* this minor.
        # Promoted holders are new old objects whose edges were already
        # forwarded during the survivor fix stages (and whose remembered
        # membership was recorded there); processing them again would
        # misread post-copy young addresses as pre-copy ones.
        promoted = set(c["promoted_new"])
        current = self._wrs if weak else self._rs
        holders = sorted(h for h in current if h not in promoted)
        c["cf_weak"] = weak
        c["cf_holders"] = holders
        c["cf_i"] = 0
        c["cf_j"] = 0
        c["cf_keep"] = set()

    def _fix_cross_edges(self, c: dict, budget: _Budget, *, weak: bool) -> bool:
        holders = c["cf_holders"]
        keep = c["cf_keep"]
        i, j = c["cf_i"], c["cf_j"]
        while i < len(holders):
            raw = holders[i]
            if raw not in self._olay:
                i += 1
                j = 0
                c["cf_i"], c["cf_j"] = i, j
                continue
            ns, nw = self._olay[raw]
            n_fields = nw if weak else ns
            while j < n_fields:
                if budget.exhausted:
                    c["cf_i"], c["cf_j"] = i, j
                    return False
                budget.take()
                base = raw + 2 + (ns if weak else 0)
                slot = base + j
                j += 1
                v = self._old[slot]
                if v is None:
                    continue
                tg, traw = self._split(v)
                if tg != YOUNG:
                    continue
                if traw in c["fwd"]:
                    fg, fr = c["fwd"][traw]
                    self._old[slot] = self._join(fg, fr)
                    if fg == YOUNG:
                        # Edge still ends in the young generation: the
                        # holder stays in the remembered set.
                        keep.add(raw)
                elif weak:
                    self._old[slot] = None
                    c["cleared_weak"].append((self._yc + raw, j - 1))
                # Strong edge to a non-surviving young object cannot occur:
                # the target would have been in the minor closure.
            i += 1
            j = 0
            c["cf_i"], c["cf_j"] = i, j
        # All pre-existing holders examined within budget.  The new
        # remembered membership is the union of:
        #   * examined holders whose edge still ends in young (``keep``);
        #   * promoted holders recorded during survivor edge fixing
        #     (``prom_rs`` / ``prom_wrs``).
        # No final re-scan is needed: both sets were built incrementally.
        target = self._wrs if weak else self._rs
        prom = c["prom_wrs"] if weak else c["prom_rs"]
        target.clear()
        target |= keep
        target |= prom
        return True

    # ------------------------------------------------------------------ #
    # Commit
    # ------------------------------------------------------------------ #
    def _commit(self, c: dict, fin_now, scan_budget, scanned) -> StepResult:
        kind = c["kind"]
        if kind == "minor":
            relocated, promoted, ly, lo = self._commit_minor(c)
            cleared_eph = tuple(c["cleared_eph"])
        else:
            relocated, promoted = {}, {}
            ly, lo = self._yalloc, self._oocc
            cleared_eph = tuple(c["cleared_eph"])
        self._epoch += 1
        return StepResult(
            kind=kind,
            phase="done",
            completed=True,
            budget=scan_budget,
            scanned=scanned,
            overscan=0,
            gray_remaining=0,
            steps_in_cycle=c["steps"],
            barrier_records=c["barrier"],
            relocated=relocated,
            promoted=promoted,
            live_young_slots=ly,
            live_old_slots=lo,
            cleared_weak_edges=tuple(sorted(c["cleared_weak"])),
            cleared_ephemerons=cleared_eph,
            finalized_now=tuple(fin_now),
            finalized=tuple(c["finalized"]),
        )

    def _commit_minor(self, c: dict) -> tuple[dict, dict, int, int]:
        self._yidx = 1 - self._yidx
        self._yspace = self._young[self._yidx]
        self._ylay = c["new_lay"]
        self._age = c["new_age"]
        # Destination layout is kept ascending as copies/tail allocs land;
        # commit does no examination or sorting.
        self._yorder = list(c["new_order"])
        self._yalloc = c["tail"]
        self._force_promote = set()
        relocated = {
            raw: self._join(*c["fwd"][raw])
            for raw in c["fwd"]
            if c["fwd"][raw][0] == YOUNG
        }
        promoted = dict(c["promoted_map"])
        return relocated, promoted, self._yalloc, self._oocc
