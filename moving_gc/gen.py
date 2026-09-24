"""Generational GC engine: young semi-space copying + old mark-sweep.

Extends the legacy v1 semi-space kernel (:mod:`moving_gc.core`) with:

* **two generations** -- a young generation collected by semi-space copying
  (with configurable promotion age) and an old, non-moving generation
  collected by mark-sweep;
* **incremental update (IU) write barriers** plus an *exact* remembered set
  of old objects that hold *any* old -> young slot (strong, weak, ephemeron
  key or value);
* **ephemeron tables** -- a value joins the closure only while its table
  and key are both reachable; entries are iterated to a fixed point *before*
  ordinary weak edges are cleared;
* **explicit finalization events** -- each object identity is finalized at
  most once; a finalizer body may re-attach the object to a root
  (resurrection); the fixed order is mark -> finalize -> remark to a fixed
  point -> weak/ephemeron clearing -> commit;
* **finite, budgeted incremental steps** -- every reclaim step states an
  edge-scan budget; at most one edge is scanned past it.  A single call
  never scans the whole heap at once.

Scope rules for a *minor* collection (documented, deterministic):

* only young objects are scanned; old objects named by a root, by a
  young strong edge, or by the remembered set are treated as atomic black
  roots;
* the remembered set is conservative for *dead* old holders: a young target
  named only from an old object that is itself garbage may be retained
  until the next major collection (standard RS conservatism);
* ordinary weak edges into the old generation, and ephemeron entries whose
  key is in the old generation, are never cleared by a minor collection
  (early clearing would be irreversible); old-keyed ephemerons activate
  conservatively when their table is reachable.  A major collection
  resolves both exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .core import Handle, _check_int
from .errors import GCError, HeapCapacityError, StaleHandleError

__all__ = [
    "GenerationalHeap",
    "GenCollectionResult",
    "CycleInfo",
    "FinalizationEvent",
    "Relocation",
    "WeakClear",
    "EphemeronClear",
    "StepReport",
    "OLD",
    "YOUNG",
    "DEFAULT_SCAN_BUDGET",
]

OLD = 0
YOUNG = 1
_GEN_NAME = {OLD: "old", YOUNG: "young"}
DEFAULT_SCAN_BUDGET = 64


def _enc(addr: int, gen: int) -> int:
    """Encode an edge target ``(address, generation)`` as one slot integer."""
    return (addr << 1) | gen


def _dec(ref: int) -> tuple[int, int]:
    """Decode a slot integer to ``(address, generation)``."""
    return ref >> 1, ref & 1


def _key(ref: int) -> tuple[int, int]:
    """Decode a slot integer to a cycle key ``(generation, address)``."""
    return ref & 1, ref >> 1


def _href(ref: int, epoch: int) -> Handle:
    addr, gen = _dec(ref)
    return Handle(addr, epoch, gen)


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Relocation:
    """One moved object: ``(from_gen, from_addr, to_gen, to_addr)``."""

    from_gen: int
    from_addr: int
    to_gen: int
    to_addr: int


@dataclass(frozen=True)
class WeakClear:
    """A cleared ordinary weak edge, in post-collection coordinates."""

    gen: int
    holder: int
    index: int


@dataclass(frozen=True)
class EphemeronClear:
    """A cleared ephemeron entry whose key died (table survived)."""

    gen: int
    holder: int
    index: int


@dataclass(frozen=True)
class GenCollectionResult:
    """Outcome of one completed minor or major collection.

    All tuples are in a fixed deterministic order so equal inputs always
    give equal results.
    """

    kind: str
    relocated: tuple[Relocation, ...]
    promoted: int
    live_old_slots: int
    live_young_slots: int
    roots_preserved: int
    cleared_weak_edges: tuple[WeakClear, ...]
    cleared_ephemerons: tuple[EphemeronClear, ...]
    # (gen, holder_post_addr, idx): entry whose key was reachable, so its
    # value was required to join the closure.
    resolved_ephemerons: tuple[tuple[int, int, int], ...]
    # Strict subset of resolved entries whose value was not reachable
    # otherwise and was pulled in by the ephemeron.
    rescued_values: tuple[tuple[int, int, int], ...]
    finalized: tuple[int, ...]   # event ids whose body ran
    reclaimed_objects: int
    edges_scanned: int           # cumulative edge scans over the whole cycle


@dataclass(frozen=True)
class FinalizationEvent:
    """A staged finalizer event; inspectable between finalization steps."""

    event_id: int
    gen: int
    address: int
    object_id: int


@dataclass(frozen=True)
class CycleInfo:
    """Read-only snapshot of an in-progress incremental cycle."""

    kind: str
    phase: str
    marked: tuple[tuple[int, int], ...]
    gray: tuple[tuple[int, int], ...]
    ephemeron_pending: tuple[tuple[tuple[int, int], int], ...]
    finalizer_pending: tuple[FinalizationEvent, ...]
    edges_scanned: int


@dataclass(frozen=True)
class StepReport:
    """Result of one incremental step.

    ``complete`` only accompanies a finished commit (``result`` set); it is
    a progress flag, never a certificate -- callers verify against
    :attr:`phase`, :attr:`edges_scanned` and the post-collection heap.
    """

    phase: str
    edges_scanned: int
    budget: int
    budget_exhausted: bool
    complete: bool
    result: GenCollectionResult | None = None


# --------------------------------------------------------------------------- #
# Internal object record
# --------------------------------------------------------------------------- #
@dataclass
class _Obj:
    oid: int
    gen: int
    addr: int
    n_strong: int
    n_weak: int
    n_eph: int
    size: int
    age: int = 0
    finalizer: int | None = None
    finalized: bool = False


class GenerationalHeap:
    """Two-generation incremental heap (see module docstring)."""

    def __init__(
        self,
        young_capacity: int = 512,
        old_capacity: int = 512,
        *,
        promotion_age: int = 1,
    ) -> None:
        _check_int(young_capacity, "young_capacity")
        _check_int(old_capacity, "old_capacity")
        _check_int(promotion_age, "promotion_age")
        for name, v in (("young_capacity", young_capacity),
                        ("old_capacity", old_capacity)):
            if not (1 <= v <= 4096):
                raise ValueError(f"{name} must be in [1, 4096], got {v}")
        if promotion_age < 1:
            raise ValueError(f"promotion_age must be >= 1, got {promotion_age}")
        self._ycap = young_capacity
        self._ocap = old_capacity
        self._promotion_age = promotion_age

        self._young: list[list] = [
            [None] * young_capacity,
            [None] * young_capacity,
        ]
        self._yactive = 0
        self._yptr = 0
        self._old: list = [None] * old_capacity
        self._free: list[tuple[int, int]] = [(0, old_capacity)]

        # _obj[OLD|YOUNG]: address -> _Obj
        self._obj: list[dict[int, _Obj]] = [{}, {}]
        self._roots: dict[str, tuple[int, int] | None] = {}
        self._root_order: list[str] = []
        self._epoch = 0
        self._next_oid = 0
        self._finalized_oids: set[int] = set()
        self._finalized_events: set[int] = set()

        # Exact remembered set: old addresses naming a young strong target
        # or a young ephemeron key.
        self._remembered: set[int] = set()

        # Incremental cycle state; None when idle.
        self._cyc: dict | None = None

    # ================================================================== #
    # Introspection
    # ================================================================== #
    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def young_capacity(self) -> int:
        return self._ycap

    @property
    def old_capacity(self) -> int:
        return self._ocap

    @property
    def promotion_age(self) -> int:
        return self._promotion_age

    def used_slots(self, generation: int | None = None) -> int:
        if generation is None:
            return self._yptr + (self._ocap - sum(s for _, s in self._free))
        _check_gen(generation)
        if generation == YOUNG:
            return self._yptr
        return self._ocap - sum(s for _, s in self._free)

    def free_slots(self, generation: int) -> int:
        _check_gen(generation)
        return self._ycap - self._yptr if generation == YOUNG \
            else sum(s for _, s in self._free)

    def objects(self, generation: int | None = None) -> tuple[tuple[int, int], ...]:
        out: list[tuple[int, int]] = []
        gens = (OLD, YOUNG) if generation is None else (generation,)
        for g in gens:
            out.extend((g, a) for a in self._obj[g])
        return tuple(sorted(out))

    def remembered_set(self) -> frozenset[int]:
        return frozenset(self._remembered)

    def generation_of(self, h: Handle) -> int:
        return self._resolve(h)[0]

    def cycle_phase(self) -> str:
        return "idle" if self._cyc is None else self._cyc["phase"]

    def cycle_info(self) -> CycleInfo | None:
        if self._cyc is None:
            return None
        c = self._cyc
        return CycleInfo(
            kind=c["kind"],
            phase=c["phase"],
            marked=tuple(sorted(c["mark"])),
            gray=tuple(k for k, _ in c["gray"]),
            ephemeron_pending=tuple(sorted(c["eph_pending"])),
            finalizer_pending=tuple(c["final_q"]),
            edges_scanned=c["scanned"],
        )

    # ================================================================== #
    # Allocation
    # ================================================================== #
    def allocate(
        self,
        payload: int = 0,
        *,
        n_strong: int = 0,
        n_weak: int = 0,
        n_ephemeron: int = 0,
        generation: int = YOUNG,
    ) -> Handle:
        """Allocate one object in ``generation`` (default young).

        Young exhaustion triggers at most one minor then, if needed, one
        major collection.  Old exhaustion triggers at most one major
        collection.  No half object is ever left behind.
        """
        _check_int(payload, "payload")
        _check_int(n_strong, "n_strong")
        _check_int(n_weak, "n_weak")
        _check_int(n_ephemeron, "n_ephemeron")
        _check_gen(generation)
        if min(n_strong, n_weak, n_ephemeron) < 0:
            raise ValueError("edge slot counts must be >= 0")
        size = 2 + n_strong + n_weak + 2 * n_ephemeron

        if self._cyc is not None and (
            (generation == YOUNG and self._yptr + size > self._ycap)
            or (generation == OLD and self._old_fit(size) is None)
        ):
            raise GCError(
                "cannot auto-collect while a collection cycle is in progress; "
                "finish the current cycle first")

        if generation == YOUNG:
            if size > self._ycap:
                raise HeapCapacityError(
                    f"object size {size} exceeds young semi-space capacity {self._ycap}")
            if self._yptr + size > self._ycap:
                self.run_collection("minor", DEFAULT_SCAN_BUDGET, finalizer=None)
                if self._yptr + size > self._ycap:
                    self.run_collection("major", DEFAULT_SCAN_BUDGET, finalizer=None)
                if self._yptr + size > self._ycap:
                    raise HeapCapacityError(
                        f"out of young capacity: need {size} slot(s), "
                        f"{self._ycap - self._yptr} free after minor+major collection")
            return self._alloc_young(payload, n_strong, n_weak, n_ephemeron, size)

        if size > self._ocap:
            raise HeapCapacityError(
                f"object size {size} exceeds old generation capacity {self._ocap}")
        addr = self._old_fit(size)
        if addr is None:
            self.run_collection("major", DEFAULT_SCAN_BUDGET, finalizer=None)
            addr = self._old_fit(size)
            if addr is None:
                raise HeapCapacityError(
                    f"out of old capacity: need {size} slot(s), "
                    f"{sum(s for _, s in self._free)} free after major collection")
        self._write_old(addr, payload, n_strong, n_weak, n_ephemeron, size)
        h = Handle(addr, self._epoch, OLD)
        self._born_during_cycle(h)
        return h

    def _alloc_young(self, payload, ns, nw, ne, size) -> Handle:
        addr = self._yptr
        space = self._young[self._yactive]
        for i in range(size):
            space[addr + i] = None
        space[addr + 1] = payload
        self._yptr = addr + size
        self._init_object(YOUNG, addr, ns, nw, ne, size)
        h = Handle(addr, self._epoch, YOUNG)
        self._born_during_cycle(h)
        return h

    def _write_old(self, addr, payload, ns, nw, ne, size) -> None:
        for i in range(size):
            self._old[addr + i] = None
        self._old[addr + 1] = payload
        self._init_object(OLD, addr, ns, nw, ne, size)

    def _init_object(self, gen, addr, ns, nw, ne, size) -> _Obj:
        obj = _Obj(self._next_oid, gen, addr, ns, nw, ne, size)
        self._next_oid += 1
        self._obj[gen][addr] = obj
        return obj

    def _old_fit(self, size: int) -> int | None:
        """First-fit over the sorted disjoint free list; consumes the hole."""
        for i, (start, hole) in enumerate(self._free):
            if hole >= size:
                if hole == size:
                    self._free.pop(i)
                else:
                    self._free[i] = (start + size, hole - size)
                return start
        return None

    def _space_of(self, gen: int):
        return self._old if gen == OLD else self._young[self._yactive]

    # ================================================================== #
    # Roots
    # ================================================================== #
    def add_root(self, name: str, target: Handle) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("root name must be a non-empty str")
        key = self._resolve(target)
        if name not in self._roots:
            self._root_order.append(name)
        self._roots[name] = key
        self._barrier_root(key)

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
        key = self._roots[name]
        return None if key is None else Handle(key[1], self._epoch, key[0])

    def roots(self) -> dict[str, Handle | None]:
        return {
            name: (None if key is None else Handle(key[1], self._epoch, key[0]))
            for name, key in self._roots.items()
        }

    # ================================================================== #
    # Handles / fields
    # ================================================================== #
    def handle(self, address: int, generation: int = YOUNG) -> Handle:
        _check_int(address, "address")
        _check_gen(generation)
        if address not in self._obj[generation]:
            raise StaleHandleError(
                f"no live object at gen {_GEN_NAME[generation]} address {address}")
        return Handle(address, self._epoch, generation)

    def _resolve(self, target: Handle) -> tuple[int, int]:
        if not isinstance(target, Handle):
            raise TypeError(f"expected Handle, got {type(target).__name__}")
        gen = target.generation
        if gen not in (OLD, YOUNG):
            raise StaleHandleError(
                f"handle does not name a heap generation (generation={gen})")
        if target.epoch != self._epoch:
            raise StaleHandleError(
                f"stale handle: epoch {target.epoch} != current epoch {self._epoch} "
                f"(gen {_GEN_NAME[gen]} address {target.address})")
        if target.address not in self._obj[gen]:
            raise StaleHandleError(
                f"no live object at gen {_GEN_NAME[gen]} address {target.address}")
        return gen, target.address

    def payload(self, h: Handle) -> int:
        gen, addr = self._resolve(h)
        return self._space_of(gen)[addr + 1]

    def set_payload(self, h: Handle, value: int) -> None:
        _check_int(value, "value")
        gen, addr = self._resolve(h)
        self._space_of(gen)[addr + 1] = value

    def object_shape(self, h: Handle) -> tuple[int, int, int]:
        gen, addr = self._resolve(h)
        o = self._obj[gen][addr]
        return o.n_strong, o.n_weak, o.n_eph

    def _idx(self, h, index, kind, count_attr):
        gen, addr = self._resolve(h)
        o = self._obj[gen][addr]
        n = getattr(o, count_attr)
        _check_int(index, "index")
        if not 0 <= index < n:
            raise IndexError(
                f"{kind} index {index} out of range for object with {n} {kind} slot(s)")
        return gen, addr, o

    def get_strong(self, h: Handle, index: int) -> Handle | None:
        gen, addr, _ = self._idx(h, index, "strong", "n_strong")
        ref = self._space_of(gen)[addr + 2 + index]
        return None if ref is None else _href(ref, self._epoch)

    def set_strong(self, h: Handle, index: int, ref: Handle | None) -> None:
        gen, addr, _ = self._idx(h, index, "strong", "n_strong")
        new = None if ref is None else self._resolve(ref)
        self._space_of(gen)[addr + 2 + index] = (
            None if new is None else _enc(new[1], new[0]))
        self._barrier_strong((gen, addr), new)
        if gen == OLD:
            self._recompute_remembered(addr)

    def get_weak(self, h: Handle, index: int) -> Handle | None:
        gen, addr, o = self._idx(h, index, "weak", "n_weak")
        ref = self._space_of(gen)[addr + 2 + o.n_strong + index]
        return None if ref is None else _href(ref, self._epoch)

    def set_weak(self, h: Handle, index: int, ref: Handle | None) -> None:
        gen, addr, o = self._idx(h, index, "weak", "n_weak")
        new = None if ref is None else self._resolve(ref)
        self._space_of(gen)[addr + 2 + o.n_strong + index] = (
            None if new is None else _enc(new[1], new[0]))
        if gen == OLD:
            self._recompute_remembered(addr)

    # ----- ephemeron slots --------------------------------------------- #
    def get_ephemeron(self, h: Handle, index: int) -> tuple[Handle | None, Handle | None]:
        """Return the raw current ``(key, value)`` pair of ephemeron slot."""
        gen, addr, o = self._idx(h, index, "ephemeron", "n_eph")
        base = addr + 2 + o.n_strong + o.n_weak + 2 * index
        space = self._space_of(gen)
        kref, vref = space[base], space[base + 1]
        return (
            None if kref is None else _href(kref, self._epoch),
            None if vref is None else _href(vref, self._epoch),
        )

    def set_ephemeron(
        self, h: Handle, index: int, key: Handle | None, value: Handle | None
    ) -> None:
        """Write ephemeron slot ``index``; a ``None`` key empties the entry."""
        gen, addr, o = self._idx(h, index, "ephemeron", "n_eph")
        k = None if key is None else self._resolve(key)
        v = None if value is None else self._resolve(value)
        base = addr + 2 + o.n_strong + o.n_weak + 2 * index
        space = self._space_of(gen)
        space[base] = None if k is None else _enc(k[1], k[0])
        space[base + 1] = None if v is None else _enc(v[1], v[0])
        self._barrier_ephemeron((gen, addr), index, k, v)
        if gen == OLD:
            self._recompute_remembered(addr)

    def clear_ephemeron(self, h: Handle, index: int) -> None:
        self.set_ephemeron(h, index, None, None)

    # ----- finalizers -------------------------------------------------- #
    def set_finalizer(self, h: Handle, event_id: int) -> None:
        """Register/replace the explicit finalization event of one object.

        An object identity whose finalizer body has already run (even when
        resurrected) cannot be re-registered: finalization happens at most
        once.
        """
        _check_int(event_id, "event_id")
        gen, addr = self._resolve(h)
        o = self._obj[gen][addr]
        if o.finalized or o.oid in self._finalized_oids:
            raise GCError(
                f"object {o.oid} was already finalized; finalizers run at most once")
        o.finalizer = event_id

    def finalizer_event_id(self, h: Handle) -> int | None:
        gen, addr = self._resolve(h)
        return self._obj[gen][addr].finalizer

    def finalized_event_ids(self) -> tuple[int, ...]:
        """Event ids whose finalizer body has executed (heap-wide ledger)."""
        return tuple(sorted(self._finalized_events))

    def pending_finalizers(self) -> tuple[FinalizationEvent, ...]:
        if self._cyc is None:
            return ()
        return tuple(self._cyc["final_q"])

    # ================================================================== #
    # Write barriers / remembered set
    # ================================================================== #
    def _recompute_remembered(self, old_addr: int) -> None:
        """Exact remembered-set maintenance for one old object.

        The set records every old object that holds *any* old -> young
        slot (strong, weak, ephemeron key or ephemeron value).  Strong
        targets are minor roots; weak/ephemeron slots of recorded holders
        are revisited by the minor clearing pass so no old -> young slot
        can dangle after young addresses are reused.
        """
        o = self._obj[OLD].get(old_addr)
        if o is None:
            self._remembered.discard(old_addr)
            return
        has_young = False
        base_w = old_addr + 2 + o.n_strong
        base_e = base_w + o.n_weak
        slots = [self._old[old_addr + 2 + j] for j in range(o.n_strong)]
        slots += [self._old[base_w + j] for j in range(o.n_weak)]
        for j in range(o.n_eph):
            slots.append(self._old[base_e + 2 * j])
            slots.append(self._old[base_e + 2 * j + 1])
        for ref in slots:
            if ref is not None and _dec(ref)[1] == YOUNG:
                has_young = True
                break
        if has_young:
            self._remembered.add(old_addr)
        else:
            self._remembered.discard(old_addr)

    def _barrier_strong(self, src: tuple[int, int],
                        dst: tuple[int, int] | None) -> None:
        """Incremental-update barrier.

        A store of a white target into a marked (gray/black) object marks
        the target immediately.  Deleted edges are separately protected by
        the per-object snapshot taken when scanning starts.
        """
        if self._cyc is not None and dst is not None:
            c = self._cyc
            if src in c["mark"] and dst not in c["mark"]:
                self._mark(dst, gray=True)

    def _barrier_root(self, key: tuple[int, int] | None) -> None:
        if self._cyc is not None and key is not None and key not in self._cyc["mark"]:
            self._mark(key, gray=True)

    def _barrier_ephemeron(self, table, idx, k, v) -> None:
        if self._cyc is None:
            return
        c = self._cyc
        if table in c["mark"]:
            # A black table is never rescanned, so register the new entry
            # explicitly for the remaining fixed-point passes.
            self._register_eph(table, idx)
            if k is not None and self._key_reachable(table, k) and v is not None:
                if v not in c["mark"]:
                    self._mark(v, gray=True)

    def _key_reachable(self, table, k) -> bool:
        """Reachability test used while resolving an ephemeron entry."""
        c = self._cyc
        if c["kind"] == "minor" and k[0] == OLD and table in c["mark"]:
            # Old-generation liveness is not traced by a minor cycle;
            # never let an old-keyed entry fail activation/clearing early.
            return True
        return k in c["mark"]

    def _born_during_cycle(self, h: Handle) -> None:
        if self._cyc is not None:
            key = (h.generation, h.address)
            self._cyc["born"].append(key)
            # Born marked; a young newborn is gray so its strong edges are
            # traced (its stores also cross the IU barrier).
            self._mark(key, gray=True)

    # ================================================================== #
    # Incremental cycle: begin / marking / finalization / finish
    # ================================================================== #
    def begin_collection(self, kind: str = "minor") -> None:
        if self._cyc is not None:
            raise GCError(f"a {self._cyc['kind']} collection is already in progress")
        if kind not in ("minor", "major"):
            raise ValueError("kind must be 'minor' or 'major'")
        self._cyc = {
            "kind": kind,
            "phase": "marking",
            "mark": set(),          # gray + black keys
            "gray": [],             # [[key, next_field_index], ...]
            "snap": {},             # key -> strong target snapshot
            "eph_pending": set(),
            "eph_round": None,   # sorted entries of the current sweep round
            "eph_cursor": 0,
            "eph_progressed": False,
            "resolved": [],
            "rescued": [],
            "final_q": [],
            "finalized_now": [],
            "born": [],
            "scanned": 0,
        }
        for name in self._root_order:
            key = self._roots[name]
            if key is not None:
                self._seed(key)
        if kind == "minor":
            self._seed_remembered()

    def _seed(self, key: tuple[int, int]) -> None:
        """Mark a root key; old objects are atomic black roots in minor cycles."""
        c = self._cyc
        if key in c["mark"]:
            return
        c["mark"].add(key)
        if c["kind"] == "major" or key[0] == YOUNG:
            c["gray"].append([key, 0])
        else:
            self._adopt_old_table(key)

    def _adopt_old_table(self, key: tuple[int, int]) -> None:
        """Register every ephemeron entry of an old table made black in a
        minor cycle.  Old objects are never scanned by a minor cycle, so
        without this their entries could never activate/resolve, which
        would leave a dangling young value slot."""
        if self._cyc["kind"] != "minor" or key[0] != OLD:
            return
        gen, addr = key
        o = self._obj[OLD].get(addr)
        if o is not None:
            for j in range(o.n_eph):
                self._register_eph(key, j)

    def _mark(self, key: tuple[int, int], *, gray: bool) -> None:
        c = self._cyc
        if key in c["mark"]:
            return
        c["mark"].add(key)
        if c["kind"] == "major" or key[0] == YOUNG:
            if gray:
                c["gray"].append([key, 0])
        else:
            self._adopt_old_table(key)

    def _register_eph(self, table: tuple[int, int], idx: int) -> None:
        """Register one ephemeron entry; invalidate the in-flight round."""
        c = self._cyc
        c["eph_pending"].add((table, idx))
        c["eph_round"] = None
        c["eph_cursor"] = 0

    def _seed_remembered(self) -> None:
        """Minor roots coming from the (exact) remembered set."""
        c = self._cyc
        for old_addr in sorted(self._remembered):
            table = (OLD, old_addr)
            o = self._obj[OLD].get(old_addr)
            if o is None:
                continue
            c["mark"].add(table)  # atomic black root
            for j in range(o.n_strong):
                ref = self._old[old_addr + 2 + j]
                if ref is not None and _dec(ref)[1] == YOUNG:
                    self._mark(_key(ref), gray=True)
            # Its ephemeron entries must participate in the fixed point.
            for j in range(o.n_eph):
                self._register_eph(table, j)

    def _read_strong_targets(self, key: tuple[int, int]) -> list[tuple[int, int] | None]:
        gen, addr = key
        o = self._obj[gen][addr]
        space = self._space_of(gen)
        return [
            (None if space[addr + 2 + j] is None else _key(space[addr + 2 + j]))
            for j in range(o.n_strong)
        ]

    # ----- marking ----------------------------------------------------- #
    def mark_step(self, scan_budget: int) -> StepReport:
        """Scan up to ``scan_budget`` edges (overrun <= 1 edge) of closure."""
        _check_int(scan_budget, "scan_budget")
        if scan_budget <= 0:
            raise ValueError("scan_budget must be >= 1")
        if self._cyc is None:
            raise GCError("no collection in progress; call begin_collection()")
        c = self._cyc
        if c["phase"] != "marking":
            raise GCError(
                f"mark_step requires the 'marking' phase, current phase "
                f"{c['phase']!r}; use finalize_step/finish_collection")
        start = c["scanned"]
        examined_all = True
        fixpoint = False

        while c["scanned"] - start < scan_budget:
            if c["gray"]:
                key, idx = c["gray"][-1]
                if key not in c["snap"]:
                    c["snap"][key] = self._read_strong_targets(key)
                    gen, addr = key
                    for j in range(self._obj[gen][addr].n_eph):
                        self._register_eph(key, j)
                edges = c["snap"][key]
                if idx < len(edges):
                    c["gray"][-1][1] = idx + 1
                    c["scanned"] += 1
                    t = edges[idx]
                    if t is not None:
                        self._mark(t, gray=True)
                    continue
                c["gray"].pop()  # fully black
                continue
            fixpoint, _examined_all = self._eph_round_step(
                scan_budget - (c["scanned"] - start))
            if not fixpoint:
                continue
            break

        charged = c["scanned"] - start
        if c["gray"] or not fixpoint:
            return StepReport("marking", charged, scan_budget,
                              charged >= scan_budget, False)

        # True marking fixed point.  Stage finalizers, or restart for remark.
        if c["phase"] == "marking":
            self._stage_finalizers()
            if c["final_q"]:
                c["phase"] = "finalizing"
                return StepReport("finalizing", charged, scan_budget, False, False)
            self._begin_remark()
            return StepReport("remark", charged, scan_budget, False, False)
        return StepReport(c["phase"], charged, scan_budget, False, False)

    def _eph_round_step(self, remaining: int) -> tuple[bool, bool]:
        """Advance one sweep round of pending ephemeron entries.

        Fair round scheduling prevents budget starvation: a round walks
        the entries in deterministic order with a cursor that survives
        across calls, so an undecided entry at the front cannot consume
        every step forever.  An activation marks a value gray and (once
        the gray set drains) starts a fresh round so entries registered by
        # newly scanned tables are seen.  The fixed point is a round that
        examined every entry and activated nothing while the gray set was
        empty.  Returns ``(fixpoint, examined_all)``.
        """
        c = self._cyc
        examined_all = True
        while True:
            if c["eph_round"] is None:
                c["eph_round"] = sorted(c["eph_pending"])
                c["eph_cursor"] = 0
                c["eph_progressed"] = False
            rnd = c["eph_round"]
            cur = c["eph_cursor"]
            while cur < len(rnd):
                entry = rnd[cur]
                table, idx = entry
                gen, taddr = table
                o = self._obj[gen].get(taddr)
                if o is None or table not in c["mark"]:
                    cur += 1  # metadata-only skip; reconsidered next round
                    continue
                if remaining <= 0:
                    examined_all = False
                    break  # cursor NOT advanced: this entry resumes later
                cur += 1
                space = self._space_of(gen)
                base = taddr + 2 + o.n_strong + o.n_weak
                kref = space[base + 2 * idx]
                vref = space[base + 2 * idx + 1]
                c["scanned"] += 1
                remaining -= 1
                if kref is None:
                    c["eph_pending"].discard(entry)  # permanently inert
                    continue
                k = _key(kref)
                if not self._key_reachable(table, k):
                    continue  # undecided this round; cursor moves on
                c["eph_pending"].discard(entry)
                c["resolved"].append((gen, taddr, idx))
                if vref is not None:
                    v = _key(vref)
                    if v not in c["mark"]:
                        c["rescued"].append((gen, taddr, idx))
                        c["eph_progressed"] = True
                        self._mark(v, gray=True)
            c["eph_cursor"] = cur
            if cur < len(rnd):
                return False, examined_all  # budget spent mid-round
            # Round complete.
            if c["eph_progressed"]:
                # Activations happened (gray may hold new work); once it
                # drains, a fresh round picks up newly registered entries.
                c["eph_round"] = None
                return False, examined_all
            if c["gray"]:
                return False, examined_all
            if set(c["eph_pending"]) - set(rnd):
                # Entries registered after the round snapshot existed.
                c["eph_round"] = None
                return False, examined_all
            return True, examined_all

    # ----- finalization ------------------------------------------------ #
    def _stage_finalizers(self) -> None:
        """Unreachable objects with a never-run finalizer become events."""
        c = self._cyc
        gens = (YOUNG,) if c["kind"] == "minor" else (OLD, YOUNG)
        q: list[FinalizationEvent] = []
        for g in gens:
            for addr in sorted(self._obj[g]):
                if (g, addr) in c["mark"]:
                    continue
                o = self._obj[g][addr]
                if o.finalizer is not None and not o.finalized:
                    q.append(FinalizationEvent(o.finalizer, g, addr, o.oid))
        q.sort(key=lambda e: (e.gen, e.address, e.event_id))
        c["final_q"] = q

    def finalize_step(self, handler: Callable[[int, Handle], object] | None) -> StepReport:
        """Run the finalizer body of the next staged event (exactly one).

        The body may attach roots, rewrite strong/weak/ephemeron edges and
        allocate; all such writes cross the normal barriers.  If the body
        raises, the event is not consumed and the exception propagates.
        """
        if self._cyc is None:
            raise GCError("no collection in progress")
        c = self._cyc
        if c["phase"] != "finalizing":
            raise GCError(
                f"finalize_step requires the 'finalizing' phase, current "
                f"phase {c['phase']!r}")
        if not c["final_q"]:
            self._begin_remark()
            return StepReport("remark", 0, 0, False, False)
        ev = c["final_q"][0]
        h = Handle(ev.address, self._epoch, ev.gen)
        if handler is not None:
            handler(ev.event_id, h)
        c["final_q"].pop(0)
        o = self._obj[ev.gen].get(ev.address)
        if o is not None:
            o.finalized = True
            if o.finalizer is not None:
                c["finalized_now"].append(o.finalizer)
            o.finalizer = None
        self._finalized_oids.add(ev.object_id)
        self._finalized_events.add(ev.event_id)
        if not c["final_q"]:
            self._begin_remark()
            return StepReport("remark", 0, 0, False, False)
        return StepReport("finalizing", 0, 0, False, False)

    def _begin_remark(self) -> None:
        """Re-mark from scratch: deletions during finalization now count."""
        c = self._cyc
        c["phase"] = "remark"
        c["mark"] = set()
        c["gray"] = []
        c["snap"] = {}
        c["eph_pending"] = set()
        c["eph_round"] = None
        c["eph_cursor"] = 0
        c["eph_progressed"] = False
        c["resolved"] = []
        c["rescued"] = []
        for name in self._root_order:
            key = self._roots[name]
            if key is not None:
                self._seed(key)
        # Objects born during the cycle are unconditionally live for it.
        for key in c["born"]:
            if key[1] in self._obj[key[0]]:
                self._mark(key, gray=True)
        if c["kind"] == "minor":
            self._seed_remembered()

    # ----- remark + weak clearing + commit ----------------------------- #
    def finish_collection(self, scan_budget: int) -> StepReport:
        """Drive the post-finalization remark, then weak clearing + commit."""
        _check_int(scan_budget, "scan_budget")
        if scan_budget <= 0:
            raise ValueError("scan_budget must be >= 1")
        if self._cyc is None:
            raise GCError("no collection in progress")
        c = self._cyc
        if c["phase"] in ("marking", "finalizing"):
            raise GCError(
                f"finish_collection requires the 'remark'/'clearing' "
                f"phase, current phase {c['phase']!r}")
        start = c["scanned"]

        if c["phase"] == "remark":
            examined_all = True
            fixpoint = False
            while c["scanned"] - start < scan_budget:
                if c["gray"]:
                    key, idx = c["gray"][-1]
                    if key not in c["snap"]:
                        c["snap"][key] = self._read_strong_targets(key)
                        gen, addr = key
                        for j in range(self._obj[gen][addr].n_eph):
                            self._register_eph(key, j)
                    edges = c["snap"][key]
                    if idx < len(edges):
                        c["gray"][-1][1] = idx + 1
                        c["scanned"] += 1
                        t = edges[idx]
                        if t is not None:
                            self._mark(t, gray=True)
                        continue
                    c["gray"].pop()
                    continue
                fixpoint, _examined_all = self._eph_round_step(
                    scan_budget - (c["scanned"] - start))
                if not fixpoint:
                    continue
                break
            if c["gray"] or not fixpoint:
                return StepReport("remark", c["scanned"] - start, scan_budget,
                                  c["scanned"] - start >= scan_budget, False)
            c["phase"] = "clearing"
            c["weak_work"] = self._build_weak_work()
            c["weak_cursor"] = 0
            c["cleared_weak"] = []
            c["cleared_eph"] = []

        if c["phase"] == "clearing":
            work = c["weak_work"]
            cur = c["weak_cursor"]
            while cur < len(work) and c["scanned"] - start < scan_budget:
                kind, key, idx = work[cur]
                cur += 1
                c["scanned"] += 1
                if kind == "weak":
                    self._clear_weak_if_dead(key, idx, c["cleared_weak"])
                else:
                    self._clear_ephemeron_if_dead(key, idx, c["cleared_eph"])
            c["weak_cursor"] = cur
            if cur < len(work):
                return StepReport("clearing", c["scanned"] - start, scan_budget,
                                  True, False)
            result = self._commit()
            return StepReport("idle", c["scanned"] - start, scan_budget,
                              False, True, result)
        return StepReport(c["phase"], 0, scan_budget, False, False)

    def _build_weak_work(self):
        """Old generation ascending, then young ascending; weak slots then
        ephemeron slots, each group by field index."""
        work = []
        for g in (OLD, YOUNG):
            for addr in sorted(self._obj[g]):
                if (g, addr) not in self._cyc["mark"]:
                    continue
                o = self._obj[g][addr]
                for j in range(o.n_weak):
                    work.append(("weak", (g, addr), j))
                for j in range(o.n_eph):
                    work.append(("eph", (g, addr), j))
        return work

    def _clear_weak_if_dead(self, key, idx, out) -> None:
        c = self._cyc
        gen, addr = key
        o = self._obj[gen][addr]
        space = self._space_of(gen)
        slot = addr + 2 + o.n_strong + idx
        ref = space[slot]
        if ref is None:
            return
        t = _key(ref)
        if c["kind"] == "minor" and t[0] == OLD:
            return  # never early-clear an edge into the old generation
        if t not in c["mark"]:
            space[slot] = None
            out.append(WeakClear(gen, addr, idx))

    def _clear_ephemeron_if_dead(self, key, idx, out) -> None:
        c = self._cyc
        gen, addr = key
        o = self._obj[gen][addr]
        space = self._space_of(gen)
        base = addr + 2 + o.n_strong + o.n_weak + 2 * idx
        kref = space[base]
        vref = space[base + 1]
        if kref is None:
            # Keyless (inert) entry: the value is never rescued, so the
            # value slot must be cleared if its object is reclaimed --
            # otherwise it would dangle into reused space.
            if vref is not None:
                v = _key(vref)
                if c["kind"] == "minor" and v[0] == OLD:
                    return
                if v not in c["mark"]:
                    space[base + 1] = None
                    out.append(EphemeronClear(gen, addr, idx))
            return
        k = _key(kref)
        if c["kind"] == "minor" and k[0] == OLD:
            return  # old-keyed entries are resolved exactly only by major
        if k not in c["mark"]:
            space[base] = None
            space[base + 1] = None
            out.append(EphemeronClear(gen, addr, idx))

    # ----- commit ------------------------------------------------------ #
    def _commit(self) -> GenCollectionResult:
        c = self._cyc
        kind = c["kind"]
        marked = c["mark"]

        pre_old = dict(self._obj[OLD])
        pre_young = dict(self._obj[YOUNG])
        dead_old = [a for a in pre_old if (OLD, a) not in marked]
        dead_young = [a for a in pre_young if (YOUNG, a) not in marked]
        reclaimed = len(dead_young) + (len(dead_old) if kind == "major" else 0)

        # Evacuation plan for young survivors: (YOUNG,addr) -> destination.
        if kind == "major":
            order = [k for k in self._major_evac_order(marked, pre_old, pre_young)
                     if k[0] == YOUNG]
            plan = {key: (OLD, -1) for key in order}
        else:
            plan = self._plan_evacuation(marked, pre_old, pre_young)

        # Pre-flight promotion capacity against the *post-sweep* free list,
        # simulated without mutating the heap (first-fit fragmentation can
        # make the total sufficient while no single hole fits).  On failure
        # nothing has been mutated: the heap is left exactly as before.
        if kind == "major":
            live_runs = [(a, pre_old[a].size) for a in sorted(pre_old)
                         if (OLD, a) in marked]
            post_free = coalesce_free(self._ocap, live_runs)
        else:
            post_free = list(self._free)
        promote_sizes = [pre_young[k[1]].size for k, dest in plan.items()
                         if dest[0] == OLD]
        if not simulate_old_fit(post_free, promote_sizes):
            self._cyc = None
            raise HeapCapacityError(
                f"old generation cannot absorb {sum(promote_sizes)} promoted "
                f"slot(s) in {len(promote_sizes)} object(s); heap left unchanged")

        relocated: list[Relocation] = []
        if kind == "major":
            # Sweep dead old objects, then coalesce (equals post_free).
            for a in dead_old:
                o = pre_old[a]
                for i in range(o.size):
                    self._old[a + i] = None
            self._free = post_free
            # Every marked young survivor promotes into old.
            fwd = {key: self._consume_old(pre_young[key[1]].size) for key in plan}
            self._bulk_promote(pre_young, fwd, relocated)
            self._reset_young()
        else:
            fwd = self._minor_copy(pre_young, plan, relocated)

        # Forwarding on destination copies and stationary old survivors.
        self._forward_all(fwd, marked, pre_old, pre_young, kind)
        if kind == "minor":
            # A minor cycle never sweeps old garbage, but young addresses
            # are reused immediately: slots of *unmarked* (unreachable) old
            # objects naming young objects must be rewritten too -- moved
            # targets forwarded, collected targets cleared -- or they
            # would dangle into reused young space before the next major
            # sweep (they may also be made reachable again via a retained
            # handle).  Such an object cannot resurrect a collected young
            # target: that target was unreachable and is already gone.
            for addr in sorted(pre_old):
                if (OLD, addr) not in marked:
                    self._forward_dead_holder(
                        self._old, addr, pre_old[addr], fwd, marked)

        # Roots.
        new_roots = {}
        for name in self._root_order:
            key = self._roots[name]
            new_roots[name] = None if key is None else fwd.get(key, key)
        self._roots = new_roots

        # Rebuild records (identity, age, finalizer state carried across).
        new_old, new_young = self._rebuild_records(
            pre_old, pre_young, fwd, marked, kind)
        self._obj = [new_old, new_young]

        # Exact remembered set for the new world.
        self._remembered = set()
        for addr in new_old:
            self._recompute_remembered(addr)

        live_old = self._ocap - sum(s for _, s in self._free)
        live_young = self._yptr

        def post(key):
            return fwd.get(key, key)

        resolved = tuple(sorted({
            (post((g, a))[0], post((g, a))[1], i) for (g, a, i) in c["resolved"]
        }))
        rescued = tuple(sorted({
            (post((g, a))[0], post((g, a))[1], i) for (g, a, i) in c["rescued"]
        }))
        cleared_weak = tuple(sorted(
            (WeakClear(post((w.gen, w.holder))[0], post((w.gen, w.holder))[1],
                       w.index) for w in c["cleared_weak"]),
            key=lambda w: (w.gen, w.holder, w.index)))
        cleared_eph = tuple(sorted(
            (EphemeronClear(post((e.gen, e.holder))[0],
                            post((e.gen, e.holder))[1], e.index)
             for e in c["cleared_eph"]),
            key=lambda e: (e.gen, e.holder, e.index)))

        result = GenCollectionResult(
            kind=kind,
            relocated=tuple(sorted(
                relocated, key=lambda r: (r.from_gen, r.from_addr,
                                          r.to_gen, r.to_addr))),
            promoted=sum(1 for r in relocated if r.from_gen == YOUNG and r.to_gen == OLD),
            live_old_slots=live_old,
            live_young_slots=live_young,
            roots_preserved=sum(1 for k in self._roots.values() if k is not None),
            cleared_weak_edges=cleared_weak,
            cleared_ephemerons=cleared_eph,
            resolved_ephemerons=resolved,
            rescued_values=rescued,
            finalized=tuple(sorted(set(c["finalized_now"]))),
            reclaimed_objects=reclaimed,
            edges_scanned=c["scanned"],
        )
        self._cyc = None
        self._epoch += 1
        return result

    def _plan_evacuation(self, marked, pre_old, pre_young):
        """Minor: decide young survivor destination (young to-space or old).

        Survivors younger than ``promotion_age`` stay young in evacuation
        order; if the young to-space overflows, the overflow survivors
        promote early (deterministic, documented).
        """
        order = self._minor_evac_order(marked, pre_old, pre_young)
        plan: dict[tuple[int, int], tuple[int, int]] = {}
        to_ptr = 0
        for key in order:
            o = pre_young[key[1]]
            if o.age + 1 >= self._promotion_age or to_ptr + o.size > self._ycap:
                plan[key] = (OLD, -1)  # old address assigned at commit time
            else:
                plan[key] = (YOUNG, to_ptr)
                to_ptr += o.size
        return plan

    def _consume_old(self, size: int) -> tuple[int, int]:
        addr = self._old_fit(size)
        assert addr is not None  # pre-flight checked the total
        return OLD, addr

    def _minor_evac_order(self, marked, pre_old, pre_young):
        """Deterministic young-only BFS order.

        Seeds (in order): young explicit roots (root registration order),
        remembered-set young strong targets (holder address ascending, then
        field order), young values rescued by RS ephemerons.  Closure then
        follows young strong edges in field order.
        """
        seen: set[tuple[int, int]] = set()
        order: list[tuple[int, int]] = []

        def add(key):
            if key[0] == YOUNG and key in marked and key not in seen \
                    and key[1] in pre_young:
                seen.add(key)
                order.append(key)

        for name in self._root_order:
            key = self._roots[name]
            if key is not None:
                add(key)
        for old_addr in sorted(self._remembered):
            o = pre_old.get(old_addr)
            if o is None:
                continue
            for j in range(o.n_strong):
                ref = self._old[old_addr + 2 + j]
                if ref is not None:
                    add(_key(ref))
        # Rescued ephemeron values (any table) that are young.
        for g, a, _i in self._cyc["rescued"]:
            add((g, a))
        # Marked young keys not reachable through the seeds above (e.g.
        # objects born during the cycle) are appended in address order.
        for addr in sorted(pre_young):
            add((YOUNG, addr))

        head = 0
        while head < len(order):
            key = order[head]
            head += 1
            o = pre_young[key[1]]
            space = self._young[self._yactive]
            for j in range(o.n_strong):
                ref = space[key[1] + 2 + j]
                if ref is not None:
                    add(_key(ref))
        return order

    def _major_evac_order(self, marked, pre_old, pre_young):
        """Root order + BFS over the full marked closure (field order)."""
        seen: set[tuple[int, int]] = set()
        order: list[tuple[int, int]] = []

        def add(key):
            if key in marked and key not in seen:
                exists = key[1] in (pre_young if key[0] == YOUNG else pre_old)
                if exists:
                    seen.add(key)
                    order.append(key)

        for name in self._root_order:
            key = self._roots[name]
            if key is not None:
                add(key)
        # Marked keys not reachable through the root seeds (e.g. objects
        # born during the cycle) are appended deterministically.
        for addr in sorted(pre_old):
            add((OLD, addr))
        for addr in sorted(pre_young):
            add((YOUNG, addr))
        head = 0
        while head < len(order):
            gen, addr = order[head]
            head += 1
            objs = pre_young if gen == YOUNG else pre_old
            o = objs[addr]
            space = self._young[self._yactive] if gen == YOUNG else self._old
            for j in range(o.n_strong):
                ref = space[addr + 2 + j]
                if ref is not None:
                    add(_key(ref))
        return order

    def _bulk_promote(self, pre_young, fwd, relocated) -> None:
        src = self._young[self._yactive]
        for key, dest in fwd.items():
            old_addr = key[1]
            o = pre_young[old_addr]
            self._old[dest[1]:dest[1] + o.size] = src[old_addr:old_addr + o.size]
            relocated.append(Relocation(YOUNG, old_addr, OLD, dest[1]))

    def _minor_copy(self, pre_young, plan, relocated) -> dict:
        src = self._young[self._yactive]
        dstidx = 1 - self._yactive
        dst = self._young[dstidx]
        for i in range(self._ycap):
            dst[i] = None
        fwd: dict[tuple[int, int], tuple[int, int]] = {}
        to_ptr = 0
        for key in plan:
            old_addr = key[1]
            o = pre_young[old_addr]
            dest_gen, dest_addr = plan[key]
            if dest_gen == YOUNG:
                dst[dest_addr:dest_addr + o.size] = src[old_addr:old_addr + o.size]
                to_ptr = max(to_ptr, dest_addr + o.size)
                fwd[key] = (YOUNG, dest_addr)
            else:
                slot = self._old_fit(o.size)
                assert slot is not None
                self._old[slot:slot + o.size] = src[old_addr:old_addr + o.size]
                fwd[key] = (OLD, slot)
            relocated.append(Relocation(YOUNG, old_addr, fwd[key][0], fwd[key][1]))
        for i in range(self._ycap):
            src[i] = None
        self._yactive = dstidx
        self._yptr = to_ptr
        return fwd

    def _reset_young(self) -> None:
        for i in range(self._ycap):
            self._young[self._yactive][i] = None
        self._yptr = 0

    def _forward_all(self, fwd, marked, pre_old, pre_young, kind) -> None:
        # Copied young survivors (now in young to-space or in old holes).
        for key, dest in fwd.items():
            ng, na = dest
            space = self._space_of(ng)
            self._forward_one(space, na, pre_young[key[1]], fwd)
        # Stationary old survivors.
        for addr in sorted(pre_old):
            if (OLD, addr) in marked:
                self._forward_one(self._old, addr, pre_old[addr], fwd)

    def _forward_dead_holder(self, space, base, o, fwd, marked) -> None:
        """Rewrite slots of an unmarked old object during a minor commit.

        Moved young targets are forwarded (conservative RS survival);
        collected young targets are cleared so no slot dangles into reused
        young space.  Old-generation targets keep their stable addresses.
        """
        def fix(slot: int) -> None:
            ref = space[slot]
            if ref is None:
                return
            t = _key(ref)
            if t in fwd:
                space[slot] = _enc(fwd[t][1], fwd[t][0])
            elif t[0] == YOUNG and t not in marked:
                space[slot] = None

        for j in range(o.n_strong):
            fix(base + 2 + j)
        for j in range(o.n_weak):
            fix(base + 2 + o.n_strong + j)
        b = base + 2 + o.n_strong + o.n_weak
        for j in range(o.n_eph):
            fix(b + 2 * j)
            fix(b + 2 * j + 1)

    def _forward_one(self, space, base: int, o: _Obj, fwd) -> None:
        def fix(slot: int) -> None:
            ref = space[slot]
            if ref is None:
                return
            t = _key(ref)
            nt = fwd.get(t)
            if nt is not None:
                space[slot] = _enc(nt[1], nt[0])

        for j in range(o.n_strong):
            fix(base + 2 + j)
        for j in range(o.n_weak):
            fix(base + 2 + o.n_strong + j)
        b = base + 2 + o.n_strong + o.n_weak
        for j in range(o.n_eph):
            fix(b + 2 * j)
            fix(b + 2 * j + 1)

    def _rebuild_records(self, pre_old, pre_young, fwd, marked, kind):
        new_old: dict[int, _Obj] = {}
        new_young: dict[int, _Obj] = {}
        for addr, o in pre_old.items():
            if kind == "major" and (OLD, addr) not in marked:
                continue
            new_old[addr] = _Obj(
                o.oid, OLD, addr, o.n_strong, o.n_weak, o.n_eph, o.size, o.age,
                None if o.finalized else o.finalizer, o.finalized)
        for addr, o in pre_young.items():
            key = (YOUNG, addr)
            if key not in marked:
                continue
            ng, na = fwd[key]
            rec = _Obj(o.oid, ng, na, o.n_strong, o.n_weak, o.n_eph, o.size,
                       o.age + 1, None if o.finalized else o.finalizer, o.finalized)
            if ng == OLD:
                new_old[na] = rec
            else:
                new_young[na] = rec
        return new_old, new_young

    # ================================================================== #
    # Whole-cycle convenience driver
    # ================================================================== #
    def run_collection(
        self,
        kind: str = "minor",
        scan_budget: int = DEFAULT_SCAN_BUDGET,
        *,
        finalizer: Callable[[int, Handle], object] | None = None,
        max_steps: int = 10_000_000,
    ) -> GenCollectionResult:
        """Run a complete minor/major collection in bounded-scan steps."""
        _check_int(scan_budget, "scan_budget")
        if scan_budget <= 0:
            raise ValueError("scan_budget must be >= 1")
        self.begin_collection(kind)
        steps = 0
        while True:
            steps += 1
            if steps > max_steps:
                raise GCError("cycle did not converge within max_steps")
            phase = self._cyc["phase"]
            if phase == "marking":
                self.mark_step(scan_budget)
            elif phase == "finalizing":
                self.finalize_step(finalizer)
            elif phase in ("remark", "clearing"):
                r = self.finish_collection(scan_budget)
                if r.complete:
                    assert r.result is not None
                    return r.result
            else:
                raise GCError(f"cycle stuck in phase {phase!r}")

    def collect(
        self,
        kind: str = "minor",
        scan_budget: int = DEFAULT_SCAN_BUDGET,
        *,
        finalizer: Callable[[int, Handle], object] | None = None,
    ) -> GenCollectionResult:
        """Convenience alias for :meth:`run_collection`."""
        return self.run_collection(kind, scan_budget, finalizer=finalizer)

    # ================================================================== #
    # Independent read-only reachability ledgers
    # ================================================================== #
    def reachable(self) -> frozenset[tuple[int, int]]:
        """Strong-only reachability from roots over the current heap."""
        seen: set[tuple[int, int]] = set()
        stack = [k for k in self._roots.values() if k is not None]
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            gen, addr = key
            o = self._obj[gen].get(addr)
            if o is None:
                continue
            space = self._space_of(gen)
            for j in range(o.n_strong):
                ref = space[addr + 2 + j]
                if ref is not None:
                    t = _key(ref)
                    if t not in seen:
                        stack.append(t)
        return frozenset(seen)

    def ephemeron_closure(
        self, base: frozenset[tuple[int, int]] | None = None
    ) -> frozenset[tuple[int, int]]:
        """Strong closure extended by the ephemeron fixed point.

        Ephemeron values are added only when both table and key are in the
        growing closure; iteration runs until no further value is added.
        Computed independently of any in-progress cycle.
        """
        reach = set(self.reachable() if base is None else base)
        while True:
            changed = False
            for g in (OLD, YOUNG):
                for addr in sorted(self._obj[g]):
                    table = (g, addr)
                    if table not in reach:
                        continue
                    o = self._obj[g][addr]
                    space = self._space_of(g)
                    b = addr + 2 + o.n_strong + o.n_weak
                    for j in range(o.n_eph):
                        kref = space[b + 2 * j]
                        vref = space[b + 2 * j + 1]
                        if kref is None or vref is None:
                            continue
                        k, v = _key(kref), _key(vref)
                        if k in reach and v not in reach:
                            reach.add(v)
                            changed = True
            if not changed:
                return frozenset(reach)


def _check_gen(gen: int) -> None:
    if isinstance(gen, bool) or not isinstance(gen, int):
        raise TypeError("generation must be a plain int (OLD=0 or YOUNG=1)")
    if gen not in (OLD, YOUNG):
        raise ValueError(f"generation must be OLD(0) or YOUNG(1), got {gen}")


def coalesce_free(capacity: int, used: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted disjoint free list derived from sorted used ``(start, size)``."""
    free: list[tuple[int, int]] = []
    cur = 0
    for start, size in used:
        if start > cur:
            free.append((cur, start - cur))
        cur = start + size
    if cur < capacity:
        free.append((cur, capacity - cur))
    return free


def simulate_old_fit(free: list[tuple[int, int]], sizes: list[int]) -> bool:
    """Simulate first-fit placement of ``sizes`` (in order) against a copy
    of the free list; True iff every size can be placed."""
    holes = list(free)
    for size in sizes:
        placed = False
        for i, (start, hole) in enumerate(holes):
            if hole >= size:
                if hole == size:
                    holes.pop(i)
                else:
                    holes[i] = (start + size, hole - size)
                placed = True
                break
        if not placed:
            return False
    return True
