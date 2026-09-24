"""Independent oracle for the generational GC (v2) tests.

This module deliberately does **not** import :mod:`moving_gc`.  It is a
second, independent implementation of the *documented* v2 semantics
(README.md, "分代引擎契约"), written in a different style: objects are
identity-keyed records holding object-id references, whereas the engine
stores encoded ``(address, generation)`` integers in flat slot arrays.

The test harness drives both implementations through the *same* operation
sequence (allocations, edge stores, root changes, finalizer registration,
incremental steps with the same budgets) and cross-checks, after every
single step:

* the marked/gray/ephemeron-pending sets (the "independent ledger"), and
* the final committed heap: survivors, relocated addresses, cleared weak
  edges, cleared/resolved ephemerons, finalized events, remembered set,
  roots and every edge of every surviving object.

Additionally :meth:`RefModel.strong_reachable` provides the plain
full-reachability baseline (roots + strong edges only, no GC semantics)
used for the soundness direction: nothing reachable may be reclaimed.
"""

from __future__ import annotations

OLD = 0
YOUNG = 1


class Rec:
    __slots__ = ("oid", "gen", "addr", "payload", "strong", "weak", "eph",
                 "age", "fin", "finalized")

    def __init__(self, oid, gen, addr, payload, ns, nw, ne):
        self.oid = oid
        self.gen = gen
        self.addr = addr
        self.payload = payload
        self.strong = [None] * ns
        self.weak = [None] * nw
        self.eph = [[None, None] for _ in range(ne)]
        self.age = 0
        self.fin = None
        self.finalized = False

    @property
    def size(self):
        return 2 + len(self.strong) + len(self.weak) + 2 * len(self.eph)


class RefModel:
    """Oracle twin of ``GenerationalHeap`` (same documented semantics)."""

    def __init__(self, young_capacity=512, old_capacity=512, promotion_age=1):
        self.ycap = young_capacity
        self.ocap = old_capacity
        self.promotion_age = promotion_age
        self.objs: dict[int, Rec] = {}
        self.next_oid = 0
        self.young_addr: dict[int, int] = {}   # oid -> address in active space
        self.old_addr: dict[int, int] = {}
        self.yptr = 0
        self.free: list[tuple[int, int]] = [(0, old_capacity)]
        self.roots: dict[str, int | None] = {}
        self.root_order: list[str] = []
        self.remembered: set[int] = set()      # oids of old holders
        self.finalized_oids: set[int] = set()
        self.epoch = 0
        self.cyc = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _gen_of(self, oid) -> int:
        return self.objs[oid].gen

    def key(self, oid):
        r = self.objs[oid]
        return (r.gen, r.addr)

    def _old_fit(self, size):
        for i, (start, hole) in enumerate(self.free):
            if hole >= size:
                if hole == size:
                    self.free.pop(i)
                else:
                    self.free[i] = (start + size, hole - size)
                return start
        return None

    def _recompute_remembered(self, oid):
        r = self.objs.get(oid)
        if r is None or r.gen != OLD:
            self.remembered.discard(oid)
            return
        targets = list(r.strong) + list(r.weak)
        for k, v in r.eph:
            targets.append(k)
            targets.append(v)
        if any(t is not None and self.objs[t].gen == YOUNG for t in targets):
            self.remembered.add(oid)
        else:
            self.remembered.discard(oid)

    # ------------------------------------------------------------------ #
    # mutator operations
    # ------------------------------------------------------------------ #
    def allocate(self, payload=0, n_strong=0, n_weak=0, n_ephemeron=0,
                 generation=YOUNG, _alloc_fail=()):
        """Returns oid.  ``_alloc_fail`` is appended with the size when the
        request cannot be satisfied after the mandated emergency
        collections (the harness then expects HeapCapacityError)."""
        size = 2 + n_strong + n_weak + 2 * n_ephemeron
        if generation == YOUNG:
            if size > self.ycap:
                _alloc_fail.append(size)
                return None
            if self.yptr + size > self.ycap:
                self.run_collection("minor", 64, None)
                if self.yptr + size > self.ycap:
                    self.run_collection("major", 64, None)
                if self.yptr + size > self.ycap:
                    _alloc_fail.append(size)
                    return None
            addr = self.yptr
            self.yptr += size
        else:
            if size > self.ocap:
                _alloc_fail.append(size)
                return None
            addr = self._old_fit(size)
            if addr is None:
                self.run_collection("major", 64, None)
                addr = self._old_fit(size)
                if addr is None:
                    _alloc_fail.append(size)
                    return None
        oid = self.next_oid
        self.next_oid += 1
        rec = Rec(oid, generation, addr, payload, n_strong, n_weak, n_ephemeron)
        self.objs[oid] = rec
        (self.young_addr if generation == YOUNG else self.old_addr)[oid] = addr
        if self.cyc is not None:
            self.cyc["born"].append(oid)
            self._mark(oid, gray=True)
        return oid

    def set_strong(self, oid, idx, tgt):
        r = self.objs[oid]
        r.strong[idx] = tgt
        if self.cyc is not None and tgt is not None:
            if oid in self.cyc["mark"] and tgt not in self.cyc["mark"]:
                self._mark(tgt, gray=True)
        if r.gen == OLD:
            self._recompute_remembered(oid)

    def set_weak(self, oid, idx, tgt):
        r = self.objs[oid]
        r.weak[idx] = tgt
        if r.gen == OLD:
            self._recompute_remembered(oid)

    def set_ephemeron(self, oid, idx, k, v):
        r = self.objs[oid]
        r.eph[idx] = [k, v]
        if self.cyc is not None and oid in self.cyc["mark"]:
            self._register_eph(oid, idx)
            if (k is not None and v is not None
                    and self._key_reachable(oid, k)
                    and v not in self.cyc["mark"]):
                self._mark(v, gray=True)
        if r.gen == OLD:
            self._recompute_remembered(oid)

    def add_root(self, name, oid):
        if name not in self.roots:
            self.root_order.append(name)
        self.roots[name] = oid
        if self.cyc is not None and oid is not None and oid not in self.cyc["mark"]:
            self._mark(oid, gray=True)

    def clear_root(self, name):
        self.roots[name] = None

    def remove_root(self, name):
        del self.roots[name]
        self.root_order.remove(name)

    def set_finalizer(self, oid, event_id):
        r = self.objs[oid]
        assert not r.finalized and oid not in self.finalized_oids
        r.fin = event_id

    # ------------------------------------------------------------------ #
    # cycle machine (same documented rules, independent code)
    # ------------------------------------------------------------------ #
    def begin(self, kind):
        assert self.cyc is None
        self.cyc = {
            "kind": kind, "phase": "marking",
            "mark": set(), "gray": [], "snap": {},
            "eph_pending": set(), "resolved": [], "rescued": [],
            "eph_round": None, "eph_cursor": 0, "eph_progressed": False,
            "final_q": [], "finalized_now": [], "born": [],
            "scanned": 0,
        }
        for name in self.root_order:
            oid = self.roots[name]
            if oid is not None:
                self._seed(oid)
        if kind == "minor":
            self._seed_remembered()

    def _seed(self, oid):
        c = self.cyc
        if oid in c["mark"]:
            return
        c["mark"].add(oid)
        r = self.objs[oid]
        if c["kind"] == "major" or r.gen == YOUNG:
            c["gray"].append([oid, 0])
        else:
            self._adopt_old_table(oid)

    def _adopt_old_table(self, oid):
        c = self.cyc
        if c["kind"] != "minor" or self.objs[oid].gen != OLD:
            return
        for j in range(len(self.objs[oid].eph)):
            self._register_eph(oid, j)

    def _register_eph(self, oid, idx):
        c = self.cyc
        c["eph_pending"].add((oid, idx))
        c["eph_round"] = None
        c["eph_cursor"] = 0

    def _mark(self, oid, gray):
        c = self.cyc
        if oid in c["mark"]:
            return
        c["mark"].add(oid)
        r = self.objs[oid]
        if c["kind"] == "major" or r.gen == YOUNG:
            if gray:
                c["gray"].append([oid, 0])
        else:
            self._adopt_old_table(oid)

    def _seed_remembered(self):
        c = self.cyc
        for oid in sorted(self.remembered, key=lambda o: self.old_addr[o]):
            r = self.objs.get(oid)
            if r is None or r.gen != OLD:
                continue
            c["mark"].add(oid)
            for t in r.strong:
                if t is not None and self.objs[t].gen == YOUNG:
                    self._mark(t, gray=True)
            for j in range(len(r.eph)):
                self._register_eph(oid, j)

    def _key_reachable(self, table_oid, k_oid):
        c = self.cyc
        if (c["kind"] == "minor" and self.objs[k_oid].gen == OLD
                and table_oid in c["mark"]):
            return True
        return k_oid in c["mark"]

    def _eph_round_step(self, remaining):
        """Fair sweep round with a call-surviving cursor (engine twin)."""
        c = self.cyc
        examined_all = True
        while True:
            if c["eph_round"] is None:
                c["eph_round"] = sorted(
                    c["eph_pending"],
                    key=lambda e: (self.objs[e[0]].gen, self.objs[e[0]].addr, e[1]))
                c["eph_cursor"] = 0
                c["eph_progressed"] = False
            rnd = c["eph_round"]
            cur = c["eph_cursor"]
            while cur < len(rnd):
                entry = rnd[cur]
                oid, idx = entry
                r = self.objs.get(oid)
                if r is None or oid not in c["mark"]:
                    cur += 1
                    continue
                if remaining <= 0:
                    examined_all = False
                    break
                cur += 1
                c["scanned"] += 1
                charged_local = 1
                remaining -= 1
                k, v = r.eph[idx]
                if k is None:
                    c["eph_pending"].discard(entry)
                    continue
                if not self._key_reachable(oid, k):
                    continue
                c["eph_pending"].discard(entry)
                c["resolved"].append((oid, idx))
                if v is not None and v not in c["mark"]:
                    c["rescued"].append((oid, idx))
                    c["eph_progressed"] = True
                    self._mark(v, gray=True)
            c["eph_cursor"] = cur
            if cur < len(rnd):
                return False, examined_all
            if c["eph_progressed"]:
                c["eph_round"] = None
                return False, examined_all
            if c["gray"]:
                return False, examined_all
            if set(c["eph_pending"]) - set(rnd):
                c["eph_round"] = None
                return False, examined_all
            return True, examined_all

    def _drain(self, budget):
        """Shared mark loop; returns (charged, finished)."""
        c = self.cyc
        charged = 0
        examined_all = True
        fixpoint = False
        while charged < budget:
            if c["gray"]:
                oid, idx = c["gray"][-1]
                if oid not in c["snap"]:
                    c["snap"][oid] = list(self.objs[oid].strong)
                    for j in range(len(self.objs[oid].eph)):
                        self._register_eph(oid, j)
                edges = c["snap"][oid]
                if idx < len(edges):
                    c["gray"][-1][1] = idx + 1
                    c["scanned"] += 1
                    charged += 1
                    t = edges[idx]
                    if t is not None:
                        self._mark(t, gray=True)
                    continue
                c["gray"].pop()
                continue
            before_scanned = c["scanned"]
            fixpoint, examined_all = self._eph_round_step(budget - charged)
            charged += c["scanned"] - before_scanned
            if not fixpoint:
                continue
            break
        return charged, (not c["gray"]) and (fixpoint and examined_all)

    def mark_step(self, budget):
        c = self.cyc
        assert c["phase"] == "marking"
        charged, done = self._drain(budget)
        if not done:
            return "marking"
        self._stage_finalizers()
        if c["final_q"]:
            c["phase"] = "finalizing"
            return "finalizing"
        self._begin_remark()
        return "remark"

    def _stage_finalizers(self):
        c = self.cyc
        gens = (YOUNG,) if c["kind"] == "minor" else (OLD, YOUNG)
        q = []
        for oid, r in self.objs.items():
            if r.gen not in gens or oid in c["mark"]:
                continue
            if r.fin is not None and not r.finalized:
                q.append(oid)
        q.sort(key=lambda o: (self.objs[o].gen, self.objs[o].addr,
                              self.objs[o].fin))
        c["final_q"] = q

    def finalize_step(self):
        c = self.cyc
        assert c["phase"] == "finalizing"
        if not c["final_q"]:
            self._begin_remark()
            return "remark"
        oid = c["final_q"].pop(0)
        r = self.objs[oid]
        r.finalized = True
        c["finalized_now"].append(r.fin)
        r.fin = None
        self.finalized_oids.add(oid)
        if not c["final_q"]:
            self._begin_remark()
            return "remark"
        return "finalizing"

    def _begin_remark(self):
        c = self.cyc
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
        for name in self.root_order:
            oid = self.roots[name]
            if oid is not None:
                self._seed(oid)
        for oid in c["born"]:
            if oid in self.objs:
                self._mark(oid, gray=True)
        if c["kind"] == "minor":
            self._seed_remembered()

    def finish(self, budget):
        c = self.cyc
        assert c["phase"] in ("remark", "clearing")
        charged = 0
        if c["phase"] == "remark":
            charged, done = self._drain(budget)
            if not done:
                return "remark"
            c["phase"] = "clearing"
            work = []
            for g in (OLD, YOUNG):
                for oid in sorted(
                        (o for o in self.objs if self.objs[o].gen == g),
                        key=lambda o: self.objs[o].addr):
                    if oid not in c["mark"]:
                        continue
                    r = self.objs[oid]
                    for j in range(len(r.weak)):
                        work.append(("weak", oid, j))
                    for j in range(len(r.eph)):
                        work.append(("eph", oid, j))
            c["weak_work"] = work
            c["weak_cursor"] = 0
            c["cleared_weak"] = []
            c["cleared_eph"] = []
        # clearing (shares this call's budget with the remark phase)
        c = self.cyc
        work = c["weak_work"]
        cur = c["weak_cursor"]
        while cur < len(work) and charged < budget:
            kind, oid, idx = work[cur]
            cur += 1
            charged += 1
            c["scanned"] += 1
            r = self.objs[oid]
            if kind == "weak":
                t = r.weak[idx]
                if t is None:
                    continue
                if c["kind"] == "minor" and self.objs[t].gen == OLD:
                    continue
                if t not in c["mark"]:
                    r.weak[idx] = None
                    c["cleared_weak"].append((oid, idx))
            else:
                k, v = r.eph[idx]
                if k is None:
                    # keyless entry: value is never rescued; clear it if
                    # its object is reclaimed (no dangling value slot)
                    if v is not None:
                        if c["kind"] == "minor" and self.objs[v].gen == OLD:
                            continue
                        if v not in c["mark"]:
                            r.eph[idx] = [None, None]
                            c["cleared_eph"].append((oid, idx))
                    continue
                if c["kind"] == "minor" and self.objs[k].gen == OLD:
                    continue
                if k not in c["mark"]:
                    r.eph[idx] = [None, None]
                    c["cleared_eph"].append((oid, idx))
        c["weak_cursor"] = cur
        if cur < len(work):
            return "clearing"
        return self._commit()

    # ------------------------------------------------------------------ #
    # commit
    # ------------------------------------------------------------------ #
    def _commit(self):
        c = self.cyc
        kind = c["kind"]
        marked = c["mark"]

        pre = dict(self.objs)
        dead_old = [o for o in pre if pre[o].gen == OLD and o not in marked]
        dead_young = [o for o in pre if pre[o].gen == YOUNG and o not in marked]
        reclaimed = len(dead_young) + (len(dead_old) if kind == "major" else 0)

        # evacuation plan for young survivors
        order = self._evac_order(marked, kind, pre)
        plan: dict[int, tuple[int, int]] = {}
        to_ptr = 0
        for oid in order:
            r = pre[oid]
            if r.gen != YOUNG:
                continue  # only young objects evacuate; old survivors stay
            if kind == "major" or r.age + 1 >= self.promotion_age \
                    or to_ptr + r.size > self.ycap:
                plan[oid] = (OLD, -1)
            else:
                plan[oid] = (YOUNG, to_ptr)
                to_ptr += r.size

        # pre-flight promotion capacity against the post-sweep free list
        if kind == "major":
            used = sorted((self.old_addr[o], pre[o].size) for o in pre
                          if pre[o].gen == OLD and o in marked)
            post_free = _coalesce(self.ocap, used)
        else:
            post_free = list(self.free)
        sizes = [pre[o].size for o, d in plan.items() if d[0] == OLD]
        if not _simulate(post_free, sizes):
            self.cyc = None
            raise RuntimeError("oracle: old generation cannot absorb promotions")

        relocated = []  # (oid, from_gen, from_addr, to_gen, to_addr)
        fwd = {}
        if kind == "major":
            for o in dead_old:
                del self.old_addr[o]
            self.free = post_free
            for oid in plan:
                addr = self._old_fit(pre[oid].size)
                assert addr is not None
                fwd[oid] = (OLD, addr)
                relocated.append((oid, YOUNG, self.young_addr[oid], OLD, addr))
        else:
            to_ptr = 0
            for oid in plan:
                r = pre[oid]
                dest_gen, dest_addr = plan[oid]
                if dest_gen == YOUNG:
                    fwd[oid] = (YOUNG, dest_addr)
                    to_ptr = max(to_ptr, dest_addr + r.size)
                else:
                    addr = self._old_fit(r.size)
                    assert addr is not None
                    fwd[oid] = (OLD, addr)
                relocated.append((oid, YOUNG, self.young_addr[oid],
                                  fwd[oid][0], fwd[oid][1]))
            self.yptr = to_ptr

        # apply moves to records (before the young address table is cleared)
        for oid, (ng, na) in fwd.items():
            r = self.objs[oid]
            del self.young_addr[oid]
            r.gen = ng
            r.addr = na
            r.age += 1
            (self.old_addr if ng == OLD else self.young_addr)[oid] = na
        if kind == "major":
            self.young_addr.clear()
            self.yptr = 0

        # drop reclaimed objects
        for oid in dead_young + (dead_old if kind == "major" else []):
            del self.objs[oid]
        self.young_addr = {o: a for o, a in self.young_addr.items()
                           if o in self.objs}

        if kind == "minor":
            # Rewrite slots of unreachable old objects as well: moved young
            # targets forward, collected young targets cleared (no dangling
            # references into reused young space); old targets are stable.
            # Rewrite slots of unreachable old objects as well: collected
            # young targets are cleared (no dangling references into reused
            # young space); moved young targets keep their oid (the record
            # was updated to its new address above); old targets are stable.
            def fix(t):
                if t is None:
                    return None
                tr = pre.get(t)
                if tr is not None and tr.gen == YOUNG and t not in marked:
                    return None
                return t

            for r in (rec for oid, rec in pre.items()
                      if rec.gen == OLD and oid not in marked):
                r.strong = [fix(t) for t in r.strong]
                r.weak = [fix(t) for t in r.weak]
                r.eph = [[fix(k), fix(v)] for k, v in r.eph]

        # roots
        for name in self.root_order:
            oid = self.roots[name]
            if oid is not None and oid not in self.objs:
                raise AssertionError("oracle: root target reclaimed")

        # rebuild remembered set exactly
        self.remembered = set()
        for oid, r in self.objs.items():
            if r.gen == OLD:
                self._recompute_remembered(oid)

        result = {
            "kind": kind,
            "relocated": tuple(sorted(
                (fg, fa, tg, ta) for (_o, fg, fa, tg, ta) in relocated)),
            "promoted": sum(1 for _o, fg, _fa, tg, _ta in relocated
                            if fg == YOUNG and tg == OLD),
            "live_old_slots": self.ocap - sum(s for _, s in self.free),
            "live_young_slots": self.yptr,
            "roots_preserved": sum(1 for v in self.roots.values() if v is not None),
            "cleared_weak": tuple(sorted(
                (self.objs[o].gen, self.objs[o].addr, i)
                for (o, i) in c["cleared_weak"])),
            "cleared_eph": tuple(sorted(
                (self.objs[o].gen, self.objs[o].addr, i)
                for (o, i) in c["cleared_eph"])),
            "resolved": tuple(sorted(set(
                (self.objs[o].gen, self.objs[o].addr, i)
                for (o, i) in c["resolved"]))),
            "rescued": tuple(sorted(set(
                (self.objs[o].gen, self.objs[o].addr, i)
                for (o, i) in c["rescued"]))),
            "finalized": tuple(sorted(set(c["finalized_now"]))),
            "reclaimed": reclaimed,
            "edges_scanned": c["scanned"],
        }
        self.cyc = None
        self.epoch += 1
        self.last_result = result
        return result

    def _evac_order(self, marked, kind, pre):
        """Evacuation order; must match the engine's deterministic order.

        minor seeds: young roots (root order), remembered-set young strong
        targets (holder address, field order), rescued young values, then
        all remaining marked young (address order).
        major seeds: roots (root order), all marked old (address order),
        all marked young (address order).
        Both then BFS strong edges in field order; only young entries are
        returned (old objects never move).
        """
        seen = set()
        order = []

        def add(oid):
            if oid in seen or oid not in marked or oid not in self.objs:
                return
            seen.add(oid)
            order.append(oid)

        for name in self.root_order:
            oid = self.roots[name]
            if oid is not None:
                add(oid)
        if kind == "minor":
            for oid in sorted(self.remembered, key=lambda o: self.old_addr[o]):
                r = self.objs.get(oid)
                if r is None or r.gen != OLD:
                    continue
                for t in r.strong:
                    if t is not None:
                        add(t)
            for oid, _i in self.cyc["rescued"]:
                add(oid)
        for oid in sorted((o for o, r in self.objs.items() if r.gen == OLD),
                          key=lambda o: self.objs[o].addr):
            add(oid)
        for oid in sorted((o for o, r in self.objs.items() if r.gen == YOUNG),
                          key=lambda o: self.objs[o].addr):
            add(oid)
        head = 0
        while head < len(order):
            oid = order[head]
            head += 1
            if kind == "minor" and self.objs[oid].gen != YOUNG:
                continue
            for t in self.objs[oid].strong:
                if t is not None:
                    add(t)
        return [o for o in order if self.objs[o].gen == YOUNG]

    # ------------------------------------------------------------------ #
    # whole-cycle driver + ledgers
    # ------------------------------------------------------------------ #
    def run_collection(self, kind, budget, finalizer):
        self.begin(kind)
        guard = 0
        while True:
            guard += 1
            assert guard < 10_000_000
            phase = self.cyc["phase"]
            if phase == "marking":
                self.mark_step(budget)
            elif phase == "finalizing":
                oid = self.cyc["final_q"][0]
                ev = self.objs[oid].fin
                if finalizer is not None:
                    finalizer(ev, oid)
                self.finalize_step()
            else:
                res = self.finish(budget)
                if not isinstance(res, str):
                    return res

    def strong_reachable(self):
        """Plain full reachability: roots + strong edges, nothing else."""
        seen = set()
        stack = [o for o in self.roots.values() if o is not None]
        while stack:
            oid = stack.pop()
            if oid in seen or oid not in self.objs:
                continue
            seen.add(oid)
            for t in self.objs[oid].strong:
                if t is not None and t not in seen:
                    stack.append(t)
        return frozenset(seen)

    def ephemeron_closure(self):
        reach = set(self.strong_reachable())
        while True:
            changed = False
            for oid, r in self.objs.items():
                if oid not in reach:
                    continue
                for k, v in r.eph:
                    if k is not None and v is not None \
                            and k in reach and v not in reach:
                        reach.add(v)
                        changed = True
            if not changed:
                return frozenset(reach)


def _coalesce(capacity, used):
    free = []
    cur = 0
    for start, size in used:
        if start > cur:
            free.append((cur, start - cur))
        cur = start + size
    if cur < capacity:
        free.append((cur, capacity - cur))
    return free


def _simulate(free, sizes):
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
