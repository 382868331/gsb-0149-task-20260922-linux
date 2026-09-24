"""Twin-driver harness: run the engine and the independent oracle in
lockstep over the same operation sequence and cross-check everything.

The harness is shared by the fast regression tests
(:mod:`tests.test_generational_gc`) and the extended verification suite
(``verification/``).  It never trusts the engine's own progress flags:
every comparison is between the engine's observable state and the
independent :class:`gen_reference.RefModel`.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_reference import OLD, YOUNG, RefModel  # noqa: E402

from moving_gc import GenerationalHeap  # noqa: E402


class Twin:
    """One engine heap + one oracle model driven by the same operations."""

    def __init__(self, young_capacity=512, old_capacity=512, promotion_age=1):
        self.heap = GenerationalHeap(young_capacity, old_capacity,
                                     promotion_age=promotion_age)
        self.ref = RefModel(young_capacity, old_capacity, promotion_age)
        self.handles = {}   # name -> engine Handle (current epoch)
        self.oids = {}      # name -> oracle oid
        self.addr_to_name = {}  # (gen, addr) -> name, rebuilt after commit

    # ------------------------------------------------------------------ #
    # mutator operations (applied to both worlds)
    # ------------------------------------------------------------------ #
    def allocate(self, name, payload=0, n_strong=0, n_weak=0,
                 n_ephemeron=0, generation=YOUNG):
        # Keep the engine's emergency collections in lockstep: if this
        # allocation would not fit, drive the mandated minor/major cycle
        # through the twin first so both worlds collect identically.
        size = 2 + n_strong + n_weak + 2 * n_ephemeron
        if self.heap.cycle_phase() == "idle":
            if generation == YOUNG:
                if (size <= self.ref.ycap
                        and self.ref.yptr + size > self.ref.ycap):
                    self.run_cycle("minor", 64, None)
                    if self.ref.yptr + size > self.ref.ycap:
                        self.run_cycle("major", 64, None)
            else:
                if size <= self.ref.ocap and not self._ref_old_fits(size):
                    self.run_cycle("major", 64, None)
        h = self.heap.allocate(payload, n_strong=n_strong, n_weak=n_weak,
                               n_ephemeron=n_ephemeron, generation=generation)
        oid = self.ref.allocate(payload, n_strong, n_weak, n_ephemeron,
                                generation)
        assert oid is not None, "oracle unexpectedly failed to allocate"
        self.handles[name] = h
        self.oids[name] = oid
        self.addr_to_name[(generation, h.address)] = name
        return name

    def _ref_old_fits(self, size):
        holes = list(self.ref.free)
        for i, (start, hole) in enumerate(holes):
            if hole >= size:
                return True
        return False

    def _tgt(self, name):
        return None if name is None else self.handles[name]

    def _tgt_oid(self, name):
        return None if name is None else self.oids[name]

    def set_strong(self, name, idx, tgt):
        self.heap.set_strong(self.handles[name], idx, self._tgt(tgt))
        self.ref.set_strong(self.oids[name], idx, self._tgt_oid(tgt))

    def set_weak(self, name, idx, tgt):
        self.heap.set_weak(self.handles[name], idx, self._tgt(tgt))
        self.ref.set_weak(self.oids[name], idx, self._tgt_oid(tgt))

    def set_ephemeron(self, name, idx, key, value):
        self.heap.set_ephemeron(self.handles[name], idx,
                                self._tgt(key), self._tgt(value))
        self.ref.set_ephemeron(self.oids[name], idx,
                               self._tgt_oid(key), self._tgt_oid(value))

    def add_root(self, root_name, name):
        self.heap.add_root(root_name, self.handles[name])
        self.ref.add_root(root_name, self.oids[name])

    def clear_root(self, root_name):
        self.heap.clear_root(root_name)
        self.ref.clear_root(root_name)

    def remove_root(self, root_name):
        self.heap.remove_root(root_name)
        self.ref.remove_root(root_name)

    def set_finalizer(self, name, event_id):
        self.heap.set_finalizer(self.handles[name], event_id)
        self.ref.set_finalizer(self.oids[name], event_id)

    # ------------------------------------------------------------------ #
    # incremental cycle driving
    # ------------------------------------------------------------------ #
    def begin(self, kind):
        self.heap.begin_collection(kind)
        self.ref.begin(kind)
        self.check_ledger()

    def mark_step(self, budget):
        r = self.heap.mark_step(budget)
        assert r.edges_scanned <= budget, (
            f"budget violated: scanned {r.edges_scanned} > {budget}")
        if self.ref.cyc["phase"] == "marking":
            self.ref.mark_step(budget)
        self.check_ledger()
        return r

    def finalize_step(self, fn=None):
        """Run one finalizer body on both sides.

        ``fn(twin, event_id, name)`` executes once and may use any twin
        mutator operation (resurrect via add_root, allocate, rewrite).
        """
        pend = self.heap.pending_finalizers()
        assert pend, "no pending finalizer on engine"
        assert self.ref.cyc["phase"] == "finalizing"
        assert len(self.ref.cyc["final_q"]) == len(pend), (
            "finalizer queues diverged: "
            f"engine {len(pend)} vs oracle {len(self.ref.cyc['final_q'])}")
        ev = pend[0]
        name = self.addr_to_name[(ev.gen, ev.address)]
        assert self.ref.cyc["final_q"][0] == self.oids[name]
        if fn is not None:
            fn(self, ev.event_id, name)
        self.heap.finalize_step(None)  # body already ran through the twin
        self.ref.finalize_step()
        self.check_ledger()

    def finish(self, budget):
        r = self.heap.finish_collection(budget)
        assert r.edges_scanned <= budget, (
            f"budget violated: scanned {r.edges_scanned} > {budget}")
        if self.ref.cyc is not None and self.ref.cyc["phase"] in ("remark", "clearing"):
            self.ref.finish(budget)
        if r.complete:
            self._compare_results(r.result)
            self._refresh_handles()
        else:
            self.check_ledger()
        return r

    def run_cycle(self, kind, budget, finalizer=None):
        """Drive a whole cycle in lockstep; ``finalizer(twin, eid, name)``."""
        self.begin(kind)
        while True:
            phase = self.heap.cycle_phase()
            if phase == "marking":
                self.mark_step(budget)
            elif phase == "finalizing":
                self.finalize_step(finalizer)
            elif phase in ("remark", "clearing"):
                r = self.finish(budget)
                if r.complete:
                    return r.result
            else:
                raise AssertionError(f"unexpected phase {phase}")

    # ------------------------------------------------------------------ #
    # cross-checks
    # ------------------------------------------------------------------ #
    def check_ledger(self):
        """Compare the in-progress cycle state (the independent ledger)."""
        info = self.heap.cycle_info()
        c = self.ref.cyc
        assert (info is None) == (c is None), "cycle presence diverged"
        if c is None:
            return
        assert info.kind == c["kind"]
        assert info.phase == c["phase"]
        ref_mark = {self.ref.key(o) for o in c["mark"]}
        assert set(info.marked) == ref_mark, (
            f"mark sets diverged:\nengine-only "
            f"{set(info.marked) - ref_mark}\nref-only {ref_mark - set(info.marked)}")
        ref_gray = {self.ref.key(o) for o, _i in c["gray"]}
        assert set(info.gray) == ref_gray
        ref_pending = {(self.ref.key(o), i) for (o, i) in c["eph_pending"]}
        assert set(info.ephemeron_pending) == ref_pending
        assert info.edges_scanned == c["scanned"], (
            f"scan counters diverged: {info.edges_scanned} vs {c['scanned']}")
        # finalizer queues (same order, same objects)
        eng_q = [(e.gen, e.address, e.event_id) for e in info.finalizer_pending]
        ref_q = [self.ref.key(o) + (self.ref.objs[o].fin,)
                 for o in c["final_q"]]
        assert eng_q == ref_q, f"finalizer queues diverged: {eng_q} vs {ref_q}"

    def _compare_results(self, res):
        ref = self.ref
                # The oracle returned its result dict from finish(); recompute the
        # comparison from the committed state instead of trusting flags.
        assert res.kind == ref.last_result["kind"]
        got_reloc = tuple(sorted((r.from_gen, r.from_addr, r.to_gen, r.to_addr)
                                 for r in res.relocated))
        assert got_reloc == ref.last_result["relocated"], (
            f"relocated diverged:\n{got_reloc}\n{ref.last_result['relocated']}")
        assert res.promoted == ref.last_result["promoted"]
        assert res.live_old_slots == ref.last_result["live_old_slots"]
        assert res.live_young_slots == ref.last_result["live_young_slots"]
        assert res.roots_preserved == ref.last_result["roots_preserved"]
        got_weak = tuple(sorted((w.gen, w.holder, w.index)
                                for w in res.cleared_weak_edges))
        assert got_weak == ref.last_result["cleared_weak"], (
            f"cleared weak diverged: {got_weak} vs {ref.last_result['cleared_weak']}")
        got_eph = tuple(sorted((e.gen, e.holder, e.index)
                               for e in res.cleared_ephemerons))
        assert got_eph == ref.last_result["cleared_eph"]
        assert res.resolved_ephemerons == ref.last_result["resolved"]
        assert res.rescued_values == ref.last_result["rescued"]
        assert res.finalized == ref.last_result["finalized"]
        assert res.reclaimed_objects == ref.last_result["reclaimed"]
        assert res.edges_scanned == ref.last_result["edges_scanned"]
        self.compare_idle()

    def compare_idle(self):
        """Full post-commit comparison of both worlds (no cycle active)."""
        heap, ref = self.heap, self.ref
        assert heap.cycle_phase() == "idle" and ref.cyc is None
        # object sets
        ref_objs = {(r.gen, r.addr) for r in ref.objs.values()}
        assert set(heap.objects()) == ref_objs, (
            f"object sets diverged:\nengine-only "
            f"{set(heap.objects()) - ref_objs}\nref-only {ref_objs - set(heap.objects())}")
        # soundness vs plain full reachability (independent of GC semantics)
        must_live = {ref.key(o) for o in ref.strong_reachable()}
        assert must_live <= set(heap.objects()), (
            f"reachable objects were reclaimed: {must_live - set(heap.objects())}")
        # ephemeron fixed-point closure agrees
        assert set(heap.ephemeron_closure()) == {
            ref.key(o) for o in ref.ephemeron_closure()}
        assert set(heap.reachable()) == must_live
        # roots
        for rn in ref.root_order:
            rh = heap.root_handle(rn)
            oid = ref.roots[rn]
            if oid is None:
                assert rh is None
            else:
                assert rh is not None
                assert (rh.generation, rh.address) == ref.key(oid)
        # remembered set
        assert set(heap.remembered_set()) == {
            ref.old_addr[o] for o in ref.remembered}
        # every slot of every object
        self.addr_to_name = {}
        for name, oid in self.oids.items():
            if oid not in ref.objs:
                continue
            r = ref.objs[oid]
            h = heap.handle(r.addr, r.gen)
            self.addr_to_name[(r.gen, r.addr)] = name
            assert heap.payload(h) == r.payload
            ns, nw, ne = heap.object_shape(h)
            assert (ns, nw, ne) == (len(r.strong), len(r.weak), len(r.eph))
            for j, t in enumerate(r.strong):
                got = heap.get_strong(h, j)
                self._check_ref(got, t, f"{name}.strong[{j}]")
            for j, t in enumerate(r.weak):
                got = heap.get_weak(h, j)
                self._check_ref(got, t, f"{name}.weak[{j}]")
            for j, (k, v) in enumerate(r.eph):
                gk, gv = heap.get_ephemeron(h, j)
                self._check_ref(gk, k, f"{name}.eph[{j}].key")
                self._check_ref(gv, v, f"{name}.eph[{j}].value")

    def _check_ref(self, handle, oid, what):
        if oid is None:
            assert handle is None, f"{what}: engine has {handle}, oracle None"
        else:
            assert handle is not None, f"{what}: engine None, oracle has target"
            assert (handle.generation, handle.address) == self.ref.key(oid), (
                f"{what}: engine {handle} vs oracle {self.ref.key(oid)}")

    def _refresh_handles(self):
        """Re-mint engine handles for surviving named objects."""
        for name in list(self.handles):
            oid = self.oids[name]
            if oid not in self.ref.objs:
                del self.handles[name]
                del self.oids[name]
                continue
            r = self.ref.objs[oid]
            self.handles[name] = self.heap.handle(r.addr, r.gen)
