"""Tests for the two-generation incremental collector (moving_gc.GenHeap).

Expected values come from :mod:`reference_gen` (never imports moving_gc):

* :class:`RefGen` gives batch post-collection worlds (minor/major,
  ephemeron fixpoint, finalizer resurrection, promotion, remembered set);
* :class:`MarkLedger` gives per-scan-unit marking/barrier state for
  interleaved write checks.

A name<->address harness mirrors every construction/mutation on both
worlds and compares them after each collection.  Mid-cycle states are
checked against the ledger one scan unit at a time.  A separate
exhaustive 5-object family enumerates strong/weak edge patterns and
barrier interleavings, and a scale test exercises >=2000 objects and
10000 interleaved updates.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_gen import MarkLedger, RefGen  # noqa: E402

from moving_gc import (  # noqa: E402
    OLD,
    YOUNG,
    CollectionInProgressError,
    FinalizerError,
    GenHeap,
    Handle,
    HeapCapacityError,
    SealedForReclamationError,
    StaleHandleError,
)


# --------------------------------------------------------------------------- #
# Dual-world harness
# --------------------------------------------------------------------------- #
class DualWorld:
    """Keep a real GenHeap and an independent RefGen in lockstep."""

    def __init__(self, yc, oc, promote_age=2):
        self.heap = GenHeap(yc, oc, promote_age=promote_age)
        self.ref = RefGen(yc, oc, promote_age=promote_age)
        self.yc = yc
        self.handles: dict[str, Handle] = {}
        # Per-object finalizer call counts (name -> int).
        self.fin_calls: dict[str, int] = {}

    # -- construction ----------------------------------------------------- #
    def alloc(self, name, payload=0, *, n_strong=0, n_weak=0):
        self.handles[name] = self.heap.allocate(
            payload, n_strong=n_strong, n_weak=n_weak)
        self.ref.allocate(name, payload, n_strong=n_strong, n_weak=n_weak)

    def set_strong(self, src, i, dst):
        self.heap.set_strong(self.handles[src], i,
                             None if dst is None else self.handles[dst])
        self.ref.set_strong(src, i, dst)

    def set_weak(self, src, i, dst):
        self.heap.set_weak(self.handles[src], i,
                           None if dst is None else self.handles[dst])
        self.ref.set_weak(src, i, dst)

    def add_root(self, rn, target):
        self.heap.add_root(rn, self.handles[target])
        self.ref.add_root(rn, target)

    def remove_root(self, rn):
        self.heap.remove_root(rn)
        self.ref.remove_root(rn)

    def clear_root(self, rn):
        self.heap.clear_root(rn)
        self.ref.clear_root(rn)

    def add_eph(self, table, key, value):
        vid = None if value is None else self.handles[value]
        eid = self.heap.add_ephemeron(self.handles[table], self.handles[key], vid)
        reid = self.ref.add_ephemeron(table, key, value)
        assert eid == reid
        return eid

    def set_finalizer(self, target, mode="noop", real_cb=None, ref_cb=None):
        calls = self.fin_calls

        def _real_cb(hp, ho):
            calls[target] = calls.get(target, 0) + 1
            if mode == "resurrect":
                hp.add_root(f"rev_{target}", ho)

        def _ref_cb(rf, name):
            calls.setdefault("__ref_" + target, 0)
            calls["__ref_" + target] += 1
            if mode == "resurrect":
                rf.add_root(f"rev_{target}", name)

        real_cb = real_cb or _real_cb
        ref_cb = ref_cb or _ref_cb
        token = self.heap.set_finalizer(self.handles[target], real_cb)
        self.ref.set_finalizer(target, ref_cb)
        # Align the oracle's token with the heap's single token space.
        self.ref.fin[target] = (self.ref.fin[target][0], token)
        return token

    def request_promotion(self, target):
        self.heap.request_promotion(self.handles[target])
        self.ref.request_promotion(target)

    # -- collection + comparison ----------------------------------------- #
    def pre_addrs(self):
        return {n: h.address for n, h in self.handles.items()}

    def run(self, kind, budget=64):
        pre = self.pre_addrs()
        if kind == "minor":
            result = self.heap.run_minor(budget)
            exp = self.ref.expected_minor()
            self.ref.commit_minor(exp)
        else:
            result = self.heap.run_major(budget)
            exp = self.ref.expected_major()
            self.ref.commit_major(exp)
        self._assert_matches(result, exp, pre)
        self._refresh_handles(exp)
        return result, exp

    def post_addr_map(self, exp, pre):
        """name -> post logical address for survivors (both worlds)."""
        m = {}
        for n, pre_addr in pre.items():
            if n in exp.addr_of:
                m[n] = exp.addr_of[n]
            elif n in exp.relocated:
                m[n] = exp.relocated[n]
            elif n in exp.promoted:
                m[n] = exp.promoted[n]
        return m

    def _assert_matches(self, result, exp, pre):
        # Survivor address sets.
        real_addrs = set(self.heap.objects())
        exp_addrs = set(exp.addr_of.values())
        assert real_addrs == exp_addrs, (real_addrs, exp_addrs)
        # relocated / promoted maps keyed by pre young addresses.
        exp_rel = {pre[n]: a for n, a in exp.relocated.items()}
        exp_prom = {pre[n]: a for n, a in exp.promoted.items()}
        assert result.relocated == exp_rel, (result.relocated, exp_rel)
        assert result.promoted == exp_prom, (result.promoted, exp_prom)
        # occupancy
        assert result.live_young_slots == exp.young_used, \
            (result.live_young_slots, exp.young_used)
        assert result.live_old_slots == exp.old_used, \
            (result.live_old_slots, exp.old_used)
        # weak clears: translate (name,field) -> (addr,field)
        exp_cw = tuple(sorted((exp.addr_of[n], j) for n, j in exp.cleared_weak))
        assert result.cleared_weak_edges == exp_cw, \
            (result.cleared_weak_edges, exp_cw)
        # ephemeron removals
        assert tuple(sorted(result.cleared_ephemerons)) == \
            tuple(sorted(exp.cleared_eph))
        # finalizer events (generation, address, token, resurrected).
        # Finalizers are delivered on stable old addresses (mark-sweep
        # never moves), so the event address is the pre-collection one
        # whether the object is later resurrected or reclaimed.
        rf = tuple(sorted(
            (g, a, tok, res) for (g, a, tok, res) in result.finalized))
        ef = tuple(sorted(
            (1, pre[n], tok, res) for (n, tok, res) in exp.finalized))
        assert rf == ef, (rf, ef)
        # roots
        for rn, t in exp.roots.items():
            rh = self.heap.root_handle(rn)
            if t is None:
                assert rh is None, rn
            else:
                assert rh is not None and rh.address == exp.addr_of[t], rn
        # edges of every survivor
        for n, addr in exp.addr_of.items():
            h = self.heap.handle(addr)
            ns = len(exp.strong[n])
            nw = len(exp.weak[n])
            assert self.heap.object_shape(h) == (ns, nw), n
            for j, t in enumerate(exp.strong[n]):
                got = self.heap.get_strong(h, j)
                if t is None:
                    assert got is None, (n, j)
                else:
                    assert got is not None and got.address == exp.addr_of[t], \
                        (n, j, t, got)
            for j, t in enumerate(exp.weak[n]):
                got = self.heap.get_weak(h, j)
                if t is None:
                    assert got is None, (n, j)
                else:
                    assert got is not None and got.address == exp.addr_of[t], \
                        (n, j, t, got)

    def _refresh_handles(self, exp):
        alive = {}
        for n, addr in exp.addr_of.items():
            alive[n] = self.heap.handle(addr)
        # Drop handles to reclaimed names, keep survivors.
        self.handles = alive


# --------------------------------------------------------------------------- #
# Batch differential tests vs the independent oracle
# --------------------------------------------------------------------------- #
class TestMinorOracle(unittest.TestCase):
    def test_basic_minor_compaction_and_ages(self):
        w = DualWorld(60, 120, promote_age=3)
        w.alloc("a", 1, n_strong=1)
        w.alloc("g", 9)            # garbage
        w.alloc("b", 2)
        w.set_strong("a", 0, "b")
        w.add_root("r", "a")
        w.run("minor")
        rb = w.handles["b"]
        self.assertEqual(w.heap.age_of(w.handles["a"]), 1)
        self.assertEqual(w.heap.age_of(rb), 1)
        self.assertNotIn("g", w.handles)
        # Two more minors promote at age 3.
        w.run("minor")
        w.run("minor")
        self.assertEqual(w.heap.generation_of(w.handles["a"]), OLD)
        self.assertEqual(w.heap.generation_of(w.handles["b"]), OLD)

    def test_age_promotion_boundary(self):
        for promote_age in (1, 2, 3):
            w = DualWorld(80, 200, promote_age=promote_age)
            w.alloc("a", 0, n_strong=1)
            w.alloc("b", 0)
            w.set_strong("a", 0, "b")
            w.add_root("r", "a")
            for _ in range(promote_age):
                w.run("minor")
            self.assertEqual(w.heap.generation_of(w.handles["a"]), OLD)
            self.assertEqual(w.heap.generation_of(w.handles["b"]), OLD)

    def test_forced_promotion(self):
        w = DualWorld(80, 200, promote_age=5)
        w.alloc("a", 0)
        w.add_root("r", "a")
        w.request_promotion("a")
        w.run("minor")
        self.assertEqual(w.heap.generation_of(w.handles["a"]), OLD)

    def test_remembered_set_old_to_young(self):
        w = DualWorld(80, 200, promote_age=1)
        w.alloc("o", 0, n_strong=2, n_weak=1)
        w.add_root("ro", "o")
        w.run("minor")                 # o promoted
        # New young objects; only the strong old->t edge keeps t alive.
        w.alloc("garbage", 5)
        w.alloc("t", 7)
        w.alloc("w", 8)
        w.set_strong("o", 0, "t")
        w.set_strong("o", 1, None)
        w.set_weak("o", 0, "w")        # weak: w dies, edge cleared
        w.run("minor")
        self.assertEqual(w.heap.generation_of(w.handles["t"]), OLD)
        self.assertNotIn("w", w.handles)
        self.assertNotIn("garbage", w.handles)
        oh = w.handles["o"]
        self.assertIsNone(w.heap.get_weak(oh, 0))


class TestMajorOracle(unittest.TestCase):
    def test_major_reclaims_old_garbage_and_clears_weak(self):
        w = DualWorld(60, 200, promote_age=1)
        w.alloc("a", 1, n_strong=1, n_weak=2)
        w.alloc("dead", 2)
        w.alloc("live", 3)
        w.set_strong("a", 0, "live")
        w.set_weak("a", 0, "dead")
        w.set_weak("a", 1, "live")
        w.add_root("r", "a")
        w.add_root("rl", "live")
        w.run("minor")                 # all promoted
        w.run("major")
        self.assertNotIn("dead", w.handles)
        ah = w.handles["a"]
        self.assertIsNone(w.heap.get_weak(ah, 0))
        self.assertEqual(w.heap.get_weak(ah, 1), w.handles["live"])

    def test_major_frees_old_slots_for_reuse(self):
        w = DualWorld(40, 80, promote_age=1)
        w.alloc("x", 0, n_strong=4)    # large old object later reclaimed
        w.add_root("rx", "x")
        w.run("minor")
        xaddr = w.handles["x"].address
        self.assertEqual(w.heap.old_used_slots, 6)
        w.remove_root("rx")
        w.run("major", budget=1)
        self.assertEqual(w.heap.old_used_slots, 0)
        # Promote a new object; it should reuse the freed low window.
        w.alloc("y", 0, n_strong=4)
        w.add_root("ry", "y")
        w.run("minor")
        self.assertEqual(w.handles["y"].address, xaddr)


class TestEphemerons(unittest.TestCase):
    def _old_world(self, specs):
        """Build objects from (name,payload,ns,nw), force all to old.

        Returns the DualWorld with every object promoted; edges, roots and
        ephemerons are configured afterwards on stable old addresses.
        """
        w = DualWorld(200, 300, promote_age=10)
        names = []
        for name, payload, ns, nw in specs:
            w.alloc(name, payload, n_strong=ns, n_weak=nw)
            names.append(name)
        for k, n in enumerate(names):
            w.request_promotion(n)
            w.add_root(f"_anchor{k}", n)
        w.run("minor")
        for k in range(len(names)):
            w.remove_root(f"_anchor{k}")
        return w

    def test_chain_requires_fixpoint(self):
        w = self._old_world([
            ("T", 0, 1, 0), ("K1", 0, 0, 0), ("K2", 0, 0, 0), ("V", 0, 0, 0)])
        w.set_strong("T", 0, "K1")
        w.add_root("rT", "T")
        # e1: T/K1 -> K2 ; e2: T/K2 -> V.  Both must activate via LFP.
        w.add_eph("T", "K1", "K2")
        w.add_eph("T", "K2", "V")
        w.run("major")
        for n in ("T", "K1", "K2", "V"):
            self.assertIn(n, w.handles, n)

    def test_ephemeron_does_not_keep_key_alive(self):
        w = self._old_world([("T", 0, 0, 0), ("K", 0, 0, 0), ("V", 0, 0, 0)])
        w.add_root("rT", "T")
        w.add_eph("T", "K", "V")
        w.run("major")
        self.assertIn("T", w.handles)
        self.assertNotIn("K", w.handles)
        self.assertNotIn("V", w.handles)

    def test_mutual_key_value_references(self):
        # K and V reference each other strongly but are otherwise
        # unreachable; the ephemeron must not resurrect them.
        w = self._old_world([
            ("T", 0, 0, 0), ("K", 0, 1, 0), ("V", 0, 1, 0)])
        w.set_strong("K", 0, "V")
        w.set_strong("V", 0, "K")
        w.add_root("rT", "T")
        w.add_eph("T", "K", "V")
        w.run("major")
        self.assertNotIn("K", w.handles)
        self.assertNotIn("V", w.handles)

    def test_ephemeron_value_reaches_key_of_another(self):
        # Table alive; K1 alive (rooted). e1 activates -> V1 alive;
        # V1 strongly references K2; e2 table/K2 -> V2 then activates.
        w = self._old_world([
            ("T", 0, 0, 0), ("K1", 0, 0, 0), ("V1", 0, 1, 0),
            ("K2", 0, 0, 0), ("V2", 0, 0, 0)])
        w.set_strong("V1", 0, "K2")
        w.add_root("rT", "T")
        w.add_root("rK1", "K1")
        w.add_eph("T", "K1", "V1")
        w.add_eph("T", "K2", "V2")
        w.run("major")
        for n in ("T", "K1", "V1", "K2", "V2"):
            self.assertIn(n, w.handles, n)

    def test_minor_old_table_key_keeps_young_value(self):
        # A minor never reclaims the old generation: an ephemeron whose
        # table and key are live OLD objects activates even though neither
        # is in the young-only closure, so its young value survives the
        # minor; an unrelated young garbage object does not.
        w = DualWorld(120, 200, promote_age=1)
        w.alloc("T", 0)
        w.alloc("K", 0)
        w.add_root("rT", "T")
        w.add_root("rK", "K")
        w.run("minor")                 # T, K -> old
        # Young value reachable only through (old T, old K) -> young V.
        w.alloc("V", 0)
        w.alloc("garbage", 0)
        w.add_eph("T", "K", "V")
        w.run("minor", budget=1)
        self.assertIn("V", w.handles)
        self.assertNotIn("garbage", w.handles)


class TestFinalizers(unittest.TestCase):
    def _oldified(self, w, *names):
        for n in names:
            pass
        w.run("minor")

    def test_resurrection_keeps_object_and_runs_once(self):
        w = DualWorld(60, 200, promote_age=1)
        w.alloc("a", 0, n_strong=1)
        w.alloc("g", 0)
        w.set_strong("a", 0, "g")
        w.add_root("r", "a")
        self._oldified(w)
        w.set_finalizer("g", mode="resurrect")
        w.set_strong("a", 0, None)
        result, _ = w.run("major", budget=1)
        self.assertEqual(len(result.finalized), 1)
        self.assertTrue(result.finalized[0][3])     # resurrected
        self.assertIn("g", w.handles)
        self.assertEqual(w.fin_calls["g"], 1)
        # Drop the resurrection root: object dies but never refinalizes.
        w.remove_root("rev_g")
        result, _ = w.run("major", budget=1)
        self.assertEqual(result.finalized, ())
        self.assertNotIn("g", w.handles)

    def test_no_resurrection_reclaims_after_one_event(self):
        w = DualWorld(60, 200, promote_age=1)
        w.alloc("a", 0, n_strong=1, n_weak=1)
        w.alloc("g", 0)
        w.set_strong("a", 0, "g")     # keeps g alive through promotion
        w.set_weak("a", 0, "g")
        w.add_root("r", "a")
        self._oldified(w)
        # g is now old; detach the strong edge (weak remains) and arm a
        # finalizer that does not resurrect it.
        w.set_strong("a", 0, None)
        w.set_finalizer("g", mode="noop")
        result, _ = w.run("major", budget=1)
        self.assertEqual(result.finalized[0][3], False)
        self.assertNotIn("g", w.handles)
        # Weak edge cleared in the same cycle after finalization.
        self.assertIsNone(w.heap.get_weak(w.handles["a"], 0))

    def test_allocation_during_finalization(self):
        # The finalizer allocates a fresh nursery object and roots it.
        w = DualWorld(120, 200, promote_age=1)
        w.alloc("a", 0, n_strong=1)
        w.alloc("g", 0)
        w.set_strong("a", 0, "g")
        w.add_root("r", "a")
        self._oldified(w)
        w.set_strong("a", 0, None)

        calls = []

        def real_cb(hp, ho):
            calls.append(1)
            new = hp.allocate(777)
            hp.add_root("born_in_fin", new)

        def ref_cb(rf, name):
            calls.append("ref")
            rf.allocate("born_in_fin", 777)
            rf.add_root("born_in_fin", "born_in_fin")

        w.set_finalizer("g", real_cb=real_cb, ref_cb=ref_cb)
        result, exp = w.run("major", budget=1)
        self.assertEqual(len(calls), 2)         # real + ref each once
        self.assertIn("born_in_fin", w.handles)
        self.assertEqual(w.heap.payload(w.handles["born_in_fin"]), 777)
        self.assertEqual(w.heap.generation_of(w.handles["born_in_fin"]), YOUNG)


# --------------------------------------------------------------------------- #
# Incremental IU-barrier ledger, lockstep per scan unit
# --------------------------------------------------------------------------- #
class TestIncrementalBarrierLedger(unittest.TestCase):
    def _build(self, seed, promote_age=10):
        rng = random.Random(seed)
        yc, oc = 200, 200
        w = DualWorld(yc, oc, promote_age=promote_age)
        names = [f"n{i}" for i in range(7)]
        for n in names:
            ns = rng.randrange(0, 3)
            nw = rng.randrange(0, 2)
            w.alloc(n, rng.randrange(0, 100), n_strong=ns, n_weak=nw)
        for n in names:
            shape_ns = w.ref.objs[n].strong.__len__()
            for j in range(len(w.ref.objs[n].strong)):
                if rng.random() < 0.6:
                    w.set_strong(n, j, rng.choice(names))
        for rn in ("r0", "r1"):
            t = rng.choice(names)
            w.add_root(rn, t)
        return w, names

    def test_major_mark_matches_ledger_with_interleaved_writes(self):
        for seed in range(30):
            w, names = self._build(1000 + seed)
            heap, ref = w.heap, w.ref

            def name_addrs(name_set):
                # Major mark-sweep never moves objects, so current handles
                # name the same addresses throughout the open phases.
                return {w.handles[n].address for n in name_set if n in w.handles}

            r0 = heap.collect_step(1, "major")
            ledger = MarkLedger(
                "major", list(ref.root_order),
                gen_of=ref.gen_of,
                root_of=lambda rn: ref.roots[rn],
                edges_of=lambda n: list(ref.objs[n].strong),
            )
            max_steps = 5000
            steps = 0
            scheduled = [(names[(seed + k) % len(names)],
                          names[(seed + 2 * k + 1) % len(names)]) for k in range(6)]
            sched_i = 0
            leaving_phase = "mark"
            while not r0.completed and steps < max_steps:
                steps += 1
                # Advance the ledger by the units this step actually spent
                # in marking.  Empty phases (no candidates / no ephemerons)
                # can be crossed for zero units, so the ledger stops itself
                # once its enumeration and grey queue are both exhausted.
                mark_units = r0.scanned if leaving_phase == "mark" else 0
                for _ in range(mark_units):
                    if not ledger.enum_done() or ledger.gray_q:
                        ledger.step_unit()
                leaving_phase = r0.phase
                if leaving_phase in ("mark", "fdiscover"):
                    self.assertEqual(set(heap.gray_pending()),
                                     name_addrs(ledger.gray()))
                    self.assertEqual(set(heap.marked_addresses()),
                                     name_addrs(ledger.marked()))
                    if (sched_i < len(scheduled) and ledger.gray() == ()
                            and steps % 3 == 0):
                        holder, target = scheduled[sched_i]
                        sched_i += 1
                        hh = w.handles[holder]
                        if heap.object_shape(hh)[0] > 0:
                            ref.set_strong(holder, 0, target)
                            ledger.write_strong(holder, 0, target)
                            heap.set_strong(hh, 0, w.handles[target])
                r0 = heap.collect_step(1)
            self.assertTrue(r0.completed)
            exp = ref.expected_major()
            ref.commit_major(exp)
            self.assertEqual(set(heap.objects()), set(exp.addr_of.values()))
            w._refresh_handles(exp)

    def test_minor_barrier_greys_old_to_young_write(self):
        w = DualWorld(200, 200, promote_age=10)
        w.alloc("o", 0, n_strong=1)
        w.add_root("ro", "o")
        w.request_promotion("o")
        w.run("minor")                       # o -> old
        # Allocate y *after* promotion: it is white young garbage with no
        # young root, until the cross-generational write logs it.
        w.alloc("y", 0)
        y_pre = w.handles["y"].address

        r = w.heap.collect_step(1, "minor")
        ledger = MarkLedger(
            "minor", list(w.ref.root_order), gen_of=w.ref.gen_of,
            root_of=lambda rn: w.ref.roots[rn],
            edges_of=lambda n: list(w.ref.objs[n].strong),
            rs_fields=[], fin_roots=[])
        if r.phase == "mark":
            for _ in range(r.scanned):
                ledger.step_unit()
        w.set_strong("o", 0, "y")
        ledger.write_strong("o", 0, "y")

        def name_addrs(name_set):
            return {w.handles[n].address for n in name_set if n in w.handles}

        self.assertEqual(set(w.heap.gray_pending()), name_addrs(ledger.gray()))
        self.assertIn("y", ledger.gray())
        result = w.heap.run_minor(64)
        # y survived (stays young, age 0->1) solely via the cross-gen write.
        self.assertIn(y_pre, result.relocated)


class TestExhaustiveFiveObjects(unittest.TestCase):
    def test_exhaustive_strong_weak_barrier(self):
        # Small reference domain: exactly 5 objects, all force-promoted to
        # the old generation so a major mark-sweep gives a fully checkable
        # result on stable addresses.  Interleaved writes are *additive
        # only* (null slot -> edge), which is exactly what incremental
        # update handles precisely: the final marked set must equal an
        # independent BFS over the final edge matrix -- no unknowns.
        cases = 0
        for seed in range(240):
            rng = random.Random(90000 + seed)
            yc, oc = 200, 200
            heap = GenHeap(yc, oc)
            names = [f"o{i}" for i in range(5)]
            NS, NW = 2, 1
            handles = {n: heap.allocate(rng.randrange(0, 50),
                                        n_strong=NS, n_weak=NW) for n in names}
            for n in names:
                heap.request_promotion(handles[n])
                heap.add_root(f"_anchor_{n}", handles[n])
            heap.run_minor(8)            # all five -> old, addresses freeze
            self.assertEqual(set(heap.objects(YOUNG)), set())
            # Addresses moved to the old generation; rebuild handles via
            # the still-installed anchor roots, then drop the anchors.
            handles = {n: heap.root_handle(f"_anchor_{n}") for n in names}
            addr = {n: handles[n].address for n in names}
            for n in names:
                heap.remove_root(f"_anchor_{n}")

            strong = {n: [None] * NS for n in names}
            weak = {n: [None] * NW for n in names}

            def put_strong(src, j, tgt):
                heap.set_strong(handles[src], j, handles[tgt])
                strong[src][j] = tgt

            def put_weak(src, j, tgt):
                heap.set_weak(handles[src], j, handles[tgt])
                weak[src][j] = tgt

            for n in names:
                for j in range(NS):
                    if rng.random() < 0.5:
                        put_strong(n, j, names[rng.randrange(0, 5)])
                for j in range(NW):
                    if rng.random() < 0.5:
                        put_weak(n, j, names[rng.randrange(0, 5)])
            root_names = []
            nroots = rng.randrange(1, 4)
            for k in range(nroots):
                rn = f"r{k}"
                t = names[rng.randrange(0, 5)]
                heap.add_root(rn, handles[t])
                root_names.append(rn)

            def bfs():
                roots = {_root_target(heap, rn, addr, names) for rn in root_names}
                seen, stk = set(), list(roots)
                while stk:
                    x = stk.pop()
                    if x in seen:
                        continue
                    seen.add(x)
                    stk.extend(t for t in strong[x] if t is not None)
                return seen

            # Pre-plan additive-only barrier actions (null strong slot).
            actions = []
            for _ in range(6):
                src = names[rng.randrange(0, 5)]
                nulls = [j for j in range(NS) if strong[src][j] is None]
                if not nulls:
                    continue
                tgt = names[rng.randrange(0, 5)]
                actions.append((src, rng.choice(nulls), tgt))
            fire_on = {rng.randrange(1, 5) for _ in range(len(actions))}

            step_no = 0
            ai = 0
            res = heap.collect_step(1, "major")
            while not res.completed:
                step_no += 1
                if ai < len(actions) and step_no in fire_on and \
                        res.phase in ("mark", "fdiscover"):
                    src, j, tgt = actions[ai]
                    if strong[src][j] is None:      # additive-only
                        put_strong(src, j, tgt)
                        ai += 1
                res = heap.collect_step(rng.choice((1, 1, 2)))
            expected = bfs()
            exp_addr = {addr[n] for n in expected}

            # (1) old object set exactly equals strong reachability.
            self.assertEqual(set(heap.objects(OLD)), exp_addr, seed)
            self.assertEqual(set(heap.objects(YOUNG)), set(), seed)
            # (2) independent in-heap traversal (no collector call) agrees.
            self.assertEqual(_heap_bfs(heap, root_names), exp_addr, seed)
            # (3) no dangling strong edges; weak edges forwarded or cleared.
            # Major advanced the epoch, so re-mint handles at stable old
            # addresses; dead names are absent and must not be dereferenced.
            for n in names:
                alive = n in expected
                if not alive:
                    self.assertNotIn(addr[n], heap.objects(OLD), seed)
                    continue
                h = heap.handle(addr[n])
                for j in range(NS):
                    got = heap.get_strong(h, j)
                    t = strong[n][j]
                    if t is None:
                        self.assertIsNone(got, (seed, n, j))
                    else:
                        self.assertIn(t, expected, (seed, n, j))
                        self.assertIsNotNone(got)
                        self.assertEqual(got.address, addr[t], (seed, n, j))
                for j in range(NW):
                    got = heap.get_weak(h, j)
                    t = weak[n][j]
                    if t is None or t not in expected:
                        self.assertIsNone(got, (seed, n, j))
                    else:
                        self.assertIsNotNone(got, (seed, n, j))
                        self.assertEqual(got.address, addr[t], (seed, n, j))
            cases += 1
        self.assertGreater(cases, 200)


def _root_target(heap, root_name, addr, names):
    h = heap.root_handle(root_name)
    for n in names:
        if addr[n] == h.address:
            return n
    raise AssertionError("root address not found")


def _heap_bfs(heap, root_names):
    """Independent strong traversal using only public heap reads."""
    seen, stk = set(), []
    for rn in root_names:
        h = heap.root_handle(rn)
        if h is not None:
            stk.append(h)
    while stk:
        h = stk.pop()
        if h.address in seen:
            continue
        seen.add(h.address)
        ns, _ = heap.object_shape(h)
        for j in range(ns):
            t = heap.get_strong(h, j)
            if t is not None and t.address not in seen:
                stk.append(t)
    return seen



# --------------------------------------------------------------------------- #
# Budget honesty + determinism
# --------------------------------------------------------------------------- #
class TestBudget(unittest.TestCase):
    def test_scanned_never_exceeds_budget(self):
        rng = random.Random(7)
        w = GenHeap(300, 300)
        hs = [w.allocate(i, n_strong=2, n_weak=1) for i in range(60)]
        for h in hs:
            w.set_strong(h, 0, hs[rng.randrange(0, 60)])
            w.set_strong(h, 1, hs[rng.randrange(0, 60)])
            w.set_weak(h, 0, hs[rng.randrange(0, 60)])
        w.add_root("r", hs[0])
        for kind in ("minor", "major"):
            r = w.collect_step(1, kind)
            while not r.completed:
                self.assertLessEqual(r.scanned, r.budget)
                self.assertEqual(r.overscan, 0)
                self.assertGreaterEqual(r.scanned, 1)
                r = w.collect_step(rng.choice((1, 2, 5)))
            self.assertLessEqual(r.scanned, r.budget)
            self.assertEqual(r.overscan, 0)

    def test_budget_one_always_makes_progress(self):
        w = GenHeap(60, 120, promote_age=1)
        for i in range(8):
            w.allocate(i, n_strong=2)
        w.add_root("r", w.handle(0))
        guard = 0
        r = w.collect_step(1, "major")
        while not r.completed:
            guard += 1
            r = w.collect_step(1)
            self.assertLess(guard, 100000)
        self.assertTrue(r.completed)

    def test_determinism_same_input_same_result(self):
        def build_and_run(seed):
            rng = random.Random(seed)
            w = GenHeap(120, 240, promote_age=2)
            hs = [w.allocate(i, n_strong=2, n_weak=1) for i in range(12)]
            for i, h in enumerate(hs):
                w.set_strong(h, 0, hs[rng.randrange(0, 12)])
                w.set_strong(h, 1, hs[rng.randrange(0, 12)])
                w.set_weak(h, 0, hs[rng.randrange(0, 12)])
            w.add_root("r", hs[0])
            out = []
            for kind in ("minor", "minor", "major"):
                r = (w.run_minor(3) if kind == "minor" else w.run_major(3))
                out.append((sorted(r.relocated.items()),
                            sorted(r.promoted.items()),
                            r.cleared_weak_edges, r.cleared_ephemerons,
                            tuple(r.finalized)))
            return out
        self.assertEqual(build_and_run(42), build_and_run(42))

    def test_kind_mismatch_midcycle_rejected(self):
        w = GenHeap(40, 80)
        w.allocate(0)
        w.collect_step(1, "minor")
        with self.assertRaises(CollectionInProgressError):
            w.collect_step(1, "major")

    def test_bad_budget(self):
        w = GenHeap(40, 80)
        with self.assertRaises(ValueError):
            w.collect_step(0, "minor")
        with self.assertRaises(TypeError):
            w.collect_step(True, "minor")
        with self.assertRaises(ValueError):
            w.collect_step(1, "full")


# --------------------------------------------------------------------------- #
# Handles, sealing, input validation
# --------------------------------------------------------------------------- #
class TestHandlesAndValidation(unittest.TestCase):
    def test_stale_handle_after_collection(self):
        w = GenHeap(40, 80, promote_age=2)
        a = w.allocate(1, n_strong=1)
        b = w.allocate(2)
        w.set_strong(a, 0, b)
        w.add_root("r", a)
        w.run_minor(4)
        with self.assertRaises(StaleHandleError):
            w.payload(a)

    def test_dangling_weak_returns_none_not_bad_address(self):
        w = GenHeap(40, 80, promote_age=1)
        a = w.allocate(1, n_weak=1)
        d = w.allocate(2)
        w.set_weak(a, 0, d)
        w.add_root("r", a)
        w.run_minor(2)       # d dies in minor (weak-only), edge cleared
        ah = w.root_handle("r")
        self.assertIsNone(w.get_weak(ah, 0))

    def test_sealed_dead_object_cannot_be_attached(self):
        # Young garbage that never survives a minor is simply stale.
        w = GenHeap(80, 160, promote_age=1)
        x = w.allocate(1, n_strong=1)
        z = w.allocate(2)
        w.add_root("rx", x)
        w.run_minor(1)
        with self.assertRaises(StaleHandleError):
            w.payload(z)

        # Sealed-phase path: two old objects, only a rooted; advance a
        # major to weak clearing/sweeping, then attaching the sealed dead
        # g off the live a must be refused with no dangling pointer left.
        w2 = GenHeap(80, 160, promote_age=1)
        a = w2.allocate(10, n_strong=1)
        g = w2.allocate(20)
        w2.set_strong(a, 0, g)
        w2.add_root("ra", a)
        w2.run_minor(1)               # both promoted to old
        ah = w2.root_handle("ra")
        gh = w2.get_strong(ah, 0)
        gaddr = gh.address
        w2.set_strong(ah, 0, None)    # g becomes old garbage
        r = w2.collect_step(1, "major")
        while not r.completed and r.phase not in ("weakclear", "sweep"):
            r = w2.collect_step(1)
        self.assertIn(r.phase, ("weakclear", "sweep"))
        with self.assertRaises(SealedForReclamationError):
            w2.set_strong(ah, 0, gh)
        self.assertIsNone(w2.get_strong(ah, 0))
        w2.run_major(1)
        with self.assertRaises(StaleHandleError):
            w2.payload(gh)

    def test_double_finalizer_registration_rejected(self):
        w = GenHeap(40, 80, promote_age=1)
        a = w.allocate(1)
        w.add_root("r", a)
        w.run_minor(1)
        ah = w.root_handle("r")
        w.set_finalizer(ah, lambda h, o: None)
        w.remove_root("r")
        w.run_major(1)
        # Object swept; attaching a finalizer is impossible (handle stale).
        with self.assertRaises(StaleHandleError):
            w.set_finalizer(ah, lambda h, o: None)

    def test_finalizer_rerun_after_delivery_rejected(self):
        # Once a finalizer has run (object still live), re-registering a
        # finalizer for it is a FinalizerError: the one-shot event cannot
        # be armed a second time.
        w = GenHeap(40, 80, promote_age=1)
        a = w.allocate(1, n_strong=1)
        g = w.allocate(2)
        w.set_strong(a, 0, g)
        w.add_root("r", a)
        w.run_minor(1)
        ah = w.root_handle("r")
        g = w.get_strong(ah, 0)
        # Run g's finalizer and resurrect it.
        w.set_finalizer(g, lambda hp, ho: hp.add_root("rev", ho))
        w.set_strong(ah, 0, None)
        w.run_major(1)
        self.assertTrue(w.finalizer_ran(w.root_handle("rev")))
        # Event already delivered: re-arming is rejected.
        with self.assertRaises(FinalizerError):
            w.set_finalizer(w.root_handle("rev"), lambda hp, ho: None)

    def test_capacity_bounds(self):
        GenHeap(1, 1)
        with self.assertRaises(ValueError):
            GenHeap(0, 10)
        with self.assertRaises(ValueError):
            GenHeap(10, 0)
        with self.assertRaises(ValueError):
            GenHeap(3000, 2000)     # sum > MAX_CAPACITY 4096
        with self.assertRaises(TypeError):
            GenHeap(True, 10)

    def test_object_larger_than_young(self):
        w = GenHeap(4, 80)
        with self.assertRaises(HeapCapacityError):
            w.allocate(0, n_strong=4)    # size 6 > 4

    def test_midcycle_finalizer_registration_rejected(self):
        w = GenHeap(40, 80)
        a = w.allocate(1)
        w.add_root("r", a)
        w.collect_step(1, "major")
        with self.assertRaises(CollectionInProgressError):
            w.set_finalizer(a, lambda h, o: None)


# --------------------------------------------------------------------------- #
# Scale: >= 2000 objects, 10000 interleaved updates
# --------------------------------------------------------------------------- #
class TestScale(unittest.TestCase):
    def test_2000_objects_10000_interleaved_updates(self):
        # Non-toy main-scale scenario within the 4096-slot interface
        # boundary: a permanently rooted old backbone, waves of nursery
        # objects (cumulatively well over 2000 allocations), 10000
        # successful strong-edge rewrites interleaved with budgeted
        # minor/major collections, exercising the remembered set and
        # promotion on a real, sparse live graph.
        yc, oc = 2000, 2090
        w = GenHeap(yc, oc, promote_age=2)
        rng = random.Random(20260924)

        BB = 40                       # backbone objects, 3 strong edges each
        backbone = []
        for i in range(BB):
            h = w.allocate(1000 + i, n_strong=3, n_weak=1)
            backbone.append(h)
            w.add_root(f"bb{i}", h)
        w.run_minor(128)
        w.run_minor(128)              # backbone promoted to old
        backbone = [w.root_handle(f"bb{i}") for i in range(BB)]

        def refresh_backbone_targets():
            """Return one current handle reachable per backbone node."""
            out = []
            for i in range(BB):
                bh = w.root_handle(f"bb{i}")
                out.append(bh)
            return out

        allocations = BB
        updates = 0
        waves = 12
        per_wave = 200               # 12*200 = 2400 nursery allocations
        target_updates = 10000
        per_wave_updates = target_updates // waves

        for wave in range(waves):
            # Allocate a wave of minimum-ish objects in the nursery.
            wave_handles = []
            for k in range(per_wave):
                h = w.allocate(wave * per_wave + k, n_strong=1)
                wave_handles.append(h)
            allocations += per_wave

            # Attach a random subset to the backbone (old -> young edges
            # enter the remembered set); chain a few wave objects strongly.
            attached = set()
            for k, h in enumerate(wave_handles):
                if rng.random() < 0.35:
                    bi = rng.randrange(BB)
                    slot = rng.randrange(3)
                    w.set_strong(backbone[bi], slot, h)
                    attached.add(k)
                if k and rng.random() < 0.2:
                    w.set_strong(wave_handles[k - 1], 0, h)

            # Interleaved rewrites between/within collection steps.
            done = 0
            while done < per_wave_updates:
                b = rng.randrange(BB)
                slot = rng.randrange(3)
                kind = rng.random()
                try:
                    if kind < 0.6 and wave_handles:
                        t = wave_handles[rng.randrange(per_wave)]
                        w.set_strong(backbone[b], slot, t)
                    else:
                        t2 = backbone[rng.randrange(BB)]
                        w.set_strong(backbone[b], slot, t2)
                    updates += 1
                    done += 1
                except StaleHandleError:
                    break
                if done % 250 == 0:
                    w.run_minor(128)
                    backbone = refresh_backbone_targets()
                    # Wave objects that survived moved; the remaining
                    # rewrites in this wave target the (old) backbone, so
                    # stale wave handles are simply abandoned.
                    wave_handles = []

            w.run_minor(128)
            backbone = refresh_backbone_targets()
            if wave % 4 == 3:
                w.run_major(128)
                backbone = refresh_backbone_targets()

        # Drain the remainder to hit exactly 10000 updates.
        while updates < target_updates:
            b = rng.randrange(BB)
            t2 = backbone[rng.randrange(BB)]
            w.set_strong(backbone[b], rng.randrange(3), t2)
            updates += 1

        w.run_major(128)
        self.assertGreaterEqual(allocations, 2000 + BB)
        self.assertEqual(updates, 10000)
        # Backbone fully present, old, and self-consistent.
        self.assertEqual(len(w.objects(OLD)), BB)
        for i in range(BB):
            rh = w.root_handle(f"bb{i}")
            self.assertIsNotNone(rh)
            self.assertEqual(w.generation_of(rh), OLD)
            self.assertEqual(w.payload(rh), 1000 + i)
            for j in range(3):
                t = w.get_strong(rh, j)
                if t is not None:
                    self.assertIn(t.address, w.objects())
        # Independent strong-reachability traversal agrees.
        live = _heap_bfs(w, [f"bb{i}" for i in range(BB)])
        self.assertEqual(live, set(w.objects(OLD)) | (live & set(w.objects(YOUNG))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
