"""Fast regression tests for the v2 generational engine.

Every semantic assertion here is cross-checked against the independent
oracle through the :class:`Twin` lockstep driver (see tests/gen_harness.py);
the engine's own progress flags are never trusted.  The heavyweight
exhaustive/scale suites live under ``verification/`` and are run separately.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_harness import Twin  # noqa: E402
from gen_reference import OLD, YOUNG, RefModel  # noqa: E402

from moving_gc import (  # noqa: E402
    GenerationalHeap,
    GCError,
    Handle,
    HeapCapacityError,
    StaleHandleError,
)


def drive_steps(twin, kind, budget, finalizer=None, budgets=None):
    """Drive a cycle manually with per-step budgets (default one budget)."""
    twin.begin(kind)
    i = 0
    while True:
        phase = twin.heap.cycle_phase()
        b = budgets[i % len(budgets)] if budgets else budget
        i += 1
        if phase == "marking":
            twin.mark_step(b)
        elif phase == "finalizing":
            twin.finalize_step(finalizer)
        elif phase in ("remark", "clearing"):
            r = twin.finish(b)
            if r.complete:
                return r.result
        else:
            raise AssertionError(f"bad phase {phase}")


class TestGenerationalBasics(unittest.TestCase):
    def test_young_minor_promotes_with_promotion_age_one(self):
        t = Twin(young_capacity=80, old_capacity=80)
        t.allocate("a", 1, n_strong=1)
        t.allocate("b", 2)
        t.set_strong("a", 0, "b")
        t.add_root("r", "a")
        r = t.run_cycle("minor", 4, None)
        self.assertEqual(r.kind, "minor")
        self.assertEqual(r.promoted, 2)
        self.assertEqual(t.heap.used_slots(YOUNG), 0)
        gens = {h.generation for h in t.handles.values()}
        self.assertEqual(gens, {OLD})
        self.assertEqual(t.heap.epoch, 1)

    def test_promotion_age_two_keeps_survivors_young_once(self):
        t = Twin(80, 80, promotion_age=2)
        t.allocate("a", 1)
        t.add_root("r", "a")
        r = t.run_cycle("minor", 4, None)
        self.assertEqual(r.promoted, 0)
        self.assertEqual(t.handles["a"].generation, YOUNG)
        # Second minor: age reaches 2 -> promotes.
        r2 = t.run_cycle("minor", 4, None)
        self.assertEqual(r2.promoted, 1)
        self.assertEqual(t.handles["a"].generation, OLD)

    def test_major_sweeps_old_garbage_and_compacts_free_list(self):
        t = Twin(80, 80)
        t.allocate("keep", 5, generation=OLD)
        t.allocate("trash", 6, generation=OLD)
        t.add_root("rk", "keep")
        r = t.run_cycle("major", 4, None)
        self.assertEqual(r.reclaimed_objects, 1)
        self.assertEqual({n for n, h in t.handles.items() if h is not None},
                         {"keep"})
        # The freed hole is reusable: allocate into the swept span.
        t.allocate("reuse", 7, n_strong=1, generation=OLD)
        self.assertIn((OLD, t.handles["reuse"].address), t.heap.objects())
        t.compare_idle()

    def test_minor_via_run_collection_alias(self):
        heap = GenerationalHeap(40, 40)
        a = heap.allocate(1)
        heap.add_root("r", a)
        r = heap.collect("minor", 8)
        self.assertEqual(r.kind, "minor")
        self.assertEqual(r.promoted, 1)


class TestRememberedSet(unittest.TestCase):
    def test_old_to_young_edge_kept_alive_by_minor(self):
        t = Twin(120, 120)
        # Promote an old holder first.
        t.allocate("holder", 1, n_strong=1)
        t.add_root("rh", "holder")
        t.run_cycle("minor", 4, None)
        # A fresh young target referenced only from the old holder.
        t.allocate("young_t", 9)
        t.set_strong("holder", 0, "young_t")
        self.assertEqual(t.heap.remembered_set(),
                         frozenset({t.handles["holder"].address}))
        r = t.run_cycle("minor", 4, None)
        # young_t must survive (promote) -- a missing RS edge would free it.
        self.assertEqual(r.promoted, 1)
        self.assertIn("young_t", t.handles)
        t.compare_idle()

    def test_overwrite_removes_stale_rs_entry(self):
        t = Twin(120, 120)
        t.allocate("holder", 1, n_strong=1)
        t.allocate("anchor", 2)
        t.add_root("rh", "holder")
        t.add_root("ra", "anchor")
        t.run_cycle("major", 4, None)   # both old
        t.allocate("yt", 3)
        t.set_strong("holder", 0, "yt")
        self.assertIn(t.handles["holder"].address, t.heap.remembered_set())
        t.set_strong("holder", 0, "anchor")  # old -> old now
        self.assertEqual(t.heap.remembered_set(), frozenset())
        t.run_cycle("minor", 4, None)
        # yt was only referenced from holder; it must now be reclaimed.
        self.assertNotIn("yt", t.handles)

    def test_old_newly_written_weak_edge_no_dangling_slot(self):
        t = Twin(160, 160)
        t.allocate("holder", 1, n_weak=1)
        t.add_root("rh", "holder")
        t.run_cycle("major", 4, None)
        t.allocate("yw", 9)
        t.set_weak("holder", 0, "yw")  # weak old -> young; yw otherwise dead
        t.run_cycle("minor", 4, None)
        # The young target died and the old holder's weak slot must be null,
        # never a dangling encoded address into reused young space.
        h = t.handles["holder"]
        self.assertIsNone(t.heap.get_weak(h, 0))
        t.compare_idle()


class TestEphemerons(unittest.TestCase):
    def test_value_rescued_when_key_reachable(self):
        t = Twin(120, 120)
        t.allocate("tab", 0, n_ephemeron=1)
        t.allocate("k", 11)
        t.allocate("v", 22)
        t.set_ephemeron("tab", 0, "k", "v")
        t.add_root("rt", "tab")
        t.add_root("rk", "k")
        r = t.run_cycle("major", 4, None)
        self.assertEqual(len(r.rescued_values), 1)
        self.assertIn("v", t.handles)

    def test_entry_cleared_when_key_dead(self):
        t = Twin(120, 120)
        t.allocate("tab", 0, n_ephemeron=1)
        t.allocate("k", 11)
        t.allocate("v", 22)
        t.set_ephemeron("tab", 0, "k", "v")
        t.add_root("rt", "tab")
        r = t.run_cycle("major", 4, None)
        self.assertEqual(len(r.cleared_ephemerons), 1)
        self.assertNotIn("v", t.handles)
        self.assertNotIn("k", t.handles)
        h = t.handles["tab"]
        self.assertEqual(t.heap.get_ephemeron(h, 0), (None, None))

    def test_chained_ephemerons_reach_fixed_point(self):
        # tab0: k0 -> tab1 ; tab1: k1 -> v1.  Only k0 and k1 rooted: tab1
        # is rescued by tab0's entry, then tab1's entry rescues v1 in a
        # later fixed-point pass.
        t = Twin(200, 200)
        t.allocate("tab1", 0, n_ephemeron=1)
        t.allocate("v1", 1)
        t.allocate("k1", 2)
        t.set_ephemeron("tab1", 0, "k1", "v1")
        t.allocate("tab0", 0, n_ephemeron=1)
        t.allocate("k0", 3)
        t.set_ephemeron("tab0", 0, "k0", "tab1")
        t.add_root("rt0", "tab0")
        t.add_root("rk0", "k0")
        t.add_root("rk1", "k1")
        # budget 1 forces the activation chain across many steps/passes
        r = drive_steps(t, "major", 1)
        self.assertIn("tab1", t.handles)
        self.assertIn("v1", t.handles)
        self.assertEqual(len(r.rescued_values), 2)

    def test_key_value_mutual_references(self):
        # ephemeron (tab): key=k, value=v ; k strongly points at v and v
        # strongly points at k; only tab and k rooted.
        t = Twin(160, 160)
        t.allocate("tab", 0, n_ephemeron=1)
        t.allocate("k", 1, n_strong=1)
        t.allocate("v", 2, n_strong=1)
        t.set_ephemeron("tab", 0, "k", "v")
        t.set_strong("k", 0, "v")
        t.set_strong("v", 0, "k")
        t.add_root("rt", "tab")
        t.add_root("rk", "k")
        r = drive_steps(t, "major", 2)
        self.assertIn("v", t.handles)
        # drop k: now the ephemeron key is dead; the k<->v cycle goes too
        t.remove_root("rk")
        r2 = drive_steps(t, "major", 2)
        self.assertEqual(len(r2.cleared_ephemerons), 1)
        self.assertNotIn("v", t.handles)
        self.assertNotIn("k", t.handles)

    def test_ephemeron_fixed_point_precedes_weak_clearing(self):
        # v is held ONLY by (a) an ephemeron value (key rooted) and
        # (b) a weak edge from the table.  After the fixed point v is live,
        # so the weak edge must be *forwarded*, not cleared.
        t = Twin(160, 160)
        t.allocate("tab", 0, n_ephemeron=1, n_weak=1)
        t.allocate("k", 1)
        t.allocate("v", 2)
        t.set_ephemeron("tab", 0, "k", "v")
        t.set_weak("tab", 0, "v")
        t.add_root("rt", "tab")
        t.add_root("rk", "k")
        r = drive_steps(t, "major", 1)
        self.assertEqual(r.cleared_weak_edges, ())
        self.assertIsNotNone(t.heap.get_weak(t.handles["tab"], 0))


class TestFinalizers(unittest.TestCase):
    def test_resurrection_and_no_double_finalization(self):
        t = Twin(80, 80)
        t.allocate("f", 1)
        t.set_finalizer("f", 100)

        def resurrect(twin, eid, name):
            self.assertEqual(eid, 100)
            twin.add_root("back", name)

        r = t.run_cycle("major", 4, resurrect)
        self.assertEqual(r.finalized, (100,))
        self.assertIn("f", t.handles)
        # Even while rooted the finalizer never runs again.
        r2 = t.run_cycle("major", 4, None)
        self.assertEqual(r2.finalized, ())
        t.remove_root("back")
        r3 = t.run_cycle("major", 4, None)
        self.assertEqual(r3.finalized, ())
        self.assertNotIn("f", t.handles)

    def test_allocation_inside_finalizer_is_born_live(self):
        t = Twin(200, 200)
        t.allocate("f", 1)
        t.set_finalizer("f", 7)

        def allocate_in_finalizer(twin, eid, name):
            twin.allocate("newborn", 42, n_strong=1)
            twin.add_root("rn", "newborn")

        r = t.run_cycle("major", 3, allocate_in_finalizer)
        self.assertEqual(r.finalized, (7,))
        self.assertIn("newborn", t.handles)
        h = t.handles["newborn"]
        self.assertEqual(t.heap.payload(h), 42)
        # The newborn survives the next collection as a normal object.
        r2 = t.run_cycle("major", 4, None)
        self.assertIn("newborn", t.handles)
        self.assertEqual(r2.reclaimed_objects, 0)

    def test_finalizer_events_are_inspectable_between_steps(self):
        t = Twin(120, 120)
        t.allocate("a", 1)
        t.allocate("b", 2)
        t.set_finalizer("a", 1)
        t.set_finalizer("b", 2)
        t.begin("major")
        while t.heap.cycle_phase() == "marking":
            t.mark_step(2)
        pending = t.heap.pending_finalizers()
        self.assertEqual([e.event_id for e in pending], [1, 2])
        t.finalize_step(None)
        self.assertEqual([e.event_id for e in t.heap.pending_finalizers()], [2])
        # drain remark/clearing in phase order
        while t.heap.cycle_phase() in ("remark", "clearing"):
            rep = t.finish(4)
            if rep.complete:
                break

    def test_finalized_event_ids_ledger(self):
        heap = GenerationalHeap(60, 60)
        a = heap.allocate(1)
        b = heap.allocate(2)
        heap.set_finalizer(a, 41)
        heap.set_finalizer(b, 42)
        self.assertEqual(heap.finalized_event_ids(), ())
        r = heap.collect("major", 4)
        self.assertEqual(r.finalized, (41, 42))
        self.assertEqual(heap.finalized_event_ids(), (41, 42))
        # Objects were reclaimed; the ledger persists across cycles.
        heap.collect("major", 4)
        self.assertEqual(heap.finalized_event_ids(), (41, 42))

    def test_reregister_after_run_rejected(self):
        heap = GenerationalHeap(40, 40)
        a = heap.allocate(1)
        heap.set_finalizer(a, 1)
        heap.collect("major", 4)  # unreachable -> finalized and reclaimed
        b = heap.allocate(2)
        with self.assertRaises(GCError):
            heap.set_finalizer(a, 2)  # stale handle anyway
        heap.add_root("r", b)
        with self.assertRaises(TypeError):
            heap.set_finalizer(b, True)


class TestMinorEphemeronAndAllocation(unittest.TestCase):
    def _promote(self, t, name):
        t.add_root("__p_" + name, name)
        t.run_cycle("minor", 4, None)
        t.remove_root("__p_" + name)

    def test_old_table_young_key_ephemeron_minor(self):
        t = Twin(220, 220)
        t.allocate("otab", 0, n_ephemeron=1)
        t.add_root("rot", "otab")
        t.run_cycle("major", 4, None)       # otab now old
        t.allocate("k", 1)
        t.allocate("v", 2)
        t.set_ephemeron("otab", 0, "k", "v")  # old table, young key/value
        t.add_root("rk", "k")
        r = t.run_cycle("minor", 3, None)
        # Old-key rule is conservative; here the key is young and rooted,
        # so the entry resolves exactly and v is rescued and promoted.
        self.assertEqual(len(r.rescued_values), 1)
        self.assertIn("v", t.handles)
        t.compare_idle()

    def test_dead_young_key_ephemeron_cleared_on_minor(self):
        t = Twin(220, 220)
        t.allocate("otab", 0, n_ephemeron=1)
        t.add_root("rot", "otab")
        t.run_cycle("major", 4, None)
        t.allocate("k", 1)  # unreachable young key
        t.allocate("v", 2)  # unreachable young value
        t.set_ephemeron("otab", 0, "k", "v")
        r = t.run_cycle("minor", 3, None)
        self.assertEqual(len(r.cleared_ephemerons), 1)
        h = t.handles["otab"]
        self.assertEqual(t.heap.get_ephemeron(h, 0), (None, None))
        self.assertNotIn("k", t.handles)
        self.assertNotIn("v", t.handles)

    def test_allocation_inside_finalizer_during_minor(self):
        t = Twin(220, 220)
        t.allocate("f", 1)
        t.set_finalizer("f", 55)

        def body(twin, eid, name):
            twin.allocate("nb", 7, n_strong=1)
            twin.add_root("rnb", "nb")

        r = t.run_cycle("minor", 2, body)
        self.assertEqual(r.finalized, (55,))
        self.assertIn("nb", t.handles)
        r2 = t.run_cycle("minor", 2, None)
        self.assertIn("nb", t.handles)
        self.assertEqual(r2.reclaimed_objects, 0)


class TestIncrementalBarrier(unittest.TestCase):
    def test_black_to_white_store_is_caught(self):
        t = Twin(120, 120)
        t.allocate("x", 1, n_strong=1)
        t.allocate("y", 2)
        t.add_root("rx", "x")
        t.begin("major")
        t.mark_step(1)  # x scanned (edge null) -> black
        info = t.heap.cycle_info()
        self.assertNotIn((YOUNG, t.handles["y"].address), set(info.marked))
        t.set_strong("x", 0, "y")  # black -> white
        info = t.heap.cycle_info()
        self.assertIn((YOUNG, t.handles["y"].address), set(info.marked))
        while t.heap.cycle_phase() == "marking":
            t.mark_step(2)
        while t.heap.cycle_phase() == "finalizing":
            t.finalize_step(None)
        rep = t.finish(2)
        while not rep.complete:
            rep = t.finish(2)
        self.assertIn("y", t.handles)

    def test_deleted_edge_before_scan_respected_by_remark(self):
        # An edge present when the holder is scanned marks its target in the
        # initial snapshot round; with no finalizers staged, the remark phase
        # re-derives liveness from the *current* heap.  Deleting the edge
        # before remark runs means the target is reclaimed (no resurrection
        # race exists without a finalizer).
        t = Twin(120, 120)
        t.allocate("x", 1, n_strong=1)
        t.allocate("y", 2)
        t.set_strong("x", 0, "y")
        t.add_root("rx", "x")
        t.begin("major")
        t.mark_step(1)  # snapshot taken, edge charged, y marked gray
        t.set_strong("x", 0, None)  # delete before the remark runs
        while t.heap.cycle_phase() == "marking":
            t.mark_step(1)
        rep = t.finish(1)
        while not rep.complete:
            rep = t.finish(1)
        self.assertNotIn("y", t.handles)


class TestIntersection(unittest.TestCase):
    """The required intersection: IU barrier x ephemeron closure x
    resurrection, all in one interwoven cycle."""

    def test_barrier_ephemeron_resurrection_intersection(self):
        # x : ephemeron table, rooted and scanned black first;
        # z : unreachable object carrying finalizer 99;
        # k/v/w: an ephemeron pair written into the *black* table x while
        #        the cycle is in progress (add root k, then write entry);
        #        v strongly references w, so activation must pull v then w.
        t = Twin(260, 260)
        t.allocate("x", 0, n_ephemeron=1)
        t.allocate("z", 1)
        t.allocate("k", 2)
        t.allocate("v", 3, n_strong=1)
        t.allocate("w", 4)
        t.set_strong("v", 0, "w")
        t.set_finalizer("z", 99)
        t.add_root("rx", "x")

        t.begin("major")
        t.mark_step(1)  # x scanned black; its (empty) eph slot inert
        self.assertEqual(t.heap.cycle_info().gray, ())
        # Mutator interleaving: make k reachable and write the entry into
        # the black table.  The IU ephemeron barrier must activate v now;
        # v is gray, so scanning it pulls w into the closure as well.
        t.add_root("rk", "k")
        t.set_ephemeron("x", 0, "k", "v")
        info = t.heap.cycle_info()
        vkey = (YOUNG, t.handles["v"].address)
        self.assertIn(vkey, set(info.marked))

        def body(twin, eid, name):
            self.assertEqual(eid, 99)
            self.assertEqual(name, "z")
            twin.add_root("zback", name)            # resurrection
            twin.allocate("newborn", 5, n_strong=1)  # allocation in finalizer
            twin.add_root("rn", "newborn")

        while t.heap.cycle_phase() == "marking":
            t.mark_step(1)
        self.assertEqual([e.event_id for e in t.heap.pending_finalizers()], [99])
        t.finalize_step(body)
        rep = t.finish(2)
        while not rep.complete:
            rep = t.finish(2)

        # Everything survived: z via resurrection, k via root, v through
        # the barrier-activated ephemeron, w through v's strong edge, and
        # the object born inside the finalizer.
        for nm in ("x", "z", "k", "v", "w", "newborn"):
            self.assertIn(nm, t.handles, nm)
        hx = t.handles["x"]
        ek, ev = t.heap.get_ephemeron(hx, 0)
        self.assertEqual(ev, t.handles["v"])
        self.assertEqual(t.heap.get_strong(t.handles["v"], 0), t.handles["w"])
        self.assertIn(99, rep.result.finalized)
        # z is never finalized again even if dropped and reclaimed later.
        t.remove_root("zback")
        r2 = drive_steps(t, "major", 2)
        self.assertNotIn("z", t.handles)
        self.assertEqual(r2.finalized, ())


class TestBudgets(unittest.TestCase):
    def test_per_step_scan_never_exceeds_budget(self):
        t = Twin(400, 400)
        t.allocate("root", 0, n_strong=3)
        for i in range(12):
            t.allocate(f"c{i}", i, n_strong=2)
        # A spanning tree over all 12 children so every edge is scanned.
        tree = {0: (3, 4), 1: (5, 6), 2: (7, 8), 3: (9, 10), 4: (11, None)}
        t.set_strong("root", 0, "c0")
        t.set_strong("root", 1, "c1")
        t.set_strong("root", 2, "c2")
        for parent, (x, y) in tree.items():
            if x is not None:
                t.set_strong(f"c{parent}", 0, f"c{x}")
            if y is not None:
                t.set_strong(f"c{parent}", 1, f"c{y}")
        t.add_root("r", "root")
        t.begin("major")
        steps = 0
        while t.heap.cycle_phase() == "marking":
            r = t.mark_step(1)  # one edge per call
            self.assertLessEqual(r.edges_scanned, 1)
            steps += 1
            self.assertLess(steps, 1000)
        while t.heap.cycle_phase() == "finalizing":
            t.finalize_step(None)
        rep = t.finish(1)
        self.assertLessEqual(rep.edges_scanned, 1)
        while not rep.complete:
            rep = t.finish(1)
            self.assertLessEqual(rep.edges_scanned, 1)
        # 3 (root) + 12*2 (children) = 27 strong edges in each of the
        # initial-mark and remark rounds -> 54 minimum.
        self.assertGreaterEqual(rep.result.edges_scanned, 54)

    def test_cycle_info_inspectable_at_every_step(self):
        heap = GenerationalHeap(80, 80)
        a = heap.allocate(1, n_strong=2)
        b = heap.allocate(2)
        heap.set_strong(a, 0, b)
        heap.add_root("r", a)
        heap.begin_collection("minor")
        phases = []
        while heap.cycle_phase() == "marking":
            phases.append(heap.cycle_info().phase)
            heap.mark_step(1)
        self.assertTrue(phases)
        with self.assertRaises(GCError):
            heap.mark_step(1)  # wrong phase raises, not silent no-op
        with self.assertRaises(GCError):
            heap.begin_collection("minor")  # nested begin rejected
        while heap.cycle_phase() == "finalizing":
            heap.finalize_step(None)
        r = heap.finish_collection(1)
        while not r.complete:
            r = heap.finish_collection(1)

    def test_zero_budget_rejected(self):
        heap = GenerationalHeap(40, 40)
        heap.begin_collection("minor")
        with self.assertRaises(ValueError):
            heap.mark_step(0)
        with self.assertRaises(ValueError):
            heap.finish_collection(0)


class TestStaleHandlesAndValidation(unittest.TestCase):
    def test_old_handle_rejected_after_minor(self):
        heap = GenerationalHeap(40, 40)
        a = heap.allocate(1, n_strong=1)
        b = heap.allocate(2)
        heap.set_strong(a, 0, b)
        heap.add_root("r", a)
        heap.collect("minor", 8)
        with self.assertRaises(StaleHandleError):
            heap.payload(a)
        with self.assertRaises(StaleHandleError):
            heap.set_strong(a, 0, None)
        fresh = heap.root_handle("r")
        self.assertEqual(heap.generation_of(fresh), OLD)
        self.assertEqual(heap.payload(fresh), 1)

    def test_capacity_bounds_and_types(self):
        GenerationalHeap(1, 1)
        GenerationalHeap(4096, 4096)
        with self.assertRaises(ValueError):
            GenerationalHeap(0, 8)
        with self.assertRaises(ValueError):
            GenerationalHeap(8, 5000)
        with self.assertRaises(ValueError):
            GenerationalHeap(8, 8, promotion_age=0)
        with self.assertRaises(TypeError):
            GenerationalHeap(True, 8)
        with self.assertRaises(ValueError):
            GenerationalHeap(8, float("nan"))
        heap = GenerationalHeap(40, 40)
        with self.assertRaises(ValueError):
            heap.allocate(0, n_weak=-1)
        with self.assertRaises(ValueError):
            heap.allocate(0, generation=2)
        with self.assertRaises(TypeError):
            heap.allocate(0, generation="young")
        h = heap.allocate(0, n_strong=1)
        with self.assertRaises(IndexError):
            heap.get_strong(h, 1)

    def test_capacity_failure_leaves_consistent_heap(self):
        # Old generation is full of a rooted object; a young survivor cannot
        # promote, so the emergency minor fails and the oversized young
        # request is rejected with the heap still fully serviceable.
        heap = GenerationalHeap(young_capacity=6, old_capacity=2)
        o1 = heap.allocate(1, generation=OLD)  # fills the 2-slot old space
        heap.add_root("ro", o1)
        a = heap.allocate(2)  # size 2 in young (2 of 6 used)
        heap.add_root("ra", a)
        before = (heap.epoch, heap.objects())
        with self.assertRaises(HeapCapacityError):
            heap.allocate(0, n_strong=3)  # size 5: young full -> minor must
                                          # promote a, but old is full
        self.assertEqual(heap.epoch, before[0])  # failed cycle never commits
        self.assertEqual(heap.objects(), before[1])
        self.assertEqual(heap.payload(heap.root_handle("ro")), 1)
        self.assertEqual(heap.payload(heap.root_handle("ra")), 2)

    def test_object_larger_than_young_rejected_directly(self):
        heap = GenerationalHeap(young_capacity=6, old_capacity=6)
        with self.assertRaises(HeapCapacityError):
            heap.allocate(0, n_strong=5)  # size 7 > young semi-space
        self.assertEqual(heap.objects(), ())

    def test_determinism_same_inputs_same_results(self):
        def build_and_run(budget):
            t = Twin(160, 160)
            t.allocate("tab", 0, n_ephemeron=1, n_weak=1)
            t.allocate("k", 1)
            t.allocate("v", 2)
            t.allocate("g", 3)
            t.set_ephemeron("tab", 0, "k", "v")
            t.set_weak("tab", 0, "g")
            t.add_root("rt", "tab")
            t.add_root("rk", "k")
            return drive_steps(t, "major", budget)
        r1 = build_and_run(3)
        r2 = build_and_run(3)
        self.assertEqual(r1, r2)
        r3 = build_and_run(1)
        self.assertEqual(r1, r3)  # budget changes scheduling, not the result


class TestSmallExhaustive(unittest.TestCase):
    """All 3-object strong graphs x root subsets, both schedules.

    The 5-object exhaustive strong/weak/barrier enumeration lives in
    verification/ (kept out of the fast regression command by design).
    """

    def test_exhaustive_three_object_strong_graphs(self):
        names = "ABC"
        checked = 0
        for ea in range(4):
            for eb in range(4):
                for ec in range(4):
                    edges = (ea, eb, ec)
                    for root_bits in range(1, 8):
                        for budget in (1, 4):
                            t = Twin(120, 120)
                            for i, nm in enumerate(names):
                                t.allocate(nm, i, n_strong=1)
                            for i, nm in enumerate(names):
                                if edges[i] < 3:
                                    t.set_strong(nm, 0, names[edges[i]])
                            for bit, nm in enumerate(names):
                                if root_bits & (1 << bit):
                                    t.add_root("r" + nm, nm)
                            t.run_cycle("major", budget, None)
                            # Independent plain reachability soundness is
                            # asserted inside compare_idle after commit.
                            checked += 1
        self.assertEqual(checked, 64 * 7 * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
