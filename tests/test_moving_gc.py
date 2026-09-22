"""Differential + contract tests for the moving_gc kernel.

Expected values come from :mod:`reference`, an independent graph model that
never imports the code under test.  Round-trip style assertions (reading
back what was written) are only used as supplementary checks.
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import GraphSpec  # noqa: E402  (independent oracle)

from moving_gc import (  # noqa: E402
    MAX_CAPACITY,
    Handle,
    Heap,
    HeapCapacityError,
    StaleHandleError,
)


# --------------------------------------------------------------------------- #
# Shared construction / verification harness
# --------------------------------------------------------------------------- #
def build_world(spec: GraphSpec) -> tuple[Heap, dict[str, Handle]]:
    """Construct a heap mirroring ``spec``; return it plus name->handle."""
    heap = Heap(spec.capacity)
    handles: dict[str, Handle] = {}
    for name, node in spec.nodes.items():
        handles[name] = heap.allocate(
            node.payload, n_strong=len(node.strong), n_weak=len(node.weak)
        )
    for name, node in spec.nodes.items():
        for j, dst in enumerate(node.strong):
            if dst is not None:
                heap.set_strong(handles[name], j, handles[dst])
        for j, dst in enumerate(node.weak):
            if dst is not None:
                heap.set_weak(handles[name], j, handles[dst])
    for rn in spec.root_order:
        t = spec.roots[rn]
        if t is not None:
            heap.add_root(rn, handles[t])
        else:
            # Named null root: register against any object, then clear it.
            anchor = next(iter(handles.values()))
            heap.add_root(rn, anchor)
            heap.clear_root(rn)
    return heap, handles


def verify_against_oracle(
    spec: GraphSpec,
    heap: Heap,
    expected,
    result,
    _old_addresses: dict[str, int],
) -> dict[str, Handle]:
    """Compare one real CollectionResult with the independent oracle."""
    # --- relocation mapping ------------------------------------------------
    exp_old = {old: new for old, new in expected.relocated.values()}
    assert exp_old == result.relocated, (exp_old, result.relocated)

    name_to_new: dict[str, int] = {}
    for name, (old, new) in expected.relocated.items():
        assert result.relocated[old] == new
        name_to_new[name] = new

    # Copy order must equal destination-address ascending order (Cheney).
    assert list(result.relocated.values()) == [
        expected.relocated[n][1] for n in expected.copy_order
    ]

    # --- counts -------------------------------------------------------------
    assert result.live_slots == expected.live_slots
    assert heap.used_slots == expected.live_slots
    assert set(heap.objects()) == set(name_to_new.values())

    # --- roots --------------------------------------------------------------
    for rn in spec.root_order:
        rh = heap.root_handle(rn)
        if expected.roots[rn] is None:
            assert rh is None
        else:
            assert rh is not None and rh.address == name_to_new[expected.roots[rn]]

    # --- cleared weak edges -------------------------------------------------
    exp_cleared = tuple(
        (name_to_new[holder], j) for holder, j in expected.cleared
    )
    assert result.cleared_weak_edges == exp_cleared

    # --- payloads and every edge -------------------------------------------
    fresh: dict[str, Handle] = {
        name: heap.handle(new) for name, new in name_to_new.items()
    }
    for name in expected.copy_order:
        node = spec.nodes[name]
        h = fresh[name]
        assert heap.payload(h) == node.payload
        for j, t in enumerate(node.strong):
            got = heap.get_strong(h, j)
            if t is None:
                assert got is None
            else:
                assert got is not None
                assert got.address == name_to_new[t]
        for j, t in enumerate(node.weak):
            got = heap.get_weak(h, j)
            if t is None or t not in name_to_new:
                assert got is None, (name, j, t)
            else:
                assert got is not None and got.address == name_to_new[t], (name, j, t)
    return fresh


def old_address_map(spec: GraphSpec) -> dict[str, int]:
    return {name: node.address for name, node in spec.nodes.items()}


# --------------------------------------------------------------------------- #
# Basic semantics
# --------------------------------------------------------------------------- #
class TestBasic(unittest.TestCase):
    def test_allocate_addresses_and_sizes(self):
        heap = Heap(20)
        h0 = heap.allocate(10, n_strong=1, n_weak=1)  # size 4
        h1 = heap.allocate(20)  # size 2
        self.assertEqual((h0.address, h1.address), (0, 4))
        self.assertEqual(heap.used_slots, 6)
        self.assertEqual(heap.free_slots, 14)
        self.assertEqual(heap.payload(h0), 10)
        self.assertEqual(heap.object_shape(h0), (1, 1))
        self.assertIsNone(heap.get_strong(h0, 0))
        self.assertIsNone(heap.get_weak(h0, 0))

    def test_simple_collection_relocates(self):
        spec = GraphSpec(20)
        spec.add_node("g", 99)  # garbage, address 0
        spec.add_node("a", 1, n_strong=1)  # address 2
        spec.link_strong("a", 0, "a")  # self loop
        spec.add_root("r", "a")
        heap, handles = build_world(spec)
        old = old_address_map(spec)
        self.assertEqual(old["a"], 2)  # not at 0 initially

        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old)
        # Address actually moved to the start of the fresh semi-space.
        self.assertEqual(fresh["a"].address, 0)
        self.assertEqual(result.relocated, {2: 0})
        # Self loop forwarded to the *new* address.
        self.assertEqual(heap.get_strong(fresh["a"], 0), fresh["a"])
        self.assertEqual(heap.epoch, 1)

    def test_collect_without_roots_reclaims_everything(self):
        heap = Heap(10)
        heap.allocate(1)
        heap.allocate(2, n_strong=1)
        result = heap.collect()
        self.assertEqual(result.relocated, {})
        self.assertEqual(result.live_slots, 0)
        self.assertEqual(heap.objects(), ())
        self.assertEqual(heap.epoch, 1)

    def test_payload_update_persists(self):
        heap = Heap(8)
        h = heap.allocate(0)
        heap.set_payload(h, 12345)
        heap.add_root("r", h)
        heap.collect()
        h2 = heap.root_handle("r")
        self.assertEqual(heap.payload(h2), 12345)


# --------------------------------------------------------------------------- #
# Required graph scenarios vs the independent oracle
# --------------------------------------------------------------------------- #
class TestGraphScenarios(unittest.TestCase):
    def _run_one(self, spec: GraphSpec):
        heap, _ = build_world(spec)
        old = old_address_map(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        return verify_against_oracle(spec, heap, expected, result, old)

    def test_self_loop(self):
        spec = GraphSpec(16)
        spec.add_node("a", n_strong=1)
        spec.link_strong("a", 0, "a")
        spec.add_root("r", "a")
        self._run_one(spec)

    def test_cycle_without_root_dies(self):
        spec = GraphSpec(16)
        spec.add_node("a", n_strong=1)
        spec.add_node("b", n_strong=1)
        spec.link_strong("a", 0, "b")
        spec.link_strong("b", 0, "a")
        # no roots: the strongly connected cycle must still be collected
        fresh = self._run_one(spec)
        self.assertEqual(fresh, {})

    def test_weak_only_reference_is_cleared(self):
        spec = GraphSpec(16)
        spec.add_node("a", n_weak=1)
        spec.add_node("b")
        spec.link_weak("a", 0, "b")  # b only weakly referenced
        spec.add_root("r", "a")
        heap, _ = build_world(spec)
        old = old_address_map(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old)
        self.assertIsNone(heap.get_weak(fresh["a"], 0))
        self.assertEqual(result.cleared_weak_edges, ((fresh["a"].address, 0),))
        self.assertEqual(set(result.relocated), {old["a"]})
        self.assertNotIn(old["b"], result.relocated)

    def test_two_weak_edges_to_one_dead_target_both_cleared(self):
        spec = GraphSpec(32)
        spec.add_node("a", n_weak=1)
        spec.add_node("c", n_weak=1)
        spec.add_node("b")
        spec.link_weak("a", 0, "b")
        spec.link_weak("c", 0, "b")
        spec.add_root("r1", "a")
        spec.add_root("r2", "c")
        self._run_one(spec)

    def test_shared_subgraph_copied_once(self):
        spec = GraphSpec(32)
        spec.add_node("a", n_strong=1)
        spec.add_node("c", n_strong=1)
        spec.add_node("b")  # shared target
        spec.link_strong("a", 0, "b")
        spec.link_strong("c", 0, "b")
        spec.add_root("r1", "a")
        spec.add_root("r2", "c")
        heap, _ = build_world(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old_address_map(spec))
        # Exactly one copy of b; both strong edges name the same new address.
        self.assertEqual(heap.get_strong(fresh["a"], 0), fresh["b"])
        self.assertEqual(heap.get_strong(fresh["c"], 0), fresh["b"])
        self.assertEqual(len(result.relocated), 3)

    def test_weak_edge_to_survivor_is_forwarded(self):
        spec = GraphSpec(32)
        spec.add_node("a", n_weak=1)
        spec.add_node("b")
        spec.link_weak("a", 0, "b")
        spec.add_root("r_a", "a")
        spec.add_root("r_b", "b")  # b survives through its own strong root
        heap, _ = build_world(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old_address_map(spec))
        self.assertEqual(heap.get_weak(fresh["a"], 0), fresh["b"])
        self.assertEqual(result.cleared_weak_edges, ())

    def test_root_and_field_copy_order(self):
        # Allocation order is b,a,c; roots are registered as a then c.
        # Standard Cheney order: every root is evacuated first, in root
        # order (a, c), then the scan queue drains strong fields in field
        # order (a's fields introduce b; c is already copied).
        spec = GraphSpec(40)
        spec.add_node("b", n_strong=0)
        spec.add_node("a", n_strong=2)
        spec.add_node("c", n_strong=0)
        spec.link_strong("a", 0, "b")
        spec.link_strong("a", 1, "c")
        spec.add_root("r_a", "a")
        spec.add_root("r_c", "c")
        heap, _ = build_world(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        # a(size4) at 0, c(size2) at 4, b(size2) at 6.
        self.assertEqual(expected.copy_order, ("a", "c", "b"))
        self.assertEqual(list(result.relocated.values()), [0, 4, 6])
        verify_against_oracle(spec, heap, expected, result, old_address_map(spec))

    def test_mixed_weak_strong_diamond(self):
        spec = GraphSpec(48)
        spec.add_node("g0")  # garbage to force real movement
        spec.add_node("a", n_strong=2, n_weak=2)
        spec.add_node("b", n_strong=1, n_weak=1)
        spec.add_node("s")  # shared
        spec.add_node("w")  # weak-only target
        spec.link_strong("a", 0, "b")
        spec.link_strong("a", 1, "s")
        spec.link_strong("b", 0, "s")
        spec.link_weak("a", 0, "w")  # dead -> cleared
        spec.link_weak("a", 1, "s")  # survivor -> forwarded
        spec.link_weak("b", 0, "w")  # dead -> cleared
        spec.add_root("r", "a")
        heap, _ = build_world(spec)
        old = old_address_map(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old)
        # Every surviving address changed (garbage used the low slots before).
        for name in ("a", "b", "s"):
            self.assertNotEqual(fresh[name].address, old[name])
        self.assertIsNone(heap.get_weak(fresh["a"], 0))
        self.assertEqual(heap.get_weak(fresh["a"], 1), fresh["s"])
        self.assertIsNone(heap.get_weak(fresh["b"], 0))


# --------------------------------------------------------------------------- #
# Capacity boundary: exactly-full heap and failed allocation
# --------------------------------------------------------------------------- #
class TestCapacity(unittest.TestCase):
    def test_exactly_full_heap_survives_collection(self):
        cap = 4096
        heap = Heap(cap)
        handles = [heap.allocate(i) for i in range(cap // 2)]  # 2048 * 2 slots
        self.assertEqual(heap.used_slots, cap)
        self.assertEqual(heap.free_slots, 0)
        for i, h in enumerate(handles):
            heap.add_root(f"r{i}", h)
        result = heap.collect()
        self.assertEqual(result.live_slots, cap)
        self.assertEqual(len(result.relocated), cap // 2)
        # Still exactly full; a minimum-size allocation must fail.
        with self.assertRaises(HeapCapacityError):
            heap.allocate(0)
        # Emergency collection ran, but no half object exists.
        self.assertEqual(heap.used_slots, cap)
        self.assertEqual(len(heap.objects()), cap // 2)

    def test_small_exactly_full_then_failure_leaves_no_half_object(self):
        heap = Heap(6)
        a = heap.allocate(1, n_strong=1)  # size 3
        b = heap.allocate(2, n_strong=1)  # size 3 -> exactly full
        heap.add_root("ra", a)
        heap.add_root("rb", b)
        snapshot = (heap.epoch, heap.used_slots, sorted(heap.objects()))
        with self.assertRaises(HeapCapacityError):
            heap.allocate(9)  # size 2: collect first (nothing dies), then fail
        # No half object: object set and occupancy unchanged post-GC.
        self.assertEqual(heap.used_slots, snapshot[1])
        self.assertEqual(sorted(heap.objects()), snapshot[2])
        # One collection did happen.
        self.assertEqual(heap.epoch, snapshot[0] + 1)

    def test_object_bigger_than_capacity_collects_then_fails(self):
        # Contract: allocation failure ALWAYS performs one collection first,
        # even when the object can never fit an empty semi-space.
        heap = Heap(4)
        heap.allocate(9)  # unrooted garbage, size 2 at slot 0
        epoch = heap.epoch
        with self.assertRaises(HeapCapacityError):
            heap.allocate(0, n_strong=4)  # size 6 > 4
        # The mandatory collection ran (epoch advanced) and reclaimed the
        # garbage, but still no half-object exists.
        self.assertEqual(heap.epoch, epoch + 1)
        self.assertEqual(heap.used_slots, 0)
        self.assertEqual(heap.objects(), ())

    def test_allocation_failure_triggers_reclaim_then_succeeds(self):
        heap = Heap(10)
        garbage = heap.allocate(77)  # size 2 at 0
        a = heap.allocate(1, n_strong=1)  # size 3 at 2
        b = heap.allocate(2)  # size 2 at 5
        heap.set_strong(a, 0, b)
        heap.add_root("r", a)
        self.assertEqual(heap.used_slots, 7)
        # 3 slots free; a size-4 object does not fit -> GC reclaims garbage
        # (2 slots) and the new object is carved from compacted to-space.
        c = heap.allocate(3, n_strong=2)  # size 4
        self.assertEqual(heap.epoch, 1)
        self.assertLessEqual(heap.used_slots, heap.capacity)
        ra = heap.root_handle("r")
        self.assertEqual(heap.payload(ra), 1)
        self.assertEqual(heap.payload(c), 3)
        self.assertEqual(heap.payload(heap.get_strong(ra, 0)), 2)

    def test_capacity_constructor_bounds(self):
        Heap(1)
        Heap(MAX_CAPACITY)
        with self.assertRaises(ValueError):
            Heap(0)
        with self.assertRaises(ValueError):
            Heap(-1)
        with self.assertRaises(ValueError):
            Heap(MAX_CAPACITY + 1)


# --------------------------------------------------------------------------- #
# Stale handles / epochs
# --------------------------------------------------------------------------- #
class TestStaleHandles(unittest.TestCase):
    def test_old_handle_rejected_everywhere(self):
        heap = Heap(20)
        heap.allocate(0)  # unrooted garbage: guarantees survivors really move
        a = heap.allocate(1, n_strong=1, n_weak=1)  # old addr 2, size 4
        b = heap.allocate(2)  # old addr 6
        heap.set_strong(a, 0, b)
        heap.set_weak(a, 0, b)
        heap.add_root("r", a)
        heap.collect()  # a,b move; epoch -> 1
        stale_a = a
        stale_b = b
        with self.assertRaises(StaleHandleError):
            heap.payload(stale_a)
        with self.assertRaises(StaleHandleError):
            heap.set_payload(stale_a, 5)
        with self.assertRaises(StaleHandleError):
            heap.get_strong(stale_a, 0)
        with self.assertRaises(StaleHandleError):
            heap.get_weak(stale_a, 0)
        fresh_a = heap.root_handle("r")
        with self.assertRaises(StaleHandleError):
            heap.set_strong(fresh_a, 0, stale_b)
        with self.assertRaises(StaleHandleError):
            heap.set_weak(fresh_a, 0, stale_b)
        with self.assertRaises(StaleHandleError):
            heap.add_root("r2", stale_a)
        # Failed writes did not mutate the edges: b was strongly reachable
        # and survived, so the strong edge must name b's *new* address (and
        # the weak edge too), never the stale one.
        fresh_b = heap.get_strong(fresh_a, 0)
        self.assertIsNotNone(fresh_b)
        self.assertNotEqual(fresh_b.address, stale_b.address)
        self.assertEqual(heap.get_weak(fresh_a, 0).address, fresh_b.address)

    def test_stale_handle_even_when_address_coincides(self):
        # Self-loop object lands at address 0 both times; epoch still invalidates.
        heap = Heap(8)
        a = heap.allocate(1, n_strong=1)
        heap.set_strong(a, 0, a)
        heap.add_root("r", a)
        self.assertEqual(a.address, 0)
        heap.collect()
        self.assertEqual(heap.root_handle("r").address, 0)  # same slot index
        stale = Handle(0, a.epoch)
        with self.assertRaises(StaleHandleError):
            heap.payload(stale)
        heap.collect()
        # An epoch-1 handle is stale at epoch 2 even with a valid address.
        old1 = Handle(0, 1)
        with self.assertRaises(StaleHandleError):
            heap.payload(old1)

    def test_dead_address_handle_rejected(self):
        heap = Heap(12)
        heap.allocate(9)  # unrooted garbage at address 0
        a = heap.allocate(1, n_strong=2)  # size 4 at address 2
        heap.add_root("r", a)
        heap.collect()
        # The survivor compacts to slots 0..3; everything from used_slots
        # (4) upward is free.  A current-epoch handle into free space must be
        # rejected even though its epoch matches.
        forged = Handle(heap.used_slots, heap.epoch)
        with self.assertRaises(StaleHandleError):
            heap.payload(forged)
        with self.assertRaises(StaleHandleError):
            heap.handle(heap.capacity - 1)

    def test_two_collections_relocation_map_complete(self):
        heap = Heap(16)
        heap.allocate(0)  # garbage
        a = heap.allocate(1)
        heap.add_root("r", a)
        r1 = heap.collect()
        self.assertIn(a.address, r1.relocated)
        addr_after_1 = r1.relocated[a.address]
        r2 = heap.collect()
        # Nothing relocates between two back-to-back compactions of one object,
        # but the map still reports the survivor and epoch advances.
        self.assertEqual(r2.relocated, {addr_after_1: 0})
        self.assertEqual(r2.live_slots, 2)
        self.assertEqual(heap.epoch, 2)


# --------------------------------------------------------------------------- #
# Input validation contract
# --------------------------------------------------------------------------- #
class TestInputValidation(unittest.TestCase):
    def test_bad_capacity_types(self):
        for bad in (True, False):
            with self.assertRaises(TypeError):
                Heap(bad)
        for bad in ("4", 1.0, 1 + 0j, None):
            with self.assertRaises(TypeError):
                Heap(bad)
        with self.assertRaises(ValueError):
            Heap(float("nan"))
        with self.assertRaises(ValueError):
            Heap(float("inf"))
        with self.assertRaises(ValueError):
            Heap(-float("inf"))

    def test_bad_allocate_arguments(self):
        heap = Heap(8)
        with self.assertRaises(TypeError):
            heap.allocate(True)
        with self.assertRaises(ValueError):
            heap.allocate(float("nan"))
        with self.assertRaises(ValueError):
            heap.allocate(float("-inf"))
        with self.assertRaises(TypeError):
            heap.allocate(1.5)
        with self.assertRaises(TypeError):
            heap.allocate(0, n_strong=True)
        with self.assertRaises(ValueError):
            heap.allocate(0, n_strong=-1)
        with self.assertRaises(ValueError):
            heap.allocate(0, n_weak=-1)

    def test_bad_field_arguments(self):
        heap = Heap(16)
        h = heap.allocate(0, n_strong=1, n_weak=1)
        with self.assertRaises(TypeError):
            heap.get_strong(h, True)
        with self.assertRaises(IndexError):
            heap.get_strong(h, 1)
        with self.assertRaises(IndexError):
            heap.get_weak(h, -1)
        with self.assertRaises(TypeError):
            heap.set_strong(h, 0, "not-a-handle")
        with self.assertRaises(TypeError):
            heap.set_weak(h, 0, 7)
        with self.assertRaises(ValueError):
            heap.set_payload(h, math.inf)

    def test_bad_root_names(self):
        heap = Heap(8)
        h = heap.allocate(0)
        with self.assertRaises(TypeError):
            heap.add_root("", h)
        with self.assertRaises(TypeError):
            heap.add_root(123, h)  # type: ignore[arg-type]
        with self.assertRaises(KeyError):
            heap.remove_root("missing")
        with self.assertRaises(KeyError):
            heap.root_handle("missing")

    def test_failed_root_replace_preserves_old_root(self):
        heap = Heap(16)
        a = heap.allocate(1)
        b = heap.allocate(2)
        heap.add_root("r", a)
        heap.collect()  # b dies anyway; keep a stale handle
        with self.assertRaises(StaleHandleError):
            heap.add_root("r", b)
        # Original root still resolves to a's copy.
        self.assertEqual(heap.payload(heap.root_handle("r")), 1)

    def test_null_roots_and_removal(self):
        heap = Heap(16)
        a = heap.allocate(1)
        heap.add_root("r", a)
        heap.clear_root("r")
        self.assertIsNone(heap.root_handle("r"))
        heap.collect()
        self.assertIsNone(heap.root_handle("r"))
        self.assertEqual(heap.objects(), ())
        heap.remove_root("r")
        with self.assertRaises(KeyError):
            heap.root_handle("r")


# --------------------------------------------------------------------------- #
# Randomized differential fuzz against the independent oracle
# --------------------------------------------------------------------------- #
class TestDifferentialFuzz(unittest.TestCase):
    def test_seeded_random_graphs(self):
        rng = random.Random(20260922)
        for trial in range(12):
            cap = 120
            spec = GraphSpec(cap)
            heap = Heap(cap)
            handles: dict[str, Handle] = {}

            def add_node(name):
                ns = rng.randrange(0, 3)
                nw = rng.randrange(0, 2)
                if 2 + ns + nw > cap:
                    return False
                try:
                    spec.add_node(name, rng.randrange(0, 1_000_000), ns, nw)
                except RuntimeError:
                    return False
                handles[name] = heap.allocate(
                    spec.nodes[name].payload, n_strong=ns, n_weak=nw
                )
                return True

            for i in range(25):
                add_node(f"n{i}")

            names = list(spec.nodes)
            for name in names:
                node = spec.nodes[name]
                for j in range(len(node.strong)):
                    if rng.random() < 0.6:
                        t = rng.choice(names)
                        spec.link_strong(name, j, t)
                        heap.set_strong(handles[name], j, handles[t])
                for j in range(len(node.weak)):
                    if rng.random() < 0.6:
                        t = rng.choice(names)
                        spec.link_weak(name, j, t)
                        heap.set_weak(handles[name], j, handles[t])

            rooted = rng.sample(names, k=max(1, len(names) // 3))
            for k, name in enumerate(rooted):
                spec.add_root(f"r{k}", name)
                heap.add_root(f"r{k}", handles[name])

            # Round 1: collection + full comparison.
            old = old_address_map(spec)
            result = heap.collect()
            expected = spec.expected_collect()
            fresh = verify_against_oracle(spec, heap, expected, result, old)

            # Mutate between GCs: drop half the roots, allocate new nodes,
            # rewire a few edges, then collect again and re-compare.
            drop = spec.root_order[::2]
            for rn in drop:
                spec.drop_root(rn)
                heap.remove_root(rn)

            for i in range(8):
                name = f"m{trial}_{i}"
                ns = rng.randrange(0, 2)
                nw = rng.randrange(0, 2)
                try:
                    spec.add_node(name, rng.randrange(0, 1_000_000), ns, nw)
                except RuntimeError:
                    break
                handles[name] = heap.allocate(
                    spec.nodes[name].payload, n_strong=ns, n_weak=nw
                )
                survivors = list(fresh)
                if survivors:
                    t = rng.choice(survivors)
                    spec.add_root(f"newr{i}", name)
                    heap.add_root(f"newr{i}", handles[name])
                    if ns > 0 and rng.random() < 0.7:
                        spec.link_strong(name, 0, t)
                        heap.set_strong(handles[name], 0, fresh[t])
                fresh[name] = handles[name]

            old2 = {name: node.address for name, node in spec.nodes.items()}
            result2 = heap.collect()
            expected2 = spec.expected_collect()
            verify_against_oracle(spec, heap, expected2, result2, old2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
