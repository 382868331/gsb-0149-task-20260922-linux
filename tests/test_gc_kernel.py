"""Tests for gc_kernel.

Expected collection results are computed by an *independent* reference:
a plain-Python Cheney-style simulation written in this file that operates
only on the test's own graph description (never on gc_kernel internals or
outputs).  Round-trip reads from the heap are used only as a supplement.
"""

import random
import unittest

from gc_kernel import (
    MAX_SLOTS,
    CollectionResult,
    Handle,
    Heap,
    HeapFullError,
    InvalidArgumentError,
    InvalidHandleError,
    StaleHandleError,
)


# ---------------------------------------------------------------------------
# Independent reference implementation (does not import gc_kernel logic).
# ---------------------------------------------------------------------------


def reference_collect(desc, roots):
    """Simulate the contracted collection on a plain graph description.

    ``desc``: {old_address: {"strong": [addr|None, ...], "weak": [...]}}.
    ``roots``: list of old addresses, in root order.
    Returns (forwarding, weak_after) where forwarding maps old -> new
    address in root-then-field breadth-first copy order, and weak_after
    maps old address -> list of new addresses / None after clearing.
    """
    forwarding = {}
    order = []

    def evacuate(addr):
        if addr not in forwarding:
            forwarding[addr] = len(order)
            order.append(addr)

    for addr in roots:
        evacuate(addr)
    scan = 0
    while scan < len(order):
        for target in desc[order[scan]]["strong"]:
            if target is not None:
                evacuate(target)
        scan += 1
    weak_after = {
        addr: [
            (forwarding[t] if t is not None and t in forwarding else None)
            for t in desc[addr]["weak"]
        ]
        for addr in order
    }
    return forwarding, weak_after


def build_heap(desc, capacity, strong_arity, weak_arity, roots):
    """Materialise ``desc`` into a Heap; returns (heap, handles_by_addr)."""
    heap = Heap(capacity, strong_arity=strong_arity, weak_arity=weak_arity)
    handles = {}
    for addr in sorted(desc):
        handles[addr] = heap.alloc(payload=desc[addr]["payload"])
    for addr, obj in desc.items():
        for i, target in enumerate(obj["strong"]):
            if target is not None:
                heap.write_strong(handles[addr], i, handles[target])
        for i, target in enumerate(obj["weak"]):
            if target is not None:
                heap.write_weak(handles[addr], i, handles[target])
    for addr in roots:
        heap.add_root(handles[addr])
    return heap, handles


class ValidationTests(unittest.TestCase):
    def test_capacity_bounds(self):
        for bad in (0, -1, MAX_SLOTS + 1):
            with self.assertRaises(InvalidArgumentError, msg=f"capacity={bad}"):
                Heap(bad)
        Heap(1)
        Heap(MAX_SLOTS)

    def test_non_int_rejected(self):
        for bad in (True, 1.5, float("nan"), float("inf"), "8", None):
            with self.assertRaises(InvalidArgumentError, msg=f"capacity={bad!r}"):
                Heap(bad)
        for bad in (True, 2.0, float("nan"), float("inf"), "7"):
            with self.assertRaises(InvalidArgumentError, msg=f"payload={bad!r}"):
                Heap(4).alloc(payload=bad)
        with self.assertRaises(InvalidArgumentError):
            Heap(4, strong_arity=False)
        with self.assertRaises(InvalidArgumentError):
            Heap(4, weak_arity=-1)

    def test_edge_list_shape(self):
        heap = Heap(4, strong_arity=2, weak_arity=1)
        with self.assertRaises(InvalidArgumentError):
            heap.alloc(strong=[None])  # wrong length
        with self.assertRaises(InvalidArgumentError):
            heap.alloc(weak="x")
        with self.assertRaises(InvalidHandleError):
            heap.alloc(strong=[None, "not-a-handle"])

    def test_index_bounds(self):
        heap = Heap(4, strong_arity=1, weak_arity=1)
        h = heap.alloc()
        for bad in (-1, 1, True, 0.0):
            with self.assertRaises(InvalidArgumentError, msg=f"index={bad!r}"):
                heap.read_strong(h, bad)
            with self.assertRaises(InvalidArgumentError, msg=f"index={bad!r}"):
                heap.write_weak(h, bad, None)

    def test_malformed_handle(self):
        heap = Heap(4)
        heap.alloc()
        for bad in (None, 7, "h", Handle(-1, 0), Handle(0, -1), Handle(True, 0)):
            with self.assertRaises(InvalidHandleError, msg=f"handle={bad!r}"):
                heap.read_payload(bad)
        with self.assertRaises(InvalidHandleError):  # slot never occupied
            heap.read_payload(Handle(3, heap.epoch))


class CollectionSemanticsTests(unittest.TestCase):
    def test_self_loop_survives_and_points_to_itself(self):
        heap = Heap(4, strong_arity=1, weak_arity=0)
        a = heap.alloc(payload=42)
        heap.write_strong(a, 0, a)  # self loop
        heap.add_root(a)
        result = heap.collect()
        self.assertEqual(result.live_slots, 1)
        self.assertEqual(result.moved, {0: 0})
        (new_a,) = result.roots
        self.assertEqual(heap.read_strong(new_a, 0), new_a)
        self.assertEqual(heap.read_payload(new_a), 42)

    def test_weak_only_target_dies_and_edge_is_cleared(self):
        heap = Heap(4, strong_arity=1, weak_arity=1)
        a = heap.alloc(payload=1)
        w = heap.alloc(payload=2)  # reachable only through a.weak[0]
        heap.write_weak(a, 0, w)
        heap.add_root(a)
        result = heap.collect()
        self.assertEqual(result.live_slots, 1)
        self.assertNotIn(w.address, result.moved)
        self.assertIsNone(heap.read_weak(result.roots[0], 0))

    def test_weak_edge_to_live_target_is_remapped(self):
        heap = Heap(8, strong_arity=2, weak_arity=1)
        a = heap.alloc(payload=1)
        b = heap.alloc(payload=2)
        heap.write_strong(a, 0, b)  # b stays alive via strong edge
        heap.write_weak(a, 0, b)  # ... and is also weakly referenced
        heap.add_root(a)
        result = heap.collect()
        self.assertEqual(result.live_slots, 2)
        new_a = result.roots[0]
        self.assertEqual(heap.read_weak(new_a, 0), heap.read_strong(new_a, 0))

    def test_shared_subgraph_copied_once(self):
        # roots r1, r2 both strongly point to shared child c.
        heap = Heap(8, strong_arity=1, weak_arity=0)
        r1 = heap.alloc(payload=11)
        r2 = heap.alloc(payload=22)
        c = heap.alloc(payload=33)
        heap.write_strong(r1, 0, c)
        heap.write_strong(r2, 0, c)
        heap.add_root(r1)
        heap.add_root(r2)
        result = heap.collect()
        self.assertEqual(result.live_slots, 3)
        new_r1, new_r2 = result.roots
        self.assertEqual(heap.read_strong(new_r1, 0), heap.read_strong(new_r2, 0))
        self.assertEqual(heap.read_payload(heap.read_strong(new_r1, 0)), 33)

    def test_unrooted_garbage_reclaimed_and_payloads_preserved(self):
        heap = Heap(8, strong_arity=1, weak_arity=0)
        live = heap.alloc(payload=100)
        for i in range(3):
            heap.alloc(payload=200 + i)  # garbage
        heap.add_root(live)
        result = heap.collect()
        self.assertEqual(result.live_slots, 1)
        self.assertEqual(result.freed_slots, 3)
        self.assertEqual(heap.read_payload(result.roots[0]), 100)

    def test_copy_order_is_root_then_field_breadth_first(self):
        # Hand-computed expectation: roots copied in registration order,
        # then strong fields scanned breadth-first in field order.
        desc = {
            0: {"payload": 0, "strong": [2, 3], "weak": []},  # root
            1: {"payload": 1, "strong": [0], "weak": []},      # root
            2: {"payload": 2, "strong": [4], "weak": []},
            3: {"payload": 3, "strong": [], "weak": []},
            4: {"payload": 4, "strong": [], "weak": []},
            5: {"payload": 5, "strong": [], "weak": []},       # garbage
        }
        forwarding, _ = reference_collect(desc, roots=[1, 0])
        self.assertEqual(forwarding, {1: 0, 0: 1, 2: 2, 3: 3, 4: 4})
        heap, _ = build_heap(desc, 8, 2, 0, roots=[1, 0])
        result = heap.collect()
        self.assertEqual(result.moved, forwarding)


class RandomisedCrossCheckTests(unittest.TestCase):
    def test_against_independent_reachability(self):
        rng = random.Random(20260922)
        n = 60
        desc = {}
        for addr in range(n):
            strong = [rng.randrange(n) if rng.random() < 0.5 else None for _ in range(2)]
            weak = [rng.randrange(n) if rng.random() < 0.4 else None]
            desc[addr] = {"payload": 1000 + addr, "strong": strong, "weak": weak}
        roots = sorted(rng.sample(range(n), 5))
        forwarding, weak_after = reference_collect(desc, roots)

        heap, handles = build_heap(desc, 128, 2, 1, roots)
        old_epoch = heap.epoch
        result = heap.collect()

        # Relocation map and surviving slot count, computed independently.
        self.assertEqual(result.moved, forwarding)
        self.assertEqual(result.live_slots, len(forwarding))
        self.assertEqual(result.freed_slots, n - len(forwarding))
        self.assertEqual(result.epoch, old_epoch + 1)

        # Every survivor: payload intact, strong edges relocated, weak edges
        # remapped or cleared exactly as the reference dictates.
        for old_addr, new_addr in forwarding.items():
            h = Handle(new_addr, result.epoch)
            self.assertEqual(heap.read_payload(h), 1000 + old_addr)
            for i, target in enumerate(desc[old_addr]["strong"]):
                got = heap.read_strong(h, i)
                if target is None:
                    self.assertIsNone(got)
                else:
                    self.assertEqual(got, Handle(forwarding[target], result.epoch))
            got_weak = heap.read_weak(h, 0)
            expected_weak = weak_after[old_addr][0]
            self.assertEqual(
                got_weak,
                None if expected_weak is None else Handle(expected_weak, result.epoch),
            )
        # Relocated roots match the forwarding table, in registration order.
        self.assertEqual(
            result.roots, tuple(Handle(forwarding[r], result.epoch) for r in roots)
        )
        # Every pre-collection handle is now stale.
        for h in handles.values():
            with self.assertRaises(StaleHandleError):
                heap.read_payload(h)


class FullHeapTests(unittest.TestCase):
    def test_exactly_full_heap_refuses_without_partial_object(self):
        heap = Heap(3, strong_arity=0, weak_arity=0)
        for i in range(3):
            heap.add_root(heap.alloc(payload=i))
        with self.assertRaises(HeapFullError):
            heap.alloc(payload=99)
        # No half object: slot count unchanged, all roots still readable.
        self.assertEqual(heap.live_slots, 3)
        self.assertEqual(
            [heap.read_payload(h) for h in heap.roots()], [0, 1, 2]
        )

    def test_failed_allocation_collects_once_first(self):
        heap = Heap(2, strong_arity=0, weak_arity=0)
        heap.alloc(payload=1)  # garbage
        heap.alloc(payload=2)  # garbage; heap now full
        h = heap.alloc(payload=3)  # triggers one collection, then succeeds
        self.assertEqual(heap.live_slots, 1)
        self.assertEqual(heap.read_payload(h), 3)

    def test_pending_strong_target_survives_alloc_triggered_collection(self):
        heap = Heap(3, strong_arity=1, weak_arity=0)
        a = heap.add_root(heap.alloc(payload=1))
        c = heap.alloc(payload=2)  # unrooted, but about to be linked
        heap.alloc(payload=3)      # garbage filler; heap now full
        d = heap.alloc(payload=4, strong=[c])
        self.assertEqual(heap.live_slots, 3)
        self.assertEqual(heap.read_payload(heap.read_strong(d, 0)), 2)
        # 'a' was minted before the alloc-triggered collection; use the
        # relocated registered root instead.
        self.assertEqual(heap.read_payload(heap.roots()[0]), 1)

    def test_pending_weak_target_is_not_kept_alive(self):
        heap = Heap(2, strong_arity=0, weak_arity=1)
        a = heap.add_root(heap.alloc(payload=1))
        w = heap.alloc(payload=2)  # weak-only, unrooted; heap now full
        b = heap.alloc(payload=3, weak=[w])
        self.assertIsNone(heap.read_weak(b, 0))
        self.assertEqual(heap.live_slots, 2)
        # 'a' is stale after the alloc-triggered collection; the registered
        # root was relocated by the heap.
        self.assertEqual(heap.read_payload(heap.roots()[0]), 1)


class StaleHandleTests(unittest.TestCase):
    def test_stale_handle_rejected_everywhere(self):
        heap = Heap(4, strong_arity=1, weak_arity=1)
        a = heap.add_root(heap.alloc(payload=1))
        heap.collect()
        with self.assertRaises(StaleHandleError):
            heap.read_payload(a)
        with self.assertRaises(StaleHandleError):
            heap.write_payload(a, 9)
        with self.assertRaises(StaleHandleError):
            heap.read_strong(a, 0)
        with self.assertRaises(StaleHandleError):
            heap.write_weak(a, 0, None)
        with self.assertRaises(StaleHandleError):
            heap.add_root(a)
        with self.assertRaises(StaleHandleError):
            heap.alloc(strong=[a])
        # The relocated root remains usable.
        self.assertEqual(heap.read_payload(heap.roots()[0]), 1)

    def test_forged_future_epoch_rejected(self):
        heap = Heap(4)
        heap.alloc()
        with self.assertRaises(StaleHandleError):
            heap.read_payload(Handle(0, heap.epoch + 1))

    def test_result_type_and_fields(self):
        heap = Heap(4)
        heap.add_root(heap.alloc(payload=7))
        result = heap.collect()
        self.assertIsInstance(result, CollectionResult)
        self.assertEqual(result.live_slots, 1)
        self.assertEqual(result.epoch, heap.epoch)


if __name__ == "__main__":
    unittest.main()
