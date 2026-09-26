"""Contract tests for the optional pinning mode of moving_gc.

Expected layouts come from an independent sliding-compaction model
(expected_compaction) that never imports the code under test;
round-trip reads are only supplementary assertions.
"""

from __future__ import annotations

import unittest

from moving_gc import (
    FragmentationError,
    GCError,
    Handle,
    Heap,
    HeapCapacityError,
    PinHandle,
    StaleHandleError,
)


# --------------------------------------------------------------------------- #
# Independent reference model (does not import moving_gc internals)
# --------------------------------------------------------------------------- #
def expected_compaction(live, pinned):
    """Sliding first-fit compaction around pinned intervals.

    live maps old address -> object size; pinned addresses keep
    their slots.  Returns the full old -> new forwarding map.
    """
    intervals = sorted(
        (addr, addr + size) for addr, size in live.items() if addr in pinned
    )
    fwd = {}
    free = 0
    idx = 0
    for old in sorted(live):
        size = live[old]
        if old in pinned:
            fwd[old] = old
            continue
        while True:
            while idx < len(intervals) and intervals[idx][1] <= free:
                idx += 1
            if idx < len(intervals) and intervals[idx][0] < free + size:
                free = intervals[idx][1]
                idx += 1
                continue
            break
        fwd[old] = free
        free += size
    return fwd


def assert_no_overlap(testcase, heap):
    """Every pair of live objects occupies disjoint slot intervals."""
    intervals = []
    for addr in heap.objects():
        n_strong, n_weak = heap.object_shape(heap.handle(addr))
        intervals.append((addr, addr + 2 + n_strong + n_weak))
    intervals.sort()
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        testcase.assertLessEqual(e1, s2, f"overlap: [{s1},{e1}) vs [{s2},{e2})")
    if intervals:
        testcase.assertLessEqual(intervals[-1][1], heap.capacity)


class PinningBasicsTest(unittest.TestCase):
    def test_pinned_object_never_moves_movable_slides(self):
        heap = Heap(20, pinning=True)
        ha = heap.allocate(11)            # addr 0, size 2
        hb = heap.allocate(22)            # addr 2 (garbage)
        hc = heap.allocate(33)            # addr 4
        hd = heap.allocate(44, n_weak=2)  # addr 6, size 4
        pa = heap.pin(ha)
        pc = heap.pin(hc)
        heap.add_root("root", hd)
        heap.set_weak(hd, 0, hc)  # weak -> pinned survivor
        heap.set_weak(hd, 1, hb)  # weak -> garbage

        expected = expected_compaction({0: 2, 4: 2, 6: 4}, pinned={0, 4})
        result = heap.collect()

        self.assertEqual(result.relocated, {o: n for o, n in expected.items() if o != n})
        self.assertEqual(result.pinned_addresses, (0, 4))
        self.assertEqual(heap.payload(pa), 11)   # pin handle still valid
        self.assertEqual(heap.payload(pc), 33)
        self.assertEqual(pa.address, 0)
        self.assertEqual(pc.address, 4)
        # d slid into the hole left by dead b.
        new_d = heap.root_handle("root")
        self.assertIsNotNone(new_d)
        self.assertEqual(new_d.address, expected[6])
        self.assertEqual(heap.payload(new_d), 44)
        # Weak edge to pinned c kept (address unchanged), weak edge to
        # dead b cleared and reported at the post-collection address.
        self.assertEqual(heap.get_weak(new_d, 0).address, 4)
        self.assertIsNone(heap.get_weak(new_d, 1))
        self.assertEqual(result.cleared_weak_edges, ((expected[6], 1),))
        assert_no_overlap(self, heap)

    def test_pin_is_a_strong_root(self):
        heap = Heap(10, pinning=True)
        hx = heap.allocate(7, n_strong=1)
        hy = heap.allocate(8)
        heap.set_strong(hx, 0, hy)
        px = heap.pin(hx)  # no explicit roots at all
        heap.collect()
        self.assertEqual(heap.payload(px), 7)
        # Strongly reachable child of a pinned object survives too.
        child = heap.get_strong(px, 0)
        self.assertEqual(heap.payload(child), 8)
        self.assertIn(px.address, heap.reachable_addresses())

    def test_repeated_pins_counted_separately(self):
        heap = Heap(10, pinning=True)
        hx = heap.allocate(5)
        p1 = heap.pin(hx)
        p2 = heap.pin(hx)
        self.assertEqual(heap.pin_count(hx.address), 2)
        self.assertNotEqual(p1.generation, p2.generation)
        heap.release(p1)
        self.assertEqual(heap.pin_count(hx.address), 1)
        # One release must not unpin: still fixed and alive, p2 usable.
        result = heap.collect()
        self.assertEqual(result.relocated, {})
        self.assertEqual(heap.payload(p2), 5)
        self.assertEqual(heap.pin_count(hx.address), 1)
        heap.release(p2)
        self.assertEqual(heap.pin_count(hx.address), 0)
        # Fully unpinned and unrooted: reclaimed by the next collection.
        heap.collect()
        self.assertEqual(heap.objects(), ())

    def test_stale_pin_handle_cannot_touch_recycled_address(self):
        heap = Heap(10, pinning=True)
        hx = heap.allocate(1)
        px = heap.pin(hx)
        addr = px.address
        heap.release(px)
        heap.collect()  # x is unpinned and unrooted -> reclaimed
        # Refill the heap so the slot is reused by a different object.
        newcomer = heap.allocate(99)
        self.assertEqual(newcomer.address, addr)
        with self.assertRaises(StaleHandleError):
            heap.payload(px)
        with self.assertRaises(StaleHandleError):
            heap.set_payload(px, 3)
        with self.assertRaises(StaleHandleError):
            heap.release(px)
        # Forged pin handles are rejected too.
        with self.assertRaises(StaleHandleError):
            heap.payload(PinHandle(addr, 12345))
        # The new object itself is unaffected.
        self.assertEqual(heap.payload(heap.handle(addr)), 99)

    def test_release_validation(self):
        heap = Heap(10, pinning=True)
        hx = heap.allocate(1)
        px = heap.pin(hx)
        heap.release(px)
        with self.assertRaises(StaleHandleError):
            heap.release(px)  # double release
        with self.assertRaises(TypeError):
            heap.release(hx)  # wrong handle type

    def test_pin_requires_pinning_mode(self):
        heap = Heap(10)
        hx = heap.allocate(1)
        with self.assertRaises(GCError):
            heap.pin(hx)

    def test_normal_handle_epoch_rules_preserved_in_pinning_mode(self):
        heap = Heap(10, pinning=True)
        hx = heap.allocate(1)
        heap.add_root("r", hx)
        heap.collect()
        with self.assertRaises(StaleHandleError):
            heap.payload(hx)  # pre-collection Handle is stale
        fresh = heap.root_handle("r")
        self.assertEqual(heap.payload(fresh), 1)

    def test_cheney_mode_unchanged_by_pinning_feature(self):
        # Same scenario without pinning: everything moves, epoch bumps.
        heap = Heap(20)
        ha = heap.allocate(11)
        hb = heap.allocate(22)
        heap.add_root("r", hb)
        result = heap.collect()
        self.assertEqual(result.relocated, {2: 0})
        self.assertEqual(result.pinned_addresses, ())
        self.assertEqual(heap.epoch, 1)
        with self.assertRaises(StaleHandleError):
            heap.payload(ha)


class FragmentationTest(unittest.TestCase):
    def _checkerboard(self):
        """Pinned objects at 0, 4, 8 of a 12-slot heap; holes of 2."""
        heap = Heap(12, pinning=True)
        pins = []
        for i in range(6):
            h = heap.allocate(i)  # each 2 slots: 0,2,4,6,8,10
            if i % 2 == 0:
                pins.append(heap.pin(h))
        heap.collect()  # odd objects die; pinned stay at 0, 4, 8
        return heap, pins

    def test_fragmentation_error_when_no_contiguous_run(self):
        heap, pins = self._checkerboard()
        self.assertEqual(heap.pinned_addresses(), (0, 4, 8))
        self.assertEqual(heap.free_slots, 6)
        # 3 contiguous slots needed, 6 free in total -> fragmented.
        with self.assertRaises(FragmentationError) as ctx:
            heap.allocate(0, n_strong=1)  # size 3
        self.assertIsInstance(ctx.exception, HeapCapacityError)
        self.assertIn("fragment", str(ctx.exception))
        # A 2-slot object still fits into a hole.
        h = heap.allocate(42)
        self.assertEqual(h.address, 2)
        self.assertEqual(heap.payload(h), 42)
        assert_no_overlap(self, heap)

    def test_release_one_layer_then_collect_merges_holes(self):
        heap, pins = self._checkerboard()
        # Release the pin at address 0; after collection the leading
        # holes merge into a run that fits a larger object.
        heap.release(pins[0])
        heap.collect()
        self.assertEqual(heap.pinned_addresses(), (4, 8))
        big = heap.allocate(7, n_strong=2)  # size 4
        self.assertEqual(big.address, 0)
        assert_no_overlap(self, heap)

    def test_genuine_capacity_error_still_reported(self):
        heap, pins = self._checkerboard()
        # Fill every hole with rooted objects, then ask for more.
        for i in range(3):
            filler = heap.allocate(0)  # fills the three 2-slot holes
            heap.add_root(f"f{i}", filler)
        self.assertEqual(heap.free_slots, 0)
        with self.assertRaises(HeapCapacityError) as ctx:
            heap.allocate(0)
        self.assertNotIsInstance(ctx.exception, FragmentationError)
        self.assertIn("out of capacity", str(ctx.exception))

    def test_oversized_object_reports_capacity_not_fragmentation(self):
        heap = Heap(6, pinning=True)
        with self.assertRaises(HeapCapacityError) as ctx:
            heap.allocate(0, n_strong=5)  # size 7 > capacity 6
        self.assertNotIsInstance(ctx.exception, FragmentationError)
        self.assertIn("exceeds semi-space capacity", str(ctx.exception))


class CompactionSequenceTest(unittest.TestCase):
    def test_stepwise_sequence_with_reference_model(self):
        """Pinned-sandwich scenario checked against the independent model
        after every step: intervals, reachability, pinned addresses."""
        heap = Heap(24, pinning=True)
        handles = [heap.allocate(100 + i, n_weak=1) for i in range(6)]
        # Weak web: every object weakly references the next one.
        for i in range(5):
            heap.set_weak(handles[i], 0, handles[i + 1])
        pins = [heap.pin(handles[1]), heap.pin(handles[4])]
        pinned_addrs = {p.address for p in pins}
        heap.add_root("first", handles[0])

        live = {h.address: 3 for h in (handles[0], handles[1], handles[4])}
        expected = expected_compaction(live, pinned_addrs)
        result = heap.collect()

        self.assertEqual(
            result.relocated, {o: n for o, n in expected.items() if o != n}
        )
        assert_no_overlap(self, heap)
        # Pinned addresses unchanged; payloads intact through pin handles.
        for p, payload in zip(pins, (101, 104)):
            self.assertIn(p.address, pinned_addrs)
            self.assertEqual(heap.payload(p), payload)
        # Root forwarded to the compacted address of object 0.
        root = heap.root_handle("first")
        self.assertEqual(root.address, expected[handles[0].address])
        # Weak edge from survivor 0 to pinned object 1 stays valid.
        self.assertEqual(heap.get_weak(root, 0).address, pins[0].address)
        # Reachability matches the live set exactly.
        self.assertEqual(
            heap.reachable_addresses(),
            frozenset(expected.values()),
        )

        # Second round: unpin object 4, collect again; it must die.
        heap.release(pins[1])
        heap.collect()
        self.assertEqual(heap.pinned_addresses(), (pins[0].address,))
        assert_no_overlap(self, heap)
        self.assertEqual(heap.payload(pins[0]), 101)
        self.assertEqual(heap.payload(heap.root_handle("first")), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
