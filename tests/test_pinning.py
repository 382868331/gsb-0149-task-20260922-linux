"""Tests for the optional pinning mode of the moving_gc kernel.

Expected layouts are computed by hand from the documented contract:
pinned objects keep their exact logical slot interval and act as strong
roots; movable reachable objects are first-fit packed into gaps in
ascending old-address order; strong and weak edges (and roots) are
forwarded normally.  These tests never rely on the implementation to
compute the expected addresses.
"""

from __future__ import annotations

import unittest

from moving_gc import (
    FragmentationError,
    GCError,
    Heap,
    HeapCapacityError,
    PinHandle,
    StaleHandleError,
    StalePinHandleError,
)


def intervals(heap: Heap) -> list[tuple[int, int]]:
    """Occupied [start, end) intervals sorted, asserted non-overlapping."""
    spans = []
    for addr in heap.objects():
        ns, nw = heap.object_shape(heap.handle(addr))
        spans.append((addr, addr + 2 + ns + nw))
    spans.sort()
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 <= s2, f"overlapping objects: {s1, e1} vs {s2, e2}"
    return spans


class PinModeBasicTests(unittest.TestCase):
    def test_pin_keeps_address_and_acts_as_root(self):
        heap = Heap(20, pinning=True)
        a = heap.allocate(10)
        junk = heap.allocate(11)
        pin = heap.pin(a)
        # No explicit roots at all: only the pin keeps ``a`` alive.
        result = heap.collect()
        self.assertIsInstance(pin, PinHandle)
        self.assertEqual(heap.pinned_addresses(), frozenset({a.address}))
        self.assertEqual(heap.objects(), (a.address,))
        self.assertNotIn(junk.address, result.relocated)
        self.assertEqual(heap.reachable_addresses(), frozenset({a.address}))
        self.assertEqual(heap.payload(heap.handle(a.address)), 10)
        heap.unpin(pin)
        heap.collect()
        self.assertEqual(heap.objects(), ())

    def test_repeated_pins_are_counted_independently(self):
        heap = Heap(10, pinning=True)
        a = heap.allocate(1)
        p1 = heap.pin(a)
        p2 = heap.pin(a)
        self.assertNotEqual(p1.pin_id, p2.pin_id)
        heap.unpin(p1)
        heap.collect()
        self.assertEqual(heap.objects(), (a.address,))
        with self.assertRaises(StalePinHandleError):
            heap.unpin(p1)
        heap.unpin(p2)
        heap.collect()
        self.assertEqual(heap.objects(), ())

    def test_compaction_between_pinned_objects(self):
        # capacity 20 layout before GC:
        #   a @0 size2 pinned; m @2 size4 rooted movable;
        #   c @6 size2 pinned; d @8 size2 movable, strongly held by m.
        heap = Heap(20, pinning=True)
        a = heap.allocate(10)
        m = heap.allocate(20, n_strong=1, n_weak=1)
        c = heap.allocate(30)
        d = heap.allocate(40)
        heap.set_strong(m, 0, d)
        heap.set_weak(m, 0, c)
        heap.pin(a)
        heap.pin(c)
        heap.add_root("r", m)

        result = heap.collect()
        # m (size4) first-fits gap 2..6 exactly; d (size2) first-fits the
        # tail gap starting at 8 (its old slot), so it stays at 8.
        self.assertEqual(result.relocated, {2: 2, 8: 8})
        self.assertEqual(heap.objects(), (0, 2, 6, 8))
        self.assertEqual(intervals(heap), [(0, 2), (2, 6), (6, 8), (8, 10)])
        self.assertEqual(heap.used_slots, 10)

        root = heap.root_handle("r")
        self.assertEqual(root.address, 2)
        self.assertEqual(heap.get_strong(root, 0).address, 8)
        weak_c = heap.get_weak(root, 0)
        self.assertIsNotNone(weak_c)
        self.assertEqual(weak_c.address, 6)
        self.assertEqual(heap.payload(heap.handle(0)), 10)
        self.assertEqual(heap.payload(heap.handle(6)), 30)

    def test_movable_packs_into_hole_left_by_garbage(self):
        # Pins clamp a hole; a rooted movable at the tail packs into it.
        heap = Heap(16, pinning=True)
        p1obj = heap.allocate(1)             # @0 size2
        heap.allocate(2)                     # @2 size2 garbage
        heap.allocate(3)                     # @4 size2 garbage
        p2obj = heap.allocate(4)             # @6 size2
        tail = heap.allocate(5, n_strong=1)  # @8 size3
        child = heap.allocate(6)             # @11 size2
        heap.set_strong(tail, 0, child)
        heap.pin(p1obj)
        heap.pin(p2obj)
        heap.add_root("r", tail)
        result = heap.collect()
        # tail size3 first-fits gap 2..6 -> @2; child size2 next-fits @8.
        self.assertEqual(result.relocated, {8: 2, 11: 8})
        self.assertEqual(intervals(heap), [(0, 2), (2, 5), (6, 8), (8, 10)])
        root = heap.root_handle("r")
        self.assertEqual(heap.get_strong(root, 0).address, 8)

    def test_release_one_pin_layer_then_collect(self):
        heap = Heap(12, pinning=True)
        a = heap.allocate(1)              # @0 size2
        b = heap.allocate(2, n_strong=1)  # @2 size3
        pa = heap.pin(a)
        pb1 = heap.pin(b)
        pb2 = heap.pin(b)
        heap.set_strong(b, 0, a)
        heap.collect()
        self.assertEqual(heap.objects(), (0, 2))
        heap.unpin(pb1)
        heap.collect()
        self.assertEqual(heap.objects(), (0, 2))
        heap.unpin(pb2)
        heap.unpin(pa)
        heap.collect()
        self.assertEqual(heap.objects(), ())

    def test_weak_edge_to_pinned_object_survives(self):
        heap = Heap(10, pinning=True)
        holder = heap.allocate(1, n_weak=1)
        pinned = heap.allocate(2)
        heap.set_weak(holder, 0, pinned)
        heap.add_root("h", holder)
        ph = heap.pin(pinned)
        result = heap.collect()
        self.assertEqual(result.cleared_weak_edges, ())
        h = heap.root_handle("h")
        weak = heap.get_weak(h, 0)
        self.assertIsNotNone(weak)
        self.assertEqual(weak.address, pinned.address)
        heap.unpin(ph)

    def test_weak_edge_to_dead_unpinned_object_cleared(self):
        heap = Heap(10, pinning=True)
        holder = heap.allocate(1, n_weak=1)
        victim = heap.allocate(2)
        heap.set_weak(holder, 0, victim)
        heap.add_root("h", holder)
        pin_h = heap.pin(holder)
        result = heap.collect()
        # holder pinned at 0; victim weakly-only and unpinned -> cleared.
        self.assertEqual(result.cleared_weak_edges, ((0, 0),))
        self.assertIsNone(heap.get_weak(heap.handle(0), 0))
        heap.unpin(pin_h)

    def test_fragmentation_error_when_no_contiguous_gap(self):
        heap = Heap(20, pinning=True)
        p1obj = heap.allocate(1)  # @0 size2
        heap.allocate(2)          # @2 garbage
        p2obj = heap.allocate(3)  # @4 size2
        heap.allocate(4)          # @6 garbage
        heap.pin(p1obj)
        heap.pin(p2obj)
        heap.collect()  # pins at 0 and 4; gaps 2..4 and 6..20
        big = heap.allocate(0, n_strong=4)  # size 6 -> tail @6
        heap.add_root("big", big)
        heap.collect()
        self.assertEqual(heap.objects(), (0, 4, 6))
        # Free: gap 2..4 (2) and 12..20 (8) = 10 total; size 9 fits neither.
        with self.assertRaises(FragmentationError):
            heap.allocate(0, n_strong=7)
        # Failed emergency cycle left no partial state.
        self.assertEqual(heap.objects(), (0, 4, 6))

    def test_capacity_error_when_total_free_insufficient(self):
        heap = Heap(10, pinning=True)
        a = heap.allocate(1)
        heap.pin(a)  # 2 slots pinned forever
        heap.collect()
        with self.assertRaises(HeapCapacityError):
            heap.allocate(0, n_strong=8)  # size 10 cannot coexist with pin

    def test_stale_pin_handle_cannot_touch_reused_address(self):
        heap = Heap(10, pinning=True)
        a = heap.allocate(1)
        ph = heap.pin(a)
        heap.unpin(ph)
        heap.collect()          # a reclaimed; slot 0 free
        b = heap.allocate(2)    # reuses address 0
        self.assertEqual(b.address, a.address)
        with self.assertRaises(StalePinHandleError):
            heap.unpin(ph)
        self.assertEqual(heap.pinned_addresses(), frozenset())
        self.assertEqual(heap.payload(b), 2)

    def test_pin_handle_validation(self):
        heap = Heap(10, pinning=True)
        with self.assertRaises(TypeError):
            heap.unpin(123)  # type: ignore[arg-type]
        with self.assertRaises(StalePinHandleError):
            heap.unpin(PinHandle(0, 999))

    def test_pinning_disabled_rejects_pin_api(self):
        heap = Heap(10)
        self.assertFalse(heap.pinning)
        a = heap.allocate(1)
        with self.assertRaises(GCError):
            heap.pin(a)
        with self.assertRaises(GCError):
            heap.unpin(PinHandle(0, 0))

    def test_disabled_mode_keeps_cheney_and_epoch_rules(self):
        heap = Heap(20)
        a = heap.allocate(1)
        b = heap.allocate(2, n_strong=1)
        heap.set_strong(b, 0, a)
        heap.add_root("r", b)
        result = heap.collect()
        # Deterministic Cheney order: root b first, then its edge a.
        self.assertEqual(list(result.relocated.values()), [0, 3])
        with self.assertRaises(StaleHandleError):
            heap.payload(a)

    def test_first_fit_allocation_uses_earliest_gap(self):
        heap = Heap(20, pinning=True)
        a = heap.allocate(1)  # @0 size2
        heap.allocate(2)      # @2 size2
        c = heap.allocate(3)  # @4 size2
        heap.pin(a)
        heap.pin(c)
        heap.collect()  # middle object gone; hole at 2..4
        small = heap.allocate(9)
        self.assertEqual(small.address, 2)
        self.assertEqual(heap.objects(), (0, 2, 4))

    def test_pinned_intervals_invariant_across_collections(self):
        heap = Heap(30, pinning=True)
        pins = []
        objs = [heap.allocate(100 + i, n_strong=1) for i in range(4)]
        for obj in objs:
            pins.append(heap.pin(obj))
        heap.add_root("r", objs[3])
        heap.set_strong(objs[3], 0, objs[1])
        addresses_before = {obj.address for obj in objs}
        for _ in range(3):
            heap.collect()
            intervals(heap)
        self.assertEqual(set(heap.objects()), addresses_before)
        for pin in pins:
            heap.unpin(pin)

    def test_constructor_rejects_bad_pinning_flag(self):
        with self.assertRaises(TypeError):
            Heap(10, pinning=1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
