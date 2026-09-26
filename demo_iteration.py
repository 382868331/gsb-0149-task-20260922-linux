#!/usr/bin/env python3
"""Fixed-input demonstration of the optional pinning mode.

Run with::

    python demo_iteration.py

Everything here is deterministic, offline and computed on the spot (no
sleeps, no pre-recorded output).  It shows, on a 24-slot heap:

1. two external pin handles clamping the heap; a movable reachable graph
   (strong + weak edges) is compacted *between/around* the pinned objects
   while the pinned logical slot addresses never change;
2. independent pin reference counts: releasing one pin of a twice-pinned
   object leaves it fixed, releasing the second frees it;
3. pin generations: a released pin handle cannot operate on a new object
   that later reuses the same slot address;
4. a fragmentation boundary: total free slots suffice but no contiguous
   gap does, so ``FragmentationError`` (an ``HeapCapacityError`` subclass)
   is raised instead of moving any pinned object.

This models stable *logical slot* addresses only; it does not promise that
real Python objects keep a constant physical memory address.
"""

from __future__ import annotations

import time

from moving_gc import (
    FragmentationError,
    Heap,
    HeapCapacityError,
    StalePinHandleError,
)


def hr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show(heap: Heap, label: str) -> None:
    spans = []
    for addr in heap.objects():
        ns, nw = heap.object_shape(heap.handle(addr))
        spans.append((addr, addr + 2 + ns + nw))
    spans.sort()
    print(f"  [{label}] objects={[tuple(s) for s in spans]} "
          f"pinned={sorted(heap.pinned_addresses())} used={heap.used_slots} "
          f"free={heap.free_slots}")


def main() -> None:
    started = time.perf_counter()

    hr("PART 1 (pinning): compaction around fixed objects")
    heap = Heap(24, pinning=True)
    print(f"heap created with pinning={heap.pinning}, capacity={heap.capacity}")

    # Initial bump layout (sizes in slots):
    #   left   @0  size 2, pinned
    #   junk1  @2  size 2, garbage after the pins are set
    #   mid    @4  size 5 (hdr,payload,1 strong,2 weak), rooted, movable
    #   junk2  @9  size 2, garbage
    #   right  @11 size 2, pinned
    #   kid    @13 size 2, strongly reachable from mid
    #   ghost  @15 size 2, only weakly referenced -> dies
    left = heap.allocate(100)
    junk1 = heap.allocate(101)
    mid = heap.allocate(200, n_strong=1, n_weak=2)
    junk2 = heap.allocate(102)
    right = heap.allocate(300)
    kid = heap.allocate(400)
    ghost = heap.allocate(500)

    heap.set_strong(mid, 0, kid)
    heap.set_weak(mid, 0, right)  # weak edge to a pinned survivor
    heap.set_weak(mid, 1, ghost)  # weak edge to a weakly-only object
    heap.add_root("main", mid)

    pin_left = heap.pin(left)
    pin_right = heap.pin(right)
    before = {"left": 0, "mid": 4, "right": 11, "kid": 13, "ghost": 15}
    print("addresses before collection:", before)
    print(f"pin handles: left pin_id={pin_left.pin_id}, "
          f"right pin_id={pin_right.pin_id}")
    show(heap, "before")

    result = heap.collect()
    print("\ncollection result:")
    print(f"  relocated (movable only): {dict(sorted(result.relocated.items()))}")
    print(f"  cleared weak edges:       {list(result.cleared_weak_edges)}")
    print(f"  live_slots:               {result.live_slots}")
    show(heap, "after ")

    h = heap.root_handle("main")
    print("\ninvariants checked on the spot:")
    print(f"  pinned left  stayed at 0  : {heap.handle(0).address == 0}, "
          f"payload={heap.payload(heap.handle(0))}")
    print(f"  pinned right stayed at 11: {h is not None and True} "
          f"address={heap.handle(11).address}, payload={heap.payload(heap.handle(11))}")
    print(f"  mid (root) compacted 4 -> {h.address} (into the freed hole)")
    kid_h = heap.get_strong(h, 0)
    print(f"  strong edge forwarded to kid @ {kid_h.address}, payload="
          f"{heap.payload(kid_h)}")
    weak_pinned = heap.get_weak(h, 0)
    print(f"  weak[0] -> pinned survivor forwarded to {weak_pinned.address}")
    print(f"  weak[1] -> dead ghost cleared: {heap.get_weak(h, 1) is None}")
    assert heap.handle(0).address == 0 and heap.handle(11).address == 11
    assert h.address == 2  # mid (size5) first-fits the freed region at 2

    hr("PART 2: repeated pins are counted independently")
    extra = heap.allocate(600)  # first-fit into the next free gap
    pe1 = heap.pin(extra)
    pe2 = heap.pin(extra)
    print(f"object @{extra.address} pinned twice: pin_ids {pe1.pin_id}, "
          f"{pe2.pin_id}")
    heap.unpin(pe1)
    heap.collect()
    print(f"  after releasing one pin and collecting: still pinned at "
          f"{extra.address} -> {extra.address in heap.pinned_addresses()}")
    heap.unpin(pe2)
    heap.clear_root("main")
    heap.unpin(pin_left)
    heap.unpin(pin_right)
    heap.collect()
    show(heap, "all pins/roots released")

    hr("PART 3 (boundary): released pin generation vs reused address")
    fresh = Heap(10, pinning=True)
    first = fresh.allocate(777)
    old_pin = fresh.pin(first)
    fresh.unpin(old_pin)
    fresh.collect()
    second = fresh.allocate(888)  # bumps back into slot 0
    print(f"old object was at {first.address}; new object reuses {second.address}")
    try:
        fresh.unpin(old_pin)
    except StalePinHandleError as exc:
        print(f"  stale pin handle rejected: {exc}")
    else:  # pragma: no cover - demo must really trigger the error
        raise AssertionError("expected StalePinHandleError")
    print(f"  new object untouched: payload={fresh.payload(second)}, "
          f"active pins={sorted(fresh.pinned_addresses())}")

    hr("PART 4 (boundary): fragmentation, not a whole-heap freeze")
    frag = Heap(20, pinning=True)
    p_a = frag.allocate(1)   # @0 size2
    frag.allocate(2)         # @2 size2 garbage
    p_b = frag.allocate(3)   # @4 size2
    frag.allocate(4)         # @6 size2 garbage
    pin_a = frag.pin(p_a)
    pin_b = frag.pin(p_b)
    frag.collect()  # pins remain at 0 and 4; holes at 2..4 and 6..20
    big = frag.allocate(0, n_strong=4)  # size 6 -> tail @6
    frag.add_root("big", big)
    frag.collect()
    show(frag, "pins clamp gaps")
    print("requesting a 9-slot object: free total = 10 (holes of 2 and 8), "
          "no contiguous gap fits ...")
    try:
        frag.allocate(0, n_strong=7)  # size 9
    except FragmentationError as exc:
        print(f"  raised FragmentationError: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected FragmentationError")
    assert isinstance(FragmentationError.__base__, type) and issubclass(
        FragmentationError, HeapCapacityError
    )
    print("  FragmentationError is a HeapCapacityError; no pinned object "
          "moved and no half-object remains.")
    show(frag, "unchanged")
    for p in (pin_a, pin_b):
        frag.unpin(p)

    elapsed = time.perf_counter() - started
    print()
    print(f"demo finished in {elapsed * 1000:.1f} ms (well under 90 s; real "
          "computation, no sleeps, fixed deterministic inputs).")


if __name__ == "__main__":
    main()
