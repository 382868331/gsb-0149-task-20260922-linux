"""Live demonstration of the optional pinning mode (moving_gc).

Runs in well under 90 seconds: every number below is computed on the
spot by the collector, nothing is pre-recorded and nothing sleeps.

Scenarios shown:
  1. pinned objects stay put while movable objects compact around them;
  2. weak edges to pinned objects survive, weak edges to garbage clear;
  3. repeated pins are counted separately;
  4. a released pin handle cannot touch the object reusing its address;
  5. fragmentation failure: enough free slots overall, no contiguous run.
"""

from __future__ import annotations

from moving_gc import (
    FragmentationError,
    Heap,
    PinHandle,
    StaleHandleError,
)


def show_layout(heap: Heap, label: str) -> None:
    objs = []
    for addr in heap.objects():
        n_strong, n_weak = heap.object_shape(heap.handle(addr))
        objs.append(f"[{addr}..{addr + 1 + n_strong + n_weak}]")
    pinned = heap.pinned_addresses()
    print(
        f"  {label}: objects at {' '.join(objs) or '(empty)'}"
        f" | pinned={pinned} used={heap.used_slots} free={heap.free_slots}"
        f" epoch={heap.epoch}"
    )


def main() -> None:
    print("== 1. compaction between pinned objects ==")
    heap = Heap(20, pinning=True)
    ha = heap.allocate(11)
    hb = heap.allocate(22, n_strong=2)     # 4-slot object, will be garbage
    hc = heap.allocate(33)
    hd = heap.allocate(44, n_weak=2)       # 4-slot survivor, slides into b's hole
    pa, pc = heap.pin(ha), heap.pin(hc)
    heap.add_root("root", hd)
    heap.set_weak(hd, 0, hc)               # weak edge to a pinned object
    heap.set_weak(hd, 1, hb)               # weak edge to garbage
    show_layout(heap, "before collect")
    result = heap.collect()
    show_layout(heap, "after collect ")
    print(f"  relocated={result.relocated} pinned stayed at {result.pinned_addresses}")
    root = heap.root_handle("root")
    print(f"  weak->pinned still valid: payload {heap.payload(heap.get_weak(root, 0))}")
    print(f"  weak->garbage cleared: {heap.get_weak(root, 1)}"
          f" (reported {result.cleared_weak_edges})")
    print(f"  pin handles still work across epoch: pa={heap.payload(pa)}"
          f" pc={heap.payload(pc)}")

    print("== 2. repeated pins counted separately ==")
    p1b = heap.pin(heap.handle(pc.address))  # second pin on the same object
    print(f"  pin_count({pc.address}) = {heap.pin_count(pc.address)}")
    heap.release(pc)
    heap.collect()
    print(f"  released one pin, object still fixed at {p1b.address},"
          f" payload {heap.payload(p1b)}")
    heap.release(p1b)
    print(f"  released last pin, pin_count({p1b.address}) ="
          f" {heap.pin_count(p1b.address)}")

    print("== 3. stale pin handle vs recycled address ==")
    heap2 = Heap(10, pinning=True)
    px = heap2.pin(heap2.allocate(1))
    addr = px.address
    heap2.release(px)
    heap2.collect()                        # unpinned + unrooted -> gone
    fresh = heap2.allocate(99)             # reuses the same slot
    print(f"  address {addr} reused by a new object (payload {heap2.payload(fresh)})")
    try:
        heap2.payload(px)
    except StaleHandleError as exc:
        print(f"  old pin handle rejected: {exc}")
    assert isinstance(px, PinHandle)

    print("== 4. fragmentation failure ==")
    heap3 = Heap(12, pinning=True)
    pins = []
    for i in range(6):
        h = heap3.allocate(i)
        if i % 2 == 0:
            pins.append(heap3.pin(h))
    heap3.collect()
    show_layout(heap3, "checkerboard")
    try:
        heap3.allocate(0, n_strong=1)      # needs 3 contiguous slots
    except FragmentationError as exc:
        print(f"  allocate(3 slots) failed: {exc}")
    small = heap3.allocate(7)              # 2 slots fit into a hole
    print(f"  allocate(2 slots) succeeded at address {small.address}")
    print("  releasing pin at 0 and collecting merges the holes...")
    heap3.release(pins[0])
    heap3.collect()
    big = heap3.allocate(9, n_strong=2)    # 4 contiguous slots now fit
    print(f"  allocate(4 slots) succeeded at address {big.address}")
    show_layout(heap3, "after merge   ")


if __name__ == "__main__":
    main()
