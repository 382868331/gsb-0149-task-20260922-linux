"""Fixed-input demo of the gc_kernel moving garbage collector.

Part 1 (normal result): builds a small object graph containing a self-loop,
a shared subgraph, a weak edge to a live object and a weak edge to a
weak-only object, then collects and prints the relocation map, showing that
addresses actually changed and that the dead weak edge was cleared.

Part 2 (rejection boundary, actually triggered): reuses a pre-collection
handle (stale) and over-allocates an exactly full heap, printing the raised
errors.

Runs in well under 8 seconds; no sleeps, no recorded output.
"""

from gc_kernel import Heap, HeapFullError, StaleHandleError


def show(heap, label, handle):
    print(f"  {label}: {heap.describe(handle)}")


def main() -> None:
    print("=== Part 1: moving collection with weak references ===")
    heap = Heap(capacity=16, strong_arity=2, weak_arity=1)

    a = heap.add_root(heap.alloc(payload=100))          # root A
    b = heap.alloc(payload=200)                          # self-loop B
    c = heap.alloc(payload=300)                          # shared child C
    d = heap.add_root(heap.alloc(payload=400))           # root D
    weak_only = heap.alloc(payload=500)                  # only weakly reachable
    heap.write_strong(b, 0, b)                           # B -> B (self loop)
    heap.write_strong(a, 0, b)                           # A -> B
    heap.write_strong(a, 1, c)                           # A -> C
    heap.write_strong(d, 0, c)                           # D -> C (shared subgraph)
    heap.write_weak(a, 0, weak_only)                     # weak -> dead target
    heap.write_weak(b, 0, c)                             # weak -> live target

    print("Before collection (epoch %d):" % heap.epoch)
    show(heap, "A", a)
    show(heap, "B", b)
    show(heap, "C", c)
    show(heap, "D", d)

    result = heap.collect()

    print(f"\nCollection: live_slots={result.live_slots} "
          f"freed_slots={result.freed_slots} epoch -> {result.epoch}")
    print(f"Relocation map (old address -> new address): {result.moved}")
    moved = {old: new for old, new in result.moved.items() if old != new}
    print(f"Objects whose address actually changed: {moved}")

    new_a, new_d = result.roots
    print("\nAfter collection (epoch %d):" % heap.epoch)
    show(heap, "A", new_a)
    show(heap, "B", heap.read_strong(new_a, 0))
    show(heap, "D", new_d)
    shared_from_a = heap.read_strong(new_a, 1)
    shared_from_d = heap.read_strong(new_d, 0)
    print(f"  shared subgraph C reachable identically from A and D: "
          f"{shared_from_a == shared_from_d} (address {shared_from_a.address})")
    print(f"  A.weak[0] to weak-only object cleared: "
          f"{heap.read_weak(new_a, 0) is None}")
    b_new = heap.read_strong(new_a, 0)
    print(f"  B.weak[0] to live C remapped to new address: "
          f"{heap.read_weak(b_new, 0)}")

    print("\n=== Part 2: triggered rejection boundaries ===")
    try:
        heap.read_payload(a)  # 'a' was minted before the collection
        print("  ERROR: stale handle was not rejected")
    except StaleHandleError as exc:
        print(f"  Stale handle rejected: {exc}")

    tiny = Heap(capacity=2, strong_arity=0, weak_arity=0)
    tiny.add_root(tiny.alloc(payload=1))
    tiny.add_root(tiny.alloc(payload=2))
    try:
        tiny.alloc(payload=3)  # exactly full, everything reachable
        print("  ERROR: full-heap allocation was not rejected")
    except HeapFullError as exc:
        print(f"  Full heap rejected: {exc}")
    print(f"  Heap left consistent after refusal: "
          f"live_slots={tiny.live_slots}, payloads="
          f"{[tiny.read_payload(h) for h in tiny.roots()]}")


if __name__ == "__main__":
    main()
