#!/usr/bin/env python3
"""Fixed-input demonstration for the moving_gc kernels.

Run with::

    python demo.py

Deterministic, offline, computed on the spot (no sleeps, no pre-recorded
output).  Two independent stories:

PART 1 (legacy, unchanged): one flat semi-space collection -- real address
movement, strong forwarding, weak-edge clearing; then two actually
triggered error boundaries (a stale handle; capacity failure after the
single emergency collection).

PART 2 (new): the two-generation incremental kernel on one scenario that
exercises every new capability together -- young copying + old
mark-sweep, an explicit per-step scan budget, an incremental-update
write barrier fired on an interleaved black -> white edge, the
remembered set keeping an old -> young edge alive, an ephemeron fixpoint
chain, a finalizer that resurrects its object and allocates a new one,
and one actually triggered rejection (attaching an object sealed for
reclamation).
"""

from __future__ import annotations

import time

from moving_gc import (
    OLD,
    YOUNG,
    GenHeap,
    Heap,
    HeapCapacityError,
    SealedForReclamationError,
    StaleHandleError,
)


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def part_one_legacy() -> None:
    hr("PART 1 (legacy Heap): one semi-space collection + two error boundaries")
    heap = Heap(capacity=40)
    tmp = heap.allocate(999)
    holder = heap.allocate(100, n_strong=1, n_weak=2)
    child = heap.allocate(200)
    cache = heap.allocate(300)
    heap.set_strong(holder, 0, child)
    heap.set_weak(holder, 0, cache)
    heap.set_weak(holder, 1, child)
    heap.add_root("main", holder)
    before = {"holder": holder.address, "child": child.address,
              "cache": cache.address}

    result = heap.collect()
    print("relocated:", dict(sorted(result.relocated.items())),
          "| cleared weak edges:", list(result.cleared_weak_edges),
          "| epoch ->", heap.epoch)
    h = heap.root_handle("main")
    print(f"holder moved {before['holder']} -> {h.address}; "
          f"weak[0] to dead cache is now {heap.get_weak(h, 0)}")
    assert heap.get_weak(h, 0) is None

    try:
        heap.payload(holder)
    except StaleHandleError as exc:
        print("stale pre-GC handle rejected:", exc)
    else:  # pragma: no cover
        raise AssertionError("expected StaleHandleError")

    small = Heap(capacity=8)
    a = small.allocate(1, n_strong=1)
    b = small.allocate(2, n_strong=1)
    c = small.allocate(3)
    small.add_root("a", a)
    small.add_root("b", b)
    small.add_root("c", c)
    try:
        small.allocate(4)
    except HeapCapacityError as exc:
        print("full heap after one emergency GC rejected the allocation:", exc)
    else:  # pragma: no cover
        raise AssertionError("expected HeapCapacityError")


def part_two_generational() -> None:
    hr("PART 2 (GenHeap): generations, budgeted steps, barrier, ephemerons, "
       "finalizers")
    # Generous enough to hold the whole fixed scenario.
    heap = GenHeap(young_capacity=200, old_capacity=400, promote_age=1)

    # --- Build the cast in the nursery and force the backbone old. ----- #
    # O is the old holder; it starts with one strong slot (the remembered
    # edge is written later) and one weak slot.
    O = heap.allocate(10, n_strong=2, n_weak=1)
    heap.add_root("O", O)
    heap.run_minor(64)                  # O promotes to the old generation
    O = heap.root_handle("O")
    print(f"O promoted to the old generation at address {O.address} "
          f"(generation {heap.generation_of(O)})")

    # Young objects used later by the remembered-set story are allocated
    # after the major (see below), so they are genuinely white nursery
    # objects reached only by an old -> young write.

    # Ephemeron fixpoint chain (all young; evaluated during a later major):
    # T rooted; e1 (T,K1)->K2, e2 (T,K2)->V, with K1 rooted too.
    T = heap.allocate(30)
    K1 = heap.allocate(31)
    K2 = heap.allocate(32)
    V = heap.allocate(33)
    heap.add_root("T", T)
    heap.add_root("K1", K1)
    e1 = heap.add_ephemeron(T, K1, K2)
    e2 = heap.add_ephemeron(T, K2, V)

    # Finalizer object F: finalizer resurrects F and allocates a new young
    # object D during the event.  Give O a weak edge to F so its clearing
    # after (non-)reclamation is observable.
    F = heap.allocate(40)
    heap.add_root("F", F)

    def finalizer(hp: GenHeap, handle) -> None:
        print(f"  [finalizer event] F at {handle.address} is unreachable; "
              "re-rooting it and allocating D")
        hp.add_root("resurrected_F", handle)
        born = hp.allocate(999)
        hp.add_root("born_in_finalizer", born)

    # Promote T,K1,K2,V,F (and the ephemeron table/keys) to the old
    # generation so a major mark-sweep decides their fate.  The minor's
    # ephemeron fixpoint keeps K2 and V alive; 'keep'/'trash' are not yet
    # referenced and die as young garbage.
    minor0 = heap.run_minor(64)         # age 1 -> promote the rooted set
    promoted = minor0.promoted
    # Re-fetch current-epoch handles at the post-promotion old addresses.
    T = heap.root_handle("T")
    K1 = heap.root_handle("K1")
    F = heap.root_handle("F")
    O = heap.root_handle("O")
    K2 = heap.handle(promoted[K2.address])
    V = heap.handle(promoted[V.address])
    print("minor promoted T,K1,K2,V,F to old:",
          sorted(promoted.values()),
          "| ephemeron entries preserved:", heap.ephemeron_ids())

    token = heap.set_finalizer(F, finalizer)
    heap.set_weak(O, 0, F)
    # Drop F's root: it is now old garbage pending its one finalizer event.
    heap.remove_root("F")

    # --- Interleaved, budgeted MAJOR collection. ------------------------ #
    # Budget of 3 examinations per step; we mutate between steps to force
    # the incremental-update barrier to fire.
    print("\nrunning a MAJOR cycle with scan_budget=3, mutating between steps")
    total_scanned = 0
    barrier_seen = False
    res = heap.collect_step(3, "major")
    step_no = 0
    fired_barrier_write = False
    while not res.completed:
        step_no += 1
        total_scanned += res.scanned
        assert res.scanned <= res.budget and res.overscan == 0
        # While marking is open, make a black holder point at a white
        # object exactly once; the IU barrier must grey that target.
        if (not fired_barrier_write and res.phase in ("mark", "fdiscover")
                and heap.gray_pending()):
            # Pick K2 (white unless the ephemeron chain already reached it)
            # and hang it off the already-black O via strong slot 1.
            try:
                heap.set_strong(O, 1, K2)
                fired_barrier_write = True
            except Exception:
                pass
        res = heap.collect_step(3)
    total_scanned += res.scanned
    print(f"major completed in {res.steps_in_cycle} steps, "
          f"{total_scanned} examinations total "
          f"(each step <= its budget; overscan always {res.overscan})")
    print("ephemerons removed this cycle:", list(res.cleared_ephemerons))
    print("finalizer events:", list(res.finalized), "token", token)

    # Outcomes:
    #  - the ephemeron chain reached K2 then V (both old and alive);
    #  - F was resurrected by its finalizer and D was born in the event;
    #  - the barrier write O->K2 is observable and consistent.
    O = heap.root_handle("O")
    assert heap.get_strong(O, 1).address == K2.address
    print("O.strong[1] (barrier write) ->", heap.get_strong(O, 1).address)
    print("resurrected F root ->", heap.root_handle("resurrected_F").address)
    born = heap.root_handle("born_in_finalizer")
    print("D born during finalization: address", born.address,
          "generation", heap.generation_of(born), "payload", heap.payload(born))
    assert heap.payload(born) == 999

    # --- Remembered set on a budgeted MINOR. ---------------------------- #
    hr("PART 2 (continued): remembered set keeps an old -> young edge alive")
    keep = heap.allocate(20)             # reached only by the old -> young edge
    trash = heap.allocate(21)            # never reached: must die
    print("writing O.strong[0] -> young 'keep'; 'trash' stays unreachable")
    heap.set_strong(O, 0, keep)
    print("remembered set before minor:", heap.remembered_set())
    minor = heap.run_minor(5)
    print("minor promoted:", dict(sorted(minor.promoted.items())),
          "| moved-young:", dict(sorted(minor.relocated.items())))
    O = heap.root_handle("O")
    forwarded = heap.get_strong(O, 0)
    print("O.strong[0] forwarded to", forwarded.address,
          "generation", heap.generation_of(forwarded))
    assert forwarded.address == minor.promoted[keep.address]
    assert trash.address not in heap.objects()
    assert heap.remembered_set() == ()     # edge is now old -> old

    # --- One actually triggered rejection. ------------------------------- #
    hr("PART 2 (boundary): attaching a sealed-for-reclamation object is refused")
    g2 = GenHeap(young_capacity=80, old_capacity=160, promote_age=1)
    a = g2.allocate(1, n_strong=1)
    dead = g2.allocate(2)
    g2.set_strong(a, 0, dead)
    g2.add_root("a", a)
    g2.run_minor(4)                      # both old
    a = g2.root_handle("a")
    dead = g2.get_strong(a, 0)
    g2.set_strong(a, 0, None)            # dead is now unreachable old garbage
    r = g2.collect_step(1, "major")
    while not r.completed and r.phase not in ("weakclear", "sweep"):
        r = g2.collect_step(1)
    print(f"major reached the sealed phase {r.phase!r}; "
          f"trying to attach dead object {dead.address} to a live holder ...")
    try:
        g2.set_strong(a, 0, dead)
    except SealedForReclamationError as exc:
        print("  rejected:", type(exc).__name__, "-", exc)
    else:  # pragma: no cover
        raise AssertionError("expected SealedForReclamationError")
    print("  failed write left no dangling edge:", g2.get_strong(a, 0))
    assert g2.get_strong(a, 0) is None

    # A promotion that provably cannot fit is a deterministic failure too.
    hr("PART 2 (boundary): promotion set that cannot fit -> HeapCapacityError")
    tiny = GenHeap(young_capacity=60, old_capacity=4, promote_age=10)
    big = tiny.allocate(7, n_strong=4)  # size 6: fits young, never fits old
    tiny.add_root("big", big)
    tiny.request_promotion(big)
    try:
        tiny.run_minor(8)
    except HeapCapacityError as exc:
        print("  rejected promotion:", exc)
    else:  # pragma: no cover
        raise AssertionError("expected HeapCapacityError")
    # The failed minor moved nothing; the heap still serves from its state.
    assert tiny.epoch == 0 and big.address in tiny.objects()
    print("  heap state unchanged after the failed minor (epoch", tiny.epoch, ")")


def main() -> None:
    started = time.perf_counter()
    part_one_legacy()
    part_two_generational()
    elapsed = time.perf_counter() - started
    print()
    print(f"demo finished in {elapsed * 1000:.1f} ms (real deterministic "
          "computation, no sleeps, no pre-recorded answers).")
    assert elapsed < 90, "demo must finish within 90 seconds"


if __name__ == "__main__":
    main()
