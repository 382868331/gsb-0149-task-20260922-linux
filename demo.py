#!/usr/bin/env python3
"""Fixed-input demonstration for the moving_gc kernel.

Run with::

    python demo.py

Everything here is deterministic, offline and computed on the spot (no
sleeps, no pre-recorded output).  It prints:

1. a *normal* v1 collection: real address movement, strong forwarding and
   weak-edge clearing on a graph containing garbage, a shared strong
   child and a weak-only target;
2. two *actually triggered* v1 error boundaries: a stale (pre-collection)
   handle being rejected, and allocation failing after the mandatory
   emergency collection on an exactly-full heap;
3. a v2 end-to-end scenario exercising, in one interwoven run, the three
   new capability groups together: young copying + promotion with an
   exact old->young remembered-set edge, an ephemeron fixed point
   (including a key/value mutual reference), finalizer resurrection with
   allocation inside the finalizer, all driven in explicit bounded scan
   steps -- plus an actually triggered promotion-capacity failure.
"""

from __future__ import annotations

import time

from moving_gc import (
    GenerationalHeap,
    Heap,
    HeapCapacityError,
    OLD,
    StaleHandleError,
    YOUNG,
)


def hr(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def parts_v1() -> None:
    started = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Part 1: normal collection with movement + weak clearing
    # ------------------------------------------------------------------ #
    hr("PART 1 (normal): one semi-space collection on a fixed graph")
    heap = Heap(capacity=40)

    # Layout (sizes in slots):
    #   tmp    size 2, no roots        -> garbage
    #   holder size 5 (hdr,payload,1 strong,2 weak), payload 100
    #   child  size 2, payload 200     -> strongly reachable via holder
    #   cache  size 2, payload 300     -> only weakly referenced, must die
    tmp = heap.allocate(999)
    holder = heap.allocate(100, n_strong=1, n_weak=2)
    child = heap.allocate(200)
    cache = heap.allocate(300)

    heap.set_strong(holder, 0, child)   # keeps child alive
    heap.set_weak(holder, 0, cache)     # weak-only: cache is reclaimed
    heap.set_weak(holder, 1, child)     # weak to a survivor: forwarded
    heap.add_root("main", holder)

    before = {
        "tmp": tmp.address,
        "holder": holder.address,
        "child": child.address,
        "cache": cache.address,
    }
    print(f"capacity={heap.capacity} slots per semi-space; "
          f"used before collect = {heap.used_slots}")
    print("addresses before collection:", before)
    print("edges before: holder.strong[0] -> child @ "
          f"{heap.get_strong(holder, 0).address}, "
          f"holder.weak[0] -> cache @ {heap.get_weak(holder, 0).address}, "
          f"holder.weak[1] -> child @ {heap.get_weak(holder, 1).address}")

    result = heap.collect()

    print("\ncollection result:")
    print(f"  epoch:                 0 -> {heap.epoch}")
    print(f"  relocated old -> new:  {dict(sorted(result.relocated.items()))}")
    print(f"  live_slots:            {result.live_slots}")
    print(f"  roots preserved:       {result.roots_preserved}")
    print(f"  cleared weak edges:    {list(result.cleared_weak_edges)}")

    h = heap.root_handle("main")
    strong_child = heap.get_strong(h, 0)
    weak_cache = heap.get_weak(h, 0)
    weak_child = heap.get_weak(h, 1)

    print("\nafter collection:")
    print(f"  holder moved {before['holder']} -> {h.address}")
    print(f"  child  moved {before['child']} -> {strong_child.address}")
    print(f"  holder.payload = {heap.payload(h)} (preserved)")
    print(f"  strong[0] now points at child's new address {strong_child.address}; "
          f"same object as weak[1]: {weak_child == strong_child}")
    print(f"  weak[0] (was -> cache @ {before['cache']}) is now: {weak_cache} "
          f"[cleared because cache was weakly referenced only]")
    print(f"  live object addresses in the compacted heap: {list(heap.objects())}")

    moved = h.address != before["holder"] and strong_child.address != before["child"]
    assert moved and weak_cache is None, "demo invariant failed"
    print("  >> addresses really changed and the dead weak edge was cleared.")

    # ------------------------------------------------------------------ #
    # Part 2a: stale (old-epoch) external handle is rejected
    # ------------------------------------------------------------------ #
    hr("PART 2a (boundary): a stale handle from before the GC is rejected")
    stale = holder  # epoch-0 handle; epoch is now 1
    print(f"stale handle: address={stale.address}, epoch={stale.epoch} "
          f"(heap epoch={heap.epoch})")
    try:
        heap.payload(stale)
    except StaleHandleError as exc:
        print(f"  heap.payload(stale) raised StaleHandleError: {exc}")
    else:  # pragma: no cover - demo must actually trigger the error
        raise AssertionError("expected StaleHandleError")
    # Failed call changed nothing: the current-epoch root still works.
    print(f"  heap unchanged; root payload via fresh handle = "
          f"{heap.payload(heap.root_handle('main'))}")

    # ------------------------------------------------------------------ #
    # Part 2b: exactly-full heap -> emergency GC, then capacity error
    # ------------------------------------------------------------------ #
    hr("PART 2b (boundary): allocation failure after one emergency GC")
    small = Heap(capacity=8)
    a = small.allocate(1, n_strong=1)  # 3 slots
    b = small.allocate(2, n_strong=1)  # 3 slots
    c = small.allocate(3)              # 2 slots -> exactly 8/8
    # Every object is rooted, so an emergency collection reclaims nothing.
    small.add_root("a", a)
    small.add_root("b", b)
    small.add_root("c", c)
    print(f"allocated objects at {a.address}, {b.address}, {c.address}; "
          f"used={small.used_slots}/{small.capacity}, all three rooted")
    print("requesting one more 2-slot object; active space is full ...")
    epoch_before = small.epoch
    try:
        small.allocate(4)
    except HeapCapacityError as exc:
        print(f"  raised HeapCapacityError: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected HeapCapacityError")
    print(f"  exactly one emergency collection ran (epoch {epoch_before} -> "
          f"{small.epoch}); all roots survived, used_slots={small.used_slots}, "
          f"no half-object left: {len(small.objects())} objects remain")
    # Once the roots are released, a manual collection empties the heap and
    # the previously-impossible allocation succeeds: clean state, no debris.
    small.remove_root("a")
    small.remove_root("b")
    small.remove_root("c")
    small.collect()
    d = small.allocate(4)
    print(f"  after dropping roots and collecting: new object allocated at "
          f"address {d.address}, payload={small.payload(d)}")
    return time.perf_counter() - started


_GEN = {OLD: "old", YOUNG: "young"}


def v2_end_to_end() -> float:
    """One scenario using all three new capability groups at once."""
    started = time.perf_counter()
    hr("PART 3 (v2): generations + IU barrier/RS + ephemerons + finalizers")
    # Small generations make the boundaries real.
    heap = GenerationalHeap(young_capacity=40, old_capacity=26)

    # --- build a young graph -------------------------------------------
    # tab  : ephemeron table (entry k -> v) with a weak edge to v as well.
    # k    : rooted key; v points strongly back at k but is reachable ONLY
    #        through the ephemeron value -> the ephemeron must rescue v;
    #        because v is live after the fixed point, tab's weak edge to v
    #        is forwarded rather than cleared.
    # doomed: finalizer object.
    tab = heap.allocate(10, n_weak=1, n_ephemeron=1)
    k = heap.allocate(11)
    v = heap.allocate(12, n_strong=1)
    doomed = heap.allocate(13)
    heap.set_ephemeron(tab, 0, k, v)
    heap.set_strong(v, 0, k)       # value refers back to its key
    heap.set_weak(tab, 0, v)       # weak edge resolved only after fixpoint
    heap.set_finalizer(doomed, 7001)
    heap.add_root("rtab", tab)
    heap.add_root("rk", k)
    print(f"  young used {heap.used_slots(YOUNG)}/{heap.young_capacity}, "
          f"old used {heap.used_slots(OLD)}/{heap.old_capacity}")

    # --- drive major cycle explicitly in small finite steps ------------
    heap.begin_collection("major")
    steps = 0
    while heap.cycle_phase() == "marking":
        r = heap.mark_step(2)
        steps += 1
    info = heap.cycle_info()
    print(f"  initial marking settled in {steps} budget-2 step(s); "
          f"phase -> {heap.cycle_phase()}")
    pending = heap.pending_finalizers()
    print(f"  staged finalizer events: "
          f"{[(e.event_id, _GEN[e.gen], e.address) for e in pending]}")

    def finalizer_body(event_id: int, h) -> None:
        assert event_id == 7001
        # Resurrect doomed on a fresh root, then allocate inside the
        # finalizer (the newborn is born marked and survives the cycle).
        heap.add_root("resurrected", h)
        nb = heap.allocate(77, n_strong=1)
        heap.add_root("rnew", nb)

    while heap.cycle_phase() == "finalizing":
        heap.finalize_step(finalizer_body)
    print("  finalizer ran exactly once: doomed resurrected, newborn made")

    total = 0
    steps = 0
    while heap.cycle_phase() in ("remark", "clearing"):
        r = heap.finish_collection(2)
        total += r.edges_scanned
        steps += 1
        if r.complete:
            res = r.result
            break
    print(f"  remark+clearing took {steps} step(s) (budget 2 each); "
          f"total edge scans cycle = {res.edges_scanned}")
    print(f"  epoch 0 -> {heap.epoch}")
    print(f"  resolved ephemerons: {res.resolved_ephemerons}")
    print(f"  rescued values:      {res.rescued_values}")
    print(f"  cleared weak edges:  {res.cleared_weak_edges}")
    print(f"  cleared ephemerons:  {res.cleared_ephemerons}")
    print(f"  finalized:           {res.finalized}")
    print(f"  reclaimed objects:   {res.reclaimed_objects}")
    print(f"  old used {heap.used_slots(OLD)}/{heap.old_capacity}, "
          f"young used {heap.used_slots(YOUNG)}/{heap.young_capacity}")

    # --- ephemeron rescued the value; weak edge resolved after fixpoint -
    rtab = heap.root_handle("rtab")
    ek, ev = heap.get_ephemeron(rtab, 0)
    # v was weak-only from the strong graph: it survives ONLY because the
    # ephemeron activated once k was known reachable.
    assert res.rescued_values and res.rescued_values[0][2] == 0
    assert res.cleared_weak_edges == ()
    assert heap.get_strong(ev, 0) == heap.root_handle("rk")
    assert heap.get_weak(rtab, 0) == ev
    assert rtab.generation == OLD
    print(f"\n  value was rescued by the ephemeron (rescued={res.rescued_values})")
    print(f"  ephemeron intact post-promotion: key {_GEN[ek.generation]}@{ek.address}, "
          f"value {_GEN[ev.generation]}@{ev.address}")
    print("  tab.weak[0] -> v was forwarded (not cleared) because the "
          "ephemeron fixed point ran first; v.strong[0] -> k intact")

    # --- remembered set catches a newly written old -> young edge ------
    yt = heap.allocate(78)  # fresh young target
    rnew = heap.root_handle("rnew")
    assert rnew.generation == OLD
    heap.set_strong(rnew, 0, yt)
    rs_before = sorted(heap.remembered_set())
    print(f"\n  old -> young strong write recorded; remembered set: {rs_before}")
    assert rs_before
    minor_res = heap.collect("minor", 2)
    print(f"  minor collection: promoted={minor_res.promoted} "
          f"(young target kept alive by the remembered set), "
          f"reclaimed={minor_res.reclaimed_objects}, "
          f"edges_scanned={minor_res.edges_scanned}")
    print(f"  remembered set after everyone promoted: "
          f"{sorted(heap.remembered_set())}")
    fresh_new = heap.root_handle("rnew")
    tgt = heap.get_strong(fresh_new, 0)
    assert tgt is not None and tgt.generation == OLD
    print(f"  young target survived the minor and promoted to "
          f"{_GEN[tgt.generation]}@{tgt.address}")

    # --- tricky boundary: promotion overflow ---------------------------
    hr("PART 4 (v2 boundary): promotion overflow is a real rejection")
    before_epoch = heap.epoch
    free_old = heap.free_slots(OLD)
    # Fill young with rooted survivors until the next allocation must
    # collect: that minor would promote more slots than the old generation
    # has free (and a major sweep cannot help -- every object is rooted).
    n_fit = heap.young_capacity // 2  # 20 size-2 objects fill 40 young slots
    try:
        for i in range(n_fit + 1):
            hh = heap.allocate(200 + i)
            heap.add_root(f"rf{i}", hh)
        raise AssertionError("expected HeapCapacityError")
    except HeapCapacityError as exc:
        print(f"  raised HeapCapacityError: {exc}")
    # The failed 21st allocation left no half object, and the failed cycle
    # never committed (rf{n_fit} is unknown; rf0..rf{n_fit-1} all exist).
    assert all(heap.root_handle(f"rf{i}") is not None for i in range(n_fit))
    try:
        heap.root_handle(f"rf{n_fit}")
    except KeyError:
        pass
    else:
        raise AssertionError("half object leaked from failed allocation")
    print(f"  epoch unchanged on failure ({before_epoch} -> {heap.epoch}); "
          f"no half object (rf0..rf{n_fit - 1} present, rf{n_fit} absent); "
          f"old had {free_old} free slot(s)")
    print(f"  heap still serviceable: rtab payload = "
          f"{heap.payload(heap.root_handle('rtab'))}")
    return time.perf_counter() - started


def main() -> None:
    t1 = parts_v1()
    t2 = v2_end_to_end()
    print()
    print(f"demo finished in {(t1 + t2) * 1000:.1f} ms (real computation, "
          "no sleeps, fixed deterministic inputs).")


if __name__ == "__main__":
    main()
