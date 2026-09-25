#!/usr/bin/env python3
"""Fixed-input demonstration for the moving_gc kernel.

Run with::

    python demo.py

Everything here is deterministic, offline and computed on the spot (no
sleeps, no pre-recorded output).  It prints:

1. a *normal* collection: real address movement, strong forwarding and
   weak-edge clearing on a graph containing garbage, a shared strong
   child and a weak-only target;
1b. weak-key entries: a two-level activation chain that only a fixpoint
   keeps alive, plus a key/value ring with no external root whose
   dead-key entry is deleted;
2. two *actually triggered* error boundaries: a stale (pre-collection)
   handle being rejected, and allocation failing after the mandatory
   emergency collection on an exactly-full heap.
"""

from __future__ import annotations

import time

from moving_gc import Heap, HeapCapacityError, StaleHandleError


def hr(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
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
    # Part 1b: weak-key entries (ephemeron-like associations)
    # ------------------------------------------------------------------ #
    hr("PART 1b (normal): weak-key entries resolved to a fixpoint")
    wh = Heap(capacity=48)

    junk = wh.allocate(99)  # unrooted garbage: forces real movement
    # t1[k1] = v1; v1 strongly holds t2 and k2; t2[k2] = v2.
    # v2 survives only because activating t1's entry makes k2 live, which
    # then activates t2's entry: a two-level chain one scan pass would miss.
    t1 = wh.allocate(10, n_entries=2)
    k1 = wh.allocate(21)
    v1 = wh.allocate(31, n_strong=2)
    t2 = wh.allocate(12, n_entries=1)
    k2 = wh.allocate(22)
    v2 = wh.allocate(32)
    # A key/value ring with no external root: k3 <- t1 entry, v3 -> k3.
    # The ring must NOT bootstrap itself; both die and the entry is deleted.
    k3 = wh.allocate(23)
    v3 = wh.allocate(33, n_strong=1)

    wh.set_entry(t1, 0, k1, v1)
    wh.set_strong(v1, 0, t2)
    wh.set_strong(v1, 1, k2)
    wh.set_entry(t2, 0, k2, v2)
    wh.set_entry(t1, 1, k3, v3)
    wh.set_strong(v3, 0, k3)  # value -> key back-reference (a ring)
    wh.add_root("table", t1)
    wh.add_root("key1", k1)

    before1b = {"t1": t1.address, "v2": v2.address, "k3": k3.address, "v3": v3.address}
    print("graph: t1[k1]=v1, v1->{t2,k2}, t2[k2]=v2  (two-level chain)")
    print("       t1[k3]=v3, v3->k3                    (key/value ring, no root)")
    print(f"addresses before: {before1b}")

    res1b = wh.collect()

    ft1 = wh.root_handle("table")
    fk1 = wh.root_handle("key1")
    fv1 = wh.get_entry_value(ft1, 0)
    ft2 = wh.get_strong(fv1, 0)
    fk2 = wh.get_strong(fv1, 1)
    fv2 = wh.get_entry_value(ft2, 0)
    print("\ncollection result:")
    print(f"  relocated old -> new: {dict(sorted(res1b.relocated.items()))}")
    print(f"  deleted entries (table_addr, idx): {list(res1b.cleared_entries)}")
    print("\nafter collection:")
    print(f"  two-level chain survived: t1@{ft1.address}, k1@{fk1.address}, "
          f"v1@{fv1.address}, t2@{ft2.address}, k2@{fk2.address}, v2@{fv2.address}")
    print(f"  t2's entry forwarded: key -> {wh.get_entry_key(ft2, 0).address}, "
          f"value -> {fv2.address} (v2 moved {before1b['v2']} -> {fv2.address})")
    ring_entry = wh.get_entry_key(ft1, 1)
    print(f"  ring entry t1[1]: key={ring_entry} value={wh.get_entry_value(ft1, 1)} "
          f"[k3 @{before1b['k3']} and v3 @{before1b['v3']} reclaimed: "
          "a key referenced only by its own value cannot stay alive]")
    assert fv2 is not None and ring_entry is None, "demo invariant failed"
    print("  >> fixpoint kept the chain alive and deleted the dead-key ring entry.")

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

    elapsed = time.perf_counter() - started
    print()
    print(f"demo finished in {elapsed * 1000:.1f} ms (real computation, "
          "no sleeps, fixed deterministic inputs).")


if __name__ == "__main__":
    main()
