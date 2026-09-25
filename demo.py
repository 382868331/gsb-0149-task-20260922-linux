#!/usr/bin/env python3
"""Fixed-input demonstration for the moving_gc kernel.

Run with::

    python demo.py

Everything here is deterministic, offline and computed on the spot (no
sleeps, no pre-recorded output).  It prints:

1. a *normal* collection: real address movement, strong forwarding and
   weak-edge clearing on a graph containing garbage, a shared strong
   child and a weak-only target;
2. the **weak-key association** contract: a two-level activation chain
   (one pass would not suffice), a key/value ring with no external root
   that cannot bootstrap itself into liveness, and forwarding of the
   surviving entries after relocation;
3. two *actually triggered* error boundaries: a stale (pre-collection)
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
    # Part 2: weak-key associations (conditional strong-ness + fixpoint)
    # ------------------------------------------------------------------ #
    hr("PART 2 (weak-key associations): liveness needs table AND key")
    wk = Heap(capacity=48)

    # Unrooted garbage occupying slot 0 so the first collection really
    # relocates every survivor (like Part 1).
    wk.allocate(999)

    # A single table object and two keys/values arranged as a two-level
    # activation chain:
    #   table[k1] = v1 ; v1 strongly references k2 ; table[k2] = v2
    # Roots are only: table and k1.  Nothing else roots k2 or v2.
    table = wk.allocate(10)
    k1 = wk.allocate(11)
    v1 = wk.allocate(21, n_strong=1)   # v1 -> k2
    k2 = wk.allocate(12)
    v2 = wk.allocate(22)
    wk.set_strong(v1, 0, k2)
    wk.association_put(table, k1, v1)
    wk.association_put(table, k2, v2)
    wk.add_root("table", table)
    wk.add_root("k1", k1)

    pre = {"table": table.address, "k1": k1.address, "v1": v1.address,
           "k2": k2.address, "v2": v2.address}
    print("roots: only table and k1; entries table[k1]=v1, table[k2]=v2;")
    print("v1 strongly references k2 (which has no root).")
    print("addresses before:", pre)

    cres = wk.collect()
    t2 = wk.root_handle("table")
    k1b = wk.root_handle("k1")
    v1b = wk.association_get(t2, k1b)
    print("\ncollection result:")
    print(f"  relocated old -> new: {dict(sorted(cres.relocated.items()))}")
    print(f"  surviving associations (new table,key,value addresses): "
          f"{list(cres.surviving_associations)}")
    print(f"  cleared associations: {list(cres.cleared_associations)}")
    print("  pass 1 activates table[k1]=v1 (k1 already alive); v1 then")
    print("  strongly reaches k2; pass 2 activates table[k2]=v2.")
    # Recover k2/v2 post-GC handles by walking from v1's strong edge.
    k2b = wk.get_strong(v1b, 0)
    v2b = wk.association_get(t2, k2b)
    print(f"  after GC: table@{t2.address}, k1@{k1b.address}, v1@{v1b.address}, "
          f"k2@{k2b.address}, v2@{v2b.address}")
    assert {t2.address, k1b.address, v1b.address, k2b.address, v2b.address} == \
        set(cres.relocated.values()), "all five must survive the two-level chain"
    assert len(cres.surviving_associations) == 2

    # A key/value ring with no external key root must NOT bootstrap:
    # table2[k3]=v3, v3 strongly -> k3, but only table2 is rooted.
    table2 = wk.allocate(30)
    k3 = wk.allocate(31)
    v3 = wk.allocate(32, n_strong=1)
    wk.set_strong(v3, 0, k3)
    wk.association_put(table2, k3, v3)
    wk.add_root("table2", table2)
    print("\nring probe: table2[k3]=v3 and v3 -> k3, but only table2 is rooted")
    print(f"  pre-GC addresses: table2={table2.address}, k3={k3.address}, "
          f"v3={v3.address}")
    ring = wk.collect()
    t2b = wk.root_handle("table2")
    print(f"  after GC: surviving objects = {list(wk.objects())}")
    print(f"  cleared associations (pre-GC table,key addresses): "
          f"{list(ring.cleared_associations)}")
    assert k3.address not in ring.relocated and v3.address not in ring.relocated, \
        "unrooted key/value ring must die"
    assert ring.cleared_associations == (
        (table2.address, k3.address),
    ), "dead-key entry reported in pre-collection addresses"
    print("  >> k3 and v3 were reclaimed: an entry cannot keep its own key "
          "alive through its not-yet-copied value.")

    # A second collection forwards the surviving entries from the chain.
    fwd2 = wk.collect()
    t3 = wk.root_handle("table")
    k1c = wk.root_handle("k1")
    v1c = wk.association_get(t3, k1c)
    print("\na further collection forwards the live entries again:")
    print(f"  table now @{t3.address}, k1 @{k1c.address}, "
          f"table[k1] -> @{v1c.address}; live slots = {fwd2.live_slots}")

    # ------------------------------------------------------------------ #
    # Part 3a: stale (old-epoch) external handle is rejected
    # ------------------------------------------------------------------ #
    hr("PART 3a (boundary): a stale handle from before the GC is rejected")
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
    # Part 3b: exactly-full heap -> emergency GC, then capacity error
    # ------------------------------------------------------------------ #
    hr("PART 3b (boundary): allocation failure after one emergency GC")
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
