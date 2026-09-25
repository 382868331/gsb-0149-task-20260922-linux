"""Differential + contract tests for the moving_gc kernel.

Expected values come from :mod:`reference`, an independent graph model that
never imports the code under test.  Round-trip style assertions (reading
back what was written) are only used as supplementary checks.
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import GraphSpec  # noqa: E402  (independent oracle)

from moving_gc import (  # noqa: E402
    MAX_CAPACITY,
    Handle,
    Heap,
    HeapCapacityError,
    StaleHandleError,
)


# --------------------------------------------------------------------------- #
# Shared construction / verification harness
# --------------------------------------------------------------------------- #
def build_world(spec: GraphSpec) -> tuple[Heap, dict[str, Handle]]:
    """Construct a heap mirroring ``spec``; return it plus name->handle."""
    heap = Heap(spec.capacity)
    handles: dict[str, Handle] = {}
    for name, node in spec.nodes.items():
        handles[name] = heap.allocate(
            node.payload, n_strong=len(node.strong), n_weak=len(node.weak)
        )
    for name, node in spec.nodes.items():
        for j, dst in enumerate(node.strong):
            if dst is not None:
                heap.set_strong(handles[name], j, handles[dst])
        for j, dst in enumerate(node.weak):
            if dst is not None:
                heap.set_weak(handles[name], j, handles[dst])
    for rn in spec.root_order:
        t = spec.roots[rn]
        if t is not None:
            heap.add_root(rn, handles[t])
        else:
            # Named null root: register against any object, then clear it.
            anchor = next(iter(handles.values()))
            heap.add_root(rn, anchor)
            heap.clear_root(rn)
    # Weak-key associations live in side metadata, not slots.
    for table, entries in spec.associations.items():
        for key, value in entries.items():
            heap.association_put(handles[table], handles[key], handles[value])
    return heap, handles


def verify_against_oracle(
    spec: GraphSpec,
    heap: Heap,
    expected,
    result,
    _old_addresses: dict[str, int],
) -> dict[str, Handle]:
    """Compare one real CollectionResult with the independent oracle."""
    # --- relocation mapping ------------------------------------------------
    exp_old = {old: new for old, new in expected.relocated.values()}
    assert exp_old == result.relocated, (exp_old, result.relocated)

    name_to_new: dict[str, int] = {}
    for name, (old, new) in expected.relocated.items():
        assert result.relocated[old] == new
        name_to_new[name] = new

    # Copy order must equal destination-address ascending order (Cheney).
    assert list(result.relocated.values()) == [
        expected.relocated[n][1] for n in expected.copy_order
    ]

    # --- counts -------------------------------------------------------------
    assert result.live_slots == expected.live_slots
    assert heap.used_slots == expected.live_slots
    assert set(heap.objects()) == set(name_to_new.values())

    # --- roots --------------------------------------------------------------
    for rn in spec.root_order:
        rh = heap.root_handle(rn)
        if expected.roots[rn] is None:
            assert rh is None
        else:
            assert rh is not None and rh.address == name_to_new[expected.roots[rn]]

    # --- cleared weak edges -------------------------------------------------
    exp_cleared = tuple(
        (name_to_new[holder], j) for holder, j in expected.cleared
    )
    assert result.cleared_weak_edges == exp_cleared

    # --- weak-key associations ---------------------------------------------
    exp_assoc_surv = tuple(
        (name_to_new[t], name_to_new[k], name_to_new[v])
        for t, k, v in expected.associations_surviving
    )
    assert result.surviving_associations == exp_assoc_surv, (
        result.surviving_associations,
        exp_assoc_surv,
    )
    # Cleared entries are reported in pre-collection addresses; the oracle
    # names the (live) table and the dead key.
    exp_assoc_clear = tuple(
        (_old_addresses[t], _old_addresses[k])
        for t, k in expected.associations_cleared
    )
    assert result.cleared_associations == exp_assoc_clear, (
        result.cleared_associations,
        exp_assoc_clear,
    )

    # --- payloads and every edge -------------------------------------------
    fresh: dict[str, Handle] = {
        name: heap.handle(new) for name, new in name_to_new.items()
    }
    for name in expected.copy_order:
        node = spec.nodes[name]
        h = fresh[name]
        assert heap.payload(h) == node.payload
        for j, t in enumerate(node.strong):
            got = heap.get_strong(h, j)
            if t is None:
                assert got is None
            else:
                assert got is not None
                assert got.address == name_to_new[t]
        for j, t in enumerate(node.weak):
            got = heap.get_weak(h, j)
            if t is None or t not in name_to_new:
                assert got is None, (name, j, t)
            else:
                assert got is not None and got.address == name_to_new[t], (name, j, t)

    # --- association metadata read-back ------------------------------------
    for tname, kname, vname in expected.associations_surviving:
        th = fresh[tname]
        got = heap.association_get(th, fresh[kname])
        assert got is not None and got.address == name_to_new[vname], (
            tname, kname, vname
        )
        listed = {
            (k.address, v.address) for k, v in heap.association_entries(th)
        }
        assert (name_to_new[kname], name_to_new[vname]) in listed
    return fresh


def old_address_map(spec: GraphSpec) -> dict[str, int]:
    return {name: node.address for name, node in spec.nodes.items()}


# --------------------------------------------------------------------------- #
# Basic semantics
# --------------------------------------------------------------------------- #
class TestBasic(unittest.TestCase):
    def test_allocate_addresses_and_sizes(self):
        heap = Heap(20)
        h0 = heap.allocate(10, n_strong=1, n_weak=1)  # size 4
        h1 = heap.allocate(20)  # size 2
        self.assertEqual((h0.address, h1.address), (0, 4))
        self.assertEqual(heap.used_slots, 6)
        self.assertEqual(heap.free_slots, 14)
        self.assertEqual(heap.payload(h0), 10)
        self.assertEqual(heap.object_shape(h0), (1, 1))
        self.assertIsNone(heap.get_strong(h0, 0))
        self.assertIsNone(heap.get_weak(h0, 0))

    def test_simple_collection_relocates(self):
        spec = GraphSpec(20)
        spec.add_node("g", 99)  # garbage, address 0
        spec.add_node("a", 1, n_strong=1)  # address 2
        spec.link_strong("a", 0, "a")  # self loop
        spec.add_root("r", "a")
        heap, handles = build_world(spec)
        old = old_address_map(spec)
        self.assertEqual(old["a"], 2)  # not at 0 initially

        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old)
        # Address actually moved to the start of the fresh semi-space.
        self.assertEqual(fresh["a"].address, 0)
        self.assertEqual(result.relocated, {2: 0})
        # Self loop forwarded to the *new* address.
        self.assertEqual(heap.get_strong(fresh["a"], 0), fresh["a"])
        self.assertEqual(heap.epoch, 1)

    def test_collect_without_roots_reclaims_everything(self):
        heap = Heap(10)
        heap.allocate(1)
        heap.allocate(2, n_strong=1)
        result = heap.collect()
        self.assertEqual(result.relocated, {})
        self.assertEqual(result.live_slots, 0)
        self.assertEqual(heap.objects(), ())
        self.assertEqual(heap.epoch, 1)

    def test_payload_update_persists(self):
        heap = Heap(8)
        h = heap.allocate(0)
        heap.set_payload(h, 12345)
        heap.add_root("r", h)
        heap.collect()
        h2 = heap.root_handle("r")
        self.assertEqual(heap.payload(h2), 12345)


# --------------------------------------------------------------------------- #
# Required graph scenarios vs the independent oracle
# --------------------------------------------------------------------------- #
class TestGraphScenarios(unittest.TestCase):
    def _run_one(self, spec: GraphSpec):
        heap, _ = build_world(spec)
        old = old_address_map(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        return verify_against_oracle(spec, heap, expected, result, old)

    def test_self_loop(self):
        spec = GraphSpec(16)
        spec.add_node("a", n_strong=1)
        spec.link_strong("a", 0, "a")
        spec.add_root("r", "a")
        self._run_one(spec)

    def test_cycle_without_root_dies(self):
        spec = GraphSpec(16)
        spec.add_node("a", n_strong=1)
        spec.add_node("b", n_strong=1)
        spec.link_strong("a", 0, "b")
        spec.link_strong("b", 0, "a")
        # no roots: the strongly connected cycle must still be collected
        fresh = self._run_one(spec)
        self.assertEqual(fresh, {})

    def test_weak_only_reference_is_cleared(self):
        spec = GraphSpec(16)
        spec.add_node("a", n_weak=1)
        spec.add_node("b")
        spec.link_weak("a", 0, "b")  # b only weakly referenced
        spec.add_root("r", "a")
        heap, _ = build_world(spec)
        old = old_address_map(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old)
        self.assertIsNone(heap.get_weak(fresh["a"], 0))
        self.assertEqual(result.cleared_weak_edges, ((fresh["a"].address, 0),))
        self.assertEqual(set(result.relocated), {old["a"]})
        self.assertNotIn(old["b"], result.relocated)

    def test_two_weak_edges_to_one_dead_target_both_cleared(self):
        spec = GraphSpec(32)
        spec.add_node("a", n_weak=1)
        spec.add_node("c", n_weak=1)
        spec.add_node("b")
        spec.link_weak("a", 0, "b")
        spec.link_weak("c", 0, "b")
        spec.add_root("r1", "a")
        spec.add_root("r2", "c")
        self._run_one(spec)

    def test_shared_subgraph_copied_once(self):
        spec = GraphSpec(32)
        spec.add_node("a", n_strong=1)
        spec.add_node("c", n_strong=1)
        spec.add_node("b")  # shared target
        spec.link_strong("a", 0, "b")
        spec.link_strong("c", 0, "b")
        spec.add_root("r1", "a")
        spec.add_root("r2", "c")
        heap, _ = build_world(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old_address_map(spec))
        # Exactly one copy of b; both strong edges name the same new address.
        self.assertEqual(heap.get_strong(fresh["a"], 0), fresh["b"])
        self.assertEqual(heap.get_strong(fresh["c"], 0), fresh["b"])
        self.assertEqual(len(result.relocated), 3)

    def test_weak_edge_to_survivor_is_forwarded(self):
        spec = GraphSpec(32)
        spec.add_node("a", n_weak=1)
        spec.add_node("b")
        spec.link_weak("a", 0, "b")
        spec.add_root("r_a", "a")
        spec.add_root("r_b", "b")  # b survives through its own strong root
        heap, _ = build_world(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old_address_map(spec))
        self.assertEqual(heap.get_weak(fresh["a"], 0), fresh["b"])
        self.assertEqual(result.cleared_weak_edges, ())

    def test_root_and_field_copy_order(self):
        # Allocation order is b,a,c; roots are registered as a then c.
        # Standard Cheney order: every root is evacuated first, in root
        # order (a, c), then the scan queue drains strong fields in field
        # order (a's fields introduce b; c is already copied).
        spec = GraphSpec(40)
        spec.add_node("b", n_strong=0)
        spec.add_node("a", n_strong=2)
        spec.add_node("c", n_strong=0)
        spec.link_strong("a", 0, "b")
        spec.link_strong("a", 1, "c")
        spec.add_root("r_a", "a")
        spec.add_root("r_c", "c")
        heap, _ = build_world(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        # a(size4) at 0, c(size2) at 4, b(size2) at 6.
        self.assertEqual(expected.copy_order, ("a", "c", "b"))
        self.assertEqual(list(result.relocated.values()), [0, 4, 6])
        verify_against_oracle(spec, heap, expected, result, old_address_map(spec))

    def test_mixed_weak_strong_diamond(self):
        spec = GraphSpec(48)
        spec.add_node("g0")  # garbage to force real movement
        spec.add_node("a", n_strong=2, n_weak=2)
        spec.add_node("b", n_strong=1, n_weak=1)
        spec.add_node("s")  # shared
        spec.add_node("w")  # weak-only target
        spec.link_strong("a", 0, "b")
        spec.link_strong("a", 1, "s")
        spec.link_strong("b", 0, "s")
        spec.link_weak("a", 0, "w")  # dead -> cleared
        spec.link_weak("a", 1, "s")  # survivor -> forwarded
        spec.link_weak("b", 0, "w")  # dead -> cleared
        spec.add_root("r", "a")
        heap, _ = build_world(spec)
        old = old_address_map(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old)
        # Every surviving address changed (garbage used the low slots before).
        for name in ("a", "b", "s"):
            self.assertNotEqual(fresh[name].address, old[name])
        self.assertIsNone(heap.get_weak(fresh["a"], 0))
        self.assertEqual(heap.get_weak(fresh["a"], 1), fresh["s"])
        self.assertIsNone(heap.get_weak(fresh["b"], 0))


# --------------------------------------------------------------------------- #
# Weak-key associations: conditional liveness + fixpoint activation
# --------------------------------------------------------------------------- #
class TestWeakKeyAssociations(unittest.TestCase):
    def _run_one(self, spec: GraphSpec):
        heap, _ = build_world(spec)
        old = old_address_map(spec)
        result = heap.collect()
        expected = spec.expected_collect()
        fresh = verify_against_oracle(spec, heap, expected, result, old)
        return heap, spec, result, expected, fresh, old

    def test_value_strong_only_when_table_and_key_both_alive(self):
        # Table alive, key alive via its own root: the entry makes value
        # strongly reachable even though nothing else references it.
        spec = GraphSpec(32)
        spec.add_node("garbage")
        spec.add_node("table")
        spec.add_node("key")
        spec.add_node("value")
        spec.assoc_put("table", "key", "value")
        spec.add_root("rt", "table")
        spec.add_root("rk", "key")
        _, _, result, expected, fresh, _old = self._run_one(spec)
        self.assertEqual(set(fresh), {"table", "key", "value"})
        self.assertEqual(
            result.surviving_associations,
            ((fresh["table"].address, fresh["key"].address,
              fresh["value"].address),),
        )
        self.assertEqual(result.cleared_associations, ())

    def test_dead_key_entry_is_deleted_and_value_reclaimed(self):
        # Table alive, key unreferenced: key dies, entry deleted, and the
        # value it protected dies with it.
        spec = GraphSpec(32)
        spec.add_node("table")
        spec.add_node("key")
        spec.add_node("value")
        spec.assoc_put("table", "key", "value")
        spec.add_root("rt", "table")  # key/value have no other reference
        heap, _, result, expected, fresh, old = self._run_one(spec)
        self.assertEqual(set(fresh), {"table"})
        self.assertEqual(result.surviving_associations, ())
        # Reported in pre-collection addresses (table old, key old).
        self.assertEqual(result.cleared_associations, ((old["table"], old["key"]),))
        # The live table carries no entry for a key handle at any epoch now;
        # its post-GC snapshot is simply empty.
        self.assertEqual(heap.association_entries(fresh["table"]), ())

    def test_dead_table_drops_entries_but_roots_keep_objects_alive(self):
        # Table itself unreachable: its entries participate in nothing,
        # so the value is NOT kept alive; independently rooted key still
        # survives on its own merits.
        spec = GraphSpec(32)
        spec.add_node("table")
        spec.add_node("key")
        spec.add_node("value")
        spec.assoc_put("table", "key", "value")
        spec.add_root("rk", "key")  # table and value unrooted
        _, _, result, _, fresh, _old = self._run_one(spec)
        self.assertEqual(set(fresh), {"key"})
        self.assertEqual(result.surviving_associations, ())
        # Entries of a dead table are not reported as cleared either.
        self.assertEqual(result.cleared_associations, ())

    def test_two_level_activation_chain(self):
        # k1 rooted; table rooted. Entry t[k1]=v1 activates on pass 1; v1
        # strongly references k2, which is otherwise unreachable; entry
        # t[k2]=v2 can only activate on pass 2. A one-pass implementation
        # would wrongly reclaim v2 (and the object k2 points at).
        spec = GraphSpec(48)
        spec.add_node("table")
        spec.add_node("k1")
        spec.add_node("v1", n_strong=1)  # v1 -> k2
        spec.add_node("k2")
        spec.add_node("v2")
        spec.link_strong("v1", 0, "k2")
        spec.assoc_put("table", "k1", "v1")
        spec.assoc_put("table", "k2", "v2")
        spec.add_root("rt", "table")
        spec.add_root("rk1", "k1")
        _, _, result, _, fresh, _old = self._run_one(spec)
        self.assertEqual(set(fresh), {"table", "k1", "v1", "k2", "v2"})
        self.assertEqual(len(result.surviving_associations), 2)

    def test_activation_chain_value_rescues_later_table(self):
        # The activated value strongly reaches a *second* table that was not
        # level-1 reachable; that table participates from the next pass on.
        # Its own entry still needs its key alive, which it is not, so the
        # entry is deleted even though its table was rescued.
        spec = GraphSpec(48)
        spec.add_node("t1")
        spec.add_node("k1")
        spec.add_node("v1", n_strong=1)  # v1 -> t2
        spec.add_node("t2")
        spec.add_node("k2")
        spec.add_node("v2")
        spec.link_strong("v1", 0, "t2")
        spec.assoc_put("t1", "k1", "v1")
        spec.assoc_put("t2", "k2", "v2")
        spec.add_root("rt1", "t1")
        spec.add_root("rk1", "k1")
        _, _, result, _, fresh, _old = self._run_one(spec)
        self.assertEqual(set(fresh), {"t1", "k1", "v1", "t2"})
        self.assertNotIn("k2", fresh)
        self.assertNotIn("v2", fresh)
        self.assertEqual(len(result.surviving_associations), 1)
        self.assertEqual(
            result.surviving_associations[0][:2],
            (fresh["t1"].address, fresh["k1"].address),
        )
        # t2 is live but k2 is dead: exactly one dead-key entry reported,
        # in pre-collection addresses (full address check is covered by the
        # oracle comparison inside _run_one).
        self.assertEqual(len(result.cleared_associations), 1)
        cleared_table_old, cleared_key_old = result.cleared_associations[0]
        self.assertNotIn(cleared_key_old, result.relocated)

    def test_key_value_ring_without_external_root_dies(self):
        # t rooted; t[k]=v; v strongly references k. Nothing roots k.
        # The activation test (is k already evacuated?) happens before v is
        # copied, so the ring cannot bootstrap: k and v both die, entry
        # deleted.
        spec = GraphSpec(32)
        spec.add_node("table")
        spec.add_node("key")
        spec.add_node("value", n_strong=1)  # value -> key
        spec.link_strong("value", 0, "key")
        spec.assoc_put("table", "key", "value")
        spec.add_root("rt", "table")
        _, _, result, _, fresh, old = self._run_one(spec)
        self.assertEqual(set(fresh), {"table"})
        self.assertEqual(result.surviving_associations, ())
        self.assertEqual(result.cleared_associations,
                         ((old["table"], old["key"]),))

    def test_mutual_entries_one_rooted_other_dies(self):
        # Two entries: t[a]=b_obj and t[b]=a_obj, where value of first
        # strongly reaches the second key. Root a: first entry activates,
        # its value reaches b, second entry then activates. Root neither
        # key: nothing activates at all.
        spec = GraphSpec(48)
        spec.add_node("table")
        spec.add_node("ka")
        spec.add_node("va", n_strong=1)  # va -> kb
        spec.add_node("kb")
        spec.add_node("vb")
        spec.link_strong("va", 0, "kb")
        spec.assoc_put("table", "ka", "va")
        spec.assoc_put("table", "kb", "vb")
        spec.add_root("rt", "table")
        # No key roots at all -> entire value/key apparatus dies.
        _, _, result_none, _, fresh_none, _old = self._run_one(spec)
        self.assertEqual(set(fresh_none), {"table"})
        self.assertEqual(result_none.surviving_associations, ())

        # With ka rooted, the two-level cascade survives everything.
        spec2 = GraphSpec(48)
        for n, kw in (
            ("table", {}),
            ("ka", {}),
            ("va", {"n_strong": 1}),
            ("kb", {}),
            ("vb", {}),
        ):
            spec2.add_node(n, **kw)
        spec2.link_strong("va", 0, "kb")
        spec2.assoc_put("table", "ka", "va")
        spec2.assoc_put("table", "kb", "vb")
        spec2.add_root("rt", "table")
        spec2.add_root("rka", "ka")
        _, _, result_all, _, fresh_all, _old = self._run_one(spec2)
        self.assertEqual(set(fresh_all), {"table", "ka", "va", "kb", "vb"})
        self.assertEqual(len(result_all.surviving_associations), 2)

    def test_value_reached_by_two_entries_copied_once(self):
        spec = GraphSpec(40)
        spec.add_node("table")
        spec.add_node("k1")
        spec.add_node("k2")
        spec.add_node("shared_value")
        spec.assoc_put("table", "k1", "shared_value")
        spec.assoc_put("table", "k2", "shared_value")
        spec.add_root("rt", "table")
        spec.add_root("r1", "k1")
        spec.add_root("r2", "k2")
        heap, _, result, _, fresh, _old = self._run_one(spec)
        self.assertEqual(len(result.relocated), 4)  # one copy of the value
        entries = heap.association_entries(fresh["table"])
        value_addrs = {v.address for _, v in entries}
        self.assertEqual(value_addrs, {fresh["shared_value"].address})

    def test_association_handles_forward_across_second_collection(self):
        spec = GraphSpec(40)
        spec.add_node("table")
        spec.add_node("key")
        spec.add_node("value")
        spec.assoc_put("table", "key", "value")
        spec.add_root("rt", "table")
        spec.add_root("rk", "key")
        heap, _ = build_world(spec)
        heap.collect()
        # Second collection: forwarded metadata must forward again.
        result2 = heap.collect()
        expected2 = spec.expected_collect()
        old2 = {name: node.address for name, node in spec.nodes.items()}
        fresh2 = verify_against_oracle(spec, heap, expected2, result2, old2)
        got = heap.association_get(fresh2["table"], fresh2["key"])
        self.assertIsNotNone(got)
        self.assertEqual(got.address, fresh2["value"].address)

    def test_plain_weak_edge_to_activated_value_is_forwarded(self):
        # A value kept alive *only* by an activated association entry must,
        # as a side effect, rescue plain weak edges that point at it.
        spec = GraphSpec(40)
        spec.add_node("table")
        spec.add_node("key")
        spec.add_node("value")
        spec.add_node("holder", n_weak=1)
        spec.link_weak("holder", 0, "value")
        spec.assoc_put("table", "key", "value")
        spec.add_root("rt", "table")
        spec.add_root("rk", "key")
        spec.add_root("rh", "holder")
        heap, _, result, _, fresh, _old = self._run_one(spec)
        self.assertIn("value", fresh)
        self.assertEqual(result.cleared_weak_edges, ())
        self.assertEqual(
            heap.get_weak(fresh["holder"], 0), fresh["value"]
        )

    def test_table_as_its_own_key(self):
        # Entry t[t]=v with t rooted: table is the key and is alive at level
        # 1, so the entry activates and keeps v alive.
        spec = GraphSpec(24)
        spec.add_node("table")
        spec.add_node("value")
        spec.assoc_put("table", "table", "value")
        spec.add_root("rt", "table")
        _, _, result, _, fresh, _old = self._run_one(spec)
        self.assertEqual(set(fresh), {"table", "value"})
        self.assertEqual(len(result.surviving_associations), 1)

    def test_key_and_value_same_object(self):
        # t[k]=k; k rooted. The entry survives and points the key at itself.
        spec = GraphSpec(24)
        spec.add_node("table")
        spec.add_node("k")
        spec.assoc_put("table", "k", "k")
        spec.add_root("rt", "table")
        spec.add_root("rk", "k")
        heap, _, result, _, fresh, _old = self._run_one(spec)
        entries = heap.association_entries(fresh["table"])
        self.assertEqual(len(entries), 1)
        kh, vh = entries[0]
        self.assertEqual(kh, vh)
        self.assertEqual(kh, fresh["k"])

    def test_entry_dropped_between_two_collections(self):
        # GC 1: entry alive.  Then the key root is dropped; GC 2 must delete
        # the entry and reclaim the now-unprotected value.
        spec = GraphSpec(32)
        spec.add_node("table")
        spec.add_node("key")
        spec.add_node("value")
        spec.assoc_put("table", "key", "value")
        spec.add_root("rt", "table")
        spec.add_root("rk", "key")
        heap, _ = build_world(spec)
        heap.collect()
        spec.expected_collect()  # advance the oracle to the same point
        # Drop the key root in both worlds.
        heap.remove_root("rk")
        spec.drop_root("rk")
        old2 = old_address_map(spec)
        result2 = heap.collect()
        expected2 = spec.expected_collect()
        fresh2 = verify_against_oracle(spec, heap, expected2, result2, old2)
        self.assertEqual(set(fresh2), {"table"})
        self.assertEqual(result2.surviving_associations, ())
        self.assertEqual(
            result2.cleared_associations, ((old2["table"], old2["key"]),)
        )

    def test_emergency_collection_on_allocate_activates_entry(self):
        # No explicit collect(): allocation pressure must itself run the
        # fixpoint and keep an association value alive before carving the
        # new object.
        heap = Heap(16)
        table = heap.allocate(1)   # 2
        key = heap.allocate(2)     # 2
        value = heap.allocate(3)   # 2 (only protected by the entry)
        garbage = heap.allocate(4, n_strong=4)  # 6 slots at 6
        heap.association_put(table, key, value)
        heap.add_root("rt", table)
        heap.add_root("rk", key)
        self.assertEqual(heap.used_slots, 12)
        # A size-5 object does not fit in 4 free slots -> emergency GC.
        new_obj = heap.allocate(5, n_strong=3)  # size 5
        self.assertEqual(heap.epoch, 1)
        # Survivors (table,key,value = 6 slots) plus the new object (5) = 11.
        self.assertEqual(heap.used_slots, 11)
        th = heap.root_handle("rt")
        kh = heap.root_handle("rk")
        vh = heap.association_get(th, kh)
        self.assertIsNotNone(vh)
        self.assertEqual(heap.payload(vh), 3)
        self.assertEqual(heap.payload(new_obj), 5)

    def test_association_put_get_remove_and_validation(self):
        heap = Heap(32)
        t = heap.allocate(1)
        k = heap.allocate(2)
        v = heap.allocate(3)
        heap.association_put(t, k, v)
        self.assertEqual(heap.association_get(t, k), v)
        self.assertEqual(
            [(kh.address, vh.address) for kh, vh in heap.association_entries(t)],
            [(k.address, v.address)],
        )
        # Replace, then remove.
        v2 = heap.allocate(4)
        heap.association_put(t, k, v2)
        self.assertEqual(heap.association_get(t, k), v2)
        heap.association_remove(t, k)
        self.assertIsNone(heap.association_get(t, k))
        self.assertEqual(heap.association_entries(t), ())
        # Removing an absent key is a no-op.
        heap.association_remove(t, k)
        # put(None) also deletes.
        heap.association_put(t, k, v)
        heap.association_put(t, k, None)
        self.assertIsNone(heap.association_get(t, k))
        # Non-handle arguments rejected, stale handles rejected post-GC.
        with self.assertRaises(TypeError):
            heap.association_put(t, k, 7)
        heap.add_root("rt", t)
        heap.collect()
        with self.assertRaises(StaleHandleError):
            heap.association_put(t, k, v)
        with self.assertRaises(StaleHandleError):
            heap.association_get(t, k)

    def test_capacity_shortfall_after_activation_keeps_no_half_object(self):
        # The activation fixpoint grows the post-collection live set; once
        # that compacted set leaves too little room, the *next* allocation
        # must fail after the mandatory single emergency collection, with
        # no half object and the activated association still intact.
        cap = 30
        heap = Heap(cap)
        table = heap.allocate(1)              # 2 slots
        key = heap.allocate(2, n_strong=0)    # 2 slots
        value = heap.allocate(909)            # 2 slots, kept alive only by
        #                                       the weak-key entry
        garbage = heap.allocate(7)            # 2 slots, unreferenced
        heap.association_put(table, key, value)
        heap.add_root("rt", table)
        heap.add_root("rk", key)
        # Active space: 8 slots used, 22 free.  Ask for a size-26 object: it
        # does not fit now (8+26>30), so the heap runs exactly one emergency
        # collection.  Post-GC live set = table+key+value = 6 slots (garbage
        # reclaimed, value activated); 6+26=32>30 so the request still fails
        # -- but the emergency GC has committed (epoch advanced) and left no
        # debris.
        epoch_before = heap.epoch
        with self.assertRaises(HeapCapacityError):
            heap.allocate(0, n_strong=24)  # size 26: fails even post-GC
        self.assertEqual(heap.epoch, epoch_before + 1)
        self.assertEqual(heap.used_slots, 6)
        self.assertEqual(len(heap.objects()), 3)  # table, key, value only
        th = heap.root_handle("rt")
        kh = heap.root_handle("rk")
        vh = heap.association_get(th, kh)
        self.assertIsNotNone(vh)
        self.assertEqual(heap.payload(vh), 909)
        # A minimum allocation now succeeds in the 24 remaining free slots.
        tiny = heap.allocate(5)  # size 2: 6+2 <= 30
        self.assertEqual(heap.payload(tiny), 5)


# --------------------------------------------------------------------------- #
# Capacity boundary: exactly-full heap and failed allocation
# --------------------------------------------------------------------------- #
class TestCapacity(unittest.TestCase):
    def test_exactly_full_heap_survives_collection(self):
        cap = 4096
        heap = Heap(cap)
        handles = [heap.allocate(i) for i in range(cap // 2)]  # 2048 * 2 slots
        self.assertEqual(heap.used_slots, cap)
        self.assertEqual(heap.free_slots, 0)
        for i, h in enumerate(handles):
            heap.add_root(f"r{i}", h)
        result = heap.collect()
        self.assertEqual(result.live_slots, cap)
        self.assertEqual(len(result.relocated), cap // 2)
        # Still exactly full; a minimum-size allocation must fail.
        with self.assertRaises(HeapCapacityError):
            heap.allocate(0)
        # Emergency collection ran, but no half object exists.
        self.assertEqual(heap.used_slots, cap)
        self.assertEqual(len(heap.objects()), cap // 2)

    def test_small_exactly_full_then_failure_leaves_no_half_object(self):
        heap = Heap(6)
        a = heap.allocate(1, n_strong=1)  # size 3
        b = heap.allocate(2, n_strong=1)  # size 3 -> exactly full
        heap.add_root("ra", a)
        heap.add_root("rb", b)
        snapshot = (heap.epoch, heap.used_slots, sorted(heap.objects()))
        with self.assertRaises(HeapCapacityError):
            heap.allocate(9)  # size 2: collect first (nothing dies), then fail
        # No half object: object set and occupancy unchanged post-GC.
        self.assertEqual(heap.used_slots, snapshot[1])
        self.assertEqual(sorted(heap.objects()), snapshot[2])
        # One collection did happen.
        self.assertEqual(heap.epoch, snapshot[0] + 1)

    def test_object_bigger_than_capacity_collects_then_fails(self):
        # Contract: allocation failure ALWAYS performs one collection first,
        # even when the object can never fit an empty semi-space.
        heap = Heap(4)
        heap.allocate(9)  # unrooted garbage, size 2 at slot 0
        epoch = heap.epoch
        with self.assertRaises(HeapCapacityError):
            heap.allocate(0, n_strong=4)  # size 6 > 4
        # The mandatory collection ran (epoch advanced) and reclaimed the
        # garbage, but still no half-object exists.
        self.assertEqual(heap.epoch, epoch + 1)
        self.assertEqual(heap.used_slots, 0)
        self.assertEqual(heap.objects(), ())

    def test_allocation_failure_triggers_reclaim_then_succeeds(self):
        heap = Heap(10)
        garbage = heap.allocate(77)  # size 2 at 0
        a = heap.allocate(1, n_strong=1)  # size 3 at 2
        b = heap.allocate(2)  # size 2 at 5
        heap.set_strong(a, 0, b)
        heap.add_root("r", a)
        self.assertEqual(heap.used_slots, 7)
        # 3 slots free; a size-4 object does not fit -> GC reclaims garbage
        # (2 slots) and the new object is carved from compacted to-space.
        c = heap.allocate(3, n_strong=2)  # size 4
        self.assertEqual(heap.epoch, 1)
        self.assertLessEqual(heap.used_slots, heap.capacity)
        ra = heap.root_handle("r")
        self.assertEqual(heap.payload(ra), 1)
        self.assertEqual(heap.payload(c), 3)
        self.assertEqual(heap.payload(heap.get_strong(ra, 0)), 2)

    def test_capacity_constructor_bounds(self):
        Heap(1)
        Heap(MAX_CAPACITY)
        with self.assertRaises(ValueError):
            Heap(0)
        with self.assertRaises(ValueError):
            Heap(-1)
        with self.assertRaises(ValueError):
            Heap(MAX_CAPACITY + 1)


# --------------------------------------------------------------------------- #
# Stale handles / epochs
# --------------------------------------------------------------------------- #
class TestStaleHandles(unittest.TestCase):
    def test_old_handle_rejected_everywhere(self):
        heap = Heap(20)
        heap.allocate(0)  # unrooted garbage: guarantees survivors really move
        a = heap.allocate(1, n_strong=1, n_weak=1)  # old addr 2, size 4
        b = heap.allocate(2)  # old addr 6
        heap.set_strong(a, 0, b)
        heap.set_weak(a, 0, b)
        heap.add_root("r", a)
        heap.collect()  # a,b move; epoch -> 1
        stale_a = a
        stale_b = b
        with self.assertRaises(StaleHandleError):
            heap.payload(stale_a)
        with self.assertRaises(StaleHandleError):
            heap.set_payload(stale_a, 5)
        with self.assertRaises(StaleHandleError):
            heap.get_strong(stale_a, 0)
        with self.assertRaises(StaleHandleError):
            heap.get_weak(stale_a, 0)
        fresh_a = heap.root_handle("r")
        with self.assertRaises(StaleHandleError):
            heap.set_strong(fresh_a, 0, stale_b)
        with self.assertRaises(StaleHandleError):
            heap.set_weak(fresh_a, 0, stale_b)
        with self.assertRaises(StaleHandleError):
            heap.add_root("r2", stale_a)
        # Failed writes did not mutate the edges: b was strongly reachable
        # and survived, so the strong edge must name b's *new* address (and
        # the weak edge too), never the stale one.
        fresh_b = heap.get_strong(fresh_a, 0)
        self.assertIsNotNone(fresh_b)
        self.assertNotEqual(fresh_b.address, stale_b.address)
        self.assertEqual(heap.get_weak(fresh_a, 0).address, fresh_b.address)

    def test_stale_handle_even_when_address_coincides(self):
        # Self-loop object lands at address 0 both times; epoch still invalidates.
        heap = Heap(8)
        a = heap.allocate(1, n_strong=1)
        heap.set_strong(a, 0, a)
        heap.add_root("r", a)
        self.assertEqual(a.address, 0)
        heap.collect()
        self.assertEqual(heap.root_handle("r").address, 0)  # same slot index
        stale = Handle(0, a.epoch)
        with self.assertRaises(StaleHandleError):
            heap.payload(stale)
        heap.collect()
        # An epoch-1 handle is stale at epoch 2 even with a valid address.
        old1 = Handle(0, 1)
        with self.assertRaises(StaleHandleError):
            heap.payload(old1)

    def test_dead_address_handle_rejected(self):
        heap = Heap(12)
        heap.allocate(9)  # unrooted garbage at address 0
        a = heap.allocate(1, n_strong=2)  # size 4 at address 2
        heap.add_root("r", a)
        heap.collect()
        # The survivor compacts to slots 0..3; everything from used_slots
        # (4) upward is free.  A current-epoch handle into free space must be
        # rejected even though its epoch matches.
        forged = Handle(heap.used_slots, heap.epoch)
        with self.assertRaises(StaleHandleError):
            heap.payload(forged)
        with self.assertRaises(StaleHandleError):
            heap.handle(heap.capacity - 1)

    def test_two_collections_relocation_map_complete(self):
        heap = Heap(16)
        heap.allocate(0)  # garbage
        a = heap.allocate(1)
        heap.add_root("r", a)
        r1 = heap.collect()
        self.assertIn(a.address, r1.relocated)
        addr_after_1 = r1.relocated[a.address]
        r2 = heap.collect()
        # Nothing relocates between two back-to-back compactions of one object,
        # but the map still reports the survivor and epoch advances.
        self.assertEqual(r2.relocated, {addr_after_1: 0})
        self.assertEqual(r2.live_slots, 2)
        self.assertEqual(heap.epoch, 2)


# --------------------------------------------------------------------------- #
# Input validation contract
# --------------------------------------------------------------------------- #
class TestInputValidation(unittest.TestCase):
    def test_bad_capacity_types(self):
        for bad in (True, False):
            with self.assertRaises(TypeError):
                Heap(bad)
        for bad in ("4", 1.0, 1 + 0j, None):
            with self.assertRaises(TypeError):
                Heap(bad)
        with self.assertRaises(ValueError):
            Heap(float("nan"))
        with self.assertRaises(ValueError):
            Heap(float("inf"))
        with self.assertRaises(ValueError):
            Heap(-float("inf"))

    def test_bad_allocate_arguments(self):
        heap = Heap(8)
        with self.assertRaises(TypeError):
            heap.allocate(True)
        with self.assertRaises(ValueError):
            heap.allocate(float("nan"))
        with self.assertRaises(ValueError):
            heap.allocate(float("-inf"))
        with self.assertRaises(TypeError):
            heap.allocate(1.5)
        with self.assertRaises(TypeError):
            heap.allocate(0, n_strong=True)
        with self.assertRaises(ValueError):
            heap.allocate(0, n_strong=-1)
        with self.assertRaises(ValueError):
            heap.allocate(0, n_weak=-1)

    def test_bad_field_arguments(self):
        heap = Heap(16)
        h = heap.allocate(0, n_strong=1, n_weak=1)
        with self.assertRaises(TypeError):
            heap.get_strong(h, True)
        with self.assertRaises(IndexError):
            heap.get_strong(h, 1)
        with self.assertRaises(IndexError):
            heap.get_weak(h, -1)
        with self.assertRaises(TypeError):
            heap.set_strong(h, 0, "not-a-handle")
        with self.assertRaises(TypeError):
            heap.set_weak(h, 0, 7)
        with self.assertRaises(ValueError):
            heap.set_payload(h, math.inf)

    def test_bad_root_names(self):
        heap = Heap(8)
        h = heap.allocate(0)
        with self.assertRaises(TypeError):
            heap.add_root("", h)
        with self.assertRaises(TypeError):
            heap.add_root(123, h)  # type: ignore[arg-type]
        with self.assertRaises(KeyError):
            heap.remove_root("missing")
        with self.assertRaises(KeyError):
            heap.root_handle("missing")

    def test_failed_root_replace_preserves_old_root(self):
        heap = Heap(16)
        a = heap.allocate(1)
        b = heap.allocate(2)
        heap.add_root("r", a)
        heap.collect()  # b dies anyway; keep a stale handle
        with self.assertRaises(StaleHandleError):
            heap.add_root("r", b)
        # Original root still resolves to a's copy.
        self.assertEqual(heap.payload(heap.root_handle("r")), 1)

    def test_null_roots_and_removal(self):
        heap = Heap(16)
        a = heap.allocate(1)
        heap.add_root("r", a)
        heap.clear_root("r")
        self.assertIsNone(heap.root_handle("r"))
        heap.collect()
        self.assertIsNone(heap.root_handle("r"))
        self.assertEqual(heap.objects(), ())
        heap.remove_root("r")
        with self.assertRaises(KeyError):
            heap.root_handle("r")


# --------------------------------------------------------------------------- #
# Randomized differential fuzz against the independent oracle
# --------------------------------------------------------------------------- #
class TestDifferentialFuzz(unittest.TestCase):
    def test_seeded_random_graphs(self):
        rng = random.Random(20260922)
        for trial in range(12):
            cap = 120
            spec = GraphSpec(cap)
            heap = Heap(cap)
            handles: dict[str, Handle] = {}

            def add_node(name):
                ns = rng.randrange(0, 3)
                nw = rng.randrange(0, 2)
                if 2 + ns + nw > cap:
                    return False
                try:
                    spec.add_node(name, rng.randrange(0, 1_000_000), ns, nw)
                except RuntimeError:
                    return False
                handles[name] = heap.allocate(
                    spec.nodes[name].payload, n_strong=ns, n_weak=nw
                )
                return True

            for i in range(25):
                add_node(f"n{i}")

            names = list(spec.nodes)
            for name in names:
                node = spec.nodes[name]
                for j in range(len(node.strong)):
                    if rng.random() < 0.6:
                        t = rng.choice(names)
                        spec.link_strong(name, j, t)
                        heap.set_strong(handles[name], j, handles[t])
                for j in range(len(node.weak)):
                    if rng.random() < 0.6:
                        t = rng.choice(names)
                        spec.link_weak(name, j, t)
                        heap.set_weak(handles[name], j, handles[t])

            # Random weak-key associations (side metadata, not slots): a
            # handful of (table, key, value) triples over the node set.
            for _ in range(14):
                tbl, key, val = rng.sample(names, 3)
                spec.assoc_put(tbl, key, val)
                heap.association_put(handles[tbl], handles[key], handles[val])

            rooted = rng.sample(names, k=max(1, len(names) // 3))
            for k, name in enumerate(rooted):
                spec.add_root(f"r{k}", name)
                heap.add_root(f"r{k}", handles[name])

            # Round 1: collection + full comparison.
            old = old_address_map(spec)
            result = heap.collect()
            expected = spec.expected_collect()
            fresh = verify_against_oracle(spec, heap, expected, result, old)

            # Mutate between GCs: drop half the roots, allocate new nodes,
            # rewire a few edges, then collect again and re-compare.
            drop = spec.root_order[::2]
            for rn in drop:
                spec.drop_root(rn)
                heap.remove_root(rn)

            for i in range(8):
                name = f"m{trial}_{i}"
                ns = rng.randrange(0, 2)
                nw = rng.randrange(0, 2)
                try:
                    spec.add_node(name, rng.randrange(0, 1_000_000), ns, nw)
                except RuntimeError:
                    break
                handles[name] = heap.allocate(
                    spec.nodes[name].payload, n_strong=ns, n_weak=nw
                )
                survivors = list(fresh)
                if survivors:
                    t = rng.choice(survivors)
                    spec.add_root(f"newr{i}", name)
                    heap.add_root(f"newr{i}", handles[name])
                    if ns > 0 and rng.random() < 0.7:
                        spec.link_strong(name, 0, t)
                        heap.set_strong(handles[name], 0, fresh[t])
                fresh[name] = handles[name]

            old2 = {name: node.address for name, node in spec.nodes.items()}
            result2 = heap.collect()
            expected2 = spec.expected_collect()
            verify_against_oracle(spec, heap, expected2, result2, old2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
