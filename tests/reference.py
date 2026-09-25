"""Independent reference oracle for the moving-GC tests.

This module deliberately does **not** import :mod:`moving_gc`.  It models
the same abstract machine on its own terms -- named objects stored in
plain dictionaries, reachability computed by an independent BFS over
strong edges, and the expected Cheney evacuation simulated with an
explicit queue.  Tests build a graph once as a :class:`GraphSpec`, feed an
equivalent heap through the public Heap API, and assert that the heap's
behaviour matches the oracle's predictions.

Address convention (mirrors the heap contract):
* objects occupy ``2 + n_strong + n_weak`` consecutive slots;
* a node's address is the slot index of its header;
* both worlds allocate in ``GraphSpec.add_node`` call order, so the test
  harness can reproduce identical pre-collection addresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Node:
    name: str
    payload: int
    strong: list[str | None]
    weak: list[str | None]
    address: int  # address in the current semi-space


@dataclass
class ExpectedCollection:
    """Oracle prediction for one collection."""

    survivors: frozenset[str]
    # node name -> (old_address, new_address)
    relocated: dict[str, tuple[int, int]]
    live_slots: int
    # (holder node name, weak field index) in deterministic holder/field order
    cleared: tuple[tuple[str, int], ...]
    # root name -> survivor node name (or None for a null root)
    roots: dict[str, str | None]
    # copy order (node names), i.e. destination-address ascending order
    copy_order: tuple[str, ...]
    # weak-key entries that stayed live: (table, key, value) names, ordered
    # by post-collection (table address, key address)
    associations_surviving: tuple[tuple[str, str, str], ...] = ()
    # weak-key entries deleted for a dead key on a live table:
    # (table, key) names, ordered by pre-collection (table address, key
    # address); entries of a dead table are not listed
    associations_cleared: tuple[tuple[str, str], ...] = ()


class GraphSpec:
    """A tiny named-object graph used both to drive the heap and the oracle."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.nodes: dict[str, _Node] = {}
        # Roots as (name, target) pairs kept in insertion order.
        self.root_order: list[str] = []
        self.roots: dict[str, str | None] = {}
        # Weak-key associations in side metadata (mirrors the heap):
        # table name -> {key name: value name}.
        self.associations: dict[str, dict[str, str]] = {}
        self._alloc = 0

    # -------------------------------------------------------------- #
    # Construction
    # -------------------------------------------------------------- #
    def add_node(
        self,
        name: str,
        payload: int = 0,
        n_strong: int = 0,
        n_weak: int = 0,
    ) -> int:
        if name in self.nodes:
            raise ValueError(f"duplicate node {name!r}")
        size = 2 + n_strong + n_weak
        if self._alloc + size > self.capacity:
            raise RuntimeError("reference graph allocation overflows its declared capacity")
        addr = self._alloc
        self._alloc += size
        self.nodes[name] = _Node(
            name=name,
            payload=payload,
            strong=[None] * n_strong,
            weak=[None] * n_weak,
            address=addr,
        )
        return addr

    def link_strong(self, src: str, index: int, dst: str | None) -> None:
        self.nodes[src].strong[index] = dst

    def link_weak(self, src: str, index: int, dst: str | None) -> None:
        self.nodes[src].weak[index] = dst

    def assoc_put(self, table: str, key: str, value: str | None) -> None:
        """Set a weak-key association; ``value=None`` deletes the entry."""
        entries = self.associations.setdefault(table, {})
        if value is None:
            entries.pop(key, None)
        else:
            entries[key] = value

    def assoc_remove(self, table: str, key: str) -> None:
        if table in self.associations:
            self.associations[table].pop(key, None)

    def add_root(self, root_name: str, target: str | None) -> None:
        if root_name not in self.roots:
            self.root_order.append(root_name)
        self.roots[root_name] = target

    def drop_root(self, root_name: str) -> None:
        del self.roots[root_name]
        self.root_order.remove(root_name)

    # -------------------------------------------------------------- #
    # Independent reachability (strong edges only)
    # -------------------------------------------------------------- #
    def _strong_close(self, seeds: set[str]) -> set[str]:
        """Close a seed set under strong edges (BFS on the object-ID graph)."""
        seen = set(seeds)
        queue = list(seeds)
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for t in self.nodes[cur].strong:
                if t is not None and t not in seen:
                    seen.add(t)
                    queue.append(t)
        return seen

    def reachable(self) -> frozenset[str]:
        """Strongly reachable node names from the registered roots."""
        seeds: set[str] = set()
        for rn in self.root_order:
            t = self.roots[rn]
            if t is not None:
                seeds.add(t)
        return frozenset(self._strong_close(seeds))

    def liveness_fixpoint(self) -> frozenset[str]:
        """Least fixpoint of GC liveness with weak-key associations.

        Computed directly on the object-ID graph, independently of any
        copying::

            X0 = strong-closure(roots)
            X_{n+1} = strong-closure(
                X_n | { value | (table, key -> value) association,
                         table in X_n, key in X_n } )

        The least fixpoint is the least set satisfying the rule, so a key
        reachable only from its own entry's not-yet-live value cannot pull
        itself (or that value) into the set.
        """
        live = set(self.reachable())
        while True:
            activated_values: set[str] = set()
            for table, entries in self.associations.items():
                if table not in live:
                    continue
                for key, value in entries.items():
                    if key in live:
                        activated_values.add(value)
            grown = self._strong_close(live | activated_values)
            if grown == live:
                return frozenset(live)
            live = grown

    # -------------------------------------------------------------- #
    # Expected collection
    # -------------------------------------------------------------- #
    def expected_collect(self) -> ExpectedCollection:
        """Simulate one collection purely from the graph description."""
        old_addr = {name: node.address for name, node in self.nodes.items()}

        fwd: dict[str, int] = {}
        copy_order: list[str] = []
        free = 0
        queue: list[str] = []  # Cheney scan queue of copied names
        head = 0

        def size(name: str) -> int:
            n = self.nodes[name]
            return 2 + len(n.strong) + len(n.weak)

        def evacuate(name: str) -> int:
            nonlocal free
            if name in fwd:
                return fwd[name]
            new_addr = free
            free += size(name)
            if free > self.capacity:
                raise RuntimeError("reference: live set does not fit (oracle)")
            fwd[name] = new_addr
            copy_order.append(name)
            queue.append(name)
            return new_addr

        def drain() -> None:
            nonlocal head
            while head < len(queue):
                cur = queue[head]
                head += 1
                for t in self.nodes[cur].strong:
                    if t is not None:
                        evacuate(t)

        # Level 1: roots in registration order, then the strong closure.
        expected_roots: dict[str, str | None] = {}
        for rn in self.root_order:
            t = self.roots[rn]
            if t is None:
                expected_roots[rn] = None
            else:
                evacuate(t)
                expected_roots[rn] = t
        drain()

        # Level 2: weak-key association fixpoint, mirroring the heap's
        # deterministic order: tables by *new* address, entries by *old*
        # key address; copy each activated value and drain strong edges
        # before the next entry.
        activated: set[tuple[str, str]] = set()
        while True:
            table_order = sorted(
                (tbl for tbl in self.associations if tbl in fwd),
                key=lambda tbl: fwd[tbl],
            )
            progressed = False
            for tbl in table_order:
                entries = sorted(
                    self.associations[tbl].items(),
                    key=lambda kv: old_addr[kv[0]],
                )
                for key, value in entries:
                    entry = (tbl, key)
                    if entry in activated:
                        continue
                    if key in fwd:
                        evacuate(value)
                        activated.add(entry)
                        progressed = True
                        drain()
            if not progressed:
                break

        # The copy-based survivor set must equal the direct least-fixpoint
        # computed on the object-ID graph (oracle self-consistency check).
        assert frozenset(copy_order) == self.liveness_fixpoint()

        survivors = frozenset(copy_order)

        # Plain weak fix-up, in copy (destination address) order.
        cleared: list[tuple[str, int]] = []
        for name in copy_order:
            for j, t in enumerate(self.nodes[name].weak):
                if t is None:
                    continue
                if t not in fwd:
                    cleared.append((name, j))

        # Association bookkeeping, in the heap's exact address orders.
        assoc_surviving: list[tuple[str, str, str]] = []
        assoc_cleared: list[tuple[str, str]] = []
        new_associations: dict[str, dict[str, str]] = {}
        for tbl, entries in self.associations.items():
            if tbl not in fwd:
                continue  # dead table: entries vanish, not reported
            new_map: dict[str, str] = {}
            for key, value in entries.items():
                if key in fwd:
                    new_map[key] = value
                    assoc_surviving.append((tbl, key, value))
                else:
                    assoc_cleared.append((tbl, key))
            if new_map:
                new_associations[tbl] = new_map
        assoc_surviving.sort(key=lambda t: (fwd[t[0]], fwd[t[1]]))
        assoc_cleared.sort(key=lambda t: (old_addr[t[0]], old_addr[t[1]]))

        relocated = {name: (old_addr[name], fwd[name]) for name in copy_order}

        # Commit the oracle's own view of the world so it can be collected
        # again: dead nodes disappear, survivors take their new addresses,
        # and weak edges that named dead targets are physically cleared
        # (strong edges only ever name survivors by construction).
        for name in copy_order:
            self.nodes[name].address = fwd[name]
            node = self.nodes[name]
            node.weak = [t if (t is None or t in fwd) else None for t in node.weak]
        for dead in [n for n in self.nodes if n not in fwd]:
            del self.nodes[dead]
        self._alloc = free
        self.roots = expected_roots
        self.associations = new_associations

        return ExpectedCollection(
            survivors=survivors,
            relocated=relocated,
            live_slots=free,
            cleared=tuple(cleared),
            roots=dict(expected_roots),
            copy_order=tuple(copy_order),
            associations_surviving=tuple(assoc_surviving),
            associations_cleared=tuple(assoc_cleared),
        )

    def address_of(self, name: str) -> int:
        """Current-space address of a live node name."""
        return self.nodes[name].address
