"""Independent reference oracle for the moving-GC tests.

This module deliberately does **not** import :mod:`moving_gc`.  It models
the same abstract machine on its own terms -- named objects stored in
plain dictionaries, reachability computed by an independent BFS over
strong edges plus a least-fixpoint over weak-key entries, and the expected
Cheney evacuation simulated with an explicit queue.  Tests build a graph
once as a :class:`GraphSpec`, feed an equivalent heap through the public
Heap API, and assert that the heap's behaviour matches the oracle's
predictions.

Address convention (mirrors the heap contract):
* objects occupy ``2 + n_strong + n_weak + 2*n_entries`` consecutive slots;
* a node's address is the slot index of its header;
* both worlds allocate in ``GraphSpec.add_node`` call order, so the test
  harness can reproduce identical pre-collection addresses.

Weak-key entry semantics (mirrors the heap contract):
* an entry ``(key, value)`` of a *live* table activates iff its key is
  already live; activation makes the value strongly reachable;
* activation is iterated to a fixpoint (values can keep other keys, and
  other tables' values, alive);
* a key referenced only by its own entry's value never becomes live;
* entries whose key stayed dead are deleted (both slots cleared).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Node:
    name: str
    payload: int
    strong: list[str | None]
    weak: list[str | None]
    # (key, value) node-name pairs; key None means a vacant entry.
    entries: list[tuple[str | None, str | None]]
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
    # (table node name, entry index) in deterministic table/entry order
    cleared_entries: tuple[tuple[str, int], ...]
    # root name -> survivor node name (or None for a null root)
    roots: dict[str, str | None]
    # copy order (node names), i.e. destination-address ascending order
    copy_order: tuple[str, ...]


class GraphSpec:
    """A tiny named-object graph used both to drive the heap and the oracle."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.nodes: dict[str, _Node] = {}
        # Roots as (name, target) pairs kept in insertion order.
        self.root_order: list[str] = []
        self.roots: dict[str, str | None] = {}
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
        n_entries: int = 0,
    ) -> int:
        if name in self.nodes:
            raise ValueError(f"duplicate node {name!r}")
        size = 2 + n_strong + n_weak + 2 * n_entries
        if self._alloc + size > self.capacity:
            raise RuntimeError("reference graph allocation overflows its declared capacity")
        addr = self._alloc
        self._alloc += size
        self.nodes[name] = _Node(
            name=name,
            payload=payload,
            strong=[None] * n_strong,
            weak=[None] * n_weak,
            entries=[(None, None)] * n_entries,
            address=addr,
        )
        return addr

    def link_strong(self, src: str, index: int, dst: str | None) -> None:
        self.nodes[src].strong[index] = dst

    def link_weak(self, src: str, index: int, dst: str | None) -> None:
        self.nodes[src].weak[index] = dst

    def link_entry(
        self, src: str, index: int, key: str | None, value: str | None = None
    ) -> None:
        if key is None and value is not None:
            raise ValueError("reference: entry with a null key cannot hold a value")
        self.nodes[src].entries[index] = (key, value)

    def add_root(self, root_name: str, target: str | None) -> None:
        if root_name not in self.roots:
            self.root_order.append(root_name)
        self.roots[root_name] = target

    def drop_root(self, root_name: str) -> None:
        del self.roots[root_name]
        self.root_order.remove(root_name)

    # -------------------------------------------------------------- #
    # Independent reachability (strong edges + entry fixpoint)
    # -------------------------------------------------------------- #
    def reachable(self) -> frozenset[str]:
        """Live node names: strong closure from roots, then the least
        fixpoint of weak-key entry activation, computed directly on the
        named-object graph."""
        live: set[str] = set()
        queue: list[str] = []
        for rn in self.root_order:
            t = self.roots[rn]
            if t is not None and t not in live:
                live.add(t)
                queue.append(t)
        activated: set[tuple[str, int]] = set()
        head = 0
        while True:
            while head < len(queue):
                cur = queue[head]
                head += 1
                for t in self.nodes[cur].strong:
                    if t is not None and t not in live:
                        live.add(t)
                        queue.append(t)
            progress = False
            for name in list(live):
                for i, (k, v) in enumerate(self.nodes[name].entries):
                    if (name, i) in activated or k is None:
                        continue
                    if k in live:
                        activated.add((name, i))
                        if v is not None and v not in live:
                            live.add(v)
                            queue.append(v)
                            progress = True
            if not progress and head >= len(queue):
                break
        return frozenset(live)

    # -------------------------------------------------------------- #
    # Expected collection
    # -------------------------------------------------------------- #
    def expected_collect(self) -> ExpectedCollection:
        """Simulate one collection purely from the graph description."""
        fwd: dict[str, int] = {}
        old_addr = {name: node.address for name, node in self.nodes.items()}
        copy_order: list[str] = []
        free = 0
        queue: list[str] = []

        def size(name: str) -> int:
            n = self.nodes[name]
            return 2 + len(n.strong) + len(n.weak) + 2 * len(n.entries)

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

        expected_roots: dict[str, str | None] = {}
        for rn in self.root_order:
            t = self.roots[rn]
            if t is None:
                expected_roots[rn] = None
            else:
                evacuate(t)
                expected_roots[rn] = t

        # Independent Cheney scan in field order.
        head = 0

        def drain_strong() -> None:
            nonlocal head
            while head < len(queue):
                cur = queue[head]
                head += 1
                for t in self.nodes[cur].strong:
                    if t is not None:
                        evacuate(t)

        drain_strong()

        # Entry fixpoint: iterate live tables in copy order (copy_order
        # grows during the pass, exactly like the heap's live_tables list);
        # an entry activates iff its key is already evacuated.  Activated
        # values are evacuated and their strong fields drained before the
        # next pass, until nothing new activates.
        activated: set[tuple[str, int]] = set()
        while True:
            progress = False
            for name in copy_order:
                node = self.nodes[name]
                for i, (k, v) in enumerate(node.entries):
                    if (name, i) in activated:
                        continue
                    if k is None:
                        activated.add((name, i))
                        continue
                    if k in fwd:
                        if v is not None:
                            evacuate(v)
                        activated.add((name, i))
                        progress = True
            drain_strong()
            if not progress:
                break

        # Dead-key entries of live tables are deleted, in table/entry order.
        cleared_entries: list[tuple[str, int]] = []
        for name in copy_order:
            node = self.nodes[name]
            for i, (k, _v) in enumerate(node.entries):
                if (name, i) not in activated and k is not None:
                    cleared_entries.append((name, i))

        # Weak fix-up: survivors in copy (== destination address) order.
        cleared: list[tuple[str, int]] = []
        for name in copy_order:
            for j, t in enumerate(self.nodes[name].weak):
                if t is None:
                    continue
                if t in fwd:
                    pass  # survives: heap must forward to fwd[t]
                else:
                    cleared.append((name, j))

        relocated = {name: (old_addr[name], fwd[name]) for name in copy_order}

        # Commit the oracle's own view of the world so it can be collected
        # again: dead nodes disappear, survivors take their new addresses,
        # weak edges that named dead targets are physically cleared, and
        # non-activated entries are physically deleted (strong edges only
        # ever name survivors by construction).
        for name in copy_order:
            self.nodes[name].address = fwd[name]
            node = self.nodes[name]
            node.weak = [t if (t is None or t in fwd) else None for t in node.weak]
            node.entries = [
                (k, v) if (name, i) in activated else (None, None)
                for i, (k, v) in enumerate(node.entries)
            ]
        for dead in [n for n in self.nodes if n not in fwd]:
            del self.nodes[dead]
        self._alloc = free
        self.roots = expected_roots

        return ExpectedCollection(
            survivors=frozenset(copy_order),
            relocated=relocated,
            live_slots=free,
            cleared=tuple(cleared),
            cleared_entries=tuple(cleared_entries),
            roots=dict(expected_roots),
            copy_order=tuple(copy_order),
        )

    def address_of(self, name: str) -> int:
        """Current-space address of a live node name."""
        return self.nodes[name].address
