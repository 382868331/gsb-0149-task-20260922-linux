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
    def reachable(self) -> frozenset[str]:
        """Strongly reachable node names from the registered roots."""
        seen: set[str] = set()
        queue: list[str] = []
        for rn in self.root_order:
            t = self.roots[rn]
            if t is not None and t not in seen:
                seen.add(t)
                queue.append(t)
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for t in self.nodes[cur].strong:
                if t is not None and t not in seen:
                    seen.add(t)
                    queue.append(t)
        return frozenset(seen)

    # -------------------------------------------------------------- #
    # Expected collection
    # -------------------------------------------------------------- #
    def expected_collect(self) -> ExpectedCollection:
        """Simulate one collection purely from the graph description."""
        survivors = self.reachable()

        fwd: dict[str, int] = {}
        old_addr = {name: node.address for name, node in self.nodes.items()}
        copy_order: list[str] = []
        free = 0
        queue: list[str] = []

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
        while head < len(queue):
            cur = queue[head]
            head += 1
            for t in self.nodes[cur].strong:
                if t is not None:
                    evacuate(t)

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

        return ExpectedCollection(
            survivors=frozenset(copy_order),
            relocated=relocated,
            live_slots=free,
            cleared=tuple(cleared),
            roots=dict(expected_roots),
            copy_order=tuple(copy_order),
        )

    def address_of(self, name: str) -> int:
        """Current-space address of a live node name."""
        return self.nodes[name].address
