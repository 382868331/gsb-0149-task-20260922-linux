"""Independent reference oracle for the generational collector tests.

This module deliberately does **not** import :mod:`moving_gc`.  It models
the same abstract machine with named objects in plain dictionaries.

:class:`RefGen` is a specification-first batch oracle.  It exposes the
same *mutation* surface as ``moving_gc.GenHeap`` (allocate / set edges /
roots / ephemerons / finalizers / forced promotion) and, for a finalizer
callback, invokes the very same test-supplied callback against itself --
so resurrection and allocation-during-finalization are modelled by
performing those operations on the oracle, never by calling the code
under test.

Expected post-collection worlds are produced by:

* **minor** -- seed young roots, remembered-set young targets and young
  objects carrying an unrun finalizer; strong closure restricted to the
  young generation; ephemeron least fixpoint; age/forced promotion;
  dense young survivor packing in source-raw order plus first-fit old
  placement; forwarding, weak clearing, ephemeron rewrite.
* **major** -- full strong closure; ephemeron fixpoint; address-ordered
  one-shot finalizer candidates, each callback followed by re-closing the
  closure and the ephemeron fixpoint; ordinary weak clearing; ephemeron
  pruning; old-only sweep with hole coalescing.

:class:`MarkLedger` is a second, independent model: an incremental marker
driven one scan unit at a time in lockstep with collector steps, applying
the same scheduled mutator writes, so tests can assert the
incremental-update barrier state (grey queue / marked set / barrier
count) after every single unit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RObj:
    name: str
    gen: int
    raw: int
    size: int
    age: int
    payload: int
    strong: list
    weak: list


@dataclass
class RExpected:
    kind: str
    survivors: frozenset
    # name -> logical address expected after the collection
    addr_of: dict
    relocated: dict          # name -> new young logical address
    promoted: dict           # name -> new old logical address
    cleared_weak: tuple      # (holder name, field)
    cleared_eph: tuple
    finalized: tuple         # (name, token, resurrected)
    young_used: int
    old_used: int
    roots: dict
    strong: dict
    weak: dict


class RefGen:
    def __init__(self, young_capacity, old_capacity, promote_age=2):
        self.yc = young_capacity
        self.oc = old_capacity
        self.promote_age = promote_age
        self.objs: dict[str, RObj] = {}
        self.ybump = 0
        self.obump = 0
        self.old_holes: list[tuple[int, int]] = []
        self.roots: dict[str, object] = {}
        self.root_order: list[str] = []
        self.eph: dict[int, tuple] = {}
        self.eph_order: list[int] = []
        self._eph_next = 0
        self.fin: dict[str, tuple] = {}      # name -> (callback, token)
        self.fin_ran: set[str] = set()
        self._fin_next = 0
        self.force: set[str] = set()

    # ------------------------------------------------------------- #
    # Mutation surface (mirrors GenHeap)
    # ------------------------------------------------------------- #
    def allocate(self, name, payload=0, *, n_strong=0, n_weak=0):
        size = 2 + n_strong + n_weak
        self.objs[name] = RObj(name, 0, self.ybump, size, 0, payload,
                               [None] * n_strong, [None] * n_weak)
        self.ybump += size
        return name

    def set_strong(self, src, index, dst):
        self.objs[src].strong[index] = dst

    def set_weak(self, src, index, dst):
        self.objs[src].weak[index] = dst

    def add_root(self, name, target):
        if name not in self.roots:
            self.root_order.append(name)
        self.roots[name] = target

    def remove_root(self, name):
        del self.roots[name]
        self.root_order.remove(name)

    def clear_root(self, name):
        self.roots[name] = None

    def add_ephemeron(self, table, key, value):
        eid = self._eph_next
        self._eph_next += 1
        self.eph[eid] = (table, key, value)
        self.eph_order.append(eid)
        return eid

    def set_finalizer(self, target, callback):
        token = self._fin_next
        self._fin_next += 1
        self.fin[target] = (callback, token)
        return token

    def request_promotion(self, target):
        self.force.add(target)

    def gen_of(self, name):
        return self.objs[name].gen

    # ------------------------------------------------------------- #
    # Reachability + ephemeron fixpoint (independent algorithms)
    # ------------------------------------------------------------- #
    def _root_seeds(self):
        out = []
        for rn in self.root_order:
            t = self.roots[rn]
            if t is not None and t in self.objs:
                out.append(t)
        return out

    def strong_closure(self, seeds, young_only):
        seen = set()
        stack = [s for s in seeds if s in self.objs]
        while stack:
            n = stack.pop()
            if n in seen:
                # Duplicate enqueue: mark membership is granted at pop, so
                # this only suppresses repeats, never the first expansion.
                continue
            seen.add(n)
            for t in self.objs[n].strong:
                if t is None or t not in self.objs or t in seen:
                    continue
                if young_only and self.objs[t].gen != 0:
                    continue
                if t not in stack:
                    stack.append(t)
        return seen

    def ephemeron_lfp(self, alive, young_only):
        def reachable(name):
            # In a minor (young_only) the old generation is never reclaimed,
            # so an old table/key is treated as live even though it is not
            # part of the young-only closure set.
            if name not in self.objs:
                return False
            if young_only and self.objs[name].gen == 1:
                return True
            return name in alive

        progressed = True
        while progressed:
            progressed = False
            for eid in self.eph_order:
                t, k, v = self.eph[eid]
                if not reachable(t) or not reachable(k):
                    continue
                if v is None or v not in self.objs or v in alive:
                    continue
                if young_only and self.objs[v].gen != 0:
                    continue
                alive.add(v)
                # Newly activated value may bring strong children.
                alive |= self.strong_closure([v], young_only)
                progressed = True
        return alive

    def _rs_young_targets(self):
        """Distinct young targets of old holders' strong edges, holder order."""
        targets = []
        seen = set()
        for holder in sorted((n for n in self.objs if self.objs[n].gen == 1),
                             key=lambda n: self.objs[n].raw):
            for t in self.objs[holder].strong:
                if t is not None and t in self.objs and self.objs[t].gen == 0 and t not in seen:
                    seen.add(t)
                    targets.append(t)
        return targets

    # ------------------------------------------------------------- #
    # Minor
    # ------------------------------------------------------------- #
    def expected_minor(self):
        seeds = []
        seen = set()

        def seed(n):
            if n in self.objs and self.objs[n].gen == 0 and n not in seen:
                seen.add(n)
                seeds.append(n)

        for t in self._root_seeds():
            seed(t)
        for t in self._rs_young_targets():
            seed(t)
        for n in sorted(self.objs):
            o = self.objs[n]
            if o.gen == 0 and n in self.fin and n not in self.fin_ran:
                seed(n)

        alive = self.strong_closure(seeds, True)
        alive = self.ephemeron_lfp(alive, True)

        young = sorted((n for n in self.objs if self.objs[n].gen == 0),
                       key=lambda n: self.objs[n].raw)
        promote = {n for n in young if n in alive and
                   (n in self.force or self.objs[n].age + 1 >= self.promote_age)}
        stay = [n for n in young if n in alive and n not in promote]

        yraw = {}
        run = 0
        for n in stay:
            yraw[n] = run
            run += self.objs[n].size
        holes = [tuple(h) for h in self.old_holes]
        hwm = self.obump
        oraw = {}

        def reserve(sz):
            nonlocal hwm
            for i, (hr, hs) in enumerate(holes):
                if hs >= sz:
                    del holes[i]
                    if hs > sz:
                        holes.append((hr + sz, hs - sz))
                        holes.sort()
                    return hr
            hr = hwm
            hwm += sz
            return hr

        for n in sorted(promote, key=lambda x: self.objs[x].raw):
            oraw[n] = reserve(self.objs[n].size)

        def target_after(src, t, weak):
            if t is None:
                return None
            if t not in self.objs:
                return None
            tg = self.objs[t].gen
            if tg == 1:
                return t
            return t if t in alive else (None if weak else t)

        new_strong, new_weak, cleared = {}, {}, []
        young_survivors = set(stay) | promote

        def rewrite_obj(n):
            o = self.objs[n]
            ns_edges = []
            for t in o.strong:
                if t is None:
                    ns_edges.append(None)
                elif self.objs[t].gen == 1:
                    ns_edges.append(t)            # old target stable
                elif t in alive:
                    ns_edges.append(t)            # young survivor, forwarded
                else:
                    ns_edges.append(t)            # strong-dying cannot occur
            wk = []
            for j, t in enumerate(o.weak):
                r = target_after(n, t, True)
                wk.append(r)
                if t is not None and r is None:
                    cleared.append((n, j))
            return ns_edges, wk

        for n in young_survivors:
            new_strong[n], new_weak[n] = rewrite_obj(n)
        # Pre-existing old holders: forward their young edges, clear weak
        # edges to young objects that died.
        for n, o in self.objs.items():
            if o.gen == 1:
                new_strong[n], new_weak[n] = rewrite_obj(n)

        new_eph, cleared_eph = {}, []
        for eid in self.eph_order:
            t, k, v = self.eph[eid]

            def minor_missing(name):
                if name is None:
                    return False
                if name not in self.objs:
                    return True
                return self.objs[name].gen == 0 and name not in alive

            if minor_missing(t) or minor_missing(k):
                cleared_eph.append(eid)
                continue
            nv = None if minor_missing(v) else v
            new_eph[eid] = (t, k, nv)

        new_roots = {}
        for rn, t in self.roots.items():
            if t is None:
                new_roots[rn] = None
            elif self.objs[t].gen == 1:
                new_roots[rn] = t
            else:
                new_roots[rn] = t if t in alive else None

        addr = {n: yraw[n] for n in stay}
        addr.update({n: self.yc + oraw[n] for n in promote})
        # A minor never reclaims the old generation: every pre-existing
        # old object survives at its stable address.
        for n, o in self.objs.items():
            if o.gen == 1:
                addr[n] = self.yc + o.raw
        # Occupied old slots after this minor = pre-existing live old
        # objects plus the objects promoted this minor (computed before the
        # commit flips generations, so use the result sets explicitly).
        occupied = sum(self.objs[n].size for n in self.objs if self.objs[n].gen == 1)
        occupied += sum(self.objs[n].size for n in promote)
        exp = RExpected(
            kind="minor", survivors=frozenset(alive), addr_of=addr,
            relocated={n: yraw[n] for n in stay},
            promoted={n: self.yc + oraw[n] for n in promote},
            cleared_weak=tuple(sorted(cleared)), cleared_eph=tuple(cleared_eph),
            finalized=(), young_used=run, old_used=occupied,
            roots=new_roots, strong=new_strong, weak=new_weak)
        exp._new_eph = new_eph
        exp._holes = holes
        exp._hwm = hwm
        exp._stay = stay
        exp._promote = promote
        exp._yraw = yraw
        exp._oraw = oraw
        exp._dead_young = [n for n in young if n not in alive]
        return exp

    def commit_minor(self, exp):
        for n in exp._dead_young:
            del self.objs[n]
        for n in exp._stay:
            o = self.objs[n]
            o.gen, o.raw, o.age = 0, exp._yraw[n], o.age + 1
        for n in exp._promote:
            o = self.objs[n]
            o.gen, o.raw, o.age = 1, exp._oraw[n], 0
        for n, edges in exp.strong.items():
            self.objs[n].strong = list(edges)
        for n, edges in exp.weak.items():
            self.objs[n].weak = list(edges)
        self.eph = dict(exp._new_eph)
        self.eph_order = [e for e in self.eph_order if e in self.eph]
        self.roots = dict(exp.roots)
        self.ybump = exp.young_used
        # obump is the old high-water mark; occupied-slot accounting is
        # derived from live object sizes, not stored here.
        self.obump = exp._hwm
        self.old_holes = exp._holes
        self.force = set()

    # ------------------------------------------------------------- #
    # Major
    # ------------------------------------------------------------- #
    def expected_major(self):
        alive = self.strong_closure(self._root_seeds(), False)
        alive = self.ephemeron_lfp(alive, False)

        candidates = [
            n for n in sorted(self.objs, key=lambda x: (self.objs[x].gen, self.objs[x].raw))
            if self.objs[n].gen == 1 and n not in alive and n in self.fin and n not in self.fin_ran
        ]
        finalized = []
        for n in candidates:
            if n in alive:
                continue
            cb, token = self.fin[n]
            self.fin_ran.add(n)
            # Explicit event: the same callback the heap runs, applied to
            # the oracle (resurrection via add_root, allocation, rewiring).
            cb(self, n)
            alive = self.strong_closure(self._root_seeds(), False) | alive
            alive = self.ephemeron_lfp(alive, False)
            finalized.append((n, token, n in alive))

        new_strong = {n: list(o.strong) for n, o in self.objs.items()}
        new_weak = {n: list(o.weak) for n, o in self.objs.items()}
        cleared = []
        # Only surviving holders have their weak edges reported as cleared;
        # weak edges of reclaimed holders vanish with the holder.
        for n in self.objs:
            if self.objs[n].gen == 1 and n not in alive:
                continue
            for j, t in enumerate(self.objs[n].weak):
                if t is not None and self.objs[t].gen == 1 and t not in alive:
                    new_weak[n][j] = None
                    cleared.append((n, j))
        # A young holder outside the closure is nursery garbage; null its
        # strong edges to reclaimed old objects so no post-cycle edge names
        # a swept address (mirrors the heap; never reported as a weak clear).
        for n in self.objs:
            if self.objs[n].gen == 0 and n not in alive:
                for j, t in enumerate(new_strong[n]):
                    if t is not None and self.objs[t].gen == 1 and t not in alive:
                        new_strong[n][j] = None
        new_eph, cleared_eph = {}, []
        for eid in self.eph_order:
            t, k, v = self.eph[eid]

            def missing(name):
                if name is None:
                    return False
                if name not in self.objs:
                    return True
                return self.objs[name].gen == 1 and name not in alive

            if missing(t) or missing(k):
                cleared_eph.append(eid)
                continue
            nv = None if missing(v) else v
            new_eph[eid] = (t, k, nv)

        old_dead = [n for n, o in self.objs.items() if o.gen == 1 and n not in alive]
        holes = list(self.old_holes)
        for n in old_dead:
            holes.append((self.objs[n].raw, self.objs[n].size))
        holes.sort()
        merged = []
        for hr, hs in holes:
            if merged and merged[-1][0] + merged[-1][1] == hr:
                merged[-1] = (merged[-1][0], merged[-1][1] + hs)
            else:
                merged.append((hr, hs))
        live_end = [self.objs[n].raw + self.objs[n].size
                    for n in self.objs if self.objs[n].gen == 1 and n in alive]
        hwm = max(live_end) if live_end else 0
        kept = [(hr, hs) for hr, hs in merged if hr + hs <= hwm]

        addr = {}
        for n, o in self.objs.items():
            if o.gen == 0:
                addr[n] = o.raw
            elif n in alive:
                addr[n] = self.yc + o.raw
        exp = RExpected(
            kind="major", survivors=frozenset(alive), addr_of=addr,
            relocated={}, promoted={}, cleared_weak=tuple(sorted(cleared)),
            cleared_eph=tuple(cleared_eph), finalized=tuple(finalized),
            young_used=self.ybump,
            old_used=sum(self.objs[n].size for n in self.objs
                         if self.objs[n].gen == 1 and n in alive),
            roots=dict(self.roots), strong=new_strong, weak=new_weak)
        exp._new_eph = new_eph
        exp._old_dead = old_dead
        exp._holes = kept
        exp._hwm = hwm
        exp._alive = alive
        return exp

    def commit_major(self, exp):
        for n in exp._old_dead:
            del self.objs[n]
        for n, edges in exp.strong.items():
            if n in self.objs:
                self.objs[n].strong = list(edges)
        for n, edges in exp.weak.items():
            if n in self.objs:
                self.objs[n].weak = list(edges)
        self.eph = dict(exp._new_eph)
        self.eph_order = [e for e in self.eph_order if e in self.eph]
        self.old_holes = exp._holes
        self.obump = exp._hwm


# --------------------------------------------------------------------------- #
# Independent incremental mark ledger
# --------------------------------------------------------------------------- #
class MarkLedger:
    """One-unit-at-a-time mirror of the open marking phases.

    ``gen_of`` / ``root_of`` / ``edges_of`` are live callables into the
    test's named world, so scheduled mutator writes between units are
    seen exactly as the heap sees them.  ``rs_fields`` (minor only) lists
    ``(holder, [young target per strong field])`` in holder raw order.
    """

    def __init__(self, kind, root_names, gen_of, root_of, edges_of,
                 rs_fields=(), fin_roots=()):
        self.kind = kind
        self.root_names = list(root_names)
        self.gen_of = gen_of
        self.root_of = root_of
        self.edges_of = edges_of
        self.rs_fields = list(rs_fields)
        self.fin_roots = list(fin_roots)
        self.gray_q = []
        self.black = set()
        self.colored = set()
        self.barrier = 0
        self.root_i = 0
        self.rs_i = 0
        self.rs_j = 0
        self.fr_i = 0
        self.cursor = {}

    def _grey(self, name):
        if name is None or name in self.colored:
            return
        self.colored.add(name)
        self.gray_q.append(name)

    def enum_done(self):
        if self.root_i < len(self.root_names):
            return False
        if self.kind == "minor" and (self.rs_i < len(self.rs_fields)
                                     or self.fr_i < len(self.fin_roots)):
            return False
        return True

    def step_unit(self):
        if not self.enum_done():
            if self.root_i < len(self.root_names):
                t = self.root_of(self.root_names[self.root_i])
                self.root_i += 1
                if t is not None and (self.kind == "major" or self.gen_of(t) == 0):
                    self._grey(t)
                return
            if self.rs_i < len(self.rs_fields):
                holder, fields = self.rs_fields[self.rs_i]
                if self.rs_j < len(fields):
                    t = fields[self.rs_j]
                    self.rs_j += 1
                    if t is not None:
                        self._grey(t)
                    return
                self.rs_i += 1
                self.rs_j = 0
                return
            t = self.fin_roots[self.fr_i]
            self.fr_i += 1
            self._grey(t)
            return
        n = self.gray_q[0]
        j = self.cursor.get(n, 0)
        edges = self.edges_of(n)
        # Mirrors the heap: one unit per strong edge, plus a single visit
        # unit for an object with no strong edges (never reads past the
        # strong window into weak slots).
        if j < len(edges):
            t = edges[j]
            if t is not None and (self.kind == "major" or self.gen_of(t) == 0):
                self._grey(t)
            j += 1
            self.cursor[n] = j
        if j >= len(edges):
            self.gray_q.pop(0)
            self.black.add(n)
            self.cursor.pop(n, None)

    def write_strong(self, holder, index, target):
        """Apply an incremental-update strong write to strong field ``index``.

        Mirrors the heap: minor old->young writes always protect the young
        target; otherwise the target is protected when the holder is black
        (all fields scanned) or grey with this specific field already
        scanned.
        """
        if target is None:
            return
        gh, gt = self.gen_of(holder), self.gen_of(target)
        fired = False
        if self.kind == "minor" and gh == 1 and gt == 0:
            if target not in self.colored:
                self._grey(target)
                fired = True
        else:
            if holder in self.black:
                protect = True
            elif holder in self.colored:  # grey
                protect = index < self.cursor.get(holder, 0)
            else:
                protect = False
            if protect and target not in self.colored:
                self._grey(target)
                fired = True
        if fired:
            self.barrier += 1

    def add_root(self, target):
        if self.kind == "major" or self.gen_of(target) == 0:
            self._grey(target)

    def allocate(self, name, gen=0):
        self._grey(name)

    def gray(self):
        return tuple(self.gray_q)

    def marked(self):
        return frozenset(self.colored)
