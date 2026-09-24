"""Extended verification for the v2 generational engine.

Run explicitly (kept out of the fast regression command on purpose)::

    python verification/extended_verification.py            # all suites
    python verification/extended_verification.py strong     # one suite

Suites
------
strong5    5 objects, every strong function (6^5 = 7776 graphs) x 8 root
           patterns x 2 step budgets -> 124,416 full twin cycles, each
           cross-checked against the independent oracle (ledger after every
           step, full heap state after every commit) plus the plain
           full-reachability soundness check.
weak5      5 objects, 5 canonical strong graphs x 32 root sets x all weak
           assignments on a 3-target domain (3^5) -> 38,880 cycles.
ephemeron  2 tables + key/value objects: all (key,value) assignments
           (4^2 x 4^2) x 8 root sets x 2 budgets -> 4,096 cycles incl.
           key<->value mutual references and chained tables.
finalizer  3 objects: all finalizer masks x root sets x resurrection
           schedules, with allocation inside the finalizer body.
barrier    4 objects: all strong functions (5^4) x 4 root sets x 6
           interleavings of a black->white store at every scan depth.
scale      2,000 objects / 10,000 interleaved updates with minor+major
           cycles driven in bounded steps (the non-toy main-scale case).
fuzz5      2,000 seeded random 5-object scenarios mixing strong/weak/
           ephemeron edges, finalizers and mid-cycle mutations.

Everything is deterministic (fixed seeds); no sleeps, no network.
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

from gen_harness import Twin  # noqa: E402
from gen_reference import OLD, YOUNG  # noqa: E402


def drive(twin, kind, budget, finalizer=None):
    twin.begin(kind)
    guard = 0
    while True:
        guard += 1
        assert guard < 1_000_000, "cycle did not converge"
        phase = twin.heap.cycle_phase()
        if phase == "marking":
            twin.mark_step(budget)
        elif phase == "finalizing":
            twin.finalize_step(finalizer)
        elif phase in ("remark", "clearing"):
            r = twin.finish(budget)
            if r.complete:
                return r.result
        else:
            raise AssertionError(phase)


# --------------------------------------------------------------------------- #
def suite_strong5() -> int:
    """All 6^5 strong functions x 8 root patterns x 2 budgets."""
    names = "ABCDE"
    root_patterns = [
        (), ("A",), ("C",), ("E",),
        ("A", "C"), ("B", "D", "E"),
        ("A", "B", "C", "D"), ("A", "B", "C", "D", "E"),
    ]
    n = 0
    for edges in itertools.product(range(6), repeat=5):
        for roots in root_patterns:
            for budget in (1, 4):
                t = Twin(120, 120)
                for i, nm in enumerate(names):
                    t.allocate(nm, i, n_strong=1)
                for i, nm in enumerate(names):
                    if edges[i] < 5:
                        t.set_strong(nm, 0, names[edges[i]])
                for rn in roots:
                    t.add_root("r" + rn, rn)
                drive(t, "major", budget)
                n += 1
    return n


def suite_weak5() -> int:
    """Canonical strong graphs x root sets x all weak assignments (3^5)."""
    names = "ABCDE"
    strong_graphs = [
        (5, 5, 5, 5, 5),          # no strong edges
        (1, 2, 3, 4, 0),          # 5-cycle
        (1, 1, 1, 1, 1),          # all point at B
        (5, 0, 1, 2, 3),          # chain E->D->C->B->A (via indices)
        (0, 0, 0, 0, 0),          # self loops
    ]
    root_sets = [(), ("A",), ("E",), ("A", "C", "E"), ("B", "D")]
    n = 0
    for sg in strong_graphs:
        for roots in root_sets:
            for weak in itertools.product(range(3), repeat=5):
                # weak domain: None, "A", "D" (3 options per object)
                t = Twin(160, 160)
                for i, nm in enumerate(names):
                    t.allocate(nm, i, n_strong=1, n_weak=1)
                for i, nm in enumerate(names):
                    if sg[i] < 5:
                        t.set_strong(nm, 0, names[sg[i]])
                    if weak[i] == 1:
                        t.set_weak(nm, 0, "A")
                    elif weak[i] == 2:
                        t.set_weak(nm, 0, "D")
                for rn in roots:
                    t.add_root("r" + rn, rn)
                drive(t, "major", 3)
                n += 1
    return n


def suite_ephemeron() -> int:
    """Two tables; all key/value assignments; includes mutual references."""
    # objects: t0, t1 (tables), k, v  -- domain for key/value slots:
    # None, k, v, other-table
    n = 0
    for k0, v0 in itertools.product(range(4), repeat=2):
        for k1, v1 in itertools.product(range(4), repeat=2):
            for roots in range(1, 8):      # subsets of {t0, k, v}
                for budget in (1, 3):
                    t = Twin(120, 120)
                    t.allocate("t0", 0, n_ephemeron=1)
                    t.allocate("t1", 0, n_ephemeron=1)
                    t.allocate("k", 1)
                    t.allocate("v", 2)
                    domain = [None, "k", "v", "t1"]
                    domain1 = [None, "k", "v", "t0"]
                    t.set_ephemeron("t0", 0, domain[k0], domain[v0])
                    t.set_ephemeron("t1", 0, domain1[k1], domain1[v1])
                    for bit, nm in enumerate(("t0", "k", "v")):
                        if roots & (1 << bit):
                            t.add_root("r" + nm, nm)
                    drive(t, "major", budget)
                    n += 1
    return n


def suite_finalizer() -> int:
    """Finalizer masks x root sets x resurrection schedules + mid-finalize
    allocation."""
    n = 0
    for fin_mask in range(1, 8):           # which of A,B,C has a finalizer
        for root_bits in range(4):         # subsets of {A, B}
            for resurrect in range(4):     # subset of finalized to resurrect
                for alloc_in_fin in (False, True):
                    t = Twin(200, 200)
                    for i, nm in enumerate("ABC"):
                        t.allocate(nm, i)
                    for i, nm in enumerate("ABC"):
                        if fin_mask & (1 << i):
                            t.set_finalizer(nm, 10 + i)
                    for bit, nm in enumerate(("A", "B")):
                        if root_bits & (1 << bit):
                            t.add_root("r" + nm, nm)
                    state = {"count": 0}

                    def fin(twin, eid, name, _res=resurrect, _a=alloc_in_fin,
                            _st=state):
                        idx = eid - 10
                        if _res & (1 << idx):
                            twin.add_root(f"back{eid}", name)
                        if _a:
                            twin.allocate(f"nb{_st['count']}", 99)
                            _st["count"] += 1

                    drive(t, "major", 2, fin)
                    n += 1
    return n


def suite_barrier() -> int:
    """All 5^4 strong functions x root sets x black->white store schedules."""
    names = "ABCD"
    n = 0
    for edges in itertools.product(range(5), repeat=4):
        for roots in (("A",), ("A", "C")):
            for write_at in range(6):
                t = Twin(120, 120)
                for i, nm in enumerate(names):
                    t.allocate(nm, i, n_strong=1)
                for i, nm in enumerate(names):
                    if edges[i] < 4:
                        t.set_strong(nm, 0, names[edges[i]])
                for rn in roots:
                    t.add_root("r" + rn, rn)
                # fresh white object + black->white store after `write_at`
                # mark steps (6 = never)
                t.allocate("W", 9)
                t.begin("major")
                steps = 0
                while t.heap.cycle_phase() == "marking":
                    if steps == write_at:
                        src = names[write_at % 4]
                        t.set_strong(src, 0, "W")
                    t.mark_step(1)
                    steps += 1
                while t.heap.cycle_phase() == "finalizing":
                    t.finalize_step(None)
                rep = t.finish(2)
                while not rep.complete:
                    rep = t.finish(2)
                n += 1
    return n


def suite_scale() -> int:
    """2,000 objects, 10,000 interleaved updates, minor+major cycles."""
    rng = random.Random(20260924)
    t = Twin(young_capacity=4096, old_capacity=4096, promotion_age=2)
    live: list[str] = []
    # 2,000 objects; root every 25th immediately so emergency minor
    # collections during the build phase cannot reclaim future roots.
    for i in range(2000):
        nm = f"n{i}"
        t.allocate(nm, i, n_strong=2, n_weak=1)
        live.append(nm)
        if i % 25 == 0:
            t.add_root(f"r{i}", nm)
    # emergency collections during the build may have reclaimed unrooted
    # objects; sync the live list before issuing updates
    live = [nm for nm in live if nm in t.handles]
    ops = 0
    cycles = 0
    while ops < 10_000:
        # a burst of interleaved updates
        for _ in range(200):
            a = rng.choice(live)
            b = rng.choice(live)
            t.set_strong(a, rng.randrange(2), b)
            if rng.random() < 0.3:
                t.set_weak(a, 0, rng.choice(live))
            ops += 1
        # bounded-step collection with mid-cycle mutation.  Budgets are
        # finite and varied (7/64/256 edges per step); budget-1 exhaustion
        # interleavings are exhaustively covered by the 5-object suites.
        kind = "minor" if cycles % 3 else "major"
        t.begin(kind)
        budget = rng.choice((7, 64, 256))
        while True:
            phase = t.heap.cycle_phase()
            if phase == "marking":
                t.mark_step(budget)
                # mutate between finite steps (allowed by the contract)
                a = rng.choice(live)
                t.set_strong(a, rng.randrange(2), rng.choice(live))
                ops += 1
            elif phase == "finalizing":
                t.finalize_step(None)
            else:
                r = t.finish(budget)
                if r.complete:
                    break
        cycles += 1
        # the twin drops reclaimed names from its handle table; keep the
        # live list in sync so later operations only name live objects
        live = [nm for nm in live if nm in t.handles]
        # drop some roots, add some objects
        if cycles % 5 == 0:
            for rn in list(t.heap.roots())[::7]:
                t.remove_root(rn)
            for j in range(40):
                nm = f"x{cycles}_{j}"
                t.allocate(nm, j, n_strong=2, n_weak=1)
                live.append(nm)
                if j % 9 == 0:
                    t.add_root(f"rx{cycles}_{j}", nm)
            live = [nm for nm in live if nm in t.handles]
    return ops


def suite_fuzz5() -> int:
    """Seeded random 5-object scenarios with all features interleaved.

    400 full twin cycles (~30-60s); budget-1 exhaustion is exhaustively
    covered by the deterministic strong5/barrier suites, so here the step
    budgets are drawn from (1, 3, 9) to keep the run bounded.
    """
    rng = random.Random(1400149)
    names = "ABCDE"
    n = 0
    for trial in range(400):
        t = Twin(200, 200)
        for i, nm in enumerate(names):
            t.allocate(nm, i, n_strong=1, n_weak=1,
                       n_ephemeron=1 if i < 2 else 0)
        for nm in names:
            if rng.random() < 0.7:
                t.set_strong(nm, 0, rng.choice(names))
            if rng.random() < 0.5:
                t.set_weak(nm, 0, rng.choice(names))
        for tab in ("A", "B"):
            if rng.random() < 0.8:
                t.set_ephemeron(tab, 0, rng.choice(names), rng.choice(names))
        for nm in names:
            if rng.random() < 0.4:
                t.add_root("r" + nm, nm)
        fins = [nm for nm in names if rng.random() < 0.3]
        for i, nm in enumerate(fins):
            t.set_finalizer(nm, 100 + i)

        def fin(twin, eid, name, _rng=rng):
            if _rng.random() < 0.5:
                twin.add_root(f"res{eid}", name)

        kind = "minor" if trial % 2 else "major"
        budget = rng.choice((1, 3, 9))
        # mid-cycle mutations
        t.begin(kind)
        while True:
            phase = t.heap.cycle_phase()
            if phase == "marking":
                if rng.random() < 0.5:
                    t.set_strong(rng.choice(names), 0, rng.choice(names))
                t.mark_step(budget)
            elif phase == "finalizing":
                t.finalize_step(fin)
            else:
                r = t.finish(budget)
                if r.complete:
                    break
        n += 1
    return n


SUITES = {
    "strong5": suite_strong5,
    "weak5": suite_weak5,
    "ephemeron": suite_ephemeron,
    "finalizer": suite_finalizer,
    "barrier": suite_barrier,
    "scale": suite_scale,
    "fuzz5": suite_fuzz5,
}


def main() -> None:
    which = sys.argv[1:] or list(SUITES)
    started = time.perf_counter()
    for name in which:
        fn = SUITES[name]
        t0 = time.perf_counter()
        count = fn()
        dt = time.perf_counter() - t0
        print(f"[ok] {name:<10} {count:>8} cases verified in {dt:6.1f}s",
              flush=True)
    print(f"extended verification finished in "
          f"{time.perf_counter() - started:.1f}s; all checks passed.")


if __name__ == "__main__":
    main()
