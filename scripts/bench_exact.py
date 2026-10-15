"""Benchmark + looseness measurement for the exact Candidate check (ticket 04, ADR 0002).

    python scripts/bench_exact.py bench --games 60 --ks 0 4 8 12 20 40
    python scripts/bench_exact.py looseness --games 150 --cap 120

Games: every Seat keeps a Deduction engine; Suggestions are uniform random, a Seat accuses
when its engine reports exactly one Candidate, else passes (the engine is queried at every
Accuse decision, which is the hot path of a training episode).
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cluedo.core import Game  # noqa: E402
from cluedo.core.deduction import Deduction  # noqa: E402


def play(n, seed, k, on_accuse=None):
    rng = random.Random(seed)
    g = Game.new(n, seed)
    engines = [Deduction.for_seat(g, s, k=k) for s in range(n)]
    while not g.done:
        d = g.decision
        e = engines[d.seat]
        e.sync(g.events_for(d.seat))
        if d.phase == "accuse":
            if on_accuse:
                on_accuse(e)
            cands = e.candidates()
            a = ("accuse", *cands[0]) if len(cands) == 1 else ("pass",)
        else:
            a = rng.choice(g.legal_actions())
        g.apply(a)
    return g


def bench(args):
    print(f"{'players':>7} " + " ".join(f"K={k:<6}" for k in args.ks))
    for n in (3, 4, 5, 6):
        row = []
        for k in args.ks:
            t = time.perf_counter()
            for seed in range(args.games):
                play(n, seed, k)
            row.append(args.games / (time.perf_counter() - t))
        print(f"{n:>7} " + " ".join(f"{r:<8.1f}" for r in row))


def looseness(args):
    for n in (3, 4, 5, 6):
        s = dict(pts=0, bad=0, extra=0, skipped=0, zpts=0, zbad=0, zextra=0)

        def on(e):
            loose = e.loose_count()
            if loose > args.cap:
                s["skipped"] += 1
                return
            exact = len(e.exact_candidates())
            s["pts"] += 1
            s["bad"] += exact < loose
            s["extra"] += loose - exact
            if 2 <= exact <= 5:
                s["zpts"] += 1
                s["zbad"] += exact < loose
                s["zextra"] += loose - exact

        for seed in range(args.games):
            play(n, seed, 0, on)
        p, z = max(s["pts"], 1), max(s["zpts"], 1)
        print(f"{n} players: {s['pts']} points (+{s['skipped']} with >{args.cap} loose, unmeasured); "
              f"impossible present {s['bad'] / p:.1%}, avg extra {s['extra'] / p:.2f} | "
              f"2-5 exact zone: {s['zpts']} points, impossible present {s['zbad'] / z:.1%}, "
              f"avg extra {s['zextra'] / z:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--games", type=int, default=60)
    b.add_argument("--ks", type=int, nargs="+", default=[0, 4, 8, 12, 20, 40])
    lo = sub.add_parser("looseness")
    lo.add_argument("--games", type=int, default=150)
    lo.add_argument("--cap", type=int, default=120)
    a = ap.parse_args()
    {"bench": bench, "looseness": looseness}[a.cmd](a)
