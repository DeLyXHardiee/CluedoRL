"""Re-run the pilot's headroom table on the rules core.

    python -m cluedo.headroom --games 6000 --players 3 4 5 6 --workers 10 --out results/headroom.md
"""
from __future__ import annotations

import argparse
import os
import time

from .core.arena import format_table, headroom_table

# The pilot's variants (suggest/reveal/accuse).
VARIANTS = ["random/random/certain", "cand/random/certain", "focused/random/certain",
            "bluff/random/certain", "cand/smart/certain", "cand/random/gamble2",
            "cand/random/gamble3", "focused/smart/gamble2"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=6000)
    p.add_argument("--players", type=int, nargs="+", default=[3, 4, 5, 6])
    p.add_argument("--base", default="cand/random/certain")
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out")
    a = p.parse_args(argv)
    t0 = time.time()
    rows = headroom_table(a.variants, a.base, a.players, a.games, a.seed, a.workers)
    table = format_table(rows, a.base, a.games)
    print(table)
    print(f"\n({time.time() - t0:.0f}s)")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
