"""Paper figures from evaluation results (ticket 14). Torch-free: reads the ticket 13 CSVs.

    python -m cluedo.figures --results results/eval --out figures/out

One PDF (vector) per entry of FIGURES; the value says whether it is sized for a single or a double
column. `figures/READING_GUIDE.md` explains how to read each one. Seeds are never pooled in the
win-rate figures (one marker per seed); the behaviour figures pool the measured games of all
cells/seeds of a Condition by summing counts.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cluedo.core.stats import wilson_ci  # noqa: E402

SINGLE_COLUMN_IN = 3.4
DOUBLE_COLUMN_IN = 7.0

FIGURES = {  # name -> column width
    "winrate": "double",
    "winrate_mixed": "single",
    "bluff_anchor": "double",
    "accusation_timing": "double",
    "gain_leakage": "single",
    "reshow": "single",
}

# Okabe-Ito (colour-blind safe); markers differ too so greyscale prints stay readable.
STYLE = {"ppo-sp": ("#0072B2", "o"), "ppo-sp-blind": ("#D55E00", "s"), "ppo-fb": ("#009E73", "^")}
_FALLBACK = [("#CC79A7", "D"), ("#E69F00", "v"), ("#56B4E9", "P"), ("#000000", "X")]
BOT_STYLE = ("#7f7f7f", "*")

_RC = {"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.5,
       "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42, "axes.linewidth": 0.6,
       "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
       "savefig.format": "pdf"}


# ------------------------------------------------------------------ data

def _read(results, name):
    p = Path(results) / f"{name}.csv"
    if not p.exists():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _style(entrant, order):
    if entrant.startswith("bot:"):
        return BOT_STYLE
    if entrant in STYLE:
        return STYLE[entrant]
    extra = [e for e in order if e not in STYLE and not e.startswith("bot:")]
    return _FALLBACK[extra.index(entrant) % len(_FALLBACK)]


def _conditions(rows):
    seen = sorted({r["entrant"] for r in rows if not r["entrant"].startswith("bot:")},
                  key=lambda e: (list(STYLE).index(e) if e in STYLE else 99, e))
    return seen


def _pooled(rows, num, den):
    """{entrant: (rate, lo, hi, n)} pooling counts over rows."""
    tot = defaultdict(lambda: [0, 0])
    for r in rows:
        tot[r["entrant"]][0] += float(r[num])
        tot[r["entrant"]][1] += float(r[den])
    out = {}
    for e, (k, n) in tot.items():
        lo, hi = wilson_ci(int(k), int(n)) if n else (0.0, 0.0)
        out[e] = (k / n if n else 0.0, lo, hi, int(n))
    return out


def _err(y, lo, hi):
    return [[max(0.0, y - lo)], [max(0.0, hi - y)]]


def _empty(ax, msg="no data in this result set"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _fig(kind, rows=1, cols=1, height=2.4, **kw):
    w = DOUBLE_COLUMN_IN if kind == "double" else SINGLE_COLUMN_IN
    return plt.subplots(rows, cols, figsize=(w, height), layout="constrained", **kw)


def _save(fig, out, name):
    path = Path(out) / f"{name}.pdf"
    fig.savefig(path)  # no bbox_inches="tight": keeps the exact column width
    plt.close(fig)
    return path


# ------------------------------------------------------------------ figures

def _truncation(rows):
    """Share of games that hit the Turn cap over `rows` (cells.csv rows); 0 when the column is absent."""
    games = sum(float(r["games"]) for r in rows)
    return sum(float(r.get("truncated") or 0) for r in rows) / games if games else 0.0


def _warn_truncation(ax, rows):
    rate = _truncation(rows)
    if rate > 0:
        ax.text(0.99, 0.98, f"\u26a0 {100 * rate:.1f}% truncated", transform=ax.transAxes, ha="right",
                va="top", fontsize=5.5, color="#D55E00")


def fig_winrate(results, out):
    cells = [r for r in _read(results, "cells") if r["table"] == "homogeneous"]
    ns = sorted({int(r["n_players"]) for r in cells})
    fig, axes = _fig("double", 1, max(1, len(ns)), height=2.5, sharey=True, squeeze=False)
    axes = axes[0]
    if not cells:
        _empty(axes[0])
    conds = _conditions(cells)
    fields_ = list(dict.fromkeys(r["field"] for r in cells))
    seeds = sorted({int(r["seed"]) for r in cells})
    labelled = set()
    for ax, n in zip(axes, ns):
        sub = [r for r in cells if int(r["n_players"]) == n]
        width = 0.8 / max(1, len(conds))
        for ci, c in enumerate(conds):
            colour, marker = _style(c, conds)
            for si, s in enumerate(seeds):
                for r in sub:
                    if r["entrant"] != c or int(r["seed"]) != s:
                        continue
                    x = (fields_.index(r["field"]) - 0.4 + width * (ci + 0.5)
                         + width * 0.6 * (si - (len(seeds) - 1) / 2) / max(1, len(seeds)))
                    y, lo, hi = float(r["win_rate"]), float(r["ci_lo"]), float(r["ci_hi"])
                    ax.errorbar(x, y, yerr=_err(y, lo, hi), fmt=marker, color=colour, ms=2.5,
                                lw=0.6, capsize=0, label=c if c not in labelled else None)
                    labelled.add(c)
        ax.axhline(1 / n, color="k", ls=":", lw=0.7)
        ax.text(len(fields_) - 0.5, 1 / n, "chance", fontsize=5.5, va="bottom", ha="right")
        ax.set_xticks(range(len(fields_)))
        ax.set_xticklabels(fields_, rotation=45, ha="right")
        ax.set_title(f"{n} players")
        _warn_truncation(ax, sub)
        ax.set_ylim(0, min(1.0, 1.15 * max([float(r["ci_hi"]) for r in cells] + [1 / min(ns)])))
    axes[0].set_ylabel("win rate (Wilson 95% CI)")
    h, l = axes[0].get_legend_handles_labels()
    if h:
        d = dict(zip(l, h))
        axes[0].legend(d.values(), d.keys(), loc="upper left", frameon=False)
    return _save(fig, out, "winrate")


def fig_winrate_mixed(results, out):
    cells = [r for r in _read(results, "cells") if r["table"] == "mixed"]
    fig, ax = _fig("single", height=2.3)
    if not cells:
        _empty(ax)
        return _save(fig, out, "winrate_mixed")
    ents = sorted({r["entrant"] for r in cells},
                  key=lambda e: (e.startswith("bot:"), list(STYLE).index(e) if e in STYLE else 99, e))
    seeds = sorted({int(r["seed"]) for r in cells})
    for i, e in enumerate(ents):
        colour, marker = _style(e, ents)
        for si, s in enumerate(seeds):
            for r in cells:
                if r["entrant"] == e and int(r["seed"]) == s:
                    y, lo, hi = float(r["win_rate"]), float(r["ci_lo"]), float(r["ci_hi"])
                    ax.errorbar(i + 0.2 * (si - (len(seeds) - 1) / 2), y, yerr=_err(y, lo, hi),
                                fmt=marker, color=colour, ms=3, lw=0.6)
    ax.axhline(float(cells[0]["chance"]), color="k", ls=":", lw=0.7)
    _warn_truncation(ax, cells)
    ax.set_xticks(range(len(ents)))
    ax.set_xticklabels(ents, rotation=30, ha="right")
    ax.set_ylabel("win rate (Wilson 95% CI)")
    ax.set_ylim(0, 1)
    return _save(fig, out, "winrate_mixed")


def _diff(a, b):
    """Difference of two pooled proportions (a - b) with a normal 95% CI."""
    (pa, _, _, na), (pb, _, _, nb) = a, b
    if not na or not nb:
        return 0.0, 0.0, 0.0
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    d = pa - pb
    return d, d - 1.96 * se, d + 1.96 * se


def fig_bluff_anchor(results, out):
    m = _read(results, "metrics")
    m = [r for r in m if not r["entrant"].startswith("bot:") and r["table"] == "homogeneous"] or \
        [r for r in m if not r["entrant"].startswith("bot:")]
    fig, axes = _fig("double", 1, 3, height=2.3, gridspec_kw={"width_ratios": [1, 1, 0.8]})
    conds = _conditions(m)
    pooled = {k: _pooled(m, num, "suggestions") for k, num in (("Bluff", "bluffs"), ("Anchor", "anchors"))}
    for ax, k in zip(axes[:2], pooled):
        if not m:
            _empty(ax)
            continue
        for i, c in enumerate(conds):
            colour, marker = _style(c, conds)
            y, lo, hi, _ = pooled[k][c]
            ax.errorbar(i, y, yerr=_err(y, lo, hi), fmt=marker, color=colour, ms=3.5, lw=0.7)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=30, ha="right")
        ax.set_ylabel(f"{k} rate per Suggestion")
        ax.set_ylim(bottom=0)
    ax = axes[2]
    if "ppo-sp" in conds and "ppo-sp-blind" in conds:
        for i, k in enumerate(pooled):
            d, lo, hi = _diff(pooled[k]["ppo-sp"], pooled[k]["ppo-sp-blind"])
            ax.errorbar(i, d, yerr=_err(d, lo, hi), fmt="o", color="k", ms=3.5, lw=0.7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(pooled)))
        ax.set_xticklabels(list(pooled))
        ax.set_xlim(-0.6, len(pooled) - 0.4)
        ax.set_ylabel("PPO-SP minus PPO-SP-blind")
    else:
        _empty(ax, "needs ppo-sp and ppo-sp-blind")
    return _save(fig, out, "bluff_anchor")


def _ecdf(ax, values, colour, label):
    if not values:
        return
    v = sorted(values)
    ax.step(v, [(i + 1) / len(v) for i in range(len(v))], where="post", color=colour, label=label)


def fig_accusation_timing(results, out):
    m = [r for r in _read(results, "metrics") if not r["entrant"].startswith("bot:")]
    games = [r for r in _read(results, "games") if r["win"] == "1" and not r["entrant"].startswith("bot:")]
    fig, (a1, a2) = _fig("double", 1, 2, height=2.3)
    conds = _conditions(m or games)
    anyd = False
    for c in conds:
        colour, _ = _style(c, conds)
        cand = [int(v) for r in m if r["entrant"] == c for v in r.get("accusation_candidates", "").split()]
        _ecdf(a1, cand, colour, c)
        _ecdf(a2, [int(r["turns"]) for r in games if r["entrant"] == c], colour, c)
        anyd = anyd or bool(cand)
    if not anyd:
        a1.text(0.5, 0.5, "no Accusations recorded", transform=a1.transAxes, ha="center", fontsize=7)
    a1.set_xscale("log")
    a1.set_xlabel("Candidates left at Accusation (1 = solved)")
    a1.set_ylabel("cumulative fraction")
    a2.set_xlabel("game length of won games (turns)")
    a2.set_ylabel("cumulative fraction")
    h, l = a1.get_legend_handles_labels()
    if h:
        a1.legend(frameon=False, loc="lower right")
    return _save(fig, out, "accusation_timing")


def fig_gain_leakage(results, out):
    m = [r for r in _read(results, "metrics") if float(r["games_measured"] or 0) > 0]
    fig, ax = _fig("single", height=2.5)
    if not m:
        _empty(ax)
        return _save(fig, out, "gain_leakage")
    ents = sorted({r["entrant"] for r in m},
                  key=lambda e: (e.startswith("bot:"), list(STYLE).index(e) if e in STYLE else 99, e))
    for e in ents:
        colour, marker = _style(e, ents)
        pts = [(float(r["leakage"]) / float(r["games_measured"]), float(r["mean_info_gain"]))
               for r in m if r["entrant"] == e]
        ax.scatter(*zip(*pts), marker=marker, s=14, color=colour, alpha=0.8, linewidths=0,
                   label=e)
    ax.set_xlabel("Leakage (hand cells newly revealed to Opponents per game)")
    ax.set_ylabel("information gain per answered Suggestion (bits)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="best")
    return _save(fig, out, "gain_leakage")


def fig_reshow(results, out):
    m = [r for r in _read(results, "metrics") if not r["entrant"].startswith("bot:")]
    fig, ax = _fig("single", height=2.3)
    conds = _conditions(m)
    if not conds:
        _empty(ax)
        return _save(fig, out, "reshow")
    pooled = _pooled(m, "reveal_reshows", "reveal_choices")
    for i, c in enumerate(conds):
        colour, marker = _style(c, conds)
        y, lo, hi, n = pooled[c]
        ax.errorbar(i, y, yerr=_err(y, lo, hi), fmt=marker, color=colour, ms=3.5, lw=0.7)
        ax.text(i, 0.02, f"n={n}", ha="center", fontsize=5.5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=30, ha="right")
    ax.set_ylabel("Reveal re-show rate")
    ax.set_ylim(0, 1)
    return _save(fig, out, "reshow")


_BUILDERS = {"winrate": fig_winrate, "winrate_mixed": fig_winrate_mixed, "bluff_anchor": fig_bluff_anchor,
             "accusation_timing": fig_accusation_timing, "gain_leakage": fig_gain_leakage, "reshow": fig_reshow}


def build_all(results, out) -> list:
    """Render every figure of FIGURES from the CSVs in `results` into `out`; returns the PDF paths."""
    Path(out).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        return [_BUILDERS[name](results, out) for name in FIGURES]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render all paper figures from evaluation CSVs.")
    ap.add_argument("--results", required=True, help="directory with games/cells/metrics .csv (ticket 13)")
    ap.add_argument("--out", default="figures/out")
    a = ap.parse_args(argv)
    for p in build_all(a.results, a.out):
        print(p)


if __name__ == "__main__":
    main()
