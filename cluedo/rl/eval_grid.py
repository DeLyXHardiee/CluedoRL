"""Evaluation harness (ticket 13): frozen Checkpoints and Bots over the full grid, tidy tables out.

    conditions = {"ppo-fb": checkpoint_entrant("ppo-fb", "runs/fb/checkpoints/step_..pt"), ...}
    tables = run_grid(conditions, EvalConfig(games=3000, seeds=(0, 1, 2)), workers=12)
    write_tables(tables, "results/eval")        # games.csv cells.csv comparisons.csv metrics.csv bluff_anchor.csv

Two kinds of table:
- homogeneous fields: one model (Entrant) vs n-1 copies of one Opponent Bot, the model's Seat rotated
  (game g -> Seat g % n), for every Condition x Opponent x player count;
- mixed 4-player tables: every Condition plus one Bot at the same table, every Seat permutation
  (game g -> permutation g % 24), field == "mixed".
Game g of seed s is always the deal ``s * 1_000_003 + g`` (``arena`` convention), so every Condition
plays the same deals with the same rotation: comparisons are paired. Seeds stay separate rows.
Games are played by ``arena.play_table``; results do not depend on the worker count.
Schemas: see the module-level ``*_COLUMNS`` constants.

Truncation (spec story 10): a game that hits the Turn cap has no winner. It counts as a loss for every
Entrant, and is reported, not hidden: ``games.csv`` has a ``truncated`` flag per row, ``cells.csv`` the
``truncated`` count and ``truncation_rate`` per cell, ``metrics.csv`` the ``truncation_rate`` over the
measured games. A nonzero rate means win rates in that cell understate every Entrant.

Statistics. ``comparisons.csv`` has one row per pair of Entrants within a cell, on the games both played:
- mixed-table and homogeneous comparisons alike are McNemar's exact test on the discordant games, i.e.
  those where exactly one of the two Entrants won (``a_only``, ``b_only``); in a mixed table the two
  Entrants sit at the same table in the same game, so at most one of them can win;
- ``diff`` is (a_only - b_only) / games with a normal 95% interval;
- the Holm family for ``p_holm`` is ALL comparisons within one seed (every table, field, player count and
  pair), not per cell; seeds are separate families.

``bluff_anchor.csv`` (spec story 45): pooled Bluff and Anchor rate per Suggestion per Condition (homogeneous
cells, Bots excluded; Wilson CI) and, when both ``ppo-sp`` and ``ppo-sp-blind`` are present, their difference
with a normal 95% CI.
"""
from __future__ import annotations

import csv
import itertools
import os
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path

from cluedo.core import arena
from cluedo.core.bots import parse_bot
from cluedo.core.metrics import Metrics, behaviour_metrics
from cluedo.core.stats import holm, mcnemar_exact, wilson_ci

HELD_OUT = {"bluff": "bluff/smart/certain", "reader": "reader/smart/reader",
            "cand/gamble3": "cand/smart/gamble3", "focused/gamble3": "focused/smart/gamble3",
            "random/certain": "random/smart/certain"}
TRAINING_BOTS = {f"{s}/{a}": f"{s}/smart/{a}" for s in ("cand", "focused") for a in ("certain", "gamble2")}
OPPONENTS = {**TRAINING_BOTS, **HELD_OUT}  # name -> Bot spec; the `field` column holds the name

GAMES_COLUMNS = ["table", "field", "n_players", "seed", "game", "deal_seed", "entrant", "seat", "win", "turns",
                 "truncated"]
CELLS_COLUMNS = ["table", "field", "n_players", "seed", "entrant", "games", "wins", "win_rate",
                 "ci_lo", "ci_hi", "chance", "mean_turns", "truncated", "truncation_rate"]
BLUFF_ANCHOR_COLUMNS = ["measure", "entrant", "suggestions", "count", "rate", "ci_lo", "ci_hi"]
COMPARISON_COLUMNS = ["table", "field", "n_players", "seed", "a", "b", "games", "a_only", "b_only", "diff",
                      "diff_lo", "diff_hi", "p", "p_holm"]
_METRIC_COUNTS = [f.name for f in fields(Metrics) if f.name not in ("accusation_candidates", "info_gains")]
METRICS_COLUMNS = (["table", "field", "n_players", "seed", "entrant", "games_measured", "truncation_rate"] + _METRIC_COUNTS +
                   ["bluff_rate", "anchor_rate", "probe_rate", "accusation_accuracy",
                    "mean_accusation_candidates", "mean_info_gain", "reshow_rate", "leakage",
                    "accusation_candidates"])  # last: space-separated Candidates left at each Accusation


@dataclass(frozen=True)
class Entrant:
    """A table participant: a Bot spec or a frozen Checkpoint file. Picklable; loads lazily."""
    name: str
    bot: str | None = None
    checkpoint: str | None = None

    def player(self, game, seat, rng):
        if self.bot is not None:
            return parse_bot(self.bot).player(game, seat, rng)
        return _load(self.checkpoint).player(game, seat, rng)


def bot_entrant(name: str, spec: str) -> Entrant:
    parse_bot(spec)
    return Entrant(name, bot=spec)


def checkpoint_entrant(name: str, path) -> Entrant:
    return Entrant(name, checkpoint=str(path))


_CKPTS: dict = {}


def _load(path):
    if path not in _CKPTS:
        import torch
        from cluedo.rl.checkpoint import load_checkpoint
        torch.set_num_threads(1)
        _CKPTS[path] = load_checkpoint(path)
    return _CKPTS[path]


@dataclass(frozen=True)
class EvalConfig:
    games: int = 3000                     # games per cell per seed
    seeds: tuple = (0,)                   # deal-seed families, kept separate
    players: tuple = (3, 4, 5, 6)         # homogeneous player counts
    opponents: dict = field(default_factory=lambda: dict(OPPONENTS))  # name -> Bot spec
    mixed_bot: str | None = "cand/smart/certain"  # Bot at the mixed table (None = no mixed tables)
    mixed_players: int = 4                # must equal len(conditions) + 1
    metric_games: int = 200               # per cell, behaviour metrics from the first N games (0 = off)
    chunk: int = 250                      # games per parallel task


@dataclass
class Tables:
    games: list
    cells: list
    comparisons: list
    metrics: list
    bluff_anchor: list


# ------------------------------------------------------------------ playing

def _layout(table, entrants, n, g):
    """Entrants by Seat for game g and the Seats whose results are recorded."""
    if table == "homogeneous":
        focus, opp = entrants
        seat = g % n
        return [focus if s == seat else opp for s in range(n)], [seat]
    perms = list(itertools.permutations(range(n)))
    perm = perms[g % len(perms)]  # perm[i] = Seat of entrant i
    seats = [None] * n
    for e, s in zip(entrants, perm):
        seats[s] = e
    return seats, list(range(n))


def _run_task(task):
    table, fld, n, seed, start, stop, entrants, metric_games = task
    rows, metrics = [], {}
    import torch
    for g in range(start, stop):
        dseed = seed * 1_000_003 + g
        seats, recorded = _layout(table, entrants, n, g)
        torch.manual_seed(dseed)  # checkpoint sampling is reproducible per game
        game = arena.play_table(seats, dseed)
        out = game.outcome
        log = game.log() if g < metric_games else None
        for s in recorded:
            rows.append({"table": table, "field": fld, "n_players": n, "seed": seed, "game": g,
                         "deal_seed": dseed, "entrant": seats[s].name, "seat": s,
                         "win": int(out["winner"] == s), "turns": out["turns"],
                         "truncated": int(out["truncated"])})
            if log is not None:
                m, c = metrics.get(seats[s].name, (Metrics(), 0))
                metrics[seats[s].name] = (m + behaviour_metrics(log, s), c + 1)
    return rows, metrics


def _tasks(conditions, cfg):
    names = list(conditions)
    jobs = []
    for seed in cfg.seeds:
        for n in cfg.players:
            for opp_name, spec in cfg.opponents.items():
                opp = bot_entrant(f"bot:{spec}", spec)
                for c in names:
                    jobs.append(("homogeneous", opp_name, n, seed, (conditions[c], opp)))
        if cfg.mixed_bot:
            if len(names) + 1 != cfg.mixed_players:
                raise ValueError(f"mixed table needs {cfg.mixed_players - 1} conditions, got {len(names)}")
            bot = bot_entrant(f"bot:{cfg.mixed_bot}", cfg.mixed_bot)
            jobs.append(("mixed", "mixed", cfg.mixed_players, seed, (*[conditions[c] for c in names], bot)))
    return [(t, f, n, seed, a, min(a + cfg.chunk, cfg.games), ents, cfg.metric_games)
            for t, f, n, seed, ents in jobs for a in range(0, cfg.games, cfg.chunk)]


# ------------------------------------------------------------------ statistics

def _cells(games):
    acc = defaultdict(lambda: [0, 0, 0, 0])
    for r in games:
        a = acc[(r["table"], r["field"], r["n_players"], r["seed"], r["entrant"])]
        a[0] += 1
        a[1] += r["win"]
        a[2] += r["turns"]
        a[3] += r["truncated"]
    out = []
    for (t, f, n, seed, e), (g, w, turns, tr) in acc.items():
        lo, hi = wilson_ci(w, g)
        out.append({"table": t, "field": f, "n_players": n, "seed": seed, "entrant": e, "games": g,
                    "wins": w, "win_rate": w / g, "ci_lo": lo, "ci_hi": hi, "chance": 1 / n,
                    "mean_turns": turns / g, "truncated": tr, "truncation_rate": tr / g})
    return out


def _comparisons(games):
    cell = defaultdict(lambda: defaultdict(dict))  # (table, field, n, seed) -> entrant -> game -> win
    for r in games:
        cell[(r["table"], r["field"], r["n_players"], r["seed"])][r["entrant"]][r["game"]] = r["win"]
    out = []
    for (t, f, n, seed), by in cell.items():
        for a, b in itertools.combinations(by, 2):
            ids = by[a].keys() & by[b].keys()
            ao = sum(by[a][i] > by[b][i] for i in ids)
            bo = sum(by[a][i] < by[b][i] for i in ids)
            N = len(ids)
            d = (ao - bo) / N
            se = max(ao + bo - (ao - bo) ** 2 / N, 0) ** 0.5 / N
            out.append({"table": t, "field": f, "n_players": n, "seed": seed, "a": a, "b": b, "games": N,
                        "a_only": ao, "b_only": bo, "diff": d, "diff_lo": d - 1.96 * se,
                        "diff_hi": d + 1.96 * se, "p": mcnemar_exact(ao, bo), "p_holm": None})
    for seed in {c["seed"] for c in out}:  # Holm family = all comparisons within one seed
        fam = [c for c in out if c["seed"] == seed]
        for c, p in zip(fam, holm([c["p"] for c in fam])):
            c["p_holm"] = p
    return out


def _metric_rows(parts, games, metric_games):
    trunc = defaultdict(lambda: [0, 0])  # measured games and truncated ones, per cell
    for r in games:
        if r["game"] < metric_games:
            k = trunc[(r["table"], r["field"], r["n_players"], r["seed"], r["entrant"])]
            k[0] += 1
            k[1] += r["truncated"]
    out = []
    for (t, f, n, seed, e), (m, c) in parts.items():
        measured, tr = trunc[(t, f, n, seed, e)]
        row = {"table": t, "field": f, "n_players": n, "seed": seed, "entrant": e, "games_measured": c,
               "truncation_rate": tr / measured if measured else 0.0}
        row.update({k: getattr(m, k) for k in _METRIC_COUNTS})
        row.update(m.summary())
        row["accusation_candidates"] = " ".join(str(c) for c in m.accusation_candidates)
        out.append(row)
    return out


def _bluff_anchor(metrics):
    """Pooled Bluff/Anchor rate per Condition (homogeneous cells, Bots excluded), plus SP minus SP-blind."""
    rows = [m for m in metrics if not m["entrant"].startswith("bot:")]
    rows = [m for m in rows if m["table"] == "homogeneous"] or rows
    pooled = {}
    for measure, col in (("bluff", "bluffs"), ("anchor", "anchors")):
        tot = defaultdict(lambda: [0, 0])
        for m in rows:
            tot[m["entrant"]][0] += m[col]
            tot[m["entrant"]][1] += m["suggestions"]
        pooled[measure] = {e: (k, n) for e, (k, n) in tot.items()}
    out = []
    for measure, by in pooled.items():
        for e, (k, n) in by.items():
            lo, hi = wilson_ci(k, n) if n else (0.0, 0.0)
            out.append({"measure": measure, "entrant": e, "suggestions": n, "count": k,
                        "rate": k / n if n else 0.0, "ci_lo": lo, "ci_hi": hi})
        if "ppo-sp" in by and "ppo-sp-blind" in by:
            (ka, na), (kb, nb) = by["ppo-sp"], by["ppo-sp-blind"]
            if na and nb:
                pa, pb = ka / na, kb / nb
                se = (pa * (1 - pa) / na + pb * (1 - pb) / nb) ** 0.5
                out.append({"measure": measure, "entrant": "ppo-sp minus ppo-sp-blind",
                            "suggestions": na + nb, "count": ka + kb, "rate": pa - pb,
                            "ci_lo": pa - pb - 1.96 * se, "ci_hi": pa - pb + 1.96 * se})
    return out


def run_grid(conditions: dict, cfg: EvalConfig = EvalConfig(), workers: int = 1) -> Tables:
    """Play the whole grid. `conditions` maps Condition name -> Entrant. Deterministic for any `workers`."""
    tasks = _tasks(conditions, cfg)
    results = arena.map_chunks(_run_task, tasks, workers)
    games, parts = [], {}
    for task, (rows, metrics) in zip(tasks, results):
        games += rows
        for e, (m, c) in metrics.items():
            key = (task[0], task[1], task[2], task[3], e)
            pm, pc = parts.get(key, (Metrics(), 0))
            parts[key] = (pm + m, pc + c)
    metrics = _metric_rows(parts, games, cfg.metric_games)
    return Tables(games, _cells(games), _comparisons(games), metrics, _bluff_anchor(metrics))


def write_tables(t: Tables, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, cols in (("games", GAMES_COLUMNS), ("cells", CELLS_COLUMNS),
                       ("comparisons", COMPARISON_COLUMNS), ("metrics", METRICS_COLUMNS),
                       ("bluff_anchor", BLUFF_ANCHOR_COLUMNS)):
        with open(out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(getattr(t, name))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate Conditions over the full grid (ticket 13).")
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH",
                    help="Condition name and checkpoint file; repeat per Condition")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--players", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--opponents", nargs="+", default=None, help="subset of opponent names")
    ap.add_argument("--mixed-bot", default="cand/smart/certain", help="'none' to skip mixed tables")
    ap.add_argument("--metric-games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    conditions = {}
    for item in a.ckpt:
        name, path = item.split("=", 1)
        conditions[name] = checkpoint_entrant(name, path)
    opps = {k: OPPONENTS[k] for k in (a.opponents or OPPONENTS)}
    mixed = None if a.mixed_bot == "none" else a.mixed_bot
    cfg = EvalConfig(a.games, tuple(a.seeds), tuple(a.players), opps, mixed, metric_games=a.metric_games)
    write_tables(run_grid(conditions, cfg, a.workers), a.out)


if __name__ == "__main__":
    main()
