"""Evaluation harness (ticket 13): frozen Checkpoints and Bots over the full grid, tidy tables out.

    conditions = {"ppo-fb": checkpoint_entrant("ppo-fb", "runs/fb/checkpoints/step_..pt"), ...}
    tables = run_grid(conditions, EvalConfig(games=3000, seeds=(0, 1, 2)), workers=12)
    write_tables(tables, "results/eval")        # games.csv cells.csv comparisons.csv metrics.csv

Two kinds of table:
- homogeneous fields: one model (Entrant) vs n-1 copies of one Opponent Bot, the model's Seat rotated
  (game g -> Seat g % n), for every Condition x Opponent x player count;
- mixed 4-player tables: every Condition plus one Bot at the same table, every Seat permutation
  (game g -> permutation g % 24), field == "mixed".
Game g of seed s is always the deal ``s * 1_000_003 + g`` (``arena`` convention), so every Condition
plays the same deals with the same rotation: comparisons are paired. Seeds stay separate rows.
Games are played by ``arena.play_table``; results do not depend on the worker count.
Schemas: see the module-level ``*_COLUMNS`` constants.
"""
from __future__ import annotations

import csv
import itertools
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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

GAMES_COLUMNS = ["table", "field", "n_players", "seed", "game", "deal_seed", "entrant", "seat", "win", "turns"]
CELLS_COLUMNS = ["table", "field", "n_players", "seed", "entrant", "games", "wins", "win_rate",
                 "ci_lo", "ci_hi", "chance", "mean_turns"]
COMPARISON_COLUMNS = ["table", "field", "n_players", "seed", "a", "b", "games", "a_only", "b_only", "diff",
                      "diff_lo", "diff_hi", "p", "p_holm"]
_METRIC_COUNTS = [f.name for f in fields(Metrics) if f.name not in ("accusation_candidates", "info_gains")]
METRICS_COLUMNS = (["table", "field", "n_players", "seed", "entrant", "games_measured"] + _METRIC_COUNTS +
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
                         "win": int(out["winner"] == s), "turns": out["turns"]})
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
    acc = defaultdict(lambda: [0, 0, 0])
    for r in games:
        a = acc[(r["table"], r["field"], r["n_players"], r["seed"], r["entrant"])]
        a[0] += 1
        a[1] += r["win"]
        a[2] += r["turns"]
    out = []
    for (t, f, n, seed, e), (g, w, turns) in acc.items():
        lo, hi = wilson_ci(w, g)
        out.append({"table": t, "field": f, "n_players": n, "seed": seed, "entrant": e, "games": g,
                    "wins": w, "win_rate": w / g, "ci_lo": lo, "ci_hi": hi, "chance": 1 / n,
                    "mean_turns": turns / g})
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


def _metric_rows(parts):
    out = []
    for (t, f, n, seed, e), (m, c) in parts.items():
        row = {"table": t, "field": f, "n_players": n, "seed": seed, "entrant": e, "games_measured": c}
        row.update({k: getattr(m, k) for k in _METRIC_COUNTS})
        row.update(m.summary())
        row["accusation_candidates"] = " ".join(str(c) for c in m.accusation_candidates)
        out.append(row)
    return out


def run_grid(conditions: dict, cfg: EvalConfig = EvalConfig(), workers: int = 1) -> Tables:
    """Play the whole grid. `conditions` maps Condition name -> Entrant. Deterministic for any `workers`."""
    tasks = _tasks(conditions, cfg)
    if workers <= 1:
        results = [_run_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_run_task, tasks))
    games, parts = [], {}
    for task, (rows, metrics) in zip(tasks, results):
        games += rows
        for e, (m, c) in metrics.items():
            key = (task[0], task[1], task[2], task[3], e)
            pm, pc = parts.get(key, (Metrics(), 0))
            parts[key] = (pm + m, pc + c)
    return Tables(games, _cells(games), _comparisons(games), _metric_rows(parts))


def write_tables(t: Tables, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, cols in (("games", GAMES_COLUMNS), ("cells", CELLS_COLUMNS),
                       ("comparisons", COMPARISON_COLUMNS), ("metrics", METRICS_COLUMNS)):
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
