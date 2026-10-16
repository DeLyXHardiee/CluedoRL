import csv
from collections import Counter

import torch

from cluedo.core.bots import parse_bot
from cluedo.rl.checkpoint import save_checkpoint
from cluedo.rl.config import TrainConfig
from cluedo.rl.eval_grid import (HELD_OUT, TRAINING_BOTS, EvalConfig, bot_entrant, checkpoint_entrant,
                                 run_grid, write_tables)
from cluedo.rl.ppo_stack import make_actor, make_critic

TWINS = {"A": bot_entrant("A", "cand/smart/certain"), "B": bot_entrant("B", "cand/smart/certain"),
         "C": bot_entrant("C", "random/smart/certain")}


def _cfg(**kw):
    base = dict(games=24, seeds=(1,), players=(3,), opponents={"cand": "cand/smart/certain"},
                mixed_bot=None, metric_games=0)
    base.update(kw)
    return EvalConfig(**base)


def test_opponent_catalogue_matches_spec_story_40():
    assert set(HELD_OUT) == {"bluff", "reader", "cand/gamble3", "focused/gamble3", "random/certain"}
    assert len(TRAINING_BOTS) == 4
    for spec in [*HELD_OUT.values(), *TRAINING_BOTS.values()]:
        parse_bot(spec)


def test_homogeneous_cells_one_row_per_condition_opponent_players_seed():
    t = run_grid(TWINS, _cfg(players=(3, 4), seeds=(1, 2)))
    keys = {(r["entrant"], r["field"], r["n_players"], r["seed"]) for r in t.cells}
    assert keys == {(e, "cand", n, s) for e in TWINS for n in (3, 4) for s in (1, 2)}
    for r in t.cells:
        assert r["games"] == 24 and 0 <= r["ci_lo"] <= r["win_rate"] <= r["ci_hi"] <= 1
        assert abs(r["chance"] - 1 / r["n_players"]) < 1e-12


def test_same_deal_seeds_and_rotation_for_every_condition():
    t = run_grid(TWINS, _cfg(players=(4,)))
    by = {}
    for r in t.games:
        by.setdefault(r["entrant"], []).append((r["game"], r["deal_seed"], r["seat"]))
    assert by["A"] == by["B"] == by["C"]
    assert [g[2] for g in by["A"]] == [g % 4 for g in range(24)]
    # identical strategies, identical deals and rng -> identical outcomes (pairing is real)
    wins = {e: [r["win"] for r in t.games if r["entrant"] == e] for e in "AB"}
    assert wins["A"] == wins["B"]


def test_seeds_are_kept_separate_and_deals_differ():
    t = run_grid(TWINS, _cfg(seeds=(1, 2)))
    deals = {s: {r["deal_seed"] for r in t.games if r["seed"] == s} for s in (1, 2)}
    assert deals[1].isdisjoint(deals[2])


def test_result_independent_of_worker_count():
    a = run_grid(TWINS, _cfg(), workers=1)
    b = run_grid(TWINS, _cfg(), workers=3)
    assert a.games == b.games and a.cells == b.cells


def test_paired_comparison_of_identical_entrants_is_null():
    t = run_grid(TWINS, _cfg())
    ab = next(c for c in t.comparisons if {c["a"], c["b"]} == {"A", "B"})
    assert ab["diff"] == 0 and ab["a_only"] == ab["b_only"] == 0 and ab["p"] == 1.0 and ab["p_holm"] == 1.0
    assert len(t.comparisons) == 3  # one per pair within the single cell
    for c in t.comparisons:
        assert c["p_holm"] >= c["p"]


def test_mixed_table_covers_all_seat_permutations_evenly():
    cfg = _cfg(games=48, mixed_players=4, mixed_bot="cand/smart/certain", opponents={}, players=())
    t = run_grid(TWINS, cfg)
    rows = [r for r in t.games if r["field"] == "mixed"]
    bot = "bot:cand/smart/certain"
    assert {r["entrant"] for r in rows} == {"A", "B", "C", bot}
    perms = {}
    for r in rows:
        perms.setdefault(r["game"], {})[r["entrant"]] = r["seat"]
    assert all(sorted(p.values()) == [0, 1, 2, 3] for p in perms.values())
    assert len({tuple(sorted(p.items())) for p in perms.values()}) == 24
    seats = Counter((r["entrant"], r["seat"]) for r in rows)
    assert set(seats.values()) == {12}
    for g in range(48):  # at most one winner per game
        assert sum(r["win"] for r in rows if r["game"] == g) <= 1
    assert {r["entrant"] for r in t.cells if r["field"] == "mixed"} == {"A", "B", "C", bot}


def test_checkpoint_entrants_play_and_metrics_are_reported(tmp_path):
    cfg = TrainConfig(hidden=16, depth=1)
    torch.manual_seed(0)
    for name, blind in (("fb", False), ("blind", True)):
        save_checkpoint(tmp_path / f"{name}.pt", make_actor(16, 1), make_critic(16, 1), cfg,
                        condition=name, learner_steps=0, blind=blind)
    ents = {n: checkpoint_entrant(n, tmp_path / f"{n}.pt") for n in ("fb", "blind")}
    t = run_grid(ents, _cfg(games=6, metric_games=3), workers=2)
    assert {r["entrant"] for r in t.cells} == {"fb", "blind"}
    for m in t.metrics:
        assert m["games_measured"] == 3 and m["suggestions"] > 0
        assert 0 <= m["bluff_rate"] <= 1 and "leakage" in m and "mean_info_gain" in m
    assert {(m["entrant"], m["field"]) for m in t.metrics} == {("fb", "cand"), ("blind", "cand")}


def test_write_tables_roundtrip(tmp_path):
    t = run_grid(TWINS, _cfg(games=6, metric_games=2))
    write_tables(t, tmp_path)
    for name in ("games", "cells", "comparisons", "metrics"):
        rows = list(csv.DictReader(open(tmp_path / f"{name}.csv")))
        assert len(rows) == len(getattr(t, name)) > 0
    cells = list(csv.DictReader(open(tmp_path / "cells.csv")))
    assert float(cells[0]["win_rate"]) == t.cells[0]["win_rate"]


def test_cli_writes_tables_for_checkpoints(tmp_path):
    from cluedo.rl.eval_grid import main
    cfg = TrainConfig(hidden=16, depth=1)
    args = []
    for name in ("fb", "sp", "blind"):
        save_checkpoint(tmp_path / f"{name}.pt", make_actor(16, 1), make_critic(16, 1), cfg,
                        condition=name, learner_steps=0, blind=name == "blind")
        args += ["--ckpt", f"{name}={tmp_path / (name + '.pt')}"]
    out = tmp_path / "out"
    main([*args, "--out", str(out), "--games", "24", "--players", "3", "--opponents", "cand/certain",
          "--seeds", "0", "1", "--metric-games", "2", "--workers", "2"])
    cells = list(csv.DictReader(open(out / "cells.csv")))
    assert {r["seed"] for r in cells} == {"0", "1"}
    assert {r["field"] for r in cells} == {"cand/certain", "mixed"}
    assert {r["entrant"] for r in cells if r["field"] == "mixed"} == {"fb", "sp", "blind", "bot:cand/smart/certain"}


def test_metrics_carry_accusation_candidate_distribution():
    t = run_grid(TWINS, _cfg(games=12, metric_games=12))
    for m in t.metrics:
        vals = [int(v) for v in m["accusation_candidates"].split()] if m["accusation_candidates"] else []
        assert len(vals) == m["accusations"] and all(v >= 1 for v in vals)
