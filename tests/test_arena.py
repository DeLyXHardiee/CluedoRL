"""Headroom harness: one test strategy against a base field, seat rotated, win rate with CI."""
import math

from cluedo.core.arena import evaluate, format_table, headroom_table
from cluedo.core.bots import parse_bot

BASE = parse_bot("cand/smart/certain")


def test_evaluate_reports_win_rate_with_confidence_interval():
    r = evaluate(parse_bot("cand/smart/gamble2"), BASE, n_players=4, games=40, seed=1)
    assert r.games == 40 and 0 <= r.wins <= 40
    assert r.win_rate == r.wins / 40
    p = r.win_rate
    assert math.isclose(r.ci, 1.96 * math.sqrt(p * (1 - p) / 40))
    assert 1 <= r.mean_turns <= 400


def test_evaluate_is_reproducible_and_independent_of_worker_count():
    a = evaluate(parse_bot("focused/smart/certain"), BASE, 3, games=30, seed=7)
    b = evaluate(parse_bot("focused/smart/certain"), BASE, 3, games=30, seed=7)
    c = evaluate(parse_bot("focused/smart/certain"), BASE, 3, games=30, seed=7, workers=2)
    assert (a.wins, a.mean_turns) == (b.wins, b.mean_turns) == (c.wins, c.mean_turns)


def test_identical_strategies_share_the_table_equally():
    # 4 seats x the same Bot: seat rotation gives every Seat the same number of games, and
    # nobody can win more than once per game.
    r = evaluate(BASE, BASE, 4, games=80, seed=3)
    assert 0.1 <= r.win_rate <= 0.45  # chance is 0.25; ties of draws/truncation only lower it


def test_headroom_table_covers_variants_and_player_counts():
    rows = headroom_table(["random/smart/certain", "cand/smart/gamble3"], "cand/smart/certain",
                          player_counts=[3, 4], games=20, seed=1)
    assert [(r.variant, r.n_players) for r in rows] == [
        ("random/smart/certain", 3), ("random/smart/certain", 4),
        ("cand/smart/gamble3", 3), ("cand/smart/gamble3", 4)]
    md = format_table(rows, base="cand/smart/certain", games=20)
    assert "cand/smart/gamble3" in md and "| 3 players |" in md and "| 4 players |" in md
