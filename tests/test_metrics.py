"""Behaviour metrics on hand-built Game logs with known counts (seam: metrics from a Game log)."""
import math
import random

import pytest

from cluedo.core import Game, random_legal
from cluedo.core.metrics import Metrics, analyse, behaviour_metrics

P = ("pass",)


def script(hands, envelope, start, moves):
    """Game log from [(seat, action), ...]; asserts each move is made by the Seat that is due."""
    g = Game.from_deal(hands, envelope, start)
    for seat, action in moves:
        assert g.decision.seat == seat, (g.decision, seat, action)
        g.apply(action)
    return g.log()


# Deal A: seat 0 holds suspects 0,1 / weapons 6,7 / rooms 12,13; Envelope (5, 11, 20).
DEAL_A = ([{0, 1, 6, 7, 12, 13}, {2, 3, 8, 9, 14, 15}, {4, 10, 16, 17, 18, 19}], (5, 11, 20))
FILL_A = [(1, ("suggest", 2, 8, 14)), (1, P), (2, ("suggest", 4, 10, 16)), (2, P)]


def log_a():
    return script(*DEAL_A, 0, [
        (0, ("suggest", 1, 8, 20)), (0, P), *FILL_A,
        (0, ("suggest", 2, 8, 13)), (1, ("reveal", 8)), (0, P), *FILL_A,
        (0, ("suggest", 2, 9, 14)), (1, ("reveal", 2)), (0, P),
    ])


def test_bluff_and_probe_counts_from_known_log():
    m = behaviour_metrics(log_a(), seat=0)
    assert m.suggestions == 3
    assert m.bluffs == 2      # suspect 1 (turn 1) and room 13 (turn 4) are own cards, categories open
    assert m.anchors == 0
    assert m.probes == 1      # turn 4 names weapon 8, known to be in Seat 1's hand since turn 1
    assert m.bluff_rate == pytest.approx(2 / 3)


def test_information_gain_is_log2_candidates_before_over_after():
    m = behaviour_metrics(log_a(), seat=0)
    # Seat 0 knows 2 suspects/2 weapons/2 rooms are its own: 4*4*7 = 112 Candidates.
    # Turn 1 shows Seat 1 holds weapon 8: 4*3*7 = 84.
    assert m.info_gains[0] == pytest.approx(math.log2(112 / 84))
    assert len(m.info_gains) == 3


def test_reveal_reshow_rate_counts_choices_that_repeat_a_card_the_asker_saw():
    m = behaviour_metrics(log_a(), seat=1)
    # Seat 1 chose twice: 8 again for Seat 0 (already shown on turn 1), then 2 (new).
    assert m.reveal_choices == 2
    assert m.reveal_reshows == 1
    assert m.reshow_rate == 0.5
    assert behaviour_metrics(log_a(), seat=0).reveal_choices == 0


# Deal B: seat 0 holds all suspects but 5, so the Envelope suspect is solved from the start.
DEAL_B = ([{0, 1, 2, 3, 4, 12}, {6, 7, 8, 9, 10, 13}, {14, 15, 16, 17, 18, 19}], (5, 11, 20))
FILL_B = [(1, ("suggest", 0, 6, 12)), (0, ("reveal", 0)), (1, P), (2, ("suggest", 5, 11, 14)), (2, P)]


def log_b(final):
    return script(*DEAL_B, 0, [
        (0, ("suggest", 0, 6, 12)), (0, P), *FILL_B,
        (0, ("suggest", 1, 7, 14)), (0, P), *FILL_B,
        (0, ("suggest", 5, 6, 20)), (0, final),
    ])


def test_anchor_is_own_card_in_a_solved_category_and_not_a_bluff():
    m = behaviour_metrics(log_b(("accuse", 5, 11, 20)), seat=0)
    assert m.suggestions == 3
    assert m.anchors == 2     # suspects 0 and 1
    assert m.bluffs == 1      # only the first: room 12 is own, rooms still open
    assert m.probes == 1      # last: weapon 6 known to be in Seat 1's hand
    assert m.anchor_rate == pytest.approx(2 / 3)


def test_accusation_timing_records_candidates_left_and_accuracy():
    right = behaviour_metrics(log_b(("accuse", 5, 11, 20)), seat=0)
    assert right.accusations == 1 and right.accusations_correct == 1
    assert right.accusation_candidates == [32]   # weapons 8-11 (4) x rooms 13-20 (8)
    wrong = behaviour_metrics(script(*DEAL_B, 0, [
        (0, ("suggest", 0, 6, 12)), (0, ("accuse", 5, 7, 13))]), seat=0)
    assert wrong.accusations == 1 and wrong.accusations_correct == 0
    assert wrong.accusation_candidates == [40]   # weapons 7-11 (5) x rooms 13-20 (8)
    assert wrong.accusation_accuracy == 0.0 and right.accusation_accuracy == 1.0


def log_c():
    """Seat 0 (the Learner) answers other Seats' Suggestions."""
    return script(*DEAL_B, 1, [
        (1, ("suggest", 0, 6, 13)), (1, P),                       # Seat 0 auto-shows suspect 0 to Seat 1
        (2, ("suggest", 0, 7, 12)), (0, ("reveal", 0)), (2, P),   # choice: suspect 0 to Seat 2 (new to them)
        (0, ("suggest", 5, 11, 20)), (0, P),
        (1, ("suggest", 0, 8, 12)), (0, ("reveal", 0)), (1, P),   # choice: suspect 0 again to Seat 1
    ])


def test_leakage_counts_hand_cells_newly_resolved_in_opponents_engines():
    m = behaviour_metrics(log_c(), seat=0)
    # 1 (Seat 1 learns suspect 0) + 1 (Seat 2 learns suspect 0) + 0 (re-show to Seat 1 teaches nothing).
    assert m.leakage_reveal == 2
    assert m.leakage_suggest == 0
    assert m.leakage == 2
    assert m.reveal_choices == 2 and m.reveal_reshows == 1


def test_metrics_add_up_across_games():
    a, c = behaviour_metrics(log_a(), 0), behaviour_metrics(log_c(), 0)
    t = a + c
    assert t.suggestions == a.suggestions + c.suggestions
    assert t.info_gains == a.info_gains + c.info_gains
    assert t.leakage == c.leakage
    assert Metrics().bluff_rate == 0.0  # no division by zero on empty


def test_analyse_flags_each_suggestion_for_the_replay():
    recs = analyse(log_a(), seat=0)
    flags = {r["index"]: r["flags"] for r in recs if r["flags"]}
    assert flags[0] == ["bluff"]
    assert [sorted(f) for f in flags.values()].count(["bluff", "probe"]) == 1


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_metrics_run_on_random_games_and_every_seat(n):
    for seed in range(5):
        g = Game.new(n, seed)
        rng = random.Random(seed)
        while not g.done:
            g.apply(random_legal(g, rng))
        for seat in range(n):
            m = behaviour_metrics(g.log(), seat)
            assert all(x >= -1e-9 for x in m.info_gains)
            assert m.leakage >= 0
