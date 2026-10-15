import json
import random

import pytest

from cluedo.core import Game, random_legal
from cluedo.core.deduction import Deduction
from cluedo.replay import OMNISCIENT, build_replay, render_html, write_replay


def play(n, seed, max_turns=400):
    g = Game.new(n, seed, max_turns=max_turns)
    rng = random.Random(seed)
    while not g.done:
        g.apply(random_legal(g, rng))
    return g.log()


def test_one_step_per_decision_plus_initial_state():
    log = play(4, 1)
    data = build_replay(log)
    assert len(data["steps"]) == len(log["actions"]) + 1
    assert data["steps"][0]["action"] is None
    assert data["steps"][1]["action"] == log["actions"][0]["action"]
    assert data["steps"][1]["seat"] == log["actions"][0]["seat"]


def test_seat_views_match_deduction_engine():
    log = play(4, 2)
    data = build_replay(log)
    g = Game.replay(log)
    last = data["steps"][-1]
    for seat in range(4):
        ded = Deduction.for_seat(g, seat)
        ded.sync(g.events_for(seat))
        view = last["views"][str(seat)]
        assert view["candidate_count"] == ded.candidate_count()
        for owner in range(5):
            for card in range(21):
                st = ded.status(owner, card)
                assert view["grid"][owner][card] == (-1 if st is None else st)


def test_omniscient_view_shows_true_hands_and_envelope():
    log = play(3, 3)
    data = build_replay(log)
    grid = data["steps"][0]["views"][OMNISCIENT]["grid"]
    for seat, hand in enumerate(log["hands"]):
        assert [c for c in range(21) if grid[seat][c] == 1] == sorted(hand)
    assert [c for c in range(21) if grid[3][c] == 1] == sorted(log["envelope"])


def test_steps_carry_public_consequences_and_annotation_slot():
    log = play(5, 4)
    data = build_replay(log)
    texts = [t for s in data["steps"] for t in s["events"]]
    assert any("suggests" in t for t in texts)
    assert all(s["annotations"] == {} for s in data["steps"])


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_html_is_self_contained_for_all_player_counts(n, tmp_path):
    path = tmp_path / "r.html"
    write_replay(play(n, 10 + n), str(path))
    html = path.read_text()
    assert "<html" in html and "<script" in html
    assert "src=" not in html and "http" not in html.replace("</", "")
    assert json.dumps(n) in html


def test_truncated_game_replays():
    log = play(4, 5, max_turns=2)
    assert log["outcome"]["truncated"]
    data = build_replay(log)
    assert data["outcome"]["truncated"]
    assert "Candidates" in render_html(data)
