import json
import random
import subprocess
import sys

import pytest

from cluedo.core import Game, IllegalAction, random_legal

ENV = (0, 6, 12)
H0 = [1, 2, 3, 4, 5, 7]
H1 = [8, 9, 10, 11, 13, 14]
H2 = [15, 16, 17, 18, 19, 20]


def g3(start=0, **kw):
    return Game.from_deal([H0, H1, H2], ENV, start, **kw)


def test_deal_sizes_and_envelope():
    expected = {3: [6, 6, 6], 4: [5, 5, 4, 4], 5: [4, 4, 4, 3, 3], 6: [3] * 6}
    for n, sizes in expected.items():
        seen_starts = set()
        for seed in range(40):
            g = Game.new(n, seed)
            assert sorted(g.hand_sizes, reverse=True) == sizes
            cards = [c for h in g.hands for c in h] + list(g.envelope)
            assert sorted(cards) == list(range(21))
            seen_starts.add(g.decision.seat)
        assert len(seen_starts) == n  # starting Seat is random


def test_dealer_varies_hand_sizes():
    firsts = {tuple(Game.new(4, s).hand_sizes) for s in range(60)}
    assert len(firsts) > 1


def test_pending_decision_and_legal_actions():
    g = g3(start=1)
    assert (g.decision.seat, g.decision.phase) == (1, "suggest")
    assert len(g.legal_actions()) == 324
    g.apply(("suggest", 0, 6, 12))
    assert (g.decision.seat, g.decision.phase) == (1, "accuse")
    acts = g.legal_actions()
    assert ("pass",) in acts and len(acts) == 325
    g.apply(("pass",))
    assert (g.decision.seat, g.decision.phase) == (2, "suggest")


def test_illegal_actions_rejected():
    g = g3()
    for bad in [("pass",), ("accuse", 0, 6, 12), ("suggest", 0, 1, 12), ("reveal", 3)]:
        with pytest.raises(IllegalAction):
            g.apply(bad)


def test_first_seat_to_left_answers_and_passes_public():
    g = g3(start=0)
    g.apply(("suggest", 0, 6, 15))  # seat1 holds none, seat2 holds 15
    e = g.events[-1]
    assert e["passes"] == [1] and e["answerer"] == 2 and e["card"] == 15
    assert e["automatic"] is True
    # public view hides the card; passes visible
    pe = g.public_events()[-1]
    assert pe["card"] is None and pe["passes"] == [1]
    assert g.events_for(0)[-1]["card"] == 15 and g.events_for(2)[-1]["card"] == 15
    assert g.events_for(1)[-1]["card"] is None


def test_wraps_around_table_order():
    g = g3(start=2)
    g.apply(("suggest", 0, 6, 12))  # nobody holds any (envelope cards)
    e = g.events[-1]
    assert e["answerer"] is None and e["passes"] == [0, 1]
    g2 = g3(start=2)
    g2.apply(("suggest", 1, 6, 12))  # seat 0 holds 1
    assert g2.events[-1]["answerer"] == 0 and g2.events[-1]["passes"] == []


def test_suggester_does_not_answer_own_suggestion():
    g = g3(start=0)
    g.apply(("suggest", 1, 7, 12))  # only seat 0 (suggester) holds them
    assert g.events[-1]["answerer"] is None and g.events[-1]["passes"] == [1, 2]


def test_reveal_is_decision_with_two_matches():
    g = g3(start=0)
    g.apply(("suggest", 0, 8, 13))  # seat1 holds 8 and 13
    assert (g.decision.seat, g.decision.phase) == (1, "reveal")
    assert g.turn_seat == 0
    assert sorted(g.legal_actions()) == [("reveal", 8), ("reveal", 13)]
    with pytest.raises(IllegalAction):
        g.apply(("reveal", 9))
    g.apply(("reveal", 13))
    e = g.events[-1]
    assert e["answerer"] == 1 and e["card"] == 13 and e["automatic"] is False
    assert (g.decision.seat, g.decision.phase) == (0, "accuse")


def test_reveal_automatic_with_one_match():
    g = g3(start=0)
    g.apply(("suggest", 0, 8, 12))
    assert g.decision.phase == "accuse" and g.events[-1]["card"] == 8


def test_wrong_accusation_eliminates_but_still_answers():
    g = g3(start=0)
    g.apply(("suggest", 0, 6, 12))
    g.apply(("accuse", 1, 6, 12))
    assert g.eliminated == [True, False, False] and not g.done
    assert g.decision.seat == 1
    g.apply(("suggest", 1, 6, 12 + 0))  # seat 2 ... nobody
    g.apply(("pass",))
    g.apply(("suggest", 0, 7, 12))  # seat 2 names seat 0 card 7
    assert g.events[-1]["answerer"] == 0  # eliminated seat still answers
    g.apply(("pass",))
    assert g.decision.seat == 1  # elimination skipped in turn order


def test_correct_accusation_wins():
    g = g3()
    g.apply(("suggest", 0, 6, 12))
    g.apply(("accuse", *ENV))
    assert g.outcome == {"reason": "win", "winner": 0, "turns": 0, "truncated": False}
    assert g.decision is None and g.legal_actions() == []
    with pytest.raises(IllegalAction):
        g.apply(("pass",))


def test_last_seat_standing_has_no_automatic_win():
    g = g3(start=0)
    for _ in range(2):  # seats 0 and 1 accuse wrongly
        g.apply(("suggest", 0, 6, 12))
        g.apply(("accuse", 1, 6, 12))
        if g.eliminated[0] and not g.eliminated[1]:
            continue
    assert g.eliminated == [True, True, False] and not g.done
    assert g.decision.seat == 2
    for _ in range(5):  # seat 2 keeps playing alone
        g.apply(("suggest", 0, 7, 12))
        g.apply(("pass",))
        assert g.decision.seat == 2 and not g.done
    g.apply(("suggest", 0, 6, 12))
    g.apply(("accuse", *ENV))
    assert g.outcome["winner"] == 2


def test_everyone_eliminated_ends_without_winner():
    g = g3(start=0)
    for _ in range(3):
        g.apply(("suggest", 0, 6, 12))
        g.apply(("accuse", 1, 6, 12))
    assert g.outcome["reason"] == "all_eliminated" and g.outcome["winner"] is None


def test_turn_cap_truncates():
    g = g3(max_turns=400)
    for _ in range(399):
        g.apply(("suggest", 0, 6, 12))
        g.apply(("pass",))
    assert not g.done
    g.apply(("suggest", 0, 6, 12))
    g.apply(("pass",))
    assert g.outcome["truncated"] and g.outcome["winner"] is None and g.outcome["turns"] == 400


def play_random(g, rng):
    while not g.done:
        g.apply(random_legal(g, rng))
    return g


def test_seed_reproducibility():
    a = play_random(Game.new(4, 7), random.Random(1))
    b = play_random(Game.new(4, 7), random.Random(1))
    assert a.log() == b.log()
    assert Game.new(4, 7).log()["hands"] == Game.new(4, 7).log()["hands"]
    assert Game.new(4, 8).log()["hands"] != Game.new(4, 7).log()["hands"]


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_log_roundtrip_through_json(n):
    for seed in range(15):
        g = Game.new(n, seed)
        rng = random.Random(seed)
        # bias toward passing so games get long and use many Reveals
        while not g.done:
            acts = g.legal_actions()
            if g.decision.phase == "accuse" and rng.random() < 0.9:
                g.apply(("pass",))
            else:
                g.apply(rng.choice(acts))
        log = json.loads(json.dumps(g.log()))
        assert log["envelope"] == list(g.envelope) and log["hands"]
        r = Game.replay(log)
        assert r.log() == g.log()
        for s in range(n):
            assert r.events_for(s) == g.events_for(s)


def test_core_never_imports_torch():
    code = ("import sys, cluedo.core, cluedo.cli; "
            "assert not any(m == 'torch' or m.startswith('torch.') for m in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_human_plays_to_the_end():
    from cluedo.cli import play
    game = Game.new(3, 5)
    seat = 0
    script = []

    def inp(prompt):
        if prompt.startswith("suggest"):
            return "0 0 0"
        if prompt.startswith("accuse"):
            return "{} {} {}".format(*[c - o for c, o in zip(game.envelope, (0, 6, 12))])
        return str(sorted(game._reveal_options)[0])

    lines = []
    play(game, seat, random.Random(0), inp=inp, out=lines.append)
    assert game.done
    assert any("Game over" in line for line in lines)
    assert game.outcome["reason"] in ("win", "all_eliminated")
