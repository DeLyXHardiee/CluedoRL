"""Bots are tested through the actions they choose in the core (seam A)."""
import random

import pytest

from cluedo.core import Game
from cluedo.core.bots import parse_bot
from cluedo.core.deduction import Deduction

ENV = (0, 6, 12)
H0 = [1, 2, 3, 4, 5, 7]
H1 = [8, 9, 10, 11, 13, 14]
H2 = [15, 16, 17, 18, 19, 20]


def scripted():
    return Game.from_deal([H0, H1, H2], ENV, start_seat=0)


def drive(game, *actions):
    for a in actions:
        game.apply(a)


def play_with_bots(spec, n, seed, watch=None):
    """Seat `watch` (default all seats) plays `spec`; returns (game, accusations) where each
    accusation is (seat, cards, independent engine's candidate count just before it)."""
    game = Game.new(n, seed)
    rng = random.Random(seed)
    bot = parse_bot(spec)
    players = [bot.player(game, s, rng) for s in range(n)]
    observers = [Deduction.for_seat(game, s) for s in range(n)]
    accusations = []
    while not game.done:
        d = game.decision
        a = players[d.seat].act(game)
        if a[0] == "accuse":
            observers[d.seat].sync(game.events_for(d.seat))
            accusations.append((d.seat, a[1:], observers[d.seat].candidates()))
        game.apply(a)  # raises IllegalAction on anything illegal
    return game, accusations


@pytest.mark.parametrize("n", [3, 4, 5, 6])
@pytest.mark.parametrize("spec", ["random/smart/certain", "cand/smart/gamble2",
                                  "focused/random/gamble3", "bluff/smart/gamble3"])
def test_bots_play_only_legal_actions_and_accuse_only_candidates(spec, n):
    for seed in range(8):
        game, accusations = play_with_bots(spec, n, seed)
        assert game.done
        for _seat, cards, cands in accusations:
            assert cards in cands


def test_bots_finish_games_with_a_winner_mostly():
    wins = sum(play_with_bots("cand/smart/certain", 4, s)[0].outcome["winner"] is not None
               for s in range(20))
    assert wins >= 15


# ---- accusation rules -------------------------------------------------

def test_certain_accuses_once_the_envelope_is_solved():
    g = scripted()
    p = parse_bot("cand/smart/certain").player(g, 0, random.Random(0))
    drive(g, ("suggest", 0, 6, 12))  # seats 1 and 2 both pass: all three are in the Envelope
    assert p.act(g) == ("accuse", 0, 6, 12)


def test_certain_passes_while_candidates_remain():
    g = scripted()
    p = parse_bot("cand/smart/certain").player(g, 0, random.Random(0))
    drive(g, ("suggest", 0, 8, 12))  # seat 1 shows 8 (the only match)
    assert p.act(g) == ("pass",)


@pytest.mark.parametrize("spec,k", [("cand/smart/certain", 1), ("cand/smart/gamble2", 2),
                                    ("cand/smart/gamble3", 3)])
def test_accuse_rules_gamble_exactly_up_to_their_candidate_limit(spec, k):
    seen_accuse = seen_pass = 0
    for seed in range(15):
        game = Game.new(4, seed)
        rng = random.Random(seed)
        bot = parse_bot(spec)
        players = [bot.player(game, s, rng) for s in range(4)]
        obs = [Deduction.for_seat(game, s) for s in range(4)]
        while not game.done:
            d = game.decision
            a = players[d.seat].act(game)
            if d.phase == "accuse":
                obs[d.seat].sync(game.events_for(d.seat))
                count = obs[d.seat].candidate_count()
                assert (a[0] == "accuse") == (count <= k)
                seen_accuse += a[0] == "accuse"
                seen_pass += a[0] == "pass"
            game.apply(a)
    assert seen_accuse and seen_pass


# ---- suggest rules ----------------------------------------------------

def suggestions(spec, seeds=60):
    out = []
    for seed in range(seeds):
        g = scripted()
        p = parse_bot(spec).player(g, 0, random.Random(seed))
        out.append(p.act(g))
    return out


def test_random_suggests_non_candidates_too():
    acts = suggestions("random/smart/certain")
    assert any(a[1] != 0 for a in acts)  # suspect 0 is the only suspect Candidate
    assert any(a[2] == 7 for a in acts) or any(a[3] in range(12, 21) for a in acts)


def test_cand_names_only_candidates():
    for a in suggestions("cand/smart/certain"):
        assert a[0] == "suggest"
        assert a[1] == 0 and a[2] != 7  # suspect solved; weapon 7 is my own card


def test_focused_uses_own_card_for_a_solved_category():
    for a in suggestions("focused/smart/certain"):
        assert a[1] in H0  # suspect 0 is solved, so name one of my suspects (nobody can show it)
        assert a[2] in (6, 8, 9, 10, 11)


def test_bluff_sometimes_names_an_own_card_in_an_open_category():
    acts = suggestions("bluff/smart/certain", seeds=200)
    assert any(a[2] == 7 for a in acts)
    assert any(a[2] != 7 for a in acts)
    assert all(a[1] in H0 for a in acts)  # solved category always uses my own card


# ---- reveal rule ------------------------------------------------------

def to_second_question(first_reveal, second_reveal, seed):
    """Seat 1 (Bot) holds 8 and 13. Returns its Reveal when Seat 0 asks (0,8,13) a second time,
    after having been shown first_reveal by seat 1, with seat 2 shown second_reveal."""
    g = scripted()
    p1 = parse_bot("cand/smart/certain").player(g, 1, random.Random(seed))
    drive(g, ("suggest", 0, 8, 13), ("reveal", first_reveal), ("pass",),
          ("suggest", 0, 6, 12), ("pass",),
          ("suggest", 0, 8, 13), ("reveal", second_reveal), ("pass",),
          ("suggest", 0, 8, 13))
    assert g.decision.seat == 1 and g.decision.phase == "reveal"
    return p1.act(g)


def test_smart_reshows_a_card_the_asker_already_saw():
    for seed in range(10):
        assert to_second_question(8, 13, seed) == ("reveal", 8)  # seat 0 saw 8
        assert to_second_question(13, 8, seed) == ("reveal", 13)


def test_smart_otherwise_shows_the_most_seen_card():
    g = scripted()
    p1 = parse_bot("cand/smart/certain").player(g, 1, random.Random(0))
    drive(g, ("suggest", 0, 8, 13), ("reveal", 8), ("pass",),  # 8 seen by seat 0
          ("suggest", 0, 6, 12), ("pass",),
          ("suggest", 0, 8, 13))  # seat 2 has seen nothing: show the most-seen card
    assert p1.act(g) == ("reveal", 8)


def test_random_reveal_is_legal_and_varies():
    seen = set()
    for seed in range(40):
        g = scripted()
        p1 = parse_bot("cand/random/certain").player(g, 1, random.Random(seed))
        drive(g, ("suggest", 0, 8, 13))
        a = p1.act(g)
        assert a in g.legal_actions()
        seen.add(a)
    assert seen == {("reveal", 8), ("reveal", 13)}


def test_parse_bot_names_and_errors():
    assert parse_bot("focused/smart/gamble2").name == "focused/smart/gamble2"
    with pytest.raises(ValueError):
        parse_bot("nonsense/smart/certain")
    with pytest.raises(ValueError):
        parse_bot("cand/smart")
