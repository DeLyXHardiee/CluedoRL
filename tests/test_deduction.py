import random

import pytest

from cluedo.core import Game
from cluedo.core.deduction import Contradiction, Deduction

# Seat 0 holds suspects 1-5 and weapon 7; the Envelope is (0, 6, 12).
ENV = (0, 6, 12)
H0 = [1, 2, 3, 4, 5, 7]
H1 = [8, 9, 10, 11, 13, 14]
H2 = [15, 16, 17, 18, 19, 20]
SIZES = [6, 6, 6]


def sug(seat, cards, passes=(), answerer=None, card=None):
    return {"type": "suggestion", "seat": seat, "cards": list(cards), "passes": list(passes),
            "answerer": answerer, "card": card, "automatic": False}


def wrong_accusation(seat, cards):
    return {"type": "accusation", "seat": seat, "cards": list(cards), "correct": False}


def fresh(me=0, hand=H0):
    return Deduction(3, me, hand, SIZES)


def test_own_hand_is_known_and_owned_once():
    d = fresh()
    for c in H0:
        assert d.status(0, c) == 1
        assert [d.status(p, c) for p in (1, 2, 3)] == [0, 0, 0]  # seat 3 is the Envelope
    assert all(d.status(0, c) == 0 for c in range(21) if c not in H0)  # hand-size counting
    assert d.status(1, 8) is None
    assert d.category_candidates() == [[0], [6, 8, 9, 10, 11], list(range(12, 21))]
    assert d.candidate_count() == 45


def test_pass_information_finds_envelope_cards():
    d = fresh()
    d.observe(sug(0, (0, 6, 12), passes=[1, 2]))
    assert d.candidates() == [(0, 6, 12)]
    assert d.status(3, 12) == 1


def test_shown_card_is_owned_by_answerer():
    d = fresh()
    d.observe(sug(0, (8, 6, 12), passes=[], answerer=1, card=8))
    assert d.status(1, 8) == 1


def test_own_shown_log():
    e = fresh(me=1, hand=H1)
    e.observe(sug(0, (8, 6, 12), answerer=1, card=8))
    e.observe(sug(2, (8, 6, 12), answerer=1, card=8))
    assert e.shown_to(8) == {0, 2}
    assert e.shown_to(9) == set()


def test_unseen_answer_makes_clause_that_collapses_to_single_live_card():
    d = fresh()
    d.observe(sug(2, (13, 6, 12), answerer=1))  # Seat 1 showed Seat 2 something unseen by me
    assert d.open_clauses() == [(1, frozenset({13, 6, 12}))]
    assert d.open_clause_count() == 1
    d.observe(sug(0, (6, 12, 1), passes=[1]))  # Seat 1 has neither 6 nor 12
    assert d.status(1, 13) == 1
    assert d.open_clauses() == []


def test_wrong_accusation_is_envelope_clause():
    d = fresh()
    d.observe(wrong_accusation(1, (0, 6, 12)))
    assert (0, 6, 12) not in d.candidates()
    assert d.candidate_count() == 44
    assert d.envelope_clauses() == [(0, 6, 12)]
    # Once two of its cards are known Envelope cards, the third is excluded.
    d.observe(sug(0, (1, 6, 2), passes=[1, 2]))  # nobody holds 6 -> Envelope
    assert d.status(3, 6) == 1 and d.status(3, 0) == 1
    assert d.status(3, 12) == 0
    assert all(c[2] != 12 for c in d.candidates())
    assert d.envelope_clauses() == []


def test_contradiction_raises():
    d = fresh()
    d.observe(sug(0, (8, 6, 12), answerer=1, card=8))
    with pytest.raises(Contradiction):
        d.observe(sug(0, (8, 6, 12), passes=[1]))


def test_sync_consumes_only_new_events():
    d = fresh()
    ev = [wrong_accusation(1, (0, 6, 12))]
    d.sync(ev)
    d.sync(ev)  # re-feeding the same list must not double-count
    assert d.envelope_clauses() == [(0, 6, 12)]
    d.sync(ev + [wrong_accusation(2, (0, 6, 13))])
    assert len(d.envelope_clauses()) == 2


def play_and_check(n, seed):
    rng = random.Random(seed)
    g = Game.new(n, seed)
    engines = [Deduction.for_seat(g, s) for s in range(n)]
    owner = {c: s for s, h in enumerate(g.hands) for c in h}
    owner.update({c: n for c in g.envelope})

    def check():
        for s, e in enumerate(engines):
            e.sync(g.events_for(s))
            for p in range(n + 1):
                for c in range(21):
                    v = e.status(p, c)
                    if v is not None:
                        assert v == (1 if owner[c] == p else 0), (seed, s, p, c)
            assert tuple(g.envelope) in e.candidates(), (seed, s)

    check()
    while not g.done:
        legal = g.legal_actions()
        d = g.decision
        e = engines[d.seat]
        if d.phase == "accuse":
            cands = e.candidates()
            if len(cands) == 1 or (len(cands) <= 3 and rng.random() < 0.5):
                a = ("accuse", *rng.choice(cands))
            else:
                a = ("pass",)
        else:
            a = rng.choice(legal)
        g.apply(a)
        check()
    return g


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_soundness_across_seeded_games(n):
    for seed in range(40):
        play_and_check(n, seed)


def test_cli_shows_knowledge_grid_and_candidate_count():
    g = Game.new(3, 1)
    seat = g.decision.seat
    from cluedo.cli import knowledge_view
    view = knowledge_view(Deduction.for_seat(g, seat))
    assert "Candidates: 45" in view or "Candidates:" in view
    assert "Envelope" in view and "Seat 0" in view
