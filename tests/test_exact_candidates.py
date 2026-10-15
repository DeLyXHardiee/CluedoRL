"""Exact Candidate check (ticket 04): agreement with brute-force enumeration of all deals."""
import itertools
import random

import pytest

from cluedo.core import Game
from cluedo.core.cards import CATS, NCARDS
from cluedo.core.deduction import Deduction

BIG = 10**9


def brute_force_envelopes(n, me, hand, sizes, events):
    """Envelopes (s, w, r) that appear in at least one deal consistent with `events`.

    Independent of the engine: enumerates whole deals and checks every event literally.
    """
    others = [p for p in range(n) if p != me]
    hand = set(hand)
    free = [c for c in range(NCARDS) if c not in hand]
    found = set()

    def consistent(hands):
        for e in events:
            if e["type"] == "suggestion":
                cards = set(e["cards"])
                for p in e["passes"]:
                    if hands[p] & cards:
                        return False
                a = e["answerer"]
                if a is not None:
                    if e["card"] is not None:
                        if e["card"] not in hands[a]:
                            return False
                    elif not hands[a] & cards:
                        return False
        return True

    for env in itertools.product(*CATS):
        if any(c in hand for c in env):
            continue
        # accusations constrain the Envelope directly
        if any((e["type"] == "accusation") and
               ((tuple(e["cards"]) == env) != e["correct"]) for e in events):
            continue
        rest = [c for c in free if c not in env]

        def rec(i, remaining, hands):
            if i == len(others) - 1:
                hands[others[i]] = set(remaining)
                if len(remaining) == sizes[others[i]] and consistent(hands):
                    return True
                return False
            for comb in itertools.combinations(sorted(remaining), sizes[others[i]]):
                hands[others[i]] = set(comb)
                if rec(i + 1, remaining - set(comb), hands):
                    return True
            return False

        hands = {me: hand}
        if rec(0, set(rest), hands):
            found.add(env)
    return found


def test_exact_agrees_with_brute_force_on_seeded_games():
    disagreements = 0
    checked = 0
    for seed in range(12):
        rng = random.Random(seed)
        g = Game.new(3, seed)
        engine = Deduction.for_seat(g, 0, k=BIG)
        loose = Deduction.for_seat(g, 0, k=0)
        step = 0
        while not g.done and step < 60:
            a = ("pass",) if g.decision.phase == "accuse" else rng.choice(g.legal_actions())
            g.apply(a)
            step += 1
            if step % 6:
                continue
            ev = g.events_for(0)
            engine.sync(ev)
            loose.sync(ev)
            truth = brute_force_envelopes(3, 0, g.hands[0], g.hand_sizes, ev)
            assert set(engine.candidates()) == truth, (seed, step)
            assert tuple(g.envelope) in truth
            assert engine.candidate_count() == len(truth)
            checked += 1
            disagreements += set(loose.candidates()) != truth
    assert checked >= 20
    assert disagreements > 0, "no sampled point where propagation alone was loose"


def test_k_zero_is_pure_propagation_and_default_refines_small_counts():
    g = Game.new(3, 3)
    pure = Deduction.for_seat(g, 0, k=0)
    assert pure.candidates() == pure.loose_candidates()


def test_is_candidate_matches_candidates_when_exact():
    rng = random.Random(7)
    g = Game.new(3, 7)
    e = Deduction.for_seat(g, 0, k=BIG)
    while not g.done:
        g.apply(("pass",) if g.decision.phase == "accuse" else rng.choice(g.legal_actions()))
        e.sync(g.events_for(0))
        cands = set(e.candidates())
        assert all(e.is_candidate(t) for t in cands)
        if len(cands) < 30:
            loose_only = set(e.loose_candidates()) - cands
            assert not any(e.is_candidate(t) for t in loose_only)


def test_exactness_never_drops_the_true_envelope_at_any_player_count():
    for n in (3, 4, 5, 6):
        for seed in range(15):
            rng = random.Random(seed)
            g = Game.new(n, seed)
            engines = [Deduction.for_seat(g, s, k=40) for s in range(n)]
            while not g.done:
                g.apply(("pass",) if g.decision.phase == "accuse" else rng.choice(g.legal_actions()))
                for s, e in enumerate(engines):
                    e.sync(g.events_for(s))
                    assert tuple(g.envelope) in e.candidates(), (n, seed, s)
