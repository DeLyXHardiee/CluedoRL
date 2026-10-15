"""The reader's gambling Accusation rule, tested on hand-built situations (rule seam)."""
import random

from cluedo.core.bots import ACCUSE


class FakeView:
    """Just enough of a SeatView for the accuse rule: Candidates and Public history."""

    def __init__(self, candidates, named, seat=0, by=1):
        self.seat, self.rng = seat, random.Random(0)
        self.ded = type("D", (), {"candidates": lambda s: candidates})()
        self._events = [self.event(by, c) for c in named]

    @staticmethod
    def event(seat, cards):
        return {"type": "suggestion", "seat": seat, "cards": list(cards), "answerer": None,
                "card": None, "passes": [], "automatic": False}

    def public_events(self):
        return self._events


def test_reader_gambles_on_the_most_named_candidate():
    cands = [(0, 6, 12), (0, 6, 13), (0, 8, 12)]
    named = [(0, 6, 13), (0, 6, 13), (0, 8, 12)]
    for _ in range(5):
        assert ACCUSE["reader"](FakeView(cands, named)) == (0, 6, 13)


def test_reader_does_not_gamble_above_three_candidates():
    cands = [(0, 6, 12), (0, 6, 13), (0, 8, 12), (0, 8, 13)]
    assert ACCUSE["reader"](FakeView(cands, [(0, 6, 12)])) is None


def test_reader_accuses_the_single_candidate_whatever_was_named():
    assert ACCUSE["reader"](FakeView([(0, 6, 12)], [(1, 7, 13)])) == (0, 6, 12)


def test_reader_ignores_its_own_suggestions_in_accusation_weights():
    cands = [(0, 6, 12), (0, 6, 13)]
    v = FakeView(cands, [(0, 6, 13)] * 3, seat=1, by=1)  # all named by the reader itself
    v._events.append(v.event(2, (0, 6, 12)))
    assert ACCUSE["reader"](v) == (0, 6, 12)
