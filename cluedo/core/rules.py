"""Torch-free rules core for card-only Cluedo (3-6 Seats).

Public interface (see Game):
    Game.new(n_players, seed)                   random deal, reproducible
    Game.from_deal(hands, envelope, start_seat) explicit (scripted) deal
    Game.replay(log)                            rebuild a game from a Game log
    game.decision                               pending Decision or None when over
    game.legal_actions()                        actions legal for the pending decision
    game.apply(action)                          apply an action, advance the game
    game.events / game.events_for(seat)         full / Seat-filtered event record
    game.public_events()                        Public history (shown card hidden)
    game.outcome, game.log()                    result and JSON-able Game log

Actions are tuples: ("suggest", s, w, r), ("accuse", s, w, r), ("pass",),
("reveal", card). Cards are ints (see cards.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product

from .cards import CAT_OF, CATS, NCARDS

SUGGEST, ACCUSE, REVEAL = "suggest", "accuse", "reveal"
MAX_TURNS = 400
TRIPLES = list(product(*CATS))  # 324 (suspect, weapon, room)


@dataclass(frozen=True)
class Decision:
    seat: int
    phase: str  # "suggest" | "accuse" | "reveal"


class IllegalAction(ValueError):
    pass


def deal_cards(n: int, rng: random.Random):
    """Return (hands, envelope, dealer). Deals the 18 non-Envelope cards round-robin
    starting from a random Seat."""
    if not 3 <= n <= 6:
        raise ValueError("player count must be 3..6")
    envelope = tuple(rng.choice(cat) for cat in CATS)
    rest = [c for c in range(NCARDS) if c not in envelope]
    rng.shuffle(rest)
    dealer = rng.randrange(n)
    hands = [[] for _ in range(n)]
    for i, c in enumerate(rest):
        hands[(dealer + i) % n].append(c)
    return [frozenset(h) for h in hands], envelope, dealer


class Game:
    def __init__(self, hands, envelope, start_seat, seed=None, max_turns=MAX_TURNS):
        self.hands = [frozenset(h) for h in hands]
        self.n = len(self.hands)
        if not 3 <= self.n <= 6:
            raise ValueError("player count must be 3..6")
        self.envelope = tuple(envelope)
        allcards = set(self.envelope).union(*self.hands)
        if (len(allcards) != NCARDS or sum(map(len, self.hands)) != 18
                or [CAT_OF[e] for e in self.envelope] != [0, 1, 2]):
            raise ValueError(f"deal must partition the {NCARDS} cards with one envelope card per category")
        self.seed = seed
        self.start_seat = start_seat
        self.max_turns = max_turns
        self.eliminated = [False] * self.n
        self.turns_taken = 0  # completed Turns
        self.events: list[dict] = []
        self.actions: list[dict] = []  # every applied action with its seat, for replay
        self.outcome: dict | None = None
        self._seat = start_seat  # Seat whose Turn it is
        self._phase = SUGGEST
        self._suggestion = None  # (cards, suggester)
        self._reveal_seat = None
        self._reveal_options: list[int] = []
        self._answer_queue: list[int] = []
        self._passes: list[int] = []

    # ---- construction -------------------------------------------------
    @classmethod
    def new(cls, n_players, seed, max_turns=MAX_TURNS):
        rng = random.Random(seed)
        hands, envelope, _dealer = deal_cards(n_players, rng)
        start = rng.randrange(n_players)
        return cls(hands, envelope, start, seed=seed, max_turns=max_turns)

    @classmethod
    def from_deal(cls, hands, envelope, start_seat=0, max_turns=MAX_TURNS):
        return cls(hands, envelope, start_seat, max_turns=max_turns)

    @classmethod
    def replay(cls, log):
        g = cls([frozenset(h) for h in log["hands"]], log["envelope"], log["start_seat"],
                seed=log.get("seed"), max_turns=log.get("max_turns", MAX_TURNS))
        for a in log["actions"]:
            g.apply(tuple(a["action"]))
        return g

    # ---- queries ------------------------------------------------------
    @property
    def decision(self) -> Decision | None:
        if self.outcome is not None:
            return None
        if self._phase == REVEAL:
            return Decision(self._reveal_seat, REVEAL)
        return Decision(self._seat, self._phase)

    @property
    def done(self) -> bool:
        return self.outcome is not None

    @property
    def hand_sizes(self):
        return [len(h) for h in self.hands]

    @property
    def turn_seat(self):
        """Seat whose Turn is in progress (differs from decision.seat during a Reveal)."""
        return self._seat

    def pending_suggestion(self):
        """During a reveal decision: (suggester, cards) being answered, else None."""
        if self._phase != REVEAL:
            return None
        return self._suggestion[1], self._suggestion[0]

    def legal_actions(self) -> list[tuple]:
        d = self.decision
        if d is None:
            return []
        if d.phase == SUGGEST:
            return [(SUGGEST, *t) for t in TRIPLES]
        if d.phase == ACCUSE:
            return [("pass",)] + [(ACCUSE, *t) for t in TRIPLES]
        return [(REVEAL, c) for c in self._reveal_options]

    # ---- transitions --------------------------------------------------
    def apply(self, action):
        action = tuple(action)
        d = self.decision
        if d is None:
            raise IllegalAction("game is over")
        if not self._is_legal(d, action):
            raise IllegalAction(f"{action} not legal for {d}")
        self.actions.append({"seat": d.seat, "action": list(action)})
        if d.phase == SUGGEST:
            self._do_suggest(d.seat, action[1:])
        elif d.phase == REVEAL:
            self._answer(self._reveal_seat, action[1], automatic=False)
        elif action[0] == "pass":
            self.events.append({"type": "pass_accuse", "seat": d.seat})
            self._end_turn()
        else:
            self._do_accuse(d.seat, action[1:])

    def _is_legal(self, d, a):
        if d.phase == REVEAL:
            return len(a) == 2 and a[0] == REVEAL and a[1] in self._reveal_options
        if a == ("pass",):
            return d.phase == ACCUSE
        return (len(a) == 4 and a[0] == d.phase and all(
            isinstance(c, int) and c in CATS[k] for k, c in enumerate(a[1:])))

    def _do_suggest(self, seat, cards):
        self._suggestion = (tuple(cards), seat)
        self._passes = []
        self._answer_queue = [(seat + i) % self.n for i in range(1, self.n)]
        cards = set(cards)
        while self._answer_queue:
            other = self._answer_queue.pop(0)
            match = sorted(self.hands[other] & cards)
            if not match:
                self._passes.append(other)
            elif len(match) == 1:
                self._answer(other, match[0], automatic=True)
                return
            else:
                self._reveal_seat, self._reveal_options = other, match
                self._phase = REVEAL
                return
        self._answer(None, None, automatic=True)

    def _answer(self, answerer, card, automatic):
        cards, seat = self._suggestion
        self.events.append({"type": "suggestion", "seat": seat, "cards": list(cards),
                            "passes": list(self._passes), "answerer": answerer,
                            "card": card, "automatic": automatic})
        self._suggestion = None
        self._reveal_seat, self._reveal_options = None, []
        self._phase = ACCUSE

    def _do_accuse(self, seat, cards):
        correct = tuple(cards) == self.envelope
        self.events.append({"type": "accusation", "seat": seat, "cards": list(cards),
                            "correct": correct})
        if correct:
            self._finish("win", winner=seat)
            return
        self.eliminated[seat] = True
        if all(self.eliminated):
            self._finish("all_eliminated")
            return
        self._end_turn()

    def _end_turn(self):
        self.turns_taken += 1
        if self.turns_taken >= self.max_turns:
            self._finish("truncated")
            return
        for i in range(1, self.n + 1):
            nxt = (self._seat + i) % self.n
            if not self.eliminated[nxt]:
                self._seat = nxt
                break
        self._phase = SUGGEST

    def _finish(self, reason, winner=None):
        self.outcome = {"reason": reason, "winner": winner, "turns": self.turns_taken,
                        "truncated": reason == "truncated"}

    # ---- records ------------------------------------------------------
    def events_for(self, seat):
        """Events as Seat `seat` observes them: the shown card is visible only to the
        suggester and the answerer."""
        return [{**e, "card": None} if e["type"] == "suggestion"
                and seat not in (e["seat"], e["answerer"]) else e for e in self.events]

    def public_events(self):
        return [{**e, "card": None} if e["type"] == "suggestion" else e for e in self.events]

    def log(self) -> dict:
        """JSON-able Game log with all hidden information."""
        return {"version": 1, "n_players": self.n, "seed": self.seed,
                "max_turns": self.max_turns, "start_seat": self.start_seat,
                "hands": [sorted(h) for h in self.hands], "envelope": list(self.envelope),
                "actions": [dict(a) for a in self.actions], "events": list(self.events),
                "outcome": self.outcome}
