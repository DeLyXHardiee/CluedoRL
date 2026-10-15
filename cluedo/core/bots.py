"""Hand-crafted Bots (ported from pilot/cluedo_pilot.py) on the rules core.

A Bot is a strategy spec "suggest/reveal/accuse", e.g. ``parse_bot("focused/smart/gamble2")``.
It is stateless; per game and Seat it makes a ``BotPlayer`` holding that Seat's Deduction engine:

    player = bot.player(game, seat, rng)
    action = player.act(game)       # for the pending decision of `seat` (any phase)

Rules are plain functions taking a ``SeatView`` and live in the SUGGEST / REVEAL / ACCUSE
registries, so new Bots (e.g. `reader`) just register more rules:

    suggest(view)                    -> (suspect, weapon, room)
    reveal(view, matching, asker)    -> card
    accuse(view)                     -> (suspect, weapon, room) or None (pass)

Accusations are only ever Candidates of the Seat's own engine.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .cards import CATS
from .deduction import Deduction


class SeatView:
    """What a Bot rule may look at: its Deduction engine (synced), hand, rng and the Game's
    Public history. Never hidden information."""

    def __init__(self, game, seat, rng):
        self.seat = seat
        self.n = game.n
        self.hand = game.hands[seat]
        self.rng = rng
        self.ded = Deduction.for_seat(game, seat)
        self.game = game  # only public queries (public_events, eliminated, ...) are allowed

    def public_events(self):
        return self.game.public_events()


# ---------------------------------------------------------------- suggest rules

def suggest_random(v):
    return tuple(v.rng.choice(cat) for cat in CATS)


def suggest_cand(v):
    return tuple(v.rng.choice(c) for c in v.ded.category_candidates())


def suggest_focused(v):
    """Unsolved category: an Envelope candidate. Solved: one of my own cards (nobody can show
    it), so every answer is about the open categories."""
    out = []
    for cat, cands in zip(CATS, v.ded.category_candidates()):
        if len(cands) == 1:
            mine = [c for c in cat if c in v.hand]
            out.append(v.rng.choice(mine) if mine else cands[0])
        else:
            out.append(v.rng.choice(cands))
    return tuple(out)


def suggest_bluff(v):
    """Like focused, but sometimes names an own card in an open category."""
    out = []
    cands_all = v.ded.category_candidates()
    open_cats = [k for k, c in enumerate(cands_all) if len(c) > 1]
    bluff_cat = v.rng.choice(open_cats) if len(open_cats) > 1 and v.rng.random() < 0.3 else None
    for k, (cat, cands) in enumerate(zip(CATS, cands_all)):
        mine = [c for c in cat if c in v.hand]
        if len(cands) == 1 or (k == bluff_cat and mine):
            out.append(v.rng.choice(mine) if mine else cands[0])
        else:
            out.append(v.rng.choice(cands))
    return tuple(out)


# ---------------------------------------------------------------- reveal rules

def reveal_random(v, matching, asker):
    return v.rng.choice(matching)


def reveal_smart(v, matching, asker):
    """Re-show a card the asker already saw; else the card most Seats have seen."""
    for c in matching:
        if asker in v.ded.shown_to(c):
            return c
    return max(matching, key=lambda c: (len(v.ded.shown_to(c)), v.rng.random()))


# ---------------------------------------------------------------- accuse rules

def make_accuse_gamble(limit):
    """Accuse a random Candidate once at most `limit` Candidates remain."""
    def accuse(v):
        cands = v.ded.candidates()
        return v.rng.choice(cands) if len(cands) <= limit else None
    return accuse


SUGGEST = {"random": suggest_random, "cand": suggest_cand,
           "focused": suggest_focused, "bluff": suggest_bluff}
REVEAL = {"random": reveal_random, "smart": reveal_smart}
ACCUSE = {"certain": make_accuse_gamble(1), "gamble2": make_accuse_gamble(2),
          "gamble3": make_accuse_gamble(3)}


@dataclass(frozen=True)
class Bot:
    suggest: str
    reveal: str = "smart"
    accuse: str = "certain"

    @property
    def name(self):
        return f"{self.suggest}/{self.reveal}/{self.accuse}"

    def player(self, game, seat: int, rng: random.Random) -> "BotPlayer":
        return BotPlayer(self, game, seat, rng)


class BotPlayer:
    def __init__(self, bot, game, seat, rng):
        self.bot, self.seat = bot, seat
        self.view = SeatView(game, seat, rng)
        self._suggest = SUGGEST[bot.suggest]
        self._reveal = REVEAL[bot.reveal]
        self._accuse = ACCUSE[bot.accuse]

    def act(self, game):
        d = game.decision
        if d is None or d.seat != self.seat:
            raise ValueError(f"no pending decision for seat {self.seat}")
        v = self.view
        v.ded.sync(game.events_for(self.seat))
        if d.phase == "suggest":
            return ("suggest", *self._suggest(v))
        if d.phase == "accuse":
            cards = self._accuse(v)
            return ("pass",) if cards is None else ("accuse", *cards)
        asker, named = game.pending_suggestion()
        matching = sorted(c for (_k, c) in game.legal_actions())
        return ("reveal", self._reveal(v, matching, asker))


def parse_bot(spec: str) -> Bot:
    """'suggest/reveal/accuse', e.g. 'cand/smart/gamble2'."""
    parts = spec.split("/")
    if len(parts) != 3:
        raise ValueError(f"bot spec must be suggest/reveal/accuse, got {spec!r}")
    for part, table, kind in zip(parts, (SUGGEST, REVEAL, ACCUSE), ("suggest", "reveal", "accuse")):
        if part not in table:
            raise ValueError(f"unknown {kind} rule {part!r}; choose from {sorted(table)}")
    return Bot(*parts)
