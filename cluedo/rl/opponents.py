"""Opponent sources for the env: who plays the non-Learner Seats of an episode.

The env asks its source, once per episode (after the Game is dealt), for a mapping
``{seat: player}`` where a player is anything with ``act(game) -> core action`` for its own Seat's
pending decision (any phase). ``BotField`` samples hand-crafted Bots per Seat (PPO-FB);
ticket 10 adds sources that wrap Checkpoint policies behind the same ``seat_players`` method
(``SeatEncoder`` in encoding.py gives such a policy its own Seat-relative observation and mask).
"""
from __future__ import annotations

from typing import Protocol

from cluedo.core.bots import parse_bot

PPO_FB_FIELD = tuple(f"{s}/smart/{a}" for s in ("cand", "focused") for a in ("certain", "gamble2"))


class OpponentSource(Protocol):
    def seat_players(self, game, learner: int, rng) -> dict: ...


class BotField:
    """Each Opponent Seat gets a Bot sampled uniformly from `specs` every episode."""

    def __init__(self, specs=PPO_FB_FIELD):
        self.bots = [parse_bot(s) if isinstance(s, str) else s for s in specs]

    def seat_players(self, game, learner, rng):
        return {seat: rng.choice(self.bots).player(game, seat, rng)
                for seat in range(game.n) if seat != learner}


def describe(player) -> str:
    bot = getattr(player, "bot", None)
    return bot.name if bot is not None else getattr(player, "name", type(player).__name__)
