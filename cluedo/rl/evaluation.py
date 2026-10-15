"""Fixed evaluation set for monitoring training (spec story 51; full eval is a later ticket).

Homogeneous fields (n-1 copies of one Opponent type), policy Seat rotated each game, win rate vs
chance 1/n. ``EVAL_FIELDS`` is the fixed set; ``random_suggest_baseline`` is the reference for the
gate "PPO beats a random-suggesting Bot": a random-suggest Bot dropped into the cand/certain field.
"""
from __future__ import annotations

from cluedo.core import arena
from cluedo.core.bots import parse_bot
from cluedo.rl.checkpoint import PolicyPlayer

EVAL_FIELDS = ("random/smart/certain", "cand/smart/certain", "focused/smart/gamble2")
REFERENCE_FIELD = "cand/smart/certain"
RANDOM_SUGGEST_BOT = "random/smart/certain"


class LivePolicy:
    """Adapter: an actor (live or frozen) as a player factory for ``arena.evaluate``."""

    def __init__(self, actor, blind: bool, name: str = "ppo"):
        self.actor, self.blind, self.name = actor, blind, name

    def player(self, game, seat, rng):
        return PolicyPlayer(self.actor, game, seat, self.blind)


def evaluate_actor(actor, blind: bool, players=(4,), games: int = 200, seed: int = 0,
                   workers: int = 1, fields=EVAL_FIELDS) -> dict:
    """{(field, n_players): arena.Result} for the policy against each fixed field."""
    policy = LivePolicy(actor, blind)
    return {(f, n): arena.evaluate(policy, parse_bot(f), n, games, seed, workers)
            for f in fields for n in players}


def random_suggest_baseline(players: int = 4, games: int = 200, seed: int = 0, workers: int = 1):
    """Win rate of a random-suggesting Bot in the cand/certain field (policy reference)."""
    return arena.evaluate(RANDOM_SUGGEST_BOT, REFERENCE_FIELD, players, games, seed, workers)
