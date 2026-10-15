"""Fixed evaluation set for monitoring training (spec story 51; full eval is a later ticket).

Homogeneous fields (n-1 copies of one Opponent type), policy Seat rotated each game, win rate vs
chance 1/n. ``EVAL_FIELDS`` is the fixed set; ``random_suggest_baseline`` is the reference for the
gate "PPO beats a random-suggesting Bot": a random-suggest Bot dropped into the cand/certain field.
"""
from __future__ import annotations

import random

from cluedo.core import Game, arena
from cluedo.core.bots import parse_bot
from cluedo.core.metrics import Metrics, behaviour_metrics
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


def logged_game(actor, blind: bool, base, n_players: int, seed: int, seat: int = 0) -> dict:
    """Play one game of the policy (at `seat`) against `base` Bots and return its Game log, with
    ``log["learner"] = seat`` and each of the policy's actions carrying ``"probs"``:
    the top-5 [[action, probability], ...] (the replay viewer shows them)."""
    base = parse_bot(base) if isinstance(base, str) else base
    game = Game.new(n_players, seed)
    rng = random.Random(seed + 1)
    players = [LivePolicy(actor, blind).player(game, s, rng) if s == seat else base.player(game, s, rng)
               for s in range(n_players)]
    probs = {}
    while not game.done:
        s = game.decision.seat
        action = players[s].act(game)
        if s == seat:
            probs[len(game.actions)] = [[list(a), p] for a, p in players[s].last_probs]
        game.apply(action)
    log = game.log()
    log["learner"] = seat
    for i, p in probs.items():
        log["actions"][i]["probs"] = p
    return log


def behaviour_rates(actor, blind: bool, players=(4,), games: int = 20, seed: int = 0,
                    base=REFERENCE_FIELD) -> dict:
    """{n_players: Metrics} (cluedo.core.metrics) summed over `games` logged games per player count,
    the policy rotating through the Seats against the `base` field."""
    out = {}
    for n in players:
        total = Metrics()
        for g in range(games):
            total = total + behaviour_metrics(logged_game(actor, blind, base, n, seed * 1_000_003 + g, g % n), g % n)
        out[n] = total
    return out


def random_suggest_baseline(players: int = 4, games: int = 200, seed: int = 0, workers: int = 1):
    """Win rate of a random-suggesting Bot in the cand/certain field (policy reference)."""
    return arena.evaluate(RANDOM_SUGGEST_BOT, REFERENCE_FIELD, players, games, seed, workers)
