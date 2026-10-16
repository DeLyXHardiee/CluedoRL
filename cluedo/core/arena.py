"""Headroom harness: one test strategy vs a base field, seat rotated each game.

Works with anything that has ``.player(game, seat, rng)`` returning an object with
``.act(game)`` (a ``Bot``; ticket 07's reader; later a Checkpoint adapter), or a Bot spec string.

    r = evaluate(test, base, n_players=4, games=6000, seed=1, workers=8)
    r.win_rate, r.ci, r.mean_turns

Game ``g`` uses ``Game.new(n, seed * 1_000_003 + g)`` and a bot rng seeded from it, so results
do not depend on the worker count.
"""
from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .bots import Bot, parse_bot
from .rules import Game


@dataclass(frozen=True)
class Result:
    variant: str
    n_players: int
    games: int
    wins: int
    mean_turns: float

    @property
    def win_rate(self):
        return self.wins / self.games

    @property
    def ci(self):
        """Half-width of the normal 95% interval."""
        p = self.win_rate
        return 1.96 * (p * (1 - p) / self.games) ** 0.5


def _as_bot(x):
    return parse_bot(x) if isinstance(x, str) else x


def _name(x):
    return getattr(x, "name", str(x))


def play_game(game, players):
    """Play `game` to the end; players[seat].act(game) decides. Returns game.outcome."""
    while not game.done:
        game.apply(players[game.decision.seat].act(game))
    return game.outcome


def play_table(factories, gseed):
    """Play one game with `factories[seat].player(...)` on the deal `gseed` (bot rng seeded from it,
    exactly as `evaluate` does). Returns the finished Game."""
    game = Game.new(len(factories), gseed)
    rng = random.Random(gseed + 1)
    play_game(game, [f.player(game, s, rng) for s, f in enumerate(factories)])
    return game


def _run_chunk(args):
    test, base, n, start, stop, seed = args
    wins = turns = 0
    for g in range(start, stop):
        gseed = seed * 1_000_003 + g
        game = Game.new(n, gseed)
        rng = random.Random(gseed + 1)
        seat = g % n
        players = [(test if s == seat else base).player(game, s, rng) for s in range(n)]
        out = play_game(game, players)
        wins += out["winner"] == seat
        turns += out["turns"]
    return wins, turns


def evaluate(test, base, n_players, games, seed, workers=1):
    test, base = _as_bot(test), _as_bot(base)
    if workers <= 1:
        chunks = [(test, base, n_players, 0, games, seed)]
        parts = [_run_chunk(chunks[0])]
    else:
        step = -(-games // (workers * 4))
        chunks = [(test, base, n_players, a, min(a + step, games), seed)
                  for a in range(0, games, step)]
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_run_chunk, chunks))
    wins = sum(p[0] for p in parts)
    turns = sum(p[1] for p in parts)
    return Result(_name(test), n_players, games, wins, turns / games)


def headroom_table(variants, base, player_counts, games, seed, workers=1):
    """Rows (Result) for every variant x player count, variants outermost."""
    rows = []
    for vi, v in enumerate(variants):
        for n in player_counts:
            rows.append(evaluate(v, base, n, games, seed + vi, workers))
    return rows


def format_table(rows, base, games):
    counts = sorted({r.n_players for r in rows})
    variants = list(dict.fromkeys(r.variant for r in rows))
    by = {(r.variant, r.n_players): r for r in rows}
    lines = [f"Base field: `{base}` ({games} games per cell; win rate % +- 95% CI; "
             f"chance = 100/n). One test seat vs n-1 base seats, seat rotated.", "",
             "| variant | " + " | ".join(f"{n} players" for n in counts) + " |",
             "|---|" + "---|" * len(counts)]
    for v in variants:
        cells = []
        for n in counts:
            r = by.get((v, n))
            cells.append(f"{100 * r.win_rate:.1f} +- {100 * r.ci:.1f}" if r else "")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("| chance | " + " | ".join(f"{100 / n:.1f}" for n in counts) + " |")
    return "\n".join(lines)
