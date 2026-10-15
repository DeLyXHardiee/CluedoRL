"""One shared hyperparameter config + the Condition registry.

Every Condition (PPO-FB, PPO-SP, PPO-SP-blind) trains with the same ``TrainConfig`` so the
comparison is fair (spec story 35). A Condition differs only in *where Opponents come from*
(``make_opponents``) and the ``blind`` flag (Public history zeroed). Ticket 10 registers the
self-play Conditions in ``CONDITIONS``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from cluedo import contract as C
from cluedo.rl.opponents import BotField


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    # network (actor and critic share the shape)
    hidden: int = 256
    depth: int = 2
    pass_bias: float = 9.0  # initial pass-logit prior (cautious early game)
    # PPO (gamma = 1 + GAE: only the real race creates urgency, spec story 38)
    gamma: float = 1.0
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    # budget / collection: total_frames = Learner-step budget
    total_frames: int = 1_000_000
    frames_per_batch: int = 12_288
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)  # one env worker per core
    update_threads: int = 0  # torch threads for the PPO update in the trainer; 0 = physical-core guess (cpus // 2)
    turn_cap: int = C.TURN_CAP
    # monitoring
    checkpoint_every: int = 100_000  # Learner steps
    eval_every: int = 50_000  # Learner steps
    eval_games: int = 200  # per Opponent field
    eval_players: tuple = (4,)
    # self-play (PPO-SP / PPO-SP-blind only; ignored by PPO-FB)
    pool_size: int = 20
    pool_every: int = 50_000  # Learner steps between Checkpoints entering the pool
    latest_fraction: float = 0.5  # share of Opponent Seats playing the latest policy
    pool_dir: str = ""  # set by train() to <out>/pool


@dataclass(frozen=True)
class Condition:
    name: str
    blind: bool
    make_opponents: Callable[[TrainConfig], object]  # -> OpponentSource (picklable for workers)
    self_play: bool = False  # trainer publishes latest/pool Checkpoints to cfg.pool_dir


def _self_play(cfg: TrainConfig):
    from cluedo.rl.selfplay import SelfPlay  # lazy: selfplay -> checkpoint -> config
    return SelfPlay(cfg.pool_dir, cfg.latest_fraction)


CONDITIONS: dict[str, Condition] = {
    "ppo-fb": Condition("ppo-fb", blind=False, make_opponents=lambda cfg: BotField()),
    "ppo-sp": Condition("ppo-sp", blind=False, make_opponents=_self_play, self_play=True),
    "ppo-sp-blind": Condition("ppo-sp-blind", blind=True, make_opponents=_self_play, self_play=True),
}


def condition(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise ValueError(f"unknown condition {name!r}; known: {sorted(CONDITIONS)}") from None
