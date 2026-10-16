"""Checkpoints: save a training snapshot, load it back as a frozen policy.

``load_checkpoint(path)`` returns a ``Checkpoint`` whose ``.actor`` has no gradients and whose
``.player(game, seat, rng)`` is a drop-in player (``act(game)``) for the arena, the evaluation
set, and ticket 10's Opponent sources (a Seat plays from its own Seat-relative observation).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from tensordict import TensorDict

from cluedo import contract as C
from cluedo.rl.config import TrainConfig
from cluedo.rl.encoding import SeatEncoder
from cluedo.rl.ppo_stack import make_actor, make_critic


def save_checkpoint(path, actor, critic, cfg: TrainConfig, condition: str, learner_steps: int,
                    blind: bool = False, extra: dict | None = None) -> None:
    cfg_d = asdict(cfg)
    cfg_d["eval_players"] = list(cfg.eval_players)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "cfg": cfg_d,
                "condition": condition, "blind": blind, "learner_steps": learner_steps,
                "extra": extra or {}}, path)


@dataclass
class Checkpoint:
    actor: object  # frozen ProbabilisticActor
    critic: object
    cfg: TrainConfig
    condition: str
    blind: bool
    learner_steps: int

    def player(self, game, seat: int, rng, greedy: bool = False) -> "PolicyPlayer":
        return PolicyPlayer(self.actor, game, seat, self.blind, greedy)


def load_checkpoint(path) -> Checkpoint:
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg_d = dict(d["cfg"])
    cfg_d["eval_players"] = tuple(cfg_d["eval_players"])
    cfg = TrainConfig(**cfg_d)
    actor, critic = make_actor(cfg.hidden, cfg.depth), make_critic(cfg.hidden, cfg.depth)
    actor.load_state_dict(d["actor"])
    critic.load_state_dict(d["critic"])
    for m in (actor, critic):
        m.eval().requires_grad_(False)
    return Checkpoint(actor, critic, cfg, d["condition"], d["blind"], d["learner_steps"])


class PolicyPlayer:
    """A policy playing one Seat: encodes that Seat's view, samples a legal action."""

    TOP_K = 5
    last_probs: list | None = None  # [(action tuple, probability)] of the latest act()

    def __init__(self, actor, game, seat: int, blind: bool, greedy: bool = False):
        from torchrl.modules import MLP
        self.actor, self.seat, self.greedy = actor, seat, greedy
        self.net = next(m for m in actor.modules() if isinstance(m, MLP))  # bypass TensorDict overhead
        self.enc = SeatEncoder(game, seat, blind=blind)

    @property
    def blind(self) -> bool:
        return self.enc.blind

    @torch.no_grad()
    def act(self, game):
        logits = self.net(torch.from_numpy(self.enc.observation()))
        logits = logits.masked_fill(~torch.from_numpy(self.enc.mask()), float("-inf"))
        p = torch.softmax(logits, -1)
        a = int(logits.argmax()) if self.greedy else int(torch.multinomial(p, 1))
        top = p.topk(min(self.TOP_K, p.numel()))  # for Game logs: top few legal actions and their probability
        self.last_probs = [(C.to_core_action(int(i)), float(v)) for v, i in zip(top.values, top.indices) if v > 0]
        return C.to_core_action(a)
