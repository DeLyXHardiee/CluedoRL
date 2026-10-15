"""The real single-Learner-seat env (ADR 0001): Opponents act inside step().

Per episode: player count uniform 3-6, random Learner Seat, Game dealt from the env rng,
Opponents from an OpponentSource (default: the PPO-FB Bot field). Reward +1 if the Learner wins,
else 0. Terminates on game end or Learner elimination; truncates at the Turn cap.
`blind=True` zeroes Public history in the observation (PPO-SP-blind).
"""
from __future__ import annotations

import random

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from cluedo import contract as C
from cluedo.core.rules import Game
from cluedo.rl.encoding import SeatEncoder
from cluedo.rl.opponents import BotField, describe


class CluedoEnv(EnvBase):
    def __init__(self, opponents=None, blind: bool = False, turn_cap: int = C.TURN_CAP,
                 n_players: int | None = None, seed: int | None = None, device="cpu"):
        super().__init__(device=device, batch_size=[])
        self.opponents = opponents if opponents is not None else BotField()
        self.blind, self.turn_cap, self.fixed_n = blind, turn_cap, n_players
        self._rng = random.Random(seed)
        self.game: Game | None = None
        self.learner = 0
        self.opponent_names: dict[int, str] = {}
        self.observation_spec = Composite(
            observation=Unbounded(shape=(C.OBS_DIM,), dtype=torch.float32),
            action_mask=Binary(n=C.N_ACTIONS, shape=(C.N_ACTIONS,), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = Categorical(n=C.N_ACTIONS, shape=(), dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Composite(
            done=Binary(n=1, shape=(1,), dtype=torch.bool),
            terminated=Binary(n=1, shape=(1,), dtype=torch.bool),
            truncated=Binary(n=1, shape=(1,), dtype=torch.bool),
            shape=(),
        )

    # ---- internals ----------------------------------------------------
    def _advance_opponents(self):
        g = self.game
        while not g.done and g.decision.seat != self.learner:
            g.apply(self._players[g.decision.seat].act(g))

    def _fill(self, td: TensorDictBase, terminated: bool, truncated: bool) -> TensorDictBase:
        enc = self._encoder
        td.set("observation", torch.from_numpy(enc.observation()))
        td.set("action_mask", torch.from_numpy(enc.mask()))
        t = torch.tensor([terminated], dtype=torch.bool)
        u = torch.tensor([truncated], dtype=torch.bool)
        td.set("terminated", t)
        td.set("truncated", u)
        td.set("done", t | u)
        return td

    def _reset(self, tensordict=None, **kwargs):
        while True:  # an Opponent winning before the Learner's first decision is resampled
            n = self.fixed_n or self._rng.randint(C.MIN_PLAYERS, C.MAX_PLAYERS)
            self.game = Game.new(n, seed=self._rng.randrange(2**31), max_turns=self.turn_cap)
            self.learner = self._rng.randrange(n)
            self._players = self.opponents.seat_players(self.game, self.learner, self._rng)
            self._encoder = SeatEncoder(self.game, self.learner, blind=self.blind)
            self._advance_opponents()
            if not self.game.done:
                break
        self.opponent_names = {s: describe(p) for s, p in self._players.items()}
        return self._fill(TensorDict({}, batch_size=[]), False, False)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        g = self.game
        g.apply(self._to_core(int(tensordict.get("action"))))
        self._advance_opponents()
        out_reason = g.outcome["reason"] if g.done else None
        truncated = out_reason == "truncated"
        terminated = (g.done and not truncated) or g.eliminated[self.learner]
        won = g.done and g.outcome["winner"] == self.learner
        out = self._fill(TensorDict({}, batch_size=[]), bool(terminated), bool(truncated))
        out.set("reward", torch.tensor([1.0 if won else 0.0], dtype=torch.float32))
        return out

    @staticmethod
    def _to_core(a: int) -> tuple:
        act = C.decode_action(a)
        if act.kind == "pass":
            return ("pass",)
        if act.kind == "reveal":
            return ("reveal", act.cards[0])
        return (act.kind, *act.cards)

    def rand_action(self, tensordict=None):
        """Uniform over legal actions (used by rollout()/check_env_specs without a policy)."""
        if tensordict is None:
            tensordict = TensorDict({}, batch_size=[])
        mask = self._encoder.mask()
        tensordict.set("action", torch.tensor(int(self._rng.choice(np.flatnonzero(mask)))))
        return tensordict

    def _set_seed(self, seed):
        self._rng = random.Random(seed)
