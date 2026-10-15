"""Stub single-Learner-seat env with the FINAL specs (no real game logic).

Used to prove the PPO stack and enforce the contract before the real env exists.
Keys: observation (float32 [OBS_DIM]), action_mask (bool [N_ACTIONS]),
action (int64, Categorical N_ACTIONS), reward (float32 [1]), done/terminated/truncated (bool [1]).
"""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from cluedo import contract as C


class StubCluedoEnv(EnvBase):
    def __init__(self, turn_cap: int = 40, win_prob: float = 0.15, seed: int | None = None, device="cpu"):
        super().__init__(device=device, batch_size=[])
        self.turn_cap = turn_cap
        self.win_prob = win_prob
        self._rng = np.random.default_rng(seed)
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
        self._turn = 0
        self._phase = C.SUGGEST
        self._n = 4

    def _make_obs(self):
        obs = C.empty_obs()
        k = self._rng.random((C.N_OWNERS, C.N_CARDS))
        grid = np.zeros((C.N_OWNERS, C.N_CARDS, C.N_KNOW), dtype=np.float32)
        np.put_along_axis(grid, (k * C.N_KNOW).astype(int)[..., None], 1.0, axis=-1)
        C.set_slice(obs, "knowledge", grid)
        C.set_slice(obs, "turn", [self._turn / C.TURN_CAP])
        C.set_slice(obs, "hand_sizes", [(18 // self._n) / 6] * self._n + [0] * (C.MAX_SEATS - self._n))
        C.set_slice(obs, "name_freq", self._rng.random((C.MAX_SEATS, C.N_CARDS)) * 0.2)
        C.encode_n_players(obs, self._n)
        C.encode_phase(obs, self._phase)
        if self._phase == C.SUGGEST:
            mask = C.suggest_mask()
        elif self._phase == C.ACCUSE:
            idx = self._rng.choice(C.N_COMBOS, 5, replace=False)
            mask = C.accuse_mask([C.combo_cards(int(i)) for i in idx])
        else:
            mask = C.reveal_mask(self._rng.choice(C.N_CARDS, 2, replace=False))
        return torch.from_numpy(obs), torch.from_numpy(mask)

    def _fill(self, td: TensorDictBase, terminated: bool, truncated: bool) -> TensorDictBase:
        obs, mask = self._make_obs()
        td.set("observation", obs)
        td.set("action_mask", mask)
        t = torch.tensor([terminated], dtype=torch.bool)
        u = torch.tensor([truncated], dtype=torch.bool)
        td.set("terminated", t)
        td.set("truncated", u)
        td.set("done", t | u)
        return td

    def _reset(self, tensordict=None, **kwargs):
        self._turn = 0
        self._phase = C.SUGGEST
        self._n = int(self._rng.integers(C.MIN_PLAYERS, C.MAX_PLAYERS + 1))
        return self._fill(TensorDict({}, batch_size=[]), False, False)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        kind = C.decode_action(int(tensordict.get("action"))).kind
        reward, terminated, truncated = 0.0, False, False
        if self._phase == C.SUGGEST:
            self._phase = C.REVEAL if self._rng.random() < 0.3 else C.ACCUSE
        elif self._phase == C.REVEAL:
            self._phase = C.ACCUSE
        elif kind == "accuse":
            terminated = True
            reward = 1.0 if self._rng.random() < self.win_prob else 0.0
        else:
            self._turn += 1
            self._phase = C.SUGGEST
            truncated = self._turn >= self.turn_cap
        out = self._fill(TensorDict({}, batch_size=[]), terminated, truncated)
        out.set("reward", torch.tensor([reward], dtype=torch.float32))
        return out

    def _set_seed(self, seed):
        self._rng = np.random.default_rng(seed)
