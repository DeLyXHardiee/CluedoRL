"""PPO building blocks shared by training and tests (TorchRL).

Actor: MLP -> logits -> MaskedCategorical (mask = "action_mask"). Critic: MLP -> state_value.
"""
from __future__ import annotations

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import MLP, MaskedCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from cluedo import contract as C


def make_actor(hidden: int = 256, depth: int = 2, pass_bias: float = 0.0) -> ProbabilisticActor:
    """`pass_bias` raises the initial pass logit: at the start of a game all 324 Accusations are legal, so a
    uniform init would accuse (and die) on turn one and sparse reward could not teach otherwise."""
    net = MLP(in_features=C.OBS_DIM, out_features=C.N_ACTIONS, num_cells=[hidden] * depth, activation_class=nn.Tanh)
    with torch.no_grad():
        net[-1].bias[C.PASS_ACTION] += pass_bias
    logits = TensorDictModule(net, in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(
        module=logits,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )


def make_critic(hidden: int = 256, depth: int = 2) -> ValueOperator:
    net = MLP(in_features=C.OBS_DIM, out_features=1, num_cells=[hidden] * depth, activation_class=nn.Tanh)
    return ValueOperator(net, in_keys=["observation"])


def make_loss(actor, critic, gamma: float = 1.0, lmbda: float = 0.95, clip_epsilon: float = 0.2,
              entropy_coeff: float = 0.01):
    advantage = GAE(gamma=gamma, lmbda=lmbda, value_network=critic, average_gae=True)
    loss = ClipPPOLoss(actor_network=actor, critic_network=critic, clip_epsilon=clip_epsilon,
                       entropy_coeff=entropy_coeff, critic_coeff=0.5, normalize_advantage=True)
    return loss, advantage
