"""PPO building blocks shared by training and tests (TorchRL).

Actor: MLP -> logits -> MaskedCategorical (mask = "action_mask"). Critic: MLP -> state_value.
"""
from __future__ import annotations

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import MLP, MaskedCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from cluedo import contract as C


def make_actor(hidden: int = 256, depth: int = 2) -> ProbabilisticActor:
    net = MLP(in_features=C.OBS_DIM, out_features=C.N_ACTIONS, num_cells=[hidden] * depth, activation_class=nn.Tanh)
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
