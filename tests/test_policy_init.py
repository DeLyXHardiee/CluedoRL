import torch
from tensordict import TensorDict

from cluedo import contract as C
from cluedo.rl.ppo_stack import make_actor


def _pass_prob(actor):
    """P(pass) in an Accuse phase with every one of the Candidates legal (start of a game)."""
    every = [(s, w, r) for s in range(6) for w in range(6, 12) for r in range(12, 21)]
    mask = C.accuse_mask(every)
    td = TensorDict({"observation": torch.zeros(1, C.OBS_DIM),
                     "action_mask": torch.from_numpy(mask).unsqueeze(0)}, [1])
    return float(actor.get_dist(td).probs[0, C.PASS_ACTION])


def test_default_init_accuses_almost_surely_with_324_candidates():
    torch.manual_seed(0)
    assert _pass_prob(make_actor(32, 1)) < 0.05


def test_pass_prior_makes_early_game_policy_cautious():
    torch.manual_seed(0)
    assert _pass_prob(make_actor(32, 1, pass_bias=9.0)) > 0.9
