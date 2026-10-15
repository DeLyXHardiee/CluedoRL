import torch
from torchrl.envs.utils import check_env_specs

from cluedo import contract as C
from cluedo.rl.stub_env import StubCluedoEnv


def test_check_env_specs():
    check_env_specs(StubCluedoEnv(seed=0))


def test_rollout_invariants():
    env = StubCluedoEnv(seed=1, turn_cap=10)
    td = env.reset()
    ended = {"term": 0, "trunc": 0}
    for _ in range(600):
        mask = td["action_mask"]
        assert mask.dtype == torch.bool and mask.shape == (C.N_ACTIONS,)
        assert td["observation"].shape == (C.OBS_DIM,)
        phase = C.mask_phase(mask.numpy())
        assert C.obs_phase(td["observation"].numpy()) == phase
        if phase == C.ACCUSE and torch.rand(()) < 0.9:  # mostly pass so truncation is reached
            td["action"] = torch.tensor(C.PASS_ACTION)
        else:
            td["action"] = torch.multinomial(mask.float(), 1).squeeze(-1)
        td = env.step(td)["next"]
        assert td["done"].item() == (td["terminated"].item() or td["truncated"].item())
        if td["done"].item():
            ended["term"] += td["terminated"].item()
            ended["trunc"] += td["truncated"].item()
            assert td["reward"].item() in (0.0, 1.0)
            td = env.reset()
    assert ended["term"] > 0 and ended["trunc"] > 0
