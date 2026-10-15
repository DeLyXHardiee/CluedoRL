import torch
from torchrl.collectors import Collector

from cluedo.rl.ppo_stack import make_actor, make_critic, make_loss
from cluedo.rl.env import CluedoEnv
from cluedo.rl.stub_env import StubCluedoEnv


def test_ppo_smoke_2k_steps():
    run_smoke(lambda: StubCluedoEnv(seed=0))


def test_ppo_smoke_real_env():
    run_smoke(lambda: CluedoEnv(seed=0))


def run_smoke(make_env):
    torch.manual_seed(0)
    actor, critic = make_actor(64, 1), make_critic(64, 1)
    loss_mod, advantage = make_loss(actor, critic, gamma=1.0)
    optim = torch.optim.Adam(loss_mod.parameters(), lr=3e-4)
    collector = Collector(
        make_env, actor, frames_per_batch=500, total_frames=2000, device="cpu"
    )
    n_batches, steps = 0, 0
    for batch in collector:
        steps += batch.numel()
        mask, act = batch["action_mask"], batch["action"]
        assert mask.gather(-1, act.unsqueeze(-1)).all()  # sampled action always legal
        with torch.no_grad():
            advantage(batch)
        flat = batch.reshape(-1)
        for _ in range(2):
            for idx in torch.randperm(flat.numel()).split(125):
                loss = loss_mod(flat[idx])
                total = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                assert torch.isfinite(total)
                optim.zero_grad()
                total.backward()
                optim.step()
        n_batches += 1
    collector.shutdown()
    assert steps >= 2000 and n_batches == 4
