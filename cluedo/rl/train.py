"""Training entry point for every Condition (PPO-FB now; PPO-SP / PPO-SP-blind in ticket 10).

    python -m cluedo.rl.train --condition ppo-fb --seed 0 --total-frames 2000000 --out runs/fb-s0
    tensorboard --logdir runs/fb-s0/tb

One env worker per CPU core (``cfg.workers``), PPO with GAE (gamma = 1), periodic Checkpoints in
``<out>/checkpoints/step_<learner steps>.pt`` (load with ``cluedo.rl.checkpoint.load_checkpoint``) and
TensorBoard curves under ``<out>/tb``. Metrics go through a ``Logger`` (``.scalar(name, value, step)``)
so ticket 12 can log Bluff / Anchor rate into the same stream. A Condition differs from another only in
its Opponent source and ``blind`` flag (``cluedo.rl.config.CONDITIONS``); everything else is ``TrainConfig``.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import time
from pathlib import Path

import torch

from cluedo.rl import evaluation
from cluedo.rl.checkpoint import save_checkpoint
from cluedo.rl.config import TrainConfig, condition as get_condition
from cluedo.rl.env import CluedoEnv
from cluedo.rl.ppo_stack import make_actor, make_critic, make_loss


class RecordingLogger:
    """In-memory logger (tests)."""

    def __init__(self):
        self.records: list[tuple[str, float, int]] = []

    def scalar(self, name: str, value: float, step: int) -> None:
        self.records.append((name, float(value), step))

    def close(self) -> None:
        pass


class TensorBoardLogger:
    def __init__(self, logdir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(logdir))

    def scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, float(value), step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


def make_env(condition_name: str, cfg: TrainConfig, seed: int) -> CluedoEnv:
    cond = get_condition(condition_name)
    return CluedoEnv(opponents=cond.make_opponents(cfg), blind=cond.blind, turn_cap=cfg.turn_cap, seed=seed)


def _evaluate(actor, blind, cfg, logger, step, seed, baseline=False):
    t = time.time()
    res = evaluation.evaluate_actor(actor, blind, cfg.eval_players, cfg.eval_games, seed, cfg.workers)
    for (field, n), r in res.items():
        logger.scalar(f"eval/win_rate/{field}/{n}p", r.win_rate, step)
    for n, m in evaluation.behaviour_rates(actor, blind, cfg.eval_players, max(1, cfg.eval_games // 8), seed).items():
        logger.scalar(f"eval/bluff_rate/{n}p", m.bluff_rate, step)
        logger.scalar(f"eval/anchor_rate/{n}p", m.anchor_rate, step)
    if baseline:
        for n in cfg.eval_players:
            b = evaluation.random_suggest_baseline(n, cfg.eval_games, seed, cfg.workers)
            logger.scalar(f"eval/baseline/random_suggest/{n}p", b.win_rate, 0)
            logger.scalar(f"eval/chance/{n}p", 1 / n, 0)
    return res, time.time() - t


def train(cfg: TrainConfig, condition_name: str, out_dir, logger=None, verbose: bool = False) -> Path:
    """Run PPO for ``cfg.total_frames`` Learner steps. Returns the final Checkpoint path."""
    from torchrl.collectors import MultiSyncCollector

    cond = get_condition(condition_name)
    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    own_logger = logger is None
    logger = logger or TensorBoardLogger(out / "tb")
    (out / "config.txt").write_text(f"condition={condition_name}\n{cfg}\n")

    torch.manual_seed(cfg.seed)
    actor, critic = make_actor(cfg.hidden, cfg.depth, cfg.pass_bias), make_critic(cfg.hidden, cfg.depth)
    loss_mod, advantage = make_loss(actor, critic, cfg.gamma, cfg.lmbda, cfg.clip_epsilon, cfg.entropy_coeff)
    optim = torch.optim.Adam(loss_mod.parameters(), lr=cfg.lr)

    env_fns = [functools.partial(make_env, condition_name, cfg, cfg.seed * 10_000 + i)
               for i in range(cfg.workers)]
    collector = MultiSyncCollector(env_fns, actor, frames_per_batch=cfg.frames_per_batch,
                                   total_frames=cfg.total_frames, device="cpu")

    def ckpt(step):
        p = out / "checkpoints" / f"step_{step:010d}.pt"
        save_checkpoint(p, actor, critic, cfg, condition_name, step, blind=cond.blind)
        return p

    steps, next_eval, next_ckpt, t0 = 0, 0, cfg.checkpoint_every, time.time()
    last = None
    try:
        for batch in collector:
            steps += batch.numel()
            with torch.no_grad():
                advantage(batch)
            flat = batch.reshape(-1)
            stats = {k: [] for k in ("entropy", "kl", "value_loss", "policy_loss", "clip_fraction")}
            for _ in range(cfg.ppo_epochs):
                for idx in torch.randperm(flat.numel()).split(cfg.minibatch_size):
                    loss = loss_mod(flat[idx])
                    total = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                    optim.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg.max_grad_norm)
                    optim.step()
                    for k, key in (("entropy", "entropy"), ("kl", "kl_approx"), ("value_loss", "loss_critic"),
                                   ("policy_loss", "loss_objective"), ("clip_fraction", "clip_fraction")):
                        stats[k].append(float(loss[key].detach()))
            collector.update_policy_weights_()
            for k, v in stats.items():
                logger.scalar(f"train/{k}", sum(v) / len(v), steps)
            done = batch["next", "done"].squeeze(-1)
            if done.any():  # fraction of finished training episodes the Learner won
                logger.scalar("train/episode_win_rate", float(batch["next", "reward"].squeeze(-1)[done].mean()), steps)
            logger.scalar("train/learner_steps_per_s", steps / (time.time() - t0), steps)
            if steps >= next_ckpt:
                last = ckpt(steps)
                next_ckpt += cfg.checkpoint_every
            if steps >= next_eval:
                res, dt = _evaluate(actor, cond.blind, cfg, logger, steps, cfg.seed, baseline=next_eval == 0)
                next_eval += cfg.eval_every
                if verbose:
                    print(f"[{steps}] {steps / (time.time() - t0):.0f} steps/s eval({dt:.0f}s) "
                          + " ".join(f"{f.split('/')[0]}:{r.win_rate:.2f}" for (f, _), r in res.items()), flush=True)
            elif verbose:
                print(f"[{steps}] {steps / (time.time() - t0):.0f} steps/s", flush=True)
        final = ckpt(steps)
        _evaluate(actor, cond.blind, cfg, logger, steps, cfg.seed)
    finally:
        collector.shutdown()
        if own_logger:
            logger.close()
    return final


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", default="ppo-fb")
    p.add_argument("--out", required=True)
    d = TrainConfig()
    for f in dataclasses.fields(TrainConfig):
        if f.name == "eval_players":
            p.add_argument("--eval-players", type=int, nargs="+", default=list(d.eval_players))
        else:
            p.add_argument("--" + f.name.replace("_", "-"), type=type(getattr(d, f.name)), default=getattr(d, f.name))
    a = vars(p.parse_args(argv))
    cond, out = a.pop("condition"), a.pop("out")
    a["eval_players"] = tuple(a["eval_players"])
    final = train(TrainConfig(**a), cond, out, verbose=True)
    print("final checkpoint:", final)


if __name__ == "__main__":
    main()
