import collections
import dataclasses
import pickle

from cluedo.rl.checkpoint import load_checkpoint
from cluedo.rl.config import TrainConfig, condition
from cluedo.rl.env import CluedoEnv
from cluedo.rl.ppo_stack import make_actor, make_critic
from cluedo.rl.selfplay import CheckpointPool
from cluedo.rl.train import RecordingLogger, train

CFG = TrainConfig(hidden=16, depth=1, turn_cap=40)


def _nets():
    return make_actor(CFG.hidden, CFG.depth), make_critic(CFG.hidden, CFG.depth)


def _pool(tmp_path, size=20, blind=False, condition_name="ppo-sp"):
    return CheckpointPool(tmp_path / "pool", CFG, condition_name, blind, size=size)


def _env(cond_name, pool, blind=False, seed=0):
    cfg = dataclasses.replace(CFG, pool_dir=str(pool.dir))
    return CluedoEnv(opponents=condition(cond_name).make_opponents(cfg), blind=blind, turn_cap=40, seed=seed)


def test_pool_keeps_the_newest_20_and_drops_the_oldest(tmp_path):
    pool, (actor, critic) = _pool(tmp_path), _nets()
    for step in range(1, 26):
        pool.add(actor, critic, step)
    assert pool.steps() == list(range(6, 26))
    assert len(list((tmp_path / "pool").glob("pool_*.pt"))) == 20


def test_self_play_conditions_differ_only_in_blind_flag():
    sp, blind = condition("ppo-sp"), condition("ppo-sp-blind")
    assert (sp.blind, blind.blind) == (False, True)
    assert sp.self_play and blind.self_play and not condition("ppo-fb").self_play
    cfg = TrainConfig(pool_dir="x")
    assert type(sp.make_opponents(cfg)) is type(blind.make_opponents(cfg))
    pickle.dumps(sp.make_opponents(cfg))  # sent to forkserver workers


def _names(env, episodes):
    out = collections.Counter()
    for _ in range(episodes):
        env.reset()
        out.update(env.opponent_names.values())
    return out


def test_opponent_seats_are_half_latest_half_uniform_from_pool_through_the_env(tmp_path):
    pool, (actor, critic) = _pool(tmp_path), _nets()
    pool.publish_latest(actor, critic, 1000)
    for step in (1, 2, 3, 4):
        pool.add(actor, critic, step)
    c = _names(_env("ppo-sp", pool), 400)
    total = sum(c.values())
    assert set(c) == {"latest:1000", "pool:1", "pool:2", "pool:3", "pool:4"}
    assert 0.45 < c["latest:1000"] / total < 0.55
    for s in (1, 2, 3, 4):  # uniform over pool
        assert 0.09 < c[f"pool:{s}"] / total < 0.16


def test_empty_pool_falls_back_to_latest_and_new_latest_is_picked_up(tmp_path):
    pool, (actor, critic) = _pool(tmp_path), _nets()
    pool.publish_latest(actor, critic, 10)
    env = _env("ppo-sp", pool, seed=1)
    assert set(_names(env, 5)) == {"latest:10"}
    pool.publish_latest(actor, critic, 20)
    assert set(_names(env, 5)) == {"latest:20"}


def test_opponents_are_stochastic_and_blind_flag_reaches_their_policies(tmp_path):
    pool, (actor, critic) = _pool(tmp_path, blind=True, condition_name="ppo-sp-blind"), _nets()
    pool.publish_latest(actor, critic, 5)
    assert load_checkpoint(pool.dir / "latest.pt").blind is True
    env = _env("ppo-sp-blind", pool, blind=True, seed=2)
    env.reset()
    assert all(not p.greedy and p.enc.blind for p in env._players.values())


def test_env_workers_run_single_threaded_torch():
    """Opponent policies run torch inside every worker; 12 workers x 12 threads thrashed (6p self-play
    collected at 0.7k steps/s instead of 1.9k)."""
    import torch
    from cluedo.rl.train import make_env
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        make_env("ppo-fb", CFG, seed=0)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


def _tiny(**kw):
    base = dict(seed=3, hidden=16, depth=1, total_frames=1000, frames_per_batch=500, workers=2,
                minibatch_size=128, ppo_epochs=1, checkpoint_every=500, eval_every=10_000, eval_games=4,
                turn_cap=60, pool_every=500)
    base.update(kw)
    return TrainConfig(**base)


def test_ppo_sp_blind_trains_end_to_end_and_fills_the_pool(tmp_path):
    final = train(_tiny(), "ppo-sp-blind", tmp_path, logger=RecordingLogger())
    ck = load_checkpoint(final)
    assert ck.blind and ck.condition == "ppo-sp-blind"
    assert len(list((tmp_path / "pool").glob("pool_*.pt"))) >= 2
