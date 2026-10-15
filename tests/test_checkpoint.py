import random

import torch
from tensordict import TensorDict

from cluedo import contract as C
from cluedo.core.rules import Game
from cluedo.rl.checkpoint import load_checkpoint, save_checkpoint
from cluedo.rl.config import TrainConfig
from cluedo.rl.ppo_stack import make_actor, make_critic


def _make(cfg):
    torch.manual_seed(1)
    return make_actor(cfg.hidden, cfg.depth), make_critic(cfg.hidden, cfg.depth)


def test_roundtrip_gives_frozen_policy_with_same_outputs(tmp_path):
    cfg = TrainConfig(hidden=32, depth=1)
    actor, critic = _make(cfg)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, actor, critic, cfg, condition="ppo-fb", learner_steps=123)
    ck = load_checkpoint(path)
    assert (ck.condition, ck.learner_steps, ck.cfg) == ("ppo-fb", 123, cfg)
    assert not any(p.requires_grad for p in ck.actor.parameters())
    td = TensorDict({"observation": torch.rand(5, C.OBS_DIM),
                     "action_mask": torch.ones(5, C.N_ACTIONS, dtype=torch.bool)}, [5])
    assert torch.allclose(actor.get_dist(td.clone()).logits, ck.actor.get_dist(td.clone()).logits)


def test_checkpoint_plays_legal_complete_games_as_a_player(tmp_path):
    cfg = TrainConfig(hidden=32, depth=1)
    actor, critic = _make(cfg)
    save_checkpoint(tmp_path / "ck.pt", actor, critic, cfg, condition="ppo-fb", learner_steps=0)
    ck = load_checkpoint(tmp_path / "ck.pt")
    game = Game.new(4, seed=3, max_turns=60)
    rng = random.Random(0)
    players = {s: ck.player(game, s, rng) for s in range(4)}
    n = 0
    while not game.done:
        game.apply(players[game.decision.seat].act(game))  # an illegal action would raise
        n += 1
    assert n > 0


def test_policy_player_samples_the_actors_masked_distribution_and_greedy_is_its_argmax():
    from cluedo.rl.checkpoint import PolicyPlayer
    from cluedo.rl.env import CluedoEnv
    cfg = TrainConfig(hidden=32, depth=1)
    actor, _ = _make(cfg)
    game = Game.new(4, seed=5, max_turns=60)
    seat = game.decision.seat
    p = PolicyPlayer(actor, game, seat, blind=False)
    td = TensorDict({"observation": torch.from_numpy(p.enc.observation()),
                     "action_mask": torch.from_numpy(p.enc.mask())}, [])
    probs = actor.get_dist(td).probs
    top = int(probs.argmax())
    assert PolicyPlayer(actor, game, seat, False, greedy=True).act(game) == CluedoEnv._to_core(top)
    draws = [p.act(game) for _ in range(4000)]
    assert abs(draws.count(CluedoEnv._to_core(top)) / 4000 - float(probs[top])) < 0.04
