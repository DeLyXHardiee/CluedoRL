import numpy as np
import torch
from torchrl.envs.utils import check_env_specs

from cluedo import contract as C
from cluedo.rl.env import CluedoEnv
from cluedo.rl.opponents import PPO_FB_FIELD, BotField


def test_check_env_specs():
    check_env_specs(CluedoEnv(seed=0))


def test_ppo_fb_field_is_the_four_bots():
    assert sorted(PPO_FB_FIELD) == sorted(
        f"{s}/smart/{a}" for s in ("cand", "focused") for a in ("certain", "gamble2"))


def run_random(env, steps, rng):
    td = env.reset()
    for _ in range(steps):
        mask = td["action_mask"]
        yield td
        td["action"] = torch.tensor(int(rng.choice(np.flatnonzero(mask.numpy()))))
        td = env.step(td)["next"]
        if td["done"].item():
            yield td
            td = env.reset()


def test_mask_invariants_over_random_rollouts():
    env = CluedoEnv(seed=3)
    rng = np.random.default_rng(0)
    ns, phases = set(), set()
    for td in run_random(env, 1500, rng):
        if td["done"].item():
            continue
        mask = td["action_mask"].numpy()
        assert mask.any()
        phase = C.mask_phase(mask)  # exactly one phase active
        assert C.obs_phase(td["observation"].numpy()) == phase
        phases.add(phase)
        ns.add(int(np.argmax(C.view(td["observation"].numpy(), "n_players"))) + 3)
        if phase == C.ACCUSE:
            assert mask[C.encode_accuse(*env.game.envelope)]  # true Envelope never masked
        if phase == C.REVEAL:
            assert mask.sum() <= 3
    assert phases == {0, 1, 2} and ns == {3, 4, 5, 6}


def test_correct_accusation_wins():
    env = CluedoEnv(seed=1)
    td = env.reset()
    for _ in range(100):
        if C.mask_phase(td["action_mask"].numpy()) == C.ACCUSE:
            td["action"] = torch.tensor(C.encode_accuse(*env.game.envelope))
        else:
            td["action"] = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
        nxt = env.step(td)["next"]
        if nxt["done"].item():
            assert nxt["terminated"].item() and not nxt["truncated"].item()
            assert nxt["reward"].item() == 1.0
            assert env.game.outcome["winner"] == env.learner
            return
        td = nxt
    raise AssertionError("never reached an Accusation")


def test_wrong_accusation_eliminates_and_terminates_with_zero():
    env = CluedoEnv(seed=2)
    td = env.reset()
    for _ in range(300):
        if C.mask_phase(td["action_mask"].numpy()) == C.ACCUSE:
            wrong = [a for a in np.flatnonzero(td["action_mask"].numpy())
                     if a != C.PASS_ACTION and C.decode_action(a).cards != tuple(env.game.envelope)]
            if wrong:
                td["action"] = torch.tensor(int(wrong[0]))
                nxt = env.step(td)["next"]
                assert nxt["terminated"].item() and nxt["reward"].item() == 0.0
                assert env.game.eliminated[env.learner]
                return
            td["action"] = torch.tensor(C.PASS_ACTION)
        else:
            td["action"] = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
        td = env.step(td)["next"]
        if td["done"].item():
            td = env.reset()
    raise AssertionError("no wrong Accusation available")


def test_turn_cap_truncates():
    env = CluedoEnv(seed=4, turn_cap=3)
    td = env.reset()
    for _ in range(200):
        m = td["action_mask"].numpy()
        td["action"] = torch.tensor(C.PASS_ACTION if C.mask_phase(m) == C.ACCUSE
                                    else int(np.flatnonzero(m)[0]))
        td = env.step(td)["next"]
        if td["done"].item():
            if td["truncated"].item():
                assert not td["terminated"].item() and td["reward"].item() == 0.0
                return
            td = env.reset()  # an Opponent won first; try again
    raise AssertionError("not truncated")


def test_blind_flag_zeroes_public_history():
    rng = np.random.default_rng(0)
    for blind in (False, True):
        env = CluedoEnv(seed=5, blind=blind)
        seen = False
        for td in run_random(env, 200, rng):
            hist = np.concatenate(
                [td["observation"].numpy()[C.SLICES[n].sl] for n in C.PUBLIC_HISTORY])
            seen |= bool(hist.any())
        assert seen != blind


class UniformPlayer:
    def __init__(self, rng):
        self.rng = rng

    def act(self, game):
        return self.rng.choice(game.legal_actions())


class CountingSource:
    """Custom Opponent source (the seam ticket 10 plugs Checkpoints into)."""
    def __init__(self):
        self.calls = []

    def seat_players(self, game, learner, rng):
        self.calls.append((game.n, learner))
        return {s: UniformPlayer(rng) for s in range(game.n) if s != learner}


def test_pluggable_opponent_source():
    src = CountingSource()
    env = CluedoEnv(opponents=src, seed=0)
    env.reset()
    env.reset()
    assert len(src.calls) == 2


def test_bot_field_samples_per_seat_from_given_specs():
    env = CluedoEnv(opponents=BotField(["focused/smart/certain"]), seed=0)
    env.reset()
    assert set(env.opponent_names.values()) == {"focused/smart/certain"}
    assert len(env.opponent_names) == env.game.n - 1
