import torch

from cluedo.rl.evaluation import EVAL_FIELDS, evaluate_actor, random_suggest_baseline
from cluedo.rl.ppo_stack import make_actor


def test_eval_set_includes_random_suggest_and_cand_certain():
    assert "random/smart/certain" in EVAL_FIELDS
    assert "cand/smart/certain" in EVAL_FIELDS


def test_evaluate_actor_reports_win_rate_per_field_and_player_count():
    torch.manual_seed(0)
    actor = make_actor(16, 1)
    res = evaluate_actor(actor, blind=False, players=(3, 4), games=12, seed=1, workers=1,
                         fields=("random/smart/certain", "cand/smart/certain"))
    assert set(res) == {(f, n) for f in ("random/smart/certain", "cand/smart/certain") for n in (3, 4)}
    for r in res.values():
        assert r.games == 12 and 0.0 <= r.win_rate <= 1.0


def test_random_suggest_baseline_is_a_bot_in_the_cand_certain_field():
    r = random_suggest_baseline(players=4, games=12, seed=1, workers=1)
    assert r.games == 12 and r.n_players == 4
