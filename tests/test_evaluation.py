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


def test_logged_game_records_learner_action_probabilities_and_replays():
    from cluedo.core import Game
    from cluedo.rl.evaluation import logged_game
    torch.manual_seed(0)
    actor = make_actor(16, 1)
    log = logged_game(actor, blind=False, base="cand/smart/certain", n_players=4, seed=5, seat=1)
    assert log["learner"] == 1 and log["outcome"] is not None
    Game.replay(log)  # still a valid Game log
    mine = [a for a in log["actions"] if a["seat"] == 1]
    assert mine and all(0 < len(a["probs"]) <= 5 for a in mine)
    for a in mine:
        assert all(0 < p <= 1 for _, p in a["probs"]) and sum(p for _, p in a["probs"]) <= 1 + 1e-6
    assert all("probs" not in a for a in log["actions"] if a["seat"] != 1)


def test_behaviour_rates_cover_each_player_count():
    from cluedo.rl.evaluation import behaviour_rates
    torch.manual_seed(0)
    res = behaviour_rates(make_actor(16, 1), blind=False, players=(3, 4), games=3, seed=1)
    assert set(res) == {3, 4}
    assert all(m.suggestions > 0 and 0 <= m.bluff_rate <= 1 for m in res.values())


def test_random_suggest_baseline_is_a_bot_in_the_cand_certain_field():
    r = random_suggest_baseline(players=4, games=12, seed=1, workers=1)
    assert r.games == 12 and r.n_players == 4
