import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from cluedo.rl.checkpoint import load_checkpoint
from cluedo.rl.config import TrainConfig, condition
from cluedo.rl.train import RecordingLogger, train


def _tiny(**kw):
    base = dict(seed=3, hidden=16, depth=1, total_frames=1000, frames_per_batch=500, workers=2,
                minibatch_size=128, ppo_epochs=1, checkpoint_every=500, eval_every=500, eval_games=4,
                turn_cap=60)
    base.update(kw)
    return TrainConfig(**base)


def test_short_run_checkpoints_and_logs_tensorboard_curves(tmp_path):
    cfg = _tiny()
    final = train(cfg, "ppo-fb", tmp_path)
    ck = load_checkpoint(final)
    assert ck.learner_steps >= cfg.total_frames and ck.condition == "ppo-fb"
    assert len(list(tmp_path.glob("checkpoints/step_*.pt"))) >= 2  # periodic, not just final

    acc = EventAccumulator(str(tmp_path / "tb"))
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    for required in ("eval/win_rate/random/smart/certain/4p", "eval/win_rate/cand/smart/certain/4p",
                     "eval/baseline/random_suggest/4p", "train/entropy", "train/kl", "train/value_loss"):
        assert required in tags, required


def test_logger_is_injectable_so_later_metrics_share_the_same_stream(tmp_path):
    log = RecordingLogger()
    train(_tiny(total_frames=500), "ppo-fb", tmp_path, logger=log)
    assert any(name == "train/entropy" for name, _, _ in log.records)


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        condition("nope")
