import re
import shutil
from pathlib import Path

import pytest
import torch

from cluedo.figures import DOUBLE_COLUMN_IN, FIGURES, SINGLE_COLUMN_IN, build_all, main
from cluedo.rl.checkpoint import save_checkpoint
from cluedo.rl.config import TrainConfig
from cluedo.rl.eval_grid import EvalConfig, checkpoint_entrant, run_grid, write_tables
from cluedo.rl.ppo_stack import make_actor, make_critic

ROOT = Path(__file__).resolve().parent.parent
CONDS = ("ppo-sp", "ppo-sp-blind", "ppo-fb")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """A tiny real result set from the ticket 13 harness (untrained Checkpoints, few games)."""
    d = tmp_path_factory.mktemp("res")
    cfg = TrainConfig(hidden=16, depth=1)
    torch.manual_seed(0)
    ents = {}
    for c in CONDS:
        save_checkpoint(d / f"{c}.pt", make_actor(16, 1), make_critic(16, 1), cfg, condition=c,
                        learner_steps=0, blind=c == "ppo-sp-blind")
        ents[c] = checkpoint_entrant(c, d / f"{c}.pt")
    ec = EvalConfig(games=16, seeds=(0, 1), players=(3, 4),
                    opponents={"cand/certain": "cand/smart/certain", "bluff": "bluff/smart/certain"},
                    mixed_bot="cand/smart/certain", metric_games=16)
    write_tables(run_grid(ents, ec, workers=2), d / "eval")
    return d / "eval"


def test_one_command_renders_every_figure_as_vector_pdf(results, tmp_path):
    paths = build_all(results, tmp_path)
    assert {p.stem for p in paths} == set(FIGURES)
    assert len(FIGURES) >= 5
    for p in paths:
        data = p.read_bytes()
        assert p.suffix == ".pdf" and data.startswith(b"%PDF") and len(data) > 1000
        assert b"/Subtype /Image" not in data  # vector, no raster plots


def test_figures_fit_paper_column_widths(results, tmp_path):
    for p in build_all(results, tmp_path):
        m = re.search(rb"/MediaBox\s*\[\s*0 0 ([\d.]+) ([\d.]+)", p.read_bytes())
        want = DOUBLE_COLUMN_IN if FIGURES[p.stem] == "double" else SINGLE_COLUMN_IN
        assert float(m.group(1)) / 72 == pytest.approx(want, abs=0.05)


def test_cli_is_a_single_command(results, tmp_path):
    main(["--results", str(results), "--out", str(tmp_path / "f")])
    assert len(list((tmp_path / "f").glob("*.pdf"))) == len(FIGURES)


def test_missing_comparisons_file_is_not_fatal(results, tmp_path):
    d = tmp_path / "partial"
    shutil.copytree(results, d)
    (d / "comparisons.csv").unlink()
    assert len(build_all(d, tmp_path / "o")) == len(FIGURES)


def test_reading_guide_covers_every_figure_and_tensorboard_tags():
    text = (ROOT / "figures" / "READING_GUIDE.md").read_text()
    for name in FIGURES:
        assert f"`{name}`" in text
    for tag in ("train/entropy", "train/kl", "train/clip_fraction", "train/value_loss",
                "train/episode_win_rate", "eval/win_rate", "eval/bluff_rate", "eval/anchor_rate",
                "eval/chance", "eval/baseline/random_suggest"):
        assert tag in text
    for phrase in ("good", "bad", "warning"):
        assert phrase in text.lower()
