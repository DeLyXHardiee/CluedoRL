"""Self-play Opponents: a Checkpoint pool on disk + the OpponentSource that draws from it.

The Learner's policy lives in the trainer process; env workers are separate processes, so weights
travel through files in ``pool_dir``: ``latest.pt`` (rewritten after every PPO update, atomically) and
``pool_<steps>.pt`` (the newest ``size`` Checkpoints, oldest dropped). ``SelfPlay`` is picklable (just
paths and numbers); each worker lazily loads Checkpoints and reloads ``latest.pt`` when it changes.
Per episode, each Opponent Seat is the latest policy with prob. ``latest_fraction``, else a uniform
pool entry. Opponents sample from their policy (stochastic).
"""
from __future__ import annotations

import os
from pathlib import Path

from cluedo.rl.checkpoint import load_checkpoint, save_checkpoint


class CheckpointPool:
    """Trainer-side writer of the pool directory."""

    def __init__(self, pool_dir, cfg, condition: str, blind: bool, size: int = 20):
        self.dir, self.cfg, self.condition, self.blind, self.size = Path(pool_dir), cfg, condition, blind, size
        self.dir.mkdir(parents=True, exist_ok=True)

    def _save(self, name, actor, critic, steps):
        tmp = self.dir / f"{name}.tmp"
        save_checkpoint(tmp, actor, critic, self.cfg, self.condition, steps, blind=self.blind)
        os.replace(tmp, self.dir / f"{name}.pt")  # atomic: readers never see a partial file

    def publish_latest(self, actor, critic, steps: int) -> None:
        self._save("latest", actor, critic, steps)

    def add(self, actor, critic, steps: int) -> None:
        self._save(f"pool_{steps:010d}", actor, critic, steps)
        for old in sorted(self.dir.glob("pool_*.pt"))[:-self.size]:
            old.unlink()

    def steps(self) -> list[int]:
        return [int(p.stem.split("_")[1]) for p in sorted(self.dir.glob("pool_*.pt"))]


class SelfPlay:
    """OpponentSource: each Opponent Seat plays the latest policy or a uniform pool member."""

    def __init__(self, pool_dir, latest_fraction: float = 0.5):
        self.pool_dir, self.latest_fraction = str(pool_dir), latest_fraction
        self._latest = None  # (mtime_ns, Checkpoint)
        self._pool: dict[str, object] = {}

    def __getstate__(self):  # ship paths only, never loaded networks
        return {"pool_dir": self.pool_dir, "latest_fraction": self.latest_fraction}

    def __setstate__(self, state):
        self.__init__(**state)

    def _load_latest(self):
        path = Path(self.pool_dir) / "latest.pt"
        m = path.stat().st_mtime_ns
        if self._latest is None or self._latest[0] != m:
            self._latest = (m, load_checkpoint(path))
        return self._latest[1]

    def _pool_entry(self, path: Path):
        ck = self._pool.get(path.name)
        if ck is None:
            ck = self._pool[path.name] = load_checkpoint(path)
        return ck

    def seat_players(self, game, learner, rng):
        latest = self._load_latest()
        paths = sorted(Path(self.pool_dir).glob("pool_*.pt"))
        live = {p.name for p in paths}
        for k in [k for k in self._pool if k not in live]:
            del self._pool[k]
        players = {}
        for seat in range(game.n):
            if seat == learner:
                continue
            ck, kind = latest, "latest"
            if paths and rng.random() >= self.latest_fraction:
                try:
                    ck, kind = self._pool_entry(rng.choice(paths)), "pool"
                except FileNotFoundError:  # dropped from the pool mid-read
                    pass
            player = ck.player(game, seat, rng)
            player.name = f"{kind}:{ck.learner_steps}"
            players[seat] = player
        return players
