"""Running mean / variance normalizers (numpy, no torch)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class RunningMeanStd:
    """Running mean and variance, merged batch by batch (Chan et al.'s parallel update)."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)   # tiny prior count avoids dividing by zero

    def update(self, x) -> None:
        batch = np.asarray(x, dtype=np.float64).reshape((-1,) + self.mean.shape)
        batch_mean, batch_var, batch_count = batch.mean(axis=0), batch.var(axis=0), batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count

        # combine the two sums of squared deviations, then renormalize
        combined_m2 = (self.var * self.count + batch_var * batch_count
                       + delta ** 2 * self.count * batch_count / total_count)
        self.var = combined_m2 / total_count
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x, clip: Optional[float] = None):
        normalized = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        if clip is not None:
            normalized = np.clip(normalized, -clip, clip)
        return normalized.astype(np.float32)

    def denormalize(self, normalized):
        return (np.asarray(normalized, dtype=np.float64) * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, saved: dict) -> None:
        self.mean = np.asarray(saved["mean"], dtype=np.float64).copy()
        self.var = np.asarray(saved["var"], dtype=np.float64).copy()
        self.count = float(saved["count"])


class Preprocessor:
    """Normalizes local observations (per agent) and the critic's global state.

    Messages are NOT normalized: SoC is already in [0, 1] and the mask is
    binary, so a dropped message stays exactly [0, 0].
    """

    def __init__(self, obs_dims: Sequence[int], state_dim: int, enabled: bool = True,
                 clip: float = 10.0):
        self.enabled = enabled
        self.clip = clip
        self.local_obs_stats = [RunningMeanStd((dim,)) for dim in obs_dims]
        self.state_stats = RunningMeanStd((state_dim,))

    def normalize_local_obs(self, obs, update: bool):
        """Normalize each agent's local observation; ``update`` adds it to the statistics."""
        normalized = []
        for stats, agent_obs in zip(self.local_obs_stats, obs):
            agent_obs = np.asarray(agent_obs, dtype=np.float32)
            if not self.enabled:
                normalized.append(agent_obs)
                continue

            if update:
                stats.update(agent_obs[None])
            normalized.append(stats.normalize(agent_obs, self.clip))
        return normalized

    def normalize_state(self, state, update: bool):
        """Normalize the critic's global state; ``update`` adds it to the statistics."""
        state = np.asarray(state, dtype=np.float32)
        if not self.enabled:
            return state

        if update:
            self.state_stats.update(state[None])
        return self.state_stats.normalize(state, self.clip)

    # "obs" / "state" keys kept so existing checkpoints still load
    def state_dict(self) -> dict:
        return {"enabled": self.enabled, "clip": self.clip,
                "obs": [stats.state_dict() for stats in self.local_obs_stats],
                "state": self.state_stats.state_dict()}

    def load_state_dict(self, saved: dict) -> None:
        self.enabled, self.clip = saved["enabled"], saved["clip"]
        for stats, saved_stats in zip(self.local_obs_stats, saved["obs"]):
            stats.load_state_dict(saved_stats)
        self.state_stats.load_state_dict(saved["state"])
