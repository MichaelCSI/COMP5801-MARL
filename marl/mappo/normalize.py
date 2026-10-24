"""Running mean / variance normalizers (numpy, no torch)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class RunningMeanStd:
    """Parallel-algorithm running moments (Chan et al.)."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64)
        x = x.reshape((-1,) + self.mean.shape)
        b_mean, b_var, b_n = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x, clip: Optional[float] = None):
        y = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        if clip is not None:
            y = np.clip(y, -clip, clip)
        return y.astype(np.float32)

    def denormalize(self, y):
        return (np.asarray(y, dtype=np.float64) * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64).copy()
        self.var = np.asarray(d["var"], dtype=np.float64).copy()
        self.count = float(d["count"])


class Preprocessor:
    """Normalizes local observations (per agent) and the critic's global state.

    Messages are NOT normalized: SoC is already in [0, 1] and the mask is
    binary, so a dropped message stays exactly [0, 0].
    """

    def __init__(self, obs_dims: Sequence[int], state_dim: int, enabled: bool = True,
                 clip: float = 10.0):
        self.enabled = enabled
        self.clip = clip
        self.obs_rms = [RunningMeanStd((d,)) for d in obs_dims]
        self.state_rms = RunningMeanStd((state_dim,))

    def local(self, obs, update: bool):
        out = []
        for rms, o in zip(self.obs_rms, obs):
            o = np.asarray(o, dtype=np.float32)
            if not self.enabled:
                out.append(o)
                continue
            if update:
                rms.update(o[None])
            out.append(rms.normalize(o, self.clip))
        return out

    def state(self, s, update: bool):
        s = np.asarray(s, dtype=np.float32)
        if not self.enabled:
            return s
        if update:
            self.state_rms.update(s[None])
        return self.state_rms.normalize(s, self.clip)

    def state_dict(self) -> dict:
        return {"enabled": self.enabled, "clip": self.clip,
                "obs": [r.state_dict() for r in self.obs_rms],
                "state": self.state_rms.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        self.enabled, self.clip = d["enabled"], d["clip"]
        for r, sd in zip(self.obs_rms, d["obs"]):
            r.load_state_dict(sd)
        self.state_rms.load_state_dict(d["state"])
