"""Actor and critic networks (torch).

One actor per building with NO parameter sharing (proposal Section 5), and
either one centralized critic on the true global state (MAPPO) or one local
critic per agent on its own input (IPPO).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .normalize import RunningMeanStd


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, out_gain: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden=(64, 64), init_log_std=-0.5):
        super().__init__()
        self.mu = mlp(in_dim, hidden, act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def dist(self, x: torch.Tensor) -> Normal:
        mean = self.mu(x)
        return Normal(mean, self.log_std.exp().expand_as(mean))


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(64, 64)):
        super().__init__()
        self.v = mlp(in_dim, hidden, 1, out_gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x).squeeze(-1)


class MAPPOPolicy(nn.Module):
    """Holds all actors and critics; numpy-in / numpy-out for the rollout loop."""

    def __init__(self, actor_dims, action_dims, critic_dims, hidden=(64, 64),
                 init_log_std=-0.5, normalize_value=True, device="cpu"):
        super().__init__()
        self.n_agents = len(actor_dims)
        self.actor_dims, self.action_dims = list(actor_dims), list(action_dims)
        self.critic_dims = list(critic_dims)
        self.actors = nn.ModuleList(
            GaussianActor(d, a, hidden, init_log_std) for d, a in zip(actor_dims, action_dims))
        self.critics = nn.ModuleList(Critic(d, hidden) for d in critic_dims)
        self.normalize_value = normalize_value
        self.value_rms = [RunningMeanStd(()) for _ in critic_dims]
        self.device = torch.device(device)
        self.to(self.device)

    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, actor_inputs, deterministic: bool = False):
        acts, logps = [], []
        for actor, x in zip(self.actors, actor_inputs):
            d = actor.dist(self._t(x)[None])
            a = d.mean if deterministic else d.sample()
            acts.append(a[0].cpu().numpy())
            logps.append(float(d.log_prob(a).sum(-1)[0]))
        width = max(self.action_dims)
        out = np.zeros((self.n_agents, width), np.float32)
        for i, a in enumerate(acts):
            out[i, : a.shape[0]] = a
        return out, np.asarray(logps, np.float32)

    @torch.no_grad()
    def value(self, critic_inputs):
        """De-normalized V for each critic, shape (K,)."""
        v = []
        for k, (critic, x) in enumerate(zip(self.critics, critic_inputs)):
            vn = float(critic(self._t(x)[None])[0])
            v.append(self.value_rms[k].denormalize(vn) if self.normalize_value else vn)
        return np.asarray(v, np.float32)

    def extra_state(self) -> dict:
        return {"value_rms": [r.state_dict() for r in self.value_rms]}

    def load_extra_state(self, d: dict) -> None:
        for r, sd in zip(self.value_rms, d["value_rms"]):
            r.load_state_dict(sd)
