"""Training configuration and the experimental arms (proposal Sections 4-6)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..comms.channel import RND_P_CHOICES

# arm -> settings that define it. Everything else is shared, so arms differ
# ONLY in these fields (proposal: "identical except for the peer-SoC input").
ARMS = {
    # primary baseline: MAPPO without messages
    "M0": dict(use_messages=False, critic="central", train_schedule="none"),
    # messages, perfect delivery in training
    "M1": dict(use_messages=True, critic="central", train_schedule="fixed", train_p=1.0),
    # messages, p ~ U{0, .5, .75, 1} per training episode
    "M1-RND": dict(use_messages=True, critic="central", train_schedule="random"),
    # optional: IPPO without messages (local critics)
    "I0": dict(use_messages=False, critic="local", train_schedule="none"),
    # optional: M1-RND without the availability mask (mask column held at 1,
    # so it carries no information and a drop looks like SoC = 0)
    "M1-RND-NM": dict(use_messages=True, critic="central", train_schedule="random",
                      force_mask_train=True),
}

# Independent random streams, keyed so they do not depend on the arm.
# Same seed => same env reset seeds (outages / windows) in every arm.
STREAM_TORCH, STREAM_ENV, STREAM_CHANNEL, STREAM_SCHEDULE, STREAM_EVAL = 0, 1, 2, 3, 4


def stream(seed: int, key: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed), int(key)])


@dataclass
class MAPPOConfig:
    arm: str = "M0"
    seed: int = 0

    # arm-defining fields (filled from ARMS by ``for_arm``)
    use_messages: bool = False
    critic: str = "central"          # "central" (MAPPO) or "local" (IPPO)
    train_schedule: str = "none"     # "none" | "fixed" | "random"
    train_p: float = 1.0
    rnd_p_choices: tuple = RND_P_CHOICES
    force_mask_train: bool = False

    # RL
    total_steps: int = 200_000       # environment steps (each = all agents act once)
    rollout_steps: int = 2_048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    bootstrap_on_truncation: bool = True

    # PPO
    ppo_epochs: int = 10
    num_minibatches: int = 4
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    lr_anneal: bool = True
    target_kl: float | None = None

    # networks
    hidden_sizes: tuple = (64, 64)
    init_log_std: float = -0.5
    normalize_obs: bool = True
    normalize_value: bool = True
    obs_clip: float = 10.0

    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    @classmethod
    def for_arm(cls, arm: str, **overrides) -> "MAPPOConfig":
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
        cfg = cls(arm=arm, **ARMS[arm])
        return replace(cfg, **overrides) if overrides else cfg

    def validate(self) -> None:
        if self.critic not in ("central", "local"):
            raise ValueError("critic must be 'central' or 'local'")
        if self.train_schedule not in ("none", "fixed", "random"):
            raise ValueError("train_schedule must be none/fixed/random")
        if self.use_messages and self.train_schedule == "none":
            raise ValueError("message arms need a training delivery schedule")
        if not self.use_messages and self.train_schedule != "none":
            raise ValueError("no-message arms must use train_schedule='none'")
        if self.rollout_steps % self.num_minibatches:
            raise ValueError("rollout_steps must be divisible by num_minibatches")

    def to_dict(self) -> dict:
        return asdict(self)
