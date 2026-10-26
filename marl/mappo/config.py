"""Experimental arms and the mapping onto on-policy's MAPPO arguments.

The learner is the reference MAPPO implementation (marlbenchmark/on-policy,
Yu et al. 2022), vendored in third_party/onpolicy. ``MAPPOConfig`` holds:

* the fields that DEFINE an arm (messages on/off, training delivery
  schedule, centralized vs local critic), and
* run settings (seed, steps, rollout length),

and ``to_onpolicy_args()`` turns it into the ``argparse.Namespace`` on-policy
expects. Everything not set below keeps on-policy's own default, and every
arm gets the same values (equal tuning budget, proposal Section 4).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..comms.channel import RND_P_CHOICES

# Arm name -> the only settings that differ between arms
ARMS = {
    # primary baseline: MAPPO without messages
    "M0": dict(use_messages=False, critic="central", train_schedule="none"),
    # messages, perfect delivery in training
    "M1": dict(use_messages=True, critic="central", train_schedule="fixed", train_p=1.0),
    # messages, p ~ U{0, .5, .75, 1} per training episode
    "M1-RND": dict(use_messages=True, critic="central", train_schedule="random"),
    # optional: IPPO without messages (each critic sees only its agent's input)
    "I0": dict(use_messages=False, critic="local", train_schedule="none"),
    # optional: M1-RND without the availability mask (mask column held at 1)
    "M1-RND-NM": dict(use_messages=True, critic="central", train_schedule="random",
                      force_mask_train=True),
}

# Independent random streams keyed so they do not depend on the arm:
# same seed => same env reset seeds (outages / windows) in every arm.
STREAM_TORCH, STREAM_ENV, STREAM_CHANNEL, STREAM_SCHEDULE, STREAM_EVAL = 0, 1, 2, 3, 4


def stream(seed: int, key: int) -> np.random.SeedSequence:
    """Independent random stream ``key`` (one of the STREAM_* constants) for a run seed."""
    return np.random.SeedSequence([int(seed), int(key)])


# on-policy settings fixed by the experimental design (not tuning knobs).
FIXED_ONPOLICY_ARGS = dict(
    algorithm_name="mappo",           # feed-forward MAPPO (on-policy's "mappo")
    share_policy=False,               # separated mode: one actor per building
    use_recurrent_policy=False,       # MLP policy, as in the proposal
    use_naive_recurrent_policy=False,
    n_rollout_threads=1,              # CityLearn is a single sequential simulator
    use_wandb=False,
    use_render=False,
    use_eval=False,
    # Training windows end by time limit, not a true terminal: on-policy's
    # bad_masks then zero the advantage of the cut-off step.
    use_proper_time_limits=True,
    # We normalize each local feature with running statistics in MessageEnv
    # and leave the [SoC, mask] message columns raw. A LayerNorm over the whole
    # actor input would mix raw-scale CityLearn features with the messages.
    use_feature_normalization=False,
)


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

    # run settings
    total_steps: int = 200_000       # env steps (each = all agents act once)
    rollout_steps: int = 1_000       # on-policy `episode_length` (buffer length)
    normalize_obs: bool = True
    obs_clip: float = 10.0
    device: str = "cpu"

    # anything else on-policy accepts, e.g. {"ppo_epoch": 10, "lr": 3e-4}
    onpolicy_overrides: dict = field(default_factory=dict)

    @classmethod
    def for_arm(cls, arm: str, **overrides) -> "MAPPOConfig":
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
        config = cls(arm=arm, **ARMS[arm])
        return replace(config, **overrides) if overrides else config

    def validate(self) -> None:
        if self.critic not in ("central", "local"):
            raise ValueError("critic must be 'central' or 'local'")
        if self.train_schedule not in ("none", "fixed", "random"):
            raise ValueError("train_schedule must be none/fixed/random")

        if self.use_messages and self.train_schedule == "none":
            raise ValueError("message arms need a training delivery schedule")
        if not self.use_messages and self.train_schedule != "none":
            raise ValueError("no-message arms must use train_schedule='none'")

        locked = set(self.onpolicy_overrides) & set(FIXED_ONPOLICY_ARGS)
        if locked:
            raise ValueError(f"these on-policy args are fixed by the design: {sorted(locked)}")

    def to_onpolicy_args(self):
        """``argparse.Namespace`` for on-policy's R_MAPPO / R_MAPPOPolicy / buffer."""
        from .onpolicy_compat import get_onpolicy_parser

        self.validate()
        args = get_onpolicy_parser().parse_args([])   # on-policy reference defaults
        for name, value in FIXED_ONPOLICY_ARGS.items():
            setattr(args, name, value)

        args.use_centralized_V = self.critic == "central"
        args.seed = int(self.seed)
        args.episode_length = int(self.rollout_steps)
        args.num_env_steps = int(self.total_steps)
        args.cuda = self.device != "cpu"
        args.experiment_name = f"{self.arm}_s{self.seed}"

        for name, value in self.onpolicy_overrides.items():
            if not hasattr(args, name):
                raise ValueError(f"on-policy has no argument {name!r}")
            setattr(args, name, value)
        return args

    def to_dict(self) -> dict:
        return asdict(self)
