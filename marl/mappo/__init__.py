"""MAPPO with peer-SoC messages.

Torch-free pieces (config, buffer, normalizers, rollout/evaluate) import
without torch; the trainer and networks need torch.
"""
from .buffer import RolloutBuffer, compute_gae
from .config import ARMS, MAPPOConfig
from .normalize import Preprocessor, RunningMeanStd
from .rollout import RandomPolicy, RolloutWorker, build_actor_inputs, evaluate

import importlib.util as _ilu

if _ilu.find_spec("torch") is not None:
    from .networks import MAPPOPolicy
    from .trainer import MAPPOTrainer
else:  # pragma: no cover - torch is optional at import time
    MAPPOPolicy = MAPPOTrainer = None

__all__ = [
    "ARMS", "MAPPOConfig", "MAPPOPolicy", "MAPPOTrainer", "Preprocessor", "RandomPolicy",
    "RolloutBuffer", "RolloutWorker", "RunningMeanStd", "build_actor_inputs",
    "compute_gae", "evaluate",
]
