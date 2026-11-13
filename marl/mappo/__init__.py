"""MAPPO with peer-SoC messages, on top of the reference on-policy MAPPO.

Learner: marlbenchmark/on-policy (Yu et al., NeurIPS 2022), vendored in
third_party/onpolicy. Torch-free pieces (config, MessageEnv, evaluate,
normalizers) import without torch; ``OnPolicyRunner`` needs torch.
"""
import importlib.util as _ilu

from .config import (ARMS, CORE_ARMS, FIXED_ONPOLICY_ARGS, OPTIONAL_ARMS, MAPPOConfig,
                     evaluation_conditions)
from .evaluate import RandomPolicy, evaluate
from .message_env import MessageEnv, build_actor_inputs, full_peer_block, make_message_env
from .normalize import Preprocessor, RunningMeanStd

if _ilu.find_spec("torch") is not None:
    from .runner import OnPolicyRunner
else:  # pragma: no cover - torch is optional at import time
    OnPolicyRunner = None

__all__ = [
    "ARMS", "CORE_ARMS", "FIXED_ONPOLICY_ARGS", "MAPPOConfig", "MessageEnv", "OPTIONAL_ARMS",
    "OnPolicyRunner", "Preprocessor", "RandomPolicy", "RunningMeanStd", "build_actor_inputs",
    "evaluate", "evaluation_conditions", "full_peer_block", "make_message_env",
]
