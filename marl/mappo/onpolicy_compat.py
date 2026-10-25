"""Access to the vendored on-policy MAPPO code (third_party/onpolicy).

Upstream: https://github.com/marlbenchmark/on-policy @ de66d7a (MIT).
Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
Games", NeurIPS 2022.
"""
from __future__ import annotations

import os
import sys

VENDOR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "onpolicy"))
ONPOLICY_COMMIT = "de66d7a4b23fac2513f56f96f73b3f5cb96695ac"
ONPOLICY_URL = "https://github.com/marlbenchmark/on-policy"


def ensure_on_path() -> None:
    if VENDOR_ROOT not in sys.path:
        sys.path.insert(0, VENDOR_ROOT)


def get_onpolicy_parser():
    """on-policy's own argparse parser with its reference defaults (no torch needed)."""
    ensure_on_path()
    from onpolicy.config import get_config
    return get_config()
