"""Minimal ``Box`` space.

on-policy inspects spaces by class name (``space.__class__.__name__ == "Box"``)
and reads ``.shape``, so this avoids a hard gym/gymnasium dependency.
"""
from __future__ import annotations

import numpy as np


class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.shape = tuple(int(size) for size in shape)
        self.dtype = dtype
        self.low = np.full(self.shape, low, dtype=dtype)
        self.high = np.full(self.shape, high, dtype=dtype)

    def __repr__(self) -> str:
        return f"Box({float(self.low.min())}, {float(self.high.max())}, {self.shape})"
