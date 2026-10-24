"""Lossy peer state-of-charge (SoC) message channel (proposal Section 1).

Before acting at step t every agent j sends its current SoC to each of its
n-1 peers. Each of the n(n-1) directed links j -> i delivers independently
with probability p. Receiver i gets, for every peer j (ascending index, self
skipped):

    [SoC_j, 1]  if the link delivered
    [0,     0]  if the link dropped

The second entry is an availability mask, needed because SoC = 0 is a valid
state. Nothing else is transmitted and there is no latency.

The channel owns its own random generator, separate from the environment,
so changing p (or the arm) never changes the environment's outage
realizations.

``force_mask=True`` is the mask check from Section 6: the mask is always 1,
so a dropped message looks like a real SoC of 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: Training delivery levels for M1-RND (proposal Section 5).
RND_P_CHOICES: tuple = (0.0, 0.5, 0.75, 1.0)

#: Evaluation delivery levels (proposal Section 6, experiment matrix).
EVAL_P_LEVELS: tuple = (1.0, 0.9, 0.5, 0.0)


def peer_indices(n_agents: int, receiver: int) -> list:
    """Peers of ``receiver`` in the order their messages appear in its input."""
    return [j for j in range(n_agents) if j != receiver]


def message_dim(n_agents: int) -> int:
    """Length of one agent's message vector: [SoC, mask] per peer."""
    return 2 * (n_agents - 1)


class PeerSoCChannel:
    """Independent Bernoulli(p) delivery on every directed link, every step."""

    def __init__(
        self,
        n_agents: int,
        p: float = 1.0,
        seed: Optional[int | np.random.SeedSequence] = None,
        force_mask: bool = False,
    ) -> None:
        if n_agents < 2:
            raise ValueError("a message channel needs at least two agents")
        self.n_agents = int(n_agents)
        self.force_mask = bool(force_mask)
        self.rng = np.random.default_rng(seed)
        self.set_p(p)
        # Pre-computed receiver -> peer index table, shape (n, n-1).
        self._peers = np.array(
            [peer_indices(self.n_agents, i) for i in range(self.n_agents)], dtype=int
        )
        self.last_delivered: Optional[np.ndarray] = None

    @property
    def dim(self) -> int:
        return message_dim(self.n_agents)

    def set_p(self, p: float) -> None:
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"delivery probability must be in [0, 1], got {p}")
        self.p = p

    def sample_delivery(self) -> np.ndarray:
        """Boolean (n, n) matrix; entry [i, j] is True if j -> i delivered.

        The diagonal is always False (no self-messages). One uniform draw is
        consumed per directed link per step whatever p is, so two channels
        with the same seed but different p stay aligned step by step.
        """
        u = self.rng.random((self.n_agents, self.n_agents))
        delivered = u < self.p
        np.fill_diagonal(delivered, False)
        return delivered

    def transmit(self, soc: Sequence[float]) -> np.ndarray:
        """Return the (n, 2(n-1)) message block for one step.

        Row i is receiver i's input: [SoC_j, mask_j] for each peer j.
        """
        soc = np.asarray(soc, dtype=np.float32).reshape(-1)
        if soc.shape[0] != self.n_agents:
            raise ValueError(f"expected {self.n_agents} SoC values, got {soc.shape[0]}")
        delivered = self.sample_delivery()
        self.last_delivered = delivered
        return self.encode(soc, delivered)

    def encode(self, soc: np.ndarray, delivered: np.ndarray) -> np.ndarray:
        rows = np.arange(self.n_agents)[:, None]
        got = delivered[rows, self._peers].astype(np.float32)  # (n, n-1)
        values = soc[self._peers] * got                        # 0 when dropped
        mask = np.ones_like(got) if self.force_mask else got
        out = np.empty((self.n_agents, 2 * (self.n_agents - 1)), dtype=np.float32)
        out[:, 0::2] = values
        out[:, 1::2] = mask
        return out


@dataclass
class DeliverySchedule:
    """Chooses the delivery probability at the start of each episode.

    ``fixed`` -> always ``p`` (M1 trains with p = 1).
    ``random`` -> uniform draw from ``choices`` per episode (M1-RND).
    """

    mode: str = "fixed"
    p: float = 1.0
    choices: tuple = RND_P_CHOICES
    seed: Optional[int | np.random.SeedSequence] = None

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "random"):
            raise ValueError(f"unknown schedule mode {self.mode!r}")
        if self.mode == "random" and len(self.choices) == 0:
            raise ValueError("random schedule needs at least one choice")
        for c in (self.choices if self.mode == "random" else (self.p,)):
            if not 0.0 <= float(c) <= 1.0:
                raise ValueError(f"delivery probability out of range: {c}")
        self.rng = np.random.default_rng(self.seed)

    def sample(self) -> float:
        if self.mode == "fixed":
            return float(self.p)
        return float(self.choices[int(self.rng.integers(len(self.choices)))])
