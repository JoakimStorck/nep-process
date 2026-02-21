# mlp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np


def act_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def act_softsign(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.abs(x))


_ACTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": act_tanh,
    "softsign": act_softsign,
}


@dataclass
class MLPGenome:
    """
    Genotype = policy network + traits.

    - Policy: (layer_sizes, act, weights, biases)
    - Traits: real-valued vector (float32), interpreted elsewhere (phenotype.py)

    NOTE:
      * This module does NOT define how traits affect behavior/life history.
      * That mapping is centralized in phenotype.py.
    """

    layer_sizes: List[int]
    act: str = "tanh"

    weights: List[np.ndarray] | None = None
    biases: List[np.ndarray] | None = None

    traits: np.ndarray | None = None  # shape [n_traits], float32

    # -------------------------
    # init/copy
    # -------------------------

    def init_traits(
        self,
        rng: np.random.Generator,
        n_traits: int,
        lo: float = -1.0,
        hi: float = 1.0,
    ) -> "MLPGenome":
        self.traits = rng.uniform(lo, hi, size=(int(n_traits),)).astype(np.float32)
        return self

    def init_random(
        self,
        rng: np.random.Generator,
        scale: float = 0.6,
        n_traits: int | None = None,
        init_traits_if_missing: bool = True,
    ) -> "MLPGenome":
        """
        Initialize policy weights/biases (uniform fan-in scaled).
        Traits are only created if missing (unless init_traits_if_missing=False).
        """
        self.weights = []
        self.biases = []
        for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            lim = float(scale) / max(1.0, float(np.sqrt(a)))
            W = rng.uniform(-lim, lim, size=(b, a)).astype(np.float32)
            bias = rng.uniform(-lim, lim, size=(b,)).astype(np.float32)
            self.weights.append(W)
            self.biases.append(bias)

        if init_traits_if_missing and self.traits is None:
            if n_traits is None:
                raise ValueError("init_random: n_traits måste anges när traits saknas.")
            self.init_traits(rng, n_traits=int(n_traits))
            
        return self

    def copy(self) -> "MLPGenome":
        g = MLPGenome(layer_sizes=list(self.layer_sizes), act=str(self.act))
        g.weights = [w.copy() for w in (self.weights or [])]
        g.biases = [b.copy() for b in (self.biases or [])]
        g.traits = None if self.traits is None else self.traits.copy()
        return g

    # -------------------------
    # forward
    # -------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass. Returns raw output (no output activation).
        """
        assert self.weights is not None and self.biases is not None
        h = x.astype(np.float32, copy=False)
        act_fn = _ACTS.get(self.act, act_tanh)
        L = len(self.weights)
        for i in range(L):
            h = (self.weights[i] @ h) + self.biases[i]
            if i < L - 1:
                h = act_fn(h)
        return h

    # -------------------------
    # lightweight introspection
    # -------------------------

    def n_traits(self) -> int:
        return 0 if self.traits is None else int(self.traits.shape[0])

    def policy_key(self) -> str:
        return f"{'-'.join(map(str, self.layer_sizes))}:{self.act}"