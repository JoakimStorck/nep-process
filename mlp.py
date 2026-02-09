# mlp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np


def act_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def act_softsign(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.abs(x))


_ACTS = {
    "tanh": act_tanh,
    "softsign": act_softsign,
}


@dataclass
class MLPGenome:
    """
    Genotype = (policy network) + (traits vector).

    Policy (NN):
      - layer_sizes: [in, h1, h2, ..., out]
      - act: activation name for hidden layers
      - weights/biases: list per layer

    Traits:
      - real-valued vector, mutated slowly
      - interpreted in Agent.apply_traits() as bounded modifiers of AgentParams
    """

    layer_sizes: List[int]
    act: str = "tanh"
    weights: List[np.ndarray] | None = None
    biases: List[np.ndarray] | None = None

    traits: np.ndarray | None = None  # shape [n_traits], float32

    def init_traits(self, rng: np.random.Generator, n_traits: int = 12, lo: float = -1.0, hi: float = 1.0) -> "MLPGenome":
        self.traits = rng.uniform(lo, hi, size=(int(n_traits),)).astype(np.float32)
        return self

    def init_random(
        self,
        rng: np.random.Generator,
        scale: float = 0.6,
        n_traits: int = 12,
        init_traits_if_missing: bool = True,
    ) -> "MLPGenome":
        """
        Initialize weights/biases. Traits are only created if missing (unless init_traits_if_missing=False).
        This keeps phenotype stable across architecture rewiring if desired.
        """
        self.weights = []
        self.biases = []
        for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            lim = scale / max(1.0, np.sqrt(a))
            W = rng.uniform(-lim, lim, size=(b, a)).astype(np.float32)
            bias = rng.uniform(-lim, lim, size=(b,)).astype(np.float32)
            self.weights.append(W)
            self.biases.append(bias)

        if init_traits_if_missing and self.traits is None:
            self.init_traits(rng, n_traits=n_traits)

        return self

    def copy(self) -> "MLPGenome":
        g = MLPGenome(layer_sizes=list(self.layer_sizes), act=str(self.act))
        g.weights = [w.copy() for w in (self.weights or [])]
        g.biases = [b.copy() for b in (self.biases or [])]
        g.traits = None if self.traits is None else self.traits.copy()
        return g

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert self.weights is not None and self.biases is not None
        h = x.astype(np.float32, copy=False)
        act_fn: Callable[[np.ndarray], np.ndarray] = _ACTS.get(self.act, act_tanh)
        L = len(self.weights)
        for i in range(L):
            h = (self.weights[i] @ h) + self.biases[i]
            if i < L - 1:
                h = act_fn(h)
        return h  # raw output

    def mutate_weights(self, rng: np.random.Generator, sigma: float = 0.08, p: float = 0.10) -> None:
        """
        Mutate only policy network parameters (weights/biases).
        Traits are mutated separately via mutate_traits().
        """
        assert self.weights is not None and self.biases is not None
        for i in range(len(self.weights)):
            W = self.weights[i]
            B = self.biases[i]
            mW = rng.random(W.shape) < p
            mB = rng.random(B.shape) < p
            if mW.any():
                W[mW] += rng.normal(0.0, sigma, size=int(mW.sum())).astype(np.float32)
            if mB.any():
                B[mB] += rng.normal(0.0, sigma, size=int(mB.sum())).astype(np.float32)

    def mutate_traits(self, rng: np.random.Generator, sigma: float = 0.02, p: float = 0.05, clip: float = 2.0) -> None:
        """
        Slow phenotypic drift: sparse, small Gaussian perturbations on traits.
        """
        if self.traits is None:
            return
        t = self.traits
        mt = rng.random(t.shape) < p
        if mt.any():
            t[mt] += rng.normal(0.0, sigma, size=int(mt.sum())).astype(np.float32)
            if clip is not None and clip > 0.0:
                t[:] = np.clip(t, -float(clip), float(clip)).astype(np.float32)
        self.traits = t

    def mutate_architecture(
        self,
        rng: np.random.Generator,
        p_layer: float = 0.15,
        p_width: float = 0.20,
        min_h: int = 6,
        max_h: int = 64,
        max_hidden_layers: int = 4,
    ) -> "MLPGenome":
        """
        Return new genome (copy) with possibly changed hidden layout.
        We keep input/output fixed; only hidden layers mutate.
        Architecture mutation discards old wiring (weights/biases reset),
        but preserves traits (phenotype) by default.
        """
        g = self.copy()
        in_dim = g.layer_sizes[0]
        out_dim = g.layer_sizes[-1]
        hidden = g.layer_sizes[1:-1]

        if rng.random() < 0.10:
            g.act = "softsign" if g.act == "tanh" else "tanh"

        if rng.random() < p_layer:
            if len(hidden) < max_hidden_layers and rng.random() < 0.60:
                w = int(rng.integers(min_h, max_h + 1))
                pos = int(rng.integers(0, len(hidden) + 1))
                hidden = hidden[:pos] + [w] + hidden[pos:]
            elif len(hidden) > 1:
                pos = int(rng.integers(0, len(hidden)))
                hidden = hidden[:pos] + hidden[pos + 1 :]

        if hidden and rng.random() < p_width:
            pos = int(rng.integers(0, len(hidden)))
            delta = int(rng.integers(-8, 9))
            hidden[pos] = int(np.clip(hidden[pos] + delta, min_h, max_h))

        g.layer_sizes = [in_dim] + list(hidden) + [out_dim]
        g.weights, g.biases = None, None
        return g