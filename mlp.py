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
    Genotype = policy network + traits + recurrent state spec.

    - Policy:  (layer_sizes, act, weights, biases)
    - Traits:  real-valued vector (float32), interpreted in phenotype.py
    - Recurrence: h_dim > 0 aktiverar rekurrent minne.

    Rekurrent design ("split output"):
      input  = concat(obs, h)          shape: (obs_dim + h_dim,)
      output = net(input)              shape: (act_dim + h_dim,)
      y      = output[:act_dim]        policy actions
      h_new  = tanh(output[act_dim:])  nytt minnestillstånd ∈ (−1,+1)

    layer_sizes lagrar de FAKTISKA nätverksdimensionerna inklusive h:
      layer_sizes[0]  = obs_dim + h_dim
      layer_sizes[-1] = act_dim + h_dim

    obs_dim() och act_dim() returnerar de rena bio-dimensionerna.

    Energikostnad:
      n_params() returnerar totalt antal vikter+biaser.
      Population skalar compute_cost proportionellt mot n_params / ref_params,
      så bredare hjärnor kostar mer metabolt per sekund.
    """

    layer_sizes: List[int]
    act: str = "tanh"
    h_dim: int = 0          # rekurrent minnesdimension; 0 = ingen rekurrens

    weights: List[np.ndarray] | None = None
    biases:  List[np.ndarray] | None = None
    traits:  np.ndarray | None = None  # shape (n_traits,), float32

    # -------------------------
    # Dimensionshjälpare
    # -------------------------

    def obs_dim(self) -> int:
        """Ren observationsdimension (layer_sizes[0] minus h_dim)."""
        return int(self.layer_sizes[0]) - max(0, int(self.h_dim))

    def act_dim(self) -> int:
        """Ren aktionsdimension (layer_sizes[-1] minus h_dim)."""
        return int(self.layer_sizes[-1]) - max(0, int(self.h_dim))

    def n_params(self) -> int:
        """
        Totalt antal träningsbara parametrar (vikter + biaser).
        Används för att skala compute_cost proportionellt mot hjärnstorlek.
        """
        if self.weights is None or self.biases is None:
            return 0
        return (
            sum(int(w.size) for w in self.weights) +
            sum(int(b.size) for b in self.biases)
        )

    # -------------------------
    # Init / copy
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
        Initialiserar vikter/biaser med uniform fan-in-skalning.

        Rekurrenta outputvikter (sista lagrets h_dim rader) dämpas med
        faktor 0.1 vid initialisering — undviker kaotisk h-dynamik i
        början av körningen innan evolution stabiliserat nätverket.
        """
        self.weights = []
        self.biases  = []
        for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            lim = float(scale) / max(1.0, float(np.sqrt(a)))
            W    = rng.uniform(-lim, lim, size=(b, a)).astype(np.float32)
            bias = rng.uniform(-lim, lim, size=(b,)).astype(np.float32)
            self.weights.append(W)
            self.biases.append(bias)

        # Dämpa rekurrenta outputvikter i sista lagret
        h = max(0, int(self.h_dim))
        if h > 0 and self.weights:
            self.weights[-1][-h:] *= np.float32(0.1)
            self.biases[-1][-h:]  *= np.float32(0.1)

        if init_traits_if_missing and self.traits is None:
            if n_traits is None:
                raise ValueError("init_random: n_traits måste anges när traits saknas.")
            self.init_traits(rng, n_traits=int(n_traits))

        return self

    def copy(self) -> "MLPGenome":
        g = MLPGenome(
            layer_sizes=list(self.layer_sizes),
            act=str(self.act),
            h_dim=int(self.h_dim),
        )
        g.weights = [w.copy() for w in (self.weights or [])]
        g.biases  = [b.copy() for b in (self.biases  or [])]
        g.traits  = None if self.traits is None else self.traits.copy()
        return g

    # -------------------------
    # Forward pass
    # -------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward-pass. Returnerar rå output utan outputaktivering.
        Vid rekurrens: x = concat(obs, h), output = concat(y, h_raw).
        Agenten ansvarar för att splitta och applicera tanh på h-delen.
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
    # Introspection
    # -------------------------

    def n_traits(self) -> int:
        return 0 if self.traits is None else int(self.traits.shape[0])

    def policy_key(self) -> str:
        return f"{'-'.join(map(str, self.layer_sizes))}:{self.act}"