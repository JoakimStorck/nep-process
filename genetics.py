# genetics.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mlp import MLPGenome


@dataclass(frozen=True)
class MutationConfig:
    """
    Controls inheritance + mutation.

    A (now):
      - copy parent genome
      - mutate traits (slow drift)
      - mutate weights/biases (policy noise)

    B (later):
      - optional architecture mutation hooks (kept off by default)
    """

    # traits mutation
    traits_sigma: float = 0.02
    traits_p: float = 0.05
    traits_clip: float = 2.0  # clip to [-clip, clip]

    # weights mutation
    weights_sigma: float = 0.08
    weights_p: float = 0.10

    # architecture mutation (off by default)
    allow_arch_mutation: bool = False
    arch_p: float = 0.0  # probability to attempt mutate_architecture when enabled

    # init ranges (only used if missing traits)
    n_traits: int = 12
    traits_init_lo: float = -1.0
    traits_init_hi: float = 1.0


def ensure_initialized(g: MLPGenome, rng: np.random.Generator, cfg: MutationConfig) -> None:
    if g.weights is None or g.biases is None:
        g.init_random(rng, scale=0.6, n_traits=cfg.n_traits, init_traits_if_missing=False)
    if g.traits is None:
        g.init_traits(rng, n_traits=cfg.n_traits, lo=cfg.traits_init_lo, hi=cfg.traits_init_hi)

def child_genome_from_parent(parent: MLPGenome, rng: np.random.Generator, cfg: MutationConfig) -> MLPGenome:
    """
    A: asexual inheritance with mutation.
    Returns a new MLPGenome (parent is not modified).
    """
    g = parent.copy()

    # Ensure child genome is initialized (do NOT touch parent)
    ensure_initialized(g, rng, cfg)

    # optional B-hook: architecture mutation (off)
    if cfg.allow_arch_mutation and cfg.arch_p > 0.0 and rng.random() < cfg.arch_p:
        g = g.mutate_architecture(rng)
        # new architecture => random init of policy; preserve traits by design
        ensure_initialized(g, rng, cfg)

    # mutate policy parameters
    _mutate_weights(g, rng, sigma=cfg.weights_sigma, p=cfg.weights_p)

    # mutate traits
    _mutate_traits(g, rng, sigma=cfg.traits_sigma, p=cfg.traits_p, clip=cfg.traits_clip)

    return g

def _mutate_weights(g: MLPGenome, rng: np.random.Generator, sigma: float, p: float) -> None:
    assert g.weights is not None and g.biases is not None
    for i in range(len(g.weights)):
        W = g.weights[i]
        B = g.biases[i]
        mW = rng.random(W.shape) < float(p)
        mB = rng.random(B.shape) < float(p)
        if mW.any():
            W[mW] += rng.normal(0.0, float(sigma), size=int(mW.sum())).astype(np.float32)
        if mB.any():
            B[mB] += rng.normal(0.0, float(sigma), size=int(mB.sum())).astype(np.float32)


def _mutate_traits(g: MLPGenome, rng: np.random.Generator, sigma: float, p: float, clip: float) -> None:
    if g.traits is None:
        return
    t = g.traits
    mt = rng.random(t.shape) < float(p)
    if mt.any():
        t[mt] += rng.normal(0.0, float(sigma), size=int(mt.sum())).astype(np.float32)
        if clip is not None and float(clip) > 0.0:
            t[:] = np.clip(t, -float(clip), float(clip)).astype(np.float32)
    g.traits = t