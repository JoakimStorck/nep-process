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
    n_traits: int = 18
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

def recombine(
    g1: MLPGenome,
    g2: MLPGenome,
    rng: np.random.Generator,
    cfg: MutationConfig,
) -> MLPGenome:
    """
    Sexuell rekombination: kombinerar gener från två föräldrar.

    Traits: uniform crossover — varje trait väljs oberoende från g1 eller g2.
    MLP-vikter: per-parameter uniform crossover (inte 50/50-medelvärde,
                vilket komprimerar viktrymden och hämmar evolution).
    Mutation appliceras efteråt, precis som vid asexuell reproduktion.
    """
    ensure_initialized(g1, rng, cfg)
    ensure_initialized(g2, rng, cfg)

    child = g1.copy()

    # --- Traits: uniform crossover ---
    if g1.traits is not None and g2.traits is not None:
        t1, t2 = g1.traits, g2.traits
        n = min(len(t1), len(t2))
        mask = rng.random(n) < 0.5
        child.traits = t1.copy()
        child.traits[:n] = np.where(mask, t1[:n], t2[:n]).astype(np.float32)

    # --- MLP-vikter: uniform crossover per parameter ---
    if (g1.weights is not None and g2.weights is not None and
            len(g1.weights) == len(g2.weights)):
        child.weights = [w.copy() for w in g1.weights]
        child.biases  = [b.copy() for b in g1.biases]
        for i in range(len(g1.weights)):
            W1, W2 = g1.weights[i], g2.weights[i]
            B1, B2 = g1.biases[i],  g2.biases[i]
            if W1.shape == W2.shape:
                mW = rng.random(W1.shape) < 0.5
                child.weights[i] = np.where(mW, W1, W2).astype(np.float32)
            if B1.shape == B2.shape:
                mB = rng.random(B1.shape) < 0.5
                child.biases[i]  = np.where(mB, B1, B2).astype(np.float32)

    # Mutation efteråt
    _mutate_weights(child, rng, sigma=cfg.weights_sigma, p=cfg.weights_p)
    _mutate_traits(child, rng, sigma=cfg.traits_sigma, p=cfg.traits_p, clip=cfg.traits_clip)

    return child


def genetic_compatibility(g1: MLPGenome, g2: MLPGenome, sigma: float = 2.0) -> float:
    """
    Beräknar parningssannolikhet (0–1) baserad på genetiskt avstånd i trait-rymden.

    Modell: Gaussisk avtagning med normaliserat kvadratiskt avstånd.

        P = exp( −d²_norm / (2·σ²) )

    där d²_norm = (1/n) · Σ(t1_i − t2_i)² är medel-kvadratavståndet per trait.

    Traits klipps till [−2, 2] via MutationConfig — maximalt avstånd per trait = 4,
    d²_norm_max ≈ 4.0.

    Intuition för σ-val (vid random-initialiserade traits, ej ännu divergerade):
        σ = 3.0 → permissiv  — P(max_dist) ≈ 0.80   (nästan alla kan para sig)
        σ = 2.0 → balanserad — P(max_dist) ≈ 0.61
        σ = 1.0 → moderat    — P(max_dist) ≈ 0.14   (tydlig preferens för likhet)
        σ = 0.5 → strikt     — P(max_dist) ≈ 0.0003 (hård artgräns)

    Rekommendation: starta med σ=2.0 tills populationen stabiliseras,
    sänk sedan gradvis mot 0.5–1.0 för att driva artbildning.

    Returnerar 1.0 om traits saknas (permissivt fallback).
    """
    t1, t2 = g1.traits, g2.traits
    if t1 is None or t2 is None:
        return 1.0
    n = min(int(t1.shape[0]), int(t2.shape[0]))
    if n == 0:
        return 1.0

    diff = t1[:n].astype(np.float64) - t2[:n].astype(np.float64)
    d_sq_norm = float(np.dot(diff, diff)) / n          # normaliserat per trait
    sigma_sq = max(float(sigma), 1e-9) ** 2
    return float(np.exp(-d_sq_norm / (2.0 * sigma_sq)))


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