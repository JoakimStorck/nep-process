# genetics.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mlp import MLPGenome
from phenotype import derive_pheno, _T_DIET, _T_PREDATION


def _arch_from_traits(
    traits: np.ndarray | None,
    obs_dim: int,
    act_dim: int,
    h_dim: int,
) -> tuple[int, int]:
    """
    Härleder de dolda lagernas bredder från traits via fenotypen.
    Returnerar (hidden_1, hidden_2).
    """
    if traits is None:
        return 24, 24
    p = derive_pheno(traits)
    return int(p.hidden_1), int(p.hidden_2)


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

    # trophic traits mutation (diet/predation)
    # Dessa traits representerar en djupare livsformsdisposition och ska därför
    # driva långsammare än övriga traits. Avkommor ska normalt likna föräldrarna
    # i trophic bias; större skiften ska kräva många generationer.
    trophic_sigma: float = 0.005
    trophic_p: float = 0.02

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
    Asexuell ärftlighet med mutation.
    Returnerar nytt MLPGenome (parent rörs ej).

    Arkitekturmutation:
      Traits muteras först. Om de muterade traits ger en annan nätverksform
      initieras vikterna slumpmässigt — kapaciteten ärvs, inte kopplingsmönstret.
      Biologisk motivering: hjärnans storlek är genetisk, men synapskopplingar
      bildas från grunden under individens liv.
    """
    g = parent.copy()
    ensure_initialized(g, rng, cfg)

    if cfg.allow_arch_mutation and cfg.arch_p > 0.0 and rng.random() < cfg.arch_p:
        g = g.mutate_architecture(rng)
        ensure_initialized(g, rng, cfg)

    # Mutera vikter och traits
    _mutate_weights(g, rng, sigma=cfg.weights_sigma, p=cfg.weights_p)
    _mutate_traits(
        g,
        rng,
        sigma=cfg.traits_sigma,
        p=cfg.traits_p,
        clip=cfg.traits_clip,
        trophic_sigma=cfg.trophic_sigma,
        trophic_p=cfg.trophic_p,
    )

    # Kontrollera om arkitekturen förändrats via trait-mutation
    h_dim = max(0, int(g.h_dim))
    obs   = int(g.layer_sizes[0]) - h_dim
    act   = int(g.layer_sizes[-1]) - h_dim
    h1_new, h2_new = _arch_from_traits(g.traits, obs, act, h_dim)
    h1_old, h2_old = int(g.layer_sizes[1]), int(g.layer_sizes[2])

    if h1_new != h1_old or h2_new != h2_old:
        # Arkitektur har driftat — ny form med slumpmässiga vikter
        in_dim  = obs + h_dim
        out_dim = act + h_dim
        new_g   = MLPGenome(
            layer_sizes=[in_dim, h1_new, h2_new, out_dim],
            act=g.act,
            h_dim=h_dim,
        )
        new_g.traits = g.traits.copy()
        new_g.init_random(rng, init_traits_if_missing=False)
        return new_g

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
    Mutation appliceras efteråt.

    Arkitekturmutation: om de rekombinerade traits ger en ny nätverksform
    initieras vikterna slumpmässigt (se child_genome_from_parent för motivering).
    Om båda föräldrarna har samma arkitektur och traits inte förändrar formen
    ärvs vikterna via crossover som tidigare.
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

    # --- MLP-vikter: uniform crossover (bara om arkitekturerna matchar) ---
    if (g1.weights is not None and g2.weights is not None and
            len(g1.weights) == len(g2.weights) and
            g1.layer_sizes == g2.layer_sizes):
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

    # Mutation
    _mutate_weights(child, rng, sigma=cfg.weights_sigma, p=cfg.weights_p)
    _mutate_traits(
        child,
        rng,
        sigma=cfg.traits_sigma,
        p=cfg.traits_p,
        clip=cfg.traits_clip,
        trophic_sigma=cfg.trophic_sigma,
        trophic_p=cfg.trophic_p,
    )

    # Kontrollera om arkitekturen förändrats efter mutation
    h_dim   = max(0, int(child.h_dim))
    obs     = int(child.layer_sizes[0]) - h_dim
    act     = int(child.layer_sizes[-1]) - h_dim
    h1_new, h2_new = _arch_from_traits(child.traits, obs, act, h_dim)
    h1_old, h2_old = int(child.layer_sizes[1]), int(child.layer_sizes[2])

    if h1_new != h1_old or h2_new != h2_old:
        in_dim  = obs + h_dim
        out_dim = act + h_dim
        new_child = MLPGenome(
            layer_sizes=[in_dim, h1_new, h2_new, out_dim],
            act=child.act,
            h_dim=h_dim,
        )
        new_child.traits = child.traits.copy()
        new_child.init_random(rng, init_traits_if_missing=False)
        return new_child

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


def _mutate_traits(
    g: MLPGenome,
    rng: np.random.Generator,
    sigma: float,
    p: float,
    clip: float,
    trophic_sigma: float,
    trophic_p: float,
) -> None:
    if g.traits is None:
        return
    t = g.traits

    # Basmutation för de flesta traits
    mt = rng.random(t.shape) < float(p)

    # Trophic traits (diet/predation) ska vara mer ärftliga och ändras långsammare.
    # Överskriv därför deras mutationsregim med separata, lägre nivåer.
    if t.shape[0] > _T_DIET:
        mt[_T_DIET] = bool(rng.random() < float(trophic_p))
    if t.shape[0] > _T_PREDATION:
        mt[_T_PREDATION] = bool(rng.random() < float(trophic_p))

    # Basmutation för icke-trophiska traits
    non_trophic = mt.copy()
    if t.shape[0] > _T_DIET:
        non_trophic[_T_DIET] = False
    if t.shape[0] > _T_PREDATION:
        non_trophic[_T_PREDATION] = False

    if non_trophic.any():
        t[non_trophic] += rng.normal(0.0, float(sigma), size=int(non_trophic.sum())).astype(np.float32)

    if t.shape[0] > _T_DIET and mt[_T_DIET]:
        t[_T_DIET] += np.float32(rng.normal(0.0, float(trophic_sigma)))

    if t.shape[0] > _T_PREDATION and mt[_T_PREDATION]:
        t[_T_PREDATION] += np.float32(rng.normal(0.0, float(trophic_sigma)))

    if clip is not None and float(clip) > 0.0:
        t[:] = np.clip(t, -float(clip), float(clip)).astype(np.float32)

    g.traits = t