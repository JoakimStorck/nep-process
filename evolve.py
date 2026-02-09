from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from mlp import MLPGenome
from run import run_episode


@dataclass
class EvoParams:
    pop: int = 32
    elites: int = 6
    children: int = 26
    generations: int = 25

    # mutation knobs
    p_arch: float = 0.20
    p_weights: float = 1.00
    w_sigma: float = 0.09
    w_p: float = 0.12

    # evaluation
    T: float = 1200.0
    size: int = 64
    base_seed: int = 10
    n_rollouts: int = 2


def eval_genome(g: MLPGenome, EP: EvoParams, gen_seed: int) -> float:
    fs = []
    for k in range(EP.n_rollouts):
        seed = EP.base_seed + 1000 * gen_seed + 17 * k
        r = run_episode(T=EP.T, seed=seed, size=EP.size, log_fp="__tmp.jsonl", genome=g)
        fs.append(float(r["fitness"]))
    return float(sum(fs) / len(fs))


def make_initial_population(rng: np.random.Generator, pop: int, in_dim: int, out_dim: int) -> List[MLPGenome]:
    P: List[MLPGenome] = []
    for _ in range(pop):
        g = MLPGenome(layer_sizes=[in_dim, 24, 24, out_dim], act="tanh").init_random(rng)
        P.append(g)
    return P


def tournament(rng: np.random.Generator, genomes: List[MLPGenome], scores: List[float], k: int = 3) -> MLPGenome:
    idx = rng.integers(0, len(genomes), size=(k,))
    best = max(idx, key=lambda i: scores[int(i)])
    return genomes[int(best)].copy()


def evolve(EP: EvoParams) -> Tuple[MLPGenome, List[float]]:
    rng = np.random.default_rng(EP.base_seed)

    in_dim = 23
    out_dim = 5
    pop = make_initial_population(rng, EP.pop, in_dim, out_dim)

    best_hist: List[float] = []

    for gen in range(EP.generations):
        scores = [eval_genome(g, EP, gen_seed=gen * 100 + i) for i, g in enumerate(pop)]
        order = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)

        best = scores[order[0]]
        best_hist.append(best)
        print(f"[gen {gen:02d}] best={best:.2f}  median={float(np.median(scores)):.2f}")

        elites = [pop[i].copy() for i in order[:EP.elites]]

        children: List[MLPGenome] = []
        while len(children) < EP.children:
            parent = tournament(rng, pop, scores, k=3)

            if rng.random() < EP.p_arch:
                parent = parent.mutate_architecture(rng).init_random(rng)

            if EP.p_weights > 0.0:
                parent.mutate_weights(rng, sigma=EP.w_sigma, p=EP.w_p)

            children.append(parent)

        pop = elites + children

    scores = [eval_genome(g, EP, gen_seed=9999 * 17 + i) for i, g in enumerate(pop)]
    i_best = int(np.argmax(scores))
    return pop[i_best], best_hist


if __name__ == "__main__":
    EP = EvoParams()
    best, hist = evolve(EP)
    print("DONE. best genome:", best.layer_sizes, best.act, "best_hist_last=", hist[-1])