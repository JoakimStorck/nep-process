from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np

from world import World, WorldParams
from mlp import MLPGenome
from agent import Agent, AgentParams
from observer import Observer


def run_episode(
    T: float = 1200.0,
    seed: int = 3,
    size: int = 64,
    log_fp: str = "steps.jsonl",
    genome: MLPGenome | None = None,
    tick: float = 2.0,
) -> dict:
    random.seed(seed)
    rng = np.random.default_rng(seed)

    WP = WorldParams(size=size, dt=0.02)
    world = World(WP)

    AP = AgentParams(dt=WP.dt)

    in_dim = 23
    out_dim = 5

    if genome is None:
        genome = MLPGenome(layer_sizes=[in_dim, 24, 24, out_dim], act="tanh").init_random(rng)

    agent = Agent(
        AP=AP,
        genome=genome,
        x=size / 2,
        y=size / 2,
        heading=random.uniform(-math.pi, math.pi),
    )
    agent.bind_world(world)

    obs = Observer(fp=log_fp, every_s=0.5)
    t = 0.0

    fitness = 0.0
    hazard_residency = 0.0

    next_tick_t = 0.0
    t0 = time.perf_counter()

    print(f"START: T={T} dt={WP.dt} size={size} seed={seed}", flush=True)

    while t < T and agent.body.alive:
        world.step()
        B0, F0, C0 = agent.step(world)
        t += WP.dt

        obs.maybe_log(t, agent, B0, F0, C0)

        E = agent.body.E_total()
        D = agent.body.D

        hazard_residency += float(F0) * WP.dt

        fitness += (
            1.0 * WP.dt
            + 0.25 * float(E) * WP.dt
            - 0.80 * float(D) * WP.dt
            - 0.15 * float(F0) * WP.dt
        )

        if tick > 0.0 and t >= next_tick_t:
            next_tick_t = t + tick
            wall = time.perf_counter() - t0
            print(
                f"t={t:8.2f}  wall={wall:7.2f}s  alive={agent.body.alive}  "
                f"E={agent.body.E_total():.3f} D={agent.body.D:.3f} Fg={agent.body.Fg:.3f} "
                f"B0={B0:.3f} F0={F0:.3f} C0={C0:.3f}",
                flush=True,
            )

    return {
        "alive": agent.body.alive,
        "t_end": float(t),
        "E_end": float(agent.body.E_total()),
        "D_end": float(agent.body.D),
        "Fg_end": float(agent.body.Fg),
        "fitness": float(fitness),
        "hazard_proxy": float(hazard_residency),
        "genome": genome,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--log", type=str, default="steps.jsonl")
    ap.add_argument("--tick", type=float, default=2.0, help="Seconds between stdout status lines (sim-time).")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    res = run_episode(T=a.T, seed=a.seed, size=a.size, log_fp=a.log, genome=None, tick=a.tick)
    print(
        f"END: alive={res['alive']} t={res['t_end']:.2f} fitness={res['fitness']:.2f} "
        f"E={res['E_end']:.3f} D={res['D_end']:.3f} Fg={res['Fg_end']:.3f}",
        flush=True,
    )