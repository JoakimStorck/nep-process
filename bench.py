"""Mätning för Steg 5: kostnad per tick och per floraindivid."""
from __future__ import annotations
import argparse, cProfile, io, pstats, time

import numpy as np

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams
import invariants


def build(seed=1, w=64, h=256, init_pop=12, ratio=None):
    WP = WorldParams(width=w, height=h, dt=0.02)
    PP = PopParams(init_pop=init_pop, max_pop=256)
    if ratio is not None:
        PP.flora_init_mass_ratio = float(ratio)
    return Population(WP=WP, AP=AgentParams(dt=WP.dt), PP=PP, seed=seed, hub=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--ratio", type=float, default=None, help="flora_init_mass_ratio")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()

    pop = build(ratio=a.ratio)
    n_flora = int(invariants.diagnostics(pop)["flora_n"])
    for _ in range(int(a.warmup)):
        pop.step()

    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
    t0 = time.perf_counter()
    for _ in range(int(a.ticks)):
        pop.step()
    ms = (time.perf_counter() - t0) / int(a.ticks) * 1000.0
    if a.profile:
        pr.disable()

    d = invariants.diagnostics(pop)
    nb = invariants.nutrient_balance(pop)
    rep = invariants.check_all(pop)
    viol = getattr(rep, "violations", rep)
    nf = int(d["flora_n"])
    print(f"flora {n_flora} -> {nf}, fauna {d['fauna_n']}: {ms:.2f} ms/tick "
          f"({ms*1000/max(1,nf):.2f} us/floraindivid) "
          f"drift={nb['unaccounted']/max(1e-12,abs(nb['total'])):.2e} "
          f"brott={len(viol)}")

    if a.profile:
        sio = io.StringIO()
        pstats.Stats(pr, stream=sio).sort_stats("cumulative").print_stats(14)
        print("\n".join(sio.getvalue().splitlines()[4:24]))


if __name__ == "__main__":
    main()
