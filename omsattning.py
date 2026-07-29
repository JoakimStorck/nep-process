"""Mäter om faunans stationära antal är samma individer eller ett flödesjämviktstal."""
from __future__ import annotations
import argparse
import numpy as np

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams
import invariants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--from-tick", type=int, default=4000,
                    help="kohorten fryses här, efter inkörningen")
    a = ap.parse_args()

    WP = WorldParams(width=64, height=256, dt=0.02)
    pop = Population(WP=WP, AP=AgentParams(dt=WP.dt),
                     PP=PopParams(init_pop=12, max_pop=256), seed=int(a.seed), hub=None)

    seen: set[int] = set()
    cohort: set[int] = set()
    ages_at_death: list[float] = []
    prev: dict[int, float] = {}
    n_deaths = 0

    for t in range(1, int(a.ticks) + 1):
        alive_before = dict(prev)
        pop.step()

        now = {}
        for x in pop.agents:
            if not x.body.alive:
                continue
            oid = int(getattr(x, "id", id(x)))
            slot = int(getattr(x, "store_slot", -1))
            age = float(pop.store.age[slot]) if slot >= 0 else 0.0
            now[oid] = age
            seen.add(oid)

        for oid, age in alive_before.items():
            if oid not in now:
                n_deaths += 1
                ages_at_death.append(age)
        prev = now

        if t == int(a.from_tick):
            cohort = set(now.keys())

    final = set(prev.keys())
    kvar = cohort & final

    print(f"seed {a.seed}, {a.ticks} tick ({a.ticks * WP.dt:.0f} s simulerad tid)")
    print(f"  individer vid tick {a.from_tick}: {len(cohort)}")
    print(f"  individer vid slutet:            {len(final)}")
    print(f"  av kohorten kvar vid slutet:     {len(kvar)}  ({100.0*len(kvar)/max(1,len(cohort)):.0f} %)")
    print(f"  unika individer totalt:          {len(seen)}")
    print(f"  dödsfall totalt:                 {n_deaths}")
    if ages_at_death:
        arr = np.array(ages_at_death)
        print(f"  ålder vid död: median {np.median(arr):.1f} s, max {arr.max():.1f} s")
    nb = invariants.nutrient_balance(pop)
    print(f"  näringsdrift: {nb['unaccounted']/max(1e-12,abs(nb['total'])):.3e} rel")


if __name__ == "__main__":
    main()
