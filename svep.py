"""
Täthetssvep: finns en täthet där både mötesfrekvens och födobas går ihop?

Under en viss täthet binder mötesfrekvensen — agenterna är reproduktionsklara
men ser aldrig varandra, och beståndet kollapsar oavsett födotillgång (Allee).
Över en viss täthet binder födan. Frågan är om fönstret mellan dem existerar.

    python svep.py --width 64 --height 256 --init-pop 40 --ticks 4000
"""
from __future__ import annotations
import argparse

import numpy as np

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams
import invariants

C = {"redo": 0, "agenttick": 0, "ser_ingen": 0, "utanfor_radie": 0, "parning": 0}


def instrument():
    ready = Population._ready_to_reproduce_slot
    dist = Population._slot_distance
    agent_for = Population._agent_for_slot
    orig = Population._try_mating

    def wrapped(self, agent, ctx, candidates=None):
        if not agent.body.alive:
            return
        a_slot = int(agent.store_slot)
        if a_slot < 0:
            return
        C["agenttick"] += 1
        if not ready(self, a_slot):
            return
        C["redo"] += 1

        hit = getattr(agent, "_cached_agent_hit", None)
        ok = isinstance(hit, tuple) and len(hit) >= 5
        best = None
        if ok:
            _, _, _, hit_slot, desired_id = hit[:5]
            s = int(hit_slot)
            store = self.store
            if (s >= 0 and int(desired_id) > 0 and s < int(store.n)
                    and bool(store.alive[s]) and int(store.kind[s]) == 0
                    and int(store.id[s]) == int(desired_id)):
                cand = agent_for(self, s)
                if cand is not None and cand is not agent and int(cand.store_slot) >= 0:
                    best = cand
        if best is None:
            C["ser_ingen"] += 1
            return
        if dist(self, a_slot, int(best.store_slot), squared=True) > float(self.PP.mating_radius) ** 2:
            C["utanfor_radie"] += 1
            return

        before = bool(agent.body.gestating) or bool(best.body.gestating)
        orig(self, agent, ctx, candidates)
        if (bool(agent.body.gestating) or bool(best.body.gestating)) and not before:
            C["parning"] += 1

    Population._try_mating = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-pop", type=int, default=12)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--max-pop", type=int, default=0, help="0 = 4x init_pop, minst 256")
    ap.add_argument("--nutrient-input", type=float, default=0.0, help="0 = WorldParams default")
    a = ap.parse_args()

    instrument()
    WP = WorldParams(width=int(a.width), height=int(a.height), dt=0.02)
    if float(a.nutrient_input) > 0.0:
        WP.nutrient_input = float(a.nutrient_input)
    max_pop = int(a.max_pop) or max(256, 4 * int(a.init_pop))
    pop = Population(WP=WP, AP=AgentParams(dt=WP.dt),
                     PP=PopParams(init_pop=int(a.init_pop), max_pop=max_pop),
                     seed=int(a.seed), hub=None)

    n_cells = int(pop.grid.n_cells)
    n0 = len([x for x in pop.agents if x.body.alive])
    ids = {int(x.id) for x in pop.agents if x.body.alive}
    traj = []

    for t in range(1, int(a.ticks) + 1):
        pop.step()
        for x in pop.agents:
            if x.body.alive:
                ids.add(int(x.id))
        if t % max(1, int(a.ticks) // 4) == 0:
            traj.append(len([x for x in pop.agents if x.body.alive]))

    n1 = len([x for x in pop.agents if x.body.alive])
    d = invariants.diagnostics(pop)
    nb = invariants.nutrient_balance(pop)
    births = int(getattr(pop, "_births_total", 0))
    deaths = int(getattr(pop, "_deaths_total", 0))

    ms = [float(x.body.M) for x in pop.agents if x.body.alive]
    mt = [float(x.pheno.M_target) for x in pop.agents if x.body.alive]
    rel_m = float(np.mean(np.array(ms) / np.array(mt))) if ms else 0.0

    r = max(1, C["redo"])
    print(
        f"N={WP.nutrient_input:.1e} {a.width}x{a.height} n0={n0:4d} tät={n0/n_cells*1000:6.2f} | "
        f"n={n1:4d} bana={'>'.join(str(v) for v in traj)} | "
        f"föd={births:4d} död={deaths:4d} unika={len(ids):4d} | "
        f"redo={100*C['redo']/max(1,C['agenttick']):4.1f}% "
        f"ser_ingen={100*C['ser_ingen']/r:4.1f}% radie={100*C['utanfor_radie']/r:4.1f}% "
        f"parn={C['parning']:4d} | flora={d['flora_n']:5d} M/M_tgt={rel_m:.2f} "
        f"| Mflora={d['flora_mass_kg']:.3f} Mfauna={d['fauna_mass_kg']:.2f} drift={nb['unaccounted']/max(1e-12,abs(nb['total'])):7.1e}"
    )


if __name__ == "__main__":
    main()
