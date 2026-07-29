"""
Var fastnar reproduktionen? Räknar utfallen i _try_mating() per tick.

Skiljer fysiologisk grind (agenten är inte redo) från rumslig gles het
(agenten är redo men ser ingen, eller ser någon utanför parningsradien).
"""
from __future__ import annotations
import argparse

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams
import invariants

C = {"ej_redo_sjalv": 0, "ingen_sensingtraff": 0, "traff_ogiltig": 0,
     "partner_ej_redo": 0, "utanfor_radie": 0, "kompat_avslag": 0, "parning": 0,
     "redo_totalt": 0, "agenttick": 0}


def instrument(Population):
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
            C["ej_redo_sjalv"] += 1
            return
        C["redo_totalt"] += 1

        hit = getattr(agent, "_cached_agent_hit", None)
        if not isinstance(hit, tuple) or len(hit) < 5:
            C["ingen_sensingtraff"] += 1
            return
        _, _, _, hit_slot, desired_id = hit[:5]
        store = self.store
        s = int(hit_slot)
        if (s < 0 or int(desired_id) <= 0 or s >= int(store.n)
                or not bool(store.alive[s]) or int(store.kind[s]) != 0
                or int(store.id[s]) != int(desired_id)):
            C["traff_ogiltig"] += 1
            return
        best = agent_for(self, s)
        if best is None or best is agent or int(best.store_slot) < 0:
            C["traff_ogiltig"] += 1
            return
        if not ready(self, int(best.store_slot)):
            C["partner_ej_redo"] += 1
            return
        if dist(self, a_slot, int(best.store_slot), squared=True) > float(self.PP.mating_radius) ** 2:
            C["utanfor_radie"] += 1
            return
        before = bool(agent.body.gestating) or bool(best.body.gestating)
        orig(self, agent, ctx, candidates)
        after = bool(agent.body.gestating) or bool(best.body.gestating)
        if after and not before:
            C["parning"] += 1
        else:
            C["kompat_avslag"] += 1

    Population._try_mating = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-pop", type=int, default=12)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--max-pop", type=int, default=256)
    a = ap.parse_args()

    instrument(Population)
    WP = WorldParams(width=int(a.width), height=int(a.height), dt=0.02)
    pop = Population(WP=WP, AP=AgentParams(dt=WP.dt),
                     PP=PopParams(init_pop=int(a.init_pop), max_pop=int(a.max_pop)),
                     seed=int(a.seed), hub=None)
    n_cells = int(pop.grid.n_cells)
    n0 = len(pop.agents)
    for _ in range(int(a.ticks)):
        pop.step()

    d = invariants.diagnostics(pop)
    print(f"seed {a.seed}, {a.width}x{a.height} = {n_cells} celler, init_pop={a.init_pop}, "
          f"{a.ticks} tick")
    print(f"  fauna {n0} -> {d['fauna_n']}, flora {d['flora_n']}, "
          f"täthet {d['fauna_n']/n_cells*1000:.2f} agenter per 1000 celler")
    tot = max(1, C["agenttick"])
    print(f"  agenttick totalt: {C['agenttick']}")
    print(f"  redo att reproducera: {C['redo_totalt']} ({100*C['redo_totalt']/tot:.1f} %)")
    for k in ("ingen_sensingtraff", "traff_ogiltig", "partner_ej_redo",
              "utanfor_radie", "kompat_avslag", "parning"):
        r = max(1, C["redo_totalt"])
        print(f"    {k:>18}: {C[k]:8d} ({100*C[k]/r:5.1f} % av de redo)")


if __name__ == "__main__":
    main()
