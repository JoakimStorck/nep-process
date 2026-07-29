"""Tillfällig mätning: hur stor är faunaläckan i massa och näring?"""
from __future__ import annotations
import argparse
import numpy as np

from world import WorldParams
from agent import AgentParams, Agent, Body
from population import Population, PopParams
from phenotype import nutrient_content
import invariants


def fauna_pools(pop):
    """Massa och näring bunden i fauna, inklusive reserv."""
    m_body = 0.0
    m_res = 0.0
    nut = 0.0
    from phenotype import NUTRIENT_PER_KG_LABILE
    for a in pop.agents:
        b = a.body
        s = float(pop.store.structure[int(a.store_slot)]) if a.store_slot >= 0 else 0.25
        m_body += float(b.M)
        m_res += float(b.M_reserve())
        nut += float(b.M) * nutrient_content(s) + b.M_reserve() * NUTRIENT_PER_KG_LABILE
    return m_body, m_res, nut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    WP = WorldParams(width=64, height=256, dt=0.02)
    AP = AgentParams(dt=WP.dt)
    PP = PopParams(init_pop=12, max_pop=256)
    pop = Population(WP=WP, AP=AP, PP=PP, seed=int(a.seed), hub=None)

    # --- instrumentering ---
    counters = {"ingest": 0.0, "excrete_fauna": 0.0, "carcass": 0.0,
                "growth": 0.0, "catab": 0.0, "burn": 0.0, "overflow": 0.0}

    orig_feed = Agent._perform_feeding

    def feed(self, world, dt, allow_eat):
        out = orig_feed(self, world, dt, allow_eat)
        counters["ingest"] += out[0] + out[1]
        return out

    Agent._perform_feeding = feed

    orig_exc = Agent._excrete

    def exc(self, world, mass_kg, energy_J, diet_eff):
        before = float(getattr(world, "_dM_excreted", 0.0))
        r = orig_exc(self, world, mass_kg, energy_J, diet_eff)
        counters["excrete_fauna"] += float(getattr(world, "_dM_excreted", 0.0)) - before
        return r

    Agent._excrete = exc

    orig_carc = type(pop.world).add_carcass

    def carc(self, x, y, amount_kg, rad=3, structure=0.45):
        counters["carcass"] += float(amount_kg)
        return orig_carc(self, x, y, amount_kg, rad, structure)

    type(pop.world).add_carcass = carc

    orig_step = Body.step

    def bstep(self, ctx, **kw):
        m0 = float(self.M) + float(self.M_reserve())
        orig_step(self, ctx, **kw)
        counters["growth"] += (float(self.M) + float(self.M_reserve())) - m0

    Body.step = bstep

    hdr = (f"{'tick':>7} {'fauna':>5} {'flora':>6} {'M_fauna':>10} {'M_flora':>10} "
           f"{'M_detr':>10} {'nut_free':>10} {'nut_fauna':>10} {'unacc':>11}")
    print(hdr)

    m0_f = fauna_pools(pop)[0] + fauna_pools(pop)[1]
    for t in range(1, int(a.ticks) + 1):
        pop.step()
        if t % int(a.every) == 0 or t == int(a.ticks):
            nb = invariants.nutrient_balance(pop)
            mb, mr, nf = fauna_pools(pop)
            d = invariants.diagnostics(pop)
            print(f"{t:7d} {d['fauna_n']:5d} {d['flora_n']:6d} "
                  f"{mb + mr:10.4e} {d['flora_mass_kg']:10.4e} "
                  f"{float(np.sum(pop.world.detritus, dtype=np.float64)):10.4e} "
                  f"{nb['free']:10.4e} {nf:10.4e} {nb['unaccounted']:11.3e}")

    print()
    print("--- faunans massaflöden över körningen (kg) ---")
    mb, mr, nf = fauna_pools(pop)
    print(f"ingesterad massa (flora+detritus): {counters['ingest']:.4e}")
    print(f"exkreterad massa (fauna):          {counters['excrete_fauna']:.4e}")
    print(f"assimilerad = ingest - exkret:     {counters['ingest'] - counters['excrete_fauna']:.4e}")
    print(f"netto ΔM i Body.step (M+reserv):   {counters['growth']:.4e}")
    print(f"kadavermassa till detritus:        {counters['carcass']:.4e}")
    print(f"levande faunamassa nu (M+reserv):  {mb + mr:.4e}   (vid start {m0_f:.4e})")
    ing = counters["ingest"] - counters["excrete_fauna"]
    made = counters["growth"] - ing
    print()
    print(f"massa skapad ur intet i Body.step: {made:.4e}")
    if ing > 0:
        print(f"kvot skapad/assimilerad:           {made / ing:.1f}x")
    bad = sum(int(getattr(x.body, "ledger_bad_steps", 0)) for x in pop.agents)
    tot = sum(int(getattr(x.body, "ledger_steps", 0)) for x in pop.agents)
    mx = max([float(getattr(x.body, "ledger_max_rel", 0.0)) for x in pop.agents] or [0.0])
    print()
    print(f"--- energiledger: {bad}/{tot} steg utanför tolerans, max_rel {mx:.2e} ---")

    nb = invariants.nutrient_balance(pop)
    print()
    print("--- näring (kg) ---")
    print(f"  {'drift (rel)':>14}: {nb['unaccounted'] / max(1e-12, abs(nb['total'])):.3e}")
    for k, v in nb.items():
        print(f"  {k:>14}: {v:.6e}")
    print(f"  {'in_fauna(mät)':>14}: {nf:.6e}")


if __name__ == "__main__":
    main()
