"""
profile_run.py — kör utan pygame, profilerar N steg.
Kör: python profile_run.py 2>&1 | tee profile_output.txt
"""
import cProfile, pstats, io, time

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams

WP = WorldParams(size=64, dt=0.02)
AP = AgentParams(dt=WP.dt)
PP = PopParams(init_pop=50, max_pop=256)   # fler startindivider — population når topp snabbare
pop = Population(WP=WP, AP=AP, PP=PP, seed=1, hub=None)

# Värm upp ett fast antal steg istf populationsvillkor
# (populationen kanske aldrig når 100 beroende på parametrar)
WARMUP_STEPS = 500
PROFILE_STEPS = 10000

print(f"Värmer upp {WARMUP_STEPS} steg...", flush=True)
t0 = time.perf_counter()
for _ in range(WARMUP_STEPS):
    pop.step()
t1 = time.perf_counter()
print(f"Uppvärmning klar: {t1-t0:.2f}s wall, pop={len(pop.agents)} agenter @ t={pop.t:.1f}s", flush=True)

if len(pop.agents) == 0:
    print("Population utdöd under uppvärmning — öka WARMUP_STEPS eller justera parametrar.")
    raise SystemExit(1)

print(f"\nStartar profilering: {PROFILE_STEPS} steg, {len(pop.agents)} agenter", flush=True)

pr = cProfile.Profile()
pr.enable()
t2 = time.perf_counter()
for _ in range(PROFILE_STEPS):
    if not pop.agents:
        break
    pop.step()
t3 = time.perf_counter()
pr.disable()

wall_s   = t3 - t2
steps_run = min(PROFILE_STEPS, PROFILE_STEPS)  # kan vara kortare vid utdöende
sim_s    = steps_run * WP.dt
print(f"Profilering klar: {wall_s:.2f}s wall → {sim_s:.1f}s sim  ({sim_s/wall_s:.1f}× realtid)\n")

# --- tottime: var CPU-tid faktiskt spenderas (den verkliga flaskhalsen) ---
print("=" * 70)
print("TOTTIME — tid i funktionen exkl. underfunktioner (hitta flaskhalsen)")
print("=" * 70)
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
print(s.getvalue())

# --- cumtime: kumulativ tid inkl. underfunktioner (förstå anropskedjan) ---
print("=" * 70)
print("CUMTIME — kumulativ tid inkl. underfunktioner (förstå var i systemet)")
print("=" * 70)
s2 = io.StringIO()
pstats.Stats(pr, stream=s2).sort_stats("cumulative").print_stats(25)
print(s2.getvalue())