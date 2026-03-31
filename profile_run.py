"""
profile_run.py — kör utan pygame, profilerar 100 steg.
Kör: python profile_run.py 2>&1 | tee profile_output.txt
"""
import cProfile, pstats, io

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams

WP = WorldParams(size=64, dt=0.02)
AP = AgentParams(dt=WP.dt)
PP = PopParams(init_pop=12, max_pop=256)
pop = Population(WP=WP, AP=AP, PP=PP, seed=1, hub=None)

# Värm upp 1000 steg (oavsett population)
print("Värmer upp tills populationen överstiger 100...", flush=True)
while len(pop.agents) < 100:
    pop.step()

print(f"\nStartar profilering: pop={len(pop.agents)} agenter", flush=True)

pr = cProfile.Profile()
pr.enable()
for _ in range(100):
    pop.step()
pr.disable()

print("Profilering klar.\n", flush=True)

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(30)
print(s.getvalue())