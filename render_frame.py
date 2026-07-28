# render_frame.py
"""
Rendera en enskild bildruta ur simuleringen till PNG, utan fönster.

Gör viewern testbar: samma ritkod som i den interaktiva körningen, men
resultatet hamnar i en fil i stället för på skärmen. Användbart för
regressionsgranskning efter geometriändringar, och för att se hur världen
faktiskt ser ut utan att starta en session.

    python render_frame.py --ticks 2000 --out frame.png
    python render_frame.py --ticks 5000 --mode BC --scale 3 --out frame.png
"""

from __future__ import annotations

import argparse
import os

# Måste sättas före pygame importeras.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams


def main() -> int:
    ap = argparse.ArgumentParser(description="Rendera en bildruta till PNG utan fönster.")
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--init_pop", type=int, default=12)
    ap.add_argument("--max_pop", type=int, default=256)
    ap.add_argument("--scale", type=int, default=4, help="pixlar per cellbredd")
    ap.add_argument("--mode", type=str, default="BC")
    ap.add_argument("--out", type=str, default="frame.png")
    a = ap.parse_args()

    if int(a.size) > 0:
        WP = WorldParams(size=int(a.size), width=0, height=0, dt=float(a.dt))
    else:
        WP = WorldParams(width=int(a.width), height=int(a.height), dt=float(a.dt))

    pop = Population(
        WP=WP,
        AP=AgentParams(dt=WP.dt),
        PP=PopParams(init_pop=int(a.init_pop), max_pop=int(a.max_pop)),
        seed=int(a.seed),
        hub=None,
    )
    for _ in range(int(a.ticks)):
        pop.step()

    import pygame
    from viewer_pygame import ViewerConfig, WorldViewer

    pygame.init()
    cfg = ViewerConfig(scale=int(a.scale), mode=str(a.mode))
    viewer = WorldViewer(cfg)
    # render_every kan hoppa över bildrutor; anropa tills en faktiskt ritas
    for _ in range(max(1, int(cfg.render_every)) + 1):
        viewer.update(pop)
        if viewer._screen is not None:
            break

    surf = viewer._screen
    if surf is None:
        print("Ingen yta renderades.")
        return 1

    pygame.image.save(surf, str(a.out))
    n_flora = int((pop.store.alive & (pop.store.kind == 1)).sum())
    print(
        f"{a.out}: {surf.get_width()}x{surf.get_height()} px, "
        f"värld {pop.grid.width}x{pop.grid.height} celler, "
        f"{len(pop.agents)} fauna, {n_flora} flora, tick {a.ticks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
