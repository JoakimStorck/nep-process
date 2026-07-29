# calibrate.py
"""
Kalibreringskörning med diagnostik till fil.

Kör simuleringen länge och skriver en rad per mätpunkt till CSV, med glesare
statusrader till skärmen. Avsedd för Steg 4:s kalibrering, där frågorna kräver
tiotusentals tick för att besvaras.

    python calibrate.py --ticks 200000 --out runs/base.csv
    python calibrate.py --ticks 200000 --nutrient-input 5e-11 --label mager --out runs/mager.csv
    python calibrate.py --ticks 200000 --max-pop 512 --label cap512 --out runs/cap512.csv

Mätpunkterna i TODO, Del E, och var de finns i utdatat:

  floran når ett stationärt antal          flora_n över tid
  oförändrat vid dubblad capacity          jämför två körningar med olika --max-pop
  överlevnadsskillnad på uptake_capacity   uptake_mean över tid; stiger den selekteras den
  näringsbalans sluten                     nutrient_unaccounted, driften snarare än nivån
  structure differentierar mellan band     struct_band0..7 och struct_temp_corr
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time

import numpy as np

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams
from invariants import check_all, nutrient_balance

N_BANDS_OUT = 8


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lång kalibreringskörning med CSV-diagnostik.")
    ap.add_argument("--ticks", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--init-pop", type=int, default=12)
    ap.add_argument("--max-pop", type=int, default=256)

    # Parametrar under kalibrering
    ap.add_argument("--nutrient-input", type=float, default=None)
    ap.add_argument("--uptake-rate-max", type=float, default=None)
    ap.add_argument("--nutrient-diffusion", type=float, default=None)

    ap.add_argument("--out", type=str, default="calibration.csv")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--every", type=int, default=500, help="mätintervall till fil")
    ap.add_argument("--status-every", type=int, default=20000, help="statusrad till skärm")
    ap.add_argument("--check-every", type=int, default=10000, help="invariantsvit, 0 = av")
    return ap.parse_args()


def build(a: argparse.Namespace) -> Population:
    wp_kw = dict(width=int(a.width), height=int(a.height), dt=float(a.dt))
    for cli, name in (
        ("nutrient_input", "nutrient_input"),
        ("uptake_rate_max", "uptake_rate_max"),
        ("nutrient_diffusion", "nutrient_diffusion"),
    ):
        v = getattr(a, cli.replace("_", "_"), None)
        if v is not None:
            wp_kw[name] = float(v)
    WP = WorldParams(**wp_kw)

    pp_kw = dict(init_pop=int(a.init_pop), max_pop=int(a.max_pop))
    PP = PopParams(**pp_kw)

    return Population(WP=WP, AP=AgentParams(dt=WP.dt), PP=PP, seed=int(a.seed), hub=None)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson, med noll för degenererade fall i stället för NaN."""
    if x.size < 3:
        return 0.0
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def sample(pop, tick: int, elapsed: float) -> dict:
    st, w, g = pop.store, pop.world, pop.grid

    live = np.flatnonzero(st.alive[: int(st.n)])
    flora = live[st.kind[live] == 1] if live.size else live
    fauna_alive = [a for a in pop.agents if a.body.alive]

    row = {
        "tick": tick,
        "t": round(float(pop.t), 3),
        "ms_per_tick": round(elapsed / max(tick, 1) * 1000.0, 4),
        "fauna_n": len(fauna_alive),
        "flora_n": int(flora.size),
        "capacity": int(st.capacity),
        "free_slots": int(len(st.free_slots)),
    }

    if flora.size:
        m = st.mass[flora].astype(np.float64)
        struct = st.structure[flora].astype(np.float64)
        upt = st.uptake_capacity[flora].astype(np.float64)
        cells = st.cell_idx[flora].astype(np.int64)
        bands = np.asarray(g.bands_of_cells(cells), dtype=np.int64)

        occupied, counts = np.unique(cells, return_counts=True)
        band_out = (bands * N_BANDS_OUT) // int(g.n_bands)

        row.update({
            "flora_mass": float(m.sum()),
            "flora_mass_mean": float(m.mean()),
            "struct_mean": float(struct.mean()),
            "struct_std": float(struct.std()),
            "uptake_mean": float(upt.mean()),
            "uptake_std": float(upt.std()),
            # Trängsel: konkurrensen biter först när flera delar cell.
            "cells_occupied": int(occupied.size),
            "per_cell_max": int(counts.max()),
            "per_cell_mean": float(counts.mean()),
            "shared_cell_frac": float(np.mean(counts[np.searchsorted(occupied, cells)] > 1)),
            # Differentiering: samvariation mellan trait och lokal miljö.
            "struct_temp_corr": _corr(struct, np.asarray(w.T_band)[bands]),
            "uptake_nutr_corr": _corr(upt, np.asarray(w.nutrient)[cells]),
            "uptake_mass_corr": _corr(upt, m),
        })
        for b in range(N_BANDS_OUT):
            sel = band_out == b
            row[f"flora_band{b}"] = int(sel.sum())
            row[f"struct_band{b}"] = float(struct[sel].mean()) if sel.any() else 0.0
    else:
        for k in ("flora_mass", "flora_mass_mean", "struct_mean", "struct_std",
                  "uptake_mean", "uptake_std", "per_cell_mean", "shared_cell_frac",
                  "struct_temp_corr", "uptake_nutr_corr", "uptake_mass_corr"):
            row[k] = 0.0
        row["cells_occupied"] = 0
        row["per_cell_max"] = 0
        for b in range(N_BANDS_OUT):
            row[f"flora_band{b}"] = 0
            row[f"struct_band{b}"] = 0.0

    nb = nutrient_balance(pop)
    row.update({
        "nutr_free": nb["free"],
        "nutr_in_flora": nb["in_flora"],
        "nutr_in_detritus": nb["in_detritus"],
        "nutr_total": nb["total"],
        "nutr_added": nb["added"],
        "nutr_lost": nb["lost"],
        "nutr_unaccounted": nb["unaccounted"],
    })

    act = w.detritus_active_cells
    row["detritus_mass"] = float(np.sum(np.asarray(w.detritus), dtype=np.float64))
    row["detritus_cells"] = int(act.size)
    row["detritus_struct"] = float(np.asarray(w.detritus_structure)[act].mean()) if act.size else 0.0
    row["fauna_mass"] = float(sum(float(a.body.M) for a in fauna_alive))

    return row


def main() -> int:
    a = parse_args()
    pop = build(a)

    print(
        f"START {a.label or 'kalibrering'}: {a.ticks} tick, {pop.grid.width}x{pop.grid.height} "
        f"({pop.grid.n_cells} celler), seed {a.seed}, max_pop {a.max_pop}\n"
        f"  nutrient_input={pop.WP.nutrient_input:.3e}  uptake_rate_max={pop.WP.uptake_rate_max:.3e}  "
        f"diffusion={pop.WP.nutrient_diffusion:.3g}\n"
        f"  flora_min_mass_frac={pop.PP.flora_min_mass_frac:.3e}\n"
        f"  -> {a.out}",
        flush=True,
    )

    every = max(1, int(a.every))
    status_every = max(0, int(a.status_every))
    check_every = max(0, int(a.check_every))

    t0 = time.perf_counter()
    violations = 0
    writer = None

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        for tick in range(0, int(a.ticks) + 1):
            if tick:
                pop.step()

            if tick % every == 0:
                row = sample(pop, tick, time.perf_counter() - t0)
                if a.label:
                    row["label"] = a.label
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)

                if status_every and tick % status_every == 0:
                    fh.flush()
                    print(
                        f"  {tick:8d}  fauna {row['fauna_n']:4d}  flora {row['flora_n']:5d}"
                        f"  M {row['flora_mass']:.3e}  näring {row['nutr_free']:.3e}"
                        f"  delad cell {row['shared_cell_frac'] * 100:4.1f}%"
                        f"  struktur {row['struct_mean']:.3f}  upptag {row['uptake_mean']:.3f}"
                        f"  {row['ms_per_tick']:.2f} ms/tick",
                        flush=True,
                    )

            if check_every and tick and tick % check_every == 0:
                rep = check_all(pop, tick)
                if not rep.ok:
                    violations += len(rep.violations)
                    print(rep.summary(), file=sys.stderr, flush=True)

    elapsed = time.perf_counter() - t0
    print(
        f"SLUT: {a.ticks} tick på {elapsed / 60.0:.1f} min "
        f"({elapsed / max(a.ticks, 1) * 1000.0:.2f} ms/tick), "
        f"{'invarianter OK' if violations == 0 else f'{violations} brott'}",
        flush=True,
    )
    return 0 if violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
