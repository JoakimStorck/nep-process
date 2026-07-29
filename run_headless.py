# run_headless.py
"""
Headless entrypoint för nep-process.

Kör simuleringen utan pygame och utan viewer. Avsedd för smoke test,
invariantkontroll, regressionskörningar och profilering.

    python run_headless.py --ticks 10000
    python run_headless.py --ticks 10000 --check-every 500 --seed 3
    python run_headless.py --ticks 2000 --profile

Med --stats blir diagnostikraden bredare och en sammanfattning skrivs vid
slutet: omsättning, var reproduktionen fastnar, näringsbalansens termer.

    python run_headless.py --ticks 4000 --stats
    python run_headless.py --ticks 4000 --stats --seeds 1,2,3
    python run_headless.py --ticks 4000 --stats --flora-ratio 47

Med --pop-log skrivs samma pop.jsonl som run_population.py producerar, så
live_pop_plot.py kan följa körningen utan att pygame behöver vara med:

    python run_headless.py --ticks 20000 --pop-log pop.jsonl
    python live_pop_plot.py --fp pop.jsonl        # i ett annat skal

Exitkod 0 om alla invarianter höll, 1 annars. Det gör kommandot användbart
direkt i CI eller som pre-commit-kontroll.
"""

from __future__ import annotations

import argparse
import sys
import time

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams

import numpy as np

from invariants import check_all, diagnostics, nutrient_balance


# ---------------------------------------------------------------------------
# Instrumentering av reproduktionen.
#
# Räknar var _try_mating() faller ifrån, utan att ändra beteende. Skiljer den
# fysiologiska grinden (agenten är inte redo) från den rumsliga (agenten är
# redo men ser ingen, eller ser någon utanför parningsradien). Installeras
# bara med --stats.
# ---------------------------------------------------------------------------

_R = {"agenttick": 0, "redo": 0, "ser_ingen": 0, "utanfor_radie": 0,
      "sag_men_parade_ej": 0, "parning": 0}
_INSTRUMENTED = False


def instrument_mating() -> None:
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return
    _INSTRUMENTED = True

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
        _R["agenttick"] += 1
        if not ready(self, a_slot):
            return
        _R["redo"] += 1

        hit = getattr(agent, "_cached_agent_hit", None)
        best = None
        if isinstance(hit, tuple) and len(hit) >= 5:
            _, _, _, hit_slot, desired_id = hit[:5]
            k = int(hit_slot)
            store = self.store
            if (k >= 0 and int(desired_id) > 0 and k < int(store.n)
                    and bool(store.alive[k]) and int(store.kind[k]) == 0
                    and int(store.id[k]) == int(desired_id)):
                cand = agent_for(self, k)
                if cand is not None and cand is not agent and int(cand.store_slot) >= 0:
                    best = cand
        if best is None:
            _R["ser_ingen"] += 1
            return
        if dist(self, a_slot, int(best.store_slot), squared=True) > float(self.PP.mating_radius) ** 2:
            _R["utanfor_radie"] += 1
            return

        before = bool(agent.body.gestating) or bool(best.body.gestating)
        orig(self, agent, ctx, candidates)
        if (bool(agent.body.gestating) or bool(best.body.gestating)) and not before:
            _R["parning"] += 1
        else:
            _R["sag_men_parade_ej"] += 1

    Population._try_mating = wrapped


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Kör nep-process headless med invariantkontroll.")
    ap.add_argument("--ticks", type=int, default=10000, help="antal simuleringssteg")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=0, help="kvadratisk värld; sätter både bredd och höjd")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=256, help="måste vara jämn")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--init_pop", type=int, default=12)
    ap.add_argument("--max_pop", type=int, default=256)

    ap.add_argument("--check-every", type=int, default=500,
                    help="kör invariantsviten var N:te tick (0 = bara vid start och slut)")
    ap.add_argument("--report-every", type=int, default=2000,
                    help="skriv diagnostikrad var N:te tick (0 = av)")
    ap.add_argument("--fail-fast", action="store_true",
                    help="avbryt vid första invariantbrottet")
    ap.add_argument("--profile", action="store_true",
                    help="kör under cProfile och skriv topplista vid slut")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--stats", action="store_true",
                    help="bredare diagnostikrad och sammanfattning vid slut")
    ap.add_argument("--seeds", type=str, default=None,
                    help="kommaseparerad lista, t.ex. 1,2,3 — kör flera och jämför")
    ap.add_argument("--flora-ratio", type=float, default=None,
                    help="PopParams.flora_init_mass_ratio; standard är modellens eget värde")
    ap.add_argument("--pop-log", type=str, default=None,
                    help="skriv pop.jsonl för live_pop_plot.py")
    ap.add_argument("--pop-every", type=float, default=1.0,
                    help="loggintervall i simulerade sekunder för --pop-log")
    ap.add_argument("--world-log", type=str, default=None,
                    help="skriv world.jsonl för live_world_plot.py (flora och näring)")
    ap.add_argument("--world-every", type=float, default=2.0,
                    help="loggintervall i simulerade sekunder för --world-log")
    return ap.parse_args()


def build_population(a: argparse.Namespace, seed: int, hub=None) -> Population:
    if int(a.size) > 0:
        WP = WorldParams(size=int(a.size), width=0, height=0, dt=float(a.dt))
    else:
        WP = WorldParams(width=int(a.width), height=int(a.height), dt=float(a.dt))
    AP = AgentParams(dt=WP.dt)
    PP = PopParams(init_pop=int(a.init_pop), max_pop=int(a.max_pop))
    if getattr(a, "flora_ratio", None) is not None:
        PP.flora_init_mass_ratio = float(a.flora_ratio)
    return Population(WP=WP, AP=AP, PP=PP, seed=int(seed), hub=hub)


def gestation_state(pop: Population) -> tuple[int, float]:
    """Antal dräktiga och deras median-andel av målmassan."""
    fr = [
        float(x.body.gest_M) / float(x.body.gest_M_target)
        for x in pop.agents
        if x.body.alive and bool(x.body.gestating) and float(x.body.gest_M_target) > 0.0
    ]
    return len(fr), (float(np.median(fr)) if fr else 0.0)


def format_stats(pop: Population, d: dict, tick: int, elapsed: float) -> str:
    nb = nutrient_balance(pop)
    ng, gfrac = gestation_state(pop)
    return (
        f"tick {tick:7d}  t={d['t']:8.1f}  "
        f"fauna={d['fauna_n']:4d}  flora={d['flora_n']:6d}  "
        f"M_fauna={d['fauna_mass_kg']:8.3f}  M_flora={d['flora_mass_kg']:9.3f}  "
        f"detritus={d['detritus_mass_kg']:8.4f}  fri_när={nb['free']:8.5f}  "
        f"föd={pop._births_total:4d}  död={pop._deaths_total:4d}  "
        f"dräkt={ng:3d}/{gfrac*100:3.0f}%  "
        f"{elapsed / max(tick, 1) * 1000.0:.2f} ms/tick"
    )


def print_summary(pop: Population, d0: dict, unika: int, worst_drift: float,
                  n_cells: int, elapsed: float, ticks: int) -> None:
    d = diagnostics(pop)
    nb = nutrient_balance(pop)
    redo = max(1, _R["redo"])

    print(f"\n--- sammanfattning ---")
    print(f"  bestånd      fauna {d0['fauna_n']:>5} -> {d['fauna_n']:<5}"
          f"   flora {d0['flora_n']:>6} -> {d['flora_n']:<6}")
    print(f"  massa (kg)   fauna {d0['fauna_mass_kg']:>8.3f} -> {d['fauna_mass_kg']:<8.3f}"
          f" flora {d0['flora_mass_kg']:>8.2f} -> {d['flora_mass_kg']:<8.2f}"
          f"  kvot {d['flora_mass_kg'] / max(1e-12, d['fauna_mass_kg']):.1f}x")
    print(f"  omsättning   {pop._births_total} födslar, {pop._deaths_total} dödsfall, "
          f"{unika} unika individer (start {d0['fauna_n']})")
    if unika <= d0["fauna_n"]:
        if pop._deaths_total == 0:
            print("               ingen omsättning alls — samma individer hela körningen, "
                  "så beståndet säger inget om jämvikt")
        else:
            print("               inga nya individer — en avtagande kohort, "
                  "inte ett flödesjämviktstal")

    if _R["agenttick"]:
        print(f"\n  reproduktion {100 * _R['redo'] / _R['agenttick']:.1f} % av agenttickarna är klara")
        print(f"    ser ingen alls {100 * _R['ser_ingen'] / redo:5.1f} %"
              f"   utanför parningsradie {100 * _R['utanfor_radie'] / redo:5.1f} %")
        print(f"    parningar      {_R['parning']:5d}"
              f"     såg partner men parade inte {_R['sag_men_parade_ej']:5d}")
        print(f"    täthet {d['fauna_n'] / max(1, n_cells) * 1000.0:.2f} agenter per 1000 celler")

    print(f"\n  näring (kg)  fri {nb['free']:.5f}  flora {nb['in_flora']:.5f}  "
          f"fauna {nb['in_fauna']:.5f}  detritus {nb['in_detritus']:.5f}")
    print(f"               tillfört {nb['added']:.5f}  förlorat {nb['lost']:.5f}  "
          f"summa {nb['total']:.5f}")
    print(f"               drift {nb['unaccounted'] / max(1e-12, abs(nb['total'])):.2e} rel "
          f"(störst under körningen {worst_drift:.2e})")

    print(f"\n  takt         {elapsed / max(1, ticks) * 1000.0:.2f} ms/tick, "
          f"{elapsed / max(1, ticks) * 1e6 / max(1, d['flora_n']):.2f} us per floraindivid")


def format_diagnostics(d: dict, tick: int, elapsed: float) -> str:
    rate = (elapsed / max(tick, 1)) * 1000.0
    return (
        f"tick {tick:7d}  t={d['t']:8.1f}  "
        f"fauna={d['fauna_n']:4d}  flora={d['flora_n']:5d}  "
        f"fria={d['free_slots']:5d}/{d['capacity']:5d}  "
        f"M_flora={d['flora_mass_kg']:.4e}  M_fauna={d['fauna_mass_kg']:.4e}  "
        f"M_detritus={d['detritus_mass_kg']:.4e}  "
        f"{rate:.2f} ms/tick"
    )


def run(a: argparse.Namespace, seed: int | None = None) -> int:
    seed = int(a.seed) if seed is None else int(seed)

    if a.stats:
        instrument_mating()
        for k in _R:
            _R[k] = 0

    # Samma loggar som run_population.py skriver, men utan pygame. Det gör
    # live_pop_plot.py och live_world_plot.py användbara mot en
    # headless-körning.
    writers = []
    observers = []
    if a.pop_log or a.world_log:
        from simlog.jsonl import JsonlWriter
        from simlog.sinks import EventHub
        from simlog.observers import PopLogger, WorldLogger

        if a.pop_log:
            w = JsonlWriter(str(a.pop_log), flush_every=1)
            w.__enter__()
            writers.append(w)
            observers.append(PopLogger(w=w, every_s=float(a.pop_every)))
        if a.world_log:
            w = JsonlWriter(str(a.world_log), flush_every=1)
            w.__enter__()
            writers.append(w)
            observers.append(WorldLogger(w=w, every_s=float(a.world_every)))

        hub = EventHub(observers)
    else:
        hub = None

    try:
        return _run_inner(a, seed, hub)
    finally:
        for w in writers:
            w.__exit__(None, None, None)


def _run_inner(a: argparse.Namespace, seed: int, hub) -> int:
    pop = build_population(a, seed, hub=hub)
    d0 = diagnostics(pop)
    n_cells = int(pop.grid.n_cells)
    unika = {int(x.id) for x in pop.agents if x.body.alive}
    worst_drift = 0.0

    if not a.quiet:
        print(
            f"START headless: ticks={a.ticks} värld={pop.grid.width}x{pop.grid.height} ({pop.grid.n_cells} celler) dt={a.dt} "
            f"init_pop={a.init_pop} max_pop={a.max_pop} seed={seed}",
            flush=True,
        )

    failures = 0
    check_every = max(0, int(a.check_every))
    report_every = max(0, int(a.report_every))

    report = check_all(pop, tick=0)
    if not report.ok:
        failures += len(report.violations)
        print(report.summary(), file=sys.stderr, flush=True)
        if a.fail_fast:
            return 1

    t0 = time.perf_counter()
    tick = 0

    for tick in range(1, int(a.ticks) + 1):
        try:
            pop.step()
        except Exception as exc:  # noqa: BLE001 — vi vill se vilket tick som small
            print(f"AVBROTT i tick {tick}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            raise

        if a.stats:
            for x in pop.agents:
                if x.body.alive:
                    unika.add(int(x.id))

        if check_every and tick % check_every == 0:
            report = check_all(pop, tick=tick)
            if not report.ok:
                failures += len(report.violations)
                print(report.summary(), file=sys.stderr, flush=True)
                if a.fail_fast:
                    return 1

        if report_every and tick % report_every == 0 and not a.quiet:
            el = time.perf_counter() - t0
            if a.stats:
                nb = nutrient_balance(pop)
                worst_drift = max(
                    worst_drift, abs(nb["unaccounted"]) / max(1e-12, abs(nb["total"]))
                )
                print(format_stats(pop, diagnostics(pop), tick, el), flush=True)
            else:
                print(format_diagnostics(diagnostics(pop), tick, el), flush=True)

    elapsed = time.perf_counter() - t0

    report = check_all(pop, tick=tick)
    if not report.ok:
        failures += len(report.violations)
        print(report.summary(), file=sys.stderr, flush=True)

    if not a.quiet:
        if a.stats:
            nb = nutrient_balance(pop)
            worst_drift = max(
                worst_drift, abs(nb["unaccounted"]) / max(1e-12, abs(nb["total"]))
            )
            print(format_stats(pop, diagnostics(pop), tick, elapsed), flush=True)
        else:
            print(format_diagnostics(diagnostics(pop), tick, elapsed), flush=True)
        print(
            f"SLUT: {tick} tick på {elapsed:.1f}s "
            f"({elapsed / max(tick, 1) * 1000.0:.2f} ms/tick), "
            f"{'invariantsvit godkänd' if failures == 0 else f'{failures} invariantbrott'}",
            flush=True,
        )
        if a.stats:
            print_summary(pop, d0, len(unika), worst_drift, n_cells, elapsed, tick)

    return 0 if failures == 0 else 1


def main() -> int:
    a = parse_args()

    if a.seeds:
        seeds = [int(x) for x in str(a.seeds).split(",") if x.strip()]
        code = 0
        for sd in seeds:
            print(f"\n{'=' * 20} seed {sd} {'=' * 20}", flush=True)
            code |= run(a, seed=sd)
        return code

    if not a.profile:
        return run(a)

    import cProfile
    import io
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    code = run(a)
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
    print(s.getvalue())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
