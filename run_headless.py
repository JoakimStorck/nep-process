# run_headless.py
"""
Headless entrypoint för nep-process.

Kör simuleringen utan pygame och utan viewer. Avsedd för smoke test,
invariantkontroll, regressionskörningar och profilering.

    python run_headless.py --ticks 10000
    python run_headless.py --ticks 10000 --check-every 500 --seed 3
    python run_headless.py --ticks 2000 --profile

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

from invariants import check_all, diagnostics


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
    return ap.parse_args()


def build_population(a: argparse.Namespace) -> Population:
    if int(a.size) > 0:
        WP = WorldParams(size=int(a.size), width=0, height=0, dt=float(a.dt))
    else:
        WP = WorldParams(width=int(a.width), height=int(a.height), dt=float(a.dt))
    AP = AgentParams(dt=WP.dt)
    PP = PopParams(init_pop=int(a.init_pop), max_pop=int(a.max_pop))
    return Population(WP=WP, AP=AP, PP=PP, seed=int(a.seed), hub=None)


def format_diagnostics(d: dict, tick: int, elapsed: float) -> str:
    rate = (elapsed / max(tick, 1)) * 1000.0
    return (
        f"tick {tick:7d}  t={d['t']:8.1f}  "
        f"fauna={d['fauna_n']:4d}  flora={d['flora_n']:5d}  "
        f"fria={d['free_slots']:5d}/{d['capacity']:5d}  "
        f"M_flora={d['flora_mass_kg']:.4e}  M_fauna={d['fauna_mass_kg']:.4e}  "
        f"M_kadaver={d['carcass_mass_kg']:.4e}  "
        f"{rate:.2f} ms/tick"
    )


def run(a: argparse.Namespace) -> int:
    pop = build_population(a)

    if not a.quiet:
        print(
            f"START headless: ticks={a.ticks} värld={pop.grid.width}x{pop.grid.height} ({pop.grid.n_cells} celler) dt={a.dt} "
            f"init_pop={a.init_pop} max_pop={a.max_pop} seed={a.seed}",
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

        if check_every and tick % check_every == 0:
            report = check_all(pop, tick=tick)
            if not report.ok:
                failures += len(report.violations)
                print(report.summary(), file=sys.stderr, flush=True)
                if a.fail_fast:
                    return 1

        if report_every and tick % report_every == 0 and not a.quiet:
            print(format_diagnostics(diagnostics(pop), tick, time.perf_counter() - t0), flush=True)

    elapsed = time.perf_counter() - t0

    report = check_all(pop, tick=tick)
    if not report.ok:
        failures += len(report.violations)
        print(report.summary(), file=sys.stderr, flush=True)

    if not a.quiet:
        print(format_diagnostics(diagnostics(pop), tick, elapsed), flush=True)
        print(
            f"SLUT: {tick} tick på {elapsed:.1f}s "
            f"({elapsed / max(tick, 1) * 1000.0:.2f} ms/tick), "
            f"{'invariantsvit godkänd' if failures == 0 else f'{failures} invariantbrott'}",
            flush=True,
        )

    return 0 if failures == 0 else 1


def main() -> int:
    a = parse_args()

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
