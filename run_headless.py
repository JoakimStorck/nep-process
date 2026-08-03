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
import os
import sys
import time

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams

import numpy as np

from invariants import check_all, diagnostics, fauna_spacing, nutrient_balance


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


# ---------------------------------------------------------------------------
# Passtidtagning.
#
# cProfile tredubblar körtiden och förvränger fördelningen mellan pass, så
# --profile duger för att hitta en hotspot men inte för att jämföra två
# körningar. Den här mätningen läser perf_counter runt varje systempass:
# tjugotvå klockavläsningar per tick, alltså långt under en promille.
#
# Frågan den ska svara på är inte var tiden går utan *vad den växer med*.
# Floran och faunan skalar oberoende, och totalen döljer det: ett pass som är
# en femtedel vid tolv djur kan vara hälften vid tvåhundra utan att någon
# enskild körning visar det. Därför normeras varje post både per djur och per
# floraindivid, och de fauna-linjära posten summeras för sig.
#
# Håll floran konstant när init_pop varieras. Sådden skalar med faunans
# startmassa, så --flora-ratio måste kompenseras: ratio × init_pop konstant.
# ---------------------------------------------------------------------------

_PASSES = (
    "_step_world_and_flora",
    "_step_metabolism_system",
    "_step_sense_system",
    "_step_decision_system",
    "_step_move_system",
    "_step_body_system",
    "_step_interaction_system",
    "_step_deaths",
    "_step_births",
    "_step_sampling",
    "_finalize_store_and_emit",
)

# Pass vars kostnad förväntas skala med antalet djur. Summeras separat,
# eftersom det är den summan som avgör hur stor en population kan bli.
_FAUNA_PASSES = frozenset({
    "_step_metabolism_system",
    "_step_sense_system",
    "_step_decision_system",
    "_step_move_system",
    "_step_body_system",
    "_step_interaction_system",
})

_INNER_CLASSES_DONE = False


class PassTimer:
    """Ackumulerar tid och anropsantal per pass, normerat mot beståndet."""

    def __init__(self, warmup: int = 50) -> None:
        self.warmup = int(warmup)
        self.acc: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.order: list[str] = []
        self.ticks = 0
        self.wall = 0.0
        self.fauna_sum = 0.0
        self.flora_sum = 0.0
        self.active = False

    # -- installation ---------------------------------------------------

    def _slot(self, label: str) -> None:
        if label not in self.acc:
            self.acc[label] = 0.0
            self.calls[label] = 0
            self.order.append(label)

    def _wrap(self, owner, name: str, label: str) -> None:
        orig = getattr(owner, name)
        self._slot(label)

        def wrapped(*args, **kwargs):
            if not self.active:
                return orig(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                self.acc[label] += time.perf_counter() - t0
                self.calls[label] += 1

        setattr(owner, name, wrapped)

    def install(self, pop: Population) -> None:
        """Passen wrappas på instansen; ingen annan körning påverkas."""
        for name in _PASSES:
            self._wrap(pop, name, name)

    def install_inner(self) -> None:
        """
        Finfördelning inuti sense- och move-passen.

        Wrappas på klassnivå och kostar en klockavläsning per agent och metod,
        alltså mer än passtidtagningen. Håll den till separata körningar.
        """
        global _INNER_CLASSES_DONE
        if _INNER_CLASSES_DONE:
            return
        from agent import Agent, RaySensors

        for cls, name in (
            (RaySensors, "sense"),
            (RaySensors, "see_agent_first_hit"),
            (Agent, "_build_obs"),
            (Agent, "_build_inputs_from_cache"),
            (Agent, "_integrate_motion"),
            (Agent, "_perform_feeding"),
        ):
            self._wrap(cls, name, f"  {cls.__name__}.{name}")
        _INNER_CLASSES_DONE = True

    # -- mätning --------------------------------------------------------

    def begin_tick(self) -> float:
        return time.perf_counter()

    def end_tick(self, pop: Population, t0: float, tick: int) -> None:
        if tick <= self.warmup:
            # Numbas JIT och de första allokeringarna hör inte till taktens
            # stationära kostnad. Nollställ i stället för att dra av.
            self.reset()
            self.active = True
            return
        self.wall += time.perf_counter() - t0
        self.ticks += 1
        self.fauna_sum += float(len(pop._fauna_slots()))
        self.flora_sum += float(len(pop._flora_slots()))

    def reset(self) -> None:
        for k in self.acc:
            self.acc[k] = 0.0
            self.calls[k] = 0
        self.ticks = 0
        self.wall = 0.0
        self.fauna_sum = 0.0
        self.flora_sum = 0.0

    # -- rapport --------------------------------------------------------

    def report(self) -> None:
        if self.ticks <= 0:
            print("\n--- passtidtagning: för få tick efter uppvärmningen ---")
            return

        n = float(self.ticks)
        fauna = self.fauna_sum / n
        flora = self.flora_sum / n
        tot_ms = self.wall / n * 1e3

        print(f"\n--- passtidtagning ---")
        print(f"  {self.ticks} tick efter {self.warmup} uppvärmning   "
              f"fauna {fauna:.1f}   flora {flora:.0f}   "
              f"totalt {tot_ms:.3f} ms/tick")
        print(f"  {'pass':32s} {'ms/tick':>9s} {'andel':>7s} "
              f"{'us/djur':>9s} {'us/planta':>10s} {'anrop':>8s}")

        fauna_ms = 0.0
        for label in self.order:
            ms = self.acc[label] / n * 1e3
            if label in _FAUNA_PASSES:
                fauna_ms += ms
            print(f"  {label:32s} {ms:9.3f} {100.0 * ms / max(1e-12, tot_ms):6.1f}% "
                  f"{ms * 1e3 / max(1.0, fauna):9.2f} "
                  f"{ms * 1e3 / max(1.0, flora):10.3f} "
                  f"{self.calls[label] / n:8.1f}")

        per_animal = fauna_ms * 1e3 / max(1.0, fauna)
        fixed_ms = tot_ms - fauna_ms
        print(f"\n  fauna-linjära pass   {fauna_ms:.3f} ms/tick, "
              f"{100.0 * fauna_ms / max(1e-12, tot_ms):.1f} % av takten, "
              f"{per_animal:.1f} us per djur")
        print(f"  övrigt               {fixed_ms:.3f} ms/tick")
        for target in (200, 500, 1000):
            est = fixed_ms + per_animal * target / 1e3
            print(f"    extrapolerat till {target:4d} djur vid samma flora: "
                  f"{est:8.1f} ms/tick")
        print("  Extrapolationen antar att floran hålls konstant och att "
              "sensingens\n  träfffrekvens inte ändras med tätheten. Den är "
              "en storleksordning, inte\n  en prognos.")


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

    ap.add_argument("--nutrient-init", type=float, default=None,
                    help="WorldParams.nutrient_init, kg fri näring per cell — bördigheten")
    ap.add_argument("--detritus-init", type=float, default=None,
                    help="WorldParams.detritus_init, kg förna per cell vid start")
    ap.add_argument("--detritus-structure-init", type=float, default=None,
                    help="WorldParams.detritus_structure_init, strukturandel i sådd förna")
    ap.add_argument("--nutrient-input", type=float, default=None,
                    help="WorldParams.nutrient_input, per cell och månad")
    ap.add_argument("--nutrient-loss-frac", type=float, default=None,
                    help="WorldParams.nutrient_loss_frac, andel som lämnar systemet vid nedbrytning")
    ap.add_argument("--uptake-rate", type=float, default=None,
                    help="WorldParams.uptake_rate_per_area, kg näring per månad och areaenhet")
    ap.add_argument("--serve", type=int, default=0,
                    help="sänd bildrutor till viewerklienter på denna port (0 = av)")
    ap.add_argument("--serve-host", type=str, default="127.0.0.1",
                    help="adress att binda till; 127.0.0.1 kräver SSH-tunnel utifrån")
    ap.add_argument("--serve-control", action="store_true",
                    help="låt anslutna viewers pausa körningen")
    ap.add_argument("--serve-fps", type=float, default=10.0,
                    help="högsta antal bildrutor per sekund att packa")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="skriv en PNG av världen var N:te tick (0 = av)")
    ap.add_argument("--snapshot-dir", type=str, default="snapshots",
                    help="katalog för bildrutorna; skapas om den saknas")
    ap.add_argument("--snapshot-scale", type=int, default=2, help="pixlar per cellbredd")
    ap.add_argument("--snapshot-mode", type=str, default="BC",
                    help="viewerns ritläge: BC, B, C, TEMP eller FLORA")
    ap.add_argument("--snapshot-crop", type=int, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"),
                    help="beskär till ett utsnitt i celler; låter en liten yta ritas stort")
    ap.add_argument("--life-log", type=str, default=None,
                    help="skriv life.jsonl med födslar och dödsfall, inklusive dödsorsak")
    ap.add_argument("--check-every", type=int, default=500,
                    help="kör invariantsviten var N:te tick (0 = bara vid start och slut)")
    ap.add_argument("--report-every", type=int, default=2000,
                    help="skriv diagnostikrad var N:te tick (0 = av)")
    ap.add_argument("--fail-fast", action="store_true",
                    help="avbryt vid första invariantbrottet")
    ap.add_argument("--profile", action="store_true",
                    help="kör under cProfile och skriv topplista vid slut")
    ap.add_argument("--pass-timing", action="store_true",
                    help="tidta varje systempass med perf_counter; oförvrängt, "
                         "till skillnad från --profile")
    ap.add_argument("--pass-timing-inner", action="store_true",
                    help="dessutom finfördelning inuti sense- och move-passen; "
                         "kostar mer och bör köras separat")
    ap.add_argument("--pass-timing-warmup", type=int, default=50,
                    help="tick som räknas som uppvärmning och utesluts")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--stats", action="store_true",
                    help="bredare diagnostikrad och sammanfattning vid slut")
    ap.add_argument("--seeds", type=str, default=None,
                    help="kommaseparerad lista, t.ex. 1,2,3 — kör flera och jämför")
    ap.add_argument("--drag-lin", type=float, default=None,
                    help="AgentParams.drag_lin — linjärt motstånd; sätter farten")
    ap.add_argument("--drag-quad", type=float, default=None,
                    help="AgentParams.drag_quad — kvadratiskt motstånd")
    ap.add_argument("--v-max", type=float, default=None,
                    help="AgentParams.v_max — hårt farttak; binder inte vid förval")
    ap.add_argument("--sense-idle", type=int, default=None,
                    help="AgentParams.sense_idle_steps — tick mellan sensingar i vila")
    ap.add_argument("--sense-alert", type=int, default=None,
                    help="AgentParams.sense_alert_steps — tick mellan sensingar i beredskap")
    ap.add_argument("--sociability-init", type=float, default=None,
                    help="grundarnas sociability, 0..1; nollpunkten i reflexen är 0,5")
    ap.add_argument("--fauna-at", type=int, default=None,
                    help="tick då faunan sätts in; 0 = vid start")
    ap.add_argument("--fauna-spawn-radius", type=float, default=None,
                    help="radie för insättningsfläcken; 0 = jämn utspridning")
    ap.add_argument("--flora-plant-mass", type=float, default=None,
                    help="PopParams.flora_init_plant_mass, medelmassa per sådd planta")
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
    # Världsparametrar som kan överskridas från kommandoraden. Bördigheten och
    # förlustandelen är härledda tal, inte fria — men de behöver kunna varieras
    # för att härledningen ska gå att pröva. Se docs/vaxternas-livscykel.md.
    wp_over: dict[str, float] = {}
    for cli, name in (
        (a.nutrient_init, "nutrient_init"),
        (a.detritus_init, "detritus_init"),
        (a.detritus_structure_init, "detritus_structure_init"),
        (a.nutrient_input, "nutrient_input"),
        (a.nutrient_loss_frac, "nutrient_loss_frac"),
        (a.uptake_rate, "uptake_rate_per_area"),
    ):
        if cli is not None:
            wp_over[name] = float(cli)

    if int(a.size) > 0:
        WP = WorldParams(size=int(a.size), width=0, height=0, dt=float(a.dt), **wp_over)
    else:
        WP = WorldParams(width=int(a.width), height=int(a.height), dt=float(a.dt), **wp_over)
    AP = AgentParams(dt=WP.dt)
    # Rörelse- och perceptionsparametrar som kommandoradsflaggor. Det som
    # avgör om perceptionen hinner med är dimensionslöst: sträcka per
    # sensingintervall genom synvidd. Vid förval är den 0,88 · 10 / 7,0 = 1,26,
    # alltså rör sig djuret längre mellan två sensingar än synfältet är långt.
    # Kvoten går att sänka från båda hållen, och båda vägarna ska kunna prövas
    # utan kodändring.
    for _cli, _name in (
        (a.drag_lin, "drag_lin"),
        (a.drag_quad, "drag_quad"),
        (a.v_max, "v_max"),
        (a.sense_idle, "sense_idle_steps"),
        (a.sense_alert, "sense_alert_steps"),
    ):
        if _cli is not None:
            setattr(AP, _name, type(getattr(AP, _name))(_cli))
    PP = PopParams(init_pop=int(a.init_pop), max_pop=int(a.max_pop))
    if getattr(a, "flora_ratio", None) is not None:
        PP.flora_init_mass_ratio = float(a.flora_ratio)
    if getattr(a, "flora_plant_mass", None) is not None:
        PP.flora_init_plant_mass = float(a.flora_plant_mass)
    if getattr(a, "sociability_init", None) is not None:
        PP.sociability_init = float(a.sociability_init)
    if getattr(a, "fauna_at", None) is not None:
        PP.fauna_at = int(a.fauna_at)
    if getattr(a, "fauna_spawn_radius", None) is not None:
        PP.fauna_spawn_radius = float(a.fauna_spawn_radius)
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


def print_summary(pop: Population, d0: dict, nb0: dict, unika: int, worst_drift: float,
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
        sp = fauna_spacing(pop)
        if sp["fauna_nn_mean"] == sp["fauna_nn_mean"]:   # inte NaN
            print(f"\n  utbredning   avstånd till närmaste artfrände: "
                  f"medel {sp['fauna_nn_mean']:.2f}, median {sp['fauna_nn_median']:.2f}")
            print(f"               {100 * sp['fauna_in_range_frac']:.1f} % har någon inom "
                  f"synhåll ({float(pop.agents[0].AP.ray_len_front):.0f}), "
                  f"mot {100 * sp['fauna_in_range_poisson']:.1f} % vid slumpmässig "
                  f"fördelning")
            print(f"               medelavstånd {sp['fauna_nn_mean']:.2f} mot "
                  f"{sp['fauna_nn_poisson']:.2f} vid slump; synellipsen täcker "
                  f"{sp['fauna_sense_area']:.0f} av {int(pop.grid.n_cells)} celler")

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

    # Takterna avgör om världen ackumulerar eller dräneras, och på vilken
    # tidsskala. Totalerna ensamma döljer det: en stock som ser stabil ut över
    # några hundra sekunder kan ha en tidskonstant på timmar.
    # Se docs/naringens-ekonomi.md.
    sim_h = float(pop.t) / 3600.0
    if sim_h > 0.0:
        add_h = (nb["added"] - nb0["added"]) / sim_h
        lost_h = (nb["lost"] - nb0["lost"]) / sim_h
        net_h = add_h - lost_h
        loss_frac = max(1e-12, float(pop.WP.nutrient_loss_frac))
        cyc_h = (lost_h / loss_frac) / max(1e-12, nb["total"])
        line = (f"               takt: +{add_h:.5f} -{lost_h:.5f} = {net_h:+.5f} kg/h")
        if abs(net_h) > 1e-12:
            line += f"   tidskonstant {abs(nb['total'] / net_h):.0f} h"
        print(line)
        print(f"               omsättning {cyc_h:.1f} varv/h, "
              f"{1.0 / loss_frac:.0f} varv innan förlust")
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
    if a.pop_log or a.world_log or a.life_log:
        from simlog.jsonl import JsonlWriter
        from simlog.sinks import EventHub
        from simlog.observers import PopLogger, WorldLogger, LifeLogger

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
        if a.life_log:
            # Ogrindad: varje födsel och dödsfall skrivs. Händelserna är få
            # jämfört med tickarna, och det är just de sällsynta vi vill se.
            w = JsonlWriter(str(a.life_log), flush_every=1)
            w.__enter__()
            writers.append(w)
            observers.append(LifeLogger(w=w))

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

    timer: PassTimer | None = None
    if a.pass_timing or a.pass_timing_inner:
        timer = PassTimer(warmup=int(a.pass_timing_warmup))
        if a.pass_timing_inner:
            timer.install_inner()
        timer.install(pop)

    d0 = diagnostics(pop)
    nb0 = nutrient_balance(pop)
    n_cells = int(pop.grid.n_cells)
    unika = {int(x.id) for x in pop.agents if x.body.alive}
    worst_drift = 0.0

    if not a.quiet:
        print(
            f"START headless: ticks={a.ticks} värld={pop.grid.width}x{pop.grid.height} ({pop.grid.n_cells} celler) dt={a.dt} "
            f"init_pop={a.init_pop} max_pop={a.max_pop} seed={seed}",
            flush=True,
        )

    # --- bildrutor utan fönster -------------------------------------------
    # Samma ritkod som den interaktiva viewern, men sparad till fil. Gör
    # rumslig fördelning granskbar över ssh, där ett fönster inte går att
    # öppna. Pygame importeras först när första bildrutan begärs, så en
    # körning utan --snapshot-every rör aldrig biblioteket.
    # --- bildruteserver ---------------------------------------------------
    server = None
    if int(a.serve or 0) > 0:
        from viewer_server import ViewerServer
        from viewframe import frame_from_pop as _frame_from_pop

        server = ViewerServer(
            host=str(a.serve_host),
            port=int(a.serve),
            fps=float(a.serve_fps),
            control=bool(a.serve_control),
        )

    snapshot_every = max(0, int(a.snapshot_every))
    _snap = {"viewer": None, "pygame": None}

    def write_snapshot(tick_no: int) -> None:
        if _snap["viewer"] is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            # SDL installerar egna hanterare för SIGTERM och SIGINT för att
            # kunna stänga fönster och ljudenheter kontrollerat. I dummy-läget
            # finns inget fönster, signalen tas emot och leder ingenstans, och
            # processen går inte att döda med `kill`. Lämna signalerna åt Python.
            os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
            import pygame as _pg
            from viewer_pygame import ViewerConfig, WorldViewer

            # Bara det som behövs. pygame.init() startar även ljud och
            # joystick — därav ALSA-varningen i loggen — utan att något av det
            # används för att spara en PNG.
            _pg.display.init()
            _pg.font.init()
            _snap["pygame"] = _pg
            _snap["viewer"] = WorldViewer(
                ViewerConfig(scale=int(a.snapshot_scale), mode=str(a.snapshot_mode))
            )
            os.makedirs(str(a.snapshot_dir), exist_ok=True)

        from viewframe import frame_from_pop

        viewer = _snap["viewer"]
        # update() kan hoppa över bildrutor enligt render_every; anropa tills
        # en faktiskt ritats.
        for _ in range(max(1, int(viewer.cfg.render_every)) + 1):
            viewer.update(frame_from_pop(pop), grid=pop.grid)
            if viewer._screen is not None:
                break
        if viewer._screen is None:
            return
        surf = viewer._screen
        if a.snapshot_crop is not None:
            # Utsnittet anges i celler och skalas till pixlar. Kartläggningen är
            # approximativ på hex — raderna är förskjutna — men räcker gott för
            # att titta på en avgränsad del av världen i hög upplösning.
            cx, cy, cw, ch = (int(v) for v in a.snapshot_crop)
            s = max(1, int(a.snapshot_scale))
            rect = _snap["pygame"].Rect(cx * s, cy * s, max(1, cw * s), max(1, ch * s))
            rect = rect.clip(surf.get_rect())
            if rect.width > 0 and rect.height > 0:
                surf = surf.subsurface(rect).copy()
        path = os.path.join(str(a.snapshot_dir), f"t{tick_no:07d}.png")
        _snap["pygame"].image.save(surf, path)

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
        if server is not None:
            # Före steget, aldrig mitt i en tick. Den pausade tiden räknas
            # bort ur den förflutna, annars blir varje ms/tick-siffra från
            # en pausad session tyst felaktig.
            server.wait_while_paused()

        tick_t0 = timer.begin_tick() if timer is not None else 0.0

        try:
            pop.step()
        except Exception as exc:  # noqa: BLE001 — vi vill se vilket tick som small
            print(f"AVBROTT i tick {tick}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            raise

        if timer is not None:
            timer.end_tick(pop, tick_t0, tick)

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

        # Fråga innan bildrutan byggs. Argumentet beräknas före anropet, så
        # ett publish() som ändå kastar bildrutan skulle kosta ett par
        # millisekunder per tick även när ingen tittar.
        if server is not None and server.wants_frame():
            server.publish(
                _frame_from_pop(
                    pop,
                    births_total=int(pop._births_total),
                    deaths_total=int(pop._deaths_total),
                )
            )

        if snapshot_every and tick % snapshot_every == 0:
            write_snapshot(tick)

        if report_every and tick % report_every == 0 and not a.quiet:
            el = time.perf_counter() - t0 - (server.paused_seconds if server else 0.0)
            if a.stats:
                nb = nutrient_balance(pop)
                worst_drift = max(
                    worst_drift, abs(nb["unaccounted"]) / max(1e-12, abs(nb["total"]))
                )
                print(format_stats(pop, diagnostics(pop), tick, el), flush=True)
            else:
                print(format_diagnostics(diagnostics(pop), tick, el), flush=True)

    elapsed = time.perf_counter() - t0 - (server.paused_seconds if server else 0.0)

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
            print_summary(pop, d0, nb0, len(unika), worst_drift, n_cells, elapsed, tick)

    if timer is not None and not a.quiet:
        timer.report()

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
