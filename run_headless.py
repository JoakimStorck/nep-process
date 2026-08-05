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
import math
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

# Kontaktmätning. Frågan "slår de följe när de möts" går inte att svara på med
# ett ögonblicksmått: kvoten mot Poisson mäter momentan täthet och låg på 0,95
# till 1,03 oavsett vad flockningen gjorde. Det som saknades var
# **varaktigheten** — hur länge en artfrände stannar inom synhåll efter första
# kontakten.
#
# Tillståndet ligger i `state`, med individens id som nyckel, inte på agenten.
# Då rör instrumenteringen ingen produktionskod och är helt borta utan
# `--stats`.
_C: dict = {"agenttick": 0, "i_kontakt": 0, "kontakter": 0,
            "langder": [], "state": {}}
_INSTRUMENTED_CONTACTS = False


def instrument_contacts() -> None:
    """
    Räkna möten och deras längd, per agenttick.

    Ett möte är en obruten följd av tick där samma artfrände ligger i
    `_cached_agent_hit`. Byter motparten identitet, eller försvinner den,
    stängs mötet och längden bokförs.

    Två tal faller ut: hur ofta djuren möts och hur länge de stannar. Det är
    det andra som säger om affiniteten gör något — ett möte som varar en tick
    är ingen flock.
    """
    global _INSTRUMENTED_CONTACTS
    if _INSTRUMENTED_CONTACTS:
        return
    _INSTRUMENTED_CONTACTS = True

    from agent import Agent

    orig = Agent._apply_reflex_drives
    st = _C["state"]

    def wrapped(self, *ar, **kw):
        out = orig(self, *ar, **kw)
        _C["agenttick"] += 1
        hit = getattr(self, "_cached_agent_hit", None)
        nid = 0
        if isinstance(hit, tuple) and len(hit) >= 5:
            hit_slot, did = hit[3], hit[4]
            if int(hit_slot) >= 0 and int(did) > 0:
                nid = int(did)
        key = int(getattr(self, "id", 0))
        prev_id, prev_n = st.get(key, (0, 0))
        if nid and nid == prev_id:
            st[key] = (nid, prev_n + 1)
            _C["i_kontakt"] += 1
        else:
            if prev_id and prev_n > 0:
                _C["langder"].append(prev_n)
                _C["kontakter"] += 1
            st[key] = (nid, 1 if nid else 0)
            if nid:
                _C["i_kontakt"] += 1
        return out

    Agent._apply_reflex_drives = wrapped


# ---------------------------------------------------------------------------
# Styrningens bidrag.
#
# `turn` bär i dag både riktning och kraft i samma tal, och riktningen väljs
# genom addition i stället för genom ett val. Nio ställen skriver additivt, på
# formen `vikt · err/π`, och `_integrate_motion` tolkar summan som ett normerat
# kursfel. Ett beteende som skriver `0,40 · err/π` påstår därmed inte "jag vill
# dit med halv kraft" utan "kursfelet är 40 procent av vad det är". Riktningen
# ljugs ned för att uttrycka svag prioritet.
#
# Två tal ska falla ut, och de är facit att bygga mot innan arbitreringen
# byggs:
#
#   kansellering — andelen agenttick där `|turn|` är väsentligt mindre än
#     summan av bidragens belopp. Drar två termer isär tar de ut varandra mot
#     noll, vilket i `_integrate_motion` betyder rakt fram. Vid en T-korsning
#     är medelvägen diket.
#
#   riktningsavvikelse — andelen agenttick där den vinnande grenens egen
#     riktning skiljer sig mer än trettio grader från det `turn` som faktiskt
#     blev efter att kyla och föda lagts på. Trettio grader är ett mjukt mått:
#     en avvikelse dit kan vara oskyldig. Nittio kan den inte vara — då styr
#     djuret in i fel halvplan mot vad grenen ville — och andelen över nittio
#     redovisas därför vid sidan av.
#
# **Kanselleringen behöver en nollhypotes.** Fyra bidrag med oberoende tecken
# tar ut varandra av sig själva; utan att veta vad ofarlig addition ger går
# talet inte att tolka. Nollfördelningen håller beloppen fasta och randomiserar
# tecknen — men den *räknas upp exakt* i stället för att dras. Med fyra bidrag
# finns sexton teckenmönster, åtta upp till en global spegling som inte ändrar
# `|Σ|`. Åtta additioner per tick är billigare än ett enda dragningsanrop och
# har ingen Monte Carlo-brusnivå alls, till skillnad från permutationsnivån i
# `genopheno_analyze.py` där rummet är för stort för uppräkning.
#
# Nollfördelningen inkluderar det observerade mönstret som ett av åtta fall.
# Det är avsiktligt: frågan är inte om additionen kancellerar mer än slumpen i
# någon strikt mening utan hur mycket av den uppmätta kanselleringen som är en
# egenskap hos konstruktionen snarare än hos beteendet.
#
# Bidragen mäts som differenser runt de tre metoder som var för sig skriver i
# `turn`, inte genom att grenlogiken replikeras här:
#
#   turn_mlp   ut ur `_decode_action_outputs`      tanh(y[0])
#   Δ_kyla     in till `_apply_reflex_drives`      minus turn_mlp
#   Δ_gren     ut ur `_apply_reflex_drives`        minus dess indata
#   Δ_föda     ut ur `_apply_food_steering`        minus dess indata
#
# Summan av de fyra är per konstruktion exakt det `turn` som blir, så
# klampningen i varje `clamp(turn + …)` ligger redan inbakad i differenserna
# och räknas som förlorad styrvilja. Parningsgrenen skriver `turn = 0,95·bias`
# i stället för `turn +=`, alltså kastar den MLP:n och kylan; det syns här som
# kansellering, vilket är rätt i sak — styrviljan går förlorad.
#
# Att jämföra magnitud som om den vore riktning är avsiktligt. I dagens
# semantik *är* magnituden riktning: `d_steer = frac · turn · π`, så en ändrad
# magnitud är en ändrad riktning djuret svänger mot. Det är hela felet.
#
# Kvantilerna tas ur histogram i stället för sparade listor. Vid tjugo djur och
# 80 000 tick vore en lista per mått 1,6 miljoner tal; facken kostar inget och
# ger medianen på ett par procent när.
#
# Tillståndet ligger i `state`, med individens id som nyckel, inte på agenten.
# Produktionskoden rörs alltså inte, och instrumenteringen är helt borta utan
# `--stats`.
_S: dict = {"agenttick": 0, "kans_n": 0, "kans_traff": 0, "kans_hist": [0] * 20,
            "null_traff": 0.0, "null_hist": [0.0] * 20,
            "gren_n": 0, "gren_traff": 0, "gren_traff90": 0, "avv_hist": [0] * 18,
            "kalla": {}, "gren": {}, "state": {},
            # Fönstret är **skilt** från totalerna ovan och nollställs vid
            # avläsning, som `gestation_window()`. Totalerna nollställs per
            # körning, som `_C`. Att blanda de två mönstren i samma räknare är
            # den fälla som gör ett block tyst tomt: läser man av en räknare
            # som bara nollställs per körning växer den obegränsat, och
            # nollställer man en total vid avläsning är den alltid noll i
            # sammanfattningen.
            #
            # [kans_n, kans_traff, null_traff, gren_n, gren_traff, gren_traff90]
            "w": [0, 0, 0.0, 0, 0, 0]}


def steering_window() -> tuple[float, float, float, float] | None:
    """
    Kansellering, överskott mot nollhypotesen och riktningsavvikelse sedan
    förra avläsningen, i procent. Nollställer fönstret.

    Över 80 000 tick är en enda total ett medelvärde över tre olika regimer:
    tjugo grundare i en flora som självgallrar, ett växande bestånd, och vad
    jämvikten nu blir. Frågan om kanselleringen beror på täthet — flockgrenen
    kan bara vinna när någon syns — går inte att ställa till ett sådant tal.
    """
    w = _S["w"]
    if w[0] <= 0 and w[3] <= 0:
        return None
    kn = max(1, w[0])
    gn = max(1, w[3])
    out = (100.0 * w[1] / kn, 100.0 * (w[1] - w[2]) / kn,
           100.0 * w[4] / gn, 100.0 * w[5] / gn)
    _S["w"] = [0, 0, 0.0, 0, 0, 0]
    return out

# De åtta teckenmönstren, första tecknet fixerat till +. `|Σ|` är oförändrat
# under global spegling, så de resterande åtta är dubletter.
_S_TECKEN = ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
             (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1))

# Grenarnas nominella vikter, speglade ur `_apply_reflex_drives` och
# `_apply_food_steering`. De används bara för att räkna ut styrningens
# tidskonstant nedan. Att de står på två ställen är avsiktligt tillfälligt:
# steg 3 och 4 flyttar vikterna till arbitreringen respektive styrkraften, och
# då har raden inga hårdkodade tal kvar att spegla.
_S_VIKTER = (("flykt", 0.95), ("jakt", 0.90), ("parning", 0.95),
             ("flock", 0.60), ("föda", 0.36))

# Reflexkedjans grenar i elif-ordning. Skild från viktlistan ovan, som också
# bär födan — den är inte en gren utan ett påslag efter kedjan.
_S_GRENAR = ("flykt", "jakt", "parning", "flock")

_INSTRUMENTED_STEERING = False


def _hist_median(h: list, lo: float, hi: float) -> float:
    """Median ur jämnbreda fack, linjärt interpolerad inom det träffade."""
    n = sum(h)
    if n <= 0:
        return float("nan")
    w = (hi - lo) / len(h)
    half = 0.5 * n
    acc = 0
    for i, c in enumerate(h):
        if c > 0 and acc + c >= half:
            return lo + w * (i + (half - acc) / c)
        acc += c
    return hi


def instrument_steering() -> None:
    """
    Bokför varje additivt bidrag till `turn`, per agenttick.

    Lindar de tre metoder som skriver i `turn`. Ingen av dem ändras, och
    grenvalet läses ur returvärdet — `flee_state` och `hunt_state` — i stället
    för att elif-kedjans villkor skrivs av en gång till.
    """
    global _INSTRUMENTED_STEERING
    if _INSTRUMENTED_STEERING:
        return
    _INSTRUMENTED_STEERING = True

    from agent import Agent

    st = _S["state"]
    orig_decode = Agent._decode_action_outputs
    orig_reflex = Agent._apply_reflex_drives
    orig_food = Agent._apply_food_steering

    def turn_in(ar, kw, pos):
        if "turn" in kw:
            return float(kw["turn"])
        return float(ar[pos]) if len(ar) > pos else 0.0

    def wrapped_decode(self, *ar, **kw):
        out = orig_decode(self, *ar, **kw)
        # [turn_mlp, Δ_kyla, Δ_gren, gren, turn_ut_ur_reflex]
        st[int(getattr(self, "id", 0))] = [float(out[0]), 0.0, 0.0, "", 0.0]
        return out

    def wrapped_reflex(self, *ar, **kw):
        t_in = turn_in(ar, kw, 0)
        out = orig_reflex(self, *ar, **kw)
        rec = st.get(int(getattr(self, "id", 0)))
        if rec is None:
            return out
        rec[1] = t_in - rec[0]
        rec[2] = float(out[0]) - t_in
        rec[4] = float(out[0])
        if float(out[3]) > 0.0:
            rec[3] = "flykt"
        elif float(out[4]) > 0.0:
            rec[3] = "jakt"
        elif bool(kw.get("in_mating_mode", False)) and kw.get("best_mate") is not None:
            rec[3] = "parning"
        elif abs(rec[2]) > 1e-12:
            # Flockgrenen kan avfyra utan att röra `turn` — då har den ingen
            # riktning att avvika från och räknas inte som vinnare.
            rec[3] = "flock"
        return out

    def wrapped_food(self, *ar, **kw):
        t_in = turn_in(ar, kw, 1)
        out = orig_food(self, *ar, **kw)
        rec = st.pop(int(getattr(self, "id", 0)), None)
        if rec is None:
            return out

        t_fin = float(out[0])
        b = (abs(rec[0]), abs(rec[1]), abs(rec[2]), abs(t_fin - t_in))
        tot = b[0] + b[1] + b[2] + b[3]
        # Läses om per anrop: `steering_window()` binder om listan vid
        # avläsning, så en bindning vid installationen hade lämnat wrappern på
        # det första fönstret för alltid.
        w = _S["w"]

        _S["agenttick"] += 1
        for namn, v in zip(("mlp", "kyla", "gren", "föda"), b):
            if v > 1e-12:
                e = _S["kalla"].setdefault(namn, [0, 0.0])
                e[0] += 1
                e[1] += v

        # Brusgolvet håller borta tick där ingen vill någonstans: en summa på
        # några tusendelar kan ta ut sig själv fullständigt utan att det
        # betyder något.
        if tot > 0.02:
            _S["kans_n"] += 1
            w[0] += 1
            f = 1.0 - abs(t_fin) / tot
            _S["kans_hist"][min(19, max(0, int(f * 20.0)))] += 1
            if abs(t_fin) < 0.5 * tot:
                _S["kans_traff"] += 1
                w[1] += 1

            # Nollfördelningen: samma belopp, alla teckenmönster, en åttondel
            # vikt vardera.
            b0, b1, b2, b3 = b
            nh = _S["null_hist"]
            for s1, s2, s3 in _S_TECKEN:
                sn = abs(b0 + s1 * b1 + s2 * b2 + s3 * b3)
                fn = 1.0 - sn / tot
                nh[min(19, max(0, int(fn * 20.0)))] += 0.125
                if sn < 0.5 * tot:
                    _S["null_traff"] += 0.125
                    w[2] += 0.125

        gren = rec[3]
        if gren:
            _S["gren_n"] += 1
            w[3] += 1
            # Referensen är grenens *egen* riktning, och den är inte samma
            # storhet för de två grentyperna.
            #
            # Åtta grenar adderar: `turn = clamp(turn + w · bias)`. Deras egen
            # riktning är differensen `Δ_gren`, eftersom det är precis vad de
            # begärde utöver vad som redan låg där.
            #
            # Parningsgrenen tilldelar: `turn = clamp(0,95 · biasN)`. Dess egen
            # riktning är hela det tilldelade värdet, alltså `turn` ut ur
            # reflexkedjan — inte differensen mot det den kastade. Mäts den mot
            # differensen läggs MLP:ns och kylans belopp till avvikelsen, trots
            # att grenen medvetet gjorde sig av med dem. Det var precis vad
            # 0136 gjorde, och det gav parningsgrenen 22,6 % i p140 mot
            # flockens 18,1 — en siffra som mätte grenens val i stället för
            # vad som stördes efteråt. För den grenen är den enda verkliga
            # störningen födotermen.
            ref = rec[4] if gren == "parning" else rec[2]
            d = (t_fin - ref) * math.pi
            d = abs((d + math.pi) % (2.0 * math.pi) - math.pi)
            grader = d * 180.0 / math.pi
            _S["avv_hist"][min(17, int(grader / 10.0))] += 1
            e = _S["gren"].setdefault(gren, [0, 0, 0])
            e[0] += 1
            if grader > 30.0:
                _S["gren_traff"] += 1
                e[1] += 1
                w[4] += 1
            if grader > 90.0:
                _S["gren_traff90"] += 1
                e[2] += 1
                w[5] += 1
        return out

    Agent._decode_action_outputs = wrapped_decode
    Agent._apply_reflex_drives = wrapped_reflex
    Agent._apply_food_steering = wrapped_food


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
    ap.add_argument("--flora-growth", type=str, default=None,
                    choices=("numpy", "numba"),
                    help="väg för florans tillväxtpass; standard är numba "
                         "när biblioteket finns")
    ap.add_argument("--bench-flora-growth", type=int, default=0,
                    help="mät tillväxtpassets vägar mot varandra efter N tick "
                         "uppvärmning; interfolierade varv i samma process")
    ap.add_argument("--bench-rounds", type=int, default=5,
                    help="antal interfolierade varv i --bench-flora-growth")
    ap.add_argument("--bench-ticks", type=int, default=8,
                    help="tick per väg och varv i --bench-flora-growth")
    ap.add_argument("--verify-flora-growth", type=int, default=0,
                    help="kör två identiska världar med var sin väg i N tick "
                         "och jämför tillståndet elementvis efter varje tick")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--stats", action="store_true",
                    help="bredare diagnostikrad och sammanfattning vid slut")
    ap.add_argument("--seeds", type=str, default=None,
                    help="kommaseparerad lista, t.ex. 1,2,3 — kör flera och jämför")
    ap.add_argument("--scenario", type=str, default=None,
                    help="YAML-fil med körningens utgångsläge; se scenario.py")
    ap.add_argument("--scenario-out", type=str, default=None,
                    help="skriv scenariot hit för spårbarhet")
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
    ap.add_argument("--sociability-init-sd", type=float, default=None,
                    help="spridning kring --sociability-init, i logit-rymden; 0 = identiska")
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
    if getattr(a, "flora_growth", None) is not None:
        PP.flora_growth_backend = str(a.flora_growth)
    if getattr(a, "flora_ratio", None) is not None:
        PP.flora_init_mass_ratio = float(a.flora_ratio)
    if getattr(a, "flora_plant_mass", None) is not None:
        PP.flora_init_plant_mass = float(a.flora_plant_mass)
    if getattr(a, "sociability_init", None) is not None:
        PP.sociability_init = float(a.sociability_init)
    if getattr(a, "sociability_init_sd", None) is not None:
        PP.sociability_init_sd = float(a.sociability_init_sd)
    if getattr(a, "fauna_at", None) is not None:
        PP.fauna_at = int(a.fauna_at)
    if getattr(a, "fauna_spawn_radius", None) is not None:
        PP.fauna_spawn_radius = float(a.fauna_spawn_radius)
    _sc = getattr(a, "_scenario", None)
    if _sc is not None:
        PP.fauna_spawn_patches = int(_sc.fauna.flackar)
        PP.founder_group_sep = float(_sc.fauna.grupp_avstand)
        PP.founder_group_spread = float(_sc.fauna.grupp_spridning)
    return Population(WP=WP, AP=AP, PP=PP, seed=int(seed), hub=hub)


# Takten mättes kumulativt från tick noll, vilket gör den till ett medel över
# hela körningen i stället för ett mått på nuläget. I p124 stod den på 44,20
# vid tick 1 000 och 22,17 vid 30 000 — och den siffran fortsatte sjunka länge
# efter att takten planat ut, eftersom historiken dominerade. Marginaltakten i
# slutet var 26,8 medan raden skrev 27,17.
#
# Fönstret är senaste rapportintervallet, alltså redan ett medel över
# `--report-every` tick. Det kumulativa medlet står kvar bredvid, eftersom det
# är rätt tal för frågan "hur lång tid tar resten".
_RATE = {"tick": 0, "elapsed": 0.0}


def tick_rate(tick: int, elapsed: float) -> tuple[float, float]:
    """Takt över senaste intervallet, och kumulativt medel. Båda i ms/tick."""
    prev_t = int(_RATE["tick"])
    prev_e = float(_RATE["elapsed"])
    d_ticks = int(tick) - prev_t
    recent = (
        (float(elapsed) - prev_e) / d_ticks * 1000.0
        if d_ticks > 0 else float(elapsed) / max(1, int(tick)) * 1000.0
    )
    _RATE["tick"] = int(tick)
    _RATE["elapsed"] = float(elapsed)
    return recent, float(elapsed) / max(1, int(tick)) * 1000.0


# Dräktigheten räknades som en ögonblicksbild i den tick raden skrevs. Det är
# fel sorts mått på en säsongsbunden process: gestationen har uppmätt period
# 12,03 månader, alltså exakt `WorldParams.year_len`, medan rapportintervallet
# på tusen tick är 20 månader. Ett år samplat var 1,67 år ger en svävning med
# period tre, inte en trend — och det är den som ligger bakom att p124 växlade
# mellan `dräkt=0/0%` och `dräkt=39/54%` mellan varannan rad.
#
# Måttet är därför tidsmedlet över rapportintervallet, plus toppen. Medlet
# svarar på hur mycket reproduktion som pågår, toppen på hur stor kullen blir.
#
# Median-andelen av målmassan är borta. Den var meningsfull bara som
# ögonblicksbild och därmed aliasad på samma sätt.
_GEST = {"sum": 0.0, "ticks": 0, "peak": 0}


def gestation_window() -> tuple[float, int]:
    """Medel och topp sedan förra avläsningen. Nollställer fönstret."""
    n = max(1, int(_GEST["ticks"]))
    out = (float(_GEST["sum"]) / n, int(_GEST["peak"]))
    _GEST["sum"] = 0.0
    _GEST["ticks"] = 0
    _GEST["peak"] = 0
    return out


def format_stats(pop: Population, d: dict, tick: int, elapsed: float) -> str:
    nb = nutrient_balance(pop)
    g_mean, g_peak = gestation_window()
    recent, mean = tick_rate(tick, elapsed)
    # Styrningen som fyra tal per fönster: kansellering, dess överskott mot
    # nollhypotesen, och andelen tick som styr mer än 30 respektive 90 grader
    # från vinnande grenens riktning. Sammanfattningen ger samma tal som en
    # total över hela körningen; det här ger dem över tid, så att en
    # täthetsberoende drift syns. Vid --report-every 600 är ett fönster exakt
    # ett år.
    sw = steering_window()
    # Utan breddspecificerare. Med `4.1f` får ett värde under tio ett inledande
    # blanksteg, och fältet blir då fyra tokens i stället för ett — varje
    # radparsare som delar på blanksteg tappar tyst just de fönster där
    # avvikelsen är låg, alltså systematiskt de välmatade. Det hände i
    # analysen av p140 och gjorde korrelationen mot massa per djur −0,65 i
    # stället för −0,82.
    styr = (f"  styr={sw[0]:.1f}/{sw[1]:+.1f}/{sw[2]:.1f}/{sw[3]:.1f}"
            if sw is not None else "")
    # Varje fält bär sitt eget prefix i stället för att ligga under en
    # gruppetikett. `n_` är antal, `M_` massa i kilo torrsubstans, `N_` näring i
    # kilo. Skillnaden mot grupper är att fältet blir självbeskrivande: `grep
    # M_flora` fungerar oavsett var i raden det står, och läsaren behöver inte
    # hålla reda på vilken grupp hen befinner sig i.
    #
    # Bakgrunden är att raden en gång skrev `detritus=325549` och
    # `fri_när=11423` bredvid varandra — kilo torrsubstans mot kilo näring, samma
    # enhet, olika storhet. Att förnan väger hundra gånger mer än den bär i
    # näring är hela poängen med strukturandelen: den är kol, inte kväve.
    #
    # `M_`-prefixen är desamma som i världsloggen (`M_detritus`, `M_carcass`),
    # så samma namn betyder samma sak i konsol och logg.
    return (
        f"tick {tick:7d}  t={d['t']:8.1f}  "
        f"n_fauna={d['fauna_n']:4d}  n_flora={d['flora_n']:6d}  "
        f"rotandel={d.get('flora_root_frac', 0.0):4.2f}  "
        f"M_fauna={d['fauna_mass_kg']:7.1f}  M_flora={d['flora_mass_kg']:9.3e}  "
        f"M_förna={d['detritus_mass_kg']:9.3e}  "
        f"M_kadaver={d.get('carcass_mass_kg', 0.0):7.2f}  "
        f"N_fri={nb['free']:8.0f}  N_flora={nb['in_flora']:7.0f}  "
        f"N_förna={nb.get('in_litter', nb['in_detritus']):7.0f}  "
        f"n_föd={pop._births_total:4d}  n_död={pop._deaths_total:4d}  "
        f"dräkt={g_mean:5.1f}/{g_peak:3d}  "
        f"{styr}  "
        f"{recent:.2f} ms/tick (medel {mean:.2f})"
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

    if _C["agenttick"]:
        dt = float(pop.WP.dt)
        L = np.asarray(_C["langder"], dtype=np.float64)
        rate = _C["kontakter"] / max(1e-9, _C["agenttick"] * dt)
        print(f"\n  kontakt      {_C['kontakter']} avslutade möten, "
              f"{rate:.3f} per djur och månad")
        if L.size:
            print(f"               längd median {np.median(L) * dt:.2f} mån, "
                  f"p90 {np.percentile(L, 90) * dt:.2f}, max {L.max() * dt:.1f}")
        print(f"               sällskap {100 * _C['i_kontakt'] / _C['agenttick']:.1f} % "
              f"av agenttickarna")

        print(f"\n  reproduktion {100 * _R['redo'] / _R['agenttick']:.1f} % av agenttickarna är klara")
        print(f"    ser ingen alls {100 * _R['ser_ingen'] / redo:5.1f} %"
              f"   utanför parningsradie {100 * _R['utanfor_radie'] / redo:5.1f} %")
        print(f"    parningar      {_R['parning']:5d}"
              f"     såg partner men parade inte {_R['sag_men_parade_ej']:5d}")
        print(f"    täthet {d['fauna_n'] / max(1, n_cells) * 1000.0:.2f} agenter per 1000 celler")

    if _S["agenttick"]:
        n = _S["agenttick"]
        print(f"\n  styrning     {n} agenttick genom hela styrkedjan")
        if _S["kans_n"]:
            n_k = _S["kans_n"]
            med = _hist_median(_S["kans_hist"], 0.0, 1.0)
            med0 = _hist_median(_S["null_hist"], 0.0, 1.0)
            print(f"    kansellering  {100 * _S['kans_traff'] / n_k:5.1f} % av "
                  f"tickarna tappar över hälften av styrviljan, median "
                  f"{100 * med:.0f} % av bidragens belopp")
            print(f"      nollhypotes {100 * _S['null_traff'] / n_k:5.1f} % och "
                  f"{100 * med0:.0f} % vid slumpmässiga tecken och samma belopp "
                  f"— överskott {100 * (_S['kans_traff'] - _S['null_traff']) / n_k:+.1f} "
                  f"respektive {100 * (med - med0):+.0f} enheter")
        if _S["gren_n"]:
            n_g = _S["gren_n"]
            medg = _hist_median(_S["avv_hist"], 0.0, 180.0)
            print(f"    riktning      {100 * _S['gren_traff'] / n_g:5.1f} % av "
                  f"tickarna styr mer än 30° från vinnande grenens egen "
                  f"riktning, median {medg:.0f}°")
            print(f"      över 90°, alltså in i fel halvplan: "
                  f"{100 * _S['gren_traff90'] / n_g:5.1f} %")
            for k in _S_GRENAR:
                e = _S["gren"].get(k)
                if e and e[0]:
                    print(f"      {k:<8} vann {e[0]:8d} tick, "
                          f"{100 * e[1] / e[0]:5.1f} % över 30°, "
                          f"{100 * e[2] / e[0]:5.1f} % över 90°")
        for k in ("mlp", "kyla", "gren", "föda"):
            e = _S["kalla"].get(k)
            if e and e[0]:
                print(f"      {k:<8} skrev {100 * e[0] / n:5.1f} % av tickarna, "
                      f"medelbelopp {e[1] / e[0]:.3f}")

        # Loopbandbredden är i dag en bieffekt av prioritetsvikten: ett
        # beteende som skriver `w · err/π` rätar ut kursfelet med
        # tidskonstanten `1/(frac · w)` tick. Kvoten mot mediankontakten är
        # måttet steg 4 ska kalibrera styrkraften mot — en flock kan inte
        # bildas om djuret svänger långsammare än mötet varar.
        AP = pop.agents[0].AP if pop.agents else None
        if AP is not None:
            dt_w = float(pop.WP.dt)
            frac = 1.0 - math.exp(-float(AP.turn_gain) * dt_w)
            rader = ", ".join(f"{k} {1.0 / (frac * w):.0f}" for k, w in _S_VIKTER)
            print(f"    bandbredd    frac {frac:.3f} per tick "
                  f"(turn_gain {float(AP.turn_gain):.1f}, dt {dt_w:.3f})")
            print(f"      tidskonstant vid full vikt: {rader} tick")
            L2 = np.asarray(_C["langder"], dtype=np.float64)
            if L2.size:
                kont = float(np.median(L2))
                tau_flock = 1.0 / (frac * 0.60)
                print(f"      mot mediankontakt {kont:.0f} tick — kvot "
                      f"{tau_flock / max(1e-9, kont):.1f} för flocken")

    print(f"\n  näring (kg)  fri {nb['free']:.2f}  flora {nb['in_flora']:.2f}  "
          f"fauna {nb['in_fauna']:.2f}  förna {nb.get('in_litter', nb['in_detritus']):.2f}  "
          f"kadaver {nb.get('in_carcass', 0.0):.2f}")
    tot_n = max(1e-12, nb['total'])
    print(f"               andelar   fri {nb['free'] / tot_n * 100:.0f} %  "
          f"flora {nb['in_flora'] / tot_n * 100:.0f} %  "
          f"död {(nb['in_detritus']) / tot_n * 100:.0f} %")
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
    recent, mean = tick_rate(tick, elapsed)
    return (
        f"tick {tick:7d}  t={d['t']:8.1f}  "
        f"n_fauna={d['fauna_n']:4d}  n_flora={d['flora_n']:5d}  "
        f"fria={d['free_slots']:5d}/{d['capacity']:5d}  "
        f"M_flora={d['flora_mass_kg']:.4e}  M_fauna={d['fauna_mass_kg']:.4e}  "
        # Samma delning som i `format_stats`: `detritus` är förnan sedan 0121,
        # och kadavret är en egen pool. Den korta raden saknade den helt.
        f"M_förna={d['detritus_mass_kg']:.4e}  "
        f"M_kadaver={d.get('carcass_mass_kg', 0.0):.2f}  "
        f"{recent:.2f} ms/tick (medel {mean:.2f})"
    )


def run(a: argparse.Namespace, seed: int | None = None) -> int:
    seed = int(a.seed) if seed is None else int(seed)

    if a.stats:
        instrument_mating()
        instrument_contacts()
        instrument_steering()
        for k in _R:
            _R[k] = 0

    # `--seeds` kör flera världar i samma process; fönstret får inte bära över.
    _RATE["tick"] = 0
    _RATE["elapsed"] = 0.0
    _GEST["sum"] = 0.0
    _GEST["ticks"] = 0
    _GEST["peak"] = 0
    _C["agenttick"] = 0
    _C["i_kontakt"] = 0
    _C["kontakter"] = 0
    _C["langder"] = []
    # In-place. `instrument_contacts()` band `st = _C["state"]` vid
    # installationen, så en ombindning här lämnade wrappern kvar på den gamla
    # dicten. Vid ett frö är det ofarligt — tillståndet börjar ändå tomt — men
    # `--seeds` installerar en gång och kör flera världar, och då bar
    # kontakterna över: en agent med samma id i nästa värld fortsatte ett möte
    # som avslutades i den förra. `langder` behöver inte samma behandling
    # eftersom den bara nås som `_C["langder"].append(...)`.
    _C["state"].clear()
    # Per körning, som `_C` — inte vid avläsning, som `gestation_window()`.
    # Hamnar en räknare i det andra mönstret är den alltid noll när blocket
    # ska skrivas ut, och blocket hoppas tyst över.
    _S["agenttick"] = 0
    _S["kans_n"] = 0
    _S["kans_traff"] = 0
    _S["kans_hist"] = [0] * 20
    _S["null_traff"] = 0.0
    _S["null_hist"] = [0.0] * 20
    _S["gren_n"] = 0
    _S["gren_traff"] = 0
    _S["gren_traff90"] = 0
    _S["avv_hist"] = [0] * 18
    _S["kalla"] = {}
    _S["gren"] = {}
    _S["w"] = [0, 0, 0.0, 0, 0, 0]
    # In-place: wrappern band `st = _S["state"]` vid installationen, så en
    # ombindning här hade lämnat den kvar på den gamla dicten och burit över
    # tillstånd mellan seeds i `--seeds`.
    _S["state"].clear()

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


def apply_scenario(a: argparse.Namespace) -> None:
    """
    Låt scenariot sätta de flaggor det äger, om de inte givits explicit.
    Uttryckliga flaggor vinner, så en fil kan användas som utgångsläge och
    enskilda tal varieras ovanpå den.
    """
    if not getattr(a, "scenario", None):
        return
    from scenario import Scenario

    sc = Scenario.load(a.scenario)
    print(sc.summary(), flush=True)

    def setdef(name, value):
        if getattr(a, name, None) in (None, 0) or name in ("width", "height"):
            setattr(a, name, value)

    a.width = int(sc.varld.bredd)
    a.height = int(sc.varld.hojd)
    a.dt = float(sc.varld.dt)
    if a.nutrient_input is None:
        a.nutrient_input = sc.nutrient_input
    if a.nutrient_init is None:
        a.nutrient_init = sc.nutrient_init
    if a.detritus_init is None:
        a.detritus_init = sc.detritus_init
    if a.drag_lin is None:
        a.drag_lin = sc.drag_lin
    if a.fauna_at is None:
        a.fauna_at = sc.fauna_at_tick
    if a.fauna_spawn_radius is None:
        a.fauna_spawn_radius = float(sc.fauna.flackradie)
    if getattr(a, "sociability_init", None) is None and sc.fysiologi.sociability is not None:
        a.sociability_init = float(sc.fysiologi.sociability)
    if getattr(a, "sociability_init_sd", None) is None:
        a.sociability_init_sd = float(sc.fysiologi.sociability_sd)
    a.init_pop = int(sc.fauna.antal)
    a.max_pop = int(sc.fauna.max_antal)
    a._scenario = sc

    out = getattr(a, "scenario_out", None)
    if out:
        sc.dump(out)


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
            n_gest = 0
            for x in pop.agents:
                if x.body.alive:
                    unika.add(int(x.id))
                    if bool(x.body.gestating):
                        n_gest += 1
            _GEST["sum"] += n_gest
            _GEST["ticks"] += 1
            if n_gest > _GEST["peak"]:
                _GEST["peak"] = n_gest

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
            # Servern vet vilka celler viewrarna zoomat in på; bara de får
            # anspråksrader. Utan urvalet är rutan 25,6 MB vid 818 000 plantor,
            # varav 24 är rader som ritas i fyra pixlar per cell.
            server.publish(
                _frame_from_pop(
                    pop,
                    births_total=int(pop._births_total),
                    deaths_total=int(pop._deaths_total),
                    claim_cells=server.claim_cells(pop.grid),
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


def bench_flora_growth(a: argparse.Namespace, seed: int, warmup: int,
                       rounds: int = 5, per: int = 8) -> int:
    """
    Tillväxtpassets vägar mätta mot varandra i samma process.

    Absoluttal är inte jämförbara mellan sessioner, maskiner eller bestånd, så
    ett tal ur en körning och ett ur en annan säger ingenting om varandra. Det
    som går att jämföra är två vägar som räknat på samma värld i samma
    process, med varven interfolierade så att drift i beståndet och i maskinens
    tillstånd träffar båda lika.

    Mätningen sker under `pop.step()`, inte genom att anropa passet i en loop.
    Ett upprepat anrop utan spridning låter floran krympa mellan mätpunkterna
    och mäter därmed ett bestånd som inte finns. Passets egen tid tas med
    `perf_counter` runt anropet, alltså samma oförvrängda metod som
    `--pass-timing` — `--profile` tredubblar körtiden och förvränger
    fördelningen.

    Uppvärmningen bär två saker som inte hör till den stationära kostnaden:
    Numbas kompilering vid första anropet, och att floran ska hinna till det
    bestånd man vill mäta vid.
    """
    import flora_growth

    backends = list(flora_growth.available_backends())
    if len(backends) < 2:
        print("bara en väg byggd i den här miljön; inget att jämföra")
        return 1

    pop = build_population(a, seed)

    acc = {b: 0.0 for b in backends}
    wall = {b: 0.0 for b in backends}
    flora = {b: 0.0 for b in backends}
    calls = {b: 0 for b in backends}
    state = {"mode": backends[-1]}

    orig = pop._growth_system_flora

    def timed():
        t0 = time.perf_counter()
        try:
            return orig()
        finally:
            acc[state["mode"]] += time.perf_counter() - t0

    pop._growth_system_flora = timed

    print(f"värmer upp {warmup} tick (kompilering och beståndets bana)", flush=True)
    for _ in range(int(warmup)):
        pop.step()

    # Uppvärmningen bokfördes på den väg som råkade vara vald, kompilering
    # och allt. Nollställ innan mätningen börjar.
    for d in (acc, wall, flora):
        for k in d:
            d[k] = 0.0
    for k in calls:
        calls[k] = 0

    n0 = len(pop._flora_slots(rebuild=True))
    print(f"mäter: {rounds} varv x {per} tick per väg, flora {n0}", flush=True)
    for r in range(int(rounds)):
        # Ordningen vänds varannat varv. Beståndet driver monotont under
        # mätningen — floran föll tre procent under en verklig mätning — och med
        # fast ordning träffar den driften vägarna olika. Det gynnade den som
        # kördes sist med ungefär en procent, vilket är i samma storleksordning
        # som skillnaden mellan njit och prange.
        for b in (backends if r % 2 == 0 else backends[::-1]):
            state["mode"] = b
            pop._flora_growth_mode = b
            t0 = time.perf_counter()
            for _ in range(int(per)):
                pop.step()
                flora[b] += float(len(pop._flora_slots()))
                calls[b] += 1
            wall[b] += time.perf_counter() - t0

    n1 = len(pop._flora_slots(rebuild=True))
    drift = abs(n1 - n0) / max(1.0, float(n0))
    print(f"\n--- tillväxtpassets vägar ---")
    print(f"  flora {n0} -> {n1} under mätningen"
          + (f"  ({drift * 100:.1f} % drift — kvoterna bär sämre än vanligt)"
             if drift > 0.05 else ""))
    print()
    print(f"  {'väg':<10} {'passet':>10} {'hel tick':>10} {'us/planta':>11} "
          f"{'passets andel':>14}")
    for b in backends:
        c = max(1, calls[b])
        p_ms = acc[b] / c * 1e3
        t_ms = wall[b] / c * 1e3
        us = wall[b] / c / max(1.0, flora[b] / c) * 1e6
        print(f"  {b:<10} {p_ms:8.2f} ms {t_ms:8.2f} ms {us:10.3f}  "
              f"{acc[b] / max(1e-12, wall[b]) * 100:12.1f} %")
    print()
    base = backends[0]
    for b in backends[1:]:
        print(f"  {base} -> {b}: passet {acc[base] / max(1e-12, acc[b]):.2f}x, "
              f"hel tick {wall[base] / max(1e-12, wall[b]):.2f}x")
    print("\n  Talen gäller det här beståndet och den här maskinen. Kvoten är "
          "det som bär\n  mellan körningar; millisekunderna gör det inte.")
    return 0


# Fält som tillväxtpasset skriver, plus de världsfält det rör. Jämförelsen
# gäller hela slotrymden och hela cellrymden, inte ett urval — ett fel som bara
# syns i en död slot är fortfarande ett fel, eftersom sloten återanvänds.
_VERIFY_STORE_FIELDS = (
    "mass", "energy", "alive", "cell_idx",
    "flora_root_mass", "flora_reserve", "flora_repro_pool", "flora_carbon_pool",
    "flora_cell_claimed", "flora_claim_share", "flora_claim_cell",
)
_VERIFY_WORLD_FIELDS = ("nutrient", "detritus", "detritus_structure")


def _max_rel(a_arr, b_arr) -> tuple[float, float]:
    """
    Största absoluta avvikelsen, och samma tal mot fältets egen skala.

    Elementvis relativfel är missvisande här: en reserv som råkar stå på 1e-11
    ger relativfel 1e-8 av en avvikelse i sista biten, vilket säger något om
    talets storlek och ingenting om räkningen. Nämnaren är därför fältets
    största magnitud, vilket är det mått som faktiskt svarar på frågan om
    avvikelsen är avrundning eller fel.
    """
    x = np.asarray(a_arr, dtype=np.float64).ravel()
    y = np.asarray(b_arr, dtype=np.float64).ravel()
    if x.shape != y.shape:
        return (float("inf"), float("inf"))
    if x.size == 0:
        return (0.0, 0.0)
    d = float(np.abs(x - y).max())
    scale = float(np.abs(x).max())
    return (d, d / scale if scale > 0.0 else 0.0)


def verify_flora_growth(a: argparse.Namespace, seed: int, ticks: int) -> int:
    """
    Två identiska världar, en per väg, jämförda elementvis varje tick.

    Det här är verifieringen som säger något. Att invariantsviten går igenom
    betyder bara att kärnan inte bryter mot bevarandelagarna — den skulle göra
    det även om upptaget fördelades fel mellan plantor, så länge summan stämde.
    Här jämförs i stället varje skrivet fält, i hela slot- och cellrymden.

    Bitidentitet är inte målet och inte möjlig: `exp` och `pow` skiljer sig i
    sista biten mellan numpys vektoriserade och libms skalära variant, och de
    tre returnerade skalärerna summeras i olika ordning. Det som prövas är att
    avvikelsen stannar på den nivån och inte växer till något strukturellt.
    """
    ref = build_population(a, seed)
    ref.PP.flora_growth_backend = "numpy"
    ref._flora_growth_mode = "numpy"
    want = str(getattr(a, "flora_growth", None) or "numba")
    if want == "numpy":
        want = "numba"
    new = build_population(a, seed)
    new.PP.flora_growth_backend = want
    new._flora_growth_mode = None

    print(f"verifierar florans tillväxtpass: numpy mot {want}, "
          f"{ticks} tick, seed {seed}", flush=True)
    worst: dict[str, tuple[float, float]] = {
        name: (0.0, 0.0)
        for name in _VERIFY_STORE_FIELDS + _VERIFY_WORLD_FIELDS
        if hasattr(ref.store, name) or hasattr(ref.world, name)
    }
    for t in range(1, int(ticks) + 1):
        ref.step()
        new.step()

        n_ref = len(ref._flora_slots(rebuild=True))
        n_new = len(new._flora_slots(rebuild=True))
        if n_ref != n_new:
            print(f"  tick {t}: STRUKTURELL AVVIKELSE, flora {n_ref} mot {n_new}")
            return 1

        for name in _VERIFY_STORE_FIELDS:
            x = getattr(ref.store, name, None)
            y = getattr(new.store, name, None)
            if x is None or y is None:
                continue
            d = _max_rel(x, y)
            if d > worst.get(name, (0.0, 0.0)):
                worst[name] = d
        for name in _VERIFY_WORLD_FIELDS:
            d = _max_rel(getattr(ref.world, name), getattr(new.world, name))
            if d > worst.get(name, (0.0, 0.0)):
                worst[name] = d

    print(f"  {ticks} tick utan strukturell avvikelse, flora {n_new}")
    print(f"  {'fält':<24} {'max abs':>12} {'mot skalan':>12}")
    bad = 0
    for name in sorted(worst):
        ab, rel = worst[name]
        flag = "" if rel <= 1e-12 else "   <-- över 1e-12"
        if rel > 1e-12:
            bad += 1
        print(f"  {name:<24} {ab:12.3e} {rel:12.3e}{flag}")
    return 1 if bad else 0


def main() -> int:
    a = parse_args()
    apply_scenario(a)

    if int(getattr(a, "bench_flora_growth", 0) or 0) > 0:
        sd = int(str(a.seeds).split(",")[0]) if a.seeds else 1
        return bench_flora_growth(a, seed=sd, warmup=int(a.bench_flora_growth),
                                  rounds=int(a.bench_rounds),
                                  per=int(a.bench_ticks))

    if int(getattr(a, "verify_flora_growth", 0) or 0) > 0:
        sd = int(str(a.seeds).split(",")[0]) if a.seeds else 1
        return verify_flora_growth(a, seed=sd, ticks=int(a.verify_flora_growth))

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
