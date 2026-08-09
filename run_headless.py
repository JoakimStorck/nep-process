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

    # Lindade `_apply_reflex_drives` fram till steg 3. Den metoden finns inte
    # längre; `_samla_anspravk` anropas på exakt samma ställe i kedjan, en gång
    # per levande agent och tick, och har samma egenskap: den ser
    # `_cached_agent_hit` efter sensingen.
    orig = Agent._samla_anspravk
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

    Agent._samla_anspravk = wrapped


# ---------------------------------------------------------------------------
# Arbitreringen.
#
# Sedan steg 3 skriver inget beteende i `turn`. Varje sensoriskt intryck lämnar
# ett anspråk — (namn, nivå, styrka, bäring, thrust_min, explore_mult) — och
# `_valj_anspravk` väljer ett. Måtten från steg 1 är därmed meningslösa:
# kanselleringen kan inte inträffa när ingenting summeras, och
# riktningsavvikelsen är noll per konstruktion eftersom vinnarens bäring *är*
# kursen. De togs bort i stället för att lämnas kvar och rapportera nollor.
#
# Det som mäts nu är valet: vem vinner, hur ofta byts vinnare, hur många
# anspråk konkurrerar, och hur styrkorna fördelar sig. Allt läses ur listan som
# produktionskoden redan bygger — ingen grenlogik replikeras här, och till
# skillnad från 0136–0148 behöver instrumenteringen inte räkna ut någonting
# själv.
_S: dict = {"tick": 0, "byten": 0, "n_anskrav": 0,
            "vinnare": {}, "styrka": {}, "forra": {}}


# ---------------------------------------------------------------------------
# Riktningsfördelningen.
#
# `_valj_anspravk` utvärderar `styrka · cos(Δ) − vikt · kostnad(b)` i varje
# kandidatbäring. Det är en fördelning över riktningar — attraktion minus
# kostnad — och den kollapsades till sitt `argmax`. Frågan den här mätningen
# ska avgöra är **om fördelningen är toppig eller platt**, eftersom det avgör
# om en representation som bär hela fördelningen ger något utöver toppen.
#
# `_W["tvekan"]` mäter något närliggande men inte samma sak: den läser
# födoperceptet `accB` före kostnaden och före vinnarens bäring. Här mäts det
# arbitreringen faktiskt jämför.
#
# Tre tal, alla i **styrkeenheter** så att de är jämförbara mellan djur och
# över tid. Att normera per djur mot dess egen min och max vore att radera
# just den skillnad som ska mätas: ett platt djur skulle se likadant ut som ett
# toppigt.
#
#   **spridning**  (max − min) över de sex sektorerna, delat med styrkan. Noll
#     betyder att riktningen inte spelar någon roll för djuret just då; två är
#     det mesta `cos` kan ge, och mer när kostnaden skiljer sektorerna åt.
#
#   **marginal**  (bästa − näst bästa) över samtliga kandidater, delat med
#     styrkan. Det är exakt den storhet `baring_marginal` jämförs mot, så
#     medianen säger direkt hur ofta hysteresen binder i stället för att välja.
#
#   **likvärdiga**  antal sektorer inom `baring_marginal` från toppen. Det är
#     den skarpaste avläsningen av platt: ligger fem av sex inom marginalen
#     finns det ingen riktning att tala om, bara en plats.
#
# Alla tre läses ur `Agent._dir_prof`, som produktionskoden redan bygger.
# Ingen grenlogik replikeras här.
#   **toppandel**  bästa sektorns andel av **perceptets** summa. Det är den
#     enda av de fyra som mäter profilen *före* arbitreringens kollaps, och
#     därmed den enda som kan vara flat på ett informativt sätt. Jämförelsen är
#     `1/S`: lika stor betyder att riktningen inte bär någon information alls.
#     Talet delas på **kust** och **inland**. Kustlinjen är den enda skarpa
#     kanten i en värld där marken är sluten överallt, och den är därför den
#     enda kontrollen som kan skilja "profilen är platt" från "profilen läser
#     inte världen". Ett djur med tre sektorer mot land och tre mot hav kan inte
#     ha en jämn profil; får det ändå det är felet i perceptet och inte i
#     världen.
# Driftens census. Kroppsdjupet är modellens enda djupskala sedan 7023, och de
# tre talen nedan är vad den gjorde med driften: hur ofta den fyrar, hur djupt
# vattnet var när den gjorde det, och hur långt steget blev.
_D: dict = {"agenttick": 0, "vata": 0, "flytande": 0,
            "steg": [0] * 20, "steg_max": 0.0, "v_max": 0.0}
_D_HI = 24.0

_P: dict = {"tick": 0, "n_sektor": 0, "likvardiga": 0, "n_percept": 0,
            "n_kust": 0,
            "spridning": [0] * 20, "marginal": [0] * 20, "toppandel": [0] * 20,
            "toppandel_kust": [0] * 20}

# Histogrammens övre gräns, i styrkeenheter. Värden över den hamnar i sista
# facket; det syns som en klump vid kanten och är avsiktligt inte tyst.
_P_HI = 2.0


# Fönstret är **skilt** från totalerna och nollställs vid avläsning, som
# `gestation_window()`. Totalerna nollställs per körning, som `_C`.
_S_W = [0, 0]


def styr_fonster():
    """Andel tick med byte av vinnande anspråk sedan förra avläsningen."""
    if _S_W[0] <= 0:
        return None
    ut = 100.0 * _S_W[1] / _S_W[0]
    _S_W[0] = 0
    _S_W[1] = 0
    return ut


def _s_hist(d: dict, namn: str, v: float) -> None:
    e = d.get(namn)
    if e is None:
        e = [0, 0.0, [0] * 20]
        d[namn] = e
    e[0] += 1
    e[1] += v
    e[2][min(19, max(0, int(v * 20.0)))] += 1


# ---------------------------------------------------------------------------
# Sektorprofilen mot vattnet.
#
# Sensingen ger redan riktningsupplöst information: `_acc_dir_B` och
# `_acc_dir_C` är födoaggregat per sektor över hela synfältet, sex sektorer om
# sextio grader i kroppsram. Djuret *skannar* alltså sin omgivning.
#
# Men `_samla_anspravk` tar `argmax` och kastar resten. Sju anspråk, sju
# kollapsade riktningar. Frågan den här mätningen ska avgöra är om det spelar
# någon roll — alltså om profilen bär information utöver sin topp, och om en
# kostnadsterm skulle ha ändrat valet.
#
# Tre tal:
#
#   **tvekan**  näst bästa sektorns värde delat med bästas. Nära noll betyder
#     entoppig profil, och då ger en ombyggnad till sektorrum lite: toppen *är*
#     svaret. Nära ett betyder att flera riktningar är likvärdiga och att
#     kostnaden kan avgöra utan att djuret ger upp något.
#
#   **våt topp**  andelen tick där den valda födosektorn leder ut i vatten.
#     Det är den direkta orsaken vi tror dödade p160–p162: djuret ser flora på
#     andra sidan en vik, får en bäring rakt över vattnet och fryser ihjäl vid
#     tjugofem gånger värmeledning.
#
#   **billigare granne**  andelen tick där någon annan sektor är torr och har
#     minst hälften av toppens födovärde. Det är precis de tick där en
#     kostnadsvägd riktning hade valt annorlunda — alltså mätningens svar på
#     om ombyggnaden är värd sin storlek.
#
# Vattnet samplas ett par cellbredder ut längs varje sektor, inte i cellen
# djuret står i: framåtblick är hela poängen. Uppslaget går via samma
# `world.water` och samma kroppsdjup som `_water_factor` läser, så
# mätningen och fysiken kan inte glida isär.
_W: dict = {"tick": 0, "vat_topp": 0, "billigare": 0, "vat_sektor": 0,
            "n_sektor": 0, "tvekan": [0] * 20, "vatandel": [0] * 20}

# Hur långt fram sektorn prövas, i cellbredder.
_W_FRAMAT = 2.0


def _vat_kostnad(agent, world, ang_kropp):
    """Vattnets hinder en bit fram längs en sektor, i [0, 1]."""
    import math as _m
    a = float(agent.heading) + float(ang_kropp)
    x = float(agent.x) + _W_FRAMAT * _m.cos(a)
    y = float(agent.y) + _W_FRAMAT * _m.sin(a)
    try:
        d = float(world.water[world.grid.cell_of(x, y)])
    except Exception:
        return 0.0
    if d <= float(world.WP.submerged_threshold):
        return 0.0
    ref = agent._body_depth()
    return min(1.0, d / ref) * (1.0 - float(getattr(agent.pheno, "buoyancy", 0.0)))


_INSTRUMENTED_VATTEN = False


def instrument_vatten() -> None:
    """Mät sektorprofilen och vattnet längs varje sektor, en gång per tick."""
    global _INSTRUMENTED_VATTEN
    if _INSTRUMENTED_VATTEN:
        return
    _INSTRUMENTED_VATTEN = True

    import numpy as _np
    import styrning as _st
    from agent import Agent

    orig = Agent._samla_anspravk

    def wrapped(self, world, *ar, **kw):
        sens = getattr(self, "sensors", None)
        aB = getattr(sens, "_acc_dir_B", None) if sens is not None else None
        aC = getattr(sens, "_acc_dir_C", None) if sens is not None else None
        ang = getattr(sens, "_acc_dir_ang", None) if sens is not None else None
        if aB is not None and aC is not None and ang is not None and len(aB) > 1:
            diet = float(getattr(self.pheno, "diet", 0.5))
            combo = (_np.asarray(aB, dtype=_np.float64) * ((1.0 - diet) ** 0.7)
                     + _np.asarray(aC, dtype=_np.float64) * (diet ** 0.7))
            kost = _np.array([_vat_kostnad(self, world, a) for a in ang])

            _W["tick"] += 1
            _W["n_sektor"] += kost.size
            _W["vat_sektor"] += int(_np.count_nonzero(kost > 0.0))
            for k in kost:
                _W["vatandel"][min(19, max(0, int(k * 20.0)))] += 1

            i = int(_np.argmax(combo))
            b = float(combo[i])
            if b > 1e-9:
                andra = _np.sort(combo)[-2]
                _W["tvekan"][min(19, max(0, int(andra / b * 20.0)))] += 1
                if kost[i] > 0.0:
                    _W["vat_topp"] += 1
                    # Finns en torr sektor med minst halva toppens värde?
                    torr = (kost <= 0.0) & (combo >= 0.5 * b)
                    if bool(_np.any(torr)):
                        _W["billigare"] += 1
        return orig(self, world, *ar, **kw)

    Agent._samla_anspravk = wrapped


# ---------------------------------------------------------------------------
# Reservuttagets strypning.
#
# `slow_mobil_frac = 0,25` per månad gör `M_slow` till fett med veckors
# tidsskala. Men p160b visar att `fast_frac` fortsätter drifta mot den snabba
# poolen även när isoleringen sänker termoregleringen från 102 till 81 procent
# av basalomsättningen — späcket lönar sig inte trots att det fungerar.
#
# Misstanken är att taket kostar på fel ställe. Det sattes för att motsvara en
# vinter, alltså fyra månaders **uttömning av hela poolen**. Men uttaget sker
# per tick, och vid dt = 0,02 släpper taket bara en halv procent av poolen per
# steg. Det är inte en vinterbudget utan en flaskhals i vardagen — och tillväxt
# och dräktighet drar ur samma pool.
#
# Frågan mätningen ska avgöra: **vem svälter på strypningen?**
#
#   underhållet   då är taket för hårt och bör lätta
#   tillväxten    då ska taket gälla underhåll men inte anabolism, vilket
#     dräktigheten  också är biologiskt rimligare — en dräktig hona omsätter
#                 sina reserver snabbare än en vilande
#
# Anroparna skiljs på anropsplats via stackens radnummer, inte på en flagga i
# produktionskoden. Det håller mätningen utanför `agent.py` helt.
_M: dict = {}

# Radnumren flyttar när `agent.py` ändras. Kartan slås därför upp mot
# funktionsnamn plus offset vid installationen i stället för mot fasta tal —
# se `instrument_reserv`.
_M_RADER: dict = {}

_INSTRUMENTED_RESERV = False


def instrument_reserv() -> None:
    """Bokför hur ofta reservuttaget ger mindre än begärt, och till vem."""
    global _INSTRUMENTED_RESERV
    if _INSTRUMENTED_RESERV:
        return
    _INSTRUMENTED_RESERV = True

    import sys as _sys
    from agent import Body

    orig = Body._take_reserve_mass

    # Bind radnumren vid installationen genom att läsa källan. Fasta tal
    # åldras med första ändring i `agent.py`, och en mätning som tyst byter
    # etikett är värre än ingen.
    import inspect as _insp
    from agent import Body as _B
    _src, _lin0 = _insp.getsourcelines(_insp.getmodule(_B))
    for _i, _rad in enumerate(_src):
        if "_take_reserve_mass(" not in _rad or "def " in _rad:
            continue
        n = _lin0 + _i
        if "strypt=False" in _rad:
            _namn = "dräktighet" if "gest" in _rad else "tillväxt"
        elif "Ecap" in _rad or "trim" in _rad:
            _namn = "trimning"
        else:
            _namn = "underhåll"
        # `f_lineno` pekar på raden där uttrycket *avslutas*, vilket kan vara
        # nästa rad när anropet är radbrutet. Registrera båda.
        _M_RADER[n] = _namn
        _M_RADER[n + 1] = _namn

    def wrapped(self, kg, dt=1.0, **kw):
        ut = orig(self, kg, dt, **kw)
        want = float(kg)
        if want > 1e-15:
            rad = _sys._getframe(1).f_lineno
            namn = _M_RADER.get(rad, f"rad {rad}")
            e = _M.get(namn)
            if e is None:
                e = [0, 0, 0.0, 0.0]      # anrop, strypta, begärt, givet
                _M[namn] = e
            e[0] += 1
            e[2] += want
            e[3] += float(ut)
            # Strypt om uttaget understiger det begärda trots att reserven
            # hade räckt — alltså att taket och inte tomheten band.
            if ut < want * 0.999 and float(self.M_reserve()) > want:
                e[1] += 1
            # Anabolismen går sedan 0162 förbi taket. Raden står kvar i
            # mätningen för att kunna visa att strypningen faktiskt upphörde.
        return ut

    Body._take_reserve_mass = wrapped


_INSTRUMENTED_STEERING = False


def _hist_median(h, lo: float, hi: float) -> float:
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


def _hist_kvantil(h, lo: float, hi: float, q: float) -> float:
    """
    Godtycklig kvantil ur samma fack som `_hist_median`.

    Skild funktion i stället för en generalisering av `_hist_median`: den
    senare har fyra anropare vars tal står i commitmeddelanden, och en
    omskrivning skulle behöva visa att de är oförändrade för att vara värd
    något. Medianen ur den här funktionen och ur den andra är samma tal.

    En fördelning kan bara dömas på sina percentiler. Medelvärdet låg åtta
    gånger över medianen i floran, och det felet är dokumenterat mer än en
    gång i det här repot.
    """
    n = sum(h)
    if n <= 0:
        return float("nan")
    w = (hi - lo) / len(h)
    mal = q * n
    acc = 0
    for i, c in enumerate(h):
        if c > 0 and acc + c >= mal:
            return lo + w * (i + (mal - acc) / c)
        acc += c
    return hi


def _mat_riktning(agent) -> None:
    """
    Riktningsfördelningen ur `Agent._dir_prof`, i styrkeenheter.

    Sektorerna ligger på plats 1 till och med `n_sektor`; plats 0 är
    anspråkets egen bäring och en eventuell bunden bäring ligger sist.
    Spridningen mäts **bara över sektorerna**, eftersom det är de som utgör
    fördelningen över riktningar — de två övriga är enskilda kandidater och
    skulle bredda spannet utan att säga något om formen. Marginalen mäts över
    samtliga kandidater, eftersom det är dem valet står mellan.
    """
    p = getattr(agent, "_dir_prof", None)
    if p is None:
        return
    _kand, varde, styrka, n_sekt = p
    if styrka <= 0.0 or n_sekt <= 1 or len(varde) < 1 + n_sekt:
        return

    sekt = varde[1:1 + n_sekt]
    _P["tick"] += 1
    _P["n_sektor"] += n_sekt

    spr = (max(sekt) - min(sekt)) / styrka
    _P["spridning"][min(19, max(0, int(spr / _P_HI * 20.0)))] += 1

    ord_ = sorted(varde, reverse=True)
    marg = (ord_[0] - ord_[1]) / styrka if len(ord_) > 1 else _P_HI
    _P["marginal"][min(19, max(0, int(marg / _P_HI * 20.0)))] += 1

    # Likvärdiga sektorer: de som ligger inom `baring_marginal` från den bästa
    # sektorn. Tröskeln tas ur `AgentParams` och inte som ett eget tal här —
    # ändras den i modellen ska mätningen följa med.
    tak = float(agent.AP.baring_marginal) * styrka
    b = max(sekt)
    _P["likvardiga"] += sum(1 for v in sekt if b - v <= tak)

    # Perceptet, före kollapsen. Skild räknare eftersom ett djur kan ha en
    # arbitrering utan att ha sensat.
    f = getattr(agent, "_dir_food", None)
    if f is not None and len(f) > 1:
        tot = float(sum(f))
        if tot > 0.0:
            b = min(19, max(0, int(float(max(f)) / tot * 20.0)))
            _P["n_percept"] += 1
            _P["toppandel"][b] += 1
            if float(getattr(agent, "_dir_vatten", 0.0)) > 0.0:
                _P["n_kust"] += 1
                _P["toppandel_kust"][b] += 1


_INSTRUMENTED_DRIFT = False


def instrument_drift() -> None:
    """
    Bokför driften. Lindar `_drift_system`, som inte ändras.

    Talen läses ur bidragen passet lämnar — `_drift_dx`, `_drift_dy` — och inte
    ur en egen beräkning, så mätningen kan inte glida isär från mekanismen.
    """
    global _INSTRUMENTED_DRIFT
    if _INSTRUMENTED_DRIFT:
        return
    _INSTRUMENTED_DRIFT = True

    from population import Population
    import numpy as _np

    orig = Population._drift_system

    def wrapped(self, *ar, **kw):
        ut = orig(self, *ar, **kw)
        alive = [a for a in self.agents if a.body.alive]
        w = getattr(self, "world", None)
        dr = getattr(w, "drainage", None)
        if dr is None or not alive:
            return ut
        _D["agenttick"] += len(alive)
        thr = float(w.WP.submerged_threshold)
        for a in alive:
            try:
                c = self.grid.cell_of(float(a.x), float(a.y))
                d = float(w.water[c])
            except Exception:
                continue
            if d > thr:
                _D["vata"] += 1
            dx = float(getattr(a, "_drift_dx", 0.0))
            dy = float(getattr(a, "_drift_dy", 0.0))
            if dx or dy:
                st = float(_np.hypot(dx, dy))
                _D["flytande"] += 1
                _D["steg"][min(19, max(0, int(st / _D_HI * 20.0)))] += 1
                if st > _D["steg_max"]:
                    _D["steg_max"] = st
        return ut

    Population._drift_system = wrapped


def instrument_steering() -> None:
    """
    Bokför arbitreringen. Lindar de två metoder som bygger och väljer
    anspråk; ingen av dem ändras.
    """
    global _INSTRUMENTED_STEERING
    if _INSTRUMENTED_STEERING:
        return
    _INSTRUMENTED_STEERING = True

    from agent import Agent

    orig_samla = Agent._samla_anspravk
    orig_valj = Agent._valj_anspravk

    def wrapped_samla(self, *ar, **kw):
        A = orig_samla(self, *ar, **kw)
        _S["tick"] += 1
        _S_W[0] += 1
        _S["n_anskrav"] += len(A)
        for a in A:
            _s_hist(_S["styrka"], a[0], float(a[2]))
        return A

    def wrapped_valj(self, anskrav, *ar, **kw):
        out = orig_valj(self, anskrav, *ar, **kw)
        _mat_riktning(self)
        namn = out[3]
        if namn:
            e = _S["vinnare"].get(namn, 0)
            _S["vinnare"][namn] = e + 1
        aid = int(getattr(self, "id", 0))
        if _S["forra"].get(aid, "") != namn:
            _S["byten"] += 1
            _S_W[1] += 1
        _S["forra"][aid] = namn
        return out

    Agent._samla_anspravk = wrapped_samla
    Agent._valj_anspravk = wrapped_valj


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
    wp_over: dict = {}
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

    # Terrängen kommer bara från scenariot. Den har elva tal och hör inte hemma
    # på kommandoraden — det var just den sortens spridning `scenario.py` finns
    # till för att stoppa.
    _sc_terr = getattr(a, "_scenario", None)
    if _sc_terr is not None and _sc_terr.terrain is not None:
        wp_over["terrain"] = _sc_terr.terrain

    # Världens position. Den hör till scenariot av samma skäl som terrängen:
    # den beskriver vilken värld som körs, inte hur den körs. Två tal, och de
    # ersätter fyra klimatparametrar som tidigare inte gick att sätta alls.
    if _sc_terr is not None:
        wp_over["latitud"] = float(_sc_terr.varld.latitud)
        wp_over["kontinentalitet"] = float(_sc_terr.varld.kontinentalitet)

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
        # `flora.sadd` som tal hade ingen läsare: `Scenario.flora_seed_kg`
        # räknades ut och kastades, och `PopParams.flora_init_target_kg` sattes
        # aldrig. Varje scenario fick alltså tyst bördighetsregeln oavsett vad
        # filen sa. Det är samma sorts fält-utan-läsare som kapacitetsmodellen,
        # och det upptäcktes när geo-scenariot bad om en hundradels sådd och
        # fick full.
        PP.flora_init_target_kg = _sc.flora_seed_kg
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
    # Styrfältet per fönster mätte kansellering och riktningsavvikelse. Båda
    # är meningslösa under arbitrering — ingenting summeras, och vinnarens
    # bäring *är* kursen — så fältet mäter nu bytesfrekvensen i stället: hur
    # ofta djuren byter vinnande anspråk. Det är det tal hysteresen påverkar.
    sbyten = styr_fonster()
    styr = f"  byten={sbyten:.1f}%" if sbyten is not None else ""
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

    if _S["tick"]:
        n = _S["tick"]
        print(f"\n  arbitrering  {n} agenttick, {_S['n_anskrav'] / n:.2f} anspråk per tick")
        print(f"    byte av vinnare  {100 * _S['byten'] / n:5.2f} % av tickarna")
        rader = sorted(_S["vinnare"].items(), key=lambda kv: -kv[1])
        for k, v in rader:
            e = _S["styrka"].get(k)
            andel = f"{100 * v / n:5.1f} %"
            if e and e[0]:
                print(f"      {k:<15} vann {andel}  av {100 * e[0] / n:5.1f} % närvaro"
                      f"   styrka median {_hist_median(e[2], 0.0, 1.0):.3f}"
                      f"  medel {e[1] / e[0]:.3f}")
            else:
                print(f"      {k:<15} vann {andel}")
        for k, e in sorted(_S["styrka"].items(), key=lambda kv: -kv[1][0]):
            if k not in _S["vinnare"] and e[0]:
                print(f"      {k:<15} vann   0.0 %  av {100 * e[0] / n:5.1f} % närvaro"
                      f"   styrka median {_hist_median(e[2], 0.0, 1.0):.3f}"
                      f"  medel {e[1] / e[0]:.3f}")

    if _D["agenttick"]:
        n = _D["agenttick"]
        print(f"\n  drift        {n} agenttick, {100 * _D['vata'] / n:.2f} % i vatten,"
              f" {100 * _D['flytande'] / n:.3f} % flytande")
        if _D["flytande"]:
            print(f"    steg         median {_hist_median(_D['steg'], 0.0, _D_HI):.2f}"
                  f"   p90 {_hist_kvantil(_D['steg'], 0.0, _D_HI, 0.90):.2f}"
                  f"   max {_D['steg_max']:.2f} cellbredder")
        else:
            print("    steg         ingen drift alls — kroppsdjupet grindade bort den")

    if _P["tick"]:
        n = _P["tick"]
        sn = max(1, _P["n_sektor"])
        print(f"\n  riktning     {n} agenttick, {_P['n_sektor'] / n:.1f} sektorer per tick"
              f"   (i styrkeenheter)")
        print(f"    spridning    p10 {_hist_kvantil(_P['spridning'], 0.0, _P_HI, 0.10):.3f}"
              f"   median {_hist_median(_P['spridning'], 0.0, _P_HI):.3f}"
              f"   p90 {_hist_kvantil(_P['spridning'], 0.0, _P_HI, 0.90):.3f}")
        print(f"    marginal     p10 {_hist_kvantil(_P['marginal'], 0.0, _P_HI, 0.10):.3f}"
              f"   median {_hist_median(_P['marginal'], 0.0, _P_HI):.3f}"
              f"   p90 {_hist_kvantil(_P['marginal'], 0.0, _P_HI, 0.90):.3f}")
        print(f"    likvärdiga   {_P['likvardiga'] / n:.2f} sektorer av "
              f"{_P['n_sektor'] / n:.1f} inom marginalen från toppen"
              f"   ({100 * _P['likvardiga'] / sn:.1f} %)")
        if _P["n_percept"]:
            jamn = n / max(1e-9, _P["n_sektor"] / n) / n
            print(f"    toppandel    p10 {_hist_kvantil(_P['toppandel'], 0.0, 1.0, 0.10):.3f}"
                  f"   median {_hist_median(_P['toppandel'], 0.0, 1.0):.3f}"
                  f"   p90 {_hist_kvantil(_P['toppandel'], 0.0, 1.0, 0.90):.3f}"
                  f"   (jämn fördelning = {jamn:.3f})")
            if _P["n_kust"]:
                print(f"      vid kust   median {_hist_median(_P['toppandel_kust'], 0.0, 1.0):.3f}"
                      f"   av {100 * _P['n_kust'] / _P['n_percept']:.1f} % av tickarna"
                      f"   — tre sektorer mot hav kan inte ge en jämn profil")
            else:
                print("      vid kust   inga tick med vatten inom två cellbredder")

    if _W["tick"]:
        n = _W["tick"]
        print(f"\n  sektorer     {n} tick, {_W['n_sektor'] / n:.1f} sektorer per tick")
        print(f"    tvekan       näst bästa sektorn är {100 * _hist_median(_W['tvekan'], 0.0, 1.0):.0f} % "
              f"av bästa i median")
        print(f"    vatten       {100 * _W['vat_sektor'] / max(1, _W['n_sektor']):5.1f} % av sektorerna "
              f"leder i vatten två cellbredder fram")
        print(f"    våt topp     {100 * _W['vat_topp'] / n:5.1f} % av tickarna pekar födotoppen i vatten")
        if _W["vat_topp"]:
            print(f"      av dem har {100 * _W['billigare'] / _W['vat_topp']:5.1f} % en torr sektor "
                  f"med minst halva toppens värde")

    if _M:
        print("\n  reservuttag   anrop      strypta    begärt kg    givet kg   andel given")
        for k, e in sorted(_M.items(), key=lambda kv: -kv[1][0]):
            print(f"    {k:<12}{e[0]:9d}{100 * e[1] / max(1, e[0]):9.1f} %"
                  f"{e[2]:12.4g}{e[3]:12.4g}{100 * e[3] / max(1e-12, e[2]):11.1f} %")

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
        instrument_reserv()
        instrument_vatten()
        instrument_drift()
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
    _M.clear()

    # Per körning, inte vid avläsning. Nollställningen låg först i
    # `styr_fonster()`, som tömmer vid läsning — samma fälla som 0140 skrev
    # varningen om, och blocket skrevs därför aldrig ut.
    _W["tick"] = 0
    _W["vat_topp"] = 0
    _W["billigare"] = 0
    _W["vat_sektor"] = 0
    _W["n_sektor"] = 0
    _W["tvekan"] = [0] * 20
    _W["vatandel"] = [0] * 20
    # Per körning, som `_C` — inte vid avläsning, som `gestation_window()`.
    # Hamnar en räknare i det andra mönstret är den alltid noll när blocket
    # ska skrivas ut, och blocket hoppas tyst över.
    # Per körning, som `_C` — inte vid avläsning, som `gestation_window()`.
    _S["tick"] = 0
    _S["byten"] = 0
    _S["n_anskrav"] = 0
    _S["vinnare"] = {}
    _S["styrka"] = {}
    _D["agenttick"] = 0
    _D["vata"] = 0
    _D["flytande"] = 0
    _D["steg"] = [0] * 20
    _D["steg_max"] = 0.0
    _P["tick"] = 0
    _P["n_sektor"] = 0
    _P["likvardiga"] = 0
    _P["n_percept"] = 0
    _P["n_kust"] = 0
    _P["spridning"] = [0] * 20
    _P["marginal"] = [0] * 20
    _P["toppandel"] = [0] * 20
    _P["toppandel_kust"] = [0] * 20
    # In-place: wrappern läser dicten per anrop, men `forra` bär tillstånd
    # mellan världar vid `--seeds` om den ombinds.
    _S["forra"].clear()
    _S_W[0] = 0
    _S_W[1] = 0

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
        # Positionen gäller varje värld, platt som kuperad, och skrivs därför
        # utanför terrängblocket. Två tal som ersätter fyra, och raden gör
        # synligt vad de blev.
        from klimat import beskriv as _beskriv_klimat

        print(
            "[klimat] "
            + _beskriv_klimat(pop.world.WP.latitud, pop.world.WP.kontinentalitet),
            flush=True,
        )
        _dr = getattr(pop.world, "drainage", None)
        if _dr is not None:
            from drainage import describe as _describe_drainage

            d = _describe_drainage(_dr)
            # Vattnets andel av världen är det tal som avgör om världen behöver
            # bli större, och den är inte densamma som andelen förlorad
            # produktion: havet ligger i polarbältet där tillväxtgrinden ändå
            # är nära noll. Därför båda talen.
            g = np.asarray(pop.world.growth_gate_field(), dtype=np.float64)
            wet = _dr.sea | (_dr.lake_id >= 0)
            g_tot = float(g.sum())
            g_lost = float(g[wet].sum())
            print(
                f"[terräng] hav {100*d['sea_frac']:.1f} %  sjö {100*d['lake_frac']:.1f} % "
                f"({d['n_lakes']} st, störst {d['largest_lake']} celler)  "
                f"land {100*d['land_frac']:.1f} %  "
                f"största flod {d['max_upslope']:.0f} celler  "
                f"grindviktad areaförlust {100*g_lost/max(1e-12, g_tot):.1f} %",
                flush=True,
            )
            _off = getattr(pop.world, "_T_offset", 0.0)
            if isinstance(_off, np.ndarray) and float(_off.min()) < 0.0:
                _gl = g[(~_dr.sea) & (_dr.lake_id < 0)]
                print(
                    f"[höjdklimat] höjdgradient {float(_off.min()):.1f} °C på högsta punkten  "
                    f"tillväxtgrind p10/median/p90 "
                    f"{np.quantile(_gl, 0.1):.2f}/{np.quantile(_gl, 0.5):.2f}/"
                    f"{np.quantile(_gl, 0.9):.2f}",
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
        _dr = getattr(pop.world, "drainage", None)
        if _dr is not None and not a.quiet:
            # Markfukten vid slutet, inte vid start: vid tick noll står den på
            # fältkapacitet överallt och säger ingenting.
            w = pop.world
            _land = (~_dr.sea) & (_dr.lake_id < 0)
            _m = np.asarray(w.soil_water)[_land] / max(1e-12, float(w.WP.soil_capacity))
            _stock = float(w.water_stock())
            _resid = ((_stock - float(w._water_stock_init))
                      - (float(w._water_added_total) - float(w._water_lost_total)))
            print(
                f"[hydro] markfukt p10/median/p90 "
                f"{np.quantile(_m, 0.1):.2f}/{np.quantile(_m, 0.5):.2f}/{np.quantile(_m, 0.9):.2f}  "
                f"fåror {100.0 * float(np.mean((np.asarray(w.water) > 0)[_land])):.2f} %  "
                f"sjömagasin {float(w.lake_storage.sum()):.2f} av {float(_dr.lake_cap.sum()):.2f}  "
                f"vattenbalans {abs(_resid) / max(1e-12, float(w._water_added_total)):.1e} rel",
                flush=True,
            )
            _land2 = (~_dr.sea) & (_dr.lake_id < 0)
            if _land2.any():
                _n = np.asarray(w.nutrient)[_land2]
                _up = np.asarray(_dr.upslope)[_land2]
                _q = np.quantile(_up, [0.25, 0.75])
                _lo = _n[_up <= _q[0]].mean()
                _hi = _n[_up >= _q[1]].mean()
                print(
                    f"[näring] fri per landcell: rygg {_lo:.4f}  dal {_hi:.4f}  "
                    f"kvot {_hi / max(1e-12, _lo):.2f}x   "
                    f"vittrat {float(w._nutrient_added_total):.1f} kg  "
                    f"urlakat {float(w._nutrient_lost_total):.1f} kg",
                    flush=True,
                )
            _wet = _dr.sea | (np.asarray(w.water) > float(w.WP.submerged_threshold))
            _d = np.asarray(w.detritus)
            print(
                f"[förna] per cell: land {_d[~_wet].mean():.2f} kg  "
                f"vatten {_d[_wet].mean():.2f} kg  "
                f"andel i vatten {100.0 * _d[_wet].sum() / max(1e-12, _d.sum()):.1f} %",
                flush=True,
            )
            print(
                f"[flora] begränsande resurs: vatten "
                f"{float(getattr(pop, '_last_flora_water_limited', 0.0)):.3f}  ljus "
                f"{float(getattr(pop, '_last_flora_light_limited', 0.0)):.3f}  "
                f"(resten näring)",
                flush=True,
            )
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
