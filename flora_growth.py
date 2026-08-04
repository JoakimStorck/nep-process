"""
Florans tillväxtpass som kompilerad kärna.

Passet är över hälften av takten vid flora i storleksordningen 300 000
individer, och profileringen i 0113 visade att den tiden ligger i passets egen
aritmetik snarare än i minnestrafiken: ett float32-experiment som halverade
varje temporär gav bara sex procent. Kvar stod alltså själva räknandet, och det
räknas elementvis över tolv numpy-uttryck som var för sig läser och skriver
hela floravektorn.

Kärnan här gör samma sak i ett svep per beroendesteg. Den tar bara arrayer och
skalärer — ingen store, ingen värld, inget grid — vilket är villkoret för att
den ska kunna kompileras. Anropande skal och
efterspel ligger kvar i `Population._growth_system_flora`, som också behåller
numpy-vägen bakom en flagga så att de två går att jämföra elementvis på samma
tillstånd.

**Vad som inte flyttat in.** Två världsanrop måste ligga utanför:
`world.temperature_of_cells()` läser temperaturen via bandindex och hissas till
skalet, och `world.excrete_cells()` deponerar förna och kadaver i detritus och
sker i efterspelet. Det senare är säkert eftersom passet aldrig läser
`detritus` — bara `nutrient`, som kärnan i stället muterar på plats i den
ordning numpy-vägen gör det.

**Ordningen är bevarad, inte omtolkad.** Näringen som frigörs när en planta dör
återförs till cellen *inuti* kärnan, före anspråksberäkningen, eftersom
numpy-vägen gör det i den ordningen och grannarna alltså kan ta upp den samma
tick. Reduktionerna ackumuleras i radordning, precis som `np.bincount` gör, så
avrundningen följer med.

Skillnaderna mot numpy-vägen är därmed begränsade till bibliotekens sista bit i
`exp` och `pow`, plus summeringsordningen i de tre returnerade skalärerna.
Uppmätt relativ avvikelse ligger kring 1e-16 per element.
"""

from __future__ import annotations

import numpy as np

# Numba är valfri, som i organism_store.py. Saknas den faller passet tillbaka
# på numpy-vägen; kärnan här körs aldrig otolkad, eftersom en Python-loop över
# 300 000 plantor vore hundra gånger långsammare än den numpy-väg den ersätter.
try:
    from numba import njit as _njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - beror på miljön
    HAVE_NUMBA = False

    def _njit(*a, **k):  # type: ignore[misc]
        def deco(f):
            return f

        return deco

def available_backends() -> tuple[str, ...]:
    """Vägar som går att välja i den här miljön, i stigande förväntad fart."""
    out = ["numpy"]
    if HAVE_NUMBA:
        out.append("numba")
    return tuple(out)


from phenotype import (
    FLORA_LIFESPAN_MAX,
    FLORA_LIFESPAN_MIN,
    FLORA_TURNOVER_LABILE,
    FLORA_TURNOVER_STRUCT,
    NUTRIENT_PER_KG_LABILE,
    NUTRIENT_PER_KG_STRUCT,
)

# Hjälpfunktionerna är enkla uttryck över strukturandelen och inlinas som
# njit-funktioner. Konstanterna hämtas från phenotype.py, som äger
# traitsemantiken — de dupliceras inte här.
_LIFESPAN_RATIO = float(FLORA_LIFESPAN_MAX) / float(FLORA_LIFESPAN_MIN)


@_njit(cache=True, nogil=True, inline="always")
def _nutrient_content(s: float) -> float:
    """`phenotype.nutrient_content`, skalärt."""
    return NUTRIENT_PER_KG_LABILE * (1.0 - s) + NUTRIENT_PER_KG_STRUCT * s


@_njit(cache=True, nogil=True, inline="always")
def _turnover_rate(s: float) -> float:
    """`phenotype.flora_turnover_rate`, skalärt."""
    return FLORA_TURNOVER_LABILE * (1.0 - s) + FLORA_TURNOVER_STRUCT * s


@_njit(cache=True, nogil=True, inline="always")
def _lifespan(s: float) -> float:
    """`phenotype.flora_lifespan`, skalärt."""
    return FLORA_LIFESPAN_MIN * _LIFESPAN_RATIO ** s


@_njit(cache=True, nogil=True, inline="always")
def _store_down(x: float) -> float:
    """
    Runda till float32 nedåt och ge tillbaka det lagrade värdet som float64.

    Massan lagras i float32. Rundas den uppåt binder plantan mer näring än som
    betalats, och cellen är ofta redan tömd av just det upptaget — nedåt är
    felet i stället alltid ett överskott som kan återföras. Samma resonemang
    som i numpy-vägen, se `docs/vaxternas-livscykel.md`.
    """
    v = np.float32(x)
    if np.float64(v) > x:
        v = np.nextafter(v, np.float32(0.0))
    return np.float64(v)


def _growth_kernel_impl(
    # --- per planta, gathrade ur store:n ---------------------------------
    m32, struct32, adult32, root32, seed32, energy32,
    topt32, twid32, uptake32, ralloc32, rcap32, rootalloc32,
    reserve, pool, carbon,
    cells, temp, draws,
    # --- värld -----------------------------------------------------------
    nutrient, neighbor_idx,
    # --- skalärer --------------------------------------------------------
    dt, BK, u_area, sra, sla, k_ext, h_ref, L_cell,
    m_floor_abs, seedling_floor, seed_cap_mult, e_labile,
    # --- utdata per planta -----------------------------------------------
    mass_out, root_out, energy_out, reserve_out, pool_out, carbon_out,
    shed_out, dying_out, dm_out, grow_out,
    # --- skrivbuffertar med längd n_cells --------------------------------
    claimed, lam, hsum, cellacc,
):
    """
    Florans livscykel per planta: förnafall, död, anspråk, upptag, allokering,
    ljus, tillväxt.

    Semantiken är oförändrad mot `Population._growth_system_flora_numpy`; se
    dess docstring för biologin.

    **Formen är delad efter vad som ackumulerar och vad som inte gör det.** De
    stora loopar som räknar per planta skriver bara egna index; summor, räknare
    och de utströdda additionerna till cellfälten ligger i korta svep emellan.
    Uppdelningen kom till för en parallellisering som visade sig inte löna sig
    (se `TODO.md`, 0119), men den är behållen på egna meriter: reduktionerna
    har en fast ordning, så `np.bincount`-avrundningen följer med och den
    elementvisa jämförelsen mot numpy-vägen står kvar.

    Returnerar `(shed_total, n_age, n_starve, produced, taken, died,
    light_limited, row_plant, row_cell, row_share)`.
    """
    n = m32.shape[0]
    n_cells = nutrient.shape[0]
    k_nb = neighbor_idx.shape[1]

    m = np.empty(n, np.float64)
    cost = np.empty(n, np.float64)
    a_root = np.empty(n, np.float64)
    gate = np.empty(n, np.float64)
    leaf = np.empty(n, np.float64)
    height = np.empty(n, np.float64)
    access = np.zeros(n, np.float64)
    take = np.empty(n, np.float64)
    eff = np.empty(n, np.float64)
    alloc = np.empty(n, np.float64)
    holds = np.zeros(n, np.bool_)

    # --- 1. förnafall, åldrande och svält, per planta ---------------------
    for i in range(n):
        # Strukturandelen används i två former, precis som i numpy-vägen:
        # klippt i näringsinnehåll, omsättning och livslängd, oklippt i
        # areorna och i energitätheten.
        su = np.float64(struct32[i])
        s = su
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        cost[i] = _nutrient_content(s)

        m0 = np.float64(m32[i])
        shed_want = _turnover_rate(s) * dt * m0
        if shed_want > m0:
            shed_want = m0
        left = _store_down(m0 - shed_want)
        shed_out[i] = m0 - left
        mass_out[i] = np.float32(left)
        m[i] = left

        keep = left / m0 if m0 > 0.0 else 1.0
        rm = np.float64(np.float32(np.float64(root32[i]) * keep))
        root_out[i] = np.float32(rm)
        energy_out[i] = energy32[i]

        floor = seedling_floor * np.float64(seed32[i])
        if floor < m_floor_abs:
            floor = m_floor_abs
        ls = _lifespan(s)
        if ls < 1e-6:
            ls = 1e-6

        gate[i] = 0.0
        a_root[i] = 0.0
        leaf[i] = 0.0
        height[i] = 0.0
        if draws[i] < dt / ls:
            dying_out[i] = 1          # åldrande
        elif left <= floor:
            dying_out[i] = 2          # svält
        else:
            dying_out[i] = 0
            reserve_out[i] = reserve[i]
            pool_out[i] = pool[i]
            carbon_out[i] = carbon[i]
            g = np.float64(temp[i]) - np.float64(topt32[i])
            w = np.float64(twid32[i])
            if w < 1e-6:
                w = 1e-6
            gate[i] = np.exp(-0.5 * (g / w) * (g / w))
            if cells[i] >= 0 and left > 0.0:
                holds[i] = True
                if rm > left:
                    rm = left
                shoot = left - rm
                if shoot < 0.0:
                    shoot = 0.0
                a_root[i] = sra * rm * (1.0 - su) / BK
                leaf[i] = sla * shoot * (1.0 - su) / BK
                height[i] = shoot * su

    # Seriellt: räknare, summa och näringen som återförs när en planta dör.
    # Ordningen är stigande slotindex, alltså densamma som numpy-vägens loop
    # över `fl[dying]` — näringen ska hinna bli tillgänglig för grannarna
    # samma tick, så detta måste ske före anspråken.
    shed_total = 0.0
    n_age = 0
    n_starve = 0
    n_holds = 0
    n_big = 0
    for i in range(n):
        shed_total += shed_out[i]
        d = dying_out[i]
        if d != 0:
            if d == 1:
                n_age += 1
            else:
                n_starve += 1
            held = reserve[i] + pool[i]
            c = cells[i]
            if c >= 0 and c < n_cells and held > 0.0 and np.isfinite(held):
                nutrient[c] += held
            reserve_out[i] = 0.0
            pool_out[i] = 0.0
            carbon_out[i] = 0.0
        elif holds[i]:
            n_holds += 1
            if a_root[i] > 1.0:
                n_big += 1

    if n_holds == 0:
        for i in range(n):
            dm_out[i] = 0.0
            grow_out[i] = 0
        died = 0.0
        for i in range(n):
            if dying_out[i] != 0:
                dv = np.float64(mass_out[i])
                if dv > 0.0:
                    died += dv
        return (shed_total, n_age, n_starve, 0.0, 0.0, died, 0.0,
                np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64))

    # --- 2. anspråk: en rad per planta och berörd cell --------------------
    n_rows = n_holds + n_big * k_nb
    row_plant = np.empty(n_rows, np.int32)
    row_cell = np.empty(n_rows, np.int32)
    row_claim = np.empty(n_rows, np.float64)

    r = 0
    for i in range(n):
        if holds[i]:
            row_plant[r] = i
            row_cell[r] = cells[i]
            a = a_root[i]
            row_claim[r] = a if a < 1.0 else 1.0
            r += 1
    # Överskjutande area fördelas jämnt över grannarna. Raderna läggs efter
    # samtliga egna celler, precis som concatenate gör i numpy-vägen.
    for i in range(n):
        if holds[i] and a_root[i] > 1.0:
            per = (a_root[i] - 1.0) / np.float64(k_nb)
            if per > 1.0:
                per = 1.0
            c = cells[i]
            for j in range(k_nb):
                row_plant[r] = i
                row_cell[r] = neighbor_idx[c, j]
                row_claim[r] = per
                r += 1

    for c in range(n_cells):
        claimed[c] = 0.0
    for t in range(n_rows):
        claimed[row_cell[t]] += row_claim[t]

    row_share = np.empty(n_rows, np.float64)
    row_avail = np.empty(n_rows, np.float64)
    for t in range(n_rows):
        c = row_cell[t]
        d = claimed[c]
        if d < 1.0:
            d = 1.0
        sh = row_claim[t] / d * (1.0 - 1e-12)
        row_share[t] = sh
        av = sh * nutrient[c]
        row_avail[t] = av
        access[row_plant[t]] += av

    # --- 3. inkomst -------------------------------------------------------
    for i in range(n):
        if holds[i] and gate[i] > 1e-6:
            adult = np.float64(adult32[i])
            if adult < 1e-12:
                adult = 1e-12
            head = (adult - m[i]) * cost[i] - reserve_out[i]
            if head < 0.0:
                head = 0.0
            uc = np.float64(uptake32[i])
            if uc < 0.0:
                uc = 0.0
            cap = uc * u_area * a_root[i] * gate[i] * dt
            t_i = cap if cap < access[i] else access[i]
            if head < t_i:
                t_i = head
            take[i] = t_i
        else:
            take[i] = 0.0

    any_take = False
    for i in range(n):
        if take[i] > 0.0:
            any_take = True
            break
    if any_take:
        for c in range(n_cells):
            cellacc[c] = 0.0
        for t in range(n_rows):
            i = row_plant[t]
            a = access[i]
            if a > 0.0:
                fr = take[i] / a
                if fr > 1.0:
                    fr = 1.0
            else:
                fr = 0.0
            cellacc[row_cell[t]] += fr * row_avail[t]
        for c in range(n_cells):
            nutrient[c] -= cellacc[c]

    # --- 4. bladarean per cell -------------------------------------------
    for c in range(n_cells):
        lam[c] = 0.0
        hsum[c] = 0.0
    for i in range(n):
        if holds[i]:
            c = cells[i]
            lam[c] += leaf[i]
            hsum[c] += leaf[i] * height[i]
    for c in range(n_cells):
        if lam[c] > 0.0:
            hsum[c] = hsum[c] / lam[c]
        else:
            hsum[c] = 0.0

    # --- 5. allokering och skugga ----------------------------------------
    for i in range(n):
        al = np.float64(ralloc32[i]) * np.float64(rcap32[i])
        if al < 0.0:
            al = 0.0
        elif al > 1.0:
            al = 1.0
        alloc[i] = al
        tp = take[i] * al
        reserve_out[i] = reserve_out[i] + (take[i] - tp)
        pool_out[i] = pool_out[i] + tp
        if holds[i]:
            c = cells[i]
            hb = hsum[c]
            if hb < 1e-12:
                hb = 1e-12
            hgt = height[i]
            r_rel = hgt / (hgt + h_ref * hb)
            eff[i] = leaf[i] * np.exp(-k_ext * lam[c] * (1.0 - r_rel))
        else:
            eff[i] = 0.0

    # `cellacc` återanvänds som bladarea vägd med skuggan.
    for c in range(n_cells):
        cellacc[c] = 0.0
    for i in range(n):
        if holds[i]:
            cellacc[cells[i]] += eff[i]

    # --- 6. ljus och tillväxt --------------------------------------------
    for i in range(n):
        dm_out[i] = 0.0
        grow_out[i] = 0
        if not holds[i]:
            continue
        c = cells[i]
        d = cellacc[c]
        if d < 1.0:
            d = 1.0
        light = L_cell * eff[i] / d * (1.0 - 1e-12)

        c_cap = seed_cap_mult * np.float64(seed32[i])
        head_c = c_cap - carbon_out[i]
        if head_c < 0.0:
            head_c = 0.0
        to_carbon = light * alloc[i]
        if to_carbon > head_c:
            to_carbon = head_c
        carbon_out[i] = carbon_out[i] + to_carbon
        light_growth = light - to_carbon

        res = reserve_out[i]
        adult = np.float64(adult32[i])
        if adult < 1e-12:
            adult = 1e-12
        if not (m[i] < adult and res > 0.0 and light_growth > 0.0):
            continue
        cst = cost[i]
        if cst < 1e-12:
            cst = 1e-12
        dm_nutrient = res / cst
        grow_out[i] = 2 if light_growth < dm_nutrient else 1
        want = adult - m[i]
        if dm_nutrient < want:
            want = dm_nutrient
        if light_growth < want:
            want = light_growth
        stored = _store_down(m[i] + want)
        dm = stored - m[i]
        if dm <= 0.0:
            continue
        dm_out[i] = dm
        reserve_out[i] = res - dm * cost[i]
        rm = np.float64(root_out[i])
        if rm > m[i]:
            rm = m[i]
        root_out[i] = np.float32(rm + np.float64(rootalloc32[i]) * dm)
        mass_out[i] = np.float32(stored)
        energy_out[i] = np.float32(
            stored * e_labile * (1.0 - np.float64(struct32[i]))
        )

    # --- 7. summorna, seriellt och i slotordning -------------------------
    produced = 0.0
    taken = 0.0
    died = 0.0
    n_grow = 0
    light_lim = 0
    for i in range(n):
        produced += dm_out[i]
        taken += dm_out[i] * cost[i]
        g = grow_out[i]
        if g != 0:
            n_grow += 1
            if g == 2:
                light_lim += 1
        if dying_out[i] != 0:
            dv = np.float64(mass_out[i])
            if dv > 0.0:
                died += dv

    ll = np.float64(light_lim) / np.float64(n_grow) if n_grow > 0 else 0.0
    return (shed_total, n_age, n_starve, produced, taken, died, ll,
            row_plant, row_cell, row_share)


growth_kernel = _njit(cache=True, nogil=True)(_growth_kernel_impl)
