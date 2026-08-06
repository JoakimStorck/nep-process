"""
Hydrologin: jämvikt per tick, inte transient.

Tidssteget är 0,02 månader, ungefär femton timmar. Faunan rör sig 0,74
cellbredder på den tiden; vatten i en fåra hinner korsa hela världen. Ett
explicit grannflöde håller informationshastigheten under ungefär en cell per
tick och skulle därför låta en flod behöva tio simulerade månader på att nå
havet — en fördröjning som är numerik utan biologisk motsvarighet. Hydro löser
i stället stationärt tillstånd varje tick längs `drainage.flow_order`. Fysiken
är oförändrat lokal, gradientdriven och kontinuitetsbevarande; det är
lösningsmetoden som skiljer. Se `docs/geologin-och-vattnet.md`.

**Tre lager med tre kadenser.**

`soil_water` är tätt dynamiskt men **punktvis** — ingen grannkoppling alls, så
ett svep utan minnestrafik mellan celler. Det är fältet biologin läser som
markfuktighet.

`discharge` är nätverksdynamiskt: ett svep över den topologiska ordningen där
varje cell rörs exakt en gång. Det är ingen av de fyra kadensklasser
`docs/varldens-kadensmodell.md` beskrev, utan en femte — tät men riktad.

`lake_storage` är per bassäng, inte per cell. Nivån slås upp ur hypsometrin med
ett `searchsorted`. Det gör sjöar exakt bevarande och exakt stabila: en sjö kan
inte darra kring en tröskel om den inte har ett per-cell-tillstånd att darra i.

**Vattenbalansen är exakt av strukturella skäl.** Routing är en ren överföring,
reservoaren en ren omfördelning mellan magasin och spill. Stocken är
`soil_water` plus sjömagasinen; kanalvatten är i transit och lagras inte, vilket
följer av jämviktsantagandet. In är nederbörd, ut är avdunstning plus det som
når havet.

**Nederbörden följer temperaturen.** Varm luft bär mer fukt, ungefär en
fördubbling per tio grader, och temperaturen är redan en bandprofil med
årstid. Det ger våta tropiker, torra poler och en regnperiod på sommaren utan
att någon av de tre kodas som en regel — och det kostar ingenting, eftersom
profilen har längd `n_bands`.

**Vad som inte finns här.** Ingen regnskugga: den kräver en förhärskande
vindriktning, och den orografiska modifieraren är tills vidare bara höjdlyft.
Ingen erosion, ingen transient översvämning — båda följer av att terrängen och
nätet är statiska.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit as _njit, prange as _prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - beror på miljön
    HAVE_NUMBA = False
    _prange = range

    def _njit(*a, **k):  # type: ignore[misc]
        def deco(f):
            return f

        return deco


@_njit(cache=True, fastmath=True, parallel=True)
def soil_pass(soil, bands, rain_band, T_band, T_off, T0, T_span, oro,
              et_max_dt, capacity, baseflow_k, sea, lake_id, water,
              submerged_thr, runoff, acc):
    """
    Markvattnet, punktvis. Returnerar via `runoff` och två summor i `acc`.

    Forcingen räknas **inne i** kärnan. Att bygga nederbörds- och
    avdunstningsfälten i numpy först kostade fem täta temporärer per tick och
    stod för sju av åtta millisekunder vid 262 144 celler — samma
    representationsfel som `docs/varldens-kadensmodell.md` hittade i
    temperaturen. Profilerna har längd `n_bands`; uppslaget är en gather.

    **Tillväxtgrinden räknas per cell, inte per band.** Temperaturen är
    bandprofilen plus höjdens statiska modifierare, och grinden är en klippt
    linjär funktion av den summan — alltså inte konstant inom ett band. Att
    räkna den här kostar ingenting: svepet går ändå över varje cell, och
    alternativet vore ett tätt fält som materialiseras och kastas.

    Ordningen är nederbörd, avdunstning, mättnadsöverskott, basflöde. Att lägga
    avdunstningen före överskottet gör att ett regn som ändå avdunstar aldrig
    blir avrinning, vilket är skillnaden mellan en torr och en våt regim vid
    samma årsnederbörd.

    acc[0] = tillförd nederbörd, acc[1] = avdunstat. Båda över land; havet har
    inget markvatten och räknas i sitt eget lager.
    """
    n = soil.shape[0]
    p_tot = 0.0
    e_tot = 0.0
    for i in _prange(n):
        # En cell under vatten har ingen mark att lagra fukt i. Havet gäller
        # alltid; en sjöcell gäller när magasinets yta faktiskt står över den,
        # så att en strand blir bar mark när nivån sjunker i stället för att
        # vara permanent dränkt av sjöns största utbredning.
        #
        # Det som fanns kvar blir avrinning i stället för att försvinna, så att
        # övergången mellan de två regimerna bevarar massa.
        if sea[i] or (lake_id[i] >= 0 and water[i] > submerged_thr):
            runoff[i] = soil[i]
            soil[i] = 0.0
            continue
        b = bands[i]
        p = rain_band[b] * oro[i]
        s = soil[i] + p
        p_tot += p

        # Avdunstningen avtar med torrhet: full potential vid mättnad, noll vid
        # tom mark. Utan den termen torkar marken ut till exakt noll och
        # markfuktigheten blir en binär grind i stället för en gradient.
        T = T_band[b] + T_off[i]
        g = (T - T0) / T_span
        if g < 0.0:
            g = 0.0
        elif g > 1.0:
            g = 1.0
        e = g * et_max_dt * (s / capacity)
        if e > s:
            e = s
        s -= e
        e_tot += e

        q = 0.0
        if s > capacity:
            q = s - capacity
            s = capacity
        # Basflödet är det som håller fåror rinnande mellan regnen. Utan det
        # torkar varje vattendrag ut samma tick som regnet slutar.
        bf = baseflow_k * s
        s -= bf
        q += bf

        soil[i] = s
        runoff[i] = q
    acc[0] = p_tot
    acc[1] = e_tot


@_njit(cache=True, fastmath=True)
def route_reservoirs(supply, flow_to, flow_order, outlet_lake, lake_id,
                     storage, capacity, sea, soil, soil_cap, reinf_max,
                     discharge, acc):
    """
    Ett svep nedströms med sjöarna som magasin och marken som mottagare.

    Sjöns celler dränerar till sin utloppscell längs grannrelationen, så när
    svepet når utloppet har hela bassängens tillflöde samlats där. Där går det
    in i magasinet, och bara spillet fortsätter nedströms. Bassängen kan därmed
    fyllas under torrår och tömmas under blöta utan att någon cell bär
    tillståndet.

    **Återinfiltrationen är inte en detalj.** Utan den är markvattnet rent
    punktvis: två celler med samma klimat blir lika våta oavsett var i
    landskapet de ligger, dalar blir inte blötare än ryggar, och den
    topografiska fuktgradienten — hela skälet att bygga hydro — uppstår aldrig.
    Marken tar därför upp det som passerar, begränsat av två tak: cellens
    omättade utrymme och en infiltrationstakt. Vattnet flyttar från transit
    till lager och balansen är exakt.

    **Taket är absolut, inte en andel.** Som andel — tjugofem procent per cell —
    var det kvar 0,3 procent av flödet efter tjugo steg, och fåror kunde inte
    bildas alls: uppmätt noll procent av landet vid regn 0,5. Med ett absolut
    tak sugs en liten bäck upp av marken medan en stor flod korsar en torka
    med försumbar förlust, vilket också är hur efemära vattendrag faktiskt
    beter sig.

    acc[0] = summa som nådde havet, acc[1] = summa som återinfiltrerade.
    """
    n = discharge.shape[0]
    for i in range(n):
        discharge[i] = supply[i]

    out_sea = 0.0
    reinf = 0.0
    for m in range(flow_order.shape[0]):
        c = flow_order[m]
        L = outlet_lake[c]
        if L >= 0:
            storage[L] += discharge[c]
            spill = storage[L] - capacity[L]
            if spill > 0.0:
                storage[L] = capacity[L]
            else:
                spill = 0.0
            discharge[c] = spill
        elif lake_id[c] < 0 and not sea[c] and discharge[c] > 0.0:
            room = soil_cap - soil[c]
            if room > 0.0:
                take = reinf_max
                if take > room:
                    take = room
                if take > discharge[c]:
                    take = discharge[c]
                soil[c] += take
                discharge[c] -= take
                reinf += take
        t = flow_to[c]
        if t >= 0:
            discharge[t] += discharge[c]
        elif sea[c]:
            # Den topologiska ordningen garanterar att allt uppströms redan
            # räknats när cellen nås, så summan kan tas här i stället för i ett
            # andra svep över hela världen.
            out_sea += discharge[c]
    acc[0] = out_sea
    acc[1] = reinf


@_njit(cache=True, fastmath=True)
def leach_pass(nutrient, soil, runoff, flow_to, flow_order, lake_id, sea,
               efficiency, acc):
    """
    Urlakning: löst näring följer vattnet ett steg nedströms per tick.

    **Takten kommer ur vattnets egen budget och inte ur en ny parameter.**
    Näringen är löst i markvattnet, så den andel av vattnet som lämnar cellen
    bär med sig samma andel av det lösta:

        andel = avrinning / (markvatten + avrinning)

    Det ger utan kalibrering det man vill ha: en brant cell med mycket
    avrinning och lite kvarhållet vatten lakas ur snabbt, en flack dalbotten
    långsamt. `efficiency` under ett svarar mot att en del av näringen är
    bunden till partiklar och inte följer med i lösning.

    **Gradienten kommer ur routingen, inte ur takten.** Varje cell skickar sin
    urlakning ett steg nedåt, så en cell långt ned i nätet tar emot allt som
    passerat ovanför. Att takten är densamma överallt hindrar alltså inte att
    stocken blir monotont växande nedströms — det är ackumulationen som gör
    lågland bördigt, precis som i en verklig flodslätt.

    Havet är sänka. Näring som når det lämnar modellen; ingenting återförs.
    Budgeten är vittring in mot urlakning ut, vilket är hur en landekologi
    faktiskt hushållar.

    acc[0] = summa som nådde havet, acc[1] = summa som flyttades.
    """
    n = nutrient.shape[0]
    out_sea = 0.0
    moved = 0.0

    # Havet töms först. Näring når det inte bara genom urlakning: diffusionen i
    # `transport_pass` trycker in den över kustlinjen, och flora som dör i en
    # havscell mineraliserar där. Uppmätt 9,88 kg efter 500 tick i en värld
    # utan biologi alls, enbart från diffusionen. Utan den här raden blir havet
    # en pool som växer monotont och som ingenting kan nå — samma sorts tyst
    # ansamling som den låsta reproduktionsreserven en gång var.
    for i in range(n):
        if sea[i] and nutrient[i] > 0.0:
            out_sea += nutrient[i]
            nutrient[i] = 0.0

    for m in range(flow_order.shape[0]):
        c = flow_order[m]
        if sea[c]:
            continue
        q = runoff[c]
        if q <= 0.0:
            continue
        avail = nutrient[c]
        if avail <= 0.0:
            continue
        denom = soil[c] + q
        if denom <= 0.0:
            continue
        frac = efficiency * q / denom
        if frac > 1.0:
            frac = 1.0
        take = avail * frac
        nutrient[c] = avail - take
        moved += take
        t = flow_to[c]
        if t < 0:
            # Ändstation utan väg ut — lägg tillbaka hellre än att tappa den
            # tyst. Kan bara inträffa i en sänka utan utlopp.
            nutrient[c] = avail
            moved -= take
        elif sea[t]:
            out_sea += take
        else:
            nutrient[t] += take
    acc[0] = out_sea
    acc[1] = moved


@_njit(cache=True, fastmath=True)
def lake_levels(storage, lake_start, lake_cells, lake_vol, elev, level, area):
    """
    Nivå och yta ur magasinet, per sjö. `searchsorted` för hand, eftersom
    hypsometrin ligger packad i en enda array.
    """
    nl = lake_start.shape[0] - 1
    for L in range(nl):
        a = lake_start[L]
        b = lake_start[L + 1]
        V = storage[L]
        m = b - a
        if m <= 0 or V <= 0.0:
            level[L] = elev[lake_cells[a]] if m > 0 else 0.0
            area[L] = 0.0
            continue
        # Största i med lake_vol[a+i] <= V
        lo = 0
        hi = m - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if lake_vol[a + mid] <= V:
                lo = mid
            else:
                hi = mid - 1
        k = lo + 1  # antal dränkta celler
        z = elev[lake_cells[a + lo]]
        level[L] = z + (V - lake_vol[a + lo]) / k
        area[L] = k


@_njit(cache=True, fastmath=True, parallel=True)
def derive_water(elev, sea, lake_id, lake_level, discharge, slope,
                 channel_k, channel_exp, slope_floor, q_min, water):
    """
    Stående vattendjup per cell. Härlett varje tick ur de tre lagren.

    Havet står vid nivå noll, sjön vid sin magasinsnivå, och en fåra bär ett
    djup som växer med genomströmningen och avtar med lutningen — formen är
    Mannings, förenklad till en potenslag eftersom bredden inte modelleras.

    `q_min` skiljer fåra från sluttning. Det är inte en optimering utan en
    modellutsaga: avrinning under tröskeln är ytavrinning och markflöde, inte
    stående vatten, och en organism möter ingen vattenyta där. Att räkna
    potenslagen för varje landcell kostade fem av åtta millisekunder vid
    262 144 celler, eftersom `pow` med bruten exponent är dyr — och nio
    tiondelar av dem var sluttning.
    """
    n = water.shape[0]
    for i in _prange(n):
        if sea[i]:
            water[i] = -elev[i]
            continue
        L = lake_id[i]
        if L >= 0:
            d = lake_level[L] - elev[i]
            water[i] = d if d > 0.0 else 0.0
            continue
        q = discharge[i]
        if q <= q_min:
            water[i] = 0.0
            continue
        s = slope[i]
        if s < slope_floor:
            s = slope_floor
        water[i] = channel_k * (q / math.sqrt(s)) ** channel_exp
