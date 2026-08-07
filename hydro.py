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
fördubbling per tio grader, och temperaturen bär årstiden. Det ger en
regnperiod på sommaren och en torr vinter utan att någon av dem kodas som en
regel — och det kostar ingenting, eftersom nederbörden är ett tal. Den rumsliga
fördelningen — våta tropiker, torra poler — föll med latituden; kvar är det
orografiska lyftet.

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
def soil_pass(soil, rain_dt, T_air, T_off, T0, T_span, oro,
              et_max_dt, capacity, baseflow_k, sea, lake_id, water,
              submerged_thr, runoff, acc):
    """
    Markvattnet, punktvis. Returnerar via `runoff` och två summor i `acc`.

    Forcingen räknas **inne i** kärnan. Att bygga nederbörds- och
    avdunstningsfälten i numpy först kostade fem täta temporärer per tick och
    stod för sju av åtta millisekunder vid 262 144 celler — samma
    representationsfel som `docs/varldens-kadensmodell.md` hittade i
    temperaturen. Nederbörd och lufttemperatur är skalärer sedan latituden
    föll; det rumsliga bärs av `oro` och `T_off`.

    **Tillväxtgrinden räknas per cell.** Temperaturen är luftens skalär plus
    höjdens statiska modifierare, och grinden är en klippt linjär funktion av
    den summan. Att räkna den här kostar ingenting: svepet går ändå över varje
    cell, och alternativet vore ett tätt fält som materialiseras och kastas.

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
        p = rain_dt * oro[i]
        s = soil[i] + p
        p_tot += p

        # Avdunstningen avtar med torrhet: full potential vid mättnad, noll vid
        # tom mark. Utan den termen torkar marken ut till exakt noll och
        # markfuktigheten blir en binär grind i stället för en gradient.
        T = T_air + T_off[i]
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
def sediment_pass(detritus, structure, slope, flow_to, flow_order,
                  sea, rate, slope_ref, struct_mobility, eps,
                  in_lab, in_str, acc):
    """
    Partikulär transport: förna följer vattnet nedströms och avsätts där
    flödet saktar.

    **Skillnaden mot löst näring är att partiklar måste ryckas med.** Löst
    material följer vattnet ovillkorligt; partiklar rör sig när skjuvningen
    räcker och sjunker när den avtar — Hjulströms samband. Takten skalas därför
    med lutningen, normerad mot `slope_ref`. En sjöcell har lutning noll och
    blir en perfekt fälla utan att det behöver kodas som en regel: bassängen
    samlar det som spolas dit och släpper det inte vidare.

    **Takten skalar med lutningen ensam, inte också med avrinningens andel.**
    Första formen multiplicerade två per-tick-storheter och blev därmed
    proportionell mot `dt²` — modellens beteende hade hängt på tidsstegets
    storlek, samma klass av fel som `water_extract_frac` var per tick i stället
    för per månad. Den gav dessutom en takt fem tiopotenser för låg: uppmätt
    2,5 procent av förnan i vatten både med och utan transport.

    **Fraktionerna rör sig olika fort**, men sorteringen syns inte i utfallet.
    Avsikten var att fint labilt material skulle transporteras lättare och ge
    vattnet lägre strukturandel. Uppmätt blir den i stället högre — 0,965 mot
    landets 0,956 — eftersom det labila bryts ned långt snabbare än det
    transporteras och material som blir liggande åldras till struktur oavsett
    var. Vattnet får mer föda, inte bättre. Mekanismen är fysiskt riktig och
    behålls, men den är i praktiken vilande.

    Ett steg per tick. Materialet samlas i `in_lab` och `in_str` och läggs på
    efter svepet, så att det som flyttats inte flyttas igen samma tick.

    Havet är sänka: det som når det lämnar modellen och bokförs som förlust.

    acc[0] = massa till havet, acc[1] = strukturmassa till havet,
    acc[2] = flyttad massa.
    """
    n = detritus.shape[0]
    for i in range(n):
        in_lab[i] = 0.0
        in_str[i] = 0.0

    to_sea = 0.0
    to_sea_str = 0.0
    moved = 0.0
    for m in range(flow_order.shape[0]):
        c = flow_order[m]
        if sea[c]:
            continue
        d = detritus[c]
        if d <= eps:
            continue
        sl = slope[c] / slope_ref
        if sl > 1.0:
            sl = 1.0
        base = rate * sl
        if base <= 0.0:
            continue

        st = structure[c]
        lab = d * (1.0 - st)
        stru = d * st
        f_lab = base
        if f_lab > 1.0:
            f_lab = 1.0
        f_str = base * struct_mobility
        if f_str > 1.0:
            f_str = 1.0
        d_lab = lab * f_lab
        d_str = stru * f_str
        if d_lab + d_str <= 0.0:
            continue

        new = (lab - d_lab) + (stru - d_str)
        detritus[c] = new
        if new > 0.0:
            structure[c] = (stru - d_str) / new
        else:
            structure[c] = 0.0
        moved += d_lab + d_str

        t = flow_to[c]
        if t < 0:
            # Ändstation utan väg ut: lägg tillbaka hellre än att tappa tyst.
            detritus[c] = d
            structure[c] = st
            moved -= d_lab + d_str
        elif sea[t]:
            to_sea += d_lab + d_str
            to_sea_str += d_str
        else:
            in_lab[t] += d_lab
            in_str[t] += d_str

    acc[0] = to_sea
    acc[1] = to_sea_str
    acc[2] = moved


@_njit(cache=True, fastmath=True)
def sediment_deposit(detritus, structure, in_lab, in_str, eps, changed):
    """
    Lägg på det transporterade materialet och blanda strukturandelen
    massviktat. Skilt från svepet så att ett steg per tick verkligen blir ett
    steg. `changed` markerar celler vars medlemskap kan ha ändrats.
    """
    n = detritus.shape[0]
    for i in range(n):
        changed[i] = False
        a = in_lab[i]
        b = in_str[i]
        if a <= 0.0 and b <= 0.0:
            continue
        d = detritus[i]
        st = structure[i]
        lab = d * (1.0 - st) + a
        stru = d * st + b
        new = lab + stru
        detritus[i] = new
        structure[i] = stru / new if new > 0.0 else 0.0
        changed[i] = True


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
