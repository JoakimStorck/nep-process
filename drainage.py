"""
Dräneringsnätet: terrängens statiska konsekvens.

Allt här beror bara på `elevation` och `Grid`s grannrelation, och terrängen
ändras inte under normala tick. Nätet byggs därför en gång vid världens
tillkomst och rörs aldrig därefter — vilket är det grepp som gör hydro billig.
Se `docs/geologin-och-vattnet.md`.

Fyra ting beräknas, i ordning:

**Fylld höjd.** Prioritetsflod från havet och inåt. Resultatet är den lägsta
höjd varje cell kan ha och ändå ha en icke-stigande väg till havet. Där den
fyllda höjden överstiger den verkliga ligger en sänka, alltså en sjö.

**Flödesriktning.** Brantaste fallet bland de sex grannarna, räknat i *fylld*
höjd. Att räkna i verklig höjd vore fel: inne i en sänka är den verkliga höjden
inte monoton mot utloppet, och nätet skulle få cykler.

**Topologisk ordning.** Kahns algoritm på trädet, källa mot mynning. Att svepa
`discharge` i den ordningen ackumulerar hela avrinningen i ett enda pass utan
att någon cell rörs mer än en gång.

**Sjöarnas hypsometri.** Cellerna i varje sänka sorterade efter höjd, med
kumulativ volym. Det gör sjönivån till en funktion av magasinet som slås upp med
`searchsorted` — och därmed sjöar som är exakt bevarande och exakt stabila, utan
per-cell-tillstånd som kan darra.

**Vad som inte finns här.** Ingen erosion. Terrängen är statisk, och att låta
vattnet forma höjden skulle göra `flow_order` dynamisk och riva hela
kostnadsargumentet.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit as _njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - beror på miljön
    HAVE_NUMBA = False

    def _njit(*a, **k):  # type: ignore[misc]
        def deco(f):
            return f

        return deco


# En sänka räknas som sjö först när den är djupare än så. Utan tröskel blir
# varje avrundningsartefakt i den fyllda höjden en sjö med noll volym.
LAKE_EPS = 1e-6


@dataclass
class Drainage:
    """
    Terrängens statiska hydrologiska struktur.

    filled       f32[n]   fylld höjd; över `elevation` inne i en sänka
    flow_to      i32[n]   granne nedströms; -1 bara i havet och i en sänka utan
                          utlopp
    slope        f32[n]   **gradient** mot flow_to: höjdfall genom grannavstånd,
                          alltså dimensionslös. 0,05 betyder fem procents lutning
    flow_order   i32[n]   topologisk ordning, källa -> mynning
    sea          bool[n]  under havsnivån
    lake_id      i32[n]   -1 utanför sjö
    outlet_lake  i32[n]   sjöns index om cellen är dess utlopp, annars -1
    lake_cells   i32[]    sjöarnas celler, sorterade efter höjd, sjö för sjö
    lake_start   i32[]    startindex per sjö i lake_cells, med slutvakt
    lake_vol     f64[]    kumulativ volym vid varje celltröskel
    lake_cap     f64[]    volym vid brädden; däröver spiller sjön
    lake_outlet  i32[]    utloppscell per sjö; dit spillet går
    upslope      f32[n]   uppströms cellantal vid enhetsavrinning
    """

    filled: np.ndarray
    flow_to: np.ndarray
    slope: np.ndarray
    flow_order: np.ndarray
    sea: np.ndarray
    lake_id: np.ndarray
    outlet_lake: np.ndarray
    lake_cells: np.ndarray
    lake_start: np.ndarray
    lake_vol: np.ndarray
    lake_cap: np.ndarray
    lake_outlet: np.ndarray
    upslope: np.ndarray

    @property
    def n_lakes(self) -> int:
        return int(self.lake_start.shape[0]) - 1


def _priority_flood(elev: np.ndarray, neighbor_idx: np.ndarray,
                    sea: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fyller sänkor så att varje cell har en icke-stigande väg till havet, och
    lämnar samtidigt trädet den byggde.

    Trädet behövs. Efter fyllningen är sjöns yta platt, och brantaste fallet
    hittar då **ingen** riktning alls inne i en sänka: alla grannar ligger på
    samma fyllda nivå, och tröskelcellen utanför likaså. Utan trädet blev varje
    sjö en ändstation — uppmätt strandade 88 procent av all avrinning där, och
    vattenbalansen fattades 27 procent.

    Floden går från havet och uppåt, så den cell en granne först nås ifrån
    ligger alltid nedströms. Trädet är rotat i havet och därmed acykliskt av
    konstruktion.

    Heapen bär cellindex som andra nyckel, så att lika höjder bryts
    deterministiskt. Utan det beror utfallet på heapens interna ordning och
    samma frö skulle kunna ge två olika världar.
    """
    n = int(elev.shape[0])
    k = int(neighbor_idx.shape[1])
    filled = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int32)
    seen = np.zeros(n, dtype=bool)

    heap: list[tuple[float, int]] = []
    for c in np.flatnonzero(sea):
        ci = int(c)
        filled[ci] = float(elev[ci])
        seen[ci] = True
        heap.append((float(elev[ci]), ci))
    if not heap:
        # Ingen havsrand: världens lägsta cell blir ändstation, annars vore
        # dräneringen odefinierad.
        ci = int(np.argmin(elev))
        filled[ci] = float(elev[ci])
        seen[ci] = True
        heap.append((float(elev[ci]), ci))
    heapq.heapify(heap)

    while heap:
        lv, c = heapq.heappop(heap)
        if lv > filled[c]:
            continue
        row = neighbor_idx[c]
        for j in range(k):
            nb = int(row[j])
            if seen[nb]:
                continue
            e = float(elev[nb])
            cand = lv if e < lv else e
            filled[nb] = cand
            parent[nb] = c
            seen[nb] = True
            heapq.heappush(heap, (cand, nb))
    return filled, parent


@_njit(cache=True)
def _flow_dirs(filled, neighbor_idx, sea, parent, nb_dist):
    """
    Brantaste fallet där ett fall finns, flodträdet där det inte gör det.

    Sluttningar dräneras därmed efter sin egen gradient, medan plana ytor —
    sjöar och utfyllda sänkor — följer trädet till tröskeln.

    `slope` är en verklig gradient: höjdfallet delat med grannavståndet. Utan
    delningen är talet en höjdskillnad och inte en lutning, och varje
    parameter som läser det tappar sin dimensionslöshet. På hex är delningen
    gratis eftersom alla sex grannar ligger på exakt samma avstånd. Blandningen är
    acyklisk: den fyllda höjden avtar aldrig längs en väg, och de plana
    partierna följer ett träd rotat i havet.
    """
    n = filled.shape[0]
    k = neighbor_idx.shape[1]
    to = np.full(n, -1, dtype=np.int32)
    slope = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if sea[i]:
            continue
        best = -1
        bestd = 0.0
        for j in range(k):
            nb = neighbor_idx[i, j]
            d = filled[i] - filled[nb]
            if d > bestd:
                bestd = d
                best = nb
        if best < 0:
            best = parent[i]
            bestd = 0.0
        to[i] = best
        slope[i] = bestd / nb_dist
    return to, slope


@_njit(cache=True)
def _topo_order(to):
    """Kahns algoritm. Returnerar ordningen och hur många celler den täckte."""
    n = to.shape[0]
    indeg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        t = to[i]
        if t >= 0:
            indeg[t] += 1
    order = np.empty(n, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)
    top = 0
    for i in range(n):
        if indeg[i] == 0:
            stack[top] = i
            top += 1
    m = 0
    while top > 0:
        top -= 1
        c = stack[top]
        order[m] = c
        m += 1
        t = to[c]
        if t >= 0:
            indeg[t] -= 1
            if indeg[t] == 0:
                stack[top] = t
                top += 1
    return order, m


@_njit(cache=True)
def route(supply, to, order, out):
    """
    Ackumulera nedströms i ett svep. Nätverkskadensens kärna: varje cell rörs
    exakt en gång, i en ordning som geometrin och terrängen bestämt en gång.
    """
    n = out.shape[0]
    for i in range(n):
        out[i] = supply[i]
    for m in range(order.shape[0]):
        c = order[m]
        t = to[c]
        if t >= 0:
            out[t] += out[c]


def _label_lakes(filled, elev, neighbor_idx, sea):
    """
    Sammanhängande sänkor med samma fyllnadsnivå bildar en sjö. Etiketten sätts
    med en flodfyllning i grannmatrisen.
    """
    n = int(elev.shape[0])
    k = int(neighbor_idx.shape[1])
    depth = filled - elev
    is_lake = (depth > LAKE_EPS) & (~sea)
    lake_id = np.full(n, -1, dtype=np.int32)

    groups: list[list[int]] = []
    stack: list[int] = []
    for c in np.flatnonzero(is_lake):
        ci = int(c)
        if lake_id[ci] >= 0:
            continue
        gid = len(groups)
        members: list[int] = []
        lake_id[ci] = gid
        stack.append(ci)
        level = float(filled[ci])
        while stack:
            a = stack.pop()
            members.append(a)
            row = neighbor_idx[a]
            for j in range(k):
                nb = int(row[j])
                if lake_id[nb] >= 0 or not is_lake[nb]:
                    continue
                if abs(float(filled[nb]) - level) > LAKE_EPS:
                    continue
                lake_id[nb] = gid
                stack.append(nb)
        groups.append(members)

    return lake_id, groups


def build(grid, elevation: np.ndarray, sea_level: float = 0.0) -> Drainage:
    """
    Bygg hela nätet. O(n log n) en gång; se `docs/geologin-och-vattnet.md` för
    uppmätta tider.
    """
    elev = np.asarray(elevation, dtype=np.float64).ravel()
    n = int(elev.shape[0])
    if n != int(grid.n_cells):
        raise ValueError(f"elevation har längd {n}, förväntat {int(grid.n_cells)}")
    idx = np.ascontiguousarray(grid.neighbor_idx, dtype=np.int32)

    sea = elev < float(sea_level)
    filled, parent = _priority_flood(elev, idx, sea)
    to, slope = _flow_dirs(filled, idx, sea, parent,
                           float(np.asarray(grid.neighbor_dist)[0]))

    lake_id, groups = _label_lakes(filled, elev, idx, sea)

    # Inne i en sjö bär inte cell-till-cell-flödet någon fysik — hela bassängen
    # är ett magasin. Men avrinningen som faller *på* sjön måste nå magasinet,
    # och magasinet ligger vid utloppscellen. Flodträdet ger den vägen gratis:
    # eftersom floden går nedifrån och upp pekar varje sjöcell mot den granne
    # som ligger närmare tröskeln, och exakt en cell per sjö pekar ut ur den.
    lake_cells_l: list[np.ndarray] = []
    lake_start_l: list[int] = [0]
    lake_vol_l: list[np.ndarray] = []
    lake_cap_l: list[float] = []
    lake_outlet_l: list[int] = []
    outlet_lake = np.full(n, -1, dtype=np.int32)

    for gid, members in enumerate(groups):
        mem = np.asarray(members, dtype=np.int32)
        outs = [int(c) for c in mem
                if int(to[c]) >= 0 and lake_id[int(to[c])] != gid]
        if len(outs) == 0:
            # Sänkan har ingen väg ut — bara möjligt om prioritetsfloden inte
            # nådde havet. Låt lägsta cellen bli ändstation.
            outlet = int(mem[np.argmin(filled[mem])])
            to[outlet] = -1
        else:
            # Flera tröskelceller på exakt samma nivå är en verklig
            # oavgjordhet: bassängen skulle spilla åt två håll. Magasinet kan
            # bara ha ett utlopp, så det brantaste vinner och cellindex bryter
            # lika — determinism före realism, eftersom skillnaden är en
            # flyttalstie.
            outlet = min(outs, key=lambda c: (-float(slope[c]), c))

        # Cellerna dräneras till utloppet med ett bredden-först-träd i
        # grannmatrisen. Utan det kan en sjöcell peka ut ur sjön förbi
        # magasinet, och tillflödet räknas två gånger.
        in_lake = set(int(c) for c in mem)
        seen_l = {outlet}
        queue = [outlet]
        qi = 0
        while qi < len(queue):
            a = queue[qi]
            qi += 1
            row = idx[a]
            for j in range(int(idx.shape[1])):
                nb = int(row[j])
                if nb in in_lake and nb not in seen_l:
                    seen_l.add(nb)
                    to[nb] = a
                    slope[nb] = 0.0
                    queue.append(nb)
        if len(seen_l) != mem.size:
            raise RuntimeError(
                f"sjö {gid} är inte sammanhängande i grannmatrisen: "
                f"{len(seen_l)} av {mem.size} celler nåddes från utloppet"
            )

        order_by_z = mem[np.argsort(elev[mem], kind="stable")]
        z = elev[order_by_z]
        # Volym när ytan står vid tröskel i: summan av (z_i - z_j) för j < i.
        # Cellarean är 1, så volym och djup delar enhet.
        vol = np.concatenate(([0.0], np.cumsum(np.arange(1, z.size) * np.diff(z))))
        # Brädden är den fyllda nivån: däröver spiller sjön genom utloppet.
        cap = float(np.sum(filled[order_by_z] - z))

        lake_cells_l.append(order_by_z)
        lake_vol_l.append(vol)
        lake_cap_l.append(cap)
        lake_start_l.append(lake_start_l[-1] + int(order_by_z.size))
        lake_outlet_l.append(outlet)
        outlet_lake[outlet] = gid

    lake_cells = (np.concatenate(lake_cells_l).astype(np.int32, copy=False)
                  if lake_cells_l else np.zeros(0, dtype=np.int32))
    lake_vol = (np.concatenate(lake_vol_l).astype(np.float64, copy=False)
                if lake_vol_l else np.zeros(0, dtype=np.float64))
    lake_start = np.asarray(lake_start_l, dtype=np.int32)
    lake_outlet = np.asarray(lake_outlet_l, dtype=np.int32)
    lake_cap = np.asarray(lake_cap_l, dtype=np.float64)

    order, m = _topo_order(to)
    if m != n:
        raise RuntimeError(
            f"dräneringsnätet har en cykel: topologisk ordning täckte {m} av {n} celler"
        )

    upslope = np.empty(n, dtype=np.float64)
    route(np.ones(n, dtype=np.float64), to, order, upslope)

    return Drainage(
        filled=filled.astype(np.float32, copy=False),
        flow_to=to,
        slope=slope,
        flow_order=order,
        sea=sea,
        lake_id=lake_id,
        outlet_lake=outlet_lake,
        lake_cells=lake_cells,
        lake_start=lake_start,
        lake_vol=lake_vol,
        lake_cap=lake_cap,
        lake_outlet=lake_outlet,
        upslope=upslope.astype(np.float32, copy=False),
    )


def describe(dr: Drainage) -> dict:
    """Sammanfattning. Underlag för loggrad och gallring av frön."""
    lake = dr.lake_id >= 0
    land = (~dr.sea) & (~lake)
    n = float(dr.sea.shape[0])
    return {
        "sea_frac": float(dr.sea.mean()),
        "lake_frac": float(lake.mean()),
        "land_frac": float(land.mean()),
        "n_lakes": int(dr.n_lakes),
        "largest_lake": (int(np.bincount(dr.lake_id[lake]).max()) if lake.any() else 0),
        "max_upslope": float(dr.upslope.max()),
        "upslope_p99": float(np.quantile(dr.upslope[land], 0.99)) if land.any() else 0.0,
    }
