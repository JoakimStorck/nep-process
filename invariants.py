# invariants.py
"""
Invariantkontroller för nep-process.

Modulen är avsiktligt läsande och beroendefri: den importerar bara numpy och
inspekterar Population/OrganismStore utifrån. Den muterar aldrig simuleringens
tillstånd och kan därför köras mellan tick i vilken körning som helst.

Kontrollerna motsvarar de invarianter manifestet formulerar som krav:

  - cell_idx är en invariant, inte ett opportunistiskt fält
  - slotindex och organism-id är distinkta begrepp
  - lediga slotar återanvänds utan dubbelallokering
  - inga NaN eller negativa bevarade storheter
  - spatialindexet är konsistent med organismernas faktiska celler
  - Agent.store_slot pekar på rätt slot

Notera vad som medvetet *inte* kontrolleras som hård invariant: total massa
och energi. Systemet är öppet by construction — floran växer ur ingenting och
kadaverfältet sönderfaller ut ur modellen — så någon sluten balans finns inte
att pröva förrän näringskretsloppet är på plats. Fram till dess spåras massa
och energi som diagnostik i `diagnostics()`, inte som assertions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


MAX_EXAMPLES = 5


@dataclass
class Violation:
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.detail}"


@dataclass
class InvariantReport:
    tick: int
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        if self.ok:
            return f"tick {self.tick}: alla invarianter uppfyllda"
        lines = [f"tick {self.tick}: {len(self.violations)} brott"]
        lines.extend(f"  {v}" for v in self.violations)
        return "\n".join(lines)


def _live_slots(store) -> np.ndarray:
    """Slotindex för levande organismer, som int-array."""
    n = int(store.n)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(store.alive[:n]).astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Enskilda kontroller
# ---------------------------------------------------------------------------

def check_slot_bookkeeping(pop) -> list[Violation]:
    """Lediga slotar är unika, inom intervall och inte levande."""
    store = pop.store
    out: list[Violation] = []

    free = list(store.free_slots)
    cap = int(store.capacity)

    if len(set(free)) != len(free):
        dupes = sorted({s for s in free if free.count(s) > 1})[:MAX_EXAMPLES]
        out.append(Violation("slot_bookkeeping", f"dubbletter i free_slots: {dupes}"))

    bad_range = [s for s in free if not (0 <= int(s) < cap)][:MAX_EXAMPLES]
    if bad_range:
        out.append(Violation("slot_bookkeeping", f"free_slots utanför [0,{cap}): {bad_range}"))

    n = int(store.n)
    live_free = [s for s in free if 0 <= int(s) < n and bool(store.alive[int(s)])][:MAX_EXAMPLES]
    if live_free:
        out.append(Violation("slot_bookkeeping", f"slotar både lediga och levande: {live_free}"))

    n_live = int(_live_slots(store).size)
    if n_live + len(free) > cap:
        out.append(
            Violation(
                "slot_bookkeeping",
                f"levande ({n_live}) + lediga ({len(free)}) överstiger capacity ({cap})",
            )
        )

    return out


def check_cell_idx(pop) -> list[Violation]:
    """cell_idx ska alltid vara konsistent med pos_x/pos_y via Grid."""
    store = pop.store
    grid = pop.grid
    out: list[Violation] = []

    slots = _live_slots(store)
    bad: list[str] = []
    for s in slots:
        i = int(s)
        expected = int(grid.cell_of(float(store.pos_x[i]), float(store.pos_y[i])))
        actual = int(store.cell_idx[i])
        if expected != actual:
            bad.append(
                f"slot {i} (id {int(store.id[i])}): cell_idx={actual}, "
                f"pos=({float(store.pos_x[i]):.3f},{float(store.pos_y[i]):.3f}) -> {expected}"
            )
            if len(bad) >= MAX_EXAMPLES:
                break

    if bad:
        out.append(Violation("cell_idx", "; ".join(bad)))
    return out


def check_identity(pop) -> list[Violation]:
    """Levande organismer har unika, giltiga id och korrekt id->slot-lookup."""
    store = pop.store
    out: list[Violation] = []

    slots = _live_slots(store)
    if slots.size == 0:
        return out

    ids = store.id[slots].astype(np.int64, copy=False)

    neg = slots[ids < 0]
    if neg.size:
        out.append(Violation("identity", f"levande slotar utan id: {neg[:MAX_EXAMPLES].tolist()}"))

    uniq, counts = np.unique(ids[ids >= 0], return_counts=True)
    dupes = uniq[counts > 1]
    if dupes.size:
        out.append(Violation("identity", f"id delade av flera levande slotar: {dupes[:MAX_EXAMPLES].tolist()}"))

    bad_lookup: list[str] = []
    for s in slots:
        i = int(s)
        oid = int(store.id[i])
        if oid < 0:
            continue
        if int(store.slot_for_id(oid)) != i:
            bad_lookup.append(f"id {oid} -> {int(store.slot_for_id(oid))}, förväntat {i}")
            if len(bad_lookup) >= MAX_EXAMPLES:
                break
    if bad_lookup:
        out.append(Violation("identity", "; ".join(bad_lookup)))

    return out


def check_finite_and_nonnegative(pop) -> list[Violation]:
    """Inga NaN/inf, och inga negativa bevarade storheter hos levande organismer."""
    store = pop.store
    out: list[Violation] = []

    slots = _live_slots(store)
    if slots.size:
        for name in ("pos_x", "pos_y", "mass", "energy", "age"):
            arr = np.asarray(getattr(store, name))[slots]
            bad = slots[~np.isfinite(arr)]
            if bad.size:
                out.append(Violation("finite", f"store.{name} icke-finit i slotar {bad[:MAX_EXAMPLES].tolist()}"))

        for name in ("mass", "energy"):
            arr = np.asarray(getattr(store, name))[slots]
            bad = slots[arr < 0.0]
            if bad.size:
                out.append(Violation("nonnegative", f"store.{name} negativ i slotar {bad[:MAX_EXAMPLES].tolist()}"))

    detritus = np.asarray(pop.world.detritus)
    if not np.all(np.isfinite(detritus)):
        out.append(Violation("finite", "world.detritus innehåller icke-finita värden"))
    if np.any(detritus < 0.0):
        out.append(Violation("nonnegative", f"world.detritus har negativa celler (min={float(detritus.min()):.3e})"))

    bad_agents: list[str] = []
    for a in pop.agents:
        body = a.body
        vals = (float(a.x), float(a.y), float(body.M), float(body.E_total()), float(body.D))
        if not all(np.isfinite(v) for v in vals):
            bad_agents.append(f"agent {int(a.id)}")
            if len(bad_agents) >= MAX_EXAMPLES:
                break
    if bad_agents:
        out.append(Violation("finite", f"icke-finit agenttillstånd: {', '.join(bad_agents)}"))

    return out


def check_spatial_index(pop) -> list[Violation]:
    """CSR-indexet ska innehålla exakt de levande slotarna, grupperade rätt."""
    store = pop.store
    out: list[Violation] = []

    offsets = np.asarray(store.cell_offsets)
    counts = np.asarray(store.cell_counts)
    n_cells = int(counts.size)

    if int(offsets[0]) != 0:
        out.append(Violation("spatial_index", f"cell_offsets[0]={int(offsets[0])}, förväntat 0"))

    expected_offsets = np.concatenate(([0], np.cumsum(counts)))
    if not np.array_equal(offsets, expected_offsets):
        out.append(Violation("spatial_index", "cell_offsets stämmer inte med cumsum(cell_counts)"))
        return out

    total = int(offsets[n_cells])
    live = _live_slots(store)
    live_placed = int(np.count_nonzero(store.cell_idx[live] >= 0)) if live.size else 0
    if total != live_placed:
        out.append(
            Violation("spatial_index", f"indexet rymmer {total} slotar, {live_placed} levande med giltig cell")
        )

    packed = np.asarray(store.cell_slots)[:total]
    if packed.size:
        if not np.all(store.alive[packed]):
            dead = packed[~store.alive[packed]]
            out.append(Violation("spatial_index", f"döda slotar i indexet: {dead[:MAX_EXAMPLES].tolist()}"))

        cell_of_packed = np.repeat(np.arange(n_cells, dtype=np.int64), counts)
        mismatch = packed[store.cell_idx[packed] != cell_of_packed]
        if mismatch.size:
            out.append(
                Violation("spatial_index", f"slotar i fel cellbucket: {mismatch[:MAX_EXAMPLES].tolist()}")
            )

    return out


def check_agent_store_binding(pop) -> list[Violation]:
    """Varje levande agent pekar på en levande faunaslot med matchande id."""
    store = pop.store
    out: list[Violation] = []

    bad: list[str] = []
    seen: dict[int, int] = {}

    for a in pop.agents:
        if not a.body.alive:
            continue
        slot = int(a.store_slot)
        if slot < 0:
            bad.append(f"agent {int(a.id)} saknar store_slot")
        elif not bool(store.alive[slot]):
            bad.append(f"agent {int(a.id)} pekar på död slot {slot}")
        elif int(store.id[slot]) != int(a.id):
            bad.append(f"agent {int(a.id)} pekar på slot {slot} med id {int(store.id[slot])}")
        elif int(store.kind[slot]) != 0:
            bad.append(f"agent {int(a.id)} pekar på floraslot {slot}")
        elif slot in seen:
            bad.append(f"slot {slot} delas av agent {seen[slot]} och {int(a.id)}")
        else:
            seen[slot] = int(a.id)

        if len(bad) >= MAX_EXAMPLES:
            break

    if bad:
        out.append(Violation("agent_binding", "; ".join(bad)))
    return out


def check_array_domains(pop) -> list[Violation]:
    """
    Varje store-array ska ligga i rätt indexdomän: slotindexerade arrayer har
    längd capacity, cellindexerade har längd n_cells. Store:n växer dynamiskt,
    och en array som växer i fel domän ger tyst korruption eller en krasch långt
    från orsaken.
    """
    store = pop.store
    out: list[Violation] = []

    try:
        store._assert_array_domains()
    except AssertionError as exc:
        out.append(Violation("array_domains", str(exc)))
    except AttributeError:
        pass

    return out


def check_body_store_mirror(pop) -> list[Violation]:
    """
    Store-fält som speglas från Body måste faktiskt stämma med Body.

    Fram till Steg 5 är Body source of truth för fauna-fysiologin, medan store
    bär en cache som passen läser slotbaserat. Skrivriktningen ska alltid vara
    Body -> store. Divergens innebär att någon skrivit åt fel håll eller missat
    en cacheuppdatering, och den vore annars tyst: grindarna skulle läsa ett
    värde som inte längre motsvarar organismen.

    Kontrolleras med float32-tolerans eftersom store lagrar float32 och Body
    räknar i float64.
    """
    store = pop.store
    out: list[Violation] = []

    rtol = 1e-6
    atol = 1e-12

    bad: list[str] = []
    for a in pop.agents:
        if not a.body.alive:
            continue
        s = int(a.store_slot)
        if s < 0 or s >= int(store.n):
            continue

        body = a.body

        if bool(store.gestating[s]) != bool(body.gestating):
            bad.append(
                f"slot {s} (id {int(a.id)}): gestating store={bool(store.gestating[s])} "
                f"body={bool(body.gestating)}"
            )

        pairs = (
            ("gest_M", float(store.gest_M[s]), float(body.gest_M)),
            ("gest_E_J", float(store.gest_E_J[s]), float(body.gest_E_J)),
            ("gest_M_target", float(store.gest_M_target[s]), float(body.gest_M_target)),
            ("mass", float(store.mass[s]), float(body.M)),
            ("energy", float(store.energy[s]), float(body.E_total())),
            ("damage", float(store.damage[s]), float(body.D)),
            ("wear", float(store.wear[s]), float(body.W)),
        )
        for name, got, want in pairs:
            if abs(got - want) > (atol + rtol * abs(want)):
                bad.append(f"slot {s} (id {int(a.id)}): {name} store={got:.9g} body={want:.9g}")

        if len(bad) >= MAX_EXAMPLES:
            break

    if bad:
        out.append(Violation("body_store_mirror", "; ".join(bad[:MAX_EXAMPLES])))
    return out


WORLD_CELL_FIELDS = (
    "elevation", "water", "nutrient", "detritus",
    "rain_input", "spring_input", "infiltration", "evaporation",
    "surface_level", "submerged", "flow_strength",
    "T_cell", "g_cell",
)


def check_world_field_domains(pop) -> list[Violation]:
    """
    Världsfälten ska vara platta per-cell-arrayer med längd n_cells.

    Ett fält som råkar behålla eller återfå rutnätsform är den regression som
    Steg 1 finns till för att förhindra: den fungerar så länge geometrin är
    kvadratisk och går sönder tyst först vid hexbytet.
    """
    world = pop.world
    n_cells = int(pop.grid.n_cells)
    out: list[Violation] = []

    bad: list[str] = []
    for name in WORLD_CELL_FIELDS:
        arr = getattr(world, name, None)
        if arr is None:
            bad.append(f"{name} saknas")
            continue
        arr = np.asarray(arr)
        if arr.ndim != 1:
            bad.append(f"{name} har form {arr.shape}, förväntat 1D")
        elif int(arr.shape[0]) != n_cells:
            bad.append(f"{name} har längd {int(arr.shape[0])}, förväntat n_cells={n_cells}")

    if bad:
        out.append(Violation("world_field_domains", "; ".join(bad[:MAX_EXAMPLES])))

    return out


ALL_CHECKS = (
    check_slot_bookkeeping,
    check_world_field_domains,
    check_body_store_mirror,
    check_array_domains,
    check_cell_idx,
    check_identity,
    check_finite_and_nonnegative,
    check_spatial_index,
    check_agent_store_binding,
)


def check_all(pop, tick: int = -1) -> InvariantReport:
    """Kör samtliga invariantkontroller och returnera en samlad rapport."""
    report = InvariantReport(tick=int(tick))
    for check in ALL_CHECKS:
        report.violations.extend(check(pop))
    return report


# ---------------------------------------------------------------------------
# Diagnostik — spåras men assertas inte
# ---------------------------------------------------------------------------

def check_grid_reference(grid, sample_stride: int = 1) -> list[Violation]:
    """
    Referensegenskaper för en geometriimplementation.

    Prövar en Grid mot egenskaper som måste hålla oavsett cellform, så att en ny
    geometri kan valideras innan den kopplas in. Grafavståndet från
    bredden-först-sökning är facit; en sluten avståndsformel måste stämma med
    det, inte tvärtom.
    """
    from collections import deque

    out: list[Violation] = []
    n = int(grid.n_cells)
    k = int(grid.neighbor_count)
    idx = np.asarray(grid.neighbor_idx)

    cells = range(0, n, max(1, int(sample_stride)))

    for c in cells:
        nbs = [int(v) for v in idx[c]]
        if len(set(nbs)) != k:
            out.append(Violation("grid_reference", f"cell {c} har {len(set(nbs))} distinkta grannar, förväntat {k}"))
            break
        if c in nbs:
            out.append(Violation("grid_reference", f"cell {c} är sin egen granne"))
            break
        for nb in nbs:
            if c not in [int(v) for v in idx[nb]]:
                out.append(Violation("grid_reference", f"grannrelation {c}->{nb} är inte ömsesidig"))
                break
        else:
            continue
        break

    cx = np.asarray(grid.cell_center_x, dtype=np.float64)
    cy = np.asarray(grid.cell_center_y, dtype=np.float64)
    back = np.asarray(grid.cell_of_many(cx, cy), dtype=np.int64)
    bad = np.flatnonzero(back != np.arange(n))
    if bad.size:
        out.append(Violation("grid_reference", f"cell_of(cellcentrum) fel för {bad.size} celler, t.ex. {bad[:MAX_EXAMPLES].tolist()}"))

    src = n // 3
    dist = np.full(n, -1, dtype=np.int32)
    dist[src] = 0
    q = deque([src])
    while q:
        c = q.popleft()
        for nb in idx[c]:
            nb = int(nb)
            if dist[nb] < 0:
                dist[nb] = dist[c] + 1
                q.append(nb)

    if int(dist.min()) < 0:
        out.append(Violation("grid_reference", "grafen är inte sammanhängande"))

    for t in range(0, n, max(1, n // 200)):
        if int(grid.distance(src, t)) != int(dist[t]):
            out.append(Violation(
                "grid_reference",
                f"distance({src},{t})={int(grid.distance(src, t))} men grafavstånd {int(dist[t])}",
            ))
            break

    for r in range(4):
        want = set(np.flatnonzero(dist <= r).tolist())
        got = set(int(v) for v in grid.cells_within(src, r))
        if want != got:
            out.append(Violation("grid_reference", f"cells_within(r={r}) ger {len(got)} celler, avstånd ger {len(want)}"))
            break

    return out


def diagnostics(pop) -> dict[str, Any]:
    """
    Storheter som är intressanta att följa men som ännu inte har en sluten
    balans att prövas mot. När näringskretsloppet är på plats kan flera av
    dessa flyttas upp till hårda invarianter.
    """
    store = pop.store
    slots = _live_slots(store)

    if slots.size:
        is_flora = store.kind[slots] == 1
        flora_slots = slots[is_flora]
        fauna_slots = slots[~is_flora]
        flora_mass = float(np.sum(store.mass[flora_slots], dtype=np.float64))
        flora_n = int(flora_slots.size)
        fauna_store_n = int(fauna_slots.size)
    else:
        flora_mass = 0.0
        flora_n = 0
        fauna_store_n = 0

    fauna_mass = float(sum(float(a.body.M) for a in pop.agents if a.body.alive))
    fauna_energy = float(sum(float(a.body.E_total()) for a in pop.agents if a.body.alive))

    return {
        "t": float(pop.t),
        "fauna_n": int(len([a for a in pop.agents if a.body.alive])),
        "fauna_store_n": fauna_store_n,
        "flora_n": flora_n,
        "free_slots": int(len(store.free_slots)),
        "capacity": int(store.capacity),
        "flora_mass_kg": flora_mass,
        "fauna_mass_kg": fauna_mass,
        "fauna_energy_J": fauna_energy,
        "detritus_mass_kg": float(np.sum(np.asarray(pop.world.detritus), dtype=np.float64)),
    }
