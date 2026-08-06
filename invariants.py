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

import math
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
    """Glest CSR-index ska innehålla exakt de levande, placerade slotarna."""
    store = pop.store
    out: list[Violation] = []

    cells = np.asarray(store.idx_cells)
    starts = np.asarray(store.idx_starts)

    if starts.size != cells.size + 1:
        out.append(Violation(
            "spatial_index",
            f"idx_starts har {starts.size} element, förväntat {cells.size + 1}",
        ))
        return out

    if int(starts[0]) != 0:
        out.append(Violation("spatial_index", f"idx_starts[0]={int(starts[0])}, förväntat 0"))

    if cells.size and not np.all(np.diff(cells) > 0):
        out.append(Violation("spatial_index", "idx_cells är inte strikt stigande"))
        return out

    if starts.size > 1 and not np.all(np.diff(starts) > 0):
        out.append(Violation("spatial_index", "idx_starts är inte strikt stigande"))
        return out

    total = int(starts[-1])
    live = _live_slots(store)
    live_placed = int(np.count_nonzero(store.cell_idx[live] >= 0)) if live.size else 0
    if total != live_placed:
        out.append(Violation(
            "spatial_index",
            f"indexet rymmer {total} slotar, {live_placed} levande med giltig cell",
        ))

    packed = np.asarray(store.cell_slots)[:total]
    if packed.size:
        if not np.all(store.alive[packed]):
            dead = packed[~store.alive[packed]]
            out.append(Violation("spatial_index", f"döda slotar i indexet: {dead[:MAX_EXAMPLES].tolist()}"))

        counts = np.diff(starts)
        cell_of_packed = np.repeat(cells, counts)
        mismatch = packed[store.cell_idx[packed] != cell_of_packed]
        if mismatch.size:
            out.append(Violation(
                "spatial_index",
                f"slotar i fel cellbucket: {mismatch[:MAX_EXAMPLES].tolist()}",
            ))

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


# Fält som ska vara platta per-cell-arrayer.
WORLD_CELL_FIELDS = ("water", "nutrient", "detritus", "carcass")

# Fält som är rumsligt konstanta och därför lagras som skalär. De promoveras
# till arrayer först när något faktiskt varierar dem — se
# docs/varldens-kadensmodell.md.
WORLD_UNIFORM_FIELDS = (
    "elevation", "rain_input", "spring_input", "infiltration", "evaporation",
    "flow_strength",
)

# Härledda fält beräknas vid läsning och ska ha per-cell-form när de läses.
WORLD_DERIVED_FIELDS = ("surface_level", "submerged")


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
    for name in WORLD_CELL_FIELDS + WORLD_DERIVED_FIELDS:
        arr = getattr(world, name, None)
        if arr is None:
            bad.append(f"{name} saknas")
            continue
        arr = np.asarray(arr)
        if arr.ndim != 1:
            bad.append(f"{name} har form {arr.shape}, förväntat 1D")
        elif int(arr.shape[0]) != n_cells:
            bad.append(f"{name} har längd {int(arr.shape[0])}, förväntat n_cells={n_cells}")

    for name in WORLD_UNIFORM_FIELDS:
        val = getattr(world, name, None)
        if val is None:
            bad.append(f"{name} saknas")
            continue
        arr = np.asarray(val)
        if arr.ndim == 0:
            continue
        # Promoverat till fält: då gäller per-cell-form.
        if arr.ndim != 1 or int(arr.shape[0]) != n_cells:
            bad.append(f"{name} är varken skalär eller per-cell, form {arr.shape}")

    if bad:
        out.append(Violation("world_field_domains", "; ".join(bad[:MAX_EXAMPLES])))

    return out


def check_sparse_fields(pop) -> list[Violation]:
    """
    Glesa fälts kontrakt: en cell som inte är med i den aktiva mängden är exakt
    noll.

    Utan den här kontrollen är glesheten en optimering som kan vara fel — ett
    pass som råkat skriva utanför den aktiva mängden ger ett värde som aldrig
    bryts ner och aldrig syns. Med kontrollen är den en prövbar egenskap.
    """
    world = pop.world
    out: list[Violation] = []

    if hasattr(world, "_detritus_flush"):
        world._detritus_flush()
    member = getattr(world, "_detritus_member", None)
    active = getattr(world, "_detritus_active", None)
    if member is None or active is None:
        return out

    detritus = np.asarray(world.detritus)
    member = np.asarray(member)

    struct = np.asarray(getattr(world, "detritus_structure", np.zeros(0)))
    if struct.size == detritus.size:
        stray_s = np.flatnonzero((~member) & (struct != 0.0))
        if stray_s.size:
            out.append(Violation(
                "sparse_fields",
                f"{stray_s.size} inaktiva celler har nollskild detritus_structure",
            ))
        bad_range = np.flatnonzero((struct < 0.0) | (struct > 1.0))
        if bad_range.size:
            out.append(Violation(
                "sparse_fields",
                f"detritus_structure utanför [0,1] i {bad_range.size} celler",
            ))

    stray = np.flatnonzero((~member) & (detritus != 0.0))
    if stray.size:
        out.append(Violation(
            "sparse_fields",
            f"{stray.size} inaktiva celler har nollskilt detritus, t.ex. "
            f"{[(int(c), float(detritus[c])) for c in stray[:MAX_EXAMPLES]]}",
        ))

    act = np.asarray(active, dtype=np.int64)
    if act.size != int(member.sum()):
        out.append(Violation("sparse_fields", f"aktiv mängd har {act.size} poster, medlemsflaggan {int(member.sum())}"))
    elif act.size and not np.all(member[act]):
        out.append(Violation("sparse_fields", "aktiv mängd innehåller celler utan medlemsflagga"))
    elif act.size != np.unique(act).size:
        out.append(Violation("sparse_fields", "aktiv mängd innehåller dubbletter"))

    return out


def check_nutrient_balance(pop, rel_tol: float = 1e-6) -> list[Violation]:
    """
    Näringen är en sluten storhet.

    Massa kan aldrig sluta sig i den här modellen: flora bygger merparten av
    sin vävnad ur luft och faunans kol lämnar kroppen som koldioxid vid
    förbränning. Näringen kan det, och är därför den storhet som prövas.

    Summan av fri näring, flora, fauna och detritus ska vara lika med extern
    tillförsel minus nedbrytningsförlust. Toleransen är relativ mot totalen och
    satt av att detritusfälten lagras i float32; `nutrient` ligger i float64
    just för att den termen inte ska dominera.
    """
    nb = nutrient_balance(pop)
    total = float(nb["total"])
    scale = max(1e-12, abs(total))
    rel = float(nb["unaccounted"]) / scale

    if abs(rel) > float(rel_tol):
        return [Violation(
            "nutrient_balance",
            f"näringsbalansen driver {rel:.3e} relativt "
            f"(obokfört {nb['unaccounted']:.6e} kg av {total:.6e} kg; "
            f"fri {nb['free']:.3e}, flora {nb['in_flora']:.3e}, "
            f"fauna {nb['in_fauna']:.3e}, detritus {nb['in_detritus']:.3e})",
        )]
    return []


def check_flora_claims(pop) -> list[Violation]:
    """
    Ytfördelningen ska vara en giltig fördelning av cellens area.

    Cellarean är 1. En plantas andel av en cell kan aldrig vara negativ, och
    summan av andelarna i en cell kan aldrig överstiga 1 — då hade marken
    delats ut två gånger. Där anspråket når eller överstiger 1 ska summan
    dessutom vara 1: marken är slut, och det finns ingen bar mark kvar att
    fördela. Under 1 ska summan vara lika med anspråket, eftersom ingen
    konkurrens skett.

    Invarianten är billig och skyddar det som viewern kommer att rita som
    bar mark. Blir den falsk visar bilden en yta som inte finns.
    """
    store = pop.store
    out: list[Violation] = []

    slots = np.asarray(store.flora_claim_slot)
    cells = np.asarray(store.flora_claim_cell)
    share = np.asarray(store.flora_claim_share, dtype=np.float64)
    claimed = np.asarray(store.flora_cell_claimed, dtype=np.float64)

    if slots.size == 0:
        if np.any(claimed > 0.0):
            out.append(Violation(
                "flora_claims",
                f"tom anspråkstabell men {int(np.count_nonzero(claimed > 0.0))} "
                f"celler har anspråk kvar",
            ))
        return out

    if not np.all(np.isfinite(share)) or not np.all(np.isfinite(claimed)):
        out.append(Violation("flora_claims", "icke-ändliga andelar eller anspråk"))
        return out

    if np.any(share < 0.0):
        out.append(Violation(
            "flora_claims",
            f"{int(np.count_nonzero(share < 0.0))} rader har negativ andel "
            f"(minst {float(share.min()):.3e})",
        ))

    n_cells = int(claimed.shape[0])
    bad_cell = (cells < 0) | (cells >= n_cells)
    if np.any(bad_cell):
        out.append(Violation(
            "flora_claims",
            f"{int(np.count_nonzero(bad_cell))} rader pekar på cell utanför världen",
        ))
        return out

    used = np.bincount(cells.astype(np.int64), weights=share, minlength=n_cells)[:n_cells]

    # Toleransen bär float32-lagringen av andelarna. Passet räknar i float64
    # och lagrar i float32; felet växer med antalet rader i cellen.
    tol = 1e-5
    over = used > 1.0 + tol
    if np.any(over):
        out.append(Violation(
            "flora_claims",
            f"{int(np.count_nonzero(over))} celler har utdelad andel > 1 "
            f"(max {float(used.max()):.6f}) — marken delades ut två gånger",
        ))

    full = claimed >= 1.0
    if np.any(full):
        short = np.abs(used[full] - 1.0) > tol
        if np.any(short):
            out.append(Violation(
                "flora_claims",
                f"{int(np.count_nonzero(short))} översökta celler delar inte ut "
                f"hela arean (störst avvikelse {float(np.abs(used[full] - 1.0).max()):.3e})",
            ))

    open_ = ~full
    if np.any(open_):
        drift = np.abs(used[open_] - claimed[open_])
        if np.any(drift > tol):
            out.append(Violation(
                "flora_claims",
                f"{int(np.count_nonzero(drift > tol))} celler under mättnad har "
                f"utdelad andel skild från anspråket (störst {float(drift.max()):.3e})",
            ))

    bad_slot = (slots < 0) | (slots >= int(store.n))
    if np.any(bad_slot):
        out.append(Violation(
            "flora_claims",
            f"{int(np.count_nonzero(bad_slot))} rader pekar på slot utanför store:n",
        ))

    return out


def check_damage_saturation(pop) -> list[Violation]:
    """
    Ett levande djur får inte ligga kvar på skadetaket tick efter tick.

    D_max är en dödströskel, inte ett viloläge. Ligger ett djur kvar strax
    under taket över flera tick betyder det att någon gräns tillämpas i fel
    ordning i förhållande till dödstestet — precis det fel som lät en individ
    i runs/p78/s2 sitta på D = 1,00000 i 186 månader.

    Kontrollen är medvetet trög: den kräver att samma individ setts mättad
    vid flera på varandra följande anrop. Ett enstaka tick på taket är
    legitimt — reparationen kan hinna ta ner skadan innan nästa tick — medan
    en individ som fastnar där är en artefakt.
    """
    out: list[Violation] = []
    AP = getattr(pop, "AP", None)
    D_max = float(getattr(AP, "D_max", 1.0) or 1.0)
    if D_max <= 0.0:
        return out

    # Hur många anrop i rad en individ måste ha varit mättad för att räknas.
    # Sviten körs typiskt var n:te tick, så tre anrop är en lång stund.
    limit = 3
    tol = 1e-4

    seen = getattr(pop, "_damage_saturation_streak", None)
    if seen is None:
        seen = {}
        setattr(pop, "_damage_saturation_streak", seen)

    now: dict[int, int] = {}
    for a in getattr(pop, "agents", []) or []:
        body = getattr(a, "body", None)
        if body is None or not bool(getattr(body, "alive", False)):
            continue
        d = float(getattr(body, "D", 0.0))
        if d < D_max - tol:
            continue
        aid = int(getattr(a, "id", getattr(a, "agent_id", id(a))))
        now[aid] = seen.get(aid, 0) + 1

    seen.clear()
    seen.update(now)

    stuck = sorted((n, aid) for aid, n in now.items() if n >= limit)
    if stuck:
        n, aid = stuck[-1]
        out.append(Violation(
            "damage_saturation",
            f"{len(stuck)} levande djur ligger kvar på skadetaket "
            f"(D >= {D_max - tol:.6f}); värst är id {aid} med {n} kontroller i rad "
            f"— dödströskeln nås aldrig",
        ))
    return out


def check_death_cause_set(pop) -> list[Violation]:
    """
    Varje dödsfall ska ha en dödsorsak.

    Sex vägar dödar ett djur: `damage`, `starvation`, `hazard`, de tre
    vakterna, och predationen. Fem av dem ligger i `Body.step()` och sätter
    `death_cause` innan de sätter `alive = False`. Predationen ligger i
    `_step_predation()` och gjorde det inte, vilket lät den dö tyst: alla
    38 dödsfall med orsak `unknown` över p75, p77 och p78 var predation, och
    nischen beskrevs därför som död kod i tre statusanalyser.

    Kontrollen är kumulativ och inte en ögonblicksbild. En död agent tas bort
    i samma tick som den dör, så en sviten som körs var n:te tick skulle
    aldrig hinna se den.
    """
    out: list[Violation] = []
    n = int(getattr(pop, "_deaths_without_cause", 0) or 0)
    if n > 0:
        out.append(Violation(
            "death_cause_unset",
            f"{n} dödsfall utan dödsorsak — någon väg dödar utanför de sex "
            f"kända och är därmed oinstrumenterad",
        ))
    return out


def check_drainage(pop) -> list[Violation]:
    """
    Dräneringsnätets struktur.

    Nätet byggs en gång och rörs aldrig, så felen här är byggfel snarare än
    drifter — men de är tysta. Ett nät med en cykel skulle få `route()` att
    ackumulera i evighet; en cell utan väg till havet skulle samla vatten som
    aldrig lämnar systemet, vilket bryter vattenbalansen först efter lång tid.
    """
    dr = getattr(pop.world, "drainage", None)
    if dr is None:
        return []

    n = int(pop.grid.n_cells)
    out: list[Violation] = []

    if int(dr.flow_order.shape[0]) != n:
        out.append(Violation(
            "drainage",
            f"flow_order täcker {int(dr.flow_order.shape[0])} av {n} celler",
        ))
        return out

    if np.unique(dr.flow_order).size != n:
        out.append(Violation("drainage", "flow_order innehåller dubbletter"))
        return out

    # Topologin: varje cell måste komma före sin nedströmsgranne i ordningen.
    # Det är cykelfrihet uttryckt som en prövbar egenskap i stället för som ett
    # antagande om hur bygget gick till.
    pos = np.empty(n, dtype=np.int64)
    pos[dr.flow_order] = np.arange(n, dtype=np.int64)
    has_to = dr.flow_to >= 0
    if has_to.any():
        src = np.flatnonzero(has_to)
        bad = src[pos[src] >= pos[dr.flow_to[src]]]
        if bad.size:
            out.append(Violation(
                "drainage",
                f"{bad.size} celler ligger efter sin nedströmsgranne i "
                f"flow_order, t.ex. {[int(c) for c in bad[:MAX_EXAMPLES]]}",
            ))

        # Nedströmsgrannen måste vara en granne.
        nb = np.asarray(pop.grid.neighbor_idx, dtype=np.int64)
        is_nb = (nb[src] == dr.flow_to[src][:, None]).any(axis=1)
        if not is_nb.all():
            stray = src[~is_nb]
            out.append(Violation(
                "drainage",
                f"{stray.size} celler pekar på en cell som inte är granne, "
                f"t.ex. {[int(c) for c in stray[:MAX_EXAMPLES]]}",
            ))

    # Havet är ändstation och får ingen riktning nedåt.
    if bool((dr.flow_to[dr.sea] >= 0).any()):
        out.append(Violation("drainage", "havsceller har en riktning nedströms"))

    # Och omvänt: ingen landcell får vara ändstation. En sådan cell tar emot
    # avrinning som aldrig når havet, vilket är vatten som försvinner tyst.
    # Felet fanns på riktigt: när sjöns fyllda yta blev platt hittade brantaste
    # fallet ingen riktning alls, 37 av 37 sjöar blev ändstationer, och 88
    # procent av all avrinning strandade utan att synas i något annat mått än
    # vattenbalansen.
    stranded = np.flatnonzero((dr.flow_to < 0) & (~dr.sea))
    if stranded.size:
        out.append(Violation(
            "drainage",
            f"{stranded.size} landceller är ändstationer utan väg till havet, "
            f"t.ex. {[int(c) for c in stranded[:MAX_EXAMPLES]]}",
        ))

    # Sjöarnas bokföring: startindex monotona och täcker lake_cells exakt.
    ls = np.asarray(dr.lake_start, dtype=np.int64)
    if ls.size < 1 or int(ls[0]) != 0:
        out.append(Violation("drainage", "lake_start börjar inte på noll"))
    elif np.any(np.diff(ls) < 0):
        out.append(Violation("drainage", "lake_start är inte växande"))
    elif int(ls[-1]) != int(dr.lake_cells.shape[0]):
        out.append(Violation(
            "drainage",
            f"lake_start slutar på {int(ls[-1])}, lake_cells har "
            f"{int(dr.lake_cells.shape[0])} poster",
        ))
    elif dr.lake_cells.size and np.unique(dr.lake_cells).size != dr.lake_cells.size:
        out.append(Violation("drainage", "en cell tillhör två sjöar"))

    # Hypsometrin ska vara växande i volym — annars går nivåuppslaget baklänges.
    for i in range(dr.n_lakes):
        a, b = int(ls[i]), int(ls[i + 1])
        v = dr.lake_vol[a:b]
        if v.size and (float(v[0]) != 0.0 or np.any(np.diff(v) < 0.0)):
            out.append(Violation("drainage", f"sjö {i} har icke-växande hypsometri"))
            break

    return out


def check_water_balance(pop, rel_tol: float = 1e-9) -> list[Violation]:
    """
    Vattnet är en sluten storhet över en absorberande rand.

    Stocken är markvatten plus sjömagasin. Kanalvatten är i transit och lagras
    inte — det följer av att hydro löser jämvikt per tick i stället för att
    integrera en transient, och är alltså en modellutsaga och inte en
    approximation.

    Balansen är exakt av strukturella skäl: routing är en ren överföring och
    reservoaren en ren omfördelning mellan magasin och spill. Toleransen är
    därför satt vid flyttalsbrus och inte vid en modelltolerans. Faller den
    isär är det ett verkligt fel — vatten som strandar i en cell utan väg ut,
    eller ett spill som räknas två gånger.
    """
    world = pop.world
    if getattr(world, "drainage", None) is None:
        return []
    if not hasattr(world, "_water_added_total"):
        return []

    stock = float(world.water_stock())
    start = float(getattr(world, "_water_stock_init", 0.0))
    added = float(world._water_added_total)
    lost = float(world._water_lost_total)

    resid = (stock - start) - (added - lost)
    scale = max(abs(start), abs(stock), abs(added), 1e-12)
    rel = abs(resid) / scale
    if rel > rel_tol:
        return [Violation(
            "water_balance",
            f"vattenbalansen sluter inte: stock {stock:.6e} start {start:.6e} "
            f"tillfört {added:.6e} förlorat {lost:.6e} rest {resid:.3e} "
            f"({rel:.2e} relativt)",
        )]
    return []


ALL_CHECKS = (
    check_sparse_fields,
    check_nutrient_balance,
    check_water_balance,
    check_drainage,
    check_slot_bookkeeping,
    check_world_field_domains,
    check_body_store_mirror,
    check_array_domains,
    check_cell_idx,
    check_identity,
    check_finite_and_nonnegative,
    check_spatial_index,
    check_agent_store_binding,
    check_flora_claims,
    check_damage_saturation,
    check_death_cause_set,
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


def _flora_root_frac(store) -> float:
    """
    Rotens andel av florans massa.

    Kom med rotens återgång i 0124 och är sedan dess en av de mer talande
    siffrorna om betningen: den säger hur stor del av beståndet som är anspråk
    utan bladverk.
    """
    live = _live_slots(store)
    flora = live[store.kind[live] == 1] if live.size else live
    if flora.size == 0:
        return 0.0
    m = store.mass[flora].astype(np.float64)
    r = np.minimum(m, store.flora_root_mass[flora].astype(np.float64))
    tot = float(m.sum())
    return float(r.sum() / tot) if tot > 0.0 else 0.0


def nutrient_balance(pop) -> dict[str, float]:
    """
    Näringens fördelning och balans.

    Näring är den enda storhet i modellen som kan cirkulera slutet: flora
    bygger merparten av sin vävnad ur luft och fauna växer ur en energibudget,
    så total massa flödar igenom snarare än cirkulerar.

    Balansen omfattar mark, flora, detritus och fauna. Faunan kom in när
    reserven blev massa och tillväxten fick ett materialkrav: därefter kan
    kroppsmassa varken uppstå ur en energibudget eller försvinna vid
    förbränning utan att näringen bokförs.

    `Body` äger fortfarande faunans massa och reserv, så de läses därifrån och
    inte ur store:n. När Steg 6b flyttat ägarskapet kan den delen bli
    arraybaserad som resten.
    """
    from phenotype import nutrient_content, NUTRIENT_PER_KG_LABILE

    world = pop.world
    store = pop.store

    free = float(np.sum(np.asarray(world.nutrient), dtype=np.float64))

    from phenotype import NUTRIENT_PER_KG_LABILE, NUTRIENT_PER_KG_STRUCT

    def _pool_nutrient(cells, v, st) -> float:
        if cells.size == 0:
            return 0.0
        d = np.asarray(v)[cells].astype(np.float64)
        ds = np.asarray(st)[cells].astype(np.float64)
        return float(np.sum(d * (NUTRIENT_PER_KG_LABILE * (1.0 - ds)
                                 + NUTRIENT_PER_KG_STRUCT * ds)))

    # Två dödpooler sedan kadavret skildes från förnan. Missas den ena syns det
    # som en läcka i takt med dödligheten, vilket är precis den storleksordning
    # som är svårast att skilja från en verklig läcka.
    in_litter = _pool_nutrient(
        np.asarray(world.detritus_active_cells, dtype=np.int64),
        world.detritus, world.detritus_structure)
    in_carcass = _pool_nutrient(
        np.asarray(getattr(world, "carcass_active_cells", np.zeros(0, np.int64)),
                   dtype=np.int64),
        getattr(world, "carcass", np.zeros(0)),
        getattr(world, "carcass_structure", np.zeros(0)))
    in_detritus = in_litter + in_carcass

    live = _live_slots(store)
    flora = live[store.kind[live] == 1] if live.size else live
    if flora.size:
        m = store.mass[flora].astype(np.float64)
        st = store.structure[flora].astype(np.float64)
        from phenotype import NUTRIENT_PER_KG_LABILE, NUTRIENT_PER_KG_STRUCT
        in_flora = float(np.sum(m * (NUTRIENT_PER_KG_LABILE * (1.0 - st)
                                     + NUTRIENT_PER_KG_STRUCT * st)))
        # Reserven och reproduktionspoolen är näring plantan tagit upp men ännu
        # inte bundit i vävnad. Utan dem läcker balansen i takt med upptaget.
        in_flora += float(np.sum(store.flora_reserve[flora], dtype=np.float64))
        in_flora += float(np.sum(store.flora_repro_pool[flora], dtype=np.float64))
    else:
        in_flora = 0.0

    # Fauna: committad vävnad bär sin egen strukturandel, reserven och ett
    # eventuellt foster är labila.
    in_fauna = 0.0
    for a in getattr(pop, "agents", ()):
        if not a.body.alive:
            continue
        slot = int(getattr(a, "store_slot", -1))
        s = float(store.structure[slot]) if slot >= 0 else 0.25
        in_fauna += float(a.body.M) * nutrient_content(s)
        in_fauna += a.body.M_reserve() * NUTRIENT_PER_KG_LABILE
        if bool(a.body.gestating):
            in_fauna += float(a.body.gest_M) * NUTRIENT_PER_KG_LABILE

    added = float(getattr(world, "_nutrient_added_total", 0.0))
    lost = float(getattr(world, "_nutrient_lost_total", 0.0))
    total = free + in_detritus + in_flora + in_fauna

    return {
        "free": free,
        "in_flora": in_flora,
        "in_fauna": in_fauna,
        "in_detritus": in_detritus,
        "in_litter": in_litter,
        "in_carcass": in_carcass,
        "total": total,
        "added": added,
        "lost": lost,
        "unaccounted": total - (added - lost),
    }


def fauna_spacing(pop) -> dict[str, float]:
    """
    Hur tätt faunan står, mätt som avstånd till närmaste artfrände.

    Måttet finns för att skilja två fel åt som annars ser likadana ut i
    populationskurvan. Ett bestånd som diffunderar fritt sprids som roten ur
    tiden och tappar mötesfrekvens tills reproduktionen upphör; ett som håller
    ihop planar ut på ett avstånd satt av kohesionen. Båda kan sluta i noll
    djur, men bara det första är ett geometriproblem.

    Räckviddsandelen är den storhet som direkt förklarar parningsfrekvensen.
    Synellipsen med `r_front` och excentricitet `e` täcker
    `π r² (1-e)² / (1-e²)^{3/2}` areaenheter — 38 av 16 384 vid r=7 och e=0,7,
    alltså 0,23 procent. Vid jämn utspridning ser tjugo djur varandra fyra
    procent av tickarna. Ligger andelen väsentligt högre håller beståndet ihop.

    Avstånden är toroidala. Kostnaden är O(n²) men beräknas bara när
    diagnostiken efterfrågas, och n är hundratal.
    """
    ag = [a for a in getattr(pop, "agents", ()) if a.body.alive]
    n = len(ag)
    out = {"fauna_nn_mean": float("nan"), "fauna_nn_median": float("nan"),
           "fauna_in_range_frac": float("nan"), "fauna_sense_area": float("nan")}
    if n < 2:
        return out

    g = pop.grid
    xs = np.fromiter((a.x for a in ag), dtype=np.float64, count=n)
    ys = np.fromiter((a.y for a in ag), dtype=np.float64, count=n)
    ex, ey = float(g.extent_x), float(g.extent_y)

    dx = np.abs(xs[:, None] - xs[None, :])
    dy = np.abs(ys[:, None] - ys[None, :])
    np.minimum(dx, ex - dx, out=dx)
    np.minimum(dy, ey - dy, out=dy)
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)

    AP = ag[0].AP
    rf = float(AP.ray_len_front)
    e = max(0.0, min(0.999, float(AP.ray_eccentricity)))
    area = math.pi * rf * rf * (1.0 - e) ** 2 / (1.0 - e * e) ** 1.5

    out["fauna_nn_mean"] = float(np.mean(nn))
    out["fauna_nn_median"] = float(np.median(nn))
    # Andel med minst en artfrände inom den längsta synriktningen. Grov mot
    # ellipsen, men den är riktningsoberoende och därför jämförbar över tid.
    out["fauna_in_range_frac"] = float(np.mean(nn <= rf))
    # Poissonförväntan vid samma täthet. Utan den är andelen inte tolkbar:
    # den stiger med beståndets storlek även vid helt slumpmässig fördelning,
    # och 70 procent vid femtio djur är 62 vid slump. Kvoten mot förväntan är
    # det som säger om beståndet faktiskt håller ihop.
    lam = n / max(1.0, float(pop.grid.n_cells))
    out["fauna_in_range_poisson"] = float(1.0 - math.exp(-lam * math.pi * rf * rf))
    out["fauna_nn_poisson"] = float(0.5 / math.sqrt(max(lam, 1e-12)))
    out["fauna_sense_area"] = float(area)
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
        # `detritus` är förnan sedan kadavret fick egen pool i 0121. Namnet
        # står kvar för läsare som inte känner till delningen; `carcass_mass_kg`
        # är den andra poolen.
        "detritus_mass_kg": float(np.sum(np.asarray(pop.world.detritus), dtype=np.float64)),
        "carcass_mass_kg": float(np.sum(np.asarray(pop.world.carcass), dtype=np.float64)),
        "flora_root_frac": _flora_root_frac(store),
    }
