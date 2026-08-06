"""
ViewFrame — det simuleringen visar, skilt från det simuleringen är.

Viewern läste tidigare `Population` direkt: den sonderade `store`-arrayer,
plockade attribut ur `Agent` och `Body` med getattr-kedjor och kände till
hur floran låg i slotarna. Det gjorde presentationen till en andra läsare
av kärnans interna form, och varje ändring i store:n riskerade att tyst
ändra bilden.

`ViewFrame` är snittet däremellan. Den innehåller bara det som ska ritas,
i den form det ska ritas, och ingenting som pekar tillbaka in i
simuleringen. Tre saker följer av det:

  * viewern går att köra mot en inspelad eller mottagen bildruta, inte
    bara mot ett levande `Population`
  * bildrutan går att serialisera och skicka över nätet, vilket är hela
    poängen med uppdelningen
  * flera viewers kan läsa samma bildruta utan att simuleringen märker
    det, eftersom den byggs en gång per tick och inte en gång per klient

Geometrin skickas inte. `Grid` härleder allt — grannmatris, cellcentrum,
bandlatituder — ur `width` och `height` i `__post_init__`, så mottagaren
rekonstruerar den exakt ur två heltal. Det är också det enda stället där
geometrin får finnas, och det gäller i mottagaränden lika mycket som i
sändaränden.

Normeringen sker här och inte i viewern. Skalorna är modellkonstanter —
`B_K`, frömassans intervall, vuxenmassans — och om viewern bar dem hade
de funnits i två exemplar. Här finns de en gång, och en mottagare behöver
inte känna till `WorldParams` för att rita rätt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# 4: klienten förhandlar sitt synliga utsnitt med `cmd: "view"`, och servern
# packar anspråksrader bara för de cellerna. Frame-formatet är oförändrat —
# radarrayerna kunde alltid vara tomma — men förvalet är nu *inga* rader tills
# en klient bett om dem, så en klient från version 3 skulle tappa floran tyst.
# Handskakningen avvisar den i stället.
# 5: bildrutan bär terrängen och vattnet. `elevation01` är normerad över
# världens eget höjdspann med spannet som skalärer, och `wet_kind` klassar varje
# cell som land, fåra, sjö eller hav. Klassningen görs i världen och inte i
# viewern: trösklarna mellan de fyra hör till modellen, och en mottagare som
# räknade om dem ur djupet skulle rita en annan värld än den som simuleras.
# 6: höjden skickas en gång vid handskakningen i stället för i varje bildruta.
# Den är statisk, och att sända om den femton gånger i sekunden var två
# tredjedelar av det terrängen kostade på tråden. `wet_kind` och `soil01`
# stannar i bildrutan — sjönivåer och fåror ändras med årstiden.
PROTOCOL_VERSION = 6

# Vattnets topologi per cell. Ordningen är bindande — den är värdet i
# `ViewFrame.wet_kind`.
WET_LAND = 0
WET_CHANNEL = 1
WET_LAKE = 2
WET_SEA = 3

# Traitaxlarna floran kan färgkodas på. Ordningen är bindande: den är
# kolumnordningen i `claim_trait` och den ordning tangenten T stegar i.
FLORA_TRAITS: tuple[str, ...] = ("temp_opt", "dispersal", "adult_mass", "growth")

# Normeringsintervall per traitaxel. `adult_mass` skalas med B_K och
# hanteras därför separat i `frame_from_pop`.
_TRAIT_RANGE = {
    "temp_opt": (-5.0, 35.0),
    "dispersal": (0.0002, 0.0200),
    "growth": (0.005, 0.050),
}

TEMP_RANGE = (-10.0, 40.0)


def _norm(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


@dataclass
class ViewFrame:
    """En bildruta. Rena arrayer, inga referenser in i simuleringen."""

    protocol: int = PROTOCOL_VERSION

    # --- geometri: allt annat härleds ur dessa av mottagarens Grid ---
    grid_width: int = 0
    grid_height: int = 0

    # --- tid och räknare ---
    tick: int = 0
    t: float = 0.0
    births_total: int = 0
    deaths_total: int = 0

    # --- körningens läge ---
    #
    # Bildrutan bär det, inte en sidokanal, så att en klient som ansluter
    # mitt i en paus ser rätt tillstånd på sin allra första bild. Servern
    # stämplar fälten vid packning.
    paused: bool = False
    paused_by: str = ""
    control_enabled: bool = False

    # --- cellindexerade fält, längd n_cells ---
    # Normerade till [0, 1] här, inte i viewern: skalorna B_K och C_K är
    # modellkonstanter, och en mottagare ska inte behöva WorldParams för att
    # rita rätt. Temperaturen skickas rå eftersom HUD:en visar den i grader.
    detritus01: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    flora_shoot01: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    temperature: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    claimed: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    # --- terräng och vatten ---
    #
    # `elevation01` är normerad över världens eget höjdspann, som följer med som
    # `elev_min` och `elev_max` så att HUD:en kan visa råa tal och viewern kan
    # placera havsnivån. Att normera här och inte i viewern följer samma regel
    # som B_K och C_K: en mottagare ska inte behöva WorldParams för att rita
    # rätt.
    #
    # `wet_kind` är klassificeringen, inte djupet. Gränsen mellan fåra och
    # sluttning är en modellutsaga — se hydro.derive_water — och den ska inte
    # räknas om av en mottagare som gissar en tröskel. Ett byte per cell.
    #
    # `soil01` är markfukten som andel av fältkapacitet, alltså det tal florans
    # tillväxt faktiskt läser.
    # `elevation01`, `elev_min` och `elev_max` är statiska och skickas inte i
    # bildrutan — se `WorldStatic` och `_WIRE_STATIC`. De finns kvar som fält
    # eftersom lokala renderingar bygger rutan direkt ur populationen och då
    # ska ha allt på ett ställe; det är bara tråden som delar upp dem.
    elevation01: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    wet_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    soil01: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    has_terrain: bool = False
    elev_min: float = 0.0
    elev_max: float = 0.0

    # --- ytfördelningen, CSR över celler ---
    #
    # `claim_starts` har längd n_cells + 1 och pekar in i radarrayerna, som
    # är sorterade efter cell. Raderna i cell c är [starts[c], starts[c+1]).
    # Summan av `claim_share` över en cell är den upptagna andelen; resten
    # upp till 1 är bar mark. En planta med rotarea över 1 når in i sina sex
    # grannar och har därför en rad per berörd cell.
    claim_starts: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))
    claim_share: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    claim_fill: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    claim_trait: np.ndarray = field(default_factory=lambda: np.zeros((0, len(FLORA_TRAITS)), np.float32))

    # Riktningen från radens cell till plantans egen cell, som en andel av
    # ett varv. -1 för rader i plantans egen cell. Viewern använder den för
    # att rikta grannkilarna in mot moderplantan, så att en planta som täcker
    # flera celler går att se som en planta.
    claim_dir: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    # --- fauna, en post per levande djur ---
    fauna_x: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fauna_y: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fauna_heading: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fauna_energy_frac: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fauna_damage_frac: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fauna_mass_frac: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fauna_predation: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fauna_gest_frac: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    fauna_ready: np.ndarray = field(default_factory=lambda: np.zeros(0, np.bool_))

    # --- HUD ---
    T_band: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    band_lat: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    flora_n: int = 0
    flora_mass: float = 0.0
    flora_summary: dict[str, float] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.claim_starts.shape[0]) - 1

    @property
    def fauna_n(self) -> int:
        return int(self.fauna_x.shape[0])


def _terrain_fields(world, n_cells: int):
    """
    Höjd, vattentopologi och markfukt, färdiga att ritas.

    En platt värld har ingen av delarna, och då skickas tomma arrayer i stället
    för nollfyllda: `has_terrain` säger åt viewern att inte erbjuda lägena, och
    bildrutan växer inte för en värld som inte har terräng.
    """
    dr = getattr(world, "drainage", None)
    if dr is None:
        z = np.zeros(0, np.float32)
        return z, np.zeros(0, np.uint8), z, 0.0, 0.0, False

    elev = np.asarray(world.elevation, dtype=np.float64)
    lo = float(elev.min())
    hi = float(elev.max())
    elev01 = ((elev - lo) / max(hi - lo, 1e-12)).astype(np.float32, copy=False)

    # Ordningen är bindande: hav slår sjö slår fåra. En sjöcell vid kusten är
    # sjö, inte hav, och en fåra som mynnar i en sjö är sjö där den gör det.
    wet = np.full(n_cells, WET_LAND, dtype=np.uint8)
    water = np.asarray(world.water, dtype=np.float64)
    thr = float(getattr(world.WP, "submerged_threshold", 1e-6))
    wet[(water > thr)] = WET_CHANNEL
    wet[(np.asarray(dr.lake_id) >= 0) & (water > thr)] = WET_LAKE
    wet[np.asarray(dr.sea)] = WET_SEA

    cap = max(1e-12, float(getattr(world.WP, "soil_capacity", 1.0)))
    soil01 = np.clip(np.asarray(world.soil_water, dtype=np.float64) / cap,
                     0.0, 1.0).astype(np.float32, copy=False)

    return elev01, wet, soil01, lo, hi, True


def _flora_slots(store) -> np.ndarray:
    n = int(store.n)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(store.alive[:n] & (store.kind[:n] == 1)).astype(np.int64)


def _claim_table(pop, keep_cells=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ytfördelningen sorterad efter cell, som CSR.

    `keep_cells` begränsar raderna till de cellerna. Tabellen är en rad per
    planta och berörd cell och dominerar bildrutan fullständigt när floran är
    stor: vid 818 128 plantor i en värld på 65 536 celler är den 24 MB av
    rutans 25,6, medan de cellindexerade fälten är 1,57 oavsett bestånd.

    Raderna går inte heller att se i det läget. En cell är då omkring fyra
    pixlar och innehåller 12,5 plantor, så servern packade tolv kilar per cell
    för att viewern skulle rita dem i fyra pixlar. `None` betyder inga rader
    alls — inte alla — eftersom det är det billiga förvalet och klienten säger
    till när den zoomat in tillräckligt för att kilarna ska synas.

    Anspråkstabellen skrivs av upptagspasset i den ordning plantorna råkar
    ligga, med grannraderna sist. Viewern ritar per cell och vill ha den
    grupperad. Sorteringen är presentation och görs därför här, inte i
    passet — annars betalar simuleringen för en ordning bara bilden
    behöver.
    """
    store = pop.store
    n_cells = int(pop.grid.n_cells)

    if keep_cells is None:
        slots = np.zeros(0, dtype=np.int64)
        cells = np.zeros(0, dtype=np.int64)
        share = np.zeros(0, dtype=np.float64)
    else:
        slots = np.asarray(store.flora_claim_slot, dtype=np.int64)
        cells = np.asarray(store.flora_claim_cell, dtype=np.int64)
        share = np.asarray(store.flora_claim_share, dtype=np.float64)
        keep = np.asarray(keep_cells, dtype=np.int64)
        if keep.size and slots.size:
            mask = np.zeros(n_cells, dtype=bool)
            mask[keep[(keep >= 0) & (keep < n_cells)]] = True
            sel = mask[np.clip(cells, 0, n_cells - 1)] & (cells >= 0) & (cells < n_cells)
            slots, cells, share = slots[sel], cells[sel], share[sel]
        else:
            slots = np.zeros(0, dtype=np.int64)
            cells = np.zeros(0, dtype=np.int64)
            share = np.zeros(0, dtype=np.float64)

    if slots.size == 0:
        return (
            np.zeros(n_cells + 1, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, len(FLORA_TRAITS)), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    # Raderna kan peka på en slot som hunnit dö efter upptagspasset —
    # betningen och dispersionen ligger senare i ticken. En död planta ska
    # inte hålla mark i bilden.
    ok = (slots >= 0) & (slots < int(store.n))
    ok[ok] &= store.alive[slots[ok]] & (store.kind[slots[ok]] == 1)
    ok &= (cells >= 0) & (cells < n_cells)
    if not np.all(ok):
        slots, cells, share = slots[ok], cells[ok], share[ok]

    order = np.argsort(cells, kind="stable")
    slots, cells, share = slots[order], cells[order], share[order]

    counts = np.bincount(cells, minlength=n_cells)[:n_cells]
    starts = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    m = np.asarray(store.mass[slots], dtype=np.float64)
    m_adult = np.maximum(1e-12, np.asarray(store.flora_adult_mass[slots], dtype=np.float64))
    fill = np.clip(m / m_adult, 0.0, 1.0)

    BK = float(pop.WP.B_K)
    trait = np.empty((slots.size, len(FLORA_TRAITS)), dtype=np.float32)
    for j, name in enumerate(FLORA_TRAITS):
        if name == "temp_opt":
            v = _norm(store.flora_temp_opt[slots], *_TRAIT_RANGE["temp_opt"])
        elif name == "dispersal":
            v = _norm(store.flora_seed_mass[slots], *_TRAIT_RANGE["dispersal"])
        elif name == "adult_mass":
            v = _norm(store.flora_adult_mass[slots], 0.25 * BK, 4.0 * BK)
        else:
            v = _norm(store.flora_repro_alloc[slots], *_TRAIT_RANGE["growth"])
        trait[:, j] = v.astype(np.float32, copy=False)

    # Riktning mot moderplantans cell. Grannraderna uppstår när en plantas
    # rotarea överstiger 1 och spiller ut i de sex grannarna; för dem pekar
    # riktningen tillbaka mot plantans egen cell. Avståndet tas kortaste
    # vägen, eftersom världen wrappar.
    grid = pop.grid
    home = np.asarray(store.cell_idx[slots], dtype=np.int64)
    cx = np.asarray(grid.cell_center_x, dtype=np.float64)
    cy = np.asarray(grid.cell_center_y, dtype=np.float64)
    ex, ey = float(grid.extent_x), float(grid.extent_y)
    dx = (cx[home] - cx[cells] + 0.5 * ex) % ex - 0.5 * ex
    dy = (cy[home] - cy[cells] + 0.5 * ey) % ey - 0.5 * ey
    direction = (np.arctan2(dy, dx) / (2.0 * np.pi)) % 1.0
    direction[home == cells] = -1.0

    return (
        starts,
        share.astype(np.float32, copy=False),
        fill.astype(np.float32, copy=False),
        trait,
        direction.astype(np.float32, copy=False),
    )


def _fauna_table(pop) -> dict[str, np.ndarray]:
    """
    Faunan som arrayer.

    Detta är enda stället som fortfarande sonderar `Agent` och `Body` med
    getattr. Det är avsiktligt: fauna är halvmigrerad, `Body` är alltjämt
    source of truth för fysiologin, och sonderingen hör hemma här i
    översättningen snarare än utspridd i viewerns ritkod. När Steg 5 flyttat
    tillståndet till store-arrayer blir funktionen en ren slicing.
    """
    agents = [a for a in getattr(pop, "agents", []) or [] if getattr(getattr(a, "body", None), "alive", True)]
    n = len(agents)
    out = {
        k: np.zeros(n, dtype=np.float32)
        for k in ("x", "y", "heading", "energy_frac", "damage_frac", "mass_frac", "predation", "gest_frac")
    }
    out["ready"] = np.zeros(n, dtype=np.bool_)

    AP = getattr(pop, "AP", None)
    D_max = float(getattr(AP, "D_max", 1.0) or 1.0)
    M0 = float(getattr(AP, "M0", 1.0) or 1.0)

    for i, a in enumerate(agents):
        b = getattr(a, "body", None)
        out["x"][i] = float(getattr(b, "x", getattr(a, "x", 0.0)))
        out["y"][i] = float(getattr(b, "y", getattr(a, "y", 0.0)))
        out["heading"][i] = float(getattr(b, "heading", getattr(a, "heading", 0.0)))

        if b is not None:
            try:
                cap = float(b.E_cap())
                out["energy_frac"][i] = float(b.E_total()) / max(cap, 1e-12)
            except Exception:
                pass
            out["damage_frac"][i] = min(1.0, float(getattr(b, "D", 0.0)) / max(D_max, 1e-12))
            out["mass_frac"][i] = float(getattr(b, "M", 0.0)) / max(M0, 1e-12)
            if bool(getattr(b, "gestating", False)):
                tgt = max(1e-12, float(getattr(b, "gest_M_target", 1e-9)))
                out["gest_frac"][i] = min(1.0, float(getattr(b, "gest_M", 0.0)) / tgt)

        out["predation"][i] = float(getattr(getattr(a, "pheno", None), "predation", 0.0))

        slot = int(getattr(a, "store_slot", -1))
        if slot >= 0 and hasattr(pop, "_ready_to_reproduce_slot"):
            try:
                out["ready"][i] = bool(pop._ready_to_reproduce_slot(slot))
            except Exception:
                pass

    return out


def frame_from_pop(pop, births_total: int = 0, deaths_total: int = 0,
                   claim_cells=None) -> ViewFrame:
    """
    Bygg en bildruta ur ett levande `Population`.

    Kostnaden är en sortering av anspråkstabellen plus några kopior. Den
    betalas en gång per tick oavsett hur många viewers som tittar.

    `claim_cells` är de celler någon viewer zoomat in tillräckligt på för att
    plantornas kilar ska vara upplösbara. `None` ger inga anspråksrader, vilket
    är förvalet: cellfälten räcker för att rita floran som täckningsgrad, och
    det är ändå allt en cell på fyra pixlar kan visa.
    """
    store = pop.store
    world = pop.world
    grid = pop.grid
    n_cells = int(grid.n_cells)

    BK = max(1e-12, float(getattr(pop.WP, "B_K", 1.0)))
    CK = max(1e-12, float(getattr(pop.WP, "C_K", getattr(world, "C_K", 1.0)) or 1.0))

    starts, share, fill, trait, cdir = _claim_table(pop, claim_cells)
    fa = _fauna_table(pop)

    fl = _flora_slots(store)
    flora_mass = float(np.sum(store.mass[fl], dtype=np.float64)) if fl.size else 0.0

    summary: dict[str, float] = {}
    if hasattr(pop, "_flora_summary"):
        try:
            summary = {k: float(v) for k, v in pop._flora_summary().items()}
        except Exception:
            summary = {}

    if hasattr(world, "temperature_field"):
        temp = np.asarray(world.temperature_field(), dtype=np.float32)
    else:
        temp = np.zeros(n_cells, dtype=np.float32)

    elev01, wet, soil01, e_lo, e_hi, has_terr = _terrain_fields(world, n_cells)

    return ViewFrame(
        grid_width=int(grid.width),
        grid_height=int(grid.height),
        tick=int(getattr(pop, "tick", 0)),
        t=float(getattr(pop, "t", getattr(world, "t", 0.0)) or 0.0),
        births_total=int(births_total),
        deaths_total=int(deaths_total),
        detritus01=np.sqrt(_norm(world.detritus, 0.0, CK)).astype(np.float32),
        flora_shoot01=_norm(store.flora_cell_mass, 0.0, BK).astype(np.float32),
        temperature=temp.copy(),
        claimed=np.asarray(store.flora_cell_claimed, dtype=np.float32).copy(),
        elevation01=elev01,
        wet_kind=wet,
        soil01=soil01,
        has_terrain=has_terr,
        elev_min=e_lo,
        elev_max=e_hi,
        claim_starts=starts,
        claim_share=share,
        claim_fill=fill,
        claim_trait=trait,
        claim_dir=cdir,
        fauna_x=fa["x"],
        fauna_y=fa["y"],
        fauna_heading=fa["heading"],
        fauna_energy_frac=fa["energy_frac"],
        fauna_damage_frac=fa["damage_frac"],
        fauna_mass_frac=fa["mass_frac"],
        fauna_predation=fa["predation"],
        fauna_gest_frac=fa["gest_frac"],
        fauna_ready=fa["ready"],
        T_band=np.asarray(getattr(world, "T_band", np.zeros(0)), dtype=np.float32).copy(),
        band_lat=np.asarray(getattr(grid, "band_lat", np.zeros(0)), dtype=np.float32).copy(),
        flora_n=int(fl.size),
        flora_mass=flora_mass,
        flora_summary=summary,
    )


# ---------------------------------------------------------------------------
# Trådformat
# ---------------------------------------------------------------------------
#
# En bildruta på nätet är en JSON-header följd av råa buffertar. Ingen ny
# dependency: klienten ska klara sig på numpy och pygame, och varje extra
# paket är ett hinder till vid uppsättningen.
#
# Formatet är avsiktligt förlustbehäftat för de fält som bara ska ritas.
# Viewern normerar ändå allt till [0, 1] och landar i en 8-bitars färgkanal,
# så uint8 över ett känt intervall är visuellt likvärdigt med float32 och
# fjärdedelar storleken. En bildruta som packas och packas upp är därför
# inte bitidentisk med originalet — den är identisk på skärmen. Fält som
# HUD:en visar som tal (T_band, skalärer) skickas orörda.

import json
import struct
import zlib

_MAGIC = b"NEPV"
_MAGIC_STATIC = b"NEPS"


@dataclass
class WorldStatic:
    """
    Det som inte ändras under en körning. Skickas en gång per klient, direkt
    efter handskakningen och före första bildrutan.

    Höjden är världens enda genuint statiska cellfält, och den är samtidigt det
    dyraste: uppmätt två tredjedelar av vad terrängen lade på tråden, sänt om
    femton gånger i sekunden utan att ändras. Att sända den en gång är inte en
    optimering utan samma princip som kadensmodellen: ett fält som inte ändras
    ska inte röras varje tick.

    Vattnet hör *inte* hit. Sjöar fylls och töms med årstiden och fåror torkar
    ut, så `wet_kind` och `soil01` stannar i bildrutan.
    """

    protocol: int = PROTOCOL_VERSION
    grid_width: int = 0
    grid_height: int = 0
    has_terrain: bool = False
    elev_min: float = 0.0
    elev_max: float = 0.0
    elevation01: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))


# Fält i ViewFrame som bärs av WorldStatic i stället och därför hoppas över vid
# packning. Mottagaren fyller dem ur den statiska posten.
_WIRE_STATIC: frozenset[str] = frozenset({"elevation01", "elev_min", "elev_max"})


def static_from_frame(frame: ViewFrame) -> WorldStatic:
    """Plocka ut den statiska delen ur en färdig bildruta."""
    return WorldStatic(
        grid_width=int(frame.grid_width),
        grid_height=int(frame.grid_height),
        has_terrain=bool(frame.has_terrain),
        elev_min=float(frame.elev_min),
        elev_max=float(frame.elev_max),
        elevation01=np.asarray(frame.elevation01, dtype=np.float32).copy(),
    )


def apply_static(frame: ViewFrame, static: "WorldStatic | None") -> ViewFrame:
    """
    Fyll bildrutans statiska fält ur den post som kom vid handskakningen.

    Utan en statisk post lämnas fälten tomma, och viewern ritar då utan
    terräng i stället för att krascha — en klient ska inte behöva veta om
    servern körde en platt värld eller om posten uteblev.
    """
    if static is None:
        return frame
    frame.elevation01 = np.asarray(static.elevation01, dtype=np.float32)
    frame.elev_min = float(static.elev_min)
    frame.elev_max = float(static.elev_max)
    if not bool(static.has_terrain):
        frame.has_terrain = False
    return frame

# Fält som kvantiseras till uint8, med det intervall de normeras över.
_WIRE_U8: dict[str, tuple[float, float]] = {
    "detritus01": (0.0, 1.0),
    "flora_shoot01": (0.0, 1.0),
    "claim_share": (0.0, 1.0),
    "claim_fill": (0.0, 1.0),
    "claim_trait": (0.0, 1.0),
    # Anspråket kan överstiga 1 — det är hela poängen med det. Taket 4 rymmer
    # den uppmätta spridningen med marginal; över det är cellen ändå mättad.
    "claimed": (0.0, 4.0),
    # -1 betyder "egen cell". Intervallet rymmer sentinelvärdet och ger
    # riktningen 1/128 varv, alltså knappt 3 grader — gott om upplösning
    # för sex grannriktningar.
    "claim_dir": (-1.0, 1.0),
    "temperature": TEMP_RANGE,
    "elevation01": (0.0, 1.0),
    "soil01": (0.0, 1.0),
}

# Fält som byter till en smalare heltalstyp utan att tappa något.
_WIRE_CAST: dict[str, str] = {"claim_starts": "int32"}


def pack(frame: ViewFrame, compress: bool = True, *, magic: bytes = _MAGIC,
         skip: frozenset[str] = _WIRE_STATIC) -> bytes:
    """
    Serialisera en bildruta eller en statisk post. Returnerar ett komplett
    ramat meddelande.

    Rutinen itererar över dataklassens fält och är därmed oberoende av vilken
    klass den får. `magic` skiljer de två meddelandetyperna åt på tråden, och
    `skip` utelämnar de fält som bärs av den andra.
    """
    from dataclasses import fields as _fields

    scalars: dict[str, object] = {}
    arrays: list[dict] = []
    bufs: list[bytes] = []

    for f in _fields(frame):
        if f.name in skip:
            continue
        v = getattr(frame, f.name)
        if isinstance(v, np.ndarray):
            q = None
            if f.name in _WIRE_U8:
                lo, hi = _WIRE_U8[f.name]
                a = np.clip((v.astype(np.float64) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
                # Ett positivt värde får aldrig rundas till noll: nollan
                # betyder frånvaro i ritkoden, och en groddplanta som
                # försvinner ur bilden är inte avrundning utan en utplåning.
                a = np.where(v > 0, np.maximum(1.0, np.rint(a * 255.0)), np.rint(a * 255.0))
                a = a.astype(np.uint8)
                q = [float(lo), float(hi)]
            elif f.name in _WIRE_CAST:
                a = v.astype(_WIRE_CAST[f.name], copy=False)
            else:
                a = np.ascontiguousarray(v)
            arrays.append({"n": f.name, "d": a.dtype.str, "s": list(a.shape), "q": q})
            bufs.append(np.ascontiguousarray(a).tobytes())
        elif isinstance(v, dict):
            scalars[f.name] = {str(k): float(x) for k, x in v.items()}
        else:
            scalars[f.name] = v

    header = json.dumps(
        {"protocol": PROTOCOL_VERSION, "scalars": scalars, "arrays": arrays},
        separators=(",", ":"),
    ).encode("utf-8")

    payload = struct.pack("<I", len(header)) + header + b"".join(bufs)
    flags = 0
    if compress:
        payload = zlib.compress(payload, 1)
        flags = 1
    return magic + struct.pack("<BI", flags, len(payload)) + payload


def pack_static(static: WorldStatic, compress: bool = True) -> bytes:
    """Serialisera den statiska posten."""
    return pack(static, compress, magic=_MAGIC_STATIC, skip=frozenset())


HEADER_SIZE = len(_MAGIC) + 5


def message_kind(head: bytes) -> str:
    """"frame" eller "static" ur meddelandets första byten."""
    if len(head) < HEADER_SIZE:
        raise ValueError("för kort huvud")
    if head[:4] == _MAGIC:
        return "frame"
    if head[:4] == _MAGIC_STATIC:
        return "static"
    raise ValueError("okänt magiskt tal på tråden")


def payload_size(head: bytes) -> int:
    """Läs nyttolastens längd ur de första HEADER_SIZE byten."""
    message_kind(head)
    _flags, n = struct.unpack("<BI", head[4:HEADER_SIZE])
    return int(n)


def unpack_static(head: bytes, payload: bytes) -> WorldStatic:
    """Återskapa den statiska posten."""
    return unpack(head, payload, cls=WorldStatic)


def unpack(head: bytes, payload: bytes, cls=ViewFrame):
    """Återskapa en bildruta eller statisk post ur ram och nyttolast."""
    flags, _n = struct.unpack("<BI", head[4:HEADER_SIZE])
    if flags & 1:
        payload = zlib.decompress(payload)

    (hlen,) = struct.unpack("<I", payload[:4])
    header = json.loads(payload[4 : 4 + hlen].decode("utf-8"))

    got = int(header.get("protocol", -1))
    if got != PROTOCOL_VERSION:
        raise ValueError(
            f"protokollversion {got} från servern, klienten talar "
            f"{PROTOCOL_VERSION} — kör samma commit i båda ändar"
        )

    frame = cls(**header["scalars"])
    off = 4 + hlen
    for spec in header["arrays"]:
        dt = np.dtype(spec["d"])
        shape = tuple(spec["s"])
        n = int(np.prod(shape)) if shape else 0
        raw = np.frombuffer(payload, dtype=dt, count=n, offset=off).reshape(shape)
        off += n * dt.itemsize
        q = spec["q"]
        if q is not None:
            lo, hi = float(q[0]), float(q[1])
            raw = (raw.astype(np.float32) / np.float32(255.0)) * np.float32(hi - lo) + np.float32(lo)
        setattr(frame, spec["n"], np.array(raw, copy=True))
    return frame
