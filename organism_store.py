from __future__ import annotations

from dataclasses import dataclass, field, fields
   
from typing import Iterable

import numpy as np

from phenotype import buoyancy_from_structure, structure_fraction

# Numba är valfri. Finns den används en counting sort för CSR-bygget; annars
# faller koden tillbaka på np.argsort. Permutationerna är bitidentiska, så
# fallbacken ger samma bana — bara långsammare.
try:
    from numba import njit as _njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - beror på miljön
    _HAVE_NUMBA = False

    def _njit(*a, **k):  # type: ignore[misc]
        def deco(f):
            return f
        return deco

import itertools as _itertools


# ---------------------------------------------------------------------------
# Biologisk identitet
# ---------------------------------------------------------------------------
# Manifestet: `id` är en stabil unik biologisk identitet — ett monotont ökande
# heltal som tilldelas vid birth och aldrig återanvänds. Det gäller ALLA
# organismer, inte en klass av dem. Flora och fauna måste därför dela samma
# id-rymd; två separata räknare ger kollisioner som korrumperar id->slot-
# uppslag, härstamningsloggar och all framtida kod som refererar en individ
# över tid.
_organism_id_counter = _itertools.count(1)


# ---------------------------------------------------------------------------
# Arrayernas indexdomäner
# ---------------------------------------------------------------------------
# Store:n innehåller arrayer i tre olika indexdomäner. Bara den första växer
# med slotkapaciteten.
#
#   slotindexerade   längd = capacity   (id, alive, mass, cell_slots, ...)
#   cellindexerade   längd = n_cells    (flora_cell_mass)
#   glest cellvisa   längd = bebodda     (idx_cells, idx_starts)
#   id-indexerade    egen tillväxt      (id_to_slot_arr)
#
# Nya fält som inte är slotindexerade MÅSTE läggas till här.
_NON_SLOT_ARRAYS = frozenset({
    "id_to_slot_arr",
    "idx_cells",
    "idx_starts",
    "flora_cell_mass",
    "flora_cell_structure",
    "flora_cell_claimed",
    "flora_claim_slot",
    "flora_claim_cell",
    "flora_claim_share",
    "_flora_cells_prev",
    "_csr_cursor",
})


@_njit(cache=True, nogil=True)
def _counting_sort_order(cells, cursor):
    """
    Stabil gruppering av begränsade heltalsnycklar. O(n + n_cells).

    Ger exakt samma permutation som `np.argsort(cells, kind="stable")`, men
    utan jämförelser: räkna, kumulera, strö ut. Elementen besöks i
    ursprunglig ordning, vilket är det som gör den stabil.

    `cursor` är en återanvänd skrivbuffert med längd n_cells. Att den finns
    återinför en O(n_cells)-term som Steg 5 tog bort ur indexet — men den
    kostar två linjära svep, uppmätt 2,0 ms vid en miljon celler, mot
    argsortens 39. Bytet är alltså kraftigt positivt även vid den skalan.
    """
    n = cells.shape[0]
    nc = cursor.shape[0]
    for c in range(nc):
        cursor[c] = 0
    for i in range(n):
        cursor[cells[i]] += 1
    run = 0
    for c in range(nc):
        k = cursor[c]
        cursor[c] = run
        run += k
    order = np.empty(n, np.int64)
    for i in range(n):
        c = cells[i]
        order[cursor[c]] = i
        cursor[c] += 1
    return order


def _group_starts(sorted_keys: np.ndarray) -> np.ndarray:
    """Startindex för varje grupp i en redan sorterad nyckelföljd."""
    if sorted_keys.size == 0:
        return np.zeros(0, dtype=np.int64)
    bounds = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    out = np.empty(bounds.size + 1, dtype=np.int64)
    out[0] = 0
    out[1:] = bounds
    return out


def next_organism_id() -> int:
    """Allokera nästa biologiska organism-id. Återanvänds aldrig."""
    return next(_organism_id_counter)


def reset_organism_ids(start: int = 1) -> None:
    """
    Nollställ id-räknaren. Avsedd för tester och reproducerbara körningar
    inom samma process — inte för produktionsbruk under pågående simulering.
    """
    global _organism_id_counter
    _organism_id_counter = _itertools.count(int(start))


@dataclass
class OrganismStore:
    """
    Kärnstore för organismer och härledda spatiala cachefält.
    
    Store-slot är den primära interna identiteten i store:n.
    Biologiskt organism-id lagras separat i `id`.
    
    Nuvarande läge:
    - flora har source of truth direkt i OrganismStore
    - fauna bär fortfarande sitt fulla dynamiska tillstånd i Agent/Body
    - store används redan för spatial indexering, lokal lookup och härledda
      perceptionsrepresentationer
    
    Härledda strukturer som byggs från store per tick:
    - spatialindex: `cell_counts`, `cell_offsets`, `cell_slots`
    - id->slot-lookup: `id_to_slot_arr`
    - flora_cell_mass: summa flora-massa per cell, härledd från levande flora
    
    `flora_cell_mass` är inte source of truth.
    Det är ett avlett perceptionsfält för sensingoptimering.
    Source of truth för flora ligger i store-slotsen (`alive`, `kind`, `cell_idx`,
    `mass`, `energy`, traits och härledda kapaciteter).
    """

    capacity: int
    n_cells: int
    n_traits: int = 32

    free_slots: list[int] = field(init=False, default_factory=list)
    
    id: np.ndarray = field(init=False)
    alive: np.ndarray = field(init=False)

    pos_x: np.ndarray = field(init=False)
    pos_y: np.ndarray = field(init=False)
    cell_idx: np.ndarray = field(init=False)

    # Glest CSR: `idx_cells` är de bebodda cellerna i stigande ordning och
    # `idx_starts` deras startpositioner i `cell_slots`, med en avslutande
    # totalsumma. Uppslag sker med `searchsorted` i stället för direkt
    # indexering, vilket tar bort n_cells-termen ur bygget.
    idx_cells: np.ndarray = field(init=False)
    idx_starts: np.ndarray = field(init=False)
    # Massviktad strukturandel för floran i cellen. Utan den kan en konsument
    # se *att* det finns föda men inte värdera den, och värdet är hela
    # skillnaden mellan seg ved och färskt blad.
    flora_cell_structure: np.ndarray = field(init=False)
    # Cellens anspråkade rotarea, summerad över de plantor som når in i den.
    # Cellarean är 1, så värdet är direkt jämförbart med 1: under betyder
    # bar mark, över betyder att marken är översökt och att varje planta
    # bara får en andel av det den gör anspråk på.
    flora_cell_claimed: np.ndarray = field(init=False)

    # Anspråkstabellen: en rad per (planta, berörd cell). En planta med
    # rotarea över 1 når in i sina sex grannar och har därför flera rader.
    # `flora_claim_share` är plantans faktiska andel av cellen efter
    # konkurrensen — summan över en cell är cellens utnyttjade andel, och
    # resten är bar mark. Tabellen är ragged och byggs om varje tick.
    flora_claim_slot: np.ndarray = field(init=False)
    flora_claim_cell: np.ndarray = field(init=False)
    flora_claim_share: np.ndarray = field(init=False)
    _flora_cells_prev: np.ndarray = field(init=False)
    # Cellmedlemskapet har ändrats sedan senaste indexbygge. Sätts av
    # alloc/clear och av rörelse som faktiskt byter cell.
    _index_dirty: bool = field(init=False, default=True)
    _csr_cursor: np.ndarray = field(init=False)
    cell_slots: np.ndarray = field(init=False)

    # Härlett perceptionsfält: summa flora-massa per cell.
    # Source of truth för flora ligger i levande store-slots.
    flora_cell_mass: np.ndarray = field(init=False)

    id_to_slot_arr: np.ndarray = field(init=False)
    id_lookup_cap: int = field(init=False, default=0)

    energy: np.ndarray = field(init=False)
    energy_cap: np.ndarray = field(init=False)
    mass: np.ndarray = field(init=False)
    age: np.ndarray = field(init=False)
    damage: np.ndarray = field(init=False)
    wear: np.ndarray = field(init=False)    

    repro_cd: np.ndarray = field(init=False)
    
    gestating: np.ndarray = field(init=False)
    gest_M: np.ndarray = field(init=False)
    gest_E_J: np.ndarray = field(init=False)
    gest_M_target: np.ndarray = field(init=False)

    genome_idx: np.ndarray = field(init=False)
    traits: np.ndarray = field(init=False)

    # Subsystemkapaciteter (fas 2: ännu skrivna från agentfenotyp)
    uptake_capacity: np.ndarray = field(init=False)
    growth_capacity: np.ndarray = field(init=False)
    dispersal_capacity: np.ndarray = field(init=False)
    sense_radius: np.ndarray = field(init=False)
    sense_rate: np.ndarray = field(init=False)
    mobility: np.ndarray = field(init=False)
    attack_capacity: np.ndarray = field(init=False)
    repair_capacity: np.ndarray = field(init=False)
    repro_capacity: np.ndarray = field(init=False)

    # Andel segt bärande material i vävnaden. Gemensam för flora och fauna:
    # samma tal beskriver ved och ben. Styr energitäthet, betningsutbyte och
    # nedbrytningstakt — se docs/substratets-struktur.md.
    structure: np.ndarray = field(init=False)

    # Mediumkapaciteter
    flood_tolerance: np.ndarray = field(init=False)
    buoyancy: np.ndarray = field(init=False)

    flora_adult_mass: np.ndarray = field(init=False)
    flora_repro_alloc: np.ndarray = field(init=False)
    flora_root_alloc: np.ndarray = field(init=False)
    flora_maturity: np.ndarray = field(init=False)
    # Rotmassa i kg. Kroppens sammansättning är integralen av tidigare
    # allokeringsbeslut, inte en andel som räknas om retroaktivt. Det gör
    # representationen redo för plasticitet utan att byta form.
    flora_root_mass: np.ndarray = field(init=False)
    # Näringsreserv och reproduktionspool, i kg näring. float64 av
    # bokföringsskäl: reserven är källa och sänka i näringsbalansen, och
    # float32 skulle kräva samma nedåtavrundningsdans som massan.
    flora_reserve: np.ndarray = field(init=False)
    flora_repro_pool: np.ndarray = field(init=False)
    flora_carbon_pool: np.ndarray = field(init=False)
    flora_temp_opt: np.ndarray = field(init=False)
    flora_temp_width: np.ndarray = field(init=False)
    flora_apparatus: np.ndarray = field(init=False)
    flora_seed_mass: np.ndarray = field(init=False)

    # Tillfällig migrationsflagga: 0=djur/nuvarande agent, 1=flora
    kind: np.ndarray = field(init=False)
    
    n: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        cap = int(self.capacity)

        n_cells = int(self.n_cells)
        self.free_slots = list(range(cap - 1, -1, -1))

        self.id = np.full(cap, -1, dtype=np.int32)
        self.alive = np.zeros(cap, dtype=np.bool_)

        self.pos_x = np.zeros(cap, dtype=np.float32)
        self.pos_y = np.zeros(cap, dtype=np.float32)
        self.cell_idx = np.full(cap, -1, dtype=np.int32)

        self.energy = np.zeros(cap, dtype=np.float32)
        self.energy_cap = np.zeros(cap, dtype=np.float32)
        self.mass = np.zeros(cap, dtype=np.float32)
        self.age = np.zeros(cap, dtype=np.float32)
        self.damage = np.zeros(cap, dtype=np.float32)
        self.wear = np.zeros(cap, dtype=np.float32)

        self.repro_cd = np.zeros((self.capacity,), dtype=np.float32)
        
        self.gestating = np.zeros((self.capacity,), dtype=bool)
        self.gest_M = np.zeros((self.capacity,), dtype=np.float32)
        self.gest_E_J = np.zeros((self.capacity,), dtype=np.float32)
        self.gest_M_target = np.zeros((self.capacity,), dtype=np.float32)

        # Fas 0: ännu ingen riktig genomstore.
        self.genome_idx = np.full(cap, -1, dtype=np.int32)
        self.traits = np.zeros((cap, int(self.n_traits)), dtype=np.float32)

        self.uptake_capacity = np.zeros(cap, dtype=np.float32)
        self.growth_capacity = np.zeros(cap, dtype=np.float32)
        self.dispersal_capacity = np.zeros(cap, dtype=np.float32)
        self.sense_radius = np.zeros(cap, dtype=np.float32)
        self.sense_rate = np.zeros(cap, dtype=np.float32)
        self.mobility = np.zeros(cap, dtype=np.float32)
        self.attack_capacity = np.zeros(cap, dtype=np.float32)
        self.repair_capacity = np.zeros(cap, dtype=np.float32)
        self.repro_capacity = np.zeros(cap, dtype=np.float32)
        self.structure = np.zeros(cap, dtype=np.float32)

        self.flood_tolerance = np.zeros(cap, dtype=np.float32)
        self.buoyancy = np.zeros(cap, dtype=np.float32)

        self.flora_adult_mass = np.zeros(cap, dtype=np.float32)
        self.flora_repro_alloc = np.zeros(cap, dtype=np.float32)
        self.flora_root_alloc = np.zeros(cap, dtype=np.float32)
        self.flora_maturity = np.zeros(cap, dtype=np.float32)
        self.flora_root_mass = np.zeros(cap, dtype=np.float32)
        self.flora_reserve = np.zeros(cap, dtype=np.float64)
        self.flora_repro_pool = np.zeros(cap, dtype=np.float64)
        self.flora_carbon_pool = np.zeros(cap, dtype=np.float64)
        self.flora_temp_opt = np.zeros(cap, dtype=np.float32)
        self.flora_temp_width = np.zeros(cap, dtype=np.float32)
        self.flora_apparatus = np.zeros(cap, dtype=np.float32)
        self.flora_seed_mass = np.zeros(cap, dtype=np.float32)

        self.kind = np.zeros(cap, dtype=np.int8)

        self.idx_cells = np.zeros(0, dtype=np.int64)
        self.idx_starts = np.zeros(1, dtype=np.int64)
        self.flora_cell_structure = np.zeros(n_cells, dtype=np.float32)
        self.flora_cell_claimed = np.zeros(n_cells, dtype=np.float32)
        self.flora_claim_slot = np.zeros(0, dtype=np.int32)
        self.flora_claim_cell = np.zeros(0, dtype=np.int32)
        self.flora_claim_share = np.zeros(0, dtype=np.float32)
        self._flora_cells_prev = np.zeros(0, dtype=np.int64)
        self._index_dirty = True
        self._csr_cursor = np.zeros(n_cells, dtype=np.int64)
        self.cell_slots = np.zeros(cap, dtype=np.int32)
        self.flora_cell_mass = np.zeros(n_cells, dtype=np.float32)

        self.id_lookup_cap = cap + 1
        self.id_to_slot_arr = np.full(self.id_lookup_cap, -1, dtype=np.int32)

        self.n = 0

    def _ensure_id_lookup_capacity(self, max_id_needed: int) -> None:
        if max_id_needed < self.id_lookup_cap:
            return
    
        new_cap = int(self.id_lookup_cap)
        while new_cap <= max_id_needed:
            new_cap = max(2 * new_cap, max_id_needed + 1)
    
        new_arr = np.full(new_cap, -1, dtype=np.int32)
        new_arr[:self.id_lookup_cap] = self.id_to_slot_arr
        self.id_to_slot_arr = new_arr
        self.id_lookup_cap = new_cap

    def _grow_array(self, arr: np.ndarray, new_cap: int) -> np.ndarray:
        """
        Allokera större array med samma dtype och trailing shape, och kopiera över innehåll.
        """
        old_cap = int(arr.shape[0])
        if new_cap <= old_cap:
            return arr
    
        new_shape = (int(new_cap),) + tuple(arr.shape[1:])
        out = np.zeros(new_shape, dtype=arr.dtype)
        out[:old_cap] = arr
        return out

    def grow(self, new_capacity: int) -> None:
        """
        Utöka store:ns slotkapacitet.

        Endast slotindexerade arrayer växer. Arrayer med annan indexdomän —
        per cell eller per organism-id — får aldrig växa med kapaciteten, och
        identifieras med namn i _NON_SLOT_ARRAYS. Att i stället gissa på
        `shape[0] == old_cap` är osäkert: när capacity råkar sammanfalla med
        n_cells blir per-cell-arrayerna felaktigt förstorade och nästa
        rebuild_spatial_index() kraschar.
        """
        old_cap = int(self.capacity)
        new_cap = int(new_capacity)
    
        if new_cap <= old_cap:
            return
    
        for f in fields(self):
            name = f.name
            if name in _NON_SLOT_ARRAYS:
                continue
    
            arr = getattr(self, name, None)
            if not isinstance(arr, np.ndarray):
                continue
    
            if arr.ndim < 1:
                continue
    
            if int(arr.shape[0]) != old_cap:
                continue
    
            setattr(self, name, self._grow_array(arr, new_cap))
    
        self.free_slots.extend(range(new_cap - 1, old_cap - 1, -1))
        self.capacity = new_cap
    
        self._assert_array_domains()

    def _assert_array_domains(self) -> None:
        """
        Kontrollera att varje array ligger i rätt indexdomän. Billig nog att
        köra efter varje grow() och fångar den vanligaste regressionen:
        ett nytt fält som läggs till utan att klassificeras.
        """
        cap = int(self.capacity)
        n_cells = int(self.n_cells)

        for _name in ("flora_cell_mass", "flora_cell_structure", "flora_cell_claimed"):
            _got = int(getattr(self, _name).shape[0])
            if _got != n_cells:
                raise AssertionError(
                    f"{_name} har längd {_got}, förväntat n_cells={n_cells}"
                )

        # Det glesa indexet har inte n_cells som längd — dess arrayer är lika
        # långa som antalet bebodda celler. Det som ska hålla är relationen
        # mellan dem.
        # Anspråkstabellens tre kolumner hör till samma rader och måste
        # därför vara lika långa. Längden i sig är fri.
        _cl = (int(self.flora_claim_slot.shape[0]),
               int(self.flora_claim_cell.shape[0]),
               int(self.flora_claim_share.shape[0]))
        if len(set(_cl)) != 1:
            raise AssertionError(
                f"anspråkstabellens kolumner har olika längd: "
                f"slot={_cl[0]}, cell={_cl[1]}, share={_cl[2]}"
            )

        if int(self.idx_starts.shape[0]) != int(self.idx_cells.shape[0]) + 1:
            raise AssertionError(
                f"idx_starts har längd {int(self.idx_starts.shape[0])}, "
                f"förväntat idx_cells + 1 = {int(self.idx_cells.shape[0]) + 1}"
            )
    
        for f in fields(self):
            name = f.name
            if name in _NON_SLOT_ARRAYS:
                continue
            arr = getattr(self, name, None)
            if not isinstance(arr, np.ndarray) or arr.ndim < 1:
                continue
            if int(arr.shape[0]) != cap:
                raise AssertionError(f"{name} har längd {int(arr.shape[0])}, förväntat capacity={cap}")
        
    def mark_index_dirty(self) -> None:
        """
        Något har flyttat, fötts eller dött sedan senaste indexbygge.

        Flaggan gäller *cellmedlemskapet*, inte de härledda florafälten. Florans
        massa ändras varje tick i tillväxtpasset, så en flagga som också täckte
        dem hade alltid varit satt och därmed varit meningslös. Det som ändras
        sällan är vem som ligger i vilken cell.
        """
        self._index_dirty = True

    def alloc_slot(self) -> int:
        if not self.free_slots:
            raise RuntimeError("OrganismStore has no free slots; caller must grow store before alloc_slot().")
        slot = self.free_slots.pop()
        if slot >= self.n:
            self.n = slot + 1
        self._index_dirty = True
        return slot
    
    def write_agent(self, slot: int, a, grid) -> None:
        """
        Init/write-once-ish store population for fauna wrappers.
    
        Denna metod används nu främst vid initialisering och spawn för att skriva
        statisk eller långlivad metadata. Löpande fauna-state skrivs i subsystempassen
        och inte via central tick-writeback.
        """
        body = a.body

        self.id[slot] = int(a.id)
        self.alive[slot] = bool(body.alive)
        
        self.energy[slot] = float(body.E_total())
        self.energy_cap[slot] = float(body.E_cap())
        self.mass[slot] = float(body.M)
        self.damage[slot] = float(body.D)
        self.wear[slot] = float(body.W)

        self.genome_idx[slot] = slot
    
        ph = a.pheno
        self.uptake_capacity[slot] = 0.0
        self.growth_capacity[slot] = 0.0
        self.dispersal_capacity[slot] = 0.0
    
        self.sense_radius[slot] = float(getattr(a.AP, "ray_len_front", 0.0))
        self.sense_rate[slot] = 1.0 / max(float(getattr(a.AP, "sense_idle_steps", 1)), 1.0)
        self.mobility[slot] = float(getattr(a.AP, "v_max", 0.0))
        self.attack_capacity[slot] = float(getattr(ph, "predation", 0.0))
        self.repair_capacity[slot] = float(getattr(ph, "repair_capacity", 0.0))
        self.repro_capacity[slot] = float(getattr(ph, "repro_rate", 0.0))

        # Strukturandel: samma axel som för flora. Kadaver ärver den, och den
        # styr både energitäthet och nedbrytningstakt.
        traits = getattr(getattr(a, "genome", None), "traits", None)
        self.structure[slot] = np.float32(structure_fraction(traits))
        if traits is not None:
            n = min(int(self.traits.shape[1]), int(np.size(traits)))
            self.traits[slot, :n] = np.asarray(traits, dtype=np.float32).ravel()[:n]
    
        self.flood_tolerance[slot] = 0.0
        # Flytförmågan är härledd ur strukturandelen och är därmed den första
        # mediumkapaciteten med både skrivare och läsare. Se
        # phenotype.buoyancy_from_structure och Agent._water_factor.
        self.buoyancy[slot] = np.float32(
            buoyancy_from_structure(float(self.structure[slot]))
        )
        self.kind[slot] = 0

    def release_slot(self, slot: int) -> None:
        self.clear_slot(slot)
        self.free_slots.append(int(slot))
        
    def clear_slot(self, slot: int) -> None:
        self._index_dirty = True
        self.id[slot] = -1
        self.alive[slot] = False
        self.pos_x[slot] = 0.0
        self.pos_y[slot] = 0.0
        self.cell_idx[slot] = -1
        self.genome_idx[slot] = -1
        self.traits[slot, :] = 0.0
        
        self.energy[slot] = 0.0
        self.energy_cap[slot] = 0.0
        self.mass[slot] = 0.0
        self.age[slot] = 0.0
        self.damage[slot] = 0.0
        self.wear[slot] = 0.0

        self.repro_cd[slot] = np.float32(0.0)
        
        self.gestating[slot] = False
        self.gest_M[slot] = np.float32(0.0)
        self.gest_E_J[slot] = np.float32(0.0)
        self.gest_M_target[slot] = np.float32(0.0)

        self.uptake_capacity[slot] = 0.0
        self.growth_capacity[slot] = 0.0
        self.dispersal_capacity[slot] = 0.0
        self.sense_radius[slot] = 0.0
        self.sense_rate[slot] = 0.0
        self.mobility[slot] = 0.0
        self.attack_capacity[slot] = 0.0
        self.repair_capacity[slot] = 0.0
        self.repro_capacity[slot] = 0.0
        self.flood_tolerance[slot] = 0.0
        self.buoyancy[slot] = 0.0
    
        self.flora_adult_mass[slot] = 0.0
        self.flora_repro_alloc[slot] = 0.0
        self.flora_root_alloc[slot] = 0.0
        self.flora_maturity[slot] = 0.0
        self.flora_root_mass[slot] = 0.0
        self.flora_reserve[slot] = 0.0
        self.flora_repro_pool[slot] = 0.0
        self.flora_carbon_pool[slot] = 0.0
        self.flora_temp_opt[slot] = 0.0
        self.flora_temp_width[slot] = 0.0
        self.flora_apparatus[slot] = 0.0
        self.flora_seed_mass[slot] = 0.0
    
        self.kind[slot] = 0
        
    def set_flora_claims(self, claimed, slots, cells, share) -> None:
        """
        Skriv upptagspassets ytfördelning som härlett tillstånd.

        Passet räknar redan ut både cellens totala anspråk och varje plantas
        andel av den — det är så konkurrensen om marken avgörs. Att behålla
        resultatet i stället för att kasta det gör fördelningen läsbar för
        viewern och för diagnostik utan att någon annan kod behöver räkna om
        formeln. En andra implementation av samma semantik är precis den
        divergens som `_flora_*` i population.py redan orsakat mot
        phenotype.py.

        Ägarskapet är entydigt: `_growth_system_flora` är enda skrivare, och
        fälten är giltiga från upptagspassets slut till nästa tick.
        """
        self.flora_cell_claimed[:] = np.asarray(claimed, dtype=np.float32)
        self.flora_claim_slot = np.asarray(slots, dtype=np.int32)
        self.flora_claim_cell = np.asarray(cells, dtype=np.int32)
        self.flora_claim_share = np.asarray(share, dtype=np.float32)

    def clear_flora_claims(self) -> None:
        """Nolla ytfördelningen. Anropas när upptagspasset inte gör anspråk."""
        self.flora_cell_claimed.fill(np.float32(0.0))
        self.flora_claim_slot = np.zeros(0, dtype=np.int32)
        self.flora_claim_cell = np.zeros(0, dtype=np.int32)
        self.flora_claim_share = np.zeros(0, dtype=np.float32)

    def rebuild_spatial_index(self, with_flora_fields: bool = True) -> None:
        """
        Bygg gemensamt spatialindex för alla levande organismer.

        Bygget gör tre saker med olika kadensbehov, och det var det som gjorde
        det till trettiotvå procent av takten vid 256x256: det kördes två
        gånger per tick och gjorde allt tre båda gångerna.

        `id -> slot` och CSR-layouten ändras när någon föds, dör eller byter
        cell. De härledda florafälten `flora_cell_mass` och
        `flora_cell_structure` ändras varje tick, eftersom florans massa gör
        det — men bara i tillväxtpasset, och betningen håller dem uppdaterade
        inkrementellt medan den äter.

        Andra anropet per tick, efter faunapassen, behöver därför bara
        medlemskapet: `with_flora_fields=False`. Och har ingen flyttat sedan
        förra bygget är det anropet en ren no-op — vilket är hela ticken i en
        värld utan fauna.

        Glest CSR-liknande layout:
          - idx_cells[k]   = bebodda cell-ID, stigande
          - idx_starts[k]  = startindex i cell_slots, med totalsumma sist
          - cell_slots[..] = platt lista av slotindex, grupperade per cell

        Samtidigt byggs snabb id->slot-lookup och det härledda
        perceptionsfältet `flora_cell_mass`, som inte är source of truth.

        Ingenting här skalar med n_cells. Den täta formen räknade och
        kumulerade över hela cellrymden, vilket vid en miljon celler och
        femtontusen organismer betydde att indexet nästan uteslutande
        arbetade med tomrum. `flora_cell_mass` är fortfarande tät, eftersom
        perceptionen läser den med fancy indexing över många celler samtidigt
        — men bara de celler som faktiskt var satta nollställs.

        Den stabila sorteringen bevarar slotordningen inom varje cell.
        """
        if not with_flora_fields and not self._index_dirty:
            return
        self._index_dirty = False

        n = int(self.n)
        empty_i64 = np.zeros(0, dtype=np.int64)

        # Nollställ bara det som var satt förra bygget.
        if with_flora_fields and self._flora_cells_prev.size:
            self.flora_cell_mass[self._flora_cells_prev] = np.float32(0.0)
            self.flora_cell_structure[self._flora_cells_prev] = np.float32(0.0)
            self._flora_cells_prev = empty_i64

        self.idx_cells = empty_i64
        self.idx_starts = np.zeros(1, dtype=np.int64)

        if n <= 0:
            self.id_to_slot_arr.fill(-1)
            return

        live = np.flatnonzero(self.alive[:n])
        if live.size == 0:
            self.id_to_slot_arr.fill(-1)
            return

        # --- id -> slot ---
        ids = self.id[live].astype(np.int64, copy=False)
        max_live_id = int(ids.max())
        if max_live_id >= 0:
            self._ensure_id_lookup_capacity(max_live_id)
        self.id_to_slot_arr.fill(-1)
        ok_id = ids >= 0
        if np.any(ok_id):
            self.id_to_slot_arr[ids[ok_id]] = live[ok_id].astype(
                self.id_to_slot_arr.dtype, copy=False
            )

        # --- celltillhörighet ---
        cells = self.cell_idx[live].astype(np.int64, copy=False)
        placed = cells >= 0
        if not np.any(placed):
            return

        live_c = live[placed]
        cells_c = cells[placed]

        # Counting sort när numba finns, annars argsort. Permutationen är
        # densamma; argsorten var uppmätt 3,9 ms vid 47 000 organismer och
        # 39 ms vid 350 000 — superlinjär, och körd två gånger per tick. Den
        # var därmed den enskilt största posten i profilen vid stora bestånd.
        if _HAVE_NUMBA:
            order = _counting_sort_order(cells_c, self._csr_cursor)
        else:
            order = np.argsort(cells_c, kind="stable")
        sorted_cells = cells_c[order]
        sorted_slots = live_c[order]
        m = int(sorted_slots.size)
        self.cell_slots[:m] = sorted_slots.astype(self.cell_slots.dtype, copy=False)

        # Gruppgränserna läses ur den redan sorterade följden. `np.unique`
        # hade sorterat om, alltså gjort samma arbete en gång till.
        starts = _group_starts(sorted_cells)
        self.idx_cells = sorted_cells[starts]
        self.idx_starts = np.append(starts, np.int64(m))

        # --- härlett florafält ---
        if not with_flora_fields:
            return
        is_flora = self.kind[sorted_slots] == 1
        if np.any(is_flora):
            fcells = sorted_cells[is_flora]
            # Skottmassa, inte total. Rötterna är under jord: de kan varken
            # ses eller ätas. Att fältet summerade hela växten betydde att
            # sensorn rapporterade något fysiskt osynligt — och sedan 0071, då
            # betningen stannar vid roten, att djuret såg två till tre gånger
            # mer mat än det kunde få i sig. Det är inte en fråga om att
            # sensorn ska bedöma ätbarhet; samma horisont definierar både vad
            # som syns och vad som går att äta.
            _fm_all = self.mass[sorted_slots][is_flora].astype(np.float64, copy=False)
            _fr_all = self.flora_root_mass[sorted_slots][is_flora].astype(np.float64, copy=False)
            fmass = np.maximum(0.0, _fm_all - _fr_all)
            fstruct = self.structure[sorted_slots][is_flora].astype(np.float64, copy=False)
            fstarts = _group_starts(fcells)
            fu = fcells[fstarts]
            msum = np.add.reduceat(fmass, fstarts)
            ssum = np.add.reduceat(fmass * fstruct, fstarts)
            self.flora_cell_mass[fu] = msum.astype(np.float32, copy=False)
            self.flora_cell_structure[fu] = (
                ssum / np.maximum(msum, 1e-30)
            ).astype(np.float32, copy=False)
            self._flora_cells_prev = fu

    def slots_in_cell(self, cell: int) -> np.ndarray:
        """
        Slotarna i en cell. Tom array om cellen är obebodd.

        Uppslaget går via `searchsorted` i listan över bebodda celler i
        stället för direkt indexering i en array av världens längd. Det gör
        bygget oberoende av n_cells, vilket är förutsättningen för miljoncells-
        världen i Steg 3 — där bär annars indexet en term som skalar med
        världen och inte med livet i den.
        """
        cells = self.idx_cells
        if cells.size == 0:
            return self.cell_slots[0:0]
        i = int(np.searchsorted(cells, int(cell)))
        if i >= cells.size or int(cells[i]) != int(cell):
            return self.cell_slots[0:0]
        return self.cell_slots[int(self.idx_starts[i]):int(self.idx_starts[i + 1])]

    def slot_for_id(self, id_: int) -> int:
        oid = int(id_)
        if oid < 0 or oid >= self.id_lookup_cap:
            return -1
        return int(self.id_to_slot_arr[oid])
    
    def slots_for_ids(self, ids: np.ndarray) -> np.ndarray:
        """
        Vektoriserad lookup: biologiska id -> slotindex.
        Ogiltiga eller okända id ger -1.
        """
        ids = np.asarray(ids, dtype=np.int32)
        out = np.full(ids.shape, -1, dtype=np.int32)
    
        if ids.size == 0:
            return out
    
        mask = (ids >= 0) & (ids < self.id_lookup_cap)
        if np.any(mask):
            out[mask] = self.id_to_slot_arr[ids[mask]]
    
        return out