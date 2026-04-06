from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


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
    world_size: int

    free_slots: list[int] = field(init=False, default_factory=list)
    
    id: np.ndarray = field(init=False)
    alive: np.ndarray = field(init=False)

    pos_x: np.ndarray = field(init=False)
    pos_y: np.ndarray = field(init=False)
    cell_idx: np.ndarray = field(init=False)

    cell_counts: np.ndarray = field(init=False)
    cell_offsets: np.ndarray = field(init=False)
    cell_slots: np.ndarray = field(init=False)

    # Härlett perceptionsfält: summa flora-massa per cell.
    # Source of truth för flora ligger i levande store-slots.
    flora_cell_mass: np.ndarray = field(init=False)

    id_to_slot_arr: np.ndarray = field(init=False)
    id_lookup_cap: int = field(init=False, default=0)

    energy: np.ndarray = field(init=False)
    mass: np.ndarray = field(init=False)
    age: np.ndarray = field(init=False)

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

    # Mediumkapaciteter
    flood_tolerance: np.ndarray = field(init=False)
    buoyancy: np.ndarray = field(init=False)

    flora_growth_rate: np.ndarray = field(init=False)
    flora_adult_mass: np.ndarray = field(init=False)
    flora_temp_opt: np.ndarray = field(init=False)
    flora_temp_width: np.ndarray = field(init=False)
    flora_dispersal_rate: np.ndarray = field(init=False)    

    # Tillfällig migrationsflagga: 0=djur/nuvarande agent, 1=flora
    kind: np.ndarray = field(init=False)
    
    n: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        cap = int(self.capacity)

        n_cells = int(self.world_size) * int(self.world_size)
        self.free_slots = list(range(cap - 1, -1, -1))

        self.id = np.full(cap, -1, dtype=np.int32)
        self.alive = np.zeros(cap, dtype=np.bool_)

        self.pos_x = np.zeros(cap, dtype=np.float32)
        self.pos_y = np.zeros(cap, dtype=np.float32)
        self.cell_idx = np.full(cap, -1, dtype=np.int32)

        self.energy = np.zeros(cap, dtype=np.float32)
        self.mass = np.zeros(cap, dtype=np.float32)
        self.age = np.zeros(cap, dtype=np.float32)

        # Fas 0: ännu ingen riktig genomstore.
        self.genome_idx = np.full(cap, -1, dtype=np.int32)
        self.traits = np.zeros((cap, 32), dtype=np.float32)

        self.uptake_capacity = np.zeros(cap, dtype=np.float32)
        self.growth_capacity = np.zeros(cap, dtype=np.float32)
        self.dispersal_capacity = np.zeros(cap, dtype=np.float32)
        self.sense_radius = np.zeros(cap, dtype=np.float32)
        self.sense_rate = np.zeros(cap, dtype=np.float32)
        self.mobility = np.zeros(cap, dtype=np.float32)
        self.attack_capacity = np.zeros(cap, dtype=np.float32)
        self.repair_capacity = np.zeros(cap, dtype=np.float32)
        self.repro_capacity = np.zeros(cap, dtype=np.float32)

        self.flood_tolerance = np.zeros(cap, dtype=np.float32)
        self.buoyancy = np.zeros(cap, dtype=np.float32)

        self.flora_growth_rate = np.zeros(cap, dtype=np.float32)
        self.flora_adult_mass = np.zeros(cap, dtype=np.float32)
        self.flora_temp_opt = np.zeros(cap, dtype=np.float32)
        self.flora_temp_width = np.zeros(cap, dtype=np.float32)
        self.flora_dispersal_rate = np.zeros(cap, dtype=np.float32)        

        self.kind = np.zeros(cap, dtype=np.int8)

        self.cell_counts = np.zeros(n_cells, dtype=np.int32)
        self.cell_offsets = np.zeros(n_cells + 1, dtype=np.int32)
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
    
    def alloc_slot(self) -> int:
        if not self.free_slots:
            raise RuntimeError("OrganismStore full")
        slot = self.free_slots.pop()
        if slot >= self.n:
            self.n = slot + 1
        return slot    
    
    def write_agent(self, slot: int, a, grid) -> None:
        body = a.body
        x = float(a.x)
        y = float(a.y)
    
        self.id[slot] = int(a.id)
        self.alive[slot] = bool(body.alive)
    
        self.pos_x[slot] = x
        self.pos_y[slot] = y
        self.cell_idx[slot] = int(grid.cell_of(x, y))
    
        self.energy[slot] = float(body.E_total())
        self.mass[slot] = float(body.M)
        self.age[slot] = float(a.age_s)
    
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
    
        self.flood_tolerance[slot] = 0.0
        self.buoyancy[slot] = 0.0
        self.kind[slot] = 0

    def release_slot(self, slot: int) -> None:
        self.clear_slot(slot)
        self.free_slots.append(int(slot))
        
    def clear_slot(self, slot: int) -> None:
        self.id[slot] = -1
        self.alive[slot] = False
        self.pos_x[slot] = 0.0
        self.pos_y[slot] = 0.0
        self.cell_idx[slot] = -1
        self.energy[slot] = 0.0
        self.mass[slot] = 0.0
        self.age[slot] = 0.0
        self.genome_idx[slot] = -1
        self.traits[slot, :] = 0.0
    
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
    
        self.flora_growth_rate[slot] = 0.0
        self.flora_adult_mass[slot] = 0.0
        self.flora_temp_opt[slot] = 0.0
        self.flora_temp_width[slot] = 0.0
        self.flora_dispersal_rate[slot] = 0.0
    
        self.kind[slot] = 0
        
    def rebuild_spatial_index(self) -> None:
        """
        Bygg gemensamt spatialindex för alla levande organismer.
    
        CSR-liknande layout:
          - cell_counts[c]   = antal levande slotar i cell c
          - cell_offsets[c]  = startindex i cell_slots för cell c
          - cell_slots[...]  = platt lista av slotindex, grupperade per cell
    
        Samtidigt byggs:
          - snabb id->slot-lookup för levande organismer
          - platt härlett flora-perceptionsfält: flora_cell_mass[cell]
            (ej source of truth, bara sensing-cache)
        """
        counts = self.cell_counts
        offsets = self.cell_offsets
        slots_out = self.cell_slots
        flora = self.flora_cell_mass
    
        counts.fill(0)
        flora.fill(np.float32(0.0))
    
        max_live_id = -1
    
        # Pass 1: counts + flora field + max id
        for slot in range(int(self.n)):
            if not bool(self.alive[slot]):
                continue
    
            cell = int(self.cell_idx[slot])
            if cell >= 0:
                counts[cell] += 1
                if int(self.kind[slot]) == 1:
                    flora[cell] = np.float32(float(flora[cell]) + float(self.mass[slot]))
    
            oid = int(self.id[slot])
            if oid > max_live_id:
                max_live_id = oid
    
        if max_live_id >= 0:
            self._ensure_id_lookup_capacity(max_live_id)
    
        self.id_to_slot_arr.fill(-1)
    
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
    
        write_ptr = offsets[:-1].copy()
    
        # Pass 2: id->slot + packed cell slots
        for slot in range(int(self.n)):
            if not bool(self.alive[slot]):
                continue
    
            oid = int(self.id[slot])
            if oid >= 0:
                self.id_to_slot_arr[oid] = int(slot)
    
            cell = int(self.cell_idx[slot])
            if cell < 0:
                continue
    
            j = write_ptr[cell]
            slots_out[j] = int(slot)
            write_ptr[cell] += 1
    
    def slots_in_cell(self, cell: int) -> np.ndarray:
        start = int(self.cell_offsets[cell])
        end = int(self.cell_offsets[cell + 1])
        return self.cell_slots[start:end]

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