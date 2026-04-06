from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class OrganismStore:
    """
    Migrerande kärnstore för organismer.

    Just nu är source of truth fortfarande i Agent/Body för djurpopulationen.
    Store:n fylls om från populationen efter varje tick, men strukturen utökas nu
    så att även flora senare kan allokeras direkt här.

    Index i arrayerna är store-slot, inte agent_id.
    agent_id lagras explicit i id-arrayerna.
    """

    capacity: int
    world_size: int

    free_slots: list[int] = field(init=False, default_factory=list)
    
    id: np.ndarray = field(init=False)
    alive: np.ndarray = field(init=False)

    pos_x: np.ndarray = field(init=False)
    pos_y: np.ndarray = field(init=False)
    cell_idx: np.ndarray = field(init=False)

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
        
        self.n = 0

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
        

