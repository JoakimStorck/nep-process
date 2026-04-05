from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class OrganismStore:
    """
    Fas 0: read-only spegel av nuvarande Agent-objekt.

    Source of truth ligger fortfarande i Agent/Body.
    Den här store:n fylls om från populationen efter varje tick.

    Index i arrayerna är store-slot, inte agent_id.
    agent_id lagras explicit i id-arrayerna.
    """

    capacity: int
    world_size: int

    id: np.ndarray = field(init=False)
    alive: np.ndarray = field(init=False)

    pos_x: np.ndarray = field(init=False)
    pos_y: np.ndarray = field(init=False)
    cell_idx: np.ndarray = field(init=False)

    energy: np.ndarray = field(init=False)
    mass: np.ndarray = field(init=False)
    age: np.ndarray = field(init=False)

    genome_idx: np.ndarray = field(init=False)

    n: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        cap = int(self.capacity)

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

        self.n = 0

    def clear(self) -> None:
        self.n = 0

    def sync_from_agents(self, agents: Iterable, grid) -> None:
        """
        Full omskrivning av levande+icke-levande agenter i nuvarande population.
        Fas 0: enkel, tydlig, korrekt. Ingen inkrementell optimering ännu.
        """
        self.clear()

        for i, a in enumerate(agents):
            if i >= self.capacity:
                raise RuntimeError(
                    f"OrganismStore capacity exceeded: {i+1} > {self.capacity}"
                )

            body = a.body

            x = float(a.x)
            y = float(a.y)

            self.id[i] = int(a.id)
            self.alive[i] = bool(body.alive)

            self.pos_x[i] = x
            self.pos_y[i] = y
            self.cell_idx[i] = int(grid.cell_of(x, y))

            self.energy[i] = float(body.E_total())
            self.mass[i] = float(body.M)
            self.age[i] = float(a.age_s)

            # Fas 0-placeholder. Senare ska detta peka mot separat genomlager.
            self.genome_idx[i] = i

            self.n = i + 1

    def assert_consistent(self) -> None:
        """
        Billiga invariants för Fas 0.
        """
        n = int(self.n)
        if n < 0 or n > int(self.capacity):
            raise AssertionError(f"Invalid store size: n={n}, capacity={self.capacity}")

        s = int(self.world_size)
        max_cell = s * s

        if n == 0:
            return

        if np.any(self.id[:n] < 0):
            raise AssertionError("Negative agent id in live prefix")

        if np.any(self.cell_idx[:n] < 0) or np.any(self.cell_idx[:n] >= max_cell):
            raise AssertionError("cell_idx out of range")

        if np.any(~np.isfinite(self.pos_x[:n])) or np.any(~np.isfinite(self.pos_y[:n])):
            raise AssertionError("Non-finite position in OrganismStore")

        if np.any(~np.isfinite(self.energy[:n])) or np.any(self.energy[:n] < 0.0):
            raise AssertionError("Invalid energy in OrganismStore")

        if np.any(~np.isfinite(self.mass[:n])) or np.any(self.mass[:n] < 0.0):
            raise AssertionError("Invalid mass in OrganismStore")

        if np.any(~np.isfinite(self.age[:n])) or np.any(self.age[:n] < 0.0):
            raise AssertionError("Invalid age in OrganismStore")