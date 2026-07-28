from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class Grid:
    """
    Världens geometri. Enda plats i systemet där cellform, grannrelationer och
    kopplingen mellan kontinuerlig position och cell-ID är känd.

    Utöver de topologiska metoderna exponerar Grid förberäknade tabeller som
    gör topologiska pass vektoriserbara utan Python-loop:

        n_cells         antal celler
        neighbor_idx    (n_cells, k) int32 — grannarnas cell-ID
        neighbor_mask   (n_cells, k) bool  — vilka grannplatser som är giltiga
        cell_lat        (n_cells,) float32 — normerad latitud i [-1, +1]
        cell_center_x   (n_cells,) float32 — cellcentrums kontinuerliga position
        cell_center_y   (n_cells,) float32

    Tabellerna är geometrispecifika men gränssnittet är det inte: en hex-
    implementation fyller samma tabeller med k=6 utan att någon anropare ändras.

    `cell_lat` är en geometrisk egenskap — cellens normerade läge längs världens
    andra axel. Vad latituden betyder klimatologiskt ägs av världslagret, inte
    här.
    """

    size: int

    n_cells: int = field(init=False, repr=False, compare=False)
    neighbor_idx: np.ndarray = field(init=False, repr=False, compare=False)
    neighbor_mask: np.ndarray = field(init=False, repr=False, compare=False)
    cell_lat: np.ndarray = field(init=False, repr=False, compare=False)
    cell_center_x: np.ndarray = field(init=False, repr=False, compare=False)
    cell_center_y: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        s = int(self.size)
        n = s * s

        cells = np.arange(n, dtype=np.int64)
        row = cells // s
        col = cells % s

        # Von Neumann-grannar i samma ordning som neighbors(): upp, ned, vänster, höger.
        up = ((row - 1) % s) * s + col
        down = ((row + 1) % s) * s + col
        left = row * s + ((col - 1) % s)
        right = row * s + ((col + 1) % s)

        neighbor_idx = np.stack((up, down, left, right), axis=1).astype(np.int32, copy=False)

        # På en torus är alla grannplatser giltiga. Masken finns för geometrier
        # där de inte är det, så att anropare kan skrivas en gång.
        neighbor_mask = np.ones(neighbor_idx.shape, dtype=np.bool_)

        denom = float(max(1, s - 1))
        cell_lat = (2.0 * (row.astype(np.float32) / np.float32(denom)) - np.float32(1.0)).astype(
            np.float32, copy=False
        )

        object.__setattr__(self, "n_cells", int(n))
        object.__setattr__(self, "neighbor_idx", neighbor_idx)
        object.__setattr__(self, "neighbor_mask", neighbor_mask)
        object.__setattr__(self, "cell_lat", cell_lat)
        object.__setattr__(self, "cell_center_x", (col + 0.5).astype(np.float32, copy=False))
        object.__setattr__(self, "cell_center_y", (row + 0.5).astype(np.float32, copy=False))

    @property
    def neighbor_count(self) -> int:
        """Antal grannplatser per cell. 4 för kvadratisk von Neumann, 6 för hex."""
        return int(self.neighbor_idx.shape[1])

    def wrap_pos(self, x: float, y: float) -> tuple[float, float]:
        s = float(self.size)
        return x % s, y % s

    def cell_of(self, x: float, y: float) -> int:
        """
        Position -> cell-ID.

        Anmärkning: trunkerar före modulo, vilket avviker från cell_of_many()
        för negativa koordinater (cell_of(-0.5) ger 0, cell_of_many ger 63).
        Positioner är wrappade till [0, size) överallt i simuleringen, så
        skillnaden är i dag inte nåbar. Semantiken bör harmoniseras när
        geometrin byts i Steg 2, men inte i samma ändring som en refaktor —
        det vore en tyst beteendeändring.
        """
        s = int(self.size)
        ix = int(x) % s
        iy = int(y) % s
        return iy * s + ix

    def cell_of_many(self, xs: object, ys: object) -> np.ndarray:
        """
        Vektoriserad position -> cell-ID.

        Toroidal semantik: modulo först, golv sedan. Det är samma ordning som
        bilinear_indices_many() använder och den enda som är korrekt för
        negativa koordinater. Den skalära cell_of() trunkerar före modulo och
        avviker därför för negativa värden — se anmärkningen där.
        """
        s = int(self.size)
        xw = np.mod(np.asarray(xs, dtype=np.float32), np.float32(s))
        yw = np.mod(np.asarray(ys, dtype=np.float32), np.float32(s))
        return (yw.astype(np.int32, copy=False) * np.int32(s)
                + xw.astype(np.int32, copy=False)).astype(np.int32, copy=False)

    def rowcol_of(self, cell: int) -> tuple[int, int]:
        s = int(self.size)
        cell = int(cell) % (s * s)
        row, col = divmod(cell, s)
        return row, col

    def cell_from_rowcol(self, row: int, col: int) -> int:
        s = int(self.size)
        return (int(row) % s) * s + (int(col) % s)

    def wrap_cell(self, cell: int) -> int:
        s = int(self.size)
        return int(cell) % (s * s)

    def neighbors(self, cell: int) -> tuple[int, int, int, int]:
        """
        Von Neumann-grannar på kvadratisk torus:
        upp, ned, vänster, höger.
        """
        r, c = self.rowcol_of(cell)
        return (
            self.cell_from_rowcol(r - 1, c),
            self.cell_from_rowcol(r + 1, c),
            self.cell_from_rowcol(r, c - 1),
            self.cell_from_rowcol(r, c + 1),
        )

    def bilinear_corners(self, x: float, y: float) -> tuple[int, int, int, int, float, float]:
        """
        Wrapad bilinjär diskretisering av kontinuerlig position.

        Returnerar:
            x0, y0, x1, y1, fx, fy

        där (x0,y0) och (x1,y1) är hörnindex i kvadratgridet och
        fx, fy är fraktionella vikter i [0,1).
        """
        s = int(self.size)
        xw, yw = self.wrap_pos(float(x), float(y))

        x0 = int(math.floor(xw)) % s
        y0 = int(math.floor(yw)) % s
        x1 = (x0 + 1) % s
        y1 = (y0 + 1) % s

        fx = xw - math.floor(xw)
        fy = yw - math.floor(yw)

        return x0, y0, x1, y1, fx, fy

    def bilinear_indices_many(
        self,
        xs: object,
        ys: object,
    ) -> tuple[object, object, object, object, object, object]:
        """
        Batchvariant av bilinjär diskretisering.

        Returnerar arrayer:
            x0, y0, x1, y1, fx, fy

        med samma semantik som bilinear_corners(), men för hela fält av punkter.
        """
        s = int(self.size)
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)

        xw = np.mod(xs, np.float32(s))
        yw = np.mod(ys, np.float32(s))

        x0 = xw.astype(np.int32, copy=False)
        y0 = yw.astype(np.int32, copy=False)

        fx = xw - x0
        fy = yw - y0

        x1 = x0 + 1
        y1 = y0 + 1
        x1[x1 == s] = 0
        y1[y1 == s] = 0

        return x0, y0, x1, y1, fx, fy
        
    def distance(self, cell_a: int, cell_b: int) -> int:
        """
        Topologiskt cellavstånd på kvadratisk torus, mätt som Manhattan-avstånd
        med toroidal wrap.
        """
        s = int(self.size)
        ra, ca = self.rowcol_of(cell_a)
        rb, cb = self.rowcol_of(cell_b)

        dr = abs(rb - ra)
        dc = abs(cb - ca)

        dr = min(dr, s - dr)
        dc = min(dc, s - dc)

        return dr + dc

    def cells_within(self, cell: int, r: int) -> tuple[int, ...]:
        """
        Alla celler inom topologiskt avstånd <= r från centrumcellen,
        under kvadratisk torus och samma metrik som distance().
        """
        rr = int(r)
        if rr < 0:
            return ()

        c0r, c0c = self.rowcol_of(cell)
        out: list[int] = []
        for dr in range(-rr, rr + 1):
            rem = rr - abs(dr)
            for dc in range(-rem, rem + 1):
                out.append(self.cell_from_rowcol(c0r + dr, c0c + dc))
        return tuple(out)

    def torus_delta_pos(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
        s = float(self.size)
        half = 0.5 * s

        dx = x2 - x1
        dy = y2 - y1

        if dx > half:
            dx -= s
        elif dx < -half:
            dx += s

        if dy > half:
            dy -= s
        elif dy < -half:
            dy += s

        return dx, dy

    def distance2_pos(self, x1: float, y1: float, x2: float, y2: float) -> float:
        dx, dy = self.torus_delta_pos(x1, y1, x2, y2)
        return dx * dx + dy * dy

    def distance_pos(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt(self.distance2_pos(x1, y1, x2, y2))