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

    # width och height är de bärande måtten. size är en bekvämlighet som sätter
    # båda för kvadratiska världar; den sätts till 0 när världen inte är
    # kvadratisk och ska inte läsas av någon — använd width, height eller shape.
    size: int = 0
    width: int = 0
    height: int = 0

    n_cells: int = field(init=False, repr=False, compare=False)
    neighbor_idx: np.ndarray = field(init=False, repr=False, compare=False)
    neighbor_mask: np.ndarray = field(init=False, repr=False, compare=False)
    cell_lat: np.ndarray = field(init=False, repr=False, compare=False)
    cell_center_x: np.ndarray = field(init=False, repr=False, compare=False)
    cell_center_y: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Bredd och höjd är de bärande måtten. `size` finns kvar som bekvämlighet
        # för kvadratiska världar och sätter båda.
        w = int(self.width) if int(self.width) > 0 else int(self.size)
        h = int(self.height) if int(self.height) > 0 else int(self.size)
        if w <= 0 or h <= 0:
            raise ValueError("Grid kräver size, eller width och height")

        object.__setattr__(self, "width", w)
        object.__setattr__(self, "height", h)
        object.__setattr__(self, "size", w if w == h else 0)

        n = w * h

        cells = np.arange(n, dtype=np.int64)
        row = cells // w
        col = cells % w

        # Von Neumann-grannar i samma ordning som neighbors(): upp, ned, vänster, höger.
        up = ((row - 1) % h) * w + col
        down = ((row + 1) % h) * w + col
        left = row * w + ((col - 1) % w)
        right = row * w + ((col + 1) % w)

        neighbor_idx = np.stack((up, down, left, right), axis=1).astype(np.int32, copy=False)

        # På en torus är alla grannplatser giltiga. Masken finns för geometrier
        # där de inte är det, så att anropare kan skrivas en gång.
        neighbor_mask = np.ones(neighbor_idx.shape, dtype=np.bool_)

        denom = float(max(1, h - 1))
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
    def shape(self) -> tuple[int, int]:
        """(höjd, bredd) i celler. För anropare som behöver rutnätsform."""
        return int(self.height), int(self.width)

    @property
    def extent_x(self) -> float:
        """
        Världens utsträckning i kontinuerligt rum längs x.

        Sammanfaller med antalet celler bara för kvadratgeometri. På hex är
        cellbredden inte 1, och utsträckningen skiljer sig därför från
        cellantalet — anropare ska aldrig anta att de är samma tal.
        """
        return float(self.width)

    @property
    def extent_y(self) -> float:
        """Världens utsträckning i kontinuerligt rum längs y."""
        return float(self.height)

    def random_position(self, rng) -> tuple[float, float]:
        """Enhetligt fördelad position i världen. Geometrin äger sin egen form."""
        return (
            float(rng.uniform(0.0, self.extent_x)),
            float(rng.uniform(0.0, self.extent_y)),
        )

    @property
    def neighbor_count(self) -> int:
        """Antal grannplatser per cell. 4 för kvadratisk von Neumann, 6 för hex."""
        return int(self.neighbor_idx.shape[1])

    def wrap_pos(self, x: float, y: float) -> tuple[float, float]:
        return x % float(self.width), y % float(self.height)

    def wrap_pos_inplace(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """
        Vektoriserad toroidal wrap av kontinuerliga positioner, in-place.

        Världens utsträckning i kontinuerligt rum är en geometrisk egenskap och
        hör därför hit. Anropare ska aldrig läsa grid.size för att wrappa själva
        — då flyttar geometrin ut ur Grid.
        """
        np.mod(xs, np.float32(self.width), out=xs)
        np.mod(ys, np.float32(self.height), out=ys)

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
        w = int(self.width)
        ix = int(x) % w
        iy = int(y) % int(self.height)
        return iy * w + ix

    def cell_of_many(self, xs: object, ys: object) -> np.ndarray:
        """
        Vektoriserad position -> cell-ID.

        Toroidal semantik: modulo först, golv sedan — den enda som är korrekt
        för negativa koordinater. Den skalära cell_of() trunkerar före modulo
        och avviker därför för negativa värden; se anmärkningen där.
        """
        w = int(self.width)
        xw = np.mod(np.asarray(xs, dtype=np.float32), np.float32(w))
        yw = np.mod(np.asarray(ys, dtype=np.float32), np.float32(self.height))
        return (yw.astype(np.int32, copy=False) * np.int32(w)
                + xw.astype(np.int32, copy=False)).astype(np.int32, copy=False)

    def rowcol_of(self, cell: int) -> tuple[int, int]:
        w = int(self.width)
        cell = int(cell) % int(self.n_cells)
        row, col = divmod(cell, w)
        return row, col

    def cell_from_rowcol(self, row: int, col: int) -> int:
        w = int(self.width)
        return (int(row) % int(self.height)) * w + (int(col) % w)

    def wrap_cell(self, cell: int) -> int:
        return int(cell) % int(self.n_cells)

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

    def distance(self, cell_a: int, cell_b: int) -> int:
        """
        Topologiskt cellavstånd på kvadratisk torus, mätt som Manhattan-avstånd
        med toroidal wrap.
        """
        ra, ca = self.rowcol_of(cell_a)
        rb, cb = self.rowcol_of(cell_b)

        dr = abs(rb - ra)
        dc = abs(cb - ca)

        dr = min(dr, int(self.height) - dr)
        dc = min(dc, int(self.width) - dc)

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
        s = float(self.width)
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