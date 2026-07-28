from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
import math

_SQRT3 = math.sqrt(3.0)

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


# ---------------------------------------------------------------------------
# Hexagonal geometri
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HexGrid:
    """
    Hexagonal toroidal geometri med samma gränssnitt som Grid.

    **Layout.** Spetsig hexagon med radoffset: udda rader förskjutna en halv
    cellbredd åt höger. Alternativet, en romb med axialkoordinater som wrappar
    i q och r var för sig, är topologiskt enklare men skevar den kontinuerliga
    inbäddningen — att gå rakt uppåt och wrappa skulle förflytta en i sidled.
    Eftersom organismer rör sig i kontinuerligt rum vore det en osynlig drift
    utan biologisk motsvarighet.

    **Villkor.** Antalet rader måste vara jämnt. Halvcellsförskjutningen matchar
    annars inte vid sömmen, och radparitet — som grannrelationerna beror på —
    blir inkonsistent över wrap.

    **Skala.** Cellarean är 1, samma som den kvadratiska cellen. Det bevarar
    alla täthetsstorheter: flora per cell, näring per cell, kadaver per cell.
    För radoffset är arean per cell exakt `w * h`, och för spetsig hexagon är
    `h / w = sqrt(3) / 2`. Världen blir därmed inte kvadratisk i kontinuerliga
    enheter även när cellantalet är det.

    **Avstånd.** `distance()` använder kubkoordinater och prövar torusens nio
    translationskandidater. `cells_within()` byggs med bredden-först-sökning i
    grannmatrisen, vilket är korrekt på torus utan specialfall och ger serien
    1, 7, 19, 37 automatiskt. De två valideras mot varandra i tester.
    """

    size: int = 0
    width: int = 0
    height: int = 0

    n_cells: int = field(init=False, repr=False, compare=False)
    neighbor_idx: np.ndarray = field(init=False, repr=False, compare=False)
    neighbor_mask: np.ndarray = field(init=False, repr=False, compare=False)
    cell_lat: np.ndarray = field(init=False, repr=False, compare=False)
    cell_center_x: np.ndarray = field(init=False, repr=False, compare=False)
    cell_center_y: np.ndarray = field(init=False, repr=False, compare=False)

    # Cellgeometri: w * h = 1 ger cellarea 1; h / w = sqrt(3)/2 ger regelbunden
    # spetsig hexagon.
    COL_SPACING: ClassVar[float] = math.sqrt(2.0 / math.sqrt(3.0))
    ROW_SPACING: ClassVar[float] = math.sqrt(2.0 / math.sqrt(3.0)) * math.sqrt(3.0) / 2.0

    def __post_init__(self) -> None:
        w = int(self.width) if int(self.width) > 0 else int(self.size)
        h = int(self.height) if int(self.height) > 0 else int(self.size)
        if w <= 0 or h <= 0:
            raise ValueError("HexGrid kräver size, eller width och height")
        if h % 2 != 0:
            raise ValueError(
                f"HexGrid kräver jämnt radantal, fick height={h}. "
                "Udda radantal gör halvcellsförskjutningen inkonsistent över sömmen."
            )

        object.__setattr__(self, "width", w)
        object.__setattr__(self, "height", h)
        object.__setattr__(self, "size", w if w == h else 0)

        n = w * h
        cells = np.arange(n, dtype=np.int64)
        row = cells // w
        col = cells % w
        odd = (row & 1).astype(np.int64)

        # odd-r offset: udda rader förskjutna åt höger.
        #   jämn rad: NW=(r-1,c-1) NE=(r-1,c) SW=(r+1,c-1) SE=(r+1,c)
        #   udda rad: NW=(r-1,c)   NE=(r-1,c+1) SW=(r+1,c) SE=(r+1,c+1)
        def cid(rr, cc):
            return (rr % h) * w + (cc % w)

        west = cid(row, col - 1)
        east = cid(row, col + 1)
        nw = cid(row - 1, col - 1 + odd)
        ne = cid(row - 1, col + odd)
        sw = cid(row + 1, col - 1 + odd)
        se = cid(row + 1, col + odd)

        neighbor_idx = np.stack((west, east, nw, ne, sw, se), axis=1).astype(np.int32, copy=False)
        neighbor_mask = np.ones(neighbor_idx.shape, dtype=np.bool_)

        lat = (-np.cos(2.0 * np.pi * row.astype(np.float64) / float(h))).astype(np.float32)

        cx = ((col.astype(np.float64) + 0.5 + 0.5 * odd) * self.COL_SPACING).astype(np.float32)
        cy = ((row.astype(np.float64) + 0.5) * self.ROW_SPACING).astype(np.float32)

        object.__setattr__(self, "n_cells", int(n))
        object.__setattr__(self, "neighbor_idx", neighbor_idx)
        object.__setattr__(self, "neighbor_mask", neighbor_mask)
        object.__setattr__(self, "cell_lat", lat)
        object.__setattr__(self, "cell_center_x", cx)
        object.__setattr__(self, "cell_center_y", cy)

    # -- form och utsträckning -------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.height), int(self.width)

    @property
    def extent_x(self) -> float:
        return float(self.width) * self.COL_SPACING

    @property
    def extent_y(self) -> float:
        return float(self.height) * self.ROW_SPACING

    @property
    def neighbor_count(self) -> int:
        return 6

    def random_position(self, rng) -> tuple[float, float]:
        return (float(rng.uniform(0.0, self.extent_x)),
                float(rng.uniform(0.0, self.extent_y)))

    # -- wrap -------------------------------------------------------------

    def wrap_pos(self, x: float, y: float) -> tuple[float, float]:
        return x % self.extent_x, y % self.extent_y

    def wrap_pos_inplace(self, xs: np.ndarray, ys: np.ndarray) -> None:
        np.mod(xs, np.float32(self.extent_x), out=xs)
        np.mod(ys, np.float32(self.extent_y), out=ys)

    def wrap_cell(self, cell: int) -> int:
        return int(cell) % int(self.n_cells)

    def torus_delta_pos(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
        ex, ey = self.extent_x, self.extent_y
        dx = (x2 - x1) % ex
        dy = (y2 - y1) % ey
        if dx > 0.5 * ex:
            dx -= ex
        if dy > 0.5 * ey:
            dy -= ey
        return dx, dy

    # -- position <-> cell -------------------------------------------------

    def _axial_round(self, qf, rf):
        """Kubavrundning. Skalär eller array."""
        xf = np.asarray(qf, dtype=np.float64)
        zf = np.asarray(rf, dtype=np.float64)
        yf = -xf - zf
        rx, ry, rz = np.round(xf), np.round(yf), np.round(zf)
        dx, dy, dz = np.abs(rx - xf), np.abs(ry - yf), np.abs(rz - zf)
        fix_x = (dx > dy) & (dx > dz)
        fix_z = (~fix_x) & (dz > dy)
        rx = np.where(fix_x, -ry - rz, rx)
        rz = np.where(fix_z, -rx - ry, rz)
        return rx.astype(np.int64), rz.astype(np.int64)

    def _pos_to_rowcol(self, xs, ys):
        # Cellcentrum ligger en halv cell in från origo, så att hela världen
        # ryms i [0, extent). Axialformeln utgår från centrum i origo, därför
        # dras förskjutningen av före konverteringen.
        xs = (np.asarray(xs, dtype=np.float64) - 0.5 * self.COL_SPACING) % self.extent_x
        ys = (np.asarray(ys, dtype=np.float64) - 0.5 * self.ROW_SPACING) % self.extent_y
        R = self.COL_SPACING / math.sqrt(3.0)          # omkretsradie
        qf = (math.sqrt(3.0) / 3.0 * xs - ys / 3.0) / R
        rf = (2.0 / 3.0 * ys) / R
        q, r = self._axial_round(qf, rf)
        # axial -> odd-r offset
        col = q + ((r - (r & 1)) // 2)
        return r % int(self.height), col % int(self.width)

    def cell_of(self, x: float, y: float) -> int:
        """
        Position -> cell-ID, skalär.

        Egen implementation i ren Python i stället för att gå via
        _pos_to_rowcol(). Numpys overhead per anrop dominerar fullständigt för
        skalärer, och den här metoden anropas i storleksordningen hundratals
        gånger per tick i sensing och interaktion.
        """
        W = int(self.width)
        H = int(self.height)
        cw = self.COL_SPACING
        rh = self.ROW_SPACING

        xs = (float(x) - 0.5 * cw) % (W * cw)
        ys = (float(y) - 0.5 * rh) % (H * rh)

        R = cw / _SQRT3
        qf = (_SQRT3 / 3.0 * xs - ys / 3.0) / R
        rf = (2.0 / 3.0 * ys) / R

        yf = -qf - rf
        rx = round(qf)
        ry = round(yf)
        rz = round(rf)
        dx = abs(rx - qf)
        dy = abs(ry - yf)
        dz = abs(rz - rf)
        if dx > dy and dx > dz:
            rx = -ry - rz
        elif dz > dy:
            rz = -rx - ry

        q = int(rx)
        r = int(rz)
        col = q + ((r - (r & 1)) // 2)
        return (r % H) * W + (col % W)

    def cell_of_many(self, xs: object, ys: object) -> np.ndarray:
        r, c = self._pos_to_rowcol(xs, ys)
        return (r * np.int64(self.width) + c).astype(np.int32, copy=False)

    def rowcol_of(self, cell: int) -> tuple[int, int]:
        w = int(self.width)
        cell = int(cell) % int(self.n_cells)
        return divmod(cell, w)

    def cell_from_rowcol(self, row: int, col: int) -> int:
        w = int(self.width)
        return (int(row) % int(self.height)) * w + (int(col) % w)

    # -- topologi ----------------------------------------------------------

    def neighbors(self, cell: int) -> tuple[int, ...]:
        return tuple(int(v) for v in self.neighbor_idx[int(cell) % int(self.n_cells)])

    def _cube(self, cell: int):
        r, c = self.rowcol_of(cell)
        x = c - ((r - (r & 1)) // 2)
        z = r
        return x, -x - z, z

    def distance(self, cell_a: int, cell_b: int) -> int:
        """
        Hexavstånd på torus. Prövar de nio translationskandidaterna som
        genereras av världens wrap: W kolumner ger kubvektorn (W, -W, 0), och
        H rader ger (-H/2, -H/2, H) eftersom radoffseten förskjuter x med H/2
        över ett helt varv. Att H är jämnt är vad som gör det heltaligt.
        """
        W = int(self.width)
        H = int(self.height)
        ax, ay, az = self._cube(cell_a)
        bx, by, bz = self._cube(cell_b)
        dx0, dy0, dz0 = bx - ax, by - ay, bz - az

        best = None
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                dx = dx0 + i * W + j * (-(H // 2))
                dz = dz0 + j * H
                dy = -dx - dz
                d = (abs(dx) + abs(dy) + abs(dz)) // 2
                if best is None or d < best:
                    best = d
        return int(best)

    def distance_pos(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Euklidiskt avstånd i kontinuerligt rum under toroidal wrap."""
        dx, dy = self.torus_delta_pos(x1, y1, x2, y2)
        return math.hypot(dx, dy)

    def distance2_pos(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Kvadrerat avstånd i kontinuerligt rum. Undviker roten i heta vägar."""
        dx, dy = self.torus_delta_pos(x1, y1, x2, y2)
        return dx * dx + dy * dy

    def cells_within(self, cell: int, r: int) -> tuple[int, ...]:
        """
        Alla celler inom topologiskt avstånd <= r, via bredden-först-sökning i
        grannmatrisen. Ger 1, 7, 19, 37 … celler och är korrekt på torus utan
        specialfall vid sömmarna.
        """
        rr = int(r)
        if rr < 0:
            return ()
        start = int(cell) % int(self.n_cells)
        seen = {start}
        frontier = [start]
        out = [start]
        for _ in range(rr):
            nxt = []
            for c in frontier:
                for nb in self.neighbor_idx[c]:
                    nb = int(nb)
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
                        out.append(nb)
            frontier = nxt
            if not frontier:
                break
        return tuple(out)
