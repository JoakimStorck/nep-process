from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Grid:
    size: int

    def wrap_pos(self, x: float, y: float) -> tuple[float, float]:
        s = float(self.size)
        return x % s, y % s

    def cell_of(self, x: float, y: float) -> int:
        s = int(self.size)
        ix = int(x) % s
        iy = int(y) % s
        return iy * s + ix

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
        import numpy as np

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