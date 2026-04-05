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