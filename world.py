# world.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class WorldParams:
    size: int = 64
    dt: float = 0.02

    # --- Plant biomass field B (food)
    B_regen: float = 0.060
    B_K: float = 1.0
    B_diff: float = 0.050

    # Germination / ecology forcing
    seed_p: float = 0.00035
    seed_amp: float = 0.55
    seed_rad_min: int = 3
    seed_rad_max: int = 7

    # --- Hazard field F
    F_decay: float = 0.010
    F_diff: float = 0.030

    hazard_event_p: float = 0.0012
    hazard_event_amp: float = 0.70
    hazard_event_rad_min: int = 5
    hazard_event_rad_max: int = 12

    # --- Carcass field C
    C_decay: float = 0.020
    C_diff: float = 0.015


@dataclass
class World:
    P: WorldParams

    def __post_init__(self) -> None:
        s = int(self.P.size)
        self.B = np.zeros((s, s), dtype=np.float32)
        self.F = np.zeros((s, s), dtype=np.float32)
        self.C = np.zeros((s, s), dtype=np.float32)

        # initial ecology
        self.B.fill(np.float32(0.08))

        for _ in range(30):
            self._add_blob(self.B, random.randrange(s), random.randrange(s), amp=0.9, rad=9)

        for _ in range(7):
            self._add_blob(self.F, random.randrange(s), random.randrange(s), amp=0.8, rad=10)

    def _add_blob(self, A: np.ndarray, cx: int, cy: int, amp: float, rad: int) -> None:
        s = int(self.P.size)
        rr = float(rad * rad)

        ys = np.arange(s, dtype=np.float32)
        xs = np.arange(s, dtype=np.float32)

        dy = ((ys - float(cy) + s / 2) % s) - s / 2
        dx = ((xs - float(cx) + s / 2) % s) - s / 2
        DX, DY = np.meshgrid(dx, dy)

        r2 = DX * DX + DY * DY
        mask = r2 <= rr
        sigma2 = max(rr / 4.0, 1e-6)
        blob = float(amp) * np.exp(-r2 / (2.0 * sigma2))
        A[mask] = np.clip(A[mask] + blob[mask], 0.0, 1.0)

    @staticmethod
    def _laplace(A: np.ndarray) -> np.ndarray:
        return (
            np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
            + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
            - 4.0 * A
        )

    def step(self) -> None:
        dt = float(self.P.dt)

        # Plants: logistic growth + diffusion
        lapB = self._laplace(self.B)
        dB = self.P.B_regen * (1.0 - self.B / self.P.B_K) * self.B + self.P.B_diff * lapB
        self.B = np.clip(self.B + dt * dB, 0.0, 1.0)

        # Hazard: decay + diffusion + stochastic events
        lapF = self._laplace(self.F)
        dF = (-self.P.F_decay * self.F) + self.P.F_diff * lapF
        self.F = np.clip(self.F + dt * dF, 0.0, 1.0)

        if self.P.hazard_event_p > 0.0 and random.random() < self.P.hazard_event_p:
            cx, cy = random.randrange(self.P.size), random.randrange(self.P.size)
            rad = random.randint(self.P.hazard_event_rad_min, self.P.hazard_event_rad_max)
            self._add_blob(self.F, cx, cy, amp=self.P.hazard_event_amp, rad=rad)

        # Carcass: decay + diffusion
        lapC = self._laplace(self.C)
        dC = (-self.P.C_decay * self.C) + self.P.C_diff * lapC
        self.C = np.clip(self.C + dt * dC, 0.0, 1.0)

        # Germination
        if self.P.seed_p > 0.0 and random.random() < self.P.seed_p:
            cx, cy = random.randrange(self.P.size), random.randrange(self.P.size)
            rad = random.randint(self.P.seed_rad_min, self.P.seed_rad_max)
            self._add_blob(self.B, cx, cy, amp=self.P.seed_amp, rad=rad)

    # -------------------------
    # Sampling
    # -------------------------
    def sample_bilinear_many(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _sample_bilinear_many_layers(self.B, self.F, self.C, xs, ys, self.P.size)

    def sample_bilinear_many_BF(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        return _sample_bilinear_many_BF(self.B, self.F, xs, ys, self.P.size)

    def sample_bilinear(self, x: float, y: float) -> Tuple[float, float, float]:
        return sample_bilinear_scalar(self.B, self.F, self.C, x, y, self.P.size)

    # -------------------------
    # Consumption + carcass
    # -------------------------
    def _consume_bilinear_from(self, A: np.ndarray, x: float, y: float, amount: float) -> float:
        s = int(self.P.size)

        x0 = int(math.floor(x)) % s
        y0 = int(math.floor(y)) % s
        x1 = x0 + 1
        y1 = y0 + 1
        if x1 == s: x1 = 0
        if y1 == s: y1 = 0

        fx = float(x - math.floor(x))
        fy = float(y - math.floor(y))

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        c00 = w00 * float(A[y0, x0])
        c10 = w10 * float(A[y0, x1])
        c01 = w01 * float(A[y1, x0])
        c11 = w11 * float(A[y1, x1])

        pool = c00 + c10 + c01 + c11
        if pool <= 1e-12 or amount <= 0.0:
            return 0.0

        got = amount if amount < pool else pool
        frac = got / pool

        # apply proportional removal
        if c00 > 0.0:
            A[y0, x0] = float(np.clip(float(A[y0, x0]) - (frac * c00) / w00, 0.0, 1.0))
        if c10 > 0.0:
            A[y0, x1] = float(np.clip(float(A[y0, x1]) - (frac * c10) / w10, 0.0, 1.0))
        if c01 > 0.0:
            A[y1, x0] = float(np.clip(float(A[y1, x0]) - (frac * c01) / w01, 0.0, 1.0))
        if c11 > 0.0:
            A[y1, x1] = float(np.clip(float(A[y1, x1]) - (frac * c11) / w11, 0.0, 1.0))

        return got

    def consume_food(self, x: float, y: float, amount: float, prefer_carcass: bool = True) -> Tuple[float, float]:
        got_c = 0.0
        if prefer_carcass:
            got_c = self._consume_bilinear_from(self.C, x, y, amount)
            amount = max(0.0, amount - got_c)
        got_b = self._consume_bilinear_from(self.B, x, y, amount)
        return (got_c + got_b), got_c

    def add_carcass(self, x: float, y: float, amount: float, rad: int = 3) -> None:
        s = int(self.P.size)

        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return
        amt = max(0.0, min(1.0, amt))

        r = int(rad)
        if r < 1:
            r = 1

        cx = int(round(x)) % s
        cy = int(round(y)) % s

        sigma = max(0.75, 0.5 * r)
        inv2sig2 = 1.0 / (2.0 * sigma * sigma)

        wsum = 0.0
        weights = []
        rr = float(r * r)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d2 = float(dx * dx + dy * dy)
                if d2 > rr:
                    continue
                w = math.exp(-d2 * inv2sig2)
                weights.append((dx, dy, w))
                wsum += w

        if wsum <= 1e-12:
            self.C[cy, cx] = float(min(1.0, self.C[cy, cx] + amt))
            return

        scale = amt / wsum
        for dx, dy, w in weights:
            ix = (cx + dx) % s
            iy = (cy + dy) % s
            self.C[iy, ix] = float(min(1.0, self.C[iy, ix] + scale * w))


# -------------------------
# Sampling kernels
# -------------------------
def _sample_bilinear_many_BF(
    B: np.ndarray, F: np.ndarray,
    xs: np.ndarray, ys: np.ndarray,
    size: int
) -> Tuple[np.ndarray, np.ndarray]:
    s = int(size)

    x0 = xs.astype(np.int32, copy=False)
    y0 = ys.astype(np.int32, copy=False)

    fx = xs - x0
    fy = ys - y0
    fx1 = np.float32(1.0) - fx
    fy1 = np.float32(1.0) - fy

    x0 = x0 % s
    y0 = y0 % s
    x1 = x0 + 1
    y1 = y0 + 1
    x1 = np.where(x1 == s, 0, x1)
    y1 = np.where(y1 == s, 0, y1)

    B00 = B[y0, x0]; B10 = B[y0, x1]; B01 = B[y1, x0]; B11 = B[y1, x1]
    F00 = F[y0, x0]; F10 = F[y0, x1]; F01 = F[y1, x0]; F11 = F[y1, x1]

    B0 = B00 * fx1 + B10 * fx
    B1 = B01 * fx1 + B11 * fx
    Bout = B0 * fy1 + B1 * fy

    F0 = F00 * fx1 + F10 * fx
    F1 = F01 * fx1 + F11 * fx
    Fout = F0 * fy1 + F1 * fy

    return Bout.astype(np.float32, copy=False), Fout.astype(np.float32, copy=False)


def _sample_bilinear_many_layers(
    B: np.ndarray, F: np.ndarray, C: np.ndarray,
    xs: np.ndarray, ys: np.ndarray,
    size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = int(size)
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have same shape")

    x0 = xs.astype(np.int32, copy=False)
    y0 = ys.astype(np.int32, copy=False)

    fx = xs - x0
    fy = ys - y0
    fx1 = np.float32(1.0) - fx
    fy1 = np.float32(1.0) - fy

    x0 = x0 % s
    y0 = y0 % s
    x1 = x0 + 1
    y1 = y0 + 1
    x1 = np.where(x1 == s, 0, x1)
    y1 = np.where(y1 == s, 0, y1)

    B00 = B[y0, x0]; B10 = B[y0, x1]; B01 = B[y1, x0]; B11 = B[y1, x1]
    F00 = F[y0, x0]; F10 = F[y0, x1]; F01 = F[y1, x0]; F11 = F[y1, x1]
    C00 = C[y0, x0]; C10 = C[y0, x1]; C01 = C[y1, x0]; C11 = C[y1, x1]

    B0 = B00 * fx1 + B10 * fx
    B1 = B01 * fx1 + B11 * fx
    Bout = B0 * fy1 + B1 * fy

    F0 = F00 * fx1 + F10 * fx
    F1 = F01 * fx1 + F11 * fx
    Fout = F0 * fy1 + F1 * fy

    C0 = C00 * fx1 + C10 * fx
    C1 = C01 * fx1 + C11 * fx
    Cout = C0 * fy1 + C1 * fy

    return (
        Bout.astype(np.float32, copy=False),
        Fout.astype(np.float32, copy=False),
        Cout.astype(np.float32, copy=False),
    )


def sample_bilinear_scalar(
    B: np.ndarray, F: np.ndarray, C: np.ndarray,
    x: float, y: float,
    s: int
) -> Tuple[float, float, float]:
    xf = float(x) % s
    yf = float(y) % s

    x0 = int(xf)
    y0 = int(yf)
    x1 = x0 + 1
    y1 = y0 + 1
    if x1 == s: x1 = 0
    if y1 == s: y1 = 0

    fx = xf - x0
    fy = yf - y0
    fx1 = 1.0 - fx
    fy1 = 1.0 - fy

    b00 = float(B[y0, x0]); b10 = float(B[y0, x1]); b01 = float(B[y1, x0]); b11 = float(B[y1, x1])
    f00 = float(F[y0, x0]); f10 = float(F[y0, x1]); f01 = float(F[y1, x0]); f11 = float(F[y1, x1])
    c00 = float(C[y0, x0]); c10 = float(C[y0, x1]); c01 = float(C[y1, x0]); c11 = float(C[y1, x1])

    b0 = b00 * fx1 + b10 * fx
    b1 = b01 * fx1 + b11 * fx
    Bv = b0 * fy1 + b1 * fy

    f0 = f00 * fx1 + f10 * fx
    f1 = f01 * fx1 + f11 * fx
    Fv = f0 * fy1 + f1 * fy

    c0 = c00 * fx1 + c10 * fx
    c1 = c01 * fx1 + c11 * fx
    Cv = c0 * fy1 + c1 * fy

    return Bv, Fv, Cv