# world.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


@dataclass
class WorldParams:
    size: int = 64
    dt: float = 0.02

    # -------------------------
    # Temperature / seasons
    # -------------------------
    # One "year" in simulation time units
    year_len: float = 256.0

    # Mean temperature profile (latitudinal):
    # T_mean(y) = T_eq - dT_pole * |lat(y)|^lat_p
    T_eq: float = 30.0          # mean at equator
    dT_pole: float = 30.0       # equator->pole mean drop (=> mean at poles ~ 0)
    lat_p: float = 1.5          # shape of mean profile

    # Seasonal amplitude profile:
    # A(y) = A_eq + (A_pole - A_eq) * |lat(y)|^amp_q
    A_eq: float = 3.0
    A_pole: float = 15.0
    amp_q: float = 1.5

    # Seasonal phase offset (radians). Controls what t=0 means.
    # Example: 0 -> sin phase starts at 0; tweak if you want "mid-summer at t=0".
    season_phase0: float = 0.0

    # Growth gating thresholds (degC): g(T) in [0,1]
    # g(T)=0 for T<=T0, g(T)=1 for T>=T1 (linear in between)
    T0: float = 0.0
    T1: float = 20.0

    # -------------------------
    # Plant biomass field B (food)
    # -------------------------
    B_regen: float = 0.060
    B_K: float = 1.0
    B_diff: float = 0.050

    # Germination / ecology forcing (blob injections)
    seed_p: float = 0.00035
    seed_amp: float = 0.55
    seed_rad_min: int = 3
    seed_rad_max: int = 7

    # -------------------------
    # Hazard field F
    # -------------------------
    F_decay: float = 0.010
    F_diff: float = 0.030

    hazard_event_p: float = 0.0012
    hazard_event_amp: float = 0.70
    hazard_event_rad_min: int = 5
    hazard_event_rad_max: int = 12

    # Optional: make hazards more frequent in cold seasons/latitudes (0 disables)
    winter_hazard_scale: float = 0.0  # e.g. 1.0 -> up to +100% when g(T)=0

    # -------------------------
    # Carcass field C
    # -------------------------
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

        # internal world time (so step() doesn't need args)
        self.t = 0.0

        # Precompute latitudinal profiles (depend only on y)
        ys = np.arange(s, dtype=np.float32)
        lat = 2.0 * (ys / np.float32(max(1, s - 1))) - 1.0  # [-1, +1]
        abs_lat = np.abs(lat)

        hemi_south = (lat < 0.0).astype(np.float32)  # 1 in south, 0 otherwise

        # mean temp profile: equator -> poles
        Tmean_y = np.float32(self.P.T_eq) - np.float32(self.P.dT_pole) * (abs_lat ** np.float32(self.P.lat_p))

        # amplitude profile: small at equator, larger at poles
        Amp_y = np.float32(self.P.A_eq) + (np.float32(self.P.A_pole) - np.float32(self.P.A_eq)) * (
            abs_lat ** np.float32(self.P.amp_q)
        )

        self._lat = lat                 # (s,)
        self._abs_lat = abs_lat         # (s,)
        self._hemi_south = hemi_south   # (s,) float32 0/1
        self._Tmean_y = Tmean_y         # (s,)
        self._Amp_y = Amp_y             # (s,)

        # For debugging/inspection (last computed)
        self.Ty = np.zeros((s,), dtype=np.float32)     # temperature per row
        self.gy = np.ones((s,), dtype=np.float32)      # growth gate per row

        # initial ecology
        self.B.fill(np.float32(0.08))

        for _ in range(30):
            self._add_blob(self.B, random.randrange(s), random.randrange(s), amp=0.9, rad=9)

        for _ in range(7):
            self._add_blob(self.F, random.randrange(s), random.randrange(s), amp=0.8, rad=10)

    # -------------------------
    # Temperature / season helpers
    # -------------------------
    def _update_temperature(self) -> None:
        """
        Updates:
          - self.Ty: temperature per y-row
          - self.gy: growth gating g(T) per y-row in [0,1]
        Uses:
          - equator at center
          - mean temperature decreases towards poles
          - seasonal sinusoid in anti-phase between hemispheres
        """
        P = self.P
        year_len = float(P.year_len)
        if year_len <= 1e-9:
            year_len = 1.0

        # phase in [0, 2pi)
        phase = 2.0 * math.pi * ((self.t % year_len) / year_len)
        phase -= float(P.season_phase0)

        # anti-phase in south hemisphere (add pi where lat<0)
        # We use hemi_south as 0/1 to avoid branching
        S_y = np.sin(np.float32(phase) + np.float32(math.pi) * self._hemi_south).astype(np.float32, copy=False)

        Ty = self._Tmean_y + self._Amp_y * S_y
        self.Ty = Ty

        # growth gate: 0 below T0, 1 above T1, linear between
        T0 = float(P.T0)
        T1 = float(P.T1)
        if T1 <= T0 + 1e-9:
            # degenerate: hard threshold at T0
            gy = (Ty >= np.float32(T0)).astype(np.float32)
        else:
            gy = (Ty - np.float32(T0)) / np.float32(T1 - T0)
            gy = np.clip(gy, 0.0, 1.0).astype(np.float32, copy=False)

        self.gy = gy

    def temperature_field(self) -> np.ndarray:
        """Returns temperature as (size,size) float32 via broadcasting from Ty."""
        s = int(self.P.size)
        return np.broadcast_to(self.Ty[:, None], (s, s)).astype(np.float32, copy=False)

    def growth_gate_field(self) -> np.ndarray:
        """Returns growth gate g(T) as (size,size) float32 via broadcasting from gy."""
        s = int(self.P.size)
        return np.broadcast_to(self.gy[:, None], (s, s)).astype(np.float32, copy=False)

    # -------------------------
    # Ecology kernels
    # -------------------------
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

        # --- Update seasonal temperature and growth gating (per row)
        self._update_temperature()
        g = self.growth_gate_field()  # (s,s) float32 in [0,1]

        # --- Plants: logistic growth + diffusion, modulated by temperature gate g(T)
        lapB = self._laplace(self.B)
        # logistic term multiplied by g: no growth when g=0
        dB = (self.P.B_regen * g) * (1.0 - self.B / self.P.B_K) * self.B + self.P.B_diff * lapB
        self.B = _clip01(self.B + np.float32(dt) * dB).astype(np.float32, copy=False)

        # --- Hazard: decay + diffusion + stochastic events
        lapF = self._laplace(self.F)
        dF = (-self.P.F_decay * self.F) + self.P.F_diff * lapF
        self.F = _clip01(self.F + np.float32(dt) * dF).astype(np.float32, copy=False)

        # optional winter scaling: more hazard when growth gate is low (cold)
        hazard_p = float(self.P.hazard_event_p)
        if self.P.winter_hazard_scale > 0.0:
            # use global mean gate as a simple proxy for "winter severity"
            gbar = float(np.mean(self.gy))
            hazard_p *= (1.0 + float(self.P.winter_hazard_scale) * (1.0 - gbar))

        if hazard_p > 0.0 and random.random() < hazard_p:
            cx, cy = random.randrange(self.P.size), random.randrange(self.P.size)
            rad = random.randint(self.P.hazard_event_rad_min, self.P.hazard_event_rad_max)
            self._add_blob(self.F, cx, cy, amp=self.P.hazard_event_amp, rad=rad)

        # --- Carcass: decay + diffusion
        lapC = self._laplace(self.C)
        dC = (-self.P.C_decay * self.C) + self.P.C_diff * lapC
        self.C = _clip01(self.C + np.float32(dt) * dC).astype(np.float32, copy=False)

        # --- Germination (blob injections), modulated by local temperature gate
        # We keep a single-event Bernoulli per step, but accept it based on local row gate.
        if self.P.seed_p > 0.0 and random.random() < self.P.seed_p:
            cx, cy = random.randrange(self.P.size), random.randrange(self.P.size)
            # accept with probability g at that latitude -> no germination in winter/poles if g~0
            if random.random() < float(self.gy[cy]):
                rad = random.randint(self.P.seed_rad_min, self.P.seed_rad_max)
                self._add_blob(self.B, cx, cy, amp=self.P.seed_amp, rad=rad)

        # advance world time
        self.t += dt

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
        if x1 == s:
            x1 = 0
        if y1 == s:
            y1 = 0

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
    if x1 == s:
        x1 = 0
    if y1 == s:
        y1 = 0

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

