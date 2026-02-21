# world.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# -------------------------
# Parameters
# -------------------------
@dataclass
class WorldParams:
    size: int = 64
    dt: float = 0.02

    # -------------------------
    # Temperature / seasons
    # -------------------------
    year_len: float = 256.0

    # Mean temperature profile (latitudinal):
    # T_mean(y) = T_eq - dT_pole * |lat(y)|^lat_p
    T_eq: float = 30.0
    dT_pole: float = 30.0
    lat_p: float = 1.5

    # Seasonal amplitude profile:
    # A(y) = A_eq + (A_pole - A_eq) * |lat(y)|^amp_q
    A_eq: float = 3.0
    A_pole: float = 15.0
    amp_q: float = 1.5

    season_phase0: float = 0.0

    # Growth gating thresholds (degC): g(T) in [0,1]
    T0: float = 0.0
    T1: float = 20.0

    # -------------------------
    # Plant biomass field B [kg per cell]
    # -------------------------
    B_K: float = 1e-3          # kg/cell at carrying capacity (tune later)
    B_regen: float = 0.025     # 1/time (logistic rate)
    B_diff: float = 0.006      # diffusion coefficient
    B_wither_base: float = 0.005  # 1/time

    # Seeding (patch driver)
    seed_p: float = 0.0010
    seed_amp: float = 0.22        # kg amplitude for seeding blobs
    seed_rad_min: int = 2
    seed_rad_max: int = 5

    # Growth window (triangular)
    T_grow_min: float = 5.0
    T_grow_opt: float = 25.0
    T_grow_max: float = 35.0

    # Temperature-driven dieoff (wither)
    T_cold: float = 0.0
    cold_width: float = 8.0
    B_wither_cold: float = 0.060

    T_hot: float = 33.0
    hot_width: float = 6.0
    B_wither_hot: float = 0.020

    # --- Perception scaling (kg -> u in [0,1]) ---
    B_sense_K: float = 5e-4  # kg where perception ≈ 0.5

    # -------------------------
    # Carcass field C [kg per cell]
    # -------------------------
    C_K: float = 1e-3        # kg/cell "practical cap" for numerics (optional)
    C_decay: float = 0.005   # 1/time
    C_diff: float = 0.00     # diffusion coefficient

    # --- Perception scaling ---
    C_sense_K: float = 5e-4


# -------------------------
# World
# -------------------------
@dataclass
class World:
    P: WorldParams

    def __post_init__(self) -> None:
        s = int(self.P.size)

        # occupancy (0=empty else agent_id)
        self.A = np.zeros((s, s), dtype=np.int32)

        # ecology fields (kg/cell)
        self.B = np.zeros((s, s), dtype=np.float32)  # plant biomass
        self.C = np.zeros((s, s), dtype=np.float32)  # carcass biomass

        # time
        self.t = 0.0

        # Precompute latitudinal profiles (depend only on y)
        ys = np.arange(s, dtype=np.float32)
        lat = np.float32(2.0) * (ys / np.float32(max(1, s - 1))) - np.float32(1.0)  # [-1, +1]
        abs_lat = np.abs(lat)

        self._lat = lat
        self._abs_lat = abs_lat
        self._Tmean_y = np.float32(self.P.T_eq) - np.float32(self.P.dT_pole) * (abs_lat ** np.float32(self.P.lat_p))
        self._Amp_y = np.float32(self.P.A_eq) + (np.float32(self.P.A_pole) - np.float32(self.P.A_eq)) * (
            abs_lat ** np.float32(self.P.amp_q)
        )

        # last-computed profiles (debug/inspection)
        self.Ty = np.zeros((s,), dtype=np.float32)  # degC per row
        self.gy = np.ones((s,), dtype=np.float32)   # gate per row in [0,1]

        # initialize temperature profiles at t=0
        self._update_temperature()

        # initial ecology
        BK = float(self.P.B_K)
        self.B.fill(np.float32(0.08 * BK))

        for _ in range(30):
            self._add_blob(
                self.B,
                random.randrange(s),
                random.randrange(s),
                amp=float(0.9 * BK),
                rad=9,
                hi=BK,
            )

    # -------------------------
    # Temperature / season
    # -------------------------
    def _update_temperature(self) -> None:
        P = self.P
        year_len = float(P.year_len) if float(P.year_len) > 1e-9 else 1.0

        phase = 2.0 * math.pi * ((self.t % year_len) / year_len)
        phase -= float(P.season_phase0)

        s = np.float32(math.sin(phase))
        S_y = self._lat * s  # (size,)

        Ty = self._Tmean_y + self._Amp_y * S_y
        self.Ty = Ty.astype(np.float32, copy=False)

        T0 = float(P.T0)
        T1 = float(P.T1)
        if T1 <= T0 + 1e-9:
            gy = (Ty >= np.float32(T0)).astype(np.float32)
        else:
            gy = (Ty - np.float32(T0)) / np.float32(T1 - T0)
            gy = np.clip(gy, 0.0, 1.0).astype(np.float32, copy=False)

        self.gy = gy

    def temperature_field(self) -> np.ndarray:
        s = int(self.P.size)
        return np.broadcast_to(self.Ty[:, None], (s, s)).astype(np.float32, copy=False)

    def growth_gate_field(self) -> np.ndarray:
        s = int(self.P.size)
        return np.broadcast_to(self.gy[:, None], (s, s)).astype(np.float32, copy=False)

    def temperature_at(self, x: float, y: float) -> float:
        s = int(self.P.size)
        yf = float(y) % s
        y0 = int(math.floor(yf)) % s
        y1 = (y0 + 1) % s
        fy = yf - math.floor(yf)
        t0 = float(self.Ty[y0])
        t1 = float(self.Ty[y1])
        return (1.0 - fy) * t0 + fy * t1

    # -------------------------
    # Agent occupancy
    # -------------------------
    def rebuild_agent_layer(self, agents: Iterable) -> None:
        A = self.A
        A.fill(0)
        s = int(self.P.size)
        for ag in agents:
            if not ag.body.alive:
                continue
            ix = int(ag.x) % s
            iy = int(ag.y) % s
            A[iy, ix] = int(ag.id)

    # -------------------------
    # Ecology kernels
    # -------------------------
    @staticmethod
    def _laplace(A: np.ndarray) -> np.ndarray:
        return (
            np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
            + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
            - 4.0 * A
        )

    def _clip_B(self, B: np.ndarray) -> np.ndarray:
        return np.clip(B, 0.0, float(self.P.B_K)).astype(np.float32, copy=False)

    def _add_blob(self, A: np.ndarray, cx: int, cy: int, amp: float, rad: int, hi: float) -> None:
        """
        Add a gaussian-ish blob with peak amplitude `amp` (same units as A),
        clipped to [0, hi].
        """
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

        A[mask] = np.clip(A[mask] + blob[mask], 0.0, float(hi)).astype(np.float32, copy=False)

    def step(self) -> None:
        dt = float(self.P.dt)
        P = self.P

        self._update_temperature()
        T = self.temperature_field()  # (s,s) degC

        # --- Growth window G(T) (triangular around an optimum)
        Tmin, Topt, Tmax = float(P.T_grow_min), float(P.T_grow_opt), float(P.T_grow_max)
        G = np.zeros_like(T, dtype=np.float32)

        if Topt > Tmin + 1e-9:
            G = np.where((T >= Tmin) & (T < Topt), (T - Tmin) / (Topt - Tmin), G)
        if Tmax > Topt + 1e-9:
            G = np.where((T >= Topt) & (T <= Tmax), (Tmax - T) / (Tmax - Topt), G)
        G = np.clip(G, 0.0, 1.0).astype(np.float32, copy=False)

        # --- Wither / dieoff rate m(T) >= 0
        m = np.full_like(T, float(P.B_wither_base), dtype=np.float32)

        if float(P.B_wither_cold) > 0.0 and float(P.cold_width) > 1e-9:
            Sc = np.clip((float(P.T_cold) - T) / float(P.cold_width), 0.0, 1.0).astype(np.float32, copy=False)
            m += float(P.B_wither_cold) * Sc

        if float(P.B_wither_hot) > 0.0 and float(P.hot_width) > 1e-9:
            Sh = np.clip((T - float(P.T_hot)) / float(P.hot_width), 0.0, 1.0).astype(np.float32, copy=False)
            m += float(P.B_wither_hot) * Sh

        # --- Plants: growth - wither + diffusion
        lapB = self._laplace(self.B)

        growth = (float(P.B_regen) * G) * (1.0 - self.B / float(P.B_K)) * self.B
        wither = m * self.B

        dB = growth - wither + float(P.B_diff) * lapB
        self.B = self._clip_B(self.B + np.float32(dt) * dB)

        # --- Carcass field C [kg/cell]: decay + diffusion
        lapC = self._laplace(self.C)
        dC = (-float(P.C_decay) * self.C) + float(P.C_diff) * lapC
        self.C = (self.C + np.float32(dt) * np.float32(dC)).astype(np.float32, copy=False)
        self.C = np.maximum(self.C, 0.0).astype(np.float32, copy=False)

        # --- Germination / seeding
        seed_p = float(P.seed_p)
        if seed_p > 0.0:
            cx, cy = random.randrange(P.size), random.randrange(P.size)
            Gcy = float(G[cy, 0])  # season factor at latitude row
            seed_p_eff = seed_p * Gcy

            if seed_p_eff > 0.0 and random.random() < seed_p_eff:
                rad = random.randint(P.seed_rad_min, P.seed_rad_max)
                amp = float(P.seed_amp) * (0.25 + 0.75 * Gcy)  # kg peak amplitude
                self._add_blob(self.B, cx, cy, amp=amp, rad=rad, hi=float(P.B_K))

        self.t += dt

    # -------------------------
    # Sampling (renodlad)
    # -------------------------
    def sample(self, x: float, y: float) -> Tuple[float, float]:
        """Bilinear sampling at a single point. Returns (B_kg, C_kg)."""
        return _bilinear_scalar_BC(self.B, self.C, x, y, int(self.P.size))

    def sample_many(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        outB: Optional[np.ndarray] = None,
        outC: Optional[np.ndarray] = None,
        tmp: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized bilinear sampling for many points.
        Returns (B_array, C_array), both float32.
        """
        return _bilinear_many_BC(
            self.B, self.C,
            xs, ys,
            int(self.P.size),
            outB=outB, outC=outC, tmp=tmp,
        )

    # -------------------------
    # Consumption + carcass
    # -------------------------
    def _consume_bilinear_from(self, A: np.ndarray, x: float, y: float, amount: float) -> float:
        """
        Bilinear consume from field A (kg/cell).
        Removes up to `amount` kg from the bilinear-sampled pool around (x,y),
        updating the four corner cells. Returns `got` in kg.
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0

        s = int(self.P.size)
        xf = float(x)
        yf = float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            return 0.0

        xf %= s
        yf %= s

        x0 = int(math.floor(xf))
        y0 = int(math.floor(yf))
        x1 = x0 + 1
        y1 = y0 + 1
        if x1 == s:
            x1 = 0
        if y1 == s:
            y1 = 0

        fx = xf - math.floor(xf)
        fy = yf - math.floor(yf)

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        a00 = float(A[y0, x0])
        a10 = float(A[y0, x1])
        a01 = float(A[y1, x0])
        a11 = float(A[y1, x1])

        pool = w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11
        if not math.isfinite(pool) or pool <= 1e-12:
            return 0.0

        got = amt if amt < pool else pool
        frac = got / pool

        d00 = frac * w00 * a00
        d10 = frac * w10 * a10
        d01 = frac * w01 * a01
        d11 = frac * w11 * a11

        if d00 > a00: d00 = a00
        if d10 > a10: d10 = a10
        if d01 > a01: d01 = a01
        if d11 > a11: d11 = a11

        A[y0, x0] = np.float32(a00 - d00)
        A[y0, x1] = np.float32(a10 - d10)
        A[y1, x0] = np.float32(a01 - d01)
        A[y1, x1] = np.float32(a11 - d11)

        return float(got)

    def consume_food(self, x: float, y: float, amount: float, prefer_carcass: bool = True) -> Tuple[float, float]:
        """
        Consume up to `amount` kg total. Returns (got_total_kg, got_carcass_kg).
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0

        got_c = 0.0
        if prefer_carcass:
            got_c = float(self._consume_bilinear_from(self.C, x, y, amt))
            amt = max(0.0, amt - got_c)

        got_b = float(self._consume_bilinear_from(self.B, x, y, amt))
        return got_b + got_c, got_c

    def add_carcass(self, x: float, y: float, amount_kg: float, rad: int = 3) -> None:
        """
        Add carcass mass to C (kg/cell). amount_kg is TOTAL kg deposited.
        """
        s = int(self.P.size)
        amt = float(amount_kg)
        if not math.isfinite(amt) or amt <= 0.0:
            return

        r = int(rad)
        if r < 1:
            r = 1

        cx = int(round(x)) % s
        cy = int(round(y)) % s

        sigma = max(0.75, 0.5 * r)
        inv2sig2 = 1.0 / (2.0 * sigma * sigma)

        wsum = 0.0
        weights: list[tuple[int, int, float]] = []
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
            self.C[cy, cx] = np.float32(float(self.C[cy, cx]) + amt)
            return

        scale = amt / wsum
        for dx, dy, w in weights:
            ix = (cx + dx) % s
            iy = (cy + dy) % s
            self.C[iy, ix] = np.float32(float(self.C[iy, ix]) + scale * w)


# -------------------------
# Sampling kernels (private)
# -------------------------
def _bilinear_scalar_BC(B: np.ndarray, C: np.ndarray, x: float, y: float, s: int) -> Tuple[float, float]:
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
    c00 = float(C[y0, x0]); c10 = float(C[y0, x1]); c01 = float(C[y1, x0]); c11 = float(C[y1, x1])

    b0 = b00 * fx1 + b10 * fx
    b1 = b01 * fx1 + b11 * fx
    Bv = b0 * fy1 + b1 * fy

    c0 = c00 * fx1 + c10 * fx
    c1 = c01 * fx1 + c11 * fx
    Cv = c0 * fy1 + c1 * fy

    return Bv, Cv


def _bilinear_many_BC(
    B: np.ndarray,
    C: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    s: int,
    outB: Optional[np.ndarray] = None,
    outC: Optional[np.ndarray] = None,
    tmp: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized bilinear sampling for two layers (B,C).
    xs,ys can be any float dtype; will be treated as float32 internally.
    """
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have same shape")

    # wrap to [0,s)
    xs = np.mod(xs, np.float32(s))
    ys = np.mod(ys, np.float32(s))

    x0 = xs.astype(np.int32, copy=False)
    y0 = ys.astype(np.int32, copy=False)

    fx = xs - x0
    fy = ys - y0
    fx1 = np.float32(1.0) - fx
    fy1 = np.float32(1.0) - fy

    x1 = x0 + 1
    y1 = y0 + 1
    x1[x1 == s] = 0
    y1[y1 == s] = 0

    if outB is None:
        outB = np.empty(xs.shape, dtype=np.float32)
    if outC is None:
        outC = np.empty(xs.shape, dtype=np.float32)
    if tmp is None:
        tmp = np.empty(xs.shape, dtype=np.float32)

    # ---- B ----
    B00 = B[y0, x0]; B10 = B[y0, x1]; B01 = B[y1, x0]; B11 = B[y1, x1]

    np.multiply(B00, fx1, out=outB)    # outB = B00*fx1
    outB += B10 * fx                   # outB = B00*fx1 + B10*fx
    outB *= fy1                        # outB *= fy1

    np.multiply(B01, fx1, out=tmp)     # tmp = B01*fx1
    tmp += B11 * fx                    # tmp = B01*fx1 + B11*fx
    tmp *= fy                          # tmp *= fy
    outB += tmp                        # outB += tmp

    # ---- C ----
    C00 = C[y0, x0]; C10 = C[y0, x1]; C01 = C[y1, x0]; C11 = C[y1, x1]

    np.multiply(C00, fx1, out=outC)
    outC += C10 * fx
    outC *= fy1

    np.multiply(C01, fx1, out=tmp)
    tmp += C11 * fx
    tmp *= fy
    outC += tmp

    return outB, outC