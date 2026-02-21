# world.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

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
    B_regen: float = 0.025    # 1/time (logistic rate)
    B_diff: float = 0.006     # diffusion coefficient (same algebra; tune)
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
    B_sense_K: float = 5e-4   # kg where perception ≈ 0.5

    # -------------------------
    # Carcass field C [kg per cell]
    # -------------------------
    C_decay: float = 0.005
    C_diff: float = 0.00
    C_unit_mass: float = 2.0   # legacy
    
    # --- Perception scaling ---
    C_sense_K: float = 5e-4   # kg where perception ≈ 0.5

    # -------------------------
    # Hazard field F [0..1]
    # -------------------------
    F_decay: float = 0.06
    F_diff: float = 0.004

    hazard_event_p: float = 0.0003
    hazard_event_amp: float = 0.25
    hazard_event_rad_min: int = 3
    hazard_event_rad_max: int = 6

    winter_hazard_scale: float = 0.0


@dataclass
class World:
    P: WorldParams

    def __post_init__(self) -> None:
        s = int(self.P.size)

        self.A = np.zeros((s, s), dtype=np.int32)     # occupancy (0=tomt, annars agent_id)
        self.B = np.zeros((s, s), dtype=np.float32)   # biomass [kg/cell]
        self.F = np.zeros((s, s), dtype=np.float32)   # hazard [0..1]
        self.C = np.zeros((s, s), dtype=np.float32)   # carcass [0..1] (tills vidare)

        self.t = 0.0

        # Precompute latitudinal profiles (depend only on y)
        ys = np.arange(s, dtype=np.float32)
        lat = np.float32(2.0) * (ys / np.float32(max(1, s - 1))) - np.float32(1.0)  # [-1, +1]
        abs_lat = np.abs(lat)

        Tmean_y = np.float32(self.P.T_eq) - np.float32(self.P.dT_pole) * (abs_lat ** np.float32(self.P.lat_p))
        Amp_y = np.float32(self.P.A_eq) + (np.float32(self.P.A_pole) - np.float32(self.P.A_eq)) * (
            abs_lat ** np.float32(self.P.amp_q)
        )

        self._lat = lat
        self._abs_lat = abs_lat
        self._Tmean_y = Tmean_y
        self._Amp_y = Amp_y

        # For debugging/inspection (last computed)
        self.Ty = np.zeros((s,), dtype=np.float32)    # temperature per row
        self.gy = np.ones((s,), dtype=np.float32)     # growth gate per row

        # initial ecology (B in kg/cell)
        BK = float(self.P.B_K)
        self.B.fill(np.float32(0.08 * BK))  # t.ex 8% av kapacitet

        for _ in range(30):
            # amp i kg: peak ~ 0.9*BK vid centrum
            self._add_blob(
                self.B,
                random.randrange(s),
                random.randrange(s),
                amp=float(0.9 * BK),
                rad=9,
                hi=BK,
            )

        for _ in range(7):
            self._add_blob(
                self.F,
                random.randrange(s),
                random.randrange(s),
                amp=0.8,
                rad=10,
                hi=1.0,
            )

    # -------------------------
    # Clips
    # -------------------------
    def _clip_B(self, B: np.ndarray) -> np.ndarray:
        return np.clip(B, 0.0, float(self.P.B_K)).astype(np.float32, copy=False)

    # -------------------------
    # Temperature / season helpers
    # -------------------------
    def _update_temperature(self) -> None:
        P = self.P
        year_len = float(P.year_len) if float(P.year_len) > 1e-9 else 1.0

        phase = 2.0 * math.pi * ((self.t % year_len) / year_len)
        phase -= float(P.season_phase0)

        s = np.float32(math.sin(phase))
        S_y = self._lat.astype(np.float32, copy=False) * s  # (size,)

        Ty = self._Tmean_y + self._Amp_y * S_y
        self.Ty = Ty

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
        if not hasattr(self, "Ty") or self.Ty.size == 0:
            return 0.0

        yf = float(y) % s
        y0 = int(math.floor(yf)) % s
        y1 = (y0 + 1) % s
        fy = yf - math.floor(yf)

        t0 = float(self.Ty[y0])
        t1 = float(self.Ty[y1])
        return (1.0 - fy) * t0 + fy * t1

    def rebuild_agent_layer(self, agents) -> None:
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
    def _add_blob(self, A: np.ndarray, cx: int, cy: int, amp: float, rad: int, hi: float = 1.0) -> None:
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
        blob = float(amp) * np.exp(-r2 / (2.0 * sigma2))  # same units as A

        A[mask] = np.clip(A[mask] + blob[mask], 0.0, float(hi)).astype(np.float32, copy=False)

    @staticmethod
    def _laplace(A: np.ndarray) -> np.ndarray:
        return (
            np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
            + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
            - 4.0 * A
        )

    def step(self) -> None:
        dt = float(self.P.dt)
        P = self.P

        self._update_temperature()
        T = self.temperature_field()  # (s,s) degC

        # --- Growth window G(T) (triangular around an optimum)
        G = np.zeros_like(T, dtype=np.float32)
        Tmin, Topt, Tmax = float(P.T_grow_min), float(P.T_grow_opt), float(P.T_grow_max)

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

        # logistic growth (units: kg/time)
        # NOTE: This keeps the old algebra; now B and B_K are kg/cell.
        growth = (float(P.B_regen) * G) * (1.0 - self.B / float(P.B_K)) * self.B

        # active dieoff / respiration / frost/heat stress (kg/time)
        wither = m * self.B

        dB = growth - wither + float(P.B_diff) * lapB
        self.B = self._clip_B(self.B + np.float32(dt) * dB)

        # --- Hazard field F: decay + diffusion + stochastic events (still 0..1)
        lapF = self._laplace(self.F)
        dF = (-float(P.F_decay) * self.F) + float(P.F_diff) * lapF
        self.F = _clip01(self.F + np.float32(dt) * dF).astype(np.float32, copy=False)

        hazard_p = float(P.hazard_event_p)
        if float(P.winter_hazard_scale) > 0.0:
            gbar = float(np.mean(self.gy))
            hazard_p *= (1.0 + float(P.winter_hazard_scale) * (1.0 - gbar))

        if hazard_p > 0.0 and random.random() < hazard_p:
            cx, cy = random.randrange(P.size), random.randrange(P.size)
            rad = random.randint(P.hazard_event_rad_min, P.hazard_event_rad_max)
            self._add_blob(self.F, cx, cy, amp=float(P.hazard_event_amp), rad=rad, hi=1.0)

        # --- Carcass field C: decay + diffusion (still 0..1)
        lapC = self._laplace(self.C)
        dC = (-float(P.C_decay) * self.C) + float(P.C_diff) * lapC
        self.C = _clip01(self.C + np.float32(dt) * dC).astype(np.float32, copy=False)

        # --- Germination / seeding
        seed_p = float(P.seed_p)
        if seed_p > 0.0:
            cx, cy = random.randrange(P.size), random.randrange(P.size)
            Gcy = float(G[cy, 0])  # local season factor at latitude
            seed_p_eff = seed_p * Gcy

            if seed_p_eff > 0.0 and random.random() < seed_p_eff:
                rad = random.randint(P.seed_rad_min, P.seed_rad_max)
                amp = float(P.seed_amp) * (0.25 + 0.75 * Gcy)  # kg peak amplitude
                self._add_blob(self.B, cx, cy, amp=amp, rad=rad, hi=float(P.B_K))

        self.t += dt

    # -------------------------
    # Sampling
    # -------------------------
    def sample_bilinear_many(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _sample_bilinear_many_layers(self.B, self.F, self.C, xs, ys, self.P.size)

    def sample_bilinear_many_BF(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        outB: Optional[np.ndarray] = None,
        outF: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _sample_bilinear_many_BF(self.B, self.F, xs, ys, self.P.size, outB=outB, outF=outF)

    def sample_bilinear(self, x: float, y: float) -> Tuple[float, float, float]:
        return sample_bilinear_scalar(self.B, self.F, self.C, x, y, self.P.size)

    # -------------------------
    # Consumption + carcass
    # -------------------------
    def _consume_bilinear_from(self, A: np.ndarray, x: float, y: float, amount: float, hi: float) -> float:
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
    
        A00 = float(A[y0, x0])
        A10 = float(A[y0, x1])
        A01 = float(A[y1, x0])
        A11 = float(A[y1, x1])
    
        pool = w00 * A00 + w10 * A10 + w01 * A01 + w11 * A11
        if pool <= 1e-12 or amount <= 0.0:
            return 0.0
    
        got = amount if amount < pool else pool
        frac = got / pool  # fraction of the *sampled* pool to remove
    
        # remove proportionally from each corner contribution => Aij *= (1-frac)
        keep = 1.0 - frac
        A[y0, x0] = np.float32(max(0.0, min(hi, A00 * keep)))
        A[y0, x1] = np.float32(max(0.0, min(hi, A10 * keep)))
        A[y1, x0] = np.float32(max(0.0, min(hi, A01 * keep)))
        A[y1, x1] = np.float32(max(0.0, min(hi, A11 * keep)))
    
        return got

    def consume_food(self, x: float, y: float, amount: float, prefer_carcass: bool = True) -> Tuple[float, float]:
        """
        amount: requested intake (kg if you pass it as kg).
        For now:
          - C is still [0..1] normalized, so "got_c" is in that normalized unit.
          - B is kg/cell, so "got_b" is kg.
        This mismatch is intentional until C is migrated to kg.
        Return: (total_got, got_c)
        """
        got_c = 0.0
        if prefer_carcass:
            got_c = self._consume_bilinear_from(self.C, x, y, amount, hi=1.0)
            amount = max(0.0, amount - got_c)

        got_b = self._consume_bilinear_from(self.B, x, y, amount, hi=float(self.P.B_K))
        return (got_c + got_b), got_c

    def add_carcass(self, x: float, y: float, amount: float, rad: int = 3) -> None:
        """
        Still writes into normalized C in [0..1].
        (När du gör C till kg: byt clamp/hi, och gör amount i kg.)
        """
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
    B: np.ndarray,
    F: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    size: int,
    outB: Optional[np.ndarray] = None,
    outF: Optional[np.ndarray] = None,
    tmp: Optional[np.ndarray] = None,  # scratch buffer (float32, same shape as xs)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assumes xs,ys are float32 and already wrapped to [0,size).
    B,F should be float32, C-contiguous.
    """
    s = int(size)

    x0 = xs.astype(np.int32, copy=False)
    y0 = ys.astype(np.int32, copy=False)

    x0 %= s
    y0 %= s

    fx = xs - x0
    fy = ys - y0
    fx1 = np.float32(1.0) - fx
    fy1 = np.float32(1.0) - fy

    x1 = x0 + 1
    y1 = y0 + 1
    x1[x1 == s] = 0
    y1[y1 == s] = 0

    B00 = B[y0, x0]
    B10 = B[y0, x1]
    B01 = B[y1, x0]
    B11 = B[y1, x1]

    F00 = F[y0, x0]
    F10 = F[y0, x1]
    F01 = F[y1, x0]
    F11 = F[y1, x1]

    if outB is None:
        outB = np.empty(xs.shape, dtype=np.float32)
    if outF is None:
        outF = np.empty(xs.shape, dtype=np.float32)
    if tmp is None:
        tmp = np.empty(xs.shape, dtype=np.float32)

    # --- B out ---
    np.multiply(B00, fx1, out=outB)
    outB += B10 * fx
    outB *= fy1

    np.multiply(B01, fx1, out=tmp)
    tmp += B11 * fx
    tmp *= fy
    outB += tmp

    # --- F out ---
    np.multiply(F00, fx1, out=outF)
    outF += F10 * fx
    outF *= fy1

    np.multiply(F01, fx1, out=tmp)
    tmp += F11 * fx
    tmp *= fy
    outF += tmp

    return outB, outF


def _sample_bilinear_many_layers(
    B: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    size: int,
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

    B00 = B[y0, x0]
    B10 = B[y0, x1]
    B01 = B[y1, x0]
    B11 = B[y1, x1]

    F00 = F[y0, x0]
    F10 = F[y0, x1]
    F01 = F[y1, x0]
    F11 = F[y1, x1]

    C00 = C[y0, x0]
    C10 = C[y0, x1]
    C01 = C[y1, x0]
    C11 = C[y1, x1]

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
    B: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    x: float,
    y: float,
    s: int,
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

    b00 = float(B[y0, x0])
    b10 = float(B[y0, x1])
    b01 = float(B[y1, x0])
    b11 = float(B[y1, x1])

    f00 = float(F[y0, x0])
    f10 = float(F[y0, x1])
    f01 = float(F[y1, x0])
    f11 = float(F[y1, x1])

    c00 = float(C[y0, x0])
    c10 = float(C[y0, x1])
    c01 = float(C[y1, x0])
    c11 = float(C[y1, x1])

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