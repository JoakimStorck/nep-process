# world.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _numba = None
    _NUMBA_AVAILABLE = False

from abiotic import step_carcass_field
from grid import Grid

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

   
# -------------------------
# Parameters
# -------------------------
@dataclass
class WorldParams:
    size: int = 64
    dt: float = 0.02

    # Flora mass scale used by Population for normalization / initialization
    B_K: float = 5e-4

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
    # Carcass field C [kg per cell]
    # -------------------------
    C_K: float = 1e-3        # kg/cell "practical cap" for numerics (optional)
    C_decay: float = 0.077   # 1% kvar efter 60s: ln(100)/60 ≈ 0.077 (var 0.03 → halveringstid 23s)
    C_diff: float = 0.00     # diffusion coefficient

    # --- Perception scaling ---
    C_sense_K: float = 5e-4

    # Energy densities for open-system ledger diagnostics (J/kg)
    E_plant_J_per_kg: float = 4.0e6
    E_carcass_J_per_kg: float = 7.0e6


# -------------------------
# World
# -------------------------
@dataclass
class World:
    WP: WorldParams
    grid: Grid = field(init=False)    

    def __post_init__(self) -> None:
        s = int(self.WP.size)
        self.grid = Grid(size=s)

        self.sample_flora_local_hook = None
        self.sample_flora_rays_hook = None        
        self.consume_food_hook = None

        # occupancy cache for current fauna pipeline
        self.A = np.zeros((s, s), dtype=np.int32)
        self._agent_by_id: dict[int, object] = {}        
        self.C = np.zeros((s, s), dtype=np.float32)  # carcass biomass

        # time
        self.t = 0.0
        self.last_flux = {
            "dM_growth": 0.0,
            "dM_wither": 0.0,
            "dM_decay": 0.0,
            "E_in_growth": 0.0,
            "E_loss_wither": 0.0,
            "E_loss_decay": 0.0,
        }

        # Precompute latitudinal profiles (depend only on y)
        ys = np.arange(s, dtype=np.float32)
        lat = np.float32(2.0) * (ys / np.float32(max(1, s - 1))) - np.float32(1.0)  # [-1, +1]
        abs_lat = np.abs(lat)

        self._lat = lat
        self._abs_lat = abs_lat
        self._Tmean_y = np.float32(self.WP.T_eq) - np.float32(self.WP.dT_pole) * (abs_lat ** np.float32(self.WP.lat_p))
        self._Amp_y = np.float32(self.WP.A_eq) + (np.float32(self.WP.A_pole) - np.float32(self.WP.A_eq)) * (
            abs_lat ** np.float32(self.WP.amp_q)
        )

        # last-computed profiles (debug/inspection)
        self.Ty = np.zeros((s,), dtype=np.float32)  # degC per row
        self.gy = np.ones((s,), dtype=np.float32)   # gate per row in [0,1]

        # initialize temperature profiles at t=0
        self._update_temperature()


    # -------------------------
    # Temperature / season
    # -------------------------
    def _update_temperature(self) -> None:
        WP = self.WP
        year_len = float(WP.year_len) if float(WP.year_len) > 1e-9 else 1.0

        phase = 2.0 * math.pi * ((self.t % year_len) / year_len)
        phase -= float(WP.season_phase0)

        s = np.float32(math.sin(phase))
        S_y = self._lat * s  # (size,)

        Ty = self._Tmean_y + self._Amp_y * S_y
        self.Ty = Ty.astype(np.float32, copy=False)

        T0 = float(WP.T0)
        T1 = float(WP.T1)
        if T1 <= T0 + 1e-9:
            gy = (Ty >= np.float32(T0)).astype(np.float32)
        else:
            gy = (Ty - np.float32(T0)) / np.float32(T1 - T0)
            gy = np.clip(gy, 0.0, 1.0).astype(np.float32, copy=False)

        self.gy = gy

    def temperature_field(self) -> np.ndarray:
        s = int(self.WP.size)
        return np.broadcast_to(self.Ty[:, None], (s, s)).astype(np.float32, copy=False)

    def growth_gate_field(self) -> np.ndarray:
        s = int(self.WP.size)
        return np.broadcast_to(self.gy[:, None], (s, s)).astype(np.float32, copy=False)

    def temperature_at(self, x: float, y: float) -> float:
        s = int(self.WP.size)
        _, yw = self.grid.wrap_pos(float(x), float(y))

        y0 = int(math.floor(yw)) % s
        y1 = (y0 + 1) % s
        fy = yw - math.floor(yw)

        t0 = float(self.Ty[y0])
        t1 = float(self.Ty[y1])
        return (1.0 - fy) * t0 + fy * t1

    # -------------------------
    # Agent occupancy
    # -------------------------
    def rebuild_agent_layer(self, agents: Iterable) -> None:
        A = self.A
        A.fill(0)
        s = int(self.WP.size)
        by_id: dict[int, object] = {}
        for ag in agents:
            if not ag.body.alive:
                continue
            aid = int(ag.id)
            by_id[aid] = ag

            cell = int(self.grid.cell_of(float(ag.x), float(ag.y)))
            iy, ix = divmod(cell, s)
            A[iy, ix] = aid

        self._agent_by_id = by_id


    # -------------------------
    # Abiotiska världspass
    # -------------------------
    def _step_temperature(self) -> np.ndarray:
        """
        Uppdatera temperaturprofilen för aktuell tid och returnera temperaturfältet T.
        """
        self._update_temperature()
        return self.temperature_field()

    def _step_carcass(self) -> float:
        """
        Kadaverpass:
          - decay + diffusion för C
          - returnera dM_decay för flux/loggning
        """
        self.C, dM_decay = step_carcass_field(
            self.C,
            WP=self.WP,
            dt=float(self.WP.dt),
        )
        return dM_decay

    def _update_world_flux(self, dM_growth: float, dM_wither: float, dM_decay: float) -> None:
        """
        Uppdatera världens öppna-system-ledger för senaste tick.
        """
        P = self.WP
        e_plant = float(getattr(P, "E_plant_J_per_kg", getattr(P, "E_bio_J_per_kg", 4.0e6)))
        e_carc = float(getattr(P, "E_carcass_J_per_kg", 7.0e6))

        self.last_flux = {
            "dM_growth": max(0.0, dM_growth),
            "dM_wither": max(0.0, dM_wither),
            "dM_decay": max(0.0, dM_decay),
            "E_in_growth": max(0.0, dM_growth) * e_plant,
            "E_loss_wither": max(0.0, dM_wither) * e_plant,
            "E_loss_decay": max(0.0, dM_decay) * e_carc,
        }
        
       
    def step(self) -> None:
        dt = float(self.WP.dt)
    
        # (1) Temperatur / säsong
        self._step_temperature()
    
        # (2) Kadaverfält
        dM_decay = self._step_carcass()
    
        # (3) Flux / öppen-system-ledger
        self._update_world_flux(0.0, 0.0, dM_decay)
    
        # (4) Tid
        self.t += dt

    # -------------------------
    # Sampling (renodlad)
    # -------------------------
    def sample_carcass(self, x: float, y: float) -> float:
        """Bilinear sampling of carcass field C at a single point."""
        return _bilinear_scalar_C(self.C, x, y, self.grid)

    def sample_many_carcass(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        outC: Optional[np.ndarray] = None,
        tmp: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vectorized bilinear sampling for carcass field C only.
        Returns float32 array.
        """
        return _bilinear_many_C(
            self.C,
            xs, ys,
            self.grid,
            outC=outC, tmp=tmp,
        )

    def sample_flora_local(self, x: float, y: float) -> float:
        hook = getattr(self, "sample_flora_local_hook", None)
        if hook is None:
            return 0.0
        return float(hook(x, y))
    
    def sample_food_local(self, x: float, y: float) -> tuple[float, float]:
        """
        Returns (B_kg, C_kg) from current world interfaces:
          - B via flora provider
          - C via carcass field
        """
        B = float(self.sample_flora_local(x, y))
        C = float(self.sample_carcass(x, y))
        return B, C    

    def sample_flora_rays(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        hook = getattr(self, "sample_flora_rays_hook", None)
        if hook is None:
            return np.zeros_like(xs, dtype=np.float32)
        return hook(xs, ys)
        
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

        xf = float(x)
        yf = float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            return 0.0

        x0, y0, x1, y1, fx, fy = self.grid.bilinear_corners(xf, yf)

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
        Fallback-consumption in World: carcass only.
        Plant food is handled by Population via consume_food_hook.
        Returns (got_total_kg, got_carcass_kg).
        """
        hook = getattr(self, "consume_food_hook", None)
        if hook is not None:
            return hook(x, y, amount, prefer_carcass)
    
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0
    
        got_c = float(self._consume_bilinear_from(self.C, x, y, amt))
        return got_c, got_c

    def add_carcass(self, x: float, y: float, amount_kg: float, rad: int = 3) -> None:
        """
        Add carcass mass to C (kg/cell). amount_kg is TOTAL kg deposited.
        """
        amt = float(amount_kg)
        if not math.isfinite(amt) or amt <= 0.0:
            return
    
        r = int(rad)
        if r < 1:
            r = 1
    
        center = self.grid.cell_of(float(x), float(y))
        cy, cx = self.grid.rowcol_of(center)
    
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
            cell = self.grid.cell_from_rowcol(cy + dy, cx + dx)
            iy, ix = self.grid.rowcol_of(cell)
            self.C[iy, ix] = np.float32(float(self.C[iy, ix]) + scale * w)


def _bilinear_scalar_C(C: np.ndarray, x: float, y: float, grid: Grid) -> float:
    x0, y0, x1, y1, fx, fy = grid.bilinear_corners(float(x), float(y))

    fx1 = 1.0 - fx
    fy1 = 1.0 - fy

    c00 = float(C[y0, x0]); c10 = float(C[y0, x1]); c01 = float(C[y1, x0]); c11 = float(C[y1, x1])

    c0 = c00 * fx1 + c10 * fx
    c1 = c01 * fx1 + c11 * fx
    Cv = c0 * fy1 + c1 * fy

    return Cv

if _NUMBA_AVAILABLE:
    @_numba.njit(cache=True, parallel=False)
    def _bilinear_kernel_c_nb(C, xs_flat, ys_flat, s, outC_flat):
        sf = np.float32(s)
        for i in range(xs_flat.size):
            xf = xs_flat[i] % sf
            yf = ys_flat[i] % sf
            x0 = int(xf)
            y0 = int(yf)
            x1 = x0 + 1 if x0 + 1 < s else 0
            y1 = y0 + 1 if y0 + 1 < s else 0
            fx = xf - np.float32(x0)
            fy = yf - np.float32(y0)
            fx1 = np.float32(1.0) - fx
            fy1 = np.float32(1.0) - fy
            outC_flat[i] = (C[y0, x0] * fx1 + C[y0, x1] * fx) * fy1 + (C[y1, x0] * fx1 + C[y1, x1] * fx) * fy    

def _bilinear_many_C(
    C: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    grid: Grid,
    outC: Optional[np.ndarray] = None,
    tmp: Optional[np.ndarray] = None,
) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have same shape")

    shape = xs.shape
    if outC is None:
        outC = np.empty(shape, dtype=np.float32)

    if _NUMBA_AVAILABLE:
        _bilinear_kernel_c_nb(C, xs.ravel(), ys.ravel(), int(grid.size), outC.ravel())
        return outC

    if tmp is None:
        tmp = np.empty(shape, dtype=np.float32)

    x0, y0, x1, y1, fx, fy = grid.bilinear_indices_many(xs, ys)
    fx1 = np.float32(1.0) - fx
    fy1 = np.float32(1.0) - fy

    C00 = C[y0, x0]; C10 = C[y0, x1]; C01 = C[y1, x0]; C11 = C[y1, x1]
    np.multiply(C00, fx1, out=outC); outC += C10 * fx; outC *= fy1
    np.multiply(C01, fx1, out=tmp);  tmp  += C11 * fx; tmp  *= fy
    outC += tmp

    return outC