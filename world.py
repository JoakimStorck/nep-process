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

from grid import Grid

# Index i Grid.neighbor_idx. Ordningen är upp, ned, vänster, höger.
_NEIGHBOR_DOWN = 1

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

   
# -------------------------
# Parameters
# -------------------------
@dataclass
class WorldParams:
    # Världens form i celler. Höjden måste vara jämn: hexgeometrins radoffset
    # blir annars inkonsistent över sömmen. Bredd 64 och höjd 256 ger fyra
    # klimatband med 64 rader mellan pol och ekvator — se TODO.md, Steg 2.
    width: int = 64
    height: int = 256
    size: int = 0      # bekvämlighet för kvadratiska världar; sätter båda
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
    # Hydrology / terrain / world fields
    # -------------------------
    sea_level: float = 0.0
    submerged_threshold: float = 1e-6

    elevation_init: float = 0.0
    water_init: float = 0.0
    nutrient_init: float = 0.0
    detritus_init: float = 0.0

    rain_input_base: float = 0.0
    spring_input_base: float = 0.0
    infiltration_base: float = 0.0
    evaporation_base: float = 0.0

    # -------------------------
    # Detritus / decay
    # -------------------------
    detritus_decay: float = 0.077

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
        self.grid = Grid(size=int(self.WP.size),
                         width=int(self.WP.width),
                         height=int(self.WP.height))

        self.sample_flora_local_hook = None
        self.sample_flora_rays_hook = None        
        self.consume_food_hook = None

        # -------------------------
        # Primary world fields
        # -------------------------
        # Världsfälten är platta per-cell-arrayer indexerade med cell_idx.
        # Geometrin lever i Grid; världslagret ser bara en lista av celler.
        nc = int(self.grid.n_cells)

        self.elevation = np.full(nc, np.float32(self.WP.elevation_init), dtype=np.float32)
        self.water = np.full(nc, np.float32(self.WP.water_init), dtype=np.float32)
        self.nutrient = np.full(nc, np.float32(self.WP.nutrient_init), dtype=np.float32)
        self.detritus = np.full(nc, np.float32(self.WP.detritus_init), dtype=np.float32)

        # self.C är en egenskap, inte ett fält — se nedan.

        # -------------------------
        # External forcing fields
        # -------------------------
        self.rain_input = np.full(nc, np.float32(self.WP.rain_input_base), dtype=np.float32)
        self.spring_input = np.full(nc, np.float32(self.WP.spring_input_base), dtype=np.float32)
        self.infiltration = np.full(nc, np.float32(self.WP.infiltration_base), dtype=np.float32)
        self.evaporation = np.full(nc, np.float32(self.WP.evaporation_base), dtype=np.float32)

        # -------------------------
        # Derived hydro fields
        # -------------------------
        self.surface_level = np.zeros(nc, dtype=np.float32)
        self.submerged = np.zeros(nc, dtype=np.bool_)
        self.flow_strength = np.zeros(nc, dtype=np.float32)

        # time
        self.t = 0.0
        self.last_flux = {
            "dM_growth": 0.0,
            "dM_wither": 0.0,
            "dM_decay": 0.0,
            "dM_detritus_decay": 0.0,
            "dM_nutrient_from_detritus": 0.0,
            "dM_water_added": 0.0,
            "dM_water_removed": 0.0,
            "dM_transport": 0.0,
            "E_in_growth": 0.0,
            "E_loss_wither": 0.0,
            "E_loss_decay": 0.0,
        }

        # Latitudprofilen är periodisk runt torusen: sydpol -> ekvator ->
        # nordpol -> ekvator -> sydpol. Det ger två kalla band åtskilda av två
        # tempererade zoner, med motfasiga årstider.
        #
        # Den tidigare linjära profilen gick -1 till +1 utan att sluta sig, och
        # eftersom världen wrappar i y hamnade de båda polerna kant i kant med
        # en säsongsdiskontinuitet på 24 degC emellan — tio gånger brantare än
        # någon verklig klimatgradient i världen, och ett artefakt snarare än en
        # barriär organismer kan anpassa sig till. Den gav dessutom ett enda
        # sammanhängande kallt band, inte två isolerade.
        #
        # Grid.cell_lat bär cellens normerade radläge; klimatets tolkning av det
        # ägs här.
        H = max(1, int(self.grid.height))
        W = max(1, int(self.grid.width))
        rows = np.arange(int(self.grid.n_cells), dtype=np.int64) // W
        lat = (-np.cos(2.0 * np.pi * rows.astype(np.float64) / float(H))).astype(np.float32)
        abs_lat = np.abs(lat)

        self._lat = lat
        self._abs_lat = abs_lat
        self._Tmean_cell = np.float32(self.WP.T_eq) - np.float32(self.WP.dT_pole) * (
            abs_lat ** np.float32(self.WP.lat_p)
        )
        self._Amp_cell = np.float32(self.WP.A_eq) + (np.float32(self.WP.A_pole) - np.float32(self.WP.A_eq)) * (
            abs_lat ** np.float32(self.WP.amp_q)
        )

        # Senast beräknade fält, per cell
        self.T_cell = np.zeros(nc, dtype=np.float32)   # degC per cell
        self.g_cell = np.ones(nc, dtype=np.float32)    # tillväxtgrind per cell i [0,1]

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
        S_cell = self._lat * s  # (n_cells,)

        T_cell = self._Tmean_cell + self._Amp_cell * S_cell
        self.T_cell = T_cell.astype(np.float32, copy=False)

        T0 = float(WP.T0)
        T1 = float(WP.T1)
        if T1 <= T0 + 1e-9:
            g_cell = (T_cell >= np.float32(T0)).astype(np.float32)
        else:
            g_cell = (T_cell - np.float32(T0)) / np.float32(T1 - T0)
            g_cell = np.clip(g_cell, 0.0, 1.0).astype(np.float32, copy=False)

        self.g_cell = g_cell

    @property
    def C(self) -> np.ndarray:
        """
        Kvadratbunden kompatibilitetsvy av detritus, för viewer och simlog som
        ännu ritar i rutnät. Delar buffert med self.detritus.

        Egenskap och inte fält med avsikt: världspassen binder om self.detritus
        till nya arrayer, och ett cachat C skulle då tyst peka på ett gammalt
        tillstånd eller få fel form. Tas bort när viewern går över till cell-ID
        i Steg 2, och namnet försvinner helt i Steg 3.
        """
        return self._as_2d(self.detritus)

    def _as_2d(self, arr: np.ndarray) -> np.ndarray:
        """
        Kvadratbunden vy av ett per-cell-fält.

        Enda platsen där världsfält får anta rutnätsform. Vyn delar minne med
        det platta fältet, så skrivningar går tillbaka till källan.

        Efter Steg 1c har den bara C-egenskapen som anropare, alltså viewer och
        simlog. Båda går över till cell-ID i Steg 2, och då försvinner den här
        metoden tillsammans med dem.
        """
        return arr.reshape(self.grid.shape)

    @property
    def Ty(self) -> np.ndarray:
        """
        Per-rad-temperatur. Kvadratbunden kompatibilitetsvy för viewer och
        simlog. Temperaturen är konstant längs en rad, så första kolumnen av
        det omformade per-cell-fältet räcker. Tas bort i Steg 2.
        """
        return self.T_cell.reshape(self.grid.shape)[:, 0]

    @property
    def gy(self) -> np.ndarray:
        """Per-rad-tillväxtgrind. Samma kompatibilitetsroll som Ty."""
        return self.g_cell.reshape(self.grid.shape)[:, 0]

    def temperature_field(self) -> np.ndarray:
        """Temperatur per cell, indexerad med cell_idx."""
        return self.T_cell

    def growth_gate_field(self) -> np.ndarray:
        """Tillväxtgrind per cell, indexerad med cell_idx."""
        return self.g_cell

    def temperature_of_cell(self, cell: int) -> float:
        """Temperatur i en cell. Den form biologin bör använda."""
        return float(self.T_cell[int(cell)])

    def temperature_at(self, x: float, y: float) -> float:
        """
        Temperatur vid en kontinuerlig position.

        Bekvämlighetsomslag kring temperature_of_cell(). Ingen interpolation
        mellan celler: cellen är miljöns enhet, och alla organismer i samma
        cell möter samma temperatur — vilket floran redan gjorde. Kroppens
        temperatur integrerar över tid och jämnar ut steget mellan celler.

        Anroparen i agent.py bör på sikt slå upp cellen själv; det hör till
        Steg 5b när fauna blir store-first.
        """
        return self.temperature_of_cell(self.grid.cell_of(float(x), float(y)))


    # -------------------------
    # Abiotiska världspass
    # -------------------------
    def temperature_pass(self) -> np.ndarray:
        """
        Uppdatera temperaturprofilen för aktuell tid och returnera temperaturfältet T.
        """
        self._update_temperature()
        return self.temperature_field()

    def hydro_pass(self) -> tuple[float, float]:
        """
        Minimal hydro-skelett för fas 1.5.

        Patch 1 gör ännu inget grannflöde. Hydro äger dock redan sina härledda fält:
          - water uppdateras av forcing-termer
          - surface_level, submerged och flow_strength lämnas i konsistent skick
        """
        dt = np.float32(self.WP.dt)

        water_before = self.water.copy()
        dwater = dt * (self.rain_input + self.spring_input - self.infiltration - self.evaporation)
        self.water = np.maximum(self.water + dwater, np.float32(0.0)).astype(np.float32, copy=False)

        self.surface_level = (self.elevation + self.water).astype(np.float32, copy=False)
        self.submerged = self.water > np.float32(self.WP.submerged_threshold)
        self.flow_strength.fill(np.float32(0.0))

        delta = self.water - water_before
        dM_water_added = float(np.sum(np.maximum(delta, 0.0), dtype=np.float64))
        dM_water_removed = float(np.sum(np.maximum(-delta, 0.0), dtype=np.float64))
        return dM_water_added, dM_water_removed

    def transport_pass(self) -> float:
        """
        Placeholder för framtida transport/diffusion av lösta ämnen.
        Ingen transport ännu i patch 1.
        """
        return 0.0

    def decomposition_pass(self) -> tuple[float, float]:
        """
        Minimal decomposition för fas 1.5.

        I patch 1 görs endast enkel decay av detritus. Ingen diffusion och ingen
        överföring till nutrient ännu. Source of truth är self.detritus.
        """
        dt = float(self.WP.dt)
        rate = float(self.WP.detritus_decay)

        decay = np.float32(rate) * self.detritus
        self.detritus = np.maximum(
            self.detritus - np.float32(dt) * decay,
            np.float32(0.0),
        ).astype(np.float32, copy=False)

        # håll kompatibilitetsaliaset pekande på source-of-truth-arrayen
        dM_detritus_decay = float(np.sum(np.float64(dt) * np.float64(decay)))
        dM_nutrient_from_detritus = 0.0
        return dM_detritus_decay, dM_nutrient_from_detritus

    def update_flux(
        self,
        *,
        dM_growth: float = 0.0,
        dM_wither: float = 0.0,
        dM_detritus_decay: float = 0.0,
        dM_nutrient_from_detritus: float = 0.0,
        dM_water_added: float = 0.0,
        dM_water_removed: float = 0.0,
        dM_transport: float = 0.0,
    ) -> None:
        """
        Uppdatera världens öppna-system-ledger för senaste tick.
        """
        P = self.WP
        e_plant = float(getattr(P, "E_plant_J_per_kg", getattr(P, "E_bio_J_per_kg", 4.0e6)))
        e_carc = float(getattr(P, "E_carcass_J_per_kg", 7.0e6))

        self.last_flux = {
            "dM_growth": max(0.0, dM_growth),
            "dM_wither": max(0.0, dM_wither),
            "dM_decay": max(0.0, dM_detritus_decay),
            "dM_detritus_decay": max(0.0, dM_detritus_decay),
            "dM_nutrient_from_detritus": max(0.0, dM_nutrient_from_detritus),
            "dM_water_added": max(0.0, dM_water_added),
            "dM_water_removed": max(0.0, dM_water_removed),
            "dM_transport": float(dM_transport),
            "E_in_growth": max(0.0, dM_growth) * e_plant,
            "E_loss_wither": max(0.0, dM_wither) * e_plant,
            "E_loss_decay": max(0.0, dM_detritus_decay) * e_carc,
        }

    def step(self) -> None:
        dt = float(self.WP.dt)

        self.temperature_pass()
        dM_water_added, dM_water_removed = self.hydro_pass()
        dM_transport = self.transport_pass()
        dM_detritus_decay, dM_nutrient_from_detritus = self.decomposition_pass()
        self.update_flux(
            dM_growth=0.0,
            dM_wither=0.0,
            dM_detritus_decay=dM_detritus_decay,
            dM_nutrient_from_detritus=dM_nutrient_from_detritus,
            dM_water_added=dM_water_added,
            dM_water_removed=dM_water_removed,
            dM_transport=dM_transport,
        )

        self.t += dt

    # -------------------------
    # Sampling (renodlad)
    # -------------------------
    def sample_carcass(self, x: float, y: float) -> float:
        """Bilinear sampling of detritus field at a single point."""
        return float(self.detritus[self.grid.cell_of(float(x), float(y))])

    def sample_many_carcass(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        outC: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vektoriserad sampling av detritus i de celler punkterna faller i.

        Ingen interpolation: fältet är styckvis konstant per cell, så en
        organism läser cellens värde. `tmp` togs bort med den bilinjära
        mellanlagringen.
        """
        cells = self.grid.cell_of_many(xs, ys)
        if outC is None:
            return self.detritus[cells].astype(np.float32, copy=False)
        outC[...] = self.detritus[cells].reshape(np.shape(outC))
        return outC

    def sample_flora_local(self, x: float, y: float) -> float:
        hook = getattr(self, "sample_flora_local_hook", None)
        if hook is None:
            return 0.0
        return float(hook(x, y))
    
    def sample_food_local(self, x: float, y: float) -> tuple[float, float]:
        """
        Returns (B_kg, detritus_kg) from current world interfaces:
          - B via flora provider
          - detritus via world detritus field
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
    def _consume_from_field(self, field: np.ndarray, x: float, y: float, amount: float) -> float:
        """
        Konsumera upp till `amount` kg ur `field` i den cell (x, y) faller i.

        Organismen befinner sig i en cell och äter ur den. Vill den åt en
        granncells innehåll måste den flytta sig dit — position har betydelse.
        `field` är en platt per-cell-array; funktionen är därmed generell och
        kan bära nutrient-upptaget i Steg 3.
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0

        xf = float(x)
        yf = float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            return 0.0

        cell = int(self.grid.cell_of(xf, yf))
        avail = float(field[cell])
        if not math.isfinite(avail) or avail <= 1e-12:
            return 0.0

        got = amt if amt < avail else avail
        field[cell] = np.float32(avail - got)
        return float(got)

    def consume_food(self, x: float, y: float, amount: float, prefer_carcass: bool = True) -> Tuple[float, float]:
        """
        Fallback-consumption in World: detritus only.
        Plant food is handled by Population via consume_food_hook.
        Returns (got_total_kg, got_detritus_kg).
        """
        hook = getattr(self, "consume_food_hook", None)
        if hook is not None:
            return hook(x, y, amount, prefer_carcass)
    
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0
    
        got_c = float(self._consume_from_field(self.detritus, x, y, amt))
        return got_c, got_c

    def add_carcass(self, x: float, y: float, amount_kg: float, rad: int = 3) -> None:
        """
        Add carcass mass to detritus field (kg/cell).
        """
        amt = float(amount_kg)
        if not math.isfinite(amt) or amt <= 0.0:
            return
    
        r = int(rad)
        if r < 1:
            r = 1
    
        center = int(self.grid.cell_of(float(x), float(y)))
    
        # Topologisk spridning: cellerna inom r steg, viktade med topologiskt
        # avstånd. Ersätter den kvadratiska dx/dy-kärnan med euklidiskt avstånd,
        # som var det sista geometriantagandet i world.py.
        cells = self.grid.cells_within(center, r)
        if not cells:
            self.detritus[center] = np.float32(float(self.detritus[center]) + amt)
            return
    
        sigma = max(0.75, 0.5 * r)
        inv2sig2 = 1.0 / (2.0 * sigma * sigma)
    
        weights = []
        wsum = 0.0
        for cell in cells:
            d = float(self.grid.distance(center, int(cell)))
            w = math.exp(-(d * d) * inv2sig2)
            weights.append((int(cell), w))
            wsum += w
    
        if wsum <= 1e-12:
            self.detritus[center] = np.float32(float(self.detritus[center]) + amt)
            return
    
        scale = amt / wsum
        for cell, w in weights:
            self.detritus[cell] = np.float32(float(self.detritus[cell]) + scale * w)


