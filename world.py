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
from phenotype import (
    DECAY_SCALE_LABILE,
    DECAY_SCALE_STRUCT,
    NUTRIENT_PER_KG_LABILE,
    NUTRIENT_PER_KG_STRUCT,
)

# Under detta värde nollställs detritus exakt och cellen lämnar den aktiva
# mängden. Utan en tröskel skulle exponentiellt avklingande celler aldrig bli
# inaktiva, och glesheten vore verkningslös.
_DETRITUS_EPS = 1e-12

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

    # Florans massaskala. En vuxen planta väger B_K gånger sin genetiska
    # adult_mass_k, alltså 0,25–4 gånger detta tal.
    #
    # Var 5e-4, vilket gjorde en vuxen planta till en halv gram mot faunans
    # kilo. Följden var att konsumentbiomassan blev 138 gånger
    # primärproduktionen — förhållandet omvänt mot hur en ekologi ser ut.
    # Ingen kalibrering av näringstillförsel eller täthet rättar en sådan
    # inversion; det är massaskalan som är fel.
    B_K: float = 5e-2

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
    # Andel av frisatt näring som lämnar systemet (urlakning, denitrifikation).
    nutrient_loss_frac: float = 0.10
    # Näringstillförsel per cell och tick. Konstant tills terrängen finns; då
    # blir den vittring som funktion av höjd, och utsköljning under havsnivå.
    nutrient_input: float = 2.0e-10
    # Maximalt näringsupptag per tick vid uptake_capacity = 1.
    # Upptagstak per individ och sekund, uttryckt relativt massaskalan.
    # Ett absolut tal skulle bli hundrafalt hårdare bundet när B_K höjs och
    # göra floran upptagsbegränsad i stället för näringsbegränsad.
    uptake_rate_max_per_BK: float = 0.10

    @property
    def uptake_rate_max(self) -> float:
        """Upptagstak i kg näring per individ och sekund."""
        return float(self.uptake_rate_max_per_BK) * float(self.B_K)
    # Diffusionstakt för löst näring. Explicit schema: D*dt måste hållas under
    # ett för stabilitet, och laplacianen är normerad med grannantalet.
    nutrient_diffusion: float = 0.20

    # --- Perception scaling ---
    C_sense_K: float = 5e-4

    # Energy densities for open-system ledger diagnostics (J/kg)
    # Energi per kilo *labil* vävnad, gemensam för allt organiskt material.
    # Den användbara energitätheten är detta gånger (1 - strukturandel);
    # strukturmaterial lagrar ingen användbar energi.
    #
    # De tidigare skilda konstanterna för växt och kadaver, 4,0e6 och 7,0e6,
    # kodade som typskillnad det som är en materialegenskap. Med initierad
    # medelstruktur 0,57 för flora och 0,25 för fauna ger den här enda
    # konstanten 4,00e6 respektive 6,98e6 — samma kvot, men härledd.
    E_labile_J_per_kg: float = 9.302e6


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
        # Världsfälten har både en ägare och en kadens. Ägaren säger vem som
        # skriver; kadensen säger hur ofta fältet behöver röras.
        #
        #   statiska      lagras som skalär tills något varierar dem rumsligt
        #   dynamiska     platta per-cell-arrayer indexerade med cell_idx
        #   härledda      beräknas vid läsning, inte varje tick
        #
        # En tom cell ska vara billig, av samma skäl som en organism utan en
        # kapacitet inte ska kosta något för den. Se docs/varldens-kadensmodell.md.
        nc = int(self.grid.n_cells)

        # --- statiska: skalära tills terräng eller väder gör dem rumsliga ---
        self.elevation = float(self.WP.elevation_init)
        self.rain_input = float(self.WP.rain_input_base)
        self.spring_input = float(self.WP.spring_input_base)
        self.infiltration = float(self.WP.infiltration_base)
        self.evaporation = float(self.WP.evaporation_base)

        # --- dynamiska: per cell ---
        self.water = np.full(nc, np.float32(self.WP.water_init), dtype=np.float32)
        # nutrient lagras i float64 till skillnad från övriga fält. Det är den
        # bevarade storhet vi vill kunna påstå balans om: näring cirkulerar
        # medan kol flödar igenom, så näringsbalansen är den enda hårda
        # invarianten som är möjlig. I float32 ackumulerar diffusionens
        # laplacian ett fel kring 5e-7 per tusen pass; i float64 är det 1e-15.
        self.nutrient = np.full(nc, float(self.WP.nutrient_init), dtype=np.float64)
        # detritus är glest dynamiskt: nollskilt i en bråkdel av cellerna, och
        # ett fullt svep skulle mest multiplicera nollor. Fältet bär därför en
        # aktiv mängd, och kontraktet är att inaktiva celler är exakt noll.
        # detritus och detritus_structure lagras i float64 av samma skäl som
        # nutrient: de ingår i näringsbalansen, som är hård invariant. I float32
        # låg balansens drift kring 1e-7 relativt och golvet sattes av just
        # dessa två fält. Kostnaden är minne, inte tid — fälten är glest
        # dynamiska och sveps bara över den aktiva mängden.
        self.detritus = np.full(nc, float(self.WP.detritus_init), dtype=np.float64)
        self._detritus_member = self.detritus > _DETRITUS_EPS
        self._detritus_active = np.flatnonzero(self._detritus_member).astype(np.int32, copy=False)

        # Massviktad medelstrukturandel i cellens detritus, ärvd från det som
        # dog. Styr nedbrytningstakten: högstrukturerat material bryts ner
        # långsammare. Samma kadensklass och samma aktiva mängd som detritus.
        self.detritus_structure = np.zeros(nc, dtype=np.float64)

        # Ackumulerad exkreterad massa sedan senaste ledgeruppdatering.
        self._dM_excreted = 0.0

        # Näringens externa flöden, ackumulerade sedan start. Näringen är den
        # enda storhet som kan cirkulera slutet — kol flödar igenom — så
        # balansen mäts mot dessa och inte mot total massa.
        self._nutrient_added_total = 0.0
        self._nutrient_lost_total = 0.0

        # Aktiveringar och avaktiveringar samlas och slås ihop en gång per tick.
        # Att göra dem direkt mot arrayen kostade O(n) per händelse, vilket dög
        # för sällsynta kadaver men inte för exkretion i varje betningshändelse.
        self._detritus_pending: list[int] = []
        self._detritus_dirty = False

        # --- härledda: flödesstyrkan är noll tills hydro räknar grannflöde ---
        self.flow_strength = 0.0

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

        # Klimatet är en bandegenskap: latituden varierar bara mellan band, och
        # celler i samma band har identiskt klimat. Profilerna lagras därför per
        # band, inte per cell — O(H) i stället för O(H*W) arbete varje tick.
        #
        # Latitudprofilen är periodisk runt torusen: sydpol -> ekvator ->
        # nordpol -> ekvator -> sydpol. Den tidigare linjära profilen slöt sig
        # inte, och eftersom världen wrappar i y hamnade båda polerna kant i
        # kant med en säsongsdiskontinuitet emellan.
        #
        # Grid äger bandindelningen och latituden; klimatets tolkning av den
        # ägs här.
        lat = np.asarray(self.grid.band_lat, dtype=np.float32)
        abs_lat = np.abs(lat)

        self._lat = lat
        self._abs_lat = abs_lat
        self._Tmean_band = np.float32(self.WP.T_eq) - np.float32(self.WP.dT_pole) * (
            abs_lat ** np.float32(self.WP.lat_p)
        )
        self._Amp_band = np.float32(self.WP.A_eq) + (np.float32(self.WP.A_pole) - np.float32(self.WP.A_eq)) * (
            abs_lat ** np.float32(self.WP.amp_q)
        )

        # Senast beräknade klimatfält, per band
        n_bands = int(self.grid.n_bands)
        self.T_band = np.zeros(n_bands, dtype=np.float32)   # degC per band
        self.g_band = np.ones(n_bands, dtype=np.float32)    # tillväxtgrind per band

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
        S_band = self._lat * s  # (n_bands,)

        T_band = self._Tmean_band + self._Amp_band * S_band
        self.T_band = T_band.astype(np.float32, copy=False)

        T0 = float(WP.T0)
        T1 = float(WP.T1)
        if T1 <= T0 + 1e-9:
            g_band = (T_band >= np.float32(T0)).astype(np.float32)
        else:
            g_band = (T_band - np.float32(T0)) / np.float32(T1 - T0)
            g_band = np.clip(g_band, 0.0, 1.0).astype(np.float32, copy=False)

        self.g_band = g_band

    def temperature_of_cell(self, cell: int) -> float:
        """Temperatur i en cell. Den form biologin ska använda."""
        return float(self.T_band[self.grid.band_of_cell(cell)])

    def temperature_of_cells(self, cells: np.ndarray) -> np.ndarray:
        """Temperatur för en mängd celler, utan att materialisera hela fältet."""
        return self.T_band[self.grid.bands_of_cells(cells)]

    def growth_gate_of_cell(self, cell: int) -> float:
        return float(self.g_band[self.grid.band_of_cell(cell)])

    def growth_gate_of_cells(self, cells: np.ndarray) -> np.ndarray:
        return self.g_band[self.grid.bands_of_cells(cells)]

    def temperature_field(self) -> np.ndarray:
        """
        Hela temperaturfältet per cell.

        Materialiserar en array med längd n_cells och kostar därmed O(n_cells).
        Avsedd för visning och diagnostik, inte för systempass — dessa ska
        använda temperature_of_cell() eller temperature_of_cells().
        """
        return self.T_band[self.grid.bands_of_cells(np.arange(int(self.grid.n_cells)))]

    def growth_gate_field(self) -> np.ndarray:
        """Hela tillväxtgrindsfältet per cell. Samma kostnadsanmärkning som ovan."""
        return self.g_band[self.grid.bands_of_cells(np.arange(int(self.grid.n_cells)))]

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
    def temperature_pass(self) -> None:
        """
        Uppdatera klimatprofilerna för aktuell tid.

        Returnerar inget fält. Att materialisera temperaturen per cell vore
        O(n_cells) arbete varje tick för information som ryms i n_bands värden,
        och ingen anropare behövde det. Läsare använder temperature_of_cell()
        eller temperature_of_cells().
        """
        self._update_temperature()

    def _detritus_activate(self, cell: int) -> None:
        """Markera en cell som nollskild. Idempotent, O(1)."""
        c = int(cell)
        if not self._detritus_member[c]:
            self._detritus_member[c] = True
            self._detritus_pending.append(c)

    def _detritus_flush(self) -> None:
        """
        Slå ihop väntande aktiveringar och släpp avaktiverade celler.

        Dedupliceringen är nödvändig och inte defensiv: en cell som töms och
        fylls igen innan flush hinner köra hamnar annars två gånger i mängden,
        eftersom medlemsflaggan är sann både före och efter.
        """
        if self._detritus_pending:
            add = np.asarray(self._detritus_pending, dtype=np.int32)
            self._detritus_active = np.concatenate((self._detritus_active, add))
            self._detritus_pending = []
            self._detritus_dirty = True
        if self._detritus_dirty:
            act = self._detritus_active
            if act.size:
                act = act[self._detritus_member[act]]
                self._detritus_active = np.unique(act).astype(np.int32, copy=False)
            self._detritus_dirty = False

    def _detritus_add(self, cell: int, amount: float, structure: float) -> None:
        """
        Lägg till massa i en cells detritus och blanda in dess strukturandel
        massviktat.

        En stor kadaver dominerar cellens sammansättning; en liten i mycket
        förna späds ut. Det är cellen som är miljöns enhet — samma val som
        ligger bakom att all perception och konsumtion är cellvis.
        """
        c = int(cell)
        add = float(amount)
        if add <= 0.0:
            return
        old = float(self.detritus[c])
        tot = old + add
        if tot <= 0.0:
            return
        s_old = float(self.detritus_structure[c])
        self.detritus_structure[c] = (s_old * old + float(structure) * add) / tot
        self.detritus[c] = tot
        self._detritus_activate(c)

    def _detritus_deactivate_if_empty(self, cell: int) -> None:
        """Nollställ exakt och lämna aktiva mängden om cellen tömts."""
        c = int(cell)
        if self._detritus_member[c] and float(self.detritus[c]) <= _DETRITUS_EPS:
            self.detritus[c] = 0.0
            self.detritus_structure[c] = 0.0
            self._detritus_member[c] = False
            self._detritus_dirty = True

    @property
    def detritus_active_cells(self) -> np.ndarray:
        """Celler med nollskilt detritus. Läsvy för pass och diagnostik."""
        self._detritus_flush()
        return self._detritus_active

    @property
    def surface_level(self) -> np.ndarray:
        """
        Fri yta, elevation + water. Härledd: beräknas vid läsning i stället för
        varje tick, eftersom inget systempass läser den ännu. Kostar O(n_cells)
        per anrop — hydro ska räkna på `water` direkt.
        """
        return (self.water + np.float32(self.elevation)).astype(np.float32, copy=False)

    @property
    def submerged(self) -> np.ndarray:
        """Bool per cell, water över tröskeln. Härledd, se surface_level."""
        return self.water > np.float32(self.WP.submerged_threshold)

    def hydro_pass(self) -> tuple[float, float]:
        """
        Minimal hydro-skelett för fas 1.5. Ännu inget grannflöde.

        Forcing-termerna är rumsligt konstanta, så nettotillskottet per tick är
        ett tal och inte ett fält. Det gör att passet gör en enda vektoriserad
        operation över `water` i stället för ett dussin. När forcing blir
        rumsligt varierande promoveras termerna till arrayer och uttrycket
        nedan fungerar oförändrat.
        """
        dt = float(self.WP.dt)
        dwater = dt * (
            float(self.rain_input)
            + float(self.spring_input)
            - float(self.infiltration)
            - float(self.evaporation)
        )

        if dwater == 0.0:
            return 0.0, 0.0

        w = self.water
        if dwater > 0.0:
            # Inget klipps bort: varje cell ökar lika mycket.
            w += np.float32(dwater)
            n = float(w.shape[0])
            return dwater * n, 0.0

        # Negativt tillskott: celler med mindre vatten än uttaget klipps mot noll,
        # så det faktiska uttaget är summan av min(water, -dwater).
        removed = float(np.sum(np.minimum(w, np.float32(-dwater)), dtype=np.float64))
        np.maximum(w + np.float32(dwater), np.float32(0.0), out=w)
        return 0.0, removed

    def nutrient_input_pass(self) -> float:
        """
        Extern näringstillförsel. Forcing, alltså rumsligt konstant tills något
        varierar den — se docs/varldens-kadensmodell.md.
        """
        add = float(self.WP.dt) * float(self.WP.nutrient_input)
        if add == 0.0:
            return 0.0
        self.nutrient += add
        total = add * float(self.grid.n_cells)
        self._nutrient_added_total += total
        return total

    def add_nutrient(self, cell: int, amount: float) -> float:
        """
        Återför näring till en cell. Returnerar tillförd mängd.

        Detta är recirkulation, inte tillförsel: kvävehaltigt avfall från
        organismernas förbränning och vävnadsomsättning. Den räknas därför
        inte in i `_nutrient_added_total`, som bara bokför extern forcing.
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0
        c = int(cell)
        if c < 0 or c >= int(self.grid.n_cells):
            return 0.0
        self.nutrient[c] += amt
        return amt

    def take_nutrient(self, cell: int, amount: float) -> float:
        """Ta upp till `amount` kg näring ur en cell. Returnerar faktiskt uttag."""
        c = int(cell)
        avail = float(self.nutrient[c])
        got = amount if amount < avail else avail
        if got <= 0.0:
            return 0.0
        self.nutrient[c] = avail - got
        return float(got)

    def transport_pass(self) -> float:
        """
        Diffusion av lösta ämnen över topologiska grannar.

        Tvåstegsmetod: flödena beräknas ur tillståndet vid passets början och
        appliceras simultant som nettoförändringar. Diskret laplacian via
        grannmatrisen, alltså geometriagnostisk — samma kod gäller för fyra
        eller sex grannar.

        Massbevarandet är exakt och inte approximativt. Grannrelationen är
        ömsesidig och graden konstant, så varje cell förekommer i exakt k
        grannlistor och summan av laplacianen över hela världen är noll.

        Kadensmässigt gör det här `nutrient` tätt dynamiskt — den fjärde
        klassen i docs/varldens-kadensmodell.md, och den enda där fullt svep
        är genuint motiverat. Näring som diffunderar har inget glest stöd.

        Returnerar summan av absolut omfördelad mängd, som diagnostik.
        """
        D = float(self.WP.nutrient_diffusion)
        dt = float(self.WP.dt)
        if D <= 0.0 or dt <= 0.0:
            return 0.0

        n = self.nutrient
        idx = self.grid.neighbor_idx
        k = int(self.grid.neighbor_count)

        # Laplacian: grannarnas summa minus k gånger egen halt.
        lap = n[idx].sum(axis=1, dtype=np.float64) - float(k) * n

        delta = lap * (D * dt / k)
        n += delta

        return float(np.sum(np.abs(delta))) * 0.5

    def decomposition_pass(self) -> tuple[float, float]:
        """
        Minimal decomposition för fas 1.5.

        I patch 1 görs endast enkel decay av detritus. Ingen diffusion och ingen
        överföring till nutrient ännu. Source of truth är self.detritus.
        """
        self._detritus_flush()
        act = self._detritus_active
        if act.size == 0:
            return 0.0, 0.0

        dt = float(self.WP.dt)
        rate = float(self.WP.detritus_decay)

        d = self.detritus[act]
        st = self.detritus_structure[act]

        # Nedbrytning per fraktion. Labilt och strukturellt material bryts ner
        # med var sin takt, räknade ur massa och strukturandel utan ett andra
        # fält. Att sakta ner hela massan i stället lät strukturandelen skena:
        # bara det labila försvann, och kvarvarande material blev asymptotiskt
        # ren struktur.
        lab = d * (1.0 - st)
        stru = d * st

        k_lab = dt * rate * float(DECAY_SCALE_LABILE)
        k_str = dt * rate * float(DECAY_SCALE_STRUCT)

        d_lab = np.minimum(lab, lab * k_lab)
        d_str = np.minimum(stru, stru * k_str)

        lab_new = lab - d_lab
        stru_new = stru - d_str
        new = lab_new + stru_new

        dM_detritus_decay = float(np.sum(d_lab + d_str, dtype=np.float64))

        # Frisatt näring är den nedbrutna massan gånger dess näringsinnehåll.
        released = float(
            np.sum(d_lab, dtype=np.float64) * NUTRIENT_PER_KG_LABILE
            + np.sum(d_str, dtype=np.float64) * NUTRIENT_PER_KG_STRUCT
        )
        retained = 1.0 - float(self.WP.nutrient_loss_frac)
        np.add.at(self.nutrient, act,
                  (d_lab * NUTRIENT_PER_KG_LABILE
                   + d_str * NUTRIENT_PER_KG_STRUCT) * retained)

        # Strukturandelen följer av vad som blev kvar.
        with np.errstate(invalid="ignore", divide="ignore"):
            st_new = np.where(new > 0.0, stru_new / np.maximum(new, 1e-300), 0.0)
        self.detritus_structure[act] = np.clip(st_new, 0.0, 1.0)

        # Celler under tröskeln nollställs exakt och lämnar den aktiva mängden,
        # så att kontraktet "inaktiv cell är noll" håller.
        empty = new <= float(_DETRITUS_EPS)
        if empty.any():
            new[empty] = 0.0
            self.detritus_structure[act[empty]] = 0.0
            self._detritus_member[act[empty]] = False
            self._detritus_dirty = True

        self.detritus[act] = new

        self._nutrient_lost_total += released * (1.0 - float(retained))
        dM_nutrient_from_detritus = released * float(retained)
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
        # Nominella energiskalor för ledgern; se anmärkningen i Population.
        e_lab = float(getattr(P, "E_labile_J_per_kg", 9.302e6))
        e_plant = e_lab * (1.0 - 0.57)
        e_carc = e_lab * (1.0 - 0.25)

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
        self.nutrient_input_pass()
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
    def _consume_from_field(self, field: np.ndarray, x: float, y: float, amount: float) -> tuple[float, float]:
        """
        Konsumera upp till `amount` kg ur `field` i den cell (x, y) faller i.

        Returnerar (kg, energi_J). Energin följer av materialets strukturandel:
        strukturmaterial lagrar ingen användbar energi, så ett kilo segt
        substrat är värt mindre än ett kilo mjukt. Konsumentens
        matsmältningsverkningsgrad tillämpas hos konsumenten, inte här.

        Organismen befinner sig i en cell och äter ur den. Vill den åt en
        granncells innehåll måste den flytta sig dit — position har betydelse.
        `field` är en platt per-cell-array; funktionen är därmed generell och
        kan bära nutrient-upptaget i Steg 3.
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0

        xf = float(x)
        yf = float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            return 0.0, 0.0

        cell = int(self.grid.cell_of(xf, yf))
        avail = float(field[cell])
        if not math.isfinite(avail) or avail <= 1e-12:
            return 0.0, 0.0

        got = amt if amt < avail else avail
        field[cell] = (avail - got) if field.dtype == np.float64 else np.float32(avail - got)

        struct = 0.0
        if field is self.detritus:
            struct = float(self.detritus_structure[cell])
            self._detritus_deactivate_if_empty(cell)

        energy = got * float(self.WP.E_labile_J_per_kg) * (1.0 - struct)
        return float(got), float(energy)

    def consume_food(self, x: float, y: float, amount: float,
                     prefer_detritus: bool = True) -> Tuple[float, float, float, float]:
        """
        Reservkonsumtion i World: bara detritus. Levande föda hanteras av
        Population via consume_food_hook.

        Returnerar (kg_levande, kg_detritus, energi_levande_J, energi_detritus_J).
        """
        hook = getattr(self, "consume_food_hook", None)
        if hook is not None:
            return hook(x, y, amount, prefer_detritus)
    
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
    
        got_d, e_d = self._consume_from_field(self.detritus, x, y, amt)
        return 0.0, float(got_d), 0.0, float(e_d)

    def excrete_at(self, x: float, y: float, amount_kg: float, structure: float) -> float:
        """
        Återför icke assimilerad massa till cellen som detritus.

        Till skillnad från add_carcass sprids inget: exkrementet hamnar där
        organismen står. Returnerar tillförd massa för ledgern.
        """
        amt = float(amount_kg)
        if not (math.isfinite(amt) and amt > 0.0):
            return 0.0
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            return 0.0
        cell = int(self.grid.cell_of(float(x), float(y)))
        self._detritus_add(cell, amt, float(structure))
        self._dM_excreted += amt
        return amt

    def add_carcass(self, x: float, y: float, amount_kg: float, rad: int = 3,
                    structure: float = 0.45) -> None:
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
            self.detritus[center] = float(self.detritus[center]) + amt
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
            self.detritus[center] = float(self.detritus[center]) + amt
            return
    
        scale = amt / wsum
        for cell, w in weights:
            self._detritus_add(cell, scale * w, structure)


