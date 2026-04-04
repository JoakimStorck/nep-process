# agent.py
from __future__ import annotations


import math, random
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Tuple

import numpy as np

from world import World, clamp
from mlp import MLPGenome
from phenotype import Phenotype, derive_pheno, phenotype_summary


def torus_wrap(x: float, size: int) -> float:
    return x % size


# -------------------------
# Agent ids
import itertools as _itertools

# Agent-ID-räknare: per-modul-instans, unik per Python-process.
# Använd _agent_id_counter.reset() finns inte — starta om processen för rena körningar.
# OBS: om flera Population-instanser körs i samma process delar de ID-rymden,
# vilket är korrekt (IDs förblir globalt unika).
_agent_id_counter = _itertools.count(1)


def _new_agent_id() -> int:
    return next(_agent_id_counter)


# -------------------------
# Sensing helpers
# -------------------------
def _sense_level(u: float) -> int:
    # u in [0,1] -> 0..3
    if u < 0.25:
        return 0
    if u < 0.50:
        return 1
    if u < 0.75:
        return 2
    return 3


def _apply_sense_to_AP(AP: "AgentParams", level: int) -> None:
    # Basnivå (level 0) = default.
    # Höj stegvis: fler strålar + längre räckvidd + mindre brus.
    if level <= 0:
        AP.n_rays = 12
        AP.ray_len_front = 7.0
        AP.noise_sigma = 0.06
    elif level == 1:
        AP.n_rays = 16
        AP.ray_len_front = 8.0
        AP.noise_sigma = 0.055
    elif level == 2:
        AP.n_rays = 24
        AP.ray_len_front = 10.0
        AP.noise_sigma = 0.050
    else:  # level 3
        AP.n_rays = 32
        AP.ray_len_front = 12.0
        AP.noise_sigma = 0.045

# reproduction helpers

def sigmoid(x: float) -> float:
    # clamp för numerisk stabilitet
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

# Generic helpers
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    
# -------------------------
# Params
# -------------------------
@dataclass
class AgentParams:
    # ------------------------
    # Time discretization
    # ------------------------
    dt: float = 0.02

    # ------------------------
    # Steering / policy kinematics
    # ------------------------
    v_max: float = 2.2
    turn_rate: float = 2.2

    # ------------------------
    # Sensing / perception
    # ------------------------
    n_rays: int = 12
    ray_len_front: float = 7.0    # max räckvidd framåt (ellipsens långa halva)
    ray_eccentricity: float = 0.7  # 0=cirkel → 1=extremt avlångt
                                   # r(θ) = r_front×(1-e)/(1-e×cos(θ))
                                   # sida(90°)=r_front×(1-e)≈2.1, bak(180°)≈1.2
    ray_step: float = 1.0
    noise_sigma: float = 0.06

    # Adaptiv sensing-rate (fladdermus-modell):
    # Agenter scannar omgivningen lågfrekvent i frånvaro av stimuli,
    # och växlar till hög frekvens + kortare räckvidd när mat eller grannar detekteras.
    # sense_idle_steps:    steg mellan fullständiga skanningar i frånvaro av stimuli
    #                      (idle=10 → 80% av skanningarna sparas i tomma miljöer)
    # sense_alert_steps:   steg mellan skanningar när stimuli detekteras
    #                      (alert=3 → tät tracking av känt objekt)
    # sense_alert_thresh:  u-signal som räknas som "detekterad" (0=ingenting, 1=max)
    sense_idle_steps: int = 10
    sense_alert_steps: int = 3
    sense_alert_thresh: float = 0.15

    # Sensing upkeep costs (per mass, per second via metabolism_scale)
    sense_cost_L1: float = 0.2
    sense_cost_L2: float = 0.5
    sense_cost_L3: float = 1.0

    # ------------------------
    # Feeding: world pool units -> internal energy units
    # ------------------------
    eat_rate: float = 1e-4  # kg/s

    # Energy densities (J/kg)
    # E_bio_J_per_kg behålls som bakåtkompatibelt alias för växtbiomassa.
    E_bio_J_per_kg: float = 4.0e6
    E_plant_J_per_kg: float = 4.0e6
    E_carcass_J_per_kg: float = 7.0e6
    E_body_J_per_kg: float = 7.0e6

    # Conversion efficiencies
    digest_eff_plant: float = 0.55
    digest_eff_carcass: float = 0.75
    anabolism_eff: float = 0.70
    catabolism_eff: float = 0.90

    # Energy storage capacity (J/kg)
    E_cap_per_M: float = 3.0e5

    # ------------------------
    # Energy ledger diagnostics
    # ------------------------
    ledger_eps_abs: float = 1e-8   # NEW: absolute drift tolerance
    ledger_eps_rel: float = 1e-12  # NEW: relative drift tolerance
    assert_ledger: bool = False    # NEW: raise if drift exceeds tolerances    

    # ------------------------
    # Initial physiological state
    # ------------------------
    M0: float = 1.0   # initial body mass
    E0: float = 1.5e5 # initial energy (J)

    # ------------------------
    # Basal metabolism (allometric)
    # ------------------------
    k_basal: float = 30.0  # [W]

    # Activity-related metabolic costs (non-locomotor)
    compute_cost: float = 5.0  # [W/kg] — skalas upp proportionellt mot n_params vid agentens start

    # Referens-nätverksstorlek för compute_cost-skalning.
    # Baseline = 23-24-24-5 = 1301 parametrar.
    # En agent med ett bredare/djupare nätverk betalar mer metabolt per sekund.
    # compute_cost_eff = compute_cost × (n_params / compute_cost_ref_params)
    compute_cost_ref_params: int = 1301

    # ------------------------
    # Thermoregulation (simple)
    # ------------------------
    Tb_init: float = 37.0
    Tb_set: float = 37.0
    Tb_min: float = 35.0

    heatcap_J_per_kgC: float = 3500.0   # thermal inertia ~ M
    thermo_k_W_per_C: float = 1.0       # heat loss coefficient ~ M^(2/3)
    thermo_mass_exp: float = 2.0 / 3.0
    thermo_Pmax_per_kg: float = 20.0    # max heat generation (W/kg)
    cold_damage_gain: float = 0.03      # damage per second at severe cold

    # ------------------------
    # Locomotion mechanics
    # ------------------------
    F0: float = 4.0
    force_mass_exp: float = 2.0 / 3.0
    drag_lin: float = 0.8
    drag_quad: float = 0.2
    locomotion_eff: float = 0.25

    # ------------------------
    # Starvation / weakness dynamics
    # ------------------------
    M_crit: float = 0.25   # under detta försvagas rörelseförmågan
    M_min: float = 0.07    # absolut minimum — lite under child_M_min för buffert
    v_weak_min: float = 0.25
    rep_weak_min: float = 0.20
    starve_stress_gain: float = 1.0

    # ------------------------
    # Fatigue dynamics
    # ------------------------
    fatigue_recover: float = 0.020
    fatigue_effort: float = 0.050

    # ------------------------
    # Damage / frailty (Body.step uses these)
    # ------------------------
    D_max: float = 1.0
    frailty_gain_cap: float = 1.0  # NEW: clamp for pheno.frailty_gain

    # Aging background damage.
    # k_age1 (linjärt med ålder) är biologiskt korrekt: unga är nästan oskadade,
    # gamla ackumulerar skada exponentiellt snabbare.
    # k_age0=0 → ingen konstant bakgrund (var 0.200 — dödade unga för snabbt).
    # k_age1=0.001 → dD/dt = 0.001×age_s: vid 250s ger 0.25 D/s → döden.
    k_age0: float = 0.000
    k_age1: float = 0.0003     # 0.0002 gav för lång livslängd; 0.0003 ger tipping ~400-500s.
    k_ageD: float = 0.4

    # Skadehastighet — grundterm i dD_eff.
    # Kalibrerat så att ung frisk agent har dD << reparationskapacitet (D ≈ 0),
    # men gammal agent (W ≈ 12-15) får D att stiga mot 1.0 → biologisk åldersdöd.
    # Analytisk kalibrering (normal aktivitet, repair_capacity = 0.055, wear_a0 = 0.04):
    #   dD_eff ≈ 0.00130/steg  →  kritisk W ≈ 12  →  naturlig livslängd ≈ 300s
    k_damage: float = 0.06     # var 0.15 → sänkt: grundskadehastigheten var för dominant mot reparation

    # ------------------------
    # Repair / pain homeostasis
    # ------------------------
    # pain
    pain_k_dD: float = 1.0
    pain_k_D: float = 0.2
    pain_k_hunger: float = 0.0
    pain_tau: float = 0.5

    # repair (energy -> D reduction)
    repair_gain: float = 1.0
    repair_E_per_D: float = 0.005  # var 0.02 → sänkt: billigare reparation gör det evolutionärt lönsamt
    repair_W_decay: float = 0.15   # var 0.3 — mjukare degradering av reparation med ålder
    repair_eta0: float = 1.0
    repair_eta_W: float = 0.1     # var 0.2

    # Legacy / currently unused (kan tas bort när du städat klart)

    # Wear / slitageackumulering (W-dynamik) — driver för åldrande.
    # W växer långsamt hela livet och försämrar reparationskapaciteten exponentiellt.
    #   R_max = repair_capacity × exp(-repair_W_decay × W)
    #
    # Tidsskaleproblem: wear_a0=0.04 gav W=4 vid t=100s (en generation) →
    # reparationen halverades redan i generation 1 → populationskollaps.
    # Mål: W-systemet ska verka på MÅNGA generationers tidsskala.
    #   wear_a0=0.004 → W=0.4 vid t=100s, W=4 vid t=1000s (10 generationer)
    #   repair_W_decay=0.15 → mjukare degradering (var 0.3)
    wear_a0: float = 0.008     # var 0.035 → sänkt: W=2.8 vid t=350s istf 12.25, reparation degraderas långsammare
    wear_aE: float = 0.0
    wear_aR: float = 0.0
    wear_aD: float = 0.002

    # ------------------------
    # Reproduction (Population)
    # ------------------------
    repro_cooldown_s: float = 8.0
    M_repro_soft: float = 0.03
    E_repro_soft: float = 0.05

    # ------------------------
    # Predation
    # ------------------------
    attack_range: float = 1.5        # rutnätsenheter — max avstånd för attack
    attack_damage_per_s: float = 0.3 # D-inflöde per sekund på bytet vid attack
    attack_energy_gain: float = 0.5  # fraktion av bytets förlorade energi som predatorn får
    attack_cost_per_s: float = 0.05  # fraktion av predatorns Ecap per sekund som attacken kostar

    # Selektiv predator-prey-logik
    predator_trait_min: float = 0.20

    # Diet/predation-koppling: jaktförmåga kräver animalisk diet.
    # hunt_eff = predation × diet^hunt_diet_exp
    # En herbivore (diet=0) får hunt_eff=0 oavsett predation-trait.
    # En generalist (diet=0.5) får hunt_eff = predation × 0.5^1.5 ≈ 0.35×predation.
    # En karnivor (diet=1.0) får hunt_eff = predation (fullt utbyte).
    # Mismatch-kostnad: predation utan diet ger extra attackkostnad (kJ slösas).
    hunt_diet_exp: float = 1.5       # exponent för diet-skalning av jaktförmåga
    hunt_mismatch_cost: float = 2.0  # multiplikator på attackkostnad vid låg diet
    threat_predation_min: float = 0.35
    hunt_score_min: float = 0.12
    attack_score_min: float = 0.18
    prey_search_radius: float = 6.0
    mate_search_radius: float = 5.0
    flee_radius: float = 6.0

    birth_E0: float = 0.0
    birth_k_E_per_M: float = 7.0e4
    birth_energy_eff: float = 0.70

    # Aktiv tillväxt mot M_target.
    # Juvenil somatisk tillväxt mot M_target.
    # Viktigt efter energikonsolideringen: tillväxt får bara ske hos omogna
    # individer och bara när reservgraden är tillräckligt hög. Annars driver
    # modellen in i en growth→catabolism-spiral direkt från warm start.
    growth_rate_per_s: float = 0.008
    growth_R_min: float = 0.30   # ingen aktiv tillväxt under denna reservgrad
    growth_R_full: float = 0.60  # full tillväxthastighet först här

    # Gestationstillväxthastighet (kg/s fetal vävnad per sekund).
    # 0.004 kg/s → 50s för ett 0.2 kg foster (var 0.002 = 100s).
    # Föräldern kataboliserar ~0.2 kg kroppsmassa under gestationen (M: 1.0→0.8). ✓
    gestation_growth_kg_per_s: float = 0.004

    # Energikostnad för att bygga fetal vävnad (J/kg).
    # OBS: Detta är INTE samma som E_body_J_per_kg (energidensitet vid katabolism).
    # Anabolisk byggkostnad ≈ metaboliskt arbete för syntes, ej lagrad energi i vävnaden.
    # Kalibrering: 0.002 kg/s × 10 000 J/kg = 20 J/s ≈ 2/3 av basalmetabolismen.
    # Det gör gestation energimässigt rimlig utan att tömma föräldern på sekunder.
    gestation_E_per_kg: float = 10_000.0  # J/kg (var implicit E_body_J_per_kg = 7 000 000)
    growth_E_per_kg:    float = 10_000.0  # J/kg somatisk tillväxt (samma princip som gestation)
    k_cat_dmg:          float = 1.0       # skada per relativ massförlust via katabolism

    # Svältskada baserad på individens massa relativt förväntad massa för åldern.
    # M_expected(age) = child_M  vid age=0
    #                 = M_target  vid age=A_mature (sedan konstant)
    # Massunderskott i relation till åldersnormen är i sig bevis på svält —
    # en agent under kurvan har per definition inte haft tillräcklig energi att växa.
    # Reservvillkor är därför överflödigt: reserven styr om agenten kan *komma ur*
    # svältläget (via tillväxt), men massunderskottet är det direkta skadesignalet.
    starve_mass_ok_frac:   float = 0.85   # ingen svältskada om M >= 85% av M_expected
    starve_mass_crit_frac: float = 0.55   # maximal svältskada om M <= 55% av M_expected
    starve_damage_gain:    float = 0.025  # max extra D/s vid full svält (var 0.08 — för aggressivt)

    # Stokastisk dödsrisk — liten och tillståndsberoende ("olyckor").
    # Biologisk princip: friska agenter har låg olycksrisk; skadade har hög.
    # death_h_base = 0.0 → ingen konstant bakgrundshazard (sätt > 0 för att aktivera).
    # death_h_D    → D-beroende hazard: skadade agenter är mer sårbara för yttre hot.
    # OBS: median_age_s används INTE längre som fallback när death_h_base = 0.
    #      Livslängd emergerar ur W/D/M-systemet, inte ur en parameter.
    median_age_s: float = 50.0   # behålls för bakåtkompatibilitet men används ej som default
    death_h_base: float = 0.0    # konstant hazard (av som default)
    death_h_age: float = 0.0
    death_h_D: float = 0.01      # D-beroende hazard (var 0.0)

# -------------------------
# Body: energy + mass (unbounded) + damage/fatigue (bounded)
# -------------------------

@dataclass
class Body:
    AP: "AgentParams"  # <-- var tidigare P: AgentParams (namnkrock)

    # homeostatic / damage state
    D: float = 0.0        # acute damage
    W: float = 0.0        # wear/frailty
    P: float = 0.0        # pain/drive
    _D_prev: float = 0.0  # for dD/dt

    # energy buffers (weighted in E_total)
    E_fast: float = 0.0
    E_slow: float = 0.0

    # structural state
    M: float = 0.0        # body mass
    Fg: float = 0.15      # fatigue level
    Tb: float = 37.0      # Target body temperature

    alive: bool = True

    # energy ledger diagnostics (per-individual)
    last_ledger: dict | None = None
    last_flux: dict | None = None
    ledger_steps: int = 0
    ledger_bad_steps: int = 0
    ledger_max_abs: float = 0.0
    ledger_max_rel: float = 0.0
    ledger_worst: dict | None = None

    # numerical guard diagnostics (per-individual)
    guard_steps: int = 0
    guard_killed: int = 0
    guard_clamp_steps: int = 0
    guard_last: dict | None = None

    # --- gestation buffer (netto-delta method) ---
    gestating: bool = False
    gest_M: float = 0.0          # accumulated fetal mass (M-units)
    gest_E_J: float = 0.0        # accumulated fetal energy in J (weighted space)
    gest_M_target: float = 0.0   # target fetal mass
    
    def E_total(self) -> float:
        return 0.6 * self.E_fast + 0.4 * self.E_slow

    def E_cap(self) -> float:
        M = self.M
        return self.AP.E_cap_per_M * (M if M > 1e-9 else 1e-9)

    def reserve_frac(self) -> float:
        Ecap = self.E_cap()
        if Ecap <= 1e-9:
            return 0.0
        r = self.E_total() / Ecap
        return 0.0 if r < 0.0 else 1.0 if r > 1.0 else r

    def hunger(self) -> float:
        Et   = self.E_total()
        Ecap = self.E_cap()
        h    = (Ecap - Et) / (Ecap if Ecap > 1e-9 else 1e-9)
        return 0.0 if h < 0.0 else 1.0 if h > 1.0 else h

    def weakness(self) -> float:
        m = float(self.M)
        mcrit = float(self.AP.M_crit)
        if m >= mcrit:
            return 1.0
        if mcrit <= 1e-9:
            return 0.0
        return clamp(m / mcrit, 0.0, 1.0)

    def step_pain_and_repair(self, ctx, pheno, *, D_before: float) -> float:    
        """
        Updates pain P and converts energy -> repair (reduces D).
        Returns repair energy spent (for logging/aging accounting).
        """
        AP = self.AP
        dt = float(ctx.dt)
        dD = (float(self.D) - float(D_before)) / max(1e-9, dt)
        dD_pos = dD if dD > 0.0 else 0.0
    
        hunger = float(self.hunger())
    
        P_target = (
            float(AP.pain_k_dD) * dD_pos +
            float(AP.pain_k_D)  * float(self.D) +
            float(AP.pain_k_hunger) * hunger
        )
    
        alpha = dt / max(1e-9, float(AP.pain_tau))
        alpha = _clamp01(alpha)
        self.P = float(self.P) + (P_target - float(self.P)) * alpha
    
        # R_max styrs av pheno.repair_capacity, inte AP
        R_max = float(pheno.repair_capacity) * math.exp(-float(AP.repair_W_decay) * float(self.W))
        R_des = max(0.0, float(AP.repair_gain) * float(self.P))
        R_des = min(R_des, R_max)
    
        E_per_D = max(1e-9, float(AP.repair_E_per_D))
        E_need = R_des * E_per_D
        E_paid = float(self.take_energy(E_need))
        R = E_paid / E_per_D
    
        eta = float(AP.repair_eta0) * math.exp(-float(AP.repair_eta_W) * float(self.W))
        self.D = max(0.0, float(self.D) - eta * R)
    
        self._D_prev = float(self.D)   # om du fortfarande vill ha den som debug/state
        return E_paid

    def step_aging(
        self,
        ctx,
        *,
        E_spent_total: float,
        repro_cost_paid: float,
        dD_pos: float,
    ) -> None:
        AP = self.AP
        dt = float(ctx.dt)
        self.W = float(self.W) + dt * (
            float(AP.wear_a0) +
            float(AP.wear_aE) * float(E_spent_total) +
            float(AP.wear_aR) * float(repro_cost_paid) +
            float(AP.wear_aD) * float(dD_pos)
        )    
        
    def move_factor(self) -> float:
        w = float(self.weakness())
        return float(self.AP.v_weak_min + (1.0 - float(self.AP.v_weak_min)) * w)

    def repair_factor(self) -> float:
        w = float(self.weakness())
        return float(self.AP.rep_weak_min + (1.0 - float(self.AP.rep_weak_min)) * w)

    def _finite(self, x: float) -> bool:
        # Behålls för bakåtkompatibilitet men används ej internt längre.
        return math.isfinite(float(x))

    def _guard_snapshot(self, where: str) -> dict:
        return {
            "where": where,
            "E_fast": float(self.E_fast),
            "E_slow": float(self.E_slow),
            "M": float(self.M),
            "D": float(self.D),
            "Fg": float(self.Fg),
            "alive": bool(self.alive),
        }

    def start_gestation(self, M_target: float) -> bool:
        Mt = max(0.0, float(M_target))
        if Mt <= 0.0:
            return False
        if self.gestating:
            return False
        self.gestating = True
        self.gest_M = 0.0
        self.gest_E_J = 0.0
        self.gest_M_target = Mt
        return True
    
    def abort_gestation(self) -> None:
        # Nothing to refund because buffers were taken from net deltas (already removed from parent)
        self.gestating = False
        self.gest_M = 0.0
        self.gest_E_J = 0.0
        self.gest_M_target = 0.0
    
    def gestation_ready(self) -> bool:
        return bool(self.gestating and (float(self.gest_M) >= float(self.gest_M_target) > 0.0))
        
    def take_energy(self, amount: float) -> float:
        amt = float(max(0.0, amount))
        if amt <= 0.0:
            return 0.0

        Et = float(self.E_total())
        if Et <= 1e-12:
            return 0.0

        share_fast = (0.6 * float(self.E_fast)) / max(Et, 1e-12)
        share_slow = (0.4 * float(self.E_slow)) / max(Et, 1e-12)

        d_fast = (amt * share_fast) / 0.6
        d_slow = (amt * share_slow) / 0.4

        d_fast = min(d_fast, float(self.E_fast))
        d_slow = min(d_slow, float(self.E_slow))

        self.E_fast = max(0.0, float(self.E_fast) - d_fast)
        self.E_slow = max(0.0, float(self.E_slow) - d_slow)

        return float(0.6 * d_fast + 0.4 * d_slow)

    def scale_energy(self, factor: float) -> None:
        f = max(0.0, float(factor))
        self.E_fast = float(self.E_fast) * f
        self.E_slow = float(self.E_slow) * f
    
    def clamp_energy_to_cap(self) -> None:
        Et = float(self.E_total())
        Ecap = float(self.E_cap())
        if Et <= Ecap:
            return
        k = Ecap / max(Et, 1e-12)
        self.scale_energy(k)
        
    def _sense_cost(self, pheno: Phenotype) -> float:
        level = _sense_level(float(getattr(pheno, "sense_strength", 0.0)))
        if level == 1:
            return float(self.AP.sense_cost_L1)
        if level == 2:
            return float(self.AP.sense_cost_L2)
        if level >= 3:
            return float(self.AP.sense_cost_L3)
        return 0.0

    def expected_mass(self, pheno: Phenotype, age_s: float) -> float:
        """Förväntad kroppsmassa givet utvecklingsstadium."""
        child_M = max(float(self.AP.M_min), float(getattr(pheno, "child_M", self.AP.M_min)))
        M_target = max(child_M, float(getattr(pheno, "M_target", float(self.AP.M0))))
        A_mature = max(1e-9, float(getattr(pheno, "A_mature", 1.0)))
        u_age = clamp(float(age_s) / A_mature, 0.0, 1.0)
        return child_M + u_age * (M_target - child_M)

    def step(
        self,
        ctx: "StepCtx",
        *,
        speed: float,
        activity: float,
        food_bio_kg: float,
        food_carcass_kg: float,
        pheno: Phenotype,
        extra_drain: float = 0.0,
        T_env: float = 0.0,
        age_s: float = 0.0,
    ) -> None:
        """
        Hazard removed.
    
        Single-pay design:
          - Intake updates stores/mass.
          - Compute all drains (including thermo + gestation overhead/build).
          - Pay drains ONCE from buffers.
          - If short -> catabolize mass above M_min to cover deficit, then pay remaining.
          - Then repair/pain (separate payment).
          - Ledger checks energy consistency.
    
        Gestation model ("Väg 2"):
          - gest_M / gest_E_J tracked separately from structural M.
          - gestation build consumes energy (and optionally catabolism) to convert into fetal tissue.
          - (optional) gestation overhead is just another drain term.
        """
        if not self.alive:
            return
    
        dt = float(ctx.dt)
        rng = ctx.rng
    
        WF = 0.6
        WS = 0.4
    
        # ---------------------------------------------------------
        # (0) Numerical guards (pre) — inlineat för att undvika metod-overhead
        # ---------------------------------------------------------
        if not (
            math.isfinite(self.E_fast)
            and math.isfinite(self.E_slow)
            and math.isfinite(self.M)
            and math.isfinite(self.D)
            and math.isfinite(self.Fg)
        ):
            self.guard_steps += 1
            self.guard_killed += 1
            self.guard_last = self._guard_snapshot("pre_state")
            self.alive = False
            return
    
        if not (
            math.isfinite(speed)
            and math.isfinite(activity)
            and math.isfinite(food_bio_kg)
            and math.isfinite(food_carcass_kg)
            and math.isfinite(extra_drain)
            and math.isfinite(age_s)
        ):
            self.guard_steps += 1
            self.guard_killed += 1
            self.guard_last = {
                **self._guard_snapshot("pre_inputs"),
                "speed": float(speed),
                "activity": float(activity),
                "food_bio_kg": float(food_bio_kg),
                "food_carcass_kg": float(food_carcass_kg),
                "extra_drain": float(extra_drain),
                "age_s": float(age_s),
            }
            self.alive = False
            return
    
        # ---------------------------------------------------------
        # (0B) Ledger baselines
        # ---------------------------------------------------------
        E_before = float(self.E_total())
        M_before = float(self.M)

        # ---------------------------------------------------------
        # (0C) Cacha AP-parametrar som lokala variabler
        # Lokala variabler (LOAD_FAST) är ~3× snabbare än attributuppslag.
        # Alla dessa parametrar är konstanta under agentens liv.
        # ---------------------------------------------------------
        AP = self.AP
        _E_bio        = float(getattr(AP, 'E_plant_J_per_kg', AP.E_bio_J_per_kg))
        _E_carcass    = float(AP.E_carcass_J_per_kg)
        _E_body       = float(AP.E_body_J_per_kg)
        _dig_eff_bio  = float(getattr(AP, 'digest_eff_plant', 1.0))
        _dig_eff_car  = float(getattr(AP, 'digest_eff_carcass', 1.0))
        _ana_eff      = max(1e-9, float(getattr(AP, 'anabolism_eff', 1.0)))
        _cat_eff      = max(0.0, float(getattr(AP, 'catabolism_eff', 1.0)))
        _E_cap_per_M  = float(AP.E_cap_per_M)
        _k_basal      = float(AP.k_basal)
        _compute_cost = float(AP.compute_cost)
        _v_max        = float(AP.v_max)
        _D_max        = float(AP.D_max)
        _M_min        = float(AP.M_min)
        _M_crit       = float(AP.M_crit)
        _starve_gain  = float(AP.starve_stress_gain)
        _frailty_cap  = float(AP.frailty_gain_cap)
        _fatigue_eff  = float(AP.fatigue_effort)
        _fatigue_rec  = float(AP.fatigue_recover)
        _Tb_set       = float(AP.Tb_set)
        _Tb_min       = float(AP.Tb_min)
        _thermo_k     = float(AP.thermo_k_W_per_C)
        _thermo_exp   = float(AP.thermo_mass_exp)
        _heatcap      = float(AP.heatcap_J_per_kgC)
        _thermo_Pmax  = float(AP.thermo_Pmax_per_kg)
        _cold_dmg     = float(AP.cold_damage_gain)
        _gest_burden  = float(getattr(AP, "gestation_mass_burden", 0.0))
        _gest_over    = float(getattr(AP, "gestation_P_overhead_per_kg", 0.0))
        _gest_rate    = float(AP.gestation_growth_kg_per_s)
        _k_damage     = float(getattr(AP, "k_damage", 0.02))
        _k_age0       = float(AP.k_age0)
        _k_age1       = float(AP.k_age1)
        _k_ageD       = float(AP.k_ageD)
        _h_base       = float(AP.death_h_base)
        _h_age        = float(AP.death_h_age)
        _h_D          = float(AP.death_h_D)
        _loco_eff     = float(AP.locomotion_eff)
        _wear_a0      = float(AP.wear_a0)
        _wear_aE      = float(AP.wear_aE)
        _wear_aD      = float(AP.wear_aD)
        _growth_rate  = float(AP.growth_rate_per_s)
        # M_target: genetiskt bestämd vuxenmassa från phenotype
        _M_target     = float(getattr(pheno, "M_target", float(AP.M0)))
        _build_E_kg        = _E_body / _ana_eff
        _gest_build_E_kg   = float(getattr(AP, 'gestation_E_per_kg', 10_000.0))
        _growth_build_E_kg = float(getattr(AP, 'growth_E_per_kg',    10_000.0))
    
        # ---------------------------------------------------------
        # (1) Intake -> buffers (up to cap), surplus -> mass
        # ---------------------------------------------------------
        m_bio = max(0.0, float(food_bio_kg))
        m_car = max(0.0, float(food_carcass_kg))

        # Kostpreferens (diet-trait): 0=herbivore, 1=scavenger.
        # herb_eff och scav_eff är negativt korrelerade — generalisten (0.5)
        # är sämre på båda än en specialist.
        # herb_eff(d) = (1-d)^0.7,  scav_eff(d) = d^0.7
        # Vid d=0: herb=1.00, scav=0.00
        # Vid d=0.5: herb=0.62, scav=0.62  (generalisten förlorar ~38%)
        # Vid d=1: herb=0.00, scav=1.00
        _diet     = float(getattr(pheno, "diet", 0.5))
        herb_eff  = (1.0 - _diet) ** 0.7
        scav_eff  = _diet ** 0.7

        E_in_gross_bio = m_bio * _E_bio * herb_eff
        E_in_gross_car = m_car * _E_carcass * scav_eff
        E_in_bio = E_in_gross_bio * _dig_eff_bio
        E_in_car = E_in_gross_car * _dig_eff_car
        E_loss_digest_bio = max(0.0, E_in_gross_bio - E_in_bio)
        E_loss_digest_car = max(0.0, E_in_gross_car - E_in_car)
        E_in = E_in_bio + E_in_car

        Et0 = float(self.E_total())
        Ecap0 = _E_cap_per_M * max(1e-9, float(self.M))
        room = max(0.0, Ecap0 - Et0)

        dE_store = min(E_in, room)

        if dE_store > 0.0:
            dE_fast_w = 0.85 * dE_store
            dE_slow_w = 0.15 * dE_store
            self.E_fast += dE_fast_w / WF
            self.E_slow += dE_slow_w / WS

        E_to_M = max(0.0, E_in - dE_store)
        # OBS: energiöverskott lagras inte längre som massa här.
        # Tillväxt sker via det genetiska M_target-drivet i sektion 2C.5.
        # Eventuellt överskott stannar i energibuffertarna (kläms mot Ecap nedan).

        # ---------------------------------------------------------
        # (2) Effective mass (carried load) + basal/compute/sense/loco
        # ---------------------------------------------------------
        M_carry = float(self.M)
        if bool(self.gestating):
            M_carry += _gest_burden * max(0.0, float(self.gest_M))

        M_eff = max(1e-9, M_carry)
        metab = float(pheno.metabolism_scale)

        # Basalmetabolism skalas med nuvarande massa (M_eff).
        # Det obligatoriska tillväxtdrivet (sektion 2C.5) är den mekanism som
        # förhindrar r-strategi via minimerad massa — inte metaboliken.
        out_basal   = dt * metab * (M_eff ** 0.75) * _k_basal
        out_compute = dt * metab * M_eff * _compute_cost * float(activity)

        sense_cost = float(self._sense_cost(pheno))
        out_sense  = dt * metab * M_eff * sense_cost

        out_loco = max(0.0, float(extra_drain))
    
        # ---------------------------------------------------------
        # (2B) Thermoregulation (ledger-consistent)
        # ---------------------------------------------------------
        Tb   = float(self.Tb)
        Tenv = float(T_env)

        K   = _thermo_k * (M_eff ** _thermo_exp)
        Cth = max(1e-9, _heatcap * M_eff)

        P_need = max(0.0, K * (_Tb_set - Tenv)) if Tb < _Tb_set else 0.0
        P_gen  = min(P_need, _thermo_Pmax * M_eff)

        out_thermo = dt * P_gen

        Qloss   = K * (Tb - Tenv)
        dTb     = (P_gen - Qloss) * dt / Cth
        self.Tb = Tb + dTb
    
        # ---------------------------------------------------------
        # (2C) Gestation (Väg 2): overhead + build energy
        # ---------------------------------------------------------
        out_gest_overhead = 0.0   # J
        out_gest_build = 0.0      # J actually paid this tick for fetal tissue
        dM_gest = 0.0             # kg fetal tissue built this tick
    
        # diagnostics for ledger
        dM_cat_gest = 0.0         # kg catabolized specifically to support gestation build
        E_from_M_gest = 0.0       # J injected via that catabolism (then spent)

        if bool(self.gestating):
            Pg_over = _gest_over * M_eff
            out_gest_overhead = dt * Pg_over

            M_tgt = max(0.0, float(self.gest_M_target))
            M_cur = max(0.0, float(self.gest_M))

            if M_tgt > 0.0 and M_cur < M_tgt:
                dM_want = min(_gest_rate * dt, M_tgt - M_cur)

                if dM_want > 0.0:
                    E_need = dM_want * _gest_build_E_kg
    
                    # pay from buffers
                    paid1 = float(self.take_energy(E_need))
                    deficit_g = max(0.0, E_need - paid1)
    
                    paid2 = 0.0
                    if deficit_g > 0.0:
                        free = max(0.0, float(self.M) - _M_min)

                        want_cat    = deficit_g / max(_cat_eff * _E_body, 1e-12)
                        dM_cat_gest = min(want_cat, free)

                        if dM_cat_gest > 0.0:
                            self.M -= dM_cat_gest
                            E_from_M_gest = dM_cat_gest * _E_body * _cat_eff
                            self.E_fast += E_from_M_gest / WF

                        paid2 = float(self.take_energy(deficit_g))

                    paid_total = paid1 + paid2
                    out_gest_build = paid_total

                    # Massa byggd = betald energi / byggkostnad per kg.
                    dM_gest = paid_total / _gest_build_E_kg
                    if dM_gest > 0.0:
                        self.gest_M = M_cur + dM_gest
                        self.gest_E_J = float(self.gest_E_J) + paid_total
    
        # ---------------------------------------------------------
        # (2C.5) Aktiv juvenil tillväxt mot M_target
        # Efter energikonsolideringen får tillväxt inte vara ett konstant drag
        # hos alla individer under M_target; det dödar warm-startade vuxna som
        # försöker "växa ikapp" genom akut katabolism. Därför krävs nu både:
        #   (a) omogen ålder, och
        #   (b) tillräcklig energireserv.
        # Reservgaten är mjuk mellan growth_R_min och growth_R_full.
        # Juvenil-gate borttagen: indeterminerad tillväxt mot M_target oavsett ålder.
        # Biologisk motivering: fiskar, reptiler och de flesta djur växer kontinuerligt
        # mot ett genetiskt storleksmål under hela livet om energi finns. Den gamla
        # juvenil-gaten innebar att A_mature (5–20s) var för kort för att hinna växa
        # från child_M (0.14 kg) till M_target (1+ kg) → permanenta miniagenterna.
        # Reservgaten (growth_R_min) är det primära skyddet mot okontrollerad tillväxt.
        # ---------------------------------------------------------
        out_growth = 0.0
        dM_growth_want = 0.0
        r_now = self.reserve_frac()
        gR0 = float(getattr(AP, 'growth_R_min', 0.30))
        gR1 = max(gR0 + 1e-9, float(getattr(AP, 'growth_R_full', 0.60)))
        if r_now <= gR0:
            growth_gate = 0.0
        elif r_now >= gR1:
            growth_gate = 1.0
        else:
            growth_gate = (r_now - gR0) / (gR1 - gR0)

        if float(self.M) < _M_target and growth_gate > 0.0:
            M_deficit      = _M_target - float(self.M)
            dM_growth_want = min(_growth_rate * dt * growth_gate, M_deficit)
            out_growth     = dM_growth_want * _growth_build_E_kg

        # ---------------------------------------------------------
        # (2D) Pay drains ONCE
        # ---------------------------------------------------------
        # OBS: out_gest_build ingår INTE — redan betald i sektion (2C).
        E_out_drain = (
            out_basal + out_compute + out_sense + out_loco + out_thermo
            + out_gest_overhead + out_growth
        )
    
        paid = float(self.take_energy(E_out_drain))
        deficit = max(0.0, E_out_drain - paid)
        E_paid_drain = paid

        # Applicera tillväxt proportionellt mot betald fraktion
        if dM_growth_want > 0.0 and E_out_drain > 1e-12:
            frac_paid  = paid / E_out_drain
            dM_growth  = dM_growth_want * frac_paid
            if dM_growth > 0.0:
                self.M = float(self.M) + dM_growth
    
        # ---------------------------------------------------------
        # (3) Catabolism: cover remaining deficit (if any), above M_min
        # ---------------------------------------------------------
        E_from_M = 0.0
        dM_cat = 0.0
        paid2 = 0.0
    
        if deficit > 0.0:
            want_cat = deficit / max(_cat_eff * _E_body, 1e-12)
            free     = max(0.0, float(self.M) - _M_min)
            dM_cat   = min(want_cat, free)

            if dM_cat > 0.0:
                self.M   -= dM_cat
                E_from_M  = dM_cat * _E_body * _cat_eff
                self.E_fast += E_from_M / WF
                _k_cat_dmg = float(getattr(AP, 'k_cat_dmg', 1.0))
                dD_cat = _k_cat_dmg * dM_cat / max(float(self.M), 1e-9)
                self.D = clamp(float(self.D) + dD_cat, 0.0, _D_max)

            paid2      = float(self.take_energy(deficit))
            deficit    = max(0.0, deficit - paid2)
            E_paid_drain += paid2
    
        # snapshot after drains/catabolism (for stress math)
        Et = float(self.E_total())
        Ecap = float(self.E_cap())
    
        # ---------------------------------------------------------
        # (4) Damage + repair + fatigue
        # ---------------------------------------------------------
        D_before = float(self.D)

        e_lack   = clamp((Ecap - Et) / max(Ecap, 1e-9), 0.0, 1.0)
        d_norm   = clamp(D_before / max(_D_max, 1e-9), 0.0, 1.0)

        speed_n = clamp(float(speed) / max(_v_max, 1e-9), 0.0, 1.0)
        effort  = speed_n + 0.6 * float(activity)
        rest    = max(0.0, 1.0 - speed_n) * max(0.0, 1.0 - float(activity))

        w = float(self.weakness())
        starve_stress = 1.0 + _starve_gain * (1.0 - w)

        susc         = float(pheno.susceptibility)
        frailty_gain = clamp(float(pheno.frailty_gain), 0.0, max(0.0, _frailty_cap))
        frail        = 1.0 + frailty_gain * d_norm

        drain_rate   = E_out_drain / max(dt, 1e-12)
        drain_rate_n = drain_rate / max(Ecap, 1e-9)

        dD_eff = dt * (_k_damage * susc * (1.0 + 1.2 * e_lack) * frail * effort * starve_stress)
        dD_met = dt * (float(pheno.stress_per_drain) * drain_rate_n * starve_stress)

        age_rate = max(0.0, _k_age0 + _k_age1 * float(age_s))
        dD_age   = dt * age_rate * (1.0 + _k_ageD * d_norm) * (1.0 + frailty_gain)

        # Svältskada: individens massa relativt förväntad massa för åldern.
        # M_expected är linjär från child_M (age=0) till M_target (age=A_mature),
        # sedan konstant. En agent under kurvan har inte kunnat växa i takt — svälter.
        M_expected = self.expected_mass(pheno, age_s)
        m_rel = float(self.M) / max(M_expected, 1e-9)
        m_ok   = float(getattr(AP, 'starve_mass_ok_frac',   0.85))
        m_crit = float(getattr(AP, 'starve_mass_crit_frac', 0.55))
        if m_rel >= m_ok:
            mass_severity = 0.0
        elif m_rel <= m_crit:
            mass_severity = 1.0
        else:
            mass_severity = (m_ok - m_rel) / max(m_ok - m_crit, 1e-9)
        dD_starve = dt * float(getattr(AP, 'starve_damage_gain', 0.025)) * mass_severity

        Tb_now = float(self.Tb)
        if Tb_now < _Tb_min:
            sev    = clamp((_Tb_min - Tb_now) / 10.0, 0.0, 1.0)
            dD_cold = dt * _cold_dmg * sev
        else:
            dD_cold = 0.0

        dD_in      = dD_eff + dD_met + dD_age + dD_starve + dD_cold
        dD_pos_rate = dD_in / max(dt, 1e-9)

        self.D = clamp(D_before + dD_in, 0.0, _D_max)

        E_pain_repair = float(self.step_pain_and_repair(ctx, pheno, D_before=D_before))
        E_out_repair  = E_pain_repair

        E_spent_total = float(E_paid_drain) + float(E_out_repair)
        self.step_aging(
            ctx,
            E_spent_total=E_spent_total,
            repro_cost_paid=0.0,
            dD_pos=float(dD_pos_rate),
        )

        d_norm2 = clamp(float(self.D) / max(_D_max, 1e-9), 0.0, 1.0)
        fatigue_effort_eff  = _fatigue_eff * (1.0 + 0.4 * d_norm2)
        fatigue_recover_eff = _fatigue_rec * max(0.0, 1.0 - 0.05 * d_norm2)

        self.Fg = clamp(
            float(self.Fg) + dt * (fatigue_effort_eff * effort - fatigue_recover_eff * rest),
            0.0, 1.0,
        )
    
        # enforce storage capacity
        Et = float(self.E_total())
        Ecap = float(self.E_cap())
        if Et > Ecap:
            overflow = Et - Ecap
    
            take_fast_w = min(WF * float(self.E_fast), overflow)
            if take_fast_w > 0.0:
                self.E_fast = float(self.E_fast) - (take_fast_w / WF)
                overflow -= take_fast_w
    
            if overflow > 0.0:
                take_slow_w = min(WS * float(self.E_slow), overflow)
                if take_slow_w > 0.0:
                    self.E_slow = float(self.E_slow) - (take_slow_w / WS)
                    overflow -= take_slow_w
    
        # ---------------------------------------------------------
        # (5) Deterministic death conditions
        # ---------------------------------------------------------
        if float(self.D) >= _D_max or float(self.M) <= _M_min:
            self.alive = False
            return

        # ---------------------------------------------------------
        # (6) Stochastic death
        # ---------------------------------------------------------
        if rng is not None:
            hazard_rate = max(0.0, _h_base + _h_age * float(age_s) + _h_D * d_norm2)
            if hazard_rate > 0.0:
                p = 1.0 - math.exp(-hazard_rate * dt)
                if rng.random() < p:
                    self.alive = False
                    return
    
        # ---------------------------------------------------------
        # (7) Numerical guard (post)
        # ---------------------------------------------------------
        clamped = False
    
        if float(self.E_fast) < 0.0:
            self.E_fast = 0.0
            clamped = True
        if float(self.E_slow) < 0.0:
            self.E_slow = 0.0
            clamped = True
        if float(self.M) < 0.0:
            self.M = 0.0
            clamped = True
    
        D0 = float(self.D)
        self.D = clamp(D0, 0.0, _D_max)
        if float(self.D) != D0:
            clamped = True
    
        Fg0 = float(self.Fg)
        self.Fg = clamp(Fg0, 0.0, 1.0)
        if float(self.Fg) != Fg0:
            clamped = True
    
        if clamped:
            self.guard_steps += 1
            self.guard_clamp_steps += 1
            self.guard_last = self._guard_snapshot("post_clamp")
    
        if not (
            math.isfinite(self.E_fast)
            and math.isfinite(self.E_slow)
            and math.isfinite(self.M)
            and math.isfinite(self.D)
            and math.isfinite(self.Fg)
        ):
            self.guard_steps += 1
            self.guard_killed += 1
            self.guard_last = self._guard_snapshot("post_state")
            self.alive = False
            return
    
        # ---------------------------------------------------------
        # (8) Ledger finalize
        # ---------------------------------------------------------
        # include gestation-catabolism diagnostics into totals
        E_from_M_total = float(E_from_M) + float(E_from_M_gest)
        dM_cat_total = float(dM_cat) + float(dM_cat_gest)
    
        E_after = float(self.E_total())
        M_after = float(self.M)
    
        expected_E_after = E_before + dE_store + E_from_M_total - E_out_drain - out_gest_build - E_out_repair
    
        drift = E_after - expected_E_after
        drift_abs = abs(drift)
    
        scale = max(
            1.0,
            abs(E_before),
            abs(E_after),
            abs(dE_store),
            abs(E_out_drain),
            abs(E_out_repair),
            abs(E_from_M_total),
        )
        drift_rel = drift / scale
        drift_rel_abs = abs(drift_rel)
    
        eps_abs = float(getattr(self.AP, "ledger_eps_abs", 1e-8))
        eps_rel = float(getattr(self.AP, "ledger_eps_rel", 1e-12))
        ok = (drift_abs <= eps_abs) or (drift_rel_abs <= eps_rel)
    
        self.ledger_steps = int(getattr(self, "ledger_steps", 0)) + 1
        if not ok:
            self.ledger_bad_steps = int(getattr(self, "ledger_bad_steps", 0)) + 1
    
        prev_max_abs = float(getattr(self, "ledger_max_abs", 0.0))
        prev_max_rel = float(getattr(self, "ledger_max_rel", 0.0))
    
        self.ledger_max_abs = max(prev_max_abs, drift_abs)
        self.ledger_max_rel = max(prev_max_rel, drift_rel_abs)
    
        self.last_ledger = {
            "ok": ok,
            "eps_abs": eps_abs,
            "eps_rel": eps_rel,
            "scale": scale,
            "drift": drift,
            "drift_abs": drift_abs,
            "drift_rel": drift_rel,
    
            "E_before": E_before,
            "E_in": E_in,
            "E_in_bio": E_in_bio,
            "E_in_carcass": E_in_car,
            "E_in_gross_bio": E_in_gross_bio,
            "E_in_gross_carcass": E_in_gross_car,
            "E_loss_digest_bio": E_loss_digest_bio,
            "E_loss_digest_carcass": E_loss_digest_car,
            "E_store": dE_store,
            "E_to_M": E_to_M,
    
            "E_out_basal": out_basal,
            "E_out_compute": out_compute,
            "E_out_sense": out_sense,
            "E_out_loco": out_loco,
            "E_out_thermo": out_thermo,
            "E_out_gest_overhead": out_gest_overhead,
            "E_out_gest_build": out_gest_build,
            "E_out_growth": out_growth,
            "E_out_drain": E_out_drain,

            "E_out_repair": E_out_repair,
            "E_from_M": E_from_M_total,
            "deficit": deficit,
            "E_after": E_after,

            "M_before": M_before,
            "dM_growth": dM_growth_want,
            "dM_cat": dM_cat_total,
            "M_after": M_after,
    
            # gestation state
            "gestating": bool(self.gestating),
            "gest_M": float(self.gest_M),
            "gest_M_target": float(self.gest_M_target),
            "gest_E_J": float(self.gest_E_J),
            "dM_gest": dM_gest,
            "dM_cat_gest": dM_cat_gest,
            "M_expected": M_expected,
            "m_rel_expected": m_rel,
            "dD_starve": dD_starve,
        }
    
        self.last_flux = {
            "food_bio_kg": float(m_bio),
            "food_carcass_kg": float(m_car),
            "E_in_bio": float(E_in_bio),
            "E_in_carcass": float(E_in_car),
            "E_in_total": float(E_in),
            "E_loss_digest_bio": float(E_loss_digest_bio),
            "E_loss_digest_carcass": float(E_loss_digest_car),
            "E_loss_basal": float(out_basal),
            "E_loss_compute": float(out_compute),
            "E_loss_sense": float(out_sense),
            "E_loss_loco": float(out_loco),
            "E_loss_thermo": float(out_thermo),
            "E_loss_gest_overhead": float(out_gest_overhead),
            "E_build_growth": float(out_growth),
            "E_build_gestation": float(out_gest_build),
            "E_loss_repair": float(E_out_repair),
            "E_from_catabolism": float(E_from_M_total),
            "E_loss_catabolism": float(max(0.0, dM_cat_total * _E_body - E_from_M_total)),
            "dM_growth": float(dM_growth_want),
            "M_expected": float(M_expected),
            "m_rel_expected": float(m_rel),
            "dD_starve": float(dD_starve),
            "reserve_frac": float(r_now),
            "growth_gate": float(growth_gate),
            "dM_gestation": float(dM_gest),
            "dM_catabolism": float(dM_cat_total),
        }

        if drift_abs >= prev_max_abs:
            self.ledger_worst = dict(self.last_ledger)
    
        if bool(getattr(self.AP, "assert_ledger", False)) and (not ok):
            raise AssertionError(
                f"Energy ledger drift: abs={drift_abs:.3e} rel={drift_rel:.3e} "
                f"(eps_abs={eps_abs:.3e}, eps_rel={eps_rel:.3e})"
            )


# -------------------------
# Ray sensors (B and C only)
# -------------------------
@dataclass
class RaySensors:
    AP: AgentParams
    world_size: int

    _n: int = field(init=False, default=0)
    _m: int = field(init=False, default=0)

    _ang_base: np.ndarray = field(init=False)
    _ang: np.ndarray = field(init=False)
    _d: np.ndarray = field(init=False)
    _w: np.ndarray = field(init=False)
    _wsum: np.float32 = field(init=False)
    _inv_wsum: np.float32 = field(init=False)

    _dx: np.ndarray = field(init=False)
    _dy: np.ndarray = field(init=False)
    _xs: np.ndarray = field(init=False)
    _ys: np.ndarray = field(init=False)

    # samples (kg); overwritten to u for integration
    _Bp: np.ndarray = field(init=False)
    _Cp: np.ndarray = field(init=False)

    # ray accumulators (u-domain)
    _accB: np.ndarray = field(init=False)
    _accC: np.ndarray = field(init=False)

    _noiseB: np.ndarray = field(init=False)
    _noiseC: np.ndarray = field(init=False)
    _noise64: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rebuild_cache()

    @staticmethod
    def _sat_u(x_kg: np.ndarray, K: float) -> np.ndarray:
        if K <= 0.0:
            return (x_kg > 0.0).astype(np.float32, copy=False)
        np.maximum(x_kg, 0.0, out=x_kg)
        x_kg /= (x_kg + np.float32(K))
        return x_kg

    @staticmethod
    def _sat1_u(x_kg: float, K: float) -> float:
        x = 0.0 if x_kg < 0.0 else float(x_kg)
        if K <= 0.0:
            return 1.0 if x > 0.0 else 0.0
        return float(x / (x + K))

    def _rebuild_cache(self) -> None:
        n    = int(self.AP.n_rays)
        step = float(self.AP.ray_step)
        r_front = float(self.AP.ray_len_front)
        e       = max(0.0, min(0.999, float(self.AP.ray_eccentricity)))

        self._n = max(0, n)

        def z1(dtype=np.float32):
            return np.zeros((0,), dtype=dtype)
        def z2(dtype=np.float32):
            return np.zeros((0, 0), dtype=dtype)

        if self._n <= 0 or step <= 0.0 or r_front <= 0.0:
            self._m = 0
            self._ray_m = np.zeros((0,), dtype=np.int32)
            self._ang_base = z1(); self._ang = z1()
            self._d = z1(); self._w = z1()
            self._wsum = np.float32(1.0); self._inv_wsum = np.float32(1.0)
            self._dx = z1(); self._dy = z1()
            self._xs = z2(); self._ys = z2()
            self._Bp = z2(); self._Cp = z2()
            self._accB = z1(); self._accC = z1()
            self._noiseB = z1(); self._noiseC = z1()
            self._noise64 = z1(dtype=np.float64)
            self._ixs = np.empty((0, 0), dtype=np.int32)
            self._iys = np.empty((0, 0), dtype=np.int32)
            return

        # Strålvinklar i [0, 2π)
        self._ang_base = (
            np.float32(2.0 * np.pi)
            * (np.arange(self._n, dtype=np.float32) / np.float32(self._n))
        )
        self._ang = np.empty((self._n,), dtype=np.float32)

        # Buffrar allokeras för maximalt djup (framåt = r_front)
        self._d = np.arange(step, r_front + 1e-6, step, dtype=np.float32)
        self._m = int(self._d.size)

        # Per-stråle djup via ellipsformeln (polär konik med fokus i origo):
        #   r(θ) = r_front × (1-e) / (1 - e × cos(θ))
        # θ=0   → r_front          (framåt, maximum)
        # θ=π/2 → r_front × (1-e) (sida)
        # θ=π   → r_front×(1-e)/(1+e) (bakåt, minimum)
        self._ray_m = np.empty((self._n,), dtype=np.int32)
        for i in range(self._n):
            ang = float(self._ang_base[i])
            if ang > math.pi:
                ang -= 2.0 * math.pi
            r_i = r_front * (1.0 - e) / (1.0 - e * math.cos(ang))
            m_i = max(1, int(r_i / step + 0.5))
            self._ray_m[i] = min(m_i, self._m)

        self._w = (np.float32(1.0) / (np.float32(1.0) + np.float32(0.25) * self._d)).astype(
            np.float32, copy=False
        )
        self._wsum    = np.sum(self._w, dtype=np.float32) + np.float32(1e-9)
        self._inv_wsum = np.float32(1.0) / self._wsum

        self._dx = np.empty((self._n,), dtype=np.float32)
        self._dy = np.empty((self._n,), dtype=np.float32)
        self._xs = np.empty((self._n, self._m), dtype=np.float32)
        self._ys = np.empty((self._n, self._m), dtype=np.float32)
        self._Bp = np.empty((self._n, self._m), dtype=np.float32)
        self._Cp = np.empty((self._n, self._m), dtype=np.float32)
        self._accB = np.empty((self._n,), dtype=np.float32)
        self._accC = np.empty((self._n,), dtype=np.float32)
        self._noiseB = np.empty((self._n,), dtype=np.float32)
        self._noiseC = np.empty((self._n,), dtype=np.float32)
        self._noise64 = np.empty((self._n,), dtype=np.float64)
        self._ixs = np.empty((self._n, self._m), dtype=np.int32)
        self._iys = np.empty((self._n, self._m), dtype=np.int32)
    def sense(
        self,
        world: World,
        x: float,
        y: float,
        heading: float,
        rng: np.random.Generator | None = None,
        m_eff: int = 0,
    ) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
        """
        Perception API (u-domain):
          - Returns (B0_u, C0_u)
          - Returns rays_B_u, rays_C_u
        m_eff: effektivt antal avståndssteg (0 = använd alla).
        World sampling (new API):
          - world.sample(x,y) -> (B_kg, C_kg)
          - world.sample_many(xs,ys) -> (B_kg_array, C_kg_array)
        """
        # ---- (A) Sample local physics ----
        B0_kg, C0_kg = world.sample(x, y)

        Pworld = getattr(world, "WP", None)
        Kb = float(getattr(Pworld, "B_sense_K", 0.0)) if Pworld is not None else 0.0
        Kc = float(getattr(Pworld, "C_sense_K", 0.0)) if Pworld is not None else 0.0

        B0_u = self._sat1_u(float(B0_kg), Kb)
        C0_u = self._sat1_u(float(C0_kg), Kc)

        n = int(self._n)
        m_full = int(self._m)
        if n <= 0 or m_full <= 0:
            return (float(B0_u), float(C0_u)), self._accB[:0], self._accC[:0]

        # ---- (B) Ray geometry — alltid full räckvidd (m_full) ----
        np.add(self._ang_base, np.float32(heading), out=self._ang)
        np.cos(self._ang, out=self._dx)
        np.sin(self._ang, out=self._dy)

        xx = np.float32(x)
        yy = np.float32(y)

        np.multiply(self._dx[:, None], self._d[None, :], out=self._xs)
        self._xs += xx
        np.multiply(self._dy[:, None], self._d[None, :], out=self._ys)
        self._ys += yy

        ws = np.float32(self.world_size)
        np.mod(self._xs, ws, out=self._xs)
        np.mod(self._ys, ws, out=self._ys)

        # ---- (C) Sample world fields ----
        Bkg, Ckg = world.sample_many(self._xs, self._ys, outB=self._Bp, outC=self._Cp)

        # ---- (D) Konvertera kg→u, maskera bortom per-stråle djup, integrera ----
        self._sat_u(Bkg, Kb)
        self._sat_u(Ckg, Kc)

        # Använd m_eff som globalt tak om angivet, annars _ray_m per stråle
        if m_eff > 0 and m_eff < m_full:
            ray_depths = np.minimum(self._ray_m, m_eff)
        else:
            ray_depths = self._ray_m

        # Mask: True där avståndssteg j < stråle i's djup
        j_idx  = np.arange(m_full, dtype=np.int32)[None, :]   # (1, m)
        dmask  = j_idx < ray_depths[:, None]                   # (n, m)

        # Viktad summa per stråle med per-stråle viktsum
        w2d    = np.where(dmask, self._w[None, :], np.float32(0.0))
        wsum_r = w2d.sum(axis=1, keepdims=True).clip(min=1e-9)
        self._accB[:] = (Bkg * w2d).sum(axis=1) / wsum_r.squeeze()
        self._accC[:] = (Ckg * w2d).sum(axis=1) / wsum_r.squeeze()

        # ---- (E) Noise in u-domain
        sig = float(self.AP.noise_sigma)
        if sig > 0.0 and (rng is not None):
            rng.standard_normal(size=n, out=self._noise64)
            self._noiseB[:] = (self._noise64 * sig).astype(np.float32, copy=False)

            rng.standard_normal(size=n, out=self._noise64)
            self._noiseC[:] = (self._noise64 * sig).astype(np.float32, copy=False)

            self._accB += self._noiseB
            self._accC += self._noiseC

            B0_u = float(B0_u + rng.normal(0.0, sig * 0.5))
            C0_u = float(C0_u + rng.normal(0.0, sig * 0.5))

        # ---- (F) Clamp outputs
        np.clip(self._accB, 0.0, 1.0, out=self._accB)
        np.clip(self._accC, 0.0, 1.0, out=self._accC)
        B0_u = clamp(B0_u, 0.0, 1.0)
        C0_u = clamp(C0_u, 0.0, 1.0)

        return (float(B0_u), float(C0_u)), self._accB, self._accC

    def see_agent_first_hit(
        self,
        world: World,
        x: float,
        y: float,
        heading: float,
        self_id: int,
        m_eff: int = 0,
    ) -> tuple[float, float, float, int, int]:
        """
        Returns (present, bearing_u, dist_u, j_hit, hit_agent_id).
        j_hit: avståndssteg-index för träffen (-1 om ingen träff).
        hit_agent_id: agent-ID vid träffpunkten (0 om ingen träff).
        m_eff: effektivt antal avståndssteg att skanna (0 = alla).
        """
        n = int(self._n)
        m_full = int(self._m)
        if n <= 0 or m_full <= 0:
            return 0.0, 0.0, 0.0, -1, 0

        np.add(self._ang_base, np.float32(heading), out=self._ang)
        np.cos(self._ang, out=self._dx)
        np.sin(self._ang, out=self._dy)

        xx = np.float32(x)
        yy = np.float32(y)

        np.multiply(self._dx[:, None], self._d[None, :], out=self._xs)
        self._xs += xx
        np.multiply(self._dy[:, None], self._d[None, :], out=self._ys)
        self._ys += yy

        ws = np.float32(self.world_size)
        s  = int(self.world_size)
        np.mod(self._xs, ws, out=self._xs)
        np.mod(self._ys, ws, out=self._ys)

        np.floor(self._xs, out=self._xs)
        np.floor(self._ys, out=self._ys)
        np.mod(self._xs, s, out=self._xs)
        np.mod(self._ys, s, out=self._ys)
        np.copyto(self._ixs, self._xs, casting="unsafe")
        np.copyto(self._iys, self._ys, casting="unsafe")

        aids = world.A[self._iys, self._ixs]
        mask = (aids != 0) & (aids != int(self_id))

        # Maskera bortom per-stråle djup (ellipsmodell)
        if m_eff > 0 and m_eff < m_full:
            ray_depths = np.minimum(self._ray_m, m_eff)
        else:
            ray_depths = self._ray_m
        j_idx    = np.arange(m_full, dtype=np.int32)[None, :]
        depth_ok = j_idx < ray_depths[:, None]
        mask     = mask & depth_ok

        hit_per_j = mask.any(axis=0)
        if not hit_per_j.any():
            return 0.0, 0.0, 0.0, -1, 0

        j_hit        = int(np.argmax(hit_per_j))
        i_hit        = int(np.argmax(mask[:, j_hit]))
        bearing_u    = float(i_hit) / float(n)
        dist_u       = float(self._d[j_hit]) / max(float(self.AP.ray_len_front), 1e-9)
        hit_agent_id = int(aids[i_hit, j_hit])
        return 1.0, bearing_u, dist_u, j_hit, hit_agent_id


# -------------------------
# Agent
# -------------------------
@dataclass
class Agent:
    """
    NEP-agent:
      - policy output -> motorik + ätande + kroppsdynamik
      - Phenotype härleds från traits och är konstant över livstid
    """

    AP: AgentParams
    genome: MLPGenome

    x: float
    y: float
    heading: float

    id: int = field(default_factory=_new_agent_id)

    OBS_DIM: ClassVar[int] = 23   # +2: predator_bearing (cos/sin), predator_dist
    OUT_DIM: ClassVar[int] = 5
    
    body: Body = field(init=False)
    sensors: RaySensors = field(init=False)

    obs_trace: np.ndarray = field(init=False)

    birth_t: float = 0.0
    pheno: Phenotype = field(init=False)

    last_speed: float = 0.0
    age_s: float = 0.0
    repro_cd_s: float = 0.0

    last_B0: float = 0.0
    last_C0: float = 0.0

    WF = 0.6
    WS = 0.4

    sense_level: int = field(init=False, default=0)

    # Adaptiv sensing-cache: lagrar senaste sensing-resultat och cooldown-räknare.
    _sense_cd: int = field(init=False, default=0)        # steg kvar tills nästa skanning
    _cached_B0: float = field(init=False, default=0.0)
    _cached_C0: float = field(init=False, default=0.0)
    _cached_x_in: np.ndarray = field(init=False)         # cachat obs-vektor
    _last_detect_j: int = field(init=False, default=-1)  # avståndssteg för senaste träff (-1=ingen)
    _sense_m_eff: int = field(init=False, default=0)     # effektivt stråldjup nästa skanning
    _cached_agent_hit: tuple = field(init=False)         # (N, Nu, Nd, hit_id) från senaste see_agent_first_hit
    _cached_predator_hit: tuple = field(init=False)      # (pred_bearing, pred_dist) — närmaste hotande predator
    _desired_mate_id: int = field(init=False, default=0) # ID för lokalt detekterad potentiell partner (0=ingen)

    def __post_init__(self) -> None:
        self.AP = replace(self.AP)

        self.body = Body(self.AP)
        self.obs_trace = np.zeros((8,), dtype=np.float32)

        self.age_s = 0.0
        self.repro_cd_s = 0.0
        self.birth_t = float(getattr(self, "birth_t", 0.0))

        self.apply_traits()

        self.sense_level = _sense_level(float(self.pheno.sense_strength))
        _apply_sense_to_AP(self.AP, self.sense_level)

        # --- Rekurrent minnestillstånd ---
        # h bärs av agenten mellan stegen; nollställs vid födseln.
        # Nätverkets input = concat(obs, h), output = concat(y, h_raw).
        _h_dim = max(0, int(getattr(self.genome, "h_dim", 0)))
        self._h: np.ndarray = np.zeros((_h_dim,), dtype=np.float32)

        # --- compute_cost skalas med nätverksstorlek ---
        # Agenter med bredare/djupare nätverk betalar mer metabolt per sekund.
        _n_params = int(self.genome.n_params())
        _ref_params = max(1, int(self.AP.compute_cost_ref_params))
        if _n_params > 0 and _ref_params > 0:
            self.AP.compute_cost = float(self.AP.compute_cost) * (_n_params / _ref_params)

        self._init_body_state_from_AP()

        # Sense-cache: börja med noll-vektor; triggar full skanning vid första steget.
        self._sense_cd = 0
        self._cached_B0 = 0.0
        self._cached_C0 = 0.0
        self._cached_x_in = np.zeros((self.OBS_DIM + _h_dim,), dtype=np.float32)
        self._last_detect_j = -1
        self._sense_m_eff = 0
        self._cached_agent_hit = (0.0, 0.0, 0.0, 0)
        self._cached_predator_hit = (0.0, 0.0)
        self._desired_mate_id = 0

    def _init_body_state_from_AP(self) -> None:
        self.body.M = max(0.0, float(self.AP.M0))

        E0 = max(0.0, float(self.AP.E0))
        self.body.E_fast = (0.85 * E0) / 0.6
        self.body.E_slow = (0.15 * E0) / 0.4

        self.body.Tb = float(getattr(self.AP, "Tb_init", 37.0))
        
        self.body.D = 0.0
        self.body.Fg = float(self.body.Fg)
        self.body.alive = True

    def bind_world(self, world: World) -> None:
        self.world = world
        size = int(world.WP.size)
        if getattr(self, "sensors", None) is None or getattr(self.sensors, "world_size", None) != size:
            self.sensors = RaySensors(self.AP, world_size=size)

    @staticmethod
    def _signed_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi


    def _torus_delta_to(self, other, size: float) -> tuple[float, float, float]:
        dx = float(other.x) - float(self.x)
        dy = float(other.y) - float(self.y)
        half = 0.5 * float(size)
        if dx > half:
            dx -= float(size)
        elif dx < -half:
            dx += float(size)
        if dy > half:
            dy -= float(size)
        elif dy < -half:
            dy += float(size)
        d2 = dx * dx + dy * dy
        return dx, dy, math.sqrt(d2) if d2 > 0.0 else 0.0

    def attack_value(self, target, dist: float) -> float:
        tb = getattr(target, 'body', None)
        if tb is None or not bool(getattr(tb, 'alive', False)):
            return -1.0e9
        tm = max(0.0, float(getattr(tb, 'M', 0.0)))
        te = max(0.0, float(tb.E_total())) if hasattr(tb, 'E_total') else 0.0
        my_ecap = max(1e-9, float(self.body.E_cap()))
        dist_term = max(0.0, 1.0 - float(dist) / max(1e-9, float(self.AP.attack_range)))
        return 0.35 * tm + 0.65 * (te / my_ecap) + 0.40 * dist_term

    def attack_risk(self, target, dist: float) -> float:
        tb = getattr(target, 'body', None)
        tp = float(getattr(getattr(target, 'pheno', None), 'predation', 0.0))
        if tb is None or not bool(getattr(tb, 'alive', False)):
            return 1.0e9
        tm = max(1e-9, float(getattr(tb, 'M', 0.0)))
        my_m = max(1e-9, float(self.body.M))
        d_norm = clamp(float(getattr(tb, 'D', 0.0)) / max(1e-9, float(self.AP.D_max)), 0.0, 1.0)
        healthy = 1.0 - d_norm
        dist_term = max(0.0, 1.0 - float(dist) / max(1e-9, float(self.AP.attack_range)))
        return 0.45 * (tm / my_m) + 0.55 * tp + 0.35 * healthy * dist_term

    def attack_score(self, target, dist: float) -> float:
        return self.attack_value(target, dist) - self.attack_risk(target, dist)

    def apply_traits(self) -> None:
        self.pheno = derive_pheno(self.genome.traits)

    def phenotype_summary(self) -> dict:
        return phenotype_summary(self.pheno)

    def _build_obs(self, B0: float, C0: float, rays_B, rays_C,
                   pred_bearing: float = 0.0, pred_dist: float = 0.0) -> np.ndarray:
        rb = np.asarray(rays_B, dtype=np.float32)
        rc = np.asarray(rays_C, dtype=np.float32)

        n = int(rb.shape[0])
        if n <= 0:
            meanB = meanC = maxB = maxC = 0.0
            aB = aC = 0.0
        else:
            iB = int(np.argmax(rb))
            iC = int(np.argmax(rc))
            aB = 2.0 * math.pi * (iB / n)
            aC = 2.0 * math.pi * (iC / n)
            meanB = float(rb.mean())
            meanC = float(rc.mean())
            maxB = float(rb[iB])
            maxC = float(rc[iC])

        hunger = float(self.body.hunger())
        fatigue = float(self.body.Fg)
        D = float(self.body.D)

        obs = np.array(
            [float(B0), float(C0), meanB, meanC, maxB, maxC, hunger, fatigue],
            dtype=np.float32,
        )

        a = 0.06
        self.obs_trace = (1.0 - a) * self.obs_trace + a * obs

        x = np.concatenate(
            [
                obs,
                self.obs_trace,
                np.array([math.cos(aB), math.sin(aB), math.cos(aC), math.sin(aC), D,
                          float(pred_bearing), float(pred_dist)], dtype=np.float32),
            ]
        )
        return x

    def build_inputs(self, world: World, rng: np.random.Generator):
        if not self.body.alive:
            return None, 0.0, 0.0

        m_full = int(self.sensors._m) if self.sensors is not None else 0

        # Initialisera m_eff vid första anropet
        if self._sense_m_eff <= 0:
            self._sense_m_eff = m_full

        # --- Adaptiv sensing: returnera cache om cooldown aktiv ---
        if self._sense_cd > 0:
            self._sense_cd -= 1

            # Body-state uppdateras varje steg även vid cache
            hunger = float(self.body.hunger())
            fatigue = float(self.body.Fg)
            D = float(self.body.D)
            self._cached_x_in[6] = hunger
            self._cached_x_in[7] = fatigue
            self._cached_x_in[20] = D

            a = 0.06
            self.obs_trace = (1.0 - a) * self.obs_trace + a * self._cached_x_in[:8]
            self._cached_x_in[8:16] = self.obs_trace

            # h uppdateras alltid — cachat obs + färskt h → korrekt nätverksinput
            _h_dim = int(self._h.shape[0])
            if _h_dim > 0:
                self._cached_x_in[self.OBS_DIM:] = self._h

            return self._cached_x_in.copy(), self._cached_B0, self._cached_C0

        # --- Full sensing med adaptivt djup ---
        m_eff = self._sense_m_eff

        # Parningsläge: kör alltid full sensing för att inte missa en potentiell partner
        if self.ready_to_reproduce():
            m_eff = m_full
            self._sense_cd = 0

        (B0, C0), rays_B, rays_C = self.sensors.sense(
            world, self.x, self.y, self.heading, rng=rng, m_eff=m_eff,
        )

        # Kontrollera om något detekterades (mat i rays eller lokalt)
        thresh = float(self.AP.sense_alert_thresh)
        food_near = (
            float(B0) > thresh
            or float(C0) > thresh
            or (len(rays_B) > 0 and float(rays_B.max()) > thresh)
            or (len(rays_C) > 0 and float(rays_C.max()) > thresh)
        )

        # Kontrollera grannar med adaptivt djup
        N_ag, Nu_ag, Nd_ag, j_agent, hit_id = self.sensors.see_agent_first_hit(
            world, self.x, self.y, self.heading, self.id, m_eff=m_eff,
        )
        agent_near = j_agent >= 0

        # Cacha för apply_outputs — inkluderar hit_id för lokal agent-lookup
        self._cached_agent_hit = (N_ag, Nu_ag, Nd_ag, hit_id)

        # Predator-proxy: närmaste agent-riktning ger nätverket flykt-signal
        pred_bearing = float(Nu_ag) * 2.0 * math.pi if N_ag > 0.5 else 0.0
        pred_dist    = float(Nd_ag) if N_ag > 0.5 else 1.0

        x_in = self._build_obs(B0, C0, rays_B, rays_C,
                               pred_bearing=pred_bearing, pred_dist=pred_dist)

        if food_near or agent_near:
            # ALERT: hög frekvens, tight djup (lite förbi senaste träffen)
            j_det = j_agent if agent_near else 0
            self._last_detect_j = j_det
            # Nästa scan: bara så långt som behövs + 2 steg marginal, max m_full
            self._sense_m_eff = min(m_full, max(2, j_det + 3))
            self._sense_cd = max(0, int(self.AP.sense_alert_steps) - 1)
        else:
            # IDLE: låg frekvens, full räckvidd (fångar nya saker i periferin)
            self._last_detect_j = -1
            self._sense_m_eff = m_full
            self._sense_cd = max(0, int(self.AP.sense_idle_steps) - 1)

        # Konkatenera rekurrent h till observationsvektorn
        _h_dim = int(self._h.shape[0])
        if _h_dim > 0:
            x_full = np.empty(self.OBS_DIM + _h_dim, dtype=np.float32)
            x_full[:self.OBS_DIM] = x_in
            x_full[self.OBS_DIM:] = self._h
            x_in = x_full

        # Uppdatera cache
        self._cached_B0 = float(B0)
        self._cached_C0 = float(C0)
        self._cached_x_in[:] = x_in

        return x_in, float(B0), float(C0)

    def _torus_delta_to(self, other: "Agent", size: float) -> tuple[float, float, float]:
        dx = float(other.x) - float(self.x)
        dy = float(other.y) - float(self.y)
        half = 0.5 * float(size)
        if dx > half:
            dx -= float(size)
        elif dx < -half:
            dx += float(size)
        if dy > half:
            dy -= float(size)
        elif dy < -half:
            dy += float(size)
        d2 = dx * dx + dy * dy
        return dx, dy, math.sqrt(d2) if d2 > 0.0 else 0.0

    def attack_value(self, target: "Agent", dist: float) -> float:
        if not target.body.alive:
            return -1e9
        d_norm = 1.0 - clamp(dist / max(float(self.AP.prey_search_radius), 1e-9), 0.0, 1.0)
        m_term = clamp(float(target.body.M) / max(float(self.body.M), 1e-9), 0.0, 2.0)
        e_term = clamp(float(target.body.reserve_frac()), 0.0, 1.0)
        weak_term = 1.0 - clamp(float(target.body.D) / max(float(target.body.AP.D_max), 1e-9), 0.0, 1.0)
        weak_term = 1.0 - weak_term  # low D => low prey value from weakness, high D => high value
        return 0.55 * d_norm + 0.25 * m_term + 0.10 * e_term + 0.10 * weak_term

    def attack_risk(self, target: "Agent", dist: float) -> float:
        d_norm = 1.0 - clamp(dist / max(float(self.AP.prey_search_radius), 1e-9), 0.0, 1.0)
        rel_mass = clamp(float(target.body.M) / max(float(self.body.M), 1e-9), 0.0, 3.0)
        target_pred = clamp(float(getattr(target.pheno, "predation", 0.0)), 0.0, 1.0)
        target_def = 1.0 - clamp(float(target.body.D) / max(float(target.body.AP.D_max), 1e-9), 0.0, 1.0)
        return 0.35 * rel_mass + 0.40 * target_pred + 0.20 * target_def + 0.05 * d_norm

    def attack_score(self, target: "Agent", dist: float) -> float:
        return float(self.attack_value(target, dist) - self.attack_risk(target, dist))

    # _local_social_targets() avvecklad — social interaktion är nu perceptionsbunden via hit_id.

    def apply_outputs(
        self,
        world: World,
        ctx: StepCtx,
        y: np.ndarray,
        B0: float,
        C0: float,
    ) -> Tuple[float, float]:
        if not self.body.alive:
            return 0.0, 0.0
    
        dt = float(ctx.dt)
    
        # --- bookkeeping (agent-level clocks) ---
        self.age_s += dt
        self.repro_cd_s = max(0.0, float(self.repro_cd_s) - dt)

        # ---------------------------------------------------------
        # 0) Splitta policy-output i aktioner (y) och nytt minnestillstånd (h)
        # Nätverket producerar concat(y, h_raw); h_new = tanh(h_raw) ∈ (−1,+1).
        # ---------------------------------------------------------
        _h_dim = int(self._h.shape[0])
        _out_dim = int(self.OUT_DIM)
        if _h_dim > 0:
            h_raw = y[_out_dim : _out_dim + _h_dim]
            self._h = np.tanh(h_raw).astype(np.float32)
            y = y[:_out_dim]          # rena aktioner, identiskt med tidigare API

        # ---------------------------------------------------------
        # 1) Decode policy outputs
        # ---------------------------------------------------------
        turn = float(np.tanh(y[0]))  # [-1,1]
        thrust = float(1.0 / (1.0 + np.exp(-float(y[1]))))         # [0,1]
        inh_move = float(1.0 / (1.0 + np.exp(-float(y[2]))))       # [0,1]
        inh_eat = float(1.0 / (1.0 + np.exp(-float(y[3]))))        # [0,1]
        explore_drive = float(1.0 / (1.0 + np.exp(-float(y[4]))))  # [0,1]
    
        allow_move = 1.0 - inh_move
        allow_eat = 1.0 - inh_eat
    
        # ---------------------------------------------------------
        # 2) Temperature + local target assessment
        # ---------------------------------------------------------
        Tloc = float(world.temperature_at(self.x, self.y)) if hasattr(world, "temperature_at") else 0.0

        soc = float(getattr(self.pheno, "sociability", 0.0))
        pred = float(getattr(self.pheno, "predation", 0.0))
        N, Nu, Nd, hit_id = self._cached_agent_hit
        in_mating_mode = self.ready_to_reproduce()

        # Lokal agent-utvärdering: bara den agent som faktiskt detekterades via sensing.
        best_prey        = None
        best_prey_score  = -1e9
        best_threat      = None
        best_threat_score = -1e9
        best_mate        = None
        hunt_state       = 0.0
        flee_state       = 0.0
        self._desired_mate_id = 0

        # hunt_eff: jaktförmåga skalas med dietanpassning.
        # En herbivore (diet≈0) kan inte jaga lönsamt oavsett predation-trait.
        _hunt_diet_exp = float(getattr(self.AP, 'hunt_diet_exp', 1.5))
        _diet_val      = float(getattr(self.pheno, 'diet', 0.5))
        hunt_eff       = pred * (_diet_val ** _hunt_diet_exp)

        if N > 0.5 and hit_id > 0:
            detected = world._agent_by_id.get(int(hit_id))
            if detected is not None and detected.body.alive and detected is not self:
                dx, dy, dist = self._torus_delta_to(detected, float(world.WP.size))
                other_pred = float(getattr(detected.pheno, "predation", 0.0))
                other_diet = float(getattr(detected.pheno, "diet", 0.5))
                other_hunt_eff = other_pred * (other_diet ** _hunt_diet_exp)
                if other_hunt_eff >= float(self.AP.threat_predation_min):
                    sc_th = detected.attack_score(self, dist)
                    if sc_th > float(self.AP.hunt_score_min):
                        best_threat = (detected, dx, dy, dist)
                        best_threat_score = sc_th
                if hunt_eff >= float(self.AP.predator_trait_min):
                    sc = self.attack_score(detected, dist)
                    if sc > float(self.AP.hunt_score_min):
                        best_prey = (detected, dx, dy, dist)
                        best_prey_score = sc
                if in_mating_mode and detected.ready_to_reproduce():
                    best_mate = (detected, dx, dy, dist)
                    self._desired_mate_id = int(hit_id)


        # ---------------------------------------------------------
        # 3) Reflexiva drivkrafter: flykt, jakt, parning, socialt, föda
        # PRIORITET: flee > hunt > mating > social > food > explore
        # ---------------------------------------------------------
        if best_threat is not None and hunt_eff < float(self.AP.threat_predation_min) and best_threat_score > float(self.AP.hunt_score_min):
            other, dx, dy, dist = best_threat
            a_th = math.atan2(dy, dx)
            err = self._signed_angle(a_th - self.heading)
            bias = clamp(err / math.pi, -1.0, 1.0)
            turn = clamp(turn - 0.95 * bias, -1.0, 1.0)
            thrust = clamp(max(thrust, 0.95), 0.0, 1.0)
            explore_drive *= 0.10
            flee_state = 1.0

        elif best_prey is not None and hunt_eff >= float(self.AP.predator_trait_min) and best_prey_score > float(self.AP.hunt_score_min):
            other, dx, dy, dist = best_prey
            a_hit = math.atan2(dy, dx)
            errN = self._signed_angle(a_hit - self.heading)
            biasN = clamp(errN / math.pi, -1.0, 1.0)
            hs = clamp(hunt_eff, 0.0, 1.0)
            turn = clamp(turn + 0.90 * hs * biasN, -1.0, 1.0)
            thrust = clamp(max(thrust, 0.85), 0.0, 1.0)
            explore_drive *= 0.25
            hunt_state = 1.0

        elif in_mating_mode and best_mate is not None:
            other, dx, dy, dist = best_mate
            a_hit = math.atan2(dy, dx)
            errN = self._signed_angle(a_hit - self.heading)
            biasN = clamp(errN / math.pi, -1.0, 1.0)
            turn = clamp(0.95 * biasN, -1.0, 1.0)
            thrust = max(thrust, 0.95)
            explore_drive = 0.0

        elif N > 0.5:
            a_hit = self.heading + (2.0 * math.pi * float(Nu))
            errN = self._signed_angle(a_hit - self.heading)
            biasN = clamp(errN / math.pi, -1.0, 1.0)
            Nd_f = float(Nd)
            REP_ZONE = 0.35
            if Nd_f < REP_ZONE:
                rs = 1.0 - (Nd_f / REP_ZONE)
                turn = clamp(turn - 0.70 * rs * biasN, -1.0, 1.0)
                explore_drive = explore_drive * (1.0 - 0.3 * rs)
            else:
                soc_bias = 2.0 * soc - 1.0
                if abs(soc_bias) > 1e-6:
                    wdist = clamp(1.0 - Nd_f, 0.0, 1.0)
                    turn = clamp(turn + 0.70 * soc_bias * wdist * biasN, -1.0, 1.0)
                    explore_drive = explore_drive * (1.0 - 0.60 * abs(soc_bias) * wdist)

        hunger_now = float(self.body.hunger())
        if flee_state < 0.5 and hunger_now > 0.4:
            sensors = getattr(self, "sensors", None)
            if sensors is not None:
                accB = getattr(sensors, "_accB", None)
                accC = getattr(sensors, "_accC", None)
                ang = getattr(sensors, "_ang_base", None)
                if accB is not None and ang is not None and len(accB) > 0:
                    _diet    = float(getattr(self.pheno, "diet", 0.5))
                    herb_eff = (1.0 - _diet) ** 0.7
                    scav_eff = _diet ** 0.7
                    # combo viktas med faktiskt energiutbyte per diettyp —
                    # en scavenger reagerar starkare på carcass-signal än en generalist.
                    combo = accB * herb_eff + accC * scav_eff
                    i_best = int(np.argmax(combo))
                    sig = float(combo[i_best])
                    if sig > 0.05:
                        food_angle = float(ang[i_best]) + float(self.heading)
                        err_food = self._signed_angle(food_angle - self.heading)
                        bias_food = clamp(err_food / math.pi, -1.0, 1.0)
                        fd = clamp(hunger_now - 0.4, 0.0, 0.6) * sig
                        turn = clamp(turn + 0.60 * fd * bias_food, -1.0, 1.0)
                        thrust = clamp(thrust + 0.3 * fd, 0.0, 1.0)

        # ---------------------------------------------------------
        # Lokal mat dämpar utforskning: hungrig agent stannar vid mat.
        # food_local: dietviktat lokalt matutbud direkt under agenten (B0/C0).
        # hunger_now=1, food_local=1 → explore_drive → 0  (svältande, stå kvar)
        # hunger_now=1, food_local=0 → ingen dämpning     (svältande, ingen mat här)
        # hunger_now=0, food_local=1 → ingen dämpning     (mätt, utforska fritt)
        _diet_local  = float(getattr(self.pheno, "diet", 0.5))
        _herb_local  = (1.0 - _diet_local) ** 0.7
        _scav_local  = _diet_local ** 0.7
        food_local   = clamp(float(B0) * _herb_local + float(C0) * _scav_local, 0.0, 1.0)
        explore_drive *= 1.0 - hunger_now * food_local


        # ---------------------------------------------------------
        # 4) Heading integration (turn + exploration jitter)
        # ---------------------------------------------------------
        jitter = float(ctx.rng.normal(0.0, 0.65)) * explore_drive
        self.heading = float(self.heading) + dt * float(self.AP.turn_rate) * (
            0.85 * allow_move * turn + 0.25 * jitter
        )
        self.heading = self._signed_angle(self.heading)

        # ---------------------------------------------------------
        # 5) Locomotion control u in [0,1]
        # ---------------------------------------------------------
        fatigue = float(self.body.Fg)
        fatigue_factor = clamp(1.0 - 0.9 * fatigue, 0.05, 1.0)
        weak_move = float(self.body.move_factor())
        u = clamp(allow_move * thrust * fatigue_factor * weak_move, 0.0, 1.0)

        # ---------------------------------------------------------
        # 6) Locomotion dynamics
        # ---------------------------------------------------------
        v_prev = max(0.0, float(self.last_speed))
        M_pre = max(1e-9, float(self.body.M))

        F0_cap = float(self.AP.F0)
        alpha = float(self.AP.force_mass_exp)
        F_prop = u * F0_cap * (M_pre ** alpha)

        c1 = float(self.AP.drag_lin)
        c2 = float(self.AP.drag_quad)

        F_drag_prev = c1 * v_prev + c2 * v_prev * v_prev
        a = (F_prop - F_drag_prev) / M_pre
        v_euler = max(0.0, v_prev + dt * a)

        speed = min(v_euler, float(self.AP.v_max))
        v_mid = 0.5 * (v_prev + speed)
        self.last_speed = float(speed)

        # ---------------------------------------------------------
        # 7) Locomotion energy (J)
        # ---------------------------------------------------------
        eta = clamp(float(self.AP.locomotion_eff), 1e-6, 1.0)
        P_mech = max(0.0, F_prop * v_mid)
        E_move = (dt * P_mech) / eta

        # ---------------------------------------------------------
        # 8) Apply translation (torus)
        # ---------------------------------------------------------
        self.x = torus_wrap(float(self.x) + dt * speed * math.cos(self.heading), world.WP.size)
        self.y = torus_wrap(float(self.y) + dt * speed * math.sin(self.heading), world.WP.size)

        # ---------------------------------------------------------
        # 9) Feeding (kg)
        # ---------------------------------------------------------
        got_bio = 0.0
        got_carcass = 0.0
        if allow_eat > 0.20:
            want_kg = float(self.AP.eat_rate) * dt * (0.25 + 0.75 * float(self.body.hunger()))
            got_total, got_carcass = world.consume_food(self.x, self.y, amount=want_kg, prefer_carcass=True)
            got_bio = max(0.0, float(got_total) - float(got_carcass))

        food_bio_kg = float(got_bio)
        food_carcass_kg = float(got_carcass)

        # ---------------------------------------------------------
        # 10) Activity proxy
        # ---------------------------------------------------------
        speed_n = clamp(speed / max(float(self.AP.v_max), 1e-9), 0.0, 1.0)
        ate = 1.0 if (allow_eat > 0.20 and (food_bio_kg + food_carcass_kg) > 0.0) else 0.0
        activity = 0.03 + 0.45 * speed_n + 0.10 * ate
        # ---------------------------------------------------------
        # 11) Body dynamics (includes gestation build model: "Väg 2")
        # ---------------------------------------------------------
        self.body.step(
            ctx,
            speed=speed,
            activity=activity,
            food_bio_kg=food_bio_kg,
            food_carcass_kg=food_carcass_kg,
            pheno=self.pheno,
            extra_drain=E_move,
            T_env=Tloc,
            age_s=self.age_s,
        )
    
        # ---------------------------------------------------------
        # 12) Tracking
        # ---------------------------------------------------------
        self.last_B0 = float(B0)
        self.last_C0 = float(C0)
    
        return float(B0), float(C0)

    # --- reproduction hooks (Population uses these) ---
    
    def ready_to_reproduce(self) -> bool:
        if not self.body.alive:
            return False
    
        # already pregnant => don't start a new one
        if bool(getattr(self.body, "gestating", False)):
            return False
    
        # cooldown gate  (FIX: repro_cd_s is on Agent, not Body)
        if float(getattr(self, "repro_cd_s", 0.0)) > 0.0:
            return False
    
        # maturity gate
        if float(self.age_s) < float(self.pheno.A_mature):
            return False
    
        # hard resource gates (parent must have buffer above minimum)
        M = float(self.body.M)
        Mreq = max(float(self.AP.M_min), float(self.pheno.M_repro_min))
        if M < Mreq:
            return False
    
        Et = float(self.body.E_total())
        Ecap = float(self.body.E_cap())
        efrac = Et / max(Ecap, 1e-12)
        if efrac < float(self.pheno.E_repro_min):
            return False
    
        return True
    
    def wants_to_reproduce(self, rng: np.random.Generator) -> bool:
        # Hard gates (alive, cooldown, maturity, M, energy fraction)
        if not self.ready_to_reproduce():
            return False
    
        # Pain / regulation gates — lösgjorda från föregående körning:
        # hunger > 0.7 blockerade 83% av populationen (kronisk svält, mean hunger=0.84)
        # D > 0.25 blockerade ytterligare 32% — för aggressivt givet att D stiger normalt
        if float(self.body.hunger()) > 0.85:
            return False
        if float(self.body.D) > 0.50:
            return False
        if float(self.body.Fg) > 0.85:
            return False
    
        # Stochastic trigger
        dt = float(self.AP.dt)
        lam = float(self.pheno.repro_rate)
        p = 1.0 - math.exp(-lam * dt)
        return bool(rng.random() < p)
    
    def start_gestation(self) -> bool:
        # child mass target from phenotype (absolute units)
        M_target = float(getattr(self.pheno, "child_M", 0.0))
        return bool(self.body.start_gestation(M_target))
    
    def pay_repro_cost(self, cost_E_J: float) -> float:
        """
        Dra energi från parent. Returnerar faktiskt betald energi (J).
        OBS: Den här är bara en transfer; pain/damage ska uppstå via
        din ordinarie energi-/stresslogik i steget.
        """
        want_J = max(0.0, float(cost_E_J))
        paid_J = float(self.body.take_energy(want_J))
        return paid_J
    
    
    def init_newborn_state(
        self,
        parent_pheno: Phenotype,
        child_M_from_parent: float | None = None,
        child_E_fast_J: float | None = None,
        child_E_slow_J: float | None = None,
    ) -> None:
        """
        Initiera nyfödd deterministiskt:
          - Massan: från parent om provisionerad, annars fallback till pheno/AP.
          - Energi: får bara komma från parent (J in), aldrig från Ecap.
          - Klipp mot Ecap.
          - Reset interna tillstånd.
        """
    
        # ---- Mass ----
        if child_M_from_parent is not None:
            child_M = float(child_M_from_parent)
        else:
            child_M = float(getattr(parent_pheno, "child_M", float(self.AP.M0) * 0.5))
    
        child_M = max(float(self.AP.M_min), child_M)
        self.body.M = float(child_M)
    
        # ---- Energy (J -> internal units via WF/WS) ----
        Ef_J = max(0.0, float(child_E_fast_J)) if child_E_fast_J is not None else 0.0
        Es_J = max(0.0, float(child_E_slow_J)) if child_E_slow_J is not None else 0.0
    
        self.body.E_fast = Ef_J / float(self.WF)
        self.body.E_slow = Es_J / float(self.WS)
    
        # ---- Clip to Ecap deterministiskt (weighted space) ----
        Et = float(self.body.E_total())
        Ecap = float(self.body.E_cap())
        overflow = Et - Ecap
        if overflow > 0.0:
            # take from fast first in weighted units
            fast_w = float(self.WF) * float(self.body.E_fast)
            take_fast_w = min(fast_w, overflow)
            if take_fast_w > 0.0:
                self.body.E_fast = float(self.body.E_fast) - (take_fast_w / float(self.WF))
                overflow -= take_fast_w
    
            if overflow > 0.0:
                slow_w = float(self.WS) * float(self.body.E_slow)
                take_slow_w = min(slow_w, overflow)
                if take_slow_w > 0.0:
                    self.body.E_slow = float(self.body.E_slow) - (take_slow_w / float(self.WS))
                    overflow -= take_slow_w
    
        # ---- Other body fields ----
        self.body.Fg = clamp(float(getattr(parent_pheno, "child_Fg", 0.15)), 0.0, 1.0)
        self.body.Tb = float(getattr(self.AP, "Tb_init", 37.0))
    
        # ---- Reset internal accumulators/state ----
        self.body.D = 0.0
        self.body.W = 0.0
        self.body.P = 0.0
        self.body._D_prev = 0.0
    
        self.body.alive = True

        # Nollställ rekurrent minnestillstånd — nyfödd börjar utan minne
        self._h.fill(0.0)

        # newborn: start with cooldown so they can't instantly reproduce
        self.repro_cd_s = float(self.AP.repro_cooldown_s)
        self.age_s = 0.0