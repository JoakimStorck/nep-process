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
# -------------------------
_NEXT_AGENT_ID = 0


def _new_agent_id() -> int:
    global _NEXT_AGENT_ID
    _NEXT_AGENT_ID += 1
    return _NEXT_AGENT_ID


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
        AP.ray_len = 7.0
        AP.noise_sigma = 0.06
    elif level == 1:
        AP.n_rays = 16
        AP.ray_len = 8.0
        AP.noise_sigma = 0.055
    elif level == 2:
        AP.n_rays = 24
        AP.ray_len = 10.0
        AP.noise_sigma = 0.050
    else:  # level 3
        AP.n_rays = 32
        AP.ray_len = 12.0
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
    ray_len: float = 7.0
    ray_step: float = 1.0
    noise_sigma: float = 0.06

    # Sensing upkeep costs (per mass, per second via metabolism_scale)
    sense_cost_L1: float = 0.2
    sense_cost_L2: float = 0.5
    sense_cost_L3: float = 1.0

    # ------------------------
    # Feeding: world pool units -> internal energy units
    # ------------------------
    eat_rate: float = 1e-4  # kg/s

    # Energy densities (J/kg)
    E_bio_J_per_kg: float = 4.0e6
    E_carcass_J_per_kg: float = 7.0e6
    E_body_J_per_kg: float = 7.0e6

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
    compute_cost: float = 5.0  # [W/kg]

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
    M_crit: float = 0.30
    M_min: float = 0.10
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

    # Aging background damage (optional; defaults -> off)
    k_age0: float = 0.0       # NEW: base background rate term
    k_age1: float = 0.0       # NEW: linear-in-age coefficient
    k_ageD: float = 0.4       # NEW: amplification by normalized damage

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
    repair_E_per_D: float = 0.02
    repair_W_decay: float = 0.3
    repair_eta0: float = 1.0
    repair_eta_W: float = 0.2

    # Legacy / currently unused (kan tas bort när du städat klart)

    # ------------------------
    # Wear / aging accumulation (W dynamics)
    # ------------------------
    wear_a0: float = 0.0
    wear_aE: float = 0.0
    wear_aR: float = 0.0
    wear_aD: float = 0.0

    # ------------------------
    # Reproduction (Population)
    # ------------------------
    repro_cooldown_s: float = 8.0
    M_repro_soft: float = 0.03
    E_repro_soft: float = 0.05

    birth_E0: float = 0.0
    birth_k_E_per_M: float = 7.0e4
    birth_energy_eff: float = 0.70
    gestation_growth_kg_per_s: float = 0.002  # exempel: 2 g/s

    # ------------------------
    # Stochastic death knobs
    # ------------------------
    median_age_s: float = 50.0
    death_h_base: float = 0.0  # if 0 -> computed from median_age_s in Body.step
    death_h_age: float = 0.0
    death_h_D: float = 0.0

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
        return 0.6 * float(self.E_fast) + 0.4 * float(self.E_slow)

    def E_cap(self) -> float:
        return float(self.AP.E_cap_per_M) * max(1e-9, float(self.M))

    def hunger(self) -> float:
        Et = float(self.E_total())
        Ecap = float(self.E_cap())
        return clamp((Ecap - Et) / max(Ecap, 1e-9), 0.0, 1.0)

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
            return float(getattr(self.AP, "sense_cost_L1", 0.0))
        if level == 2:
            return float(getattr(self.AP, "sense_cost_L2", 0.0))
        if level >= 3:
            return float(getattr(self.AP, "sense_cost_L3", 0.0))
        return 0.0

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
        # (0) Numerical guards (pre)
        # ---------------------------------------------------------
        if not (
            self._finite(self.E_fast)
            and self._finite(self.E_slow)
            and self._finite(self.M)
            and self._finite(self.D)
            and self._finite(self.Fg)
        ):
            self.guard_steps += 1
            self.guard_killed += 1
            self.guard_last = self._guard_snapshot("pre_state")
            self.alive = False
            return
    
        if not (
            self._finite(speed)
            and self._finite(activity)
            and self._finite(food_bio_kg)
            and self._finite(food_carcass_kg)
            and self._finite(extra_drain)
            and self._finite(age_s)
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
        # (1) Intake -> buffers (up to cap), surplus -> mass
        # ---------------------------------------------------------
        m_bio = max(0.0, float(food_bio_kg))
        m_car = max(0.0, float(food_carcass_kg))
    
        E_in = (
            m_bio * float(self.AP.E_bio_J_per_kg)
            + m_car * float(self.AP.E_carcass_J_per_kg)
        )
    
        Et0 = float(self.E_total())
        Ecap0 = float(self.E_cap())
        room = max(0.0, Ecap0 - Et0)
    
        dE_store = min(E_in, room)
    
        if dE_store > 0.0:
            # store split in weighted J-space
            dE_fast_w = 0.85 * dE_store
            dE_slow_w = 0.15 * dE_store
            self.E_fast += dE_fast_w / WF
            self.E_slow += dE_slow_w / WS
    
        E_to_M = max(0.0, E_in - dE_store)
        dM_grow = 0.0
        if E_to_M > 0.0:
            E_body = max(1e-12, float(self.AP.E_body_J_per_kg))
            dM_grow = E_to_M / E_body
            self.M = float(self.M) + dM_grow
    
        # ---------------------------------------------------------
        # (2) Effective mass (carried load) + basal/compute/sense/loco
        # ---------------------------------------------------------
        burden = max(0.0, float(getattr(self.AP, "gestation_mass_burden", 0.0)))
        M_carry = float(self.M)
        if bool(self.gestating):
            M_carry += burden * max(0.0, float(self.gest_M))
    
        M_eff = max(1e-9, M_carry)
        metab = float(getattr(pheno, "metabolism_scale", 1.0))
    
        out_basal = dt * metab * (M_eff ** 0.75) * float(self.AP.k_basal)
        out_compute = dt * metab * M_eff * float(self.AP.compute_cost) * float(activity)
    
        sense_cost = float(self._sense_cost(pheno))
        out_sense = dt * metab * M_eff * sense_cost
    
        out_loco = max(0.0, float(extra_drain))
    
        # ---------------------------------------------------------
        # (2B) Thermoregulation (ledger-consistent)
        # ---------------------------------------------------------
        Tb = float(getattr(self, "Tb", float(getattr(self.AP, "Tb_set", 37.0))))
        Tset = float(getattr(self.AP, "Tb_set", 37.0))
        Tenv = float(T_env)
    
        k0 = float(getattr(self.AP, "thermo_k_W_per_C", 0.0))
        expA = float(getattr(self.AP, "thermo_mass_exp", 2.0 / 3.0))
        K = k0 * (M_eff ** expA)
    
        c0 = float(getattr(self.AP, "heatcap_J_per_kgC", 0.0))
        Cth = max(1e-9, c0 * M_eff)
    
        P_need = 0.0
        if Tb < Tset:
            P_need = max(0.0, K * (Tset - Tenv))
    
        Pmax = float(getattr(self.AP, "thermo_Pmax_per_kg", 0.0)) * M_eff
        P_gen = min(P_need, Pmax)
    
        out_thermo = dt * P_gen  # J
    
        Qloss = K * (Tb - Tenv)
        dTb = (P_gen - Qloss) * dt / Cth
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

        # i Body.step() tidigt
        if bool(getattr(self, "gestating", False)) and not hasattr(self, "_dbg_gest_started"):
            self._dbg_gest_started = True
            print(f"[DBG] gestating became true: agent_id={int(getattr(ctx,'agent_id',-1))} t={float(ctx.t):.2f}")
            
        if bool(self.gestating):
            # overhead power drain (optional)
            Pg_over = max(0.0, float(getattr(self.AP, "gestation_P_overhead_per_kg", 0.0))) * M_eff
            out_gest_overhead = dt * Pg_over
    
            # build toward target
            M_tgt = max(0.0, float(self.gest_M_target))
            M_cur = max(0.0, float(self.gest_M))
    
            if M_tgt > 0.0 and M_cur < M_tgt:
                rate = max(0.0, float(getattr(self.AP, "gestation_growth_kg_per_s", 0.0)))
                dM_want = min(rate * dt, M_tgt - M_cur)
    
                if dM_want > 0.0:
                    E_body = max(1e-12, float(self.AP.E_body_J_per_kg))
                    E_need = dM_want * E_body
    
                    # pay from buffers
                    paid1 = float(self.take_energy(E_need))
                    deficit_g = max(0.0, E_need - paid1)
    
                    paid2 = 0.0
                    if deficit_g > 0.0:
                        # catabolize above M_min to fund remaining gestation need
                        Mmin = float(self.AP.M_min)
                        free = max(0.0, float(self.M) - Mmin)
    
                        want_cat = deficit_g / E_body
                        dM_cat_gest = min(want_cat, free)
    
                        if dM_cat_gest > 0.0:
                            self.M -= dM_cat_gest
                            E_from_M_gest = dM_cat_gest * E_body
                            # inject to fast buffer in same weighted convention, then immediately pay
                            self.E_fast += E_from_M_gest / WF
    
                        paid2 = float(self.take_energy(deficit_g))
    
                    paid_total = paid1 + paid2
                    out_gest_build = paid_total
    
                    dM_gest = paid_total / E_body
                    if dM_gest > 0.0:
                        self.gest_M = M_cur + dM_gest
                        self.gest_E_J = float(self.gest_E_J) + paid_total
    
        # ---------------------------------------------------------
        # (2D) Pay drains ONCE (this is the critical fix)
        # ---------------------------------------------------------
        E_out_drain = (
            out_basal + out_compute + out_sense + out_loco + out_thermo
            + out_gest_overhead + out_gest_build
        )
    
        paid = float(self.take_energy(E_out_drain))
        deficit = max(0.0, E_out_drain - paid)
        E_paid_drain = paid
    
        # ---------------------------------------------------------
        # (3) Catabolism: cover remaining deficit (if any), above M_min
        # ---------------------------------------------------------
        E_from_M = 0.0
        dM_cat = 0.0
        paid2 = 0.0
    
        if deficit > 0.0:
            E_body = max(1e-12, float(self.AP.E_body_J_per_kg))
            want_cat = deficit / E_body
    
            Mmin = float(self.AP.M_min)
            free = max(0.0, float(self.M) - Mmin)
            dM_cat = min(want_cat, free)
    
            if dM_cat > 0.0:
                self.M -= dM_cat
                E_from_M = dM_cat * E_body
                self.E_fast += E_from_M / WF
    
            paid2 = float(self.take_energy(deficit))
            deficit = max(0.0, deficit - paid2)
            E_paid_drain += paid2
    
        # snapshot after drains/catabolism (for stress math)
        Et = float(self.E_total())
        Ecap = float(self.E_cap())
    
        # ---------------------------------------------------------
        # (4) Damage + repair + fatigue
        # ---------------------------------------------------------
        D_before = float(self.D)
        D_max = float(self.AP.D_max)
    
        # (we no longer use dD_rep precomputed; repair handled in step_pain_and_repair)
        E_out_repair = 0.0
    
        e_lack = clamp((Ecap - Et) / max(Ecap, 1e-9), 0.0, 1.0)
        d_norm = clamp(D_before / max(D_max, 1e-9), 0.0, 1.0)
    
        speed_n = clamp(float(speed) / max(float(self.AP.v_max), 1e-9), 0.0, 1.0)
        effort = speed_n + 0.6 * float(activity)
        rest = max(0.0, 1.0 - speed_n) * max(0.0, 1.0 - float(activity))
    
        w = float(self.weakness())
        starve_stress = 1.0 + float(self.AP.starve_stress_gain) * (1.0 - w)
    
        susc = float(getattr(pheno, "susceptibility", 1.0))
        frailty_gain = float(getattr(pheno, "frailty_gain", 0.0))
        frailty_gain_cap = float(getattr(self.AP, "frailty_gain_cap", 1.0))
        frailty_gain = clamp(frailty_gain, 0.0, max(0.0, frailty_gain_cap))
        frail = 1.0 + frailty_gain * d_norm
    
        # normalized drain rate (for stress_per_drain)
        drain_rate = E_out_drain / max(dt, 1e-12)
        drain_rate_n = drain_rate / max(Ecap, 1e-9)  # 1/s
    
        k_E = 1.2
        k_eff = 0.02
        dD_eff = dt * (k_eff * susc * (1.0 + k_E * e_lack) * frail * effort * starve_stress)
        dD_met = dt * (float(getattr(pheno, "stress_per_drain", 0.0)) * drain_rate_n * starve_stress)
    
        # aging background (optional)
        k_age0 = float(getattr(self.AP, "k_age0", 0.0))
        k_age1 = float(getattr(self.AP, "k_age1", 0.0))
        k_ageD = float(getattr(self.AP, "k_ageD", 0.4))
        age_rate = max(0.0, k_age0 + k_age1 * float(age_s))
        dD_age = dt * age_rate * (1.0 + k_ageD * d_norm) * (1.0 + frailty_gain)
    
        # cold stress
        Tb_now = float(getattr(self, "Tb", float(getattr(self.AP, "Tb_set", 37.0))))
        Tb_min = float(getattr(self.AP, "Tb_min", 35.0))
        cold_damage_gain = float(getattr(self.AP, "cold_damage_gain", 0.0))
        if Tb_now < Tb_min:
            sev = clamp((Tb_min - Tb_now) / 10.0, 0.0, 1.0)
            dD_cold = dt * cold_damage_gain * sev
        else:
            dD_cold = 0.0
    
        dD_in = dD_eff + dD_met + dD_age + dD_cold
        dD_pos_rate = dD_in / max(dt, 1e-9)
    
        self.D = clamp(D_before + dD_in, 0.0, D_max)
    
        # repair/pain spend (separate)
        E_pain_repair = float(self.step_pain_and_repair(ctx, pheno, D_before=D_before))
        E_out_repair = E_pain_repair
    
        # aging / wear accumulation
        E_spent_total = float(E_paid_drain) + float(E_out_repair)
        self.step_aging(
            ctx,
            E_spent_total=E_spent_total,
            repro_cost_paid=0.0,
            dD_pos=float(dD_pos_rate),
        )
    
        # fatigue update
        d_norm2 = clamp(float(self.D) / max(D_max, 1e-9), 0.0, 1.0)
        fatigue_effort_eff = float(self.AP.fatigue_effort) * (1.0 + 0.4 * d_norm2)
        fatigue_recover_eff = float(self.AP.fatigue_recover) * max(0.0, (1.0 - 0.05 * d_norm2))
    
        self.Fg = clamp(
            float(self.Fg) + dt * (fatigue_effort_eff * effort - fatigue_recover_eff * rest),
            0.0,
            1.0,
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
        if float(self.D) >= D_max or float(self.M) <= float(self.AP.M_min):
            self.alive = False
            return
    
        # ---------------------------------------------------------
        # (6) Stochastic death (optional)
        # ---------------------------------------------------------
        if rng is not None:
            target_med = max(1e-6, float(getattr(self.AP, "median_age_s", 50.0)))
            h_base_cfg = float(getattr(self.AP, "death_h_base", 0.0))
            h_base = h_base_cfg if h_base_cfg > 0.0 else (math.log(2.0) / target_med)
    
            h_age = float(getattr(self.AP, "death_h_age", 0.0))
            h_D = float(getattr(self.AP, "death_h_D", 0.0))
    
            hazard_rate = max(0.0, h_base + h_age * float(age_s) + h_D * d_norm2)
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
        self.D = clamp(D0, 0.0, D_max)
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
            self._finite(self.E_fast)
            and self._finite(self.E_slow)
            and self._finite(self.M)
            and self._finite(self.D)
            and self._finite(self.Fg)
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
    
        expected_E_after = E_before + dE_store + E_from_M_total - E_out_drain - E_out_repair
    
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
            "E_store": dE_store,
            "E_to_M": E_to_M,
    
            "E_out_basal": out_basal,
            "E_out_compute": out_compute,
            "E_out_sense": out_sense,
            "E_out_loco": out_loco,
            "E_out_thermo": out_thermo,
            "E_out_gest_overhead": out_gest_overhead,
            "E_out_gest_build": out_gest_build,
            "E_out_drain": E_out_drain,
    
            "E_out_repair": E_out_repair,
            "E_from_M": E_from_M_total,
            "deficit": deficit,
            "E_after": E_after,
    
            "M_before": M_before,
            "dM_grow": dM_grow,
            "dM_cat": dM_cat_total,
            "M_after": M_after,
    
            # gestation state
            "gestating": bool(self.gestating),
            "gest_M": float(self.gest_M),
            "gest_M_target": float(self.gest_M_target),
            "gest_E_J": float(self.gest_E_J),
            "dM_gest": dM_gest,
            "dM_cat_gest": dM_cat_gest,
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
        n = int(self.AP.n_rays)
        step = float(self.AP.ray_step)
        ray_len = float(self.AP.ray_len)

        self._n = max(0, n)

        def z1(dtype=np.float32):
            return np.zeros((0,), dtype=dtype)

        def z2(dtype=np.float32):
            return np.zeros((0, 0), dtype=dtype)

        if self._n <= 0 or step <= 0.0 or ray_len <= 0.0:
            self._m = 0
            self._ang_base = z1()
            self._ang = z1()
            self._d = z1()
            self._w = z1()
            self._wsum = np.float32(1.0)
            self._inv_wsum = np.float32(1.0)
            self._dx = z1()
            self._dy = z1()
            self._xs = z2()
            self._ys = z2()
            self._Bp = z2()
            self._Cp = z2()
            self._accB = z1()
            self._accC = z1()
            self._noiseB = z1()
            self._noiseC = z1()
            self._noise64 = z1(dtype=np.float64)
            return

        self._ang_base = (
            np.float32(2.0 * np.pi)
            * (np.arange(self._n, dtype=np.float32) / np.float32(self._n))
        )
        self._ang = np.empty((self._n,), dtype=np.float32)

        self._d = np.arange(step, ray_len + 1e-6, step, dtype=np.float32)
        self._m = int(self._d.size)

        self._w = (np.float32(1.0) / (np.float32(1.0) + np.float32(0.25) * self._d)).astype(
            np.float32, copy=False
        )
        self._wsum = np.sum(self._w, dtype=np.float32) + np.float32(1e-9)
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

    def sense(
        self,
        world: World,
        x: float,
        y: float,
        heading: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
        """
        Perception API (u-domain):
          - Returns (B0_u, C0_u)
          - Returns rays_B_u, rays_C_u
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
        m = int(self._m)
        if n <= 0 or m <= 0:
            return (float(B0_u), float(C0_u)), self._accB[:0], self._accC[:0]

        # ---- (B) Ray geometry ----
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

        # ---- (C) Sample world fields along rays (kg) ----
        Bkg, Ckg = world.sample_many(self._xs, self._ys, outB=self._Bp, outC=self._Cp)

        # ---- (D) Convert kg -> u in-place, integrate in u-domain
        self._sat_u(Bkg, Kb)
        self._sat_u(Ckg, Kc)

        np.matmul(Bkg, self._w, out=self._accB)
        np.matmul(Ckg, self._w, out=self._accC)

        self._accB *= self._inv_wsum
        self._accC *= self._inv_wsum

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
    ) -> tuple[float, float, float]:
        """
        Returns (present, bearing_u, dist_u)
        First-hit semantics: scan distance outward; first encountered agent stops scan.
        """
        n = int(self._n)
        m = int(self._m)
        if n <= 0 or m <= 0:
            return 0.0, 0.0, 0.0

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

        A = world.A
        s = int(self.world_size)

        for j in range(m):
            for i in range(n):
                ix = int(self._xs[i, j]) % s
                iy = int(self._ys[i, j]) % s
                aid = int(A[iy, ix])
                if aid != 0 and aid != int(self_id):
                    bearing_u = float(i) / float(n)
                    dist_u = float(self._d[j] / max(float(self.AP.ray_len), 1e-9))
                    return 1.0, bearing_u, dist_u

        return 0.0, 0.0, 0.0


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

    OBS_DIM: ClassVar[int] = 21
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

    def __post_init__(self) -> None:
        self.AP = replace(self.AP)

        self.body = Body(self.AP)
        # OBS: dimension updated (see _build_obs)
        self.obs_trace = np.zeros((8,), dtype=np.float32)

        self.age_s = 0.0
        self.repro_cd_s = 0.0
        self.birth_t = float(getattr(self, "birth_t", 0.0))

        self.apply_traits()

        self.sense_level = _sense_level(float(self.pheno.sense_strength))
        _apply_sense_to_AP(self.AP, self.sense_level)

        self._init_body_state_from_AP()

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

    def apply_traits(self) -> None:
        self.pheno = derive_pheno(self.genome.traits)

    def phenotype_summary(self) -> dict:
        return phenotype_summary(self.pheno)

    def _build_obs(self, B0: float, C0: float, rays_B, rays_C) -> np.ndarray:
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
                np.array([math.cos(aB), math.sin(aB), math.cos(aC), math.sin(aC), D], dtype=np.float32),
            ]
        )
        return x

    def build_inputs(self, world: World, rng: np.random.Generator):
        if not self.body.alive:
            return None, 0.0, 0.0

        (B0, C0), rays_B, rays_C = self.sensors.sense(world, self.x, self.y, self.heading, rng=rng)
        x_in = self._build_obs(B0, C0, rays_B, rays_C)
        return x_in, float(B0), float(C0)

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
        # 2) Temperature at location (NO policy modulation)
        # ---------------------------------------------------------
        Tloc = float(world.temperature_at(self.x, self.y)) if hasattr(world, "temperature_at") else 0.0
    
        # ---------------------------------------------------------
        # 3) Heading integration (turn + exploration jitter)
        # ---------------------------------------------------------
        jitter = float(ctx.rng.normal(0.0, 0.65)) * explore_drive
        self.heading = float(self.heading) + dt * float(self.AP.turn_rate) * (
            0.85 * allow_move * turn + 0.25 * jitter
        )
        self.heading = self._signed_angle(self.heading)
    
        # ---------------------------------------------------------
        # 4) Locomotion control u in [0,1]
        # ---------------------------------------------------------
        fatigue = float(self.body.Fg)
        fatigue_factor = clamp(1.0 - 0.9 * fatigue, 0.05, 1.0)
        weak_move = float(self.body.move_factor())
        u = clamp(allow_move * thrust * fatigue_factor * weak_move, 0.0, 1.0)
    
        # ---------------------------------------------------------
        # 5) Locomotion dynamics
        # ---------------------------------------------------------
        v_prev = max(0.0, float(self.last_speed))
        M_pre = max(1e-9, float(self.body.M))
    
        F0_cap = float(getattr(self.AP, "F0", 4.0))
        alpha = float(getattr(self.AP, "force_mass_exp", 2.0 / 3.0))
        F_prop = u * F0_cap * (M_pre ** alpha)
    
        c1 = float(getattr(self.AP, "drag_lin", 0.8))
        c2 = float(getattr(self.AP, "drag_quad", 0.2))
    
        F_drag_prev = c1 * v_prev + c2 * v_prev * v_prev
        a = (F_prop - F_drag_prev) / M_pre
        v_euler = max(0.0, v_prev + dt * a)
    
        speed = min(v_euler, float(self.AP.v_max))
        v_mid = 0.5 * (v_prev + speed)
        self.last_speed = float(speed)
    
        # ---------------------------------------------------------
        # 6) Locomotion energy (J)
        # ---------------------------------------------------------
        eta = clamp(float(getattr(self.AP, "locomotion_eff", 0.25)), 1e-6, 1.0)
        P_mech = max(0.0, F_prop * v_mid)
        E_move = (dt * P_mech) / eta
    
        # ---------------------------------------------------------
        # 7) Apply translation (torus)
        # ---------------------------------------------------------
        self.x = torus_wrap(float(self.x) + dt * speed * math.cos(self.heading), world.WP.size)
        self.y = torus_wrap(float(self.y) + dt * speed * math.sin(self.heading), world.WP.size)
    
        # ---------------------------------------------------------
        # 8) Feeding (kg)
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
        # 9) Activity proxy
        # ---------------------------------------------------------
        speed_n = clamp(speed / max(float(self.AP.v_max), 1e-9), 0.0, 1.0)
        ate = 1.0 if (allow_eat > 0.20 and (food_bio_kg + food_carcass_kg) > 0.0) else 0.0
        activity = 0.03 + 0.45 * speed_n + 0.10 * ate
    
        # ---------------------------------------------------------
        # 10) Social attraction (unchanged semantics)
        # ---------------------------------------------------------
        soc = float(getattr(self.pheno, "sociability", 0.0))
        SOC_TURN_GAIN = 0.55
        SOC_DIST_GAIN = 1.00
        SOC_JITTER_DAMP = 0.60
    
        N, Nu, Nd = self.sensors.see_agent_first_hit(world, self.x, self.y, self.heading, self.id)
        if N > 0.5 and soc > 1e-6:
            a_hit = self.heading + (2.0 * math.pi * float(Nu))
            errN = self._signed_angle(a_hit - self.heading)
            biasN = clamp(errN / math.pi, -1.0, 1.0)
            wdist = clamp(1.0 - SOC_DIST_GAIN * float(Nd), 0.0, 1.0)
            turn = clamp(turn + SOC_TURN_GAIN * soc * wdist * biasN, -1.0, 1.0)
            explore_drive = explore_drive * (1.0 - SOC_JITTER_DAMP * soc * wdist)
    
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
    
        # Pain / regulation gates
        if float(self.body.hunger()) > 0.7:
            return False
        if float(self.body.D) > 0.25:
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
    
    
    def provide_child_mass(self, child_M: float) -> float:
        """
        Flytta massa från parent till barn utan att gå under M_min.
        Returnerar faktiskt given massa.
        """
        want = max(0.0, float(child_M))
        M = float(self.body.M)
        Mmin = float(self.AP.M_min)
    
        free = max(0.0, M - Mmin)
        got = min(want, free)
    
        if got > 0.0:
            self.body.M = M - got
    
        return float(got)
    
    
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
    
        # newborn: start with cooldown so they can't instantly reproduce
        self.repro_cd_s = float(self.AP.repro_cooldown_s)
        self.age_s = 0.0
        