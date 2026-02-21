# agent.py
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Tuple

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


# -------------------------
# Params
# -------------------------
from dataclasses import dataclass

@dataclass
class AgentParams:
    # ------------------------
    # Time discretization
    # ------------------------
    dt: float = 0.02

    # ------------------------
    # Sensing / perception
    # ------------------------
    n_rays: int = 12
    ray_len: float = 7.0
    ray_step: float = 1.0
    noise_sigma: float = 0.06

    # ------------------------
    # Steering / policy kinematics
    # (v_max is a policy cap; actual speed is set by the locomotion model)
    # ------------------------
    v_max: float = 2.2
    turn_rate: float = 2.2

    # ------------------------
    # Feeding: world pool units -> internal energy units
    # ------------------------
    # Feeding (kg/s)
    eat_rate: float = 1e-4  # kg/s
    
    # Energy densities (J/kg)
    E_bio_J_per_kg: float = 4.0e6
    E_carcass_J_per_kg: float = 7.0e6
    E_body_J_per_kg: float = 7.0e6   # kroppsvävnad (våt, ”biologisk”) ~ 5–7 MJ/kg. Välj och håll konsekvent.
    
    # Energy storage capacity (J/kg)
    E_cap_per_M: float = 3.0e5        # J/kg (EXEMPEL)
    
    # Initial energy (J)
    E0: float = 1.5e5                 # EXEMPEL: 100 kJ
    
    # ------------------------
    # Initial physiological state
    # ------------------------
    M0: float = 1.0     # initial body mass

    # ------------------------
    # Basal metabolism (allometric)
    # ------------------------
    k_basal: float = 30.0 # [W]

    # ------------------------
    # Activity-related metabolic costs (non-locomotor)
    # ------------------------
    compute_cost: float = 5.0 # [W/kg]

    # ------------------------
    # Sensing upkeep costs (per mass, per second via metabolism_scale)
    # ------------------------
    sense_cost_L1: float = 0.2
    sense_cost_L2: float = 0.5
    sense_cost_L3: float = 1.0


    # ------------------------
    # Locomotion mechanics (single source of truth for movement energetics)
    #
    # Interpretation:
    #   thrust_u in [0,1] from policy gives propulsive force:
    #       F_prop = thrust_u * (F0 * M^alpha)
    #   Resistive force (direction-opposed):
    #       F_res = drag_lin*v + drag_quad*v^2
    #   Net mechanical power demand (only if moving / accelerating against resistance):
    #       P_mech = max(0, F_prop - F_res) * v
    #   Metabolic power:
    #       P_met = P_mech / locomotion_eff
    #
    # Choose alpha=2/3 for geometric similarity (cross-section ∝ M^(2/3)).
    # ------------------------
    F0: float = 4.0              # propulsive force coefficient (scales force capacity)
    force_mass_exp: float = 2/3  # alpha in F_prop = thrust_u * F0 * M^alpha
    drag_lin: float = 0.8        # c1 (viscous-like)
    drag_quad: float = 0.2       # c2 (inertial / turbulent-like)
    locomotion_eff: float = 0.25 # eta in (0,1]

    # ------------------------
    # Starvation and weakness dynamics
    # ------------------------
    M_crit: float = 0.30
    M_min: float = 0.10
    v_weak_min: float = 0.25
    rep_weak_min: float = 0.20
    starve_stress_gain: float = 1.0

    # ------------------------
    # Damage and fatigue dynamics
    # ------------------------
    hazard_to_damage: float = 0.003
    hazard_to_fatigue: float = 0.08
    fatigue_recover: float = 0.020
    fatigue_effort: float = 0.050
    D_max: float = 1.0

    # ------------------------
    # Repair energy cost
    # ------------------------
    k_rep: float = 0.45

    # ------------------------
    # Reproduction (handled by Population)
    # ------------------------
    repro_cooldown_s: float = 8.0


# -------------------------
# Body: energy + mass (unbounded) + damage/fatigue (bounded)
# -------------------------
@dataclass
class Body:
    P: AgentParams

    # energy buffers (weighted in E_total)
    E_fast: float = 0.0
    E_slow: float = 0.0

    # structural state
    M: float = 0.0        # body mass
    D: float = 0.0        # accumulated damage
    Fg: float = 0.15      # fatigue level

    # life state
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
    
    def E_total(self) -> float:
        return 0.6 * float(self.E_fast) + 0.4 * float(self.E_slow)

    def E_cap(self) -> float:
        return float(self.P.E_cap_per_M) * max(1e-9, float(self.M))

    def hunger(self) -> float:
        # hunger = relativ energibrist jämfört med E_cap(M)
        Et = float(self.E_total())
        Ecap = float(self.E_cap())
        return clamp((Ecap - Et) / max(Ecap, 1e-9), 0.0, 1.0)

    def weakness(self) -> float:
        """
        Weakness factor w in [0,1]:
          w=1 => full function
          w->0 as M approaches M_crit from below
        """
        m = float(self.M)
        mcrit = float(self.P.M_crit)
        if m >= mcrit:
            return 1.0
        if mcrit <= 1e-9:
            return 0.0
        return clamp(m / mcrit, 0.0, 1.0)

    def move_factor(self) -> float:
        w = float(self.weakness())
        return float(self.P.v_weak_min + (1.0 - float(self.P.v_weak_min)) * w)

    def repair_factor(self) -> float:
        w = float(self.weakness())
        return float(self.P.rep_weak_min + (1.0 - float(self.P.rep_weak_min)) * w)

    def _finite(self, x: float) -> bool:
        return math.isfinite(float(x))

    def _guard_snapshot(self, where: str) -> dict:
        # minimal snapshot: state only (no huge dicts)
        return {
            "where": where,
            "E_fast": float(self.E_fast),
            "E_slow": float(self.E_slow),
            "M": float(self.M),
            "D": float(self.D),
            "Fg": float(self.Fg),
            "alive": bool(self.alive),
        }
        
    def take_energy(self, amount: float) -> float:
        """
        Remove 'amount' from the body energy stores (unbounded),
        split proportionally. Returns actual removed (in E-units).
        """
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

    def _sense_cost(self, pheno: Phenotype) -> float:
        level = _sense_level(float(getattr(pheno, "sense_strength", 0.0)))
        if level == 1:
            return float(getattr(self.P, "sense_cost_L1", 0.0))
        if level == 2:
            return float(getattr(self.P, "sense_cost_L2", 0.0))
        if level >= 3:
            return float(getattr(self.P, "sense_cost_L3", 0.0))
        return 0.0
        
    def step(
        self,
        speed: float,
        activity: float,
        hazard: float,
        food_bio_kg: float,
        food_carcass_kg: float,
        pheno: Phenotype,
        extra_drain: float = 0.0,
    ) -> None:
        """
        Energy + mass (SI):
          - intake:   E_in [J] = m_bio*E_bio_J_per_kg + m_car*E_carcass_J_per_kg
          - store up to E_cap(M) [J]; surplus energy -> body mass via E_body_J_per_kg
          - drains [J] paid from E buffers; deficit covered by catabolizing mass via E_body_J_per_kg
          - death: D>=D_max or M<=M_min
    
        Damage/fatigue (bounded):
          - damage inflow from hazard, effort, and normalized metabolic drain
          - repair outflow gated by rest/energy/intake/freshness; paid in energy
        """
        if not self.alive:
            return
    
        dt = float(self.P.dt)
    
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
            and self._finite(hazard)
            and self._finite(food_bio_kg)
            and self._finite(food_carcass_kg)
            and self._finite(extra_drain)
        ):
            self.guard_steps += 1
            self.guard_killed += 1
            self.guard_last = {
                **self._guard_snapshot("pre_inputs"),
                "speed": float(speed),
                "activity": float(activity),
                "hazard": float(hazard),
                "food_bio_kg": float(food_bio_kg),
                "food_carcass_kg": float(food_carcass_kg),
                "extra_drain": float(extra_drain),
            }
            self.alive = False
            return
    
        # ---------------------------------------------------------
        # (0B) Ledger baselines
        # ---------------------------------------------------------
        E_before = float(self.E_total())
        M_before = float(self.M)
    
        # ---------------------------------------------------------
        # (1) Intake -> E buffers (up to cap), surplus -> mass
        # ---------------------------------------------------------
        DBG = False  # sätt False när du är klar

        # försök plocka upp "t" och "agent_id" om du skickar in dem i Body.step(...)
        t_now = locals().get("t", locals().get("t_now", None))
        a_id = locals().get("agent_id", locals().get("a_id", None))
        t_s = f"{float(t_now):.2f}" if t_now is not None else "?"
        id_s = f"{a_id}" if a_id is not None else "?"

        m_bio = max(0.0, float(food_bio_kg))
        m_car = max(0.0, float(food_carcass_kg))

        E_in = (
            m_bio * float(self.P.E_bio_J_per_kg)
            + m_car * float(self.P.E_carcass_J_per_kg)
        )

        Et0 = float(self.E_total())
        Ecap0 = float(self.E_cap())
        room = max(0.0, Ecap0 - Et0)

        if DBG:
            print(
                f"[MASSDBG][IN] t={t_s} id={id_s} "
                f"m_bio={m_bio:.6g} m_car={m_car:.6g} E_in={E_in:.6g} "
                f"Et0={Et0:.6g} Ecap0={Ecap0:.6g} room={room:.6g} "
                f"E_fast0={float(self.E_fast):.6g} E_slow0={float(self.E_slow):.6g} M0={float(self.M):.6g}",
                flush=True
            )

        dE_store = min(E_in, room)
        if dE_store > 0.0:
            dE_fast = 0.85 * dE_store
            dE_slow = 0.15 * dE_store
            self.E_fast += dE_fast
            self.E_slow += dE_slow

            if DBG:
                print(
                    f"[MASSDBG][STORE] t={t_s} id={id_s} "
                    f"dE_store={dE_store:.6g} -> dE_fast={dE_fast:.6g} dE_slow={dE_slow:.6g} "
                    f"E_fast={float(self.E_fast):.6g} E_slow={float(self.E_slow):.6g}",
                    flush=True
                )
        else:
            if DBG:
                print(
                    f"[MASSDBG][STORE] t={t_s} id={id_s} dE_store=0 (E_in={E_in:.6g}, room={room:.6g})",
                    flush=True
                )

        E_to_M = max(0.0, E_in - dE_store)
        if E_to_M > 0.0:
            E_body = max(1e-12, float(self.P.E_body_J_per_kg))
            dM_grow = E_to_M / E_body
            M_pre = float(self.M)
            self.M = float(self.M) + dM_grow

            if DBG:
                print(
                    f"[MASSDBG][GROW] t={t_s} id={id_s} "
                    f"E_to_M={E_to_M:.6g} E_body={E_body:.6g} dM_grow={dM_grow:.6g} "
                    f"M={M_pre:.6g}->{float(self.M):.6g}",
                    flush=True
                )
        else:
            dM_grow = 0.0
            if DBG:
                print(
                    f"[MASSDBG][GROW] t={t_s} id={id_s} E_to_M=0 (E_in={E_in:.6g}, dE_store={dE_store:.6g})",
                    flush=True
                )

        # ---------------------------------------------------------
        # (2) Drains (basal + compute + sense + locomotion(extra))
        # ---------------------------------------------------------
        M_eff = max(1e-9, float(self.M))
        metab = float(getattr(pheno, "metabolism_scale", 1.0))

        k_basal = float(self.P.k_basal)
        out_basal = dt * metab * (M_eff ** 0.75) * k_basal

        out_compute = dt * metab * M_eff * float(self.P.compute_cost) * float(activity)

        sense_cost = float(self._sense_cost(pheno))
        out_sense = dt * metab * M_eff * sense_cost

        out_loco = max(0.0, float(extra_drain))

        E_out_drain = out_basal + out_compute + out_sense + out_loco

        if DBG:
            print(
                f"[MASSDBG][DRAIN] t={t_s} id={id_s} "
                f"M_eff={M_eff:.6g} metab={metab:.6g} "
                f"out_basal={out_basal:.6g} out_compute={out_compute:.6g} "
                f"out_sense={out_sense:.6g} out_loco={out_loco:.6g} "
                f"E_out_drain={E_out_drain:.6g}",
                flush=True
            )

        paid = float(self.take_energy(E_out_drain))
        deficit = max(0.0, E_out_drain - paid)

        if DBG:
            Et_after_pay = float(self.E_total())
            print(
                f"[MASSDBG][PAY]  t={t_s} id={id_s} "
                f"paid={paid:.6g} deficit={deficit:.6g} "
                f"Et={Et0:.6g}->{Et_after_pay:.6g} "
                f"E_fast={float(self.E_fast):.6g} E_slow={float(self.E_slow):.6g} M={float(self.M):.6g}",
                flush=True
            )

        # ---------------------------------------------------------
        # (3) Catabolism: convert body mass -> energy to cover deficit
        # ---------------------------------------------------------
        E_from_M = 0.0
        dM_cat = 0.0

        if deficit > 0.0:
            E_body = max(1e-12, float(self.P.E_body_J_per_kg))
            want_cat = deficit / E_body

            if DBG:
                print(
                    f"[MASSDBG][CAT_PRE] t={t_s} id={id_s} "
                    f"deficit={deficit:.6g} E_body={E_body:.6g} want_cat={want_cat:.6g} M_avail={float(self.M):.6g}",
                    flush=True
                )

            dM_cat = min(want_cat, max(0.0, float(self.M)))
            M_pre = float(self.M)
            self.M = float(self.M) - dM_cat

            E_from_M = dM_cat * E_body

            if DBG:
                print(
                    f"[MASSDBG][CAT] t={t_s} id={id_s} "
                    f"dM_cat={dM_cat:.6g} M={M_pre:.6g}->{float(self.M):.6g} "
                    f"E_from_M={E_from_M:.6g}",
                    flush=True
                )

            if E_from_M > 0.0:
                Efast_pre = float(self.E_fast)
                self.E_fast = float(self.E_fast) + E_from_M
                if DBG:
                    print(
                        f"[MASSDBG][CAT_BUF] t={t_s} id={id_s} "
                        f"E_fast={Efast_pre:.6g}->{float(self.E_fast):.6g} (added {E_from_M:.6g})",
                        flush=True
                    )

                paid2 = float(self.take_energy(deficit))
                if DBG:
                    print(
                        f"[MASSDBG][CAT_PAY] t={t_s} id={id_s} "
                        f"paid2={paid2:.6g} (requested={deficit:.6g}) "
                        f"E_fast={float(self.E_fast):.6g} E_slow={float(self.E_slow):.6g} M={float(self.M):.6g}",
                        flush=True
                    )
        else:
            if DBG:
                print(f"[MASSDBG][CAT] t={t_s} id={id_s} deficit=0 -> no catabolism", flush=True)

        Et = float(self.E_total())
        Ecap = float(self.E_cap())

        if DBG:
            print(
                f"[MASSDBG][AFTER] t={t_s} id={id_s} "
                f"Et={Et:.6g} Ecap={Ecap:.6g} "
                f"E_fast={float(self.E_fast):.6g} E_slow={float(self.E_slow):.6g} "
                f"M={float(self.M):.6g} dM_grow={float(dM_grow):.6g} dM_cat={float(dM_cat):.6g}",
                flush=True
            )

        # refresh after energy dynamics
        Et = float(self.E_total())
        Ecap = float(self.E_cap())

        if DBG:
            t_now = locals().get("t_now", None)
            a_id  = locals().get("agent_id", None)

            t_s = f"{float(t_now):.2f}" if t_now is not None else "?"
            id_s = f"{a_id}" if a_id is not None else "?"

            print(
                f"[MASSDBG][AFTER] t={t_s} id={id_s} "
                f"Et={Et:.6g} Ecap={Ecap:.6g} "
                f"E_fast={float(self.E_fast):.6g} E_slow={float(self.E_slow):.6g} "
                f"M={float(self.M):.6g} dM_grow={float(dM_grow):.6g} dM_cat={float(dM_cat):.6g}",
                flush=True
            )

        # ---------------------------------------------------------
        # (4) Damage + repair + fatigue
        # ---------------------------------------------------------
        e_lack = clamp((Ecap - Et) / max(Ecap, 1e-9), 0.0, 1.0)
        d_norm = clamp(float(self.D) / max(float(self.P.D_max), 1e-9), 0.0, 1.0)

        speed_n = clamp(float(speed) / max(float(self.P.v_max), 1e-9), 0.0, 1.0)
        effort = speed_n + 0.6 * float(activity)
        rest = (
            max(0.0, 1.0 - speed_n)
            * max(0.0, 1.0 - float(activity))
            * (0.25 + 0.75 * max(0.0, 1.0 - float(hazard)))
        )

        # intake normalization (uses E_in); reference intake per tick, in J
        I_ref = max(1e-12, float(self.P.eat_rate) * dt * float(self.P.E_bio_J_per_kg))
        intake_n = clamp(E_in / I_ref, 0.0, 1.0)

        # weakness coupling
        w = float(self.weakness())
        starve_stress = 1.0 + float(self.P.starve_stress_gain) * (1.0 - w)

        susc = float(pheno.susceptibility)
        frail = 1.0 + float(pheno.frailty_gain) * d_norm
        k_E = 1.2

        # metabolic drain term MUST be normalized in J-world (else instant D blow-up)
        drain_rate = (out_basal + out_compute + out_sense + out_loco) / max(dt, 1e-12)  # J/s
        drain_rate_n = drain_rate / max(Ecap, 1e-9)                                     # 1/s

        dD_haz = dt * (
            float(self.P.hazard_to_damage)
            * susc
            * (1.0 + k_E * e_lack)
            * frail
            * float(hazard)
            * starve_stress
        )

        k_eff = 0.02
        dD_eff = dt * (k_eff * susc * (1.0 + k_E * e_lack) * frail * effort * starve_stress)

        dD_met = dt * (float(pheno.stress_per_drain) * drain_rate_n * starve_stress)

        dD_in = dD_haz + dD_eff + dD_met

        # --- repair outflow ---
        E_rep_min = float(pheno.E_rep_min)
        E_thr = E_rep_min * Ecap
        G_E = clamp((Et - E_thr) / max(Ecap - E_thr, 1e-9), 0.0, 1.0)

        G_rest = 0.2 + 0.8 * rest
        G_int = 0.3 + 0.7 * intake_n
        fresh = 1.0 - float(self.Fg)
        G_fresh = 0.75 + 0.25 * fresh

        frailty_damp = 1.0 / (1.0 + float(pheno.frailty_gain) * d_norm)
        rep_fac = float(self.repair_factor())

        dD_rep = dt * float(pheno.repair_capacity) * rep_fac * G_rest * G_E * G_int * G_fresh * frailty_damp

        # energy cost for repair (k_rep is J per unit damage repaired; keep as-is)
        E_out_repair = 0.0
        k_rep = float(getattr(self.P, "k_rep", 0.45))
        E_need = max(0.0, k_rep * dD_rep)
        if E_need > 0.0:
            E_paid = float(self.take_energy(E_need))
            E_out_repair = E_paid
            if E_paid < E_need:
                dD_rep *= (E_paid / max(E_need, 1e-12))

        self.D = clamp(float(self.D) + dD_in - dD_rep, 0.0, float(self.P.D_max))

        # fatigue
        fatigue_effort_eff = float(self.P.fatigue_effort) * (1.0 + 0.4 * d_norm)
        fatigue_recover_eff = float(self.P.fatigue_recover) * max(0.0, (1.0 - 0.05 * d_norm))

        self.Fg = clamp(
            float(self.Fg)
            + dt
            * (
                fatigue_effort_eff * effort
                + float(self.P.hazard_to_fatigue) * float(hazard)
                - fatigue_recover_eff * rest
            ),
            0.0,
            1.0,
        )

        # ---------------------------------------------------------
        # (4B) Numerical guard (post): clamps + finiteness
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

        # D bounded
        D0 = float(self.D)
        self.D = clamp(D0, 0.0, float(self.P.D_max))
        if float(self.D) != D0:
            clamped = True

        # Fg bounded
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
        # (5) Death conditions
        # ---------------------------------------------------------
        if float(self.D) >= float(self.P.D_max):
            self.alive = False
            return
        if float(self.M) <= float(self.P.M_min):
            self.alive = False
            return
    
        # ---------------------------------------------------------
        # (6) Ledger finalize (drift accounting)
        # ---------------------------------------------------------
        E_after = float(self.E_total())
        M_after = float(self.M)
    
        expected_E_after = (
            E_before
            + dE_store
            + E_from_M
            - E_out_drain
            - E_out_repair
        )
    
        drift = E_after - expected_E_after
        drift_abs = abs(drift)
    
        scale = max(
            1.0,
            abs(E_before),
            abs(E_after),
            abs(dE_store),
            abs(E_out_drain),
            abs(E_out_repair),
            abs(E_from_M),
        )
        drift_rel = drift / scale
        drift_rel_abs = abs(drift_rel)
    
        eps_abs = float(getattr(self.P, "ledger_eps_abs", 1e-8))
        eps_rel = float(getattr(self.P, "ledger_eps_rel", 1e-12))
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
            "E_out_drain": E_out_drain,
            "E_out_repair": E_out_repair,
            "E_from_M": E_from_M,
            "deficit": deficit,
            "E_after": E_after,
            "M_before": M_before,
            "dM_grow": dM_grow,
            "dM_cat": dM_cat,
            "M_after": M_after,
        }
    
        if drift_abs >= prev_max_abs:
            self.ledger_worst = dict(self.last_ledger)
    
        if bool(getattr(self.P, "assert_ledger", False)) and (not ok):
            raise AssertionError(
                f"Energy ledger drift: abs={drift_abs:.3e} rel={drift_rel:.3e} "
                f"(eps_abs={eps_abs:.3e}, eps_rel={eps_rel:.3e})"
            )


# -------------------------
# Ray sensors
# -------------------------
@dataclass
class RaySensors:
    P: "AgentParams"
    world_size: int

    # cached geometry/buffers
    _n: int = field(init=False, default=0)
    _m: int = field(init=False, default=0)

    _ang_base: np.ndarray = field(init=False)  # (n,) float32
    _ang: np.ndarray = field(init=False)       # (n,) float32
    _d: np.ndarray = field(init=False)         # (m,) float32
    _w: np.ndarray = field(init=False)         # (m,) float32
    _wsum: np.float32 = field(init=False)
    _inv_wsum: np.float32 = field(init=False)

    _dx: np.ndarray = field(init=False)
    _dy: np.ndarray = field(init=False)
    _xs: np.ndarray = field(init=False)
    _ys: np.ndarray = field(init=False)

    # sample buffers (B in kg initially; later overwritten to B_u for ray integration)
    _Bp: np.ndarray = field(init=False)
    _Fp: np.ndarray = field(init=False)

    # ray accumulators (u-domain)
    _accB: np.ndarray = field(init=False)
    _accF: np.ndarray = field(init=False)

    _noiseB: np.ndarray = field(init=False)
    _noiseF: np.ndarray = field(init=False)
    _noise64: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rebuild_cache()

    @staticmethod
    def _sat_u(x_kg: np.ndarray, K: float) -> np.ndarray:
        """
        Saturating mapping kg -> u in [0,1], vectorized.
        u = x/(x+K) with x clipped to >=0. If K<=0 => hard step at 0.
        """
        if K <= 0.0:
            return (x_kg > 0.0).astype(np.float32, copy=False)
        # x/(x+K) with x>=0
        np.maximum(x_kg, 0.0, out=x_kg)
        x_kg /= (x_kg + np.float32(K))
        return x_kg

    @staticmethod
    def _sat1_u(x_kg: float, K: float) -> float:
        """Scalar version of _sat_u."""
        x = 0.0 if x_kg < 0.0 else float(x_kg)
        if K <= 0.0:
            return 1.0 if x > 0.0 else 0.0
        return float(x / (x + K))

    def _rebuild_cache(self) -> None:
        n = int(self.P.n_rays)
        step = float(self.P.ray_step)
        ray_len = float(self.P.ray_len)

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
            self._Fp = z2()
            self._accB = z1()
            self._accF = z1()
            self._noiseB = z1()
            self._noiseF = z1()
            self._noise64 = z1(dtype=np.float64)
            return

        self._ang_base = (
            np.float32(2.0 * np.pi)
            * (np.arange(self._n, dtype=np.float32) / np.float32(self._n))
        )
        self._ang = np.empty((self._n,), dtype=np.float32)

        self._d = np.arange(step, ray_len + 1e-6, step, dtype=np.float32)
        self._m = int(self._d.size)

        # distance weights (nearer = higher)
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
        self._Fp = np.empty((self._n, self._m), dtype=np.float32)

        self._accB = np.empty((self._n,), dtype=np.float32)
        self._accF = np.empty((self._n,), dtype=np.float32)

        self._noiseB = np.empty((self._n,), dtype=np.float32)
        self._noiseF = np.empty((self._n,), dtype=np.float32)
        self._noise64 = np.empty((self._n,), dtype=np.float64)

    def sense(
        self,
        world: "World",
        x: float,
        y: float,
        heading: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
        """
        Perception API (u-domain):
          - Returns (B0_u, F0_u, C0_u) where:
              B0_u = percept(B0_kg) in [0,1]
              F0_u = hazard in [0,1] (world-native)
              C0_u = percept(C0_kg) in [0,1]
          - Returns rays_B_u, rays_F_u (both in [0,1])
        World is authoritative physics:
          - world.sample_bilinear(x,y) -> (B_kg, F_u, C_kg)
          - world.sample_bilinear_many_BF(xs,ys, outB, outF) -> outB kg, outF u
        """
        # ---- (A) Sample local physics ----
        B0_kg, F0_u, C0_kg = world.sample_bilinear(x, y)

        # perception parameters (NOT B_K; separate sense scales)
        Pworld = getattr(world, "P", None)
        Kb = float(getattr(Pworld, "B_sense_K", 0.0)) if Pworld is not None else 0.0
        Kc = float(getattr(Pworld, "C_sense_K", 0.0)) if Pworld is not None else 0.0

        B0_u = self._sat1_u(float(B0_kg), Kb)
        C0_u = self._sat1_u(float(C0_kg), Kc)
        F0_u = float(F0_u)  # already u-domain

        n = int(self._n)
        m = int(self._m)
        if n <= 0 or m <= 0:
            if n > 0:
                self._accB[:n].fill(0.0)
                self._accF[:n].fill(0.0)
                # noise/clamp in u-domain below (kept consistent)
            else:
                return (B0_u, F0_u, C0_u), self._accB[:0], self._accF[:0]

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

        # ---- (C) Sample world fields along rays ----
        # outB: kg, outF: u
        world.sample_bilinear_many_BF(self._xs, self._ys, outB=self._Bp, outF=self._Fp)

        # ---- (D) Convert biomass samples kg -> u in-place, then integrate in u-domain ----
        # _Bp overwritten from kg to u
        self._sat_u(self._Bp, Kb)
        np.matmul(self._Bp, self._w, out=self._accB)
        np.matmul(self._Fp, self._w, out=self._accF)

        self._accB *= self._inv_wsum
        self._accF *= self._inv_wsum

        # ---- (E) Noise in u-domain (same sigma semantics for all u-channels) ----
        sig = float(self.P.noise_sigma)
        if sig > 0.0 and (rng is not None):
            rng.standard_normal(size=n, out=self._noise64)
            self._noiseB[:] = (self._noise64 * sig).astype(np.float32, copy=False)
            rng.standard_normal(size=n, out=self._noise64)
            self._noiseF[:] = (self._noise64 * sig).astype(np.float32, copy=False)

            self._accB += self._noiseB
            self._accF += self._noiseF

            B0_u = float(B0_u + rng.normal(0.0, sig * 0.5))
            F0_u = float(F0_u + rng.normal(0.0, sig * 0.5))
            C0_u = float(C0_u + rng.normal(0.0, sig * 0.5))

        # ---- (F) Clamp ONLY u-domain outputs ----
        np.clip(self._accB, 0.0, 1.0, out=self._accB)
        np.clip(self._accF, 0.0, 1.0, out=self._accF)

        if B0_u < 0.0:
            B0_u = 0.0
        elif B0_u > 1.0:
            B0_u = 1.0

        if F0_u < 0.0:
            F0_u = 0.0
        elif F0_u > 1.0:
            F0_u = 1.0

        if C0_u < 0.0:
            C0_u = 0.0
        elif C0_u > 1.0:
            C0_u = 1.0

        return (float(B0_u), float(F0_u), float(C0_u)), self._accB, self._accF

    def see_agent_first_hit(
        self,
        world: "World",
        x: float,
        y: float,
        heading: float,
        self_id: int,
    ) -> tuple[float, float, float]:
        """
        Returns (present, bearing_u, dist_u)
          present in {0,1}
          bearing_u in [0,1)
          dist_u in [0,1]
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
                    dist_u = float(self._d[j] / max(float(self.P.ray_len), 1e-9))
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
      - Phenotype (tolkbara parametrar) härleds från traits och är konstant över livstid
      - Population sköter reproduktion + mutation; Agent håller cooldown
    """

    AP: AgentParams
    genome: MLPGenome

    x: float
    y: float
    heading: float

    id: int = field(default_factory=_new_agent_id)

    body: Body = field(init=False)
    sensors: RaySensors = field(init=False)

    # lightweight local memory
    obs_trace: np.ndarray = field(init=False)

    # life origin (absolute sim time at birth; set by Population)
    birth_t: float = 0.0

    # derived phenotype (fixed for lifetime)
    pheno: Phenotype = field(init=False)

    # local life history
    last_speed: float = 0.0
    age_s: float = 0.0
    repro_cd_s: float = 0.0

    # for tracking
    last_B0: float = 0.0
    last_F0: float = 0.0
    last_C0: float = 0.0

    sense_level: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.AP = replace(self.AP)

        self.body = Body(self.AP)
        self.obs_trace = np.zeros((9,), dtype=np.float32)

        self.age_s = 0.0
        self.repro_cd_s = 0.0
        self.birth_t = float(getattr(self, "birth_t", 0.0))

        # phenotype fixed for lifetime
        self.apply_traits()

        # apply sensing phenotype -> AP knobs, then rebuild sensors
        self.sense_level = _sense_level(float(self.pheno.sense_strength))
        _apply_sense_to_AP(self.AP, self.sense_level)
        self.sensors = RaySensors(self.AP, world_size=64)

        # init mass + energy (unbounded)
        self._init_body_state_from_AP()
    
    def _init_body_state_from_AP(self) -> None:
        self.body.M = max(0.0, float(self.AP.M0))

        E0 = max(0.0, float(self.AP.E0))
        # split total E0 into fast/slow respecting weights
        self.body.E_fast = (0.85 * E0) / 0.6
        self.body.E_slow = (0.15 * E0) / 0.4

        self.body.D = 0.0
        self.body.Fg = float(self.body.Fg)
        self.body.alive = True

    def bind_world(self, world: World) -> None:
        self.sensors = RaySensors(self.AP, world_size=world.P.size)

    @staticmethod
    def _signed_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def apply_traits(self) -> None:
        self.pheno = derive_pheno(self.genome.traits)

    def phenotype_summary(self) -> dict:
        return phenotype_summary(self.pheno)

    def _build_obs(self, B0: float, F0: float, C0: float, rays_B, rays_F) -> np.ndarray:
        rb = np.asarray(rays_B, dtype=np.float32)
        rf = np.asarray(rays_F, dtype=np.float32)

        n = int(rb.shape[0])
        if n <= 0:
            meanB = meanF = maxB = maxF = 0.0
            aB = aF = 0.0
        else:
            iB = int(np.argmax(rb))
            iF = int(np.argmax(rf))
            aB = 2.0 * math.pi * (iB / n)
            aF = 2.0 * math.pi * (iF / n)
            meanB = float(rb.mean())
            meanF = float(rf.mean())
            maxB = float(rb[iB])
            maxF = float(rf[iF])

        hunger = float(self.body.hunger())
        fatigue = float(self.body.Fg)
        D = float(self.body.D)

        obs = np.array(
            [float(B0), float(F0), float(C0), meanB, meanF, maxB, maxF, hunger, fatigue],
            dtype=np.float32,
        )

        a = 0.06
        self.obs_trace = (1.0 - a) * self.obs_trace + a * obs

        x = np.concatenate(
            [
                obs,
                self.obs_trace,
                np.array([math.cos(aB), math.sin(aB), math.cos(aF), math.sin(aF), D], dtype=np.float32),
            ]
        )
        return x

    def build_inputs(self, world: World, rng: np.random.Generator):
        if not self.body.alive:
            return None, 0.0, 0.0, 0.0

        (B0, F0, C0), rays_B, rays_F = self.sensors.sense(world, self.x, self.y, self.heading, rng=rng)
        x_in = self._build_obs(B0, F0, C0, rays_B, rays_F)
        return x_in, float(B0), float(F0), float(C0)

    def apply_outputs(
        self,
        world: World,
        y: np.ndarray,
        B0: float,
        F0: float,
        C0: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float, float]:
        # ------------------------------------------------------------
        # 0) Early exit + clocks
        # ------------------------------------------------------------
        if not self.body.alive:
            return 0.0, 0.0, 0.0
    
        dt = float(self.AP.dt)
        self.age_s += dt
        self.repro_cd_s = max(0.0, float(self.repro_cd_s) - dt)
    
        # ------------------------------------------------------------
        # 1) Decode policy outputs (bounded controls)
        # ------------------------------------------------------------
        turn = float(np.tanh(y[0]))  # [-1,1]
        thrust = float(1.0 / (1.0 + np.exp(-float(y[1]))))         # [0,1]
        inh_move = float(1.0 / (1.0 + np.exp(-float(y[2]))))       # [0,1]
        inh_eat = float(1.0 / (1.0 + np.exp(-float(y[3]))))        # [0,1]
        explore_drive = float(1.0 / (1.0 + np.exp(-float(y[4]))))  # [0,1]
    
        allow_move = 1.0 - inh_move
        allow_eat = 1.0 - inh_eat
    
        # ------------------------------------------------------------
        # 2) Environment/phenotype modulation (cold avoidance)
        # ------------------------------------------------------------
        Tloc = world.temperature_at(self.x, self.y) if hasattr(world, "temperature_at") else 0.0
        T_comfort = 10.0
        T_width = 12.0
        coldness = clamp((T_comfort - float(Tloc)) / max(T_width, 1e-9), 0.0, 1.0)
        cold_drive = float(self.pheno.cold_aversion) * float(coldness)
    
        thrust_eff = thrust * (1.0 - 0.85 * cold_drive)
    
        # weak centering bias when cold (unchanged semantics)
        s = int(world.P.size)
        mid = 0.5 * (s - 1)
        dy = (mid - float(self.y))
        target_heading = math.atan2(dy, 0.0)
        err = self._signed_angle(target_heading - self.heading)
        bias_turn = clamp(err / math.pi, -1.0, 1.0)
        turn = clamp(turn + 0.80 * cold_drive * bias_turn, -1.0, 1.0)
    
        # ------------------------------------------------------------
        # 3) Heading integration (turn + exploration jitter)
        # ------------------------------------------------------------
        jitter = float(rng.normal(0.0, 0.65)) * explore_drive
        self.heading = float(self.heading) + dt * float(self.AP.turn_rate) * (
            0.85 * allow_move * turn + 0.25 * jitter
        )
        self.heading = self._signed_angle(self.heading)
    
        # ------------------------------------------------------------
        # 4) Locomotion control signal u in [0,1]
        # ------------------------------------------------------------
        fatigue = float(self.body.Fg)
        fatigue_factor = clamp(1.0 - 0.9 * fatigue, 0.05, 1.0)
        weak_move = float(self.body.move_factor())
        u = clamp(allow_move * thrust_eff * fatigue_factor * weak_move, 0.0, 1.0)
    
        # ------------------------------------------------------------
        # 5) Locomotion dynamics
        # ------------------------------------------------------------
        v_prev = max(0.0, float(self.last_speed))
        M_pre = max(1e-9, float(self.body.M))  # mass pre Body.step
    
        F0_cap = float(getattr(self.AP, "F0", 4.0))
        alpha = float(getattr(self.AP, "force_mass_exp", 2.0 / 3.0))
        F_prop = u * F0_cap * (M_pre ** alpha)
    
        c1 = float(getattr(self.AP, "drag_lin", 0.8))
        c2 = float(getattr(self.AP, "drag_quad", 0.2))
    
        F_drag_prev = c1 * v_prev + c2 * v_prev * v_prev
        a = (F_prop - F_drag_prev) / M_pre
        v_euler = max(0.0, v_prev + dt * a)
    
        speed = min(v_euler, float(self.AP.v_max))
    
        # midpoint speed for power accounting
        v_mid = 0.5 * (v_prev + speed)
        self.last_speed = float(speed)
    
        # ------------------------------------------------------------
        # 6) Locomotion energy
        # ------------------------------------------------------------
        eta = clamp(float(getattr(self.AP, "locomotion_eff", 0.25)), 1e-6, 1.0)
        P_mech = max(0.0, F_prop * v_mid)
        E_move = (dt * P_mech) / eta
    
        # ------------------------------------------------------------
        # 7) Apply translation (torus)
        # ------------------------------------------------------------
        self.x = torus_wrap(float(self.x) + dt * speed * math.cos(self.heading), world.P.size)
        self.y = torus_wrap(float(self.y) + dt * speed * math.sin(self.heading), world.P.size)
    
        # ------------------------------------------------------------
        # 8) Feeding (kg)
        # ------------------------------------------------------------
        got_bio = 0.0
        got_carcass = 0.0
        if allow_eat > 0.20:
            want_kg = float(self.AP.eat_rate) * dt * (0.25 + 0.75 * float(self.body.hunger()))
            got_bio, got_carcass = world.consume_food(
                self.x, self.y, amount=want_kg, prefer_carcass=True
            )
    
        food_bio_kg = float(got_bio)
        food_carcass_kg = float(got_carcass)
    
        # ------------------------------------------------------------
        # 9) Activity proxy (for compute/fatigue/stress coupling)
        # ------------------------------------------------------------
        speed_n = clamp(speed / max(float(self.AP.v_max), 1e-9), 0.0, 1.0)
        ate = 1.0 if (allow_eat > 0.20 and (food_bio_kg + food_carcass_kg) > 0.0) else 0.0
        activity = 0.03 + 0.45 * speed_n + 0.10 * ate
    
        # ------------------------------------------------------------
        # 10) Social attraction (unchanged semantics)
        # ------------------------------------------------------------
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
    
        # ------------------------------------------------------------
        # 11) Body dynamics
        # ------------------------------------------------------------
        self.body.step(
            speed=speed,
            activity=activity,
            hazard=F0,
            food_bio_kg=food_bio_kg,
            food_carcass_kg=food_carcass_kg,
            pheno=self.pheno,
            extra_drain=E_move,
        )
    
        # ------------------------------------------------------------
        # 12) Tracking outputs
        # ------------------------------------------------------------
        self.last_B0 = float(B0)
        self.last_F0 = float(F0)
        self.last_C0 = float(C0)
    
        return float(B0), float(F0), float(C0)

    # optional legacy step() if used somewhere else
    def step(self, world: World, rng: np.random.Generator) -> Tuple[float, float, float]:
        x_in, B0, F0, C0 = self.build_inputs(world, rng=rng)
        y = self.genome.forward(x_in)
        return self.apply_outputs(world, y, B0, F0, C0, rng=rng)

    # --- reproduction hooks (Population uses these) ---
    def ready_to_reproduce(self) -> bool:
        if not self.body.alive:
            return False
        if float(self.repro_cd_s) > 0.0:
            return False
        if float(self.age_s) < float(self.pheno.A_mature):
            return False
    
        # --- Mass gate (råvara) ---
        M = float(self.body.M)
        Mreq = float(self.pheno.M_repro_min)
        if M < max(float(self.AP.M_min), Mreq):
            return False
    
        # --- Energy gate (andel av E_cap) ---
        Et = float(self.body.E_total())
        Ecap = float(self.body.E_cap())
        if Et < float(self.pheno.E_repro_min) * Ecap:
            return False
    
        return True

    def wants_to_reproduce(self, rng: np.random.Generator) -> bool:
        if not self.ready_to_reproduce():
            return False
        lam = float(self.pheno.repro_rate)
        p = 1.0 - math.exp(-lam * float(self.AP.dt))
        return bool(rng.random() < p)

    def pay_repro_cost(self, cost_E: float) -> float:
        Ecap = float(self.body.E_cap())
        # pheno.repro_cost är fraktion av Ecap
        costE = max(float(cost_E), float(self.pheno.repro_cost) * Ecap)
    
        paidE = float(self.body.take_energy(costE))
        deficit = max(0.0, costE - paidE)
    
        # Rekommendation: bränn INTE massa här – låt Body.step sköta M->E under drift.
        # Om du vill ha masskostnad, gör den explicit som "barn-massa" (se punkt 3).
        # if deficit > 0.0: ...
    
        self.repro_cd_s = float(self.AP.repro_cooldown_s)
        return float(paidE)
    
    def provide_child_mass(self, child_M: float) -> float:
        want = max(0.0, float(child_M))
        got = min(want, max(0.0, float(self.body.M) - float(self.AP.M_min)))
        self.body.M = float(self.body.M) - got
        return float(got)
        
    def init_newborn_state(self, parent_pheno: Phenotype, child_M_from_parent: float | None = None) -> None:
        # Energi: tolka child_E_fast/slow som TOTAL-E-andelar (0..1-ish) av barnets E_cap(M)
        child_M = float(child_M_from_parent) if child_M_from_parent is not None else float(getattr(parent_pheno, "child_M", self.AP.M0 * 0.5))
        child_M = max(float(self.AP.M_min), child_M)
        self.body.M = child_M
    
        # Barnets energikapacitet beror på barnets M
        Ecap = float(self.body.E_cap())
        Ef_u = clamp(float(getattr(parent_pheno, "child_E_fast", 0.50)), 0.0, 1.0)  # bidrag till E_total i units av Ecap
        Es_u = clamp(float(getattr(parent_pheno, "child_E_slow", 0.20)), 0.0, 1.0)
                
        # Sätt lagren så att deras bidrag blir exakt Ef_u*Ecap respektive Es_u*Ecap
        self.body.E_fast = (Ef_u * Ecap) / 0.6
        self.body.E_slow = (Es_u * Ecap) / 0.4
        
        self.body.Fg = clamp(float(getattr(parent_pheno, "child_Fg", 0.15)), 0.0, 1.0)
        self.body.D = 0.0
        self.body.alive = True
        self.repro_cd_s = float(self.AP.repro_cooldown_s)
        self.age_s = 0.0

