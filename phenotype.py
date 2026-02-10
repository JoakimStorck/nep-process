# phenotype.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -8.0, 8.0))
    return 1.0 / (1.0 + float(np.exp(-x)))


def _lerp(a: float, b: float, u: float) -> float:
    return float(a) + (float(b) - float(a)) * float(u)


def _get_trait(traits: np.ndarray | None, i: int, default: float = 0.0) -> float:
    if traits is None:
        return float(default)
    if i < 0 or i >= int(traits.shape[0]):
        return float(default)
    return float(traits[i])


@dataclass(frozen=True)
class Phenotype:
    # Life history (A)
    A_mature: float
    repro_rate: float
    E_repro_min: float
    repro_cost: float

    # Metabolism + aging via damage (no internal clock)
    metabolism_scale: float     # scales energy drain_rate
    susceptibility: float       # scales hazard/effort damage inflow
    stress_per_drain: float     # damage per unit drain_rate
    repair_capacity: float      # max repair rate
    frailty_gain: float         # how strongly D amplifies inflow / reduces repair

    # Children
    child_E_fast: float
    child_E_slow: float
    child_Fg: float
    
    # Placeholders / later
    risk_aversion: float
    sociability: float
    mobility: float


# ---- Fixed trait indices (explicit + stable) ----
_T_A_MATURE        = 0
_T_P_REPRO         = 1
_T_E_REPRO_MIN     = 2
_T_REPRO_COST      = 3

_T_METAB           = 4
_T_SUSC            = 5
_T_STRESS_PER_D    = 6
_T_REPAIR_CAP      = 7
_T_FRAILTY_GAIN    = 8

_T_RISK_AV         = 9
_T_SOC             = 10
_T_MOB             = 11

_T_CHILD_E_FAST = 12
_T_CHILD_E_SLOW = 13
_T_CHILD_FG     = 14

@dataclass(frozen=True)
class PhenoRanges:
    # maturity
    A_mature_min: float = 5.0
    A_mature_max: float = 40.0

    # reproduction
    repro_rate_min: float = 0.20
    repro_rate_max: float = 0.50

    E_repro_min_min: float = 0.40
    E_repro_min_max: float = 1.20

    repro_cost_min: float = 0.05
    repro_cost_max: float = 0.40

    # metabolism + damage/repair (MV0 conservative)
    metabolism_min: float = 0.85
    metabolism_max: float = 1.15

    susceptibility_min: float = 0.70
    susceptibility_max: float = 1.40

    stress_per_drain_min: float = 0.02
    stress_per_drain_max: float = 0.20

    repair_capacity_min: float = 0.01
    repair_capacity_max: float = 0.1

    frailty_gain_min: float = 0.0
    frailty_gain_max: float = 3.0

    child_E_fast_min: float = 0.10
    child_E_fast_max: float = 0.70
    child_E_slow_min: float = 0.10
    child_E_slow_max: float = 0.70
    child_Fg_min: float = 0.00
    child_Fg_max: float = 0.40

def derive_pheno(traits: np.ndarray | None, R: PhenoRanges = PhenoRanges()) -> Phenotype:
    # u in [0,1]
    u_mature   = _sigmoid(_get_trait(traits, _T_A_MATURE))
    u_prepro   = _sigmoid(_get_trait(traits, _T_P_REPRO))
    u_emin     = _sigmoid(_get_trait(traits, _T_E_REPRO_MIN))
    u_cost     = _sigmoid(_get_trait(traits, _T_REPRO_COST))

    u_metab    = _sigmoid(_get_trait(traits, _T_METAB))
    u_susc     = _sigmoid(_get_trait(traits, _T_SUSC))
    u_spd      = _sigmoid(_get_trait(traits, _T_STRESS_PER_D))
    u_rep      = _sigmoid(_get_trait(traits, _T_REPAIR_CAP))
    u_frail    = _sigmoid(_get_trait(traits, _T_FRAILTY_GAIN))

    u_risk     = _sigmoid(_get_trait(traits, _T_RISK_AV))
    u_soc      = _sigmoid(_get_trait(traits, _T_SOC))
    u_mob      = _sigmoid(_get_trait(traits, _T_MOB))

    u_cef  = _sigmoid(_get_trait(traits, _T_CHILD_E_FAST))
    u_ces  = _sigmoid(_get_trait(traits, _T_CHILD_E_SLOW))
    u_cfg  = _sigmoid(_get_trait(traits, _T_CHILD_FG))
    
    return Phenotype(
        A_mature=float(_lerp(R.A_mature_min, R.A_mature_max, u_mature)),
        repro_rate=float(_lerp(R.repro_rate_min, R.repro_rate_max, u_prepro)),
        E_repro_min=float(_lerp(R.E_repro_min_min, R.E_repro_min_max, u_emin)),
        repro_cost=float(_lerp(R.repro_cost_min, R.repro_cost_max, u_cost)),

        metabolism_scale=float(_lerp(R.metabolism_min, R.metabolism_max, u_metab)),
        susceptibility=float(_lerp(R.susceptibility_min, R.susceptibility_max, u_susc)),
        stress_per_drain=float(_lerp(R.stress_per_drain_min, R.stress_per_drain_max, u_spd)),
        repair_capacity=float(_lerp(R.repair_capacity_min, R.repair_capacity_max, u_rep)),
        frailty_gain=float(_lerp(R.frailty_gain_min, R.frailty_gain_max, u_frail)),

        risk_aversion=float(u_risk),
        sociability=float(u_soc),
        mobility=float(u_mob),

        child_E_fast=float(_lerp(R.child_E_fast_min, R.child_E_fast_max, u_cef)),
        child_E_slow=float(_lerp(R.child_E_slow_min, R.child_E_slow_max, u_ces)),
        child_Fg=float(_lerp(R.child_Fg_min, R.child_Fg_max, u_cfg)),
    )


def phenotype_summary(p: Phenotype) -> dict[str, float]:
    return {
        "A_mature": float(p.A_mature),
        "repro_rate": float(p.repro_rate),
        "E_repro_min": float(p.E_repro_min),
        "repro_cost": float(p.repro_cost),

        "metabolism_scale": float(p.metabolism_scale),
        "susceptibility": float(p.susceptibility),
        "stress_per_drain": float(p.stress_per_drain),
        "repair_capacity": float(p.repair_capacity),
        "frailty_gain": float(p.frailty_gain),

        "risk_aversion": float(p.risk_aversion),
        "sociability": float(p.sociability),
        "mobility": float(p.mobility),
    }