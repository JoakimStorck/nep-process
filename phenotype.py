# phenotype.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sigmoid(x: float) -> float:
    # numeriskt stabil för rimliga x; traits klipps ändå i genetics
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
    # Fas 1 (A): selektion via livshistoria
    A_mature: float          # time to maturity (sim-time seconds)
    p_repro_base: float      # base reproduction probability per attempt (0..1-ish)
    E_repro_min: float       # min energy to reproduce
    repro_cost: float        # energy cost paid by parent at birth

    # Fas 3–4 (C): placeholders / senare kopplingar
    metabolism_scale: float  # multiplier on baseline metabolism
    risk_aversion: float     # 0..1
    sociability: float       # 0..1
    mobility: float          # 0..1

    senescence_rate: float   # >0 ger åldrande via damage-drift
    senescence_shape: float  # 0..1 styr när rampen slår i

# Fixed trait indices (explicit + stable)
# You can change these later without touching agent/population logic.
_T_A_MATURE     = 0
_T_P_REPRO      = 1
_T_E_REPRO_MIN  = 2
_T_REPRO_COST   = 3
_T_METAB        = 4
_T_RISK_AV      = 5
_T_SOC          = 6
_T_MOB          = 7
_T_SENESC_RATE  = 8
_T_SENESC_SHAPE = 9

@dataclass(frozen=True)
class PhenoRanges:
    """
    Conservative defaults for MV0 to avoid runaway dynamics.
    Tune once reproduction gate is in place.
    """
    A_mature_min: float = 10.0
    A_mature_max: float = 180.0

    p_repro_min: float = 0.2
    p_repro_max: float = 0.50

    # These should match your body's typical energy scale.
    # Start modest; adjust after first run.
    E_repro_min_min: float = 0.40
    E_repro_min_max: float = 1.20

    repro_cost_min: float = 0.05
    repro_cost_max: float = 0.40

    metabolism_min: float = 0.85
    metabolism_max: float = 1.15

    senesc_rate_min: float = 0.0005
    senesc_rate_max: float = 0.0040
    
def derive_pheno(traits: np.ndarray | None, R: PhenoRanges = PhenoRanges()) -> Phenotype:
    """
    Deterministic mapping: traits -> bounded phenotype params.

    Uses sigmoid to map trait_i to [0,1], then scales to configured ranges.
    Keeps everything stable and inspectable.
    """
    u_mature = _sigmoid(_get_trait(traits, _T_A_MATURE))
    u_prepro = _sigmoid(_get_trait(traits, _T_P_REPRO))
    u_emin   = _sigmoid(_get_trait(traits, _T_E_REPRO_MIN))
    u_cost   = _sigmoid(_get_trait(traits, _T_REPRO_COST))
    u_metab  = _sigmoid(_get_trait(traits, _T_METAB))
    u_risk   = _sigmoid(_get_trait(traits, _T_RISK_AV))
    u_soc    = _sigmoid(_get_trait(traits, _T_SOC))
    u_mob    = _sigmoid(_get_trait(traits, _T_MOB))
    u_senr  = _sigmoid(_get_trait(traits, _T_SENESC_RATE))
    u_sensh = _sigmoid(_get_trait(traits, _T_SENESC_SHAPE))

    A_mature = _lerp(R.A_mature_min, R.A_mature_max, u_mature)
    p_repro  = _lerp(R.p_repro_min, R.p_repro_max, u_prepro)
    E_min    = _lerp(R.E_repro_min_min, R.E_repro_min_max, u_emin)
    cost     = _lerp(R.repro_cost_min, R.repro_cost_max, u_cost)
    metab    = _lerp(R.metabolism_min, R.metabolism_max, u_metab)
    senr = _lerp(R.senesc_rate_min, R.senesc_rate_max, u_senr)
    sensh = float(u_sensh)
    
    return Phenotype(
        A_mature=float(A_mature),
        p_repro_base=float(p_repro),
        E_repro_min=float(E_min),
        repro_cost=float(cost),
        metabolism_scale=float(metab),
        risk_aversion=float(u_risk),
        sociability=float(u_soc),
        mobility=float(u_mob),
        senescence_rate=float(senr),
        senescence_shape=float(sensh),        
    )


def phenotype_summary(p: Phenotype) -> dict[str, float]:
    """
    Canonical dict used for logging (stable key names).
    """
    return {
        "A_mature": float(p.A_mature),
        "p_repro_base": float(p.p_repro_base),
        "E_repro_min": float(p.E_repro_min),
        "repro_cost": float(p.repro_cost),
        "metabolism_scale": float(p.metabolism_scale),
        "risk_aversion": float(p.risk_aversion),
        "sociability": float(p.sociability),
        "mobility": float(p.mobility),
        "senescence_rate":float(p.senescence_rate),
        "senescence_shape":float(p.senescence_shape),
    }