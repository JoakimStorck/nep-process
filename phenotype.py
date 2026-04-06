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
    M_repro_min: float

    # Metabolism + aging via damage (no internal clock)
    metabolism_scale: float
    susceptibility: float
    stress_per_drain: float
    repair_capacity: float
    frailty_gain: float
    E_rep_min: float

    # Children
    child_E_fast: float
    child_E_slow: float
    child_Fg: float
    child_M: float   # NEW

    # Genetiskt tillväxtprogram
    M_target: float

    # Kostpreferens: 0=herbivore, 0.5=generalist, 1=scavenger
    diet: float

    # Predation: benägenhet att attackera levande byten (0=fredsam, 1=rovdjur)
    predation: float

    # Nätverksarkitektur — dolda lagrets bredder (diskreta steg, genetiskt bestämda)
    # Evolution väljer kapacitet mot energikostnad: small=billig+snabb, large=dyr+kraftfull
    hidden_1: int   # bredd på lager 1
    hidden_2: int   # bredd på lager 2

    # Placeholders / later
    risk_aversion: float
    sociability: float
    mobility: float
    cold_aversion: float
    sense_strength: float


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

_T_CHILD_E_FAST    = 12
_T_CHILD_E_SLOW    = 13
_T_CHILD_FG        = 14
_T_COLD_AV         = 15

_T_E_REP_MIN       = 16
_T_SENSE           = 17

_T_CHILD_M         = 18
_T_M_REPRO_MIN     = 19
_T_M_TARGET        = 20   # genetiskt bestämd vuxenmassa
_T_DIET            = 21   # 0=ren herbivore, 1=ren scavenger
_T_PREDATION       = 22   # benägenhet att attackera levande byten
_T_HIDDEN_1        = 23   # bredd på första dolda lagret
_T_HIDDEN_2        = 24   # bredd på andra dolda lagret

# ---- Shared organism trait block (phase 2+) ----
# Dessa loci är avsedda att vara gemensamma på organismnivå.
# Alla system behöver inte använda alla loci.
_T_AUTOTROPHY      = 25
_T_GROWTH          = 26
_T_ADULT_MASS      = 27
_T_TEMP_OPT        = 28
_T_TEMP_WIDTH      = 29
_T_SEXUAL_MODE     = 30
_T_DISPERSAL       = 31

@dataclass(frozen=True)
class PhenoRanges:
    # maturity
    A_mature_min: float = 5.0
    A_mature_max: float = 20.0   # var 40.0 — kortare mognadsperiod relativt livslängden

    # reproduction
    repro_rate_min: float = 1.00
    repro_rate_max: float = 2.50

    # reproduction
    E_repro_min_min: float = 0.05   # var 0.10
    E_repro_min_max: float = 0.25   # var 0.40 — sänkt så att mean_R≈0.33 räcker
    
    repro_cost_min: float = 0.01
    repro_cost_max: float = 0.05    # sänkt tak — extra cost vid födseln (var 0.08)

    # metabolism + damage/repair
    metabolism_min: float = 0.85
    metabolism_max: float = 1.15

    susceptibility_min: float = 0.70
    susceptibility_max: float = 1.40

    stress_per_drain_min: float = 0.01
    stress_per_drain_max: float = 0.05

    repair_capacity_min: float = 0.002
    repair_capacity_max: float = 0.030   # höjt — k_age1 ger lägre inflöde än k_age0=0.2

    frailty_gain_min: float = 0.0
    frailty_gain_max: float = 3.0

    E_rep_min_min: float = 0.00
    E_rep_min_max: float = 0.35

    # children
    # Energi till barnet: fraktion av barnets Ecap som föräldern betalar.
    child_E_fast_min: float = 0.05   # var 0.10
    child_E_fast_max: float = 0.40   # var 0.70 — sänkt tak
    child_E_slow_min: float = 0.05   # var 0.10
    child_E_slow_max: float = 0.40   # var 0.70 — sänkt tak
    child_Fg_min: float = 0.00
    child_Fg_max: float = 0.40

    # Barnets massa vid födseln.
    # Kompromiss: litet nog att kosta föräldern rimligt, stort nog att överleva.
    child_M_min: float = 0.08   # var 0.10 → 0.05 (för litet gav utrotning)
    child_M_max: float = 0.20   # var 0.30 → 0.15 (gav utrotning) → 0.20

    cold_aversion_min: float = 0.0
    cold_aversion_max: float = 1.0

    # Reproduktions-mass-tröskel.
    # Genetiskt bestämd vuxenmassa
    # Brett intervall — evolution hittar r- och K-strateger
    M_target_min: float = 0.10
    M_target_max: float = 2.00

    # Kostpreferens — brett intervall för maximal nischuppdelning
    diet_min: float = 0.0
    diet_max: float = 1.0

    # Predation — brett för att ge evolution chans att hitta rovdjur
    predation_min: float = 0.0
    predation_max: float = 1.0

    M_repro_min_min: float = 0.15   # var 0.20 — lite lägre för att hinna reproducera
    M_repro_min_max: float = 0.45   # var 0.45

    # Nätverksarkitektur — tillåtna bredder för dolda lager.
    # Diskreta steg via snap-funktion i derive_pheno.
    # sigmoid(0)=0.5 → lerp(8,40)=24 → snap=24 (logisk default/mittenvärde).
    # Med logit-initialisering täcks hela spannet 8-40 uniformt.
    # Min 8 → mycket litet och billigt; max 40 → kraftfullt men metabolt dyrt.
    hidden_min: int = 8
    hidden_max: int = 40

def derive_pheno(traits: np.ndarray | None, R: PhenoRanges = PhenoRanges()) -> Phenotype:
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

    u_cef      = _sigmoid(_get_trait(traits, _T_CHILD_E_FAST))
    u_ces      = _sigmoid(_get_trait(traits, _T_CHILD_E_SLOW))
    u_cfg      = _sigmoid(_get_trait(traits, _T_CHILD_FG))

    u_cold     = _sigmoid(_get_trait(traits, _T_COLD_AV))
    u_erep     = _sigmoid(_get_trait(traits, _T_E_REP_MIN))
    u_sense    = _sigmoid(_get_trait(traits, _T_SENSE))

    # NEW
    u_childM   = _sigmoid(_get_trait(traits, _T_CHILD_M))

    u_mrepro   = _sigmoid(_get_trait(traits, _T_M_REPRO_MIN))
    u_Mtarget  = _sigmoid(_get_trait(traits, _T_M_TARGET))
    u_diet       = _sigmoid(_get_trait(traits, _T_DIET))
    u_predation  = _sigmoid(_get_trait(traits, _T_PREDATION))

    # Arkitektur-traits: kontinuerligt u → diskret bredd med steg om 4.
    # round-till-närmaste-4 ger jämna storlekar och begränsar antalet
    # unika ParamBank-nycklar (viktigt för batch-prestanda).
    u_h1 = _sigmoid(_get_trait(traits, _T_HIDDEN_1, default=0.0))
    u_h2 = _sigmoid(_get_trait(traits, _T_HIDDEN_2, default=0.0))

    def _snap_hidden(u: float, lo: int, hi: int) -> int:
        raw = _lerp(float(lo), float(hi), u)
        snapped = int(round(raw / 4.0)) * 4
        return max(lo, min(hi, snapped))

    hidden_1 = _snap_hidden(u_h1, int(R.hidden_min), int(R.hidden_max))
    hidden_2 = _snap_hidden(u_h2, int(R.hidden_min), int(R.hidden_max))

    return Phenotype(
        A_mature=float(_lerp(R.A_mature_min, R.A_mature_max, u_mature)),
        repro_rate=float(_lerp(R.repro_rate_min, R.repro_rate_max, u_prepro)),
        E_repro_min=float(_lerp(R.E_repro_min_min, R.E_repro_min_max, u_emin)),
        repro_cost=float(_lerp(R.repro_cost_min, R.repro_cost_max, u_cost)),
    
        # NEW
        M_repro_min=float(_lerp(R.M_repro_min_min, R.M_repro_min_max, u_mrepro)),
        M_target=float(_lerp(R.M_target_min, R.M_target_max, u_Mtarget)),
    
        metabolism_scale=float(_lerp(R.metabolism_min, R.metabolism_max, u_metab)),
        susceptibility=float(_lerp(R.susceptibility_min, R.susceptibility_max, u_susc)),
        stress_per_drain=float(_lerp(R.stress_per_drain_min, R.stress_per_drain_max, u_spd)),
        repair_capacity=float(_lerp(R.repair_capacity_min, R.repair_capacity_max, u_rep)),
        frailty_gain=float(_lerp(R.frailty_gain_min, R.frailty_gain_max, u_frail)),
        E_rep_min=float(_lerp(R.E_rep_min_min, R.E_rep_min_max, u_erep)),
    
        child_E_fast=float(_lerp(R.child_E_fast_min, R.child_E_fast_max, u_cef)),
        child_E_slow=float(_lerp(R.child_E_slow_min, R.child_E_slow_max, u_ces)),
        child_Fg=float(_lerp(R.child_Fg_min, R.child_Fg_max, u_cfg)),
        child_M=float(_lerp(R.child_M_min, R.child_M_max, u_childM)),
    
        risk_aversion=float(u_risk),
        sociability=float(u_soc),
        mobility=float(u_mob),
        cold_aversion=float(_lerp(R.cold_aversion_min, R.cold_aversion_max, u_cold)),
        sense_strength=float(u_sense),
        diet=float(_lerp(R.diet_min, R.diet_max, u_diet)),
        predation=float(_lerp(R.predation_min, R.predation_max, u_predation)),
        hidden_1=int(hidden_1),
        hidden_2=int(hidden_2),
    )


def phenotype_summary(p: Phenotype) -> dict[str, float]:
    return {
        "A_mature": float(p.A_mature),
        "repro_rate": float(p.repro_rate),
        "E_repro_min": float(p.E_repro_min),
        "repro_cost": float(p.repro_cost),
        "M_repro_min": float(p.M_repro_min),
        "M_target": float(p.M_target),
        "diet": float(p.diet),
        "predation": float(p.predation),
        "E_rep_min": float(p.E_rep_min),

        "metabolism_scale": float(p.metabolism_scale),
        "susceptibility": float(p.susceptibility),
        "stress_per_drain": float(p.stress_per_drain),
        "repair_capacity": float(p.repair_capacity),
        "frailty_gain": float(p.frailty_gain),

        "child_E_fast": float(p.child_E_fast),
        "child_E_slow": float(p.child_E_slow),
        "child_Fg": float(p.child_Fg),
        "child_M": float(p.child_M),   # NEW

        "risk_aversion": float(p.risk_aversion),
        "sociability": float(p.sociability),
        "mobility": float(p.mobility),
        "cold_aversion": float(p.cold_aversion),
        "sense_strength": float(p.sense_strength),
        "hidden_1": int(p.hidden_1),
        "hidden_2": int(p.hidden_2),
    }

def trait_unit(traits: np.ndarray | None, i: int, default: float = 0.0) -> float:
    return _sigmoid(_get_trait(traits, i, default))


def trait_lerp(
    traits: np.ndarray | None,
    i: int,
    lo: float,
    hi: float,
    default: float = 0.0,
) -> float:
    return _lerp(lo, hi, trait_unit(traits, i, default))


def autotrophy_from_traits(traits: np.ndarray | None) -> float:
    return trait_unit(traits, _T_AUTOTROPHY, default=0.0)


def growth_from_traits(traits: np.ndarray | None) -> float:
    return trait_unit(traits, _T_GROWTH, default=0.0)


def adult_mass_from_traits(traits: np.ndarray | None) -> float:
    return trait_unit(traits, _T_ADULT_MASS, default=0.0)


def temp_opt_from_traits(traits: np.ndarray | None) -> float:
    return trait_unit(traits, _T_TEMP_OPT, default=0.0)


def temp_width_from_traits(traits: np.ndarray | None) -> float:
    return trait_unit(traits, _T_TEMP_WIDTH, default=0.0)


def sexual_mode_from_traits(traits: np.ndarray | None) -> float:
    return trait_unit(traits, _T_SEXUAL_MODE, default=0.0)


def dispersal_from_traits(traits: np.ndarray | None) -> float:
    return trait_unit(traits, _T_DISPERSAL, default=0.0)