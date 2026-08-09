# phenotype.py
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _sigmoid(x: float) -> float:
    """
    Logistisk funktion på en skalär.

    Ren python och inte numpy: `np.clip` plus `np.exp` på ett enda tal kostar
    4,42 µs mot 0,116 µs, alltså trettioåtta gånger mer, och funktionen anropas
    tio gånger per floraetablering via `_init_flora_slot`. Vid trettiotusen
    plantor stod den för 6,6 procent av hela ticken. Resultatet är bitidentiskt
    — båda vägar går till libm:s exp.
    """
    v = float(x)
    if v < -8.0:
        v = -8.0
    elif v > 8.0:
        v = 8.0
    return 1.0 / (1.0 + math.exp(-v))


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
    reserve_cap: float
    fast_frac: float
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
    breed_phase: float
    breed_sync: float
    breed_pull: float
    sense_strength: float

    # Vävnadens strukturandel. Body behöver den för katabolismens utbyte och
    # för kadavrets sammansättning; utan den skulle fysiologin få gissa sin
    # egen komposition ur en konstant.
    structure: float = 0.25
    # Härledd ur strukturandelen, inte ett eget locus. Se
    # `buoyancy_from_structure`.
    buoyancy: float = 0.0


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

# Gemensamt för flora och fauna: andel av vävnaden som är segt bärande material
# — lignin och cellulosa hos växter, kitin och ben hos djur — mot lättomsatt
# protein, socker och fett. Samma tal beskriver båda ontologierna.
_T_STRUCTURE       = 32

# Näringsupptagets effektivitet. Eget locus, inte härlett ur autotrofin.
# Att vara växt och att vara effektiv på näringsupptag är olika egenskaper:
# autotrofin initieras högt för att göra flora till flora, vilket lämnade
# upptaget klustrat i övre änden av sin skala med bara 18 % av intervallet
# utnyttjat — selektionen hade nästan ingenting att gripa i.
_T_UPTAKE          = 33

# Reservkapacitet: hur mycket energi organismen kan bära som mobiliserbar
# reserv, i joule per kilo kroppsmassa. Var konstanten AP.E_cap_per_M.
_T_RESERVE         = 34

# Propagulmassa i absoluta tal. Var tidigare en fast andel av moderns
# vuxenmassa, vilket är både orimligt — ett träd på 44 kg gav ett frö på 4,4 —
# och skalfritt, så att storleksaxeln bara kunde falla. Det här är det enda
# locus som tillkommer i serien; `_T_DISPERSAL` och `_T_GROWTH` bytte jobb.
_T_SEED_MASS       = 35

# Rotandel: hur stor del av tillväxten som går till rot i stället för skott.
# Den enda mekanism som kan uttrycka balansen mellan två begränsande resurser,
# och två begränsande resurser existerar först sedan 0061. Båda ändarna är
# dödliga — ingen rot ger ingen kväveupptagning, inget skott ingen kolfixering
# — så ett inre optimum följer av mekaniken och inte av parametrarna.
# Funktionell jämvikt: Brouwer 1963; Bloom, Chapin & Mooney 1985.
_T_ROOT_ALLOC      = 36

# Mognadströskel: vid vilken andel av sin vuxenmassa plantan börjar reproducera
# sig. Den enda mekanism som kan uttrycka generationstid. Tidskostnaden faller
# ut ur tillväxthastighet och målstorlek och kodas inte: örten når sin lilla
# tröskel på månader, det vedartade sin stora på decennier.
#
# Evolverbar och inte konstant, av ett skäl utöver flexibilitet: en hårdkodad
# tröskel kan sterilisera en population, vilket den gjorde i w1 där
# medianplantan låg på en sjundedel av `0,20 · vuxenmassa`. En linje som inte
# kan reproducera sig försvinner omedelbart, så felläget blir strukturellt
# omöjligt.
_T_MATURITY        = 37

# Säsongsbunden parning. Två loci: när på året och hur snävt.
#
# Beredskapen är i dag asynkron — varje individ blir redo på sin egen
# tidtabell. Uppmätt i p97 var 14,8 procent av agenttickarna reproduktivt
# klara, och två djur måste vara det *samtidigt*: 0,148² = 2,2 procent, innan
# de ens ska hitta varandra. Det är kvadreringen som dödar reproduktionen.
#
# Grindas beredskapen till en period blir alla redo samtidigt. Ett
# tvåmånadersfönster ger 1,0² × 2 = 2,0 månadsekvivalenter per år mot dagens
# 0,148² × 12 = 0,26 — ungefär åtta gånger fler tillfällen.
#
# `breed_phase` är fasen i årscykeln, `breed_sync` skärpan. Vid skärpa noll är
# grinden konstant ett, alltså dagens asynkrona beteende — nuvarande läge är en
# punkt i rummet snarare än något som tas bort.
#
# Avvägningen är verklig: en strikt säsongsparare som missar fönstret väntar
# ett år. Ju snävare synkronisering, desto större vinst i mötesfrekvens och
# desto större risk att individen inte är i kondition just då.
_T_BREED_PHASE     = 38
_T_BREED_SYNC      = 39

# Social synkronisering av häckningsfasen.
#
# Spridningen i `breed_phase` är ±0,19 år mot ett fönster på 2,7 månader vid
# skärpa 3 — spridningen är alltså **större än fönstret**, och bara halva
# överlappningen återstår: 10,9 procent mot de 22 som identiska faser skulle
# ge. Att stänga glappet genom smalare spridning eller bredare fönster är
# kalibrering.
#
# Social synkronisering av brunst är väldokumenterad hos får, getter, möss och
# flera flockdjur: kemisk signalering mellan individer som ofta är nära drar
# gruppens honor mot en gemensam cykel. Affinitetsmatrisen är precis den
# viktning det ska ske över — ett djur drar sin fas mot dem det verkligen
# umgås med, inte mot vem som helst som passerar.
#
# Locus behåller individens *anlag*; det som styr grinden är anlaget plus
# flockdragningen. Tre saker som kalibrering inte kan ge: synkroniseringen
# blir lokal, så varje flock konvergerar mot sin egen fas och två flockar i
# olika delar av världen får reproduktiv isolering; slingan är
# självförstärkande, eftersom de som synkroniserar får fler parningar och
# avkomman ärver dragningen; och avvägningen finns inbyggd, eftersom stark
# dragning betyder att man följer flocken även när dess fas är dålig för den
# egna konditionen — och en ensam individ med hög dragning har ingen att dra
# mot.
_T_BREED_PULL      = 40

# Andelen av assimilerad massa som går till den snabba reserven. Resten blir
# späck.
#
# Poolerna fick olika egenskaper i 0157 och 0158 — `M_slow` mobiliseras med tak
# och isolerar — och först då finns en avvägning att selektera på. Axeln är
# självbegränsande åt båda håll: för mycket snabbt ger ingen vinterbuffert och
# ingen isolering, för mycket långsamt ger en varm kropp som är trög att växa
# och föröka sig, eftersom både tillväxt och dräktighet drar ur reserven och
# `M_slow` bara släpper `slow_mobil_frac · dt` per steg.
#
# Därför behövs inget tak: överdriften straffar sig själv i båda riktningarna.
_T_FAST_FRAC       = 41

@dataclass(frozen=True)
class PhenoRanges:
    # maturity
    # Säsongsparning. `breed_phase` är en **förskjutning** mot årets
    # temperaturtopp, inte en absolut fas i cykeln.
    #
    # Som absolut fas var locus fritt och pekade var som helst i året. Två djur
    # fick därmed oberoende fönster, och grinden blev ett filter i stället för
    # en synkronisering: om var och en är i säsong en fjärdedel av året och
    # faserna är oberoende är sannolikheten att två är det samtidigt en
    # sextondel — sämre än den asynkrona utgångspunkten den skulle förbättra.
    # Uppmätt i p108: 75 födslar mot p96:s 211-862, med allt annat lika eller
    # bättre.
    #
    # I naturen är häckningstiden inte en godtycklig fas utan låst till att
    # ungarna ska födas när fodret är rikligt, och djuret läser en signal —
    # dagslängd eller temperatur — som är gemensam för alla i samma trakt.
    # Synkroniseringen uppstår då gratis: alla läser samma värld, och ingen
    # delad genetik behövs. Två djur på samma latitud får samma säsong även om
    # de aldrig träffats; två på olika latitud får olika, vilket är biologiskt
    # riktigt och ger rumslig struktur på köpet.
    #
    # Kvar att evolvera är förskjutningen: rätt läge beror på dräktighetstid
    # och på när floran faktiskt producerar, vilket djuret inte vet i förväg.
    # Intervallet spänner en fjärdedels år åt vardera hållet kring toppen.
    breed_offset_min: float = -0.25
    breed_offset_max: float = 0.25

    # Andel av avståndet till flockens fas som stängs per observation.
    breed_pull_min: float = 0.0
    breed_pull_max: float = 0.20

    breed_sync_min: float = 0.0
    breed_sync_max: float = 6.0    # von Mises-liknande skärpa; 0 = ingen säsong

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

    # Reservkapacitet i J/kg kroppsmassa. 0,5e6 är ungefär 5 % av
    # kroppsmassan som mobiliserbar reserv, 4,0e6 är drygt 40 % — magert
    # respektive vinterfett. Att bära reserven kostar, eftersom den räknas
    # in i M_carry och därmed i basal, rörelse och värmeförlust.
    # Spännvidden runt det gamla hårdkodade 0,85. Nedre änden gör späcket till
    # huvudpoolen, övre nästan renodlar den snabba.
    fast_frac_min: float = 0.50
    fast_frac_max: float = 0.95

    reserve_cap_min: float = 0.5e6
    reserve_cap_max: float = 4.0e6

    repair_capacity_min: float = 0.10
    repair_capacity_max: float = 1.50   # höjt — k_age1 ger lägre inflöde än k_age0=0.2

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
    child_M_min: float = 0.16   # var 0.10 → 0.05 (för litet gav utrotning)
    child_M_max: float = 0.40   # var 0.30 → 0.15 (gav utrotning) → 0.20

    cold_aversion_min: float = 0.0
    cold_aversion_max: float = 1.0

    # Reproduktions-mass-tröskel.
    # Genetiskt bestämd vuxenmassa
    # Brett intervall — evolution hittar r- och K-strateger
    M_target_min: float = 0.20
    M_target_max: float = 4.00

    # Kostpreferens — brett intervall för maximal nischuppdelning
    diet_min: float = 0.0
    diet_max: float = 1.0

    # Predation — brett för att ge evolution chans att hitta rovdjur
    predation_min: float = 0.0
    predation_max: float = 1.0

    M_repro_min_min: float = 0.30   # var 0.20 — lite lägre för att hinna reproducera
    M_repro_min_max: float = 0.90   # var 0.45

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
        reserve_cap=float(_lerp(R.reserve_cap_min, R.reserve_cap_max,
                                _sigmoid(_get_trait(traits, _T_RESERVE)))),
        fast_frac=float(_lerp(R.fast_frac_min, R.fast_frac_max,
                              _sigmoid(_get_trait(traits, _T_FAST_FRAC)))),
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
        breed_phase=float(_lerp(R.breed_offset_min, R.breed_offset_max,
                                _sigmoid(_get_trait(traits, _T_BREED_PHASE)))),
        breed_pull=float(_lerp(R.breed_pull_min, R.breed_pull_max,
                               _sigmoid(_get_trait(traits, _T_BREED_PULL)))),
        breed_sync=float(_lerp(R.breed_sync_min, R.breed_sync_max,
                               _sigmoid(_get_trait(traits, _T_BREED_SYNC)))),
        sense_strength=float(u_sense),
        diet=float(_lerp(R.diet_min, R.diet_max, u_diet)),
        predation=float(_lerp(R.predation_min, R.predation_max, u_predation)),
        hidden_1=int(hidden_1),
        hidden_2=int(hidden_2),
        structure=float(structure_fraction(traits)),
        buoyancy=float(buoyancy_from_structure(structure_fraction(traits))),
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
        "reserve_cap": float(p.reserve_cap),
        "frailty_gain": float(p.frailty_gain),

        "child_E_fast": float(p.child_E_fast),
        "child_E_slow": float(p.child_E_slow),
        "child_Fg": float(p.child_Fg),
        "child_M": float(p.child_M),   # NEW

        "risk_aversion": float(p.risk_aversion),
        "sociability": float(p.sociability),
        "mobility": float(p.mobility),
        "cold_aversion": float(p.cold_aversion),
        "breed_phase": float(p.breed_phase),
        "breed_sync": float(p.breed_sync),
        "breed_pull": float(p.breed_pull),
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


# ---------------------------------------------------------------------------
# Locusaccessorer — normaliserat uttryck i [0, 1]
# ---------------------------------------------------------------------------
# Dessa mappar locus -> normaliserat uttryck. De känner till locuskartan men
# inga skalor. Skalning till fysiska storheter sker i lagret nedanför.

def autotrophy_from_traits(traits: np.ndarray | None, default: float = 0.0) -> float:
    return trait_unit(traits, _T_AUTOTROPHY, default=default)


def growth_from_traits(traits: np.ndarray | None, default: float = 0.0) -> float:
    return trait_unit(traits, _T_GROWTH, default=default)


def adult_mass_from_traits(traits: np.ndarray | None, default: float = 0.0) -> float:
    return trait_unit(traits, _T_ADULT_MASS, default=default)


def temp_opt_from_traits(traits: np.ndarray | None, default: float = 0.0) -> float:
    return trait_unit(traits, _T_TEMP_OPT, default=default)


def temp_width_from_traits(traits: np.ndarray | None, default: float = 0.0) -> float:
    return trait_unit(traits, _T_TEMP_WIDTH, default=default)


def sexual_mode_from_traits(traits: np.ndarray | None, default: float = 0.0) -> float:
    return trait_unit(traits, _T_SEXUAL_MODE, default=default)


def dispersal_from_traits(traits: np.ndarray | None, default: float = 0.0) -> float:
    return trait_unit(traits, _T_DISPERSAL, default=default)


def maturity_from_traits(traits: np.ndarray | None, default: float = 0.3) -> float:
    return trait_unit(traits, _T_MATURITY, default=default)


def root_alloc_from_traits(traits: np.ndarray | None, default: float = 0.5) -> float:
    return trait_unit(traits, _T_ROOT_ALLOC, default=default)


def seed_mass_from_traits(traits: np.ndarray | None, default: float = 0.5) -> float:
    return trait_unit(traits, _T_SEED_MASS, default=default)


def structure_from_traits(traits: np.ndarray | None, default: float = 0.0) -> float:
    return trait_unit(traits, _T_STRUCTURE, default=default)


def uptake_from_traits(traits: np.ndarray | None, default: float = 0.0) -> float:
    return trait_unit(traits, _T_UPTAKE, default=default)


# ---------------------------------------------------------------------------
# Strukturandel — gemensam för alla organismer
# ---------------------------------------------------------------------------

STRUCTURE_MIN = 0.05
STRUCTURE_MAX = 0.85


def structure_fraction(traits: np.ndarray | None) -> float:
    """
    Andel strukturmaterial i vävnaden, i [STRUCTURE_MIN, STRUCTURE_MAX].

    Intervallet är öppet i båda ändar med avsikt. Vid noll vore organismen ren
    reserv utan bärande vävnad; vid ett vore den ren struktur utan energi, och
    då skulle både dess egen katabolism och dess värde som föda vara noll. Båda
    ändarna är degenererade snarare än extrema strategier.
    """
    u = structure_from_traits(traits, default=0.5)
    return STRUCTURE_MIN + (STRUCTURE_MAX - STRUCTURE_MIN) * float(u)


def energy_density(traits: np.ndarray | None, e_labile_per_kg: float) -> float:
    """
    Användbar energi per kilo vävnad.

    Energin sitter i den labila fraktionen; strukturmaterial lagrar ingen
    användbar energi. Därmed följer betningsutbytet av samma tal som organismens
    egen reserv — den som är seg att äta har också mindre att tära på när maten
    tryter.
    """
    return float(e_labile_per_kg) * (1.0 - structure_fraction(traits))


def decay_rate_scale(structure: float) -> float:
    """
    Faktor på nedbrytningstakten som funktion av strukturandel.

    Högstrukturerat material bryts ner långsammare. Skalan är linjär mellan
    full takt vid ren labil vävnad och DECAY_MIN_SCALE vid ren struktur.
    """
    s = min(1.0, max(0.0, float(structure)))
    return DECAY_MIN_SCALE + (1.0 - DECAY_MIN_SCALE) * (1.0 - s)


# Nedbrytningstakt per fraktion, som andel av WorldParams.detritus_decay.
# Labilt material bryts ner i full takt, strukturellt betydligt långsammare.
# Ersätter den tidigare decay_rate_scale(), som approximerade samma sak genom
# att sakta ner hela massan i stället för dess delar — och som därför lät
# strukturandelen skena mot ett, eftersom bara det labila försvann.
DECAY_SCALE_LABILE = 1.00
DECAY_SCALE_STRUCT = 0.15

# Näringsinnehåll per kilo vävnad, per fraktion. Speglar kol-kväve-förhållandet:
# blad ligger kring 30:1, ved kring 500:1. Strukturmaterial är alltså kolrikt och
# näringsfattigt.
#
# Samma tal används i båda riktningarna: det som frisätts när materialet bryts
# ner är det som kostade att bygga det. Därav byggkostnadens inversion — seg
# vävnad är billig i näring men fattig på energi.
NUTRIENT_PER_KG_LABILE = 1.0 / 30.0
NUTRIENT_PER_KG_STRUCT = 1.0 / 500.0


def nutrient_content(structure: float) -> float:
    """Kilo näring per kilo vävnad vid given strukturandel."""
    s = min(1.0, max(0.0, float(structure)))
    return NUTRIENT_PER_KG_LABILE * (1.0 - s) + NUTRIENT_PER_KG_STRUCT * s


# Vävnadens täthet relativt vatten. Ben och kitin är tunga, fett och muskel
# nära neutrala — 1,9 respektive 0,97 är rimliga tal för de två ändarna.
#
# **Tätheten är härledd, inte ett eget locus.** Samma disciplin som florans
# livslängd och som att dess höjd är `m · s`: en egenskap som faller ut ur
# kroppens sammansättning ska inte också gå att mutera oberoende av den.
#
# Och den ger `structure` sin sjätte konsument — på den ände som saknat
# motkraft. En vattenaxel behöver ingen egen nedsida när den ärver en
# avvägning som redan är kalibrerad: seg vävnad ger seghet, lång livslängd,
# låg nedbrytningstakt och billig näring, men gör kroppen tung i vatten.
DENSITY_LABILE = 0.95
DENSITY_STRUCT = 1.35

# Täthetsavvikelse som ger flytförmåga noll.
#
# Talen är satta mot faunans **uppmätta** strukturspann, inte mot dess
# nominella. Uppmätt vid 64x128 efter 2 000 tick: p5 0,211, median 0,567, p95
# 0,684. Över det spannet går tätheten från 1,03 till 1,23 och flytförmågan
# från 0,90 till 0,23 — en levande gradient över hela den fördelning som
# faktiskt finns.
#
# Ett första försök hade ändpunkterna 0,97 och 1,90, alltså ren benmineral i
# den ena änden. Det gav flytförmåga exakt noll för varje strukturandel över
# 0,5, alltså en död halva av axeln och samma klippta tak som mutationsklippet
# en gång skapade. Verkliga ryggradsdjur ligger mellan 0,95 och 1,10 i
# helkroppstäthet; 1,35 för ren stödjevävnad är en sammansättning av ben,
# kitin och kollagen snarare än mineralet ensamt.
DENSITY_DEV_REF = 0.30


def body_density(structure: float) -> float:
    """Kroppens täthet relativt vatten vid given strukturandel."""
    s = min(1.0, max(0.0, float(structure)))
    return DENSITY_LABILE * (1.0 - s) + DENSITY_STRUCT * s


def body_density_array(structure):
    s = np.clip(np.asarray(structure, dtype=np.float64), 0.0, 1.0)
    return DENSITY_LABILE * (1.0 - s) + DENSITY_STRUCT * s


def buoyancy_from_structure(structure):
    """
    Flytförmåga i [0, 1]: ett vid neutral täthet, noll vid full avvikelse.

    `buoyancy` i manifestet är den kapacitet som styr hur en organism förhåller
    sig till vatten. Här får den äntligen en skrivare — härledd ur kroppen — och
    därmed en betydelse som inte kan glida isär från fysiologin.

    Avvikelsen mäts åt båda håll. En kropp som är för lätt är inte bättre än en
    som är för tung: den ligger i ytan och drivs med strömmen i stället för att
    ta sig dit den vill. Att formen är tvåsidig är vad som gör ett inre optimum
    möjligt i stället för en ände att kollapsa mot.
    """
    rho = body_density_array(structure)
    return np.clip(1.0 - np.abs(rho - 1.0) / DENSITY_DEV_REF, 0.0, 1.0)


# --- Kroppens djupmått -----------------------------------------------------
#
# Längdenheten i meter. En cell har arean 100 m² enligt
# `docs/varldens-skala.md`, alltså är längdenheten 10 m och centrumavståndet
# 10,746. Talet har funnits som text i dokumentet och som underförstådd
# omräkning i kommentarer; här får det ett namn, eftersom en storhet utan enhet
# inte är en fysisk storhet.
LENGTH_UNIT_M = 10.0

# Vattnets täthet i kg/m³. `body_density` är relativ mot den.
WATER_DENSITY = 1000.0


def body_depth(mass, structure):
    """
    Kroppens karaktäristiska djup, i längdenheter.

    Ett djur står i vattnet med sin egen kropp som måttstock: `L = (M/ρ)^(1/3)`
    är den linjära skalan hos en kropp med massan M och tätheten ρ. Vid den
    djupleken är kroppen täckt, under den vadar den.

    Det är den enda djupskala i modellen som inte är vald. Den följer ur massa
    och täthet, som båda redan finns, och den skalar rätt: ett åtta gånger
    tyngre djur behöver dubbelt så djupt vatten.

    Uppmätt för modellens median — 2 kg vid strukturandel 0,567, alltså relativ
    täthet 1,177 — blir djupet 0,0119 längdenheter, eller tolv centimeter. Det
    kan jämföras med vad `water_drag_depth_ref` stod på: 0,20, alltså **två
    meter**, vilket är den djuplek där en tvåkilos organism först skulle räknas
    som helt nedsänkt. Skillnaden är sjutton gånger, och den är skälet att
    vattnet aldrig kostade något: fårorna är sjutton centimeter djupa.
    """
    rho = body_density(structure) * WATER_DENSITY
    m = max(1e-9, float(mass))
    return float((m / rho) ** (1.0 / 3.0) / LENGTH_UNIT_M)


DECAY_MIN_SCALE = 0.15

# Matsmältningens verkningsgrad som funktion av substratets strukturandel.
# Ersätter de tidigare kategoriska konstanterna digest_eff_plant och
# digest_eff_carcass: skillnaden mellan växt och kadaver var aldrig två sorters
# mage, utan samma mage mot olika segt material.
#
# Lutningen är tills vidare densamma för alla. I Steg 6 blir den en kapacitet
# med egen underhållskostnad, och då blir specialisering på segt material ett
# verkligt val i stället för en universell nackdel.
DIGEST_EFF_LABILE = 0.80
DIGEST_EFF_STRUCT = 0.45


def digestion_efficiency(structure: float) -> float:
    """
    Hela substratets smältbarhet: `0,80·(1−s) + 0,45·s`.

    Den fullständigare formen, och den som ska ersätta `assimilated_fraction`
    när reserven fått en strukturandel. Läs den där innan den används här — den
    kan inte tas i bruk förrän upptaget får innehålla strukturmaterial utan att
    näringsbalansen driver.
    """
    s = min(1.0, max(0.0, float(structure)))
    return DIGEST_EFF_LABILE - (DIGEST_EFF_LABILE - DIGEST_EFF_STRUCT) * s


# Kostpreferensens verkningsgrad. herb_eff och scav_eff är negativt
# korrelerade — generalisten på 0,5 är sämre på båda än en specialist.
DIET_EFF_EXP = 0.7


def nutrient_content_array(structure):
    """Vektoriserad `nutrient_content`. Samma uttryck, elementvis."""
    import numpy as _np
    s = np.clip(np.asarray(structure, dtype=np.float64), 0.0, 1.0)
    return NUTRIENT_PER_KG_LABILE * (1.0 - s) + NUTRIENT_PER_KG_STRUCT * s


def diet_efficiency(diet: float) -> tuple[float, float]:
    """(herb_eff, scav_eff) för given kostpreferens; 0 = herbivor, 1 = asätare."""
    d = min(1.0, max(0.0, float(diet)))
    return (1.0 - d) ** DIET_EFF_EXP, d ** DIET_EFF_EXP


def assimilated_fraction(structure: float, diet_eff: float) -> float:
    """
    Andel av ingesterad massa som passerar tarmväggen.

    Tre faktorer, alla på massan: strukturmaterialet passerar orört, av det
    labila tas matsmältningens andel upp, och kostpreferensen avgör hur väl
    konsumenten är rustad för just det substratet.

    **Mellanfaktorn var `digestion_efficiency(s)` och straffade segheten två
    gånger.** `(1 − s)` har redan tagit bort strukturmaterialet; att sedan sänka
    verkningsgraden på det som återstår *därför att substratet var segt* är samma
    påstående en gång till. De två sade dessutom oförenliga saker om samma
    material: `(1 − s)` att strukturmaterial ger noll, `digestion_efficiency(s)`
    att rent strukturmaterial ger 0,45.

        s        (1−s)·dig(s)   (1−s)·0,80
        0,000       0,800         0,800
        0,500       0,313         0,400
        0,567       0,260         0,346      <- florans uppmätta median
        0,750       0,134         0,200

    Följden var att en herbivor tillgodogjorde sig **16 procent** av det den åt,
    mot 45–70 procent för ett verkligt betesdjur på gräs. Behovet blev 0,54 kg
    per tick, alltså sjuttiotre procent av kroppsvikten per dygn, och hela
    födobudgeten — aptit, funktionell respons, betesyta — kalibrerades mot det.

    Det som återstår är det labila materialets verkningsgrad, en konstant. Den
    beror inte på hur segt det *omgivande* materialet var.

    **Varför inte hela blandningen.** `digestion_efficiency(s)` är i själva
    verket redan tvåpoolsformen — `0,80·(1−s) + 0,45·s` är exakt "av det labila
    tas 80 procent, av det strukturella 45" — och den vore den fullständigare
    modellen. Den kan inte användas här, och skälet är en premiss någon
    annanstans: **reserven är modellerad som ren labil massa** och bokförs med
    `NUTRIENT_PER_KG_LABILE` per kilo. Så snart upptaget får innehålla
    strukturmaterial stämmer inte det, och näringsbalansen driver — uppmätt
    2,1e-03 relativt mot invariantens 1e-06. Att låta magen komma åt segt
    material kräver att reserven får en strukturandel, och det hör till Steg 6
    där matsmältningen blir en kapacitet med egen underhållskostnad.

    Att kostpreferensen sitter på massan och inte på energin är oförändrat: som
    separat multiplikator på energin försvann skillnaden ur bokföringen — massan
    var varken kropp eller exkrement.
    """
    s = min(1.0, max(0.0, float(structure)))
    d = min(1.0, max(0.0, float(diet_eff)))
    return (1.0 - s) * DIGEST_EFF_LABILE * d


# Initieringsintervall för strukturlocus, i logit-rymdens enhetsskala.
# Växtvävnad är till större delen cellulosa och lignin; djurvävnad har mindre
# bärande material. Skillnaden är en startpunkt, inte en gräns — selektionen
# kan flytta båda.
STRUCTURE_INIT_FLORA = (0.35, 0.95)
STRUCTURE_INIT_FAUNA = (0.05, 0.45)

# Upptagslocus initieras brett, så att selektionen har något att gripa i från
# start i stället för att först behöva vänta ut mutationen.
UPTAKE_INIT = (0.05, 0.95)


# ---------------------------------------------------------------------------
# Floras traituttryck — skalade storheter
# ---------------------------------------------------------------------------
# Manifestet: traitsemantiken ägs av phenotype.py. Skalningsintervallen hör
# därför hit och inte till Population. Locus refereras via _T_*-konstanterna,
# aldrig via råa heltal, så att en ändring i locuskartan slår igenom överallt
# i stället för att gå sönder tyst hos en anropare.


# ---------------------------------------------------------------------------
# Florans omsättning och livslängd — härledda ur strukturandelen
# ---------------------------------------------------------------------------
# Bladekonomispektrumet: tunna billiga blad omsätts fort och ger hög tillväxt,
# seg vävnad omsätts långsamt och lever länge. Samma tal styr båda, vilket är
# vad som gör `structure` till en långsam–snabb-axel i stället för en
# näringsrabatt. Se docs/vaxternas-livscykel.md.

# Omsättningstakt per månad: andel av massan som fälls som förna.
FLORA_TURNOVER_LABILE = 0.083    # ren labil vävnad, uppehållstid ~12 månader
FLORA_TURNOVER_STRUCT = 0.0083   # ren struktur, uppehållstid ~120 månader

# Livslängd i månader, log-linjär i strukturandelen. Med structure i
# [0,05, 0,85] blir spannet omkring 7 månader till 11 år — örter till björkar.
# Ekänden ryms inte i ett fönster på 120 månader och ska inte finnas som en
# parameter som aldrig får verka.
FLORA_LIFESPAN_MIN = 6.0
FLORA_LIFESPAN_MAX = 240.0


def flora_turnover_rate(structure):
    """Förnafall per månad som andel av massan. Skalär eller array."""
    s = np.clip(structure, 0.0, 1.0)
    return FLORA_TURNOVER_LABILE * (1.0 - s) + FLORA_TURNOVER_STRUCT * s


def flora_lifespan(structure):
    """Förväntad livslängd i månader. Skalär eller array."""
    s = np.clip(structure, 0.0, 1.0)
    return FLORA_LIFESPAN_MIN * (FLORA_LIFESPAN_MAX / FLORA_LIFESPAN_MIN) ** s


# ---------------------------------------------------------------------------
# Rörelsens riktningspersistens — härledd ur `mobility`
# ---------------------------------------------------------------------------
# `_T_MOB` bar `pheno.mobility` utan en enda läsare. Den äger nu hur långa
# raksträckor organismen håller **i färdregimen** — kringgående sök på fläcken
# har en gemensam och mycket kortare tid, satt av `AgentParams.dir_tau_local`.
# Axeln är alltså inte "hur rakt djuret rör sig" utan "hur långt det förflyttar
# sig när det har bestämt sig för att förflytta sig".
#
# Log-linjär av samma skäl som florans livslängd: axeln spänner en tiopotens
# och en linjär skala hade lagt merparten av intervallet i den övre änden.
#
# **Avvägningen är tunn i det här läget och det är avsiktligt redovisat.** En
# lång färdsträcka kostar ingenting så länge farten är oberoende av axeln, och
# den enda nedsidan är att en organism som mislläser sin omgivning — percepten
# bär brus — färdas längre bort från en god fläck innan den vänder. Den
# verkliga motkraften är energisk och kommer när farten kopplas till regimen:
# att färdas ska vara snabbt och därmed dyrt, att söka lokalt ska vara
# långsamt och billigt. Se docs/rorelsens-arkitektur.md, Del 1.


def direction_tau(mobility, tau_min: float = 1.5, tau_max: float = 15.0):
    """Färdregimens riktningspersistens i månader. Skalär eller array."""
    u = np.clip(mobility, 0.0, 1.0)
    return float(tau_min) * (float(tau_max) / float(tau_min)) ** u


@dataclass(frozen=True)
class FloraRanges:
    # Andel av inkomsten som går till reproduktion i stället för till kropp.
    # Övre änden är avsiktligt hög: den som lägger nästan allt på frön kan inte
    # betala sitt förnafall, krymper och dör. Semelpari är ytterligheten av
    # axeln, inte ett specialfall.
    repro_alloc_min: float = 0.0
    repro_alloc_max: float = 0.9

    # Vuxenmassa uttrycks som multipel av världens massaskala (WorldParams.B_K)
    # och skalas därför vid anropet, inte här.
    adult_mass_k_min: float = 0.25
    adult_mass_k_max: float = 4.0

    temp_opt_min: float = -5.0
    temp_opt_max: float = 35.0

    temp_width_min: float = 4.0
    temp_width_max: float = 18.0

    uptake_capacity_min: float = 0.25
    uptake_capacity_max: float = 1.0


# ---------------------------------------------------------------------------
# Fröet: storlek, apparat och etablering
# ---------------------------------------------------------------------------
# Propagulmassan spänner fyra tiopotenser och skalas logaritmiskt: verkliga
# kvoter frö mot vuxen spänner sex, och en linjär skala skulle lägga nästan
# hela intervallet i den grova änden.
SEED_MASS_MIN = 1.0e-4
SEED_MASS_MAX = 0.5

# Apparatandel: den del av propagulen som ligger i vinge eller pappus i stället
# för i näringsförrådet. Ökar avståndet, minskar etableringen.
APPARATUS_MAX = 0.5

# Etableringens halvmättnad i kg förråd, i en tom cell.
#
# Formen är inte fri. Fitness går som antal gånger etableringssannolikhet,
# alltså f(m)/m, och en vanlig mättande funktion har inget inre optimum — då
# vinner alltid det minsta fröet. Villkoret för inre optimum är f'(m) = f(m)/m,
# vilket kräver inflexion. Med Hillexponent 2 hamnar optimum exakt på h.
# Smith, C. C. & Fretwell, S. D. 1974. The optimal balance between size and
# number of offspring. The American Naturalist 108: 499–506.
ESTABLISH_HALF = 5.0e-2

# Trängselns effekt på halvmättnaden. Utan den konvergerar fröaxeln mot ett
# enda optimum i stället för att differentiera: stort frö ska vinna där det är
# trångt, många små där det är öppet.
#
# Formen är exponentiell och inte linjär. Det är Beer–Lambert: ljuset som når
# marken avtar exponentiellt med arealindexet ovanför, och förrådet måste
# kompensera för underskottet. Den linjära termen med koefficient 1,0 var
# uppmätt verkningslös — vid den observerade tätheten ΣA = 1,44 etablerade sig
# ett frö med 0,15 kg förråd fortfarande med sextio procents sannolikhet, och
# beståndet fylldes med plantor som ingen näring räckte till.
#
# Talet är kalibrerat, inte härlett: 2,0 ger fri etablering på öppen mark,
# omkring hälften vid halvt sluten och några procent vid sluten.
ESTABLISH_CROWD = 2.0

# Hur många gånger sin egen propagulmassa en planta måste väga för att kunna
# reproducera sig. Ersätter grinden `m >= 0,20 · vuxenmassa`, som var skalfri
# och därför samma fälla som den 70-procentströskel 0056 tog bort: i ett hämmat
# bestånd låg medianplantan på en sjundedel av tröskeln, floran blev steril —
# 160 frön per månad från 344 000 individer — och 944 kg näring låstes i
# reproduktionspooler som aldrig kunde tömmas.
#
# Villkoret är nu absolut i båda leden och biologiskt enkelt: man måste vara
# flera gånger större än sin egen avkomma. Poolgrinden gör det mesta av
# arbetet ändå, eftersom poolen fylls ur inkomsten och inkomsten följer arean.
FLORA_REPRO_MASS_MULT = 3.0

# Skalfaktor för spridningsavståndet i cellbredder. Formen är
# L = L0 · M_vuxen^(1/3) · a^(1/3) / m_frö^(1/6): frigöringshöjd ur vuxenmassan,
# och en apparat vars massa växer brantare än sin area.
#
# Den sista termen är svag, och den är svag i verkligheten också — att stora
# frön faller nära är ett syndrom selektionen byggt, inte en fysisk
# nödvändighet. Korrelationen kodas därför inte in.
DISPERSAL_SCALE = 0.6


# Fröets egen strukturandel. Ett frö är inte gjort av moderns vävnad — det är
# näringsrikt förråd oavsett om föräldern är vedartad. Att räkna kostnaden ur
# moderns struktur gav en vedartad mor 4,5 gångers rabatt på varje frö, vilket
# är precis fel håll: den snabba strategin betalade mest i just den valuta
# reproduktionen drogs ur.
SEED_STRUCTURE = 0.15


# Rotandelens intervall. Ändarna är inte nåbara i praktiken — mutationens clip
# vid ±2,5 i logit-rymden ger 0,076 till 0,924 i normerad skala — men de ska
# ändå vara dödliga, så att optimum är inre och inte en gräns.
ROOT_ALLOC_MIN = 0.02
ROOT_ALLOC_MAX = 0.98


# Mognadströskelns intervall, som andel av vuxenmassan. Övre änden ska rymma en
# strategi som växer färdigt först; nedre en som blommar nästan direkt.
MATURITY_MIN = 0.01
MATURITY_MAX = 0.50


def flora_maturity_frac(traits: np.ndarray | None) -> float:
    """Andel av vuxenmassan vid vilken plantan blir reproduktionsklar."""
    return _lerp(MATURITY_MIN, MATURITY_MAX, maturity_from_traits(traits))


def flora_root_alloc(traits: np.ndarray | None) -> float:
    """Andel av tillväxten som går till rot i stället för skott."""
    return _lerp(ROOT_ALLOC_MIN, ROOT_ALLOC_MAX, root_alloc_from_traits(traits))


def flora_seed_mass(traits: np.ndarray | None) -> float:
    """Propagulmassa i kg, logaritmiskt skalad."""
    u = seed_mass_from_traits(traits)
    lo, hi = np.log(SEED_MASS_MIN), np.log(SEED_MASS_MAX)
    return float(np.exp(lo + (hi - lo) * float(u)))


def flora_apparatus(traits: np.ndarray | None) -> float:
    """Andel av propagulen som är spridningsapparat. Tar över `_T_DISPERSAL`."""
    return float(APPARATUS_MAX * dispersal_from_traits(traits, default=0.5))


def establish_p(provision, crowd):
    """
    Etableringssannolikhet ur fröets förråd. Hillfunktion med exponent 2, så
    att optimum finns och ligger på halvmättnaden. Skalär eller array.
    """
    h = ESTABLISH_HALF * np.exp(ESTABLISH_CROWD * np.maximum(0.0, crowd))
    pr = np.maximum(0.0, provision)
    return (pr * pr) / (pr * pr + h * h)


def dispersal_scale(adult_mass, apparatus, seed_mass):
    """Avståndskärnans skala i cellbredder. Skalär eller array."""
    a = np.maximum(1e-6, apparatus)
    return (DISPERSAL_SCALE
            * np.maximum(1e-9, adult_mass) ** (1.0 / 3.0)
            * a ** (1.0 / 3.0)
            / np.maximum(1e-9, seed_mass) ** (1.0 / 6.0))


def flora_repro_alloc(traits: np.ndarray | None, R: FloraRanges = FloraRanges()) -> float:
    """
    Andel av näringsinkomsten som avsätts till reproduktion.

    Tar över `_T_GROWTH`, som blev ledigt när tillväxten blev inkomstbegränsad.
    Locuset kodade en logistisk takt vars `regen · dt` låg mellan 260 och 2 600
    och därför saturerade mot vuxenmassetaket varje tick — det hade i praktiken
    ingen konsekvens. Genomet växer alltså inte.
    """
    return _lerp(R.repro_alloc_min, R.repro_alloc_max, growth_from_traits(traits, default=0.5))


def flora_adult_mass(
    traits: np.ndarray | None,
    mass_scale: float,
    R: FloraRanges = FloraRanges(),
) -> float:
    """Vuxenmassa i kg. `mass_scale` är världens massaskala, WorldParams.B_K."""
    k = _lerp(R.adult_mass_k_min, R.adult_mass_k_max, adult_mass_from_traits(traits, default=0.5))
    return float(k) * float(mass_scale)


def flora_temp_opt(traits: np.ndarray | None, R: FloraRanges = FloraRanges()) -> float:
    return _lerp(R.temp_opt_min, R.temp_opt_max, temp_opt_from_traits(traits, default=0.5))


def flora_temp_width(traits: np.ndarray | None, R: FloraRanges = FloraRanges()) -> float:
    return _lerp(R.temp_width_min, R.temp_width_max, temp_width_from_traits(traits, default=0.5))


def flora_uptake_capacity(traits: np.ndarray | None, R: FloraRanges = FloraRanges()) -> float:
    """Näringsupptagets effektivitet, ur eget locus."""
    return _lerp(
        R.uptake_capacity_min,
        R.uptake_capacity_max,
        uptake_from_traits(traits, default=0.5),
    )


def flora_repro_capacity(traits: np.ndarray | None) -> float:
    """
    sexual_mode nära 0 ger asexuell flora med hög lokal reproduktionsbenägenhet.
    Skalfritt mått i [0, 1] — ingen range behövs.
    """
    sexual = sexual_mode_from_traits(traits, default=0.0)
    return float(max(0.0, 1.0 - sexual))