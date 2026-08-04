# agent.py
from __future__ import annotations


import math, random
from dataclasses import dataclass, field, replace
from typing import ClassVar, Iterable, Optional, Tuple

import numpy as np

from world import World, clamp
from mlp import MLPGenome
from phenotype import (
    Phenotype,
    derive_pheno,
    phenotype_summary,
    diet_efficiency,
    assimilated_fraction,
    nutrient_content,
    direction_tau,
    NUTRIENT_PER_KG_LABILE,
)

from grid import Grid
from organism_store import next_organism_id


# -------------------------
# Agent ids
# -------------------------
# Agent-ID hämtas ur den gemensamma organism-id-rymden i organism_store.
# Fauna och flora får aldrig ha egna räknare: id ska vara unikt över alla
# organismer, oavsett kapacitetsprofil. Se organism_store.next_organism_id().


def _new_agent_id() -> int:
    return next_organism_id()


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
    v_max: float = 100.0

    # --- Styrningens kinematik -------------------------------------------
    #
    # `turn_rate` var 300 rad/månad, vilket vid dt = 0,02 ger 6,0 rad per tick
    # mot ett helt varv på 6,28. Riktningen kunde alltså slumpas om helt på ett
    # tick, och gjorde det: uppmätt rakhet — nettoförflyttning genom bansträcka
    # över livet — var 0,069, alltså 1 563 cellbredders bana för 85 cellbredders
    # förflyttning. Se docs/rorelsens-arkitektur.md, Del 1.
    #
    # Taket på vridhastigheten, i rad per månad. Vid 25 vänder en organism 180°
    # på ungefär åtta tick i stället för på ett.
    turn_rate_max: float = 25.0

    # Relaxationstakt mot önskad riktning, 1/månad. Styrningen är analytisk
    # relaxation och inte proportionell förstärkning per tick: den gamla formen
    # hade förstärkningen 1,54 per tick, alltså översläng med teckenbyte varje
    # tick. Formen nedan kan inte slå över vid något dt.
    turn_gain: float = 6.0

    # Lateral acceleration i cellbredder per månad². Svängradien följer av
    # centripetalvillkoret: ω ≤ a_lat / v, alltså r = v²/a_lat. Fart köper
    # räckvidd och kostar manöverförmåga, vilket är mekanism och inte
    # kostnadsparameter. Talet är satt så att svängradien vid uppmätt marschfart
    # (37 cellbredder per månad) blir omkring 1,5 cellbredder — en organism ska
    # kunna vända inom sitt eget synfält, annars går födostyrningen sönder.
    lat_accel_max: float = 900.0

    # --- Riktningens persistens: områdessökning mot färd -------------------
    #
    # Rotationsdiffusionen uttrycks som en persistenstid och inte som en
    # brusamplitud, och bruset skalar med √dt så att diffusionen blir
    # tidsstegsinvariant. Den gamla formen skalade med dt och halverades därför
    # när tidssteget halverades.
    #
    # Persistensen är **tillståndsberoende**, inte konstant. Kringgående sök i
    # ett område är rätt beteende när födan är riklig — det finns ingen
    # anledning att färdas då, och den slingrande banan håller organismen kvar
    # på fläcken. Rak färd är rätt när det finns ett incitament att ta sig
    # någon annanstans. Det är områdesbegränsad sökning, och den är en av de
    # bäst belagda rörelsemönstren i naturen.
    #
    # `explore_drive` byter därmed roll: den höjde tidigare bruset, alltså gav
    # mer utforskning *mindre* effektiv förflyttning. Nu väljer den regim.
    dir_tau_local: float = 0.35   # månader; kringgående sök på fläcken
    dir_tau_min: float = 1.5      # månader; färdregim vid mobility = 0
    dir_tau_max: float = 15.0     # månader; färdregim vid mobility = 1

    # ------------------------
    # Sensing / perception
    # ------------------------
    n_rays: int = 12
    ray_len_front: float = 7.0    # max räckvidd framåt (ellipsens långa halva)
    # Artfrändesynens form. Sedan 0084 läser världskanalerna sitt grannskap
    # isotropt via sektoraggregat; ellipsen styr numera bara var en artfrände
    # kan upptäckas.
    #
    # Vid 0,7 gav r(θ) = r_front(1−e)/(1−e·cos θ) tio enheter rakt fram, tre
    # åt sidan och 1,8 rakt bak. En flockkamrat som färdas jämsides på fyra
    # enheters avstånd var alltså osynlig — och det är just där en flockkamrat
    # befinner sig. Visuellt syntes det som att djur upptäckte varandra
    # frontalt, gjorde en ömsesidig riktningsändring och sedan fortsatte i
    # tangentens riktning, eftersom motparten försvann ur perceptionen i samma
    # ögonblick som de passerade.
    #
    # Vid 0,3 blir det tio fram, sju åt sidan och 5,4 bak. Framåtriktningen är
    # kvar men sidosynen räcker för att hålla sällskap. Samma ändring bör också
    # höja parningsfrekvensen, som legat på sex till sju procent av alla
    # tillfällen då en partner setts.
    # Kvar bara som historik: synfältet är cirkulärt och delat mellan aggregat
    # och identitet sedan 0101, och ingen kod läser detta värde.
    ray_eccentricity: float = 0.0  # oanvänd sedan 0101
    # Hur många tick en sedd artfrände minns efter att den lämnat synfältet.
    # 0 = av. Sensingen körs var tionde tick i vila, och djuret förflyttar sig
    # 7,4 enheter däremellan — nästan hela synfältets längd. Utan minne finns
    # grannen bara de tick då den råkar synas, och alignment hinner ett enda
    # samplingstillfälle per möte. Minnet dödräknar grannens position framåt
    # längs dess senast sedda kurs, så att den inte tappas mellan sensingar
    # eller när den glider åt sidan. Styr aldrig parning eller predation —
    # bara styrningen.
    social_memory_ticks: int = 12
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
    eat_rate: float = 90.0  # kg/s

    # Energy densities (J/kg)
    # E_bio_J_per_kg behålls som bakåtkompatibelt alias för växtbiomassa.
    E_bio_J_per_kg: float = 4.0e6
    # Energi per kilo labil vävnad, gemensam för allt organiskt material.
    # Materialets strukturandel avgör hur stor den labila fraktionen är; se
    # docs/substratets-struktur.md.
    E_labile_J_per_kg: float = 9.302e6
    E_body_J_per_kg: float = 7.0e6

    # Conversion efficiencies
    # Matsmältningens verkningsgrad härleds ur substratets strukturandel via
    # phenotype.digestion_efficiency(). De tidigare skilda konstanterna för
    # växt och kadaver kodade som typskillnad det som är en materialegenskap.
    anabolism_eff: float = 0.70
    catabolism_eff: float = 0.90

    # Energy storage capacity (J/kg)
    E_cap_per_M: float = 1.4e6

    # ------------------------
    # Energy ledger diagnostics
    # ------------------------
    ledger_eps_abs: float = 1e-8   # NEW: absolute drift tolerance
    ledger_eps_rel: float = 1e-12  # NEW: relative drift tolerance
    assert_ledger: bool = False    # NEW: raise if drift exceeds tolerances    

    # ------------------------
    # Initial physiological state
    # ------------------------
    M0: float = 2.0   # initial body mass
    E0: float = 1.4e6 # initial energy (J)

    # ------------------------
    # Basal metabolism (allometric)
    # ------------------------
    k_basal: float = 9.0e6  # [W]

    # Activity-related metabolic costs (non-locomotor)
    compute_cost: float = 1.5e6  # [W/kg] — skalas upp proportionellt mot n_params vid agentens start

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
    thermo_k_W_per_C: float = 3.0e5       # heat loss coefficient ~ M^(2/3)
    thermo_mass_exp: float = 2.0 / 3.0
    thermo_Pmax_per_kg: float = 5.0e7    # max heat generation (W/kg)
    cold_damage_gain: float = 0.03      # damage per second at severe cold

    # ------------------------
    # Locomotion mechanics
    # ------------------------
    F0: float = 5.0e4
    force_mass_exp: float = 2.0 / 3.0
    drag_lin: float = 220.0
    drag_quad: float = 1.2
    locomotion_eff: float = 0.25

    # ------------------------
    # Starvation / weakness dynamics
    # ------------------------
    M_crit: float = 0.50   # under detta försvagas rörelseförmågan
    M_min: float = 0.14    # absolut minimum — lite under child_M_min för buffert
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
    repair_gain: float = 50.0
    repair_E_per_D: float = 2.0e6  # var 0.02 → sänkt: billigare reparation gör det evolutionärt lönsamt
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
    # Slitagets åldersterm — den enda posten som tickar för ett oskadat djur.
    #
    # Var 0.008, ett tal från sekundskalan (kommentarerna ovan räknar i "t=100s"
    # och "t=350s"). Uppmätt konsekvens: medianskadan i en matt population är
    # exakt noll vid varje mätpunkt, och 97 procent av alla dödsfall är svält.
    # Ett djur med repair_capacity 0,80 dog aldrig av ålder över 600 månader.
    #
    # Slitaget matar både reparationstaket, exp(-0,15·W), och verkningsgraden,
    # exp(-0,10·W). Ett djur som aldrig skadas slits därför aldrig, och ett som
    # aldrig slits reparerar för evigt — åldrandet fanns som inflöde men aldrig
    # som ackumulation. Ålderstermen är det som bryter den loopen.
    #
    # 0,12 är härlett, inte fittat: det placerar åldersdöden på 121, 134 och
    # 153 månader vid repair_capacity 0,15, 0,30 och 0,80. Ingen är odödlig,
    # och axeln köper uppskjuten död i stället för ingenting — vilket är
    # anledningen till att den selekterades nedåt, 0,554 till 0,312, i den
    # uppmätta körningen.
    # --- Betningens funktionella respons -----------------------------------
    #
    # Betaren tog tidigare exakt sin efterfrågan ur cellerna inom räckhåll,
    # utan söktid och utan hanteringstid: `take = min(m, amt)`. Den sista
    # plantan i grannskapet var därmed lika lätt att hitta och äta som den
    # första, och intaget föll inte förrän den lokala biomassan understeg
    # aptiten — då tvärt till noll. Bytet hade ingen lågtäthetsrefug.
    #
    # Hollings typ II följer av att betandet tar tid. Under en tick fördelas
    # tiden mellan att söka och att beta:
    #
    #     intag = dt · a · B / (1 + a · h · B)
    #
    # där B är tillgänglig växtmassa inom räckhåll, `a` är sökeffektiviteten
    # och `h` hanteringstiden per kilo. Vid stor B går intaget mot dt/h, alltså
    # samma tak som förut; vid liten B faller det proportionellt mot B i
    # stället för att ligga kvar på aptiten.
    #
    # h är satt så att taket blir 1,8 kg per tick, vilket är den uppmätta
    # maximala aptiten: h = dt / 1,8. `a` är satt så att halvmättnaden ligger
    # vid 20 kg inom räckhåll — mellan mättnadslägets omkring 76 kg och
    # kollapslägets omkring 6, så att responsen biter under en nedgång utan att
    # märkas under normala förhållanden.
    graze_handle_h: float = 0.0111   # månader per kg
    graze_search_a: float = 4.5      # 1 / (h · B_halv)

    wear_a0: float = 0.12
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
    attack_cost_per_s: float = 5.0  # fraktion av predatorns Ecap per sekund som attacken kostar

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
    growth_rate_per_s: float = 0.19
    growth_R_min: float = 0.30   # ingen aktiv tillväxt under denna reservgrad
    growth_R_full: float = 0.60  # full tillväxthastighet först här

    # Gestationstillväxthastighet (kg/s fetal vävnad per sekund).
    # 0.004 kg/s → 50s för ett 0.2 kg foster (var 0.002 = 100s).
    # Föräldern kataboliserar ~0.2 kg kroppsmassa under gestationen (M: 1.0→0.8). ✓
    gestation_growth_kg_per_s: float = 0.085

    # Energikostnad för att bygga fetal vävnad (J/kg).
    # OBS: Detta är INTE samma som E_body_J_per_kg (energidensitet vid katabolism).
    # Anabolisk byggkostnad ≈ metaboliskt arbete för syntes, ej lagrad energi i vävnaden.
    # Kalibrering: 0.002 kg/s × 10 000 J/kg = 20 J/s ≈ 2/3 av basalmetabolismen.
    # Det gör gestation energimässigt rimlig utan att tömma föräldern på sekunder.
    gestation_E_per_kg: float = 10_000.0  # J/kg (var implicit E_body_J_per_kg = 7 000 000)
    growth_E_per_kg:    float = 10_000.0  # J/kg somatisk tillväxt (samma princip som gestation)
    # Skada per relativ massförlust via katabolism. Var 1,0, vilket innebar
    # att varje mobilisering av egen vävnad kostade skada i proportion till
    # sin storlek — att bränna en procent av kroppen tog en procent av hela
    # skadebudgeten, oåterkalleligt.
    #
    # Det är dubbelräkning. Utmärgling modelleras redan av dD_starve, som
    # utgår från massan relativt förväntad massa för åldern och alltså mäter
    # *utfallet*. k_cat_dmg straffade dessutom *mekanismen*. En organism som
    # tär på reserven under en svacka och äter upp sig igen är inte skadad,
    # och uppmätt kataboliserade agenterna varje tick trots positiv
    # energibalans — vilket ensamt räckte för att nå D_max på fem månader.
    k_cat_dmg:          float = 0.05

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
    # Reservmassa i kilo, uppdelad i en snabbt och en långsamt tillgänglig pool.
    # Tidigare låg de i viktade energienheter, vilket gjorde att energi kunde
    # skapas och förstöras utan att någon massa rörde sig. Med massa som enhet
    # följer näringen med, och energin härleds ur E_labile.
    M_fast: float = 0.0
    M_slow: float = 0.0

    # Utflöden till cellen, ackumulerade sedan senaste tömning av body-passet.
    out_excreta_kg: float = 0.0          # massa till detritus
    out_excreta_struct_kg: float = 0.0   # dess massviktade strukturandel, i kg
    out_nutrient_kg: float = 0.0         # fri näring till cellen

    # Sant om organismen kataboliserade egen vävnad under senaste steget.
    _catabolized_last_step: bool = False
    # Förväntad massa för åldern, cachad i step() så hunger() kan läsa den.
    _M_expected: float = 0.0
    # Reservkapacitet per kilo, cachad från fenotypen så E_cap() kan läsa den.
    _reserve_cap: float = 0.0

    # structural state
    M: float = 0.0        # body mass
    Fg: float = 0.15      # fatigue level
    Tb: float = 37.0      # Target body temperature

    alive: bool = True

    # energy ledger diagnostics (per-individual)
    last_ledger: dict | None = None
    last_flux: dict | None = None

    # Skadeinflödets termer från senaste steget, för aggregering i pop-loggen.
    last_damage_terms: dict | None = None
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
    # Varför individen dog. Sätts vid varje dödsväg och läses av death_record.
    # Utan den går de fem vägarna inte att skilja åt i efterhand, och vi har
    # gissat fel på dödsorsak tre gånger i rad.
    death_cause: str = ""
    gest_M_target: float = 0.0   # target fetal mass
    
    def M_reserve(self) -> float:
        """Total mobiliserbar reservmassa i kilo."""
        return float(self.M_fast) + float(self.M_slow)

    def M_body(self) -> float:
        """
        Hela kroppens massa: strukturell och funktionell vävnad plus reserv.

        `M` bär den committade vävnaden och reservpoolerna det mobiliserbara.
        Den bevarade storheten är summan — det är den som kadavret bär och som
        näringsbokföringen räknar på.
        """
        return float(self.M) + self.M_reserve()

    # --- utflöden till världen, ackumulerade per tick ---------------------
    # Body har ingen världsreferens och ska inte få en. Den bokför i stället
    # vad som lämnat kroppen, och body-passet tömmer posterna till rätt cell.

    def _void(self, kg: float, structure: float) -> None:
        """Materia som lämnar kroppen som exkrement, alltså till detritus."""
        m = float(kg)
        if m <= 0.0:
            return
        self.out_excreta_kg += m
        self.out_excreta_struct_kg += m * min(1.0, max(0.0, float(structure)))

    def _burn(self, kg: float) -> None:
        """
        Labil massa som oxideras för energi.

        Kolet går till atmosfären och lämnar modellen — massa är aldrig en
        sluten storhet här. Näringen i den brända massan är däremot kvar och
        utsöndras till cellen som kvävehaltigt avfall, direkt växttillgängligt.
        """
        m = float(kg)
        if m <= 0.0:
            return
        self.out_nutrient_kg += m * NUTRIENT_PER_KG_LABILE

    def _release_nutrient(self, kg: float) -> None:
        """Fri näring rakt ut, utan att någon massa lämnar kroppen."""
        n = float(kg)
        if n > 0.0:
            self.out_nutrient_kg += n

    def _take_reserve_mass(self, kg: float) -> float:
        """
        Ta reservmassa proportionellt ur båda poolerna. Returnerar uttaget.

        Ingen näringsbokföring här — anroparen avgör om massan brändes,
        byggdes in i vävnad eller överfördes till en avkomma.
        """
        want = float(kg)
        if want <= 0.0:
            return 0.0
        Mr = self.M_reserve()
        if Mr <= 1e-15:
            return 0.0
        take = want if want < Mr else Mr
        d_fast = take * (float(self.M_fast) / Mr)
        self.M_fast = max(0.0, float(self.M_fast) - d_fast)
        self.M_slow = max(0.0, float(self.M_slow) - (take - d_fast))
        return float(take)

    def _catabolize(self, dM_cat: float, structure: float) -> float:
        """
        Mobilisera `dM_cat` kg vävnad till reserven. Returnerar frigjord energi.

        Bara den labila fraktionen kan mobiliseras; strukturmaterialet passerar
        ut som exkrement tillsammans med katabolismens förlust. Vid faunans
        typiska strukturandel 0,25 ger det samma energiutbyte som den tidigare
        konstanten E_body_J_per_kg — vilket inte är en slump, eftersom
        7,0e6 ≈ 9,302e6 · (1 − 0,25). Skillnaden är att utbytet nu följer
        individens egen sammansättning i stället för en global konstant.
        """
        dM = float(dM_cat)
        if dM <= 0.0:
            return 0.0
        s = min(1.0, max(0.0, float(structure)))
        cat_eff = max(0.0, float(getattr(self.AP, "catabolism_eff", 1.0)))

        lab = dM * (1.0 - s) * cat_eff
        self.M = float(self.M) - dM
        self.M_fast = float(self.M_fast) + lab

        rest = max(0.0, dM - lab)
        if rest > 1e-18:
            self._void(rest, min(1.0, dM * s / rest))

        return float(lab * float(self.AP.E_labile_J_per_kg))

    def E_total(self) -> float:
        """Tillgänglig energi, härledd ur reservmassan."""
        return self.M_reserve() * float(self.AP.E_labile_J_per_kg)

    def E_cap(self) -> float:
        """
        Reservens tak i joule.

        Kapaciteten per kilo är en genetisk axel, cachad från fenotypen i
        step(). Ett fast tak gjorde upplagring omöjlig: en vuxen individ vid
        M_target kunde varken växa eller lagra, så allt överskott
        exkreterades medan varje svacka kostade massa — ett tak uppåt utan
        golv nedåt, vilket ger en långsam nedgång oavsett hur god
        försörjningen är.

        Med säsonger som verkar över fyra vintrar per liv är förmågan att
        lägga på hull dessutom det som avgör om vintern går att överleva.
        """
        M = self.M
        cap = float(getattr(self, "_reserve_cap", 0.0)) or float(self.AP.E_cap_per_M)
        return cap * (M if M > 1e-9 else 1e-9)

    def hunger(self) -> float:
        """
        Aptit, 0 till 1.

        Reservmåttet ensamt duger inte, av två skäl.

        Reserven läses i födosöket, alltså efter förra tickens intag och före
        den här tickens dräneringar, medan den bara rymmer några få ticks
        underhåll. En organism kunde därför avläsa en halvfull reserv, äta på
        halv kapacitet, och ändå stå på noll när ticken var slut. Katabolism är
        ett entydigt underskottsbesked: den som bryter ner sin egen vävnad för
        att betala underhållet är hungrig oavsett ögonblicksbilden.

        Och Ecap = reserve_cap · M, så när organismen magrar krymper både
        reserven och taket och kvoten står nästan still. Uppmätt rapporterade
        ett djur på väg mot M_min hunger 0,33 — lägre än ett välmatat djur
        borde ligga över. Aptiten måste därför också mätas mot vad organismen
        *borde* väga, inte bara mot vad den råkar rymma.
        """
        if bool(getattr(self, "_catabolized_last_step", False)):
            return 1.0

        Et   = self.E_total()
        Ecap = self.E_cap()
        h    = (Ecap - Et) / (Ecap if Ecap > 1e-9 else 1e-9)
        h    = 0.0 if h < 0.0 else 1.0 if h > 1.0 else h

        M_exp = float(getattr(self, "_M_expected", 0.0))
        if M_exp > 1e-9:
            span = max(1e-9, M_exp - float(self.AP.M_min))
            hm = (M_exp - float(self.M)) / span
            hm = 0.0 if hm < 0.0 else 1.0 if hm > 1.0 else hm
            if hm > h:
                h = hm
        return h

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
        # Reparationen saknade dt och var därmed per *tick* medan skadan är
        # per tidsenhet. Modellens beteende berodde på tidsstegets storlek:
        # halverad dt halverade skadan men lämnade reparationen orörd.
        # Med dt är R_max och repair_gain takter, som allt annat.
        R_des = max(0.0, float(AP.repair_gain) * float(self.P))
        R_des = min(R_des, R_max) * dt

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

    def clamp_energy_to_cap(self) -> None:
        Et = float(self.E_total())
        Ecap = float(self.E_cap())
        if Et <= Ecap:
            return
        k = Ecap / max(Et, 1e-12)
        self.scale_energy(k)

    def scale_energy(self, factor: float) -> None:
        f = max(0.0, float(factor))
        self.M_fast = float(self.M_fast) * f
        self.M_slow = float(self.M_slow) * f

    def take_energy(self, amount: float, *, burn: bool = True) -> float:
        """
        Ta ut energi ur reserven. Returnerar faktiskt uttag i joule.

        Uttaget fördelas proportionellt mot poolernas massa. Skillnaden mellan
        snabb och långsam ligger tills vidare bara i insättningskvoten; att
        göra den till en åtkomsttakt är en egen ändring.

        `burn=True` betyder att massan oxideras och att dess näring utsöndras.
        `burn=False` används när massan i stället överförs någon annanstans —
        till en avkomma — och alltså inte lämnar systemet.
        """
        amt = float(max(0.0, amount))
        if amt <= 0.0:
            return 0.0

        e_lab = float(self.AP.E_labile_J_per_kg)
        take_kg = self._take_reserve_mass(amt / e_lab)
        if take_kg <= 0.0:
            return 0.0

        if burn:
            self._burn(take_kg)

        return float(take_kg * e_lab)
        
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

    def reserve_frac(self) -> float:
        Ecap = self.E_cap()
        if Ecap <= 1e-9:
            return 0.0
        r = self.E_total() / Ecap
        return 0.0 if r < 0.0 else 1.0 if r > 1.0 else r

    def _finite(self, x: float) -> bool:
        # Behålls för bakåtkompatibilitet men används ej internt längre.
        return math.isfinite(float(x))

    def _guard_snapshot(self, where: str) -> dict:
        return {
            "where": where,
            "M_fast": float(self.M_fast),
            "M_slow": float(self.M_slow),
            "M": float(self.M),
            "D": float(self.D),
            "Fg": float(self.Fg),
            "alive": bool(self.alive),
        }
        
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
        food_bio_J: float,
        food_carcass_J: float,
        assim_bio_kg: float = 0.0,
        assim_carcass_kg: float = 0.0,
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
            math.isfinite(self.M_fast)
            and math.isfinite(self.M_slow)
            and math.isfinite(self.M)
            and math.isfinite(self.D)
            and math.isfinite(self.Fg)
        ):
            self.guard_steps += 1
            self.guard_killed += 1
            self.guard_last = self._guard_snapshot("pre_state")
            self.death_cause = "guard_pre"
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
            self.death_cause = "guard_energy"
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
        _E_labile     = float(AP.E_labile_J_per_kg)
        _E_body       = float(AP.E_body_J_per_kg)
        _ana_eff      = max(1e-9, float(getattr(AP, 'anabolism_eff', 1.0)))
        _cat_eff      = max(0.0, float(getattr(AP, 'catabolism_eff', 1.0)))
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
        # Egen strukturandel: styr katabolismens utbyte, exkrementets
        # sammansättning och hur mycket näring vävnaden binder per kilo.
        _structure    = min(1.0, max(0.0, float(getattr(pheno, "structure", 0.25))))
        _nut_tissue   = nutrient_content(_structure)
        _build_E_kg        = _E_body / _ana_eff
        _gest_build_E_kg   = float(getattr(AP, 'gestation_E_per_kg', 10_000.0))
        _growth_build_E_kg = float(getattr(AP, 'growth_E_per_kg',    10_000.0))
    
        # Reservmassa som lämnat poolerna utan att brännas: inbyggd i vävnad
        # eller överförd till foster. Behövs för att energiledgern ska sluta.
        E_material = 0.0
        E_overflow = 0.0
        self._catabolized_last_step = False

        # ---------------------------------------------------------
        # (1) Intake -> reserv (massa), överskott exkreteras i (5)
        # ---------------------------------------------------------
        m_bio = max(0.0, float(food_bio_kg))
        m_car = max(0.0, float(food_carcass_kg))

        E_raw_bio = max(0.0, float(food_bio_J))
        E_raw_car = max(0.0, float(food_carcass_J))

        # Assimilationen är redan avgjord i _perform_feeding, som också har
        # exkreterat resten till cellen. Här kommer bara den massa som faktiskt
        # passerat tarmväggen. Den är labil per konstruktion — strukturmaterial
        # assimileras inte — så dess energi är massan gånger E_labile.
        m_assim_bio = max(0.0, float(assim_bio_kg))
        m_assim_car = max(0.0, float(assim_carcass_kg))
        m_assim = m_assim_bio + m_assim_car

        E_in_bio = m_assim_bio * _E_labile
        E_in_car = m_assim_car * _E_labile
        E_in = E_in_bio + E_in_car

        # Förlusterna är nu massa som ligger i detrituspoolen, inte energi som
        # försvann. Termerna behålls som diagnostik.
        E_in_gross_bio = E_raw_bio
        E_in_gross_car = E_raw_car
        E_loss_digest_bio = max(0.0, E_raw_bio - E_in_bio)
        E_loss_digest_car = max(0.0, E_raw_car - E_in_car)

        # Massan går in i reserven. Taket mot E_cap tillämpas inte här utan i
        # sektion (5), där överskottet exkreteras i stället för att raderas —
        # att klippa intaget mot taket lät massa försvinna ur bokföringen.
        if m_assim > 0.0:
            self.M_fast += 0.85 * m_assim
            self.M_slow += 0.15 * m_assim

        dE_store = E_in
        E_to_M = 0.0

        # ---------------------------------------------------------
        # (2) Effective mass (carried load) + basal/compute/sense/loco
        # ---------------------------------------------------------
        # Reserven är massa organismen faktiskt bär, och ska belasta basal,
        # rörelse och värmeförlust som all annan massa. Att utesluta den
        # gjorde fett gratis — och utan kostnad har en evolverbar
        # reservkapacitet ingen avvägning att selekteras på.
        self._reserve_cap = float(getattr(pheno, "reserve_cap", 0.0))
        M_carry = float(self.M) + self.M_reserve()
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

        # Analytisk relaxation i stället för explicit Euler. Den termiska
        # tidskonstanten Cth/K är kortare än ett tick — vid månadsskalan
        # ungefär en halv tick — och explicit integrering divergerar då:
        # kroppstemperaturen sköt iväg till hundratals minusgrader. Samma
        # klass av fel som den newtonska rörelsedynamiken, där relaxationen
        # mot terminalhastighet också är mycket kortare än tidssteget.
        #
        # Lösningen på dT/dt = (P_gen - K·(T - Tenv))/Cth är exakt för
        # konstant P_gen över steget och ovillkorligt stabil.
        T_inf = Tenv + (P_gen / K if K > 1e-30 else 0.0)
        decay = math.exp(-K * dt / Cth) if Cth > 1e-30 else 0.0
        self.Tb = T_inf + (Tb - T_inf) * decay

        # Ledgern ska bära den värme som faktiskt avgavs, inte den som
        # begärdes; vid jämvikt är de lika.
        Qloss = K * (0.5 * (Tb + float(self.Tb)) - Tenv)
    
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
                    # Fostervävnad byggs av moderns reservmassa, ett kilo per
                    # kilo. Byggkostnaden är syntesarbetet ovanpå materialet,
                    # inte i stället för det — de är två termer.
                    kg_per_kg = 1.0 + (_gest_build_E_kg / _E_labile)
                    need_kg = dM_want * kg_per_kg
                    have_kg = self.M_reserve()

                    if have_kg < need_kg:
                        # Katabolisera egen vävnad för att fylla på reserven.
                        yield_kg = max(1e-12, (1.0 - _structure) * _cat_eff)
                        free = max(0.0, float(self.M) - _M_min)
                        dM_cat_gest = min((need_kg - have_kg) / yield_kg, free)
                        if dM_cat_gest > 0.0:
                            E_from_M_gest = self._catabolize(dM_cat_gest, _structure)
                        have_kg = self.M_reserve()

                    build_kg = min(dM_want, have_kg / kg_per_kg)
                    if build_kg > 0.0:
                        out_gest_build = float(self.take_energy(build_kg * _gest_build_E_kg))
                        # Fostret är ännu odifferentierad, labil vävnad; dess
                        # struktur läggs på först vid födseln.
                        dM_gest = self._take_reserve_mass(build_kg)
                        E_material += dM_gest * _E_labile
                        if dM_gest > 0.0:
                            self.gest_M = M_cur + dM_gest
                            self.gest_E_J = float(self.gest_E_J) + out_gest_build
    
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
        # Tillväxten flyttad till sektion (3B), efter att de obligatoriska
        # dräneringarna och katabolismen är avklarade. Låg den kvar här blev
        # den en obligatorisk post, och när reserven inte räckte täckte
        # katabolismen även tillväxten — organismen bröt ner sin egen kropp
        # för att bygga sin egen kropp. Materialet kom tillbaka som massa, så
        # nettot syntes knappt, men varje varv kostade katabolismens utbyte
        # och lade på skada. Uppmätt kataboliserade en agent varje tick trots
        # fyrtio procents energiöverskott.
        out_growth = 0.0
        dM_growth = 0.0
        growth_gate = 0.0

        # ---------------------------------------------------------
        # (2D) Pay drains ONCE
        # ---------------------------------------------------------
        # OBS: out_gest_build ingår INTE — redan betald i sektion (2C).
        E_out_drain = (
            out_basal + out_compute + out_sense + out_loco + out_thermo
            + out_gest_overhead
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
            # Utbytet följer individens egen strukturandel i stället för den
            # globala konstanten E_body_J_per_kg. Vid s = 0,25 är de samma tal.
            yield_J_per_kg = max(1e-12, (1.0 - _structure) * _cat_eff * _E_labile)
            want_cat = deficit / yield_J_per_kg
            free     = max(0.0, float(self.M) - _M_min)
            dM_cat   = min(want_cat, free)

            if dM_cat > 0.0:
                self._catabolized_last_step = True
                E_from_M = self._catabolize(dM_cat, _structure)
                _k_cat_dmg = float(getattr(AP, 'k_cat_dmg', 1.0))
                dD_cat = _k_cat_dmg * dM_cat / max(float(self.M), 1e-9)
                self.D = clamp(float(self.D) + dD_cat, 0.0, _D_max)

            paid2      = float(self.take_energy(deficit))
            deficit    = max(0.0, deficit - paid2)
            E_paid_drain += paid2

        # ---------------------------------------------------------
        # (3B) Somatisk tillväxt — ur det som faktiskt återstår
        # ---------------------------------------------------------
        # Tillväxt är diskretionär. Den får bara ta av reserven efter att
        # underhållet är betalt, och kan därför aldrig skapa det underskott
        # som katabolismen sedan täcker. En svältande organism växer inte;
        # den överlever, och växer när det finns något över.
        #
        # Reservgraden läses här, efter dräneringarna, så grinden bedömer det
        # som finns kvar i stället för det som var på väg att spenderas.
        if deficit <= 0.0 and float(self.M) < _M_target:
            r_now = self.reserve_frac()
            gR0 = float(getattr(AP, 'growth_R_min', 0.30))
            gR1 = max(gR0 + 1e-9, float(getattr(AP, 'growth_R_full', 0.60)))
            if r_now <= gR0:
                growth_gate = 0.0
            elif r_now >= gR1:
                growth_gate = 1.0
            else:
                growth_gate = (r_now - gR0) / (gR1 - gR0)

            if growth_gate > 0.0:
                # Somatisk vävnad byggs av reservmassa, ett kilo per kilo, med
                # syntesarbetet som tilläggskostnad. Materialet stryper, inte
                # byggkostnaden.
                _kg_per_kg_growth = 1.0 + (_growth_build_E_kg / _E_labile)
                dM_want = min(_growth_rate * dt * growth_gate,
                              _M_target - float(self.M))
                dM_want = min(dM_want, self.M_reserve() / _kg_per_kg_growth)

                if dM_want > 0.0:
                    out_growth = float(self.take_energy(dM_want * _growth_build_E_kg))
                    mat = self._take_reserve_mass(dM_want)
                    if mat > 0.0:
                        self.M = float(self.M) + mat
                        dM_growth = mat
                        E_material += mat * _E_labile
                        # Reserven är labil och näringsrik; vävnaden binder
                        # mindre per kilo eftersom en del är strukturmaterial.
                        # Mellanskillnaden utsöndras som kvävehaltigt avfall.
                        self._release_nutrient(
                            mat * (NUTRIENT_PER_KG_LABILE - _nut_tissue)
                        )

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

        # Normera dräneringen mot basalmetabolismen, inte mot reservens
        # storlek. Ecap är en stock och drain_rate en takt; kvoten mellan dem
        # har enheten 1/tid och ändrar värde när tidsenheten byts. Mot basalen
        # blir kvoten dimensionslös och skalfri.
        drain_rate   = E_out_drain / max(dt, 1e-12)
        basal_rate   = max(1e-30, _k_basal * (M_eff ** 0.75))
        drain_rate_n = drain_rate / basal_rate

        dD_eff = dt * (_k_damage * susc * (1.0 + 1.2 * e_lack) * frail * effort * starve_stress)
        dD_met = dt * (float(pheno.stress_per_drain) * drain_rate_n * starve_stress)

        age_rate = max(0.0, _k_age0 + _k_age1 * float(age_s))
        dD_age   = dt * age_rate * (1.0 + _k_ageD * d_norm) * (1.0 + frailty_gain)

        # Svältskada: individens massa relativt förväntad massa för åldern.
        # M_expected är linjär från child_M (age=0) till M_target (age=A_mature),
        # sedan konstant. En agent under kurvan har inte kunnat växa i takt — svälter.
        M_expected = self.expected_mass(pheno, age_s)
        self._M_expected = float(M_expected)
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

        # Skadeinflödets termer var för sig. Utan uppdelningen går det bara att
        # se att skada dödar, inte vilken term som byggde den — och `effort`
        # normeras mot `v_max`, som är ett arkitektoniskt tak och inte en
        # biologisk fart. Omnormeringen ska göras mot en mätning och inte mot en
        # gissning, så termerna exporteras först.
        self.last_damage_terms = {
            "dD_eff": float(dD_eff),
            "dD_met": float(dD_met),
            "dD_age": float(dD_age),
            "dD_starve": float(dD_starve),
            "dD_cold": float(dD_cold),
            "effort": float(effort),
            "rest": float(rest),
            "speed_n": float(speed_n),
        }

        # Skadan före taket. Den, och inte det klampade värdet, är det som
        # avgör om djuret överlever ticken.
        #
        # Klampen låg tidigare före reparationen, och reparationen drar alltid
        # av en flisa. Ett djur vars inflöde slog i taket fick därför D satt
        # till exakt D_max, sedan reparerat till D_max - epsilon, och
        # dödstestet `D >= D_max` blev falskt — varje tick, hur stort inflödet
        # än var. Det gav en fixpunkt exakt på gränsen: maximalt skadad och
        # odödlig. Uppmätt i runs/p78/s2 satt den sista individen på
        # mean_D = 1,00000 i 186 månader och blev 343 månader gammal, mot 140
        # för det äldsta djur som någonsin dött i samma körning.
        D_raw = D_before + dD_in
        self.D = clamp(D_raw, 0.0, _D_max)

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
            # Vad som inte får plats i reserven lämnar kroppen som exkrement.
            # Att radera det vore att förstöra massa: djuret hann äta upp det.
            # Att i stället begränsa intaget vid källan är biologiskt renare men
            # rör födosöket, och tas när kalibreringen är stabil.
            trim_kg = self._take_reserve_mass((Et - Ecap) / _E_labile)
            if trim_kg > 0.0:
                self._void(trim_kg, 0.0)
                E_overflow = trim_kg * _E_labile
    
        # ---------------------------------------------------------
        # (5) Deterministic death conditions
        # ---------------------------------------------------------
        # Skadedöden prövas mot D_raw, alltså inflödet innan taket kapade det.
        # Reparationen har verkat däremellan och kan sänka D under D_max, men
        # den kan inte göra ogjort att skadan under ticken översteg vad
        # kroppen bär. Att pröva mot self.D vore att låta klampen bestämma
        # över biologin.
        if float(D_raw) >= _D_max or float(self.M) <= _M_min:
            self.death_cause = "damage" if float(D_raw) >= _D_max else "starvation"
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
                    self.death_cause = "hazard"
                    self.alive = False
                    return
    
        # ---------------------------------------------------------
        # (7) Numerical guard (post)
        # ---------------------------------------------------------
        clamped = False
    
        if float(self.M_fast) < 0.0:
            self.M_fast = 0.0
            clamped = True
        if float(self.M_slow) < 0.0:
            self.M_slow = 0.0
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
            math.isfinite(self.M_fast)
            and math.isfinite(self.M_slow)
            and math.isfinite(self.M)
            and math.isfinite(self.D)
            and math.isfinite(self.Fg)
        ):
            self.guard_steps += 1
            self.guard_killed += 1
            self.guard_last = self._guard_snapshot("post_state")
            self.death_cause = "guard_post"
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
    
        expected_E_after = (
            E_before
            + dE_store
            + E_from_M_total
            - E_out_drain
            - out_gest_build
            - E_out_repair
            - E_material
            - E_overflow
            - out_growth
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
            "E_material": E_material,
            "E_overflow": E_overflow,
    
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
            "dM_growth": dM_growth,
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
            # Katabolismens förlust: den labila fraktion som mobiliserades
            # men inte blev energi. Strukturmaterialet räknas inte som förlust
            # — det lämnade kroppen som exkrement och finns kvar i detritus.
            "E_loss_catabolism": float(max(
                0.0,
                dM_cat_total * (1.0 - _structure) * _E_labile - E_from_M_total,
            )),
            "dM_growth": float(dM_growth),
            "M_expected": float(M_expected),
            "m_rel_expected": float(m_rel),
            "dD_starve": float(dD_starve),
            "reserve_frac": float(self.reserve_frac()),
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
    grid: Grid
    store: object | None = None

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

    def _wrap_points_inplace(self, xs: np.ndarray, ys: np.ndarray) -> None:
        self.grid.wrap_pos_inplace(xs, ys)
        
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

    def sense_local(
        self,
        world: World,
        x: float,
        y: float,
        secB: np.ndarray,
        secC: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
        """
        Perception när grannskapet redan aggregerats till sektorer.

        Strålarnas geometri behövs inte längre: sektorerna kommer färdiga från
        `Population._build_sector_percept()`, som räknar dem för alla djur på en
        gång. Kvar här är det som är lokalt för individen — cellens eget
        innehåll och sensorbruset.

        Bruset läggs på i u-domänen precis som förut, med samma sigma. Antalet
        dragningar följer antalet sektorer i stället för antalet strålar, så
        slumpströmmen är inte densamma som med strålsensorn.
        """
        B0_kg, C0_kg = world.sample_food_local(x, y)
        Pworld = getattr(world, "WP", None)
        Kb = float(getattr(Pworld, "B_K", 0.0)) if Pworld is not None else 0.0
        Kc = float(getattr(Pworld, "C_sense_K", 0.0)) if Pworld is not None else 0.0

        B0_u = self._sat1_u(float(B0_kg), Kb)
        C0_u = self._sat1_u(float(C0_kg), Kc)

        accB = np.array(secB, dtype=np.float32, copy=True)
        accC = np.array(secC, dtype=np.float32, copy=True)

        sig = float(self.AP.noise_sigma)
        if sig > 0.0 and (rng is not None) and accB.size:
            accB += (rng.standard_normal(size=accB.size) * sig).astype(np.float32, copy=False)
            accC += (rng.standard_normal(size=accC.size) * sig).astype(np.float32, copy=False)
            B0_u = float(B0_u + rng.normal(0.0, sig * 0.5))
            C0_u = float(C0_u + rng.normal(0.0, sig * 0.5))

        np.clip(accB, 0.0, 1.0, out=accB)
        np.clip(accC, 0.0, 1.0, out=accC)

        # Riktningskanalen som reflexerna läser. Strålsensorn exponerade
        # `_accB` och `_ang_base`; sektorpercepten har färre och bredare
        # riktningar, och reflexen ska läsa den som faktiskt kördes.
        # Sektor k pekar (k + 0,5) sektorbredder från nosen.
        S = int(accB.size)
        if getattr(self, "_sector_ang", None) is None or self._sector_ang.size != S:
            self._sector_ang = (
                (np.arange(S, dtype=np.float32) + np.float32(0.5))
                * np.float32(2.0 * math.pi / max(S, 1))
            )
        self._acc_dir_B = accB
        self._acc_dir_C = accC
        self._acc_dir_ang = self._sector_ang

        return (clamp(B0_u, 0.0, 1.0), clamp(C0_u, 0.0, 1.0)), accB, accC

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
        B0_kg, C0_kg = world.sample_food_local(x, y)
        
        Pworld = getattr(world, "WP", None)
        Kb = float(getattr(Pworld, "B_K", 0.0)) if Pworld is not None else 0.0
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
        
        self._wrap_points_inplace(self._xs, self._ys)
        
        # ---- (C) Sample flora + carcass separately ----
        self._Bp[:] = world.sample_flora_rays(self._xs, self._ys)
        world.sample_many_carcass(self._xs, self._ys, outC=self._Cp)
        
        Bkg = self._Bp
        Ckg = self._Cp

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

        self._acc_dir_B = self._accB
        self._acc_dir_C = self._accC
        self._acc_dir_ang = self._ang_base

        return (float(B0_u), float(C0_u)), self._accB, self._accC

    def see_agent_first_hit(
        self,
        world: World,
        x: float,
        y: float,
        heading: float,
        self_id: int,
        m_eff: int = 0,
    ) -> tuple[float, float, float, int, int, int]:
        """
        Returns (present, bearing_u, dist_u, j_hit, hit_slot, hit_agent_id).
        j_hit: avståndssteg-index för träffen (-1 om ingen träff).
        hit_slot: store-slot för träffen (-1 om ingen träff).
        hit_agent_id: biologiskt agent-ID vid träffpunkten (0 om ingen träff).
    
        Fauna upptäcks via store.spatial_index.
        """
        n = int(self._n)
        m_full = int(self._m)
        if n <= 0 or m_full <= 0:
            return 0.0, 0.0, 0.0, -1, -1, 0
    
        store = getattr(self, "store", None)
        if store is None:
            return 0.0, 0.0, 0.0, -1, -1, 0
    
        np.add(self._ang_base, np.float32(heading), out=self._ang)
        np.cos(self._ang, out=self._dx)
        np.sin(self._ang, out=self._dy)
    
        xx = np.float32(x)
        yy = np.float32(y)
    
        np.multiply(self._dx[:, None], self._d[None, :], out=self._xs)
        self._xs += xx
        np.multiply(self._dy[:, None], self._d[None, :], out=self._ys)
        self._ys += yy
    
        self._wrap_points_inplace(self._xs, self._ys)
    
        if m_eff > 0 and m_eff < m_full:
            ray_depths = np.minimum(self._ray_m, m_eff)
        else:
            ray_depths = self._ray_m
    
        for j in range(m_full):
            hit_i = -1
            hit_slot = -1
            hit_id = 0
    
            for i in range(n):
                if j >= int(ray_depths[i]):
                    continue
    
                cell = int(self.grid.cell_of(float(self._xs[i, j]), float(self._ys[i, j])))
                slots = store.slots_in_cell(cell)
                if slots.size == 0:
                    continue
    
                for slot in slots:
                    s = int(slot)
    
                    if not bool(store.alive[s]):
                        continue
                    if int(store.kind[s]) != 0:
                        continue
    
                    oid = int(store.id[s])
                    if oid == int(self_id):
                        continue
                    if oid <= 0:
                        continue
    
                    hit_i = i
                    hit_slot = s
                    hit_id = oid
                    break
    
                if hit_i >= 0:
                    break
    
            if hit_i >= 0:
                bearing_u = float(hit_i) / float(n)
                dist_u = float(self._d[j]) / max(float(self.AP.ray_len_front), 1e-9)
                return 1.0, bearing_u, dist_u, j, hit_slot, hit_id
    
        return 0.0, 0.0, 0.0, -1, -1, 0



@dataclass
class ActionPlan:
    turn: float
    thrust: float
    allow_move: float
    allow_eat: float
    explore_drive: float
    B0: float
    C0: float
    Tloc: float


@dataclass
class BodyStepInput:
    speed: float
    activity: float
    food_bio_kg: float
    food_carcass_kg: float
    food_bio_J: float
    food_carcass_J: float
    assim_bio_kg: float
    assim_carcass_kg: float
    E_move: float
    Tloc: float
    B0: float
    C0: float
    
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
    store_slot: int = -1
    wrapper_lookup: object = None

    OBS_DIM: ClassVar[int] = 23   # +2: predator_bearing (cos/sin), predator_dist
    OUT_DIM: ClassVar[int] = 5
    
    body: Body = field(init=False)
    grid: Grid = field(init=False)
    sensors: RaySensors = field(init=False)

    obs_trace: np.ndarray = field(init=False)

    birth_t: float = 0.0
    pheno: Phenotype = field(init=False)

    last_speed: float = 0.0

    # Bansträcka och nettoförflyttning sedan födseln, det senare utan
    # torusvikning. Kvoten mellan dem är Del 1:s mätpunkt i
    # docs/rorelsens-arkitektur.md — 0,034 innan riktningen fick tröghet, alltså
    # 33 cellbredders bana för 1,1 cellbredders förflyttning.
    path_len: float = 0.0
    disp_x: float = 0.0
    disp_y: float = 0.0

    last_B0: float = 0.0
    last_C0: float = 0.0

    WF = 0.6
    WS = 0.4

    sense_level: int = field(init=False, default=0)

    # Adaptiv sensing-cache: lagrar senaste sensing-resultat och cooldown-räknare.
    _sense_cd: int = field(init=False, default=0)        # steg kvar tills nästa skanning
    # Minne av senast sedda artfrände: [x, y, kurs, ålder i tick]. None = tomt.
    _nb_mem: object = field(init=False, default=None, repr=False, compare=False)
    # id på den artfrände djuret senast följde. Förstahandsval nästa gång den
    # syns, så att den sociala reflexen får ett stabilt mål över hela mötet.
    _follow_id: int = field(init=False, default=0)
    # (antal, kurs-x, kurs-y) per riktningssektor i kroppsram.
    _soc_sectors: object = field(init=False, default=None, repr=False, compare=False)
    # Temperatur per riktningssektor i kroppsram, satt av sensingpasset.
    _temp_sectors: object = field(init=False, default=None, repr=False, compare=False)
    _cached_B0: float = field(init=False, default=0.0)
    _cached_C0: float = field(init=False, default=0.0)
    _cached_x_in: np.ndarray = field(init=False)         # cachat obs-vektor
    _last_detect_j: int = field(init=False, default=-1)  # avståndssteg för senaste träff (-1=ingen)
    _sense_m_eff: int = field(init=False, default=0)     # effektivt stråldjup nästa skanning
    _cached_agent_hit: tuple = field(init=False)         # (N, Nu, Nd, hit_slot, hit_id) från senaste see_agent_first_hit
    _cached_predator_hit: tuple = field(init=False)      # (pred_bearing, pred_dist) — närmaste hotande predator
    _mating_mode: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.AP = replace(self.AP)

        self.body = Body(self.AP)
        self.obs_trace = np.zeros((8,), dtype=np.float32)

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
        self._cached_agent_hit = (0.0, 0.0, 0.0, -1, 0)
        self._cached_predator_hit = (0.0, 0.0)

        self._mating_mode = False

    def _init_body_state_from_AP(self) -> None:
        self.body.M = max(0.0, float(self.AP.M0))

        E0 = max(0.0, float(self.AP.E0))
        e_lab = float(self.AP.E_labile_J_per_kg)
        self.body.M_fast = (0.85 * E0) / e_lab
        self.body.M_slow = (0.15 * E0) / e_lab

        self.body.Tb = float(getattr(self.AP, "Tb_init", 37.0))
        
        self.body.D = 0.0
        self.body.Fg = float(self.body.Fg)
        self.body.alive = True

    def bind_world(self, world: World) -> None:
        self.world = world
        if getattr(self, "sensors", None) is None:
            self.sensors = RaySensors(self.AP, grid=self.grid, store=None)

    def bind_grid(self, grid: Grid) -> None:
        self.grid = grid

    def bind_store(self, store) -> None:
        self.store = store
        if getattr(self, "sensors", None) is not None:
            self.sensors.store = store

    def bind_wrapper_lookup(self, fn) -> None:
        self.wrapper_lookup = fn

    def apply_traits(self) -> None:
        self.pheno = derive_pheno(self.genome.traits)
        
    def phenotype_summary(self) -> dict:
        return phenotype_summary(self.pheno)

    @staticmethod
    def _signed_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

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

    def _torus_delta_to(self, other: "Agent") -> tuple[float, float, float]:
        dx, dy = self.grid.torus_delta_pos(
            float(self.x), float(self.y),
            float(other.x), float(other.y),
        )
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

    def _build_inputs_from_cache(self) -> tuple[np.ndarray, float, float]:
        """
        Returnera cachad observationsvektor, men uppdatera de delar som måste
        spegla aktuell kroppsstatus även när ingen full sensing körs.
        """
        self._sense_cd -= 1
    
        hunger = float(self.body.hunger())
        fatigue = float(self.body.Fg)
        D = float(self.body.D)
    
        # obs = [B0, C0, meanB, meanC, maxB, maxC, hunger, fatigue]
        self._cached_x_in[6] = hunger
        self._cached_x_in[7] = fatigue
    
        # sista delen av obs-vektorn innehåller D på index 20
        self._cached_x_in[20] = D
    
        a = 0.06
        self.obs_trace = (1.0 - a) * self.obs_trace + a * self._cached_x_in[:8]
        self._cached_x_in[8:16] = self.obs_trace
    
        # Rekurrentt state ska alltid vara färskt
        _h_dim = int(self._h.shape[0])
        if _h_dim > 0:
            self._cached_x_in[self.OBS_DIM:] = self._h
    
        return self._cached_x_in.copy(), self._cached_B0, self._cached_C0
    
    
    def _run_full_sensing(
        self,
        world: World,
        rng: np.random.Generator,
        m_eff: int,
        sectors: tuple[np.ndarray, np.ndarray] | None = None,
        neighbour: object = False,
    ) -> tuple[
        float, float, np.ndarray, np.ndarray,
        float, float, float, int, int, int,
        bool, bool, float, float
    ]:
        """
        Kör full sensing och returnerar allt som build_inputs behöver vidare.
        """
        # Sektorpercepten är enda vägen sedan 0084, och strålreserven togs
        # aldrig: 23 209 anrop till `_run_full_sensing` gav noll anrop till
        # `RaySensors.sense`, verifierat över två parameteruppsättningar innan
        # grenen togs bort. Att låta den ligga kvar dolde att `sense()` var död.
        secB, secC = sectors
        (B0, C0), rays_B, rays_C = self.sensors.sense_local(
            world, self.x, self.y, secB, secC, rng=rng,
        )
    
        thresh = float(self.AP.sense_alert_thresh)
        food_near = (
            float(B0) > thresh
            or float(C0) > thresh
            or (len(rays_B) > 0 and float(rays_B.max()) > thresh)
            or (len(rays_C) > 0 and float(rays_C.max()) > thresh)
        )
    
        # Batchad artfrändeuppslagning är enda vägen sedan 0085.
        # `see_agent_first_hit` togs aldrig i samma verifiering.
        if neighbour is None or neighbour is False:
            N_ag, Nu_ag, Nd_ag, j_agent, hit_slot, hit_id = 0.0, 0.0, 0.0, -1, -1, 0
        else:
            N_ag, Nu_ag, Nd_ag, j_agent, hit_slot, hit_id = neighbour
        agent_near = j_agent >= 0
    
        pred_bearing = float(Nu_ag) * 2.0 * math.pi if N_ag > 0.5 else 0.0
        pred_dist = float(Nd_ag) if N_ag > 0.5 else 1.0
    
        return (
            float(B0), float(C0), rays_B, rays_C,
            float(N_ag), float(Nu_ag), float(Nd_ag),
            int(j_agent), int(hit_slot), int(hit_id),
            bool(food_near), bool(agent_near),
            float(pred_bearing), float(pred_dist),
        )
    
    
    def _update_sensing_schedule(
        self,
        *,
        food_near: bool,
        agent_near: bool,
        j_agent: int,
        m_full: int,
    ) -> None:
        """
        Uppdatera adaptiv sensingfrekvens och nästa effektiva djup.
        """
        # Avståndsberoende frekvens, som ekolokalisering: långsamma ping när
        # ingenting är i närheten, tätare ju närmare grannen kommer.
        #
        # Två fasta steg — tio i vila, tre i beredskap — räcker inte för
        # alignment. Riktningsbruset verkar varje tick och ackumulerar 61
        # grader över tio tick, mot en alignmentkorrigering på två grader per
        # sensing. Kursen randomiseras helt mellan två observationer, och den
        # regeln kan därför inte verka oavsett vikt.
        #
        # Med avståndsberoende intervall betalar djuret för upplösning bara när
        # den behövs. Sensing kostar energi, så avvägningen finns redan: ett
        # djur som ständigt har grannar nära betalar mer.
        if agent_near:
            j_det = max(0, int(j_agent))
            self._last_detect_j = j_det
            self._sense_m_eff = min(m_full, max(2, j_det + 3))
            # Linjär interpolation mellan alert och idle efter hur nära
            # grannen är i förhållande til synvidden.
            idle = max(1, int(self.AP.sense_idle_steps))
            alert = max(1, int(self.AP.sense_alert_steps))
            frac = clamp(float(j_det) / max(1.0, float(m_full)), 0.0, 1.0)
            steps = alert + (idle - alert) * frac
            self._sense_cd = max(0, int(round(steps)) - 1)
        elif food_near:
            self._last_detect_j = 0
            self._sense_m_eff = min(m_full, 3)
            self._sense_cd = max(0, int(self.AP.sense_alert_steps) - 1)
        else:
            self._last_detect_j = -1
            self._sense_m_eff = m_full
            self._sense_cd = max(0, int(self.AP.sense_idle_steps) - 1)
            
    def build_inputs(self, world: World, rng: np.random.Generator,
                     sectors: tuple[np.ndarray, np.ndarray] | None = None,
                     neighbour: object = False):
        if not self.body.alive:
            return None, 0.0, 0.0
    
        m_full = int(self.sensors._m) if self.sensors is not None else 0
    
        if self._sense_m_eff <= 0:
            self._sense_m_eff = m_full
    
        # Cacheväg
        if self._sense_cd > 0:
            return self._build_inputs_from_cache()
    
        # Full sensing
        m_eff = self._sense_m_eff
    
        # Parningsläge: tvinga full räckvidd
        if self._mating_mode:
            m_eff = m_full
            self._sense_cd = 0
    
        (
            B0, C0, rays_B, rays_C,
            N_ag, Nu_ag, Nd_ag,
            j_agent, hit_slot, hit_id,
            food_near, agent_near,
            pred_bearing, pred_dist,
        ) = self._run_full_sensing(world, rng, m_eff, sectors=sectors,
                                   neighbour=neighbour)
    
        self._cached_agent_hit = (N_ag, Nu_ag, Nd_ag, hit_slot, hit_id)
    
        x_in = self._build_obs(
            B0, C0, rays_B, rays_C,
            pred_bearing=pred_bearing,
            pred_dist=pred_dist,
        )
    
        self._update_sensing_schedule(
            food_near=food_near,
            agent_near=agent_near,
            j_agent=j_agent,
            m_full=m_full,
        )
    
        # Lägg till rekurrentt state
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
        
    # ------------------------------
    # apply_outputs: Helpers
    # ------------------------------
    
    def _split_recurrent_output(self, y: np.ndarray) -> np.ndarray:
        _h_dim = int(self._h.shape[0])
        _out_dim = int(self.OUT_DIM)
        if _h_dim > 0:
            h_raw = y[_out_dim : _out_dim + _h_dim]
            self._h = np.tanh(h_raw).astype(np.float32)
            return y[:_out_dim]
        return y
    
    
    def _decode_action_outputs(
        self,
        y: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        turn = float(np.tanh(y[0]))
        thrust = float(1.0 / (1.0 + np.exp(-float(y[1]))))
        inh_move = float(1.0 / (1.0 + np.exp(-float(y[2]))))
        inh_eat = float(1.0 / (1.0 + np.exp(-float(y[3]))))
        explore_drive = float(1.0 / (1.0 + np.exp(-float(y[4]))))
    
        allow_move = 1.0 - inh_move
        allow_eat = 1.0 - inh_eat
        return turn, thrust, allow_move, allow_eat, explore_drive
    
    
    def _resolve_detected_agent(
        self,
        hit_slot: int,
        hit_id: int,
        N: float,
    ):
        store = getattr(self, "store", None)
        lookup = getattr(self, "wrapper_lookup", None)
    
        if N <= 0.5:
            return None
        if store is None or not callable(lookup):
            return None
        if int(hit_slot) < 0:
            return None
    
        s = int(hit_slot)
        if s < 0 or s >= int(store.n):
            return None
        if not bool(store.alive[s]):
            return None
        if int(store.kind[s]) != 0:
            return None
    
        detected_id = int(store.id[s])
        if detected_id <= 0 or detected_id == int(self.id):
            return None
        if int(hit_id) > 0 and detected_id != int(hit_id):
            return None
    
        detected = lookup(s)
        if detected is None or detected is self or (not detected.body.alive):
            return None
        return detected
    
    
    def _evaluate_local_agent_drives(
        self,
        detected,
        hunt_eff: float,
        in_mating_mode: bool,
    ) -> tuple[object, float, object, float, object]:
        best_prey = None
        best_prey_score = -1e9
        best_threat = None
        best_threat_score = -1e9
        best_mate = None
    
        if detected is None:
            return best_prey, best_prey_score, best_threat, best_threat_score, best_mate
    
        dx, dy, dist = self._torus_delta_to(detected)
    
        _hunt_diet_exp = float(getattr(self.AP, "hunt_diet_exp", 1.5))
    
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
    
        if in_mating_mode and bool(getattr(detected, "_mating_mode", False)):
            best_mate = (detected, dx, dy, dist)
    
        return best_prey, best_prey_score, best_threat, best_threat_score, best_mate
    
    
    def _apply_reflex_drives(
        self,
        turn: float,
        thrust: float,
        explore_drive: float,
        soc: float,
        hunt_eff: float,
        in_mating_mode: bool,
        N: float,
        Nu: float,
        Nd: float,
        best_prey,
        best_prey_score: float,
        best_threat,
        best_threat_score: float,
        best_mate,
        neighbour_heading: float | None = None,
        neighbour_memory=None,
    ) -> tuple[float, float, float, float, float]:
        hunt_state = 0.0
        flee_state = 0.0
    
        if (
            best_threat is not None
            and hunt_eff < float(self.AP.threat_predation_min)
            and best_threat_score > float(self.AP.hunt_score_min)
        ):
            other, dx, dy, dist = best_threat
            a_th = math.atan2(dy, dx)
            err = self._signed_angle(a_th - self.heading)
            bias = clamp(err / math.pi, -1.0, 1.0)
    
            turn = clamp(turn - 0.95 * bias, -1.0, 1.0)
            thrust = clamp(max(thrust, 0.95), 0.0, 1.0)
            explore_drive *= 0.10
            flee_state = 1.0
    
        elif (
            best_prey is not None
            and hunt_eff >= float(self.AP.predator_trait_min)
            and best_prey_score > float(self.AP.hunt_score_min)
        ):
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
    
        elif N > 0.5 or neighbour_memory is not None:
            if N > 0.5:
                a_hit = self.heading + (2.0 * math.pi * float(Nu))
                Nd_mem = float(Nd)
                head_mem = neighbour_heading
                trust = 1.0
            else:
                # Minnet styr bara. Det ger aldrig ett mål för parning eller
                # predation — de kräver en verkligt detekterad motpart.
                a_hit, Nd_mem, head_mem, trust = neighbour_memory
            _ = a_hit
            errN = self._signed_angle(a_hit - self.heading)
            biasN = clamp(errN / math.pi, -1.0, 1.0) * trust
            Nd_f = Nd_mem
            neighbour_heading = head_mem
    
            # Reynolds tre regler, med var sin räckvidd. Separation nära,
            # alignment på mellanavstånd, kohesion långt bort.
            #
            # Tidigare fanns bara två av dem, och kohesionen var dessutom
            # viktad tvärtom: `wdist = 1 - Nd` är störst på nära håll, alltså
            # starkast precis där separationen borde ta över och svagast där
            # gruppen behöver dras ihop. Två djur svängde mot varandra, möttes,
            # stöttes isär i repulsionszonen och divergerade omedelbart —
            # eftersom ingenting sa åt dem att fortsätta åt samma håll. Det
            # ger möten, inte flockar, och kohesionskvoten mot Poisson mättes
            # till 1,0 oavsett `sociability`: ingen aggregering alls.
            #
            # Alignment är regeln som omvandlar ett möte till ett sällskap.
            # Riktningen ett djur rör sig i är observerbar — den läses ur
            # motpartens kurs, inte ur dess arvsmassa.
            REP_ZONE = 0.35
            soc_bias = 2.0 * soc - 1.0
            if Nd_f < REP_ZONE:
                rs = 1.0 - (Nd_f / REP_ZONE)
                turn = clamp(turn - 0.70 * rs * biasN, -1.0, 1.0)
                explore_drive = explore_drive * (1.0 - 0.3 * rs)
            elif abs(soc_bias) > 1e-6:
                # Kohesion mot **grannskapets tyngdpunkt**, inte mot den
                # närmaste. En enskild granne ger bara riktningen mot den
                # grannen, och djur som svänger mot närmaste artfrände roterar
                # runt varandra i stället för att konvergera. 0099 gjorde
                # målvalet stabilt utan att kohesionskvoten rörde sig från 1,0.
                #
                # Reynolds tre regler får dela på **samma** totala vikt som den
                # enda kohesionstermen hade. Flockningen ska vara en drift
                # bland flera — föda, värme, flykt och parning verkar
                # oförändrat — inte ta över styrningen.
                wcoh = clamp((Nd_f - REP_ZONE) / max(1e-6, 1.0 - REP_ZONE), 0.0, 1.0)
                _soc = getattr(self, "_soc_sectors", None)
                did_group = False
                if _soc is not None and soc_bias > 0.0:
                    _F, _HX, _HY = _soc
                    _S = len(_F)
                    if _S:
                        _ang = (np.arange(_S) + 0.5) * (2.0 * math.pi / _S)
                        _w = np.asarray(_F, dtype=np.float64)
                        _cx = float(np.sum(_w * np.cos(_ang)))
                        _cy = float(np.sum(_w * np.sin(_ang)))
                        if abs(_cx) + abs(_cy) > 1e-9:
                            _e = self._signed_angle(math.atan2(_cy, _cx))
                            turn = clamp(turn + 0.40 * soc_bias * wcoh
                                         * clamp(_e / math.pi, -1.0, 1.0), -1.0, 1.0)
                            did_group = True
                        # Alignment mot grannskapets medelkurs. Vektorerna är
                        # redan i kroppsram, så nollriktningen är egen kurs.
                        _hx = float(np.sum(_HX)); _hy = float(np.sum(_HY))
                        if abs(_hx) + abs(_hy) > 1e-9:
                            _ea = self._signed_angle(math.atan2(_hy, _hx))
                            turn = clamp(turn + 0.20 * soc_bias
                                         * clamp(_ea / math.pi, -1.0, 1.0), -1.0, 1.0)
                if not did_group:
                    # Ingen granne i aggregatet — falla tillbaka på den
                    # detekterade individen.
                    turn = clamp(turn + 0.40 * soc_bias * wcoh * biasN, -1.0, 1.0)

                # Alignment: starkast på mellanavstånd, där grannen är nära nog
                # att vara värd att följa men inte så nära att man måste väja.
                if (not did_group) and neighbour_heading is not None and soc_bias > 0.0:
                    w = clamp((Nd_f - REP_ZONE) / max(1e-6, 1.0 - REP_ZONE), 0.0, 1.0)
                    walign = 4.0 * w * (1.0 - w)
                    errA = self._signed_angle(float(neighbour_heading) - self.heading)
                    turn = clamp(turn + 0.20 * soc_bias * walign
                                 * clamp(errA / math.pi, -1.0, 1.0), -1.0, 1.0)

                # Utforskningen dämpas bara av att *söka sällskap*. Tidigare
                # användes abs(soc_bias), så även ett djur som undviker
                # artfränder slutade söka föda.
                if soc_bias > 0.0:
                    explore_drive = explore_drive * (1.0 - 0.60 * soc_bias * wcoh)
    
        return turn, thrust, explore_drive, flee_state, hunt_state
    
    
    def _apply_food_steering(
        self,
        ctx: "StepCtx",
        turn: float,
        thrust: float,
        explore_drive: float,
        flee_state: float,
        B0: float,
        C0: float,
    ) -> tuple[float, float, float]:
        hunger_now = float(self.body.hunger())
    
        if flee_state < 0.5 and hunger_now > 0.4:
            sensors = getattr(self, "sensors", None)
            if sensors is not None:
                accB = getattr(sensors, "_acc_dir_B", None)
                accC = getattr(sensors, "_acc_dir_C", None)
                ang = getattr(sensors, "_acc_dir_ang", None)
    
                if accB is not None and accC is not None and ang is not None and len(accB) > 0:
                    _diet = float(getattr(self.pheno, "diet", 0.5))
                    herb_eff = (1.0 - _diet) ** 0.7
                    scav_eff = _diet ** 0.7
    
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
    
        _diet_local = float(getattr(self.pheno, "diet", 0.5))
        _herb_local = (1.0 - _diet_local) ** 0.7
        _scav_local = _diet_local ** 0.7
        food_local = clamp(float(B0) * _herb_local + float(C0) * _scav_local, 0.0, 1.0)
    
        explore_drive *= 1.0 - hunger_now * food_local
        return turn, thrust, explore_drive
    
    
    def _integrate_motion(
        self,
        ctx: "StepCtx",
        turn: float,
        thrust: float,
        allow_move: float,
        explore_drive: float,
    ) -> tuple[float, float]:
        """
        Riktning och fart som tillstånd med tröghet.

        **Riktningen.** Tre saker var fel i den gamla formen. Vridhastigheten
        hade inget tak — `turn_rate · dt = 6,0` rad per tick mot ett varv på
        6,28. Styrningen var proportionell med förstärkningen 1,54 per tick,
        alltså översläng med teckenbyte varje tick. Och bruset skalade med `dt`
        i stället för `√dt`, vilket gör en slumpvandrings diffusion beroende av
        tidsstegets storlek.

        Följden var att headingen dekorrelerade på ett tick. Organismen rörde
        sig fort — uppmätt 37 cellbredder per månad — men kom ingenstans:
        rakheten över livet låg på 0,069.

        **Persistensen är tillståndsberoende, inte hög.** Att göra rörelsen rak
        vore fel svar. Kringgående sök i ett område är rätt beteende när födan
        är riklig, och den slingrande banan är det som håller organismen kvar på
        fläcken. Det som saknades var inte raka linjer utan förmågan att
        **välja** — att färdas när det finns skäl och söka lokalt när det inte
        finns. Se `explore_drive` i steg 2 nedan.

        **Farten** löses ur kraftbalansen i stället för att integreras explicit
        mot en relaxationstid som är kortare än tidssteget. Se steg 1 nedan.
        """
        dt = float(ctx.dt)

        # --- 1. fart: kvasistatisk kraftbalans --------------------------------
        # Explicit Euler mot dragkraften har förstärkningsfaktorn
        # `|1 − dt·c₁/M|`, som passerar ett vid `M = dt·c₁/2 = 2,2 kg`. Uppmätt
        # medianmassa är 1,3–1,7 kg, så schemat var divergent för merparten av
        # populationen och hölls ändligt bara av klampningen mot noll och
        # `v_max`. Farten svängde alltså mellan noll och taket i stället för att
        # följa gaspådraget, och `F_prop` och `v` var aldrig i kraftbalans —
        # vilket också är skälet till att farten inte gick att härleda ur
        # `E_loss_loco` under antagande om stationaritet.
        #
        # Relaxationstiden `M/c₁ ≈ 0,45 tick` är kortare än tidssteget, så
        # kvasistatisk form är den riktiga approximationen: terminalfarten löses
        # direkt ur `F_prop = c₁v + c₂v²`. Samma fälla och samma lösning som den
        # termiska relaxationen; se `docs/metabolismen.md`.
        fatigue = float(self.body.Fg)
        fatigue_factor = clamp(1.0 - 0.9 * fatigue, 0.05, 1.0)
        weak_move = float(self.body.move_factor())
        u = clamp(allow_move * thrust * fatigue_factor * weak_move, 0.0, 1.0)

        M_pre = max(1e-9, float(self.body.M))
        F_prop = u * float(self.AP.F0) * (M_pre ** float(self.AP.force_mass_exp))

        c1 = float(self.AP.drag_lin)
        c2 = float(self.AP.drag_quad)

        if F_prop <= 0.0:
            speed = 0.0
        elif c2 > 0.0:
            speed = (math.sqrt(c1 * c1 + 4.0 * c2 * F_prop) - c1) / (2.0 * c2)
        else:
            speed = F_prop / max(c1, 1e-12)
        speed = min(max(0.0, speed), float(self.AP.v_max))
        self.last_speed = float(speed)

        # Vid kraftbalans är den mekaniska effekten exakt dragkraftens
        # dissipation. Uttrycket är oförändrat men beror inte längre på
        # föregående ticks numeriska transient.
        eta = clamp(float(self.AP.locomotion_eff), 1e-6, 1.0)
        E_move = (dt * max(0.0, F_prop * speed)) / eta

        # --- 2. riktning: relaxation med tak, plus persistent brus ------------
        # Centripetalvillkoret: en snabb organism svänger trögt. Vid låg fart
        # binder i stället det absoluta taket, så att uttrycket inte divergerar
        # när farten går mot noll.
        w_max = min(
            float(self.AP.turn_rate_max),
            float(self.AP.lat_accel_max) / max(speed, 1e-6),
        )

        # Analytisk relaxation mot den önskade riktningen. `turn` tolkas som ett
        # riktningsanspråk i (−π, π) relativt nuvarande kurs; andelen som tas ut
        # per tick ligger alltid i (0, 1) och kan därför aldrig slå över.
        frac = 1.0 - math.exp(-float(self.AP.turn_gain) * dt)
        d_steer = frac * clamp(float(allow_move) * float(turn), -1.0, 1.0) * math.pi

        # Rotationsdiffusion uttryckt som persistenstid. σ = √(2·D·dt) med
        # D = 1/τ gör diffusionen oberoende av tidssteget.
        #
        # τ interpolerar mellan kringgående sök och rak färd. `explore_drive`
        # är redan dämpad av `hunger · food_local` i födostyrningen, alltså låg
        # när organismen står på föda den vill ha och hög annars. Den är därmed
        # rätt signal för regimvalet — men den användes tvärtom: mer utforskning
        # gav mer brus och därmed *sämre* förflyttning.
        tau_local = max(1e-6, float(self.AP.dir_tau_local))
        tau_run = max(tau_local, float(self.pheno_dir_tau()))
        e = clamp(float(explore_drive), 0.0, 1.0)

        # Ett djur som följer flocken irrar inte. `tau_dir` interpolerar redan
        # mot `explore_drive` — låg utforskning ger rakare färd — och att följa
        # en granne är lika mycket "har ett mål" som att stå på föda.
        #
        # Utan det raderas alignment av bruset mellan två observationer: 19
        # grader per tick mot en korrigering på två.
        _soc = getattr(self, "_soc_sectors", None)
        if _soc is not None and float(getattr(self.pheno, "sociability", 0.0)) > 0.5:
            _F = _soc[0]
            if len(_F) and float(np.sum(_F)) > 0.0:
                e *= 1.0 - clamp(2.0 * (float(self.pheno.sociability) - 0.5), 0.0, 1.0)

        tau_dir = tau_local + (tau_run - tau_local) * e

        d_noise = math.sqrt(2.0 * dt / tau_dir) * float(ctx.rng.normal(0.0, 1.0))

        d_theta = clamp(d_steer + d_noise, -w_max * dt, w_max * dt)
        self.heading = self._signed_angle(float(self.heading) + d_theta)

        # --- 3. förflyttning och mätning -------------------------------------
        step_x = dt * speed * math.cos(self.heading)
        step_y = dt * speed * math.sin(self.heading)

        # Bansträcka och nettoförflyttning ackumuleras utan torusvikning, så att
        # kvoten mellan dem går att läsa i life-loggen. Den kvoten är Del 1:s
        # enda mätpunkt: 0,034 före den här ändringen.
        self.path_len += abs(dt * speed)
        self.disp_x += step_x
        self.disp_y += step_y

        self.x, self.y = self.grid.wrap_pos(float(self.x) + step_x, float(self.y) + step_y)

        return float(speed), float(E_move)

    def pheno_dir_tau(self) -> float:
        """
        Riktningens persistenstid i månader, ur `_T_MOB`.

        Locuset hade noll läsare. Avvägningen behöver ingen egen kostnad för att
        vara tvåsidig: hög persistens ger effektiv förflyttning men dålig lokal
        genomsökning — organismen lämnar en god fläck och hittar inte tillbaka —
        medan låg persistens ger tvärtom. Det är skillnaden mellan en vandrare
        och en betare, och den biter bara i en fläckvis värld, vilket floran
        efter Steg 4 faktiskt ger.
        """
        return direction_tau(
            float(getattr(self.pheno, "mobility", 0.5)),
            float(self.AP.dir_tau_min),
            float(self.AP.dir_tau_max),
        )
    
    
    def _perform_feeding(
        self,
        world: World,
        dt: float,
        allow_eat: float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Returnerar (kg_levande, kg_detritus, energi_levande_J, energi_detritus_J,
        assimilerat_levande_kg, assimilerat_detritus_kg).

        Energin bär substratets faktiska strukturandel med sig. Att bara skicka
        vidare kilon och multiplicera med en konstant hos konsumenten skulle
        kasta bort informationen om vad som faktiskt åts.

        Assimilationen avgörs här och bara här. Body.step() får den upptagna
        massan färdig och behöver inte räkna om verkningsgrader — det var den
        dubbleringen som lät kostpreferensen slå på energin men inte på massan.
        """
        if allow_eat <= 0.20:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        want_kg = float(self.AP.eat_rate) * dt * (0.25 + 0.75 * float(self.body.hunger()))
        diet = float(getattr(self.pheno, "diet", 0.5))
        # Betningen når så långt som organismen färdats under ticken. Den äter
        # medan den går; att bara sampla slutpunkten lämnade föda orörd längs
        # vägen. Antagandet höll när förflyttningen var fyra hundradels cell
        # per tick och blev fel när den är två.
        reach = int(math.ceil(float(getattr(self, "last_speed", 0.0)) * float(dt)))
        got_l, got_d, e_l, e_d = world.consume_food(
            self.x,
            self.y,
            amount=want_kg,
            diet=diet,
            reach=max(1, reach),
        )

        herb_eff, scav_eff = diet_efficiency(diet)

        a_l = self._excrete(world, got_l, e_l, herb_eff)
        a_d = self._excrete(world, got_d, e_d, scav_eff)

        return float(got_l), float(got_d), float(e_l), float(e_d), float(a_l), float(a_d)

    def _excrete(self, world: World, mass_kg: float, energy_J: float,
                 diet_eff: float) -> float:
        """
        Återför den massa som inte assimileras till cellen som detritus.
        Returnerar den assimilerade massan i kilo.

        Utan detta försvinner allt ätet ur modellen: kroppsmassan växer ur
        energibudgeten och den ingesterade massan bokförs ingenstans.

        Assimilationsandelen är (1 - struktur) * matsmältning * kostpreferens.
        Resten passerar igenom. Eftersom strukturmaterialet passerar i sin
        helhet medan bara en del av det labila gör det, är exkrementet mer
        strukturrikt än födan — betning koncentrerar segt material i
        detrituspoolen.
        """
        m = float(mass_kg)
        if m <= 1e-15:
            return 0.0

        e_lab = float(self.AP.E_labile_J_per_kg)
        s_in = 1.0 - (float(energy_J) / max(m * e_lab, 1e-30))
        s_in = min(1.0, max(0.0, s_in))

        m_assim = m * assimilated_fraction(s_in, diet_eff)
        out_kg = max(0.0, m - m_assim)
        if out_kg <= 1e-15:
            return float(m_assim)

        s_out = min(1.0, max(0.0, m * s_in / out_kg))
        world.excrete_at(self.x, self.y, out_kg, s_out)
        return float(m_assim)
    
    
    def _activity_proxy(
        self,
        speed: float,
        allow_eat: float,
        food_bio_kg: float,
        food_carcass_kg: float,
    ) -> float:
        speed_n = clamp(speed / max(float(self.AP.v_max), 1e-9), 0.0, 1.0)
        ate = 1.0 if (allow_eat > 0.20 and (food_bio_kg + food_carcass_kg) > 0.0) else 0.0
        return 0.03 + 0.45 * speed_n + 0.10 * ate

    
    def plan_actions(
        self,
        world: World,
        ctx: StepCtx,
        y: np.ndarray,
        B0: float,
        C0: float,
    ) -> ActionPlan:
        """
        Planeringsdel av gamla apply_outputs():
          - tolka policy-output
          - lokala reflexer/social steering
          - food steering
    
        Ingen state-exekvering här:
          - ingen rörelse
          - ingen feeding
          - ingen body.step()
    
        Returnerar ett litet beslutspaket som move_system senare verkställer.
        """
        y = self._split_recurrent_output(y)
    
        turn, thrust, allow_move, allow_eat, explore_drive = self._decode_action_outputs(y)
    
        Tloc = float(world.temperature_at(self.x, self.y)) if hasattr(world, "temperature_at") else 0.0
    
        soc = float(getattr(self.pheno, "sociability", 0.0))
        pred = float(getattr(self.pheno, "predation", 0.0))
        N, Nu, Nd, hit_slot, hit_id = self._cached_agent_hit
        in_mating_mode = bool(self._mating_mode)
    
        _hunt_diet_exp = float(getattr(self.AP, "hunt_diet_exp", 1.5))
        _diet_val = float(getattr(self.pheno, "diet", 0.5))
        hunt_eff = pred * (_diet_val ** _hunt_diet_exp)
    
        detected = self._resolve_detected_agent(hit_slot, hit_id, N)
    
        (
            best_prey,
            best_prey_score,
            best_threat,
            best_threat_score,
            best_mate,
        ) = self._evaluate_local_agent_drives(
            detected,
            hunt_eff,
            in_mating_mode,
        )
    
        # Termisk gradient. `cold_aversion` hade ingen läsare alls — djuren
        # bar en ärftlig köldaversion som inte gjorde någonting, och drev
        # därför mot kylan: 68 procent dog kallare än de föddes, och
        # svältdöden hade median 2,6 grader mot världens 12,2.
        #
        # Riktningen är en viktad vektorsumma över sektorernas avvikelse från
        # sitt eget medelvärde, alltså den lokala gradienten. Styrkan skalas av
        # köldaversionen och av hur mycket djuret faktiskt fryser — ett djur i
        # trettio grader har ingen anledning att söka värme.
        _ts = getattr(self, "_temp_sectors", None)
        if _ts is not None and len(_ts):
            _ca = float(getattr(self.pheno, "cold_aversion", 0.0))
            _Tloc = float(world.temperature_at(self.x, self.y)) if hasattr(world, "temperature_at") \
                else float(np.mean(_ts))
            _stress = clamp((float(self.AP.Tb_set) - _Tloc) / float(self.AP.Tb_set), 0.0, 1.0)
            if _ca > 1e-6 and _stress > 1e-6:
                _S = len(_ts)
                _dev = np.asarray(_ts, dtype=np.float64) - float(np.mean(_ts))
                _ang = (np.arange(_S) + 0.5) * (2.0 * math.pi / _S)
                _gx = float(np.sum(_dev * np.cos(_ang)))
                _gy = float(np.sum(_dev * np.sin(_ang)))
                if abs(_gx) + abs(_gy) > 1e-9:
                    _errT = self._signed_angle(math.atan2(_gy, _gx))
                    turn = clamp(turn + 0.70 * _ca * _stress
                                 * clamp(_errT / math.pi, -1.0, 1.0), -1.0, 1.0)

        # Minnet av senast sedda artfrände. Uppdateras när någon syns och
        # dödräknas annars framåt längs dess senast sedda kurs. Det överbryggar
        # både sensingintervallet och den sida där synfältet är kortast.
        mem_lim = int(getattr(self.AP, "social_memory_ticks", 0))
        if detected is not None:
            self._nb_mem = [float(detected.x), float(detected.y),
                            float(detected.heading), 0]
        elif self._nb_mem is not None:
            m = self._nb_mem
            m[3] += 1
            if m[3] > mem_lim:
                self._nb_mem = None
            else:
                v = float(getattr(self, "last_speed", 0.0)) * float(ctx.dt)
                m[0] += v * math.cos(m[2])
                m[1] += v * math.sin(m[2])

        nb_mem = None
        if detected is None and self._nb_mem is not None and mem_lim > 0:
            mx, my, mh, mage = self._nb_mem
            dxm = mx - self.x
            dym = my - self.y
            ex, ey = float(self.grid.extent_x), float(self.grid.extent_y)
            dxm -= ex * round(dxm / ex)
            dym -= ey * round(dym / ey)
            dm = math.hypot(dxm, dym)
            rf = float(self.AP.ray_len_front)
            if dm <= rf:
                # Tilltron avtar linjärt med minnets ålder.
                nb_mem = (math.atan2(dym, dxm), dm / max(rf, 1e-9), mh,
                          max(0.0, 1.0 - mage / max(1.0, float(mem_lim))))

        turn, thrust, explore_drive, flee_state, hunt_state = self._apply_reflex_drives(
            turn=turn,
            thrust=thrust,
            explore_drive=explore_drive,
            soc=soc,
            hunt_eff=hunt_eff,
            in_mating_mode=in_mating_mode,
            N=N,
            Nu=Nu,
            Nd=Nd,
            best_prey=best_prey,
            best_prey_score=best_prey_score,
            best_threat=best_threat,
            best_threat_score=best_threat_score,
            best_mate=best_mate,
            neighbour_heading=(float(detected.heading) if detected is not None else None),
            neighbour_memory=nb_mem,
        )
    
        turn, thrust, explore_drive = self._apply_food_steering(
            ctx=ctx,
            turn=turn,
            thrust=thrust,
            explore_drive=explore_drive,
            flee_state=flee_state,
            B0=B0,
            C0=C0,
        )
    
        return ActionPlan(
            turn=float(turn),
            thrust=float(thrust),
            allow_move=float(allow_move),
            allow_eat=float(allow_eat),
            explore_drive=float(explore_drive),
            B0=float(B0),
            C0=float(C0),
            Tloc=float(Tloc),
        )
    
    def execute_action_plan(
        self,
        world: World,
        ctx: StepCtx,
        plan: ActionPlan,
    ) -> BodyStepInput:
        """
        Exekveringsdel av action plan:
          - rörelse
          - feeding
          - activity proxy
    
        Returnerar ett explicit underlag för body_system.
        """
        dt = float(ctx.dt)
    
        speed, E_move = self._integrate_motion(
            ctx=ctx,
            turn=float(plan.turn),
            thrust=float(plan.thrust),
            allow_move=float(plan.allow_move),
            explore_drive=float(plan.explore_drive),
        )
    
        (
            food_bio_kg,
            food_carcass_kg,
            food_bio_J,
            food_carcass_J,
            assim_bio_kg,
            assim_carcass_kg,
        ) = self._perform_feeding(
            world=world,
            dt=dt,
            allow_eat=float(plan.allow_eat),
        )
    
        activity = self._activity_proxy(
            speed=speed,
            allow_eat=float(plan.allow_eat),
            food_bio_kg=food_bio_kg,
            food_carcass_kg=food_carcass_kg,
        )
    
        return BodyStepInput(
            speed=float(speed),
            activity=float(activity),
            food_bio_kg=float(food_bio_kg),
            food_carcass_kg=float(food_carcass_kg),
            food_bio_J=float(food_bio_J),
            food_carcass_J=float(food_carcass_J),
            assim_bio_kg=float(assim_bio_kg),
            assim_carcass_kg=float(assim_carcass_kg),
            E_move=float(E_move),
            Tloc=float(plan.Tloc),
            B0=float(plan.B0),
            C0=float(plan.C0),
        )
        
    # --- reproduction hooks (Population uses these) ---
    
    def start_gestation(self) -> bool:
        # child mass target from phenotype (absolute units)
        M_target = float(getattr(self.pheno, "child_M", 0.0))
        return bool(self.body.start_gestation(M_target))
    
    def pay_repro_cost(self, cost_E_J: float, *, transfer: bool = False) -> float:
        """
        Dra energi från parent. Returnerar faktiskt betald energi (J).
        OBS: Den här är bara en transfer; pain/damage ska uppstå via
        din ordinarie energi-/stresslogik i steget.

        `transfer=True` när massan går vidare till en avkomma i stället för att
        oxideras. Reproduktionens overhead brinner; barnets startreserv gör det
        inte.
        """
        want_J = max(0.0, float(cost_E_J))
        paid_J = float(self.body.take_energy(want_J, burn=not bool(transfer)))
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
    
        e_lab = float(self.AP.E_labile_J_per_kg)
        self.body.M_fast = Ef_J / e_lab
        self.body.M_slow = Es_J / e_lab
    
        # ---- Clip to Ecap deterministiskt ----
        # Överskottet raderas inte utan bokförs som exkrement, precis som i
        # Body.step(). Barnet kan inte bära mer reserv än dess massa tillåter.
        Et = float(self.body.E_total())
        Ecap = float(self.body.E_cap())
        if Et > Ecap:
            e_lab_c = float(self.AP.E_labile_J_per_kg)
            trim_kg = self.body._take_reserve_mass((Et - Ecap) / e_lab_c)
            if trim_kg > 0.0:
                self.body._void(trim_kg, 0.0)
    
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
