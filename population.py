from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from world import World, WorldParams
from mlp import MLPGenome
from agent import Agent, AgentParams
from genetics import (
    child_genome_from_parent,
    recombine,
    MutationConfig,
    genetic_compatibility,
    init_organism_traits,
    mutate_trait_vector,
)
import flora_growth
from phenotype import (
    _T_SOC,
    derive_pheno,
    flora_adult_mass,
    dispersal_scale,
    establish_p,
    FLORA_REPRO_MASS_MULT,
    SEED_STRUCTURE,
    flora_apparatus,
    flora_lifespan,
    flora_seed_mass,
    flora_repro_alloc,
    flora_maturity_frac,
    flora_root_alloc,
    flora_turnover_rate,
    flora_repro_capacity,
    flora_temp_opt,
    flora_temp_width,
    flora_uptake_capacity,
    structure_fraction,
    energy_density,
    nutrient_content,
    nutrient_content_array,
    diet_efficiency,
    assimilated_fraction,
    NUTRIENT_PER_KG_LABILE,
)

from organism_store import OrganismStore, next_organism_id
from grid import Grid

# new logging
from simlog.events import Event, EventName
from simlog.sinks import EventHub
from simlog import records


_NAN_DICT = {
    "mean": float("nan"), "median": float("nan"),
    "p10":  float("nan"), "p25":    float("nan"),
    "p75":  float("nan"), "p90":    float("nan"),
}

def _stats_1d(x: np.ndarray) -> dict[str, float]:
    """
    Snabb statistik för små arrayer (typiskt n=10–100).
    Använder sort+indexering istf np.percentile — 17× snabbare för n=50.
    """
    if x.size == 0:
        return dict(_NAN_DICT)

    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return dict(_NAN_DICT)

    xs = np.sort(x)
    mean = float(xs.sum()) / n
    mid  = n >> 1
    median = float(xs[mid]) if n & 1 else float(xs[mid - 1] + xs[mid]) * 0.5

    # Linjär interpolation för percentiler (matchar np.percentile default)
    def _pct(p: float) -> float:
        idx = p * (n - 1)
        lo  = int(idx)
        hi  = lo + 1
        if hi >= n:
            return float(xs[n - 1])
        frac = idx - lo
        return float(xs[lo]) + frac * float(xs[hi] - xs[lo])

    return {
        "mean":   mean,
        "median": median,
        "p10":    _pct(0.10),
        "p25":    _pct(0.25),
        "p75":    _pct(0.75),
        "p90":    _pct(0.90),
    }

def _occupied_cells(store, fl) -> int:
    """
    Antal celler med minst en levande planta.

    Bincount och inte unique: unique sorterar, och vid trehundratusen plantor
    kostar det 13,3 ms mot 0,60 ms — tjugotvå gånger. Funktionen anropas två
    gånger per florasammanfattning.
    """
    if fl.size == 0:
        return 0
    c = store.cell_idx[fl]
    c = c[c >= 0]
    if c.size == 0:
        return 0
    return int(np.count_nonzero(np.bincount(c, minlength=int(store.n_cells))))


@dataclass
class PopParams:
    init_pop: int = 12
    # Tick då faunan sätts in. 0 = vid start.
    #
    # Floran behöver omkring 15 000 tick för att nå jämvikt från sådd. Sätts
    # faunan in före det möter den en halvfärdig värld: sådden ger 56 procent
    # av jämviktens stående gröda, produktionen skalar med den, och ett bestånd
    # som ligger under bärkraften i den färdiga världen ligger över den i den
    # halvfärdiga.
    fauna_at: int = 0
    # Radie i kontinuerliga enheter för insättningsfläcken. 0 = jämn
    # utspridning över hela världen. Se `_fauna_spawn_pos`.
    fauna_spawn_radius: float = 0.0
    fauna_spawn_patches: int = 1
    # Grupperna i traitrymden. `founder_group_sep` är avståndet mellan
    # gruppernas tyngdpunkter i logit-rymden, `founder_group_spread` en skala
    # på spridningen inom gruppen. 0 = alla ur samma fördelning.
    founder_group_sep: float = 0.0
    founder_group_spread: float = 1.0
    # Jämviktsdetektion för fördröjd insättning. Se `_fauna_release_now`.
    flora_eq_window: int = 1000
    flora_eq_tol: float = 0.01
    flora_eq_max_ticks: int = 40000
    # Flockmedlemskap som relation: affiniteten stiger med `flock_gain` vid
    # varje observation och avtar med `flock_decay` mellan dem. Vid 0,25 och
    # 0,90 krävs ungefär fyra observationer för full medlem, och en granne som
    # försvinner tappas efter ungefär trettio.
    flock_gain: float = 0.25
    flock_decay: float = 0.90
    # Startvärde för `sociability` hos grundarna. None = normal slumpning.
    #
    # Reflexen använder `soc_bias = 2·soc − 1`, så nollpunkten ligger vid 0,5:
    # halva en uniformt slumpad startpopulation styr *bort* från artfränder.
    # Med detektion i storleksordningen tre procent av agenttickarna uttrycks
    # driften sällan nog att selektionen ska hinna verka, så egenskapen driver
    # i praktiken neutralt medan beståndet avgörs.
    #
    # Sätts värdet får alla grundare samma. Locus muteras normalt vidare, så
    # avkomman kan avvika — det är startfördelningen som styrs, inte taket.
    sociability_init: float | None = None
    # Spridning kring `sociability_init`, i logit-rymden. 0 = alla identiska.
    # Förvalet motsvarar ungefär ±0,08 i fenotyp kring 0,8 — samma
    # storleksordning som den stående variation en verklig grundarpopulation
    # bär, och drygt hundra gånger den effektiva mutations-sd:n per generation.
    sociability_init_sd: float = 0.5
    max_pop: int = 500
    # Initial floramassa som multipel av faunans initiala massa.
    #
    # Talet hör ihop med B_K och måste följa med när den ändras: såddet är
    # massastyrt, så större växter ger färre av dem. Med B_K på 11 och kvoten
    # kvar på 10 sås ett trettiotal jätteväxter i hela världen.
    #
    # Nivån 2000 är kalibrerad, inte härledd. Masslagret visade att
    # assimilerat och förbränt låg inom två procent av varandra, medan en
    # växande population kräver att intaget täcker underhåll plus egen
    # tillväxt plus avkomma — omkring fyra procent ovanpå underhållet.
    # Glappet på sex procent satt i primärproduktionen. En ekologi
    # har mer primärproduktion än konsumenter; modellen hade förhållandet
    # omvänt med faktor 138. Talet sätter utgångsläget, inte jämvikten —
    # om dynamiken inte bär det faller floran tillbaka, och då är det den
    # dynamiken som ska rättas.
    flora_init_mass_ratio: float = 2000.0
    # Medelmassa per sådd planta. Antalet faller ut ur måltotalen dividerat med
    # det här talet, i stället för ur världens cellantal.
    #
    # 1,32 kg är den uppmätta jämviktsmassan per individ: 118 104 kg fördelat
    # på 89 319 plantor i en floraköring utan fauna över 30 000 tick. Sådden
    # hamnar därmed nära den struktur som ändå uppstår, i stället för att
    # behöva växa dit från fel håll.
    #
    # Tätheten är dessutom självbegränsande sedan sådden betalas ur marken: en
    # cell med `nutrient_init` = 0,117 kg fri näring räcker till omkring fem
    # plantor av den här storleken, vilket är samma storleksordning som
    # jämviktens 5,45 per cell. Plantor som inte får betalt krymps.
    flora_init_plant_mass: float = 1.32
    #
    # Efter Steg 4b betyder talet vad det säger. Bördigheten ligger i
    # `WorldParams.nutrient_init` och sådden debiteras marken, så det här är en
    # starttäthet och inte längre världens näringsmängd i förklädnad. Vid
    # M_1 = B_K är jämviktens stående biomassa omkring 180 000 kg, så 2000 sår
    # ungefär en sjundedel av den och låter världen växa in i resten.

    # `flora_mortality` och `flora_seedling_mort_mult` är borta. Livslängden
    # härleds nu ur strukturandelen — se phenotype.flora_lifespan — och
    # groddplantor som inte får näring försvinner av att förnafallet äter upp
    # dem, inte av en påslagen risk. Båda var postulat där det nu finns
    # mekanism.
    # Massa under vilken en planta räknas som död. Förnafallet är
    # multiplikativt och når därför aldrig noll av sig självt.
    flora_min_mass_frac: float = 1.0e-4
    # Tak på antal frön per planta och tick. Bara en beräkningsgräns: med
    # propagulmassa ner till 1e-4 kg kan en full pool annars ge tusentals frön
    # i ett svep, och etableringsutfallet dras ändå innan slots allokeras.
    flora_max_seeds_per_tick: int = 64
    # Andel av propagulmassan under vilken en groddplanta räknas som förlorad.
    flora_seedling_floor: float = 0.5

    # Tillväxtpassets väg. "numba" kompilerar kärnan vid första anropet;
    # "numpy" kör referensvägen. Båda ska ge samma utfall — flaggan finns för
    # att kunna jämföra dem elementvis på samma tillstånd, vilket är den enda
    # verifiering som säger något. Faller tillbaka på numpy om numba saknas.
    flora_growth_backend: str = "numba"

    store_growth_min_chunk: int = 256
    store_growth_factor: float = 2.0
    
    n_traits: int = 41   # +1: _T_BREED_PULL = 40, social fassynkronisering

    spawn_jitter_r: float = 1.5

    carcass_yield: float = 0.65  # currently unused (carcass mass = remaining M)
    carcass_rad: int = 2

    # sampling
    sample_dt: float = 1.0
    sample_avoid_repeat_k: int = 0

    warm_age_max_s: float = 60.0
    warm_cd_max_s: float = 8.0

    mating_radius: float = 3.0   # var 1.5 — vid 50 agenter på 64×64 gav det bara ~9% chans att mötas; 3.0 ger ~35%

    # --- Rekurrent minnesdimension ---
    # Varje agent bär en h-vektor av denna storlek mellan stegen.
    # 0 = ingen rekurrens (bakåtkompatibelt).
    # Rekommendation: 8–16 för ett balanserat minne/kostnad-förhållande.
    h_dim: int = 8

    # --- Genetisk kompatibilitet (reproduktiv isolering) ---
    # Parningssannolikhet P = exp(-d2_norm / 2*sigma2) dar d ar normaliserat
    # avstand i trait-rymden. Se genetics.genetic_compatibility() for detaljer.
    #
    # Rekommenderat flode: borja med compat_sigma=2.0 (permissivt) tills
    # populationen ar stabil, sank sedan mot 0.5-1.0 for artbildning.
    compat_sigma: float = 2.0    # bredden pa kompatibilitetsklockan
    compat_enabled: bool = True  # False = alla kan para sig med alla (debug)


@dataclass
class SenseBatch:
    alive: list[Agent]
    alive_slots: np.ndarray
    X: np.ndarray
    BC_list: list[tuple[float, float]]


@dataclass
class DecisionBatch:
    plans: list[tuple[Agent, object]]
    plan_slots: np.ndarray


@dataclass
class BodyBatch:
    body_inputs: list[tuple[Agent, object]]
    body_slots: np.ndarray
    
@dataclass
class Population:
    WP: WorldParams
    AP: AgentParams
    PP: PopParams
    MC: MutationConfig = field(default_factory=MutationConfig)
    seed: int = 0

    store: OrganismStore = field(init=False)
    grid: Grid = field(init=False)
    
    # optional: pass from runner; if None, no logging
    hub: Optional[EventHub] = None

    # optional: only emit "step" events for this id (keeps logging cheap)
    track_step_id: Optional[int] = None

    world: World = field(init=False)
    agents: List[Agent] = field(init=False, default_factory=list)

    t: float = 0.0
    rng: np.random.Generator = field(init=False)

    _banks: dict[tuple, "ParamBank"] = field(init=False, default_factory=dict)

    _slot_to_agent: list[Agent | None] = field(init=False, default_factory=list)

    # agent sampling
    _next_sample_t: float = 0.0
    _recent_sample_ids: List[int] = field(default_factory=list)

    # Kumulativa totaler (aldrig nollställda) — samma semantik som konsolutskriften.
    # Analysverktyg kan diffa konsekutiva poster för att få per-period-värden.
    _births_total: int = 0
    _deaths_total: int = 0

    _last_flora_growth: float = 0.0
    _last_flora_established: int = 0
    _last_flora_dispersed_mass: float = 0.0
    # Flöden och dödsorsaker per tick. De finns för att kunna läsas i
    # world-loggen: utan dem syns förnafallet bara indirekt som att detritus
    # växer, och dödsorsakerna inte alls.
    _last_flora_shed: float = 0.0
    _last_flora_died_age: int = 0
    _last_flora_died_starve: int = 0
    _last_flora_seeds: int = 0
    # Andel växande plantor där ljuset är den knappare resursen. Är den nära
    # noll eller ett är den ena resursen inert och kalibreringen fel.
    _last_flora_light_limited: float = 0.0
    # Hur många plantor som passerar varje reproduktionsgrind var för sig.
    # Summan av poolen kan inte skilja en tom median från en annan flaskhals.
    _last_gate_alive: int = 0
    _last_gate_nutrient: int = 0
    _last_gate_carbon: int = 0
    _last_gate_size: int = 0
    _last_gate_all: int = 0
    _flora_slots_cache: object = None
    _fauna_slots_cache: object = None
    # Vald väg för tillväxtpasset, avgjord vid första anropet, och de
    # cellindexerade skrivbuffertar kärnan strör ut i. Buffertarna återanvänds
    # mellan tick: fyra allokeringar med längd n_cells per tick är gratis vid
    # sextontusen celler men inte vid en miljon.
    _flora_growth_mode: object = None
    _flora_cell_buf: object = None
    
    _flora_summary_cache: dict[str, float] | None = field(init=False, default=None)

    world_log_with_percentiles: bool = True

    def __post_init__(self) -> None:
        random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.world = World(self.WP)
        # Geometrin är en enda: dela världens Grid i stället för att instansiera
        # en andra med samma parametrar. Grid bär förberäknade tabeller och ska
        # inte finnas i två exemplar.
        self.grid = self.world.grid
        self._banks = {}

        self.world.consume_food_hook = self.consume_food
        self.world.sample_flora_local_hook = self.sample_flora_local
        self.world.sample_flora_rays_hook = self.sample_flora_rays        
        self._sector_cache: dict = {}
        
        self.store = OrganismStore(
            capacity=int(self.PP.max_pop),
            n_cells=int(self.grid.n_cells),
            n_traits=int(self.PP.n_traits),
        )
        self._slot_to_agent = [None] * int(self.PP.max_pop)        

        self._next_sample_t = 0.0
        self._recent_sample_ids = []
        self._births_total = 0
        self._deaths_total = 0
        # Kumulativt antal dödsfall där ingen dödsorsak sattes. Ska förbli noll;
        # se check_death_cause_set i invariants.py.
        self._deaths_without_cause = 0

        self._flora_summary_cache = None

        # ensure MC uses PP.n_traits (single source of truth for this run)
        if int(self.MC.n_traits) != int(self.PP.n_traits):
            self.MC = replace(self.MC, n_traits=int(self.PP.n_traits))

        # IO dims: obs/act är rena bio-dimensioner; nätverket får h_dim extra i/o.
        _h_dim  = max(0, int(self.PP.h_dim))
        in_dim  = int(Agent.OBS_DIM) + _h_dim
        out_dim = int(Agent.OUT_DIM) + _h_dim

        self.agents = []
        self._tick = 0
        self._fauna_spawn_centre = None
        self._flora_eq_prev = None
        self._founder_centroids = None
        # Faunan kan hållas tillbaka tills floran nått jämvikt.
        self._fauna_pending = int(self.PP.init_pop) if int(self.PP.fauna_at) != 0 else 0
        if self._fauna_pending == 0:
            self.seed_fauna(int(self.PP.init_pop))

        fauna_mass0 = float(sum(
            float(x.body.M) + x.body.M_reserve()
            for x in self.agents if x.body.alive
        ))
        if fauna_mass0 <= 0.0:
            # Sådden skalas mot **markens bördighet**, inte mot konsumenterna.
            #
            # Kopplingen till `fauna_mass0` har nu brustit fyra gånger: vid
            # fördröjd fauna, vid ändrad världsstorlek, vid ändrad init_pop och
            # senast vid `--init_pop 0`, där en floraköring utan djur sådde en
            # enda planta på 7,64 kg och därmed mätte ingenting alls.
            #
            # Rätt storhet fanns hela tiden. Sedan 0087 köps sådden ur den fria
            # näringspoolen, så taket är redan bördigheten — det här gör det
            # till *målet* i stället för till en spärr. En värld med
            # `nutrient_init` = 0,117 och 16 384 celler bär 1 917 kg näring,
            # vilket vid typisk stökiometri räcker till omkring 108 000 kg
            # vävnad mot jämviktens uppmätta 118 100.
            free_n = float(np.sum(np.asarray(self.world.nutrient), dtype=np.float64))
            target = free_n / max(1e-9, nutrient_content(0.5))
        else:
            target = float(self.PP.flora_init_mass_ratio) * fauna_mass0

        _ = self._seed_initial_flora(target_mass=target)
        self.store.rebuild_spatial_index()

        self._book_initial_nutrient()

    def _book_initial_nutrient(self) -> None:
        """
        Bokför näringen i utgångstillståndet som tillförd.

        Sådd flora och warm-startad fauna får sin massa gratis vid t=0. Utan
        den här posten startar ledgern med ett konstant glapp som ser ut som
        en läcka men bara är initialvillkoret. Efter den här raden ska
        `nutrient_balance()['unaccounted']` vara noll vid tick 0.
        """
        world = self.world
        store = self.store

        total = float(np.sum(np.asarray(world.nutrient), dtype=np.float64))

        # Sådd förna bär näring och måste bokföras som tillförd, precis som den
        # fria poolen. Posten var noll så länge `detritus_init` var det, och
        # blev synlig först när världen började sås vid sin jämviktsfördelning:
        # 1 454 kg av 3 371 låg i marken och såg ut som en läcka på 43 procent.
        act = np.asarray(world.detritus_active_cells, dtype=np.int64)
        if act.size:
            d = np.asarray(world.detritus)[act].astype(np.float64)
            ds = np.asarray(world.detritus_structure)[act].astype(np.float64)
            total += float(np.sum(d * nutrient_content_array(ds)))
        cact = np.asarray(world.carcass_active_cells, dtype=np.int64)
        if cact.size:
            c = np.asarray(world.carcass)[cact].astype(np.float64)
            cs = np.asarray(world.carcass_structure)[cact].astype(np.float64)
            total += float(np.sum(c * nutrient_content_array(cs)))

        for slot in range(int(store.n)):
            if not bool(store.alive[slot]):
                continue
            total += float(store.mass[slot]) * nutrient_content(float(store.structure[slot]))

        e_lab = float(self.WP.E_labile_J_per_kg)
        for a in self.agents:
            if not a.body.alive:
                continue
            total += a.body.M_reserve() * NUTRIENT_PER_KG_LABILE
            total += float(a.body.gest_M) * NUTRIENT_PER_KG_LABILE

        world._nutrient_added_total += total

    # -----------------------
    # logging helpers
    # -----------------------
    def _emit(self, name: EventName, t: float, payload: dict) -> None:
        if self.hub is None:
            return
        self.hub.emit(Event(name=name, t=float(t), payload=payload))

    def _emit_wanted(self, name: EventName, t: float) -> bool:
        """
        Skulle någon lyssnare ta emot det här eventet nu?

        Frågan finns för poster som är dyra att bygga. Den ska ställas före
        nyttolasten, inte efter — det var hela felet med världsposten.
        """
        hub = self.hub
        if hub is None:
            return False
        w = getattr(hub, "wants", None)
        return True if w is None else bool(w(name, float(t)))

    def _emit_birth(self, t: float, child: Agent, parent: Optional[Agent]) -> None:
        self._emit("birth", t, records.birth_record(t, child, parent))

    def _emit_death(
        self,
        t: float,
        agent: Agent,
        carcass_amount: float,  # kg
        carcass_rad: int,
    ) -> None:
        self._emit(
            "death",
            t,
            records.death_record(
                t=t,
                agent=agent,
                carcass_amount=float(carcass_amount),
                carcass_rad=int(carcass_rad),
            ),
        )

    def _emit_population(self, t: float, births: int, deaths: int) -> None:
        # Räkna bara levande för statistik + pop (mer semantiskt korrekt)
        alive = [a for a in self.agents if a.body.alive]
        pop_n = int(len(alive))
    
        E = np.fromiter((float(a.body.E_total()) for a in alive), dtype=np.float64, count=pop_n)
        D = np.fromiter((float(a.body.D) for a in alive),       dtype=np.float64, count=pop_n)
        M = np.fromiter((float(a.body.M) for a in alive),       dtype=np.float64, count=pop_n)
        G = np.fromiter((float(getattr(a.body, "gest_M", 0.0)) for a in alive), dtype=np.float64, count=pop_n)
        R = np.fromiter(
            (
                float(a.body.E_total()) / max(float(a.body.E_cap()), 1e-12)
                for a in alive
            ),
            dtype=np.float64,
            count=pop_n,
        )
    
        sE = _stats_1d(E)
        sD = _stats_1d(D)
        sM = _stats_1d(M)
        sR = _stats_1d(R)

        M_sum = float(np.nansum(M)) if pop_n > 0 else 0.0
        gest_M_sum = float(np.nansum(G)) if pop_n > 0 else 0.0
        E_store_sum = float(np.nansum(E)) if pop_n > 0 else 0.0
        e_body = float(getattr(self.AP, "E_body_J_per_kg", 0.0))
        E_body_equiv = e_body * M_sum
        E_gest_equiv = e_body * gest_M_sum

        flow_keys = [
            "food_bio_kg", "food_carcass_kg", "E_in_bio", "E_in_carcass", "E_in_total",
            "E_loss_digest_bio", "E_loss_digest_carcass", "E_loss_basal", "E_loss_compute",
            "E_loss_sense", "E_loss_loco", "E_loss_thermo", "E_loss_gest_overhead",
            "E_build_growth", "E_build_gestation", "E_loss_repair", "E_from_catabolism",
            "E_loss_catabolism", "dM_growth", "dM_gestation", "dM_catabolism",
        ]
        flow_sums = {k: 0.0 for k in flow_keys}
        for a in alive:
            fl = getattr(getattr(a, "body", None), "last_flux", None)
            if isinstance(fl, dict):
                for k in flow_keys:
                    try:
                        flow_sums[k] += float(fl.get(k, 0.0))
                    except Exception:
                        pass

        # Skadeinflödets termer, summerade över levande. Att veta vilken term
        # som faktiskt bygger skadan är förutsättningen för att kunna byta
        # `effort`-normeringen mot något mätt: den går i dag mot `v_max = 100`,
        # som är en klampningsgräns och ingen biologisk fart.
        dmg_keys = ["dD_eff", "dD_met", "dD_age", "dD_starve", "dD_cold",
                    "effort", "rest", "speed_n"]
        dmg_sums = {k: 0.0 for k in dmg_keys}
        for a in alive:
            dm = getattr(getattr(a, "body", None), "last_damage_terms", None)
            if isinstance(dm, dict):
                for k in dmg_keys:
                    try:
                        dmg_sums[k] += float(dm.get(k, 0.0))
                    except Exception:
                        pass
        if pop_n > 0:
            # De tre sista är tillstånd och inte flöden; de rapporteras som
            # medelvärde över beståndet.
            for k in ("effort", "rest", "speed_n"):
                dmg_sums[k] /= float(pop_n)
        flow_sums.update(dmg_sums)
    
        # Backward compatible: mean_* som tidigare
        # Nya fält: median_* och pXX_* + mass/energy ledgers.
        payload = records.population_record(
            t=t,
            pop_n=pop_n,
            births=int(births),
            deaths=int(deaths),
        
            mean_E=sE["mean"],
            mean_D=sD["mean"],
            mean_M=sM["mean"],
            mean_R=sR["mean"],
        
            median_E=sE["median"],
            p10_E=sE["p10"],
            p25_E=sE["p25"],
            p75_E=sE["p75"],
            p90_E=sE["p90"],
        
            median_D=sD["median"],
            p10_D=sD["p10"],
            p25_D=sD["p25"],
            p75_D=sD["p75"],
            p90_D=sD["p90"],
        
            median_M=sM["median"],
            p10_M=sM["p10"],
            p25_M=sM["p25"],
            p75_M=sM["p75"],
            p90_M=sM["p90"],
        )
        if isinstance(payload, dict):
            payload.update({
                "M_sum": float(M_sum),
                "gest_M_sum": float(gest_M_sum),
                "E_store_sum": float(E_store_sum),
                "E_body_equiv": float(E_body_equiv),
                "E_gest_equiv": float(E_gest_equiv),
                **{k: float(v) for k, v in flow_sums.items()},
            })
    
        self._emit("population", t, payload)

    def _emit_sample(self, t: float, a: Agent) -> None:
        self._emit("sample", t, records.sample_record(t, a, pop_n=len(self.agents)))

    def _emit_world(self, t: float) -> None:
        # Fråga först. Posten kostar en full florasammanfattning med kvantiler
        # plus två svep över cell- och floravektorerna, och byggdes varje tick
        # oavsett om någon världslogg fanns kopplad — vid 256x256 en fjärdedel
        # av takten, kastad i `_emit`. Världsloggen skriver dessutom i sin egen
        # kadens, som standard var annan simulerad sekund, alltså vart
        # hundrade tick.
        #
        # Gatningen i 0113 lade percentilerna bakom `_flora_want_quantiles`;
        # den här raden slog på flaggan igen varje tick och tog tillbaka det
        # mesta av den vinsten.
        if not self._emit_wanted("world", t):
            return

        # Kvantilerna behövs bara här, och bara när posten faktiskt skrivs.
        # Cachen byggs om i samma anrop nedan om den är ogiltig.
        self._flora_want_quantiles = True
        self._flora_summary_cache = None
        payload = records.world_record(
            t,
            self.world,
            with_percentiles=self.world_log_with_percentiles,
        )
        if isinstance(payload, dict):
            litter_sum = float(np.nansum(self.world.detritus))
            carcass_sum = float(np.nansum(self.world.carcass))
            # `M_C` är summan av båda dödpoolerna och behålls som totalsiffra;
            # `M_detritus` och `M_carcass` är de två var för sig. Poolerna har
            # olika strukturandel, olika nedbrytningstakt och olika roll i
            # födovalet, så en gemensam kurva döljer det som är intressant.
            detritus_sum = litter_sum + carcass_sum
            # Ledgerns energiskalor: nominell labil energitäthet vid
            # respektive ontologis initierade medelstrukturandel. Diagnostik,
            # inte fysik — den verkliga omvandlingen sker per organism.
            e_lab = float(self.WP.E_labile_J_per_kg)
            e_plant = e_lab * (1.0 - 0.57)
            e_carc = e_lab * (1.0 - 0.25)
            wf = getattr(self.world, "last_flux", {})
    
            flora_info = self._flora_summary()
            flora_n = int(flora_info["flora_n"])
            flora_mass = float(flora_info["flora_mass_store"])
            flora_energy = float(flora_info["flora_energy_store"])
    
            payload.update({
                # Flora-store är source of truth
                "E_B": e_plant * flora_mass,
                "E_C": e_carc * detritus_sum,
                "BC_sum": flora_mass + detritus_sum,
                "M_B": flora_mass,
                "M_C": detritus_sum,
                "M_detritus": litter_sum,
                "M_carcass": carcass_sum,
    
                "flora_n": flora_n,
                "flora_mass_store": flora_mass,
                "flora_energy_store": flora_energy,
    
                "flora_mean_repro_alloc": float(flora_info["flora_mean_repro_alloc"]),
                "flora_mean_adult_mass": float(flora_info["flora_mean_adult_mass"]),
                "flora_mean_temp_opt": float(flora_info["flora_mean_temp_opt"]),
                "flora_mean_temp_width": float(flora_info["flora_mean_temp_width"]),
                "flora_mean_apparatus": float(flora_info["flora_mean_apparatus"]),
                "flora_mean_seed_mass": float(flora_info["flora_mean_seed_mass"]),
    
                "E_in_growth": float(wf.get("E_in_growth", 0.0)),
                "E_loss_wither": float(wf.get("E_loss_wither", 0.0)),
                "E_loss_decay": float(wf.get("E_loss_decay", 0.0)),
                "dM_growth": float(wf.get("dM_growth", 0.0)),
                "dM_wither": float(wf.get("dM_wither", 0.0)),
                "dM_decay": float(wf.get("dM_decay", 0.0)),
                "flora_dM_growth": float(getattr(self, "_last_flora_growth", 0.0)),
                "flora_established": int(getattr(self, "_last_flora_established", 0)),
                "flora_dispersed_mass": float(getattr(self, "_last_flora_dispersed_mass", 0.0)),
                "flora_shed": float(getattr(self, "_last_flora_shed", 0.0)),
                "flora_died_age": int(getattr(self, "_last_flora_died_age", 0)),
                "flora_died_starve": int(getattr(self, "_last_flora_died_starve", 0)),
                "flora_seeds": int(getattr(self, "_last_flora_seeds", 0)),
                "gate_alive": int(getattr(self, "_last_gate_alive", 0)),
                "gate_nutrient": int(getattr(self, "_last_gate_nutrient", 0)),
                "gate_carbon": int(getattr(self, "_last_gate_carbon", 0)),
                "gate_size": int(getattr(self, "_last_gate_size", 0)),
                "gate_all": int(getattr(self, "_last_gate_all", 0)),
                "flora_light_limited": float(getattr(self, "_last_flora_light_limited", 0.0)),
                "flora_mean_structure": float(flora_info["flora_mean_structure"]),
                "flora_cells_occupied": float(flora_info["flora_cells_occupied"]),
                "flora_per_cell": float(flora_info["flora_per_cell"]),
                "flora_reserve_total": float(flora_info["flora_reserve_total"]),
                "flora_pool_p25": float(flora_info["flora_pool_p25"]),
                "flora_pool_median": float(flora_info["flora_pool_median"]),
                "flora_pool_p75": float(flora_info["flora_pool_p75"]),
                "flora_carbon_median": float(flora_info["flora_carbon_median"]),
                "flora_mass_median": float(flora_info["flora_mass_median"]),
                "flora_mass_p90": float(flora_info["flora_mass_p90"]),
                "flora_pool_total": float(flora_info["flora_pool_total"]),
                "flora_carbon_pool_total": float(flora_info["flora_carbon_pool_total"]),
                "flora_mean_root_alloc": float(flora_info["flora_mean_root_alloc"]),
                "flora_mean_maturity": float(flora_info["flora_mean_maturity"]),
                "flora_mature_frac": float(flora_info["flora_mature_frac"]),
                "flora_mean_root_frac": float(flora_info["flora_mean_root_frac"]),
                "flora_mean_seed_mass": float(flora_info["flora_mean_seed_mass"]),
            })

            # Näringskretsloppet. Näringen är modellens bevarade valuta och det
            # som begränsar floran, men den fanns inte i loggen alls — det gick
            # alltså inte att följa den utan att köra invariantsviten separat.
            # Sveparna är O(celler + flora) och sker i world-loggens kadens,
            # alltså tiotals tick isär.
            fl = self._flora_slots()
            det = np.asarray(self.world.detritus, dtype=np.float64)
            det_s = np.asarray(self.world.detritus_structure, dtype=np.float64)
            car = np.asarray(self.world.carcass, dtype=np.float64)
            car_s = np.asarray(self.world.carcass_structure, dtype=np.float64)
            payload.update({
                "nutrient_free": float(np.sum(self.world.nutrient, dtype=np.float64)),
                "nutrient_in_flora": float(np.sum(
                    self.store.mass[fl].astype(np.float64)
                    * nutrient_content_array(self.store.structure[fl]),
                    dtype=np.float64,
                )) if fl.size else 0.0,
                # `nutrient_in_detritus` är fortfarande båda poolerna, så att
                # summan fri + flora + fauna + detritus stänger som förut för
                # läsare som inte känner till delningen. `nutrient_in_litter`
                # och `nutrient_in_carcass` är de två var för sig.
                "nutrient_in_detritus": float(np.sum(
                    det * nutrient_content_array(det_s), dtype=np.float64
                )) + float(np.sum(
                    car * nutrient_content_array(car_s), dtype=np.float64
                )),
                "nutrient_in_litter": float(np.sum(
                    det * nutrient_content_array(det_s), dtype=np.float64
                )),
                "nutrient_in_carcass": float(np.sum(
                    car * nutrient_content_array(car_s), dtype=np.float64
                )),
                "carcass_cells": int(np.count_nonzero(car)),
                "detritus_cells": int(np.count_nonzero(det)),
                "nutrient_added": float(getattr(self.world, "_nutrient_added_total", 0.0)),
                "nutrient_lost": float(getattr(self.world, "_nutrient_lost_total", 0.0)),
            })
        self._emit("world", t, payload)

        self._flora_want_quantiles = False

    def _emit_step_if_tracked(self, t: float, a: Agent, B0: float, C0: float) -> None:
        if self.track_step_id is None:
            return
        if int(getattr(a, "id", -1)) != int(self.track_step_id):
            return
        # NOTE: records.step_record must be updated accordingly (no F0)
        self._emit("step", t, records.step_record(t, a, B0, C0))

    # -----------------------
    # policy net batch
    # -----------------------
    @staticmethod
    def _act_hidden(x: np.ndarray, act: str) -> np.ndarray:
        if act == "softsign":
            return x / (1.0 + np.abs(x))
        return np.tanh(x)
        
    # -----------------------
    # look-up
    # -----------------------
    def _agent_for_slot(self, slot: int) -> Agent | None:
        s = int(slot)
        if s < 0 or s >= len(self._slot_to_agent):
            return None
    
        ag = self._slot_to_agent[s]
        if ag is None:
            return None
        if ag.store_slot != s:
            return None
        if not ag.body.alive:
            return None
    
        return ag

    def _ensure_store_capacity(self, extra_slots: int = 1) -> None:
        """
        Säkerställ att OrganismStore har minst `extra_slots` lediga slots.
    
        Detta växer bara den tekniska store-kapaciteten för flora + fauna.
        Det ändrar inte djurtaket PP.max_pop.
        """
        need = int(extra_slots)
        if need <= 0:
            return
    
        free_now = len(self.store.free_slots)
        if free_now >= need:
            return
    
        old_cap = int(self.store.capacity)
        min_chunk = int(self.PP.store_growth_min_chunk)
        growth_factor = max(1.1, float(self.PP.store_growth_factor))
    
        target = max(
            old_cap + min_chunk,
            int(math.ceil(old_cap * growth_factor)),
            old_cap + (need - free_now),
        )
    
        self.store.grow(target)
    
        if len(self._slot_to_agent) < target:
            self._slot_to_agent.extend([None] * (target - len(self._slot_to_agent)))
            
    def _slot_distance(self, slot_a: int, slot_b: int, *, squared: bool = False) -> float:
        """
        Store-first geometrihjälpare för fauna-interaktioner.
    
        Interaction-systemet ska läsa position från OrganismStore, inte från wrapperns
        x/y. Wrapperposition finns tills vidare kvar för agentintern logik.
        """
        sa = int(slot_a)
        sb = int(slot_b)
        store = self.store
    
        if sa < 0 or sb < 0:
            return float("inf")
        if sa >= int(store.n) or sb >= int(store.n):
            return float("inf")
        if not bool(store.alive[sa]) or not bool(store.alive[sb]):
            return float("inf")
    
        xa = float(store.pos_x[sa])
        ya = float(store.pos_y[sa])
        xb = float(store.pos_x[sb])
        yb = float(store.pos_y[sb])
    
        if squared:
            return float(self.grid.distance2_pos(xa, ya, xb, yb))
        return float(self.grid.distance_pos(xa, ya, xb, yb))

    def _write_spatial_to_store(self, slot: int, x: float, y: float) -> None:
        """
        Uppdatera den rumsliga fauna-klungan direkt i store.
    
        Detta gör store till source of truth för position/cell efter move_system.
        Wrappern hålls tills vidare kvar som kompatibilitetsyta.
        """
        s = int(slot)
        if s < 0 or s >= int(self.store.n):
            return
    
        self.store.pos_x[s] = np.float32(float(x))
        self.store.pos_y[s] = np.float32(float(y))
        self.store.cell_idx[s] = np.int32(self.grid.cell_of(float(x), float(y)))

    def _write_alive_to_store(self, slot: int, alive: bool) -> None:
        """
        Uppdatera levande-status direkt i store.
    
        Detta gör store.alive till omedelbar sanning under ticken, i stället för att
        vänta på senare writeback eller slot-release.
        """
        s = int(slot)
        if s < 0 or s >= int(self.store.n):
            return
        self.store.alive[s] = bool(alive)

    def _write_body_surface_to_store(self, slot: int, a: Agent) -> None:
        """
        Uppdatera kroppens yttre store-state direkt från wrappern.
    
        Detta omfattar nu den fysiologiska surface som andra pass kan läsa:
          - mass
          - total energy
          - energy capacity
          - damage
          - wear
        """
        s = int(slot)
        if s < 0 or s >= int(self.store.n):
            return
    
        self.store.mass[s] = np.float32(float(a.body.M))
        self.store.energy[s] = np.float32(float(a.body.E_total()))
        self.store.energy_cap[s] = np.float32(float(a.body.E_cap()))
        self.store.damage[s] = np.float32(float(a.body.D))
        self.store.wear[s] = np.float32(float(a.body.W))

    def _write_gestation_to_store(self, slot: int, a: Agent) -> None:
        """
        Uppdatera store:ns gestationscache från Body.

        Ägarskapet är entydigt: Body är source of truth för gestating, gest_M,
        gest_E_J och gest_M_target fram till Steg 5, eftersom gest_M ackumuleras
        inne i Body.step():s energibudget och fostermassan belastar buren massa
        i samma beräkning. Store-fälten finns för att reproduktionsgrindarna ska
        kunna arbeta slotbaserat utan att gå via wrapperobjektet.

        Skrivriktningen är alltid Body -> store. Ingen kod får skriva åt andra
        hållet; divergens fångas av invariants.check_body_store_mirror().
        """
        s = int(slot)
        if s < 0 or s >= int(self.store.n):
            return

        body = a.body
        self.store.gestating[s] = bool(body.gestating)
        self.store.gest_M[s] = np.float32(body.gest_M)
        self.store.gest_E_J[s] = np.float32(body.gest_E_J)
        self.store.gest_M_target[s] = np.float32(body.gest_M_target)

    def _ready_to_reproduce_slot(self, slot: int) -> bool:
        """
        Store-first reproduktionsgate för fauna.
    
        Detta är nu den enda reproduktionsreadiness-källan i den heta loopen.
        """
        s = int(slot)
        store = self.store
    
        if s < 0 or s >= int(store.n):
            return False
        if not bool(store.alive[s]):
            return False
        if int(store.kind[s]) != 0:
            return False
    
        ag = self._agent_for_slot(s)
        if ag is None:
            return False
    
        if bool(store.gestating[s]):
            return False
        if float(store.repro_cd[s]) > 0.0:
            return False
    
        if float(store.age[s]) < float(ag.pheno.A_mature):
            return False
    
        M = float(store.mass[s])
        Mreq = max(float(ag.AP.M_min), float(ag.pheno.M_repro_min))
        if M < Mreq:
            return False
    
        E = float(store.energy[s])
        Ecap = max(float(store.energy_cap[s]), 1e-12)
        efrac = E / Ecap
        if efrac < float(ag.pheno.E_repro_min):
            return False

        # Säsongsgrind. Beredskapen var asynkron: varje individ blev redo på
        # sin egen tidtabell, och två djur måste vara det samtidigt. Uppmätt i
        # p97 var 14,8 procent klara vid en given tick, alltså 2,2 procents
        # sannolikhet att båda är det — innan de ens ska hitta varandra.
        #
        # Grinden är von Mises-liknande: exp(k·(cos Δ − 1)), som är ett vid
        # fasens topp och avtar med skärpan k. Vid k = 0 är den konstant ett,
        # så det asynkrona beteendet är bevarat som specialfall och inte
        # borttaget.
        #
        # Kuen är årscykeln som redan finns — `year_len` med sinusformad
        # temperatur — så ingen ny perception behövs.
        k = float(getattr(ag.pheno, "breed_sync", 0.0))
        if k > 1e-6:
            # Fasen mäts mot årets **temperaturtopp**, inte mot en absolut
            # punkt i cykeln. Signalen är gemensam för alla på samma latitud,
            # så synkroniseringen uppstår ur världen i stället för ur delad
            # genetik. `breed_phase` är individens förskjutning mot toppen.
            yl = max(1e-9, float(getattr(self.WP, "year_len", 12.0)))
            ph0 = float(getattr(self.WP, "season_phase0", 0.0))
            frac = ((float(self.t) / yl) - ph0) % 1.0
            _bp = float(getattr(ag, "_breed_phase_real", ag.pheno.breed_phase))
            d = 2.0 * math.pi * (frac - 0.25 - _bp)
            if math.exp(k * (math.cos(d) - 1.0)) < 0.5:
                return False

        return True

    def _mating_mode_slot(self, slot: int) -> bool:
        """
        Mjuk store-baserad signal för mating-orienterat beteende.
    
        Detta används för sensing/steering, inte som slutlig biologisk gate.
        Den hårda gaten ligger i _ready_to_reproduce_slot().
        """
        s = int(slot)
        store = self.store
    
        if s < 0 or s >= int(store.n):
            return False
        if not bool(store.alive[s]):
            return False
        if int(store.kind[s]) != 0:
            return False
    
        ag = self._agent_for_slot(s)
        if ag is None:
            return False
    
        if bool(store.gestating[s]):
            return False
        if float(store.repro_cd[s]) > 0.0:
            return False
        if float(store.age[s]) < float(ag.pheno.A_mature):
            return False
    
        return True
        
    # -----------------------
    # births
    # -----------------------
    def _spawn_child(
        self,
        parent: Agent,
        ctx: StepCtx,
        child_M_from_parent: float,
        child_E_fast_J: float,
        child_E_slow_J: float,
        other_parent: Optional[Agent] = None,
    ) -> Agent:
        dx = float(self.rng.normal(0.0, float(self.PP.spawn_jitter_r)))
        dy = float(self.rng.normal(0.0, float(self.PP.spawn_jitter_r)))

        if other_parent is not None:
            g_child = recombine(parent.genome, other_parent.genome, rng=self.rng, cfg=self.MC)
        else:
            g_child = child_genome_from_parent(parent.genome, rng=self.rng, cfg=self.MC)

        x_child, y_child = self.grid.wrap_pos(
            float(parent.x) + dx,
            float(parent.y) + dy,
        )
        child = Agent(
            AP=self.AP,
            genome=g_child,
            x=x_child,
            y=y_child,
            heading=float(self.rng.uniform(-math.pi, math.pi)),
        )
        child.bind_grid(self.grid)
        child.bind_world(self.world)
        child.bind_store(self.store)        
        child.bind_wrapper_lookup(self._agent_for_slot)

        child.birth_t = float(self.t)

        key = (tuple(g_child.layer_sizes), str(g_child.act))
        bank = self._banks.get(key)
        if bank is None:
            bank = ParamBank.create(key[0], key[1], capacity=int(self.PP.max_pop))
            self._banks[key] = bank

        slot = bank.alloc()
        bank.write_genome(slot, g_child)

        child._policy_key = key
        child._policy_slot = slot

        child.init_newborn_state(
            parent.pheno,
            child_M_from_parent=child_M_from_parent,
            child_E_fast_J=child_E_fast_J,
            child_E_slow_J=child_E_slow_J,
        )

        self._ensure_store_capacity(1)
        store_slot = self.store.alloc_slot()
        child.store_slot = int(store_slot)
        self.store.write_agent(store_slot, child, self.grid)
        self._write_spatial_to_store(store_slot, child.x, child.y)
        self.store.age[store_slot] = np.float32(0.0)
        self._slot_to_agent[int(store_slot)] = child

        self.store.repro_cd[store_slot] = np.float32(float(self.AP.repro_cooldown_s))
        self.store.gestating[store_slot] = False
        self.store.gest_M[store_slot] = np.float32(0.0)
        self.store.gest_E_J[store_slot] = np.float32(0.0)
        self.store.gest_M_target[store_slot] = np.float32(0.0)

        self._emit_birth(self.t, child, parent)
        return child

    def _try_mating(self, agent: Agent, ctx: StepCtx, candidates: list | None = None) -> None:
        """
        Sexuell reproduktion: verkställer parning med den agent som apply_outputs()
        lokalt detekterade och markerade via sensing-cachen.
        Ingen global sökning — agenten kan bara para sig med någon den uppfattat via sensing.
        """
        if not agent.body.alive:
            return
        
        a_slot = int(agent.store_slot)
        if a_slot < 0:
            return
        
        store = self.store
        
        if not self._ready_to_reproduce_slot(a_slot):
            return
    
        hit = getattr(agent, "_cached_agent_hit", None)
        if not isinstance(hit, tuple) or len(hit) < 5:
            return
    
        _, _, _, hit_slot, desired_id = hit[:5]
        if int(hit_slot) < 0 or int(desired_id) <= 0:
            return
    
        store = self.store
        s = int(hit_slot)
        if s >= int(store.n):
            return
        if not bool(store.alive[s]) or int(store.kind[s]) != 0:
            return
        if int(store.id[s]) != int(desired_id):
            return
    
        best = self._agent_for_slot(s)
        if best is None or best is agent:
            return
        
        b_slot = int(best.store_slot)
        if b_slot < 0:
            return
        if not self._ready_to_reproduce_slot(b_slot):
            return
    
        if self._slot_distance(a_slot, b_slot, squared=True) > float(self.PP.mating_radius) ** 2:
            return
    
        if self.PP.compat_enabled:
            compat = genetic_compatibility(
                agent.genome, best.genome, sigma=float(self.PP.compat_sigma)
            )
            if self.rng.random() > compat:
                return
    
        if best.body.M > agent.body.M:
            bearer, partner = best, agent
        else:
            bearer, partner = agent, best
    
        bearer._mating_partner = partner
        bearer.start_gestation()
        
        b_slot = int(bearer.store_slot)
        p_slot = int(partner.store_slot)
        
        if b_slot >= 0:
            self._write_gestation_to_store(b_slot, bearer)
        
        mating_cost = 0.05 * partner.body.E_cap()
        partner.pay_repro_cost(mating_cost)
        
        if p_slot >= 0:
            self._write_body_surface_to_store(p_slot, partner)
            store.repro_cd[p_slot] = np.float32(float(self.AP.repro_cooldown_s))
        
    def _try_birth(self, parent: Agent, ctx: StepCtx) -> Optional[Agent]:
        if not parent.body.alive:
            return None

        p_slot = int(parent.store_slot)
        if p_slot < 0:
            return None
        
        store = self.store
        b = parent.body
        
        # Ägarskap: Body äger gestationstillståndet fram till Steg 5, eftersom
        # gest_M ackumuleras inne i Body.step():s energibudget. Store-fälten är
        # en envägscache som uppdateras efter body-passet och efter parning.
        # Grinden nedan läser cachen; själva värdena tas ur Body, som är källan.
        if not bool(store.gestating[p_slot]):
            return None
        if not (float(store.gest_M[p_slot]) >= float(store.gest_M_target[p_slot]) > 0.0):
            return None
        
        child_M = float(b.gest_M)
        
        b.abort_gestation()
        
        store.gestating[p_slot] = False
        store.gest_M[p_slot] = np.float32(0.0)
        store.gest_E_J[p_slot] = np.float32(0.0)
        store.gest_M_target[p_slot] = np.float32(0.0)

        # --- 1.4: Energi till barnet (dras från föräldern) ---
        # child_E_fast/slow är fraktioner av barnets energikapacitet — en livshistoriestrategi.
        # Barnets Ecap beräknas från dess massa och den delade AP-konstanten.
        child_Ecap = float(self.AP.E_cap_per_M) * max(child_M, float(self.AP.M_min))
        child_E_fast_J = float(parent.pheno.child_E_fast) * child_Ecap
        child_E_slow_J = float(parent.pheno.child_E_slow) * child_Ecap

        # Föräldern betalar barnenergin ur sina egna buffrar.
        # pay_repro_cost() anropar body.take_energy() — aldrig mer än vad som finns.
        total_child_E = child_E_fast_J + child_E_slow_J
        # Överföring, inte förbränning: massan lämnar föräldern och hamnar i
        # barnet. Att bokföra den som bränd vore att utsöndra den två gånger.
        paid_to_child = float(parent.pay_repro_cost(total_child_E, transfer=True))
        
        if total_child_E > 1e-12:
            scale = paid_to_child / total_child_E
        else:
            scale = 0.0
        child_E_fast_J *= scale
        child_E_slow_J *= scale
        
        repro_cost_J = float(parent.pheno.repro_cost) * float(parent.body.E_cap())
        parent.pay_repro_cost(repro_cost_J)
        
        self._write_body_surface_to_store(p_slot, parent)

        other_parent = getattr(parent, "_mating_partner", None)
        # Rensa referensen så den inte hänger kvar
        parent._mating_partner = None

        child = self._spawn_child(
            parent,
            ctx,
            child_M_from_parent=child_M,
            child_E_fast_J=child_E_fast_J,
            child_E_slow_J=child_E_slow_J,
            other_parent=other_parent,
        )

        # Fostret bars som labil vävnad. Vid födseln får barnet sin egen
        # strukturandel, och vävnaden binder mindre näring per kilo än
        # reserven gjorde. Mellanskillnaden utsöndras till moderns cell.
        s_child = float(getattr(child.pheno, "structure", 0.25))
        d_nut = float(child_M) * (NUTRIENT_PER_KG_LABILE - nutrient_content(s_child))
        if d_nut > 0.0:
            self.world.add_nutrient(int(self.grid.cell_of(float(parent.x), float(parent.y))), d_nut)

        self._flush_body_outputs(parent)
        self._flush_body_outputs(child)

        store.repro_cd[p_slot] = np.float32(float(self.AP.repro_cooldown_s))
        return child


    def _slot_energy_per_kg(self, slot: int) -> float:
        """
        Användbar energi per kilo för en organism, given dess strukturandel.

        Samma tal styr organismens egen reserv och vad en betare får ut per
        kilo — den som är seg att äta har också mindre att tära på själv.
        """
        struct = float(self.store.structure[int(slot)])
        return float(self.WP.E_labile_J_per_kg) * (1.0 - struct)

    def _init_flora_slot(
        self,
        slot: int,
        cell: int,
        mass: float,
        traits: np.ndarray,
    ) -> None:
        struct = structure_fraction(traits)
        e_per_kg = energy_density(traits, float(self.WP.E_labile_J_per_kg))

        r_alloc = np.float32(flora_repro_alloc(traits))
        rho = np.float32(flora_root_alloc(traits))
        mat = np.float32(flora_maturity_frac(traits))
        a_mass = np.float32(flora_adult_mass(traits, mass_scale=float(self.WP.B_K)))
        t_opt = np.float32(flora_temp_opt(traits))
        t_width = np.float32(flora_temp_width(traits))
        appar = np.float32(flora_apparatus(traits))
        s_mass = np.float32(flora_seed_mass(traits))

        self.store.id[slot] = int(next_organism_id())
    
        self.store.alive[slot] = True
        self.store.kind[slot] = 1
        self.store.cell_idx[slot] = int(cell)
        self.store.pos_x[slot] = self.grid.cell_center_x[int(cell)]
        self.store.pos_y[slot] = self.grid.cell_center_y[int(cell)]
    
        m = float(mass)
        self.store.mass[slot] = np.float32(m)
        self.store.energy[slot] = np.float32(m * e_per_kg)
        self.store.age[slot] = np.float32(0.0)
        self.store.genome_idx[slot] = -1
        self.store.traits[slot, :] = np.asarray(traits, dtype=np.float32)
    
        self.store.flora_repro_alloc[slot] = r_alloc
        self.store.flora_root_alloc[slot] = rho
        self.store.flora_maturity[slot] = mat
        self.store.flora_root_mass[slot] = np.float32(float(mass) * float(rho))
        self.store.flora_adult_mass[slot] = a_mass
        self.store.flora_reserve[slot] = 0.0
        self.store.flora_repro_pool[slot] = 0.0
        self.store.flora_carbon_pool[slot] = 0.0
        self.store.flora_temp_opt[slot] = t_opt
        self.store.flora_temp_width[slot] = t_width
        self.store.flora_apparatus[slot] = appar
        self.store.flora_seed_mass[slot] = s_mass
        
        # Härled enkla store-kapaciteter från traits istället för hårdkodade 1.0/0.0
        self.store.uptake_capacity[slot] = np.float32(flora_uptake_capacity(traits))
        # growth_capacity har ingen läsare och sedan tillväxten blev
        # inkomstbegränsad inte heller något locus. Den hör till Steg 6b, där
        # kapacitetsfälten får läsare och kostnader.
        self.store.growth_capacity[slot] = np.float32(0.0)
        self.store.dispersal_capacity[slot] = appar

        self.store.sense_radius[slot] = np.float32(0.0)
        self.store.sense_rate[slot] = np.float32(0.0)
        self.store.mobility[slot] = np.float32(0.0)
        self.store.attack_capacity[slot] = np.float32(0.0)
        self.store.repair_capacity[slot] = np.float32(0.0)
        self.store.repro_capacity[slot] = np.float32(flora_repro_capacity(traits))
        self.store.structure[slot] = np.float32(struct)
    
        self.store.flood_tolerance[slot] = np.float32(0.0)
        self.store.buoyancy[slot] = np.float32(0.0)
        
    def _fauna_release_now(self) -> bool:
        """
        Är världen redo att ta emot faunan?

        `fauna_at >= 0` är ett tickvärde. `fauna_at < 0` betyder **jämvikt**:
        sätt in djuren när floramassan slutat förändras.

        Skälet att detektera i stället för att gissa: faunan mätt mot en
        halvfärdig flora mäter fel sak. Sådden ger omkring hälften av
        jämviktens stående gröda, produktionen skalar med den, och ett bestånd
        som ligger under bärkraften i den färdiga världen ligger över den i den
        halvfärdiga. Det gjorde både p87 och p97 ogiltiga.

        Ett fast tal räcker inte heller, eftersom jämvikten infaller olika sent
        vid olika bördighet: sådden ligger närmare målet ju rikare marken är,
        men gallringen tar också längre tid när plantorna är fler. Det tal som
        låg i `scenario.py` var mätt vid faktor 1 och 4 och var en gissning
        utanför det intervallet.

        Kriteriet är relativ förändring per fönster. Vid `flora_eq_tol = 0,01`
        och `flora_eq_window = 1000` krävs att floramassan rör sig mindre än en
        procent per tusen tick — uppmätt låg den på 2,5 procent per 5 000 tick
        vid tick 30 000, alltså långt under, medan den under uppbyggnaden
        ändras tiotals procent per fönster.
        """
        want = int(self.PP.fauna_at)
        if want >= 0:
            return self._tick >= want

        win = max(1, int(self.PP.flora_eq_window))
        if self._tick < win or (self._tick % win) != 0:
            return False

        n_ = int(self.store.n)
        fl = (self.store.alive[:n_]) & (self.store.kind[:n_] == 1)
        m = float(np.sum(self.store.mass[:n_][fl], dtype=np.float64))
        cnt = float(np.count_nonzero(fl))

        prev = self._flora_eq_prev
        self._flora_eq_prev = (m, cnt)
        if prev is None or m <= 0.0 or cnt <= 0.0:
            return False
        pm, pc = prev

        # **Båda** måste stå still. Massan ensam räcker inte: under
        # gallringsfasen dör många små plantor bort medan de överlevande växer
        # lika mycket som de döda tappar, så massan passerar en platå mitt i
        # förloppet. I p107 utlöste detektorn vid tick 4 000 på 0,45 procents
        # massändring — medan antalet samtidigt föll nitton procent, och den
        # verkliga jämvikten låg trettio procent högre i massa.
        #
        # Antalet är monotont under gallringen och avslöjar den direkt.
        rel_m = abs(m - pm) / max(m, 1e-9)
        rel_c = abs(cnt - pc) / max(cnt, 1e-9)
        rel = max(rel_m, rel_c)
        hard = int(self.PP.flora_eq_max_ticks)
        if rel <= float(self.PP.flora_eq_tol) or (hard > 0 and self._tick >= hard):
            print(f"[flora] jämvikt vid tick {self._tick}: {m:.0f} kg i "
                  f"{int(cnt)} plantor, ändring {100*rel_m:.2f} % massa / "
                  f"{100*rel_c:.2f} % antal per {win} tick",
                  flush=True)
            return True
        return False

    def _fauna_spawn_pos(self) -> tuple[float, float]:
        """
        Var ett insatt djur hamnar.

        Jämn utspridning garanterar Allee-fällan från tick noll. Synellipsen
        täcker 38 celler av 16 384 — 0,23 procent — så tjugo djur utspridda
        över hela världen ser varandra fyra procent av tickarna. Uppmätt i
        rökprovet: 99,3 procent av agenttickarna utan en enda artfrände i
        sikte, och reproduktionen upphör under omkring tio individer.

        Ett grundarbestånd anländer tillsammans. Med `fauna_spawn_radius` satt
        sätts djuren i en fläck, och den lokala tätheten blir världens täthet
        gånger kvoten mellan världsarea och fläckarea. Tjugo djur i tusen
        celler ger ungefär femtio procents mötesfrekvens i stället för fyra.

        Fläcken köper ett etableringsfönster, inte en lösning: tusen celler
        producerar omkring 3 400 kg per år och bär ett par individer, så
        beståndet måste sprida ut sig och möter då samma geometri igen. Vad
        den gör är att skilja de två felen åt — dör beståndet trots parningar
        under fönstret är det spridningen och inte fodret.

        Radie <= 0 ger jämn utspridning över hela världen.
        """
        r = float(self.PP.fauna_spawn_radius)
        if r <= 0.0:
            return self.grid.random_position(self.rng)

        # Flera grundargrupper. En enda fläck ger en enda linje: mätt i p91
        # tog två grundarlinjer av tjugo 73 procent av avkommorna, och den
        # effektiva populationsstorleken låg på fem. Skilda fläckar ger
        # genetisk struktur från start.
        nfl = max(1, int(getattr(self.PP, "fauna_spawn_patches", 1)))
        if self._fauna_spawn_centre is None:
            self._fauna_spawn_centre = [self.grid.random_position(self.rng)
                                        for _ in range(nfl)]
        cent = self._fauna_spawn_centre
        cx, cy = cent[int(self.rng.integers(0, len(cent)))]

        # Likformigt i skivan, inte i polära koordinater — sqrt-transformen
        # hindrar klumpning mot centrum.
        ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
        rad = r * math.sqrt(float(self.rng.uniform(0.0, 1.0)))
        return self.grid.wrap_pos(cx + rad * math.cos(ang), cy + rad * math.sin(ang))

    def seed_fauna(self, n: int) -> int:
        """
        Skapa `n` djur och sätt in dem i världen. Returnerar antalet skapade.

        Bruten ut ur `__init__` för att insättningen ska kunna fördröjas.
        Faunan mätt mot en halvfärdig flora mäter fel sak: sådden gav 56
        procent av jämviktens stående gröda, produktionen skalar med den, och
        tjugo djur låg därför över bärkraften just då fastän de legat under
        den i den färdiga världen.
        """
        _h_dim = max(0, int(self.PP.h_dim))
        in_dim = int(Agent.OBS_DIM) + _h_dim
        out_dim = int(Agent.OUT_DIM) + _h_dim
        made = 0
        for _ in range(int(n)):
            # Initiera traits först, härleda fenotyp för att få rätt arkitektur,
            # skapa sedan nätverket med korrekt form.
            import numpy as _np
            # Initiera traits med uniform fördelning i FENOTYPRYMDEN (u-domänen).
            # Standardmetoden uniform(-1,1) + sigmoid komprimerar till u∈(0.27,0.73)
            # och gör extremfenotyper (litet nätverk, hög reparation, etc.) praktiskt
            # omöjliga vid start. Med logit-transformationen är alla fenotypvärden
            # lika sannolika, vilket ger en verkligt diversifierad startpopulation.
            #
            # u ~ uniform(eps, 1-eps)  →  trait = logit(u) = log(u/(1-u))
            # sigmoid(trait) = u  →  lerp(min, max, u) är uniformt fördelat i [min,max]
            _eps = 0.02   # marginaler för att undvika logit(0) och logit(1)
            _u   = self.rng.uniform(_eps, 1.0 - _eps, int(self.PP.n_traits)).astype(_np.float64)
            raw_traits = _np.log(_u / (1.0 - _u)).astype(_np.float32)  # logit

            # Grupper med egen tyngdpunkt i traitrymden.
            #
            # Varje grundares genom slumpades oberoende, så tre fläckar
            # innehöll tre stickprov ur samma fördelning: geografiskt åtskilda
            # men genetiskt identiska. Med en egen tyngdpunkt per grupp och
            # liten spridning inom den får man verkliga raser redan vid
            # introduktionen.
            #
            # Kvoten mellan `grupp_avstand` och `grupp_spridning` avgör om det
            # blir raser eller arter, och det är just den kvoten som är
            # intressant att variera: den säger när grupperna slutar utbyta
            # gener. Kombinerat med den sociala fassynkroniseringen blir
            # isoleringen dubbel — grupperna skiljs både genetiskt och
            # reproduktivt, eftersom varje flock drar mot sin egen säsong.
            # Det är ungefär så artbildning faktiskt börjar.
            _gsep = float(getattr(self.PP, "founder_group_sep", 0.0))
            _gspr = float(getattr(self.PP, "founder_group_spread", 1.0))
            _ngrp = max(1, int(getattr(self.PP, "fauna_spawn_patches", 1)))
            if _gsep > 0.0 and _ngrp > 1:
                if self._founder_centroids is None:
                    self._founder_centroids = [
                        self.rng.normal(0.0, _gsep, int(self.PP.n_traits)).astype(_np.float32)
                        for _ in range(_ngrp)
                    ]
                _g = int(len(self.agents)) % _ngrp
                raw_traits = (self._founder_centroids[_g]
                              + raw_traits * _gspr).astype(_np.float32)

            # Grundarnas sociability kan sättas explicit. Reflexen använder
            # `soc_bias = 2·soc − 1`, så nollpunkten ligger vid 0,5: halva en
            # uniformt slumpad startpopulation styr *bort* från artfränder.
            # Vid detektion i storleksordningen tre procent av agenttickarna
            # uttrycks driften för sällan för att selektionen ska hinna verka
            # innan beståndets öde avgjorts, så locus driver neutralt.
            #
            # Samma logit-invers som ovan. Locus muteras normalt vidare, så
            # avkomman kan avvika — det är startfördelningen som styrs.
            # Grundarnas sociability kan styras. Reflexen använder
            # `soc_bias = 2·soc − 1`, så nollpunkten ligger vid 0,5: halva en
            # uniformt slumpad startpopulation styr *bort* från artfränder.
            #
            # Värdet sätter **medelvärdet**, inte alla individers värde. Att ge
            # varje grundare samma tal var fel: i p91 mättes standardavvikelsen
            # bland 264 avkommor till 0,0017 efter hundra år, och de tre
            # nivåerna i experimentet var därmed tre konstanter snarare än tre
            # startfördelningar. Selektionen hade ingenting att arbeta på.
            #
            # `sociability_init_sd` är spridningen i **logit-rymden**, samma
            # skala som mutationssteget. En grundarpopulation har en stående
            # genetisk variation som speglar artens historia, och den ska vara
            # väsentligt större än vad mutation hinner skapa: uppmätt effektiv
            # mutations-sd är σ√p per generation, och startspridningen bör ligga
            # en till två tiopotenser över.
            _soc0 = self.PP.sociability_init
            if _soc0 is not None:
                _us = min(max(float(_soc0), 1e-4), 1.0 - 1e-4)
                _sd = max(0.0, float(self.PP.sociability_init_sd))
                _l = math.log(_us / (1.0 - _us))
                if _sd > 0.0:
                    _l += float(self.rng.normal(0.0, _sd))
                raw_traits[_T_SOC] = _np.float32(_l)

            pheno_tmp  = derive_pheno(raw_traits)
            h1 = int(pheno_tmp.hidden_1)
            h2 = int(pheno_tmp.hidden_2)

            g = MLPGenome(
                layer_sizes=[in_dim, h1, h2, out_dim],
                act="tanh",
                h_dim=_h_dim,
            )
            g.traits = raw_traits
            g.init_random(self.rng, init_traits_if_missing=False)

            x, y = self._fauna_spawn_pos()

            a = Agent(
                AP=self.AP,
                genome=g,
                x=x,
                y=y,
                heading=float(self.rng.uniform(-math.pi, math.pi)),
            )
            a.bind_grid(self.grid)
            a.bind_world(self.world)
            a.bind_store(self.store)
            a.bind_wrapper_lookup(self._agent_for_slot)

            # --- Warm start: åldersstrukturerad population ---
            # Ålder uniform från nyfödd till 3× mognadsåldern —
            # ger realistisk blandning av unga, vuxna och gamla.
            A_mature  = float(a.pheno.A_mature)
            age_s     = float(self.rng.uniform(0.0, 3.0 * A_mature))
            a.birth_t = float(self.t - age_s)

            # Massa korrelerad med ålder och M_target
            child_M_mid = 0.12
            # Vuxenmassa hämtas från agentens genetiska program
            adult_M = float(getattr(a.pheno, "M_target", float(self.AP.M0)))
            adult_M *= float(self.rng.uniform(0.7, 1.0))  # lite variation
            if age_s < A_mature:
                frac = age_s / max(A_mature, 1e-9)
                M0   = child_M_mid + frac * (adult_M - child_M_mid)
            elif age_s < 2.0 * A_mature:
                M0 = adult_M
            else:
                shrink = (age_s - 2.0 * A_mature) / max(A_mature, 1e-9)
                M0     = adult_M * max(0.5, 1.0 - 0.2 * shrink)
            a.body.M = max(float(self.AP.M_min),
                           M0 * float(self.rng.uniform(0.9, 1.1)))

            # Slitage W: ackumulerat vid denna ålder
            a.body.W = (float(self.AP.wear_a0) * age_s
                        * float(self.rng.uniform(0.8, 1.2)))

            # Skada D: låg för unga, stigande för gamla
            import math as _math
            R_frac = _math.exp(-float(self.AP.repair_W_decay) * a.body.W)
            D_bg   = max(0.0, 1.0 - R_frac) * float(self.rng.uniform(0.0, 0.5))
            a.body.D = min(float(self.AP.D_max) * 0.8, D_bg)

            # Energi: varierat
            a.body.scale_energy(self.rng.uniform(0.4, 0.95))
            a.body.clamp_energy_to_cap()

            # Cooldown: spridd
            init_repro_cd = float(
                self.rng.uniform(0.0, float(self.AP.repro_cooldown_s)))

            # allocate slot in bank and write genome params once
            key = (tuple(g.layer_sizes), str(g.act))
            bank = self._banks.get(key)
            if bank is None:
                bank = ParamBank.create(key[0], key[1], capacity=int(self.PP.max_pop))
                self._banks[key] = bank

            slot = bank.alloc()
            bank.write_genome(slot, g)

            a._policy_key = key
            a._policy_slot = slot

            self._ensure_store_capacity(1)
            store_slot = self.store.alloc_slot()
            a.store_slot = int(store_slot)
            self.store.write_agent(store_slot, a, self.grid)
            self._write_spatial_to_store(store_slot, a.x, a.y)
            self.store.age[store_slot] = np.float32(age_s)
            self._slot_to_agent[int(store_slot)] = a

            self.store.repro_cd[store_slot] = np.float32(init_repro_cd)
            self._write_gestation_to_store(store_slot, a)

            self._emit_birth(self.t, a, parent=None)
            self.agents.append(a)
            made += 1

        if made > 0 and int(getattr(self, "_tick", 0)) > 0:
            # Djur som sätts in efter tick noll får sin massa gratis, precis
            # som utgångstillståndet gör. Utan den här posten ser insättningen
            # ut som en läcka i näringsbalansen — invariantsviten fångade det
            # på 2,2e-4 relativt första gången.
            add = 0.0
            for a in self.agents[-made:]:
                if not a.body.alive:
                    continue
                slot = int(getattr(a, "store_slot", -1))
                st = float(self.store.structure[slot]) if slot >= 0 else 0.25
                add += float(a.body.M) * nutrient_content(st)
                add += a.body.M_reserve() * NUTRIENT_PER_KG_LABILE
                add += float(a.body.gest_M) * NUTRIENT_PER_KG_LABILE
            self.world._nutrient_added_total += add

        return made

    def _seed_initial_flora(
        self,
        n_flora: int | None = None,
        target_mass: float | None = None,
        init_mass_frac_lo: float = 0.4,
        init_mass_frac_hi: float = 1.0,
    ) -> int:
        """
        Skapa initial diskret flora direkt i OrganismStore.

        Med `target_mass` sås individer tills den totala floramassan nås, i
        stället för till ett fast antal. Det gör utgångsläget till en
        ekologisk storhet — primärproduktion relativt konsumenter — och håller
        det invariant när `B_K` eller `init_pop` ändras. Antalet faller ut ur
        massaskalan i stället för att sättas separat.
        """
        n_cells = int(self.grid.n_cells)
        BK = float(self.WP.B_K)
        if BK <= 0.0:
            return 0

        spread_mean = 0.0
        if target_mass is not None:
            want = max(0.0, float(target_mass))
            # Sprid över **alla** celler och låt massan per individ falla ut,
            # i stället för tvärtom. Med fast massandel 0,4–1,0 av B_K blev
            # plantorna 7,7 kg och rymdes i 4 172 celler av 16 384: tre
            # fjärdedelar av världen fångade inget ljus alls, medan varje sådd
            # planta hade bladarea 3,0 i en cell som mättas vid 1,0 och därmed
            # kastade bort två tredjedelar av sitt eget.
            #
            # Men steget stannade vid en planta per cell, och det är en planta
            # för lite. Antalet sattes då av världens storlek och massan per
            # individ föll ut som kvot — vid 130 170 kg blev plantorna 7,94 kg
            # mot jämviktens 1,32, alltså sextusen jättar där det borde stå
            # nittiotusen småplantor. Betningshorisonten verkar per planta, så
            # en betare mötte en helt annan värld än den som faktiskt uppstår.
            #
            # Hur många plantor en cell rymmer ska avgöras av konkurrensen
            # mellan rotsystemen, inte av sådden. Den mekanismen finns redan:
            # anspråket är rotarea mot cellarea 1, överskott spiller till de
            # sex grannarna, och summan skalas ner där marken är slut. Sådden
            # sätter därför bara en rimlig plantstorlek och låter antalet följa.
            per_plant = max(1e-9, float(self.PP.flora_init_plant_mass))
            n_flora = max(1, int(round(want / per_plant)))

            # Allokera inte plantor marken inte kan betala för. Sådden köps ur
            # den fria näringspoolen, så taket är känt i förväg: fri näring
            # dividerat med vad en planta av den här storleken binder. Utan
            # kapet allokerades 333 333 slots för en sådd som marken sänkte
            # till 82 885 plantor, och store:n växte till en halv miljon.
            free = float(np.sum(np.asarray(self.world.nutrient), dtype=np.float64))
            cost_each = per_plant * nutrient_content(0.5)
            if cost_each > 0.0:
                n_flora = min(n_flora, max(1, int(free / cost_each)))
            spread_mean = want / n_flora
        elif n_flora is None:
            n_flora = max(16, int(self.PP.max_pop) // 2)

        n_flora = max(0, int(n_flora))
        if n_flora <= 0:
            return 0

        # Med återläggning. Utan den kunde en cell få högst en planta, vilket
        # var hela begränsningen. Cellerna blir olika täta av slumpen, vilket
        # är rimligare än exakt en var — och rotkonkurrensen sorterar det inom
        # några hundra tick, precis som den gör med spridningen i drift.
        cells = self.rng.choice(n_cells, size=n_flora, replace=True)

        created = 0
        placed = 0.0
        for cell in cells:
            self._ensure_store_capacity(1)
            slot = self.store.alloc_slot()

            traits = init_organism_traits(
                self.rng,
                int(self.PP.n_traits),
                mode="flora",
            )

            if spread_mean > 0.0:
                # Spridning kring medelvärdet behålls, så att sådden inte är
                # en klon av identiska individer.
                mass = spread_mean * float(self.rng.uniform(0.7, 1.3))
            else:
                mass = BK * float(self.rng.uniform(init_mass_frac_lo, init_mass_frac_hi))
            self._init_flora_slot(int(slot), int(cell), float(mass), traits)

            # Sådden betalas ur marken. Tidigare myntade den näring: hela
            # stocken kom in i världen som vävnad i den sådda floran, vilket
            # gjorde `flora_init_mass_ratio` till bördighetsreglage. Nu bär
            # `nutrient_init` bördigheten och sådden är en starttäthet.
            need = float(self.store.mass[slot]) * nutrient_content(
                float(self.store.structure[slot])
            )
            paid = self.world.take_nutrient(int(cell), need)
            if paid < need:
                # Cellen kunde inte betala. Krymp plantan så att bunden näring
                # svarar mot vad som faktiskt fanns.
                nc = max(1e-12, nutrient_content(float(self.store.structure[slot])))
                self.store.mass[slot] = np.float32(paid / nc)
                self.store.energy[slot] = np.float32(max(
                    0.0,
                    float(self.store.mass[slot]) * self._slot_energy_per_kg(int(slot)),
                ))

            created += 1
            placed += float(self.store.mass[slot])

        if target_mass is not None and placed < float(target_mass) * 0.999:
            # Marken kunde inte betala. Sedan sådden köps ur den fria
            # näringspoolen är det budgeten och inte cellrymden som sätter
            # taket, och taket är då världens verkliga bördighet. Tyst
            # avvikelse här vore ett dolt designfel; hellre en synlig
            # anmärkning.
            print(f"[flora] sådd nådde {placed:.3f} kg av begärda "
                  f"{float(target_mass):.3f} kg i {created} plantor — "
                  f"marken hade inte näring för mer")

        return created
        
            
    def _consume_flora_from_store(self, x: float, y: float, amount: float, max_radius: int = 1) -> tuple[float, float]:
        """
        Konsumera växtmassa från diskret flora i OrganismStore via gemensamt spatialindex.

        Returnerar (kg, energi_J). Varje betad individ bidrar med energi enligt
        sin egen strukturandel, så en seg växt ger mindre per kilo än en mjuk.
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0
    
        cell0 = int(self.grid.cell_of(float(x), float(y)))
        got = 0.0
        energy = 0.0

        # Grannskapet en gång, ur grannskapstabellen. Det gamla anropet gjorde
        # en bredden-först-sökning per radie och slog sedan ihop resultaten med
        # en mängd; `cells_within` ger redan ringarna i ordning utan dubbletter.
        ordered_cells = self.grid.cells_within(cell0, int(max_radius))

        # Funktionell respons av Hollings typ II, härledd ur att betandet tar
        # tid. Tillgänglig massa inom räckhåll räknas först; efterfrågan kapas
        # sedan av vad söktid och hanteringstid medger under en tick. Se
        # AgentParams.graze_handle_h för härledningen av talen.
        #
        # Tillgängligheten läses ur `store.flora_cell_mass`, som redan är
        # summan av `max(0, massa - rotmassa)` per cell — alltså exakt det
        # skottförråd betningshorisonten ser. Den byggdes förut om plantvis i
        # en Python-loop vid varje tugga: 13,5 plantor och 12,5 celluppslag per
        # anrop, 91 av betningens 101 mikrosekunder, för ett tal som redan låg
        # färdigt. Fältet är fräscht under faunapassen eftersom bara betningen
        # ändrar floramassa efter florapasset, och betningen skriver av sig
        # nedan.
        cells = np.asarray(ordered_cells, dtype=np.int64)
        cell_avail = self.store.flora_cell_mass[cells]
        B = float(cell_avail.sum())
        if B > 0.0:
            AP = self.AP
            a_s = float(AP.graze_search_a)
            h_t = float(AP.graze_handle_h)
            cap = float(self.WP.dt) * a_s * B / (1.0 + a_s * h_t * B)
            if cap < amt:
                amt = cap
        if amt <= 1e-12:
            return 0.0, 0.0

        # Girigheten är oförändrad men slås upp lat. Hollings tak gör tuggan
        # liten mot vad som står inom räckhåll, och mätt rör den 1,22 plantor
        # per anrop. Att slå upp alla tolv cellernas slotar för att sedan bryta
        # i den första var tolv gånger för mycket arbete. Tomma celler hoppas
        # över på fältet i stället för via spatialindexet.
        for ci, cell in enumerate(ordered_cells):
            if float(cell_avail[ci]) <= 1e-12:
                continue
            for slot in self.store.slots_in_cell(cell):
                s = int(slot)
    
                if amt <= 1e-12:
                    break
                if not bool(self.store.alive[s]) or int(self.store.kind[s]) != 1:
                    continue
    
                m = float(self.store.mass[s])
                if m <= 1e-12:
                    continue
    
                # Betningshorisonten. Betaren tar skott och lämnar roten, precis
                # som en verklig betare tar blad och lämnar meristem. Plantan
                # överlever varje passage och skjuter igen ur sin reserv.
                #
                # Skälet är mätt. Betaren tömmer inte ett grannskap — den lämnar
                # det två till sex gånger snabbare än den hinner: 18 tick att
                # passera mot 44 att beta ner ett mediangrannskap. Men den tar
                # `min(m, amt)` per *planta*, och medianplantan väger 0,197 kg
                # mot en tugga på 0,907. Varje tugga svepte därför upp flera
                # hela småplantor, och i p70 föll floran från 37 324 till 5 396
                # individer under en betningstopp — antalet i takt med massan,
                # vilket bara sker om hela växter försvinner.
                #
                # Refugen som saknades var alltså inte att lämna plantor i
                # cellen, utan att lämna en del av varje planta. Det ger
                # tröskeleffekten typ II saknar, ur en fysisk begränsning i
                # stället för en postulerad kurva — och det ger `root_alloc` en
                # ny konsekvens: rot är betesskydd, och den som satsat på skott
                # betalar för det när trycket kommer.
                edible = m - float(self.store.flora_root_mass[s])
                if edible <= 1e-12:
                    continue
    
                take = edible if edible < amt else amt
                new_m = m - take
    
                self.store.mass[s] = np.float32(new_m)
                e_kg = self._slot_energy_per_kg(s)
                self.store.energy[s] = np.float32(max(0.0, new_m * e_kg))
                got += take
                energy += take * e_kg
                amt -= take

                # Cellens skottförråd skrivs av direkt. Utan det skulle nästa
                # betare inom samma tick se massa som redan är uppäten, och
                # tillgängligheten vore inte längre den verkliga.
                self.store.flora_cell_mass[cell] = np.float32(
                    max(0.0, float(self.store.flora_cell_mass[cell]) - take)
                )
    
                if new_m <= 1e-12:
                    # Nås numera bara av plantor utan rotmassa alls. Betning kan
                    # inte längre döda en planta med rot; svältdöden i
                    # tillväxtpasset tar den som betats under sitt golv.
                    self._release_flora_slot(s)
    
            if amt <= 1e-12:
                break
    
        if got > 0.0:
            self._flora_summary_cache = None
    
        return float(got), float(energy)

    def _add_or_create_flora_in_cell(
        self,
        cell: int,
        add_mass: float,
        traits: np.ndarray | None = None,
    ) -> int:
        """Returnerar den nya individens slot, eller -1 om den inte kunde skapas."""
        dm = float(add_mass)
        if not math.isfinite(dm) or dm <= 0.0:
            return -1

        # Varje frö blir en egen individ, även om cellen redan är bebodd.
        #
        # Tidigare slogs ny massa ihop med befintlig flora i målcellen, vilket
        # gjorde varje cell till en enda organism. Då fanns ingen konkurrens:
        # individen hade cellens hela näringspool för sig själv, och
        # uptake_capacity kunde inte selekteras eftersom snabbare upptag inte
        # vann något från någon.
        #
        # Antalet begränsas inte här utan av näringen. En groddplanta som inte
        # får näring växer inte, tappar mot senescensen och försvinner.
        if len(self.store.free_slots) < 2:
            self._ensure_store_capacity(1)
        try:
            slot = self.store.alloc_slot()
        except RuntimeError:
            return -1

        if traits is None:
            traits = init_organism_traits(
                self.rng,
                int(self.PP.n_traits),
                mode="flora",
            )

        self._init_flora_slot(slot, int(cell), dm, traits)
        self._flora_summary_cache = None
        return int(slot)

    def _fauna_slots(self, rebuild: bool = False) -> np.ndarray:
        """
        Aktiv delmängd: slotindex för levande fauna.

        Fauna och flora delar store, så en fauna-loop över `range(store.n)`
        kostar efter florans antal. Vid tiotusen växter och tolv djur betalade
        metabolismpasset tusen gånger för lite arbete.
        """
        if rebuild or self._fauna_slots_cache is None:
            n = int(self.store.n)
            if n <= 0:
                self._fauna_slots_cache = np.empty(0, dtype=np.int64)
            else:
                self._fauna_slots_cache = np.flatnonzero(
                    self.store.alive[:n] & (self.store.kind[:n] == 0)
                ).astype(np.int64, copy=False)
        return self._fauna_slots_cache

    def _flora_slots(self, rebuild: bool = False) -> np.ndarray:
        """
        Aktiv delmängd: slotindex för levande flora.

        Byggs en gång per tick och betraktas som immutabel under ticken, enligt
        manifestets ticksemantik. Senescens och spridning under ticken ändrar
        inte delmängden — de pass som körs efter maskar i stället bort de
        slotar som hunnit dö.
        """
        if rebuild or self._flora_slots_cache is None:
            n = int(self.store.n)
            if n <= 0:
                self._flora_slots_cache = np.empty(0, dtype=np.int64)
            else:
                live = self.store.alive[:n]
                self._flora_slots_cache = np.flatnonzero(
                    live & (self.store.kind[:n] == 1)
                ).astype(np.int64, copy=False)
        return self._flora_slots_cache

    def _growth_system_flora_numpy(self) -> tuple[float, float, float]:
        """
        Florans livscykel i ett pass: förnafall, död, upptag, allokering, tillväxt.

        Referensimplementationen. `_growth_system_flora_numba` gör samma sak
        med en kompilerad kärna och ska ge samma utfall; se `flora_growth.py`.

        **Inkomst skild från allokering.** En planta tog tidigare upp exakt den
        näring den i samma tick kunde omsätta till massa, och kunde därför
        varken spara, underhålla eller avsätta. Nu går inkomsten till en reserv,
        `flora_reserve`, och en andel `flora_repro_alloc` av den viks av till
        `flora_repro_pool`, som spridningen betalar ur. Reserven är florans
        motsvarighet till faunans energireserv.

        **Förnafallet** är det underhållsflöde som saknades. En levande planta
        fäller förna varje tick i en takt som avtar med strukturandelen — tunna
        billiga blad omsätts fort, seg vävnad långsamt. Det ger tre saker på en
        gång: en underhållskostnad, näring som cirkulerar utan att någon behöver
        dö, och `structure` som verklig långsam–snabb-axel i stället för bara en
        näringsrabatt.

        **Döden** har två vägar. Åldrandet är en hazard ur `flora_lifespan`,
        härledd ur strukturandelen i stället för ur en konstant: `flora_mortality`
        på 2e-5 per månad gav en medellivslängd på 4 167 år. Svälten är
        emergent — den som inte kan växa snabbare än sitt förnafall krymper, och
        under `flora_min_mass_frac · B_K` räknas den som död. Det ersätter den
        påslagna groddplanterisken med en mekanism.

        **Semelpari** faller ut som ytterligheten av allokeringsandelen: den som
        lägger nästan allt på frön kan inte betala sitt förnafall och dör efter
        att ha reproducerat sig. Ingen artgräns kodas.

        Ordningen är förnafall, åldrande, anspråk, inkomst, allokering, tillväxt,
        svältdöd. Anspråket räknas efter förnafallet, så en krympande planta
        släpper mark samma tick som den tappar massa.

        Returnerar (producerad kg, upptagen näring kg, död massa kg).
        """
        dt = float(self.WP.dt)
        BK = float(self.WP.B_K)
        if BK <= 0.0 or dt <= 0.0:
            self.store.clear_flora_claims()
            return 0.0, 0.0, 0.0

        fl = self._flora_slots(rebuild=True)
        if fl.size == 0:
            self.store.clear_flora_claims()
            return 0.0, 0.0, 0.0

        world = self.world
        store = self.store
        grid = self.grid
        u_area = float(self.WP.uptake_rate_max)
        # Svältgolvet är relativt fröet plantan kom ur, inte bara absolut. En
        # groddplanta som tappat halva sitt förråd är död; en fullvuxen ligger
        # tiopotenser över sitt eget frö och berörs inte. Utan den relativa
        # termen krymper en groddplanta med förnafallets takt och lever i
        # storleksordningen åttio månader utan att någonsin växa, vilket fyller
        # store:n med ett groddplantslager som varken dör eller producerar.
        m_floor_abs = float(self.PP.flora_min_mass_frac) * BK

        m = store.mass[fl].astype(np.float64, copy=False)
        struct = store.structure[fl].astype(np.float64, copy=False)
        m_adult = np.maximum(1e-12, store.flora_adult_mass[fl].astype(np.float64, copy=False))
        cells = store.cell_idx[fl].astype(np.int64, copy=False)
        cost = nutrient_content_array(struct)

        died = 0.0

        # --- 1. förnafall -----------------------------------------------------
        # Massan lagras i float32. Avrundningen görs nedåt och det faktiskt
        # fällda beräknas ur den lagrade massan, så att detritus får exakt det
        # plantan förlorade — varken mer eller mindre.
        shed_want = np.minimum(m, flora_turnover_rate(struct) * dt * m)
        m_left = m - shed_want
        stored_left = m_left.astype(np.float32)
        # Samma konvertering användes fyra gånger. Varje `astype` från float32
        # till float64 kopierar hela arrayen — vid 300 000 plantor är det 2,4 MB
        # per anrop, och tillväxtpasset gjorde 29 sådana per tick. Att hissa den
        # hit ger identiskt resultat och sparar tre fulla genomgångar av minnet.
        stored_left64 = stored_left.astype(np.float64)
        up = stored_left64 > m_left
        if np.any(up):
            stored_left[up] = np.nextafter(stored_left[up], np.float32(0.0))
            stored_left64 = stored_left.astype(np.float64)
        shed = m - stored_left64
        # Förnafallet fäller proportionellt ur båda facken, så sammansättningen
        # är oförändrad. Fina rötter omsätts i verkligheten ungefär lika fort
        # som blad, så förenklingen håller hyggligt.
        keep_frac = np.where(m > 0.0, stored_left64 / np.maximum(m, 1e-30), 1.0)
        store.flora_root_mass[fl] = (
            store.flora_root_mass[fl].astype(np.float64, copy=False) * keep_frac
        ).astype(np.float32)
        shedding = shed > 0.0
        self._last_flora_shed = float(shed[shedding].sum()) if np.any(shedding) else 0.0
        if np.any(shedding):
            store.mass[fl[shedding]] = stored_left[shedding]
            world.excrete_cells(cells[shedding], shed[shedding], struct[shedding])
        m = stored_left64

        # --- 2. åldrande och svält -------------------------------------------
        hazard = dt / np.maximum(1e-6, flora_lifespan(struct))
        draws = self.rng.random(fl.size)
        m_floor = np.maximum(
            m_floor_abs,
            float(self.PP.flora_seedling_floor)
            * store.flora_seed_mass[fl].astype(np.float64, copy=False),
        )
        by_age = draws < hazard
        by_starve = (~by_age) & (m <= m_floor)
        dying = np.flatnonzero(by_age | by_starve)
        self._last_flora_died_age = int(np.count_nonzero(by_age))
        self._last_flora_died_starve = int(np.count_nonzero(by_starve))

        # Dödsfallen är få per tick och varje deponering blandar strukturandelar
        # massviktat, så de hanteras individuellt efter det vektoriserade urvalet.
        if dying.size:
            # Deponeringen görs i ett svep. Ett anrop per döende planta gav
            # tolvtusen anrop på femtonhundra tick, var och en med en egen
            # aggregering över en enda rad.
            dm = store.mass[fl[dying]].astype(np.float64, copy=False)
            keep = dm > 0.0
            if np.any(keep):
                world.excrete_cells(cells[dying][keep], dm[keep], struct[dying][keep])
                died += float(dm[keep].sum())
            for slot in fl[dying]:
                self._release_flora_slot(int(slot))

        alive_mask = np.ones(fl.size, dtype=bool)
        alive_mask[dying] = False

        T = world.temperature_of_cells(cells).astype(np.float64, copy=False)
        Topt = store.flora_temp_opt[fl].astype(np.float64, copy=False)
        Twid = np.maximum(1e-6, store.flora_temp_width[fl].astype(np.float64, copy=False))
        gate = np.exp(-0.5 * ((T - Topt) / Twid) ** 2)

        # --- 3. anspråk: en rad per planta och berörd cell ---------------------
        # Rotarean följer aktuell massa, A = m / B_K, och är härledd varje tick.
        # Fullvuxna plantor ingår men tar ingenting: marken är upptagen även av
        # den som slutat växa, och det är den mekanismen som låter ett bestånd
        # hålla undan groddplantor.
        holds = alive_mask & (cells >= 0) & (m > 0.0)
        vi = np.flatnonzero(holds)
        if vi.size == 0:
            store.clear_flora_claims()
            if died > 0.0:
                self._flora_summary_cache = None
            return 0.0, 0.0, float(died)

        n_cells = int(world.nutrient.shape[0])

        # Kroppen har en sammansättning. Ytorna följer av vad den består av,
        # och båda skalas med (1 − s): grov rot absorberar lika lite som ved
        # fotosyntetiserar. Strukturandelen får därmed sin andra nedsida.
        root = np.minimum(m, store.flora_root_mass[fl].astype(np.float64, copy=False))
        shoot = np.maximum(0.0, m - root)
        sra = float(self.WP.root_area_per_kg)
        a_root = sra * root * (1.0 - struct) / BK

        A = a_root[vi]

        row_plant = vi
        row_cell = cells[vi]
        row_claim = np.minimum(A, 1.0)

        # Överskjutande area fördelas jämnt över de sex grannarna. Att en enda
        # ring räcker följer av FloraRanges.adult_mass_k_max: med taket 4,0
        # blir A högst 4, alltså ett överskott på 3 mot ringens kapacitet 6.
        big = np.flatnonzero(A > 1.0)
        if big.size:
            nb = grid.neighbor_idx[row_cell[big]].astype(np.int64, copy=False)
            per = np.minimum(1.0, (A[big] - 1.0) / float(nb.shape[1]))
            row_plant = np.concatenate((row_plant, np.repeat(vi[big], nb.shape[1])))
            row_cell = np.concatenate((row_cell, nb.reshape(-1)))
            row_claim = np.concatenate((row_claim, np.repeat(per, nb.shape[1])))

        claimed = np.bincount(row_cell, weights=row_claim, minlength=n_cells)[:n_cells]
        # Marginalen på en ulp håller summan strikt under det som finns, så att
        # poolen aldrig kan bli negativ av avrundning.
        share_row = row_claim / np.maximum(1.0, claimed)[row_cell] * (1.0 - 1e-12)

        # Ytfördelningen behålls i stället för att kastas. `claimed` säger var
        # marken är full och `share_row` vad varje planta faktiskt håller —
        # samma tal som näringsandelen räknas ur nedan. Ingen annan kod ska
        # härleda dem på nytt; se OrganismStore.set_flora_claims.
        store.set_flora_claims(claimed, fl[row_plant], row_cell, share_row)
        avail_row = share_row * world.nutrient[row_cell]
        access = np.bincount(row_plant, weights=avail_row, minlength=fl.size)[:fl.size]

        # --- 4. inkomst -------------------------------------------------------
        # Taket är per areaenhet och grindas av temperaturen: upptag är arbete,
        # och arbetet avstannar i kyla. Efterfrågan är vad reserven ännu har
        # plats för — en planta som redan står vid vuxenmassa med full reserv
        # tar inget, men den behåller sin mark.
        A_full = a_root
        reserve = store.flora_reserve[fl].copy()
        head = np.maximum(0.0, (m_adult - m) * cost - reserve)
        cap = (np.maximum(0.0, store.uptake_capacity[fl].astype(np.float64, copy=False))
               * u_area * A_full * gate * dt)

        take = np.where(holds & (gate > 1e-6),
                        np.minimum(np.minimum(cap, access), np.maximum(head, 0.0)),
                        0.0)
        if np.any(take > 0.0):
            frac = np.zeros(fl.size, dtype=np.float64)
            ok = access > 0.0
            frac[ok] = np.minimum(1.0, take[ok] / access[ok])
            draw = frac[row_plant] * avail_row
            world.nutrient -= np.bincount(
                row_cell, weights=draw, minlength=n_cells
            )[:n_cells]

        # --- 5. allokering ----------------------------------------------------
        # Andelen viks av från inkomsten, inte från reserven. Den som avsätter
        # mycket får därför mindre kvar att växa och underhålla sig med, vilket
        # är hela avvägningen.
        alloc = np.clip(store.flora_repro_alloc[fl].astype(np.float64, copy=False)
                        * store.repro_capacity[fl].astype(np.float64, copy=False), 0.0, 1.0)
        to_pool = take * alloc
        reserve = reserve + (take - to_pool)
        store.flora_repro_pool[fl] += to_pool

        # --- 6. ljus som andra begränsande resurs -----------------------------
        #
        # Bara labil vävnad fotosyntetiserar, så bladarean följer m · (1 − s)
        # medan rotarean följer hela massan. Det är strukturandelens motkraft,
        # och den är mekanism och inte kostnadsparameter: en vedartad planta
        # bygger billigt i näring men bär mindre blad per kilo.
        #
        # Höjden är m · s enligt docs/substratets-struktur.md — strukturmassan
        # är det som håller organismen uppe. Skuggan är Beer–Lambert i cellens
        # bladarealindex, dämpad av hur hög plantan står i förhållande till
        # cellens arealviktade medelhöjd. Det ger asymmetrin utan att kräva en
        # sortering per cell: två bincounts räcker.
        k_ext = float(self.WP.light_extinction)
        h_ref = max(1e-12, float(self.WP.light_height_ref))
        L_cell = float(self.WP.light_input) * dt

        sla = float(self.WP.leaf_area_per_kg)
        leaf = sla * shoot * (1.0 - struct) / BK
        # Höjden kommer ur stammen, inte ur hela kroppen: en planta blir inte
        # längre av att bygga rot. Det var fel redan i 0061 men syns först när
        # rot och skott skiljs åt.
        height = shoot * struct

        lam = np.bincount(cells[vi], weights=leaf[vi], minlength=n_cells)[:n_cells]
        hsum = np.bincount(cells[vi], weights=leaf[vi] * height[vi], minlength=n_cells)[:n_cells]
        hbar = np.zeros(n_cells, dtype=np.float64)
        nz = lam > 0.0
        hbar[nz] = hsum[nz] / lam[nz]

        r_rel = height / (height + h_ref * np.maximum(hbar[cells], 1e-12))
        shade = np.exp(-k_ext * lam[cells] * (1.0 - r_rel))
        eff = np.where(holds, leaf * shade, 0.0)

        eff_cell = np.bincount(cells[vi], weights=eff[vi], minlength=n_cells)[:n_cells]
        light = np.where(
            holds,
            L_cell * eff / np.maximum(1.0, eff_cell)[cells] * (1.0 - 1e-12),
            0.0,
        )

        # Allokeringen delar **båda** inkomsterna. Kolet gick tidigare enbart
        # till kropp, så en bladrik planta kunde inte omsätta sin ljusfördel i
        # frön — och eftersom fröet dessutom kostade näring efter moderns
        # struktur betalade den snabba strategin 4,5 gånger mer per frö än den
        # vedartade. Avvägningen var vänd bakochfram.
        # Kolpoolen har tak. Utan det ackumulerar den obegränsat — uppmätt
        # 8 310 kg mot näringspoolens 40 efter 8 000 tick — eftersom fröet är
        # näringsrikt och därför nästan alltid näringsbegränsat. Kol som inte
        # ryms går till kropp i stället för att gå förlorat, vilket är samma
        # fel som den låsta näringspoolen var före 0059.
        c_cap = float(self.PP.flora_max_seeds_per_tick) * store.flora_seed_mass[fl].astype(
            np.float64, copy=False
        )
        head_c = np.maximum(0.0, c_cap - store.flora_carbon_pool[fl])
        to_carbon = np.minimum(light * alloc, head_c)
        store.flora_carbon_pool[fl] += to_carbon
        light_growth = light - to_carbon

        # --- 7. tillväxt: min() över resurser ---------------------------------
        # Näring ur reserven, kol ur ljuset. Liebigs minimumlag: den knappaste
        # sätter takten. Kolet lagras bara i reproduktionspoolen — fotosyntat
        # som varken byggs in eller avsätts samma tick är borta.
        can_grow = holds & (m < m_adult) & (reserve > 0.0) & (light_growth > 0.0)
        room = np.where(can_grow, m_adult - m, 0.0)
        dm_nutrient = np.where(can_grow, reserve / np.maximum(cost, 1e-12), 0.0)
        dm_want = np.minimum(room, np.minimum(dm_nutrient, np.where(can_grow, light_growth, 0.0)))

        target = m + dm_want
        stored = target.astype(np.float32)
        up = stored.astype(np.float64) > target
        if np.any(up):
            stored[up] = np.nextafter(stored[up], np.float32(0.0))

        dm = np.where(can_grow, stored.astype(np.float64) - m, 0.0)
        grew = dm > 0.0
        spent = np.where(grew, dm * cost, 0.0)
        reserve = reserve - spent

        store.flora_reserve[fl] = reserve
        if np.any(grew):
            g = np.flatnonzero(grew)
            g_slots = fl[g]
            # Tillskottet fördelas enligt locuset; kroppens sammansättning är
            # integralen av besluten. `ρ` gäller alltså tillväxten, inte
            # kroppen retroaktivt — det är formen plasticitet senare kräver.
            rho_g = store.flora_root_alloc[g_slots].astype(np.float64, copy=False)
            store.flora_root_mass[g_slots] = (
                root[g] + rho_g * dm[g]
            ).astype(np.float32)
            store.mass[g_slots] = stored[g]
            store.energy[g_slots] = (
                stored[g].astype(np.float64)
                * float(self.WP.E_labile_J_per_kg)
                * (1.0 - struct[g])
            ).astype(np.float32)

        self._last_flora_light_limited = float(
            np.mean(light_growth[can_grow] < dm_nutrient[can_grow])
        ) if np.any(can_grow) else 0.0

        produced = float(dm[grew].sum()) if np.any(grew) else 0.0
        taken = float(spent.sum())

        if produced > 0.0 or died > 0.0 or np.any(shedding):
            self._flora_summary_cache = None

        return float(produced), float(taken), float(died)

    def _flora_cell_buffers(self, n_cells: int) -> tuple:
        """Fyra återanvända cellindexerade skrivbuffertar åt tillväxtkärnan."""
        buf = self._flora_cell_buf
        if buf is None or int(buf[0].shape[0]) != int(n_cells):
            buf = tuple(np.zeros(int(n_cells), dtype=np.float64) for _ in range(4))
            self._flora_cell_buf = buf
        return buf

    def _growth_system_flora(self) -> tuple[float, float, float]:
        """
        Florans tillväxtpass. Väljer väg och delegerar.

        Semantiken ägs av `_growth_system_flora_numpy`; `_growth_system_flora_numba`
        är samma pass med aritmetiken i en kompilerad kärna. Valet görs en gång
        och faller tillbaka på den snabbaste byggda vägen om den begärda
        saknas i miljön — se `flora_growth.available_backends`.
        """
        if self._flora_growth_mode is None:
            want = str(getattr(self.PP, "flora_growth_backend", "numba")).strip().lower()
            ok = flora_growth.available_backends()
            self._flora_growth_mode = want if want in ok else ok[-1]
        if self._flora_growth_mode == "numpy":
            return self._growth_system_flora_numpy()
        return self._growth_system_flora_numba()

    def _growth_system_flora_numba(self) -> tuple[float, float, float]:
        """
        Tillväxtpasset som skal, kärna och efterspel.

        Skalet plockar ut slots och celltillstånd, kärnan i `flora_growth.py`
        räknar per planta, och efterspelet deponerar till detritus och skriver
        tillbaka. Uppdelningen följer av att kärnan inte får göra världsanrop:
        `world.temperature_of_cells()` hissas hit, och `world.excrete_cells()`
        skjuts till efterspelet. Det senare är säkert eftersom passet aldrig
        läser `detritus` — bara `nutrient`, som kärnan muterar på plats i den
        ordning numpy-vägen gör det, så att näring från en död planta hinner
        bli tillgänglig för grannarna samma tick.

        Returnerar (producerad kg, upptagen näring kg, död massa kg).
        """
        dt = float(self.WP.dt)
        BK = float(self.WP.B_K)
        if BK <= 0.0 or dt <= 0.0:
            self.store.clear_flora_claims()
            return 0.0, 0.0, 0.0

        fl = self._flora_slots(rebuild=True)
        if fl.size == 0:
            self.store.clear_flora_claims()
            return 0.0, 0.0, 0.0

        world = self.world
        store = self.store
        n = int(fl.size)
        n_cells = int(world.nutrient.shape[0])

        cells = store.cell_idx[fl].astype(np.int64, copy=False)
        struct32 = store.structure[fl]
        temp = world.temperature_of_cells(cells).astype(np.float64, copy=False)
        draws = self.rng.random(n)

        mass_out = np.empty(n, dtype=np.float32)
        root_out = np.empty(n, dtype=np.float32)
        energy_out = np.empty(n, dtype=np.float32)
        reserve_out = np.empty(n, dtype=np.float64)
        pool_out = np.empty(n, dtype=np.float64)
        carbon_out = np.empty(n, dtype=np.float64)
        shed_out = np.empty(n, dtype=np.float64)
        dying_out = np.zeros(n, dtype=np.uint8)
        dm_out = np.empty(n, dtype=np.float64)
        grow_out = np.zeros(n, dtype=np.uint8)
        claimed, lam, hsum, cellacc = self._flora_cell_buffers(n_cells)

        (shed_total, n_age, n_starve, produced, taken, died, light_lim,
         row_plant, row_cell, row_share) = flora_growth.growth_kernel(
            store.mass[fl], struct32, store.flora_adult_mass[fl],
            store.flora_root_mass[fl], store.flora_seed_mass[fl], store.energy[fl],
            store.flora_temp_opt[fl], store.flora_temp_width[fl],
            store.uptake_capacity[fl], store.flora_repro_alloc[fl],
            store.repro_capacity[fl], store.flora_root_alloc[fl],
            store.flora_reserve[fl], store.flora_repro_pool[fl],
            store.flora_carbon_pool[fl],
            cells, temp, draws,
            world.nutrient, self.grid.neighbor_idx,
            dt, BK,
            float(self.WP.uptake_rate_max),
            float(self.WP.root_area_per_kg),
            float(self.WP.leaf_area_per_kg),
            float(self.WP.light_extinction),
            max(1e-12, float(self.WP.light_height_ref)),
            float(self.WP.light_input) * dt,
            float(self.PP.flora_min_mass_frac) * BK,
            float(self.PP.flora_seedling_floor),
            float(self.PP.flora_max_seeds_per_tick),
            float(self.WP.E_labile_J_per_kg),
            mass_out, root_out, energy_out,
            reserve_out, pool_out, carbon_out,
            shed_out, dying_out, dm_out, grow_out,
            claimed, lam, hsum, cellacc,
        )

        store.mass[fl] = mass_out
        store.flora_root_mass[fl] = root_out
        store.energy[fl] = energy_out
        store.flora_reserve[fl] = reserve_out
        store.flora_repro_pool[fl] = pool_out
        store.flora_carbon_pool[fl] = carbon_out

        # Förnafallet först, kadavret sedan: strukturandelen blandas massviktat
        # per cell, och ordningen mellan de två deponeringarna följer med.
        # `excrete_cells` filtrerar själv bort noll och negativa, så hela
        # vektorn kan skickas in — urvalet blir detsamma.
        world.excrete_cells(cells, shed_out, struct32)
        self._last_flora_shed = float(shed_total)
        self._last_flora_died_age = int(n_age)
        self._last_flora_died_starve = int(n_starve)

        dyi = np.flatnonzero(dying_out)
        if dyi.size:
            world.excrete_cells(
                cells[dyi], mass_out[dyi].astype(np.float64), struct32[dyi]
            )
            for slot in fl[dyi]:
                # Reserven och poolen är redan återförda av kärnan, i rätt läge
                # i passet. Här återstår bara bokföringen i store:n.
                self._release_flora_slot(int(slot), return_nutrient=False)

        if row_plant.size == 0:
            store.clear_flora_claims()
            if died > 0.0:
                self._flora_summary_cache = None
            return 0.0, 0.0, float(died)

        store.set_flora_claims(claimed, fl[row_plant], row_cell, row_share)
        self._last_flora_light_limited = float(light_lim)

        if produced > 0.0 or died > 0.0 or shed_total > 0.0:
            self._flora_summary_cache = None

        return float(produced), float(taken), float(died)

    def _release_flora_slot(self, slot: int, return_nutrient: bool = True) -> None:
        """
        Avregistrera en floraindivid och frigör dess slot.

        Reserven och reproduktionspoolen är löst näring i plantans vävnader och
        återförs till cellen. Utan det försvinner de ur balansen, och sedan
        reserven finns är det den vanligaste vägen ut ur systemet.

        `return_nutrient=False` när återföringen redan skett på annat håll.
        Tillväxtkärnan gör den inuti passet, före anspråksberäkningen, eftersom
        det är där numpy-vägen gör den — näringen ska hinna bli tillgänglig för
        grannarna samma tick.
        """
        s = int(slot)
        if return_nutrient:
            held = (float(self.store.flora_reserve[s])
                    + float(self.store.flora_repro_pool[s]))
            cell = int(self.store.cell_idx[s])
            if held > 0.0 and cell >= 0:
                self.world.add_nutrient(cell, held)
        self.store.flora_reserve[s] = 0.0
        self.store.flora_repro_pool[s] = 0.0
        # Kolpoolen är inte näring och har ingen ledger att återföras till.
        self.store.flora_carbon_pool[s] = 0.0
        self.store.alive[s] = False
        self.store.mass[s] = np.float32(0.0)
        self.store.energy[s] = np.float32(0.0)
        self.store.release_slot(s)

    def _dispersal_system_flora(self) -> tuple[int, float]:
        """
        Fröproduktion, spridning och etablering.

        Tre saker byter form här. **Propagulmassan** är en egen absolut axel i
        stället för tio procent av moderns vuxenmassa — den gamla regeln gav ett
        träd på 44 kg ett frö på 4,4, och var dessutom skalfri, så att en mindre
        planta nådde sin egen tröskel snabbare och fröet krympte med henne.
        Storleksaxeln kunde bara falla.

        **Antalet** följer av poolen dividerad med propagulmassan, så "många små
        eller få stora" blir ett val i stället för en konstant.

        **Etableringen** är en Hillfunktion med exponent två på fröets förråd.
        Formen är inte fri: fitness går som f(m)/m, och en vanlig mättande
        funktion har inget inre optimum — då vinner alltid det minsta fröet.
        Halvmättnaden växer med målcellens anspråkade area, så stort frö vinner
        där det är trångt och många små där det är öppet. Utan den kopplingen
        konvergerar axeln i stället för att differentiera.

        Utfallet dras **innan** sloten allokeras. Frön som inte etablerar sig
        blir detritus i sin målcell — fröregn föder marken — och antalet slots
        får ett tak utan att antalet frön behöver begränsas.

        Avståndet dras ur en tungsvansad kärna i kontinuerligt rum och slås upp
        med `grid.cell_of_many()`. O(1) per frö, vektoriserbart och
        geometriagnostiskt. Det ersätter en BFS per händelse över en skiva med
        två diskreta radier.

        Returnerar (antal etableringar, totalt utspridd massa i kg).
        """
        dt = float(self.WP.dt)
        BK = float(self.WP.B_K)
        if BK <= 0.0 or dt <= 0.0:
            return 0, 0.0

        store = self.store
        grid = self.grid
        fl = self._flora_slots()
        if fl.size == 0:
            return 0, 0.0

        m_all = store.mass[fl].astype(np.float64, copy=False)
        cap_all = np.maximum(1e-12, store.flora_adult_mass[fl].astype(np.float64, copy=False))
        struct_all = store.structure[fl].astype(np.float64, copy=False)
        pool_all = store.flora_repro_pool[fl]
        seed_all = np.maximum(1e-9, store.flora_seed_mass[fl].astype(np.float64, copy=False))
        appar_all = store.flora_apparatus[fl].astype(np.float64, copy=False)
        cost_all = nutrient_content_array(struct_all)

        # Fröet kostar i båda valutorna. Näringen räknas ur fröets egen
        # sammansättning och inte ur moderns: ett frö är näringsrikt förråd
        # oavsett om föräldern är vedartad, och att använda moderns struktur gav
        # den vedartade 4,5 gångers rabatt. Kolkostnaden är fröets massa, samma
        # växelkurs som när ljus bygger kropp — ett frö är alltså exakt lika
        # dyrt som lika mycket vävnad, vilket gör att samma resurs begränsar
        # reproduktion som tillväxt.
        carbon_all = store.flora_carbon_pool[fl]
        seed_cost_all = seed_all * nutrient_content(SEED_STRUCTURE)
        eligible = (
            store.alive[fl]
            & (pool_all >= seed_cost_all)
            & (carbon_all >= seed_all)
            & (m_all >= np.maximum(
                FLORA_REPRO_MASS_MULT * seed_all,
                store.flora_maturity[fl].astype(np.float64, copy=False) * cap_all,
            ))
        )
        # Grindredovisning. Hundrasextio frön per månad från 92 590 plantor med
        # i genomsnitt mer pool än ett frö kostar är tre tiopotenser fel, och
        # summan kan inte skilja "poolen är tom hos de flesta" från "en annan
        # grind stoppar dem". Här räknas varje villkor för sig.
        g_alive = store.alive[fl]
        g_nut = g_alive & (pool_all >= seed_cost_all)
        g_carbon = g_alive & (carbon_all >= seed_all)
        g_size = g_alive & (m_all >= np.maximum(
            FLORA_REPRO_MASS_MULT * seed_all,
            store.flora_maturity[fl].astype(np.float64, copy=False) * cap_all,
        ))
        self._last_gate_alive = int(np.count_nonzero(g_alive))
        self._last_gate_nutrient = int(np.count_nonzero(g_nut))
        self._last_gate_carbon = int(np.count_nonzero(g_carbon))
        self._last_gate_size = int(np.count_nonzero(g_size))
        self._last_gate_all = int(np.count_nonzero(g_nut & g_carbon & g_size))

        chosen = np.flatnonzero(eligible)
        if chosen.size == 0:
            return 0, 0.0

        # Cellernas anspråkade area, för etableringens trängselterm. Samma
        # storhet som upptaget delas efter, alltså samma trängsel.
        n_cells = int(grid.n_cells)
        live = np.flatnonzero(store.alive & (store.kind == 1))
        crowd = np.zeros(n_cells, dtype=np.float64)
        if live.size:
            lc = store.cell_idx[live].astype(np.int64, copy=False)
            ok = lc >= 0
            crowd = np.bincount(
                lc[ok],
                weights=np.minimum(1.0, store.mass[live][ok].astype(np.float64) / BK),
                minlength=n_cells,
            )[:n_cells]

        # Allt utom slotallokeringen görs över alla frön på en gång. En loop
        # per moder kostade 348 ms per tick vid trettontusen plantor: femton
        # numpy-anrop på 64-elementsarrayer, tusentals gånger. Den serielle
        # delen är bara etableringarna, och de är få.
        n_seed = np.minimum(
            int(self.PP.flora_max_seeds_per_tick),
            np.minimum(
                (pool_all[chosen] / np.maximum(1e-30, seed_cost_all[chosen])).astype(np.int64),
                (carbon_all[chosen] / np.maximum(1e-30, seed_all[chosen])).astype(np.int64),
            ),
        )
        keep = n_seed > 0
        sel = chosen[keep]
        n_seed = n_seed[keep]
        if sel.size == 0:
            return 0, 0.0

        midx = np.repeat(sel, n_seed)
        slots = fl[midx]
        seed_m = seed_all[midx]
        appar = appar_all[midx]
        prov = seed_m * (1.0 - appar)

        L = dispersal_scale(cap_all[midx], appar, seed_m)
        # Stretchad exponentialkärna: tyngre svans än exponentialen, vilket är
        # den kvalitativa egenskap verkliga spridningskärnor har.
        d = L * self.rng.standard_exponential(midx.size) ** (1.0 / 0.7)
        th = self.rng.uniform(0.0, 2.0 * np.pi, midx.size)
        px = store.pos_x[slots].astype(np.float64) + d * np.cos(th)
        py = store.pos_y[slots].astype(np.float64) + d * np.sin(th)
        grid.wrap_pos_inplace(px, py)
        targets = grid.cell_of_many(px, py).astype(np.int64, copy=False)

        pr = establish_p(prov, crowd[targets])
        wins = self.rng.random(midx.size) < pr

        # Poolen debiteras en gång per moder, oavsett utfall: fröna byggdes.
        store.flora_repro_pool[fl[sel]] -= n_seed * seed_cost_all[sel]
        store.flora_carbon_pool[fl[sel]] -= n_seed * seed_all[sel]
        dispersed_mass = float(seed_m.sum())
        self._last_flora_seeds = int(midx.size)

        # Frön som inte etablerar sig blir förna där de landade.
        lost = ~wins
        if np.any(lost):
            # Förnan får fröets egen strukturandel, inte moderns. Poolen
            # debiterades fröets sammansättning, så någon annan struktur här
            # skapar eller förstör näring — uppmätt 89 kg på 1 500 tick innan
            # det rättades.
            self.world.excrete_cells(
                targets[lost], seed_m[lost], np.full(int(lost.sum()), SEED_STRUCTURE)
            )

        established = 0
        for j in np.flatnonzero(wins):
            mother = int(slots[j])
            if not bool(store.alive[mother]):
                continue
            target = int(targets[j])
            paid_each = float(seed_cost_all[midx[j]])

            child_traits = mutate_trait_vector(
                store.traits[mother, :], self.rng, sigma=0.05, p=0.10, clip=2.5,
            )
            child_slot = self._add_or_create_flora_in_cell(
                target, float(seed_m[j]), traits=child_traits
            )
            if child_slot < 0:
                continue

            # Näringen balanseras mot cellen: fröet ärver muterad struktur, så
            # dess innehåll per kilo är inte moderns.
            nut_chi = float(store.mass[child_slot]) * nutrient_content(
                float(store.structure[child_slot])
            )
            d_nut = paid_each - nut_chi
            if d_nut > 0.0:
                self.world.add_nutrient(target, d_nut)
            elif d_nut < 0.0:
                got_n = self.world.take_nutrient(target, -d_nut)
                if got_n < -d_nut:
                    deficit = -d_nut - got_n
                    nc = max(1e-12, nutrient_content(float(store.structure[child_slot])))
                    shrink = min(float(store.mass[child_slot]), deficit / nc)
                    store.mass[child_slot] = np.float32(
                        float(store.mass[child_slot]) - shrink
                    )
                    if float(store.flora_root_mass[child_slot]) > float(store.mass[child_slot]):
                        store.flora_root_mass[child_slot] = store.mass[child_slot]
                    store.energy[child_slot] = np.float32(max(
                        0.0,
                        float(store.mass[child_slot]) * self._slot_energy_per_kg(child_slot),
                    ))
            established += 1

        if established > 0 or dispersed_mass > 0.0:
            self._flora_summary_cache = None

        return int(established), float(dispersed_mass)
    def sample_flora_local(self, x: float, y: float) -> float:
        """
        Lokal flora-sampling från härlett perceptionsfält.
        
        Detta läser `store.flora_cell_mass`, som byggs om från levande flora i
        OrganismStore. Source of truth för flora ligger fortfarande i store-slotsen.
        """
        cell = int(self.grid.cell_of(float(x), float(y)))
        return float(self.store.flora_cell_mass[cell])

    def sample_flora_rays(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Ray-sampling av flora från härlett perceptionsfält.
        
        Returnerar flora-massa per punkt via celllookup mot `store.flora_cell_mass`.
        Detta är en sensing-cache; source of truth för flora ligger i store-slotsen.
        """
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        if xs.shape != ys.shape:
            raise ValueError("xs and ys must have same shape")
    
        out = np.empty(xs.shape, dtype=np.float32)
    
        cells = self.grid.cell_of_many(xs, ys)
        out[:] = self.store.flora_cell_mass[cells]
    
        return out
        
    def consume_food(self, x: float, y: float, amount: float,
                     diet: float = 0.5, reach: int = 1) -> tuple[float, float, float, float]:
        """
        Konsumera upp till `amount` kg, från det mest värdefulla först.

        Returnerar (kg_levande, kg_detritus, energi_levande_J, energi_detritus_J).

        Värdet är `assimilated_fraction(struktur, verkningsgrad) × E_labile`,
        alltså joule per ingesterat kilo — för det här djuret och det här
        substratet. Talet finns redan i modellen; det som saknades var att
        någon läste det innan valet gjordes.

        Ordningen är maximerande, inte blandande: den bästa födan tas först,
        och den näst bästa rörs bara med det som återstår när den första inte
        räcker. Det är skillnaden mot en preferensvikt, som alltid äter en
        andel av det sämre alternativet även när det bättre står bredvid.

        Preferensen behöver därmed inte lagras. En asätare och en betare får
        olika tal ur samma föda genom sina verkningsgrader, och valet följer.
        `diet` är alltså bara en avvägning mellan verkningsgrader, inte
        dessutom en smakriktning.

        `reach` är hur många celler ut betningen når. Den följer sträckan
        organismen färdats under ticken: ett betande djur äter medan det går,
        och när ticket är över fjorton timmar hinner det korsa flera celler.

        Uppdelningen levande/detritus finns kvar därför att den är verklig i
        anskaffningen: levande vävnad tillhör en organism som växer tillbaka
        och senare kan försvara sig, medan detritus är en pool som bara
        sönderfaller. Den finns inte i energiomvandlingen — där avgör
        strukturandelen, oavsett varifrån materialet kom.
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        cell = int(self.grid.cell_of(float(x), float(y)))
        herb_eff, scav_eff = diet_efficiency(float(diet))

        v_detritus = assimilated_fraction(
            float(self.world.detritus_structure[cell]), scav_eff
        )
        v_carcass = assimilated_fraction(
            float(self.world.carcass_structure[cell]), scav_eff
        )
        v_flora = assimilated_fraction(
            float(self.store.flora_cell_structure[cell]), herb_eff
        )

        got_l = got_d = e_l = e_d = 0.0

        def take_pool(field, want: float) -> float:
            nonlocal got_d, e_d
            if want <= 0.0:
                return 0.0
            g, e = self.world._consume_from_field(field, x, y, want)
            got_d += g
            e_d += e
            return float(g)

        def take_detritus(want: float) -> float:
            return take_pool(self.world.detritus, want)

        def take_carcass(want: float) -> float:
            return take_pool(self.world.carcass, want)

        def take_flora(want: float) -> float:
            nonlocal got_l, e_l
            if want <= 0.0:
                return 0.0
            g, e = self._consume_flora_from_store(x, y, want, max_radius=int(reach))
            got_l += g
            e_l += e
            return float(g)

        # Tre källor, tagna i värdeordning. Kadaver och förna är båda döda men
        # inte längre samma sak: ett kadaver vid strukturandel 0,25 är värt
        # mångdubbelt mer per kilo än förna vid 0,83, och det var just den
        # skillnaden som försvann när de delade pool.
        order = sorted(
            ((v_flora, take_flora), (v_carcass, take_carcass),
             (v_detritus, take_detritus)),
            key=lambda kv: kv[0], reverse=True,
        )
        left = amt
        for _, take in order:
            if left <= 1e-15:
                break
            left -= take(left)

        return float(got_l), float(got_d), float(e_l), float(e_d)

    @staticmethod
    def _flora_quantiles(store, fl) -> dict:
        """
        Kvartiler och median, inte bara summan. En pool på 667 kg kan ligga
        jämnt hos alla eller samlad hos några få stora plantor, och skillnaden
        avgör om reproduktionen är strypt eller normal.

        Dyr: varje anrop partitionerar hela floravektorn. Anropas bara när
        `_flora_want_quantiles` är satt, alltså när diagnostiken efterfrågas.
        """
        return {
            "flora_pool_p25": float(np.percentile(store.flora_repro_pool[fl], 25)),
            "flora_pool_median": float(np.median(store.flora_repro_pool[fl])),
            "flora_pool_p75": float(np.percentile(store.flora_repro_pool[fl], 75)),
            "flora_carbon_median": float(np.median(store.flora_carbon_pool[fl])),
            "flora_mass_median": float(np.median(store.mass[fl])),
            "flora_mass_p90": float(np.percentile(store.mass[fl], 90)),
        }

    def _rebuild_flora_summary(self) -> None:
        """
        Bygg flora-summary-cache för loggning och diagnostik.
        Spatialt uppslag sker via store.spatial_index, inte via separat cache.

        Vektoriserad över den aktiva floradelmängden. Loopen som fanns här
        kostade efter store-kapacitet i stället för efter antal levande, och
        kördes varje tick oavsett om något ändrats.
        """
        fl = self._flora_slots()
        if fl.size == 0:
            nan = float("nan")
            self._flora_summary_cache = {
                "flora_n": 0,
                "flora_mass_store": 0.0,
                "flora_energy_store": 0.0,
                "flora_mean_repro_alloc": nan,
                "flora_mean_adult_mass": nan,
                "flora_mean_temp_opt": nan,
                "flora_mean_temp_width": nan,
                "flora_mean_structure": nan,
                "flora_cells_occupied": 0.0,
                "flora_per_cell": nan,
                "flora_reserve_total": 0.0,
                "flora_pool_p25": 0.0,
                "flora_pool_median": 0.0,
                "flora_pool_p75": 0.0,
                "flora_carbon_median": 0.0,
                "flora_mass_median": 0.0,
                "flora_mass_p90": 0.0,
                "flora_pool_total": 0.0,
                "flora_carbon_pool_total": 0.0,
                "flora_mean_root_alloc": nan,
                "flora_mean_maturity": nan,
                "flora_mature_frac": nan,
                "flora_mean_root_frac": nan,
                "flora_mean_apparatus": nan,
                "flora_mean_seed_mass": nan,
            }
            return

        store = self.store
        self._flora_summary_cache = {
            "flora_n": int(fl.size),
            "flora_mass_store": float(np.sum(store.mass[fl], dtype=np.float64)),
            "flora_energy_store": float(np.sum(store.energy[fl], dtype=np.float64)),
            "flora_mean_repro_alloc": float(np.mean(store.flora_repro_alloc[fl], dtype=np.float64)),
            "flora_mean_adult_mass": float(np.mean(store.flora_adult_mass[fl], dtype=np.float64)),
            "flora_mean_temp_opt": float(np.mean(store.flora_temp_opt[fl], dtype=np.float64)),
            "flora_mean_temp_width": float(np.mean(store.flora_temp_width[fl], dtype=np.float64)),
            "flora_mean_structure": float(np.mean(store.structure[fl], dtype=np.float64)),
            "flora_cells_occupied": float(_occupied_cells(store, fl)),
            "flora_per_cell": float(fl.size / max(1.0, _occupied_cells(store, fl))),
            "flora_reserve_total": float(np.sum(store.flora_reserve[fl], dtype=np.float64)),
            "flora_pool_total": float(np.sum(store.flora_repro_pool[fl], dtype=np.float64)),
            # Kvartiler och median, inte bara summan. En pool på 667 kg kan
            # ligga jämnt hos alla eller samlad hos några få stora plantor, och
            # skillnaden avgör om reproduktionen är strypt eller normal.
            # Kvantilerna beräknas bara när någon faktiskt läser dem.
            #
            # Sex percentiler är sex fulla partitioneringar av hela
            # floravektorn. Vid 292 459 plantor mätte cProfile 240 anrop till
            # `ndarray.partition` över 30 tick — 0,68 sekunder, alltså sju
            # procent av takten — för statistik som bara skrivs vid
            # rapportintervall. Summorna och medelvärdena är enkla genomgångar
            # och kostar en bråkdel; det är sorteringen som är dyr.
            **(self._flora_quantiles(store, fl) if getattr(self, "_flora_want_quantiles", False) else {}),
            "flora_carbon_pool_total": float(np.sum(store.flora_carbon_pool[fl], dtype=np.float64)),
            "flora_mean_root_alloc": float(np.mean(store.flora_root_alloc[fl], dtype=np.float64)),
            "flora_mean_maturity": float(np.mean(store.flora_maturity[fl], dtype=np.float64)),
            "flora_mature_frac": float(np.mean(
                store.mass[fl].astype(np.float64)
                >= store.flora_maturity[fl].astype(np.float64)
                * np.maximum(1e-12, store.flora_adult_mass[fl].astype(np.float64))
            )),
            "flora_mean_root_frac": float(
                np.sum(store.flora_root_mass[fl], dtype=np.float64)
                / max(1e-12, np.sum(store.mass[fl], dtype=np.float64))
            ),
            "flora_mean_apparatus": float(np.mean(store.flora_apparatus[fl], dtype=np.float64)),
            "flora_mean_seed_mass": float(np.mean(store.flora_seed_mass[fl], dtype=np.float64)),
        }

    def _flora_summary(self) -> dict[str, float]:
        if self._flora_summary_cache is None:
            # Delmängden byggs om här: senescens och spridning har hunnit
            # ändra beståndet sedan tickens början, och summeringen ska visa
            # läget som det är, inte som det var.
            self._flora_slots(rebuild=True)
            self._rebuild_flora_summary()
        return self._flora_summary_cache
        
    # -----------------------------
    # step-methods
    # -----------------------------
    def _step_world_and_flora(self) -> tuple[float, int, float]:
        """
        Kör världspass + florapass och bygger avledda store-index.
        Returnerar:
          (dM_growth_flora, flora_established, flora_dispersed_mass)
        """
        self.world.step()
    
        dM_growth_flora, dM_uptake, dM_flora_death = self._growth_system_flora()
        flora_established, flora_dispersed_mass = self._dispersal_system_flora()
    
        self.store.rebuild_spatial_index()
        self._flora_summary_cache = None
    
        return float(dM_growth_flora), int(flora_established), float(flora_dispersed_mass)
        
    def _build_sector_percept(
        self,
        alive: list,
        n_sectors: int = 6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perceptionens världskanaler som sektoraggregat, för alla djur på en gång.

        Strålarna sampladeepunkter: tolv riktningar gånger sju avståndssteg, med
        `grid.cell_of` per punkt i en nästlad Python-loop. De översamplade nära —
        alla tolv strålarna landar i samma sex celler vid avstånd ett — och
        undersamplade långt, där grannskapet är brett och strålarna glesa.

        Här aggregeras i stället *varje* cell inom räckvidden till sin
        riktningssektor. Räckvidden är oförändrad; det är täckningen inom den
        som blir hel. Sex sektorer, en per hexgranne.

        Aggregeringen sker i världsram med en gemensam offsettabell och roteras
        sedan till kroppsram per djur. Rotationen är en viktad blandning mellan
        grannsektorer — samma operation som en akuitetsoskärpa, och de
        komponerar därför när den kommer.

        Mättnaden `x / (x + K)` tillämpas per cell före medelvärdet, precis som
        strålarna gjorde per samplingspunkt. Vikten faller med avståndet så att
        närmare celler väger tyngre, vilket strålarnas viktprofil också gjorde.

        Returnerar `(B_u, C_u)`, båda `(n, n_sectors)` i kroppsram.
        """
        S = int(n_sectors)
        n = len(alive)
        if n == 0:
            z = np.zeros((0, S), dtype=np.float32)
            return z, z, z, z, z, z

        AP0 = alive[0].AP
        r_max = max(1, int(round(float(AP0.ray_len_front))))
        dq, dr, dist, sec, w = self._sector_tables(r_max, S)

        xs = np.fromiter((a.x for a in alive), dtype=np.float64, count=n)
        ys = np.fromiter((a.y for a in alive), dtype=np.float64, count=n)
        cell0 = self.grid.cell_of_many(xs, ys).astype(np.int64, copy=False)
        cells = self.grid.cells_within_many(cell0, r_max).astype(np.int64, copy=False)

        Kb = float(getattr(self.WP, "B_K", 0.0))
        Kc = float(getattr(self.WP, "C_sense_K", 0.0))

        def sat(field: np.ndarray, K: float) -> np.ndarray:
            v = np.maximum(field[cells].astype(np.float32, copy=False), np.float32(0.0))
            if K <= 0.0:
                return (v > 0.0).astype(np.float32, copy=False)
            return v / (v + np.float32(K))

        Bu = sat(self.store.flora_cell_mass, Kb)
        Cu = sat(np.asarray(self.world.detritus)
                 + np.asarray(self.world.carcass), Kc)

        # Tredje världskanal: temperatur. Samma gather, ingen mättnad —
        # temperaturen är inte en mängd utan ett värde, och det som ska styra
        # är skillnaden mellan sektorerna.
        #
        # `cold_aversion` fanns som ärftlig egenskap men hade noll läsare:
        # fem förekomster i phenotype.py, alla deklaration eller export. Djuren
        # kunde alltså inte söka värme. Uppmätt i p89 dog 68 procent kallare än
        # de föddes, median 11,5 -> 3,8 grader, och svältdöden hade median
        # 2,6 grader mot världens 12,2. Vid noll grader kostar termoregleringen
        # mer än hela basalmetabolismen; i de varmaste banden 43 procent mindre.
        Tw_cells = np.asarray(self.world.temperature_of_cells(cells.ravel()),
                              dtype=np.float32).reshape(cells.shape)

        # Artfränder som täta per-cell-fält: antal och hastighetssumma.
        #
        # Reynolds tre regler verkar på ett *grannskap*, inte på en granne, och
        # skälet är riktningen. En enskild granne kan bara ge riktningen mot
        # den grannen — djur som svänger mot närmaste artfrände roterar runt
        # varandra. Ett aggregat ger riktningen mot gruppens tyngdpunkt, och
        # det är en annan storhet: den konvergerar.
        #
        # 0099 stabiliserade målvalet — grannbyte 32,9 -> 8,6 procent — utan att
        # kohesionskvoten rörde sig från 1,0. Stabiliteten var nödvändig men
        # inte tillräcklig, och det som återstår är just tyngdpunkten.
        #
        # Fälten byggs med bincount över fauna_slots, samma mönster som
        # flora_cell_mass. Nollställningen är O(n_cells) och försumbar vid
        # 16 384; vid en miljon celler bör samma glesningstrick användas.
        nc = int(self.grid.n_cells)
        fslots = np.asarray(self._fauna_slots(), dtype=np.int64)
        f_cnt = np.zeros(nc, dtype=np.float32)
        f_vx = np.zeros(nc, dtype=np.float32)
        f_vy = np.zeros(nc, dtype=np.float32)
        if fslots.size:
            fc = self.store.cell_idx[fslots].astype(np.int64, copy=False)
            f_cnt += np.bincount(fc, minlength=nc).astype(np.float32, copy=False)
            hd = np.zeros(fslots.size, dtype=np.float64)
            by_slot = {}
            for a in self.agents:
                sl = int(getattr(a, "store_slot", -1))
                if sl >= 0 and a.body.alive:
                    by_slot[sl] = float(a.heading)
            for j, sv in enumerate(fslots):
                hd[j] = by_slot.get(int(sv), 0.0)
            f_vx += np.bincount(fc, weights=np.cos(hd), minlength=nc).astype(np.float32, copy=False)
            f_vy += np.bincount(fc, weights=np.sin(hd), minlength=nc).astype(np.float32, copy=False)

        # Egen cell är B0/C0 och hör inte till någon sektor.
        m = sec >= 0
        base = np.arange(n, dtype=np.int64)[:, None] * S
        flat = (base + sec[None, :])[:, m].ravel()
        wm = w[m]
        wsum = np.bincount(flat, weights=np.broadcast_to(wm, (n, wm.size)).ravel(),
                           minlength=n * S).reshape(n, S)
        np.maximum(wsum, 1e-9, out=wsum)

        def agg(u: np.ndarray) -> np.ndarray:
            acc = np.bincount(flat, weights=(u[:, m] * wm[None, :]).ravel(),
                              minlength=n * S).reshape(n, S)
            return acc / wsum

        Bw = agg(Bu)
        Cw = agg(Cu)
        Tw = agg(Tw_cells)
        Fw = agg(f_cnt[cells])
        Hx = agg(f_vx[cells])
        Hy = agg(f_vy[cells])

        # Rotation till kroppsram: fraktionell cirkulär förskjutning.
        head = np.fromiter((a.heading for a in alive), dtype=np.float64, count=n)
        shift = head / (2.0 * math.pi / S)
        i0 = np.floor(shift).astype(np.int64)
        frac = (shift - i0)[:, None].astype(np.float32)
        k = np.arange(S, dtype=np.int64)[None, :]
        a0 = (k + i0[:, None]) % S
        a1 = (a0 + 1) % S
        rows = np.arange(n, dtype=np.int64)[:, None]

        B_out = ((1.0 - frac) * Bw[rows, a0] + frac * Bw[rows, a1]).astype(np.float32, copy=False)
        C_out = ((1.0 - frac) * Cw[rows, a0] + frac * Cw[rows, a1]).astype(np.float32, copy=False)
        T_out = ((1.0 - frac) * Tw[rows, a0] + frac * Tw[rows, a1]).astype(np.float32, copy=False)
        F_out = ((1.0 - frac) * Fw[rows, a0] + frac * Fw[rows, a1]).astype(np.float32, copy=False)
        # Kurserna roteras inte som sektorer utan som vektorer: kroppsram
        # betyder att egen kurs dras av.
        HX = ((1.0 - frac) * Hx[rows, a0] + frac * Hx[rows, a1]).astype(np.float64, copy=False)
        HY = ((1.0 - frac) * Hy[rows, a0] + frac * Hy[rows, a1]).astype(np.float64, copy=False)
        ch = np.cos(-head)[:, None]; sh = np.sin(-head)[:, None]
        HXb = (HX * ch - HY * sh).astype(np.float32, copy=False)
        HYb = (HX * sh + HY * ch).astype(np.float32, copy=False)
        return B_out, C_out, T_out, F_out, HXb, HYb

    def _sector_tables(self, r_max: int, S: int):
        """Offset, avstånd, sektorindex och vikt per cell i grannskapet. Cachad."""
        key = (int(r_max), int(S))
        cached = self._sector_cache.get(key)
        if cached is not None:
            return cached
        dq, dr = self.grid._within_offsets(int(r_max))
        dist = (np.abs(dq) + np.abs(dq + dr) + np.abs(dr)) // 2
        gx = float(self.grid.COL_SPACING) * (dq + 0.5 * dr)
        gy = float(self.grid.ROW_SPACING) * dr
        ang = np.arctan2(gy, gx) % (2.0 * math.pi)
        sec = np.floor(ang / (2.0 * math.pi) * S).astype(np.int64) % S
        sec[dist == 0] = -1
        w = (1.0 / (1.0 + dist.astype(np.float32))).astype(np.float32)
        cached = (dq, dr, dist, sec, w)
        self._sector_cache[key] = cached
        return cached

    def _acquire_neighbours(self, alive: list, idx: list) -> dict:
        """
        Närmaste synliga artfrände för varje sensande djur, i ett svep.

        Strålmarschen gick avståndssteg för avståndssteg och stråle för stråle
        med `grid.cell_of` per punkt. Den kostade mest när den *inte* hittade
        något, eftersom den då gick hela vägen ut — alltså mest i den glesa
        regim modellen ska lämna. Vid 200 djur var den 176 av sensingens 246
        mikrosekunder per djur.

        Här slås grannskapet upp mot ett faunaeget cellindex. Faunan är hundratal
        mot florans tiotusental, så ett eget index är litet och gör att uppslaget
        inte behöver sålla bort växter en och en. Kandidaterna är typiskt noll
        eller ett par per grannskap.

        Lokaliteten är oförändrad: kandidater hämtas ur celler inom
        synellipsens räckvidd, aldrig ur en global lista. Räckvidden är samma
        ellips som strålarna hade — r(θ) = r_front (1-e) / (1 - e cos θ) — så
        synfältet är fortsatt framåtriktat och den bakre blindzonen består.

        Skillnaden mot strålmarschen är att träffen nu är den *verkligt* närmaste
        inom räckhåll, inte den första som råkade ligga på en stråle. Strålarna
        hade hål mellan sig som växte med avståndet.
        """
        out: dict = {}
        if not idx:
            return out

        fslots = np.asarray(self._fauna_slots(), dtype=np.int64)
        if fslots.size == 0:
            return out

        st = self.store
        fcells = st.cell_idx[fslots].astype(np.int64, copy=False)
        order = np.argsort(fcells, kind="stable")
        fcells_s = fcells[order]
        fslots_s = fslots[order]
        counts = np.bincount(fcells_s, minlength=int(self.grid.n_cells))

        AP0 = alive[idx[0]].AP
        r_front = float(AP0.ray_len_front)
        step = float(AP0.ray_step)
        r_top = max(1, int(round(r_front)))

        sub = [alive[i] for i in idx]
        n = len(sub)
        xs = np.fromiter((a.x for a in sub), dtype=np.float64, count=n)
        ys = np.fromiter((a.y for a in sub), dtype=np.float64, count=n)
        cell0 = self.grid.cell_of_many(xs, ys).astype(np.int64, copy=False)
        cells = self.grid.cells_within_many(cell0, r_top).astype(np.int64, copy=False)

        occ = counts[cells] > 0
        rows, cols = np.nonzero(occ)
        if rows.size == 0:
            return out

        # Ragged uppslag av de få bebodda cellerna.
        cc = cells[rows, cols]
        lo = np.searchsorted(fcells_s, cc, side="left")
        hi = np.searchsorted(fcells_s, cc, side="right")
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot == 0:
            return out
        starts = np.zeros(cnt.size + 1, dtype=np.int64)
        np.cumsum(cnt, out=starts[1:])
        flat = np.repeat(lo - starts[:-1], cnt) + np.arange(tot, dtype=np.int64)
        cand_slot = fslots_s[flat]
        cand_row = np.repeat(rows, cnt)

        ax = xs[cand_row]
        ay = ys[cand_row]
        bx = st.pos_x[cand_slot].astype(np.float64, copy=False)
        by = st.pos_y[cand_slot].astype(np.float64, copy=False)
        ex = float(self.grid.extent_x)
        ey = float(self.grid.extent_y)
        dx = (bx - ax) % ex
        dy = (by - ay) % ey
        dx = np.where(dx > 0.5 * ex, dx - ex, dx)
        dy = np.where(dy > 0.5 * ey, dy - ey, dy)
        dist = np.sqrt(dx * dx + dy * dy)

        heads = np.fromiter((a.heading for a in sub), dtype=np.float64, count=n)
        self_ids = np.fromiter((int(a.id) for a in sub), dtype=np.int64, count=n)
        rel = (np.arctan2(dy, dx) - heads[cand_row]) % (2.0 * math.pi)

        # Ett synfält, inte två. Individidentifieringen ärvde strålmodellens
        # ellips från 0085, medan världskanalerna sedan 0084 läser en cirkel
        # via `cells_within_many`. Samma djur hade därmed två olika synfält för
        # artfränder: aggregatet såg dem i en cirkel med radie tio, identiteten
        # bara i ellipsen — tio framåt, sju åt sidan, 5,4 bakåt. Ett djur kunde
        # känna att det fanns artfränder rakt bakom sig utan att kunna
        # identifiera någon av dem, och de två kanalerna gick inte att jämföra.
        #
        # Den framåtriktade biasen var motiverad för jakt och födosök, men
        # födosöket använder den inte längre. Ellipsen påverkade i praktiken
        # ingenting utom vilka artfränder som kunde identifieras, och för
        # flockning är den aktivt skadlig: en flockkamrat färdas jämsides, och
        # där var räckvidden kortast.
        #
        # Kvar är en enda parameter för synvidd. `ray_eccentricity` har därmed
        # ingen läsare alls och kan tas bort när strålmodellens övriga rester
        # städas.
        r_lim = np.full(rel.shape, r_front, dtype=np.float64)

        # m_eff kapar räckvidden precis som ray_depths gjorde.
        m_eff = np.fromiter(
            (max(0, int(alive[i]._sense_m_eff)) for i in idx), dtype=np.int64, count=n
        )
        cap = np.where(m_eff[cand_row] > 0, m_eff[cand_row] * step, r_front)
        r_lim = np.minimum(r_lim, cap)

        # Parningsvillig som egen sökkanal.
        #
        # Detektionen var odiskriminerande: närmaste artfrände oavsett
        # tillstånd, och filtret på parningsvillighet kom först efteråt i
        # `_resolve_detected_agent`. Var den närmaste ovillig hade djuret ingen
        # partner alls — även om en villig stod tio procent längre bort. Det är
        # defekten `docs/rorelsens-arkitektur.md` beskriver: fel granne närmast
        # blockerar parningsdriften.
        #
        # Med 32,7 procent grannbyten mellan på varandra följande detektioner
        # byttes målet dessutom under approachen. En villig partner byter inte
        # tillstånd varje tick, så en egen kanal ger något stabilt att hålla
        # fast vid under de åtta till fyrtio tick det tar att sluta avståndet.
        #
        # Biologiskt är det en separat perceptuell kanal och inte en genväg:
        # läten, dofter och uppvisning är evolverade just för att vara märkbara
        # på avstånd. Lokaliteten är orörd — samma celler, samma synellips.
        want_mate = np.zeros(n, dtype=bool)
        for k, i in enumerate(idx):
            want_mate[k] = bool(getattr(alive[i], "_mating_mode", False))
        cand_ready = np.zeros(cand_slot.size, dtype=bool)
        if want_mate.any():
            by_slot = {}
            for a in alive:
                sl = int(getattr(a, "store_slot", -1))
                if sl >= 0:
                    by_slot[sl] = bool(getattr(a, "_mating_mode", False))
            cand_ready = np.fromiter(
                (by_slot.get(int(sv), False) for sv in cand_slot),
                dtype=bool, count=cand_slot.size,
            )

        ok = (
            (dist > 0.0)
            & (dist <= r_lim)
            & (st.id[cand_slot].astype(np.int64, copy=False) != self_ids[cand_row])
            & (st.id[cand_slot] > 0)
            & (st.alive[cand_slot])
            & (st.kind[cand_slot] == 0)
        )
        if not np.any(ok):
            return out

        cand_row = cand_row[ok]
        cand_slot = cand_slot[ok]
        dist = dist[ok]
        rel = rel[ok]
        cand_ready = cand_ready[ok]

        # Ett parningsberett djur söker närmaste **villiga**; övriga söker
        # närmaste artfrände. Sorteringsnyckeln gör det utan en andra gather:
        # ovilliga kandidater läggs efter de villiga för sökare i parningsläge.
        prio = np.zeros(dist.size, dtype=np.float64)
        seeker_ready = want_mate[cand_row]
        prio[seeker_ready & ~cand_ready] = 1.0

        # Följ samma individ. Valet var alltid "närmaste artfrände", och
        # närmaste granne är en instabil referens: två djur på snarlikt avstånd
        # byter plats i rangordningen vid minsta rörelse.
        #
        # Uppmätt: 32,8 procent grannbyte mellan på varandra följande
        # detektioner vid fart 37,5, och 32,9 procent vid 23,2 — helt
        # oberoende av samplingstätheten, som samtidigt förbättrades en
        # tredjedel. Det är alltså inte perceptionen som är för gles utan
        # måttet som är instabilt.
        #
        # Följden är att den sociala reflexen svänger mot A, sedan B, sedan A.
        # Amplituden är stor — median 0,128 i styrkommandot, aldrig mättad —
        # men riktningen inkoherent, och kohesionskvoten mot Poisson har legat
        # på 1,0 genom fyra patchar.
        #
        # Ändringen gör den tidigare följda individen till förstahandsval så
        # länge den är synlig. Målet blir stabilt över hela mötet i stället för
        # att bytas var tredje gång. Ballerini m.fl. 2008 visade att starar
        # följer bestämda grannar över tid snarare än de närmaste.
        #
        # Ingen global koordination: samma celler, samma synellips, samma
        # gather. Bara valet inom grannskapet ändras.
        prev = np.zeros(n, dtype=np.int64)
        for k, i in enumerate(idx):
            prev[k] = int(getattr(alive[i], "_follow_id", 0) or 0)
        if np.any(prev > 0):
            same = st.id[cand_slot].astype(np.int64, copy=False) == prev[cand_row]
            # Halvt steg: går före andra kandidater med samma parningsprioritet,
            # men parningsvillighet väger tyngre än vanan.
            prio[same] -= 0.5

        # Flocken som relation, inte som ögonblicksurval.
        #
        # "Alla inom synhåll" ger en flock som byter medlemmar varje gång någon
        # passerar: två flockar som möts smälter omedelbart samman, och en
        # ensam vandrare fångas av den första grupp den korsar. Det finns ingen
        # identitet över tid.
        #
        # Med minne byggs medlemskapet upp — affiniteten stiger vid varaktig
        # närhet och avtar när man skiljs åt. En främling som passerar får låg
        # vikt tills den varit där ett tag. Det ger flockar som håller ihop
        # genom möten, delas när de dras isär och gradvis blandas i stället för
        # att slås samman. Medlemskapsmatrisen *är* flocken och går att mäta.
        #
        # Kohesionen behåller det täta per-cell-fältet — att dras mot där det
        # finns artfränder alls är rimligt även för icke-medlemmar — medan
        # alignment blir medlemsviktad. Det ger en naturlig asymmetri: man dras
        # mot främlingar men följer bara sina egna.
        #
        # Aggregatet i `_build_sector_percept` kan inte viktas per observatör,
        # eftersom alla läser samma fält. Därför räknas alignment här, över
        # varje djurs egna kandidater — listan finns redan.
        head_of = {}
        agent_of = {}
        for a in alive:
            sl = int(getattr(a, "store_slot", -1))
            if sl >= 0 and a.body.alive:
                head_of[sl] = float(a.heading)
                agent_of[sl] = a

        gain = float(getattr(self.PP, "flock_gain", 0.25))
        decay = float(getattr(self.PP, "flock_decay", 0.90))
        floor = 0.05
        order_r = np.argsort(cand_row, kind="stable")
        bounds = np.searchsorted(cand_row[order_r], np.arange(n + 1))
        for r in range(n):
            a = alive[idx[r]]
            fl = getattr(a, "_flock", None)
            if fl is None:
                fl = {}
                a._flock = fl
            for k in list(fl):
                v = fl[k] * decay
                if v < floor:
                    del fl[k]
                else:
                    fl[k] = v
            hx = hy = 0.0
            for t in order_r[bounds[r]:bounds[r + 1]]:
                sl = int(cand_slot[t])
                cid = int(st.id[sl])
                w = min(1.0, fl.get(cid, 0.0) + gain)
                fl[cid] = w
                h = head_of.get(sl)
                if h is not None:
                    hx += w * math.cos(h)
                    hy += w * math.sin(h)
            ch = math.cos(-float(a.heading)); sh_ = math.sin(-float(a.heading))
            a._flock_align = (hx * ch - hy * sh_, hx * sh_ + hy * ch)

            # Social synkronisering av häckningsfasen, viktad med samma
            # affinitet. Faser är cirkulära och måste medelvärdesbildas som
            # vektorer. Dragningen sker mot dem djuret verkligen umgås med,
            # så varje flock konvergerar mot sin egen fas — vilket ger
            # reproduktiv isolering mellan grupper utan att koda den.
            pull = float(getattr(a.pheno, "breed_pull", 0.0))
            if pull > 1e-6:
                px = py = 0.0
                for t in order_r[bounds[r]:bounds[r + 1]]:
                    sl = int(cand_slot[t])
                    o = agent_of.get(sl)
                    if o is None:
                        continue
                    w = fl.get(int(st.id[sl]), 0.0)
                    if w <= 0.0:
                        continue
                    ang = 2.0 * math.pi * float(getattr(o, "_breed_phase_real",
                                                        o.pheno.breed_phase))
                    px += w * math.cos(ang)
                    py += w * math.sin(ang)
                if abs(px) + abs(py) > 1e-12:
                    own = float(getattr(a, "_breed_phase_real", a.pheno.breed_phase))
                    tgt = math.atan2(py, px) / (2.0 * math.pi)
                    d_ = (tgt - own + 0.5) % 1.0 - 0.5      # kortaste vägen runt
                    a._breed_phase_real = (own + pull * d_) % 1.0

        # Närmaste per djur: sortera på (rad, prioritet, avstånd).
        srt = np.lexsort((dist, prio, cand_row))
        cand_row = cand_row[srt]
        cand_slot = cand_slot[srt]
        dist = dist[srt]
        rel = rel[srt]
        first = np.ones(cand_row.size, dtype=bool)
        first[1:] = cand_row[1:] != cand_row[:-1]
        sel = np.flatnonzero(first)

        for k in sel:
            r = int(cand_row[k])
            s_ = int(cand_slot[k])
            d = float(dist[k])
            j = min(int(AP0.n_rays) * 0 + int(d / max(step, 1e-9)), 10_000)
            alive[idx[r]]._follow_id = int(st.id[s_])
            out[idx[r]] = (
                1.0,
                float(rel[k] / (2.0 * math.pi)),
                float(d / max(r_front, 1e-9)),
                int(j),
                s_,
                int(st.id[s_]),
            )
        return out

    def _step_sense_system(
        self,
        ctx: "StepCtx",
    ) -> SenseBatch:
        """
        Kör sensing/input-byggande för alla levande agenter.
    
        Returnerar ett explicit batch-objekt för nästa pass.
        """
        alive: list[Agent] = [a for a in self.agents if a.body.alive]
        alive_slots = np.asarray([int(a.store_slot) for a in alive], dtype=np.int32)
        
        for a in alive:
            a._mating_mode = self._mating_mode_slot(a.store_slot)
        
        if not alive:
            return SenseBatch(
                alive=[],
                alive_slots=np.zeros((0,), dtype=np.int32),
                X=np.zeros((0, 0), dtype=np.float32),
                BC_list=[],
            )
    
        n = len(alive)
        in_dim = int(alive[0].genome.layer_sizes[0])
        X = np.empty((n, in_dim), dtype=np.float32)
    
        BC_list: list[tuple[float, float]] = [(0.0, 0.0)] * n
    
        # Sektorpercepten byggs bara för dem som faktiskt sensar den här
        # ticken. Cachevägen rör den inte, och att räkna för alla vore tre
        # gånger för mycket arbete vid nuvarande sensingfrekvens.
        sensing = [i for i, a in enumerate(alive) if int(a._sense_cd) <= 0]
        secB = secC = None
        if sensing:
            secB, secC, secT, secF, secHX, secHY = self._build_sector_percept(
                [alive[i] for i in sensing])
        sec_row = {}
        for k, i in enumerate(sensing):
            sec_row[i] = (secB[k], secC[k])
            alive[i]._temp_sectors = secT[k]
            alive[i]._soc_sectors = (secF[k], secHX[k], secHY[k])
        # Tomt uppslag betyder "ingen artfrände inom räckhåll", inte "ej
        # beräknat". Utan den skillnaden faller just de djur som inget ser
        # tillbaka på strålmarschen — och det är precis de dyraste fallen.
        nb_row = {i: None for i in sensing}
        if sensing:
            nb_row.update(self._acquire_neighbours(alive, sensing))

        for i, a in enumerate(alive):
            x_in, B0, C0 = a.build_inputs(self.world, rng=self.rng,
                                          sectors=sec_row.get(i),
                                          neighbour=nb_row.get(i, False))
    
            if x_in is None:
                X[i] = 0.0
                BC_list[i] = (0.0, 0.0)
                self._emit_step_if_tracked(self.t, a, 0.0, 0.0)
                continue
    
            X[i] = x_in
            BC_list[i] = (float(B0), float(C0))
            self._emit_step_if_tracked(self.t, a, float(B0), float(C0))
    
        return SenseBatch(
            alive=alive,
            alive_slots=alive_slots,
            X=X,
            BC_list=BC_list,
        )
        
    def _step_decision_system(
        self,
        ctx: "StepCtx",
        sense_batch: SenseBatch,
    ) -> DecisionBatch:
        """
        Kör decision-systemet för de agenter som redan passerat sensing:
          - policy-forward
          - action planning
    
        Returnerar ett explicit batch-objekt för move_system.
        """
        alive = sense_batch.alive
        X = sense_batch.X
        BC_list = sense_batch.BC_list
        alive_slots = sense_batch.alive_slots
    
        if not alive:
            return DecisionBatch(
                plans=[],
                plan_slots=np.zeros((0,), dtype=np.int32),
            )
    
        groups: dict[tuple, list[int]] = {}
        for i, a in enumerate(alive):
            groups.setdefault(a._policy_key, []).append(i)
    
        out_dim = int(alive[0].genome.layer_sizes[-1])
        Y = np.zeros((X.shape[0], out_dim), dtype=np.float32)
    
        for key, idxs in groups.items():
            bank = self._banks[key]
            idxs_arr = np.asarray(idxs, dtype=np.int32)
            H = X[idxs_arr]
            slots = np.asarray([alive[i]._policy_slot for i in idxs], dtype=np.int32)
    
            L = len(bank.W)
            for li in range(L):
                W = bank.W[li][slots]
                b = bank.b[li][slots]
                Z = np.einsum("noi,ni->no", W, H) + b
                H = self._act_hidden(Z, bank.act) if li < L - 1 else Z
    
            Y[idxs_arr] = H

        plans = []
        plan_slots: list[int] = []
        
        for i, a in enumerate(alive):
            B0, C0 = BC_list[i]
            plan = a.plan_actions(self.world, ctx, Y[i], B0, C0)
            plans.append((a, plan))
            plan_slots.append(int(alive_slots[i]))
        
        return DecisionBatch(
            plans=plans,
            plan_slots=np.asarray(plan_slots, dtype=np.int32),
        )

           
    def _step_move_system(
        self,
        ctx: "StepCtx",
        decision_batch: DecisionBatch,
    ) -> BodyBatch:
        """
        Verkställ rörelse och feeding för planerade agenter.
    
        Returnerar ett explicit batch-objekt för body_system.
        """
        plans = decision_batch.plans
        plan_slots = decision_batch.plan_slots
        if not plans:
            return BodyBatch(
                body_inputs=[],
                body_slots=np.zeros((0,), dtype=np.int32),
            )
    
        body_inputs = []
        body_slots: list[int] = []
        
        for i, (a, plan) in enumerate(plans):
            if not a.body.alive:
                continue
        
            body_in = a.execute_action_plan(self.world, ctx, plan)
            self._write_spatial_to_store(a.store_slot, a.x, a.y)
        
            body_inputs.append((a, body_in))
            body_slots.append(int(plan_slots[i]))
        
        return BodyBatch(
            body_inputs=body_inputs,
            body_slots=np.asarray(body_slots, dtype=np.int32),
        )

    def _flush_body_outputs(self, a: Agent) -> None:
        """
        Töm kroppens ackumulerade utflöden till den cell agenten står i.

        `Body` har ingen världsreferens och ska inte få en — fysiologikärnan
        bokför vad som lämnat kroppen, passet placerar det i världen. Exkrement
        går till detritus och måste brytas ner innan det blir tillgängligt;
        förbränningens kväve går direkt till den fria näringspoolen.
        """
        b = a.body

        out_kg = float(b.out_excreta_kg)
        if out_kg > 1e-15:
            s_out = float(b.out_excreta_struct_kg) / out_kg
            self.world.excrete_at(float(a.x), float(a.y), out_kg, s_out)
        b.out_excreta_kg = 0.0
        b.out_excreta_struct_kg = 0.0

        out_n = float(b.out_nutrient_kg)
        if out_n > 1e-18:
            self.world.add_nutrient(int(self.grid.cell_of(float(a.x), float(a.y))), out_n)
        b.out_nutrient_kg = 0.0

    def _step_body_system(
        self,
        ctx: "StepCtx",
        body_batch: BodyBatch,
    ) -> None:
        """
        Kör fysiologisk kroppsdynamik som eget pass efter move_system.
    
        Detta håller fortfarande Body.step() intakt, men flyttar subsystemgränsen
        ut ur move_system så att rörelse och fysiologi inte längre är samma pass.
        """
        body_inputs = body_batch.body_inputs
        body_slots = body_batch.body_slots
        if not body_inputs:
            return
    
        for i, (a, body_in) in enumerate(body_inputs):
            if not a.body.alive:
                continue
        
            slot = int(body_slots[i])
            age_s = float(self.store.age[slot]) if slot >= 0 else 0.0
            
            a.body.step(
                ctx,
                speed=body_in.speed,
                activity=body_in.activity,
                food_bio_kg=body_in.food_bio_kg,
                food_carcass_kg=body_in.food_carcass_kg,
                food_bio_J=body_in.food_bio_J,
                food_carcass_J=body_in.food_carcass_J,
                assim_bio_kg=body_in.assim_bio_kg,
                assim_carcass_kg=body_in.assim_carcass_kg,
                pheno=a.pheno,
                extra_drain=body_in.E_move,
                T_env=body_in.Tloc,
                age_s=age_s,
            )

            self._flush_body_outputs(a)

            self._write_alive_to_store(slot, a.body.alive)
            self._write_body_surface_to_store(slot, a)
            
            a.last_B0 = float(body_in.B0)
            a.last_C0 = float(body_in.C0)
            
            if slot >= 0:
                self._write_gestation_to_store(slot, a)

    def _step_interaction_system(self, ctx: "StepCtx") -> None:
        """
        Hantera organism–organism-interaktioner efter movement/body:
          - predation
          - mating-initiering
    
        Detta pass ska innehålla sådant som kräver att individerna redan har
        uppdaterat position och kroppstillstånd för ticken, men ännu inte har
        städats bort via death/birth-pass.
        """
        # Interaction-systemets geometri ska nu läsas store-first via slotar.
        # Wrapperobjekten används här främst för kvarvarande intern biologilogik.
        self._step_predation(ctx)
    
        for a in self.agents:
            if not a.body.alive:
                continue
            self._try_mating(a, ctx)
            
    def _step_predation(self, ctx: "StepCtx") -> None:
        """
        Verkställ predation mot lokalt detekterade mål.
        """
        dt = float(ctx.dt)
    
        attack_range = float(self.AP.attack_range)
        dmg_per_s = float(self.AP.attack_damage_per_s)
        cost_frac = float(self.AP.attack_cost_per_s)
    
        attack_score_min = float(getattr(self.AP, "attack_score_min", 0.18))
        predator_trait_min = float(getattr(self.AP, "predator_trait_min", 0.20))
    
        for predator in self.agents:
            if not predator.body.alive:
                continue
    
            pred_trait = float(getattr(predator.pheno, "predation", 0.0))
            pred_diet = float(getattr(predator.pheno, "diet", 0.5))
            hunt_diet_exp = float(getattr(predator.AP, "hunt_diet_exp", 1.5))
            hunt_eff = pred_trait * (pred_diet ** hunt_diet_exp)
            if hunt_eff < predator_trait_min:
                continue
    
            hit = getattr(predator, "_cached_agent_hit", None)
            if not isinstance(hit, tuple) or len(hit) < 5:
                continue
    
            _, _, _, hit_slot, hit_id = hit[:5]
            if int(hit_slot) < 0:
                continue
    
            store = self.store
            s = int(hit_slot)
            if s >= int(store.n):
                continue
            if not bool(store.alive[s]) or int(store.kind[s]) != 0:
                continue
            if int(store.id[s]) != int(hit_id):
                continue
    
            prey = self._agent_for_slot(s)
            if prey is None or prey is predator:
                continue
            
            pred_slot = int(predator.store_slot)
            prey_slot = int(prey.store_slot)
            
            dist = self._slot_distance(pred_slot, prey_slot)
            if dist > attack_range:
                continue
    
            score = predator.attack_score(prey, dist)
            if score <= attack_score_min:
                continue
    
            mismatch_cost = float(getattr(predator.AP, "hunt_mismatch_cost", 2.0))
            cost_mult = 1.0 + (mismatch_cost - 1.0) * max(0.0, 1.0 - pred_diet)
            predator.body.take_energy(
                cost_frac * hunt_eff * cost_mult * float(predator.body.E_cap()) * dt
            )
            self._write_body_surface_to_store(predator.store_slot, predator)
    
            dD = dmg_per_s * max(0.25, score) * hunt_eff * (float(predator.body.M) ** 0.5) * dt
            prey.body.D = min(float(prey.body.D) + dD, float(prey.body.AP.D_max))
            self._write_body_surface_to_store(prey.store_slot, prey)
            
            if float(prey.body.D) >= float(prey.body.AP.D_max):
                # Dödsorsaken måste sättas här. Det här är den enda av sex
                # dödsvägar som dödar utanför `Body.step()`, och den satte
                # tidigare ingenting — `records.death_record` föll då tillbaka
                # på "unknown". Följden var att predationen var osynlig i
                # life-loggen och felaktigt beskrevs som död kod i tre
                # statusanalyser. Signaturen är entydig: samtliga 38 `unknown`
                # över p75, p77 och p78 har D = 1,000000 exakt, vilket bara
                # `min(D + dD, D_max)` ovan producerar.
                prey.body.death_cause = "predation"
                prey.body.alive = False
                self._write_alive_to_store(prey.store_slot, False)
    
    
    def _step_deaths(self) -> int:
        """
        Hantera död, carcass, slot-release och death events.
        Returnerar antal döda denna tick.
        """
        deaths = 0
        survivors: list[Agent] = []
    
        for a in self.agents:
            if not a.body.alive:
                # En död utan orsak betyder att någon väg dödar utanför de sex
                # kända, och den vägen är per definition oinstrumenterad.
                # Räknaren läses av check_death_cause_set i invariantsviten.
                if not str(getattr(a.body, "death_cause", "") or ""):
                    self._deaths_without_cause += 1

                self._banks[a._policy_key].release(a._policy_slot)
    
                # Strukturandelen måste läsas innan sloten frigörs; kadavret
                # ärver den från organismen som dog.
                struct = (
                    float(self.store.structure[int(a.store_slot)])
                    if a.store_slot >= 0
                    else 0.45
                )

                if a.store_slot >= 0:
                    self._write_alive_to_store(a.store_slot, False)
                    self._slot_to_agent[int(a.store_slot)] = None
                    self.store.release_slot(a.store_slot)
                a.store_slot = -1
    
                body = a.body

                # Töm kroppens utflöden innan den nollställs, annars går
                # exkrement och kväve från sista ticken förlorade.
                self._flush_body_outputs(a)

                M_tissue = float(body.M)
                M_res = float(body.M_reserve())
                M_fetus = float(body.gest_M) if bool(body.gestating) else 0.0
                carcass_kg = M_tissue + M_res + M_fetus

                # Reserven och fostret är labil vävnad; bara den committade
                # massan bär strukturmaterial. Kadavrets strukturandel blir
                # därmed massviktad, och en välgödd kropp lämnar ett mjukare
                # kadaver än en utsvulten. Den tidigare divisionen med
                # (1 - struktur) skalade upp reserven till kadaverekvivalenter
                # och skapade massa vid varje dödsfall.
                if carcass_kg > 1e-15:
                    s_carc = min(1.0, max(0.0, M_tissue * struct / carcass_kg))
                    self.world.add_carcass(
                        float(a.x),
                        float(a.y),
                        amount_kg=carcass_kg,
                        rad=int(self.PP.carcass_rad),
                        structure=s_carc,
                    )

                body.M = 0.0
                body.M_fast = 0.0
                body.M_slow = 0.0
                body.gest_M = 0.0
                body.gestating = False
    
                deaths += 1
    
                self._emit_death(
                    self.t,
                    a,
                    carcass_amount=carcass_kg,
                    carcass_rad=int(self.PP.carcass_rad),
                )
            else:
                survivors.append(a)
    
        self.agents = survivors
        return int(deaths)
    
    
    def _step_births(self, ctx: "StepCtx") -> int:
        """
        Hantera enbart födslar från redan etablerad gestation.
        Returnerar antal födda denna tick.
        """
        births = 0
        if len(self.agents) >= int(self.PP.max_pop):
            return 0
    
        children: list[Agent] = []
        cap = int(self.PP.max_pop)
    
        for a in self.agents:
            if len(self.agents) + len(children) >= cap:
                break
            if not a.body.alive:
                continue
    
            child = self._try_birth(a, ctx)
            if child is not None:
                children.append(child)
    
        if children:
            self.agents.extend(children)
        births = len(children)
        return int(births)
    
    def _step_metabolism_system(self, ctx: StepCtx) -> None:
        """
        Store-first metabolism-pass för enkel fauna-state.
        
        I detta steg ägs nu:
          - store.age
          - store.repro_cd
        
        Båda uppdateras direkt här och läses senare från store av övriga pass.
        """
        dt = float(ctx.dt)
        if dt <= 0.0:
            return
    
        store = self.store
        fa = self._fauna_slots(rebuild=True)
        if fa.size == 0:
            return

        store.age[fa] = (store.age[fa].astype(np.float64, copy=False) + dt).astype(np.float32)
        store.repro_cd[fa] = np.maximum(
            0.0, store.repro_cd[fa].astype(np.float64, copy=False) - dt
        ).astype(np.float32)
            
            
    def _step_sampling(self) -> None:
        """
        Emit sample event enligt samplinginställningar.
        """
        sd = float(self.PP.sample_dt)
        if not (sd > 0.0 and self.t + 1e-12 >= self._next_sample_t):
            return
    
        alive_now = [a for a in self.agents if a.body.alive]
        if alive_now:
            if int(self.PP.sample_avoid_repeat_k) > 0 and self._recent_sample_ids:
                k = int(self.PP.sample_avoid_repeat_k)
                recent = set(self._recent_sample_ids[-k:])
                pool = [a for a in alive_now if int(a.id) not in recent]
                if pool:
                    alive_now = pool
    
            a_pick = alive_now[int(self.rng.integers(0, len(alive_now)))]
            self._emit_sample(self.t, a_pick)
    
            if int(self.PP.sample_avoid_repeat_k) > 0:
                self._recent_sample_ids.append(int(a_pick.id))
    
        while self._next_sample_t <= self.t + 1e-12:
            self._next_sample_t += sd
    
    
    def _finalize_store_and_emit(
        self,
        *,
        dM_growth_flora: float,
        flora_established: int,
        flora_dispersed_mass: float,
        births: int,
        deaths: int,
    ) -> None:
        """
        Skriv tillbaka fauna till store, rebuild derived fields och emit loggar.
        """
        # OBS:
        # Faunas löpande store-state skrivs nu i subsystempassen:
        #   - spatial state i move_system
        #   - alive i body/interaction/death
        #   - mass/energy/damage/wear i body_system samt relevanta händelsepass
        # Därför görs ingen generell fauna-writeback här längre.
    
        self.store.rebuild_spatial_index()
        self._flora_summary_cache = None
    
        self._last_flora_growth = float(dM_growth_flora)
        self._last_flora_established = int(flora_established)
        self._last_flora_dispersed_mass = float(flora_dispersed_mass)
    
        self._emit_world(self.t)
        self._emit_population(self.t, births=self._births_total, deaths=self._deaths_total)
        
    # -----------------------
    # main loop
    # -----------------------
    def step(self) -> Tuple[int, int]:
        dt = float(self.WP.dt)
        self.t += dt
        ctx = StepCtx(t=float(self.t), dt=dt, rng=self.rng)

        # Fördröjd insättning. Sker före världspasset, så att djuren möter en
        # värld i samma tillstånd som om de stått där hela tiden.
        if self._fauna_pending > 0 and self._fauna_release_now():
            n = self._fauna_pending
            self._fauna_pending = 0
            made = self.seed_fauna(n)
            print(f"[fauna] {made} djur insatta vid tick {self._tick}"
                  + (f" i en fläck med radie {float(self.PP.fauna_spawn_radius):.1f}"
                     if float(self.PP.fauna_spawn_radius) > 0.0 else " jämnt utspridda"))
        self._tick += 1

        dM_growth_flora, flora_established, flora_dispersed_mass = self._step_world_and_flora()

        self._step_metabolism_system(ctx)
        sense_batch = self._step_sense_system(ctx)
        decision_batch = self._step_decision_system(ctx, sense_batch)
        body_batch = self._step_move_system(ctx, decision_batch)
        self._step_body_system(ctx, body_batch)
        self._step_interaction_system(ctx)
    
        deaths = self._step_deaths()
        births = self._step_births(ctx)
    
        self._births_total += births
        self._deaths_total += deaths
    
        self._step_sampling()
    
        self._finalize_store_and_emit(
            dM_growth_flora=dM_growth_flora,
            flora_established=flora_established,
            flora_dispersed_mass=flora_dispersed_mass,
            births=births,
            deaths=deaths,
        )
    
        return births, deaths

    def mean_stats(self) -> tuple[float, float, float, float, float]:
        if not self.agents:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        Es: list[float] = []
        Ds: list[float] = []
        Ms: list[float] = []
        Ecs: list[float] = []
        Rs: list[float] = []

        for a in self.agents:
            body = a.body
            Et = float(body.E_total())
            Ecap = float(body.E_cap())
            M = float(body.M)
            D = float(body.D)

            Es.append(Et)
            Ds.append(D)
            Ms.append(M)
            Ecs.append(Ecap)
            Rs.append(Et / max(Ecap, 1e-12))

        n = float(len(Es))
        return (
            float(sum(Es) / n),
            float(sum(Ds) / n),
            float(sum(Ms) / n),
            float(sum(Ecs) / n),
            float(sum(Rs) / n),
        )


@dataclass
class ParamBank:
    layer_sizes: tuple[int, ...]
    act: str
    capacity: int

    W: list[np.ndarray]
    b: list[np.ndarray]
    free: list[int]

    @classmethod
    def create(cls, layer_sizes: tuple[int, ...], act: str, capacity: int) -> "ParamBank":
        Ls = list(layer_sizes)
        W: list[np.ndarray] = []
        b: list[np.ndarray] = []
        for a, o in zip(Ls[:-1], Ls[1:]):
            W.append(np.zeros((capacity, o, a), dtype=np.float32))
            b.append(np.zeros((capacity, o), dtype=np.float32))
        free = list(range(capacity - 1, -1, -1))
        return cls(layer_sizes=layer_sizes, act=act, capacity=capacity, W=W, b=b, free=free)

    def alloc(self) -> int:
        if not self.free:
            raise RuntimeError("ParamBank full (increase capacity or handle growth).")
        return self.free.pop()

    def release(self, slot: int) -> None:
        self.free.append(int(slot))

    def write_genome(self, slot: int, g: MLPGenome) -> None:
        assert g.weights is not None and g.biases is not None
        for i in range(len(self.W)):
            self.W[i][slot] = g.weights[i]
            self.b[i][slot] = g.biases[i]


@dataclass(frozen=True)
class StepCtx:
    t: float
    dt: float
    rng: np.random.Generator