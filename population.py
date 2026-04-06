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
from phenotype import (
    derive_pheno,
    trait_lerp,
)

from organism_store import OrganismStore
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

@dataclass
class PopParams:
    init_pop: int = 12
    max_pop: int = 500   # höjt — naturlig matbrist sätter taket nu, inte detta

    n_traits: int = 32   # +2 arkitekturtraits (hidden_1=23, hidden_2=24)

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

    _next_store_id: int = field(init=False, default=1)

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
    
    _flora_by_cell_cache: dict[int, list[int]] = field(init=False, default_factory=dict)
    _flora_summary_cache: dict[str, float] | None = field(init=False, default=None)
    
    world_log_with_percentiles: bool = True

    def __post_init__(self) -> None:
        random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.world = World(self.WP)
        self.grid = Grid(size=int(self.WP.size))
        self._banks = {}

        self.world.consume_food_hook = self.consume_food
        self.world.sample_flora_local_hook = self.sample_flora_local
        self.world.sample_flora_rays_hook = self.sample_flora_rays        
        
        self.store = OrganismStore(
            capacity=int(self.PP.max_pop),
            world_size=int(self.WP.size),
        )
        
        self._next_sample_t = 0.0
        self._recent_sample_ids = []
        self._births_total = 0
        self._deaths_total = 0

        self._flora_by_cell_cache = {}
        self._flora_summary_cache = None

        # ensure MC uses PP.n_traits (single source of truth for this run)
        if int(self.MC.n_traits) != int(self.PP.n_traits):
            self.MC = replace(self.MC, n_traits=int(self.PP.n_traits))

        # IO dims: obs/act är rena bio-dimensioner; nätverket får h_dim extra i/o.
        _h_dim  = max(0, int(self.PP.h_dim))
        in_dim  = int(Agent.OBS_DIM) + _h_dim
        out_dim = int(Agent.OUT_DIM) + _h_dim

        self.agents = []
        for _ in range(int(self.PP.init_pop)):
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

            # Sprid ut agenter över hela världen
            x = float(self.rng.uniform(0.0, float(self.WP.size)))
            y = float(self.rng.uniform(0.0, float(self.WP.size)))

            a = Agent(
                AP=self.AP,
                genome=g,
                x=x,
                y=y,
                heading=float(self.rng.uniform(-math.pi, math.pi)),
            )
            a.bind_grid(self.grid)
            a.bind_world(self.world)

            # --- Warm start: åldersstrukturerad population ---
            # Ålder uniform från nyfödd till 3× mognadsåldern —
            # ger realistisk blandning av unga, vuxna och gamla.
            A_mature  = float(a.pheno.A_mature)
            age_s     = float(self.rng.uniform(0.0, 3.0 * A_mature))
            a.age_s   = age_s
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
            a.repro_cd_s = float(
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

            store_slot = self.store.alloc_slot()
            a.store_slot = int(store_slot)
            self.store.write_agent(store_slot, a, self.grid)            
            
            self._emit_birth(self.t, a, parent=None)
            self.agents.append(a)

        self._next_store_id = max((int(a.id) for a in self.agents), default=0) + 1

        _ = self._seed_initial_flora(
            n_flora=max(16, int(self.PP.max_pop) // 2),
        )

    # -----------------------
    # logging helpers
    # -----------------------
    def _emit(self, name: EventName, t: float, payload: dict) -> None:
        if self.hub is None:
            return
        self.hub.emit(Event(name=name, t=float(t), payload=payload))

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
        payload = records.world_record(
            t,
            self.world,
            with_percentiles=self.world_log_with_percentiles,
        )
        if isinstance(payload, dict):
            C_sum = float(np.nansum(self.world.C))
            e_plant = float(getattr(self.AP, "E_plant_J_per_kg", getattr(self.AP, "E_bio_J_per_kg", 0.0)))
            e_carc = float(getattr(self.AP, "E_carcass_J_per_kg", 0.0))
            wf = getattr(self.world, "last_flux", {})
    
            flora_info = self._flora_summary()
            flora_n = int(flora_info["flora_n"])
            flora_mass = float(flora_info["flora_mass_store"])
            flora_energy = float(flora_info["flora_energy_store"])
    
            payload.update({
                # Flora-store är source of truth
                "E_B": e_plant * flora_mass,
                "E_C": e_carc * C_sum,
                "BC_sum": flora_mass + C_sum,
                "M_B": flora_mass,
                "M_C": C_sum,
    
                "flora_n": flora_n,
                "flora_mass_store": flora_mass,
                "flora_energy_store": flora_energy,
    
                "flora_mean_growth_rate": float(flora_info["flora_mean_growth_rate"]),
                "flora_mean_adult_mass": float(flora_info["flora_mean_adult_mass"]),
                "flora_mean_temp_opt": float(flora_info["flora_mean_temp_opt"]),
                "flora_mean_temp_width": float(flora_info["flora_mean_temp_width"]),
                "flora_mean_dispersal_rate": float(flora_info["flora_mean_dispersal_rate"]),
    
                "E_in_growth": float(wf.get("E_in_growth", 0.0)),
                "E_loss_wither": float(wf.get("E_loss_wither", 0.0)),
                "E_loss_decay": float(wf.get("E_loss_decay", 0.0)),
                "dM_growth": float(wf.get("dM_growth", 0.0)),
                "dM_wither": float(wf.get("dM_wither", 0.0)),
                "dM_decay": float(wf.get("dM_decay", 0.0)),
                "flora_dM_growth": float(getattr(self, "_last_flora_growth", 0.0)),
                "flora_established": int(getattr(self, "_last_flora_established", 0)),
                "flora_dispersed_mass": float(getattr(self, "_last_flora_dispersed_mass", 0.0)),
            })
        self._emit("world", t, payload)

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

        store_slot = self.store.alloc_slot()
        child.store_slot = int(store_slot)
        self.store.write_agent(store_slot, child, self.grid)
        
        self._emit_birth(self.t, child, parent)
        return child

    def _try_mating(self, agent: Agent, ctx: StepCtx, candidates: list | None = None) -> None:
        """
        Sexuell reproduktion: verkställer parning med den agent som apply_outputs()
        lokalt detekterade och markerade via _desired_mate_id.
        Ingen global sökning — agenten kan bara para sig med någon den uppfattat via sensing.
        """
        if not agent.body.alive or agent.body.gestating:
            return
        if not agent.ready_to_reproduce():
            return

        desired_id = int(getattr(agent, '_desired_mate_id', 0))
        if desired_id <= 0:
            return

        best = self.world._agent_by_id.get(desired_id)
        if best is None or not best.body.alive or best is agent:
            return
        if not best.ready_to_reproduce():
            return

        # Avståndskontroll via Grid: Population ska inte bära egen torusmatematik.
        if self._pair_distance2(agent, best) > float(self.PP.mating_radius) ** 2:
            return

        # Genetisk kompatibilitet: P(parning lyckas) = exp(-d2_norm / 2*sigma2)
        if self.PP.compat_enabled:
            compat = genetic_compatibility(
                agent.genome, best.genome, sigma=float(self.PP.compat_sigma)
            )
            if self.rng.random() > compat:
                return  # genetiskt inkompatibla denna omgång — försök igen senare

        # Den tyngste bär fostret — mer resurser → bättre förälder
        if best.body.M > agent.body.M:
            bearer, partner = best, agent
        else:
            bearer, partner = agent, best

        # Starta gestation på bäraren med rekombinerat genom
        bearer._mating_partner = partner   # temporär referens för _try_birth
        bearer.start_gestation()

        # Partnern betalar liten parningskostnad och får cooldown
        mating_cost = 0.05 * partner.body.E_cap()
        partner.pay_repro_cost(mating_cost)
        partner.repro_cd_s = float(self.AP.repro_cooldown_s)
        
    def _try_birth(self, parent: Agent, ctx: StepCtx) -> Optional[Agent]:
        if not parent.body.alive:
            return None

        b = parent.body

        # Body är den enda auktoriteten: gestation_ready() kontrollerar gest_M >= gest_M_target.
        if not b.gestation_ready():
            return None

        # Hämta den massa som Body byggt upp under gestationen.
        child_M = float(b.gest_M)

        # Återställ gestationstillstånd via Body innan spawn
        # (abort_gestation() nollställer gestating, gest_M, gest_E_J, gest_M_target).
        b.abort_gestation()

        # --- 1.4: Energi till barnet (dras från föräldern) ---
        # child_E_fast/slow är fraktioner av barnets energikapacitet — en livshistoriestrategi.
        # Barnets Ecap beräknas från dess massa och den delade AP-konstanten.
        child_Ecap = float(self.AP.E_cap_per_M) * max(child_M, float(self.AP.M_min))
        child_E_fast_J = float(parent.pheno.child_E_fast) * child_Ecap
        child_E_slow_J = float(parent.pheno.child_E_slow) * child_Ecap

        # Föräldern betalar barnenergin ur sina egna buffrar.
        # pay_repro_cost() anropar body.take_energy() — aldrig mer än vad som finns.
        total_child_E = child_E_fast_J + child_E_slow_J
        paid_to_child = float(parent.pay_repro_cost(total_child_E))

        # Skala ner proportionellt om föräldern inte hade råd.
        if total_child_E > 1e-12:
            scale = paid_to_child / total_child_E
        else:
            scale = 0.0
        child_E_fast_J *= scale
        child_E_slow_J *= scale

        # --- 1.5: Reproduktionskostnad (extra föräldrastress vid födseln) ---
        # repro_cost är en fraktion av förälderns Ecap — kostnad utöver energin till barnet.
        repro_cost_J = float(parent.pheno.repro_cost) * float(parent.body.E_cap())
        parent.pay_repro_cost(repro_cost_J)

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

        parent.repro_cd_s = float(self.AP.repro_cooldown_s)
        return child

    def _pair_distance(self, a: Agent, b: Agent) -> float:
        return self.grid.distance_pos(
            float(a.x), float(a.y),
            float(b.x), float(b.y),
        )

    def _pair_distance2(self, a: Agent, b: Agent) -> float:
        return self.grid.distance2_pos(
            float(a.x), float(a.y),
            float(b.x), float(b.y),
        )


    def _rebuild_flora_caches(self) -> None:
        """
        Bygg per-tick-cacher för flora-tailen:
          - flora per cell
          - summeringar för world-logg
        """
        by_cell: dict[int, list[int]] = {}
    
        flora_n = 0
        flora_mass = 0.0
        flora_energy = 0.0
    
        vals_growth = 0.0
        vals_adult_mass = 0.0
        vals_temp_opt = 0.0
        vals_temp_width = 0.0
        vals_dispersal = 0.0
    
        for slot in range(int(self.store.n)):
            if not bool(self.store.alive[slot]):
                continue
            if int(self.store.kind[slot]) != 1:
                continue
    
            cell = int(self.store.cell_idx[slot])
            if cell >= 0:
                by_cell.setdefault(cell, []).append(int(slot))
    
            flora_n += 1
    
            m = float(self.store.mass[slot])
            e = float(self.store.energy[slot])
            flora_mass += m
            flora_energy += e
    
            vals_growth += float(self.store.flora_growth_rate[slot])
            vals_adult_mass += float(self.store.flora_adult_mass[slot])
            vals_temp_opt += float(self.store.flora_temp_opt[slot])
            vals_temp_width += float(self.store.flora_temp_width[slot])
            vals_dispersal += float(self.store.flora_dispersal_rate[slot])
    
        self._flora_by_cell_cache = by_cell
    
        if flora_n <= 0:
            nan = float("nan")
            self._flora_summary_cache = {
                "flora_n": 0,
                "flora_mass_store": 0.0,
                "flora_energy_store": 0.0,
                "flora_mean_growth_rate": nan,
                "flora_mean_adult_mass": nan,
                "flora_mean_temp_opt": nan,
                "flora_mean_temp_width": nan,
                "flora_mean_dispersal_rate": nan,
            }
        else:
            inv = 1.0 / float(flora_n)
            self._flora_summary_cache = {
                "flora_n": int(flora_n),
                "flora_mass_store": float(flora_mass),
                "flora_energy_store": float(flora_energy),
                "flora_mean_growth_rate": float(vals_growth * inv),
                "flora_mean_adult_mass": float(vals_adult_mass * inv),
                "flora_mean_temp_opt": float(vals_temp_opt * inv),
                "flora_mean_temp_width": float(vals_temp_width * inv),
                "flora_mean_dispersal_rate": float(vals_dispersal * inv),
            }

    def _flora_slots_grouped_by_cell(self) -> dict[int, list[int]]:
        if not self._flora_by_cell_cache:
            self._rebuild_flora_caches()
        return self._flora_by_cell_cache
    
    
    def _flora_summary(self) -> dict[str, float]:
        if self._flora_summary_cache is None:
            self._rebuild_flora_caches()
        return self._flora_summary_cache
        
    def _consume_flora_from_store(self, x: float, y: float, amount: float, max_radius: int = 1) -> float:
        """
        Konsumera växtmassa från diskret flora i OrganismStore.
        Returnerar faktiskt konsumerad kg.
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0
    
        cell0 = int(self.grid.cell_of(float(x), float(y)))
        flora_by_cell = self._flora_slots_grouped_by_cell()
    
        got = 0.0
        e_per_kg = float(self.WP.E_plant_J_per_kg)
    
        # Börja i aktuell cell, gå sedan utåt i topologiskt avstånd
        candidate_cells: list[int] = [cell0]
        for r in range(1, int(max_radius) + 1):
            for cell in self.grid.cells_within(cell0, r):
                if cell != cell0:
                    candidate_cells.append(int(cell))
    
        seen: set[int] = set()
        ordered_cells: list[int] = []
        for c in candidate_cells:
            if c not in seen:
                seen.add(c)
                ordered_cells.append(c)
    
        for cell in ordered_cells:
            slots = flora_by_cell.get(cell)
            if not slots:
                continue
    
            for slot in list(slots):
                if amt <= 1e-12:
                    break
                if not bool(self.store.alive[slot]) or int(self.store.kind[slot]) != 1:
                    continue
    
                m = float(self.store.mass[slot])
                if m <= 1e-12:
                    continue
    
                take = m if m < amt else amt
                new_m = m - take
    
                self.store.mass[slot] = np.float32(new_m)
                self.store.energy[slot] = np.float32(max(0.0, new_m * e_per_kg))
                got += take
                amt -= take
    
                # Dö/frigör flora som blivit i praktiken tom
                if new_m <= 1e-12:
                    old_cell = int(cell)                    
                    self.store.release_slot(slot)

            if amt <= 1e-12:
                break
                
        if got > 0.0:
            self._flora_summary_cache = None
            
        return float(got)

    def _add_or_create_flora_in_cell(
        self,
        cell: int,
        add_mass: float,
        traits: np.ndarray | None = None,
    ) -> bool:
        dm = float(add_mass)
        if not math.isfinite(dm) or dm <= 0.0:
            return False
    
        e_per_kg = float(self.WP.E_plant_J_per_kg)
    
        flora_by_cell = self._flora_slots_grouped_by_cell()
        slots = flora_by_cell.get(int(cell), None)
        if slots:
            for slot in slots:
                if not bool(self.store.alive[slot]):
                    continue
                if int(self.store.kind[slot]) != 1:
                    continue
        
                new_m = float(self.store.mass[slot]) + dm
                self.store.mass[slot] = np.float32(new_m)
                self.store.energy[slot] = np.float32(new_m * e_per_kg)
        
                self._flora_summary_cache = None
                return True
    
        try:
            slot = self.store.alloc_slot()
        except RuntimeError:
            return False
    
        if traits is None:
            traits = init_organism_traits(
                self.rng,
                int(self.PP.n_traits),
                mode="flora",
            )
    
        self._init_flora_slot(slot, int(cell), dm, traits)
        self._flora_by_cell_cache.setdefault(int(cell), []).append(int(slot))
        self._flora_summary_cache = None        
        return True
        
    def _flora_growth_rate(self, traits: np.ndarray | None) -> float:
        return trait_lerp(traits, 26, 0.005, 0.050, default=0.0)
    
    def _flora_adult_mass(self, traits: np.ndarray | None) -> float:
        return trait_lerp(traits, 27, 0.25 * float(self.WP.B_K), 4.0 * float(self.WP.B_K), default=0.5)
    
    def _flora_temp_opt(self, traits: np.ndarray | None) -> float:
        return trait_lerp(traits, 28, -5.0, 35.0, default=0.5)
    
    def _flora_temp_width(self, traits: np.ndarray | None) -> float:
        return trait_lerp(traits, 29, 4.0, 18.0, default=0.5)
    
    def _flora_dispersal_rate(self, traits: np.ndarray | None) -> float:
        return trait_lerp(traits, 31, 0.0002, 0.020, default=0.5)

    def _flora_uptake_capacity(self, traits: np.ndarray | None) -> float:
        # autotrophy-locus: högre värde => högre upptagskapacitet
        return trait_lerp(traits, 25, 0.25, 1.0, default=0.5)
    
    def _flora_repro_capacity(self, traits: np.ndarray | None) -> float:
        # sexual_mode nära 0 => asexuell flora med hög lokal reproduktionsbenägenhet
        sexual = trait_lerp(traits, 30, 0.0, 1.0, default=0.0)
        return float(max(0.0, 1.0 - sexual))
        
    def _init_flora_slot(
        self,
        slot: int,
        cell: int,
        mass: float,
        traits: np.ndarray,
    ) -> None:
        y, x = self.grid.rowcol_of(int(cell))
        e_per_kg = float(self.WP.E_plant_J_per_kg)

        g_rate = np.float32(self._flora_growth_rate(traits))
        a_mass = np.float32(self._flora_adult_mass(traits))
        t_opt = np.float32(self._flora_temp_opt(traits))
        t_width = np.float32(self._flora_temp_width(traits))
        d_rate = np.float32(self._flora_dispersal_rate(traits))

        self.store.id[slot] = int(self._next_store_id)
        self._next_store_id += 1
    
        self.store.alive[slot] = True
        self.store.kind[slot] = 1
        self.store.cell_idx[slot] = int(cell)
        self.store.pos_x[slot] = np.float32(x + 0.5)
        self.store.pos_y[slot] = np.float32(y + 0.5)
    
        m = float(mass)
        self.store.mass[slot] = np.float32(m)
        self.store.energy[slot] = np.float32(m * e_per_kg)
        self.store.age[slot] = np.float32(0.0)
        self.store.genome_idx[slot] = -1
        self.store.traits[slot, :] = np.asarray(traits, dtype=np.float32)
    
        self.store.flora_growth_rate[slot] = g_rate
        self.store.flora_adult_mass[slot] = a_mass
        self.store.flora_temp_opt[slot] = t_opt
        self.store.flora_temp_width[slot] = t_width
        self.store.flora_dispersal_rate[slot] = d_rate
        
        # Härled enkla store-kapaciteter från traits istället för hårdkodade 1.0/0.0
        self.store.uptake_capacity[slot] = np.float32(self._flora_uptake_capacity(traits))
        self.store.growth_capacity[slot] = np.float32(g_rate / 0.050)
        self.store.dispersal_capacity[slot] = np.float32(d_rate / 0.020)

        self.store.sense_radius[slot] = np.float32(0.0)
        self.store.sense_rate[slot] = np.float32(0.0)
        self.store.mobility[slot] = np.float32(0.0)
        self.store.attack_capacity[slot] = np.float32(0.0)
        self.store.repair_capacity[slot] = np.float32(0.0)
        self.store.repro_capacity[slot] = np.float32(self._flora_repro_capacity(traits))
    
        self.store.flood_tolerance[slot] = np.float32(0.0)
        self.store.buoyancy[slot] = np.float32(0.0)
        
    def _seed_initial_flora(
        self,
        n_flora: int | None = None,
        init_mass_frac_lo: float = 0.4,
        init_mass_frac_hi: float = 1.0,
    ) -> int:
        """
        Skapa initial diskret flora direkt i OrganismStore, utan world.B som mellanlager.
        """
        size = int(self.WP.size)
        n_cells = size * size
        BK = float(self.WP.B_K)
        if BK <= 0.0:
            return 0
    
        if n_flora is None:
            n_flora = max(16, int(self.PP.max_pop) // 2)
        n_flora = max(0, min(int(n_flora), n_cells))
    
        cells = self.rng.choice(n_cells, size=n_flora, replace=False)
    
        created = 0
        for cell in cells:
            try:
                slot = self.store.alloc_slot()
            except RuntimeError:
                break
    
            traits = init_organism_traits(
                self.rng,
                int(self.PP.n_traits),
                mode="flora",
            )
    
            mass = BK * float(self.rng.uniform(init_mass_frac_lo, init_mass_frac_hi))
            self._init_flora_slot(int(slot), int(cell), float(mass), traits)
            created += 1
    
        return created
        
    def _growth_system_flora(self) -> float:
        """
        Enkel första tillväxt för diskret flora i OrganismStore.
        Returnerar total producerad biomassa (kg) detta tick.
        """
        dt = float(self.WP.dt)
        BK = float(self.WP.B_K)
        if BK <= 0.0 or dt <= 0.0:
            return 0.0
    
        produced = 0.0
        e_per_kg = float(self.WP.E_plant_J_per_kg)
    
        for slot in range(int(self.store.n)):
            if not bool(self.store.alive[slot]):
                continue
            if int(self.store.kind[slot]) != 1:
                continue
        
            cell = int(self.store.cell_idx[slot])
            if cell < 0:
                continue
        
            row, _ = self.grid.rowcol_of(cell)
            T = float(self.world.Ty[row])
        
            m_cap = max(1e-12, float(self.store.flora_adult_mass[slot]))
            regen = max(0.0, float(self.store.flora_growth_rate[slot]))
        
            Topt = float(self.store.flora_temp_opt[slot])
            Tw = max(1e-6, float(self.store.flora_temp_width[slot]))
            gate = max(0.0, 1.0 - abs(T - Topt) / Tw)
            gate = max(0.0, min(1.0, gate))
        
            m = float(self.store.mass[slot])
            if m <= 0.0:
                continue
    
            # Enkel logistisk tillväxt
            dm = regen * gate * m * max(0.0, 1.0 - m / m_cap) * dt
            if dm <= 0.0:
                continue
    
            new_m = m + dm
            if new_m > m_cap:
                dm = m_cap - m
                new_m = m_cap
    
            if dm > 0.0:
                self.store.mass[slot] = np.float32(new_m)
                self.store.energy[slot] = np.float32(new_m * e_per_kg)
                produced += dm
    
        return float(produced)

    def _dispersal_system_flora(self) -> tuple[int, float]:
        """
        Enkel första spridning för diskret flora.
        Returnerar (antal etableringar/påfyllningar, totalt utspridd massa i kg).
        """
        dt = float(self.WP.dt)
        BK = float(self.WP.B_K)
        if BK <= 0.0 or dt <= 0.0:
            return 0, 0.0
    
        established = 0
        dispersed_mass = 0.0
    
        # Frys listan över kandidater detta tick så att nyfödda spridare
        # inte sprider vidare samma tick.
        parent_slots: list[int] = []
        for slot in range(int(self.store.n)):
            if bool(self.store.alive[slot]) and int(self.store.kind[slot]) == 1:
                parent_slots.append(slot)
    
        for slot in parent_slots:
            if not bool(self.store.alive[slot]):
                continue
            if int(self.store.kind[slot]) != 1:
                continue
    
            traits = self.store.traits[slot, :]
            
            m_cap = max(1e-12, float(self.store.flora_adult_mass[slot]))
            base_p = max(0.0, float(self.store.flora_dispersal_rate[slot]))
    
            repro_threshold = 0.70 * m_cap
            seed_mass = 0.10 * m_cap
            min_parent_mass_after = 0.20 * m_cap
            radius = 1 if m_cap < 1.5 * float(self.WP.B_K) else 2
    
            m = float(self.store.mass[slot])
            if m < repro_threshold:
                continue
            if m - seed_mass < min_parent_mass_after:
                continue
    
            # Enkel första sannolikhet per tick
            repro_cap = float(self.store.repro_capacity[slot])
            p = base_p * repro_cap * dt
            if self.rng.random() >= p:
                continue
    
            origin = int(self.store.cell_idx[slot])
            if origin < 0:
                continue
    
            candidates = [int(c) for c in self.grid.cells_within(origin, radius) if int(c) != origin]
            if not candidates:
                continue
    
            target = int(candidates[int(self.rng.integers(0, len(candidates)))])
    
            child_traits = mutate_trait_vector(
                traits,
                self.rng,
                sigma=0.05,
                p=0.10,
                clip=2.5,
            )
            ok = self._add_or_create_flora_in_cell(target, seed_mass, traits=child_traits)
            if not ok:
                continue
    
            # Betala från moderorganismen
            new_m = m - seed_mass
            e_per_kg = float(self.WP.E_plant_J_per_kg)
            self.store.mass[slot] = np.float32(new_m)
            self.store.energy[slot] = np.float32(max(0.0, new_m * e_per_kg))
    
            established += 1
            dispersed_mass += seed_mass
    
        return established, float(dispersed_mass)
        
    # -----------------------
    # public methods
    # -----------------------    
    def sample_flora_local(self, x: float, y: float) -> float:
        """
        Lokal biomassasampling av flora från store.
        Just nu: summerad massa i aktuell cell.
        """
        cell = int(self.grid.cell_of(float(x), float(y)))
        slots = self._flora_slots_grouped_by_cell().get(cell)
        if not slots:
            return 0.0
    
        total = 0.0
        for slot in slots:
            if not bool(self.store.alive[slot]):
                continue
            if int(self.store.kind[slot]) != 1:
                continue
            total += float(self.store.mass[slot])
        return float(total)

    def sample_flora_rays(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Enkel ray-sampling av flora.
        Returnerar biomassan i cellen för varje punkt (ingen bilinjär interpolation ännu).
        """
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        if xs.shape != ys.shape:
            raise ValueError("xs and ys must have same shape")
    
        out = np.zeros(xs.shape, dtype=np.float32)
        flora_by_cell = self._flora_slots_grouped_by_cell()
    
        flat_x = xs.ravel()
        flat_y = ys.ravel()
        flat_o = out.ravel()
    
        for i in range(flat_x.size):
            cell = int(self.grid.cell_of(float(flat_x[i]), float(flat_y[i])))
            slots = flora_by_cell.get(cell)
            if not slots:
                continue
    
            total = 0.0
            for slot in slots:
                if not bool(self.store.alive[slot]):
                    continue
                if int(self.store.kind[slot]) != 1:
                    continue
                total += float(self.store.mass[slot])
    
            flat_o[i] = np.float32(total)
    
        return out
        
    def consume_food(self, x: float, y: float, amount: float, prefer_carcass: bool = True) -> tuple[float, float]:
        """
        Konsumera upp till `amount` kg totalt.
        Kadaver tas fortsatt från world.C.
        Växtföda tas från diskret flora i OrganismStore.
        Returnerar (got_total_kg, got_carcass_kg).
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0
    
        got_c = 0.0
        if prefer_carcass:
            got_c = float(self.world._consume_bilinear_from(self.world.C, x, y, amt))
            amt = max(0.0, amt - got_c)
    
        got_f = float(self._consume_flora_from_store(x, y, amt, max_radius=1))
        return got_c + got_f, got_c
        
    # -----------------------
    # main loop
    # -----------------------
    def step(self) -> Tuple[int, int]:
        """
        One global tick:
          - world dynamics
          - agent sensing + batched policy forward + apply outputs
          - hazards -> acute damage D
          - pain+repair (E -> D)
          - metabolism/maintenance drain (E ->)
          - aging (W)
          - deaths -> carcass (+ death events)
          - births -> children (+ birth events)
          - emits: world/population/step/birth/death (gating in observers)
        """
        dt = float(self.WP.dt)
        self.t += dt
        ctx = StepCtx(t=float(self.t), dt=dt, rng=self.rng)
    
        # (A) world fields
        self.world.step()

        dM_growth_flora = self._growth_system_flora()
        flora_established, flora_dispersed_mass = self._dispersal_system_flora()
        self._rebuild_flora_caches()        
        
        # (B) occupancy from current positions
        self.world.rebuild_agent_layer(self.agents)
    
        # (C) agent step (sense + policy + act)
        alive: List[Agent] = [a for a in self.agents if a.body.alive]
        if alive:
            n = len(alive)
    
            # in_dim hämtas från nätverket — inkluderar h_dim om rekurrens är aktiv
            in_dim = int(alive[0].genome.layer_sizes[0])
            X = np.empty((n, in_dim), dtype=np.float32)
    
            # store (B0, C0) per-agent for apply_outputs
            BC_list: List[Tuple[float, float]] = [None] * n  # type: ignore
    
            for i, a in enumerate(alive):
                x_in, B0, C0 = a.build_inputs(self.world, rng=self.rng)
    
                # build_inputs returns (None,0,0) for dead; but we filtered alive, so assert-ish:
                if x_in is None:
                    X[i] = 0.0
                    BC_list[i] = (0.0, 0.0)
                    self._emit_step_if_tracked(self.t, a, 0.0, 0.0)
                    continue
    
                X[i] = x_in
                BC_list[i] = (float(B0), float(C0))
                self._emit_step_if_tracked(self.t, a, float(B0), float(C0))
    
            # group by bank key
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
    
            for i, a in enumerate(alive):
                B0, C0 = BC_list[i]
                _ = a.apply_outputs(self.world, ctx, Y[i], B0, C0)

        # (C2) Predation: verkställ bara lokalt uppfattade mål.
        # Populationen väljer inte byte globalt; den validerar bara den
        # granne agenten faktiskt detekterade via sensing i samma tick.
        attack_range = float(self.AP.attack_range)
        dmg_per_s    = float(self.AP.attack_damage_per_s)
        cost_frac    = float(self.AP.attack_cost_per_s)
        
        attack_score_min = float(getattr(self.AP, 'attack_score_min', 0.18))
        predator_trait_min = float(getattr(self.AP, 'predator_trait_min', 0.20))

        by_id = getattr(self.world, "_agent_by_id", {})
        for predator in self.agents:
            if not predator.body.alive:
                continue

            pred_trait = float(getattr(predator.pheno, "predation", 0.0))
            pred_diet  = float(getattr(predator.pheno, "diet",      0.5))
            hunt_diet_exp = float(getattr(predator.AP, 'hunt_diet_exp', 1.5))
            hunt_eff   = pred_trait * (pred_diet ** hunt_diet_exp)
            if hunt_eff < predator_trait_min:
                continue

            # build_inputs() cachear (N, Nu, Nd, hit_id) för faktiskt detekterad granne.
            hit = getattr(predator, "_cached_agent_hit", None)
            if not isinstance(hit, tuple) or len(hit) < 4:
                continue
            _, _, _, hit_id = hit[:4]
            if int(hit_id) < 0:
                continue

            prey = by_id.get(int(hit_id))
            if prey is None or prey is predator or (not prey.body.alive):
                continue

            dist = self._pair_distance(predator, prey)
            if dist > attack_range:
                continue            

            score = predator.attack_score(prey, dist)
            if score <= attack_score_min:
                continue

            # Jakt kostar energi oavsett utfall.
            # Mismatch-kostnad: herbivorer med hög predation betalar extra för otillräcklig utrustning.
            mismatch_cost = float(getattr(predator.AP, 'hunt_mismatch_cost', 2.0))
            cost_mult = 1.0 + (mismatch_cost - 1.0) * max(0.0, 1.0 - pred_diet)
            predator.body.take_energy(
                cost_frac * hunt_eff * cost_mult * float(predator.body.E_cap()) * dt
            )

            dD = dmg_per_s * max(0.25, score) * hunt_eff * (float(predator.body.M) ** 0.5) * dt
            prey.body.D = min(float(prey.body.D) + dD, float(prey.body.AP.D_max))

            # Om attacken driver bytet till dödströskeln dör det direkt.
            # Energiutbytet sker via carcass-konsumtion efter död — inte direkttransfer.
            if float(prey.body.D) >= float(prey.body.AP.D_max):
                prey.body.alive = False

        # (D) deaths -> carcass (kg) + release bank slot + emit death
        deaths = 0
        survivors: List[Agent] = []
    
        for a in self.agents:
            if not a.body.alive:
                # release policy slot
                self._banks[a._policy_key].release(a._policy_slot)

                if a.store_slot >= 0:
                    self.store.release_slot(a.store_slot)
                a.store_slot = -1
                
                body = a.body
    
                # carcass mass = structural mass + energy buffer converted to kg
                M_struct = float(body.M)
                E_buf = float(body.E_total())
                E_carcass = float(a.AP.E_carcass_J_per_kg)
    
                M_buf_equiv = E_buf / max(E_carcass, 1e-12)
                carcass_kg = M_struct + M_buf_equiv
    
                if carcass_kg > 0.0:
                    self.world.add_carcass(
                        float(a.x),
                        float(a.y),
                        amount_kg=carcass_kg,
                        rad=int(self.PP.carcass_rad),
                    )
    
                # Zero out body deterministically
                body.M = 0.0
                body.E_fast = 0.0
                body.E_slow = 0.0
    
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
    
        # (E) births (simple, deterministic pass; enforce cap)
        births = 0
        if len(self.agents) < int(self.PP.max_pop):
            children: List[Agent] = []
            cap = int(self.PP.max_pop)

            for a in self.agents:
                if len(self.agents) + len(children) >= cap:
                    break
                if not a.body.alive:
                    continue

                # (1) om gravid och klar -> föd
                child = self._try_birth(a, ctx)
                if child is not None:
                    children.append(child)
                    continue

                # (2) försök para sig — bygger på lokalt detekterad partner
                self._try_mating(a, ctx)

            if children:
                self.agents.extend(children)
            births = len(children)

        self._births_total += births
        self._deaths_total += deaths

        # (F) sampling (one agent per sample_dt)
        sd = float(self.PP.sample_dt)
        if sd > 0.0 and self.t + 1e-12 >= self._next_sample_t:
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

        for a in self.agents:
            if not a.body.alive:
                continue
            if a.store_slot < 0:
                continue
            self.store.write_agent(a.store_slot, a, self.grid)
            
        self._last_flora_growth = float(dM_growth_flora)
        self._last_flora_established = int(flora_established)
        self._last_flora_dispersed_mass = float(flora_dispersed_mass)

        # (G) emit world + population
        self._emit_world(self.t)
        self._emit_population(self.t, births=self._births_total, deaths=self._deaths_total)
    
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