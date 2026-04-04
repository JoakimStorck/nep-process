from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from world import World, WorldParams
from mlp import MLPGenome
from agent import Agent, AgentParams, torus_wrap
from genetics import child_genome_from_parent, recombine, MutationConfig, genetic_compatibility
from phenotype import derive_pheno

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

    n_traits: int = 25   # +2 arkitekturtraits (hidden_1=23, hidden_2=24)

    spawn_jitter_r: float = 1.5

    carcass_yield: float = 0.65  # currently unused (carcass mass = remaining M)
    carcass_rad: int = 3

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

    # optional: pass from runner; if None, no logging
    hub: Optional[EventHub] = None

    # optional: only emit "step" events for this id (keeps logging cheap)
    track_step_id: Optional[int] = None

    world: World = field(init=False)
    agents: List[Agent] = field(init=False, default_factory=list)

    t: float = 0.0
    rng: np.random.Generator = field(init=False)

    _banks: dict[tuple, "ParamBank"] = field(init=False, default_factory=dict)

    # agent sampling
    _next_sample_t: float = 0.0
    _recent_sample_ids: List[int] = field(default_factory=list)

    # Kumulativa totaler (aldrig nollställda) — samma semantik som konsolutskriften.
    # Analysverktyg kan diffa konsekutiva poster för att få per-period-värden.
    _births_total: int = 0
    _deaths_total: int = 0

    # world logging knobs (gating lives in logger)
    world_log_with_percentiles: bool = True

    def __post_init__(self) -> None:
        random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.world = World(self.WP)
        self._banks = {}

        self._next_sample_t = 0.0
        self._recent_sample_ids = []
        self._births_total = 0
        self._deaths_total = 0

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

            self._emit_birth(self.t, a, parent=None)
            self.agents.append(a)

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
        payload = records.world_record(t, self.world, with_percentiles=self.world_log_with_percentiles)
        if isinstance(payload, dict):
            B_sum = float(np.nansum(self.world.B))
            C_sum = float(np.nansum(self.world.C))
            e_plant = float(getattr(self.AP, "E_plant_J_per_kg", getattr(self.AP, "E_bio_J_per_kg", 0.0)))
            e_carc = float(getattr(self.AP, "E_carcass_J_per_kg", 0.0))
            wf = getattr(self.world, "last_flux", {})
            payload.update({
                "E_B": e_plant * B_sum,
                "E_C": e_carc * C_sum,
                "BC_sum": B_sum + C_sum,
                "M_B": B_sum,
                "M_C": C_sum,
                "E_in_growth": float(wf.get("E_in_growth", 0.0)),
                "E_loss_wither": float(wf.get("E_loss_wither", 0.0)),
                "E_loss_decay": float(wf.get("E_loss_decay", 0.0)),
                "dM_growth": float(wf.get("dM_growth", 0.0)),
                "dM_wither": float(wf.get("dM_wither", 0.0)),
                "dM_decay": float(wf.get("dM_decay", 0.0)),
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
    def _spawn_child(self, 
        parent: Agent, 
        ctx: StepCtx, 
        child_M_from_parent: float | None,
        child_E_fast_J: float,
        child_E_slow_J: float,
    ) -> Agent:
        dx = float(self.rng.normal(0.0, float(self.PP.spawn_jitter_r)))
        dy = float(self.rng.normal(0.0, float(self.PP.spawn_jitter_r)))
    
        g_child = child_genome_from_parent(parent.genome, rng=self.rng, cfg=self.MC)
    
        child = Agent(
            AP=self.AP,
            genome=g_child,
            x=torus_wrap(float(parent.x) + dx, int(self.WP.size)),
            y=torus_wrap(float(parent.y) + dy, int(self.WP.size)),
            heading=float(self.rng.uniform(-math.pi, math.pi)),
        )
        child.bind_world(self.world)
        child.birth_t = float(self.t)
    
        # ParamBank slot
        key = (tuple(g_child.layer_sizes), str(g_child.act))
        bank = self._banks.get(key)
        if bank is None:
            bank = ParamBank.create(key[0], key[1], capacity=int(self.PP.max_pop))
            self._banks[key] = bank
    
        slot = bank.alloc()
        bank.write_genome(slot, g_child)
    
        child._policy_key = key
        child._policy_slot = slot
    
        # newborn physiology (mass + ENERGY comes from parent)
        child.init_newborn_state(
            parent.pheno,
            child_M_from_parent=child_M_from_parent,
            child_E_fast_J=child_E_fast_J,
            child_E_slow_J=child_E_slow_J,
        )
    
        self._emit_birth(self.t, child, parent)
        return child

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

        child = Agent(
            AP=self.AP,
            genome=g_child,
            x=torus_wrap(float(parent.x) + dx, int(self.WP.size)),
            y=torus_wrap(float(parent.y) + dy, int(self.WP.size)),
            heading=float(self.rng.uniform(-math.pi, math.pi)),
        )
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

        self._emit_birth(self.t, child, parent)
        return child

    def _try_mating(self, agent: Agent, ctx: StepCtx, candidates: list | None = None) -> None:
        """
        Sexuell reproduktion: agent maste mota en annan redo individ inom mating_radius.
        Den tyngste bararen startar gestation med rekombinerat barnets genom.
        Den andre betalar en liten parningskostnad och far cooldown.

        candidates: foerfiltrerad lista av verkligt redo agenter fran population.step().
                    Listan ar redan filtrerad pa alla ready_to_reproduce-villkor, sa
                    inre loopen bara behoever kolla distans och kompatibilitet.
                    Om None anvands self.agents (fallback med full kontroll).
        """
        if not agent.body.alive or agent.body.gestating:
            return

        # Om candidates ar None (fallback): kolla full gate
        if candidates is None and not agent.ready_to_reproduce():
            return

        r_max     = float(self.PP.mating_radius)
        size_f    = float(self.WP.size)
        half_size = size_f * 0.5
        r_sq      = r_max * r_max
        best      = None
        best_d_sq = r_sq

        px = float(agent.x)
        py = float(agent.y)

        search = candidates if candidates is not None else self.agents
        for other in search:
            if other is agent or not other.body.alive:
                continue
            # candidates ar foer-filtrerade: hoppa over ready_to_reproduce() haer
            if candidates is None and not other.ready_to_reproduce():
                continue

            # Torus-distans utan abs()/min()/sqrt() — snabbare med halvstorlek-jämförelse
            ddx = float(other.x) - px
            ddy = float(other.y) - py
            if ddx >  half_size: ddx -= size_f
            elif ddx < -half_size: ddx += size_f
            if ddy >  half_size: ddy -= size_f
            elif ddy < -half_size: ddy += size_f
            d_sq = ddx * ddx + ddy * ddy

            if d_sq < best_d_sq:
                best_d_sq = d_sq
                best      = other

        if best is None:
            return   # ingen lämplig partner hittad

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
    
        # (B) occupancy from current positions
        self.world.rebuild_agent_layer(self.agents)
        self.world._agents = self.agents
    
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

        # (C2) Predation: verklig jakt = attack på levande byte.
        # Energi stjäls inte längre direkt ur levande byte.
        # Predatorn orsakar skada; om bytet dör blir kroppen carcass i (D),
        # och energiutbytet sker senare via vanlig konsumtion av carcass.
        attack_range = float(self.AP.attack_range)
        dmg_per_s    = float(self.AP.attack_damage_per_s)
        cost_frac    = float(self.AP.attack_cost_per_s)
        size_f       = float(self.WP.size)
        dt_val       = float(self.WP.dt)

        alive_now = [a for a in self.agents if a.body.alive]
        for predator in alive_now:
            pred_trait = float(getattr(predator.pheno, "predation", 0.0))
            if pred_trait < float(getattr(self.AP, 'predator_trait_min', 0.20)) or not predator.body.alive:
                continue

            best_prey = None
            best_score = -1e9
            for prey in alive_now:
                if prey is predator or not prey.body.alive:
                    continue
                _, _, dist = predator._torus_delta_to(prey, size_f)
                if dist > attack_range:
                    continue
                sc = predator.attack_score(prey, dist)
                if sc > best_score:
                    best_score = sc
                    best_prey = prey

            if best_prey is None or best_score <= float(getattr(self.AP, 'attack_score_min', 0.18)):
                continue

            # Jakt kostar energi oavsett utfall.
            predator.body.take_energy(
                cost_frac * pred_trait * float(predator.body.E_cap()) * dt_val
            )

            prey = best_prey
            dD = dmg_per_s * max(0.25, best_score) * pred_trait * (float(predator.body.M) ** 0.5) * dt_val
            prey.body.D = min(float(prey.body.D) + dD, float(prey.body.AP.D_max))

            # Om attacken driver bytet till dödströskeln dör det direkt.
            # Själva energiutbytet kommer inte här utan via carcass-steget nedan.
            if float(prey.body.D) >= float(prey.body.AP.D_max):
                prey.body.alive = False

        # (D) deaths -> carcass (kg) + release bank slot + emit death
        deaths = 0
        survivors: List[Agent] = []
    
        for a in self.agents:
            if not a.body.alive:
                # release policy slot
                self._banks[a._policy_key].release(a._policy_slot)
    
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

            # En enda auktoritet för reproduktiv readiness.
            _mating_candidates = [a for a in self.agents if a.ready_to_reproduce()]

            _cand_ids = {id(a) for a in _mating_candidates}

            for a in self.agents:
                if len(self.agents) + len(children) >= cap:
                    break
                if not a.body.alive:
                    continue

                # (1) om gravid och klar -> foerd
                child = self._try_birth(a, ctx)
                if child is not None:
                    children.append(child)
                    continue

                # (2) foersoek para sig -- bara foer verkligt redo kandidater
                if id(a) in _cand_ids:
                    self._try_mating(a, ctx, candidates=_mating_candidates)

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