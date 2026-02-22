from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from world import World, WorldParams
from mlp import MLPGenome
from agent import Agent, AgentParams
from genetics import child_genome_from_parent, MutationConfig

# new logging
from simlog.events import Event, EventName
from simlog.sinks import EventHub
from simlog import records


def torus_wrap(x: float, size: int) -> float:
    return x % size

_PCTS = (10, 25, 75, 90)

def _stats_1d(x: np.ndarray) -> dict[str, float]:
    # x: float64 array
    if x.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }

    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }

    p = np.percentile(x, _PCTS)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(p[0]),
        "p25": float(p[1]),
        "p75": float(p[2]),
        "p90": float(p[3]),
    }

@dataclass
class PopParams:
    init_pop: int = 12
    max_pop: int = 2000

    n_traits: int = 20

    spawn_jitter_r: float = 1.5

    carcass_yield: float = 0.65  # currently unused (carcass mass = remaining M)
    carcass_rad: int = 3

    # sampling
    sample_dt: float = 1.0
    sample_avoid_repeat_k: int = 0

    warm_age_max_s: float = 60.0
    warm_cd_max_s: float = 8.0  # eller använd AP.repro_cooldown_s och skippa denna


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

    # world logging knobs (gating lives in logger)
    world_log_with_percentiles: bool = True

    def __post_init__(self) -> None:
        random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.world = World(self.WP)
        self._banks = {}

        self._next_sample_t = 0.0
        self._recent_sample_ids = []

        # ensure MC uses PP.n_traits (single source of truth for this run)
        if int(self.MC.n_traits) != int(self.PP.n_traits):
            self.MC = replace(self.MC, n_traits=int(self.PP.n_traits))

        # IO dims (from Agent._build_obs):
        # obs(8) + trace(8) + extra([cos aB, sin aB, cos aC, sin aC, D]) = 21
        in_dim = Agent.OBS_DIM
        out_dim = Agent.OUT_DIM

        self.agents = []
        for _ in range(int(self.PP.init_pop)):
            g = MLPGenome(layer_sizes=[in_dim, 24, 24, out_dim], act="tanh").init_random(
                self.rng, n_traits=int(self.PP.n_traits)
            )

            a = Agent(
                AP=self.AP,
                genome=g,
                x=float(self.WP.size / 2 + self.rng.normal(0.0, 3.0)),
                y=float(self.WP.size / 2 + self.rng.normal(0.0, 3.0)),
                heading=float(self.rng.uniform(-math.pi, math.pi)),
            )
            a.bind_world(self.world)

            # birth time bookkeeping (optional; age is tracked inside Agent)
            # warm-start: spread ages so they don't all mature at once
            warm_age_max = float(self.PP.warm_age_max_s)  # t.ex. 60.0
            # gör en del individer “äldre” direkt; skala med A_mature så de inte alla når maturity samtidigt
            age0 = float(self.rng.uniform(0.0, 2.0 * float(a.pheno.A_mature)))
            a.birth_t = float(self.t - age0)

            a.body.M *= float(self.rng.uniform(0.9, 1.05))
            a.body.scale_energy(self.rng.uniform(0.85, 1.10))
            a.body.clamp_energy_to_cap()
            a.repro_cd_s = float(self.rng.uniform(0.0, float(self.AP.repro_cooldown_s)))

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

            # optional: log initial agents as births (parent=None)
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
    
        sE = _stats_1d(E)
        sD = _stats_1d(D)
        sM = _stats_1d(M)
    
        # Backward compatible: mean_* som tidigare
        # Nya fält: median_* och pXX_*
        payload = records.population_record(
            t=t,
            pop_n=pop_n,
            births=int(births),
            deaths=int(deaths),
        
            mean_E=sE["mean"],
            mean_D=sD["mean"],
            mean_M=sM["mean"],
        
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
    
        self._emit("population", t, payload)

    def _emit_sample(self, t: float, a: Agent) -> None:
        self._emit("sample", t, records.sample_record(t, a, pop_n=len(self.agents)))

    def _emit_world(self, t: float) -> None:
        self._emit(
            "world",
            t,
            records.world_record(t, self.world, with_percentiles=self.world_log_with_percentiles),
        )

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

    def _try_reproduce(self, parent: Agent, ctx: StepCtx) -> Optional[Agent]:
        if len(self.agents) >= int(self.PP.max_pop):
            return None
        if not parent.body.alive:
            return None
        if not parent.ready_to_reproduce():
            return None
        if not parent.wants_to_reproduce(self.rng):
            return None
    
        # 1) Provisionera barnmassa (faktisk)
        child_M_target = float(getattr(parent.pheno, "child_M", 0.0))
        child_M_from_parent = None
        m_got = 0.0
        if child_M_target > 0.0:
            m_got = float(parent.provide_child_mass(child_M_target))
            child_M_from_parent = m_got if m_got > 0.0 else None
    
        if m_got <= 0.0:
            return None
    
        # 2) Betala energikostnad proportionell mot faktisk barnmassa
        E0 = float(getattr(self.AP, "birth_E0", 0.0))
        k  = float(self.AP.birth_k_E_per_M)
        costE = E0 + k * m_got
    
        paid = float(parent.pay_repro_cost(costE))
        if paid + 1e-9 < costE:
            parent.body.M = float(parent.body.M) + m_got
            return None
    
        # 2B) Allokera del av paid som barnets initiala energireserv
        birth_energy_eff = float(getattr(self.AP, "birth_energy_eff", 0.70))  # ny knob, default 0.70
        birth_energy_eff = max(0.0, min(1.0, birth_energy_eff))
        E_child = birth_energy_eff * paid
    
        child_E_fast_J = 0.85 * E_child
        child_E_slow_J = 0.15 * E_child
    
        # 3) Spawn
        child = self._spawn_child(
            parent,
            ctx,
            child_M_from_parent=child_M_from_parent,
            child_E_fast_J=child_E_fast_J,
            child_E_slow_J=child_E_slow_J,
        )
        
        parent.repro_cd_s = float(parent.AP.repro_cooldown_s)
        
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
    
        # (C) agent step (sense + policy + act)
        alive: List[Agent] = [a for a in self.agents if a.body.alive]
        if alive:
            n = len(alive)
    
            # derive in_dim from first alive agent's genome (safer than hardcoding)
            in_dim = int(Agent.OBS_DIM)
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
    
#        # ------------------------------------------------------------
#        # (C2) hazards/incidents -> D  (koppla in din befintliga logik här)
#        # ------------------------------------------------------------
#        # Om du redan har hazard-logik som uppdaterar body.D: anropa den här.
#        # Exempel: a.apply_hazards(self.world, ctx, rng=self.rng) eller body.apply_hazards(...)
#        # Lämnad som hook för att undvika att vi gissar fel funktionsnamn.
#        alive = [a for a in self.agents if a.body.alive]
#        if alive:
#            for a in alive:
#                # Hook 1: Agent-nivå
#                if hasattr(a, "apply_hazards"):
#                    a.apply_hazards(self.world, ctx, rng=self.rng)  # type: ignore
#                # Hook 2: Body-nivå
#                elif hasattr(a.body, "apply_hazards"):
#                    a.body.apply_hazards(self.world, ctx, a.AP, rng=self.rng)  # type: ignore
#    
#        # ------------------------------------------------------------
#        # (C3) pain + repair (E -> D) + aging (W)
#        # ------------------------------------------------------------
#        # Vi vill ha dD_pos före step_pain_and_repair (eftersom den uppdaterar _D_prev).
#        if alive:
#            # (valfritt) om du vill kunna logga/ackumulera total repair spend per tick
#            repair_E_spent_tick = 0.0
#    
#            # Om du har metabolism/maintenance-drain som separat steg: hook här.
#            # Vi kör pain/repair först (hazard -> pain -> repair), sen metabolism.
#            for a in alive:
#                b = a.body
#    
#                # dD_pos baserat på "rå" D-ändring från hazardfasen
#                dD = (float(b.D) - float(b._D_prev)) / max(1e-9, dt)
#                dD_pos = dD if dD > 0.0 else 0.0
#    
#                if hasattr(b, "step_pain_and_repair"):
#                    repair_E_spent = b.step_pain_and_repair(ctx)
#                    repair_E_spent_tick += float(repair_E_spent)
#                else:
#                    repair_E_spent = 0.0
#    
#                # Metabolism/maintenance hook (om du har den explicit)
#                # Ex: b.step_metabolism(ctx, a.AP) eller a.step_metabolism(...)
#                E_spent_other = 0.0
#                if hasattr(b, "step_metabolism"):
#                    E_spent_other = float(b.step_metabolism(ctx, a.AP))  # type: ignore
#                elif hasattr(a, "step_metabolism"):
#                    E_spent_other = float(a.step_metabolism(self.world, ctx))  # type: ignore
#    
#                # Aging/W
#                if hasattr(b, "step_aging"):
#                    # repro_cost_paid: koppla in när reproduktion faktiskt betalas
#                    b.step_aging(
#                        ctx,
#                        E_spent_total=float(repair_E_spent) + float(E_spent_other),
#                        repro_cost_paid=0.0,
#                        dD_pos=float(dD_pos),
#                    )
#    
#            # Om du vill: spara tick-summan för observers/loggar
#            self._last_repair_E_spent_tick = repair_E_spent_tick  # valfritt attribut
    
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
    
            for a in self.agents:
                if len(self.agents) + len(children) >= cap:
                    break
                if not a.body.alive:
                    continue
    
                # IMPORTANT: ctx in signature to stop churn
                child = self._try_reproduce(a, ctx)
                if child is not None:
                    children.append(child)
    
            if children:
                self.agents.extend(children)
            births = len(children)
    
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
        self._emit_population(self.t, births=births, deaths=deaths)
    
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