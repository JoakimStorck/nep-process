# population.py
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


@dataclass
class PopParams:
    init_pop: int = 12
    max_pop: int = 2000

    n_traits: int = 20

    spawn_jitter_r: float = 1.5

    carcass_yield: float = 0.65
    carcass_rad: int = 3
    
    sample_dt: float = 1.0        # logga var 1.0 s (simtid)
    sample_avoid_repeat_k: int = 0 # 0=off, annars undvik senaste k ids

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

    # Agent sampling
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
        if self.MC.n_traits != int(self.PP.n_traits):
            self.MC = replace(self.MC, n_traits=int(self.PP.n_traits))

        # standard IO dims for NEP agent
        in_dim = 23
        out_dim = 5

        self.agents = []
        for _ in range(self.PP.init_pop):
            g = MLPGenome(layer_sizes=[in_dim, 24, 24, out_dim], act="tanh").init_random(
                self.rng, n_traits=self.PP.n_traits
            )

            a = Agent(
                AP=self.AP,
                genome=g,
                x=self.WP.size / 2 + float(self.rng.normal(0.0, 3.0)),
                y=self.WP.size / 2 + float(self.rng.normal(0.0, 3.0)),
                heading=float(self.rng.uniform(-math.pi, math.pi)),
            )
            a.bind_world(self.world)

            # birth time for age
            a.birth_t = float(self.t)
            a.init_newborn_state(a.pheno)            
            
            # allocate slot in bank and write genome params once
            key = (tuple(g.layer_sizes), str(g.act))
            bank = self._banks.get(key)
            if bank is None:
                bank = ParamBank.create(key[0], key[1], capacity=self.PP.max_pop)
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
        carcass_amount: float,
        carcass_rad: int,
        carcass_amount_field: float | None = None,
    ) -> None:
        self._emit(
            "death",
            t,
            records.death_record(
                t,
                agent,
                carcass_amount=carcass_amount,
                carcass_rad=carcass_rad,
                carcass_amount_field=carcass_amount_field,
            ),
        )
            
    def _emit_population(self, t: float, births: int, deaths: int) -> None:
        mean_E, mean_D = self.mean_stats()
    
        payload = records.population_record(
            t=t,
            pop_n=len(self.agents),
            births=births,
            deaths=deaths,
            mean_E=mean_E,
            mean_D=mean_D,
        )
    
        # Optional: birth-loop diagnostics (set in step(); absent => defaults)
        diag = getattr(self, "_last_birth_diag", None) or {}
        payload.update(
            {
                "repro_alive": int(diag.get("alive", 0)),
                "repro_wants": int(diag.get("wants", 0)),
                "repro_fail_mass": int(diag.get("fail_mass", 0)),
                "repro_spawned": int(diag.get("spawned", births)),
            }
        )
    
        self._emit("population", t, payload)
        
    def _emit_sample(self, t: float, a: Agent) -> None:
        self._emit("sample", t, records.sample_record(t, a, pop_n=len(self.agents)))
    
    def _emit_world(self, t: float) -> None:
        self._emit("world", t, records.world_record(t, self.world, with_percentiles=self.world_log_with_percentiles))

    def _emit_step_if_tracked(self, t: float, a: Agent, B0: float, F0: float, C0: float) -> None:
        if self.track_step_id is None:
            return
        if int(getattr(a, "id", -1)) != int(self.track_step_id):
            return
        self._emit("step", t, records.step_record(t, a, B0, F0, C0))

    # Genetics helpers
    def _age_s(self, a: Agent) -> float:
        return float(self.t - float(getattr(a, "birth_t", self.t)))

    def _can_attempt_repro(self, a: Agent) -> bool:
        if not a.body.alive:
            return False

        # local cooldown gate (kept in Agent)
        if float(getattr(a, "repro_cd_s", 0.0)) > 0.0:
            return False

        # maturity gate (phenotype)
        age = self._age_s(a)
        if age < float(a.pheno.A_mature):
            return False

        # energy gate (phenotype)
        if float(a.body.E_total()) < float(a.pheno.E_repro_min):
            return False

        # optional "health" gates (defaults: off)
        D_birth_max  = float(getattr(self.PP, "D_birth_max", 1e9))
        Fg_birth_max = float(getattr(self.PP, "Fg_birth_max", 1e9))

        if float(a.body.D) > D_birth_max:
            return False
        if float(a.body.Fg) > Fg_birth_max:
            return False

        return True
        
    # -----------------------
    # policy net + genetics
    # -----------------------
    def _act_hidden(self, x: np.ndarray, act: str) -> np.ndarray:
        if act == "softsign":
            return x / (1.0 + np.abs(x))
        return np.tanh(x)

    def _spawn_child(self, parent: Agent) -> Agent:
        dx = float(self.rng.normal(0.0, self.PP.spawn_jitter_r))
        dy = float(self.rng.normal(0.0, self.PP.spawn_jitter_r))
    
        g_child = child_genome_from_parent(parent.genome, rng=self.rng, cfg=self.MC)
    
        child = Agent(
            AP=self.AP,
            genome=g_child,
            x=torus_wrap(parent.x + dx, self.WP.size),
            y=torus_wrap(parent.y + dy, self.WP.size),
            heading=float(self.rng.uniform(-math.pi, math.pi)),
        )
    
        child.bind_world(self.world)
        child.birth_t = float(self.t)
    
        # ParamBank slot (policy weights cached by architecture+act)
        key = (tuple(g_child.layer_sizes), str(g_child.act))
        bank = self._banks.get(key)
        if bank is None:
            bank = ParamBank.create(key[0], key[1], capacity=self.PP.max_pop)
            self._banks[key] = bank
    
        slot = bank.alloc()
        bank.write_genome(slot, g_child)
    
        child._policy_key = key
        child._policy_slot = slot
    
        # newborn physiology: provision from parent phenotype
        # (requires: Phenotype has child_E_fast/child_E_slow/child_Fg and derive_pheno sets them)
        child.init_newborn_state(parent.pheno)
    
        self._emit_birth(self.t, child, parent)
        return child

    def _child_M_target(self, a: Agent) -> float:
        m = float(getattr(a.pheno, "child_M", 0.0))
        return max(float(a.AP.M_min), m)
    
    def _can_fund_child_mass(self, a: Agent, child_M: float) -> bool:
        # Parent must not go below M_min after funding child's mass
        return (float(a.body.M) - float(child_M)) >= float(a.AP.M_min)
    
    def _try_spawn_child(self, parent: Agent) -> Optional[Agent]:
        """
        Execute reproduction side-effects in a single place:
          - pay energy cost (sets cooldown)
          - transfer mass budget
          - spawn child + initialize newborn state
        Returns child if successful, else None.
        """
        child_M = self._child_M_target(parent)
    
        if not self._can_fund_child_mass(parent, child_M):
            return None
    
        # Pay reproduction energy cost (sets cooldown)
        _ = parent.pay_repro_cost(float(parent.pheno.repro_cost))
    
        # Transfer mass to child (hard budget)
        parent.body.M = float(parent.body.M) - child_M
    
        # Spawn child and initialize with funded mass
        child = self._spawn_child(parent)
        child.init_newborn_state(parent.pheno, child_M_from_parent=child_M)
        return child
    
    # -----------------------
    # main loop
    # -----------------------
    def step(self) -> Tuple[int, int]:
        """
        One global tick:
          - world dynamics
          - agent sensing + batched policy forward + apply outputs
          - deaths -> carcass (+ death events)
          - births -> children (+ birth events)
          - emits: world/population/step/birth/death (gating in observers)
        """
        dt = float(self.WP.dt)

         # (A) uppdatera världens fält
        self.world.step()
    
        # (B) bygg agent-occupancy från aktuella agentpositioner
        self.world.rebuild_agent_layer(self.agents)
    
        # (C) agentsteg (sensing + policy + move + eat + body)
        # 1) alive list (prealloc + no stack)
        alive: List[Agent] = [a for a in self.agents if a.body.alive]
        if alive:
            n = len(alive)
            X = np.empty((n, 23), dtype=np.float32)  # 23 = in_dim
            BFC_list: List[Tuple[float, float, float]] = [None] * n  # type: ignore

            for i, a in enumerate(alive):
                x_in, B0, F0, C0 = a.build_inputs(self.world, rng=self.rng)

                # Expect x_in to already be float32 and shape (23,).
                # If not, this assignment will still coerce/copy only when needed.
                X[i] = x_in
                BFC_list[i] = (B0, F0, C0)

                self._emit_step_if_tracked(self.t, a, B0, F0, C0)

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
                B0, F0, C0 = BFC_list[i]
                a.apply_outputs(self.world, Y[i], B0, F0, C0, rng=self.rng)

        # 2) deaths
        deaths = 0
        survivors: List[Agent] = []
        for a in self.agents:
            if not a.body.alive:
                self._banks[a._policy_key].release(a._policy_slot)
        
                M = float(getattr(a.body, "M", -1.0))
                Et = float(a.body.E_total())
                D  = float(a.body.D)
        
                # "mass carcass yield" (true mass units)
                amp_mass = float(self.PP.carcass_yield) * max(0.0, M)
        
                # map mass -> C field amplitude (0..1-ish)
                C_unit = float(getattr(self.world.P, "C_unit_mass", 1.0))  # WorldParams
                amp_field = amp_mass / max(1e-9, C_unit)
        
                # Optional visibility floor in *field units* (not mass units)
                AMP_FIELD_MIN = 0.02
                if amp_field > 0.0 and amp_field < AMP_FIELD_MIN:
                    amp_field = AMP_FIELD_MIN
        
                self.world.add_carcass(a.x, a.y, amount=amp_field, rad=self.PP.carcass_rad)        
                deaths += 1
        
                # IMPORTANT: log mass (true) and field amp (for debugging)
                self._emit_death(
                    self.t,
                    a,
                    carcass_amount=amp_mass,
                    carcass_amount_field=amp_field,
                    carcass_rad=self.PP.carcass_rad,
                )
            else:
                survivors.append(a)
        self.agents = survivors

        # 3) births (agent-decided; population executes + enforces max_pop)
        births = 0
        if len(self.agents) < self.PP.max_pop:
            children: List[Agent] = []

            n_alive = 0
            n_wants = 0
            n_fail_mass = 0
            n_spawned = 0

            for a in self.agents:
                if len(self.agents) + len(children) >= self.PP.max_pop:
                    break
                if not a.body.alive:
                    continue

                n_alive += 1

                if not a.wants_to_reproduce(rng=self.rng):
                    continue
                n_wants += 1

                child = self._try_spawn_child(a)
                if child is None:
                    n_fail_mass += 1
                    continue

                children.append(child)
                n_spawned += 1

            if children:
                self.agents.extend(children)
            births = len(children)

            self._last_birth_diag = {
                "alive": n_alive,
                "wants": n_wants,
                "fail_mass": n_fail_mass,
                "spawned": n_spawned,
            }

        if len(self.agents) >= self.PP.max_pop:
            self._last_birth_diag = {"alive": 0, "wants": 0, "fail_mass": 0, "spawned": 0}
    
        # 3.5) random cross-section sampling (one agent per sample_dt)
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

        # 4) advance time
        self._emit_world(self.t)
        self._emit_population(self.t, births=births, deaths=deaths)

        self.t += dt
        return births, deaths

    def mean_stats(self) -> Tuple[float, float]:
        if not self.agents:
            return 0.0, 0.0
        Es = [a.body.E_total() for a in self.agents]
        Ds = [a.body.D for a in self.agents]
        n = float(len(self.agents))
        return float(sum(Es) / n), float(sum(Ds) / n)


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

