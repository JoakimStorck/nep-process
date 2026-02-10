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

    n_traits: int = 14

    #D_birth_max: float = 0.40
    #Fg_birth_max: float = 0.70

    #E_margin_scale: float = 0.25
    #hunger_weight: float = 1.00

    spawn_jitter_r: float = 1.5

    carcass_yield: float = 0.65
    carcass_rad: int = 3


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

    # world logging knobs (gating lives in logger)
    world_log_with_percentiles: bool = True

    def __post_init__(self) -> None:
        random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.world = World(self.WP)
        self._banks = {}
    
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

    def _emit_death(self, t: float, agent: Agent, carcass_amount: float, carcass_rad: int) -> None:
        self._emit("death", t, records.death_record(t, agent, carcass_amount, carcass_rad))

    def _emit_population(self, t: float, births: int, deaths: int) -> None:
        mean_E, mean_D = self.mean_stats()
        self._emit(
            "population",
            t,
            records.population_record(
                t=t,
                pop_n=len(self.agents),
                births=births,
                deaths=deaths,
                mean_E=mean_E,
                mean_D=mean_D,
            ),
        )

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

        # optional "health" gates (keep for now)
        if float(a.body.D) > float(self.PP.D_birth_max):
            return False
        if float(a.body.Fg) > float(self.PP.Fg_birth_max):
            return False

        return True

    def _p_birth(self, a: Agent) -> float:
        """
        MV0: phenotype base probability times simple monotone modifiers.
        Keep cheap and stable. Clamp [0,1].
        """
        p0 = float(a.pheno.p_repro_base)

        # Optional shaping by energy margin above minimum
        if self.PP.E_margin_scale > 0.0:
            E = float(a.body.E_total())
            Emin = float(a.pheno.E_repro_min)
            margin = max(0.0, E - Emin)
            # saturating ramp: margin / (margin + scale)
            s = float(self.PP.E_margin_scale)
            m = margin / (margin + s)
        else:
            m = 1.0

        # Hunger modifier (prefer reproduction when not starving)
        h = float(a.body.hunger())
        hw = float(self.PP.hunger_weight)
        hm = max(0.0, 1.0 - h) ** max(0.0, hw)

        p = p0 * m * hm
        return float(max(0.0, min(1.0, p)))
        
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

        # 0) advance world first
        self.world.step()

        # 1) alive list
        alive: List[Agent] = [a for a in self.agents if a.body.alive]
        if alive:
            X_list: List[np.ndarray] = []
            BFC_list: List[Tuple[float, float, float]] = []

            for a in alive:
                x_in, B0, F0, C0 = a.build_inputs(self.world, rng=self.rng)
                X_list.append(x_in)
                BFC_list.append((B0, F0, C0))
                self._emit_step_if_tracked(self.t, a, B0, F0, C0)

            X = np.stack(X_list, axis=0).astype(np.float32, copy=False)

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

                amp = self.PP.carcass_yield * float(a.body.E_total())
                self.world.add_carcass(a.x, a.y, amount=amp, rad=self.PP.carcass_rad)

                deaths += 1
                self._emit_death(self.t, a, carcass_amount=amp, carcass_rad=self.PP.carcass_rad)
            else:
                survivors.append(a)
        self.agents = survivors

        # 3) births (agent-decided; population just executes + enforces max_pop)
        births = 0
        if len(self.agents) < self.PP.max_pop:
            children: List[Agent] = []
            for a in self.agents:
                if len(self.agents) + len(children) >= self.PP.max_pop:
                    break
                if not a.body.alive:
                    continue
        
                if a.wants_to_reproduce(rng=self.rng):   # agent decides
                    a.pay_repro_cost(float(a.pheno.repro_cost))
                    children.append(self._spawn_child(a))
        
            if children:
                self.agents.extend(children)
            births = len(children)

        # 4) advance time (convention: events refer to state at this.t, before increment)
        # If you prefer "after", move this up and pass self.t after increment.
        # Here we keep your previous convention stable.
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

