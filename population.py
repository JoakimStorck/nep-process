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

    carcass_yield: float = 0.65  # currently unused (carcass mass = remaining M)
    carcass_rad: int = 3

    # sampling
    sample_dt: float = 1.0
    sample_avoid_repeat_k: int = 0


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
        in_dim = 21
        out_dim = 5

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
            a.birth_t = float(self.t)

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
        mean_E, mean_D, mean_M, _, _ = self.mean_stats()

        payload = records.population_record(
            t=t,
            pop_n=len(self.agents),
            births=int(births),
            deaths=int(deaths),
            mean_E=float(mean_E),
            mean_D=float(mean_D),
            mean_M=float(mean_M),
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
    def _spawn_child(self, parent: Agent, child_M_from_parent: float | None = None) -> Agent:
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

        # newborn physiology (may incorporate mass provision)
        child.init_newborn_state(parent.pheno, child_M_from_parent=child_M_from_parent)

        self._emit_birth(self.t, child, parent)
        return child

    def _try_reproduce(self, parent: Agent) -> Optional[Agent]:
        if len(self.agents) >= int(self.PP.max_pop):
            return None
        if not parent.body.alive:
            return None
        if not parent.ready_to_reproduce():
            # (OBS: ready bör vara deterministisk gate, wanting stokastisk)
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
    
        # Om ingen massa kunde provisioneras: abort
        if m_got <= 0.0:
            return None
    
        # 2) Betala energikostnad proportionell mot faktisk barnmassa
        # Parametrar i AP (eller PP): birth_k_E_per_M och ev birth_E0
        E0 = float(getattr(self.AP, "birth_E0", 0.0))
        k  = float(self.AP.birth_k_E_per_M)
        costE = E0 + k * m_got
    
        paid = float(parent.pay_repro_cost(costE))
        if paid + 1e-9 < costE:
            # rollback massan (annars "förlorar" parent massa utan barn)
            parent.body.M = float(parent.body.M) + m_got
            return None
    
        # 3) Spawn
        return self._spawn_child(parent, child_M_from_parent=child_M_from_parent)

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

        # (A) world fields
        self.world.step()

        # (B) occupancy from current positions
        self.world.rebuild_agent_layer(self.agents)

        # (C) agent step (sense + policy + act)
        alive: List[Agent] = [a for a in self.agents if a.body.alive]
        if alive:
            n = len(alive)

            # derive in_dim from first alive agent's genome (safer than hardcoding)
            in_dim = int(alive[0].genome.layer_sizes[0])
            X = np.empty((n, in_dim), dtype=np.float32)

            # store (B0, C0) per-agent for apply_outputs
            BC_list: List[Tuple[float, float]] = [None] * n  # type: ignore

            for i, a in enumerate(alive):
                x_in, B0, C0 = a.build_inputs(self.world, rng=self.rng)

                # build_inputs returns (None,0,0) for dead; but we filtered alive, so assert-ish:
                if x_in is None:
                    # extremely defensive: treat as zero-input to avoid crash
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
                _ = a.apply_outputs(self.world, Y[i], B0, C0, rng=self.rng)

        # (D) deaths -> carcass (kg) + release bank slot + emit death
        deaths = 0
        survivors: List[Agent] = []
        for a in self.agents:
            if not a.body.alive:
                # release policy slot
                self._banks[a._policy_key].release(a._policy_slot)

                # carcass mass = remaining body mass (kg)
                carcass_kg = max(0.0, float(getattr(a.body, "M", 0.0)))
                if carcass_kg > 0.0:
                    self.world.add_carcass(
                        float(a.x),
                        float(a.y),
                        amount_kg=carcass_kg,
                        rad=int(self.PP.carcass_rad),
                    )

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

                child = self._try_reproduce(a)
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

        # (G) emit world + population, advance time
        self._emit_world(self.t)
        self._emit_population(self.t, births=births, deaths=deaths)

        self.t += dt
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