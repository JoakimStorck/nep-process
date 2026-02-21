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
            #a.init_newborn_state(a.pheno)            
            
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
        carcass_amount: float,  # kg
        carcass_rad: int,
    ) -> None:
        self._emit(
            "death",
            t,
            records.death_record(
                t=t,
                agent=agent,
                carcass_amount=float(carcass_amount),  # kg
                carcass_rad=int(carcass_rad),
            ),
        )
            
    def _emit_population(self, t: float, births: int, deaths: int) -> None:
        mean_E, mean_D, mean_M, mean_Ecap, mean_R = self.mean_stats()
    
        payload = records.population_record(
            t=t,
            pop_n=len(self.agents),
            births=int(births),
            deaths=int(deaths),
            mean_E=float(mean_E),
            mean_D=float(mean_D),
            mean_M=float(mean_M),
        )
    
        diag = getattr(self, "_last_birth_diag", None) or {}
    
        s = payload.get("summary", {})
        if isinstance(s, dict):
            s["repro"] = {
                "alive": int(diag.get("alive", 0)),
                "eligible": int(diag.get("eligible", 0)),
                "block_cd": int(diag.get("block_cd", 0)),
                "block_age": int(diag.get("block_age", 0)),
                "block_mass": int(diag.get("block_mass", 0)),
                "block_energy": int(diag.get("block_energy", 0)),
                "attempts": int(diag.get("attempts", 0)),
                "spawned": int(diag.get("spawned", births)),
                "fail_spawn": int(diag.get("fail_spawn", 0)),
                "p_mean": float(diag.get("p_mean", float("nan"))),
                "p_sum": float(diag.get("p_sum", float("nan"))),
            }
    
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

    def _can_fund_child_mass(self, a: Agent, child_M: float) -> bool:
        eps = 1e-9
        return (float(a.body.M) - float(child_M)) >= (float(a.AP.M_min) - eps)
    
    def _child_M_target(self, parent: Agent) -> float:
        target = float(parent.pheno.child_M)
    
        M = float(parent.body.M)
        Mmin = float(parent.AP.M_min)
        free = M - Mmin
    
        # välj en hård min-gräns (helst från pheno/ranges, annars konstant)
        child_min = 1e-3  # börja här; byt till pheno.child_M_min om du har
        if free <= child_min:
            return 0.0
    
        return min(target, free)
    
    def _try_spawn_child(self, parent: Agent) -> Optional[Agent]:
        child_M = float(self._child_M_target(parent))
    
        # STOPPA 0/negativ massa direkt
        if not math.isfinite(child_M) or child_M <= 0.0:
            return None
    
        if not self._can_fund_child_mass(parent, child_M):
            return None
    
        # --- Reproduction cost model (mass-only) ---
        # The child is funded by transferring structural mass from the parent.
        # Do NOT also charge a generic energy cost here, otherwise reproduction
        # is typically double-penalized (energy + mass).
        parent.body.M = float(parent.body.M) - child_M
        if parent.body.M < 0.0:
            # numerical safety; should not happen if _can_fund_child_mass is correct
            parent.body.M = 0.0
    
        child = self._spawn_child(parent)
        child.init_newborn_state(parent.pheno, child_M_from_parent=child_M)
        return child
        
    def _try_spawn_child_diag(self, parent: Agent) -> tuple[Agent | None, str]:
        # 0) compute target
        try:
            child_M = float(self._child_M_target(parent))
        except Exception as e:
            return None, f"child_M_target_exc:{type(e).__name__}"
    
        # sanity
        if not math.isfinite(child_M) or child_M <= 0.0:
            return None, "child_M_target_invalid"
    
        # 1) mass funding gate (primary None path)
        try:
            ok_mass = bool(self._can_fund_child_mass(parent, child_M))
        except Exception as e:
            return None, f"can_fund_child_mass_exc:{type(e).__name__}"
    
        if not ok_mass:
            # Grov men informativ klassning (du kan förbättra om du visar _can_fund_child_mass)
            M = float(getattr(parent.body, "M", float("nan")))
            Mmin = float(getattr(parent.AP, "M_min", float("nan")))
            # Ex: om du har en reserv / max-fraktion, lägg in här senare
            if math.isfinite(M) and math.isfinite(Mmin) and M - child_M < Mmin:
                return None, "fund_mass_hits_M_min"
            return None, "fund_mass_denied"
    
        # 2) pay repro cost (may set cooldown; may fail)
        try:
            _ = parent.pay_repro_cost(float(parent.pheno.repro_cost))
        except Exception as e:
            return None, f"pay_repro_cost_exc:{type(e).__name__}"
    
        # 3) transfer mass
        try:
            parent.body.M = float(parent.body.M) - child_M
        except Exception as e:
            return None, f"parent_mass_transfer_exc:{type(e).__name__}"
    
        # 4) spawn child
        try:
            child = self._spawn_child(parent)
        except Exception as e:
            return None, f"spawn_child_exc:{type(e).__name__}"
    
        if child is None:
            return None, "spawn_child_returned_none"
    
        # 5) init newborn state
        try:
            child.init_newborn_state(parent.pheno, child_M_from_parent=child_M)
        except Exception as e:
            return None, f"init_newborn_exc:{type(e).__name__}"
    
        return child, "ok"
            
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
            X = np.empty((n, 23), dtype=np.float32)  # 23 = in_dim
            BFC_list: List[Tuple[float, float, float]] = [None] * n  # type: ignore

            for i, a in enumerate(alive):
                x_in, B0, F0, C0 = a.build_inputs(self.world, rng=self.rng)
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

        # (2) deaths -> carcass in kg
        deaths = 0
        survivors: List[Agent] = []
        for a in self.agents:
            if not a.body.alive:
                # release policy slot immediately
                self._banks[a._policy_key].release(a._policy_slot)

                # carcass mass = remaining body mass (kg)
                carcass_kg = max(0.0, float(getattr(a.body, "M", 0.0)))

                # deposit carcass into world (kg, not normalized units)
                if carcass_kg > 0.0:
                    self.world.add_carcass(a.x, a.y, amount_kg=carcass_kg, rad=self.PP.carcass_rad)

                deaths += 1

                # log carcass in kg only
                self._emit_death(
                    self.t,
                    a,
                    carcass_amount=carcass_kg,
                    carcass_rad=self.PP.carcass_rad,
                )
            else:
                survivors.append(a)

        self.agents = survivors
    
        # 3) births (agent-decided; population executes + enforces max_pop)
        births = 0
        self._last_birth_diag = {}  # reset varje tick
    
        # --- DIAG knobs ---
        DIAG_EVERY_TICK = True      # skriv en rad per tick
        DIAG_FIRST_FAILS = True     # skriv första exempel per gate per tick
        DIAG_PRINT_ATTEMPTS = True  # skriv vid rng-hit (attempt)
        DIAG_PRINT_SPAWN = True     # skriv spawn-resultat (reason)
        DIAG_MAX_FAIL_EXAMPLES = 1  # antal exempel per gate per tick
    
        # För diagnos: initiera counters även om max_pop är fullt,
        # så vi kan lämna konsekvent diag (du kan ta bort om du vill).
        n_alive = 0
        n_eligible = 0
        n_block_cd = 0
        n_block_age = 0
        n_block_mass = 0
        n_block_energy = 0
        n_attempts = 0
        n_spawned = 0
        n_fail_spawn = 0
        p_sum = 0.0
        p_n = 0
        fail_reasons: dict[str, int] = {}
    
        if len(self.agents) < self.PP.max_pop:
            children: List[Agent] = []
    
            # dt: ta från world.P om den finns, annars fallback
            dt = float(getattr(getattr(self.world, "P", None), "dt", self.WP.dt))
    
            # First-example buckets per gate
            ex = {
                "cd": [],
                "age": [],
                "mass": [],
                "energy": [],
                "eligible": [],
                "attempt": [],
                "spawn_fail": [],
                "spawn_ok": [],
            }
    
            # Optional: snapshot pop-size at loop start (for stable debug prints)
            pop0 = len(self.agents)
            cap = int(self.PP.max_pop)
    
            # ---- births scan ----
            for a in self.agents:
                # hard cap check (children not yet merged)
                if pop0 + len(children) >= cap:
                    break
    
                if not a.body.alive:
                    continue
                n_alive += 1
    
                # ---------------------------
                # Gate 0: cooldown
                # ---------------------------
                cd = float(getattr(a, "repro_cd_s", 0.0))
                if cd > 0.0:
                    n_block_cd += 1
                    if DIAG_FIRST_FAILS and len(ex["cd"]) < DIAG_MAX_FAIL_EXAMPLES:
                        ex["cd"].append(
                            f"id={a.id} cd={cd:.3f} age={a.age_s:.2f} A_mature={float(a.pheno.A_mature):.2f}"
                        )
                    continue
    
                # ---------------------------
                # Gate 1: maturity
                # ---------------------------
                age = float(getattr(a, "age_s", 0.0))
                A_mature = float(getattr(a.pheno, "A_mature", 0.0))
                if age < A_mature:
                    n_block_age += 1
                    if DIAG_FIRST_FAILS and len(ex["age"]) < DIAG_MAX_FAIL_EXAMPLES:
                        ex["age"].append(
                            f"id={a.id} age={age:.2f} < A_mature={A_mature:.2f} (cd={cd:.3f})"
                        )
                    continue
    
                # ---------------------------
                # Gate 2: mass
                # ---------------------------
                M = float(getattr(a.body, "M", 0.0))
                M_min = float(getattr(a.AP, "M_min", 0.0))
                M_repro_min = float(getattr(a.pheno, "M_repro_min", 0.0))
                M_req = max(M_min, M_repro_min)
                if M < M_req:
                    n_block_mass += 1
                    if DIAG_FIRST_FAILS and len(ex["mass"]) < DIAG_MAX_FAIL_EXAMPLES:
                        ex["mass"].append(
                            f"id={a.id} M={M:.4f} < M_req={M_req:.4f} (M_min={M_min:.4f}, M_repro_min={M_repro_min:.4f})"
                        )
                    continue
    
                # ---------------------------
                # Gate 3: energy
                # ---------------------------
                Et = float(a.body.E_total())
                Ecap = float(a.body.E_cap())
                E_repro_min = float(getattr(a.pheno, "E_repro_min", 0.0))
                thr = E_repro_min * Ecap  # IMPORTANT: assumes E_repro_min is fraction
    
                if Et < thr:
                    n_block_energy += 1
                    if DIAG_FIRST_FAILS and len(ex["energy"]) < DIAG_MAX_FAIL_EXAMPLES:
                        ex["energy"].append(
                            f"id={a.id} Et={Et:.2f} < thr={thr:.2f} (E_repro_min={E_repro_min:.3f}, Ecap={Ecap:.2f})"
                        )
                    continue
    
                # ---- eligible ----
                n_eligible += 1
                lam = float(getattr(a.pheno, "repro_rate", 0.0))
                p = 1.0 - math.exp(-lam * dt)
                p_sum += p
                p_n += 1
    
                # ---------------------------
                # RNG decision
                # ---------------------------
                r = float(self.rng.random())
                if r >= p:
                    continue
    
                n_attempts += 1
    
                # ---------------------------
                # Spawn child
                # ---------------------------
                child, reason = self._try_spawn_child_diag(a)
    
                if child is None:
                    n_fail_spawn += 1
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                    if DIAG_FIRST_FAILS and len(ex["spawn_fail"]) < DIAG_MAX_FAIL_EXAMPLES:
                        ex["spawn_fail"].append(f"id={a.id} reason={reason}")
                    continue
    
                children.append(child)
                n_spawned += 1
                if DIAG_FIRST_FAILS and len(ex["spawn_ok"]) < DIAG_MAX_FAIL_EXAMPLES:
                    ex["spawn_ok"].append(f"id={a.id} ok")
    
            # === FIX #1: merge children AFTER loop (prevents duplication) ===
            if children:
                self.agents.extend(children)
            births = len(children)
    
        # === FIX #2: always set diag AFTER loop (also when births==0) ===
        self._last_birth_diag = {
            "alive": int(n_alive),
            "eligible": int(n_eligible),
            "block_cd": int(n_block_cd),
            "block_age": int(n_block_age),
            "block_mass": int(n_block_mass),
            "block_energy": int(n_block_energy),
            "attempts": int(n_attempts),
            "wants": int(n_attempts),
            "spawned": int(n_spawned),
            "fail_spawn": int(n_fail_spawn),
            "p_mean": (p_sum / p_n) if p_n else float("nan"),
            "p_sum": float(p_sum),
            "fail_reasons": dict(fail_reasons),
        }
    
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

    def mean_stats(self) -> tuple[float, float, float, float, float]:
        if not self.agents:
            return 0.0, 0.0, 0.0, 0.0, 0.0
    
        Es = []
        Ds = []
        Ms = []
        Ecs = []
        Rs = []
    
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

