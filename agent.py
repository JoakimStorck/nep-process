# agent.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import List, Tuple

import numpy as np

from world import World, clamp
from mlp import MLPGenome
from phenotype import Phenotype, derive_pheno, phenotype_summary

def torus_wrap(x: float, size: int) -> float:
    return x % size


def _sig01(z: float) -> float:
    # stable logistic -> [0,1]
    z = float(np.clip(z, -8.0, 8.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def _lerp(a: float, b: float, u01: float) -> float:
    u = float(np.clip(u01, 0.0, 1.0))
    return a + (b - a) * u

# agent.py (top-level, nära övriga helpers)
_NEXT_AGENT_ID = 0

def _new_agent_id() -> int:
    global _NEXT_AGENT_ID
    _NEXT_AGENT_ID += 1
    return _NEXT_AGENT_ID
    
@dataclass
class AgentParams:
    dt: float = 0.02

    # sensors
    n_rays: int = 12
    ray_len: float = 7.0
    ray_step: float = 1.0
    noise_sigma: float = 0.06

    # body
    basal: float = 0.010
    move_cost: float = 0.045
    compute_cost: float = 0.006
    slow_to_fast: float = 0.030
    fast_to_slow: float = 0.006
    hazard_to_damage: float = 0.03

    hazard_to_fatigue: float = 0.08
    fatigue_recover: float = 0.020
    fatigue_effort: float = 0.050
    D_max: float = 1.0

    # kinematics
    v_max: float = 2.2
    turn_rate: float = 2.2

    # eating
    eat_rate: float = 0.10  # biomass pool/sec scaling (times dt)
    eat_gain: float = 0.6

    # reproduction (used by Population; Agent maintains cooldown locally)
    repro_cooldown_s: float = 8.0


@dataclass
class Body:
    P: AgentParams
    E_fast: float = 0.55
    E_slow: float = 0.55
    D: float = 0.0
    Fg: float = 0.15
    alive: bool = True

    def E_total(self) -> float:
        return 0.6 * self.E_fast + 0.4 * self.E_slow

    def hunger(self) -> float:
        return clamp(1.0 - self.E_total(), 0.0, 1.0)

    def take_energy(self, amount: float) -> float:
        """
        Remove 'amount' from the body energy stores, split proportionally.
        Returns actual removed.
        """
        amt = float(max(0.0, amount))
        if amt <= 0.0:
            return 0.0
        Et = self.E_total()
        if Et <= 1e-9:
            return 0.0

        share_fast = (0.6 * self.E_fast) / max(Et, 1e-9)
        share_slow = (0.4 * self.E_slow) / max(Et, 1e-9)

        d_fast = (amt * share_fast) / 0.6
        d_slow = (amt * share_slow) / 0.4

        d_fast = min(d_fast, self.E_fast)
        d_slow = min(d_slow, self.E_slow)

        self.E_fast = clamp(self.E_fast - d_fast, 0.0, 1.0)
        self.E_slow = clamp(self.E_slow - d_slow, 0.0, 1.0)

        return float(0.6 * d_fast + 0.4 * d_slow)

    def step(self, speed, activity, hazard, intake, pheno: Phenotype):
        """
        Damage dynamics (no internal clock):
          D += inflow(hazard + effort + metabolic stress per drain) - repair(rest + E + intake),
        with frailty coupling via D itself.
        """
        if not self.alive:
            return
        dt = self.P.dt

        # --- energy dynamics (metabolism scales drain_rate) ---
        drain_rate = float(pheno.metabolism_scale) * (
            self.P.basal + self.P.move_cost * float(speed) + self.P.compute_cost * float(activity)
        )
        drain = dt * drain_rate

        self.E_fast = clamp(self.E_fast + 0.95 * float(intake) - 0.7 * drain, 0.0, 1.0)
        self.E_slow = clamp(self.E_slow + 0.30 * float(intake) - 0.3 * drain, 0.0, 1.0)

        pull = self.P.slow_to_fast * dt * max(0.0, 0.6 - self.E_fast)
        push = self.P.fast_to_slow * dt * max(0.0, self.E_fast - 0.75)
        pull = min(pull, self.E_slow)
        push = min(push, self.E_fast)
        self.E_slow -= pull
        self.E_fast += pull
        self.E_fast -= push
        self.E_slow += push

        # --- normalized state ---
        Et = self.E_total()
        e_lack = clamp(1.0 - Et, 0.0, 1.0)
        d_norm = clamp(self.D / max(self.P.D_max, 1e-9), 0.0, 1.0)

        # motion/activity decomposition
        speed_n = clamp(float(speed) / max(self.P.v_max, 1e-9), 0.0, 1.0)
        effort = speed_n + 0.6 * float(activity)
        rest = (
            max(0.0, 1.0 - speed_n)
            * max(0.0, 1.0 - float(activity))
            * (0.25 + 0.75 * max(0.0, 1.0 - float(hazard)))
        )

        # intake normalization (as before)
        max_intake = max(1e-9, self.P.eat_gain * self.P.eat_rate * dt)
        intake_n = clamp(float(intake) / max_intake, 0.0, 1.0)

        # --- damage inflow ---
        susc = float(pheno.susceptibility)
        frail = 1.0 + float(pheno.frailty_gain) * d_norm
        k_E = 1.2  # energy lack amplifies damage

        dD_haz = dt * (self.P.hazard_to_damage * susc * (1.0 + k_E * e_lack) * frail * float(hazard))

        k_eff = 0.02  # MV0: small activity damage channel
        dD_eff = dt * (k_eff * susc * (1.0 + k_E * e_lack) * frail * effort)

        dD_met = dt * (float(pheno.stress_per_drain) * drain_rate)

        dD_in = dD_haz + dD_eff + dD_met

        # --- repair outflow ---
        E_rep_min = 0.20
        G_E = clamp((Et - E_rep_min) / (1.0 - E_rep_min), 0.0, 1.0)
        G_rest = 0.2 + 0.8 * rest
        G_int = 0.3 + 0.7 * intake_n
        fresh = 1.0 - self.Fg
        G_fresh = 0.75 + 0.25 * fresh

        frailty_damp = 1.0 / (1.0 + float(pheno.frailty_gain) * d_norm)

        dD_rep = dt * float(pheno.repair_capacity) * G_rest * G_E * G_int * G_fresh * frailty_damp

        self.D = clamp(self.D + dD_in - dD_rep, 0.0, self.P.D_max)

        # --- fatigue dynamics (kept; optionally make it depend on frailty too) ---
        fatigue_effort_eff = self.P.fatigue_effort * (1.0 + 0.4 * d_norm)
        fatigue_recover_eff = self.P.fatigue_recover * max(0.0, (1.0 - 0.05 * d_norm))

        self.Fg = clamp(
            self.Fg + dt * (
                fatigue_effort_eff * effort
                + self.P.hazard_to_fatigue * float(hazard)
                - fatigue_recover_eff * rest
            ),
            0.0,
            1.0,
        )

        if self.E_total() <= 0.0 or self.D >= self.P.D_max:
            self.alive = False


@dataclass
class RaySensors:
    P: "AgentParams"
    world_size: int

    # cached geometry/buffers
    _n: int = field(init=False, default=0)
    _m: int = field(init=False, default=0)
    _ang_base: np.ndarray = field(init=False)
    _d: np.ndarray = field(init=False)
    _w: np.ndarray = field(init=False)
    _wsum: np.float32 = field(init=False)
    _dx: np.ndarray = field(init=False)
    _dy: np.ndarray = field(init=False)
    _xs: np.ndarray = field(init=False)
    _ys: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        n = int(self.P.n_rays)
        step = float(self.P.ray_step)
        ray_len = float(self.P.ray_len)

        self._n = max(0, n)

        if self._n <= 0 or step <= 0.0 or ray_len <= 0.0:
            self._m = 0
            self._ang_base = np.zeros((0,), dtype=np.float32)
            self._d = np.zeros((0,), dtype=np.float32)
            self._w = np.zeros((0,), dtype=np.float32)
            self._wsum = np.float32(1.0)
            self._dx = np.zeros((0,), dtype=np.float32)
            self._dy = np.zeros((0,), dtype=np.float32)
            self._xs = np.zeros((0, 0), dtype=np.float32)
            self._ys = np.zeros((0, 0), dtype=np.float32)
            return

        self._ang_base = (np.float32(2.0 * np.pi) *
                          (np.arange(self._n, dtype=np.float32) / np.float32(self._n)))

        self._d = np.arange(step, ray_len + 1e-6, step, dtype=np.float32)
        self._m = int(self._d.size)

        self._w = (np.float32(1.0) / (np.float32(1.0) + np.float32(0.25) * self._d)).astype(np.float32, copy=False)
        self._wsum = np.sum(self._w, dtype=np.float32) + np.float32(1e-9)

        self._dx = np.empty((self._n,), dtype=np.float32)
        self._dy = np.empty((self._n,), dtype=np.float32)
        self._xs = np.empty((self._n, self._m), dtype=np.float32)
        self._ys = np.empty((self._n, self._m), dtype=np.float32)

    def sense(self, world: "World", x: float, y: float, heading: float,
              rng: np.random.Generator | None = None) -> Tuple[Tuple[float, float, float], List[float], List[float]]:

        # base sample: fast scalar
        B0, F0, C0 = world.sample_bilinear(x, y)

        n = self._n
        m = self._m
        if n <= 0 or m <= 0:
            return (B0, F0, C0), [0.0] * max(0, n), [0.0] * max(0, n)

        # angles -> dx/dy (no new allocations)
        ang = self._ang_base + np.float32(heading)
        np.cos(ang, out=self._dx)
        np.sin(ang, out=self._dy)

        ws = np.float32(self.world_size)
        xx = np.float32(x)
        yy = np.float32(y)

        # fill sample grids (reuse buffers)
        # xs = (x + dx[:,None]*d[None,:]) % ws
        self._xs[:] = xx + self._dx[:, None] * self._d[None, :]
        self._ys[:] = yy + self._dy[:, None] * self._d[None, :]
        # wrap
        self._xs[:] = np.mod(self._xs, ws)
        self._ys[:] = np.mod(self._ys, ws)

        # sample only B and F for rays
        Bp, Fp = world.sample_bilinear_many_BF(self._xs, self._ys)  # (n,m)

        # weighted averages (no new w)
        accB = (Bp * self._w[None, :]).sum(axis=1, dtype=np.float32) / self._wsum
        accF = (Fp * self._w[None, :]).sum(axis=1, dtype=np.float32) / self._wsum
        
        sig = float(self.P.noise_sigma)
        if sig > 0.0:
            noiseB = rng.normal(0.0, sig, size=n).astype(np.float32, copy=False)
            noiseF = rng.normal(0.0, sig, size=n).astype(np.float32, copy=False)
            accB = accB + noiseB
            accF = accF + noiseF
        
            B0 = float(np.clip(B0 + rng.normal(0.0, sig * 0.5), 0.0, 1.0))
            F0 = float(np.clip(F0 + rng.normal(0.0, sig * 0.5), 0.0, 1.0))
            C0 = float(np.clip(C0 + rng.normal(0.0, sig * 0.5), 0.0, 1.0))

        accB = np.clip(accB, 0.0, 1.0).astype(np.float32, copy=False)
        accF = np.clip(accF, 0.0, 1.0).astype(np.float32, copy=False)

        return (B0, F0, C0), accB.tolist(), accF.tolist()


@dataclass
class Agent:
    """
    NEP-agent:
      - Ingen PatchManager, ingen eventkö, ingen undo.
      - NN styr negativ kontroll (inhibition) + motorik + “explore drive”.
      - Phenotype (tolkbara parametrar) härleds från traits och används i reproduction/strategi.
      - Anpassning sker via reproduktion med mutation (Population).
    """

    AP: AgentParams
    genome: MLPGenome

    x: float
    y: float
    heading: float

    id: int = field(default_factory=_new_agent_id)
    
    body: Body = field(init=False)
    sensors: RaySensors = field(init=False)

    # lightweight local memory
    obs_trace: np.ndarray = field(init=False)

    # life origin (absolute sim time at birth; set by Population when spawning)
    birth_t: float = 0.0

    # derived phenotype (from genome.traits; fixed for lifetime)
    pheno: Phenotype = field(init=False)
    
    # local life history
    last_speed: float = 0.0
    age_s: float = 0.0
    repro_cd_s: float = 0.0

    def __post_init__(self) -> None:
        # Ensure per-agent AP instance (NOT trait-modified anymore; keep stable baseline).
        self.AP = replace(self.AP)
    
        self.body = Body(self.AP)
        self.sensors = RaySensors(self.AP, world_size=64)  # overwritten in bind_world
        self.obs_trace = np.zeros((9,), dtype=np.float32)  # obs has 9 dims (includes C0)
    
        # Local life history clocks
        self.age_s = 0.0
        self.repro_cd_s = 0.0
    
        # Birth time (Population should overwrite for newborns)
        self.birth_t = float(getattr(self, "birth_t", 0.0))
    
        # Derive phenotype once (fixed for lifetime unless you later add plasticity)
        self.apply_traits()

    def bind_world(self, world: World) -> None:
        self.sensors = RaySensors(self.AP, world_size=world.P.size)

    @staticmethod
    def _signed_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def apply_traits(self) -> None:
        """
        New meaning (vNext):
        Map genome.traits -> interpretable Phenotype (NOT AgentParams).
        Phenotype is fixed over lifetime and used by Population/Agent reproduction logic
        and later (C) for behavior modulation.
        """
        self.pheno = derive_pheno(self.genome.traits)

    def phenotype_summary(self) -> dict:
        return phenotype_summary(self.pheno)
        
    def _build_obs(self, B0: float, F0: float, C0: float, rays_B: List[float], rays_F: List[float]) -> np.ndarray:
        n = len(rays_B)
        iB = max(range(n), key=lambda i: rays_B[i])
        iF = max(range(n), key=lambda i: rays_F[i])

        aB = 2.0 * math.pi * (iB / n)
        aF = 2.0 * math.pi * (iF / n)

        hunger = self.body.hunger()
        fatigue = self.body.Fg
        D = self.body.D

        obs = np.array(
            [
                B0,
                F0,
                C0,
                float(sum(rays_B) / n),
                float(sum(rays_F) / n),
                float(max(rays_B)),
                float(max(rays_F)),
                hunger,
                fatigue,
            ],
            dtype=np.float32,
        )

        # update trace (local time)
        a = 0.06
        self.obs_trace = (1.0 - a) * self.obs_trace + a * obs

        # final input:
        # obs (9) + trace (9) + dirs(4) + D(1) = 23
        x = np.concatenate(
            [
                obs,
                self.obs_trace,
                np.array([math.cos(aB), math.sin(aB), math.cos(aF), math.sin(aF), D], dtype=np.float32),
            ]
        )
        return x

    def build_inputs(self, world: World, rng: np.random.Generator) -> Tuple[np.ndarray, float, float, float]:
        """
        Sense + build NN input. Returns (x_in, B0, F0, C0).
        """
        if not self.body.alive:
            # caller should skip dead agents
            return np.zeros((23,), dtype=np.float32), 0.0, 0.0, 0.0
    
        (B0, F0, C0), rays_B, rays_F = self.sensors.sense(world, self.x, self.y, self.heading, rng=rng)
        x_in = self._build_obs(B0, F0, C0, rays_B, rays_F)
        return x_in.astype(np.float32, copy=False), float(B0), float(F0), float(C0)
    
    def apply_outputs(self, world: World, y: np.ndarray, B0: float, F0: float, C0: float,
                      rng: np.random.Generator) -> Tuple[float, float, float]:
        """
        Apply policy outputs y to kinematics + eating + body update.
        Returns (B0,F0,C0) for logging symmetry (matches old step()).
        """
        if not self.body.alive:
            return 0.0, 0.0, 0.0
    
        dt = self.AP.dt
        self.age_s += dt
        self.repro_cd_s = max(0.0, self.repro_cd_s - dt)
    
        # outputs (raw):
        turn = float(np.tanh(y[0]))
        thrust = float(1.0 / (1.0 + np.exp(-float(y[1]))))
        inh_move = float(1.0 / (1.0 + np.exp(-float(y[2]))))
        inh_eat = float(1.0 / (1.0 + np.exp(-float(y[3]))))
        explore_drive = float(1.0 / (1.0 + np.exp(-float(y[4]))))
    
        allow_move = 1.0 - inh_move
        allow_eat = 1.0 - inh_eat
    
        jitter = float(rng.normal(0.0, 0.65)) * explore_drive
    
        self.heading += dt * self.AP.turn_rate * (0.85 * allow_move * turn + 0.25 * jitter)
        self.heading = self._signed_angle(self.heading)
    
        fatigue = self.body.Fg
        fatigue_factor = clamp(1.0 - 0.9 * fatigue, 0.05, 1.0)
        speed = self.AP.v_max * allow_move * fatigue_factor * thrust
        self.last_speed = speed
    
        self.x = torus_wrap(self.x + dt * speed * math.cos(self.heading), world.P.size)
        self.y = torus_wrap(self.y + dt * speed * math.sin(self.heading), world.P.size)
    
        got = 0.0
        got_carcass = 0.0
        intake = 0.0
        if allow_eat > 0.20:
            want = self.AP.eat_rate * dt * (0.25 + 0.75 * self.body.hunger())
            got, got_carcass = world.consume_food(self.x, self.y, amount=want, prefer_carcass=True)
            intake = self.AP.eat_gain * got
    
        speed_n = clamp(speed / max(self.AP.v_max, 1e-9), 0.0, 1.0)
        eat_act = 1.0 if (allow_eat > 0.20 and got > 0.0) else 0.0
        activity = 0.03 + 0.45 * speed_n + 0.10 * eat_act
    
        self.body.step(
            speed=speed,
            activity=activity,
            hazard=F0,
            intake=intake,
            pheno=self.pheno,
        )

        return float(B0), float(F0), float(C0)
        
    def step(self, world: World) -> Tuple[float, float, float]:
        x_in, B0, F0, C0 = self.build_inputs(world)
        y = self.genome.forward(x_in)
        return self.apply_outputs(world, y, B0, F0, C0)

    # --- reproduction hooks (Population uses these) ---

    def ready_to_reproduce(self) -> bool:
        if not self.body.alive: return False
        if self.repro_cd_s > 0: return False
        if self.age_s < self.pheno.A_mature: return False
        if self.body.E_total() < self.pheno.E_repro_min: return False
        # ev. fertility shutdown:
        # if self.body.D > ... or self.body.Fg > ...: return False
        return True
    
    def wants_to_reproduce(self, rng: np.random.Generator) -> bool:
        if not self.ready_to_reproduce():
            return False
        # rate per second -> probability per dt
        # interpret p_repro_base as "attempt rate" (1/s), NOT per-attempt prob
        lam = float(self.pheno.repro_rate)
        p = 1.0 - math.exp(-lam * float(self.AP.dt))
        return bool(rng.random() < p)

    def pay_repro_cost(self, cost_E: float) -> float:
        # vNext: phenotype-driven cost (keep compat: allow caller to increase it)
        cost = max(float(cost_E), float(self.pheno.repro_cost))
    
        removed = self.body.take_energy(cost)
        self.repro_cd_s = float(self.AP.repro_cooldown_s)
        return removed

    def init_newborn_state(self, parent_pheno: Phenotype) -> None:
        # newborn body state from parent's phenotype
        self.body.E_fast = clamp(float(parent_pheno.child_E_fast), 0.0, 1.0)
        self.body.E_slow = clamp(float(parent_pheno.child_E_slow), 0.0, 1.0)
        self.body.Fg     = clamp(float(parent_pheno.child_Fg), 0.0, 1.0)
        self.body.D      = 0.0
        self.repro_cd_s  = float(self.AP.repro_cooldown_s)
        self.age_s       = 0.0

