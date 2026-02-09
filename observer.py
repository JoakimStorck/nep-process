# observer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from agent import Agent
from mlp import MLPGenome
from world import World
from jsonl import JsonlWriter


@dataclass
class Observer:
    w: JsonlWriter
    every_s: float = 0.5
    _next_t: float = 0.0

    def maybe_log(self, t: float, agent: Agent, B0: float, F0: float, C0: float) -> bool:
        if t < self._next_t:
            return False
        self._next_t = t + self.every_s
        s = {
            "t": float(t),
            "agent_id": int(getattr(agent, "id", -1)),
            "E": float(agent.body.E_total()),
            "D": float(agent.body.D),
            "Fg": float(agent.body.Fg),
            "hunger": float(agent.body.hunger()),
            "speed": float(agent.last_speed),
            "x": float(agent.x),
            "y": float(agent.y),
            "B0": float(B0),
            "F0": float(F0),
            "C0": float(C0),
            "alive": bool(agent.body.alive),
        }
        self.w.write({"event": "step", "summary": s})
        return True


@dataclass
class PopObserver:
    w: JsonlWriter
    every_s: float = 1.0
    _next_t: float = 0.0

    def maybe_log(self, t: float, pop_n: int, births: int, deaths: int, mean_E: float, mean_D: float) -> bool:
        if t < self._next_t:
            return False
        self._next_t = t + self.every_s
        self.log_now(t, pop_n, births, deaths, mean_E, mean_D)
        return True

    def log_now(self, t: float, pop_n: int, births: int, deaths: int, mean_E: float, mean_D: float) -> None:
        s = {
            "t": float(t),
            "pop": int(pop_n),
            "births": int(births),
            "deaths": int(deaths),
            "mean_E": float(mean_E),
            "mean_D": float(mean_D),
        }
        self.w.write({"event": "population", "summary": s})


def _policy_key(g: MLPGenome) -> str:
    return f"{'-'.join(map(str, g.layer_sizes))}:{g.act}"


def _weight_norms(g: MLPGenome) -> Dict[str, Any]:
    # små, billiga fingerprints; räcker för att se drift/skillnad
    assert g.weights is not None and g.biases is not None
    w_norms = [float(np.linalg.norm(W)) for W in g.weights]
    b_norms = [float(np.linalg.norm(b)) for b in g.biases]
    return {"w_norms": w_norms, "b_norms": b_norms}


@dataclass
class LifeObserver:
    w: JsonlWriter

    def log_birth(self, t: float, child: Agent, parent: Optional[Agent]) -> None:
        g = child.genome
        d = {
            "event": "birth",
            "t": float(t),
            "agent_id": int(child.id),
            "parent_id": None if parent is None else int(parent.id),
            "pos": {"x": float(child.x), "y": float(child.y), "heading": float(child.heading)},
            "policy": {"key": _policy_key(g), "layer_sizes": list(g.layer_sizes), "act": str(g.act)},
            "traits": None if g.traits is None else [float(x) for x in g.traits.tolist()],
            "state": {
                "E_fast": float(child.body.E_fast),
                "E_slow": float(child.body.E_slow),
                "Fg": float(child.body.Fg),
                "D": float(child.body.D),
            },
            "phenotype": child.phenotype_summary() if hasattr(child, "phenotype_summary") else {},
            "weights": _weight_norms(g) if (g.weights is not None and g.biases is not None) else {},
        }
        self.w.write(d)

    def log_death(self, t: float, agent: Agent, carcass_amount: float, carcass_rad: int) -> None:
        d = {
            "event": "death",
            "t": float(t),
            "agent_id": int(agent.id),
            "age": float(t - getattr(agent, "birth_t", t)),
            "pos": {"x": float(agent.x), "y": float(agent.y)},
            "state": {
                "E": float(agent.body.E_total()),
                "E_fast": float(agent.body.E_fast),
                "E_slow": float(agent.body.E_slow),
                "Fg": float(agent.body.Fg),
                "D": float(agent.body.D),
                "hunger": float(agent.body.hunger()),
            },
            "carcass": {"amount": float(carcass_amount), "rad": int(carcass_rad)},
        }
        self.w.write(d)


@dataclass
class WorldObserver:
    w: JsonlWriter
    every_s: float = 2.0
    _next_t: float = 0.0

    def maybe_log(self, t: float, world: World) -> bool:
        if t < self._next_t:
            return False
        self._next_t = t + self.every_s
        self.log_now(t, world)
        return True

    def log_now(self, t: float, world: World) -> None:
        B = world.B
        F = world.F
        C = world.C

        def stats(A: np.ndarray) -> Dict[str, float]:
            flat = A.ravel()
            # kvantiler är lite dyrare; om det stör: logga bara mean/sum.
            p10, p50, p90 = np.percentile(flat, [10, 50, 90])
            return {
                "mean": float(flat.mean()),
                "sum": float(flat.sum()),
                "p10": float(p10),
                "p50": float(p50),
                "p90": float(p90),
            }

        s = {
            "t": float(t),
            "B": stats(B),
            "F": {**stats(F), "hazard_frac_0p35": float((F >= 0.35).mean())},
            "C": {"mean": float(C.mean()), "sum": float(C.sum())},
        }
        self.w.write({"event": "world", "summary": s})