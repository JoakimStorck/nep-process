from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import math

from agent import Agent
from world import World
from mlp import MLPGenome


# -----------------------
# helpers
# -----------------------
def _policy_key(g: MLPGenome) -> str:
    return f"{'-'.join(map(str, g.layer_sizes))}:{g.act}"


def _weight_norms(g: MLPGenome) -> Dict[str, Any]:
    assert g.weights is not None and g.biases is not None
    w_norms = [float(np.linalg.norm(W)) for W in g.weights]
    b_norms = [float(np.linalg.norm(b)) for b in g.biases]
    return {"w_norms": w_norms, "b_norms": b_norms}


def _traits_list(g: MLPGenome):
    tr = getattr(g, "traits", None)
    if tr is None:
        return None
    # stöd både np.ndarray och list
    try:
        return [float(x) for x in tr.tolist()]
    except Exception:
        return [float(x) for x in tr]


def _body_state(a: Agent) -> Dict[str, float]:
    """Flat-ish body state used across events."""
    b = a.body
    return {
        "E": float(b.E_total()),
        "E_fast": float(b.E_fast),
        "E_slow": float(b.E_slow),
        "M": float(getattr(b, "M", float("nan"))),
        "Fg": float(b.Fg),
        "D": float(b.D),
        "hunger": float(b.hunger()),
    }


# -----------------------
# records
# -----------------------
def step_record(t: float, agent: Agent, B0: float, C0: float) -> Dict[str, Any]:
    s = {
        "t": float(t),
        "agent_id": int(getattr(agent, "id", -1)),
        "x": float(agent.x),
        "y": float(agent.y),

        # body
        "E": float(agent.body.E_total()),
        "E_fast": float(agent.body.E_fast),
        "E_slow": float(agent.body.E_slow),
        "M": float(getattr(agent.body, "M", float("nan"))),
        "D": float(agent.body.D),
        "Fg": float(agent.body.Fg),
        "hunger": float(agent.body.hunger()),

        # action-ish
        "speed": float(getattr(agent, "last_speed", 0.0)),

        # local fields (hazard removed)
        "B0": float(B0),
        "C0": float(C0),

        "alive": bool(agent.body.alive),
    }
    return {"event": "step", "summary": s}


def population_record(
    t: float,
    pop_n: int,
    births: int,
    deaths: int,
    mean_E: float,
    mean_D: float,
    mean_M: float,
    repro: dict | None = None,
    **extra_stats: float,
) -> Dict[str, Any]:
    s = {
        "t": float(t),
        "pop": int(pop_n),
        "births": int(births),
        "deaths": int(deaths),
        "mean_E": float(mean_E),
        "mean_D": float(mean_D),
        "mean_M": float(mean_M),
    }

    # Lägg till alla extra stats som inte är None
    for k, v in extra_stats.items():
        if v is not None:
            s[k] = float(v)

    if repro is not None:
        s["repro"] = repro

    return {"event": "population", "summary": s}


def birth_record(t: float, child: Agent, parent: Optional[Agent]) -> Dict[str, Any]:
    g = child.genome
    d = {
        "event": "birth",
        "t": float(t),
        "agent_id": int(child.id),
        "parent_id": None if parent is None else int(parent.id),

        "pos": {"x": float(child.x), "y": float(child.y), "heading": float(child.heading)},

        "policy": {"key": _policy_key(g), "layer_sizes": list(g.layer_sizes), "act": str(g.act)},
        "traits": _traits_list(g),

        # include M at birth
        "state": {
            "E_fast": float(child.body.E_fast),
            "E_slow": float(child.body.E_slow),
            "M": float(getattr(child.body, "M", float("nan"))),
            "Fg": float(child.body.Fg),
            "D": float(child.body.D),
        },

        "phenotype": child.phenotype_summary() if hasattr(child, "phenotype_summary") else {},
        "weights": _weight_norms(g) if (g.weights is not None and g.biases is not None) else {},
    }
    return d


def death_record(
    t: float,
    agent,
    carcass_amount: float,   # kg
    carcass_rad: int,
) -> dict:
    body = getattr(agent, "body", None)

    M  = float(getattr(body, "M", float("nan"))) if body is not None else float("nan")
    Et = float(body.E_total()) if (body is not None and hasattr(body, "E_total")) else float("nan")
    D  = float(getattr(body, "D", float("nan"))) if body is not None else float("nan")

    birth_t = float(getattr(agent, "birth_t", float("nan")))
    age = float(t - birth_t) if math.isfinite(birth_t) else float("nan")

    return {
        "event": "death",
        "t": float(t),

        "agent_id": int(getattr(agent, "id", -1)),

        "birth_t": birth_t,
        "age": age,

        "x": float(getattr(agent, "x", float("nan"))),
        "y": float(getattr(agent, "y", float("nan"))),

        "M": M,              # kg (body mass at death)
        "E": Et,
        "D": D,

        # Carcass deposited (kg, same unit system as M)
        "carcass_amount": float(carcass_amount),
        "carcass_rad": int(carcass_rad),
    }


def sample_record(t: float, a: Agent, pop_n: int) -> Dict[str, Any]:
    b = a.body
    return {
        "event": "sample",
        "t": float(t),
        "agent_id": int(a.id),
        "age": float(getattr(a, "age_s", 0.0)),
        "birth_t": float(getattr(a, "birth_t", float("nan"))),
        "alive": bool(a.body.alive),

        "pos": {"x": float(a.x), "y": float(a.y), "heading": float(a.heading)},
        "state": {
            **_body_state(a),
            "speed": float(getattr(a, "last_speed", 0.0)),
            "repro_cd_s": float(getattr(a, "repro_cd_s", 0.0)),
            "Ecap": float(b.E_cap()),
        },
        "local": {
            "B0": float(getattr(a, "last_B0", 0.0)),
            "C0": float(getattr(a, "last_C0", 0.0)),
        },
        "phenotype": a.phenotype_summary() if hasattr(a, "phenotype_summary") else {},
        "traits": _traits_list(a.genome),
        "pop_n": int(pop_n),
    }


def world_record(t: float, world: World, with_percentiles: bool = True) -> Dict[str, Any]:
    # hazard removed: world expected to have B and C (and possibly A)
    B = world.B
    C = world.C

    def stats(A: np.ndarray) -> Dict[str, float]:
        flat = A.ravel()
        out = {"mean": float(flat.mean()), "sum": float(flat.sum())}
        if with_percentiles:
            p10, p50, p90 = np.percentile(flat, [10, 50, 90])
            out.update({"p10": float(p10), "p50": float(p50), "p90": float(p90)})
        return out

    s = {
        "t": float(t),
        "B": stats(B),
        "C": stats(C) if with_percentiles else {"mean": float(C.mean()), "sum": float(C.sum())},
    }
    return {"event": "world", "summary": s}