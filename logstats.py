# logstats.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


# -----------------------
# jsonl utils
# -----------------------

def iter_jsonl(fp: Path) -> Iterator[Dict[str, Any]]:
    if not fp.exists():
        return
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def get_event(obj: Dict[str, Any]) -> str:
    ev = obj.get("event", "")
    return ev if isinstance(ev, str) else ""


def get_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    s = obj.get("summary")
    return s if isinstance(s, dict) else obj


def fget(d: Any, *path: str, default: float = float("nan")) -> float:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return float(default)
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return float(default)


def iget(d: Any, *path: str, default: int = -1) -> int:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return int(default)
        cur = cur[k]
    try:
        return int(cur)
    except Exception:
        return int(default)


def finite(x: float) -> bool:
    return (x == x) and (not math.isinf(x))


def pct(xs: List[float], p: float) -> float:
    """Nearest-rank percentile; xs must be finite-only."""
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    k = max(0, min(len(ys) - 1, k))
    return float(ys[k])


# -----------------------
# data containers
# -----------------------

@dataclass
class PopPoint:
    t: float
    pop: int
    mean_E: float
    mean_D: float
    births: int
    deaths: int


@dataclass
class WorldPoint:
    t: float
    B_mean: float
    F_mean: float
    C_mean: float
    hazard_frac: float
    B_sum: float
    F_sum: float
    C_sum: float


@dataclass
class AgentLife:
    agent_id: int
    parent_id: int
    birth_t: float
    death_t: float
    policy_key: str
    traits: List[float]
    n_children: int = 0

    def lifespan(self) -> float:
        if finite(self.birth_t) and finite(self.death_t):
            return float(self.death_t - self.birth_t)
        return float("nan")


@dataclass
class RunStats:
    pop: List[PopPoint]
    world: List[WorldPoint]
    agents: Dict[int, AgentLife]

    def __init__(self) -> None:
        self.pop = []
        self.world = []
        self.agents = {}


# -----------------------
# parsers
# -----------------------

def parse_pop_log(fp: Path) -> List[PopPoint]:
    out: List[PopPoint] = []
    for obj in iter_jsonl(fp):
        if get_event(obj) != "population":
            continue
        s = get_summary(obj)
        t = fget(s, "t")
        out.append(
            PopPoint(
                t=t,
                pop=iget(s, "pop", default=-1),
                mean_E=fget(s, "mean_E"),
                mean_D=fget(s, "mean_D"),
                births=iget(s, "births", default=0),
                deaths=iget(s, "deaths", default=0),
            )
        )
    out.sort(key=lambda z: z.t)
    return out


def parse_world_log(fp: Path) -> List[WorldPoint]:
    out: List[WorldPoint] = []
    for obj in iter_jsonl(fp):
        if get_event(obj) != "world":
            continue
        s = get_summary(obj)
        out.append(
            WorldPoint(
                t=fget(s, "t"),
                B_mean=fget(s, "B", "mean"),
                F_mean=fget(s, "F", "mean"),
                C_mean=fget(s, "C", "mean"),
                hazard_frac=fget(s, "F", "hazard_frac_0p35"),
                B_sum=fget(s, "B", "sum"),
                F_sum=fget(s, "F", "sum"),
                C_sum=fget(s, "C", "sum"),
            )
        )
    out.sort(key=lambda z: z.t)
    return out


def parse_life_log(fp: Path) -> Dict[int, AgentLife]:
    agents: Dict[int, AgentLife] = {}

    # temp storage for births before deaths
    birth_state: Dict[int, Tuple[float, int, str, List[float]]] = {}  # id -> (t, parent_id, policy_key, traits)

    for obj in iter_jsonl(fp):
        ev = get_event(obj)
        if ev == "birth":
            t = fget(obj, "t")
            aid = iget(obj, "agent_id", default=-1)
            pid = iget(obj, "parent_id", default=-1)
            pol = obj.get("policy", {})
            pkey = pol.get("key", "") if isinstance(pol, dict) else ""
            traits = obj.get("traits", [])
            traits = traits if isinstance(traits, list) else []
            if aid >= 0:
                birth_state[aid] = (t, pid, str(pkey), [float(x) for x in traits if isinstance(x, (int, float))])
                # increment children count on parent (if already known)
                if pid >= 0 and pid in agents:
                    agents[pid].n_children += 1

        elif ev == "death":
            t = fget(obj, "t")
            aid = iget(obj, "agent_id", default=-1)
            if aid < 0:
                continue

            # link back to birth if present
            if aid in birth_state:
                bt, pid, pkey, traits = birth_state[aid]
            else:
                bt, pid, pkey, traits = (float("nan"), -1, "", [])

            agents[aid] = AgentLife(
                agent_id=aid,
                parent_id=pid,
                birth_t=bt,
                death_t=t,
                policy_key=pkey,
                traits=traits,
                n_children=agents.get(aid, AgentLife(aid, -1, bt, t, pkey, traits)).n_children,
            )

            # increment children count on parent if parent already known in agents
            if pid >= 0 and pid in agents:
                agents[pid].n_children += 0  # already handled on birth when possible

    # also register agents that are born but not dead (alive at end)
    for aid, (bt, pid, pkey, traits) in birth_state.items():
        if aid not in agents:
            agents[aid] = AgentLife(
                agent_id=aid,
                parent_id=pid,
                birth_t=bt,
                death_t=float("nan"),
                policy_key=pkey,
                traits=traits,
            )

    return agents


# -----------------------
# joins / derived
# -----------------------

def join_pop_at_time(pop: List[PopPoint], t: float) -> int:
    """Nearest-left (last known pop at or before t). pop must be sorted."""
    if not pop:
        return -1
    lo, hi = 0, len(pop) - 1
    if t < pop[0].t:
        return -1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pop[mid].t <= t:
            lo = mid
        else:
            hi = mid - 1
    return pop[lo].pop


def derive_series(stats: RunStats, *, alpha: float = 1.0) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns dict of named (t, value) lists. These are convenient for plots/exports.
    """
    out: Dict[str, List[Tuple[float, float]]] = {
        "R_pc": [],
        "C_pc": [],
        "B_pc": [],
    }
    for w in stats.world:
        popn = join_pop_at_time(stats.pop, w.t)
        if popn is None or popn <= 0:
            out["R_pc"].append((w.t, float("nan")))
            out["C_pc"].append((w.t, float("nan")))
            out["B_pc"].append((w.t, float("nan")))
            continue
        R = (float(w.B_sum) + float(alpha) * float(w.C_sum)) / float(popn)
        out["R_pc"].append((w.t, R))
        out["C_pc"].append((w.t, float(w.C_sum) / float(popn)))
        out["B_pc"].append((w.t, float(w.B_sum) / float(popn)))
    return out


# -----------------------
# summary computation
# -----------------------

def summarize(stats: RunStats, *, alpha: float = 1.0) -> Dict[str, Any]:
    # life
    lifespans = [a.lifespan() for a in stats.agents.values() if finite(a.lifespan())]
    lifespans_f = [x for x in lifespans if finite(x) and x >= 0.0]

    # births/deaths from life
    n_birth = sum(1 for a in stats.agents.values() if finite(a.birth_t))
    n_death = sum(1 for a in stats.agents.values() if finite(a.death_t))

    # pop summary (from last point)
    pop_last = stats.pop[-1].pop if stats.pop else -1
    pop_max = max((p.pop for p in stats.pop), default=-1)
    pop_min = min((p.pop for p in stats.pop if p.pop >= 0), default=-1)

    # world summary
    def smean(xs: List[float]) -> float:
        ys = [x for x in xs if finite(x)]
        return float(sum(ys) / len(ys)) if ys else float("nan")

    Bm = [w.B_mean for w in stats.world]
    Fm = [w.F_mean for w in stats.world]
    Cm = [w.C_mean for w in stats.world]
    Hf = [w.hazard_frac for w in stats.world]

    # derived
    deriv = derive_series(stats, alpha=alpha)
    Rpc_vals = [v for _, v in deriv["R_pc"] if finite(v)]

    out: Dict[str, Any] = {
        "files": {
            "life": str(stats._life_fp) if hasattr(stats, "_life_fp") else "",
            "pop": str(stats._pop_fp) if hasattr(stats, "_pop_fp") else "",
            "world": str(stats._world_fp) if hasattr(stats, "_world_fp") else "",
        },
        "counts": {
            "agents_seen": len(stats.agents),
            "births_life": n_birth,
            "deaths_life": n_death,
        },
        "population": {
            "pop_last": pop_last,
            "pop_min": pop_min,
            "pop_max": pop_max,
            "t_end_pop": stats.pop[-1].t if stats.pop else float("nan"),
        },
        "world": {
            "t_end_world": stats.world[-1].t if stats.world else float("nan"),
            "B_mean_avg": smean(Bm),
            "F_mean_avg": smean(Fm),
            "C_mean_avg": smean(Cm),
            "hazard_frac_avg": smean(Hf),
        },
        "life": {
            "lifespan_mean": smean(lifespans_f),
            "lifespan_p10": pct(lifespans_f, 10.0),
            "lifespan_p50": pct(lifespans_f, 50.0),
            "lifespan_p90": pct(lifespans_f, 90.0),
        },
        "derived": {
            "alpha": float(alpha),
            "R_pc_avg": smean(Rpc_vals),
            "R_pc_p50": pct(Rpc_vals, 50.0),
        },
    }
    return out


# -----------------------
# main API
# -----------------------

def load_run_logs(
    *,
    life_fp: Path = Path("life.jsonl"),
    pop_fp: Path = Path("pop.jsonl"),
    world_fp: Path = Path("world.jsonl"),
) -> RunStats:
    rs = RunStats()
    rs.pop = parse_pop_log(pop_fp)
    rs.world = parse_world_log(world_fp)
    rs.agents = parse_life_log(life_fp)
    rs._life_fp = life_fp
    rs._pop_fp = pop_fp
    rs._world_fp = world_fp
    return rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", default="life.jsonl")
    ap.add_argument("--pop", default="pop.jsonl")
    ap.add_argument("--world", default="world.jsonl")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="stats.json")
    args = ap.parse_args()

    stats = load_run_logs(
        life_fp=Path(args.life),
        pop_fp=Path(args.pop),
        world_fp=Path(args.world),
    )
    summ = summarize(stats, alpha=float(args.alpha))

    out_fp = Path(args.out)
    out_fp.write_text(json.dumps(summ, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_fp}")
    print(json.dumps(summ["counts"], indent=2))
    print(json.dumps(summ["population"], indent=2))
    print(json.dumps(summ["world"], indent=2))
    print(json.dumps(summ["life"], indent=2))


if __name__ == "__main__":
    main()