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


def unwrap_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliserar de vanligaste loggformaten:
      A) {"event":..., "summary": {...}}
      B) {"event":..., "payload": {"event":..., "summary": {...}}}
      C) {"event":..., "payload": {...fields...}}  (ibland utan summary)
      D) payload direkt: {"event":..., "summary": {...}}  (hub skriver payload rakt av)
    Returnerar alltid den dict som innehåller "summary" eller fälten direkt.
    """
    if not isinstance(obj, dict):
        return {}

    # vanlig wrapper
    p = obj.get("payload")
    if isinstance(p, dict):
        return p

    return obj


def get_event(obj: Dict[str, Any]) -> str:
    o = unwrap_obj(obj)

    # event kan finnas på outer eller payload
    ev = o.get("event", "")
    if isinstance(ev, str) and ev:
        return ev

    ev = obj.get("event", "")
    return ev if isinstance(ev, str) else ""


def get_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    o = unwrap_obj(obj)

    s = o.get("summary")
    if isinstance(s, dict):
        return s

    # fallback: ibland är själva payload/obj summary-lik (fält direkt)
    return o if isinstance(o, dict) else {}


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
    repro_alive: int = -1

    # optional reproduction telemetry (may be missing in old logs)
    repro_eligible: int = -1
    repro_attempts: int = 0
    repro_block_cd: int = 0
    repro_block_age: int = 0
    repro_block_mass: int = 0
    repro_block_energy: int = 0


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
class SamplePoint:
    t: float
    agent_id: int
    alive: bool
    age: float
    repro_cd_s: float
    E: float
    M: float
    D: float
    Fg: float
    hunger: float
    # phenotype gates
    A_mature: float
    E_repro_min: float
    M_repro_min: float
    # convenience
    Ecap: float
    repro_rate: float

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

        repro = s.get("repro", {})
        if not isinstance(repro, dict):
            repro = {}

        out.append(
            PopPoint(
                t=fget(s, "t"),
                pop=iget(s, "pop", default=-1),
                mean_E=fget(s, "mean_E"),
                mean_D=fget(s, "mean_D"),
                births=iget(s, "births", default=0),
                deaths=iget(s, "deaths", default=0),

                repro_eligible=iget(repro, "eligible", default=-1),
                repro_attempts=iget(repro, "attempts", default=0),
                repro_block_cd=iget(repro, "block_cd", default=0),
                repro_block_age=iget(repro, "block_age", default=0),
                repro_block_mass=iget(repro, "block_mass", default=0),
                repro_block_energy=iget(repro, "block_energy", default=0),
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

def parse_sample_log(fp: Path) -> List[SamplePoint]:
    out: List[SamplePoint] = []
    for obj in iter_jsonl(fp):
        if get_event(obj) != "sample":
            continue

        # sample_record har INTE "summary", men om hubben wrappar payload vill vi ändå nå rätt nivå
        o = unwrap_obj(obj)

        t = fget(o, "t")
        aid = iget(o, "agent_id", default=-1)
        alive = bool(o.get("alive", True)) if isinstance(o.get("alive", True), bool) else True

        st = o.get("state", {})
        ph = o.get("phenotype", {})

        out.append(SamplePoint(
            t=t,
            agent_id=aid,
            alive=alive,
            age=fget(o, "age"),
            repro_cd_s=fget(st, "repro_cd_s"),

            E=fget(st, "E"),
            M=fget(st, "M"),
            D=fget(st, "D"),
            Fg=fget(st, "Fg"),
            hunger=fget(st, "hunger"),

            A_mature=fget(ph, "A_mature"),
            E_repro_min=fget(ph, "E_repro_min"),
            M_repro_min=fget(ph, "M_repro_min"),

            Ecap=fget(st, "Ecap"),
            repro_rate=fget(ph, "repro_rate"),
        ))
    out.sort(key=lambda z: z.t)
    return out
    
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
def _birth_droughts(pop: List[PopPoint]) -> List[Tuple[float, float, float]]:
    """
    Returns list of (t_start, t_end, duration) for contiguous stretches with zero births,
    using PopPoint spacing. Requires pop sorted.
    """
    if not pop:
        return []
    droughts: List[Tuple[float, float, float]] = []
    in_drought = pop[0].births == 0
    t0 = pop[0].t

    for i in range(1, len(pop)):
        if in_drought:
            if pop[i].births > 0:
                t1 = pop[i].t
                droughts.append((t0, t1, t1 - t0))
                in_drought = False
        else:
            if pop[i].births == 0:
                in_drought = True
                t0 = pop[i-1].t  # start at last tick before zeros begin
    # if drought continues to end
    if in_drought:
        t1 = pop[-1].t
        droughts.append((t0, t1, t1 - t0))
    return droughts


def _window_world_stats(world: List[WorldPoint], t0: float, t1: float) -> Dict[str, float]:
    ws = [w for w in world if finite(w.t) and (w.t >= t0) and (w.t <= t1)]
    if not ws:
        return {}
    def mean(vals: List[float]) -> float:
        vs = [v for v in vals if finite(v)]
        return float(sum(vs)/len(vs)) if vs else float("nan")
    return {
        "B_mean": mean([w.B_mean for w in ws]),
        "C_mean": mean([w.C_mean for w in ws]),
        "F_mean": mean([w.F_mean for w in ws]),
        "hazard_frac": mean([w.hazard_frac for w in ws]),
        "B_sum": mean([w.B_sum for w in ws]),
        "C_sum": mean([w.C_sum for w in ws]),
    }

def summarize_repro_from_samples(stats: RunStats) -> Dict[str, Any]:
    samples = getattr(stats, "samples", [])
    if not samples:
        return {"ok": False, "reason": "no samples loaded"}

    # vi tittar bara på senaste N samples (annars blir allt “historia”)
    N_last = 2000
    ss = samples[-N_last:] if len(samples) > N_last else samples

    # Gate counters (count agent-samples)
    cnt = {
        "n": 0,
        "alive": 0,
        "cd_block": 0,
        "age_block": 0,
        "mass_block": 0,
        "energy_block": 0,
        "eligible": 0,
    }

    # distributions for debugging
    M_all: List[float] = []
    E_all: List[float] = []
    Ecap_all: List[float] = []
    # “closest to eligible” group: passes cd+age+mass but fails energy
    M_ne: List[float] = []
    E_ne: List[float] = []
    Ecap_ne: List[float] = []
    needE_ne: List[float] = []

    for s in ss:
        cnt["n"] += 1
        if not s.alive:
            continue
        cnt["alive"] += 1

        # Compute Ecap from agent.py: E_cap = E_cap_per_M * M
        # Vi har inte E_cap_per_M i loggen, så vi måste hämta den via en konstant.
        # Rekommendation: logga AP.E_cap_per_M i sample senare.
        # Tills vidare: försök läsa från phenotype? finns inte. -> fallback: NaN
        # MEN: din reproduction-gate använder Ecap, så för riktig analys behöver vi E_cap_per_M.
        # Så: här antar vi att du sätter en global konstant. Byt till CLI arg eller logga i sample.
        M = float(s.M)
        Et = float(s.E)
        Ecap = float(s.Ecap)
        needE = float(s.E_repro_min) * Ecap
        
        M_all.append(M); E_all.append(Et); Ecap_all.append(Ecap)

        if float(s.repro_cd_s) > 0.0:
            cnt["cd_block"] += 1
            continue
        if float(s.age) < float(s.A_mature):
            cnt["age_block"] += 1
            continue
        if M < float(s.M_repro_min):
            cnt["mass_block"] += 1
            continue
        if Et < needE:
            cnt["energy_block"] += 1
            M_ne.append(M); E_ne.append(Et); Ecap_ne.append(Ecap); needE_ne.append(needE)
            continue

        cnt["eligible"] += 1

    def sm(xs: List[float]) -> Dict[str, float]:
        xs = [x for x in xs if finite(x)]
        return {
            "n": float(len(xs)),
            "mean": float(sum(xs)/len(xs)) if xs else float("nan"),
            "p10": pct(xs, 10.0),
            "p50": pct(xs, 50.0),
            "p90": pct(xs, 90.0),
        }

    return {
        "ok": True,
        "window_samples": len(ss),
        "gates": cnt,
        "all_alive": {
            "M": sm(M_all),
            "E": sm(E_all),
            "Ecap": sm(Ecap_all),
        },
        "near_energy_gate": {
            "n": len(M_ne),
            "M": sm(M_ne),
            "E": sm(E_ne),
            "Ecap": sm(Ecap_ne),
            "needE": sm(needE_ne),
        },
    }
    
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

    # births from pop log + droughts
    births_total_pop = sum(max(0, int(p.births)) for p in stats.pop)
    deaths_total_pop = sum(max(0, int(p.deaths)) for p in stats.pop)

    droughts = _birth_droughts(stats.pop)
    droughts_sorted = sorted(droughts, key=lambda z: z[2], reverse=True)
    top = droughts_sorted[:3]

    drought_report = []
    for (t0, t1, dur) in top:
        wstats = _window_world_stats(stats.world, t0, t1)
        drought_report.append({
            "t0": float(t0), "t1": float(t1), "dur": float(dur),
            **wstats
        })

    # poplog repro totals
    repro_alive = sum(max(0, p.repro_eligible if p.repro_eligible >= 0 else 0) for p in stats.pop)  # valfritt
    repro_attempts = sum(max(0, p.repro_attempts) for p in stats.pop)
    repro_spawned  = sum(max(0, getattr(p, "repro_spawned", 0)) for p in stats.pop)
    
    out: Dict[str, Any] = {
        "files": {
            "life": str(stats._life_fp) if hasattr(stats, "_life_fp") else "",
            "pop": str(stats._pop_fp) if hasattr(stats, "_pop_fp") else "",
            "world": str(stats._world_fp) if hasattr(stats, "_world_fp") else "",
            "steps": str(stats._steps_fp) if hasattr(stats, "_steps_fp") else "",
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
    
        # ---- NEW: Reproduction diagnostics ----
        "reproduction": summarize_repro_from_samples(stats),
    
        # ---- NEW: Birth dynamics ----
        "births": {
            "births_total_poplog": int(births_total_pop),
            "deaths_total_poplog": int(deaths_total_pop),
            "n_droughts": int(len(droughts)),
            "top_droughts": drought_report,
        },
    }

    out["repro_poplog"] = {
        "attempts_total": int(repro_attempts),
        "spawned_total": int(repro_spawned),
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
    steps_fp: Path = Path("steps.jsonl"),
) -> RunStats:
    rs = RunStats()
    rs.pop = parse_pop_log(pop_fp)
    rs.world = parse_world_log(world_fp)
    rs.agents = parse_life_log(life_fp)
    rs.samples = parse_sample_log(steps_fp)
    rs._life_fp = life_fp
    rs._pop_fp = pop_fp
    rs._world_fp = world_fp
    rs._steps_fp = steps_fp
    return rs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", default="life.jsonl")
    ap.add_argument("--pop", default="pop.jsonl")
    ap.add_argument("--world", default="world.jsonl")
    ap.add_argument("--sample", default="sample.jsonl")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="stats.json")

    # utskrift
    ap.add_argument("--print", dest="print_mode", default="summary",
                    choices=["summary", "all", "keys"],
                    help="summary=vanliga block, all=allt, keys=lista nycklar")
    ap.add_argument("--pretty", action="store_true", help="pretty terminal output")
    args = ap.parse_args()

    stats = load_run_logs(
        life_fp=Path(args.life),
        pop_fp=Path(args.pop),
        world_fp=Path(args.world),
        steps_fp=Path(args.sample),
    )

    summ = summarize(stats, alpha=float(args.alpha))

    out_fp = Path(args.out)
    out_fp.write_text(json.dumps(summ, indent=2, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out_fp}")

    def dump(title: str, obj) -> None:
        print()
        print(f"=== {title} ===")
        if args.pretty:
            print(json.dumps(obj, indent=2, sort_keys=False))
        else:
            print(obj)

    if args.print_mode == "keys":
        dump("top-level keys", sorted(list(summ.keys())))
        return

    if args.print_mode == "all":
        dump("FULL REPORT", summ)
        return

    # summary mode: skriv mycket mer än tidigare, men fortfarande strukturerat
    dump("files", summ.get("files", {}))
    dump("counts", summ.get("counts", {}))
    dump("population", summ.get("population", {}))
    dump("world", summ.get("world", {}))
    dump("life", summ.get("life", {}))
    dump("derived", summ.get("derived", {}))

    # nya block (om de finns)
    if "reproduction" in summ:
        dump("reproduction", summ["reproduction"])
    if "births" in summ:
        dump("births", summ["births"])

    # om du vill ha ett litet “executive snapshot” längst ned
    snap = {
        "pop_last": summ.get("population", {}).get("pop_last"),
        "births_life": summ.get("counts", {}).get("births_life"),
        "deaths_life": summ.get("counts", {}).get("deaths_life"),
        "hazard_frac_avg": summ.get("world", {}).get("hazard_frac_avg"),
        "R_pc_avg": summ.get("derived", {}).get("R_pc_avg"),
    }
    dump("snapshot", snap)


if __name__ == "__main__":
    main()