# genopheno_analyze.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np


# ----------------------------
# utils
# ----------------------------

def _load_jsonl(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _pcts(x: np.ndarray, ps=(10, 50, 90)) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps} | {"mean": float("nan")}
    return {"mean": float(x.mean()), **{f"p{p}": float(np.percentile(x, p)) for p in ps}}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    # minimal, robust enough for analysis runs
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.linalg.norm(xm) * np.linalg.norm(ym))
    if denom <= 1e-12:
        return float("nan")
    return float((xm @ ym) / denom)


def _as_float(x: Any, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ----------------------------
# per-agent aggregation
# ----------------------------

@dataclass
class AgentAgg:
    agent_id: int
    parent_id: Optional[int] = None

    birth_t: float = float("nan")
    death_t: float = float("nan")
    age: float = float("nan")

    # genome
    policy_key: str = ""
    act: str = ""
    layer_sizes: List[int] = None
    traits: Optional[List[float]] = None
    w_norms: Optional[List[float]] = None
    b_norms: Optional[List[float]] = None

    # phenotype summary at birth (whatever user provides)
    phenotype_birth: Dict[str, Any] = None

    # offspring count (derived from birth events)
    offspring: int = 0

    # NEW: maturity / mature fitness
    mature_t: float = float("nan")
    matured: bool = False
    offspring_after_mature: int = 0
    
    # step aggregates (derived phenotypes)
    n_steps: int = 0
    t_first: float = float("nan")
    t_last: float = float("nan")

    mean_speed: float = float("nan")
    mean_hunger: float = float("nan")
    mean_E: float = float("nan")
    mean_B0: float = float("nan")
    mean_F0: float = float("nan")
    mean_C0: float = float("nan")

    # last seen step state (useful for debugging)
    last_E: float = float("nan")
    last_hunger: float = float("nan")
    last_alive: bool = True

    def __post_init__(self):
        if self.layer_sizes is None:
            self.layer_sizes = []
        if self.phenotype_birth is None:
            self.phenotype_birth = {}


class RunningMean:
    def __init__(self) -> None:
        self.n = 0
        self.s = 0.0

    def add(self, x: float) -> None:
        if x != x:  # NaN
            return
        self.n += 1
        self.s += float(x)

    def mean(self) -> float:
        return float("nan") if self.n == 0 else self.s / self.n


def analyze(
    life_fp: Path,
    pop_fp: Optional[Path] = None,
    world_fp: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    agents: Dict[int, AgentAgg] = {}

    # NEW: births timeline (for mature offspring and drift)
    births_by_parent: Dict[int, List[float]] = {}
    parent_child_pairs: List[Tuple[int, int]] = []  # (parent_id, child_id)
    t_end_seen: float = float("nan")
    
    # per-agent running means from step records
    rm_speed: Dict[int, RunningMean] = {}
    rm_hunger: Dict[int, RunningMean] = {}
    rm_E: Dict[int, RunningMean] = {}
    rm_B0: Dict[int, RunningMean] = {}
    rm_F0: Dict[int, RunningMean] = {}
    rm_C0: Dict[int, RunningMean] = {}

    def get_agent(aid: int) -> AgentAgg:
        if aid not in agents:
            agents[aid] = AgentAgg(agent_id=aid)
        return agents[aid]

    # ---- read life.jsonl (birth/death/step)
    births = 0
    deaths = 0
    steps = 0

    for obj in _load_jsonl(life_fp):
        t_obj = obj.get("t", None)
        if t_obj is not None:
            tt = _as_float(t_obj)
            if tt == tt:
                t_end_seen = tt if (t_end_seen != t_end_seen) else max(t_end_seen, tt)
                
        ev = obj.get("event", "")
        if ev == "birth":
            births += 1
            aid = int(obj.get("agent_id", -1))
            a = get_agent(aid)
            a.birth_t = _as_float(obj.get("t"))
            pid = obj.get("parent_id", None)
            a.parent_id = None if pid is None else int(pid)

            # track parent->child relationships and birth times per parent
            if a.parent_id is not None and a.parent_id >= 0:
                births_by_parent.setdefault(a.parent_id, []).append(a.birth_t)
                parent_child_pairs.append((a.parent_id, aid))
                
            pol = obj.get("policy", {}) or {}
            a.policy_key = str(pol.get("key", ""))
            a.act = str(pol.get("act", ""))
            ls = pol.get("layer_sizes", []) or []
            a.layer_sizes = [int(x) for x in ls] if isinstance(ls, list) else []

            tr = obj.get("traits", None)
            a.traits = None if tr is None else [float(x) for x in tr]

            w = obj.get("weights", {}) or {}
            a.w_norms = w.get("w_norms", None)
            a.b_norms = w.get("b_norms", None)

            ph = obj.get("phenotype", {}) or {}
            a.phenotype_birth = ph

            # offspring increment on parent (if present)
            if a.parent_id is not None and a.parent_id >= 0:
                get_agent(a.parent_id).offspring += 1

        elif ev == "death":
            deaths += 1
            aid = int(obj.get("agent_id", -1))
            a = get_agent(aid)
            a.death_t = _as_float(obj.get("t"))
            a.age = _as_float(obj.get("age"))

        elif ev == "step":
            steps += 1
            s = obj.get("summary", {}) or {}
            aid = int(s.get("agent_id", -1))
            if aid < 0:
                continue
            a = get_agent(aid)

            t = _as_float(s.get("t"))
            if a.n_steps == 0:
                a.t_first = t
            a.t_last = t
            a.n_steps += 1

            # init means
            rm_speed.setdefault(aid, RunningMean()).add(_as_float(s.get("speed")))
            rm_hunger.setdefault(aid, RunningMean()).add(_as_float(s.get("hunger")))
            rm_E.setdefault(aid, RunningMean()).add(_as_float(s.get("E")))
            rm_B0.setdefault(aid, RunningMean()).add(_as_float(s.get("B0")))
            rm_F0.setdefault(aid, RunningMean()).add(_as_float(s.get("F0")))
            rm_C0.setdefault(aid, RunningMean()).add(_as_float(s.get("C0")))

            a.last_E = _as_float(s.get("E"))
            a.last_hunger = _as_float(s.get("hunger"))
            a.last_alive = bool(s.get("alive", True))

    # finalize step means
    for aid, a in agents.items():
        a.mean_speed = rm_speed.get(aid, RunningMean()).mean()
        a.mean_hunger = rm_hunger.get(aid, RunningMean()).mean()
        a.mean_E = rm_E.get(aid, RunningMean()).mean()
        a.mean_B0 = rm_B0.get(aid, RunningMean()).mean()
        a.mean_F0 = rm_F0.get(aid, RunningMean()).mean()
        a.mean_C0 = rm_C0.get(aid, RunningMean()).mean()

    # ---- finalize maturity + mature offspring metrics
    # if we didn't see any timestamps, try fallback from deaths
    if t_end_seen != t_end_seen:
        ts = [a.death_t for a in agents.values() if a.death_t == a.death_t]
        t_end_seen = max(ts) if ts else float("nan")

    for aid, a in agents.items():
        # need birth phenotype A_mature
        A_m = _as_float((a.phenotype_birth or {}).get("A_mature"))
        if a.birth_t != a.birth_t or A_m != A_m:
            continue

        a.mature_t = float(a.birth_t + A_m)

        # matured if survived past mature_t; if alive at end, use t_end_seen
        if a.death_t == a.death_t:
            a.matured = bool(a.death_t >= a.mature_t)
        else:
            a.matured = bool(t_end_seen == t_end_seen and t_end_seen >= a.mature_t)

        # offspring after maturity = count child births with t >= mature_t
        child_ts = births_by_parent.get(aid, [])
        if child_ts:
            a.offspring_after_mature = int(sum(1 for bt in child_ts if bt == bt and bt >= a.mature_t))
        else:
            a.offspring_after_mature = 0
            
    # ---- population.jsonl summary (optional)
    pop_stats = {}
    if pop_fp and pop_fp.exists():
        ts = []
        pops = []
        births_p = []
        deaths_p = []
        mean_E = []
        for obj in _load_jsonl(pop_fp):
            if obj.get("event") != "population":
                continue
            s = obj.get("summary", {}) or {}
            ts.append(_as_float(s.get("t")))
            pops.append(int(s.get("pop", -1)))
            births_p.append(int(s.get("births", 0)))
            deaths_p.append(int(s.get("deaths", 0)))
            mean_E.append(_as_float(s.get("mean_E")))
        if len(ts) > 0:
            pop_stats = {
                "t_end_pop": float(ts[-1]),
                "pop_last": int(pops[-1]),
                "pop_min": int(np.min(pops)),
                "pop_max": int(np.max(pops)),
                "mean_E_avg": float(np.nanmean(mean_E)) if len(mean_E) else float("nan"),
            }

    # ---- world.jsonl summary (optional)
    world_stats = {}
    if world_fp and world_fp.exists():
        ts = []
        Bm = []
        Fm = []
        Cm = []
        hf = []
        for obj in _load_jsonl(world_fp):
            if obj.get("event") != "world":
                continue
            s = obj.get("summary", {}) or {}
            ts.append(_as_float(s.get("t")))
            Bm.append(_as_float((s.get("B", {}) or {}).get("mean")))
            Fm.append(_as_float((s.get("F", {}) or {}).get("mean")))
            Cm.append(_as_float((s.get("C", {}) or {}).get("mean")))
            hf.append(_as_float((s.get("F", {}) or {}).get("hazard_frac_0p35")))
        if len(ts) > 0:
            world_stats = {
                "t_end_world": float(ts[-1]),
                "B_mean_avg": float(np.nanmean(Bm)),
                "F_mean_avg": float(np.nanmean(Fm)),
                "C_mean_avg": float(np.nanmean(Cm)),
                "hazard_frac_avg": float(np.nanmean(hf)),
            }

    # ---- per-agent output records (jsonl)
    agent_rows: List[Dict[str, Any]] = []
    for aid in sorted(agents.keys()):
        a = agents[aid]
        d = asdict(a)
        # make it smaller / stable
        d["traits_dim"] = None if a.traits is None else len(a.traits)
        d["layer_sizes"] = a.layer_sizes
        agent_rows.append(d)

    # ---- summary stats + geno->pheno links
    ages = np.array([a.age for a in agents.values() if a.age == a.age], dtype=float)
    offspr = np.array([a.offspring for a in agents.values()], dtype=float)

    matured_flags = np.array([1.0 if a.matured else 0.0 for a in agents.values()], dtype=float)
    offspr_m = np.array([float(a.offspring_after_mature) for a in agents.values()], dtype=float)
    # NEW: fitness conditional on reaching maturity
    matured_mask = np.array([bool(a.matured) for a in agents.values()], dtype=bool)
    offspr_all = offspr
    offspr_cond = offspr_all[matured_mask] if matured_mask.size else np.array([], dtype=float)
    
    summary: Dict[str, Any] = {
        "files": {
            "life": str(life_fp),
            "pop": None if pop_fp is None else str(pop_fp),
            "world": None if world_fp is None else str(world_fp),
        },
        "counts": {
            "agents_seen": int(len(agents)),
            "births_life": int(births),
            "deaths_life": int(deaths),
            "steps_life": int(steps),
        },
        "life": {
            "lifespan": _pcts(ages),
            "offspring": _pcts(offspr),
            "offspring_share_zero": float(np.mean(offspr == 0.0)) if offspr.size else float("nan"),
            "matured_share": float(np.mean(matured_flags)) if matured_flags.size else float("nan"),
            "offspring_after_mature": _pcts(offspr_m),
            "offspring_after_mature_share_zero": float(np.mean(offspr_m == 0.0)) if offspr_m.size else float("nan"),
            "offspring_cond_matured": _pcts(offspr_cond),
            "matured_n": int(matured_mask.sum()) if matured_mask.size else 0,
        },
        "population": pop_stats,
        "world": world_stats,
    }

    # trait-fitness correlations (if traits exist)
    # We'll compute corr(trait_i, age) for i in first K dims, plus corr(trait_i, offspring).
    trait_mat = [a.traits for a in agents.values() if a.traits is not None and a.age == a.age]
    if trait_mat:
        X = np.asarray(trait_mat, dtype=float)
        y_age = np.asarray([a.age for a in agents.values() if a.traits is not None and a.age == a.age], dtype=float)
        y_off = np.asarray([a.offspring for a in agents.values() if a.traits is not None and a.age == a.age], dtype=float)
        y_off_m = np.asarray(
            [a.offspring_after_mature for a in agents.values() if a.traits is not None and a.age == a.age],
            dtype=float,
        )
        y_matured = np.asarray(
            [1.0 if a.matured else 0.0 for a in agents.values() if a.traits is not None and a.age == a.age],
            dtype=float,
        )
        # conditional correlations among matured only
        m_mask = np.asarray(
            [bool(a.matured) for a in agents.values() if a.traits is not None and a.age == a.age],
            dtype=bool,
        )
        if m_mask.size and np.any(m_mask):
            X_m = X[m_mask]
            y_off_matured = y_off[m_mask]
            K2 = min(12, X_m.shape[1])
            corr_off_cond = {f"trait_{i}": _pearson(X_m[:, i], y_off_matured) for i in range(K2)}
        else:
            corr_off_cond = {}
        
        K = min(12, X.shape[1])
        corr_age = {f"trait_{i}": _pearson(X[:, i], y_age) for i in range(K)}
        corr_off = {f"trait_{i}": _pearson(X[:, i], y_off) for i in range(K)}
        corr_off_m = {f"trait_{i}": _pearson(X[:, i], y_off_m) for i in range(K)}
        corr_matured = {f"trait_{i}": _pearson(X[:, i], y_matured) for i in range(K)}        
        summary["geno_pheno"] = {
            "corr_trait_vs_age": corr_age,
            "corr_trait_vs_offspring": corr_off,
            "corr_trait_vs_offspring_after_mature": corr_off_m,
            "corr_trait_vs_matured": corr_matured,
            "n_with_traits_and_age": int(X.shape[0]),
            "trait_dim": int(X.shape[1]),
            "corr_trait_vs_offspring_cond_matured": corr_off_cond
        }
    else:
        summary["geno_pheno"] = {
            "corr_trait_vs_age": {},
            "corr_trait_vs_offspring": {},
            "n_with_traits_and_age": 0,
            "trait_dim": 0,
        }

    # ---- parent->child trait regression (heritability proxy)
    # Uses birth traits only; skips if missing.
    pc = []
    for pid, cid in parent_child_pairs:
        p = agents.get(pid)
        c = agents.get(cid)
        if p is None or c is None:
            continue
        if p.traits is None or c.traits is None:
            continue
        pc.append((np.asarray(p.traits, dtype=float), np.asarray(c.traits, dtype=float)))

    if pc:
        P = np.stack([x for x, _ in pc], axis=0)
        C = np.stack([y for _, y in pc], axis=0)
        K = min(P.shape[1], 12)

        corr_pc = {f"trait_{i}": _pearson(P[:, i], C[:, i]) for i in range(K)}
        mean_abs_d = {f"trait_{i}": float(np.nanmean(np.abs(C[:, i] - P[:, i]))) for i in range(K)}

        summary["heritability"] = {
            "n_pairs": int(P.shape[0]),
            "trait_dim": int(P.shape[1]),
            "corr_parent_child": corr_pc,
            "mean_abs_delta": mean_abs_d,
        }
    else:
        summary["heritability"] = {
            "n_pairs": 0,
            "trait_dim": 0,
            "corr_parent_child": {},
            "mean_abs_delta": {},
        }

    # ---- drift over time: trait/phenotype stats per time bin (birth events)
    # crude but very informative to see evolution direction.
    bin_w = 500.0
    if t_end_seen == t_end_seen and t_end_seen > 0:
        nb = int(math.ceil(t_end_seen / bin_w))
        bins = [(i * bin_w, (i + 1) * bin_w) for i in range(nb)]
    else:
        bins = [(0.0, 500.0), (500.0, 1000.0), (1000.0, 1500.0), (1500.0, 2000.0)]

    # collect births only (agents w/ birth_t finite)
    born_agents = [a for a in agents.values() if a.birth_t == a.birth_t and a.traits is not None]
    drift_rows = []
    for lo, hi in bins:
        bucket = [a for a in born_agents if lo <= a.birth_t < hi]
        if not bucket:
            continue

        Xb = np.asarray([a.traits for a in bucket], dtype=float)
        K = min(12, Xb.shape[1])

        # phenotype fields we care about (if present)
        def phv(a: AgentAgg, k: str) -> float:
            return _as_float((a.phenotype_birth or {}).get(k))

        ph_keys = ["A_mature", "p_repro_base", "E_repro_min", "repro_cost", "metabolism_scale"]
        ph = {k: np.asarray([phv(a, k) for a in bucket], dtype=float) for k in ph_keys}

        drift_rows.append({
            "t_lo": float(lo),
            "t_hi": float(hi),
            "n_births": int(len(bucket)),
            "traits_mean": {f"trait_{i}": float(np.nanmean(Xb[:, i])) for i in range(K)},
            "traits_p10":  {f"trait_{i}": float(np.nanpercentile(Xb[:, i], 10)) for i in range(K)},
            "traits_p50":  {f"trait_{i}": float(np.nanpercentile(Xb[:, i], 50)) for i in range(K)},
            "traits_p90":  {f"trait_{i}": float(np.nanpercentile(Xb[:, i], 90)) for i in range(K)},
            "pheno_mean": {k: float(np.nanmean(v)) for k, v in ph.items()},
        })

    summary["drift"] = {
        "bin_width": float(bin_w),
        "bins": drift_rows,
    }
    
    return summary, agent_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", default="life.jsonl", type=str)
    ap.add_argument("--pop", default="pop.jsonl", type=str)
    ap.add_argument("--world", default="world.jsonl", type=str)

    ap.add_argument("--out_summary", default="genopheno_stats.json", type=str)
    ap.add_argument("--out_agents", default="genopheno_agents.jsonl", type=str)

    args = ap.parse_args()

    life_fp = Path(args.life)
    pop_fp = Path(args.pop) if args.pop else None
    world_fp = Path(args.world) if args.world else None

    summary, agent_rows = analyze(life_fp, pop_fp, world_fp)

    # write files
    Path(args.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with Path(args.out_agents).open("w", encoding="utf-8") as f:
        for row in agent_rows:
            f.write(json.dumps(row) + "\n")

    # print concise report
    print("== Genopheno postprocess ==")
    print(f"agents_seen: {summary['counts']['agents_seen']}")
    print(f"births/deaths/steps: {summary['counts']['births_life']} / {summary['counts']['deaths_life']} / {summary['counts']['steps_life']}")

    ls = summary["life"]["lifespan"]
    print(f"lifespan mean/p10/p50/p90: {ls['mean']:.3f} / {ls['p10']:.3f} / {ls['p50']:.3f} / {ls['p90']:.3f}")

    os_ = summary["life"]["offspring"]
    print(f"offspring mean/p10/p50/p90: {os_['mean']:.3f} / {os_['p10']:.3f} / {os_['p50']:.3f} / {os_['p90']:.3f}")
    print(f"offspring share zero: {summary['life']['offspring_share_zero']:.3f}")

    if summary.get("population"):
        p = summary["population"]
        if p:
            print(f"pop last/min/max: {p.get('pop_last')} / {p.get('pop_min')} / {p.get('pop_max')}  (t_end={p.get('t_end_pop')})")

    if summary.get("world"):
        w = summary["world"]
        if w:
            print(f"world means B/F/C: {w.get('B_mean_avg'):.4f} / {w.get('F_mean_avg'):.4f} / {w.get('C_mean_avg'):.6f}  hazard_frac: {w.get('hazard_frac_avg'):.4f}")

    gp = summary.get("geno_pheno", {})
    n = gp.get("n_with_traits_and_age", 0)
    if n > 0:
        # print top 3 abs corr for age and offspring (first K only)
        def topk(d: Dict[str, float], k=3):
            items = [(k_, v) for k_, v in d.items() if v == v]
            items.sort(key=lambda kv: abs(kv[1]), reverse=True)
            return items[:k]

        top_age = topk(gp.get("corr_trait_vs_age", {}), 3)
        top_off = topk(gp.get("corr_trait_vs_offspring", {}), 3)
        print(f"traits available: n={n}, dim={gp.get('trait_dim')}")
        print("top corr trait vs age:", ", ".join([f"{k}={v:+.3f}" for k, v in top_age]) or "NA")
        print("top corr trait vs offspring:", ", ".join([f"{k}={v:+.3f}" for k, v in top_off]) or "NA")

    print(f"wrote: {args.out_summary}")
    print(f"wrote: {args.out_agents}")


if __name__ == "__main__":
    main()