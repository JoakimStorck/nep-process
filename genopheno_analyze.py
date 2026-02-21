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


def _as_float(x: Any, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _isfinite(x: float) -> bool:
    return (x == x) and (abs(x) != float("inf"))


def _pcts(x: np.ndarray, ps=(10, 50, 90)) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps} | {"mean": float("nan")}
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps} | {"mean": float("nan")}
    return {"mean": float(x.mean()), **{f"p{p}": float(np.percentile(x, p)) for p in ps}}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.linalg.norm(xm) * np.linalg.norm(ym))
    if denom <= 1e-12:
        return float("nan")
    return float((xm @ ym) / denom)


def _lin_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y ~ a + b x (returns b)."""
    x = x.astype(float)
    y = y.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    vx = float(np.var(x))
    if vx <= 1e-12:
        return float("nan")
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    return cov / vx


def _topk_abs(d: Dict[str, float], k: int = 5) -> List[Tuple[str, float]]:
    items = [(kk, vv) for kk, vv in d.items() if _isfinite(vv)]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return items[:k]


def _md_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


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

    # phenotype summary at birth
    phenotype_birth: Dict[str, Any] = None

    # offspring count
    offspring: int = 0

    # maturity / mature fitness
    mature_t: float = float("nan")
    matured: bool = False
    offspring_after_mature: int = 0

    # step aggregates
    n_steps: int = 0
    t_first: float = float("nan")
    t_last: float = float("nan")

    mean_speed: float = float("nan")
    mean_hunger: float = float("nan")
    mean_E: float = float("nan")
    mean_B0: float = float("nan")
    mean_F0: float = float("nan")
    mean_C0: float = float("nan")

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
        if not _isfinite(x):
            return
        self.n += 1
        self.s += float(x)

    def mean(self) -> float:
        return float("nan") if self.n == 0 else self.s / self.n


def analyze(
    life_fp: Path,
    pop_fp: Optional[Path] = None,
    world_fp: Optional[Path] = None,
    sample_fp: Optional[Path] = None,
    out_report_md: Optional[Path] = None,
    out_birth_cohorts_json: Optional[Path] = None,
    out_lineages_json: Optional[Path] = None,
    out_sample_stats_json: Optional[Path] = None,
    cohort_bin_w: float = 500.0,
    top_quantile: float = 0.80,   # top 20% fitness group cutoff
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    agents: Dict[int, AgentAgg] = {}
    births_by_parent: Dict[int, List[float]] = {}
    parent_child_pairs: List[Tuple[int, int]] = []
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

    births = 0
    deaths = 0
    steps = 0

    # ---- read life.jsonl
    for obj in _load_jsonl(life_fp):
        t_obj = obj.get("t", None)
        if t_obj is not None:
            tt = _as_float(t_obj)
            if _isfinite(tt):
                t_end_seen = tt if not _isfinite(t_end_seen) else max(t_end_seen, tt)

        ev = obj.get("event", "")
        if ev == "birth":
            births += 1
            aid = int(obj.get("agent_id", -1))
            a = get_agent(aid)
            a.birth_t = _as_float(obj.get("t"))
            pid = obj.get("parent_id", None)
            a.parent_id = None if pid is None else int(pid)

            if a.parent_id is not None and a.parent_id >= 0:
                births_by_parent.setdefault(a.parent_id, []).append(a.birth_t)
                parent_child_pairs.append((a.parent_id, aid))
                get_agent(a.parent_id).offspring += 1

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

    # fallback t_end
    if not _isfinite(t_end_seen):
        ts = [a.death_t for a in agents.values() if _isfinite(a.death_t)]
        t_end_seen = max(ts) if ts else float("nan")

    # maturity + offspring after maturity
    for aid, a in agents.items():
        A_m = _as_float((a.phenotype_birth or {}).get("A_mature"))
        if not _isfinite(a.birth_t) or not _isfinite(A_m):
            continue
        a.mature_t = float(a.birth_t + A_m)
        if _isfinite(a.death_t):
            a.matured = bool(a.death_t >= a.mature_t)
        else:
            a.matured = bool(_isfinite(t_end_seen) and t_end_seen >= a.mature_t)

        child_ts = births_by_parent.get(aid, [])
        a.offspring_after_mature = int(sum(1 for bt in child_ts if _isfinite(bt) and bt >= a.mature_t))

    # optional pop/world summaries (unchanged-ish)
    pop_stats = {}
    if pop_fp and pop_fp.exists():
        ts, pops, mean_E = [], [], []
        for obj in _load_jsonl(pop_fp):
            if obj.get("event") != "population":
                continue
            s = obj.get("summary", {}) or {}
            ts.append(_as_float(s.get("t")))
            pops.append(int(s.get("pop", -1)))
            mean_E.append(_as_float(s.get("mean_E")))
        if ts:
            pop_stats = {
                "t_end_pop": float(ts[-1]),
                "pop_last": int(pops[-1]),
                "pop_min": int(np.min(pops)),
                "pop_max": int(np.max(pops)),
                "mean_E_avg": float(np.nanmean(mean_E)),
            }

    world_stats = {}
    if world_fp and world_fp.exists():
        ts, Bm, Fm, Cm, hf = [], [], [], [], []
        for obj in _load_jsonl(world_fp):
            if obj.get("event") != "world":
                continue
            s = obj.get("summary", {}) or {}
            ts.append(_as_float(s.get("t")))
            Bm.append(_as_float((s.get("B", {}) or {}).get("mean")))
            Fm.append(_as_float((s.get("F", {}) or {}).get("mean")))
            Cm.append(_as_float((s.get("C", {}) or {}).get("mean")))
            hf.append(_as_float((s.get("F", {}) or {}).get("hazard_frac_0p35")))
        if ts:
            world_stats = {
                "t_end_world": float(ts[-1]),
                "B_mean_avg": float(np.nanmean(Bm)),
                "F_mean_avg": float(np.nanmean(Fm)),
                "C_mean_avg": float(np.nanmean(Cm)),
                "hazard_frac_avg": float(np.nanmean(hf)),
            }

    # per-agent rows
    agent_rows: List[Dict[str, Any]] = []
    for aid in sorted(agents.keys()):
        a = agents[aid]
        d = asdict(a)
        d["traits_dim"] = None if a.traits is None else len(a.traits)
        agent_rows.append(d)

    # arrays for stats
    ages = np.array([a.age for a in agents.values() if _isfinite(a.age)], dtype=float)
    offspr = np.array([float(a.offspring) for a in agents.values()], dtype=float)
    offspr_m = np.array([float(a.offspring_after_mature) for a in agents.values()], dtype=float)
    matured_flags = np.array([1.0 if a.matured else 0.0 for a in agents.values()], dtype=float)

    matured_mask = np.array([bool(a.matured) for a in agents.values()], dtype=bool)
    offspr_cond = offspr[matured_mask] if matured_mask.size else np.array([], dtype=float)

    summary: Dict[str, Any] = {
        "files": {
            "life": str(life_fp),
            "pop": None if pop_fp is None else str(pop_fp),
            "world": None if world_fp is None else str(world_fp),
            "sample": None if sample_fp is None else str(sample_fp),
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

    # -------- phenotype fields: auto-detect numeric keys from birth phenotypes
    pheno_keys = set()
    for a in agents.values():
        ph = a.phenotype_birth or {}
        for k, v in ph.items():
            vv = _as_float(v)
            if _isfinite(vv):
                pheno_keys.add(str(k))
    pheno_keys = sorted(pheno_keys)

    # ----------------------------
    # sample.jsonl drift (cross-sectional longitudinal)
    # ----------------------------
    sample_stats = {}
    
    if sample_fp is not None and sample_fp.exists():
        ts = []
        popn = []
        aid = []
    
        # state
        sE = []
        sH = []
        sD = []
    
        # phenotype-at-sample (om du loggar den; annars fallback till birth-pheno)
        ph_cols = {k: [] for k in pheno_keys}
    
        for obj in _load_jsonl(sample_fp):
            if obj.get("event") != "sample":
                continue
    
            t = _as_float(obj.get("t"))
            if not _isfinite(t):
                continue
    
            ts.append(t)
            aid.append(int(obj.get("agent_id", -1)))
    
            popn.append(_as_float(obj.get("pop_n")))
    
            st = obj.get("state", {}) or {}
            sE.append(_as_float(st.get("E")))
            sH.append(_as_float(st.get("hunger")))
            sD.append(_as_float(st.get("D")))
    
            ph = obj.get("phenotype", None)
            if ph is None:
                # fallback: använd birth phenotype om vi känner agenten
                a = agents.get(int(obj.get("agent_id", -1)))
                ph = (a.phenotype_birth if a is not None else {}) or {}
    
            for k in pheno_keys:
                ph_cols[k].append(_as_float((ph or {}).get(k)))
    
        t_arr = np.asarray(ts, dtype=float)
        pop_arr = np.asarray(popn, dtype=float)
        E_arr = np.asarray(sE, dtype=float)
        H_arr = np.asarray(sH, dtype=float)
        D_arr = np.asarray(sD, dtype=float)
    
        # drift: slope vs t + tidig/sen diff (median split)
        if t_arr.size >= 5:
            t_med = float(np.nanmedian(t_arr))
            early = t_arr <= t_med
            late  = t_arr >  t_med
    
            drift = {}
            for k in pheno_keys:
                x = np.asarray(ph_cols[k], dtype=float)
                drift[k] = {
                    "slope_per_time": _lin_slope(t_arr, x),                    # b i x ~ a + b t
                    "corr_vs_time": _pearson(t_arr, x),
                    "mean_early": float(np.nanmean(x[early])) if np.any(early) else float("nan"),
                    "mean_late":  float(np.nanmean(x[late]))  if np.any(late)  else float("nan"),
                    "diff_late_minus_early": (
                        float(np.nanmean(x[late]) - np.nanmean(x[early]))
                        if (np.any(early) and np.any(late)) else float("nan")
                    ),
                }
    
            # regim: quantiler på pop_n
            reg = {}
            if np.any(np.isfinite(pop_arr)):
                q50 = float(np.nanquantile(pop_arr, 0.50))
                lo = pop_arr <= q50
                hi = pop_arr >  q50
                for k in pheno_keys:
                    x = np.asarray(ph_cols[k], dtype=float)
                    reg[k] = {
                        "mean_low_pop": float(np.nanmean(x[lo])) if np.any(lo) else float("nan"),
                        "mean_high_pop": float(np.nanmean(x[hi])) if np.any(hi) else float("nan"),
                        "diff_high_minus_low": (
                            float(np.nanmean(x[hi]) - np.nanmean(x[lo]))
                            if (np.any(lo) and np.any(hi)) else float("nan")
                        ),
                        "corr_vs_pop_n": _pearson(pop_arr, x),
                    }
    
            # koppling fenotyp ↔ state (E/H/D)
            ph_state = {}
            for k in pheno_keys:
                x = np.asarray(ph_cols[k], dtype=float)
                ph_state[k] = {
                    "corr_vs_E": _pearson(x, E_arr),
                    "corr_vs_hunger": _pearson(x, H_arr),
                    "corr_vs_D": _pearson(x, D_arr),
                }
    
            sample_stats = {
                "n_rows": int(t_arr.size),
                "n_unique_agents": int(len(set(aid))),
                "t_min": float(np.nanmin(t_arr)),
                "t_max": float(np.nanmax(t_arr)),
                "pop_n": _pcts(pop_arr),
                "state_E": _pcts(E_arr),
                "state_hunger": _pcts(H_arr),
                "state_D": _pcts(D_arr),
                "drift_vs_time": drift,
                "regime_vs_pop_n": reg,
                "pheno_state_corr": ph_state,
            }
    
    summary["sample"] = sample_stats
    
    if out_sample_stats_json is not None and sample_stats:
        out_sample_stats_json.write_text(json.dumps(sample_stats, indent=2), encoding="utf-8")
        
    # build phenotype matrix aligned with agents
    # only agents with death age defined (for fitness outcomes)
    valid = [a for a in agents.values() if _isfinite(a.age) and (a.phenotype_birth is not None)]
    if valid:
        y_age = np.array([a.age for a in valid], dtype=float)
        y_R0 = np.array([float(a.offspring) for a in valid], dtype=float)
        y_R0m = np.array([float(a.offspring_after_mature) for a in valid], dtype=float)
        y_matured = np.array([1.0 if a.matured else 0.0 for a in valid], dtype=float)

        # compute per-key correlations + slopes
        pheno_corr = {}
        pheno_slope = {}
        pheno_sel = {}  # selection differential vs top quantile in R0m (or R0 if no R0m)
        # choose fitness axis for selection differential
        fit = y_R0m.copy()
        if np.nanmax(fit) <= 0.0:
            fit = y_R0.copy()

        # cutoff for top group
        if fit.size:
            q = np.nanquantile(fit, top_quantile) if np.any(np.isfinite(fit)) else float("nan")
        else:
            q = float("nan")

        for k in pheno_keys:
            x = np.array([_as_float((a.phenotype_birth or {}).get(k)) for a in valid], dtype=float)

            pheno_corr[k] = {
                "corr_vs_age": _pearson(x, y_age),
                "corr_vs_R0": _pearson(x, y_R0),
                "corr_vs_R0_mature": _pearson(x, y_R0m),
                "corr_vs_matured": _pearson(x, y_matured),
            }
            pheno_slope[k] = {
                "slope_age_per_unit": _lin_slope(x, y_age),
                "slope_R0_per_unit": _lin_slope(x, y_R0),
                "slope_R0m_per_unit": _lin_slope(x, y_R0m),
            }

            if _isfinite(q):
                top = x[fit >= q]
                allx = x[np.isfinite(x)]
                if top.size and allx.size:
                    pheno_sel[k] = float(np.nanmean(top) - np.nanmean(allx))
                else:
                    pheno_sel[k] = float("nan")

        summary["pheno_fitness"] = {
            "keys": pheno_keys,
            "n": int(len(valid)),
            "top_quantile": float(top_quantile),
            "corr": pheno_corr,
            "slope": pheno_slope,
            "selection_differential": pheno_sel,
        }
    else:
        summary["pheno_fitness"] = {"keys": [], "n": 0, "corr": {}, "slope": {}, "selection_differential": {}}

    # -------- traits: correlations (use full trait dim, not min(12,...))
    trait_mat = [(a.traits, a.age, a.offspring, a.offspring_after_mature, a.matured)
                 for a in agents.values() if (a.traits is not None and _isfinite(a.age))]
    if trait_mat:
        X = np.asarray([t for t, *_ in trait_mat], dtype=float)
        y_age = np.asarray([age for _, age, *_ in trait_mat], dtype=float)
        y_off = np.asarray([off for *_, off, __, ___ in [(None,)]*0], dtype=float)  # dummy to keep lint quiet
        y_off = np.asarray([off for _, __, off, ___, ____ in trait_mat], dtype=float)
        y_off_m = np.asarray([offm for _, __, ___, offm, ____ in trait_mat], dtype=float)
        y_matured = np.asarray([1.0 if m else 0.0 for _, __, ___, ____, m in trait_mat], dtype=float)

        K = int(X.shape[1])
        corr_age = {f"trait_{i}": _pearson(X[:, i], y_age) for i in range(K)}
        corr_off = {f"trait_{i}": _pearson(X[:, i], y_off) for i in range(K)}
        corr_off_m = {f"trait_{i}": _pearson(X[:, i], y_off_m) for i in range(K)}
        corr_matured = {f"trait_{i}": _pearson(X[:, i], y_matured) for i in range(K)}

        summary["geno_pheno"] = {
            "n_with_traits_and_age": int(X.shape[0]),
            "trait_dim": int(X.shape[1]),
            "corr_trait_vs_age": corr_age,
            "corr_trait_vs_offspring": corr_off,
            "corr_trait_vs_offspring_after_mature": corr_off_m,
            "corr_trait_vs_matured": corr_matured,
        }
    else:
        summary["geno_pheno"] = {"n_with_traits_and_age": 0, "trait_dim": 0, "corr_trait_vs_age": {}, "corr_trait_vs_offspring": {}}

    # -------- heritability proxy on traits
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
        K = int(min(P.shape[1], C.shape[1]))
        corr_pc = {f"trait_{i}": _pearson(P[:, i], C[:, i]) for i in range(K)}
        mean_abs_d = {f"trait_{i}": float(np.nanmean(np.abs(C[:, i] - P[:, i]))) for i in range(K)}
        summary["heritability"] = {"n_pairs": int(P.shape[0]), "trait_dim": int(P.shape[1]), "corr_parent_child": corr_pc, "mean_abs_delta": mean_abs_d}
    else:
        summary["heritability"] = {"n_pairs": 0, "trait_dim": 0, "corr_parent_child": {}, "mean_abs_delta": {}}

    # -------- drift / birth cohorts (traits + phenotype + fitness of cohort)
    if _isfinite(t_end_seen) and t_end_seen > 0:
        nb = int(math.ceil(t_end_seen / cohort_bin_w))
        bins = [(i * cohort_bin_w, (i + 1) * cohort_bin_w) for i in range(nb)]
    else:
        bins = [(0.0, 500.0), (500.0, 1000.0), (1000.0, 1500.0), (1500.0, 2000.0)]

    born = [a for a in agents.values() if _isfinite(a.birth_t) and (a.traits is not None)]
    drift_rows = []
    for lo, hi in bins:
        bucket = [a for a in born if lo <= a.birth_t < hi]
        if not bucket:
            continue

        Xb = np.asarray([a.traits for a in bucket], dtype=float)
        Kt = int(Xb.shape[1])

        # cohort fitness outcomes (only those with observed age)
        ages_b = np.asarray([a.age for a in bucket if _isfinite(a.age)], dtype=float)
        R0_b = np.asarray([float(a.offspring) for a in bucket if _isfinite(a.age)], dtype=float)
        R0m_b = np.asarray([float(a.offspring_after_mature) for a in bucket if _isfinite(a.age)], dtype=float)
        matured_b = np.asarray([1.0 if a.matured else 0.0 for a in bucket if _isfinite(a.age)], dtype=float)

        # phenotype stats per key
        ph_stats = {}
        for k in pheno_keys:
            xv = np.asarray([_as_float((a.phenotype_birth or {}).get(k)) for a in bucket], dtype=float)
            ph_stats[k] = _pcts(xv)

        drift_rows.append({
            "t_lo": float(lo),
            "t_hi": float(hi),
            "n_births": int(len(bucket)),
            "traits_mean": {f"trait_{i}": float(np.nanmean(Xb[:, i])) for i in range(Kt)},
            "traits_p10":  {f"trait_{i}": float(np.nanpercentile(Xb[:, i], 10)) for i in range(Kt)},
            "traits_p50":  {f"trait_{i}": float(np.nanpercentile(Xb[:, i], 50)) for i in range(Kt)},
            "traits_p90":  {f"trait_{i}": float(np.nanpercentile(Xb[:, i], 90)) for i in range(Kt)},
            "cohort_fitness": {
                "age": _pcts(ages_b),
                "R0": _pcts(R0_b),
                "R0_mature": _pcts(R0m_b),
                "matured_share": float(np.mean(matured_b)) if matured_b.size else float("nan"),
            },
            "pheno": ph_stats,
        })

    cohorts = {"bin_width": float(cohort_bin_w), "bins": drift_rows}
    summary["birth_cohorts"] = cohorts

    # -------- lineage: top parents by R0 and by R0_mature
    parents = [a for a in agents.values() if a.offspring > 0]
    parents.sort(key=lambda a: a.offspring, reverse=True)
    top_parents = parents[:25]

    parents_m = [a for a in agents.values() if a.offspring_after_mature > 0]
    parents_m.sort(key=lambda a: a.offspring_after_mature, reverse=True)
    top_parents_m = parents_m[:25]

    lineage = {
        "top_parents_R0": [
            {"agent_id": a.agent_id, "R0": int(a.offspring), "age": a.age, "birth_t": a.birth_t, "pheno": a.phenotype_birth}
            for a in top_parents
        ],
        "top_parents_R0_mature": [
            {"agent_id": a.agent_id, "R0_mature": int(a.offspring_after_mature), "age": a.age, "birth_t": a.birth_t, "pheno": a.phenotype_birth}
            for a in top_parents_m
        ],
    }
    summary["lineage"] = lineage

    # optionally write cohorts/lineages files
    if out_birth_cohorts_json is not None:
        out_birth_cohorts_json.write_text(json.dumps(cohorts, indent=2), encoding="utf-8")
    if out_lineages_json is not None:
        out_lineages_json.write_text(json.dumps(lineage, indent=2), encoding="utf-8")

    # -------- markdown report (standard)
    if out_report_md is not None:
        md = []
        md.append("# Genopheno standard report\n")
        md.append(f"- life: `{life_fp}`\n")
        if pop_fp: md.append(f"- pop: `{pop_fp}`\n")
        if world_fp: md.append(f"- world: `{world_fp}`\n")
        md.append("\n## Counts\n")
        md.append(_md_table([
            ["metric", "value"],
            ["agents_seen", str(summary["counts"]["agents_seen"])],
            ["births_life", str(summary["counts"]["births_life"])],
            ["deaths_life", str(summary["counts"]["deaths_life"])],
            ["steps_life", str(summary["counts"]["steps_life"])],
        ]))

        ls = summary["life"]["lifespan"]
        md.append("\n## Lifespan\n")
        md.append(_md_table([
            ["stat", "value"],
            ["mean", f"{ls['mean']:.3f}"],
            ["p10", f"{ls['p10']:.3f}"],
            ["p50", f"{ls['p50']:.3f}"],
            ["p90", f"{ls['p90']:.3f}"],
        ]))

        os_ = summary["life"]["offspring"]
        md.append("\n## Offspring\n")
        md.append(_md_table([
            ["stat", "value"],
            ["mean", f"{os_['mean']:.3f}"],
            ["p10", f"{os_['p10']:.3f}"],
            ["p50", f"{os_['p50']:.3f}"],
            ["p90", f"{os_['p90']:.3f}"],
            ["share_zero", f"{summary['life']['offspring_share_zero']:.3f}"],
        ]))

        md.append("\n## Maturity gating\n")
        md.append(_md_table([
            ["metric", "value"],
            ["matured_share", f"{summary['life']['matured_share']:.3f}"],
            ["matured_n", str(summary["life"]["matured_n"])],
        ]))
        om = summary["life"]["offspring_after_mature"]
        md.append(_md_table([
            ["offspring_after_mature stat", "value"],
            ["mean", f"{om['mean']:.3f}"],
            ["p10", f"{om['p10']:.3f}"],
            ["p50", f"{om['p50']:.3f}"],
            ["p90", f"{om['p90']:.3f}"],
            ["share_zero", f"{summary['life']['offspring_after_mature_share_zero']:.3f}"],
        ]))

        # phenotype-fitness
        pf = summary.get("pheno_fitness", {})
        keys = pf.get("keys", [])
        if keys:
            md.append("\n## Phenotype → fitness (correlations)\n")
            # show top 10 by abs corr vs R0_mature, fallback to R0
            corr = pf.get("corr", {})
            scores = []
            for k in keys:
                v = corr.get(k, {}).get("corr_vs_R0_mature", float("nan"))
                if not _isfinite(v):
                    v = corr.get(k, {}).get("corr_vs_R0", float("nan"))
                if _isfinite(v):
                    scores.append((k, v))
            scores.sort(key=lambda kv: abs(kv[1]), reverse=True)
            top = scores[:10]
            md.append(_md_table([["phenotype", "corr_vs_R0m_or_R0"]] + [[k, f"{v:+.3f}"] for k, v in top]))

            md.append("\n## Phenotype selection differential (top group vs all)\n")
            sd = pf.get("selection_differential", {})
            sd_items = [(k, sd.get(k, float("nan"))) for k in keys]
            sd_items = [(k, v) for k, v in sd_items if _isfinite(v)]
            sd_items.sort(key=lambda kv: abs(kv[1]), reverse=True)
            md.append(_md_table([["phenotype", "Δ mean(top) - mean(all)"]] + [[k, f"{v:+.4f}"] for k, v in sd_items[:10]]))

        # lineage
        md.append("\n## Lineage dominance\n")
        topR0 = summary["lineage"]["top_parents_R0"][:10]
        md.append(_md_table([["agent_id", "R0", "age", "birth_t"]] + [[str(r["agent_id"]), str(r["R0"]), f"{r['age']:.2f}", f"{r['birth_t']:.2f}"] for r in topR0]))
        topR0m = summary["lineage"]["top_parents_R0_mature"][:10]
        md.append(_md_table([["agent_id", "R0_mature", "age", "birth_t"]] + [[str(r["agent_id"]), str(r["R0_mature"]), f"{r['age']:.2f}", f"{r['birth_t']:.2f}"] for r in topR0m]))

        # cohorts
        md.append("\n## Birth cohorts (drift + cohort fitness)\n")
        md.append(f"- bin_width: {cohort_bin_w}\n")
        for b in cohorts["bins"]:
            md.append(f"\n### t ∈ [{b['t_lo']:.0f}, {b['t_hi']:.0f})  (n_births={b['n_births']})\n")
            cf = b["cohort_fitness"]
            md.append(_md_table([
                ["metric", "mean", "p10", "p50", "p90"],
                ["age", f"{cf['age']['mean']:.2f}", f"{cf['age']['p10']:.2f}", f"{cf['age']['p50']:.2f}", f"{cf['age']['p90']:.2f}"],
                ["R0", f"{cf['R0']['mean']:.2f}", f"{cf['R0']['p10']:.2f}", f"{cf['R0']['p50']:.2f}", f"{cf['R0']['p90']:.2f}"],
                ["R0_mature", f"{cf['R0_mature']['mean']:.2f}", f"{cf['R0_mature']['p10']:.2f}", f"{cf['R0_mature']['p50']:.2f}", f"{cf['R0_mature']['p90']:.2f}"],
            ]))
            md.append(f"- matured_share: {cf['matured_share']:.3f}\n")

                # ---- sample drift summary (NEW) ----
        samp = summary.get("sample", {})
        if samp:
            md.append("\n## Sample drift (cross-sectional longitudinal)\n")
            md.append(_md_table([
                ["metric", "value"],
                ["n_rows", str(samp.get("n_rows", ""))],
                ["n_unique_agents", str(samp.get("n_unique_agents", ""))],
                ["t_min", f"{samp.get('t_min', float('nan')):.2f}"],
                ["t_max", f"{samp.get('t_max', float('nan')):.2f}"],
            ]))

            drift = samp.get("drift_vs_time", {}) or {}
            items = [(k, (v or {}).get("slope_per_time", float("nan"))) for k, v in drift.items()]
            items = [(k, v) for k, v in items if _isfinite(v)]
            items.sort(key=lambda kv: abs(kv[1]), reverse=True)
            if items:
                md.append("\n### Top |slope| vs time\n")
                md.append(_md_table([["phenotype", "slope_per_time"]] + [[k, f"{v:+.6g}"] for k, v in items[:10]]))
                
        out_report_md.write_text("\n".join(md), encoding="utf-8")

    return summary, agent_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", default="life.jsonl", type=str)
    ap.add_argument("--pop", default="pop.jsonl", type=str)
    ap.add_argument("--world", default="world.jsonl", type=str)

    ap.add_argument("--out_summary", default="genopheno_stats.json", type=str)
    ap.add_argument("--out_agents", default="genopheno_agents.jsonl", type=str)

    # new outputs
    ap.add_argument("--out_report", default="genopheno_report.md", type=str)
    ap.add_argument("--out_birth_cohorts", default="genopheno_birth_cohorts.json", type=str)
    ap.add_argument("--out_lineages", default="genopheno_lineages.json", type=str)

    ap.add_argument("--cohort_bin_w", default=500.0, type=float)
    ap.add_argument("--top_quantile", default=0.80, type=float)

    ap.add_argument("--sample", default="sample.jsonl", type=str)
    ap.add_argument("--out_sample_stats", default="genopheno_sample_stats.json", type=str)

    args = ap.parse_args()

    life_fp = Path(args.life)
    pop_fp = Path(args.pop) if args.pop else None
    world_fp = Path(args.world) if args.world else None
    sample_fp = Path(args.sample) if args.sample else None

    summary, agent_rows = analyze(
        life_fp,
        pop_fp if pop_fp and pop_fp.exists() else None,
        world_fp if world_fp and world_fp.exists() else None,
        sample_fp=sample_fp if sample_fp and sample_fp.exists() else None,
        out_sample_stats_json=Path(args.out_sample_stats) if args.out_sample_stats else None,
        out_report_md=Path(args.out_report) if args.out_report else None,
        out_birth_cohorts_json=Path(args.out_birth_cohorts) if args.out_birth_cohorts else None,
        out_lineages_json=Path(args.out_lineages) if args.out_lineages else None,
        cohort_bin_w=float(args.cohort_bin_w),
        top_quantile=float(args.top_quantile),
    )

    Path(args.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with Path(args.out_agents).open("w", encoding="utf-8") as f:
        for row in agent_rows:
            f.write(json.dumps(row) + "\n")

    # concise console output
    print("== Genopheno postprocess ==")
    print(f"agents_seen: {summary['counts']['agents_seen']}")
    print(f"births/deaths/steps: {summary['counts']['births_life']} / {summary['counts']['deaths_life']} / {summary['counts']['steps_life']}")
    ls = summary["life"]["lifespan"]
    print(f"lifespan mean/p10/p50/p90: {ls['mean']:.3f} / {ls['p10']:.3f} / {ls['p50']:.3f} / {ls['p90']:.3f}")
    os_ = summary["life"]["offspring"]
    print(f"offspring mean/p10/p50/p90: {os_['mean']:.3f} / {os_['p10']:.3f} / {os_['p50']:.3f} / {os_['p90']:.3f}  share_zero={summary['life']['offspring_share_zero']:.3f}")
    print(f"matured_share={summary['life']['matured_share']:.3f}  matured_n={summary['life']['matured_n']}")
    print(f"wrote: {args.out_summary}")
    print(f"wrote: {args.out_agents}")
    print(f"wrote: {args.out_report}")
    print(f"wrote: {args.out_birth_cohorts}")
    print(f"wrote: {args.out_lineages}")
    print(f"wrote: {args.out_sample_stats}")


if __name__ == "__main__":
    main()