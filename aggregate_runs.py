import json, glob
import numpy as np

files = sorted(glob.glob("genopheno_stats_s*.json"))
runs = [json.load(open(f)) for f in files]

def collect(path):
    xs = []
    for r in runs:
        cur = r
        for p in path:
            cur = cur.get(p, None)
        if isinstance(cur, (int, float)):
            xs.append(cur)
    return np.array(xs, dtype=float)

summary = {
    "n_runs": len(runs),

    "lifespan_mean": collect(["life","lifespan","mean"]),
    "lifespan_p90": collect(["life","lifespan","p90"]),

    "matured_share": collect(["life","matured_share"]),
    "offspring_cond_matured_mean": collect(["life","offspring_cond_matured","mean"]),

    "pop_last": collect(["population","pop_last"]),
    "pop_max": collect(["population","pop_max"]),
}

def stat(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }

report = {k: stat(v) for k,v in summary.items() if k!="n_runs"}
report["n_runs"] = summary["n_runs"]

print(json.dumps(report, indent=2))