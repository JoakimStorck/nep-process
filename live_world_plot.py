# live_world_plot.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt


# ============================================================
# Parsing helpers
# ============================================================

def _get_summary(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return obj['summary'] if event=world and summary exists; else None."""
    if obj.get("event") != "world":
        return None
    s = obj.get("summary")
    return s if isinstance(s, dict) else None


def _f(d: Any, *path: str, default: float = float("nan")) -> float:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return float(default)
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return float(default)


def _safe_load_json(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ============================================================
# Series (world only; no pop join here)
# ============================================================

@dataclass
class WorldSeries:
    # time
    t: list[float] = field(default_factory=list)

    # densities (0..1)
    B_mean: list[float] = field(default_factory=list)
    F_mean: list[float] = field(default_factory=list)
    C_mean: list[float] = field(default_factory=list)

    # totals
    B_sum: list[float] = field(default_factory=list)
    F_sum: list[float] = field(default_factory=list)
    C_sum: list[float] = field(default_factory=list)

    # percentiles (B + C only; F percentiles not used)
    B_p10: list[float] = field(default_factory=list)
    B_p50: list[float] = field(default_factory=list)
    B_p90: list[float] = field(default_factory=list)

    C_p10: list[float] = field(default_factory=list)
    C_p50: list[float] = field(default_factory=list)
    C_p90: list[float] = field(default_factory=list)

    # hazard coverage
    hazard_frac_0p35: list[float] = field(default_factory=list)

    def append(self, s: Dict[str, Any]) -> None:
        self.t.append(_f(s, "t", default=float("nan")))

        self.B_mean.append(_f(s, "B", "mean", default=float("nan")))
        self.B_sum.append(_f(s, "B", "sum", default=float("nan")))
        self.B_p10.append(_f(s, "B", "p10", default=float("nan")))
        self.B_p50.append(_f(s, "B", "p50", default=float("nan")))
        self.B_p90.append(_f(s, "B", "p90", default=float("nan")))

        self.F_mean.append(_f(s, "F", "mean", default=float("nan")))
        self.F_sum.append(_f(s, "F", "sum", default=float("nan")))
        self.hazard_frac_0p35.append(_f(s, "F", "hazard_frac_0p35", default=float("nan")))

        self.C_mean.append(_f(s, "C", "mean", default=float("nan")))
        self.C_sum.append(_f(s, "C", "sum", default=float("nan")))
        self.C_p10.append(_f(s, "C", "p10", default=float("nan")))
        self.C_p50.append(_f(s, "C", "p50", default=float("nan")))
        self.C_p90.append(_f(s, "C", "p90", default=float("nan")))


# ============================================================
# Robust tailer (read-all-at-start then tail)
#   - uses readline() only (no iterator protocol)
#   - handles rotation and truncate
# ============================================================

def _read_all(fp: str, series: WorldSeries) -> Tuple[int, int]:
    """
    Read whole file. Returns (inode, position).
    """
    if not os.path.exists(fp):
        return (0, 0)

    with open(fp, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            obj = _safe_load_json(line)
            if not obj:
                continue
            s = _get_summary(obj)
            if s is not None:
                series.append(s)
        inode = os.fstat(f.fileno()).st_ino
        pos = f.tell()
    return (inode, pos)


def _tail_loop(
    fp: str,
    series: WorldSeries,
    *,
    poll_s: float,
    on_line,
) -> None:
    """
    Tail file forever. Calls on_line() after each appended world event.
    """
    # wait until file exists
    while not os.path.exists(fp):
        time.sleep(max(0.05, float(poll_s)))

    f = open(fp, "r", encoding="utf-8")
    try:
        # start from end? no: read all once, then continue at EOF
        inode = os.fstat(f.fileno()).st_ino
        f.seek(0)
        for line in f.read().splitlines():
            obj = _safe_load_json(line)
            if not obj:
                continue
            s = _get_summary(obj)
            if s is not None:
                series.append(s)
        on_line()

        while True:
            line = f.readline()
            if line:
                obj = _safe_load_json(line)
                if obj:
                    s = _get_summary(obj)
                    if s is not None:
                        series.append(s)
                        on_line()
                continue

            time.sleep(max(0.05, float(poll_s)))

            # rotation / truncate checks
            try:
                st = os.stat(fp)
            except FileNotFoundError:
                continue

            # rotation: replaced inode
            if st.st_ino != inode:
                f.close()
                f = open(fp, "r", encoding="utf-8")
                inode = os.fstat(f.fileno()).st_ino
                series.__dict__.update(WorldSeries().__dict__)  # reset in-place
                for line2 in f.read().splitlines():
                    obj2 = _safe_load_json(line2)
                    if not obj2:
                        continue
                    s2 = _get_summary(obj2)
                    if s2 is not None:
                        series.append(s2)
                on_line()
                continue

            # truncate: file size smaller than current position
            cur = f.tell()
            if st.st_size < cur:
                f.seek(0)
                series.__dict__.update(WorldSeries().__dict__)  # reset
                for line2 in f.read().splitlines():
                    obj2 = _safe_load_json(line2)
                    if not obj2:
                        continue
                    s2 = _get_summary(obj2)
                    if s2 is not None:
                        series.append(s2)
                on_line()
    finally:
        try:
            f.close()
        except Exception:
            pass


# ============================================================
# Plotting
#   - NO twinx inside redraw (axes created once)
#   - KPI panel: B_mean, F_mean (left), C_mean (right) to make C visible
#   - Hazard frac gets its own small panel (no twinx)
#   - Status box includes sums + sanity mean-from-sum
# ============================================================

@dataclass
class PlotHandles:
    fig: Any
    ax_kpi: Any
    ax_c: Any          # right y-axis for C_mean (created once)
    ax_h: Any          # hazard frac panel
    txt: Any           # status text artist


def _make_plot(*, alpha_box: float = 1.0) -> PlotHandles:
    plt.ion()
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.15)

    ax_kpi = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[1, 0], sharex=ax_kpi)

    # C gets its own y-scale via twinx(), but created ONCE here.
    ax_c = ax_kpi.twinx()

    txt = ax_kpi.text(
        0.01, 0.02, "",
        transform=ax_kpi.transAxes,
        fontsize="small",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", alpha=float(alpha_box))
    )

    return PlotHandles(fig=fig, ax_kpi=ax_kpi, ax_c=ax_c, ax_h=ax_h, txt=txt)


def _fmt(x: float, nd: int = 3) -> str:
    return "nan" if (x != x) else f"{x:.{nd}f}"


def _status(series: WorldSeries, *, size: int) -> str:
    if not series.t:
        return ""
    i = -1
    t = series.t[i]
    cells = int(size) * int(size)

    Bsum = series.B_sum[i]
    Fsum = series.F_sum[i]
    Csum = series.C_sum[i]

    Bmean = series.B_mean[i]
    Fmean = series.F_mean[i]
    Cmean = series.C_mean[i]

    hfrac = series.hazard_frac_0p35[i]

    meanB_from_sum = (Bsum / cells) if (cells > 0 and Bsum == Bsum) else float("nan")
    meanC_from_sum = (Csum / cells) if (cells > 0 and Csum == Csum) else float("nan")

    lines = [
        f"t={t:.2f}  cells={cells}",
        f"B_sum={_fmt(Bsum,2)}  B_mean={_fmt(Bmean,4)}  B_sum/cells={_fmt(meanB_from_sum,4)}",
        f"F_sum={_fmt(Fsum,2)}  F_mean={_fmt(Fmean,4)}  hazard_frac≥0.35={_fmt(hfrac,4)}",
        f"C_sum={_fmt(Csum,4)}  C_mean={_fmt(Cmean,6)}  C_sum/cells={_fmt(meanC_from_sum,6)}",
    ]
    return "\n".join(lines)


def redraw(h: PlotHandles, series: WorldSeries, *, size: int, show_percentiles: bool = True) -> None:
    if not series.t:
        return

    t = series.t

    # --- clear axes but do NOT recreate twinx ---
    h.ax_kpi.clear()
    h.ax_c.clear()
    h.ax_h.clear()

    # --- KPI: B_mean, F_mean on left axis ---
    h.ax_kpi.plot(t, series.B_mean, label="B_mean")
    h.ax_kpi.plot(t, series.F_mean, label="F_mean")
    h.ax_kpi.set_ylabel("mean density (0..1)")
    h.ax_kpi.set_xlabel("")  # shared with bottom

    # --- KPI: C_mean on right axis (separate scale) ---
    #h.ax_c.plot(t, series.C_mean, label="C_mean", linestyle="--")
    #h.ax_c.set_ylabel("C_mean (carcass density)")

    # --- optional percentiles (B + C only) ---
    if show_percentiles:
        # B percentiles on left axis
        h.ax_kpi.plot(t, series.B_p10, linestyle=":", label="B_p10")
        h.ax_kpi.plot(t, series.B_p50, linestyle=":", label="B_p50")
        h.ax_kpi.plot(t, series.B_p90, linestyle=":", label="B_p90")
        # C percentiles on right axis
        #h.ax_c.plot(t, series.C_p10, linestyle=":", label="C_p10")
        #h.ax_c.plot(t, series.C_p50, linestyle=":", label="C_p50")
        #h.ax_c.plot(t, series.C_p90, linestyle=":", label="C_p90")

    # --- hazard frac in its own panel (no twinx) ---
    h.ax_h.plot(t, series.hazard_frac_0p35, linestyle=":", label="hazard_frac≥0.35")
    h.ax_h.set_ylabel("hazard frac")
    h.ax_h.set_xlabel("t")

    # --- legends: keep separate (cleaner) ---
    h.ax_kpi.legend(loc="upper left", fontsize="small")
    #h.ax_c.legend(loc="upper right", fontsize="small")
    h.ax_h.legend(loc="upper left", fontsize="small")

    # --- status box: update existing artist (no re-add) ---
    h.txt.set_text(_status(series, size=size))

    h.fig.suptitle("World: KPI (densities) + hazard coverage (panel) + status (totals)")
    h.fig.canvas.draw()
    h.fig.canvas.flush_events()


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", default="world.jsonl")
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--no_percentiles", action="store_true")
    ap.add_argument("--alpha_box", type=float, default=1.0)  # status bbox alpha
    args = ap.parse_args()

    series = WorldSeries()
    h = _make_plot(alpha_box=float(args.alpha_box))

    def on_line() -> None:
        redraw(h, series, size=int(args.size), show_percentiles=(not args.no_percentiles))
        plt.pause(0.001)

    _tail_loop(args.fp, series, poll_s=float(args.poll), on_line=on_line)


if __name__ == "__main__":
    main()