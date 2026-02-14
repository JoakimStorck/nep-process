# live_world_plot.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import queue
import threading

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parsing helpers
# ============================================================

def _safe_load_json(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _get_world_summary(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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


# ============================================================
# Data model
# ============================================================

@dataclass
class WorldSeries:
    # time
    t: List[float] = field(default_factory=list)

    # densities (0..1)
    B_mean: List[float] = field(default_factory=list)
    F_mean: List[float] = field(default_factory=list)
    C_mean: List[float] = field(default_factory=list)

    # totals
    B_sum: List[float] = field(default_factory=list)
    F_sum: List[float] = field(default_factory=list)
    C_sum: List[float] = field(default_factory=list)

    # percentiles (B + C only; F percentiles not used)
    B_p10: List[float] = field(default_factory=list)
    B_p50: List[float] = field(default_factory=list)
    B_p90: List[float] = field(default_factory=list)

    C_p10: List[float] = field(default_factory=list)
    C_p50: List[float] = field(default_factory=list)
    C_p90: List[float] = field(default_factory=list)

    # hazard coverage
    hazard_frac_0p35: List[float] = field(default_factory=list)

    window: int = 4000  # 0 => keep all

    def reset(self) -> None:
        self.t.clear()
        self.B_mean.clear(); self.F_mean.clear(); self.C_mean.clear()
        self.B_sum.clear();  self.F_sum.clear();  self.C_sum.clear()
        self.B_p10.clear();  self.B_p50.clear();  self.B_p90.clear()
        self.C_p10.clear();  self.C_p50.clear();  self.C_p90.clear()
        self.hazard_frac_0p35.clear()

    def _trim(self) -> None:
        w = int(self.window)
        if w <= 0:
            return
        n = len(self.t)
        if n <= w:
            return
        sl = slice(n - w, n)
        self.t[:] = self.t[sl]
        self.B_mean[:] = self.B_mean[sl]; self.F_mean[:] = self.F_mean[sl]; self.C_mean[:] = self.C_mean[sl]
        self.B_sum[:]  = self.B_sum[sl];  self.F_sum[:]  = self.F_sum[sl];  self.C_sum[:]  = self.C_sum[sl]
        self.B_p10[:]  = self.B_p10[sl];  self.B_p50[:]  = self.B_p50[sl];  self.B_p90[:]  = self.B_p90[sl]
        self.C_p10[:]  = self.C_p10[sl];  self.C_p50[:]  = self.C_p50[sl];  self.C_p90[:]  = self.C_p90[sl]
        self.hazard_frac_0p35[:] = self.hazard_frac_0p35[sl]

    def append_summary(self, s: Dict[str, Any]) -> bool:
        try:
            tt = float(_f(s, "t", default=float("nan")))
        except Exception:
            return False

        self.t.append(tt)

        self.B_mean.append(_f(s, "B", "mean"))
        self.B_sum.append(_f(s, "B", "sum"))
        self.B_p10.append(_f(s, "B", "p10"))
        self.B_p50.append(_f(s, "B", "p50"))
        self.B_p90.append(_f(s, "B", "p90"))

        self.F_mean.append(_f(s, "F", "mean"))
        self.F_sum.append(_f(s, "F", "sum"))
        self.hazard_frac_0p35.append(_f(s, "F", "hazard_frac_0p35"))

        self.C_mean.append(_f(s, "C", "mean"))
        self.C_sum.append(_f(s, "C", "sum"))
        self.C_p10.append(_f(s, "C", "p10"))
        self.C_p50.append(_f(s, "C", "p50"))
        self.C_p90.append(_f(s, "C", "p90"))

        self._trim()
        return True


# ============================================================
# Tail thread (same as pop/pheno)
# ============================================================

def start_tail_thread(fp: str, q: "queue.Queue[dict]", *, poll_s: float = 0.25) -> threading.Thread:
    def worker() -> None:
        while not os.path.exists(fp):
            time.sleep(max(0.05, float(poll_s)))

        f = open(fp, "r", encoding="utf-8")
        try:
            inode = os.fstat(f.fileno()).st_ino

            def read_all_open() -> None:
                f.seek(0)
                for line in f:
                    obj = _safe_load_json(line)
                    if obj:
                        q.put(obj)
                q.put({"_event": "batch_done"})

            # initial batch
            read_all_open()

            while True:
                line = f.readline()
                if line:
                    obj = _safe_load_json(line)
                    if obj:
                        q.put(obj)
                    continue

                time.sleep(max(0.05, float(poll_s)))

                try:
                    st = os.stat(fp)
                except FileNotFoundError:
                    continue

                # rotation
                if st.st_ino != inode:
                    f.close()
                    f = open(fp, "r", encoding="utf-8")
                    inode = os.fstat(f.fileno()).st_ino
                    q.put({"_event": "reset"})
                    read_all_open()
                    continue

                # truncate
                cur = f.tell()
                if st.st_size < cur:
                    q.put({"_event": "reset"})
                    read_all_open()

        finally:
            try:
                f.close()
            except Exception:
                pass

    th = threading.Thread(target=worker, name="tail-thread", daemon=True)
    th.start()
    return th


# ============================================================
# Plot/UI
# ============================================================

def _fmt(x: float, nd: int = 3) -> str:
    return "nan" if (x != x) else f"{x:.{nd}f}"


def _status(series: WorldSeries, *, size: int) -> str:
    if not series.t:
        return "no world events yet"

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

    meanB_from_sum = (Bsum / cells) if (cells > 0 and np.isfinite(Bsum)) else float("nan")
    meanC_from_sum = (Csum / cells) if (cells > 0 and np.isfinite(Csum)) else float("nan")

    return "\n".join(
        [
            f"t={t:.2f}  n={len(series.t)}  cells={cells}",
            f"B_sum={_fmt(Bsum,2)}  B_mean={_fmt(Bmean,4)}  B_sum/cells={_fmt(meanB_from_sum,4)}",
            f"F_sum={_fmt(Fsum,2)}  F_mean={_fmt(Fmean,4)}  hazard_frac≥0.35={_fmt(hfrac,4)}",
            f"C_sum={_fmt(Csum,4)}  C_mean={_fmt(Cmean,6)}  C_sum/cells={_fmt(meanC_from_sum,6)}",
        ]
    )


def run_ui_loop(fig, ax_kpi, ax_h, txt, series: WorldSeries, args, Q: "queue.Queue[dict]"):
    last_redraw = 0.0
    redraw_min_dt = float(args.redraw_min_dt)
    max_items_per_tick = int(args.max_items_per_tick)
    show_percentiles = not bool(args.no_percentiles)

    def redraw() -> None:
        ax_kpi.clear()
        ax_h.clear()

        if not series.t:
            ax_kpi.set_title("World (waiting for events)")
            txt.set_text(_status(series, size=int(args.size)))
            fig.canvas.draw_idle()
            return

        t = series.t

        # KPI panel
        ax_kpi.plot(t, series.B_mean, label="B_mean")
        ax_kpi.plot(t, series.F_mean, label="F_mean")
        if show_percentiles:
            ax_kpi.plot(t, series.B_p10, linestyle=":", label="B_p10")
            ax_kpi.plot(t, series.B_p50, linestyle=":", label="B_p50")
            ax_kpi.plot(t, series.B_p90, linestyle=":", label="B_p90")

        ax_kpi.set_ylabel("mean density (0..1)")
        ax_kpi.set_title("World: KPI (densities)")
        ax_kpi.legend(loc="upper left", fontsize="small")

        # hazard panel
        ax_h.plot(t, series.hazard_frac_0p35, linestyle=":", label="hazard_frac≥0.35")
        ax_h.set_ylabel("hazard frac")
        ax_h.set_xlabel("t")
        ax_h.legend(loc="upper left", fontsize="small")

        txt.set_text(_status(series, size=int(args.size)))
        fig.suptitle(f"Live world plot ({args.fp})")
        fig.canvas.draw_idle()

    def on_timer() -> None:
        nonlocal last_redraw

        changed = False
        batch_done = False
        did_reset = False

        for _ in range(max_items_per_tick):
            try:
                obj = Q.get_nowait()
            except queue.Empty:
                break

            if obj.get("_event") == "reset":
                series.reset()
                did_reset = True
                continue
            if obj.get("_event") == "batch_done":
                batch_done = True
                continue

            s = _get_world_summary(obj)
            if s is None:
                continue

            if series.append_summary(s):
                changed = True

        now = time.time()
        if did_reset or batch_done:
            redraw()
            last_redraw = now
            return

        if changed and (now - last_redraw) >= redraw_min_dt:
            redraw()
            last_redraw = now

    redraw()
    tmr = fig.canvas.new_timer(interval=int(args.timer_ms))
    tmr.add_callback(on_timer)
    tmr.start()
    return tmr


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", default="world.jsonl")
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--no_percentiles", action="store_true")
    ap.add_argument("--alpha_box", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=4000)          # 0 => keep all
    ap.add_argument("--timer_ms", type=int, default=50)
    ap.add_argument("--redraw_min_dt", type=float, default=0.20)
    ap.add_argument("--max_items_per_tick", type=int, default=5000)
    args = ap.parse_args()

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.15)
    ax_kpi = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[1, 0], sharex=ax_kpi)

    txt = ax_kpi.text(
        0.01,
        0.02,
        "",
        transform=ax_kpi.transAxes,
        fontsize="small",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", alpha=float(args.alpha_box)),
    )

    series = WorldSeries(window=int(args.window))
    Q: "queue.Queue[dict]" = queue.Queue()

    th = start_tail_thread(str(args.fp), Q, poll_s=float(args.poll))
    tmr = run_ui_loop(fig, ax_kpi, ax_h, txt, series, args, Q)

    # håll referenser vid liv (macOS/backends kan annars GC:a timers)
    fig._tail_thread = th
    fig._live_timer = tmr
    fig._series = series

    plt.show()  # blockar => fönstret lever


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[live_world_plot] stopped.", flush=True)
        sys.exit(0)