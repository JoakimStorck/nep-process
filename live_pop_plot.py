# live_pop_plot_threaded.py
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

import math


# ============================================================
# Data model
# ============================================================

PCTS = (10, 25, 75, 90)


def _get_first(d: Dict[str, Any], *keys: str, default=float("nan")):
    for k in keys:
        if k in d:
            return d.get(k, default)
    return default


@dataclass
class PopSeries:
    t: List[float] = field(default_factory=list)
    pop: List[int] = field(default_factory=list)

    # damage (D)
    mean_D: List[float] = field(default_factory=list)
    median_D: List[float] = field(default_factory=list)
    p10_D: List[float] = field(default_factory=list)
    p25_D: List[float] = field(default_factory=list)
    p75_D: List[float] = field(default_factory=list)
    p90_D: List[float] = field(default_factory=list)

    # mass (M)
    mean_M: List[float] = field(default_factory=list)
    median_M: List[float] = field(default_factory=list)
    p10_M: List[float] = field(default_factory=list)
    p25_M: List[float] = field(default_factory=list)
    p75_M: List[float] = field(default_factory=list)
    p90_M: List[float] = field(default_factory=list)

    # energy (E)
    mean_E: List[float] = field(default_factory=list)
    median_E: List[float] = field(default_factory=list)
    p10_E: List[float] = field(default_factory=list)
    p25_E: List[float] = field(default_factory=list)
    p75_E: List[float] = field(default_factory=list)
    p90_E: List[float] = field(default_factory=list)

    window: int = 4000  # 0 => keep all

    def reset(self) -> None:
        self.t.clear()
        self.pop.clear()

        self.mean_D.clear()
        self.median_D.clear()
        self.p10_D.clear()
        self.p25_D.clear()
        self.p75_D.clear()
        self.p90_D.clear()

        self.mean_M.clear()
        self.median_M.clear()
        self.p10_M.clear()
        self.p25_M.clear()
        self.p75_M.clear()
        self.p90_M.clear()

        self.mean_E.clear()
        self.median_E.clear()
        self.p10_E.clear()
        self.p25_E.clear()
        self.p75_E.clear()
        self.p90_E.clear()

    def _append_window(self) -> None:
        if int(self.window) <= 0:
            return
        w = int(self.window)
        if len(self.t) <= w:
            return

        def trim(xs: List[Any]) -> None:
            xs[:] = xs[-w:]

        trim(self.t)
        trim(self.pop)

        trim(self.mean_D); trim(self.median_D); trim(self.p10_D); trim(self.p25_D); trim(self.p75_D); trim(self.p90_D)
        trim(self.mean_M); trim(self.median_M); trim(self.p10_M); trim(self.p25_M); trim(self.p75_M); trim(self.p90_M)
        trim(self.mean_E); trim(self.median_E); trim(self.p10_E); trim(self.p25_E); trim(self.p75_E); trim(self.p90_E)

    def append_population_event(self, obj: Dict[str, Any]) -> bool:
        if obj.get("event") != "population":
            return False

        s = obj.get("summary")
        if not isinstance(s, dict):
            s = obj

        try:
            tt = float(s.get("t", float("nan")))
            pp = int(s.get("pop", 0))
        except Exception:
            return False

        self.t.append(tt)
        self.pop.append(pp)

        # ---- D ----
        self.mean_D.append(float(_get_first(s, "mean_D", default=float("nan"))))
        self.median_D.append(float(_get_first(s, "median_D", "med_D", default=float("nan"))))
        self.p10_D.append(float(_get_first(s, "p10_D", "pct10_D", default=float("nan"))))
        self.p25_D.append(float(_get_first(s, "p25_D", "pct25_D", default=float("nan"))))
        self.p75_D.append(float(_get_first(s, "p75_D", "pct75_D", default=float("nan"))))
        self.p90_D.append(float(_get_first(s, "p90_D", "pct90_D", default=float("nan"))))

        # ---- M ----
        self.mean_M.append(float(_get_first(s, "mean_M", default=float("nan"))))
        self.median_M.append(float(_get_first(s, "median_M", "med_M", default=float("nan"))))
        self.p10_M.append(float(_get_first(s, "p10_M", "pct10_M", default=float("nan"))))
        self.p25_M.append(float(_get_first(s, "p25_M", "pct25_M", default=float("nan"))))
        self.p75_M.append(float(_get_first(s, "p75_M", "pct75_M", default=float("nan"))))
        self.p90_M.append(float(_get_first(s, "p90_M", "pct90_M", default=float("nan"))))

        # ---- E ----
        self.mean_E.append(float(_get_first(s, "mean_E", default=float("nan"))))
        self.median_E.append(float(_get_first(s, "median_E", "med_E", default=float("nan"))))
        self.p10_E.append(float(_get_first(s, "p10_E", "pct10_E", default=float("nan"))))
        self.p25_E.append(float(_get_first(s, "p25_E", "pct25_E", default=float("nan"))))
        self.p75_E.append(float(_get_first(s, "p75_E", "pct75_E", default=float("nan"))))
        self.p90_E.append(float(_get_first(s, "p90_E", "pct90_E", default=float("nan"))))

        self._append_window()
        return True


# ============================================================
# Tail thread
# ============================================================

def start_tail_thread(fp: str, q: "queue.Queue[dict]", *, poll_s: float = 0.25) -> threading.Thread:
    def safe_load(line: str) -> Optional[Dict[str, Any]]:
        line = line.strip()
        if not line:
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    def worker() -> None:
        while not os.path.exists(fp):
            time.sleep(max(0.05, float(poll_s)))

        f = open(fp, "r", encoding="utf-8")
        try:
            inode = os.fstat(f.fileno()).st_ino

            def read_all_open() -> None:
                f.seek(0)
                for line in f:
                    obj = safe_load(line)
                    if obj:
                        q.put(obj)
                q.put({"_event": "batch_done"})

            read_all_open()

            while True:
                line = f.readline()
                if line:
                    obj = safe_load(line)
                    if obj:
                        q.put(obj)
                    continue

                time.sleep(max(0.05, float(poll_s)))

                try:
                    st = os.stat(fp)
                except FileNotFoundError:
                    continue

                if st.st_ino != inode:
                    f.close()
                    f = open(fp, "r", encoding="utf-8")
                    inode = os.fstat(f.fileno()).st_ino
                    q.put({"_event": "reset"})
                    read_all_open()
                    continue

                cur = f.tell()
                if st.st_size < cur:
                    q.put({"_event": "reset"})
                    read_all_open()

        finally:
            try:
                f.close()
            except Exception:
                pass

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return th


# ============================================================
# Plot/UI
# ============================================================

def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.3f}"


def _status(series: PopSeries) -> str:
    if not series.t:
        return "no population events yet"
    i = -1
    return (
        f"t={series.t[i]:.2f}  n={len(series.t)}\n"
        f"pop={series.pop[i]}\n"
        f"D mean/med={_fmt(series.mean_D[i])}/{_fmt(series.median_D[i])}\n"
        f"M mean/med={_fmt(series.mean_M[i])}/{_fmt(series.median_M[i])}\n"
        f"E mean/med={_fmt(series.mean_E[i])}/{_fmt(series.median_E[i])}"
    )


def _plot_stats(ax, t, mean, med, p10, p25, p75, p90, *, label_prefix: str, color: str, mean_lw=2.6, other_lw=1.1, ls="-"):
    ax.plot(t, mean, label=f"{label_prefix} mean", linewidth=mean_lw, color=color, linestyle=ls)
    ax.plot(t, med,  label=f"{label_prefix} median", linewidth=other_lw, color=color, linestyle=ls)

    ax.plot(t, p25, label=f"{label_prefix} p25", linewidth=other_lw, color=color, linestyle="--")
    ax.plot(t, p75, label=f"{label_prefix} p75", linewidth=other_lw, color=color, linestyle="--")

    ax.plot(t, p10, label=f"{label_prefix} p10", linewidth=other_lw, color=color, linestyle=":")
    ax.plot(t, p90, label=f"{label_prefix} p90", linewidth=other_lw, color=color, linestyle=":")


def run_ui_loop(fig, ax_top, ax_top2, ax_bot, ax_bot2, series: PopSeries, args, Q: "queue.Queue[dict]"):
    last_redraw = 0.0
    redraw_min_dt = float(args.redraw_min_dt)
    max_items_per_tick = int(args.max_items_per_tick)

    # user-requested: separate palettes for primary/secondary axes
    col_primary = getattr(args, "color_primary", "#1f77b4")    # blue-ish
    col_secondary = getattr(args, "color_secondary", "#d62728") # red-ish

    def redraw() -> None:
        ax_top.clear()
        ax_top2.clear()
        ax_bot.clear()
        ax_bot2.clear()

        if not series.t:
            ax_top.set_title("Population (waiting for events)")
            fig.canvas.draw_idle()
            return

        t = series.t

        # -------------------------
        # TOP: pop (primary) + damage (secondary)
        # -------------------------
        ax_top.plot(t, series.pop, label="pop", linewidth=2.6, color=col_primary)
        ax_top.set_ylabel("population")
        ax_top.set_title("Population + Damage")

        _plot_stats(
            ax_top2, t,
            series.mean_D, series.median_D,
            series.p10_D, series.p25_D, series.p75_D, series.p90_D,
            label_prefix="D", color=col_secondary,
        )
        ax_top2.set_ylabel("damage (D)")

        # legends (separate)
        ax_top.legend(loc="upper left", fontsize="small")
        ax_top2.legend(loc="upper right", fontsize="small")

        ax_top.text(
            0.01,
            0.02,
            _status(series),
            transform=ax_top.transAxes,
            fontsize="small",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", alpha=float(args.alpha_box)),
        )

        # -------------------------
        # BOTTOM: mass (primary) + energy (secondary)
        # -------------------------
        _plot_stats(
            ax_bot, t,
            series.mean_M, series.median_M,
            series.p10_M, series.p25_M, series.p75_M, series.p90_M,
            label_prefix="M", color=col_primary,
        )
        ax_bot.set_ylabel("mass (M)")

        _plot_stats(
            ax_bot2, t,
            series.mean_E, series.median_E,
            series.p10_E, series.p25_E, series.p75_E, series.p90_E,
            label_prefix="E", color=col_secondary,
        )
        ax_bot2.set_ylabel("energy (E)")

        ax_bot.set_xlabel("t")

        ax_bot.legend(loc="upper left", fontsize="small")
        ax_bot2.legend(loc="upper right", fontsize="small")

        fig.suptitle(f"Live pop plot ({args.fp})")
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

            if series.append_population_event(obj):
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
    ap.add_argument("--fp", default="pop.jsonl")
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--alpha_box", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=4000)
    ap.add_argument("--timer_ms", type=int, default=50)
    ap.add_argument("--redraw_min_dt", type=float, default=0.20)
    ap.add_argument("--max_items_per_tick", type=int, default=5000)

    # user: separate primary/secondary palettes
    ap.add_argument("--color_primary", default="#1f77b4")
    ap.add_argument("--color_secondary", default="#d62728")

    args = ap.parse_args()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(11, 7.5))
    ax_top, ax_bot = ax

    # secondary axes
    ax_top2 = ax_top.twinx()
    ax_bot2 = ax_bot.twinx()

    series = PopSeries(window=int(args.window))
    Q: "queue.Queue[dict]" = queue.Queue()

    th = start_tail_thread(str(args.fp), Q, poll_s=float(args.poll))
    tmr = run_ui_loop(fig, ax_top, ax_top2, ax_bot, ax_bot2, series, args, Q)

    # keep refs alive
    fig._tail_thread = th
    fig._live_timer = tmr
    fig._series = series

    try:
        fig.canvas.manager.set_window_title("NEP – Pop Plot")
    except Exception:
        pass

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[live_pop_plot] stopped.", flush=True)
        sys.exit(0)