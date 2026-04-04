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

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Data model
# ============================================================

def _get_first(d: Dict[str, Any], *keys: str, default=float("nan")):
    for k in keys:
        if k in d:
            return d.get(k, default)
    return default


@dataclass
class PopSeries:
    t: List[float] = field(default_factory=list)
    pop: List[int] = field(default_factory=list)

    births_total: List[float] = field(default_factory=list)
    deaths_total: List[float] = field(default_factory=list)

    births_rate: List[float] = field(default_factory=list)
    deaths_rate: List[float] = field(default_factory=list)
    net_rate: List[float] = field(default_factory=list)

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

    # reserve fraction if present
    mean_R: List[float] = field(default_factory=list)

    window: int = 4000

    def reset(self) -> None:
        self.t.clear()
        self.pop.clear()

        self.births_total.clear()
        self.deaths_total.clear()
        self.births_rate.clear()
        self.deaths_rate.clear()
        self.net_rate.clear()

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

        self.mean_R.clear()

    def _trim(self) -> None:
        w = int(self.window)
        if w <= 0 or len(self.t) <= w:
            return

        def trim(xs: List[Any]) -> None:
            xs[:] = xs[-w:]

        for xs in [
            self.t, self.pop,
            self.births_total, self.deaths_total, self.births_rate, self.deaths_rate, self.net_rate,
            self.mean_D, self.median_D, self.p10_D, self.p25_D, self.p75_D, self.p90_D,
            self.mean_M, self.median_M, self.p10_M, self.p25_M, self.p75_M, self.p90_M,
            self.mean_E, self.median_E, self.p10_E, self.p25_E, self.p75_E, self.p90_E,
            self.mean_R,
        ]:
            trim(xs)

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

        births_tot = float(_get_first(s, "births", "births_total", default=float("nan")))
        deaths_tot = float(_get_first(s, "deaths", "deaths_total", default=float("nan")))

        self.t.append(tt)
        self.pop.append(pp)
        self.births_total.append(births_tot)
        self.deaths_total.append(deaths_tot)

        # rates from cumulative totals
        if len(self.t) < 2:
            self.births_rate.append(float("nan"))
            self.deaths_rate.append(float("nan"))
            self.net_rate.append(float("nan"))
        else:
            dt = self.t[-1] - self.t[-2]
            if dt > 1e-12 and np.isfinite(births_tot) and np.isfinite(deaths_tot):
                db = births_tot - self.births_total[-2]
                dd = deaths_tot - self.deaths_total[-2]
                br = db / dt
                dr = dd / dt
                nr = (db - dd) / dt
            else:
                br = dr = nr = float("nan")
            self.births_rate.append(float(br))
            self.deaths_rate.append(float(dr))
            self.net_rate.append(float(nr))

        # D
        self.mean_D.append(float(_get_first(s, "mean_D", default=float("nan"))))
        self.median_D.append(float(_get_first(s, "median_D", "med_D", default=float("nan"))))
        self.p10_D.append(float(_get_first(s, "p10_D", "pct10_D", default=float("nan"))))
        self.p25_D.append(float(_get_first(s, "p25_D", "pct25_D", default=float("nan"))))
        self.p75_D.append(float(_get_first(s, "p75_D", "pct75_D", default=float("nan"))))
        self.p90_D.append(float(_get_first(s, "p90_D", "pct90_D", default=float("nan"))))

        # M
        self.mean_M.append(float(_get_first(s, "mean_M", default=float("nan"))))
        self.median_M.append(float(_get_first(s, "median_M", "med_M", default=float("nan"))))
        self.p10_M.append(float(_get_first(s, "p10_M", "pct10_M", default=float("nan"))))
        self.p25_M.append(float(_get_first(s, "p25_M", "pct25_M", default=float("nan"))))
        self.p75_M.append(float(_get_first(s, "p75_M", "pct75_M", default=float("nan"))))
        self.p90_M.append(float(_get_first(s, "p90_M", "pct90_M", default=float("nan"))))

        # E
        self.mean_E.append(float(_get_first(s, "mean_E", default=float("nan"))))
        self.median_E.append(float(_get_first(s, "median_E", "med_E", default=float("nan"))))
        self.p10_E.append(float(_get_first(s, "p10_E", "pct10_E", default=float("nan"))))
        self.p25_E.append(float(_get_first(s, "p25_E", "pct25_E", default=float("nan"))))
        self.p75_E.append(float(_get_first(s, "p75_E", "pct75_E", default=float("nan"))))
        self.p90_E.append(float(_get_first(s, "p90_E", "pct90_E", default=float("nan"))))

        # reserve fraction
        self.mean_R.append(float(_get_first(s, "mean_R", default=float("nan"))))

        self._trim()
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
# Plot helpers
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
        f"birth_rate={_fmt(series.births_rate[i])}  death_rate={_fmt(series.deaths_rate[i])}\n"
        f"D med={_fmt(series.median_D[i])}  M med={_fmt(series.median_M[i])}\n"
        f"E med={_fmt(series.median_E[i])}  mean_R={_fmt(series.mean_R[i])}"
    )


def _plot_band(ax, t, p10, p25, p50, p75, p90, *, label: str, color: str):
    ax.plot(t, p50, label=f"{label} median", linewidth=2.2, color=color)
    ax.fill_between(t, p25, p75, alpha=0.18, color=color, label=f"{label} p25–p75")
    ax.fill_between(t, p10, p90, alpha=0.08, color=color, label=f"{label} p10–p90")


# ============================================================
# UI
# ============================================================

def run_ui_loop(fig, ax_demo, ax_state, ax_energy, series: PopSeries, args, Q: "queue.Queue[dict]"):
    last_redraw = 0.0
    redraw_min_dt = float(args.redraw_min_dt)
    max_items_per_tick = int(args.max_items_per_tick)

    def redraw() -> None:
        ax_demo.clear()
        ax_state.clear()
        ax_energy.clear()

        if not series.t:
            ax_demo.set_title("Population (waiting for events)")
            fig.canvas.draw_idle()
            return

        t = series.t

        # Panel 1: demography / flow
        ax_demo.plot(t, series.pop, label="pop", linewidth=2.4)
        ax_demo.plot(t, series.births_rate, label="births / dt", linestyle="--")
        ax_demo.plot(t, series.deaths_rate, label="deaths / dt", linestyle="--")
        ax_demo.plot(t, series.net_rate, label="net / dt", linestyle=":")
        ax_demo.set_ylabel("population / rate")
        ax_demo.set_title("Demography and turnover")
        ax_demo.legend(loc="upper left", fontsize="small")

        ax_demo.text(
            0.01,
            0.02,
            _status(series),
            transform=ax_demo.transAxes,
            fontsize="small",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", alpha=float(args.alpha_box)),
        )

        # Panel 2: body state
        _plot_band(
            ax_state, t,
            series.p10_D, series.p25_D, series.median_D, series.p75_D, series.p90_D,
            label="D", color="#d62728"
        )
        _plot_band(
            ax_state, t,
            series.p10_M, series.p25_M, series.median_M, series.p75_M, series.p90_M,
            label="M", color="#1f77b4"
        )
        ax_state.set_ylabel("state")
        ax_state.set_title("Damage and mass")
        ax_state.legend(loc="upper left", fontsize="small", ncol=2)

        # Panel 3: energy / reserve
        _plot_band(
            ax_energy, t,
            series.p10_E, series.p25_E, series.median_E, series.p75_E, series.p90_E,
            label="E", color="#2ca02c"
        )
        if any(np.isfinite(x) for x in series.mean_R):
            ax_energy.plot(t, series.mean_R, label="mean_R", color="#9467bd", linestyle="--", linewidth=2.0)

        ax_energy.set_ylabel("energy / reserve")
        ax_energy.set_xlabel("t")
        ax_energy.set_title("Energy and reserve")
        ax_energy.legend(loc="upper left", fontsize="small", ncol=2)

        fig.suptitle(f"Live population plot ({args.fp})")
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
    args = ap.parse_args()

    fig, (ax_demo, ax_state, ax_energy) = plt.subplots(3, 1, sharex=True, figsize=(11, 9))

    series = PopSeries(window=int(args.window))
    Q: "queue.Queue[dict]" = queue.Queue()

    th = start_tail_thread(str(args.fp), Q, poll_s=float(args.poll))
    tmr = run_ui_loop(fig, ax_demo, ax_state, ax_energy, series, args, Q)

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