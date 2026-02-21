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


# ============================================================
# Data model
# ============================================================

@dataclass
class PopSeries:
    t: List[float] = field(default_factory=list)
    pop: List[int] = field(default_factory=list)
    mean_E: List[float] = field(default_factory=list)
    mean_D: List[float] = field(default_factory=list)

    window: int = 4000  # 0 => keep all

    def reset(self) -> None:
        self.t.clear()
        self.pop.clear()
        self.mean_E.clear()
        self.mean_D.clear()

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
        self.mean_E.append(float(s.get("mean_E", float("nan"))))
        self.mean_D.append(float(s.get("mean_D", float("nan"))))

        if int(self.window) > 0:
            w = int(self.window)
            if len(self.t) > w:
                self.t[:] = self.t[-w:]
                self.pop[:] = self.pop[-w:]
                self.mean_E[:] = self.mean_E[-w:]
                self.mean_D[:] = self.mean_D[-w:]

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

            # initial batch
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

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return th


# ============================================================
# Plot/UI
# ============================================================

def _status(series: PopSeries) -> str:
    if not series.t:
        return "no population events yet"
    i = -1
    return (
        f"t={series.t[i]:.2f}  n={len(series.t)}\n"
        f"pop={series.pop[i]}\n"
        f"mean_E={series.mean_E[i]:.3f}  mean_D={series.mean_D[i]:.3f}"
    )


def run_ui_loop(fig, ax_pop, ax_state, series: PopSeries, args, Q: "queue.Queue[dict]"):
    last_redraw = 0.0
    redraw_min_dt = float(args.redraw_min_dt)
    max_items_per_tick = int(args.max_items_per_tick)

    def redraw() -> None:
        ax_pop.clear()
        ax_state.clear()

        if not series.t:
            ax_pop.set_title("Population (waiting for events)")
            fig.canvas.draw_idle()
            return

        t = series.t

        # --- population ---
        ax_pop.plot(t, series.pop, label="pop")
        ax_pop.set_ylabel("population")
        ax_pop.set_title("Population")
        ax_pop.legend(loc="upper left", fontsize="small")

        # --- mean state ---
        ax_state.plot(t, series.mean_E, label="mean_E")
        ax_state.plot(t, series.mean_D, label="mean_D")
        ax_state.set_xlabel("t")
        ax_state.set_ylabel("mean")
        ax_state.set_title("Mean health")
        ax_state.legend(loc="upper left", fontsize="small")

        ax_pop.text(
            0.01,
            0.02,
            _status(series),
            transform=ax_pop.transAxes,
            fontsize="small",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", alpha=float(args.alpha_box)),
        )

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
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    ax_pop, ax_state = ax

    series = PopSeries(window=int(args.window))
    Q: "queue.Queue[dict]" = queue.Queue()

    th = start_tail_thread(str(args.fp), Q, poll_s=float(args.poll))
    tmr = run_ui_loop(fig, ax_pop, ax_state, series, args, Q)

    # keep refs alive
    fig._tail_thread = th
    fig._live_timer = tmr
    fig._series = series

    # Set window title
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