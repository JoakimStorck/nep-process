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

    # means
    B_mean: List[float] = field(default_factory=list)
    C_mean: List[float] = field(default_factory=list)

    # totals
    B_sum: List[float] = field(default_factory=list)
    C_sum: List[float] = field(default_factory=list)
    BC_sum: List[float] = field(default_factory=list)

    # mix
    carrion_share: List[float] = field(default_factory=list)

    # percentiles
    B_p10: List[float] = field(default_factory=list)
    B_p50: List[float] = field(default_factory=list)
    B_p90: List[float] = field(default_factory=list)

    C_p10: List[float] = field(default_factory=list)
    C_p50: List[float] = field(default_factory=list)
    C_p90: List[float] = field(default_factory=list)

    window: int = 4000  # 0 => keep all

    def reset(self) -> None:
        self.t.clear()
        self.B_mean.clear()
        self.C_mean.clear()
        self.B_sum.clear()
        self.C_sum.clear()
        self.BC_sum.clear()
        self.carrion_share.clear()
        self.B_p10.clear()
        self.B_p50.clear()
        self.B_p90.clear()
        self.C_p10.clear()
        self.C_p50.clear()
        self.C_p90.clear()

    def _trim(self) -> None:
        w = int(self.window)
        if w <= 0:
            return
        n = len(self.t)
        if n <= w:
            return
        sl = slice(n - w, n)
        self.t[:] = self.t[sl]
        self.B_mean[:] = self.B_mean[sl]
        self.C_mean[:] = self.C_mean[sl]
        self.B_sum[:] = self.B_sum[sl]
        self.C_sum[:] = self.C_sum[sl]
        self.BC_sum[:] = self.BC_sum[sl]
        self.carrion_share[:] = self.carrion_share[sl]
        self.B_p10[:] = self.B_p10[sl]
        self.B_p50[:] = self.B_p50[sl]
        self.B_p90[:] = self.B_p90[sl]
        self.C_p10[:] = self.C_p10[sl]
        self.C_p50[:] = self.C_p50[sl]
        self.C_p90[:] = self.C_p90[sl]

    def append_summary(self, s: Dict[str, Any]) -> bool:
        try:
            tt = float(_f(s, "t", default=float("nan")))
        except Exception:
            return False

        B_mean = _f(s, "B", "mean")
        B_sum = _f(s, "B", "sum")
        B_p10 = _f(s, "B", "p10")
        B_p50 = _f(s, "B", "p50")
        B_p90 = _f(s, "B", "p90")

        C_mean = _f(s, "C", "mean")
        C_sum = _f(s, "C", "sum")
        C_p10 = _f(s, "C", "p10")
        C_p50 = _f(s, "C", "p50")
        C_p90 = _f(s, "C", "p90")

        BC_sum = B_sum + C_sum
        carrion_share = (C_sum / BC_sum) if np.isfinite(BC_sum) and BC_sum > 1e-12 else 0.0

        self.t.append(tt)

        self.B_mean.append(B_mean)
        self.C_mean.append(C_mean)

        self.B_sum.append(B_sum)
        self.C_sum.append(C_sum)
        self.BC_sum.append(BC_sum)

        self.carrion_share.append(carrion_share)

        self.B_p10.append(B_p10)
        self.B_p50.append(B_p50)
        self.B_p90.append(B_p90)

        self.C_p10.append(C_p10)
        self.C_p50.append(C_p50)
        self.C_p90.append(C_p90)

        self._trim()
        return True


# ============================================================
# Tail thread
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
    Csum = series.C_sum[i]
    BCsum = series.BC_sum[i]

    Bmean = series.B_mean[i]
    Cmean = series.C_mean[i]
    carrion_share = series.carrion_share[i]

    meanB_from_sum = (Bsum / cells) if (cells > 0 and np.isfinite(Bsum)) else float("nan")
    meanC_from_sum = (Csum / cells) if (cells > 0 and np.isfinite(Csum)) else float("nan")

    return "\n".join(
        [
            f"t={t:.2f}  n={len(series.t)}  cells={cells}",
            f"B_sum={_fmt(Bsum,2)}  B_mean={_fmt(Bmean,6)}  B_sum/cells={_fmt(meanB_from_sum,6)}",
            f"C_sum={_fmt(Csum,2)}  C_mean={_fmt(Cmean,6)}  C_sum/cells={_fmt(meanC_from_sum,6)}",
            f"B+C={_fmt(BCsum,2)}  carrion_share={_fmt(carrion_share,4)}",
        ]
    )


def run_ui_loop(fig, ax_pool, ax_mix, ax_dist, txt, series: WorldSeries, args, Q: "queue.Queue[dict]"):
    last_redraw = 0.0
    redraw_min_dt = float(args.redraw_min_dt)
    max_items_per_tick = int(args.max_items_per_tick)
    show_percentiles = not bool(args.no_percentiles)

    def redraw() -> None:
        ax_pool.clear()
        ax_mix.clear()
        ax_dist.clear()

        if not series.t:
            ax_pool.set_title("World (waiting for events)")
            txt.set_text(_status(series, size=int(args.size)))
            fig.canvas.draw_idle()
            return

        t = series.t

        # Panel 1: resource pools
        ax_pool.plot(t, series.B_sum, label="B_sum")
        ax_pool.plot(t, series.C_sum, label="C_sum")
        ax_pool.plot(t, series.BC_sum, label="B_sum + C_sum", linestyle="--")
        ax_pool.set_ylabel("total mass")
        ax_pool.set_title("World resource pools")
        ax_pool.legend(loc="upper left", fontsize="small")

        # Panel 2: ecological mix
        ax_mix.plot(t, series.carrion_share, label="carrion_share = C / (B + C)")
        ax_mix.set_ylabel("share")
        ax_mix.set_ylim(0.0, 1.0)
        ax_mix.set_title("Ecological mix")
        ax_mix.legend(loc="upper left", fontsize="small")

        # Panel 3: field distributions / means
        ax_dist.plot(t, series.B_mean, label="B_mean")
        ax_dist.plot(t, series.C_mean, label="C_mean")

        if show_percentiles:
            ax_dist.plot(t, series.B_p10, linestyle=":", label="B_p10")
            ax_dist.plot(t, series.B_p50, linestyle=":", label="B_p50")
            ax_dist.plot(t, series.B_p90, linestyle=":", label="B_p90")
            ax_dist.plot(t, series.C_p10, linestyle="--", label="C_p10")
            ax_dist.plot(t, series.C_p50, linestyle="--", label="C_p50")
            ax_dist.plot(t, series.C_p90, linestyle="--", label="C_p90")

        ax_dist.set_ylabel("density")
        ax_dist.set_xlabel("t")
        ax_dist.set_title("Field means and percentiles")
        ax_dist.legend(loc="upper left", fontsize="small", ncol=2)

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

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 2.5], hspace=0.22)
    ax_pool = fig.add_subplot(gs[0, 0])
    ax_mix = fig.add_subplot(gs[1, 0], sharex=ax_pool)
    ax_dist = fig.add_subplot(gs[2, 0], sharex=ax_pool)

    txt = ax_pool.text(
        0.01,
        0.02,
        "",
        transform=ax_pool.transAxes,
        fontsize="small",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", alpha=float(args.alpha_box)),
    )

    series = WorldSeries(window=int(args.window))
    Q: "queue.Queue[dict]" = queue.Queue()

    th = start_tail_thread(str(args.fp), Q, poll_s=float(args.poll))
    tmr = run_ui_loop(fig, ax_pool, ax_mix, ax_dist, txt, series, args, Q)

    fig._tail_thread = th
    fig._live_timer = tmr
    fig._series = series

    try:
        fig.canvas.manager.set_window_title("NEP – World Plot")
    except Exception:
        pass

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[live_world_plot] stopped.", flush=True)
        sys.exit(0)