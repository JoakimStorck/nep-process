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
# Data model
# ============================================================

@dataclass
class PhenoSeries:
    t: List[float] = field(default_factory=list)
    agent_id: List[int] = field(default_factory=list)

    raw: Dict[str, List[float]] = field(default_factory=dict)
    p10: Dict[str, List[float]] = field(default_factory=dict)
    p50: Dict[str, List[float]] = field(default_factory=dict)
    p90: Dict[str, List[float]] = field(default_factory=dict)

    recompute_every: int = 25
    window: int = 2000  # rolling window over last W samples; 0 => full history

    _n_seen: int = 0

    def reset(self) -> None:
        self.t.clear()
        self.agent_id.clear()
        self.raw.clear()
        self.p10.clear()
        self.p50.clear()
        self.p90.clear()
        self._n_seen = 0

    def ensure_key(self, k: str) -> None:
        self.raw.setdefault(k, [])
        self.p10.setdefault(k, [])
        self.p50.setdefault(k, [])
        self.p90.setdefault(k, [])

    def append_sample(self, obj: Dict[str, Any], keys: List[str]) -> None:
        tt = float(obj.get("t", float("nan")))
        aid = int(obj.get("agent_id", -1))

        ph = obj.get("phenotype", {})
        if not isinstance(ph, dict):
            return

        self.t.append(tt)
        self.agent_id.append(aid)
        self._n_seen += 1

        for k in keys:
            self.ensure_key(k)
            v = ph.get(k, float("nan"))
            try:
                self.raw[k].append(float(v))
            except Exception:
                self.raw[k].append(float("nan"))

        do_recompute = (self._n_seen % max(1, int(self.recompute_every)) == 0)

        for k in keys:
            if do_recompute or not self.p50[k]:
                data = self.raw[k]
                if int(self.window) > 0 and len(data) > int(self.window):
                    data = data[-int(self.window):]

                arr = np.asarray(data, dtype=np.float32)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    q10 = q50 = q90 = float("nan")
                else:
                    q10, q50, q90 = np.quantile(arr, [0.10, 0.50, 0.90]).tolist()

                self.p10[k].append(float(q10))
                self.p50[k].append(float(q50))
                self.p90[k].append(float(q90))
            else:
                self.p10[k].append(self.p10[k][-1])
                self.p50[k].append(self.p50[k][-1])
                self.p90[k].append(self.p90[k][-1])


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
                    try:
                        f.close()
                    except Exception:
                        pass
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
# Plot helpers
# ============================================================

DEFAULT_GROUPS = {
    "body_life": ["M_target", "child_M", "A_mature"],
    "reproduction": ["M_repro_min", "E_repro_min", "repro_rate"],
    "trophic": ["diet", "predation"],
}


def _parse_key_list(s: str) -> List[str]:
    return [k.strip() for k in s.split(",") if k.strip()]


def _status(series: PhenoSeries, groups: Dict[str, List[str]]) -> str:
    if not series.t:
        return "no samples yet"

    i = -1
    t = series.t[i]
    aid = series.agent_id[i]
    n = len(series.t)

    lines = [f"t={t:.2f}  samples={n}  last_agent_id={aid}"]

    focus_keys = (
        groups.get("body_life", [])[:2]
        + groups.get("reproduction", [])[:2]
        + groups.get("trophic", [])[:2]
    )

    seen = set()
    for k in focus_keys:
        if k in seen:
            continue
        seen.add(k)
        v = series.p50.get(k, [float("nan")])[-1]
        lines.append(f"{k}: p50={v:.3f}" if np.isfinite(v) else f"{k}: p50=nan")

    return "\n".join(lines)


def _available_numeric_keys(ph: Dict[str, Any]) -> List[str]:
    kk: List[str] = []
    for k, v in ph.items():
        try:
            float(v)
            kk.append(k)
        except Exception:
            pass
    return kk


def _filter_existing(keys: List[str], available: List[str]) -> List[str]:
    avail = set(available)
    return [k for k in keys if k in avail]


def _plot_group(ax, t: List[float], series: PhenoSeries, keys: List[str], title: str) -> None:
    ax.clear()

    if not t or not keys:
        ax.set_title(title)
        return

    for k in keys:
        ax.plot(t, series.p50[k], label=f"{k} (p50)")
        ax.fill_between(t, series.p10[k], series.p90[k], alpha=0.10)

    ax.set_title(title)
    ax.legend(loc="upper left", fontsize="small", ncol=2)


# ============================================================
# UI
# ============================================================

def run_ui_loop(fig, axes, series: PhenoSeries, groups: Dict[str, List[str]], args, Q: "queue.Queue[dict]"):
    ax_body, ax_repr, ax_trophic = axes
    last_redraw = 0.0
    redraw_min_dt = float(args.redraw_min_dt)
    max_items_per_tick = int(args.max_items_per_tick)

    def init_groups_from_ph(ph: Dict[str, Any]) -> None:
        available = _available_numeric_keys(ph)

        if args.body_keys.strip():
            groups["body_life"] = _filter_existing(_parse_key_list(args.body_keys), available)
        else:
            groups["body_life"] = _filter_existing(DEFAULT_GROUPS["body_life"], available)

        if args.repro_keys.strip():
            groups["reproduction"] = _filter_existing(_parse_key_list(args.repro_keys), available)
        else:
            groups["reproduction"] = _filter_existing(DEFAULT_GROUPS["reproduction"], available)

        if args.trophic_keys.strip():
            groups["trophic"] = _filter_existing(_parse_key_list(args.trophic_keys), available)
        else:
            groups["trophic"] = _filter_existing(DEFAULT_GROUPS["trophic"], available)

    def redraw() -> None:
        if not series.t:
            ax_body.clear()
            ax_repr.clear()
            ax_trophic.clear()
            ax_body.set_title("Live phenotype trends (waiting for samples)")
            fig.canvas.draw_idle()
            return

        t = series.t

        _plot_group(ax_body, t, series, groups.get("body_life", []), "Body / life history")
        _plot_group(ax_repr, t, series, groups.get("reproduction", []), "Reproduction strategy")
        _plot_group(ax_trophic, t, series, groups.get("trophic", []), "Trophic strategy")

        ax_trophic.set_xlabel("t")
        ax_body.set_ylabel("value")
        ax_repr.set_ylabel("value")
        ax_trophic.set_ylabel("value")

        ax_body.text(
            0.01,
            0.02,
            _status(series, groups),
            transform=ax_body.transAxes,
            fontsize="small",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", alpha=float(args.alpha_box)),
        )

        fig.suptitle("Live phenotype trends from sampled agents")
        fig.canvas.draw_idle()

    def on_timer() -> None:
        nonlocal last_redraw

        changed = False
        batch_done = False
        did_reset = False

        all_keys = (
            groups.get("body_life", [])
            + groups.get("reproduction", [])
            + groups.get("trophic", [])
        )

        for _ in range(max_items_per_tick):
            try:
                obj = Q.get_nowait()
            except queue.Empty:
                break

            ev = obj.get("_event")
            if ev == "reset":
                series.reset()
                groups["body_life"] = []
                groups["reproduction"] = []
                groups["trophic"] = []
                did_reset = True
                continue
            if ev == "batch_done":
                batch_done = True
                continue

            if obj.get("event") != "sample":
                continue

            ph = obj.get("phenotype", {})
            if not isinstance(ph, dict):
                continue

            if not any(groups.values()):
                init_groups_from_ph(ph)
                all_keys = (
                    groups.get("body_life", [])
                    + groups.get("reproduction", [])
                    + groups.get("trophic", [])
                )
                for k in all_keys:
                    series.ensure_key(k)

            if not all_keys:
                continue

            series.append_sample(obj, all_keys)
            changed = True

        now = time.time()

        if did_reset:
            redraw()
            last_redraw = now
            return

        if batch_done:
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
    ap.add_argument("--fp", default="sample.jsonl")
    ap.add_argument("--poll", type=float, default=0.25)

    ap.add_argument("--body_keys", default="")
    ap.add_argument("--repro_keys", default="")
    ap.add_argument("--trophic_keys", default="")

    ap.add_argument("--alpha_box", type=float, default=1.0)
    ap.add_argument("--recompute_every", type=int, default=25)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--timer_ms", type=int, default=50)
    ap.add_argument("--redraw_min_dt", type=float, default=0.20)
    ap.add_argument("--max_items_per_tick", type=int, default=2000)
    args = ap.parse_args()

    fig, (ax_body, ax_repr, ax_trophic) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True
    )

    series = PhenoSeries(
        recompute_every=int(args.recompute_every),
        window=int(args.window),
    )

    groups: Dict[str, List[str]] = {
        "body_life": [],
        "reproduction": [],
        "trophic": [],
    }

    Q: "queue.Queue[dict]" = queue.Queue()

    th = start_tail_thread(str(args.fp), Q, poll_s=float(args.poll))
    tmr = run_ui_loop(fig, (ax_body, ax_repr, ax_trophic), series, groups, args, Q)

    fig._tail_thread = th
    fig._live_timer = tmr
    fig._series = series
    fig._groups = groups
    fig._queue = Q

    try:
        fig.canvas.manager.set_window_title("NEP – Pheno Plot")
    except Exception:
        pass

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[live_pheno_plot] stopped.", flush=True)
        sys.exit(0)