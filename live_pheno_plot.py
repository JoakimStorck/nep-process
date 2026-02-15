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

    recompute_every: int = 25   # recompute percentiles every N samples
    window: int = 2000          # percentile window over last W samples; 0 => full history

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
# Tail thread: JSONL -> queue
#   - reads full file once (batch)
#   - then tails by readline() + poll on EOF
#   - on rotation/truncate: emits reset + re-batch
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

                # EOF: poll
                time.sleep(max(0.05, float(poll_s)))

                try:
                    st = os.stat(fp)
                except FileNotFoundError:
                    continue

                # rotation
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

def _status(series: PhenoSeries, keys: List[str]) -> str:
    if not series.t:
        return "no samples yet"
    i = -1
    t = series.t[i]
    aid = series.agent_id[i]
    n = len(series.t)
    lines = [f"t={t:.2f}  samples={n}  last_agent_id={aid}"]
    for k in keys[:8]:
        v = series.p50.get(k, [float("nan")])[-1]
        lines.append(f"{k}: p50={v:.3f}" if np.isfinite(v) else f"{k}: p50=nan")
    return "\n".join(lines)


def run_ui_loop(fig, ax, series: PhenoSeries, keys: List[str], args, Q: "queue.Queue[dict]"):
    last_redraw = 0.0
    redraw_min_dt = 0.20
    max_items_per_tick = 2000

    def pick_keys_from_ph(ph: Dict[str, Any]) -> List[str]:
        if args.keys.strip():
            return [k.strip() for k in args.keys.split(",") if k.strip()]
        kk: List[str] = []
        for k, v in ph.items():
            try:
                float(v)
                kk.append(k)
            except Exception:
                pass
        if int(args.max_keys) > 0:
            kk = kk[: int(args.max_keys)]
        return kk

    def redraw() -> None:
        ax.clear()

        if not series.t:
            ax.set_title("Live phenotype cross-section (waiting for samples)")
            fig.canvas.draw_idle()
            return

        t = series.t
        for k in keys:
            ax.plot(t, series.p50[k], label=f"{k} (p50)")
            ax.fill_between(t, series.p10[k], series.p90[k], alpha=0.10)

        ax.set_xlabel("t")
        ax.set_ylabel("phenotype (rolling percentiles over samples)")
        ax.legend(loc="upper left", fontsize="small", ncol=2)

        ax.text(
            0.01, 0.02, _status(series, keys),
            transform=ax.transAxes,
            fontsize="small",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", alpha=float(args.alpha_box)),
        )

        fig.suptitle("Live phenotype cross-section (sample.jsonl)")
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

            ev = obj.get("_event")
            if ev == "reset":
                series.reset()
                did_reset = True
                # Om du vill auto-picka keys igen efter reset: keys.clear()
                # keys.clear()
                continue
            if ev == "batch_done":
                batch_done = True
                continue

            if obj.get("event") != "sample":
                continue

            ph = obj.get("phenotype", {})
            if not isinstance(ph, dict):
                continue

            if not keys:
                keys[:] = pick_keys_from_ph(ph)
                for k in keys:
                    series.ensure_key(k)

            if not keys:
                continue

            series.append_sample(obj, keys)
            changed = True

        now = time.time()

        # Vid reset vill man ofta uppdatera direkt (även om batch inte är klar)
        if did_reset:
            redraw()
            last_redraw = now
            return

        # Efter batch: rita direkt en gång
        if batch_done:
            redraw()
            last_redraw = now
            return

        # Live: throttla redraw
        if changed and (now - last_redraw) >= redraw_min_dt:
            redraw()
            last_redraw = now

    # initial paint (så du ser att fönstret finns)
    redraw()

    # Viktigt: behåll referensen till timern (annars GC på macOS är vanligt)
    tmr = fig.canvas.new_timer(interval=50)  # 20 Hz tick
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
    ap.add_argument("--keys", default="")
    ap.add_argument("--max_keys", type=int, default=6)
    ap.add_argument("--alpha_box", type=float, default=1.0)
    ap.add_argument("--recompute_every", type=int, default=25)
    ap.add_argument("--window", type=int, default=2000)  # 0 => full history
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))

    series = PhenoSeries(
        recompute_every=int(args.recompute_every),
        window=int(args.window),
    )
    keys: List[str] = []
    Q: "queue.Queue[dict]" = queue.Queue()

    th = start_tail_thread(str(args.fp), Q, poll_s=float(args.poll))
    tmr = run_ui_loop(fig, ax, series, keys, args, Q)

    # håll starka referenser på fig (för att undvika GC)
    fig._tail_thread = th         # type: ignore[attr-defined]
    fig._live_timer = tmr         # type: ignore[attr-defined]
    fig._series = series          # type: ignore[attr-defined]
    fig._keys = keys              # type: ignore[attr-defined]
    fig._queue = Q                # type: ignore[attr-defined]

    # Set window title
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