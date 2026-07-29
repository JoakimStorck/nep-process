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
import matplotlib
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
    """
    Serier för floran och näringskretsloppet.

    Ritade tidigare `B` och `C`, alltså det kontinuerliga biomassfältet och
    kadaverfältet. `B` ersattes av diskret flora i Fas 2 och `C` av `detritus`
    i näringsstängningen, så panelerna visade fält som inte längre finns medan
    hela florapayloaden i loggen låg oanvänd.
    """

    t: List[float] = field(default_factory=list)

    # bestånd och massa
    flora_n: List[float] = field(default_factory=list)
    flora_mass: List[float] = field(default_factory=list)
    detritus_mass: List[float] = field(default_factory=list)
    established: List[float] = field(default_factory=list)

    # näringskretsloppet
    nut_free: List[float] = field(default_factory=list)
    nut_flora: List[float] = field(default_factory=list)
    nut_detritus: List[float] = field(default_factory=list)
    nut_added: List[float] = field(default_factory=list)
    nut_lost: List[float] = field(default_factory=list)

    # traitmedelvärden — selektionens signal
    growth_rate: List[float] = field(default_factory=list)
    adult_mass: List[float] = field(default_factory=list)
    dispersal_rate: List[float] = field(default_factory=list)
    temp_opt: List[float] = field(default_factory=list)
    temp_width: List[float] = field(default_factory=list)

    window: int = 4000

    _TRAITS = ("growth_rate", "adult_mass", "dispersal_rate", "temp_opt", "temp_width")

    def _trim(self) -> None:
        w = int(self.window)
        if w <= 0:
            return
        for name, val in list(self.__dict__.items()):
            if isinstance(val, list) and len(val) > w:
                del val[: len(val) - w]


    def _reset_if_time_went_backwards(self, tt: float) -> bool:
        """
        En ny körning i samma fil visar sig som att tiden hoppar bakåt.

        Serierna nollställs då i stället för att fortsätta, annars binder
        matplotlib ihop sista punkten i den gamla körningen med första i den
        nya och ritar en diagonal tvärs över hela bilden.
        """
        if not self.t or tt >= self.t[-1]:
            return False
        for val in self.__dict__.values():
            if isinstance(val, list):
                val.clear()
        return True

    def append_world_event(self, obj: Dict[str, Any]) -> bool:
        if obj.get("event") != "world":
            return False

        # Florafälten ligger på toppnivå; `summary` bär fältstatistiken.
        s = obj.get("summary")
        tt = float(obj.get("t", (s or {}).get("t", float("nan")) if isinstance(s, dict) else float("nan")))
        if not np.isfinite(tt):
            return False

        self._reset_if_time_went_backwards(tt)

        g = lambda k: float(obj.get(k, float("nan")))  # noqa: E731

        self.t.append(tt)
        self.flora_n.append(g("flora_n"))
        self.flora_mass.append(g("flora_mass_store"))
        self.detritus_mass.append(g("M_C"))
        self.established.append(g("flora_established"))

        self.nut_free.append(g("nutrient_free"))
        self.nut_flora.append(g("nutrient_in_flora"))
        self.nut_detritus.append(g("nutrient_in_detritus"))
        self.nut_added.append(g("nutrient_added"))
        self.nut_lost.append(g("nutrient_lost"))

        self.growth_rate.append(g("flora_mean_repro_alloc"))
        self.adult_mass.append(g("flora_mean_adult_mass"))
        self.dispersal_rate.append(g("flora_mean_apparatus"))
        self.temp_opt.append(g("flora_mean_temp_opt"))
        self.temp_width.append(g("flora_mean_temp_width"))

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
    tot = series.nut_free[i] + series.nut_flora[i] + series.nut_detritus[i]
    bal = series.nut_added[i] - series.nut_lost[i]
    drift = (tot - bal) / bal if (np.isfinite(bal) and abs(bal) > 1e-12) else float("nan")

    return "\n".join([
        f"t={series.t[i]:.1f}  n={len(series.t)}",
        f"flora={_fmt(series.flora_n[i],0)}  massa={_fmt(series.flora_mass[i],2)} kg"
        f"  detritus={_fmt(series.detritus_mass[i],3)} kg",
        f"näring: fri={_fmt(series.nut_free[i],5)}  flora={_fmt(series.nut_flora[i],5)}"
        f"  detritus={_fmt(series.nut_detritus[i],5)}",
        f"drift={drift:.2e} rel   (fauna ingår ej i world-loggen)",
    ])


def run_ui_loop(fig, ax_pool, ax_mix, ax_dist, txt, series: WorldSeries, args, Q: "queue.Queue[dict]"):
    last_redraw = 0.0
    redraw_min_dt = float(args.redraw_min_dt)
    max_items_per_tick = int(args.max_items_per_tick)

    def redraw() -> None:
        ax_pool.clear()
        ax_mix.clear()
        ax_dist.clear()

        if not series.t:
            ax_pool.set_title("Flora (waiting for events)")
            fig.canvas.draw_idle()
            return

        t = series.t

        # Panel 1: bestånd och massa. Antal och massa har olika skala och
        # rör sig olika — antalet kan stiga medan massan står still, vilket
        # betyder att beståndet fylls med groddplantor.
        ax_pool.plot(t, series.flora_n, label="antal individer", linewidth=2.2, color="tab:green")
        ax_pool.set_ylabel("antal")
        ax_pool.set_title("Floran: bestånd och massa")
        ax_pool.legend(loc="upper left", fontsize="small")

        ax_mass = ax_pool.twinx()
        ax_mass.plot(t, series.flora_mass, label="floramassa kg", color="tab:olive", linestyle="--")
        ax_mass.plot(t, series.detritus_mass, label="detritus kg", color="tab:brown", linestyle=":")
        ax_mass.set_ylabel("kg")
        ax_mass.legend(loc="upper right", fontsize="small")

        # Panel 2: näringskretsloppet. Summan ska följa tillfört minus
        # förlorat — avvikelsen är bokföringens drift.
        ax_mix.stackplot(
            t,
            series.nut_free, series.nut_flora, series.nut_detritus,
            labels=["fri", "i flora", "i detritus"],
            colors=["tab:blue", "tab:green", "tab:brown"],
            alpha=0.75,
        )
        ax_mix.set_ylabel("näring (kg)")
        ax_mix.set_title("Näringskretsloppet")
        ax_mix.legend(loc="upper left", fontsize="small", ncol=3)

        # Panel 3: traitmedelvärden, normerade mot sitt första värde.
        # Absolutvärdena har olika enheter och skala; det intressanta är
        # driften, alltså om selektionen flyttar fördelningen.
        for name, label, style in (
            ("growth_rate", "tillväxttakt", "-"),
            ("adult_mass", "vuxenmassa", "-"),
            ("dispersal_rate", "spridningstakt", "--"),
            ("temp_opt", "temp_opt", ":"),
            ("temp_width", "temp_bredd", ":"),
        ):
            v = np.asarray(getattr(series, name), dtype=float)
            base = next((x for x in v if np.isfinite(x) and abs(x) > 1e-12), float("nan"))
            if np.isfinite(base):
                ax_dist.plot(t, v / base, label=label, linestyle=style)

        ax_dist.axhline(1.0, color="0.7", linewidth=0.8)
        ax_dist.set_ylabel("relativt startvärde")
        ax_dist.set_xlabel("t")
        ax_dist.set_title("Floras traitmedelvärden — selektionens drift")
        ax_dist.legend(loc="upper left", fontsize="small", ncol=3)

        txt.set_text(_status(series, size=int(args.size)))
        fig.canvas.draw_idle()

    def on_timer() -> None:
        nonlocal last_redraw
        changed = False
        n = 0
        while n < max_items_per_tick:
            try:
                obj = Q.get_nowait()
            except queue.Empty:
                break
            n += 1
            if series.append_world_event(obj):
                changed = True

        now = time.time()
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

def _backend_is_interactive(name: str) -> bool:
    """Kan det här backendet öppna ett fönster?"""
    try:  # matplotlib >= 3.9
        from matplotlib.backends import backend_registry, BackendFilter
        names = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    except Exception:
        names = getattr(matplotlib.rcsetup, "interactive_bk", [])
    return str(name).lower() in {str(n).lower() for n in names}


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
    ap.add_argument("--save", type=str, default=None,
                    help="rita en gång ur befintlig fil och spara som bild i "
                         "stället för att öppna fönster; fungerar utan GUI-backend")
    args = ap.parse_args()

    backend = matplotlib.get_backend()
    if not args.save and not _backend_is_interactive(backend):
        print(
            f"[live_world_plot] matplotlib använder backend '{backend}', som inte kan\n"
            f"                  öppna fönster. Inget skulle visas.\n"
            f"\n"
            f"  Installera ett GUI-backend i din venv:   pip install PyQt5\n"
            f"  Eller spara en bild i stället:            python live_world_plot.py --fp {args.fp} --save world.png\n",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

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

    if args.save:
        n = 0
        try:
            with open(str(args.fp), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = _safe_load_json(line)
                    if obj is not None and series.append_world_event(obj):
                        n += 1
        except FileNotFoundError:
            print(f"[live_world_plot] hittar inte {args.fp}", file=sys.stderr, flush=True)
            sys.exit(1)

        run_ui_loop(fig, ax_pool, ax_mix, ax_dist, txt, series, args, Q)
        fig.savefig(str(args.save), dpi=110, bbox_inches="tight")
        print(f"[live_world_plot] {n} world-händelser -> {args.save}", flush=True)
        return

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