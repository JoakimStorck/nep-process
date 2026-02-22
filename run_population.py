# run_population.py
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from viewer_pygame import WorldViewer, ViewerConfig

from world import WorldParams
from agent import AgentParams
from population import Population, PopParams

from simlog.jsonl import JsonlWriter
from simlog.sinks import EventHub
from simlog.observers import StepLogger, PopLogger, LifeLogger, WorldLogger, SampleLogger

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--init_pop", type=int, default=12)
    ap.add_argument("--max_pop", type=int, default=256)

    # log files
    ap.add_argument("--step_log", type=str, default="steps.jsonl")
    ap.add_argument("--pop_log", type=str, default="pop.jsonl")
    ap.add_argument("--life_log", type=str, default="life.jsonl")
    ap.add_argument("--world_log", type=str, default="world.jsonl")
    ap.add_argument("--sample_log", type=str, default="sample.jsonl")

    # periods (sim-time seconds)
    ap.add_argument("--step_every", type=float, default=0.5)
    ap.add_argument("--pop_every", type=float, default=1.0)
    ap.add_argument("--world_every", type=float, default=2.0)
    ap.add_argument("--sample_every", type=float, default=1.0)

    # step tracking
    ap.add_argument("--track_id", type=int, default=-1, help="Agent id to track in steps.jsonl (-1 = no filter).")
    ap.add_argument("--track_i", type=int, default=0, help="Fallback: pick agent by index at t=0 to set track_id.")

    # stdout ticker
    ap.add_argument("--tick", type=float, default=2.0, help="Seconds between stdout status lines (sim-time).")
    ap.add_argument("--wall_tick", type=float, default=1.0, help="Seconds between stdout keepalive dots (wall-time).")

    # writer flushing
    ap.add_argument("--flush_steps", type=int, default=10)
    ap.add_argument("--flush_pop", type=int, default=1)
    ap.add_argument("--flush_life", type=int, default=1)
    ap.add_argument("--flush_world", type=int, default=1)
    ap.add_argument("--flush_sample", type=int, default=1)

    return ap.parse_args()


def _truncate(fp: str) -> None:
    open(fp, "w", encoding="utf-8").close()


if __name__ == "__main__":
    a = parse_args()

    # truncate logs
    _truncate(a.step_log)
    _truncate(a.pop_log)
    _truncate(a.life_log)
    _truncate(a.world_log)
    _truncate(a.sample_log)

    viewer = WorldViewer(ViewerConfig(
        title="NEP World (PyGame)",
        scale=10,
        fps_cap=0,
        render_every=2,   # render var 2:a simstep
        mode="CBF",
        draw_agents=True,
    ))
    
    WP = WorldParams(size=a.size, dt=0.02)
    AP = AgentParams(dt=WP.dt)
    PP = PopParams(init_pop=a.init_pop, max_pop=a.max_pop)

    print(
        f"START: T={a.T} dt={WP.dt} size={a.size} init_pop={a.init_pop} max_pop={a.max_pop} seed={a.seed}\n"
        f"  step_log={a.step_log} pop_log={a.pop_log} life_log={a.life_log} world_log={a.world_log}, sample_log={a.sample_log}",
        flush=True,
    )

    with (
        JsonlWriter(a.step_log, flush_every=int(a.flush_steps)) as w_steps,
        JsonlWriter(a.pop_log, flush_every=int(a.flush_pop)) as w_pop,
        JsonlWriter(a.life_log, flush_every=int(a.flush_life)) as w_life,
        JsonlWriter(a.world_log, flush_every=int(a.flush_world)) as w_world,
        JsonlWriter(a.sample_log, flush_every=int(a.flush_sample)) as w_sample,
    ):
        # create loggers
        step_logger = StepLogger(w=w_steps, every_s=float(a.step_every), track_id=None)
        pop_logger = PopLogger(w=w_pop, every_s=float(a.pop_every))
        life_logger = LifeLogger(w=w_life)          # no gating
        world_logger = WorldLogger(w=w_world, every_s=float(a.world_every))
        sample_logger = SampleLogger(w=w_sample, every_s=float(a.sample_every))

        hub = EventHub([step_logger, pop_logger, life_logger, world_logger, sample_logger])

        # build population with hub
        pop = Population(WP=WP, AP=AP, PP=PP, seed=a.seed, hub=hub)

        # establish track_id if requested via index (only once, cheap)
        track_id: Optional[int] = None
        if int(a.track_id) >= 0:
            track_id = int(a.track_id)
        else:
            i = int(a.track_i)
            if pop.agents:
                ag = pop.agents[i] if 0 <= i < len(pop.agents) else pop.agents[0]
                track_id = int(getattr(ag, "id", -1))

        # tell population which agent to emit step events for
        pop.track_step_id = track_id if (track_id is not None and track_id >= 0) else None

        if pop.track_step_id is not None:
            step_logger.set_track_id(pop.track_step_id)
            
        # apply to step logger (if supported)
        if track_id is not None and track_id >= 0:
            # StepLogger should implement set_track_id or have attr
            if hasattr(step_logger, "set_track_id"):
                step_logger.set_track_id(track_id)
            else:
                step_logger.track_id = track_id  # type: ignore[attr-defined]
            
        births_total = 0
        deaths_total = 0

        next_tick_t = 0.0
        last_wall = time.perf_counter()
        next_wall = last_wall + max(0.05, float(a.wall_tick))

        # -------------------------
        # SIM LOOP (runs until T, extinction, stall, or user quits)
        # -------------------------
        
        SIM_T_LIMIT = float(a.T)
        TICK_S = float(a.tick)
        WALL_TICK_S = float(a.wall_tick)
        
        stall_limit_s = 2.0          # wall-time stall trigger
        slow_step_limit_s = 0.2      # wall-time threshold for "slow" pop.step
        
        user_quit = False
        stop_reason: str | None = None
        
        # Stall detection: detect if pop.t stops advancing
        stall_wall0 = time.perf_counter()
        last_t = float(pop.t)
        
        while True:
            # --- stop conditions (checked before step) ---
            if pop.t >= SIM_T_LIMIT:
                stop_reason = f"TIME_LIMIT (t={pop.t:.3f} >= T={SIM_T_LIMIT:.3f})"
                break
        
            if len(pop.agents) == 0:
                stop_reason = f"EXTINCTION (t={pop.t:.3f})"
                break
        
            # --- simulate one step (wall-timed) ---
            t0 = time.perf_counter()
            b, d = pop.step()
            dtw = time.perf_counter() - t0
        
            if dtw > slow_step_limit_s:
                print(
                    f"\nSLOW pop.step: dt_wall={dtw:.3f}s  t={pop.t:.3f} pop={len(pop.agents)} b={b} d={d}",
                    flush=True,
                )
        
            # --- stall detection: sim-time not advancing ---
            t_now = float(pop.t)
            if t_now <= last_t + 1e-12:
                if time.perf_counter() - stall_wall0 > stall_limit_s:
                    stop_reason = f"STALL (t not advancing: t={t_now:.6f}, last_t={last_t:.6f})"
                    break
            else:
                last_t = t_now
                stall_wall0 = time.perf_counter()
        
            births_total += int(b)
            deaths_total += int(d)

            # --- viewer update (pumps events + render) ---
            if not viewer.update(pop, births_total=births_total, deaths_total=deaths_total):
                stop_reason = f"USER_QUIT (t={pop.t:.3f})"
                user_quit = True
                break            
        
            # --- periodic status line (sim-time cadence) ---
            if TICK_S > 0.0 and pop.t >= next_tick_t:
                next_tick_t = pop.t + TICK_S
                mean_E, mean_D, mean_M, mean_Ecap, mean_R = pop.mean_stats()
                print(
                    f"t={pop.t:8.2f}  pop={len(pop.agents):4d}  "
                    f"b={births_total:6d} d={deaths_total:6d}  "
                    f"mean_E={mean_E:.1f} mean_Ecap={mean_Ecap:.1f} mean_R={mean_R:.3f} "
                    f"mean_M={mean_M:.4f} mean_D={mean_D:.3f}",
                    flush=True,
                )
        
            # --- wall-time keepalive dot ---
            now = time.perf_counter()
            if now >= next_wall:
                next_wall = now + max(0.05, WALL_TICK_S)
                sys.stdout.write(".")
                sys.stdout.flush()
        
        if stop_reason is None:
            stop_reason = "UNKNOWN_STOP"
        
        # -------------------------
        # STOP MESSAGE (always)
        # -------------------------
        mean_E, mean_D, mean_M, mean_Ecap, mean_R = pop.mean_stats()
        print(
            "\n"
            f"STOP: {stop_reason}\n"
            f"  t={pop.t:.3f} pop={len(pop.agents)} births_total={births_total} deaths_total={deaths_total}\n"
            f"  mean_E={mean_E:.3f} mean_Ecap={mean_Ecap:.3f} mean_R={mean_R:.3f} "
            f"mean_M={mean_M:.4f} mean_D={mean_D:.3f}",
            flush=True,
        )
        
        # -------------------------
        # VIEWER LOOP (keep window alive after stop)
        # -------------------------
        while True:
            if not viewer.update(pop, births_total=births_total, deaths_total=deaths_total):
                user_quit = True
                break
            time.sleep(0.01)
        
        viewer.close()

