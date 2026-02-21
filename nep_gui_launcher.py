from __future__ import annotations

import os
import sys
import time
import signal
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pygame
import json


# =========================
# Small UI toolkit (PyGame)
# =========================

Color = Tuple[int, int, int]


def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x


@dataclass
class Theme:
    bg: Color = (20, 20, 24)
    panel: Color = (28, 28, 34)
    panel2: Color = (34, 34, 42)
    border: Color = (70, 70, 85)
    text: Color = (235, 235, 235)
    muted: Color = (170, 170, 180)
    accent: Color = (90, 170, 255)
    danger: Color = (255, 100, 100)
    ok: Color = (120, 220, 140)


class Widget:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, th: Theme) -> None:
        raise NotImplementedError

    def handle_event(self, ev: pygame.event.Event) -> bool:
        return False


class Button(Widget):
    def __init__(self, rect: pygame.Rect, label: str, *, kind: str = "normal"):
        super().__init__(rect)
        self.label = label
        self.kind = kind
        self._down = False
        self.clicked = False

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, th: Theme) -> None:
        bg = th.panel2
        if self.kind == "danger":
            bg = (60, 25, 25)
        elif self.kind == "ok":
            bg = (25, 55, 30)

        mx, my = pygame.mouse.get_pos()
        hover = self.rect.collidepoint(mx, my)
        shade = 1.10 if hover else 1.00
        col = tuple(int(clamp(c * shade, 0, 255)) for c in bg)

        pygame.draw.rect(surf, col, self.rect, border_radius=8)
        pygame.draw.rect(surf, th.border, self.rect, 1, border_radius=8)

        txt = font.render(self.label, True, th.text)
        surf.blit(
            txt,
            (self.rect.centerx - txt.get_width() // 2, self.rect.centery - txt.get_height() // 2),
        )

    def handle_event(self, ev: pygame.event.Event) -> bool:
        self.clicked = False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self._down = True
                return True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            was_down = self._down
            self._down = False
            if was_down and self.rect.collidepoint(ev.pos):
                self.clicked = True
                return True
        return False


class Checkbox(Widget):
    def __init__(self, rect: pygame.Rect, label: str, value: bool = False):
        super().__init__(rect)
        self.label = label
        self.value = bool(value)

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, th: Theme) -> None:
        box = pygame.Rect(self.rect.x, self.rect.y, 22, 22)
        pygame.draw.rect(surf, th.panel2, box, border_radius=4)
        pygame.draw.rect(surf, th.border, box, 1, border_radius=4)
        if self.value:
            pygame.draw.rect(surf, th.accent, box.inflate(-6, -6), border_radius=3)

        txt = font.render(self.label, True, th.text)
        surf.blit(txt, (box.right + 10, box.y + 2))

    def handle_event(self, ev: pygame.event.Event) -> bool:
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.value = not self.value
                return True
        return False


class TextField(Widget):
    """
    Minimal text field:
      - click to focus
      - type (TEXTINPUT)
      - Backspace
      - Enter commits (no-op)
      - Esc unfocus
    """

    def __init__(self, rect: pygame.Rect, label: str, text: str):
        super().__init__(rect)
        self.label = label
        self.text = text
        self.focus = False

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, th: Theme) -> None:
        lab = font.render(self.label, True, th.muted)
        surf.blit(lab, (self.rect.x, self.rect.y - 18))

        bg = th.panel2 if not self.focus else (40, 40, 54)
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        pygame.draw.rect(surf, th.accent if self.focus else th.border, self.rect, 1, border_radius=8)

        txt = font.render(self.text, True, th.text)
        tx = self.rect.x + 10
        ty = self.rect.y + (self.rect.height - txt.get_height()) // 2
        surf.blit(txt, (tx, ty))

        if self.focus:
            now = time.time()
            if int(now * 2) % 2 == 0:
                caret_x = tx + txt.get_width() + 2
                pygame.draw.line(surf, th.text, (caret_x, self.rect.y + 6), (caret_x, self.rect.bottom - 6), 1)

    def handle_event(self, ev: pygame.event.Event) -> bool:
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.focus = self.rect.collidepoint(ev.pos)
            return self.focus

        if not self.focus:
            return False

        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.focus = False
                return True
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return True
            if ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                return True
            return False

        if ev.type == pygame.TEXTINPUT:
            ch = ev.text
            if not ch:
                return False

            # allow numeric tokens: digits . - + e E
            for c in ch:
                if not (c.isdigit() or c in ".-+eE"):
                    return True  # consume but ignore

            self.text += ch
            return True

        return False


# =========================
# Simulation launcher logic
# =========================

def truncate(fp: str) -> None:
    open(fp, "w", encoding="utf-8").close()


def parse_int(s: str, default: int) -> int:
    try:
        return int(float(s.strip()))
    except Exception:
        return default


def parse_float(s: str, default: float) -> float:
    try:
        return float(s.strip())
    except Exception:
        return default


@dataclass
class RunCfg:
    T: float = 2000.0
    seed: int = 1
    size: int = 64
    init_pop: int = 12
    max_pop: int = 256

    step_log: str = "steps.jsonl"
    pop_log: str = "pop.jsonl"
    life_log: str = "life.jsonl"
    world_log: str = "world.jsonl"
    sample_log: str = "sample.jsonl"

    show_pop_plot: bool = True
    show_world_plot: bool = True
    show_pheno_plot: bool = True


class Launcher:
    def __init__(self, cfg: RunCfg):
        self.cfg = cfg
        self.sim_proc: Optional[subprocess.Popen] = None
        self.plot_procs: List[subprocess.Popen] = []
        self.paused = False

    @property
    def running(self) -> bool:
        return self.sim_proc is not None and (self.sim_proc.poll() is None)

    def reset_logs(self) -> None:
        for fp in [self.cfg.step_log, self.cfg.pop_log, self.cfg.life_log, self.cfg.world_log, self.cfg.sample_log]:
            truncate(fp)

    def _spawn_plots_with_env(self, env: dict) -> None:
        self.plot_procs.clear()
        py = sys.executable or "python"

        env2 = dict(env)
        env2.setdefault("MPLBACKEND", "TkAgg")

        if self.cfg.show_pop_plot:
            self.plot_procs.append(
                subprocess.Popen([py, "live_pop_plot.py", "--fp", self.cfg.pop_log], env=env2, start_new_session=True)
            )
        if self.cfg.show_world_plot:
            self.plot_procs.append(
                subprocess.Popen(
                    [py, "live_world_plot.py", "--fp", self.cfg.world_log, "--size", str(self.cfg.size)],
                    env=env2,
                    start_new_session=True,
                )
            )
        if self.cfg.show_pheno_plot:
            self.plot_procs.append(
                subprocess.Popen([py, "live_pheno_plot.py", "--fp", self.cfg.sample_log], env=env2, start_new_session=True)
            )

    def _terminate_proc(self, p: subprocess.Popen, *, timeout_s: float = 0.5) -> None:
        try:
            if p.poll() is not None:
                return
        except Exception:
            return

        try:
            p.terminate()
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                if p.poll() is not None:
                    try:
                        p.wait(timeout=0.0)
                    except Exception:
                        pass
                    return
            except Exception:
                return
            time.sleep(0.05)

        try:
            p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=timeout_s)
        except Exception:
            pass

    def _kill_plots(self) -> None:
        for p in self.plot_procs:
            try:
                self._terminate_proc(p, timeout_s=0.75)
            except Exception:
                pass
        self.plot_procs.clear()

    def start(self) -> None:
        if self.running:
            return

        self.reset_logs()

        py = sys.executable or "python"
        args = [
            py, "run_population.py",
            "--T", str(self.cfg.T),
            "--seed", str(self.cfg.seed),
            "--size", str(self.cfg.size),
            "--init_pop", str(self.cfg.init_pop),
            "--max_pop", str(self.cfg.max_pop),
            "--step_log", self.cfg.step_log,
            "--pop_log", self.cfg.pop_log,
            "--life_log", self.cfg.life_log,
            "--world_log", self.cfg.world_log,
            "--sample_log", self.cfg.sample_log,
        ]

        env = os.environ.copy()
        env.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

        # start sim first
        self.sim_proc = subprocess.Popen(args, env=env, start_new_session=True)
        self.paused = False

        time.sleep(0.25)

        # then plots
        self._spawn_plots_with_env(env)

    def stop(self) -> None:
        if self.sim_proc is not None:
            try:
                self._terminate_proc(self.sim_proc, timeout_s=0.75)
            except Exception:
                pass
            self.sim_proc = None

        self.paused = False
        self._kill_plots()

    def pause(self) -> None:
        if not self.running or self.paused:
            return
        if os.name != "posix" or self.sim_proc is None:
            return
        try:
            os.kill(self.sim_proc.pid, signal.SIGSTOP)
            self.paused = True
        except ProcessLookupError:
            self.sim_proc = None
            self.paused = False
        except Exception:
            pass

    def resume(self) -> None:
        if not self.running or not self.paused:
            return
        if os.name != "posix" or self.sim_proc is None:
            return
        try:
            os.kill(self.sim_proc.pid, signal.SIGCONT)
            self.paused = False
        except ProcessLookupError:
            self.sim_proc = None
            self.paused = False
        except Exception:
            pass


# =========================
# GUI
# =========================

def draw_header(surf: pygame.Surface, font: pygame.font.Font, th: Theme, title: str, status: str) -> None:
    surf.fill(th.bg)
    pygame.draw.rect(surf, th.panel, pygame.Rect(12, 12, surf.get_width() - 24, 64), border_radius=12)
    pygame.draw.rect(surf, th.border, pygame.Rect(12, 12, surf.get_width() - 24, 64), 1, border_radius=12)

    t1 = font.render(title, True, th.text)
    surf.blit(t1, (26, 22))

    st_col = th.ok if status.startswith("RUNNING") else th.muted
    t2 = font.render(status, True, st_col)
    surf.blit(t2, (26, 44))


def main() -> None:
    pygame.init()
    th = Theme()

    W, H = 860, 520
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("NEP Launcher (PyGame GUI)")

    pygame.key.start_text_input()

    font = pygame.font.SysFont("Menlo", 16)
    font_small = pygame.font.SysFont("Menlo", 14)

    cfg = RunCfg()
    launcher = Launcher(cfg)

    x0, y0 = 24, 100
    col_w = 250
    row_h = 44
    pad_y = 18

    fields = [
        TextField(pygame.Rect(x0, y0 + 0 * (row_h + pad_y), col_w, row_h), "T (sim-time)", "2000.0"),
        TextField(pygame.Rect(x0, y0 + 1 * (row_h + pad_y), col_w, row_h), "seed", "1"),
        TextField(pygame.Rect(x0, y0 + 2 * (row_h + pad_y), col_w, row_h), "size", "64"),
        TextField(pygame.Rect(x0, y0 + 3 * (row_h + pad_y), col_w, row_h), "init_pop", "12"),
        TextField(pygame.Rect(x0, y0 + 4 * (row_h + pad_y), col_w, row_h), "max_pop", "256"),
    ]

    cbx_x = x0 + col_w + 60
    cbs = [
        Checkbox(pygame.Rect(cbx_x, y0 + 0 * (row_h + pad_y), 260, 24), "Show pop plot", True),
        Checkbox(pygame.Rect(cbx_x, y0 + 1 * (row_h + pad_y), 260, 24), "Show world plot", True),
        Checkbox(pygame.Rect(cbx_x, y0 + 2 * (row_h + pad_y), 260, 24), "Show phenotype plot", True),
    ]

    btn_x = cbx_x
    btn_y = y0 + 3 * (row_h + pad_y)
    btn_w = 220
    btn_h = 44

    btn_start = Button(pygame.Rect(btn_x, btn_y + 0 * (btn_h + 14), btn_w, btn_h), "START", kind="ok")
    btn_stop = Button(pygame.Rect(btn_x, btn_y + 1 * (btn_h + 14), btn_w, btn_h), "STOP", kind="danger")
    btn_logs = Button(pygame.Rect(btn_x, btn_y + 2 * (btn_h + 14), btn_w, btn_h), "RESET LOGS")

    widgets: List[Widget] = []
    widgets += fields
    widgets += cbs
    widgets += [btn_start, btn_stop, btn_logs]

    pop_fp = cfg.pop_log
    pop_f = None
    pop_pos = 0
    last_pop = None
    last_pop_read_wall = 0.0

    clock = pygame.time.Clock()

    def sync_cfg_from_ui() -> None:
        cfg.T = parse_float(fields[0].text, cfg.T)
        cfg.seed = parse_int(fields[1].text, cfg.seed)
        cfg.size = parse_int(fields[2].text, cfg.size)
        cfg.init_pop = parse_int(fields[3].text, cfg.init_pop)
        cfg.max_pop = parse_int(fields[4].text, cfg.max_pop)

        cfg.show_pop_plot = cbs[0].value
        cfg.show_world_plot = cbs[1].value
        cfg.show_pheno_plot = cbs[2].value

    def reset_pop_tail() -> None:
        nonlocal pop_f, pop_pos, last_pop, last_pop_read_wall, pop_fp
        try:
            if pop_f is not None:
                pop_f.close()
        except Exception:
            pass
        pop_f = None
        pop_pos = 0
        last_pop = None
        last_pop_read_wall = 0.0
        pop_fp = cfg.pop_log

    def poll_pop_tail() -> None:
        nonlocal pop_f, pop_pos, last_pop, last_pop_read_wall

        now = time.time()
        if now - last_pop_read_wall < 0.25:
            return
        last_pop_read_wall = now

        try:
            if pop_f is None:
                if os.path.exists(pop_fp):
                    pop_f = open(pop_fp, "r", encoding="utf-8")
                    pop_f.seek(0, 0)
                    pop_pos = pop_f.tell()

            if pop_f is None:
                return

            pop_f.seek(pop_pos)
            while True:
                line = pop_f.readline()
                if not line:
                    break
                pop_pos = pop_f.tell()
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("event") == "population":
                    s = obj.get("summary", obj)
                    if isinstance(s, dict):
                        last_pop = s
        except Exception:
            reset_pop_tail()

    running = True
    while running:
        # dynamic start label
        if launcher.running:
            btn_start.label = "RESUME" if launcher.paused else "PAUSE"
        else:
            btn_start.label = "START"

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
                break

            for wdg in widgets:
                if wdg.handle_event(ev):
                    if isinstance(wdg, TextField) and wdg.focus:
                        for f in fields:
                            if f is not wdg:
                                f.focus = False
                    break

        if btn_start.clicked:
            sync_cfg_from_ui()
            if not launcher.running:
                launcher.start()
                reset_pop_tail()
            else:
                if launcher.paused:
                    launcher.resume()
                else:
                    launcher.pause()

        if btn_stop.clicked:
            launcher.stop()

        if btn_logs.clicked:
            sync_cfg_from_ui()
            launcher.stop()
            launcher.reset_logs()
            reset_pop_tail()

        poll_pop_tail()

        if launcher.running:
            st = "PAUSED" if launcher.paused else "RUNNING"
            pid = launcher.sim_proc.pid if launcher.sim_proc else "NA"
            if last_pop and "t" in last_pop:
                try:
                    t_sim = float(last_pop.get("t", float("nan")))
                    pop_n = int(last_pop.get("pop", 0))
                    mean_E = float(last_pop.get("mean_E", float("nan")))
                    mean_D = float(last_pop.get("mean_D", float("nan")))
                    status = f"{st}  pid={pid}  t={t_sim:.2f}  pop={pop_n}  mean_E={mean_E:.3f}  mean_D={mean_D:.3f}"
                except Exception:
                    status = f"{st}  pid={pid}  (status parse error)"
            else:
                status = f"{st}  pid={pid}  (waiting for pop events)"
        else:
            status = "IDLE"

        draw_header(screen, font, th, "NEP Launcher", status)

        pygame.draw.rect(screen, th.panel, pygame.Rect(12, 88, W - 24, H - 100), border_radius=12)
        pygame.draw.rect(screen, th.border, pygame.Rect(12, 88, W - 24, H - 100), 1, border_radius=12)

        tA = font.render("Run parameters", True, th.muted)
        screen.blit(tA, (x0, y0 - 34))

        tB = font.render("Plots + actions", True, th.muted)
        screen.blit(tB, (cbx_x, y0 - 34))

        for f in fields:
            f.draw(screen, font, th)
        for cb in cbs:
            cb.draw(screen, font, th)
        for b in (btn_start, btn_stop, btn_logs):
            b.draw(screen, font, th)

        hint = "ESC quits launcher. START toggles pause/resume when running. World window still has its own controls."
        tH = font_small.render(hint, True, th.muted)
        screen.blit(tH, (24, H - 28))

        pygame.display.flip()
        clock.tick(60)

    try:
        launcher.stop()
    finally:
        try:
            if pop_f is not None:
                pop_f.close()
        except Exception:
            pass
        pygame.quit()


if __name__ == "__main__":
    main()