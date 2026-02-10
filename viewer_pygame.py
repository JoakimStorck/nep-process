# viewer_pygame.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


# ---------- Utilities ----------
def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _as_u8_rgb(img01: np.ndarray) -> np.ndarray:
    """img01: (H,W,3) float in [0,1] -> (H,W,3) uint8."""
    return (255.0 * _clip01(img01)).astype(np.uint8, copy=False)


def _try_get_body(agent):
    return getattr(agent, "body", agent)


def _get_xy_heading(agent) -> Tuple[float, float, float]:
    """
    Tries common names:
      x/y: body.x, body.y
      heading: body.heading, body.theta, body.h, agent.heading
    """
    b = _try_get_body(agent)

    x = getattr(b, "x", getattr(agent, "x", 0.0))
    y = getattr(b, "y", getattr(agent, "y", 0.0))

    heading = (
        getattr(b, "heading", None)
        or getattr(b, "theta", None)
        or getattr(b, "h", None)
        or getattr(agent, "heading", 0.0)
    )
    try:
        heading = float(heading)
    except Exception:
        heading = 0.0

    return float(x), float(y), heading


def _is_alive(agent) -> bool:
    b = _try_get_body(agent)
    alive = getattr(b, "alive", getattr(agent, "alive", True))
    return bool(alive)


# ---------- Viewer ----------
@dataclass
class ViewerConfig:
    title: str = "NEP World"
    scale: int = 10              # pixel scaling: world cell -> scale x scale pixels
    fps_cap: int = 60            # render cap
    render_every: int = 2        # only render every N simulation steps (performance)
    draw_agents: bool = True
    draw_heading: bool = True
    agent_radius_px: int = 3     # radius at scale=10; keep small
    agent_heading_len_px: int = 6
    show_hud: bool = True

    # layer mapping:
    # default uses RGB = (C, B, F) so:
    #   green ~ plants, red ~ carcass, blue ~ hazard
    mode: str = "CBF"            # "CBF", "B", "F", "C", "BF0", "TEMP"
    gamma: float = 1.0           # 1.0 = linear; >1 darkens mids; <1 brightens mids


class WorldViewer:
    def __init__(self, cfg: ViewerConfig):
        self.cfg = cfg
        self._step = 0
        self._paused = False
        self._speed_mul = 1.0  # just for HUD display / future extension

        # lazy init pygame so importing doesn't require it
        import pygame  # noqa

        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(cfg.title)

        self._screen = None
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Menlo", 14)

    # ---------- input ----------
    def _handle_events(self) -> bool:
        """Returns False if user wants to quit."""
        pygame = self.pg
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE or ev.key == pygame.K_q:
                    return False
                if ev.key == pygame.K_SPACE:
                    self._paused = not self._paused
                if ev.key == pygame.K_1:
                    self.cfg.mode = "CBF"
                if ev.key == pygame.K_2:
                    self.cfg.mode = "B"
                if ev.key == pygame.K_3:
                    self.cfg.mode = "F"
                if ev.key == pygame.K_4:
                    self.cfg.mode = "C"
                if ev.key == pygame.K_5:
                    self.cfg.mode = "TEMP"
                if ev.key == pygame.K_a:
                    self.cfg.draw_agents = not self.cfg.draw_agents
                if ev.key == pygame.K_h:
                    self.cfg.show_hud = not self.cfg.show_hud
                if ev.key == pygame.K_EQUALS or ev.key == pygame.K_PLUS:
                    self.cfg.gamma = max(0.20, self.cfg.gamma * 0.90)  # brighten mids
                if ev.key == pygame.K_MINUS:
                    self.cfg.gamma = min(5.00, self.cfg.gamma * 1.10)   # darken mids
        return True

    @property
    def paused(self) -> bool:
        return self._paused

    # ---------- rendering ----------
    def _ensure_screen(self, size: int) -> None:
        if self._screen is not None:
            return
        w = int(size) * int(self.cfg.scale)
        h = int(size) * int(self.cfg.scale)
        self._screen = self.pg.display.set_mode((w, h))

    def _gamma(self, x01: np.ndarray) -> np.ndarray:
        g = float(self.cfg.gamma)
        if abs(g - 1.0) < 1e-6:
            return x01
        # perceptual-ish gamma (avoid NaNs)
        return np.power(_clip01(x01), g, dtype=np.float32)

    def _make_rgb(self, world) -> np.ndarray:
        """Returns (H,W,3) uint8."""
        B = np.asarray(world.B, dtype=np.float32)
        F = np.asarray(world.F, dtype=np.float32)
        C = np.asarray(world.C, dtype=np.float32)

        mode = self.cfg.mode.upper().strip()

        if mode == "B":
            img = np.dstack([B, B, B])
        elif mode == "F":
            img = np.dstack([F, F, F])
        elif mode == "C":
            img = np.dstack([C, C, C])
        elif mode == "TEMP":
            # expects world.Ty (size,) float32 degC, created by your new world.py
            if hasattr(world, "Ty"):
                Ty = np.asarray(world.Ty, dtype=np.float32)
                # normalize to visible range; pick a fixed span around [ -10 .. +40 ] for stability
                Tmin, Tmax = -10.0, 40.0
                t01 = np.clip((Ty - Tmin) / (Tmax - Tmin), 0.0, 1.0)
                T = np.broadcast_to(t01[:, None], B.shape).astype(np.float32, copy=False)
            else:
                T = np.zeros_like(B)
            img = np.dstack([T, T, T])
        else:
            # default "CBF": R=C, G=B, B=F
            img = np.dstack([C, B, F])

        img = self._gamma(img)
        return _as_u8_rgb(img)

    def _blit_field(self, rgb_u8: np.ndarray) -> None:
        pygame = self.pg
        s = int(rgb_u8.shape[0])
        self._ensure_screen(s)

        # pygame.surfarray expects (W,H,3), so transpose
        surf = pygame.surfarray.make_surface(np.transpose(rgb_u8, (1, 0, 2)))
        if self.cfg.scale != 1:
            surf = pygame.transform.scale(
                surf,
                (s * self.cfg.scale, s * self.cfg.scale),
            )
        self._screen.blit(surf, (0, 0))

    def _draw_agents(self, pop) -> None:
        if not self.cfg.draw_agents:
            return

        pygame = self.pg
        s = int(pop.world.P.size) if hasattr(pop, "world") else None
        if s is None:
            return

        scale = int(self.cfg.scale)
        r = int(self.cfg.agent_radius_px)
        hl = int(self.cfg.agent_heading_len_px)

        # Try to get sequence of agents
        agents = getattr(pop, "agents", None)
        if agents is None:
            return

        for a in agents:
            if not _is_alive(a):
                continue
            x, y, h = _get_xy_heading(a)

            # map world coords to pixels
            px = int(x * scale)
            py = int(y * scale)

            # keep on screen (torus wrap)
            px %= (s * scale)
            py %= (s * scale)

            # alive = white
            pygame.draw.circle(self._screen, (255, 255, 255), (px, py), r)

            if self.cfg.draw_heading:
                ex = int(px + hl * math.cos(h))
                ey = int(py + hl * math.sin(h))
                pygame.draw.line(self._screen, (220, 220, 220), (px, py), (ex, ey), 1)

    def _draw_hud(self, pop, births: int, deaths: int) -> None:
        if not self.cfg.show_hud:
            return
        pygame = self.pg

        t = getattr(pop, "t", None)
        if t is None and hasattr(pop, "world"):
            t = getattr(pop.world, "t", 0.0)

        n = 0
        if hasattr(pop, "agents"):
            n = sum(1 for a in pop.agents if _is_alive(a))

        mode = self.cfg.mode.upper()
        paused = "PAUSED" if self._paused else ""
        text = f"t={t:8.2f}  pop={n:4d}  b={births:3d} d={deaths:3d}  mode={mode}  gamma={self.cfg.gamma:.2f}  {paused}"

        surf = self._font.render(text, True, (0, 0, 0))
        self._screen.blit(surf, (6, 6))

        # draw shadow/background for readability
        surf2 = self._font.render(text, True, (255, 255, 255))
        self._screen.blit(surf2, (5, 5))

    def update(self, pop, births: int = 0, deaths: int = 0) -> bool:
        """
        Call this from the simulation loop.
        Returns False if the user quit.
        """
        self._step += 1

        if not self._handle_events():
            return False

        # pause = stop advancing sim outside, but still render here
        if self.cfg.render_every > 1 and (self._step % self.cfg.render_every != 0):
            # still cap fps a bit so input stays responsive
            self._clock.tick(self.cfg.fps_cap)
            return True

        rgb = self._make_rgb(pop.world)
        self._blit_field(rgb)

        self._draw_agents(pop)
        self._draw_hud(pop, births=births, deaths=deaths)

        self.pg.display.flip()
        self._clock.tick(self.cfg.fps_cap)
        return True

    def close(self) -> None:
        self.pg.quit()