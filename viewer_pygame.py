# viewer_pygame.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

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
    scale: int = 10
    fps_cap: int = 60
    render_every: int = 2

    draw_agents: bool = True
    draw_heading: bool = True
    agent_radius_px: int = 3
    agent_heading_len_px: int = 6

    show_hud: bool = True

    # Modes:
    #   CBF  : RGB=(C,B,F)
    #   B/F/C: grayscale single field
    #   TEMP : grayscale temperature
    #   VEG  : vegetation health (green<->brown based on stress)
    mode: str = "VEG"
    gamma: float = 1.0


class WorldViewer:
    def __init__(self, cfg: ViewerConfig):
        self.cfg = cfg
        self._step = 0
        self._paused = False

        import pygame  # noqa
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(cfg.title)

        self._screen = None
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Menlo", 14)

    # ---------- input ----------
    def _handle_events(self) -> bool:
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
                if ev.key == pygame.K_6:
                    self.cfg.mode = "VEG"

                if ev.key == pygame.K_a:
                    self.cfg.draw_agents = not self.cfg.draw_agents
                if ev.key == pygame.K_h:
                    self.cfg.show_hud = not self.cfg.show_hud

                if ev.key == pygame.K_EQUALS or ev.key == pygame.K_PLUS:
                    self.cfg.gamma = max(0.20, self.cfg.gamma * 0.90)
                if ev.key == pygame.K_MINUS:
                    self.cfg.gamma = min(5.00, self.cfg.gamma * 1.10)

        return True

    @property
    def paused(self) -> bool:
        return self._paused

    def _throttle(self) -> None:
        cap = int(getattr(self.cfg, "fps_cap", 0) or 0)
        if cap > 0:
            self._clock.tick(cap)
        else:
            pass
            # ingen throttling (ingen sleep)
            #self._clock.tick(0)
            
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
        return np.power(_clip01(x01), g, dtype=np.float32)

    # ----- helpers for VEG mode -----
    @staticmethod
    def _temp_field(world, shape_like: np.ndarray) -> np.ndarray:
        """Broadcast Ty -> (H,W). If missing Ty, return zeros."""
        if hasattr(world, "Ty"):
            Ty = np.asarray(world.Ty, dtype=np.float32)
            return np.broadcast_to(Ty[:, None], shape_like.shape).astype(np.float32, copy=False)
        return np.zeros_like(shape_like, dtype=np.float32)

    @staticmethod
    def _veg_G_and_m(T: np.ndarray, P) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct G(T) and m(T) using WorldParams-compatible fields."""
        # G(T) triangular window
        G = np.zeros_like(T, dtype=np.float32)
        Tmin, Topt, Tmax = float(P.T_grow_min), float(P.T_grow_opt), float(P.T_grow_max)

        if Topt > Tmin + 1e-9:
            G = np.where((T >= Tmin) & (T < Topt), (T - Tmin) / (Topt - Tmin), G)
        if Tmax > Topt + 1e-9:
            G = np.where((T >= Topt) & (T <= Tmax), (Tmax - T) / (Tmax - Topt), G)
        G = np.clip(G, 0.0, 1.0).astype(np.float32, copy=False)

        # m(T) wither rate
        m = np.full_like(T, float(P.B_wither_base), dtype=np.float32)

        if float(getattr(P, "B_wither_cold", 0.0)) > 0.0 and float(getattr(P, "cold_width", 0.0)) > 1e-9:
            Sc = np.clip((float(P.T_cold) - T) / float(P.cold_width), 0.0, 1.0).astype(np.float32, copy=False)
            m += float(P.B_wither_cold) * Sc

        if float(getattr(P, "B_wither_hot", 0.0)) > 0.0 and float(getattr(P, "hot_width", 0.0)) > 1e-9:
            Sh = np.clip((T - float(P.T_hot)) / float(P.hot_width), 0.0, 1.0).astype(np.float32, copy=False)
            m += float(P.B_wither_hot) * Sh

        return G, m

    def _make_rgb(self, world) -> np.ndarray:
        """Returns (H,W,3) uint8."""
        B = np.asarray(world.B, dtype=np.float32)
        P = getattr(world, "P", None)
        BK = float(getattr(P, "B_K", 1.0)) if P is not None else 1.0
        B01 = np.clip(B / max(BK, 1e-12), 0.0, 1.0).astype(np.float32, copy=False)
        F = np.asarray(world.F, dtype=np.float32)
        C = np.asarray(world.C, dtype=np.float32)
        mode = self.cfg.mode.upper().strip()

        if mode == "B":
            img = np.dstack([B01, B01, B01])

        elif mode == "F":
            img = np.dstack([F, F, F])

        elif mode == "C":
            img = np.dstack([C, C, C])

        elif mode == "TEMP":
            # stable normalization span for readability
            T = self._temp_field(world, B)
            Tmin, Tmax = -10.0, 40.0
            t01 = np.clip((T - Tmin) / (Tmax - Tmin), 0.0, 1.0).astype(np.float32, copy=False)
            img = np.dstack([t01, t01, t01])

        elif mode == "VEG":
            # vegetation "health": green<->brown based on stress = wither/(growth+wither)
            P = world.P if hasattr(world, "P") else getattr(world, "params", None)
            if P is None:
                img = np.dstack([B, B, B])
            else:
                BK = float(getattr(P, "B_K", 1.0))
                invBK = 1.0 / max(BK, 1e-12)
                B01 = np.clip(B * invBK, 0.0, 1.0).astype(np.float32, copy=False)
        
                T = self._temp_field(world, B)
                G, m = self._veg_G_and_m(T, P)
        
                # mirror World.step() terms (no diffusion needed for "health" coloring)
                growth = (float(P.B_regen) * G) * (1.0 - B * invBK) * B
                wither = m * B
        
                eps = np.float32(1e-9)
                stress = wither / (growth + wither + eps)  # 0..1
        
                # brightness scales with normalized biomass (kg -> 0..1)
                green = B01 * (1.0 - stress)
                brown = B01 * stress
        
                R = brown
                Gc = green + 0.35 * brown
                Bl = 0.10 * brown
        
                img = np.dstack([R, Gc, Bl]).astype(np.float32, copy=False)

        else:
            # default "CBF": R=C, G=B, B=F
            img = np.dstack([C, B01, F])

        img = self._gamma(img)
        return _as_u8_rgb(img)

    def _blit_field(self, rgb_u8: np.ndarray) -> None:
        pygame = self.pg
        s = int(rgb_u8.shape[0])
        self._ensure_screen(s)

        surf = pygame.surfarray.make_surface(np.transpose(rgb_u8, (1, 0, 2)))
        if self.cfg.scale != 1:
            surf = pygame.transform.scale(surf, (s * self.cfg.scale, s * self.cfg.scale))
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

        agents = getattr(pop, "agents", None)
        if agents is None:
            return

        for a in agents:
            if not _is_alive(a):
                continue
            x, y, h = _get_xy_heading(a)

            px = int(x * scale) % (s * scale)
            py = int(y * scale) % (s * scale)

            pygame.draw.circle(self._screen, (255, 255, 255), (px, py), r)

            if self.cfg.draw_heading:
                ex = int(px + hl * math.cos(h))
                ey = int(py + hl * math.sin(h))
                pygame.draw.line(self._screen, (220, 220, 220), (px, py), (ex, ey), 1)

    def _draw_hud(self, pop, births_total: int, deaths_total: int) -> None:
        if not self.cfg.show_hud:
            return
    
        # time
        t = getattr(pop, "t", None)
        if t is None and hasattr(pop, "world"):
            t = getattr(pop.world, "t", 0.0)
    
        # population (alive)
        n = 0
        if hasattr(pop, "agents"):
            n = sum(1 for a in pop.agents if _is_alive(a))
    
        # temperature stats (global + hemispheres)
        tmean = tmin = tmax = float("nan")
        tmeanN = tmeanS = float("nan")
        if hasattr(pop, "world") and hasattr(pop.world, "Ty"):
            Ty = np.asarray(pop.world.Ty, dtype=np.float32)
            if Ty.size:
                tmean = float(np.mean(Ty))
                tmin = float(np.min(Ty))
                tmax = float(np.max(Ty))
                mid = Ty.size // 2
                if 0 < mid < Ty.size:
                    tmeanN = float(np.mean(Ty[:mid]))
                    tmeanS = float(np.mean(Ty[mid:]))
    
        mode = self.cfg.mode.upper()
        paused = "PAUSED" if self._paused else ""
    
        # Compose 2 lines to avoid clipping off-screen
        line1 = (
            f"t={t:8.2f}  pop={n:4d}  born={int(births_total):6d}  dead={int(deaths_total):6d}  "
            f"mode={mode}  gamma={self.cfg.gamma:.2f}  {paused}"
        )
    
        if math.isfinite(tmean):
            if math.isfinite(tmeanN) and math.isfinite(tmeanS):
                line2 = (
                    f"T(mean/min/max)={tmean:5.1f}/{tmin:5.1f}/{tmax:5.1f}   "
                    f"T(N/S)={tmeanN:5.1f}/{tmeanS:5.1f}"
                )
            else:
                line2 = f"T(mean/min/max)={tmean:5.1f}/{tmin:5.1f}/{tmax:5.1f}"
        else:
            line2 = "T(mean/min/max)=NA"
    
        # Draw with shadow for readability
        x0, y0 = 5, 5
        dy = 18
    
        for i, text in enumerate([line1, line2]):
            y = y0 + i * dy
            surf_shadow = self._font.render(text, True, (0, 0, 0))
            self._screen.blit(surf_shadow, (x0 + 1, y + 1))
            surf = self._font.render(text, True, (255, 255, 255))
            self._screen.blit(surf, (x0, y))

    def update(self, pop, births_total: int = 0, deaths_total: int = 0) -> bool:
        """
        Call from the simulation loop.
        births_total / deaths_total should be accumulated totals (not per-step).
        Returns False if the user quit.
        """
        self._step += 1

        if not self._handle_events():
            return False

        if self.cfg.render_every > 1 and (self._step % self.cfg.render_every != 0):
            self._throttle()
            return True

        rgb = self._make_rgb(pop.world)
        self._blit_field(rgb)

        self._draw_agents(pop)
        self._draw_hud(pop, births_total=births_total, deaths_total=deaths_total)

        self.pg.display.flip()
        self._throttle()
        return True

    def close(self) -> None:
        self.pg.quit()