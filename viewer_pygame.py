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


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """HSV (0-1 each) → RGB tuple (0-255 each)."""
    if s == 0.0:
        c = int(v * 255)
        return c, c, c
    h6 = (h % 1.0) * 6.0
    i = int(h6)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r, g, b = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][i % 6]
    return int(r * 255), int(g * 255), int(b * 255)


def _agent_visuals(agent) -> Tuple[Tuple[int,int,int], int]:
    """
    Returnerar (rgb_color, radius_px) baserat på agentens fysiologi:
      - Färg (hue): skada D  → grön (frisk/ung) till röd (döende)
      - Ljusstyrka (value):  energireserv → mörk (svält) till ljus (välmådd)
      - Storlek (radius):    massa M → liten (nyfödd) till stor (vuxen)
    """
    body = getattr(agent, "body", None)
    if body is None:
        return (200, 200, 200), 3

    # --- Färg: hue från D (0=grön, 0.33→gul vid D=0.5, röd vid D=1) ---
    D     = float(getattr(body, "D",   0.0))
    D_max = float(getattr(getattr(agent, "AP", None), "D_max", 1.0) or 1.0)
    d_norm = max(0.0, min(1.0, D / max(D_max, 1e-9)))
    hue = 0.33 * (1.0 - d_norm)   # 0.33=grön → 0.0=röd

    # --- Ljusstyrka: energireserv ---
    try:
        Et   = float(body.E_total())
        Ecap = float(body.E_cap())
        e_frac = max(0.0, min(1.0, Et / max(Ecap, 1e-9)))
    except Exception:
        e_frac = 0.5
    value = 0.35 + 0.65 * e_frac   # 0.35 (svält) → 1.0 (full)

    saturation = 0.85

    rgb = _hsv_to_rgb(hue, saturation, value)

    # --- Storlek: massa M (klammad till 2–8 px) ---
    M    = float(getattr(body, "M", 0.2))
    M0   = float(getattr(getattr(agent, "AP", None), "M0", 1.0) or 1.0)
    m_n  = max(0.0, min(1.0, M / max(M0, 1e-9)))
    radius = max(2, min(8, int(2 + 6 * m_n)))

    return rgb, radius
@dataclass
class ViewerConfig:
    title: str = "NEP World"
    scale: int = 10
    fps_cap: int = 60
    render_every: int = 2

    draw_agents: bool = True
    draw_heading: bool = True
    draw_rays: bool = False        # visa sensing-strålar (tangent R)
    agent_radius_px: int = 3
    agent_heading_len_px: int = 6

    show_hud: bool = True

    # Modes:
    #   CB   : RGB=(C,B,0)
    #   B/C  : grayscale single field
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

                # modes (hazard removed)
                if ev.key == pygame.K_1:
                    self.cfg.mode = "CB"
                if ev.key == pygame.K_2:
                    self.cfg.mode = "B"
                if ev.key == pygame.K_3:
                    self.cfg.mode = "C"
                if ev.key == pygame.K_4:
                    self.cfg.mode = "TEMP"
                if ev.key == pygame.K_5:
                    self.cfg.mode = "VEG"

                if ev.key == pygame.K_a:
                    self.cfg.draw_agents = not self.cfg.draw_agents
                if ev.key == pygame.K_r:
                    self.cfg.draw_rays = not self.cfg.draw_rays
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
            # ingen throttling (ingen sleep)
            pass

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
    def _veg_G_and_m(T: np.ndarray, WP) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct G(T) and m(T) using WorldParams-compatible fields."""
        # G(T) triangular window
        G = np.zeros_like(T, dtype=np.float32)
        Tmin, Topt, Tmax = float(WP.T_grow_min), float(WP.T_grow_opt), float(WP.T_grow_max)

        if Topt > Tmin + 1e-9:
            G = np.where((T >= Tmin) & (T < Topt), (T - Tmin) / (Topt - Tmin), G)
        if Tmax > Topt + 1e-9:
            G = np.where((T >= Topt) & (T <= Tmax), (Tmax - T) / (Tmax - Topt), G)
        G = np.clip(G, 0.0, 1.0).astype(np.float32, copy=False)

        # m(T) wither rate
        m = np.full_like(T, float(WP.B_wither_base), dtype=np.float32)

        if float(getattr(WP, "B_wither_cold", 0.0)) > 0.0 and float(getattr(WP, "cold_width", 0.0)) > 1e-9:
            Sc = np.clip((float(WP.T_cold) - T) / float(WP.cold_width), 0.0, 1.0).astype(np.float32, copy=False)
            m += float(WP.B_wither_cold) * Sc

        if float(getattr(WP, "B_wither_hot", 0.0)) > 0.0 and float(getattr(WP, "hot_width", 0.0)) > 1e-9:
            Sh = np.clip((T - float(WP.T_hot)) / float(WP.hot_width), 0.0, 1.0).astype(np.float32, copy=False)
            m += float(WP.B_wither_hot) * Sh

        return G, m

    def _make_rgb(self, world) -> np.ndarray:
        """Returns (H,W,3) uint8."""
        B = np.asarray(world.B, dtype=np.float32)
        C = np.asarray(world.C, dtype=np.float32)

        WP = getattr(world, "WP", None)
        BK = float(getattr(WP, "B_K", 1.0)) if WP is not None else 1.0
        B01 = np.clip(B / max(BK, 1e-12), 0.0, 1.0).astype(np.float32, copy=False)

        mode = self.cfg.mode.upper().strip()

        if mode == "B":
            img = np.dstack([B01, B01, B01])

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
            WP = world.WP
            if WP is None:
                img = np.dstack([B01, B01, B01])
            else:
                BK = float(getattr(WP, "B_K", 1.0))
                invBK = 1.0 / max(BK, 1e-12)
                B01 = np.clip(B * invBK, 0.0, 1.0).astype(np.float32, copy=False)

                T = self._temp_field(world, B)
                G, m = self._veg_G_and_m(T, WP)

                # mirror World.step() terms (no diffusion needed for "health" coloring)
                growth = (float(WP.B_regen) * G) * (1.0 - B * invBK) * B
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
            # default "CB": R=C, G=B, B=0
            Z = np.zeros_like(B01, dtype=np.float32)
            img = np.dstack([C, B01, Z])

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

    def _draw_rays(self, pop) -> None:
        """
        Visualiserar sensing-strålarna för varje levande agent.
        Strållängden per stråle reflekterar ellipsmodellen via sensors._ray_m.
        Grönt = mat (B), cyan = kadaver (C), svag grå = ingen signal.
        """
        pygame = self.pg
        s     = int(pop.world.WP.size)
        scale = int(self.cfg.scale)
        W_px  = s * scale

        ray_surf = pygame.Surface((W_px, W_px), pygame.SRCALPHA)

        for a in pop.agents:
            if not _is_alive(a):
                continue

            sensors = getattr(a, "sensors", None)
            if sensors is None:
                continue

            n    = int(getattr(sensors, "_n", 0))
            m    = int(getattr(sensors, "_m", 0))
            if n <= 0 or m <= 0:
                continue

            accB  = getattr(sensors, "_accB",    None)
            accC  = getattr(sensors, "_accC",    None)
            ang   = getattr(sensors, "_ang_base", None)
            d     = getattr(sensors, "_d",        None)
            ray_m = getattr(sensors, "_ray_m",    None)
            if ang is None or d is None:
                continue

            ax, ay, heading = _get_xy_heading(a)
            px = int(ax * scale) % W_px
            py = int(ay * scale) % W_px

            for i in range(n):
                # Per-stråle räckvidd från ellipsmodellen
                if ray_m is not None and i < len(ray_m):
                    depth = max(1, min(int(ray_m[i]), m))
                else:
                    depth = m
                ray_len_px = float(d[depth - 1]) * scale

                angle = float(ang[i]) + heading
                ex = px + ray_len_px * math.cos(angle)
                ey = py + ray_len_px * math.sin(angle)

                if ex < 0 or ex >= W_px or ey < 0 or ey >= W_px:
                    continue

                sig_B = float(accB[i]) if accB is not None else 0.0
                sig_C = float(accC[i]) if accC is not None else 0.0
                sig   = max(sig_B, sig_C)

                if sig < 0.02:
                    color = (60, 80, 60, 25)
                else:
                    alpha = int(40 + 160 * min(sig, 1.0))
                    if sig_B >= sig_C:
                        g = int(80 + 175 * sig_B)
                        color = (20, min(255, g), 30, alpha)
                    else:
                        gb = int(80 + 175 * sig_C)
                        color = (20, min(255, gb), min(255, gb), alpha)

                pygame.draw.line(ray_surf, color, (px, py), (int(ex), int(ey)), 1)

        pop.world._viewer_ray_surf = ray_surf

    def _draw_agents(self, pop) -> None:
        if not self.cfg.draw_agents:
            return

        pygame = self.pg
        s = int(pop.world.WP.size) if hasattr(pop, "world") else None
        if s is None:
            return

        scale = int(self.cfg.scale)
        hl    = int(self.cfg.agent_heading_len_px)

        agents = getattr(pop, "agents", None)
        if agents is None:
            return

        # Rita strålar först (under agenterna)
        if self.cfg.draw_rays:
            self._draw_rays(pop)
            ray_surf = getattr(getattr(pop, "world", None), "_viewer_ray_surf", None)
            if ray_surf is not None:
                self._screen.blit(ray_surf, (0, 0))

        for a in agents:
            if not _is_alive(a):
                continue
            x, y, h = _get_xy_heading(a)

            px = int(x * scale) % (s * scale)
            py = int(y * scale) % (s * scale)

            color, radius = _agent_visuals(a)

            # Fyllda cirklar med biologisk färg
            pygame.draw.circle(self._screen, color, (px, py), radius)

            # --- Parningsläge: pulserande ring ---
            ready = False
            try:
                ready = a.ready_to_reproduce()
            except Exception:
                pass
            if ready:
                pulse = 0.5 + 0.5 * math.sin(self._step * 0.25)
                ring_alpha = int(120 + 120 * pulse)
                ring_r = radius + 2 + int(pulse * 2)
                ring_color = (255, 220, 50, ring_alpha)
                ring_surf = pygame.Surface((ring_r*2+2, ring_r*2+2), pygame.SRCALPHA)
                pygame.draw.circle(ring_surf, ring_color,
                                   (ring_r+1, ring_r+1), ring_r, 2)
                self._screen.blit(ring_surf, (px - ring_r - 1, py - ring_r - 1))

            # --- Graviditet: liten inre prick för fostret ---
            body = getattr(a, "body", None)
            if body is not None and getattr(body, "gestating", False):
                gest_M      = float(getattr(body, "gest_M", 0.0))
                gest_M_tgt  = float(getattr(body, "gest_M_target", 1e-9))
                frac        = min(1.0, gest_M / max(gest_M_tgt, 1e-9))
                fetus_r     = max(1, int(radius * 0.35 + frac * radius * 0.25))
                # Färg: grön-vit baserat på energi/massa-fraktion
                g_val       = int(180 + 75 * frac)
                fetus_color = (200, g_val, 120)
                pygame.draw.circle(self._screen, fetus_color, (px, py), fetus_r)

            # Riktningslinje i samma färg men lite mörkare
            if self.cfg.draw_heading and hl > 0:
                dim = tuple(max(0, int(c * 0.6)) for c in color)
                ex = int(px + (radius + hl * 0.5) * math.cos(h))
                ey = int(py + (radius + hl * 0.5) * math.sin(h))
                pygame.draw.line(self._screen, dim, (px, py), (ex, ey), 1)

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

        rays_str = "strålar:PÅ" if self.cfg.draw_rays else "strålar:av"
        line3 = f"grön=frisk→röd=döende  ljus=energi  gul ring=parningsredo  vit prick=gravid  [{rays_str} R]"

        x0, y0 = 5, 5
        dy = 18

        for i, text in enumerate([line1, line2, line3]):
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