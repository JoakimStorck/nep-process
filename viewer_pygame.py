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

def _is_repro_ready(pop, agent) -> bool:
    """
    Viewer-helper: fråga Populations store-first reproduktionsgate.
    """
    try:
        slot = int(getattr(agent, "store_slot", -1))
    except Exception:
        return False

    if slot < 0:
        return False

    fn = getattr(pop, "_ready_to_reproduce_slot", None)
    if callable(fn):
        try:
            return bool(fn(slot))
        except Exception:
            return False

    return False
    
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


def _agent_visuals(agent) -> Tuple[Tuple[int, int, int], int]:
    """
    Returnerar (rgb_color, radius_px) baserat på agentens fysiologi:
      - Färg (hue): skada D  → grön (frisk/ung) till röd (döende)
      - Ljusstyrka (value):  energireserv → mörk (svält) till ljus (välmådd)
      - Storlek (radius):    massa M → liten (nyfödd) till stor (vuxen)
    """
    body = getattr(agent, "body", None)
    if body is None:
        return (200, 200, 200), 3

    D = float(getattr(body, "D", 0.0))
    D_max = float(getattr(getattr(agent, "AP", None), "D_max", 1.0) or 1.0)
    d_norm = max(0.0, min(1.0, D / max(D_max, 1e-9)))
    hue = 0.33 * (1.0 - d_norm)

    try:
        Et = float(body.E_total())
        Ecap = float(body.E_cap())
        e_frac = max(0.0, min(1.0, Et / max(Ecap, 1e-9)))
    except Exception:
        e_frac = 0.5
    value = 0.35 + 0.65 * e_frac

    saturation = 0.85
    rgb = _hsv_to_rgb(hue, saturation, value)

    M = float(getattr(body, "M", 0.2))
    M0 = float(getattr(getattr(agent, "AP", None), "M0", 1.0) or 1.0)
    m_n = max(0.0, min(1.0, M / max(M0, 1e-9)))
    radius = max(2, min(8, int(2 + 6 * m_n)))

    return rgb, radius


def _iter_live_flora_slots(pop):
    store = getattr(pop, "store", None)
    if store is None:
        return
    n0 = int(getattr(store, "n_agents", 0))
    n1 = int(getattr(store, "n", 0))
    for slot in range(n0, n1):
        if not bool(store.alive[slot]):
            continue
        if int(store.kind[slot]) != 1:
            continue
        yield slot

def _flora_mass_field(pop) -> np.ndarray:
    world = getattr(pop, "world", None)
    if world is None:
        return np.zeros((1, 1), dtype=np.float32)

    s = int(world.WP.size)
    B = np.zeros((s, s), dtype=np.float32)

    store = getattr(pop, "store", None)
    grid = getattr(pop, "grid", None)
    if store is None or grid is None:
        return B

    n0 = int(getattr(store, "n_agents", 0))
    n1 = int(getattr(store, "n", 0))

    for slot in range(n0, n1):
        if not bool(store.alive[slot]):
            continue
        if int(store.kind[slot]) != 1:
            continue

        cell = int(store.cell_idx[slot])
        if cell < 0:
            continue

        row, col = grid.rowcol_of(cell)
        B[row, col] += float(store.mass[slot])

    return B    

@dataclass
class ViewerConfig:
    title: str = "NEP World"
    scale: int = 10
    fps_cap: int = 60
    render_every: int = 2

    draw_agents: bool = True
    draw_heading: bool = True
    draw_rays: bool = False
    agent_radius_px: int = 3
    agent_heading_len_px: int = 6

    show_hud: bool = True

    # Modes:
    #   CB    : RGB=(C,B,0)
    #   B/C   : grayscale single field
    #   TEMP  : grayscale temperature
    #   FLORA : diskret flora-overlay ovanpå neutral bakgrund
    mode: str = "FLORA"
    gamma: float = 1.0

    # Hur flora färgkodas i FLORA-läget:
    #   temp_opt | dispersal | adult_mass | growth
    flora_color_by: str = "temp_opt"


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
                    self.cfg.mode = "CB"
                if ev.key == pygame.K_2:
                    self.cfg.mode = "B"
                if ev.key == pygame.K_3:
                    self.cfg.mode = "C"
                if ev.key == pygame.K_4:
                    self.cfg.mode = "FLORA"
                if ev.key == pygame.K_5:
                    self.cfg.mode = "TEMP"
                    
                if ev.key == pygame.K_a:
                    self.cfg.draw_agents = not self.cfg.draw_agents
                if ev.key == pygame.K_r:
                    self.cfg.draw_rays = not self.cfg.draw_rays
                if ev.key == pygame.K_h:
                    self.cfg.show_hud = not self.cfg.show_hud
                if ev.key == pygame.K_t:
                    order = ["temp_opt", "dispersal", "adult_mass", "growth"]
                    cur = str(getattr(self.cfg, "flora_color_by", "temp_opt"))
                    try:
                        i = order.index(cur)
                    except ValueError:
                        i = 0
                    self.cfg.flora_color_by = order[(i + 1) % len(order)]

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

    @staticmethod
    def _temp_field(world, shape_like: np.ndarray) -> np.ndarray:
        if hasattr(world, "Ty"):
            Ty = np.asarray(world.Ty, dtype=np.float32)
            return np.broadcast_to(Ty[:, None], shape_like.shape).astype(np.float32, copy=False)
        return np.zeros_like(shape_like, dtype=np.float32)

    def _make_rgb(self, pop) -> np.ndarray:
        """Returns (H,W,3) uint8."""
        world = pop.world
    
        B = _flora_mass_field(pop)
        C = np.asarray(world.C, dtype=np.float32)
    
        WP = getattr(world, "WP", None)
        BK = float(getattr(WP, "B_K", 1.0)) if WP is not None else 1.0
        CK = float(getattr(WP, "C_K", 1.0)) if WP is not None else 1.0
    
        B01 = np.clip(B / max(BK, 1e-12), 0.0, 1.0).astype(np.float32, copy=False)
        C01 = np.clip(C / max(CK, 1e-12), 0.0, 1.0).astype(np.float32, copy=False)
        C01 = np.sqrt(C01, dtype=np.float32)
    
        mode = self.cfg.mode.upper().strip()
    
        if mode == "B":
            img = np.dstack([B01, B01, B01])
    
        elif mode == "C":
            img = np.dstack([C01, C01, C01])
    
        elif mode == "TEMP":
            T = self._temp_field(world, C)
            Tmin, Tmax = -10.0, 40.0
            t01 = np.clip((T - Tmin) / (Tmax - Tmin), 0.0, 1.0).astype(np.float32, copy=False)
            img = np.dstack([t01, t01, t01])
    
        elif mode == "FLORA":
            T = self._temp_field(world, C)
            Tmin, Tmax = -10.0, 40.0
            t01 = np.clip((T - Tmin) / (Tmax - Tmin), 0.0, 1.0).astype(np.float32, copy=False)
    
            base = 0.08 + 0.10 * t01
            img = np.dstack([0.10 * base, 0.18 * base, 0.22 * base]).astype(np.float32, copy=False)
    
        else:  # "CB"
            Z = np.zeros_like(B01, dtype=np.float32)
            img = np.dstack([C01, B01, Z])
    
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

    def _draw_flora(self, pop) -> None:
        store = getattr(pop, "store", None)
        grid = getattr(pop, "grid", None)
        if store is None or grid is None:
            return
    
        pygame = self.pg
        scale = int(self.cfg.scale)
        color_by = str(getattr(self.cfg, "flora_color_by", "temp_opt")).lower()
    
        for slot in _iter_live_flora_slots(pop):
            cell = int(store.cell_idx[slot])
            if cell < 0:
                continue
    
            row, col = grid.rowcol_of(cell)
            x = col * scale
            y = row * scale
    
            m = float(store.mass[slot])
    
            try:
                m_cap = float(pop._flora_adult_mass[slot])
            except Exception:
                m_cap = max(m, 1e-9)
            frac = max(0.0, min(1.0, m / max(m_cap, 1e-9)))
            a = int(80 + 175 * frac)
    
            # default
            u = 0.5
    
            try:
                if color_by == "temp_opt":
                    val = float(pop._flora_temp_opt[slot])
                    u = max(0.0, min(1.0, (val + 5.0) / 40.0))
    
                elif color_by == "dispersal":
                    val = float(pop._flora_dispersal_rate[slot])
                    u = max(0.0, min(1.0, (val - 0.0002) / (0.0200 - 0.0002)))
    
                elif color_by == "adult_mass":
                    val = float(pop._flora_adult_mass[slot])
                    lo = 0.25 * float(pop.WP.B_K)
                    hi = 4.0 * float(pop.WP.B_K)
                    u = max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9)))
    
                elif color_by == "growth":
                    val = float(pop._flora_growth_rate[slot])
                    u = max(0.0, min(1.0, (val - 0.005) / (0.050 - 0.005)))
    
            except Exception:
                u = 0.5
    
            # kall/låg -> blå/cyan, varm/hög -> gul/röd
            r = int(40 + 180 * u)
            g = int(80 + 140 * (1.0 - abs(u - 0.5) * 2.0))
            b = int(40 + 180 * (1.0 - u))
    
            color = (r, g, b, a)
    
            flora_surf = pygame.Surface((scale, scale), pygame.SRCALPHA)
            flora_surf.fill(color)
            self._screen.blit(flora_surf, (x, y))

    def _draw_rays(self, pop) -> None:
        pygame = self.pg
        s = int(pop.world.WP.size)
        scale = int(self.cfg.scale)
        W_px = s * scale

        ray_surf = pygame.Surface((W_px, W_px), pygame.SRCALPHA)

        for a in pop.agents:
            if not _is_alive(a):
                continue

            sensors = getattr(a, "sensors", None)
            if sensors is None:
                continue

            n = int(getattr(sensors, "_n", 0))
            m = int(getattr(sensors, "_m", 0))
            if n <= 0 or m <= 0:
                continue

            accB = getattr(sensors, "_accB", None)
            accC = getattr(sensors, "_accC", None)
            ang = getattr(sensors, "_ang_base", None)
            d = getattr(sensors, "_d", None)
            ray_m = getattr(sensors, "_ray_m", None)
            if ang is None or d is None:
                continue

            ax, ay, heading = _get_xy_heading(a)
            px = int(ax * scale) % W_px
            py = int(ay * scale) % W_px

            for i in range(n):
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

                if not math.isfinite(sig_B):
                    sig_B = 0.0
                if not math.isfinite(sig_C):
                    sig_C = 0.0

                sig = max(sig_B, sig_C)

                if sig < 0.02:
                    color = (60, 80, 60, 25)
                else:
                    sig_clamped = max(0.0, min(sig, 1.0))
                    alpha = int(40 + 160 * sig_clamped)
                    if sig_B >= sig_C:
                        g = int(80 + 175 * max(0.0, min(sig_B, 1.0)))
                        color = (20, min(255, g), 30, alpha)
                    else:
                        gb = int(80 + 175 * max(0.0, min(sig_C, 1.0)))
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
        hl = int(self.cfg.agent_heading_len_px)

        agents = getattr(pop, "agents", None)
        if agents is None:
            return

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
            pygame.draw.circle(self._screen, color, (px, py), radius)

            ready = _is_repro_ready(pop, a)
            if ready:
                pulse = 0.5 + 0.5 * math.sin(self._step * 0.25)
                ring_alpha = int(120 + 120 * pulse)
                ring_r = radius + 2 + int(pulse * 2)
                ring_color = (255, 220, 50, ring_alpha)
                ring_surf = pygame.Surface((ring_r * 2 + 2, ring_r * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(ring_surf, ring_color, (ring_r + 1, ring_r + 1), ring_r, 2)
                self._screen.blit(ring_surf, (px - ring_r - 1, py - ring_r - 1))

            pred_trait = 0.0
            try:
                pred_trait = float(getattr(getattr(a, "pheno", None), "predation", 0.0))
            except Exception:
                pass
            if pred_trait > 0.15:
                pred_r = radius + 1
                pred_alpha = int(60 + 180 * pred_trait)
                pred_color = (220, 30, 30, pred_alpha)
                pred_surf = pygame.Surface((pred_r * 2 + 2, pred_r * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(pred_surf, pred_color, (pred_r + 1, pred_r + 1), pred_r, max(1, int(pred_trait * 3)))
                self._screen.blit(pred_surf, (px - pred_r - 1, py - pred_r - 1))

            body = getattr(a, "body", None)
            if body is not None and getattr(body, "gestating", False):
                gest_M = float(getattr(body, "gest_M", 0.0))
                gest_M_tgt = float(getattr(body, "gest_M_target", 1e-9))
                frac = min(1.0, gest_M / max(gest_M_tgt, 1e-9))
                fetus_r = max(1, int(radius * 0.35 + frac * radius * 0.25))
                g_val = int(180 + 75 * frac)
                fetus_color = (200, g_val, 120)
                pygame.draw.circle(self._screen, fetus_color, (px, py), fetus_r)

            if self.cfg.draw_heading and hl > 0:
                dim = tuple(max(0, int(c * 0.6)) for c in color)
                ex = int(px + (radius + hl * 0.5) * math.cos(h))
                ey = int(py + (radius + hl * 0.5) * math.sin(h))
                pygame.draw.line(self._screen, dim, (px, py), (ex, ey), 1)

    def _draw_hud(self, pop, births_total: int, deaths_total: int) -> None:
        if not self.cfg.show_hud:
            return

        t = getattr(pop, "t", None)
        if t is None and hasattr(pop, "world"):
            t = getattr(pop.world, "t", 0.0)

        n = 0
        if hasattr(pop, "agents"):
            n = sum(1 for a in pop.agents if _is_alive(a))

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

        flora_n = 0
        flora_mass = 0.0
        line5 = ""
        if hasattr(pop, "store"):
            store = pop.store
            n0 = int(getattr(store, "n_agents", 0))
            n1 = int(getattr(store, "n", 0))
            mask = (store.kind[n0:n1] == 1) & store.alive[n0:n1]
            flora_n = int(np.sum(mask))
            if flora_n > 0:
                flora_mass = float(np.sum(store.mass[n0:n1][mask]))

        if hasattr(pop, "_flora_summary"):
            ft = pop._flora_summary()
            color_by = str(getattr(self.cfg, "flora_color_by", "temp_opt"))
            line5 = (
                f"flora[{color_by}]: "
                f"g={ft['flora_mean_growth_rate']:.3f}  "
                f"M*={ft['flora_mean_adult_mass']:.4f}  "
                f"Topt={ft['flora_mean_temp_opt']:.1f}  "
                f"Tw={ft['flora_mean_temp_width']:.1f}  "
                f"d={ft['flora_mean_dispersal_rate']:.4f}"
            )

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

        line3 = f"flora_n={flora_n:4d}  flora_mass={flora_mass:.4f} kg"

        rays_str = "strålar:PÅ" if self.cfg.draw_rays else "strålar:av"
        line4 = f"grön=frisk→röd=döende  ljus=energi  gul ring=parningsredo  vit prick=gravid  [{rays_str} R]  [trait T]"

        lines = [line1, line2, line3, line4]
        if line5:
            lines.append(line5)

        x0, y0 = 5, 5
        dy = 18

        for i, text in enumerate(lines):
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

        rgb = self._make_rgb(pop)
        self._blit_field(rgb)

        if self.cfg.mode.upper().strip() == "FLORA":
            self._draw_flora(pop)

        self._draw_agents(pop)
        self._draw_hud(pop, births_total=births_total, deaths_total=deaths_total)

        self.pg.display.flip()
        self._throttle()
        return True

    def close(self) -> None:
        self.pg.quit()