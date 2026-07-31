from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from viewframe import FLORA_TRAITS, TEMP_RANGE


# ---------- Utilities ----------
def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _as_u8_rgb(img01: np.ndarray) -> np.ndarray:
    """img01: (H,W,3) float in [0,1] -> (H,W,3) uint8."""
    return (255.0 * _clip01(img01)).astype(np.uint8, copy=False)





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
    #   FLORA : ytfördelningen ovanpå neutral bakgrund
    #   CLAIM : anspråket per cell, bar mark mot översökt mark
    mode: str = "FLORA"
    gamma: float = 1.0

    # Hur flora färgkodas i FLORA-läget:
    #   temp_opt | dispersal | adult_mass | growth
    flora_color_by: str = "temp_opt"

    # Hur cellens yta fördelas mellan plantorna:
    #   wedge   : vinkelkilar kring cellcentrum, rena kanter
    #   stipple : stabil per-pixel-hash, exakt proportionella areor
    flora_fill: str = "wedge"


class WorldViewer:
    def __init__(self, cfg: ViewerConfig):
        self.cfg = cfg
        self._step = 0
        self._paused = False

        import pygame  # noqa
        self.pg = pygame
        # Bara video och font. pygame.init() startar även ljud och joystick,
        # vilket ger en ALSA-varning på maskiner utan ljudkort och är onödigt
        # för att rita. Interaktiva körningar påverkas inte — inget i viewern
        # använder mixer eller joystick.
        pygame.display.init()
        pygame.font.init()
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
                if ev.key == pygame.K_6:
                    self.cfg.mode = "CLAIM"
                    
                if ev.key == pygame.K_a:
                    self.cfg.draw_agents = not self.cfg.draw_agents
                if ev.key == pygame.K_f:
                    self.cfg.flora_fill = "stipple" if self.cfg.flora_fill == "wedge" else "wedge"
                if ev.key == pygame.K_h:
                    self.cfg.show_hud = not self.cfg.show_hud
                if ev.key == pygame.K_t:
                    order = list(FLORA_TRAITS)
                    cur = str(getattr(self.cfg, "flora_color_by", order[0]))
                    i = order.index(cur) if cur in order else 0
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
    def _ensure_projection(self, grid) -> None:
        """
        Bygg avbildningen mellan värld, pixlar och celler.

        Varje pixel slås upp mot den cell den faller i via grid.cell_of_many().
        Det gör renderingen geometriagnostisk: hexceller ritas korrekt utan att
        viewern känner till hexagoner, och en framtida geometri fungerar utan
        ändring här. Uppslaget beräknas en gång och är sedan en gather per bild.
        """
        key = (int(grid.n_cells), int(grid.width), int(grid.height), int(self.cfg.scale))
        if getattr(self, "_proj_key", None) == key:
            return

        cell_w = float(grid.extent_x) / float(grid.width)
        ppu = float(self.cfg.scale) / cell_w          # pixlar per världsenhet
        w_px = max(1, int(round(float(grid.extent_x) * ppu)))
        h_px = max(1, int(round(float(grid.extent_y) * ppu)))

        xs = (np.arange(w_px, dtype=np.float64) + 0.5) / ppu
        ys = (np.arange(h_px, dtype=np.float64) + 0.5) / ppu
        XX, YY = np.meshgrid(xs, ys)
        self._pixel_cell = np.asarray(
            grid.cell_of_many(XX.ravel(), YY.ravel()), dtype=np.int64
        ).reshape(h_px, w_px)

        self._ppu = ppu
        self._w_px = w_px
        self._h_px = h_px
        self._proj_key = key

        # Ytfördelningen inom cellen avgörs av ett tal u i [0, 1) per pixel.
        # Båda lägena är statiska för en given projektion och beräknas därför
        # en gång: per bildruta återstår ett searchsorted och en gather.
        cx = np.asarray(grid.cell_center_x, dtype=np.float64)[self._pixel_cell]
        cy = np.asarray(grid.cell_center_y, dtype=np.float64)[self._pixel_cell]
        ex, ey = float(grid.extent_x), float(grid.extent_y)
        # Cellen kan ligga på andra sidan världsranden än pixeln, eftersom
        # cell_of_many wrappar. Deltat måste därför tas kortaste vägen.
        dx = (XX - cx + 0.5 * ex) % ex - 0.5 * ex
        dy = (YY - cy + 0.5 * ey) % ey - 0.5 * ey
        self._u_wedge = (np.arctan2(dy, dx) / (2.0 * math.pi)) % 1.0

        # Stipplingen behöver ett stabilt men rumsligt okorrelerat värde per
        # pixel. En heltalshash ger samma bild varje bildruta — brus som
        # flimrar mellan rutorna är oläsbart.
        iy, ix = np.meshgrid(
            np.arange(h_px, dtype=np.uint64), np.arange(w_px, dtype=np.uint64), indexing="ij"
        )
        hsh = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663))
        hsh = (hsh ^ (hsh >> np.uint64(13))) * np.uint64(1274126177)
        self._u_stipple = ((hsh >> np.uint64(11)) & np.uint64(0xFFFFF)).astype(np.float64) / float(1 << 20)

        self._probe_cache = {}

    def _ensure_screen(self) -> None:
        """
        Se till att fönstret har världens mått.

        Storleken följer projektionen och kan därför ändras under körning —
        klienten öppnar ett litet väntefönster innan den vet något om
        världen, och får sina mått först med den första bildrutan.
        """
        want = (int(self._w_px), int(self._h_px))
        if self._screen is not None and getattr(self, "_screen_size", None) == want:
            return
        self._screen = self.pg.display.set_mode(want)
        self._screen_size = want

    def _gamma(self, x01: np.ndarray) -> np.ndarray:
        g = float(self.cfg.gamma)
        if abs(g - 1.0) < 1e-6:
            return x01
        return np.power(_clip01(x01), g, dtype=np.float32)

    @staticmethod
    def _temp_field(world, like: np.ndarray) -> np.ndarray:
        """Temperatur per cell, platt."""
        if not hasattr(world, "temperature_field"):
            return np.zeros_like(like, dtype=np.float32)
        return np.asarray(world.temperature_field(), dtype=np.float32)

    def _make_rgb(self, frame) -> np.ndarray:
        """Bakgrundsfärg per cell, (n_cells, 3) float32 i [0, 1]."""
        B01 = np.asarray(frame.flora_shoot01, dtype=np.float32)
        C01 = np.asarray(frame.detritus01, dtype=np.float32)

        mode = self.cfg.mode.upper().strip()

        if mode == "B":
            img = np.stack([B01, B01, B01], axis=-1)

        elif mode == "C":
            img = np.stack([C01, C01, C01], axis=-1)

        elif mode in ("TEMP", "FLORA", "CLAIM"):
            lo, hi = TEMP_RANGE
            t01 = np.clip(
                (np.asarray(frame.temperature, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0
            ).astype(np.float32, copy=False)

            if mode == "TEMP":
                img = np.stack([t01, t01, t01], axis=-1)
            elif mode == "CLAIM":
                # Anspråket som eget fält. Under 1 är marken ledig och visas
                # mörkt grön; över 1 är den översökt och visas i rött. Det är
                # samma tal som konkurrensen avgörs med.
                cl = np.asarray(frame.claimed, dtype=np.float32)
                free = np.clip(cl, 0.0, 1.0)
                over = np.clip(cl - 1.0, 0.0, 2.0) * 0.5
                img = np.stack([over, free * 0.75, free * 0.25 + 0.05 * t01], axis=-1)
            else:
                base = 0.08 + 0.10 * t01
                img = np.stack([0.10 * base, 0.18 * base, 0.22 * base], axis=-1)
            img = img.astype(np.float32, copy=False)

        else:  # "CB"
            Z = np.zeros_like(B01, dtype=np.float32)
            img = np.stack([C01, B01, Z], axis=-1)

        return self._gamma(img).astype(np.float32, copy=False)

    def _blit_rgb01(self, img01: np.ndarray) -> None:
        """Måla en färdig pixelbild i [0, 1]."""
        pygame = self.pg
        self._ensure_screen()
        img = _as_u8_rgb(img01)
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        self._screen.blit(surf, (0, 0))

    def _compose_flora(self, frame, base01: np.ndarray) -> np.ndarray:
        """
        Lägg ytfördelningen ovanpå bakgrunden.

        Plantorna har ingen position i cellen — modellen ger dem cellcentrum,
        och det är ärligt, för rötterna konkurrerar om cellens area och inte
        om en punkt i den. Viewern fördelar därför arean i stället för att
        hitta på koordinater: varje pixel får ett tal u i [0, 1) och slås upp
        i cellens kumulativa andelar. En planta som håller 30 % av cellen får
        30 % av dess pixlar, och den obesatta andelen 1 − claimed blir bar
        mark. Grannspillet följer med av sig självt, eftersom en planta med
        rotarea över 1 redan har en rad i varje granncell.

        Två sätt att välja u, båda samma gather:

          KILAR      pixelns vinkel kring cellcentrum. Rena kanter, läsbart
                     utzoomat, men areorna störs av hexagonens hörn.
          STIPPLING  en stabil hash av pixelindex. Areorna blir exakt
                     proportionella och bilden visar blandning i stället för
                     gränser som inte finns.

        Uppslaget görs med ett enda searchsorted över hela pixelfältet. Det
        fungerar för att nycklarna görs globalt växande: rad j i cell c får
        nyckeln c + (kumulativ andel), och varje cell avslutas med en
        vaktpost på c + 1 som representerar bar mark. Sonden c + u kan då
        aldrig hamna utanför sin egen cell.

        Det som försvinner med detta är per-planta-loopen. Den allokerade en
        pygame-yta per levande planta — vid 29 000 plantor blev det 29 000
        ytor per bildruta, och den ritade dessutom en axelriktad kvadrat på
        ett hexgrid.
        """
        starts = np.asarray(frame.claim_starts, dtype=np.int64)
        share = np.asarray(frame.claim_share, dtype=np.float64)
        n_cells = starts.shape[0] - 1
        if n_cells <= 0:
            return base01

        counts = np.diff(starts)
        n_rows = int(share.shape[0])

        # Kumulativ andel inom varje cell.
        cum = np.cumsum(share)
        if n_rows:
            # Kumulativ summa inom cellen: dra bort summan fram till cellens
            # start. Prefixet med en nolla gör att start = 0 faller ut rätt.
            before = np.concatenate(([0.0], cum))[starts[:-1]]
            cum = cum - np.repeat(before, counts)

        # Nycklar: cellindex + kumulativ andel, plus en vaktpost per cell.
        cell_of_row = np.repeat(np.arange(n_cells, dtype=np.int64), counts)
        keys = np.empty(n_rows + n_cells, dtype=np.float64)
        row_at = np.empty(n_rows + n_cells, dtype=np.int64)
        sentinel_at = starts[1:] + np.arange(1, n_cells + 1, dtype=np.int64) - 1
        is_sent = np.zeros(n_rows + n_cells, dtype=bool)
        is_sent[sentinel_at] = True
        keys[sentinel_at] = np.arange(n_cells, dtype=np.float64) + 1.0
        row_at[sentinel_at] = -1
        if n_rows:
            keys[~is_sent] = cell_of_row + np.minimum(cum, 1.0)
            row_at[~is_sent] = np.arange(n_rows, dtype=np.int64)

        u = self._u_stipple if self.cfg.flora_fill == "stipple" else self._u_wedge
        probe = self._pixel_cell.astype(np.float64) + u
        sel = np.searchsorted(keys, probe, side="left")
        np.clip(sel, 0, keys.shape[0] - 1, out=sel)
        row = row_at[sel]

        img = np.array(base01[self._pixel_cell], dtype=np.float32, copy=True)
        hit = row >= 0
        if not np.any(hit):
            return img

        r = row[hit]
        axis = FLORA_TRAITS.index(self.cfg.flora_color_by) \
            if self.cfg.flora_color_by in FLORA_TRAITS else 0
        t = np.asarray(frame.claim_trait[:, axis], dtype=np.float32)[r]
        fill = np.asarray(frame.claim_fill, dtype=np.float32)[r]

        # Kall/låg -> blå, varm/hög -> röd. Samma axel som förut.
        col = np.stack([
            0.16 + 0.70 * t,
            0.31 + 0.55 * (1.0 - np.abs(t - 0.5) * 2.0),
            0.16 + 0.70 * (1.0 - t),
        ], axis=-1).astype(np.float32)

        # Fyllnadsgraden mot vuxenmassa styr opaciteten: en groddplanta håller
        # sin mark men syns knappt, en fullvuxen dominerar sin yta.
        alpha = (0.31 + 0.69 * fill).astype(np.float32)[:, None]
        img[hit] = img[hit] * (1.0 - alpha) + col * alpha
        return img

    def _draw_agents(self, frame) -> None:
        if not self.cfg.draw_agents:
            return

        pygame = self.pg
        ppu = float(self._ppu)
        W_px, H_px = int(self._w_px), int(self._h_px)
        hl = int(self.cfg.agent_heading_len_px)
        r0 = int(self.cfg.agent_radius_px)

        px_all = (np.asarray(frame.fauna_x, dtype=np.float64) * ppu).astype(np.int64) % W_px
        py_all = (np.asarray(frame.fauna_y, dtype=np.float64) * ppu).astype(np.int64) % H_px

        for i in range(frame.fauna_n):
            px, py = int(px_all[i]), int(py_all[i])
            dmg = float(frame.fauna_damage_frac[i])
            e = float(frame.fauna_energy_frac[i])

            # Grön frisk -> röd döende, ljusstyrkan bär energin.
            v = 0.45 + 0.55 * max(0.0, min(1.0, e))
            col = _hsv_to_rgb((1.0 - max(0.0, min(1.0, dmg))) * 0.33, 0.85, v)
            radius = max(2, int(r0 * (0.7 + 0.6 * min(2.0, float(frame.fauna_mass_frac[i])))))
            pygame.draw.circle(self._screen, col, (px, py), radius)

            if bool(frame.fauna_ready[i]):
                pulse = 0.5 + 0.5 * math.sin(self._step * 0.25)
                rr = radius + 2 + int(pulse * 2)
                surf = pygame.Surface((rr * 2 + 2, rr * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (255, 220, 50, int(120 + 120 * pulse)), (rr + 1, rr + 1), rr, 2)
                self._screen.blit(surf, (px - rr - 1, py - rr - 1))

            pred = float(frame.fauna_predation[i])
            if pred > 0.15:
                pr = radius + 1
                surf = pygame.Surface((pr * 2 + 2, pr * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (220, 30, 30, int(60 + 180 * pred)),
                                   (pr + 1, pr + 1), pr, max(1, int(pred * 3)))
                self._screen.blit(surf, (px - pr - 1, py - pr - 1))

            gf = float(frame.fauna_gest_frac[i])
            if gf > 0.0:
                fr = max(1, int(radius * 0.35 + gf * radius * 0.25))
                pygame.draw.circle(self._screen, (200, int(180 + 75 * gf), 120), (px, py), fr)

            if self.cfg.draw_heading and hl > 0:
                h = float(frame.fauna_heading[i])
                dim = tuple(max(0, int(c * 0.6)) for c in col)
                ex = int(px + (radius + hl * 0.5) * math.cos(h))
                ey = int(py + (radius + hl * 0.5) * math.sin(h))
                pygame.draw.line(self._screen, dim, (px, py), (ex, ey), 1)

    def _draw_hud(self, frame) -> None:
        if not self.cfg.show_hud:
            return

        tmean = tmin = tmax = float("nan")
        tmeanN = tmeanS = float("nan")
        Tb = np.asarray(frame.T_band, dtype=np.float32)
        if Tb.size:
            # Banden är likstora, så bandstatistik är cellstatistik.
            tmean, tmin, tmax = float(np.mean(Tb)), float(np.min(Tb)), float(np.max(Tb))
            lat = np.asarray(frame.band_lat, dtype=np.float32)
            if lat.size == Tb.size:
                north = lat > 0.0
                if north.any() and (~north).any():
                    tmeanN = float(np.mean(Tb[north]))
                    tmeanS = float(np.mean(Tb[~north]))

        claimed = np.asarray(frame.claimed, dtype=np.float64)
        closed = float(np.mean(claimed >= 1.0)) * 100.0 if claimed.size else 0.0
        bare = float(np.mean(np.clip(1.0 - claimed, 0.0, 1.0))) * 100.0 if claimed.size else 0.0

        lines = [
            f"t={frame.t:8.2f}  pop={frame.fauna_n:4d}  born={frame.births_total:6d}  "
            f"dead={frame.deaths_total:6d}  mode={self.cfg.mode.upper()}  "
            f"gamma={self.cfg.gamma:.2f}  {'PAUSED' if self._paused else ''}"
        ]
        if math.isfinite(tmean):
            l2 = f"T(mean/min/max)={tmean:5.1f}/{tmin:5.1f}/{tmax:5.1f}"
            if math.isfinite(tmeanN):
                l2 += f"   T(N/S)={tmeanN:5.1f}/{tmeanS:5.1f}"
            lines.append(l2)
        else:
            lines.append("T(mean/min/max)=NA")

        lines.append(
            f"flora_n={frame.flora_n:5d}  flora_mass={frame.flora_mass:.4f} kg  "
            f"mark: bar={bare:4.1f}%  slutna celler={closed:4.1f}%"
        )
        lines.append(
            f"grön=frisk→röd=döende  ljus=energi  gul ring=parningsredo  "
            f"[yta {self.cfg.flora_fill} F]  [trait {self.cfg.flora_color_by} T]"
        )

        ft = frame.flora_summary
        if ft:
            lines.append(
                f"flora: a={ft.get('flora_mean_repro_alloc', float('nan')):.3f}  "
                f"M*={ft.get('flora_mean_adult_mass', float('nan')):.4f}  "
                f"Topt={ft.get('flora_mean_temp_opt', float('nan')):.1f}  "
                f"Tw={ft.get('flora_mean_temp_width', float('nan')):.1f}  "
                f"d={ft.get('flora_mean_apparatus', float('nan')):.4f}"
            )

        for i, text in enumerate(lines):
            y = 5 + i * 18
            self._screen.blit(self._font.render(text, True, (0, 0, 0)), (6, y + 1))
            self._screen.blit(self._font.render(text, True, (255, 255, 255)), (5, y))

    def update(self, frame, grid=None) -> bool:
        """
        Rita en bildruta.

        `frame` är en ViewFrame. `grid` kan skickas med om anroparen redan
        har ett — annars byggs det ur bildrutans width/height, vilket är det
        som gör viewern körbar utan simulering.

        Returnerar False om användaren stängt fönstret.
        """
        self._step += 1
        if not self._handle_events():
            return False
        if self.cfg.render_every > 1 and (self._step % self.cfg.render_every != 0):
            self._throttle()
            return True

        if grid is None:
            grid = self._grid_for(frame)
        self._ensure_projection(grid)
        self._ensure_screen()

        base = self._make_rgb(frame)
        if self.cfg.mode.upper().strip() == "FLORA":
            img = self._compose_flora(frame, base)
        else:
            img = base[self._pixel_cell]
        self._blit_rgb01(img)

        self._draw_agents(frame)
        self._draw_hud(frame)

        self.pg.display.flip()
        self._throttle()
        return True

    def _grid_for(self, frame):
        """Rekonstruera geometrin ur bildrutan. Grid härleder resten själv."""
        key = (int(frame.grid_width), int(frame.grid_height))
        if getattr(self, "_grid_key", None) != key:
            from grid import Grid
            self._grid = Grid(width=key[0], height=key[1])
            self._grid_key = key
        return self._grid

    def close(self) -> None:
        self.pg.quit()