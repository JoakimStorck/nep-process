from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from viewframe import (FLORA_TRAITS, TEMP_RANGE,
                       WET_CHANNEL, WET_LAKE, WET_LAND, WET_SEA)

# Bakgrund där fönstret sträcker sig utanför världen. Mörk men inte svart:
# den ska gå att skilja från en cell utan liv i den.
OUTSIDE_RGB = np.array([0.055, 0.065, 0.080], dtype=np.float32)


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

    # Sektorperceptet som kilar kring varje djur: vad djuret ser åt varje håll.
    #
    # **Två kanaler, inte en.** 1014 lade hela informationen i opaciteten, och
    # det var fel: med sex sektorer får en jämn profil en sjättedel av skalan
    # var, alltså allt i den blekaste sjättedelen, och en måttligt toppig profil
    # skiljer sig med några få steg av 255. Regeln som skulle bevara bläcket
    # komprimerade bort dynamiken. Skärmbilderna visade det direkt — sex jämnt
    # bleka kilar oavsett vad djuret stod på.
    #
    # Nu bär **radien** fördelningen och **opaciteten** mängden:
    #
    #   `r_k = R · sqrt(andel_k · S)` ger arean proportionell mot andelen, så
    #   den totala arean är `π R²` oavsett form. Bläcket är fortfarande bevarat,
    #   men i geometrin i stället för i alfakanalen — och en form läses av ögat
    #   på en tiondels sekund där fyra procents alfaskillnad inte läses alls.
    #   En jämn profil blir en cirkel, en toppig en lob.
    #
    #   Opaciteten bär profilens medelvärde, alltså hur mycket föda som finns
    #   runt djuret över huvud taget. Den informationen fanns inte i bilden
    #   tidigare, och den är precis vad man vill se bredvid riktningen.
    #
    # `R` är djurets **verkliga synvidd**, inte ett pixeltal. En halo som inte
    # går att lägga mot terrängen säger bara att djuret ser något.
    #
    # Det är perceptet som ritas och inte arbitreringens uttryck. 1013 mätte att
    # det senare är en cosinus kring ett `argmax` som redan tagits.
    show_dir: bool = False
    # Synhorisonten som tunn ring, så att kilarnas skala går att läsa av.
    dir_ring: bool = True

    # Modes:
    #   CB    : RGB=(C,B,0)
    #   B/C   : grayscale single field
    #   TEMP  : grayscale temperature
    #   FLORA : ytfördelningen ovanpå neutral bakgrund
    #   CLAIM : anspråket per cell, bar mark mot översökt mark
    #   TERRANG : höjden med backljus, hav och sjöar efter djup
    #   VATTEN  : markfukten, med vattendrag och sjöar ovanpå
    mode: str = "FLORA"

    # Vattnet ritas över bakgrunden i alla lägen. Utan det ser en karta i
    # FLORA-läge ut som om havet vore torr mark utan växtlighet, vilket är
    # precis vad det inte är.
    show_water: bool = True

    # Höjdkurvor som på en orienteringskarta: bruna linjer med jämn
    # ekvidistans, var femte som tjockare huvudkurva. De ritas ovanpå vattnet
    # och inte under, eftersom hela poängen är att kunna läsa av **på vilken
    # höjd vattnet ligger** — var en sjöyta möter en kurva.
    show_contours: bool = False
    # Ekvidistans i meter. 0 betyder att den väljs ur världens egen relief, så
    # att kartan får ungefär tolv kurvor oavsett hur kuperad den är.
    contour_step_m: float = 0.0

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

        # Sätts av en fjärrklient. Är den satt går mellanslag till servern
        # i stället för att pausa den lokala renderingen: i fjärrläget finns
        # ingen lokal simulering att pausa, och en tangent som ser ut att
        # göra något men inte gör det är värre än ingen tangent.
        self.on_command = None

        # Backljuset beror bara på höjden, som är statisk. Cachen sparar ett
        # svep över grannmatrisen per bildruta i terränglägena.
        self._shade_key = None
        self._shade_cache = np.ones(0, dtype=np.float32)
        self._contour_key = None
        self._contour_cache = None

        # Utsnitt. Av som förval: fönstret är då hela världen gånger `scale`,
        # vilket ögonblicksbilderna förlitar sig på.
        self._viewport = False
        self._win_size = (0, 0)
        self._ppu_view = None
        self._view_cx = 0.0
        self._view_cy = 0.0
        self._drag = None
        self._pixel_valid = None

        self._screen = None
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Menlo", 14)

    # ---------- input ----------
    def _handle_events(self) -> bool:
        pygame = self.pg
        grid = getattr(self, "_grid", None)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False

            if ev.type == pygame.VIDEORESIZE and self._viewport:
                self._win_size = (max(160, ev.w), max(120, ev.h))
                self._screen = pygame.display.set_mode(self._win_size, pygame.RESIZABLE)
                self._screen_size = self._win_size

            if self._viewport and grid is not None:
                if ev.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    self.zoom_at(mx, my, 1.15 ** ev.y, grid)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self._drag = ev.pos
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self._drag = None
                elif ev.type == pygame.MOUSEMOTION and self._drag is not None:
                    self.pan_px(ev.rel[0], ev.rel[1], grid)
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE or ev.key == pygame.K_q:
                    return False
                if self._viewport and grid is not None:
                    step = max(8, int(0.1 * self._win_size[1]))
                    if ev.key == pygame.K_LEFT:
                        self.pan_px(step, 0, grid)
                    if ev.key == pygame.K_RIGHT:
                        self.pan_px(-step, 0, grid)
                    if ev.key == pygame.K_UP:
                        self.pan_px(0, step, grid)
                    if ev.key == pygame.K_DOWN:
                        self.pan_px(0, -step, grid)
                    if ev.key == pygame.K_0:
                        self.fit_to_world(grid)
                    if ev.key in (pygame.K_PERIOD, pygame.K_COMMA):
                        c = (self._win_size[0] // 2, self._win_size[1] // 2)
                        self.zoom_at(c[0], c[1], 1.3 if ev.key == pygame.K_PERIOD else 1 / 1.3, grid)

                if ev.key == pygame.K_SPACE:
                    if self.on_command is not None:
                        self.on_command("toggle_pause")
                    else:
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
                if ev.key == pygame.K_7:
                    self.cfg.mode = "TERRANG"
                if ev.key == pygame.K_8:
                    self.cfg.mode = "VATTEN"
                if ev.key == pygame.K_w:
                    self.cfg.show_water = not self.cfg.show_water
                if ev.key == pygame.K_k:
                    self.cfg.show_contours = not self.cfg.show_contours
                    
                if ev.key == pygame.K_a:
                    self.cfg.draw_agents = not self.cfg.draw_agents
                if ev.key == pygame.K_f:
                    self.cfg.flora_fill = "stipple" if self.cfg.flora_fill == "wedge" else "wedge"
                if ev.key == pygame.K_h:
                    self.cfg.show_hud = not self.cfg.show_hud
                if ev.key == pygame.K_d:
                    self.cfg.show_dir = not self.cfg.show_dir
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
    # ---------- utsnitt ----------
    def fit_to_world(self, grid) -> None:
        """Ställ utsnittet så att hela världen ryms i fönstret."""
        ww, wh = self._win_size
        self._ppu_view = min(ww / float(grid.extent_x), wh / float(grid.extent_y))
        self._view_cx = 0.5 * float(grid.extent_x)
        self._view_cy = 0.5 * float(grid.extent_y)

    def enable_viewport(self, win_w: int, win_h: int) -> None:
        """
        Frikoppla fönstret från världen.

        Utan detta är fönstret alltid världen gånger `scale`, vilket är rätt
        för ögonblicksbilder — de ska vara jämförbara mellan körningar — men
        oanvändbart interaktivt: en 64x256-värld blir högre än skärmen långt
        innan cellerna blir läsbara.
        """
        self._viewport = True
        self._win_size = (int(win_w), int(win_h))
        self._ppu_view = None      # sätts av fit_to_world vid första bildrutan

    def zoom_at(self, px: int, py: int, factor: float, grid) -> None:
        """
        Zooma kring en punkt i fönstret.

        Punkten under pekaren ska ligga still. Utan det glider bilden undan
        medan man zoomar och man tappar det man siktade på.
        """
        if not self._viewport or self._ppu_view is None:
            return
        ww, wh = self._win_size
        wx, wy = self._px_to_world(px, py)
        lo = 0.25 * min(ww / float(grid.extent_x), wh / float(grid.extent_y))
        self._ppu_view = float(np.clip(self._ppu_view * factor, lo, 400.0))
        wx2, wy2 = self._px_to_world(px, py)
        self._view_cx += wx - wx2
        self._view_cy += wy - wy2
        self._clamp_view(grid)

    def pan_px(self, dx: int, dy: int, grid) -> None:
        if not self._viewport or self._ppu_view is None:
            return
        self._view_cx -= dx / self._ppu_view
        self._view_cy -= dy / self._ppu_view
        self._clamp_view(grid)

    def _clamp_view(self, grid) -> None:
        """
        Håll utsnittet innanför världen.

        Världen är toroidal och skulle kunna kaklas i all oändlighet, men det
        är inte vad ett fönster är. Utsnittet klampas därför mot världens
        kanter, och när världen är mindre än fönstret centreras den i stället
        — att panorera i tomrum är bara ett sätt att tappa bort sig.
        """
        ww, wh = self._win_size
        for axis, extent, win in (("x", float(grid.extent_x), ww), ("y", float(grid.extent_y), wh)):
            half = 0.5 * win / self._ppu_view
            name = "_view_c" + axis
            if 2.0 * half >= extent:
                setattr(self, name, 0.5 * extent)
            else:
                setattr(self, name, float(np.clip(getattr(self, name), half, extent - half)))

    # Under så här många pixlar per cellbredd är en plantas kil inte
    # upplösbar, och då är det slöseri att be servern packa raderna. Vid
    # fyra pixlar och tolv plantor i cellen ritas de ovanpå varandra.
    DETAIL_MIN_PPU = 6.0

    def view_request(self, grid) -> dict | None:
        """
        Det synliga utsnittet, som servern kan packa anspråksrader för.

        `detail: False` när cellerna är för små för att kilarna ska synas —
        då räcker celltäckningen, och bildrutan går från tiotals megabyte till
        ett par. Marginalen på tjugo procent gör att en panorering inte
        blottar oritad mark innan nästa utsnitt hunnit fram.
        """
        if not self._viewport or self._ppu_view is None:
            return None
        ppu = float(self._ppu_view) * float(getattr(grid, "COL_SPACING", 1.0))
        if ppu < self.DETAIL_MIN_PPU:
            return {"cmd": "view", "detail": False}
        ww, wh = self._win_size
        return {
            "cmd": "view",
            "detail": True,
            "cx": float(self._view_cx),
            "cy": float(self._view_cy),
            "hw": 0.6 * ww / float(self._ppu_view),
            "hh": 0.6 * wh / float(self._ppu_view),
        }

    def _px_to_world(self, px: int, py: int) -> tuple[float, float]:
        ww, wh = self._win_size
        return (
            self._view_cx + (px - 0.5 * ww) / self._ppu_view,
            self._view_cy + (py - 0.5 * wh) / self._ppu_view,
        )

    # ---------- rendering ----------
    def _ensure_projection(self, grid) -> None:
        """
        Bygg avbildningen mellan värld, pixlar och celler.

        Varje pixel slås upp mot den cell den faller i via grid.cell_of_many().
        Det gör renderingen geometriagnostisk: hexceller ritas korrekt utan att
        viewern känner till hexagoner, och en framtida geometri fungerar utan
        ändring här. Uppslaget beräknas en gång per utsnitt och är sedan en
        gather per bild.

        Två lägen delar samma kod. Utan utsnitt är fönstret hela världen gånger
        `scale` — det är ögonblicksbildernas läge och ger bit för bit samma
        resultat som förut. Med utsnitt bestämmer fönstret storleken och
        `_ppu_view` hur mycket av världen som ryms.
        """
        if self._viewport:
            if self._ppu_view is None:
                self.fit_to_world(grid)
            w_px, h_px = self._win_size
            ppu = float(self._ppu_view)
            x0 = self._view_cx - 0.5 * w_px / ppu
            y0 = self._view_cy - 0.5 * h_px / ppu
            key = (int(grid.n_cells), int(grid.width), int(grid.height),
                   w_px, h_px, round(ppu, 6), round(x0, 6), round(y0, 6))
        else:
            cell_w = float(grid.extent_x) / float(grid.width)
            ppu = float(self.cfg.scale) / cell_w
            w_px = max(1, int(round(float(grid.extent_x) * ppu)))
            h_px = max(1, int(round(float(grid.extent_y) * ppu)))
            x0 = y0 = 0.0
            key = (int(grid.n_cells), int(grid.width), int(grid.height), int(self.cfg.scale))

        if getattr(self, "_proj_key", None) == key:
            return

        xs = x0 + (np.arange(w_px, dtype=np.float64) + 0.5) / ppu
        ys = y0 + (np.arange(h_px, dtype=np.float64) + 0.5) / ppu
        XX, YY = np.meshgrid(xs, ys)
        self._pixel_cell = np.asarray(
            grid.cell_of_many(XX.ravel(), YY.ravel()), dtype=np.int64
        ).reshape(h_px, w_px)

        # cell_of_many wrappar, så en pixel utanför världen får ändå en cell —
        # den på andra sidan. Det är rätt för en torus men fel för ett
        # fönster: världen ska ritas en gång och resten vara tom. Masken är
        # separabel i x och y, så den kostar två jämförelser per axel.
        if self._viewport:
            ok_x = (xs >= 0.0) & (xs < float(grid.extent_x))
            ok_y = (ys >= 0.0) & (ys < float(grid.extent_y))
            self._pixel_valid = None if (ok_x.all() and ok_y.all()) else np.outer(ok_y, ok_x)
        else:
            self._pixel_valid = None

        self._ppu = ppu
        self._w_px = w_px
        self._h_px = h_px
        self._proj_key = key

        # Ytfördelningen inom cellen avgörs av ett tal u i [0, 1) per pixel.
        # Fälten byggs inte här utan vid första användningen: tillsammans
        # kostar de 21 ms av projektionens 56 vid ett fönster på 360x1000,
        # och i alla lägen utom FLORA används inget av dem alls. Under en
        # dragning byggs projektionen om varje bildruta, så det är skillnaden
        # mellan att kunna panorera och att inte kunna det.
        self._u_cache = {}
        self._proj_x0 = x0
        self._proj_y0 = y0

    def _pixel_world(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self._proj_x0 + (np.arange(self._w_px, dtype=np.float64) + 0.5) / self._ppu
        ys = self._proj_y0 + (np.arange(self._h_px, dtype=np.float64) + 0.5) / self._ppu
        return np.meshgrid(xs, ys)

    def _u_field(self, kind: str, grid) -> np.ndarray:
        """Pixelns plats i cellens andelsordning. Byggs en gång per projektion."""
        cached = self._u_cache.get(kind)
        if cached is not None:
            return cached

        if kind == "wedge":
            XX, YY = self._pixel_world()
            cx = np.asarray(grid.cell_center_x, dtype=np.float64)[self._pixel_cell]
            cy = np.asarray(grid.cell_center_y, dtype=np.float64)[self._pixel_cell]
            ex, ey = float(grid.extent_x), float(grid.extent_y)
            # Cellen kan ligga på andra sidan världsranden än pixeln, eftersom
            # cell_of_many wrappar. Deltat måste tas kortaste vägen.
            dx = (XX - cx + 0.5 * ex) % ex - 0.5 * ex
            dy = (YY - cy + 0.5 * ey) % ey - 0.5 * ey
            u = (np.arctan2(dy, dx) / (2.0 * math.pi)) % 1.0
        else:
            # Hashen tar pixelns läge i världen och inte i fönstret, så att
            # mönstret sitter fast i marken under panorering — ett brus som
            # glider med kameran läses som rörelse i beståndet.
            iy, ix = np.meshgrid(
                np.arange(self._h_px, dtype=np.int64),
                np.arange(self._w_px, dtype=np.int64),
                indexing="ij",
            )
            ix = (ix + int(round(self._proj_x0 * self._ppu))).astype(np.uint64)
            iy = (iy + int(round(self._proj_y0 * self._ppu))).astype(np.uint64)
            h = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663))
            h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
            u = ((h >> np.uint64(11)) & np.uint64(0xFFFFF)).astype(np.float64) / float(1 << 20)

        self._u_cache[kind] = u
        return u

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
        flags = self.pg.RESIZABLE if self._viewport else 0
        self._screen = self.pg.display.set_mode(want, flags)
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

    def _hillshade(self, frame, grid) -> np.ndarray:
        """
        Backljus per cell ur grannmatrisen, cachat på bildrutans höjdfält.

        Geometriagnostiskt: lutningen tas som skillnaden mot grannarnas medel,
        viktad så att ena halvan av grannringen lyser upp och andra skuggar.
        Ingen kod här vet vad en rad eller en kolumn är — bara att `Grid` har
        sex grannar i en bestämd ordning.

        Höjden detrendas mot latitudbandet innan skuggan räknas. Utan det
        dominerar den kontinentala lutningen bilden fullständigt och reliefen
        syns inte alls: lutningen är tre höjdenheter mot den lokala reliefens
        knappa halva.
        """
        z = np.asarray(frame.elevation01, dtype=np.float32)
        if z.size == 0:
            return np.ones(0, dtype=np.float32)
        key = (int(z.shape[0]), float(z[::max(1, z.size // 97)].sum()))
        if self._shade_key == key:
            return self._shade_cache

        idx = np.asarray(grid.neighbor_idx, dtype=np.int64)
        bands = np.asarray(grid.bands_of_cells(np.arange(z.shape[0])), dtype=np.int64)
        nb_count = np.bincount(bands, minlength=int(grid.n_bands))
        bmean = np.bincount(bands, weights=z.astype(np.float64),
                            minlength=int(grid.n_bands)) / np.maximum(1, nb_count)
        d = z.astype(np.float64) - bmean[bands]

        # Grannordningen i Grid är väst, öst, NV, NO, SV, SO. Ljuset kommer
        # från nordväst, alltså lyser den sida som vetter dit.
        lit = (d[idx[:, 2]] + d[idx[:, 0]]) * 0.5
        dark = (d[idx[:, 5]] + d[idx[:, 1]]) * 0.5
        rel = (lit - dark)
        # Normeringen mot 95:e percentilen gör styrkan oberoende av hur
        # kuperad världen är, och det klippta intervallet håller den från att
        # blåsa ut bilden. Första försöket gick till 1,75 och gjorde
        # terränglaget vitt där det var ljust.
        scale = float(np.percentile(np.abs(rel), 95)) if rel.size else 0.0
        if scale < 1e-9:
            sh = np.ones_like(z)
        else:
            sh = np.clip(1.0 + 0.55 * (rel / scale), 0.55, 1.35).astype(np.float32)

        self._shade_key = key
        self._shade_cache = sh
        return sh

    # Ekvidistanser som får stå på en karta. En kurva var 3,7 meter är
    # ingenting en människa läser av; talen nedan är de en orienteringskarta
    # eller en fjällkarta faktiskt använder.
    CONTOUR_STEPS_M = (0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0)

    def _contour_hud(self, frame) -> str:
        """Ekvidistansen i HUD:en. Utan den går kurvorna inte att läsa av."""
        if not self.cfg.show_contours:
            return ""
        ut = self._contour_cache
        if ut is None:
            return "  [kurvor K]"
        return f"  [kurvor {float(ut[3]):g} m K]"

    def _contours(self, frame, grid):
        """
        Höjdkurvor som **linjesegment i världskoordinater**, inte som färgade
        celler.

        Höjden är ett sampel vid cellcentrum och inte ett styckvis konstant
        fält — den tolkningen är vad som gör en kurva mellan celler
        meningsfull, och den gör linjen oberoende av cellstorleken. Och
        hexgittrets **dual är ett triangelnät**: tre inbördes grannar spänner
        en triangel, inom vilken linjär interpolation är entydig. Nivåkurvan
        blir ett rakt segment per triangel, utan marching squares tvetydiga
        fall.

        Två trianglar per cell täcker planet: (i, öst, sydost) och
        (i, sydost, sydväst).

        Segmenten är statiska med terrängen och räknas en gång per värld.
        Returnerar (p0, p1, huvud, step) med punkterna i världskoordinater.
        """
        z01 = np.asarray(frame.elevation01, dtype=np.float64)
        if z01.size == 0 or grid is None:
            return None
        lo = float(frame.elev_min)
        hi = float(frame.elev_max)
        z_m = (lo + z01 * (hi - lo)) * 10.0

        step = float(self.cfg.contour_step_m)
        if step <= 0.0:
            relief = float(z_m.max() - z_m.min())
            mal = max(1e-6, relief / 12.0)
            step = min(self.CONTOUR_STEPS_M, key=lambda v: abs(math.log(v / mal)))

        key = (int(z01.size), float(z01[::max(1, z01.size // 97)].sum()), step)
        if self._contour_key == key:
            return self._contour_cache

        idx = np.asarray(grid.neighbor_idx, dtype=np.int64)
        cx = np.asarray(grid.cell_center_x, dtype=np.float64)
        cy = np.asarray(grid.cell_center_y, dtype=np.float64)
        Lx = float(grid.extent_x)
        Ly = float(grid.extent_y)
        n = z_m.size
        alla = np.arange(n)

        # Grannordningen i Grid är väst, öst, NV, NO, SV, SO.
        O, SV, SO = idx[:, 1], idx[:, 4], idx[:, 5]
        trianglar = (np.stack([alla, O, SO], axis=1),
                     np.stack([alla, SO, SV], axis=1))

        p0s, p1s, huvuds = [], [], []
        niv_lo = math.floor(float(z_m.min()) / step) + 1
        niv_hi = math.ceil(float(z_m.max()) / step)

        def punkt(ia, ib, lvl):
            za, zb = z_m[ia], z_m[ib]
            d = zb - za
            t = np.clip((lvl - za) / np.where(np.abs(d) < 1e-12, 1e-12, d), 0.0, 1.0)
            return np.stack([cx[ia] + t * (cx[ib] - cx[ia]),
                             cy[ia] + t * (cy[ib] - cy[ia])], axis=1)

        for tri in trianglar:
            a_, b_, c_ = tri[:, 0], tri[:, 1], tri[:, 2]
            # Trianglar som wrappar över sömmen hoppas över: ett segment ritat
            # rakt över världen skulle se ut som ett fel.
            ok = ((np.abs(cx[b_] - cx[a_]) < 0.5 * Lx) & (np.abs(cy[b_] - cy[a_]) < 0.5 * Ly)
                  & (np.abs(cx[c_] - cx[a_]) < 0.5 * Lx) & (np.abs(cy[c_] - cy[a_]) < 0.5 * Ly))
            hj = np.stack([a_[ok], b_[ok], c_[ok]], axis=1)
            zt = z_m[hj]
            for k in range(int(niv_lo), int(niv_hi) + 1):
                lvl = k * step
                over = zt > lvl
                antal = over.sum(axis=1)
                w = np.flatnonzero((antal == 1) | (antal == 2))
                if w.size == 0:
                    continue
                ov = over[w]
                # Det ensamma hörnet: det som ligger på andra sidan nivån än de
                # två övriga. Segmentet går mellan de två kanterna ut från det.
                ensam = np.where(ov.sum(axis=1) == 1, np.argmax(ov, axis=1),
                                 np.argmin(ov, axis=1))
                rad = np.arange(w.size)
                hh = hj[w]
                i_e = hh[rad, ensam]
                i_1 = hh[rad, (ensam + 1) % 3]
                i_2 = hh[rad, (ensam + 2) % 3]
                p0s.append(punkt(i_e, i_1, lvl))
                p1s.append(punkt(i_e, i_2, lvl))
                huvuds.append(np.full(w.size, k % 5 == 0, dtype=bool))

        if not p0s:
            self._contour_key = key
            self._contour_cache = None
            return None
        ut = (np.concatenate(p0s), np.concatenate(p1s), np.concatenate(huvuds), step)
        self._contour_key = key
        self._contour_cache = ut
        return ut

    def _draw_contours(self, frame) -> None:
        """
        Rita kurvorna som linjer, efter att cellbilden blittats.

        De hamnar **ovanpå vattnet**, eftersom en kurva som löper ut i en sjö
        visar vilken nivå sjöytan står på — hela poängen med att ha dem.
        """
        if not self.cfg.show_contours or not bool(frame.has_terrain):
            return
        ut = self._contours(frame, getattr(self, "_grid", None))
        if ut is None or self._screen is None:
            return
        p0, p1, huvud, _step = ut
        pygame = self.pg

        ppu = float(self._ppu)
        x0 = (p0[:, 0] - self._proj_x0) * ppu
        y0 = (p0[:, 1] - self._proj_y0) * ppu
        x1 = (p1[:, 0] - self._proj_x0) * ppu
        y1 = (p1[:, 1] - self._proj_y0) * ppu
        vis = ((np.maximum(x0, x1) >= -2) & (np.minimum(x0, x1) <= self._w_px + 2)
               & (np.maximum(y0, y1) >= -2) & (np.minimum(y0, y1) <= self._h_px + 2))
        if not vis.any():
            return

        fin = (74, 42, 14)
        grov = (52, 28, 8)
        bredd_grov = max(2, int(round(ppu * 0.3)))
        for mask, col, bredd in ((vis & ~huvud, fin, 1), (vis & huvud, grov, bredd_grov)):
            xs0, ys0, xs1, ys1 = x0[mask], y0[mask], x1[mask], y1[mask]
            for j in range(xs0.size):
                pygame.draw.line(self._screen, col,
                                 (int(xs0[j]), int(ys0[j])),
                                 (int(xs1[j]), int(ys1[j])), bredd)

    def _water_overlay(self, img: np.ndarray, frame) -> np.ndarray:
        """
        Måla hav, sjöar och vattendrag ovanpå en färdig bakgrund.

        Klassningen kommer från världen som `wet_kind`; viewern väljer bara
        färg. Havet mörknar med djupet, sjöar är en enda ton, och vattendrag
        får den ljusaste — de är smala och behöver sticka ut mot allt annat.
        """
        wet = np.asarray(frame.wet_kind)
        if wet.size != img.shape[0]:
            return img
        z = np.asarray(frame.elevation01, dtype=np.float32)
        out = img.copy()

        sea = wet == WET_SEA
        if sea.any():
            zs = z[sea]
            deep = 1.0 - (zs - zs.min()) / max(1e-9, float(zs.max() - zs.min()))
            out[sea, 0] = 0.04 + 0.06 * (1.0 - deep)
            out[sea, 1] = 0.16 + 0.22 * (1.0 - deep)
            out[sea, 2] = 0.36 + 0.42 * (1.0 - deep)
        lake = wet == WET_LAKE
        out[lake] = np.array([0.13, 0.42, 0.78], dtype=np.float32)
        ch = wet == WET_CHANNEL
        out[ch] = np.array([0.24, 0.60, 0.92], dtype=np.float32)
        return out

    def _make_rgb(self, frame, grid=None) -> np.ndarray:
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

        elif mode in ("TERRANG", "TERRÄNG", "VATTEN") and bool(frame.has_terrain):
            if mode == "VATTEN":
                # Markfukten som färgaxel: torrt är sandgult, blött är grönt.
                # Det är det tal florans tillväxt faktiskt läser, så bilden
                # visar bördigheten och inte en proxy för den.
                s01 = np.asarray(frame.soil01, dtype=np.float32)
                img = np.stack([0.72 - 0.42 * s01,
                                0.60 - 0.06 * s01,
                                0.34 - 0.10 * s01], axis=-1).astype(np.float32)
            else:
                z = np.asarray(frame.elevation01, dtype=np.float32)
                # Sträck kontrasten över land: havet tar annars halva skalan
                # och kontinenten blir en enda ton.
                lo = float(np.quantile(z, 0.20)) if z.size else 0.0
                t = np.clip((z - lo) / max(1e-9, 1.0 - lo), 0.0, 1.0)
                img = np.stack([0.26 + 0.46 * t,
                                0.32 + 0.34 * t ** 1.3,
                                0.20 + 0.34 * t ** 3], axis=-1).astype(np.float32)
            if grid is not None:
                # Backljuset hör till höjdlaget. I VATTEN-läget är fukten
                # storheten som ska läsas, och en skugga ovanpå den gör den
                # svårare att jämföra mellan två platser — det är samma skäl
                # som att en karta inte reliefskuggar sin temperaturskala.
                sh = self._hillshade(frame, grid)
                if mode == "VATTEN":
                    sh = 1.0 + 0.35 * (sh - 1.0)
                img = img * sh[:, None]
            img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

        else:  # "CB"
            Z = np.zeros_like(B01, dtype=np.float32)
            img = np.stack([C01, B01, Z], axis=-1)

        if self.cfg.show_water and bool(frame.has_terrain):
            img = self._water_overlay(img, frame)

        return self._gamma(img).astype(np.float32, copy=False)

    def _blit_rgb01(self, img01: np.ndarray) -> None:
        """Måla en färdig pixelbild i [0, 1]."""
        pygame = self.pg
        self._ensure_screen()
        img = _as_u8_rgb(img01)
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        self._screen.blit(surf, (0, 0))

    def _claim_layout(self, frame):
        """
        Lägg ut cellernas ytfördelning som kilar och returnera uppslaget.

        Returnerar (keys, vals, row_src): `keys` är segmentens startvinklar som
        globalt växande tal `cell + vinkel`, `vals` raden segmentet tillhör
        (-1 för bar mark) och `row_src` avbildningen tillbaka till bildrutans
        radordning. Uppslaget görs sedan med ett searchsorted över hela
        pixelfältet.
        """
        starts = np.asarray(frame.claim_starts, dtype=np.int64)
        share = np.asarray(frame.claim_share, dtype=np.float64)
        n_cells = starts.shape[0] - 1
        if n_cells <= 0:
            return None

        counts = np.diff(starts)
        n_rows = int(share.shape[0])
        cell_of_row = np.repeat(np.arange(n_cells, dtype=np.int64), counts)

        cdir = np.asarray(getattr(frame, "claim_dir", np.full(n_rows, -1.0)), dtype=np.float64)
        if cdir.shape[0] != n_rows:
            cdir = np.full(n_rows, -1.0)
        is_spill = cdir >= -0.5

        # Ordna raderna inom cellen: grannrader först, i riktningsordning, och
        # cellens egna plantor efter. Det gör att flera grannkilar hamnar i
        # rätt inbördes vridning kring cellen.
        order = np.lexsort((np.where(is_spill, cdir, 2.0), cell_of_row))
        share = share[order]
        cdir = cdir[order]
        is_spill = is_spill[order]
        row_src = order

        # Startvinkel för varje rads kil, räknat från cellens egen nollpunkt.
        cum = np.cumsum(share)
        before = np.concatenate(([0.0], cum))[starts[:-1]]
        seg_start = cum - share - np.repeat(before, counts)

        used = np.bincount(cell_of_row, weights=share, minlength=n_cells)[:n_cells]
        bare = np.clip(1.0 - used, 0.0, 1.0)

        # Vrid hela cellens kilkrans så att den grannrad som håller mest yta
        # får sin kil centrerad rakt mot moderplantans cell. Med en enda
        # grannrad blir det exakt; med flera pekar den största rätt och
        # resten ligger i rätt ordning kring den. Det är skillnaden mellan
        # att se sex lösryckta kilar och att se en planta som sträcker sig
        # ut över sina grannar.
        phi = np.zeros(n_cells)
        si = np.flatnonzero(is_spill)
        if si.size:
            # Den grannrad som håller mest yta i cellen får styra vridningen.
            # Sortera på (cell, andel) och ta den sista i varje cellgrupp.
            o = np.lexsort((share[si], cell_of_row[si]))
            si = si[o]
            c = cell_of_row[si]
            pick = si[np.append(np.flatnonzero(np.diff(c)), c.size - 1)]
            phi[cell_of_row[pick]] = (
                cdir[pick] - seg_start[pick] - 0.5 * share[pick]
            ) % 1.0

        # Segmenten är raderna plus cellens bara mark, som ett eget segment
        # sist i varje cell.
        seg_cell = np.concatenate([cell_of_row, np.arange(n_cells, dtype=np.int64)])
        seg_row = np.concatenate([np.arange(n_rows, dtype=np.int64), np.full(n_cells, -1)])
        seg_off = np.concatenate([seg_start, used])
        seg_w = np.concatenate([share, bare])

        keep = seg_w > 1e-9
        seg_cell, seg_row, seg_off, seg_w = seg_cell[keep], seg_row[keep], seg_off[keep], seg_w[keep]

        # Absolut vinkel för segmentets början, efter vridningen.
        b = (phi[seg_cell] + seg_off) % 1.0

        # Nyckeln cell + vinkel är ett enda växande tal, så en argsort räcker
        # — ingen lexsort behövs för att gruppera per cell.
        keys = seg_cell + b
        srt = np.argsort(keys, kind="stable")
        keys, vals, seg_cell = keys[srt], seg_row[srt], seg_cell[srt]
        if keys.size == 0:
            return None

        # Ett segment per cell sträcker sig över nollpunkten: det med störst
        # startvinkel, alltså gruppens sista. Det måste finnas med en gång
        # till vid vinkel 0, annars blir en tårtbit per cell osynlig.
        gstart = np.concatenate(([0], np.flatnonzero(np.diff(seg_cell)) + 1))
        gend = np.append(gstart[1:] - 1, keys.size - 1)
        keys = np.insert(keys, gstart, seg_cell[gend].astype(np.float64))
        vals = np.insert(vals, gstart, vals[gend])


        return keys, vals, row_src

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
        lay = self._claim_layout(frame)
        if lay is None:
            return base01
        keys, vals, row_src = lay

        kind = "stipple" if self.cfg.flora_fill == "stipple" else "wedge"
        probe = self._pixel_cell.astype(np.float64) + self._u_field(kind, self._grid)
        sel = np.searchsorted(keys, probe, side="right") - 1
        np.clip(sel, 0, keys.shape[0] - 1, out=sel)
        row = vals[sel]

        img = np.array(base01[self._pixel_cell], dtype=np.float32, copy=True)
        hit = row >= 0
        if not np.any(hit):
            return img

        r = row_src[row[hit]]

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

    def _screen_positions(self, wx, wy):
        """
        Världskoordinater till fönsterkoordinater.

        Wrappen hör till världen och inte till fönstret: origo dras av först,
        och därefter finns inga kopior. Världen ritas en gång, så ett djur har
        en plats. Djur utanför utsnittet faller bort här i stället för att
        ritas utanför fönsterkanten.
        """
        g = getattr(self, "_grid", None)
        wx = np.asarray(wx, dtype=np.float64)
        wy = np.asarray(wy, dtype=np.float64)
        if g is None or wx.size == 0:
            return []

        ppu = float(self._ppu)
        sx = (wx - self._proj_x0) * ppu
        sy = (wy - self._proj_y0) * ppu

        margin = 4 + int(self.cfg.agent_radius_px) * 3
        vis = (
            (sx >= -margin) & (sx <= self._w_px + margin)
            & (sy >= -margin) & (sy <= self._h_px + margin)
        )
        idx = np.flatnonzero(vis)
        return [(int(i), int(sx[i]), int(sy[i])) for i in idx]

    # Anspråkens kulörer, i `viewframe.FAUNA_CLAIMS` ordning, och grammatiken är
    # **angelägenhetsgrad**: nödlägena i den röda bågen, vardagen i den gröna och
    # blå. Nivåerna i behovstrappan går 1 (flykt) till 6 (vardag), och kulören
    # följer dem, så att en flock som slår om från bete till flykt syns som en
    # färgvåg genom molnet.
    #
    # 1014 satte `nedkylning` i blått, vilket är semantiskt lockande — kallt är
    # blått — men bryter grammatiken: den ligger på nivå 3 och är ett nödläge.
    # Blått är vardagens flock. Rättat.
    #
    # Index 0 är "inget anspråk vann", vilket är ett verkligt utfall. Det ritas
    # omättat i stället för i en egen kulör, eftersom en kulör hade sagt att det
    # var ett anspråk bland de andra.
    _CLAIM_HUE = (0.00, 0.00, 0.05, 0.10, 0.93, 0.85, 0.30, 0.45, 0.58)
    _CLAIM_SAT = (0.00, 0.85, 0.85, 0.85, 0.85, 0.85, 0.80, 0.80, 0.80)

    def _draw_dir(self, frame, reps) -> None:
        """
        Sektorperceptet som kilar: vad djuret ser åt varje håll.

        Radien bär fördelningen, opaciteten mängden. Se `ViewerConfig.show_dir`
        för varför det är två kanaler och inte en.

        Kilarna ritas **under** djuret och före det, så att kroppen förblir
        läsbar. Sektor k pekar `(k + 0,5)` sektorbredder från nosen, som
        `_acc_dir_ang`, och roteras hit med kursen.
        """
        pygame = self.pg
        prof = np.asarray(getattr(frame, "fauna_dir_food", None))
        if prof.ndim != 2 or prof.shape[0] == 0 or prof.shape[1] < 2:
            return
        S = int(prof.shape[1])
        claim = np.asarray(getattr(frame, "fauna_claim", None))
        senser = np.asarray(getattr(frame, "fauna_sense_r", None))
        ppu = float(self._ppu)
        halv = math.pi / S
        # En sektor som bär allt når `sqrt(S)` gånger synvidden. Ytan är
        # bevarad, så loben *ska* sticka utanför ringen — det är hur en toppig
        # profil ser toppig ut.
        maxr = math.sqrt(float(S))

        for i, px, py in reps:
            if i >= prof.shape[0]:
                continue
            w = prof[i].astype(np.float64)
            tot = float(w.sum())
            if tot <= 0.0:
                continue
            R = float(senser[i]) * ppu if senser.size > i else 0.0
            if R < 3.0:
                continue
            ci = int(claim[i]) % len(self._CLAIM_HUE) if claim.size > i else 0
            col = _hsv_to_rgb(self._CLAIM_HUE[ci], self._CLAIM_SAT[ci], 1.0)
            # Mängden: profilens medelvärde i [0, 1]. Ett djur på bar mark blir
            # nästan osynligt, ett i tät växtlighet solitt.
            alpha = int(round(35.0 + 145.0 * min(1.0, tot / S)))
            head = float(frame.fauna_heading[i])

            RR = int(R * maxr) + 2
            surf = pygame.Surface((2 * RR + 2, 2 * RR + 2), pygame.SRCALPHA)
            for k in range(S):
                # Arean proportionell mot andelen ger radien som roten ur den.
                rk = R * math.sqrt(float(w[k]) / tot * S)
                if rk < 1.0:
                    continue
                a0 = head + (k + 0.5) * (2.0 * math.pi / S) - halv
                pts = [(RR + 1, RR + 1)]
                for j in range(7):
                    a = a0 + 2.0 * halv * j / 6.0
                    pts.append((RR + 1 + rk * math.cos(a), RR + 1 + rk * math.sin(a)))
                pygame.draw.polygon(surf, (col[0], col[1], col[2], alpha), pts)
            if self.cfg.dir_ring and R >= 6.0:
                pygame.draw.circle(surf, (col[0], col[1], col[2], 70),
                                   (RR + 1, RR + 1), int(R), 1)
            self._screen.blit(surf, (px - RR - 1, py - RR - 1))

    def _draw_agents(self, frame) -> None:
        if not self.cfg.draw_agents:
            return

        pygame = self.pg
        hl = int(self.cfg.agent_heading_len_px)
        r0 = int(self.cfg.agent_radius_px)

        reps = self._screen_positions(frame.fauna_x, frame.fauna_y)

        if self.cfg.show_dir:
            self._draw_dir(frame, reps)

        for i, px, py in reps:
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
            f"gamma={self.cfg.gamma:.2f}  {'dir ' if self.cfg.show_dir else ''}"
            f"{self._pause_text(frame)}"
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
        if bool(getattr(frame, "has_terrain", False)):
            wet = np.asarray(frame.wet_kind)
            n = max(1, wet.size)
            soil = np.asarray(frame.soil01, dtype=np.float64)
            land = wet == WET_LAND
            lines.append(
                f"höjd {frame.elev_min:+.2f}..{frame.elev_max:+.2f}  "
                f"hav={100.0 * np.mean(wet == WET_SEA):4.1f}%  "
                f"sjö={100.0 * np.mean(wet == WET_LAKE):4.1f}%  "
                f"fåra={100.0 * np.mean(wet == WET_CHANNEL):4.2f}%  "
                f"markfukt={float(np.median(soil[land])) if land.any() else 0.0:.2f}"
                + ("" if self.cfg.show_water else "  [vatten dolt W]")
                + self._contour_hud(frame)
            )
        lines.append(
            f"grön=frisk→röd=döende  ljus=energi  gul ring=parningsredo  "
            f"[yta {self.cfg.flora_fill} F]  [trait {self.cfg.flora_color_by} T]"
            + ("  [paus mellanslag]" if getattr(frame, "control_enabled", False) else "")
            + (f"  [{self._ppu / max(1e-9, self._cell_w(frame)):.1f} px/cell  0 = hela världen]"
               if self._viewport else "")
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

    def _cell_w(self, frame) -> float:
        g = getattr(self, "_grid", None)
        return float(g.extent_x) / float(g.width) if g is not None else 1.0

    def _pause_text(self, frame) -> str:
        """
        Vad HUD:en säger om körningens läge.

        Två pauser kan råda. Den lokala stoppar bara renderingsloopen i
        run_population; den fjärrstyrda stoppar simuleringen och gäller alla
        anslutna tittare. De ska inte se likadana ut i HUD:en.
        """
        if getattr(frame, "paused", False):
            who = getattr(frame, "paused_by", "")
            return f"PAUSAD{' av ' + who if who else ''}"
        return "PAUSAD (lokalt)" if self._paused else ""

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
        self._grid = grid
        self._ensure_projection(grid)
        self._ensure_screen()

        base = self._make_rgb(frame, grid)
        if self.cfg.mode.upper().strip() == "FLORA":
            img = self._compose_flora(frame, base)
        else:
            img = base[self._pixel_cell]
        if self._pixel_valid is not None:
            img = np.where(self._pixel_valid[..., None], img, OUTSIDE_RGB)
        self._blit_rgb01(img)

        self._draw_contours(frame)
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