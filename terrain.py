"""
Terrängen som statiskt världsfält.

**Geometrin läcker inte.** En FFT över ett `(h, w)`-rutnät vore exakt det
kvadratantagande Steg 1 grävde ut, och skulle gå sönder tyst vid nästa
geometribyte. Höjden byggs därför som **spektralsyntes över `Grid`s
kontinuerliga koordinater**: en summa av plana vågor med heltaliga vågtal över
`extent_x` och `extent_y`. Periodiciteten på torusen följer av konstruktionen i
stället för av rutnätets form, och modulen känner bara till `cell_center_x`,
`cell_center_y`, `extent_x`, `extent_y` och `cell_lat`.

**Moderna räknas i våglängd, inte i modindex.** Första försöket band `kmax` till
heltalsindex per axel, vilket vid 64x256 gav helt olika fysiska våglängder i x
och y — sex moder i x är elva cellers våglängd, sex i y är fyrtio. Tillsammans
med en amplitudvikt som lät den längsta vågen dominera blev resultatet en slät
gradient utan relief. Bandet uttrycks nu i cellbredder, och en mod tas med om
dess våglängd faller inom det. Det gör också att `lambda_max` under världens
kortaste sida utesluter den världsspännande gradienten, som annars skulle
korrelera med latituden och därmed med klimatet.

Höjden är

    elevation = relief * (fraktalt brus - polarsänka)

Bruset har en enda spektrallutning över hela bandet. Den tidigare uppdelningen i
"kontinent" och "strävhet" var fyra parametrar för vad ett brett band med en
lutning gör lika bra: `beta` avgör både om floderna samlar sig och hur många
sänkor som finns att fylla, eftersom det är samma egenskap sedd i två skalor.

**Polarsänkan** ger havet. Latituden är redan 1 vid projektionens över- och
underkant och 0 i mitten, så en smoothstep i `|lat|` sänker terrängen under
havsnivån i ett bälte som omsluter båda polerna — vilka på torusen är ett och
samma bälte. Havet blir en enda sammanhängande sänka som all dränering mynnar i,
och den ligger där tillväxtgrinden ändå är nära noll.

Havsnivån är noll per definition. Enheten är godtycklig men delas med `water`,
eftersom fri yta är `elevation + water`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit as _njit, prange as _prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - beror på miljön
    HAVE_NUMBA = False
    _prange = range

    def _njit(*a, **k):  # type: ignore[misc]
        def deco(f):
            return f

        return deco


@dataclass
class TerrainParams:
    """
    Terrängens form. Våglängderna är i cellbredder; `relief` sätter skalan mot
    `water`, eftersom de två adderas till fri yta.
    """

    seed: int = 1

    # Brusets standardavvikelse i **cellängder**.
    #
    # Höjd, vattendjup och vågrätt avstånd delar en enda längdenhet: cellarean
    # är 1, alltså är cellbredden 1,07 och grannavståndet detsamma. Att uttrycka
    # höjden i samma enhet gör lutningen till en verklig dimensionslös gradient
    # och varje höjdtal läsbart — en bassäng på djup 0,8 är fyra femtedelar av
    # en cellbredd djup, och det går att resonera om.
    #
    # Tidigare normerades bruset till spannet [0, 1] och skalades med `relief`.
    # Det gav fast spann men **seedberoende spridning** — uppmätt 0,142 till
    # 0,197 över fem frön — så amplituden var inte en höjd utan ett spann, och
    # ett tal som `tilt: 1.2` betydde olika saker i olika världar.
    #
    # Normeringen sker nu mot den *analytiska* standardavvikelsen,
    # `sqrt(Σ amp² / 2)` för en summa cosinus med slumpfaser. Den är exakt och
    # seedoberoende, så fältets spridning blir `noise_sd` oavsett frö och
    # oavsett hur bandet väljs.
    noise_sd: float = 0.17

    # Kontinentens grundhöjd över havsnivån, i cellängder, före lutning och
    # brus. Egen parameter sedan bruset blev nollcentrerat: tidigare låg det i
    # [0, 1] och bar en halv enhets grundhöjd som bieffekt av normeringen.
    # Hur högt landet ligger över havet är en storhet i sin egen rätt och ska
    # inte vara en artefakt av hur bruset skalades.
    base: float = 0.5

    # Global multiplikator på hela höjdfältet. Kvar för bakåtkompatibilitet och
    # för att kunna skala en hel värld i ett grepp; 1,0 betyder att talen ovan
    # gäller som de står.
    relief: float = 1.0

    # Bandets ändar i cellbredder. `lambda_max` bör ligga under världens
    # kortaste sida, annars får landskapet en världsspännande gradient som
    # korrelerar med latituden. `lambda_min` sätter finkornigheten; under två
    # celler finns ingenting att upplösa.
    lambda_max: float = 48.0
    lambda_min: float = 3.0

    # Spektrallutning: amplituden går som våglängden upphöjt till beta. Låga
    # värden ger ett finkornigt landskap där varje sänka är sin egen
    # ändstation; höga ger få stora former och inga sjöar.
    beta: float = 2.2

    # Antalet moder växer som kvadraten på bandets bredd i vågtal, alltså till
    # tiotusentals i en stor värld. Ett slumpmässigt urval bevarar spektrets
    # form och håller kostnaden linjär i cellantalet.
    max_modes: int = 4096

    # Kontinental lutning: höjden stiger från kust mot inland som
    # `tilt * (1 - |lat|)`. Utan den är landskapet en platå av gupp utan
    # övergripande gradient, och vattnet blir stående i varje stor bassäng i
    # stället för att nå kusten — uppmätt 33 procent av världen som sjö och
    # största flod 168 celler.
    #
    # Termen byter sjöar mot floder monotont. Uppmätt vid 64x256, β 2,4:
    #
    #     lutning 0   sjö 32,9 %   största sjö 3 897   maxflod   166
    #     lutning 1   sjö 14,6 %   största sjö   590   maxflod   353
    #     lutning 3   sjö  3,1 %   största sjö    91   maxflod   856
    #     lutning 5   sjö  0,8 %   största sjö    26   maxflod  2 092
    #
    # Den är också fysiskt riktig: kontinenter dräneras mot sina kuster, och
    # den vattendelare som uppstår vid ekvatorn är vad en landmassa mellan två
    # hav ska ha.
    tilt: float = 3.0

    # Kusten i |lat|: 1 vid polen, 0 vid ekvatorn. Smoothstep över
    # [coast_lat - coast_width, coast_lat + coast_width].
    coast_lat: float = 0.93
    coast_width: float = 0.06
    sea_depth: float = 1.0

    # Placerade former. `None` betyder de två inbyggda — polarhav och lutning —
    # med parametrarna ovan, alltså exakt dagens terräng. Sätts listan tar den
    # över helt, och `tilt`, `coast_lat`, `coast_width` och `sea_depth` läses
    # inte längre: det finns då bara ett ställe som beskriver strukturen.
    #
    # Se `_shapes` för formtyperna.
    former: list | None = None


@_njit(cache=True, fastmath=True, parallel=True)
def _synth(x, y, kx, ky, amp, phase, inv_Lx, inv_Ly, out):
    n = x.shape[0]
    m = kx.shape[0]
    two_pi = 2.0 * math.pi
    for i in _prange(n):
        s = 0.0
        xi = x[i] * inv_Lx
        yi = y[i] * inv_Ly
        for j in range(m):
            s += amp[j] * math.cos(two_pi * (kx[j] * xi + ky[j] * yi) + phase[j])
        out[i] = s


def _modes(tp: TerrainParams, Lx: float, Ly: float, rng):
    """
    Halva vågtalsplanet — den andra halvan är komplexkonjugatet och skulle bara
    dubbla amplituden. En mod tas med om dess fysiska våglängd faller i bandet.
    """
    lam_min = max(1e-6, float(tp.lambda_min))
    lam_max = max(lam_min, float(tp.lambda_max))
    amax = int(Lx / lam_min) + 1
    bmax = int(Ly / lam_min) + 1

    kxs: list[float] = []
    kys: list[float] = []
    amps: list[float] = []
    for a in range(0, amax + 1):
        for b in range(-bmax, bmax + 1):
            if a == 0 and b <= 0:
                continue
            k = math.hypot(a / Lx, b / Ly)
            if k <= 0.0:
                continue
            lam = 1.0 / k
            if lam < lam_min or lam > lam_max:
                continue
            kxs.append(float(a))
            kys.append(float(b))
            amps.append(lam ** float(tp.beta))

    kx = np.asarray(kxs, dtype=np.float64)
    ky = np.asarray(kys, dtype=np.float64)
    amp = np.asarray(amps, dtype=np.float64)
    if kx.size == 0:
        raise ValueError(
            f"terrängbandet [{lam_min:g}, {lam_max:g}] innehåller inga moder i "
            f"en värld med utsträckning {Lx:.1f} x {Ly:.1f}"
        )

    cap = int(tp.max_modes)
    if cap > 0 and kx.size > cap:
        # Likformigt urval: modtätheten växer redan som k², så urvalet bevarar
        # spektrets form. Amplituderna skalas så att variansen består.
        pick = rng.choice(kx.size, size=cap, replace=False)
        scale = math.sqrt(kx.size / cap)
        kx, ky, amp = kx[pick], ky[pick], amp[pick] * scale

    phase = rng.uniform(0.0, 2.0 * math.pi, size=amp.shape[0])
    return kx, ky, amp, phase


def _shapes(grid, tp) -> np.ndarray:
    """
    De placerade formerna: terrängens strategiska lager.

    **Bruset är textur, inte plan.** Ett fraktalt fält har ingen avsikt — det
    ger detalj men kan inte ombes lägga en sjö någonstans. Verkliga landskap är
    tektonik som ger struktur plus erosion som ger detalj, och den
    arbetsfördelningen är den här funktionens hela poäng.

    Polarhavet och kontinentallutningen var redan sådana former; de var bara
    hårdkodade specialfall. Här blir de två poster i en lista bland andra, och
    utan `former` i scenariot återges exakt de två.

    **Positionerna är normerade världskoordinater**, alltså andel av
    utsträckningen. En bassäng på (0,40, 0,55) betyder samma sak vid 64x128 som
    vid 512x512 — nödvändigt eftersom världsstorleken ska bytas. Avstånden tas
    med `grid.torus_delta_pos`, så former wrappar korrekt över sömmen.

    **Formerna adderas med mjuk avtoning.** Addition bevarar den spektrala
    tolkningen — bruset är också en summa — och två överlappande former
    komponerar i stället för att den ena maskera den andra. Avtoningen är en
    smoothstep, så ingen form har en kant.

    Höjder och djup är i cellängder, samma enhet som allt annat vågrätt.
    """
    n = int(grid.n_cells)
    out = np.zeros(n, dtype=np.float64)
    forms = getattr(tp, "former", None)
    if not forms:
        return out

    x = np.asarray(grid.cell_center_x, dtype=np.float64)
    y = np.asarray(grid.cell_center_y, dtype=np.float64)
    Lx = float(grid.extent_x)
    Ly = float(grid.extent_y)
    lat = np.abs(np.asarray(grid.cell_lat, dtype=np.float64))

    def _wrapped_delta(px, py):
        dx = x - px
        dy = y - py
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        return dx, dy

    for f in forms:
        typ = str(f.get("typ", "")).strip().lower()

        if typ in ("polarhav", "polar_sea"):
            out -= float(f.get("djup", 1.0)) * _smoothstep(
                float(f.get("lat", 0.93)) - float(f.get("bredd", 0.06)),
                float(f.get("lat", 0.93)) + float(f.get("bredd", 0.06)),
                lat,
            )

        elif typ in ("lutning", "tilt"):
            # Stiger från polerna mot ekvatorn. Den form som organiserar
            # dräneringen; se TerrainParams.tilt.
            out += float(f.get("styrka", 1.0)) * (1.0 - lat)

        elif typ in ("bassang", "bassäng", "basin", "kon", "cone"):
            # En rund sänka eller ett massiv. Tecknet på `djup` respektive
            # `hojd` avgör vilket, så det är en formtyp och inte två.
            px = float(f.get("x", 0.5)) * Lx
            py = float(f.get("y", 0.5)) * Ly
            r = max(1e-9, float(f.get("radie", 10.0)))
            amp = float(f.get("hojd", -float(f.get("djup", 1.0))))
            dx, dy = _wrapped_delta(px, py)
            d = np.hypot(dx, dy) / r
            out += amp * (1.0 - _smoothstep(0.0, 1.0, np.minimum(d, 1.0)))

        elif typ in ("rygg", "ridge"):
            # En höjdrygg längs en sträcka. Avståndet tas till segmentet, så
            # ryggen har ändar i stället för att vara en oändlig vägg.
            ax, ay = [float(v) for v in f.get("fran", (0.2, 0.5))]
            bx, by = [float(v) for v in f.get("till", (0.8, 0.5))]
            ax *= Lx; ay *= Ly; bx *= Lx; by *= Ly
            vx = bx - ax
            vy = by - ay
            vv = vx * vx + vy * vy
            dxa, dya = _wrapped_delta(ax, ay)
            t = 0.0 if vv <= 0.0 else np.clip((-dxa * vx - dya * vy) / vv, 0.0, 1.0)
            dx = dxa + t * vx
            dy = dya + t * vy
            r = max(1e-9, float(f.get("bredd", 6.0)))
            d = np.hypot(dx, dy) / r
            out += float(f.get("hojd", 1.0)) * (
                1.0 - _smoothstep(0.0, 1.0, np.minimum(d, 1.0))
            )

        else:
            raise ValueError(f"okänd terrängform: {typ!r}")

    return out


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    if b <= a:
        return (x >= b).astype(np.float64)
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def generate_elevation(grid, tp: TerrainParams) -> np.ndarray:
    """
    Höjd per cell, float32, med havsnivån vid noll.

    Deterministisk i `tp.seed`: samma frö och samma geometri ger samma värld.
    """
    x = np.asarray(grid.cell_center_x, dtype=np.float64)
    y = np.asarray(grid.cell_center_y, dtype=np.float64)
    Lx = float(grid.extent_x)
    Ly = float(grid.extent_y)

    rng = np.random.default_rng(int(tp.seed))
    kx, ky, amp, phase = _modes(tp, Lx, Ly, rng)

    out = np.empty(x.shape[0], dtype=np.float64)
    if HAVE_NUMBA:
        _synth(x, y, kx, ky, amp, phase, 1.0 / Lx, 1.0 / Ly, out)
    else:  # pragma: no cover - endast utan numba
        out[:] = 0.0
        for j in range(amp.shape[0]):
            out += amp[j] * np.cos(
                2.0 * math.pi * (kx[j] * x / Lx + ky[j] * y / Ly) + phase[j]
            )

    # Analytisk normering: variansen hos en summa av cosinus med oberoende
    # slumpfaser är summan av amplituderna i kvadrat genom två. Att dela med
    # den ger ett fält med standardavvikelse ett oavsett frö och band, till
    # skillnad från en normering mot det realiserade min och max.
    sd = math.sqrt(float((amp ** 2).sum()) * 0.5)
    land = (out / sd) if sd > 1e-300 else np.zeros_like(out)
    land = land * float(tp.noise_sd)

    if tp.former is None:
        # De två inbyggda formerna, uttryckta som en lista. Samma fält som
        # förut, bara via samma väg som allt annat.
        lat = np.abs(np.asarray(grid.cell_lat, dtype=np.float64))
        sea = _smoothstep(
            float(tp.coast_lat) - float(tp.coast_width),
            float(tp.coast_lat) + float(tp.coast_width),
            lat,
        )
        struct = (float(tp.tilt) * (1.0 - lat)
                  - float(tp.sea_depth) * (1.0 + float(tp.tilt)) * sea)
    else:
        struct = _shapes(grid, tp)

    z = float(tp.base) + land + struct
    return (float(tp.relief) * z).astype(np.float32, copy=False)


def describe(elevation: np.ndarray) -> dict:
    """Sammanfattning av en terräng. Underlag för loggrad och gallring av frön."""
    z = np.asarray(elevation, dtype=np.float64)
    sea = z < 0.0
    land = ~sea
    return {
        "sea_frac": float(sea.mean()),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "land_z_mean": float(z[land].mean()) if land.any() else 0.0,
    }
