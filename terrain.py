"""
Terrängen som statiskt världsfält.

**Geometrin läcker inte.** En FFT över ett `(h, w)`-rutnät vore exakt det
kvadratantagande Steg 1 grävde ut, och skulle gå sönder tyst vid nästa
geometribyte. Höjden byggs därför som **spektralsyntes över `Grid`s
kontinuerliga koordinater**: en summa av plana vågor med heltaliga vågtal över
`extent_x` och `extent_y`. Periodiciteten på torusen följer av konstruktionen i
stället för av rutnätets form, och modulen känner bara till `cell_center_x`,
`cell_center_y`, `extent_x` och `extent_y`. Latituden lästes en gång härifrån
och gör det inte längre — den var en klimatstorhet på villovägar i geometrin.

**Moderna räknas i våglängd, inte i modindex.** Första försöket band `kmax` till
heltalsindex per axel, vilket vid 64x256 gav helt olika fysiska våglängder i x
och y — sex moder i x är elva cellers våglängd, sex i y är fyrtio. Tillsammans
med en amplitudvikt som lät den längsta vågen dominera blev resultatet en slät
gradient utan relief. Bandet uttrycks nu i cellbredder, och en mod tas med om
dess våglängd faller inom det. Bandets övre gräns
avgör var bruset slutar och de placerade formerna tar vid.

Höjden är

    elevation = relief * (grundhöjd + fraktalt brus + placerade former)

Bruset har en enda spektrallutning över hela bandet. Den tidigare uppdelningen i
"kontinent" och "strävhet" var fyra parametrar för vad ett brett band med en
lutning gör lika bra: `beta` avgör både om floderna samlar sig och hur många
sänkor som finns att fylla, eftersom det är samma egenskap sedd i två skalor.

**Havet ger basnivån.** Det placeras som en bred bassäng och inte som ett
polarbälte: latituden föll ur världsmodellen när skalan fastställdes, och en
värld på någon kilometer har ingen pol. Vad som gör ett hav till ett hav är att
allt annat dräneras till det och att det självt dräneras ingenstans — en
hydrologisk definition och inte en geografisk, och därmed skalfri.

Bassängens mjuka kant gör två jobb: den är havet, och den är den regionala
lutningen mot basnivån. Den kontinentala lutning som tidigare låg här som egen
form generaliseras därmed från "mot polerna" till "mot basnivån", vilket är vad
den alltid var.

Havet är också den enda form som skalar med världen, eftersom en basnivå måste
omsluta det den är basnivå för. Övriga former är tektonik och har absoluta mått:
ett berg blir inte högre för att kartan blir större.

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

    # Brusets standardavvikelse i **cellängder**, vid referensvåglängden
    # `lambda_ref`.
    #
    # Höjd, vattendjup och vågrätt avstånd delar en enda längdenhet: cellarean
    # är 1, alltså är cellbredden 1,07 och grannavståndet detsamma. Att uttrycka
    # höjden i samma enhet gör lutningen till en verklig dimensionslös gradient
    # och varje höjdtal läsbart.
    #
    # Normeringen sker mot den *analytiska* standardavvikelsen,
    # `sqrt(Σ amp² / 2)` för en summa cosinus med slumpfaser. Den är exakt och
    # seedoberoende, så fältets spridning blir det begärda talet oavsett frö.
    #
    # **Men den normeringen gjorde amplituden oberoende av bandet**, alltså av
    # hur stora landformerna är. Modellen låg därmed på Hurstexponent noll: en
    # värld på hundra kilometer fick samma tjugo meters relief som en på en
    # kilometer. Talet nedan är därför inte längre fältets spridning utan dess
    # spridning **vid `lambda_ref`**; den faktiska följer av `hurst`.
    noise_sd: float = 0.17

    # Hurstexponenten: brusets amplitud går som `(lambda_max / lambda_ref)^H`.
    #
    # **Verkliga landskap är självaffina, inte självsimilära.** De blir flackare
    # ju större bit man tittar på, och det är skillnaden mellan H = 0,64 och
    # H = 1. Skandinaviska halvön är 1 800 km lång och 2,5 km hög, alltså 1:700
    # i medellutning, medan en enskild dalsida är 1:3. Med H = 1 — konstant
    # lutning i alla skalor — skulle halvön ha varit tiotals kilometer hög.
    #
    # Anpassning mot relief-mot-skala för verklig topografi:
    #
    #        1 km  ->    30 m          H = 0,64
    #       10 km  ->   200 m          relief(1 km) = 37 m
    #      100 km  ->   900 m
    #     1000 km  ->  2500 m
    #
    # Vad de tre lägena ger, med bandet skalat mot världen:
    #
    #     värld        lambda km    H=0 (förut)   H=0,64   H=1 (förkastat)
    #     64x128          0,52          10 m        10 m        10 m
    #     512x512         4,13          10 m        38 m        80 m
    #     1024x1024       8,25          10 m        59 m       160 m
    #     4096x4096      33,0           10 m       142 m       640 m
    #
    # Följden är att **bruset inte kan bära en höjdgradient på flera grader.**
    # Inte ens 4096² ger mer än en grad vid 6,5 °C/km. Det är riktigt:
    # femhundra meter på tio kilometer är en tektonisk struktur och inte en
    # eroderad yta, och sådana byggs som placerade former med höjd i meter.
    hurst: float = 0.64

    # Kontinentens grundhöjd över havsnivån, i cellängder, före lutning och
    # brus. Egen parameter sedan bruset blev nollcentrerat: tidigare låg det i
    # [0, 1] och bar en halv enhets grundhöjd som bieffekt av normeringen.
    # Hur högt landet ligger över havet är en storhet i sin egen rätt och ska
    # inte vara en artefakt av hur bruset skalades.
    base: float = 1.5

    # Global multiplikator på hela höjdfältet. Kvar för bakåtkompatibilitet och
    # för att kunna skala en hel värld i ett grepp; 1,0 betyder att talen ovan
    # gäller som de står.
    relief: float = 1.0

    # Bandets övre ände som **andel av världens kortaste sida**, alltså hur
    # stor den största landformen är i förhållande till världen.
    #
    # Var 48 cellbredder oavsett värld. Det är 0,75 av sidan vid bredd 64 men
    # 0,09 vid 512, så en större värld blev *finkornigare* i stället för större:
    # en slätt med krusningar där varje sänka är sin egen ändstation. Uppmätt
    # gav det 10,6 till 14,0 procent sjö vid 256x256 mot 1,6 till 2,0 vid
    # 64x128, med samma parametrar i övrigt.
    #
    # Den gamla invändningen mot ett brett band — att den världsspännande vågen
    # korrelerade med latituden och därmed med klimatet — föll med latituden.
    #
    # `lambda_min` sätter finkornigheten och står kvar i celler, eftersom den är
    # upplösningens gräns och inte världens: under två celler finns ingenting
    # att upplösa oavsett hur stor världen är.
    lambda_max_frac: float = 0.75
    lambda_min: float = 3.0

    # Referensvåglängd i cellbredder för `noise_sd`. Förankrad i den värld
    # amplituden en gång kalibrerades i — 0,75 av 64 celler — så att 64x128 är
    # bitidentisk med före Hurstskalningen.
    lambda_ref: float = 48.0

    # Spektrallutning: amplituden går som våglängden upphöjt till beta. Låga
    # värden ger ett finkornigt landskap där varje sänka är sin egen
    # ändstation; höga ger få stora former och inga sjöar.
    beta: float = 2.2

    # Antalet moder växer som kvadraten på bandets bredd i vågtal, alltså till
    # tiotusentals i en stor värld. Ett slumpmässigt urval bevarar spektrets
    # form och håller kostnaden linjär i cellantalet.
    max_modes: int = 4096

    # Havet som bred bassäng. **Havet är inte en landform utan världens
    # basnivå** — allt annat dräneras till det och det dräneras ingenstans — och
    # det är därför den enda form som skalar med världen. Ett berg blir inte
    # högre för att kartan blir större, men en basnivå måste omsluta det den är
    # basnivå för.
    #
    # Det som anges är därför **havsandelen och inte radien.** Radien löses fram
    # så att andelen träffas, vilket gör talet skalfritt på ett sätt en radie
    # inte kan vara: en radie som andel av bredden ger 21 procent hav vid 64x128
    # men bara 12 vid 64x256, eftersom en cirkel inte når hörnen i en avlång
    # värld. Med andelen som mål blir bassängen automatiskt ett bälte tvärs den
    # avlånga världen och en cirkel i den kvadratiska — samma sak uttryckt i
    # geometrin den råkar ha.
    #
    # Bassängens mjuka kant gör två jobb: den är havet, och den är den regionala
    # lutningen mot basnivån. Kontinentallutningen behövs inte som egen form —
    # den generaliseras från "mot polerna" till "mot basnivån", vilket är vad
    # den alltid var.
    #
    # Uppmätt vid 64x128 över tre frön, grundhöjd 1,5 och djup 2,0:
    #
    #     radie 0,9   hav 15–17 %   sjö 2,1–4,8 %   flod 1 224–1 909   kust 1,16–1,51
    #     radie 1,0   hav 18–23 %   sjö 1,4–3,4 %   flod 1 161–1 629   kust 1,27–1,38
    #     radie 1,1   hav 22–28 %   sjö 1,2–3,2 %   flod 1 173–1 558   kust 1,31–1,39
    #
    # Kusttalet är kustlinjens längd delad med omkretsen hos en cirkel med samma
    # havsarea, alltså hur mycket kusten avviker från bassängens egen form. Den
    # styrs av kvoten mellan brusets lutning och bassängkantens: en **flack och
    # bred** bassäng ger levande kust, en djup och smal ger en cirkel. Den
    # tidigare kandidaten radie 40 och djup 4,0 gav kusttal 1,05–1,09 och
    # dessutom 5,8–11,7 procent sjö mot målets tio.
    #
    # Andelen 0,20 ligger nära vad polarhavet gav (19,3 %), så landytan är i
    # praktiken oförändrad mot före bytet.
    hav_andel: float = 0.20
    hav_djup: float = 2.0

    # Placerade former. `None` ger havsbassängen ovan, alltså världens basnivå
    # och ingenting annat. Sätts listan tar den över helt, och `hav_radie` och
    # `hav_djup` läses inte längre: det finns då bara ett ställe som beskriver
    # strukturen.
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


def band_max(tp: TerrainParams, grid) -> float:
    """
    Bandets övre ände i cellbredder, ur världens kortaste sida.

    Att uttrycka den som en andel i stället för ett celltal är skillnaden
    mellan en större värld och en finkornigare: 48 celler är 0,75 av sidan vid
    bredd 64 men 0,09 vid 512.
    """
    kort = min(int(grid.width), int(grid.height))
    return max(float(tp.lambda_min), float(tp.lambda_max_frac) * float(kort))


def amp_scale(tp: TerrainParams, lam_max: float) -> float:
    """
    Höjdskalans faktor vid ett band, ur Hurstexponenten.

    Den gäller **hela basnivån** — bruset, grundhöjden och havsdjupet — och
    inte de placerade formerna. Skälet står i `docs/geologin-och-vattnet.md`:
    bruset är erosion och skalar fraktalt, medan en placerad form är tektonik
    med en storlek i meter. Ett berg blir inte högre för att kartan blir
    större.

    Grundhöjden och havsdjupet hör till basnivån och inte till tektoniken. Att
    skala bruset men inte dem gav ett hav som dränktes i brus: uppmätt vid
    512x512 föll den sammanhängande havsmassan till 10,8 procent av begärda 20,
    medan sjöandelen steg till 21,3 och största sjön till 21 486 celler —
    kustlinjen fragmenterades och bassängen upphörde att vara en basnivå.

    Vid referensbandet är faktorn exakt ett, så den värld amplituden
    kalibrerades i är oförändrad.
    """
    lam_ref = max(1e-9, float(tp.lambda_ref))
    return (max(1e-9, lam_max) / lam_ref) ** float(tp.hurst)


def _modes(tp: TerrainParams, Lx: float, Ly: float, lam_max: float, rng):
    """
    Halva vågtalsplanet — den andra halvan är komplexkonjugatet och skulle bara
    dubbla amplituden. En mod tas med om dess fysiska våglängd faller i bandet.
    """
    lam_min = max(1e-6, float(tp.lambda_min))
    lam_max = max(lam_min, float(lam_max))
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


def _default_shapes(grid, tp, land: np.ndarray, skala: float) -> list:
    """
    Formlistan när scenariot inte anger någon: **havet, och ingenting annat.**

    Havet är världens basnivå — allt annat dräneras till det, det dräneras
    ingenstans — och det är en hydrologisk definition och inte en geografisk.
    Den är skalfri och fungerar lika bra i en dalgång på en kilometer som på en
    planet. Kaspiska havet är ett hav i exakt den meningen.

    Tidigare låg här två former: en polarsänka och en kontinental lutning, båda
    funktioner av latituden. Polarsänkan placerade havet där klimatet var
    obeboeligt, och det argumentet finns inte kvar sedan latituden föll ur
    världsmodellen. Lutningen organiserade dräneringen mot polerna, och den
    behövs inte som egen form: bassängens mjuka kant *är* den regionala
    lutningen mot basnivån, vilket är vad kontinentallutningen alltid var.

    **Radien löses fram ur havsandelen** med bisektion mot det färdiga
    brusfältet, eftersom en radie inte är skalfri men en andel är det. Sökningen
    kostar ett tjugotal svep över `hypot` och ligger i världens engångsuppbygge.
    """
    forms = getattr(tp, "former", None)
    if forms:
        return list(forms)

    x = np.asarray(grid.cell_center_x, dtype=np.float64)
    y = np.asarray(grid.cell_center_y, dtype=np.float64)
    Lx = float(grid.extent_x)
    Ly = float(grid.extent_y)
    dx = x - 0.5 * Lx
    dy = y - 0.5 * Ly
    dx -= Lx * np.round(dx / Lx)
    dy -= Ly * np.round(dy / Ly)
    # Avståndet normeras mot utsträckningen i varje led, så bassängen blir en
    # ellips som följer världens form: en cirkel i en kvadratisk värld, ett
    # avlångt hav i en avlång. En rund bassäng når inte hörnen när världen är
    # fyra gånger högre än bred — uppmätt gav den 14 procent sjö vid 64x256 mot
    # 1,5 vid 64x128, eftersom landet längst från mitten saknade väg till havet.
    d = np.hypot(dx / Lx, dy / Ly)

    mal = min(max(float(tp.hav_andel), 0.0), 0.95)
    djup = float(tp.hav_djup) * float(skala)
    grund = float(tp.base) * float(skala)

    def andel(r: float) -> float:
        z = grund + land - djup * (1.0 - _smoothstep(0.0, 1.0, np.minimum(d / r, 1.0)))
        return float((z < 0.0).mean())

    lo, hi = 1e-4, float(d.max()) * 2.0
    if mal <= 0.0:
        r = lo
    elif andel(hi) < mal:
        r = hi
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if andel(mid) < mal:
                lo = mid
            else:
                hi = mid
        r = 0.5 * (lo + hi)

    return [{
        "typ": "bassang",
        "x": 0.5,
        "y": 0.5,
        "radie_x": r * Lx,
        "radie_y": r * Ly,
        "djup": djup,
    }]


def _shapes(grid, tp, forms) -> np.ndarray:
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
    if not forms:
        return out

    x = np.asarray(grid.cell_center_x, dtype=np.float64)
    y = np.asarray(grid.cell_center_y, dtype=np.float64)
    Lx = float(grid.extent_x)
    Ly = float(grid.extent_y)
    def _wrapped_delta(px, py):
        dx = x - px
        dy = y - py
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        return dx, dy

    for f in forms:
        typ = str(f.get("typ", "")).strip().lower()

        if typ in ("bassang", "bassäng", "basin", "kon", "cone"):
            # En sänka eller ett massiv. Tecknet på `djup` respektive `hojd`
            # avgör vilket, så det är en formtyp och inte två.
            #
            # `radie` ger en cirkel; `radie_x` och `radie_y` var för sig ger en
            # ellips. Ellipsen behövs för havet, som ska följa världens form:
            # en rund bassäng når inte hörnen i en värld som är fyra gånger
            # högre än bred. En landform har ingen anledning att vara elliptisk
            # och anger bara `radie`.
            px = float(f.get("x", 0.5)) * Lx
            py = float(f.get("y", 0.5)) * Ly
            r = max(1e-9, float(f.get("radie", 10.0)))
            rx = max(1e-9, float(f.get("radie_x", r)))
            ry = max(1e-9, float(f.get("radie_y", r)))
            amp = float(f.get("hojd", -float(f.get("djup", 1.0))))
            dx, dy = _wrapped_delta(px, py)
            d = np.hypot(dx / rx, dy / ry)
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
    lam_max = band_max(tp, grid)
    kx, ky, amp, phase = _modes(tp, Lx, Ly, lam_max, rng)

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
    skala = amp_scale(tp, lam_max)
    land = land * (float(tp.noise_sd) * skala)

    struct = _shapes(grid, tp, _default_shapes(grid, tp, land, skala))

    z = float(tp.base) * skala + land + struct
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
