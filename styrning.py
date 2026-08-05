"""
Behovstrappan: nivåer, vikter och styrkor.

Varje sensoriskt intryck med påverkan på rörelseriktningen ska lämna ett
förslag — **(bäring, nivå, styrka)** — och vinnaren väljas på

    score = λ^(N − nivå) · styrka

Det här är steg 2 av den ombyggnaden, och det gör bara en sak: ger varje styrka
ett namn, ett dokumenterat intervall och ett motiverat mättnadsvärde.
Urvalsregeln är oförändrad — `elif`-kedjan i `_apply_reflex_drives` bestämmer
fortfarande vem som vinner, och amplituden bärs fortfarande av vikten i samma
uttryck som förut.

**Bitidentitet är ett krav, inte en ambition.** Steg 2:s eget kriterium är att
talen från steg 1 ska vara oförändrade, och simuleringen är kaotisk: en
omassociering på sista biten — `(w·a)·b` mot `w·(a·b)` — driver isär banorna
inom några hundra tick och gör kriteriet omätbart. Funktionerna här returnerar
därför **exakt de uttryck som stod på plats**, och anropen ersätter en
mellanvariabel i taget utan att röra raden runt omkring. De är flyttade, inte
omskrivna.

Det avgör också vad steg 2 *inte* kan göra. Tre normeringar kräver att
uttrycket associeras om och skjuts därför till steg 3, när vikterna ändå
försvinner in i trappan och bitidentitet inte längre går att kräva av något:

  **Födans skala.** `0,36 · (q/0,6)` är inte bitidentisk med `0,60 · q`, så
  styrkan ligger kvar på 0–0,6. Se `styrka_foda()`; frågan om taket är däremot
  avgjord.

  **Sällskaplighetens tecken.** `soc_bias = 2·sociability − 1` är tecknad, och
  negativ betyder undvikande. Under arbitrering finns ingen negativ styrka —
  ett anspråk med omvänt tecken är ett anspråk på motsatt bäring. Uppdelningen
  `|soc_bias|` plus tecken i bäringen är exakt i IEEE-754, men den kräver att
  `|soc_bias| · wcoh` bildas som en produkt, och det byter association mot
  dagens `(0,40 · soc_bias) · wcoh`.

  **Köldaversionen.** Samma sak: `0,70 · _ca · _stress` skulle bli
  `0,70 · (_ca · _stress)`. Dessutom är det oklart om `cold_aversion` hör
  hemma i styrkan alls — den uttrycker hur mycket djuret bryr sig, inte hur
  kallt det är, och det är närmare `_T_SOC`:s roll i steg 4.

Två asymmetrier som mätningen blottade och som inte heller rättas här, för att
båda är beteendeändringar:

  **Alignment har två skalor.** Grannvarianten bär avståndsvikten `4w(1−w)`,
  gruppvarianten bär den inte. Samma nivå och samma vikt ger alltså olika
  amplitud beroende på om sektoraggregatet fanns, och gruppvarianten är i
  genomsnitt starkare.

  **MLP:n har ingen bäring.** De åtta övriga anspråken pekar mot något i
  världen; MLP:n lämnar ett kursfel direkt. Dess styrka är `|tanh(y[0])|`, men
  under arbitrering måste den antingen få en bäring eller sluta vara kandidat
  och i stället sätta *värdena* de andra tävlar om. Det senare ligger närmare
  A10. Eget beslut; se `TODO.md`.

Varje funktion dokumenterar tre saker: vad styrkan uttrycker, vilket intervall
den faktiskt antar, och varifrån mättnadsvärdet kommer. Det sista är där ett
tyst kalibreringsfel gömmer sig — går en styrka 0–50 medan en annan går 0–1 gör
trappan bara det skalskillnaden redan gjorde.
"""

from __future__ import annotations


# --- Behovstrappans nivåer ------------------------------------------------
#
# Låg nivå = mer angeläget. `score = λ^(N − nivå) · styrka`, så nivå 1 får den
# största potensen. Utforskningen har ingen bäring i dag — den styr
# persistensen via `explore_drive`, inte kursen — och står med bara för att
# trappan ska vara komplett.
NIVA_FLYKT = 1
NIVA_SVALT = 2
NIVA_TERMISK = 3
NIVA_JAKT = 4
NIVA_PARNING = 5
NIVA_FLOCK = 6
NIVA_UTFORSKNING = 7

N_NIVAER = 7


# --- Dagens amplituder ----------------------------------------------------
#
# Vikterna är oförändrade och står här bara för att de ska ha ett ställe. I
# steg 4 ersätts de av en gemensam styrkraft — `frac = 1 − exp(−turn_gain ·
# effort · dt)` — och då är det bara nivån och styrkan som är kvar av dem.
#
# De används ännu inte som konstanter i `agent.py`: att byta ut en literal mot
# ett namn är bitidentiskt, men att flytta den i uttrycket är det inte, och
# raderna ska röras en gång i steg 3 i stället för två gånger nu.
W_FLYKT = 0.95
W_JAKT = 0.90
W_PARNING = 0.95
W_SEPARATION = 0.70
W_KOHESION = 0.40
W_ALIGNMENT = 0.20
W_FODA = 0.60          # mot en styrka i 0–0,6; blir 0,36 mot en normerad
W_KYLA = 0.70


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# --- Nivå 1: flykt --------------------------------------------------------

def styrka_flykt(threat_score: float, sc_min: float, sc_sat: float) -> float:
    """
    Hotets närhet och styrka, 0–1.

    Den enda styrkan som inte fanns. Flykten avfyrade på ett booleskt
    tröskelvillkor och skrev sedan full amplitud oavsett om hotet var
    marginellt eller akut. Den ligger överst i trappan, så en gren vars styrka
    alltid är ett vinner alltid när den avfyrar, oavsett λ. Det är rätt för ett
    verkligt hot och fel för ett som nätt och jämnt passerade tröskeln.

    `threat_score` är motpartens `attack_score` mot mig, alltså dess värdering
    av mig som byte. Avbildningen är linjär från tröskeln till mättnad.

    **Mättnadsvärdet är provisoriskt.** 0,30 kommer från den högsta observerade
    `attack_score` i 9 736 tick med byte i sikte: max +0,313, median −0,307,
    p90 −0,030. Det är ett empiriskt tak, inte ett härlett. Två saker kan
    flytta det — en omkalibrering av `attack_risk` i Steg 6c, som är planerad
    och vidgar fördelningen uppåt, och att fördelningen mättes i en annan värld
    än f4-flock. Mättnaden ligger därför i `AgentParams` och ska mätas om innan
    styrkan får driva amplitud.

    I steg 2 driver den ingenting. Den bärs ut ur reflexkedjan som `flee_state`
    för att kunna mätas.
    """
    if not (sc_sat > sc_min):
        return 1.0
    return _clamp((threat_score - sc_min) / (sc_sat - sc_min), 0.0, 1.0)


# --- Nivå 2: svält --------------------------------------------------------

def styrka_foda(hunger: float, sig: float, h_min: float = 0.4) -> float:
    """
    Hungern gånger födosignalens styrka, **0–0,6**.

    Uttrycket är oförändrat: `clamp(hunger − 0,4; 0; 0,6) · sig`.

    **Taket 0,6 är avgjort.** Det är inte en avkortning utan en skalning:
    `hunger` går 0–1, så `hunger − 0,4` når som mest exakt 0,6 och klampen
    binder aldrig uppåt. Den är alltså en gammal skalning som hör hemma i
    vikten, och `styrka_foda_normerad()` är samma storhet på 0–1.

    Golvet 0,4 är däremot verkligt och avsiktligt: under det svarar djuret inte
    på föda alls. Det är en dödzon, inte en normering, och ska överleva som
    styrkans nollpunkt.

    `sig` är födosignalen ur sektoraggregatet, redan 0–1.
    """
    return _clamp(hunger - h_min, 0.0, 0.6) * sig


def styrka_foda_normerad(hunger: float, sig: float, h_min: float = 0.4) -> float:
    """
    Samma styrka på 0–1, mot vikten 0,36 i stället för 0,60.

    **Noll anropare tills steg 3**, av skälet i modulens docstring. Att lämna
    en skalningsfunktion utan anropare är precis det mönster som gav
    `phenotype.py` sju döda accessorer och `population.py` en parallell
    implementation med hårdkodade index. Risken hanteras på ett sätt: när
    steg 3 tar den i bruk ska `styrka_foda()` tas bort i samma patch, så att
    det aldrig finns två levande definitioner av samma styrka.
    """
    return _clamp((hunger - h_min) / 0.6, 0.0, 1.0) * sig


# --- Nivå 3: termisk stress -----------------------------------------------

def kold_stress(Tb_set: float, T_lokal: float) -> float:
    """
    Köldpåfrestning, 0–1. Noll vid kroppens börtemperatur, ett vid absolut noll
    lokalt. Mättnaden är fysikalisk och behöver ingen kalibrering.

    Uppmätt i p140 skriver köldtermen i 99,2 procent av alla tick, med
    medelbeloppet 0,028. En term som alltid är på är en förskjutning, inte ett
    anspråk — under trappan blir den nivå 3 med en poäng som normalt är liten.
    """
    return _clamp((Tb_set - T_lokal) / Tb_set, 0.0, 1.0)


# --- Nivå 4: jakt ---------------------------------------------------------

def styrka_jakt(hunt_eff: float) -> float:
    """
    Jaktförmågan, 0–1 per konstruktion.

    `hunt_eff = predation · diet^1,5`, klampad. Mättnaden 1 är formell:
    uppmätt fördelning över 78 930 agenttick är median 0,124, p90 0,365, max
    0,529, så den övre halvan av skalan besöks aldrig.

    Trappan säger att jaktens styrka ska vara *bytets värde mot risken*, alltså
    `attack_score` — inte jägarens eget anlag. De två är olika storheter, och
    `attack_score` är den som avgör om angreppet alls kan ske. Bytet av storhet
    är en beteendeändring och hör till steg 3; se `TODO.md` om jaktens dubbla
    tröskel.
    """
    return _clamp(hunt_eff, 0.0, 1.0)


# --- Nivå 6: flock --------------------------------------------------------

def styrka_separation(nd: float, rep_zone: float) -> float:
    """
    Trängseln inne i repulsionszonen, 0–1. Ett vid noll avstånd, noll vid
    zonens kant, och grenen avfyrar bara innanför zonen. Mättnaden är
    geometrisk och behöver ingen kalibrering.
    """
    return 1.0 - (nd / rep_zone)


def avstandsvikt(nd: float, rep_zone: float) -> float:
    """
    Kohesionens avståndsvikt, 0–1: noll vid repulsionszonens kant, ett på full
    synradie. Kohesionen ska dra ihop där gruppen är gles och lämna plats där
    den är tät. Alignment bygger sin vikt på samma storhet.
    """
    return _clamp((nd - rep_zone) / max(1e-6, 1.0 - rep_zone), 0.0, 1.0)


def alignment_vikt(w: float) -> float:
    """
    Kursanpassningens avståndsvikt, 0–1. `4w(1−w)` toppar på mellanavstånd:
    nära nog att vara värd att följa, inte så nära att man måste väja.
    Mättnaden är formell — uttrycket når exakt ett vid `w = 0,5`.
    """
    return 4.0 * w * (1.0 - w)
