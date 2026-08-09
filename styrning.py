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

import math


# --- Behovstrappans nivåer ------------------------------------------------
#
# Låg nivå = mer angeläget. `score = λ^(N − nivå) · styrka`, så nivå 1 får den
# största potensen. Utforskningen har ingen bäring i dag — den styr
# persistensen via `explore_drive`, inte kursen — och står med bara för att
# trappan ska vara komplett.
NIVA_FLYKT = 1
NIVA_SVALT = 2
NIVA_NEDKYLNING = 3
NIVA_JAKT = 4
NIVA_PARNING = 5
NIVA_VARDAG = 6

N_NIVAER = 6

# Trappans skärpa. `score = λ^(N − nivå) · styrka`, så λ = 1 räknar bara
# styrkan och λ → ∞ är en strikt elif-kedja. Vid λ = 2 slår ett maximalt
# socialt önskemål en svag hunger men aldrig en verklig flykt — det uttalade
# kravet — och p153 visade att λ = 2 uppfyller det mot verkliga fördelningar:
# i drift ligger allt mellan 0,23 och 1,78, akut skjuter varje nivå till 4–128.
#
# Valt ur avsikten, inte kalibrerat mot p153; de fördelningarna är en överbetad
# världs. Se `TODO.md`.
LAMBDA = 2.0

# Påslag på den sittande vinnaren. Två nästan lika förslag byter annars vinnare
# varje tick och djuret vibrerar i stället för att välja väg. 1,25 betyder att
# utmanaren måste vara tjugofem procent bättre — mindre än ett halvt trappsteg,
# så hysteresen kan aldrig hålla kvar en vinnare mot ett anspråk från en högre
# nivå.
HYSTERES = 1.25


# Repulsionszonens gräns i normerat grannavstånd. Flockens tre regler delar på
# den: innanför gäller separation, utanför kohesion och kursanpassning.
REP_ZONE = 0.35


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
W_FODA = 0.36          # mot en styrka i 0–1 sedan dödzonen togs bort
W_KYLA = 0.70


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# --- Nivå 1: flykt --------------------------------------------------------

def narhet(dist: float, d_akut: float, d_noll: float) -> float:
    """
    Yttre faktor, 0–1: hur nära det objekt anspråket gäller.

    Ett på `d_akut` och närmare, noll på `d_noll` och längre bort, linjärt
    emellan.

    Den hör till styrkan bara när den ändrar hur **brådskande** saken är, inte
    när den bara ändrar hur säker riktningen är. Avstånd till ett hot minskar
    faran; avstånd till en födosignal minskar inte hungern. Därför bär flykt,
    jakt och parning en närhetsfaktor medan födosöket inte gör det.
    """
    if dist <= d_akut:
        return 1.0
    if dist >= d_noll:
        return 0.0
    return (d_noll - dist) / max(d_noll - d_akut, 1e-9)


def styrka_angrepp(poang_vid_kontakt: float, dist: float,
                   sc_min: float, sc_sat: float,
                   d_akut: float, d_noll: float) -> float:
    """
    Angreppets styrka, 0–1 — samma tal läst från båda hållen.

    Flykt och jakt är inte två storheter utan en: `attack_score` mellan samma
    två djur. Bytet fruktar exakt det jägaren värderar. Enda skillnaden är
    golvet — `flee_score_min` respektive `hunt_score_min` — alltså var
    respektive part börjar bry sig.

        styrka = kapacitet · närhet

    **Kapaciteten utvärderas vid kontakt.** `attack_value` bär `0,55 · d_norm`
    och `attack_risk` `0,05 · d_norm`, båda mot `prey_search_radius` — avstånd
    är alltså redan över halva poängen. Läggs en närhetsfaktor på den råa
    poängen dubbelräknas avståndet, och skalan mäts dessutom mot sensorradien
    i stället för mot angreppsradien, som är den enda distans där något
    faktiskt händer. Anropas poängen med `dist = attack_range` blir den ren:
    *skulle den anfalla om den stod här?*

    Tre lägen faller ut utan att kodas var för sig. En oförmögen motståndare
    ger nära noll på varje avstånd. En förmögen på håll ger en låg styrka som
    stiger medan den närmar sig. En förmögen inom `attack_range` ger ett — och
    det är samma tillstånd som "anfaller faktiskt", eftersom den som skadar mig
    per definition är förmögen och inom räckhåll. Det fjärde läget behöver
    alltså ingen egen term.
    """
    return (_clamp((poang_vid_kontakt - sc_min) / max(sc_sat - sc_min, 1e-9), 0.0, 1.0)
            * narhet(dist, d_akut, d_noll))


def parningsdrift(t_sedan_beredskap: float, tau: float,
                  massoverskott: float, reservandel: float) -> float:
    """
    Inre drivkraft för parning, 0–1: från beredd till starkt motiverad.

    `_mating_mode` är i dag en boolean — inte dräktig, avsvalningen ute,
    könsmogen — och ger därför ingen gradering alls. Driften graderar den på
    tre storheter:

      **tidsfaktor** `t/(t+τ)`, där `t` är tiden sedan beredskapen inföll och
      `τ` är **motivationens** tidskonstant. Den byggs upp medan djuret går
      oparat och nollställs vid parning. Det är den term som faktiskt beter sig
      som stigande motivation, och den mättar mjukt i stället för att slå i tak.

      `τ` var avsvalningstiden `repro_cooldown_s`, alltså åtta månader, och den
      är en annan storhet: hur länge kroppen behöver efter en dräktighet, inte
      hur fort lusten byggs. Uppmätt gav det tidsfaktorn medianen 0,188 medan
      kapacitetstermen låg på 1,000 — anspråket hölls nere av tiden och av
      ingenting annat. Se `AgentParams.repro_motivation_tau`.

      **kapaciteten** som den bindande av massöverskottet över `M_repro_min`
      och reservandelen. `min` och inte produkt: djuret begränsas av det som
      är knappast, och ska inte straffas två gånger för att ha gott om det ena.

    Termen är neutral med avsikt. `parningsberedskap` är tillståndet,
    `parningsdrift` graden — motsvarigheten i litteraturen är *sexual
    motivation*. "Brunst" sparas till en säsongsterm om projektet någon gång
    får dagslängd att härleda den ur.
    """
    t = max(0.0, t_sedan_beredskap)
    tidsfaktor = t / (t + max(tau, 1e-9))
    return tidsfaktor * min(_clamp(massoverskott, 0.0, 1.0),
                            _clamp(reservandel, 0.0, 1.0))


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

    **Mättnaden är `attack_score_min`, inte ett mätvärde.** Den första
    versionen satte 0,30 efter det högsta `attack_score` som observerats i
    testvärlden. Det band i 4,3 procent av p145, där svansen gick till 0,515 —
    ett observerat maximum är ett stickprov ur en fördelning som dessutom rör
    sig, och `predation` gick från 0,394 till 0,687 inom samma körning.

    Båda ändpunkterna är därför beslutskonstanter. Under `flee_score_min` har
    jag inte upptäckt hotet; vid `attack_score_min` kan motparten anfalla nu,
    och mer brådskande än så blir det inte — man kan inte fly hårdare än
    maximalt. Driften försvinner per konstruktion: kalibrerar Steg 6c om
    `attack_risk` flyttas fördelningen och trösklarna tillsammans, eftersom
    trösklarna är det fördelningen mäts emot.

    Upplösningen hamnar också där massan ligger. Med mättnad 0,52 föll
    medianhotet 0,163 på styrkan 0,11 och skalan användes i sin nedersta
    femtedel; med 0,18 får samma hot 0,72. Priset är att allt över 0,18 blir
    likvärdigt — för *valet* betydelselöst, eftersom nivå 1 vinner så snart
    styrkan är skild från noll, och för *kraften* i steg 4 avsiktligt.

    I steg 2 driver den ingenting. Den bärs ut ur reflexkedjan som `flee_state`
    för att kunna mätas.
    """
    if not (sc_sat > sc_min):
        return 1.0
    return _clamp((threat_score - sc_min) / (sc_sat - sc_min), 0.0, 1.0)


# --- Nivå 2: svält --------------------------------------------------------
#
# Aptit är inte ett nödläge. `styrka_foda()` nedan är närvarande i 63 procent
# av alla agenttick med medianstyrkan 0,385, och ett anspråk med den profilen
# kan inte ligga näst överst i en trappa — det äter allt under sig oavsett λ.
# Uppmätt i p145 skulle flockgrenen gå från att vinna 53 procent av tickarna
# till omkring 5.
#
# Svält är något annat än hunger, och måttet finns redan: `mass_severity`
# driver `dD_starve` och är noll så länge djuret följer sin tillväxtkurva.
# Principen är att **ett nödläges styrka ska vara samma storhet som faktiskt
# skadar djuret.** Då är kalibreringen redan gjord, nollpunkten är fysiologisk
# i stället för vald, och styrkan är per konstruktion noll i drift.


def styrka_svalt(underskott_J: float, underhall_J: float) -> float:
    """
    Svält, 0–1: hur stor del av underhållet som betalas med egen vävnad.

    **Svält är inte att vara liten.** Det är ihållande negativ energibalans,
    där underhållet inte kan betalas ur intaget och organismen får ta av sig
    själv. Kronisk låg tillgång ger *mindre vuxna*, inte sjuka vuxna —
    utvecklingsplasticitet är regel snarare än undantag, och ett litet djur i
    energibalans är friskt. `hunger()` säger redan det rakt ut: katabolism är
    ett entydigt underskottsbesked.

    Måttet är alltså ett förlopp, inte ett tillstånd — och det uttrycks som en
    **andel, inte en takt**:

        styrka = underskott / underhåll

    `underhall_J` är `E_out_drain`, alltså steget obligatoriska dräneringar:
    basal, beräkning, sensing, rörelse, termoreglering, dräktighetsöverhead.
    `underskott_J` är den del av det som reserven inte räckte till och som
    därför måste tas ur kroppen.

    Noll för ett djur som betalar ur intaget, hur litet det än är. Ett för ett
    djur vars hela underhåll kommer ur den egna vävnaden. Däremellan graderat.

    Formen valdes efter att en takt prövats och underkänts. `(dM_kat/dt) /
    (M − M_min) · horisont` krävde en horisontkonstant, och uppmätt spelade
    den knappt någon roll: fördelningen bestod av numeriskt damm plus en
    handfull verkliga händelser, så andelen mättade gick från 0,2 till 0,4
    procent när horisonten ändrades från en månad till femtio år. En andel
    behöver ingen konstant alls, och dammet blir en försumbar andel i stället
    för en tröskelpassage.
    """
    if underhall_J <= 1e-30:
        return 0.0
    return _clamp(underskott_J / underhall_J, 0.0, 1.0)


def massunderskott(m_rel: float, m_ok: float, m_crit: float) -> float:
    """
    Massunderskott, 0–1: massan relativt förväntad massa för åldern.

    Noll vid `starve_mass_ok_frac` (0,85) och uppåt, ett vid
    `starve_mass_crit_frac` (0,55) och neråt, linjärt emellan.

    Driver `dD_starve` och gör det fortfarande. Hette `styrka_svalt` tills
    p148 visade att den inte mäter svält: hela populationen låg stadigt på 69
    procent av `expected_mass` med stigande massa per djur och växande
    bestånd, alltså anpassad och inte döende. Medianstyrkan 0,52 i 61 procent
    av tickarna är ingen nödlägesprofil.

    **Om skademodellen också bör byta storhet är en egen fråga.** Den ändrar
    dödligheten och därmed hela ekologin, och den är dessutom sammanflätad med
    florabristen som håller nere massan. Se `TODO.md`.
    """
    if m_rel >= m_ok:
        return 0.0
    if m_rel <= m_crit:
        return 1.0
    return (m_ok - m_rel) / max(m_ok - m_crit, 1e-9)


def styrka_nedkylning(Tb: float, Tb_min: float, span: float = 10.0) -> float:
    """
    Nedkylning, 0–1: kroppstemperaturens underskott mot `Tb_min`.

    Noll så länge termoregleringen håller, ett vid `span` grader under.
    Samma uttryck som driver `dD_cold`.

    Skilj den från `kold_stress()`, som mäter *omgivningens* underskott mot
    börtemperaturen och därför är påslagen i 100 procent av tickarna. Den är
    vanlig termoreglering — en normal aktivitet, inte ett nödläge — och hör
    hemma längst ner i trappan, inte på nivå 3.
    """
    if Tb >= Tb_min:
        return 0.0
    return _clamp((Tb_min - Tb) / span, 0.0, 1.0)

def styrka_foda(hunger: float, sig: float) -> float:
    """
    Födosökets styrka, 0–1: aptiten gånger födosignalens styrka.

    **Dödzonen är borta.** Uttrycket var `clamp(hunger − 0,4; 0; 0,6) · sig`,
    alltså noll tills reserven var fyrtio procent tömd. Ett djur som såg en
    äng med halv reserv gick förbi den.

    Uppmätt passerar aptiten sin egen grind i 4,68 procent av kroppsstegen.
    Det avsiktliga födosöket var i praktiken avstängt — och när flyttalsdammet
    som gav full aptit i fyra procent av stegen togs bort i 0151 halverades
    impulsen att söka mat. Djuren i p151 var lättare genom hela
    grundarkraschen, 0,911 mot 0,081 kg vid tick 18000, och beståndet dog ut.

    Ett lågt anspråk i trappan ska ha den här formen: vanligt, svagt, alltid
    riktat mot maten, och förlorande mot flocken när djuret är mätt. Inte en
    reflex som slår på fullt när djuret redan ligger efter. Samma fel som i
    svälten, en nivå ner — en tröskel där det borde vara en gradient.

    Amplituden vid full aptit är oförändrad: vikten går från 0,60 mot en
    styrka i 0–0,6 till 0,36 mot en styrka i 0–1. Det som ändras är att
    intervallet däremellan inte längre är noll.

    `sig` ligger kvar i amplituden i det här steget. Att flytta den till
    bäringen är beslut C och hör till arbitreringen, där bäringen finns.
    """
    return _clamp(hunger, 0.0, 1.0) * sig


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


def styrka_kohesion(soc_bias: float, nd: float, rep_zone: float = REP_ZONE) -> float:
    """
    Sammanhållningen som **belopp**, 0–1: sällskapligheten gånger
    avståndsvikten.

    `soc_bias = 2·sociability − 1` är tecknad, och negativ betyder undvikande.
    Under arbitrering finns ingen negativ styrka — ett anspråk med omvänt
    tecken är ett anspråk på motsatt bäring. Beloppet hör alltså till styrkan
    och tecknet till bäringen.

    Anropas ännu inte från styrpassen, av skälet i modulens docstring: den
    bildar produkten `|soc_bias| · wcoh`, och det byter association mot dagens
    `(0,40 · soc_bias) · wcoh`. Den mäts däremot, och tas i bruk i steg 3.
    """
    return abs(soc_bias) * avstandsvikt(nd, rep_zone)


def styrka_alignment(soc_bias: float, nd: float, rep_zone: float = REP_ZONE) -> float:
    """
    Kursanpassningen som belopp, 0–1. Samma villkor som ovan.

    Gäller grannvarianten. Gruppvarianten saknar avståndsvikten helt och är
    därför i genomsnitt starkare — den asymmetrin står i modulens docstring och
    ska lösas i steg 3.
    """
    return abs(soc_bias) * alignment_vikt(avstandsvikt(nd, rep_zone))


def foda_signal(accB, accC, diet: float):
    """
    Födosignalen ur sektoraggregatet: `(styrka, sektorindex)`.

    Örtätande och asätande effektivitet viktar var sitt aggregat. Uttrycket är
    oförändrat och ligger här för att styrpasset och mätningen ska läsa samma
    definition — inte två.
    """
    herb_eff = (1.0 - diet) ** 0.7
    scav_eff = diet ** 0.7
    combo = accB * herb_eff + accC * scav_eff
    i_best = int(combo.argmax())
    return float(combo[i_best]), i_best


# --- Arbitreringen -------------------------------------------------------
#
# Fuzzy i ordningen, hårt i valet. Poängen är kontinuerlig, så vinnaren byts
# vid en väldefinierad korsning i stället för vid en boolesk tröskel — men det
# är alltid *en* riktning som väljs. Inget viktat medelvärde av bäringar.
#
# Nivåerna skiljer nödlägen från vardag, inte vardag från vardag. Föda, flock,
# termoreglering och utforskning ligger alla på `NIVA_VARDAG` och avgörs på
# styrka allena. Det är därför de kan dela på tiden: hungern faller när djuret
# betar och kohesionen växer med avståndet till grannarna, så de två är
# motriktade återkopplingar på olika variabler. Ingen kan vinna permanent.
#
# Betande flockdjur väljer inte mellan att äta och att hålla ihop. Låg födan en
# nivå över flocken förlorade ett *maximalt* flockanspråk mot ett *typiskt*
# födoanspråk — 1,63 mot 1,78 i p153 — och flockningen hade upphört att finnas.


def score(niva: int, styrka: float, lam: float = LAMBDA) -> float:
    """Anspråkets poäng. Låg nivå = mer angeläget = större potens."""
    return (lam ** (N_NIVAER - niva)) * styrka


def blanda(bidrag) -> tuple:
    """
    Slå ihop flera bidrag inom *samma* beteende till en bäring och en styrka.

    `bidrag` är `[(vikt, bäring), …]` där bäringen är ett normerat kursfel i
    [−1, 1]. Summan tas som vektorer; resultatets riktning blir bäringen och
    dess belopp styrkan.

    Att blanda här är legitimt och inte samma sak som det förbjudna
    medelvärdet: Reynolds tre regler är **ett** beteende med tre termer, precis
    som i den ursprungliga boids-formuleringen. Förbudet gäller mellan nivåer,
    där två anspråk vill åt olika håll av olika skäl och medelvägen är diket.

    Beloppet faller dessutom av sig självt när reglerna motsäger varandra —
    står djuret på sitt önskade avstånd tar separation och kohesion ut varandra
    och det finns inget att begära. Det är rätt: ett anspråk med styrkan noll
    ska inte vinna.
    """
    ex = 0.0
    ey = 0.0
    for w, b in bidrag:
        if w <= 0.0:
            continue
        a = float(b) * math.pi
        ex += w * math.cos(a)
        ey += w * math.sin(a)
    r = math.hypot(ex, ey)
    if r <= 1e-12:
        return 0.0, 0.0
    return _clamp(math.atan2(ey, ex) / math.pi, -1.0, 1.0), _clamp(r, 0.0, 1.0)


def valj(anskrav, sittande: str = "", hysteres: float = HYSTERES):
    """
    Välj vinnande anspråk. `anskrav` är `[(namn, nivå, styrka, …), …]`.

    Returnerar `(index, poäng)`, eller `(-1, 0.0)` om ingen har styrka. Den
    sittande vinnaren får `hysteres` som påslag.
    """
    bast = -1
    bast_p = 0.0
    for i, a in enumerate(anskrav):
        st = float(a[2])
        if st <= 0.0:
            continue
        p = score(int(a[1]), st)
        if a[0] == sittande:
            p *= hysteres
        if p > bast_p:
            bast_p = p
            bast = i
    return bast, bast_p
