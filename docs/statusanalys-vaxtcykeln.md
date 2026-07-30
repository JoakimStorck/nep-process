# Statusanalys — efter växtcykeln

*Juli 2026. Underlag: patch 0053–0067 och de fem 40 000-tickskörningarna w1, w2, p064, p065, p066.*

*Den tredje i serien efter `nep-process-statusanalys-och-reviderad-plan` v1 och v2. Samma syfte: säga vad som faktiskt gäller, inte vad som var tänkt.*

---

## Sammanfattande bedömning

Femton patchar har byggt om florans hela livscykel. Fem av dem var mekanismer, fyra var kalibreringar, tre var instrumentering, två var rena buggrättningar och en var prestanda.

**Det som fungerar** är näringskretsloppet, arearegeln, fröaxeln och rot–skott-axeln. Näringen cirkulerar med en drift under 1e-9 relativt genom betningskatastrofer och beståndskollapser. Två traits — frömassa och rotandel — har hittat inre optima och stannat där, vilket ingen trait i modellen gjorde före det här arbetet.

**Det som inte fungerar** är strukturandelen. Den har fått fyra motkrafter i fyra patchar och kryper fortfarande uppåt: 0,78 före arbetet, 0,69 efter. Bromsad, inte vänd.

**Och faunan är oförändrat trasig.** Varje körning slutar med utdöende, av samma skäl som i w1: ingen åldersoberoende dödlighet.

Den viktigaste metodiska lärdomen är dyrare än någon enskild bugg. **Jag har fem gånger läst ett tillstånd före mättnad som en jämvikt**, och tre gånger dragit fel slutsats av ett medelvärde över en sned fördelning. Båda felen ledde till patchar som inte behövdes.

---

## Del A — Vad som byggdes

| Patch | Innehåll | Utfall |
|---|---|---|
| 0054 | Rotarea som anspråk, näringens ekonomi | Fungerar. `uptake_capacity` binder, samexistens uppstod |
| 0055 | Reserv, allokering, härledd livslängd, förnafall | Fungerar. Generationsomsättning finns |
| 0056 | Fröaxel, etablering, spridningskärna | **Bäst i serien.** Inre optimum, bekräftat över 800 månader |
| 0057 | Instrumentering | — |
| 0058, 0060 | Prestanda | 1,5× snabbare, bitidentisk bana |
| 0059 | Reproduktionsgrind, trängsel, tillförsel | Fungerar. Hämningen bruten, apparatandelen vaknade |
| 0061 | Ljus som andra resurs | Fungerar mekaniskt, men gav struktur en ny uppsida |
| 0062 | Tvåvalutareproduktion | Halvt. Vedrabatten borta, kolet binder sällan |
| 0063 | Rot mot skott | Fungerar, men först efter 0065 |
| 0064 | Betning tar skott | Buggrättning |
| 0065 | Ljusnivå och såddfördelning | Fungerar. Transienten borta, rot–skott vaknade |
| 0066 | Mognadströskel | **Neutral.** Kostade storlek men gynnade struktur |
| 0067 | Grindredovisning | Motbevisade min egen bugghypotes |

---

## Del B — Axlarnas status

| Trait | Status | Slutvärde | Kommentar |
|---|---|---|---|
| frömassa | **Inre optimum** | 0,183 kg | Steg 4×, överskjöt, la sig. Smith–Fretwell bekräftad |
| rotandel | **Inre optimum** | 0,65–0,70 | Svarar på vilken resurs som binder. Funktionell jämvikt |
| apparatandel | Aktiv | 0,22–0,32 | Död till 0059; vaknade när trängseln blev verklig |
| vuxenmassa | Bromsad | 27 kg | Steg till 30 utan mognadströskel, 27 med |
| `repro_alloc` | Faller | 0,34 | Faller i varje körning. Symptom, inte anpassning |
| mognadströskel | **Neutral** | 0,115 | Rör sig inte. Ligger på sitt initieringsvärde |
| **strukturandel** | **Kryper uppåt** | **0,69** | Fyra motkrafter, ingen räcker |

### Varför strukturandelen inte går att stoppa

Den har en uppsida under **vardera** resursregimen. Binder ljuset vinner den på höjd, eftersom höjden är strukturmassa. Binder kvävet vinner den på att seg vävnad är 4,5 gånger billigare per kilo. Uppmätt: `light_input` 0,6 gav struktur 0,673, `light_input` 1,5 gav 0,595 — men båda steg.

Ingen balans mellan de två kan därför trycka ner den. Kalibreringen väljer bara vilken uppsida som dominerar.

Dessutom: långt liv och lågt förnafall är båda härledda ur strukturandelen, och mognadströskeln i 0066 straffar **kort liv** snarare än seg vävnad. Den gjorde axeln värre, inte bättre — 0,595 utan tröskeln, 0,692 med.

Motkraften är `(1 − s)` på båda ytorna. Det är en mot fyra.

---

## Del C — Vad som är motsagt i övriga dokument

### `docs/substratets-struktur.md`

> "Näringskonkurrensen i Steg 4 ska alltså vara symmetrisk, vilket den är — upptaget begränsas proportionellt av `uptake_capacity`."

**Fel.** Fördelningen skedde mot tillväxtunderskottet, inte mot upptagsförmågan. `uptake_capacity` band i noll av alla individer, fem tiopotenser från taket. Rättat i 0054; upptaget delas nu efter rotarea.

Avsnittet om höjd och ljus är däremot **genomfört** i 0061 och 0063, med en annan form än den föreslagna: Beer–Lambert i cellens bladarealindex i stället för en normerad potensdelning, och höjden ur skottet i stället för hela massan.

### `docs/vaxternas-livscykel.md`

> "Ljus löser båda med samma grepp: det gör struktur dyrt och storlek lönsamt."

**Falsifierat.** Ljuset gav strukturandelen en ny uppsida via höjden. Se Del B.

> "`uptake_capacity` … underhållskostnad (Steg 6)"

Fortfarande utestående. Kapaciteten binder nu men kostar ingenting att bära.

Talen i dokumentets kalibreringsavsnitt gäller: bördigheten 0,32 kg per cell står, `nutrient_loss_frac` 0,01 står, `nutrient_input` gick 7,1e-5 → 4,6e-5 efter mätning över 40 000 tick.

### `docs/naringens-ekonomi.md` och `docs/metabolismen.md`

Inget motsagt. Metabolismdokumentets konstaterande att massbalansen aldrig kan sluta sig är fortsatt korrekt och blev viktigare när ljuset infördes — kolet kommer ur luften och bokförs inte.

### Manifestet

> "En organism bär ett fast antal genloci — initialt i storleksordningen 8–16"

Faktiskt 38. Tre loci har tillkommit under det här arbetet (`_T_SEED_MASS`, `_T_ROOT_ALLOC`, `_T_MATURITY`) och två har bytt jobb i stället för att adderas (`_T_GROWTH` → allokering, `_T_DISPERSAL` → apparatandel). Avvikelsen bör accepteras skriftligt eller åtgärdas; den har noterats i tre statusanalyser utan att beslutas.

---

## Del D — Metodiska lärdomar

Dessa har kostat mer tid än buggarna.

**Ingenting bedöms före mättnad.** Fem gånger har ett tillstånd före jämvikt lästs som ett resultat. Två gånger gav det motsatt tecken mot sanningen: 0059 såg ut att misslyckas vid tick 3 000 och lyckades vid 8 000, och ljusnivån såg ut att vara tio gånger fel vid tick 6 000 när den var rätt inom en faktor två. Mättnad inträffar kring månad 600 av 800, och kriteriet är att fri näring och antal båda ligger still.

**Summor över sneda fördelningar är vilseledande.** Näringspoolens medelvärde är åtta gånger dess median. Det ledde till påståendet att reproduktionen var strypt av en bugg i tre tiopotenser, vilket 0067 motbevisade. Florans fördelningar är alltid sneda.

**Mutationens clip är taket, inte traitens intervall.** `mutate_trait_vector` klipper vid ±2,5 i logit-rymden, vilket ger 0,076 till 0,924 i normerad skala. En trait vid 0,789 av sitt intervall är pinnad, inte optimerad. Det förklarade strukturandelens asymptot och gjorde fröresultatet starkare än det såg ut.

**En trait med nominell tvåsidighet kan ändå vara död.** Apparatandelen låg stilla i 800 månader med en fullt rimlig avvägning på pappret, och vaknade först när trängseln blev verklig. Rot–skott gjorde samma sak: död i 0063, aktiv efter 0065. **En axel behöver något att svara på, inte bara en kostnad.**

**Att lossa en fastnaglad axel ogiltigförklarar kalibreringen bakom den.** Talen sattes i det låsta läget. Det inträffade med förnafallet när strukturandelen släppte.

**Tal som kan sterilisera en population ska inte vara konstanter.** Den hårdkodade `0,20 · vuxenmassa` gjorde floran steril i w1 och låste 944 kg näring. Ett locus kan inte göra det, eftersom en linje som inte reproducerar sig försvinner omedelbart.

---

## Del E — Vad som återstår, i ordning

### 1. Faunans dödlighet

Den allvarligaste kvarvarande defekten, och den är oförändrad sedan w1. Åldrandet går genom skada och det finns ingen inre klocka, så ett välmatt djur ackumulerar nästan ingen skada och blir funktionellt odödligt. Kumulativa dödsfall gick från 17 till 18 på 10 000 tick medan beståndet växte från 5 till 181.

Följden är att varje population är en överskjutning: den växer till väggen och dör samlat. Alla fem körningarna slutar med utdöende. Ingen faunamätning är meningsfull förrän detta är åtgärdat.

### 2. Kol och kväve på faunasidan

Floran har två valutor och `min()` över dem. Faunan har en — energi i joule — och kvävet passerar rakt igenom och utsöndras utan att någonsin begränsa. Ekologisk stökiometri säger att herbivorer typiskt är kvävebegränsade, eftersom växtvävnad ligger på C:N 30–500 och djurvävnad på 5–10.

Att införa symmetrin gör också massbalansen granskbar över trofigränsen, vilket den inte är i dag: florans kol mäts i kilo vävnad och faunans i joule.

Bör komma **efter** dödligheten, av samma skäl som ovan.

### 3. Strukturandelens fjärde motkraft

Ingen kandidat är övertygande. Bygg inte förrän det finns en som är mekanism och inte parameter. Möjligen kommer den från faunasidan när betningen blir kvävedriven, men det är en förhoppning och inte en plan.

### 4. `uptake_capacity` som kostnad

A2 i manifestet. Kapaciteten binder nu men kostar ingenting att bära, så det finns inget tryck mot enkelhet.

### 5. Hydro

Oförändrat läge. `flood_tolerance` och `buoyancy` har fortfarande noll läsare.

---

## Del F — Vad som ska mätas

| Egenskap | Målvärde | Nuläge |
|---|---|---|
| Näringsdrift över 40 000 tick | < 1e-6 relativt | **4,6e-10** |
| Traits med inre optimum | fler än noll | **2** (frömassa, rotandel) |
| Strukturandelens lutning vid mättnad | 0 | +1,3e-04/mån |
| `flora_light_limited` vid mättnad | 0,4–0,6 | 0,22 |
| Faunans överlevnad över 40 000 tick | > 0 | **0 i fem av fem** |
| Kapacitetsfält utan läsare | 0 | `flood_tolerance`, `buoyancy`, `growth_capacity` |
| Kostnad per floraindivid | < 1 µs | **0,29 µs** |

---

*Skriven efter 0067. Ersätts när nästa arbetsblock är klart. Det som står i Del D är det som är värt att behålla längst.*
