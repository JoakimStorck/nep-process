# nep-process — Utvecklingsplan

*Juli 2026. Kompletterar arkitekturmanifestet i README.md: manifestet beskriver vad vi bygger, det här dokumentet var vi står och vad som kommer härnäst.*

---

## Sammanfattande bedömning

Arkitekturen rör sig åt rätt håll. Fauna har en verklig passkedja, världen har primära fält och en delegerande `World.step()`, och store:n bär mer tillstånd än tidigare.

Men det finns ett systematiskt glapp mellan manifestets mekanismer och koden: **kapacitetsmodellen finns som arrayer och driver ingenting.** Samtliga kapacitetsfält har noll läsare, med ett undantag. Ontologin är deklarerad, inte verksam.

Världslagret har samtidigt fått fält utan dynamik och utan konsumenter: `nutrient` allokeras och rörs aldrig, `transport_pass()` returnerar noll, `decomposition_pass()` har `dM_nutrient_from_detritus = 0.0` hårdkodat, `flow_strength` nollställs varje tick. Skelettet är rätt byggt — det är bara tomt, och tomrummet växer.

Och den ekologiska defekten är oförändrad: **floran har ingen begränsande resurs och ingen dödlighet.** Sedan store:n blev dynamiskt växande har den inte längre något tak alls.

---

# Del A — Målsättningar

Manifestets principer formulerade som prövbara påståenden om vad koden ska kunna.

**A1. Enhetlig ontologi.** Skillnaden mellan växt och rovdjur är kapacitetsprofilens värden, inte klasstillhörighet.

**A2. Kapaciteter kostar.** Underhålls-, aktiverings- och strukturkostnad. En kapacitet som är noll ska varken kosta energi eller CPU.

**A3. Enkla organismer är billiga.** Tiotusentals enkla samtidigt med hundratals komplexa.

**A4. Fasbaserad exekvering.** Systempass mot förberedda, immutabla delmängder. Pass arbetar mot arrayer, inte objekt.

**A5. Dataorienterad kärna.** SoA, minimal kärna, subsystemstate i separata tilläggsarrayer.

**A6. Abstrakt geometri.** `Grid` är den enda plats där geometrin existerar.

**A7. Fysik / värld / biologi som skilda lager.**

**A8. Lokal upptäckt.** Ingen global iteration.

**A9. Levande primärproduktion.** Floran konkurrerar, differentierar sig, koevolverar.

**A10. Emergent beteende som primärt evolutionärt mål.**

**A11. Terrängburen hydrologi.** Primära fält, forcing-fält, delegator, hydro över topografi, `detritus` ersätter `C`.

---

# Del B — Status

| # | Målsättning | Status |
|---|---|---|
| A1 | Enhetlig ontologi | Deklarerad, inte verksam |
| A2 | Kapaciteter kostar | Saknas |
| A3 | Enkla organismer är billiga | ~50–100× från mål |
| A4 | Fasbaserad exekvering | Delvis — passkedja finns, delmängder saknas |
| A5 | Dataorienterad kärna | Delvis |
| A6 | Abstrakt geometri | Delvis, försämras med varje nytt världsfält |
| A7 | Lagerseparation | Strukturellt på plats |
| A8 | Lokal upptäckt | **Uppfylld** |
| A9 | Levande primärproduktion | Strukturellt ja, funktionellt nej |
| A10 | Emergent beteende | **Uppfylld i mekanik** |
| A11 | Hydrologi | Etapp A klar, B och C tomma |

## B1. Kapacitetsprofilen är kopplad till ingenting

Läsare respektive skrivare per kapacitetsfält i `OrganismStore`:

```
uptake_capacity     0 läsare     growth_capacity     0 läsare
dispersal_capacity  0 läsare     sense_radius        0 läsare
sense_rate          0 läsare     mobility            0 läsare
attack_capacity     0 läsare     repair_capacity     0 läsare
flood_tolerance     0 läsare     buoyancy            0 läsare
genome_idx          0 läsare     repro_capacity      1 läsare
```

Detta är kärnpunkten i hela analysen. A1 och A2 är inte delvis implementerade — de är inte implementerade. Det finns ingen plats där en kapacitets underhållskostnad skulle kunna dras, eftersom ingen kostnadsfunktion känner till kapaciteterna.

`age` har tre läsare och `energy_cap` en, vilket är verklig progress från Fas 4.

## B2. Fauna: verklig passkedja, halvmigrerat tillstånd

`Population.step()` kör sex namngivna fauna-pass med explicita batch-objekt mellan sig — `_step_metabolism_system`, `_step_sense_system`, `_step_decision_system`, `_step_move_system`, `_step_body_system`, `_step_interaction_system`. Den generella per-tick-speglingen via `write_agent()` är borta; den anropas bara vid init och födsel. `Body.step()` är intakt som fysiologikärna men anropas från ett eget pass.

Det är verklig arkitektur och motsvarar manifestets Fas 4-ordning. Tillståndet är däremot bara delvis migrerat, i tre lägen:

**Genuint store-ägda:** `age`, `repro_cd`. Skrivs i passen, läses därifrån.

**Speglade från Body:** `mass`, `energy`, `energy_cap`, `damage`, `wear`. En synkhjälpare kopierar från `a.body.*`. `Body` är source of truth. Ärligt men omigrerat.

**Delat ägarskap:** `gestating`, `gest_M`, `gest_E_J`. Skrivs från `a.body.*` och läses som auktoritativa ur store. Två skrivare till samma fält är exakt det mönster manifestet förbjuder. Det är i dag tre fält; det bör lösas innan det blir tio.

## B3. Aktiva delmängder finns inte

Ingen förekomst av delmängdslogik. Per tick körs tre fulla Python-loopar över `range(store.n)` i florapassen, plus `rebuild_spatial_index` — som anropas **två gånger per tick** och själv innehåller två fulla loopar. Kostnaden skalar med kapaciteten, inte med antalet levande organismer.

## B4. Floran är en per-cell-biomassa i organismform

Tre observationer som tillsammans betyder att A9 ännu inte har prövats:

**Max en floraindivid per cell.** `_add_or_create_flora_in_cell()` adderar massa till befintlig flora i målcellen; ny slot allokeras bara om cellen är tom. Ingen konkurrens inom cell.

**Ingen begränsande resurs.** Logistisk tillväxt mot individens egen `flora_adult_mass`, grindad på temperatur. Ordet `nutrient` förekommer inte en enda gång i `population.py`. `uptake_capacity` beräknas vid födsel och läses aldrig.

**Ingen dödlighet.** Ingen wither, ingen senescens. Flora försvinner endast genom att ätas ner under en tröskel.

Konsekvensen är mätbar. Körning, 40 000 tick, `size=64`, `max_pop=256`, seed 1:

```
tick        fauna   flora   capacity
    0          12     128        256
20000          21     194        256
30000          16     323        512
40000           3     529       1024
```

Floran växer monotont, och sedan store:n blev dynamiskt växande finns inget som hejdar den — det arkitektoniska taket är borta och inget ekologiskt har ersatt det. Seed 2 och 3 över 20 000 tick ger samma monotona floratillväxt (till 172 respektive 233) med stabil fauna, så faunanedgången i seed 1 är en sen händelse som inte är belagd som systematisk.

Fas 2:s valideringsfrågor — *uppstår stabila florapopulationer? differentierar sig florastrategier via selektion?* — har därmed aldrig fått ett meningsfullt svar. Selektionstrycket är för svagt för att differentiera något: allt som växer överlever tills det äts.

## B5. Skalningsmålet är 50–100× bort

Uppmätt, 12 fauna, varierande store-kapacitet:

```
flora  130  ->   3.96 ms/tick
flora  517  ->   7.59 ms/tick
flora 2074  ->  19.81 ms/tick
```

Marginellt ungefär **8 µs per floraindivid och tick**. Extrapolerat till 10 000 flora: ~80 ms/tick. A3 kräver att siffran ner under en mikrosekund, vilket är vad vektoriserade array-pass ger.

## B6. Geometrin — och fältrepresentationen bakom den

`grid.py` är oförändrad sedan fas 3. `grid.neighbors()` har noll anropare. Fyra `np.roll`-laplacianer kvar i `abiotic.py`. Sjutton direkta koordinatläckor utanför `Grid` (`rowcol_of`, `bilinear_*`, `grid.size`).

Det avgörande talet: **elva 2D-fält** med formen `(s, s)` i `world.py`. Temperaturen är en radvektor `Ty[row]`, och på hex finns ingen rad. Hexbytet begränsas alltså inte av passlogiken utan av fältrepresentationen, och varje nytt världsfält som tillkommer före cellindexeringen ökar konverteringskostnaden linjärt.

## B7. Traitsemantiken

Locuskartan ägs korrekt av `phenotype.py`, som definierar `_T_*` för samtliga 32 loci med flora på 25–31. Skalningsintervallen låg tidigare duplicerade i `population.py` med hårdkodade heltalsindex; det är löst i Steg 0.

## B8. Testinfrastruktur

`run_headless.py` kör utan pygame med exitkod vid invariantbrott. `invariants.py` prövar sju invarianter: slotbokföring, arrayernas indexdomäner, `cell_idx`-konsistens, id-unikhet och id→slot-bijektion, finita och icke-negativa storheter, spatialindexets integritet, samt `Agent.store_slot`-bindning.

Massa- och energibalans är ännu diagnostik, inte assertion: systemet är öppet by construction och får en sluten balans först när näringskretsloppet finns.

## B9. Genomet

Manifestet anger 8–16 loci initialt. Faktiskt `n_traits = 32`. Bör antingen accepteras skriftligt eller åtgärdas.

---

# Del C — Ordningsfrågan

## Näringskretsloppet före hydro

Det ligger nära till hands att bygga hydro först: det är den svåraste world-processen och tvingar fram rätt lagerindelning. Men hydro har inga konsumenter. `flood_tolerance` och `buoyancy` läses inte, passiv drift är uppskjuten, framkomlighet likaså.

Vi behöver inte spekulera om vad som händer när man bygger produktion utan konsumtion — det har redan hänt med `nutrient`, som ligger allokerat och orört.

`nutrient` och `detritus` har däremot en konsument som väntar och en defekt som behöver dem. Kedjan **död → detritus → nedbrytning → näring → upptag → tillväxt → betning → död** är den minsta slutna slingan i modellen, och den kan byggas utan hydro. Den ger samtidigt:

- den första verkliga läsaren av ett kapacitetsfält (`uptake_capacity`)
- selektionstryck som kan differentiera florastrategier — Fas 2:s hypotes blir prövbar
- ett ekologiskt tak på floran i stället för inget tak alls
- en sluten massbalans, vilket gör ledgern till en hård invariant i stället för diagnostik

Och hydro blir lättare efteråt: när `nutrient` redan transporteras över topologiska grannar med tvåstegsmetod är hydro samma mönster med en annan drivande gradient.

## Hex före ekologin

Manifestet säger att hex ska komma när `Grid`-abstraktionen är ren, så att geometriombyggnad inte blandas med biologisk kalibrering. Villkoret handlar om att de två arbetena inte ska överlappa — inte om att geometrin ska komma sist.

Just nu finns nästan ingen ekologisk kalibrering att skydda: floran är degenererad, näringskretsloppet finns inte, hydro är tomt. Faunans fysiologi är kalibrerad, men den är geometrioberoende — energi, massa och skada bryr sig inte om cellform.

Byter vi geometri efter att ekologin kalibrerats får vi göra om kalibreringen. Grannantalet går från 4 till 6, `cells_within(1)` från 5 celler till 7, och diffusions- och spridningstakter skiftar med dem. Det är dubbelarbete som kan undvikas.

Men inte som nästa handling. Det dyra med hex är inte hexgeometrin utan fältrepresentationen. Byter vi `Grid`-implementation i dag måste `world.py`, `population.py`, `agent.py` och viewern ändras — vilket per planens eget mätkriterium betyder att abstraktionen inte bär bytet. Vi skulle konvertera kvadratbunden 2D-kod till hexbunden 2D-kod.

**Därför: cellindexerade fält först, hex omedelbart därefter, all ekologi sedan i slutgeometrin.**

Bilinjär sampling har ingen naturlig hexmotsvarighet och bör tas bort snarare än översättas. Det är i linje med manifestet, som säger att biologin ska arbeta via cell-ID och aldrig direkt mot råa koordinater. En förenkling, inte en uppoffring.

---

# Del D — Plan

## Steg 0 — Stabilisering och mätbarhet

- ~~Headless entrypoint utan pygame-import.~~ **Klart.**
- ~~Invariantsvit som smoke test.~~ **Klart.**
- ~~Gemensam organism-id-rymd för flora och fauna.~~ **Klart** — fynd från sviten.
- ~~Indexdomäner vid dynamisk store-tillväxt.~~ **Klart** — latent krasch när `capacity` sammanföll med `n_cells`.
- ~~Floras traitsemantik flyttad till `phenotype.py`.~~ **Klart.**
- ~~Arbetsgrenen insmält i `main`.~~ **Klart.** Långlivade parallella grenar är avvecklade som arbetssätt.
- Lös delat ägarskap för `gestating`, `gest_M`, `gest_E_J`. Välj en ägare per fält och dokumentera valet.

**Klart när:** ett kommando kör 10 000 tick headless med godkänd invariantsvit, och inget fält har två skrivare.

## Steg 1 — Cellindexerade fält och grannmatris

*Förutsättningen för hex. Blir dyrare för varje världsfält som tillkommer.*

- Alla världsfält blir platta arrayer med längd `n_cells`, indexerade med `cell_idx`. Ingen `[y, x]`-indexering utanför `Grid`. Det gäller samtliga elva fält.
- Temperatur blir ett per-cell-fält i stället för `Ty[row]`. Latitudprofilen genereras en gång av `Grid` som en per-cell-egenskap.
- `Grid` får en förberäknad grannmatris: `neighbor_idx` med form `(n_cells, k)` i `int32` plus giltighetsmask. Det gör topologiska pass vektoriserbara utan Python-loop och gör `neighbors()` användbar i praktiken.
- Bilinjär sampling avvecklas. Konsumtion och perception läser den innehållande cellen via `cell_idx` och grannar via grannmatrisen.

**Klart när:** ingen kod utanför `Grid` refererar rad, kolumn eller `[y, x]`, och `bilinear_*` är borta.

## Steg 2 — Hex

- `Grid` implementeras om med axialkoordinater `(q, r)`. Cell-ID är heltal mappade från axialkoordinater.
- Grannmatrisens generering ger sex grannar per cell.
- `distance()` blir hexavstånd, `cells_within(r)` ger 1, 7, 19 … celler.
- Viewern översätter cell-ID via `Grid` och ritar hexceller.
- Diffusions- och spridningsparametrar justeras för det nya grannantalet — en gång, innan ekologin byggs.

**Klart när:** allt utanför `grid.py` och viewern är oförändrat. Krävs större ändringar i world eller biologi är Steg 1 inte färdigt, och felet ska rättas där.

## Steg 3 — Näringskretsloppet

- `detritus` blir ensam source of truth. `C`-aliaset tas bort.
- `decomposition_pass()`: `detritus → nutrient` plus förlustterm. Reaktion, ingen transport.
- `transport_pass()`: diffusion av `nutrient` via grannmatrisen, tvåstegsmetod.
- `uptake_pass()`: flora tar upp `nutrient` från sin cell, begränsat av `uptake_capacity` och lokal tillgång. **Första verkliga läsaren av ett kapacitetsfält.**
- Floran får dödlighet: senescens eller temperaturberoende mortalitet, så att `detritus` fylls på från floran och inte bara från kadaver.
- Flera floraindivider per cell tillåts, konkurrerande om samma cellnäring.

**Klart när:** floran når ett stationärt tillstånd satt av näringstillgång; massan sluter sig i ledgern så att balansen kan bli hård invariant; och flora med olika `uptake_capacity` uppvisar mätbart olika överlevnad.

Här får Fas 2:s ekologiska hypotes sitt första riktiga svar. Blir svaret nej — revidera floramodellen, inte kärnan.

## Steg 4 — Aktiva delmängder och vektoriserade florapass

- Delmängder byggs en gång per tick, immutabla under ticken: `flora_slots`, `fauna_slots`, `sensing_slots`.
- Florapassen skrivs om som numpy-operationer över `flora_slots`.
- `rebuild_spatial_index` vektoriseras med `bincount` + `argsort` och anropas **en** gång per tick.
- Flora-tilläggsfält flyttas till komprimerade tilläggsarrayer indexerade via flora-delmängden. Uppfyller A5.

**Klart när:** 10 000 flora körs under 10 ms/tick.

## Steg 5 — Fauna store-first och kapacitetskostnader

*Störst risk, störst utdelning.*

- `Body`:s skalära tillstånd flyttas fält för fält till store-arrayer med dokumenterat ägarskap. `Body.step()` behålls som sammanhållen fysiologikärna men opererar på store-slices.
- Kapacitetsfälten kopplas till läsare: `sense_radius` och `sense_rate` styr sensing-delmängden, `mobility` styr rörelsekostnad, `attack_capacity` styr predation. Efter detta ska listan i B1 vara tom.
- Underhållskostnad per buren kapacitet införs i metabolismen. Det är A2, och först här får evolutionen ett tryck mot enkelhet.
- Synkhjälparen avvecklas när sista fältet bytt ägare.

**Klart när:** inget fauna-tillstånd har två skrivare, och en organism med `sense_radius = 0` aldrig berör sensing-koden.

## Steg 6 — Hydro

- `elevation` får en terränggenerator: lutande plan, bassänger, höjdryggar.
- `hydro_pass()` över fri yta `elevation + water` med grannflöde, tvåstegsmetod, strikt kontinuitet. Härledda fält som del av samma passkontrakt.
- `flood_tolerance` och `buoyancy` får läsare i locomotion och rörelsekostnad. Passiv drift i hydro-passet.

Att `nutrient` sedan Steg 3 transporteras med samma mönster gör hydro till en variant snarare än en nyhet.

## Steg 7 — Acceleration

Profilera fasmodellen under blandad realistisk belastning. Numba för sensing, CuPy för fältpass om de dominerar. Rustkärna endast om mätvärden motiverar det.

---

# Del E — Vad som ska mätas

| Steg | Mätpunkt | Målvärde |
|---|---|---|
| 0 | Invariantsvit över 10 000 tick | 0 brott |
| 0 | Fält med två skrivare | 0 |
| 1 | Referenser till rad, kolumn eller `[y, x]` utanför `Grid` | 0 |
| 2 | Filer ändrade vid hexbytet | endast `grid.py` och viewer |
| 3 | Floran når ett stationärt antal | ja |
| 3 | Stationärt antal vid dubblad `capacity` | oförändrat ±10 % |
| 3 | Överlevnadsskillnad mellan hög och låg `uptake_capacity` | statistiskt skild |
| 3 | Massbalans i ledgern | sluten inom 1e-9 relativt |
| 4 | Kostnad per floraindivid och tick | < 1 µs |
| 4 | 10 000 flora | < 10 ms/tick |
| 5 | Kapacitetsfält utan läsare | 0 |
| 5 | Kostnad per fauna och tick | mät baseline, tillåt ej regression |
| 6 | Massbevarande i hydro över 10 000 tick | drift < 1e-6 relativt |

Att floran över huvud taget når ett stationärt tillstånd, och att det tillståndet är okänsligt för `capacity`, är den viktigaste enskilda mätningen i planen. Den avgör om ekologin har tagit över från arkitekturen.

**Baseline:** 2,50 ms/tick vid 22 fauna och 149 flora, `size=64`, seed 1, 12 000 tick.

---

## Om ordningen ändå ska vara en annan

Vill man hålla fast vid hydro tidigt är det försvarbart — men gör Steg 3:s floradödlighet och konkurrens ändå först, som ett minimalt ingrepp. Utan dem mäter man hydro i en värld där floran växer obehindrat.

Vill man skjuta hex till efter ekologin är också det försvarbart, men räkna då med att kalibrera diffusion, spridning och celltäthet två gånger.

Det som däremot inte bör flyttas är Steg 1. Cellindexerade fält är den enda ändringen i planen som blir strikt dyrare för varje pass som byggs innan den är gjord.
