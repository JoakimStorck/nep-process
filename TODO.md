# nep-process — Utvecklingsplan

*Juli 2026. Kompletterar arkitekturmanifestet i README.md: manifestet beskriver vad vi bygger, det här dokumentet var vi står och vad som kommer härnäst.*

---

## Sammanfattande bedömning

Arkitekturen rör sig åt rätt håll. Fauna har en verklig passkedja, världen har primära fält och en delegerande `World.step()`, och store:n bär mer tillstånd än tidigare.

Men det finns ett systematiskt glapp mellan manifestets mekanismer och koden: **kapacitetsmodellen finns som arrayer och driver ingenting.** Samtliga kapacitetsfält har noll läsare, med ett undantag. Ontologin är deklarerad, inte verksam.

Världslagret har samtidigt fått fält utan dynamik och utan konsumenter: `nutrient` allokeras och rörs aldrig, `transport_pass()` returnerar noll, `decomposition_pass()` har `dM_nutrient_from_detritus = 0.0` hårdkodat, `flow_strength` nollställs varje tick. Skelettet är rätt byggt — det är bara tomt, och tomrummet växer.

Näringskretsloppet är sedan Steg 4 slutet hela vägen runt, fauna inräknad, och prövas som hård invariant. Det som återstår i steget är kalibreringen — och den är nu meningsfull, eftersom poolen inte längre fylls på av en bokföringsbugg.

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

**Gestationscache:** `gestating`, `gest_M`, `gest_E_J`, `gest_M_target`. `Body` äger dem fram till Steg 6b, eftersom `gest_M` ackumuleras inne i `Body.step()`:s energibudget där fostermassan också belastar buren massa. Store-fälten är en envägscache för de slotbaserade reproduktionsgrindarna, skriven via `_write_gestation_to_store()` och bevakad av en invariant.

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

`run_headless.py` kör utan pygame med exitkod vid invariantbrott. `invariants.py` prövar nio invarianter: slotbokföring, speglingen mellan `Body` och store, arrayernas indexdomäner, `cell_idx`-konsistens, id-unikhet och id→slot-bijektion, finita och icke-negativa storheter, spatialindexets integritet, `Agent.store_slot`-bindning, samt näringsbalansen.

Näringsbalansen är hård sedan faunabudgeten stängdes. Massbalansen förblir diagnostik och ska så förbli: flora bygger merparten av sin vävnad ur luft och faunans kol lämnar kroppen som koldioxid, så total massa kan aldrig sluta sig. Det som cirkulerar är näringen.

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
- ~~Entydigt ägarskap för gestationstillståndet.~~ **Klart** — `Body` äger, store speglar envägs, divergens fångas av invariant.

**Steg 0 är avslutat.** `python run_headless.py --ticks 12000 --check-every 500` kör grönt, och inget fält har två skrivare.

## Steg 1 — Cellindexerade fält och grannmatris

*Förutsättningen för hex. Blir dyrare för varje världsfält som tillkommer.*

- ~~Alla världsfält blir platta arrayer med längd `n_cells`.~~ **Klart** — samtliga elva plus `T_cell` och `g_cell`.
- ~~Temperatur blir ett per-cell-fält i stället för `Ty[row]`.~~ **Klart** — latitudprofilen kommer från `Grid.cell_lat`.
- ~~`Grid` får en förberäknad grannmatris.~~ **Klart** — `neighbor_idx` `(n_cells, k)` int32 med mask, plus `cell_lat`, `cell_center_x/y`, `cell_of_many` och `wrap_pos_inplace`.
- ~~Bilinjär sampling avvecklas.~~ **Klart** — perception, konsumtion, temperatur och kadaverdeposition läser celler. `bilinear_*` borta ur både `world.py` och `grid.py`.

**Steg 1 är avslutat.** World och biologi refererar inte längre rad, kolumn eller `[y, x]`. Kvarvarande rutnätsberoende är isolerat till `World._as_2d()` med dess enda anropare `World.C`, samt `Ty`/`gy`-egenskaperna — alla tre finns bara för viewer och simlog, och avvecklas i Steg 2.

Mätt: 0 träffar på `rowcol_of`, `cell_from_rowcol`, `grid.size` och `bilinear_*` utanför `grid.py`, viewern och simlog.

## Steg 2 — Hex

- ~~`Grid` implementeras om med hexgeometri.~~ **Klart.** Spetsig hexagon med radoffset, jämnt radantal, cellarea exakt 1 så att täthetsstorheter bevaras.
- ~~Grannmatrisens generering ger sex grannar per cell.~~ **Klart.**
- ~~`distance()` blir hexavstånd, `cells_within(r)` ger 1, 7, 19 … celler.~~ **Klart.** `cells_within` byggs med BFS i grannmatrisen; `distance` med kubkoordinater och torusens nio translationskandidater, validerad mot BFS.
- ~~Viewern översätter cell-ID via `Grid` och ritar hexceller.~~ **Klart.** Varje pixel slås upp mot sin cell via `cell_of_many`, vilket gör renderingen geometriagnostisk.
- ~~`Grid` tar `width` och `height` separat.~~ **Klart.** Hexvärlden är aldrig kvadratisk i kontinuerliga enheter, och radoffset kräver jämnt radantal.
- ~~Latituden görs periodisk, `lat = -cos(2π·r/H)`.~~ **Klart.** Den linjära profilen gav ett enda kallt band som wrappade, med en 24-gradig säsongsdiskontinuitet mitt i sig — ingen isolering plus ett artefakt. Den periodiska ger två kalla band åtskilda av två tempererade zoner, med motfasiga årstider.
- ~~Världsstorleken sätts till 64 × 256 som standard.~~ **Klart.** Periodisk latitud halverar bandbredden, så H=128 hade bara återställt nuläget medan H=256 fördubblar sträckan pol till ekvator. `--size N` finns kvar för kvadratiska världar.

**Steg 2 är avslutat.**

**Klart när:** allt utanför `grid.py` och viewern är oförändrat. Krävs större ändringar i world eller biologi är Steg 1 inte färdigt, och felet ska rättas där.

**Utfall:** geometribytet ändrade enbart `grid.py`. Kriteriet höll. Efter viewerns omskrivning har `world.C`, `Ty`, `gy` och `_as_2d` inga anropare kvar och är borttagna — det sista kvadratberoendet är ute ur `world.py`.

**Uppmätt vid klimatbytet:** sömmen går från 24,10 °C till 0,02 °C vid 64×256, och största steg mellan intilliggande rader från 2,24 till 0,96 °C — gradienten blir alltså både sammanhängande och jämnare än med den linjära profilen. Andel agenter som byter klimatband under livstiden faller från 79 % till 50 %.

## Steg 3 — Världens kadensmodell

*Förutsättningen för en stor värld. Underlag: `docs/varldens-kadensmodell.md`.*

Terrängburen hydrologi kräver en värld stor nog att bära höjdskillnader och vattendrag. Vid en miljon celler kostar världspasset ~26 ms per tick, och ungefär 15 av dem är representationsfel snarare än beräkning: temperaturen räknas om per cell trots att den bara beror på latitud, decomposition sveper hela världen för att multiplicera nollor, och samtliga fält lagras som arrayer trots att de är rumsligt konstanta.

- ~~Varje världsfält får en deklarerad **kadensklass**.~~ **Klart** — statisk, profilberoende, glest dynamisk eller tätt dynamisk.
- ~~Statiska fält sveps aldrig och lagras som skalär.~~ **Klart** — `elevation` och de fyra forcing-fälten. `surface_level`, `submerged` och `flow_strength` beräknas vid läsning i stället för varje tick.
- ~~Klimatet blir en profil per latitudband.~~ **Klart** — `T_band`/`g_band` med längd `n_bands`, lästa via `Grid.band_of_cell()` och `bands_of_cells()`.
- ~~`detritus` blir glest dynamiskt med aktiv mängd.~~ **Klart** — kontraktet att inaktiva celler är exakt noll prövas av `invariants.check_sparse_fields()`.
- ~~Ledgern underhålls inkrementellt.~~ **Klart** — hydro och decomposition räknar sina termer ur den aktiva mängden respektive analytiskt ur det konstanta tillskottet, utan fulla summeringar.

**Steg 3 är avslutat.** Världspasset vid en miljon celler gick från 14,4 ms till 0,047 ms — målet var storleksordningen 10.

```
                   före      efter
temperature_pass   2,02  ->  0,010 ms
hydro_pass         6,12  ->  0,000 ms   (0,17 med regn)
decomposition      3,78  ->  0,029 ms   (1480 aktiva celler)
hela step()       14,41  ->  0,047 ms
```

Bilden ändras i Steg 4, där `nutrient` blir tätt dynamiskt och transport införs. Det är avsiktligt: den klassen finns för fält där fullt svep är genuint motiverat.

Kadensklasserna ska in i manifestet innan hydro byggs, så att `hydro_pass` och `transport_pass` skrivs mot rätt struktur från början. Glesning av hydro självt hör till Steg 8 — den kräver en tät referensimplementation att välja epsilon mot och validera massbevarande emot.

## Steg 4 — Näringskretsloppet

- ~~`detritus` blir ensam source of truth.~~ **Klart.**
- ~~`decomposition_pass()`: `detritus → nutrient` plus förlustterm.~~ **Klart**, och per fraktion: labilt och strukturellt material bryts ner med var sin takt, räknade ur massa och strukturandel utan ett andra fält.
- ~~`transport_pass()`: diffusion via grannmatrisen, tvåstegsmetod.~~ **Klart.** Massbevarandet är exakt av strukturella skäl; `nutrient` lagras i float64 för att avvikelsen ska hamna kring 1e-15 i stället för 5e-7.
- ~~`uptake_pass()` med `uptake_capacity` som begränsning.~~ **Klart** — kapacitetsmodellens första verkliga läsare. Fusionerad med tillväxten, eftersom upptaget avgör exakt hur mycket tillväxt som kan betalas.
- ~~Floran får dödlighet.~~ **Klart.** Senescens med förhöjd risk för individer långt under vuxenmassa, så att groddplantor utan näring försvinner.
- ~~Flera floraindivider per cell.~~ **Klart.** Varje frö blir en egen individ; antalet begränsas av näringen, inte av en regel.
- ~~Substratets strukturandel.~~ **Klart.** `structure` som gemensamt locus med fem konsumenter: energitäthet, betningsutbyte, nedbrytningstakt, näringskostnad vid uppbyggnad och exkrementets sammansättning. Se `docs/substratets-struktur.md`.
- ~~Exkretion.~~ **Klart.** Icke assimilerad massa återförs som detritus, mer strukturrik än födan.
- ~~Sluten näringsbudget över fauna.~~ **Klart.** Reserven bär massa som är en del av kroppen, tillväxt och gestation kräver material 1:1 ur reserven utöver syntesarbetet, och förbränningens näring utsöndras till cellen. `check_nutrient_balance()` är hård invariant.
- Kalibrering mot mätpunkterna. **Återstår** — se överlämningsanteckningarna nedan. Punkten låg först i Steg 2 men flyttades hit: `transport_pass()` returnerade noll, så det fanns ingen diffusion att kalibrera, och att finjustera floras spridningstakt i en modell utan mortalitet vore att kalibrera fel sak. Hela tillväxtdynamiken byter karaktär här ändå.

### Substratets struktur

*Underlag: `docs/substratets-struktur.md`.*

När floran får mortalitet hamnar dött växtmaterial i samma `detritus` som kadaver. För att asätare och detritivorer ska kunna differentiera sig bär detritus en ärvd egenskap i stället för att delas i två fält — ett typfält vore just den artgräns manifestet säger att vi inte kodar.

Egenskapen är **strukturandel**: hur stor del av vävnaden som är segt bärande material — lignin och cellulosa hos växter, kitin och ben hos djur — mot lättomsatt protein, socker och fett. Samma tal beskriver båda ontologierna, vilket är vad manifestet kräver.

Manifestets `digestibility` utgår. Den har ingen uppsida: ingen organism vinner på att vara lättare att äta, så selektionen driver den mot noll och axeln dör. `defense` lämnas orörd — aktivt avskräckande är inte samma sak som passiv seghet.

Följande konsekvenser införs i det här steget, alla med omedelbara läsare:

- `structure` som locus, gemensamt för flora och fauna
- byggkostnad i tillväxt: strukturmaterial kostar mer energi per kilo massa
- energitäthet: strukturmassa lagrar ingen användbar energi, så reserven per kilo sjunker
- nedbrytningstakt som avtar med strukturandel
- näringsfrisättning som följer nedbrytningen, alltså utdragen i stället för pulsad
- betningsutbyte per kilo som sjunker med strukturandel

`detritus` får ett andra glest fält med massviktad medelstrukturandel i samma aktiva mängd och under samma kontrakt.

Underhållskostnad som *sjunker* med strukturandel, och matsmältningskapacitet på konsumentsidan, hör till Steg 6 tillsammans med övriga kapacitetskostnader.

**Krav på varje ny trait.** En trait med en enda konsekvens är ett reglage, inte en anpassning — selektionen hittar optimum och stannar. Diversitet uppstår först när samma tal har motverkande konsekvenser, så att olika nischer gynnar olika värden. Varje ny trait ska prövas mot det: kan den dras åt båda håll av olika selektionstryck? Kan den inte det är den antingen en konstant i förklädnad eller en axel som kommer att kollapsa.

**Klart när:** floran når ett stationärt tillstånd satt av näringstillgång; **närings**balansen sluter sig så att den kan bli hård invariant — total massa kan aldrig sluta sig, eftersom flora bygger merparten av sin vävnad ur luft och fauna växer ur en energibudget; det som cirkulerar är näringen; flora med olika `uptake_capacity` uppvisar mätbart olika överlevnad; och `structure` uppvisar spridning i stället för att kollapsa mot en ände.

Här får Fas 2:s ekologiska hypotes sitt första riktiga svar. Blir svaret nej — revidera floramodellen, inte kärnan.

## Steg 5 — Aktiva delmängder och vektoriserade florapass

- Delmängder byggs en gång per tick, immutabla under ticken: `flora_slots`, `fauna_slots`, `sensing_slots`.
- Florapassen skrivs om som numpy-operationer över `flora_slots`.
- `rebuild_spatial_index` indexerar bara bebodda celler i stället för hela cellrymden, och anropas **en** gång per tick. CSR över `n_cells` gör `cumsum` över en miljon element för att indexera tiotusen organismer; sortering plus `searchsorted` tar bort `n_cells`-termen helt.
- Flora-tilläggsfält flyttas till komprimerade tilläggsarrayer indexerade via flora-delmängden. Uppfyller A5.

**Klart när:** 10 000 flora körs under 10 ms/tick.

## Steg 6a — Sensing som evolverbar kapacitet

*Kräver Steg 4 för selektionstrycket och Steg 5 för delmängdsmaskineriet, men inte hela fauna-migreringen. Underlag: `docs/synens-axlar.md`.*

Synen är i dag evolverbar i fyra diskreta steg längs en sammanslagen axel. Det som saknas är att axlarna kan handlas mot varandra och att kapaciteten kostar även när den inte används.

- `sense_radius` och `sense_rate` blir kontinuerliga genetiska axlar med läsare: radien avgör vilka celler som läses, frekvensen avgör vilka slots som ingår i sensing-delmängden.
- Vinkelupplösning och synfältsform blir egna axlar. Fix antal riktningssektorer, där akuiteten styr hur mycket de blandas — så att MLP:ns indimension är oberoende av genotypen och vikter förblir ärftliga.
- Strukturkostnad införs: att bära kapaciteten kostar även oanvänd. Aktiveringskostnaden faller ut ur geometrin, eftersom `cells_within(r)` växer som ~3r² på hex.
- Strålbaserad sampling ersätts av aggregering över `cells_within()` grupperad via grannmatrisen.

**Klart när:** en organism med `sense_radius → 0` aldrig berör sensing-koden, och spridningen i `sense_radius` differentierar mot nisch med kostnadsmodellen påslagen men driftar neutralt utan den.

## Steg 6b — Fauna store-first

*Störst risk, störst utdelning.*

- `Body`:s skalära tillstånd flyttas fält för fält till store-arrayer med dokumenterat ägarskap. `Body.step()` behålls som sammanhållen fysiologikärna men opererar på store-slices.
- Gestationstillståndet byter ägare från `Body` till store när `gest_M`-ackumulationen flyttat med.
- Återstående kapacitetsfält kopplas till läsare: `mobility` styr rörelsekostnad, `attack_capacity` styr predation. Efter detta ska listan i B1 vara tom.
- Underhållskostnad per buren kapacitet generaliseras från sensing till övriga kapaciteter. Det är A2 fullt ut.
- Synkhjälparen och gestationscachen avvecklas när sista fältet bytt ägare.

**Klart när:** inget fauna-tillstånd har två skrivare, och inget kapacitetsfält saknar läsare.

## Steg 7 — Hydro

- `elevation` får en terränggenerator: lutande plan, bassänger, höjdryggar.
- `hydro_pass()` över fri yta `elevation + water` med grannflöde, tvåstegsmetod, strikt kontinuitet. Härledda fält som del av samma passkontrakt.
- `flood_tolerance` och `buoyancy` får läsare i locomotion och rörelsekostnad. Passiv drift i hydro-passet.

Att `nutrient` sedan Steg 4 transporteras med samma mönster gör hydro till en variant snarare än en nyhet.

## Steg 8 — Acceleration

Profilera fasmodellen under blandad realistisk belastning. Numba för sensing, CuPy för fältpass om de dominerar. Rustkärna endast om mätvärden motiverar det.

---

# Del E — Vad som ska mätas

| Steg | Mätpunkt | Målvärde |
|---|---|---|
| 0 | Invariantsvit över 10 000 tick | 0 brott |
| 0 | Fält med två skrivare | 0 |
| 1 | Referenser till rad, kolumn eller `[y, x]` utanför `Grid` | 0 |
| 2 | Filer ändrade vid hexbytet | endast `grid.py` och viewer |
| 3 | Världspass vid 1 000 000 celler | ~10 ms/tick, från 26 |
| 3 | Nollskilda värden i inaktiva celler i glesa fält | 0 |
| 4 | Floran når ett stationärt antal | ja |
| 4 | Stationärt antal vid dubblad `capacity` | oförändrat ±10 % |
| 4 | Överlevnadsskillnad mellan hög och låg `uptake_capacity` | statistiskt skild |
| 4 | Näringsbalans i ledgern | hård invariant vid 1e-6; uppmätt 1e-7 vid 8 000 tick, växer linjärt |
| 4 | Populationsdifferentiering mellan klimatband | mätbar, annars är världen för smal |
| 5 | Kostnad per floraindivid och tick | < 1 µs |
| 5 | 10 000 flora | < 10 ms/tick |
| 5 | `n_cells`-beroende termer i spatialindexet | 0 |
| 6a | Sensing-kostnad för organism med `sense_radius → 0` | 0, och koden aldrig berörd |
| 6a | Spridning i `sense_radius` med kostnad kontra utan | differentierar mot nisch kontra neutral drift |
| 6b | Kapacitetsfält utan läsare | 0 |
| 6b | Kostnad per fauna och tick | mät baseline, tillåt ej regression |
| 7 | Massbevarande i hydro över 10 000 tick | drift < 1e-6 relativt |

Att floran över huvud taget når ett stationärt tillstånd, och att det tillståndet är okänsligt för `capacity`, är den viktigaste enskilda mätningen i planen. Den avgör om ekologin har tagit över från arkitekturen.

**Baseline:** 2,50 ms/tick vid 22 fauna och 149 flora, `size=64`, seed 1, 12 000 tick.

---

## Överlämning — läget i Steg 4

*Skrivet för att tråden ska vara utbytbar. Uppdatera när något av nedanstående ändras.*

### Preliminära värden som behöver kalibreras tillsammans

`WorldParams.nutrient_input = 2.0e-10` sattes efter mätning att det ursprungliga värdet var fyra tiopotenser för generöst. Floran är nu näringsbegränsad — massan är i praktiken platt över 24 000 tick — men näringen ackumulerar fortfarande långsamt, så tillförseln är något högre än omsättningen.

**Floratätheten är för låg för att konkurrensen ska bita.** Uppmätt: 596 individer över 575 celler, som mest 4 i samma cell. De flesta frön landar i tomma celler. Innan differentieringsmätningarna säger något måste floran bli tät nog att faktiskt trängas, och det kräver att `nutrient_input`, spridningstakten och `flora_mortality` justeras i samma svep.

`PopParams.flora_seedling_mort_mult = 20.0` är satt på känsla och inte mätt.

**Titta på `shared_cell_frac` först.** Det är andelen floraindivider som delar cell med någon annan, och den enda siffra som säger om konkurrensen alls biter. Är den nära noll mäter differentieringssiffrorna ingenting, hur länge körningen än pågår.

### Kända gap med känd lösning

**Faunaläckan är stängd.** Kvar står kalibreringen, som den blockerade.

Läckan var två storleksordningar värre än den först beskrivna kadaverfaktorn. Uppmätt före stängningen, 6 000 tick och tolv agenter: assimilerad massa 8,2e-05 kg mot ett netto på +2,175 kg i `Body.step()`. **Kvoten skapad mot assimilerad massa var 26 500×**, och faunan bar 0,182 kg näring mot 1,10e-03 kg i hela resten av systemet. Kadavrets 1,6 gånger högre näringsinnehåll per kilo var en riktig men underordnad term.

Den dominerande mekanismen var att materialet aldrig bokfördes. `growth_E_per_kg = 10 000 J/kg` är ett medvetet val och inget dimensionsfel — byggkostnaden är syntesarbetet, inte den lagrade energin, precis som kommentaren vid `gestation_E_per_kg` säger. Men utan ett materialkrav byggde tillväxten vävnad ur enbart det arbetet, och `growth_rate_per_s = 0,008` mot `eat_rate = 1e-4` lät massan skapas åttio gånger snabbare än den maximalt kunde ätas.

Vad som gjordes:

- **Assimilationen på ett ställe.** `phenotype.assimilated_fraction(struktur, dietverkningsgrad)` är den enda definitionen, och den verkar på massan: `(1 − struktur) × matsmältning × diet`. `_perform_feeding()` avgör upptaget och exkreterar resten; `Body.step()` får den upptagna massan färdig. Dietverkningsgraden är därmed inte längre en separat multiplikator på energin — det var hålet som lät 38 % av en generalists föda bli varken kropp eller exkrement. Energimängden är oförändrad, exkretionen ökar.
- **Materialet är 1:1.** Tillväxt och gestation drar ett kilo reservmassa per byggt kilo vävnad, och syntesarbetet ovanpå. De två är termer, inte alternativ. Byggkostnadens kalibrering är därmed orörd; det är materialkravet som tillkommit.
- **Förbränningen utsöndrar.** `take_energy()` bokför den brända massans näring; kolet lämnar modellen som koldioxid. `take_energy(..., burn=False)` används när massan i stället överförs till en avkomma. Vid syntes utsöndras skillnaden mellan reservens labila näringsinnehåll och vävnadens, eftersom strukturmaterial binder mindre per kilo.
- **Katabolismen följer strukturen.** `Body._catabolize()` mobiliserar bara den labila fraktionen och exkreterar resten. Vid `s = 0,25` ger det samma energiutbyte som konstanten `E_body_J_per_kg`, vilket inte är en slump: 7,0e6 ≈ 9,302e6 · (1 − 0,25). Utbytet följer nu individens egen sammansättning.
- **Kadavret.** `M + reserv + eventuellt foster`, med massviktad strukturandel. Divisionen med `(1 − struktur)` är borta — det var den som skalade upp reserven till kadaverekvivalenter.
- **`Body` bokför, passet placerar.** Kroppen har ingen världsreferens och har inte fått en. Den ackumulerar `out_excreta_kg`, `out_excreta_struct_kg` och `out_nutrient_kg`, som `_flush_body_outputs()` tömmer till rätt cell efter body-passet, vid död och efter födsel.

**Uppmätt efter stängningen.** Näringsbalansen sluter sig till 1,4e-08 till 2,3e-07 relativt över 8 000 tick i seed 1, 2 och 3; energiledgern har 0 av 72 000 steg utanför tolerans med `max_rel` 6,4e-16. Faunan går från tolv till nio individer och stabiliseras där, floran stiger från 128 till omkring 150. `python run_headless.py --ticks 12000 --check-every 500` kör grönt.

**Kvarvarande gap i just den här delen:**

- Toleransen i `check_nutrient_balance()` är 1e-6 relativt, inte planens 1e-9. `detritus` och `detritus_structure` flyttades till float64 i samma patch, men **det var inte golvet** — driften låg kvar på ~1e-7. Bisektion per pass lokaliserar restposten till `_growth_system_flora` (systematiskt +1,1e-11 per tick) och `_dispersal_system_flora` (enstaka händelser kring 8e-9). Ackumulationen är linjär i tid: ~2,4e-12 absolut per tick, alltså omkring 1e-6 relativt vid 10⁵ tick. **Det betyder att långa körningar kan slå i toleransen.** Rätt plats att åtgärda är Steg 5, där båda passen ändå skrivs om som vektoriserade arraypass och aritmetiken byter form. Ett försök att korrigera float32-avrundningen i skrivningen till store:n gjorde driften konsekvent sämre och är återställt.
- Överskott över `E_cap` exkreteras i stället för att begränsa intaget vid källan. Biologiskt sämre, men födosöket är faunans mest kalibrerade beteende och ska inte röras i samma svep. Flyttas när kalibreringen är stabil.
- `anabolism_eff = 0,70` är fortsatt död kod. Materialutbytet vid syntes är 1,0. Att aktivera den nu lägger en broms till på en fauna som redan har smal marginal; den kan införas när svältmarginalen är mätt.

### Vad kalibreringen ska sikta på

Efter stängningen stannar faunan på nio individer i seed 1. Det talet är **inte** ett flödesjämviktstal. Uppmätt över 12 000 tick: tolv unika individer totalt, alltså enbart den warm-startade kohorten, tre dödsfall och **noll födslar**. De nio är samma nio.

Att öka födobasen är därför inte den första åtgärden. Fysiologin är inte grinden: `reproduktion.py` visar att **67 % av alla agenttick är reproduktionsklara**. Av dem faller 91 % på att agenten inte har någon giltig sensingträff alls — den ser ingen — och ytterligare 3 % på att den den ser ligger utanför `mating_radius`. Det är mötesfrekvensen som binder, inte energin.

Bekräftat genom täthet, seed 1:

```
värld      agenter   täthet/1000 celler   ser ingen   utanför radie   parningar
64×256          12                 0,67       91,2 %           3,3 %      2/4000 tick
64×64           12                 1,95       90,8 %           8,8 %      4/4000 tick
64×64           60                 9,03       57,5 %          16,0 %     25/2000 tick
```

Parningsfrekvensen stiger ungefär tjugofemfaldigt med fjortonfaldig täthet, och andelen som inte ser någon faller från 91 till 58 %. Vid den högsta tätheten binder däremot födan i stället — faunan går från 60 till 37 på 2 000 tick.

Det ger ordningen: **hitta först den täthet där mötesfrekvensen räcker, och kalibrera födobasen mot den tätheten.** Att skruva på `nutrient_input` och floras produktivitet vid 0,67 agenter per 1000 celler mäter något som ändå inte reproducerar sig. Kandidatreglagen är `init_pop` mot världsstorlek, `mating_radius` och `sense_radius` — och den sista hör till Steg 6a, vilket är ett argument för att ta 6a före den ekologiska finjusteringen.

Anmärkning: mötesproblemet är inte skapat av näringsstängningen. Före den fanns också bara enstaka födslar; skillnaden är att faunan då bar sig själv på manufakturerad massa och därför inte behövde reproducera sig för att synas som stabil.

**Verktyg:** `measure_leak.py` (faunans massaflöden, energiledgern, näringsbalansens drift), `reproduktion.py` (var reproduktionen fastnar, per utfall), `omsattning.py` (unika individer, dödsfall, ålder vid död — skiljer flödesjämvikt från stillastående kohort).

**Kroppen har tre fraktioner, inte två.** `structure` beskriver *sammansättningen* och avgör näringsinnehåll och kadavrets egenskaper. `E_cap` beskriver hur mycket som är *fritt mobiliserbart*, och tillåter en reserv på 3,2 % av kroppsmassan. Kvoten mot den labila fraktionen på 75 % är 23 gånger, och båda är riktiga: strukturell vävnad mobiliseras aldrig, funktionell vävnad bara vid svält, reserven fritt. Den nuvarande tvåstegskatabolismen — först energilagren, sedan kroppsmassa ner till `M_min` — motsvarar reserv först och funktionell vävnad sedan, och är alltså biologiskt riktig snarare än godtycklig.

**Strukturratchen.** Exkretion driver detritusens strukturandel uppåt, eftersom det labila tas ut och det sega passerar. Riktningen är korrekt — gammalt material *är* ligninrikt — men den dämpas bara av färsk förna från floramortaliteten. Håll ett öga på `detritus_structure` när mortaliteten kalibreras.

**Ljuskonkurrens.** `structure` saknar sin främsta verkliga fördel: höjd. Designen är utredd och nedskriven i `docs/substratets-struktur.md` med referens, men medvetet inte byggd — två begränsande resurser samtidigt gör kalibreringen oattribuerbar. Bygg den först om näringskonkurrensen ensam visar sig inte differentiera `structure`.

**Diffusionen är Steg 8:s tydligaste mål.** 0,77 ms vid 16 384 celler, 61 ms vid en miljon. `nutrient` är tätt dynamiskt och har inget glest stöd, så kostnaden är oundviklig utan acceleration.

### Beslut fattade under vägen som inte syns i koden

- `digestibility` utgår ur manifestets floraloci. Ingen organism vinner på att vara lättare att äta, så axeln kollapsar. Ersatt av `structure`.
- Kategorin växt kontra kadaver finns kvar i *anskaffningen* men inte i energiomvandlingen. Skillnaden i energitäthet faller ut ur strukturandelen.
- Kriteriet för Steg 4 är näringsbalans, inte massbalans. Total massa kan aldrig sluta sig: flora bygger ur luft, fauna växer ur en energibudget.
- Varje ny trait ska ha motverkande konsekvenser. En trait med en enda konsekvens är ett reglage, inte en anpassning.
- `uptake_capacity` har eget locus i stället för att härledas ur autotrofin. Autotrofin initieras högt för att göra flora till flora, vilket lämnade upptaget klustrat i övre änden med 18 % av sitt intervall utnyttjat. Att vara växt och att vara effektiv på upptag är olika egenskaper. Genomet är därmed 34 loci mot manifestets avsedda 8–16 — avvikelsen är medveten och bör antingen accepteras skriftligt eller åtgärdas genom att skära någon annanstans.

---

## Om ordningen ändå ska vara en annan

Vill man hålla fast vid hydro tidigt är det försvarbart — men gör Steg 4:s floradödlighet och konkurrens ändå först, som ett minimalt ingrepp. Utan dem mäter man hydro i en värld där floran växer obehindrat.

Vill man skjuta hex till efter ekologin är också det försvarbart, men räkna då med att kalibrera diffusion, spridning och celltäthet två gånger.

Det som däremot inte bör flyttas är Steg 1. Cellindexerade fält är den enda ändringen i planen som blir strikt dyrare för varje pass som byggs innan den är gjord.
