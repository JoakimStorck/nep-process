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

## Steg 4b — Växternas livscykel

*Underlag: `docs/vaxternas-livscykel.md`. Ersätter den kvarvarande kalibreringspunkten i Steg 4: hela tillväxtdynamiken byter karaktär här, så att finjustera den gamla vore att kalibrera fel sak.*

Mätningen inför steget visade att flera mekanismer vi trodde var verksamma inte är det. `uptake_capacity` binder aldrig — noll av alla individer, fem tiopotenser från taket — och fördelningen inne i cellen sker mot tillväxtunderskottet i stället för mot upptagsförmågan. Flera flora per cell tilläts i 0032 men inträffar i 1,2 procent av cellerna, eftersom den första plantan tar hela flödet oavsett storlek. Den fria näringen ligger till 99,9 procent i obebodda celler medan bebodda är rentvättade till maskinprecision. Floran är alltså inte mätt utan svälter: 1,4 procent når vuxenmassa, medianplantan står på 26 procent. Och näringsekonomin är icke förnybar — `nutrient_init = 0`, tillförseln fem tusendels procent av stocken, förlusten tio procent per varv, vilket tömmer världen på 37 år simulerad tid.

- ~~**Rotarea som anspråk.**~~ **Klart.** `A = m / B_K` ur aktuell massa, härledd per tick, delning `A_i / max(1, ΣA)` med spill över grannringen för `A > 1`. Fullvuxna plantor ingår i anspråket men tar ingenting, vilket är det som låter ett bestånd hålla undan groddplantor.
- ~~**Näringsekonomin.**~~ **Klart.** Bördigheten ligger i `nutrient_init`, sådden debiteras marken, tillförsel och förlust är kalibrerade som par, `uptake_rate_max` är per areaenhet.
- ~~**Inkomst skild från allokering.**~~ **Klart.** `flora_reserve` och `flora_repro_pool` i kg näring, float64 av bokföringsskäl. Allokeringsandelen tar över `_T_GROWTH`, som blev ledigt när tillväxten blev inkomstbegränsad.
- ~~**Omsättning.**~~ **Klart.** Förnafall per tick i en takt som avtar med strukturandel, deponerat vektoriserat via `World.excrete_cells()`.
- ~~**Härledd livslängd**~~ **Klart** ur strukturandel, 7 till 138 månader. Svältdöden är emergent: den som inte växer snabbare än sitt förnafall krymper under `flora_min_mass_frac · B_K`. `flora_mortality` och `flora_seedling_mort_mult` utgår.
- ~~**Fröet.**~~ **Klart.** Propagulmassa som eget locus `_T_SEED_MASS` i absoluta tal, log-skalad över fyra tiopotenser; antalet ur poolen dividerad med den; etablering som Hillfunktion med exponent två på förrådet, halvmättnad växande med målcellens anspråkade area; apparatandel på det frigjorda `_T_DISPERSAL`; spridning ur en stretchad exponentialkärna i kontinuerligt rum via `grid.cell_of_many()`. Etableringsutfallet dras innan slots allokeras och förlorade frön blir förna i sin målcell.

**Efter 0056:** fröaxeln differentierar och rör sig. Medianen går från 6,6 till 120 gram över 4 000 tick medan p10 stiger från 0,33 till 37 — fördelningen samlas kring etableringsoptimum, alltså precis den inre optimum Smith–Fretwell förutsäger. Det är den första trait i modellen som bevisligen selekteras mot ett *inre* värde i stället för mot en ände.

Tre saker återstår att döma, och de kräver längre körningar än sandlådan bär:

- **Apparatandelen driftar neutralt.** p10/median/p90 ligger på 0,095/0,246/0,417 mot initieringens 0,07/0,247/0,429. Antingen tar avståndsvinsten och etableringsförlusten ut varandra, eller är syskonkonkurrensen för svag för att avståndet ska betala. Axeln är alltså införd men ännu inte verksam.
- **`structure` fortsätter mot taket**, 0,567 → 0,763 med p90 på 0,789. Oförändrat sedan 0055 och väntat: ljuset saknas.
- **Faunan går ned till ensiffrigt** mot 256 efter 0055. Floran är nu ett bestånd av många små plantor — 63 154 individer på 62 221 kg, alltså ett kilo i medel — och en betare som tar 1,8 kg per tick raderar hela individer. Systemet är dessutom inte i vila vid tick 4 000: floran växer fortfarande brant. Det behöver 40 000 tick att döma.
- ~~**Tillväxten begränsas av ett `min()` över resurser**~~ **Klart**, med näring som enda post, så att ljus kan läggas till utan att passet skrivs om. Temperaturgrinden sitter numera på upptaget i stället för på en logistisk term som ändå aldrig band.

**Efter 0055:** floran samexisterar på riktigt — 4,06 plantor per bebodd cell i medel, median 3, mest 49, och 79,8 procent av de bebodda cellerna har fler än en. Allokeringsaxeln bär en verklig fördelning (p10 0,30, median 0,48, p90 0,67) i stället för att kollapsa. Faunan går inte längre under: den når `max_pop` på 256 och stannar där, mot en topp på 135 följd av utdöende före steget.

**Men `structure` har kollapsat mot den vedartade änden** — median 0,771 av taket 0,85, p10 till p90 bara 0,690 till 0,789. Det var förutsett i `docs/vaxternas-livscykel.md`, men 0055 gjorde det värre snarare än bättre: axeln fick två *nya* uppsidor, lång livslängd och lågt förnafall, medan den snabba änden fortfarande saknar sin — hög tillväxt är ingen fördel när tillväxten är inkomstbegränsad och inkomsten följer arean, inte vävnadstypen. Seg vävnad är dessutom sex gånger billigare i näring per kilo. Strukturaxeln har alltså fyra uppsidor och noll nedsidor, och kommer att ligga i taket tills ljuset finns. **Den ska inte läsas som ett resultat.**

**Efter 0054:** floran når stationärt beteende satt av näring i stället för av `capacity`, och systemet svänger i stället för att växa monotont. Fauna toppar på 135 individer mot tidigare 29 och överlever till tick 11 000 mot tidigare 5 000, men kollapsar därefter och tar floran med sig ner. Det är boom–bust utan reglering: 0054 höjer produktionen men inför ingen återkoppling. Regleringen är 0055:s uppgift — mortalitet, omsättning och allokering — och faunans egen reglering är en senare fråga.

**Beslut som är låsta:** `M₁ = B_K = 11`, alltså elva gånger dagens växtlighet; area ur aktuell massa; livslängd härledd, inte eget locus; ljus uppskjutet.

**Klart när:** `uptake_capacity` binder mätbart, näring i bebodda celler skalar med obesatt area, floras stationära antal är oförändrat vid dubblad `capacity`, näringsstocken saknar monoton trend över 20 000 tick, och kostnaden per floraindivid ligger kvar under 1 µs.

**Döms inte här:** spridningen i vuxenmassa. Utan ljus är näringskonkurrensen symmetrisk, alla i en cell växer med samma relativa takt, och byggkostnaden i näring gynnar dessutom seg vävnad sexfaldigt. Storleksaxeln mäts men får ingen dom förrän ljuset finns.

## Steg 4c — Instrumentering av växtcykeln

*Litet, men förutsättningen för att 4b ska gå att döma. Tre av dess utfall kunde bara mätas med engångsprobar.*

- ~~Strukturandel, täthet och fröaxel i world-loggen.~~ **Klart** — `flora_mean_structure`, `flora_mean_seed_mass`, `flora_mean_apparatus`, `flora_per_cell`, `flora_cells_occupied`, `flora_reserve_total`, `flora_pool_total`.
- ~~Flöden och dödsorsaker.~~ **Klart** — `flora_shed`, `flora_died_age`, `flora_died_starve`, `flora_seeds`. Förnafallet syntes tidigare bara indirekt som att detritus växte, och dödsorsakerna inte alls.
- ~~Fjärde panelen i `live_world_plot.py`.~~ **Klart** — täthet, dödsorsak och förnafall. Panel tre visar nu allokering, frömassa och strukturandel med rätta etiketter; den sa fortfarande "tillväxttakt" och "spridningstakt" om loci som bytt jobb.
- ~~Världsparametrar på kommandoraden.~~ **Klart** — `--nutrient-init`, `--nutrient-input`, `--nutrient-loss-frac`, `--uptake-rate`. Bördigheten är ett härlett tal, men härledningen går inte att pröva utan att kunna variera den.

**Bördighetstestet** är nu körbart och är den viktigaste enskilda mätningen i planen: `--nutrient-init 0.64` mot standardens 0,32 ska ge omkring dubbel stående biomassa medan dynamikens *form* är oförändrad. Håller det är taket ekologiskt; håller det inte sitter det någon annanstans.

## Steg 5a — Faunans åldrande

*Första steget i faunaarbetet. Mekanismen fanns; talet var kvar från sekundskalan.*

- ~~Slitagets åldersterm kalibreras.~~ **Klart.** `wear_a0` 0,008 → 0,12.

Uppmätt före: medianskadan i en matt population var **exakt noll** vid varje mätpunkt, individer nådde 157 månader, och **97 procent av alla dödsfall var svält**. Ett djur med `repair_capacity = 0,80` dog aldrig av ålder över 600 månader.

Orsaken var inte att åldersklockan saknades — `dD_age` tickar för alla — utan att slitaget byggdes ur *skadetakten* i stället för ur tiden. Ett djur som aldrig skadas slits aldrig, och ett som aldrig slits reparerar för evigt. Inflödet fanns, ackumulationen inte.

Uppmätt efter: `W` går från 0,26 till 2,6–4,1, maximal ålder från 157 till 58–89 månader, och **andelen dödsfall av skada går från 2,9 till 69,7 procent**. Svälten är kvar som andra orsak, vilket är rätt — den ska finnas, men inte vara den enda.

Och `repair_capacity` slutade selekteras nedåt. Den föll 0,554 → 0,312 i den gamla körningen därför att den kostade energi och köpte ingenting när skadan ändå var noll; nu ligger den stilla på 0,554. Axeln köper uppskjuten död, och avvägningen mot reproduktion är den klassiska.

**Svälten är äkta.** Katabolism mot floramassa korrelerar −0,20, och under de månader floran låg över 50 000 kg inträffade katabolism i 1,7 procent av fallen — noll under den långa återhämtningsfasen. Aptit- och katabolismlogiken tippar alltså inte för lätt; djuren svälte för att maten faktiskt tog slut.

## Statusanalys efter växtcykeln

`docs/statusanalys-vaxtcykeln.md` sammanfattar 0053–0067 och de fem 40 000-tickskörningarna: vad som fungerar, vilka förutsägelser som föll, vilka avsnitt i övriga dokument som är motsagda, och vad som återstår i vilken ordning.

Del D i den — de metodiska lärdomarna — är det som är värt att behålla längst. Särskilt att ingenting bedöms före mättnad, och att summor över florans sneda fördelningar är vilseledande.

## Steg 4l — Grindredovisning och fördelningar

*Instrumentering. Svarade direkt på frågan den byggdes för, och svaret var att jag hade fel.*

- ~~Varje reproduktionsgrind räknas för sig.~~ **Klart.** `gate_alive`, `gate_size`, `gate_nutrient`, `gate_carbon`, `gate_all`.
- ~~Kvartiler i stället för summor.~~ **Klart.** `flora_pool_p25/median/p75`, `flora_carbon_median`, `flora_mass_median/p90`.

Uppmätt vid tick 2 500:

```
levande            20 033
passerar storlek    3 322   16,6 %
passerar näring     7 288   36,4 %
passerar kol       14 857   74,2 %
passerar alla tre   1 233    6,2 %

pool p25/median/p75   0,00016 / 0,00085 / 0,00336 kg   (ett frö kostar 0,005)
massa median/p90       0,197 / 5,635 kg
```

**Ingen bugg.** Jag hävdade att fjortontusen mogna plantor med tillräcklig pool bara gav hundrasextio frön, och att det var tre tiopotenser fel. Talet byggde på **medelvärdet** av poolen, och fördelningen är kraftigt sned: medianen är en sjättedel av vad ett frö kostar, och medelvärdet åtta gånger medianen.

Beståndet domineras av småplantor — median 0,197 kg mot p90 på 5,6. De flesta individer är långt från varje grind, och den bindande är storleken.

Det är också en generell lärdom: **summor över en sned fördelning är vilseledande, och florans fördelningar är alltid sneda.** Alla `*_total`-fält i loggen bör läsas med den reservationen.

## Steg 4k — Mognadströskeln

*Den andra halvan av r/K-axeln, utlovad sedan 0062 och skjuten två gånger.*

- ~~`_T_MATURITY` som eget locus.~~ **Klart.** Reproduktion kräver `m >= mognadsandel · vuxenmassa`, utöver golvet `3 · propagulmassa`. Andelen spänner 0,01 till 0,50. Evolverbar och inte konstant: en hårdkodad tröskel kan sterilisera en population, vilket den gjorde i w1, medan ett locus inte kan det — en linje som inte reproducerar sig försvinner omedelbart.

Tidskostnaden kodas inte. Den faller ut ur tillväxthastighet och målstorlek: örten når sin lilla tröskel på månader, det vedartade sin stora på decennier.

**Utfall vid tick 6 000, mot samma tick utan tröskeln:**

```
                        utan       med
vuxenmassa kg             30        23
flora antal           41 998     7 902
biomassa kg           45 513    11 526
mognadsandel               —     0,120  (init 0,120)
andel mogna                —     0,124
```

**Vuxenmassan slutade stiga.** Utan tröskeln gick den 23 → 30 och fortsatte; med den ligger den kvar på 23. Det är den avsedda effekten: stor målstorlek kostar nu tid, och tiden är den valuta strukturbygge faktiskt är dyrt i.

**Men två saker oroar.** Beståndet drog ihop sig kraftigt, och bara tolv procent av plantorna är reproduktionsklara — sterilitetens signatur, mildare än w1 men samma form. Och **locuset rör sig inte**: 0,1207 vid start, 0,1195 efter sextusen tick, alltså exakt sin initieringsmedel. Ingen selektion syns.

Sannolik orsak: poolgrinden binder oftare än massagrinden. Frön ligger på 366 per månad, reproduktionen är näringssvält, och då spelar det liten roll var mognadströskeln står. Locuset skulle i så fall vara nästan neutralt av samma skäl som rot–skott var det i 0063 — det har ingenting att svara på.

Reglagen om det bekräftas är `MATURITY_MAX` och initieringsbandet i `genetics.py`. Men sextusen tick är före mättnad, och fyra gånger har en transient lästs som jämvikt. Ingen justering före en lång körning.

## Steg 4j — Ljusnivån och såddens fördelning

*Två tal, ingen ny mekanism. Båda registrerade som förutsägelser i förväg.*

- ~~`light_input` 0,60 → 1,5.~~ **Klart.** Vid 0,60 låg `flora_light_limited` på 0,886 vid mättnad: ljuset band för nio plantor av tio och kvävet knappt alls. Och ju knappare ljuset är, desto mer är höjd värd — vilket ger strukturandelen en uppsida som förstärker just det ljuset skulle motverka.
- ~~Sådden sprids över alla celler.~~ **Klart.** Massan per individ faller nu ut ur måltotalen och cellantalet i stället för tvärtom. Med fast massandel 0,4–1,0 av `B_K` blev plantorna 7,66 kg och rymdes i 4 172 celler av 16 384: tre fjärdedelar av världen fångade inget ljus, medan varje sådd planta hade bladarea 3,0 i en cell som mättas vid 1,0.

**Utfall vid tick 6 000, mot samma tick före patchen:**

```
                       före      efter
flora antal            4 745     41 998
biomassa kg            3 157     45 513
light_limited           0,65       0,43
root_alloc              0,537      0,641
structure               0,575      0,569
```

Tre saker som inte hänt förut. Relaxationssvackan är borta — biomassan faller inte under sitt startvärde. `flora_light_limited` kommer underifrån och lägger sig kring 0,43 i stället för att klättra mot 0,89. Och **rot–skott-axeln rör sig för första gången**, 0,50 → 0,64, alltså mot mer rot när ljuset blivit rikligare och kvävet relativt knappare. Det är den funktionella jämviktens respons, och axeln var död i alla tidigare körningar.

`structure` ligger stilla på 0,569, samma som vid start.

**Inget av detta är en dom.** Vid tick 6 000 är `per cell` 5,1 och stiger, biomassan stiger, fri näring sjunker. Mättnaden inträffade kring månad 600 i förra körningen. Fyra gånger har jag läst en transient som jämvikt; talen ovan är riktningar, inte jämviktsvärden.

**Attributionen är medvetet offrad.** De två ändringarna verkar dock i olika tid — sådden på transienten, ljusnivån på jämvikten — så de går delvis att skilja i efterhand ändå.

## Steg 4i — Rot mot skott

*Den axel som gör tvåresursbegränsning produktiv i stället för bara inskränkande.*

- ~~`_T_ROOT_ALLOC` som eget locus.~~ **Klart.** Andelen av tillväxten som går till rot. Kroppens sammansättning är integralen av besluten, lagrad i `flora_root_mass`, inte en andel som räknas om retroaktivt — den formen är vad plasticitet senare kräver.
- ~~Ytorna följer kroppens sammansättning.~~ **Klart.** `A_rot = SRA · rot · (1−s) / B_K`, `A_blad = SLA · skott · (1−s) / B_K`. Båda skalas med `(1−s)`: grov rot absorberar lika lite som ved fotosyntetiserar. Strukturandelen får därmed sin andra nedsida.
- ~~Höjden ur stammen.~~ **Klart.** `skott · s` i stället för `m · s`. En planta blir inte längre av att bygga rot — fel redan i 0061, syns först när facken skiljs åt.

`SRA = 6,7` och `SLA = 20` är satta så att inkomsterna vid ρ = 0,5 och s = 0,70 är desamma som före uppdelningen. Axeln startar neutralt.

**Utfall vid tick 8 000:** `structure` 0,5689 → 0,6242 → **0,5757**. För första gången springer axeln inte mot mutationens clip — den vänder och återvänder. Mot 0,6992 med 0062, 0,7582 med 0061 och 0,78 i båda långkörningarna. Och `flora_light_limited` ligger kring 0,42–0,46 större delen av körningen, alltså nära den funktionella jämviktens 0,5.

**Men produktionen föll kraftigt**, 6 440 kg mot 18 149 med 0062, med en djup svacka ned till 2 291 kg. Orsaken är inte svält och inte näring — dödsorsakerna är nästan uteslutande åldrande och fri näring ligger på 5 100 kg av 5 200. Den är att **förnafallet är kalibrerat för fel strukturandel**: `FLORA_TURNOVER_LABILE = 0,083` sattes när strukturandelen låg på 0,78, där omsättningen är 0,0247 per månad. Vid 0,57 är den 0,0406, alltså nästan dubbelt, och tillväxten hinner inte betala underhållet.

Det är mekanismen som fungerar och kalibreringen som inte hängt med: när strukturaxeln äntligen slutade springa mot taket flyttade jämvikten dit talen inte gäller. Nästa steg är att räkna om förnafallet och `light_input` mot den nya jämvikten, inte att backa mekanismen.

## Steg 4h — Tvåvalutareproduktion

*Första halvan av r/K-axeln. Andra halvan är mognadslocuset.*

- ~~Fröet kostar i båda valutorna.~~ **Klart.** Kolpool vid sidan av näringspoolen, matad ur ljusinkomsten med samma allokeringsandel. Ett frö kräver `seed_mass` kol och `seed_mass · nutrient_content(0,15)` näring.
- ~~Fröets näringskostnad räknas ur fröets egen sammansättning.~~ **Klart.** Den räknades tidigare ur moderns strukturandel, vilket gav en vedartad mor 4,5 gångers rabatt per frö — den snabba strategin betalade alltså mest i just den valuta reproduktionen drogs ur.

**Uppmätt vid tick 8 000:** `structure` 0,5696 → 0,6992, mot 0,7582 redan vid tick 6 000 med bara 0061. Första ändringen som bromsar axeln. Och `repro_alloc` ligger stilla på 0,433 mot 0,451 vid start, där den föll till 0,256 i w1 och 0,311 i w2 — reproduktionen lönar sig nu.

**Men kolvalutan binder sällan.** Kolpoolen står på 8 111 kg mot näringspoolens 44: reproduktionen är näringsbegränsad medan tillväxten är ljusbegränsad för 77 procent av plantorna. Det följer av att fröet är näringsrikt, vilket är biologiskt riktigt — fröproduktion är i verkligheten ofta kväve- eller fosforbegränsad — men det betyder att den bladrika plantans kolfördel bara svagt omsätts i frön. Det som faktiskt flyttade talen var att den vedartade rabatten försvann, inte att kolet började betala.

Kolpoolen har tak vid `flora_max_seeds_per_tick · seed_mass`; överskjutande kol går till kropp. Utan taket ackumulerade den obegränsat, vilket är samma fel som den låsta näringspoolen var före 0059.

En näringsläcka rättades under arbetet: fröregnets förna deponerades med moderns strukturandel medan poolen debiterades fröets, vilket förstörde 89 kg näring på 1 500 tick. Invariantsviten fångade den direkt.

## Steg 4g — Ljus som andra begränsande resurs

*Mekanismen är byggd. Kalibreringen är inte gjord, och strukturaxeln har ännu inte fått sin dom.*

- ~~Ljusinkomst per cell och asymmetrisk delning.~~ **Klart.** Bladarean följer `SLA · m · (1 − s) / B_K` — bara labil vävnad fotosyntetiserar. Höjden är `m · s` enligt `docs/substratets-struktur.md`. Skuggan är Beer–Lambert i cellens bladarealindex, dämpad av hur hög plantan står mot cellens arealviktade medelhöjd, vilket ger asymmetrin utan en sortering per cell — två `bincount` räcker.
- ~~`min()` över resurser blir verkligt.~~ **Klart.** Näring ur reserven, kol ur ljuset, Liebigs minimumlag. Formen låg färdig sedan 0054.
- ~~Mätpunkt för att ingen resurs är inert.~~ **Klart.** `flora_light_limited` är andelen växande plantor där ljuset är det knappare. Nära noll eller ett betyder att den ena resursen inte gör något.

**Kalibreringen tog två försök och är inte färdig.** Första ansatsen satte bladarean lika med rotarean; då täcker ett kilo blad en elftedels cell, en groddplanta växer långsammare än sitt eget förnafall, och världen dog ut — 11 741 plantor vid tick 1 000, 491 vid tick 8 000. Specifik bladarea är i verkligheten stor, och `leaf_area_per_kg = 10` löser det. Med den kommer `flora_light_limited` upp från 0,14 till 0,70 medan beståndet sluter sig, alltså ett verkligt tvåresursläge.

**Vad som ännu inte är avgjort:**

- `structure` stiger fortfarande, 0,5697 → 0,7582 vid tick 6 000. Men beståndet är då inte mättat — `flora_per_cell` är 10,8 och stiger — och lärdomen från 0059 är att täthetsberoende effekter har fel tecken före mättnad. Domen kräver 40 000 tick.
- Ljuset ger strukturandelen en **ny uppsida**: höjd undgår skugga. Nedsidan är mindre bladarea per kilo. Vilken som väger tyngre är en kalibreringsfråga i `light_extinction` och `light_height_ref`, inte en avgjord sak.
- Primärproduktionen sjönk — 31 691 kg vid tick 4 000 mot 81 083 utan ljus — och faunan dog tidigare. `light_input` är satt till 0,60 mot en uppmätt näringsbegränsad produktion på 0,46 kg per cell och månad, men bara en bråkdel av cellens ljus fångas upp av ett glest bestånd.
- Kolet lagras inte. Fotosyntat som inte används samma tick är borta. Reproduktionen betalas fortfarande enbart ur näringspoolen.

## Steg 4f — Counting sort i spatialindexet

*Enda posten i profilen som kunde accelereras utan att röra biologi. `rebuild_spatial_index` är oförändrad sedan Steg 5 och helt oberoende av modellen.*

- ~~`argsort` ersätts av en counting sort.~~ **Klart.** Cellindex är begränsade heltal, så grupperingen är O(n + n_cells) i stället för O(n log n): räkna, kumulera, strö ut. Elementen besöks i ursprunglig ordning, vilket gör den stabil, och permutationen är **bitidentisk** med `np.argsort(kind="stable")` — verifierat över fyra beståndsstorlekar.

```
plantor    counting sort   argsort stable    kvot
 47 000        0,311 ms        4,334 ms     13,9x
200 000        1,660 ms       21,236 ms     12,8x
350 000        2,452 ms       38,859 ms     15,8x
```

Vid en miljon celler och 350 000 plantor: 4,478 mot 38,765 ms, alltså 8,7 gånger. Kärnan återinför en O(n_cells)-term som Steg 5 tog bort ur indexet, men den kostar två linjära svep — 2,0 ms vid en miljon celler mot argsortens 39. Bufferten allokeras en gång och återanvänds.

Numba är valfri: saknas den faller koden tillbaka på `argsort`, med samma bana och sämre tid. `numba` står redan i `requirements.txt`.

**Uppmätt:** `rebuild_spatial_index` går från 5,55 till 1,98 ms vid 46 843 plantor, och ticken från 24,90 till 21,00 ms — femton procent vid den skalan. Vinsten växer superlinjärt med beståndet, eftersom argsorten gjorde det.

**Kvar i profilen** är nu tillväxtpasset, 12,2 ms vid 46 843 plantor. Det är bandbreddsbundet snarare än beräkningsbundet: passet materialiserar omkring tjugofem temporära arrayer, och vid 350 000 plantor är varje 2,8 MB. En elementvis operation kostar 0,71 ms och en gather 1,90 vid den skalan — kärnorna väntar på minnet. En fusionerad kernel som läser in en gång är sannolikt värd tre till fem gånger, men den fryser kod vi ändrar varannan patch och hör därför till Steg 8, efter ljuset.

## Steg 4e — Reproduktionsgrinden, trängseln och tillförseln

*Tre tal, ingen ny mekanism. Utlöst av att w1-körningen visade en steril flora.*

- ~~Reproduktionsgrinden blir absolut.~~ **Klart.** `m >= 0,20 · vuxenmassa` var skalfri och därmed samma fälla som den 70-procentströskel 0056 tog bort. I det hämmade beståndet låg medianplantan på en sjundedel av tröskeln: 160 frön per månad från 344 000 individer, och 944 kg näring — sjutton procent av världens stock — låst i pooler som aldrig kunde tömmas. Villkoret är nu `m >= 3 · propagulmassa`, absolut i båda leden.
- ~~Trängseltermen i etableringen görs exponentiell.~~ **Klart.** Den linjära formen med koefficient 1,0 var uppmätt verkningslös: vid ΣA = 1,44 etablerade sig ett frö med 0,163 kg förråd med 60 procents sannolikhet. Nu 3,2 procent. Formen är Beer–Lambert — ljuset som når marken avtar exponentiellt med arealindexet ovanför — vilket är samma mekanism ljussteget senare gör explicit.
- ~~`nutrient_input` räknas om.~~ **Klart.** 7,1e-5 → 4,6e-5. Mätt över 40 000 tick gav tillförseln 930 kg mot förlustens 600, en kvot på 1,55, och stocken växte med 331 kg. Härledningen antog 24 månaders levandeomsättning; den verkliga blev längre.

**Uppmätt mot w1-baslinjen, samma seed och tick:**

```
                 tick 8 000              tick 10 000
              baslinje    0059         baslinje    0059
flora          258 556  219 378         273 796  197 865
biomassa (kg)  245 529  263 360         243 716  241 800
medelmassa      0,95     1,20 kg         0,89     1,22 kg
fauna              11       14              23       28
```

Antalet vänder nedåt mellan tick 8 000 och 10 000, vilket baslinjen aldrig gjorde annat än genom faunakollapsen. Medelmassan stiger 37 procent vid oförändrad biomassa: samma näring bärs av färre och större plantor. Det var precis vad hämningen krävde.

**Kostnaden:** den lösare grinden gör att många fler plantor producerar frön, och med tre procents etablering i ett slutet bestånd genereras omkring trettio frön per etablering. Fröregn är verklig biologi, men det syns i profilen. Om posten dominerar kan antalet vinnare dras ur en binomial per moder i stället för per frö — det kräver att trängseln approximeras med moderns grannskap i stället för varje fröas målcell, alltså en approximation att motivera. Hör till Steg 8.

## Steg 4d — Prestanda i florapassen

*Utlöst av att en 40 000-tickskörning tog en timme vid 344 000 plantor. Profilerat, inte gissat.*

- ~~`_sigmoid` i ren python.~~ **Klart.** `np.clip` plus `np.exp` på en skalär kostade 4,42 µs mot 0,116 — trettioåtta gånger — och funktionen anropades tio gånger per floraetablering via `_init_flora_slot`. 422 510 anrop på 1 500 tick, 6,6 procent av ticken. Resultatet är bitidentiskt: båda vägar går till libm.
- ~~`World.excrete_cells()` aggregerar med `bincount` i stället för `np.unique`.~~ **Klart.** Unique sorterar; bincount gör samma jobb med ett svep. 19,9 mot 1,66 ms på 300 000 rader.
- ~~`_occupied_cells` likaså.~~ **Klart.** 13,3 mot 0,60 ms.
- ~~Dödsloopen deponerar i ett svep.~~ **Klart.** Ett anrop per döende planta gav tolvtusen anrop på 1 500 tick, var och en med egen aggregering över en enda rad.

**Uppmätt:** 27,13 → 23,71 ms per tick vid 3 000 tick, alltså 12,6 procent, och 25,56 → 21,36 vid 1 500. Banan är bitidentisk — samma antal, samma massor, samma detritus — så baslinjen är bevarad. Vinsten växer med beståndet, eftersom både unique-kostnaden och antalet etableringar gör det.

**Kvar, och nu de tre största posterna:**

- ~~`rebuild_spatial_index` och dess `argsort`.~~ **Klart i Steg 4f.** Counting sort i en Numba-kernel; permutationen är bitidentisk med `np.argsort(kind="stable")`, så banan är oförändrad.
- Att slopa det andra anropet per tick är nu värt betydligt mer än när frågan stängdes i Steg 5 — då kostade de två anropen 0,38 ms, nu 2,5 ms vid trettiotusen plantor och mer därefter. Det kräver dock att florapassen flyttas sist i ticken, vilket bryter manifestets passordning.
- `grid.cell_of` anropas 495 500 gånger på 1 500 tick, praktiskt taget allt från `see_agent_first_hit` som slår upp en cell per stråle och agent. Det försvinner i Steg 6a, där strålbaserad sampling ersätts av aggregering över `cells_within()`.

## Steg 5 — Aktiva delmängder och vektoriserade florapass

- ~~Delmängder byggs en gång per tick, immutabla under ticken.~~ **Klart** för `flora_slots` och `fauna_slots`. `sensing_slots` hör till Steg 6a, där `sense_rate` får en läsare.
- ~~Florapassen skrivs om som numpy-operationer över `flora_slots`.~~ **Klart.** Tillväxt, senescens, spridningens behörighet och tärningskast, samt floras summering. Sällsynta händelser — dödsfall kring en hundradel av populationen per tick, etableringar färre — hanteras individuellt efter den vektoriserade urvalet, eftersom var och en allokerar en slot eller blandar strukturandelar massviktat.
- ~~`rebuild_spatial_index` vektoriseras.~~ **Klart** med stabil `argsort` och gruppgränser ur den sorterade följden.
- ~~`n_cells`-termen tas bort ur indexet.~~ **Klart.** CSR är nu glest: `idx_cells` bär de bebodda cellerna i stigande ordning, `idx_starts` deras startpositioner, och `slots_in_cell()` slår upp med `searchsorted` i stället för direkt indexering. `flora_cell_mass` förblir tät, eftersom perceptionen läser den med fancy indexing över många celler samtidigt — men bara de celler som var satta nollställs.

  ```
  celler       tät form     gles form
   16 384      0,269 ms      0,185 ms
  1 048 576    5,906 ms      0,191 ms
  ```

  Indexet är därmed platt i världsstorlek, vilket var förutsättningen för miljoncellsvärlden i Steg 3. Vid en miljon celler går ticken från 110 till 93 ms enbart på den här ändringen.

- ~~`rebuild_spatial_index` anropas två gånger per tick.~~ **Stängd genom mätning, inte genom ändring.** Med det glesa indexet kostar de två anropen tillsammans 0,38 ms, alltså knappt 6 % av ticken vid dagens skala och under en halv procent vid en miljon celler. Att ta bort det andra anropet kräver att florapassen flyttas sist i ticken, vilket bryter manifestets passordning och byter fauna–flora-kopplingen mot en tickfördröjning. Det är en beteendeändring som kräver statistisk omvalidering för några få procent. Posten återupptas om profilen någon gång pekar tillbaka hit.

- **Flora-tilläggsfält i komprimerade tilläggsarrayer: medvetet uppskjuten.** Uppmätt kostnad för den okomprimerade formen: 20 B per slot för florafälten och 25 B för faunafälten, mot 263 B totalt — alltså 17 %, eller 9 MB vid 200 000 organismer. Den dominerande posten är `traits` på 128 B per slot, som är gemensam och inte går att komprimera bort. Komprimering kräver ett indirektionslager med egen frilista och allokeringsordning, alltså en verklig buggyta, för en budget som inte binder. A5:s syfte — att kärnan inte växer för att rymma subsystemstate — är uppfyllt: fälten *är* tilläggsarrayer med dokumenterat ägarskap, de är bara inte packade. Återupptas om minnet binder, eller om florapassen visar sig betala mätbart för gather i stället för slice.

**Klart när:** 10 000 flora körs under 10 ms/tick. **Uppmätt 10,1 ms** på referensmaskinen, alltså inom mätbrus från kriteriet.

Marginalkostnaden är uppfylld med god marginal: från 2 147 till omkring 10 000 floraindivider ökar ticken från 6,7 till 10,1 ms, alltså **0,44 µs per floraindivid** mot planens krav på under 1 µs. De resterande dryga 6 ms är fauna- och världsarbete som inte skalar med floran, och sensing är den största enskilda posten där. Den hör till Steg 6a.

**Uppmätt före och efter, seed 1, 64×256:**

```
flora        före        efter     kvot
 2 132     33,8 ms       6,7 ms     5,0x
10 026    ~115 ms *     10,7 ms    ~11x
```

\* extrapolerat från 11,5 µs per individ; den gamla formen kördes aldrig vid den skalan.

Dynamiken är statistiskt oförändrad. Vid 3 000 tick: flora 2 207 mot 2 214 före, floramassa 73,31 mot 73,48 kg, faunamassa 7,342 mot 7,362 kg. Identisk bana går inte att kräva — konsumtionsordningen har ändrats, se nedan.

**Näringen i en cell delas nu proportionellt.** Tidigare betjänade `take_nutrient` individerna i slotordning, så den med lägst slotindex hade första tjing på cellens pool. Det var en artefakt av loopen, inte en biologisk regel, och `docs/substratets-struktur.md` beskrev redan konkurrensen som symmetrisk. Nu får alla i cellen samma andel av sin efterfrågan, viktat med `uptake_capacity`. Det gör dokumentet sant och selektionen på `uptake_capacity` ren: en långsam planta kan inte längre vinna på att råka ha lågt slotindex. En marginal på en ulp i delningsfaktorn håller summan strikt under det som finns, så poolen aldrig kan bli negativ av avrundning.

Näringsdriften är oförändrat god efter omskrivningen: 7,7e-10 relativt vid 10 000 flora, 5,3e-09 vid 2 100. Den exakta avräkningen från 0040 — nedåtavrundning av lagrad massa och återföring av mellanskillnaden — är överförd till arrayform.

**Mätverktyg:** `run_headless.py --stats --flora-ratio N` rapporterar ms per tick och µs per floraindivid tillsammans med näringsdrift och invariantstatus; `--profile` ger hotspots.

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

### Massaskalan — designfel, rättat

Svepet över täthet och näringstillförsel hittade något annat än det letade efter. Uppmätt vid den gamla skalan: **konsumentbiomassan var 138 gånger primärproduktionen** — 6,68 kg fauna mot 0,048 kg flora, med flora på 0,86 % av cellerna. Förhållandet är omvänt mot varje verklig ekologi, och ingen kalibrering av täthet eller näringstillförsel rättar en inversion i den storleksordningen. Hundrafaldigad `nutrient_input` flyttade floran från 0,048 till 0,075 kg: floran var inte näringsbegränsad utan begränsad av sin egen massaskala.

`B_K` var 5e-4, vilket gjorde en vuxen planta till en halv gram mot faunans kilo. Den är nu 5e-2, och florans sådd är massastyrd i stället för antalsstyrd: `PopParams.flora_init_mass_ratio = 10.0` sår tills floran väger tio gånger faunans initiala massa. Antalet faller ut ur skalan i stället för att sättas separat, och utgångsläget förblir invariant när `B_K` eller `init_pop` ändras. `uptake_rate_max` uttrycks relativt `B_K` — ett absolut tak hade blivit hundrafalt hårdare bundet vid höjningen.

Utfall vid 64×256: 2 150 floraindivider, 73,5 kg flora mot 7,36 kg fauna, stabilt över 3 000 tick. **Dödsfallen går till noll** — svälttrycket är borta.

**Två latenta fel blev synliga först vid rätt skala**, båda i näringsbokföringen:

- `_growth_system_flora` drog näring *före* att massan kapades mot vuxenmassan. Det som trunkerades var redan taget ur cellen. Felet var proportionellt mot hur nära vuxenmassan individerna låg, och vid den gamla skalan kom de aldrig dit.
- `_dispersal_system_flora` flyttade massa från moder till frö utan att avräkna att fröet ärver *muterad* struktur. Ett kilo seg vävnad binder en bråkdel av vad ett kilo labil gör, så näring skapades eller förstördes vid varje spridning, proportionellt mot spridningsaktiviteten.

Dessutom avrundas florans massa nu alltid nedåt vid skrivningen till store:n. Rundar den uppåt binder massan mer näring än som betalats, och cellen är ofta redan tömd av just det upptaget så skulden inte går att driva in; nedåt är felet i stället alltid ett överskott som kan återföras.

Näringsdriften faller därmed från 2,7e-06 till 4,9e-09 relativt mätt per pass. Restposten ligger nu i fauna­passen och är omkring 2e-11 per tick.

**Kostnaden var prestanda.** 34 ms/tick vid 2 200 flora mot 5,5 ms vid 150. Det var Steg 5:s argument i konkret form, och Steg 5 har sedan tagit ner samma punkt till 6,7 ms.

### Tidsskalan omskalad till månader — se `docs/metabolismen.md`

Modellens tidsenhet var sekunder, vilket gav en livstidsomsättning på 2,4e-04 kroppsinnehåll mot verkliga 30–100. Ämnesomsättningen var komprimerad nio gånger och livslängden tvåhundratusen; följden var att det tog 568 livstider att bygga sin egen kropp, och att ett rovdjur hann konsumera en procent av vad det dödade.

Enheten är nu månader. `k_basal = 9,0e6` är Kleiber uttryckt i den, ett tick är 14,4 timmar, `year_len = 12` ger fyra vintrar per liv. Eftersom omsättningen beror på massan som M⁻⁰·²⁵ väljer tidsenheten också kroppsmasseskalan — månader ger djur på ett par kilo.

Utfallet över serien 0049–0052, seed 1:

```
                         median livslängd   max pop   födslar
utgångsläge                     0,7 mån        12        0
omskalning (0049)               0,3            12        0
biologiska invarianter (0050)   9,9            12        2
evolverbar reserv (0051)       13,2            12        3
kalibrering (0052)             61,6            29       32
```

**Populationen växer och reproducerar sig för första gången.** Den kollapsar fortfarande vid 101 månader, men beteendet ser ut som boom-bust: beståndet växer till 29, betar ner födobasen och faller. Det är ekologi utan reglering, inte en trasig mekanism. Nästa fråga är vad som ska dämpa svängningen, och där finns kandidater i modellen som ännu inte fått verka — framför allt att `sense_radius` och rörelse gör betningen rumsligt ojämn så att fläckar hinner återhämta sig. Det hör till Steg 6a.

`docs/metabolismen.md` beskriver hela metabolismen, de tre valutorna och deras olika lagar, samt de sex fällor som kostade mätbar tid under omskalningen.

Kvar på listan: den newtonska rörelseintegrationen är numeriskt brus vid femton timmars tick — relaxationen mot terminalhastighet är sekunder — och bör förenklas till kvasistatisk form.

### Näringens ekonomi — granskad, se `docs/naringens-ekonomi.md`

`flora_init_mass_ratio = 10.0` visade sig inte vara en ekologisk kvot utan systemets bärkraft. `nutrient_init = 0`, så den sådda biomassan **är** hela näringsbudgeten — extern tillförsel över en 400-sekunderskörning är 0,13 % av stocken. Fördubblad sådd ger fördubblad stående gröda, och kausaliteten går fel väg: primärproduktionen härleds ur faunans massa.

Den större felkalibreringen sitter dock i `nutrient_loss_frac = 0.10`, som säger att en näringsatom klarar tio varv genom levande vävnad innan den försvinner. I verkligheten är talet hundratals till tusentals. Uppmätt vid 12 fauna:

```
loss_frac = 0,100    tillfört 0,0118 kg/h   förlorat 0,1244 kg/h   netto -0,1126 kg/h
loss_frac = 0,002    tillfört 0,0118 kg/h   förlorat 0,0028 kg/h   netto +0,0090 kg/h
```

Vid 0,100 dräneras världen med tidskonstant kring elva simulerade timmar, vilket är precis horisonten för långkörningar. Vid 0,002 ackumulerar den.

Designskissen lägger fyra påståenden som var för sig går att motivera utan att fittas mot en önskad populationsstorlek: källan är en takt per cell och inte en stock, stocken sås och representerar prebiotisk ackumulation, den interna cirkulationen är nästan förlustfri, och sänkan är rumslig — begravning vid havsranden när hydron finns, inte en tiondels skatt på varje nedbrytning överallt.

Den visar också varför lokaliteten i upptag, exkretion och kadaverdeponering är bärande och inte bara en prestandaprincip: vatten flyttar näring strikt utför, och den biotiska kedjan är det enda som flyttar den uppför.

**Att göra i Steg 4:s kalibrering:**

- `nutrient_loss_frac` till storleksordningen 10⁻³.
- `nutrient_input` blir ett per-cell-fält i stället för en skalär, så punktkällor blir möjliga.
- Näringsbudgeten sätts explicit; flora sås ur budgeten och fauna ur floran, i stället för tvärtom.

### Födsloproblemet — orsakskedjan kartlagd

Uppmätt vid 100 agenter, 8 000 tick: 45 påbörjade gestationer, 0 födslar, 44 slutade med att bäraren dog. Samtliga dödsfall i körningen var dräktiga. De dog inte av svält i vanlig mening utan vid `D_max`: median D = 0,998 med massan på 0,858 kg, långt över `M_min` = 0,07, och reserven på noll.

Kedjan visade sig ligga tre led före gestationen.

**1. Djuren åt sopor.** `_perform_feeding` anropade `consume_food` med `prefer_detritus=True`, hårdkodat. Uppmätt gick 98,1 % av intaget till detritus vid härledd strukturandel 0,958, som assimileras till 1,2 %, medan levande flora gav 17,2 %. Total assimilation 1,5 %. Det var rimligt när detritus betydde kadaver, men sedan floran fick mortalitet och exkretionen koncentrerar strukturmaterial består poolen av nästan osmältbar förna — och slingan förstärker sig själv, eftersom det de äter blir det de exkreterar.

  **Åtgärdat:** födovalet maximerar värde i stället för att följa en fast preferens. Värdet är `assimilated_fraction(struktur, verkningsgrad) × E_labile`, alltså joule per ingesterat kilo för just det djuret och det substratet — talet fanns redan, det som saknades var att någon läste det före valet. Den bästa födan tas först och den näst bästa rörs bara med det som återstår. Assimilationen gick från 1,5 % till 15,8 % och överskottet från −1,2e-06 till +6,8e-06 kg/s.

  Preferensen behöver därmed inte lagras. `diet` är bara en avvägning mellan verkningsgrader, och beteendet härleds. Den styrde tidigare verkningsgraden men inte valet, så en herbivor med `herb_eff = 1,0` åt ändå detritus där hennes `scav_eff` var noll — traiten kunde straffa men aldrig löna sig. Nu är den en levande axel:

  ```
  diet-kvartil   andel detritus   assimilation
   0,03–0,27          0,1 %          24,1 %
   0,27–0,48          0,4 %          19,0 %
   0,48–0,74         16,6 %          13,9 %
   0,74–0,98         74,2 %           7,9 %
  ```

  **Men asätarnischen är inte livskraftig, och orsaken är poolningen.** En ren asätare får 0,087 ur detritus vid strukturandel 0,80 medan en ren betare får 0,258 ur flora vid 0,57. Verkliga asätare lever på kadaver, som är strukturfattiga och rika — men kadaver hälls i samma `detritus`-fält som förna och exkrement, och strukturandelen massviktas. Ett kadaver vid 0,25 som landar i en cell med ett kilo förna vid 0,83 blir 0,80. Asätaren ser aldrig ett rikt kadaver.

  Det är en representationsfråga, inte en substratfråga: kadaver och förna behöver hållas isär, antingen som skilda pooler eller genom att inte massvikta strukturen. Tas som eget beslut.

**2. Tillväxtdrivet går i futil cykel. Återstår.** Ledgern per djur och sekund: intag 100 J, obligatoriska dräneringar 33 J, men `E_material` 2 694 J och `E_from_M` 1 670 J. Tillväxten begär alltså tjugosju gånger mer material än födan ger och finansierar det med katabolism av den egna kroppen. Materialet återvänder som kroppsmassa, så nettomassan ändras knappt — men varje varv kostar 32 % i katabolismens utbyte och lägger på skada via `k_cat_dmg`.

  Orsaken är att `out_growth` behandlas som en obligatorisk dränering i `Body.step()`, så när reserven inte räcker täcker katabolismen även tillväxten. Tillväxten skapar då själv det underskott den finansieras ur. Rätt regel är att tillväxt bara får ta av det som återstår när de obligatoriska dräneringarna är betalda, och aldrig utlösa katabolism.

**3. Gestationstakten är fortfarande fyrtio gånger intaget.** `gestation_growth_kg_per_s = 0,004` mot `eat_rate = 1e-4`. Reservtaket på 3,2 % av kroppsmassan töms på sju sekunder, varefter modern kataboliserar sig själv. Takten kalibrerades när fostermassa kostade 10 kJ/kg och inget material.

  Den ska bli en evolverbar axel, men intervallet går inte att sätta förrän punkt 2 är åtgärdad — vilket överskott som helst äts upp av den futila cykeln. Storleken hjälper inte heller: andelen av fostret som kan ätas in under gestationen är 2,5 % oavsett `child_M`, eftersom det är takten och inte massan som är fel.

**Ordning:** tillväxtcykeln först, sedan mät om budgeten, och sätt gestationsintervallet mot ett verkligt överskott.

### Vad kalibreringen ska sikta på

Efter stängningen stannar faunan på nio individer i seed 1. Det talet är **inte** ett flödesjämviktstal. Uppmätt över 12 000 tick: tolv unika individer totalt, alltså enbart den warm-startade kohorten, tre dödsfall och **noll födslar**. De nio är samma nio.

Efter att massaskalan rättats är svälttrycket borta — noll dödsfall över 4 000 tick — men födslarna är fortfarande noll, och nu av en enda anledning. Fysiologin är inte grinden: `reproduktion.py` visar att **67 % av alla agenttick är reproduktionsklara**. Av dem faller 91 % på att agenten inte har någon giltig sensingträff alls — den ser ingen — och ytterligare 3 % på att den den ser ligger utanför `mating_radius`. Det är mötesfrekvensen som binder, inte energin.

Bekräftat genom täthet, seed 1:

```
värld      agenter   täthet/1000 celler   ser ingen   utanför radie   parningar
64×256          12                 0,67       91,2 %           3,3 %      2/4000 tick
64×64           12                 1,95       90,8 %           8,8 %      4/4000 tick
64×64           60                 9,03       57,5 %          16,0 %     25/2000 tick
```

Parningsfrekvensen stiger ungefär tjugofemfaldigt med fjortonfaldig täthet, och andelen som inte ser någon faller från 91 till 58 %. Vid den högsta tätheten binder däremot födan i stället — faunan går från 60 till 37 på 2 000 tick.

Det ger ordningen: **hitta först den täthet där mötesfrekvensen räcker, och kalibrera födobasen mot den tätheten.** Med `nutrient_loss_frac` korrigerad bär världen omkring trettio djur mot dagens tolv, och de hundra som mötesfrekvensen kräver ligger ungefär fyra till fem gånger dagens tillförsel bort — mot åttio gånger räknat med det gamla läckaget. Se `docs/naringens-ekonomi.md`. Att skruva på `nutrient_input` och floras produktivitet vid 0,67 agenter per 1000 celler mäter något som ändå inte reproducerar sig. Kandidatreglagen är `init_pop` mot världsstorlek, `mating_radius` och `sense_radius` — och den sista hör till Steg 6a, vilket är ett argument för att ta 6a före den ekologiska finjusteringen.

Med rätt massaskala står alltså Allee-effekten kvar ensam som blockerare: agenterna är mätta, reproduktionsklara och hittar inte varandra. Det gör `sense_radius` till nästa verkliga reglage, alltså Steg 6a.

Anmärkning: mötesproblemet är inte skapat av näringsstängningen. Före den fanns också bara enstaka födslar; skillnaden är att faunan då bar sig själv på manufakturerad massa och därför inte behövde reproducera sig för att synas som stabil.

**Följa en körning:** `run_headless.py --pop-log pop.jsonl --world-log world.jsonl` skriver samma loggar som `run_population.py`, utan pygame. `live_pop_plot.py` ritar faunans demografi och kroppstillstånd, `live_world_plot.py` floran och näringskretsloppet. Båda tar `--save FIL` för en stillbild utan GUI, vilket är vägen som fungerar över ssh. Vid långa körningar på fjärrmaskin kan loggen strömmas till den lokala maskinen och plottas där — plottrarna tail:ar filen och märker ingen skillnad.

**Verktyg:** `run_headless.py --stats` ger allt detta i en körning — bestånd, massakvot flora mot fauna, omsättning i unika individer, var reproduktionen fastnar uppdelat på ser-ingen mot utanför-radie, näringsbalansens termer och takt. `--seeds 1,2,3` för spridning mellan körningar, `--size` och `--init_pop` för täthetssvep, `--flora-ratio` för att skala primärproduktionen. `measure_leak.py` finns kvar för faunans massaflöden och energiledgern, som ligger utanför det `--stats` visar.

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
