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

~~Kadensklasserna ska in i manifestet innan hydro byggs.~~ **Klart i 7001** — README.md har nu ett eget avsnitt, och en femte klass tillkom när hydro fick sin form: **nätverksdynamisk**, ett fält som är tätt men riktat och rörs en gång per cell i en förberäknad ordning.

Glesning av hydro självt är däremot sannolikt onödig. Den motiverades av att ett tätt grannflöde skulle kosta för mycket, men jämviktslösningen i Steg 7 kostar 0,68 ms vid 262 144 celler — mindre än den täta laplacianen i `transport_pass`. Punkten flyttas till Steg 7:s sista patch och görs bara om en mätning kräver den.

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

## Steg 5g — Bildrutorna blir dödbara och läsbara

- ~~`SDL_NO_SIGNAL_HANDLERS=1`.~~ **Klart.** SDL installerar egna hanterare för SIGTERM och SIGINT. I dummy-läge finns inget fönster att stänga, signalen tas emot och leder ingenstans, och processen gick inte att döda med `kill` — bara med `kill -9`.
- ~~Bara video och font initieras.~~ **Klart.** `pygame.init()` startade även ljud och joystick, därav ALSA-varningen. Rättat både i `run_headless.py` och i `viewer_pygame.py`; inget i viewern använder mixer eller joystick, så interaktiva körningar påverkas inte.
- ~~`--snapshot-crop X Y W H`.~~ **Klart.** Utsnitt i celler, så att en liten yta kan ritas stort. Vid skala 8 och 48×48 celler blir bilden 384 px och enskilda celler går att skilja åt.

Motivet: den första bildserien från en 256×256-värld i skala 2 var oläsbar. Detritusfältet på 178 000 till 255 000 kg mättade färgskalan så att allt blev rött, och med fyra pixlar per cell fanns ingen upplösning kvar för den variation som betyder något — `B` inom räckhåll har p10 = 0 kg mot median 20,8.

Det gick ändå att läsa två saker ur den: agenterna klumpar sig inte, utan ligger jämnt utspridda, och de sista överlevarna ser friska ut snarare än utmärglade. Båda stämmer med mätningarna.

`--snapshot-mode FLORA` finns redan i viewern och ritar floran utan detritusfältet.

## Steg 5f — Bildrutor från en huvudlös körning

*Rumslig fördelning går inte att läsa ur tidsserier. Viewern fanns men krävde ett fönster.*

- ~~`--snapshot-every` i `run_headless.py`.~~ **Klart.** Plus `--snapshot-dir`, `--snapshot-scale`, `--snapshot-mode`.

Samma ritkod som den interaktiva viewern, sparad till `t0000200.png` och så vidare. `render_frame.py` kunde redan rendera *en* bildruta men körde sin egen simulering fram till den — det gick inte att följa en pågående körning.

Pygame importeras först när första bildrutan begärs, och `SDL_VIDEODRIVER=dummy` sätts strax innan. En körning utan flaggan rör aldrig biblioteket, så den huvudlösa vägen är oförändrad.

## Steg 5e — Perceptionsfältet visar skottmassa

*Rättar ett fel som 0071 införde och som ingen mätning fångade förrän dödsorsakerna loggades.*

- ~~`flora_cell_mass` summerar `mass − flora_root_mass`.~~ **Klart.**

`flora_cell_mass` är fältet strålarna samplar. Det summerade **hela** växtmassan, alltså även rötter. Sedan 0071 stannar betningen vid roten, så djuret navigerade mot en signal som vid den uppmätta rotandelen 0,55–0,70 överdrev måltiden **två till tre gånger**.

Det är inte en fråga om att sensorn ska bedöma ätbarhet. Rötterna är under jord: de kan varken ses eller ätas, och samma horisont definierar båda. Att fältet visade dem betydde att sensorn rapporterade något fysiskt osynligt.

Tidpunkten stämmer med symptomet. Före 0071 var hela plantan ätbar och fältet korrekt. Efter blev svält den dominerande dödsorsaken — 60,2 procent av 221 dödsfall i p71b — hos djur som dog med floran på 47 000 individer omkring sig.

**Principbeslut infört i `docs/synens-axlar.md`:** sensorn detekterar, organismen tolkar. Diskriminering mellan flora, kadaver och artfrände är i dag gratis och felfri — tre färdigsorterade kanaler, alltså en sensor som redan dömt. Den ska bli en kapacitet med kostnad och med brus som gör misstag möjliga. Arbetet prioriteras inte nu, men premissen är beslutad så att Steg 6a slipper ta om frågan.

**Ogjort:** de två navigationsmätningarna — styr agenterna mot det de ser, och hinner de fram. De bör göras efter rättningen; att mäta navigationsförmåga mot ett fält som ljuger om måltiden ger ett svar som inte går att tolka.

## Steg 5d — Dödsorsak i life-loggen

*Använder befintlig infrastruktur. `LifeLogger` och `death_record` fanns redan; de var bara aldrig exponerade.*

- ~~`death_cause` sätts vid varje dödsväg.~~ **Klart.** Fem vägar i `Body.step()`: `damage`, `starvation`, `hazard`, `guard_pre`, `guard_post`, `guard_energy`.
- ~~`--life-log` i `run_headless.py`.~~ **Klart.** Ogrindad; händelserna är få jämfört med tickarna.

Skälet: i p71 dog 86 djur på fyrtio månader utan svält — katabolismen var 0,16 procent av intaget — och utan att åldersdöden borde ha slagit. Vi loggade födslar och dödsfall men aldrig varför, och dödsorsaken har gissats fel tre gånger i det här arbetet.

Två vakter kan dessutom döda agenter vid numeriska avvikelser, och de syntes inte alls i någon logg.

**Första mätningen, 4 000 tick:** 11 `damage`, 3 `unknown`, 1 `hazard`, noll svält och noll vaktdödsfall.

Två saker att notera. `damage` inträffar vid medianålder 57 månader, alltså långt under åldersdödens kalibrerade 121–153 — skadan kommer från ansträngning, köld eller svältstress, inte från klockan. Och `unknown` betyder att det finns minst en dödsväg utanför `Body.step()` som inte sätter orsaken; den bör spåras.

## Steg 5c — Betningshorisonten

*Betaren tar skott och lämnar rot. Ingen ny parameter, inget nytt locus — bara en åtskillnad modellen redan hade.*

- ~~`take = min(m − rot, amt)` i stället för `min(m, amt)`.~~ **Klart.** Den funktionella responsen räknar också bara skottmassa som tillgänglig resurs.

**Mätningen som ledde hit.** Betaren tömmer inte ett grannskap — den lämnar det två till sex gånger snabbare än den hinner: 18 tick att passera mot 44 att beta ner ett mediangrannskap på 20,8 kg. Men den tar `min(m, amt)` per *planta*, och medianplantan väger 0,197 kg mot en tugga på 0,907. Varje tugga svepte därför upp flera hela småplantor.

Det syns i p70: floran föll från 37 324 till 5 396 individer under betningstoppen, alltså antalet i takt med massan. Det sker bara om hela växter försvinner.

Refugen som saknades var alltså inte att lämna plantor i cellen — det gjorde betaren redan — utan att **lämna en del av varje planta**.

**Utfall, 14 000 tick:**

```
tick    fauna   flora antal   flora kg
 2000      22        27 671     42 178
 4000      41        29 437     37 050
 6000      28        21 598     24 877
 8000       8        22 254     28 101
10000      13        52 819     71 061
12000      19        74 567    100 820
```

Faunan toppar på 41 mot p70:s 58 och 100, bottnar på 8 och **återhämtar sig**. Och floran tappar 27 procent av individerna mot p70:s 86 vid andra toppen — antalet håller sig medan massan svänger, vilket var hela avsikten: plantor betas ner men dör inte.

`root_alloc` får dessutom en ny konsekvens. Rot är betesskydd, så den som satsat på skott betalar när trycket kommer. Axeln var redan tvåsidig; nu är den det på tre sätt.

## Steg 5b — Betningens funktionella respons

*Härledd ur att betandet tar tid, inte påsatt som kurva.*

- ~~Hollings typ II i `_consume_flora_from_store`.~~ **Klart.** `intag = dt · a · B / (1 + a·h·B)`, där `B` är växtmassa inom räckhåll, `a` sökeffektivitet och `h` hanteringstid per kilo.

Tidigare: `take = min(m, amt)` utan söktid och utan hanteringstid. Den sista plantan i grannskapet var lika lätt att hitta och äta som den första, och intaget föll inte förrän den lokala massan understeg aptiten — då tvärt till noll. **Bytet hade ingen lågtäthetsrefug.**

Talen är härledda: `h = dt / 1,8` ger samma tak som den uppmätta maximala aptiten, och `a` sätter halvmättnaden vid 20 kg inom räckhåll — mellan mättnadslägets ~76 kg och kollapslägets ~6.

```
B inom räckhåll   intag/tick   andel av taket
        2 kg        0,164          9 %
        6 kg        0,416         23 %
       20 kg        0,901         50 %
       76 kg        1,426         79 %
      200 kg        1,638         91 %
```

**Inte testat i en lång körning.** Typ II är dessutom klassiskt *destabiliserande* i sig — den ger en refug men ingen tröskeleffekt. Typ III, som stabiliserar, kräver byte av föda eller en fysisk refug, och den senare hör till terrängen.

Kostnad: ett extra svep över slots i räckvidden per betningshändelse, omkring 8 procent på ticken.

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

## Steg 5h — Riktningen som tillstånd

*Numerisk rättning plus en beteendemekanism. Oberoende av allt annat, och först: varje beteendemätning dessförinnan mäter brus. Underlag: `docs/rorelsens-arkitektur.md`, Del 1.*

`turn_rate · dt = 6,0` rad per tick mot ett varv på 6,28, och styrningens förstärkning var 1,54 per tick. Riktningen dekorrelerade alltså på ett tick. Djuren rörde sig fort — uppmätt 37 cellbredder per månad — men kom ingenstans: rakheten över livet låg på 0,069, alltså 1 563 cellbredders bana för 85 cellbredders förflyttning.

Rätt svar är inte rak rörelse. Kringgående sök i ett område är rätt beteende när födan är riklig, och den kurviga banan håller organismen kvar på fläcken. Det som saknades var förmågan att välja regim.

- Headingen blir ett tillstånd med persistenstid. `τ_dir` interpolerar mellan kort söktid och lång färdtid, med `explore_drive` som regimval. Bruset skalar som `√dt` så att rotationsdiffusionen blir tidsstegsinvariant — den gamla formen skalade med `dt`.
- Styrningen blir analytisk relaxation med tak på vridhastigheten, ovillkorligt stabil vid godtyckligt `dt`.
- Vridhastighetens tak följer centripetalvillkoret `ω ≤ a_lat / v`. Fart köper räckvidd och kostar manöverförmåga.
- `_T_MOB` (locus 11, utan läsare) får äga färdregimens persistenstid.
- Life-loggen bär bansträcka, nettoförflyttning och deras kvot.

Fartintegrationen rättas **inte** här. Den är divergent för merparten av populationen — förstärkningsfaktorn `|1 − dt·c₁/M|` passerar ett vid 2,2 kg mot en medianmassa på 1,3–1,7 — men att rätta den höjer den realiserade farten och `effort = speed / v_max` matar skademodellen. Uppmätt när båda ändrades samtidigt: dödsorsakerna gick från 68 % svält till 63 % skada och seed 1 dog vid tick 6 000. Farten byter form i Steg 5i med sin omkalibrering.

**Klart när:** rakheten är mätbart högre än 0,069 och varierar med tillståndet, och den är oförändrad vid halverat `dt`.

## Steg 5i — Farten som kraftbalans

*Kräver Steg 5h. Underlag: `docs/rorelsens-arkitektur.md`, Del 1.*

Explicit Euler mot dragkraften har förstärkningsfaktorn `|1 − dt·c₁/M|`, som passerar ett vid 2,2 kg mot en uppmätt medianmassa på 1,3–1,7. Schemat var divergent för merparten av populationen och hölls ändligt bara av klampningen, så farten svängde mellan noll och taket i stället för att följa gaspådraget.

- Terminalfarten löses ur `F_prop = c₁v + c₂v²` i stället för att integreras. Relaxationstiden är 0,45 tick, alltså kortare än tidssteget — samma fälla och samma lösning som den termiska relaxationen.
- Skadeinflödets fem termer plus `effort`, `rest` och `speed_n` exporteras till pop-loggen. `effort` normeras mot `v_max = 100`, som är en klampningsgräns och ingen biologisk fart, och omnormeringen ska göras mot en mätning.

Uppmätt, 8 000 tick seed 1: dödsorsakerna går från 61 % skada och 25 % svält till 43 % och 45 %, alltså tillbaka mot baslinjens 58/32. Skadeinflödet fördelar sig `dD_met` 49 %, `dD_eff` 29 %, `dD_age` 15 %, `dD_starve` 6 %, `dD_cold` 1 % — rörelsen står för knappt en tredjedel.

**Klart när:** invariantsviten är godkänd och skadeinflödets fördelning är känd. Omnormeringen av `effort` och kopplingen mellan fart och rörelseregim ligger efter p78.

## Steg 6a — Sensing som evolverbar kapacitet

*Kräver Steg 4 för selektionstrycket och Steg 5 för delmängdsmaskineriet, men inte hela fauna-migreringen. Underlag: `docs/synens-axlar.md`.*

Prioriteringen inom steget är satt av rörelsemotorns behov: sektorformatet och observerbara kännetecken först, `sense_rate` och `sense_fov` sedan. Räckvidden ökas inte — det som krävs är aggregering inom befintlig radie.

Synen är i dag evolverbar i fyra diskreta steg längs en sammanslagen axel. Det som saknas är att axlarna kan handlas mot varandra och att kapaciteten kostar även när den inte används.

- `sense_radius` och `sense_rate` blir kontinuerliga genetiska axlar med läsare: radien avgör vilka celler som läses, frekvensen avgör vilka slots som ingår i sensing-delmängden.
- Vinkelupplösning och synfältsform blir egna axlar. Fix antal riktningssektorer, där akuiteten styr hur mycket de blandas — så att MLP:ns indimension är oberoende av genotypen och vikter förblir ärftliga.
- Strukturkostnad införs: att bära kapaciteten kostar även oanvänd. Aktiveringskostnaden faller ut ur geometrin, eftersom `cells_within(r)` växer som ~3r² på hex.
- Strålbaserad sampling ersätts av aggregering över `cells_within()` grupperad via grannmatrisen.

**Klart när:** en organism med `sense_radius → 0` aldrig berör sensing-koden, och spridningen i `sense_radius` differentierar mot nisch med kostnadsmodellen påslagen men driftar neutralt utan den.

### Mätningen inför steget

*64×256, seed 1, 160 tick efter uppvärmning, `--pass-timing`. Floran hålls på ~16 300 individer i alla tre körningarna: sådden skalar med faunans startmassa, så `--flora-ratio` kompenserar `--init_pop`. Utan den kompensationen mäter man sextonfaldigad flora och inte fler djur.*

```
                        12 djur    50 djur   200 djur      us/djur
totalt                   23,5 ms    40,5 ms    100,1 ms
_step_world_and_flora    13,9       14,3       14,6         konstant
_finalize_store_and_emit  4,4        4,8        6,1         nästan konstant
_step_sense_system        2,40      10,24      37,05       200 → 185
_step_move_system         1,50       6,27      24,69       125 → 123
_step_body_system         0,65       2,76      10,75        54 →  54
_step_decision_system     0,53       1,90       5,99        44 →  30
```

Sensingen är strikt linjär i antalet djur och blir inte billigare per individ med tätheten: den adaptiva frekvensen ger exakt en tredjedel full sensing per tick i alla tre körningarna (4/12, 16,7/50, 66,3/200). Extrapolationen från tolv djur förutsäger 104 ms vid 200 mot uppmätta 100, så den linjära modellen håller. Vid 500 djur och samma flora: ~220 ms/tick, varav sensingen ensam ~93.

**Sensingen är 46 procent av problemet, inte hela.** De fauna-linjära passen kostar tillsammans 395 µs per djur. Finfördelningen med `--pass-timing-inner` vid 200 djur:

```
RaySensors.see_agent_first_hit   19,3 ms    96 us/djur   53 % av sensingen
RaySensors.sense                 13,1 ms    66 us/djur   35 %
Agent._build_obs                  2,4 ms    12 us/djur
Agent._perform_feeding           19,4 ms    97 us/djur   78 % av move-passet
Agent._integrate_motion           3,5 ms    17 us/djur
```

Två fynd som ändrar steget.

`see_agent_first_hit` är den dyraste enskilda posten i hela faunakedjan och levererar **en granne**. Den kostar dessutom mest när den inte hittar något — 349 µs per anrop vid 12 och 50 djur mot 278 vid 200, eftersom den bryter tidigare vid hög täthet. Alltså dyrast precis i den glesa regim vi vill lämna.

`_step_move_system` är i praktiken inte rörelse. `_integrate_motion` är 17 µs; betningen via `world.consume_food` är 97. Betningen är alltså lika dyr som artfrändesensingen och ligger utanför det här steget som det är formulerat. Att lämna den betyder att 500 djur landar på ~130 ms i stället för ~220 — bättre, men inte tillräckligt.

### Vad aggregeringen kostar

Mikrobänk på den aritmetik omarbetningen skulle göra: vektoriserad axialoffsettabell, en gather av `(n_djur, K)` cell-ID, `bincount` till sex sektorer. Fem fält delar en gather.

```
r=7, K=169 celler       12 djur   0,116 ms/tick    9,7 us/djur
                       200 djur   1,361 ms/tick    6,8 us/djur
                       500 djur   3,891 ms/tick    7,8 us/djur
```

169 celler är fler celler än dagens 84 strålpunkter, och ändå ~7 µs mot 185. Strålarna översamplar nära och undersamplar långt; de är både redundanta och ofullständiga. Vinsten ligger inte i att läsa mindre utan i att sluta göra det i Python.

Två designkonsekvenser som mätningen pekar ut:

- **Sektorerna aggregeras i världsram och roteras till kroppsram efteråt.** Rotationen är en viktad blandning mellan grannsektorer, alltså samma operation som akuitetens oskärpa i `docs/synens-axlar.md`. De komponerar, och offsettabellen blir headingoberoende och delas av alla djur.
- **Artfrändekanalen går via täta per-cell-faunafält** byggda med `bincount` över `fauna_slots` en gång per tick: antal, massa, hastighetssumma. Då blir artfränder samma kodväg som floran — en gather, inget CSR-uppslag per cell — och percepten bär observerbara storheter i stället för `pheno.predation`. Defekt 8 i `docs/rorelsens-arkitektur.md` faller ut som biprodukt. Samma glesningstrick som `flora_cell_mass` redan använder håller kostnaden nere vid en miljon celler.

### Betningen — mätningen som sa emot designen

*0082. Underlag: samma A/B-mätning, samma process, interfolierade varv.*

Betningen är 97 µs per djur och tick och därmed lika dyr som artfrändesensingen. Uppdelningen inom den, mätt med en probe:

```
slots_in_cell x 7        48 us   45 %   sju searchsorted, ett Python-anrop i taget
B-loopen                 30 us   28 %   ~28 plantor, skalär numpy-indexering
cells_within             21 us   20 %   bredden-först med mängd och frontier
cell_of                   3 us
```

Det såg ut som ett lärobokstillfälle för vektorisering. Det var det inte.

En per-djur-vektorisering — grannskapet som en gather, slotarna i ett svep, girigheten som kumulativsumma — mättes mot originalet i samma process med identiskt utfall (noll avvikelser i B över 200 djur) och blev **1,7 gånger långsammare**: 80,5 mot 48,2 µs. Skälet är storleken. `reach = ceil(v · dt)` är 1 vid uppmätt fart, alltså sju celler och omkring 28 plantor. Numpys anropsoverhead är ~1–2 µs per operation, och femton operationer på 28 element kostar mer än Python-loopen de ersätter. Uppslaget mot store:n har ensamt ~30 µs fast overhead oavsett hur få celler det gäller.

Korsningen ligger vid r = 2. Uppslaget av slotar, µs per djur:

```
K = 7 (r=1)     gammalt 27      vektoriserat 34
K = 19 (r=2)    gammalt 70      vektoriserat 39
K = 37 (r=3)    gammalt 130     vektoriserat 51
```

**Vinsten ligger inte i att vektorisera per djur utan i att batcha över djur.** Samma aritmetik för tvåhundra djur i ett svep — `cell_of_many`, ett `cells_within_many` med formen (n, K), ett uppslag mot store:n, en `reduceat` per djur — kostar 2,26 µs per djur vid r = 1. Mot dagens 48 för samma arbete är det tjugoen gånger.

Det kräver att uppslaget skiljs från förbrukningen: alla djur rör sig, sedan slås grannskapen upp i ett svep, sedan betar djuren i tur och ordning på det redan uppslagna. Ordningen mellan djuren bevaras då exakt, och bara uppslaget batchas. Det är en uppdelning av `_step_move_system` i locomotion och feeding, vilket är den ordning manifestets fasmodell ändå anger.

**0082 levererar därför bara grannskapstabellen**, inte betningen. `cells_within()` byggs ur en cachad axialoffsettabell i stället för en sökning per anrop: 5,6 → 4,3 µs vid r = 1, 20,5 → 7,1 vid r = 2, 49 → 11 vid r = 3. Tabellen härleds ur `_bfs_within()` en gång per radie och ger identisk cellordning — verifierat mot sökningen för 2 500 celler och fem radier. `cells_within_many()` är samma topologi i batchad form och är förutsättningen både för feeding-passet och för sektorpercepten.

`slots_in_cells()` byggdes men ingår inte. Den har ingen vinnande anropare förrän passet delas, och att lägga in produktion utan konsumtion är precis det fel den här planens statusanalys gång på gång pekar ut i världslagret.

### Betningen, andra försöket

*0083. Mätning före design, igen — och den sa något annat än första gången.*

Girigheten rör **1,22 plantor per anrop**. Uppslaget som föregår den rör 12,5 celler och 13,5 plantor. Nittio procent av betningens kostnad ligger alltså i att ta reda på vad som finns, inte i att äta det.

```
uppslag + tillgänglighet   91 us
girighetsloopen            10 us
plantor girigheten rör    1,22
```

Och tillgängligheten låg redan färdig. `store.flora_cell_mass` är summan av `max(0, massa - rotmassa)` per cell — exakt det skottförråd betningshorisonten ser. Den byggdes om plantvis i en Python-loop vid varje tugga för ett tal som fanns i en array.

Ändringen är därför liten: tillgängligheten läses ur fältet, girigheten slås upp lat och hoppar tomma celler på fältet i stället för via spatialindexet, och varje tugga skriver av cellens förråd så att nästa betare inom samma tick ser rätt värde. Ingen batchning, ingen passdelning.

```
uppslag + tillgänglighet   115,7 -> 12,7 us      9,1 ggr   (samma process, interfolierat)
_perform_feeding i drift    kvot mot orörd motion 5,6 -> 2,7
B:s avvikelse mot plantsumman   median 3,2e-8, max 1,4e-7
```

Avvikelsen är float32-epsilon: `flora_cell_mass` lagras i float32, plantsumman räknades i float64. Ändringen är alltså **inte bit-för-bit**. Banorna följs åt inom ett par individer och några procent av floramassan över 600 tick på två seeds, utan systematiskt tecken.

**Passdelningen behövdes inte.** Slutsatsen från 0082 — att vinsten kräver batchning över djur — gällde uppslaget av *slotar*. När tillgängligheten inte behöver slotarna alls faller behovet bort. Batchningen står kvar som nödvändig för sektorpercepten i 0084, där varje djur verkligen behöver läsa hela sitt grannskap.

### Sektorpercepten — världskanalerna

*0084. Första patchen i serien som inte kan vara neutral.*

Strålarna samplade punkter: tolv riktningar gånger sju avståndssteg, med `grid.cell_of` per punkt i en nästlad Python-loop. De översamplade nära — alla tolv strålarna landar i samma sex celler vid avstånd ett — och undersamplade långt, där grannskapet är brett och strålarna glesa.

Nu aggregeras varje cell inom räckvidden till sin riktningssektor. **Räckvidden är oförändrad; det är täckningen inom den som blir hel.** 169 celler mot 84 samplingspunkter, sex sektorer, en per hexgranne. Aggregeringen sker i världsram med en gemensam offsettabell för alla djur på en gång och roteras sedan till kroppsram med en fraktionell cirkulär förskjutning — samma operation som en akuitetsoskärpa, så de komponerar när den kommer.

Mättnaden `x / (x + K)` läggs per cell före medelvärdet, precis som strålarna gjorde per samplingspunkt, med `B_K` och `C_sense_K` oförändrade. Vikten faller med avståndet.

```
_step_sense_system     338,9 -> 246,3 us/djur
fauna-linjära pass     580,3 -> 496,3 us/djur
sektoraggregeringen     16,8 us/djur vid 200 djur (mot sense 118)
```

Referenspassen låg still under mätningen — `_integrate_motion` 25,0 → 26,0, `_perform_feeding` 67,8 → 69,2, `_step_world_and_flora` 112,3 → 110,8 — så maskinstaten är jämförbar och skillnaden är verklig.

**Varför OBS_DIM inte ändrades.** `_build_obs` kollapsade redan strålarna till mean, max och bäringen till max: tolv tal blev sex. Sektorerna går därför in i samma kontrakt utan att observationsvektorn byter längd, utan att hjärnorna ogiltigförklaras och utan att cachevägens indexering skrivs om. Att exponera de sex sektorvärdena direkt för MLP:n är ett eget steg, där dimensionsbytet blir den enda ändringen.

**Inte neutral.** Bruset drar nu sex tal per kanal i stället för tolv, så slumpströmmen skiljer sig, och percepten bär annan information. Banorna över 600 tick på tre seeds ligger inom ett par individer och några procent av floramassan, utan systematiskt tecken — men jämförelser mot äldre loggar på tickavstånd gäller inte längre.

**Kvar:** `see_agent_first_hit` är orörd och är nu 176 av sensingens 246 µs. Den uppfattar exakt en artfrände, kostar mest när den inte hittar något, och läser motpartens arvsmassa i riskvärderingen. Artfrändekanalen återanvänder samma gather och är nästa patch.

>>>>>>> Stashed changes
### Artfrändekanalen

*0085. Strålmarschen är borta.*

`see_agent_first_hit` gick avståndssteg för avståndssteg och stråle för stråle med `grid.cell_of` per punkt. Den kostade **mest när den inte hittade något**, eftersom den då gick hela vägen ut — alltså mest i den glesa regim modellen ska lämna. Vid 200 djur var den 176 av sensingens 246 mikrosekunder.

Grannskapet slås nu upp mot ett faunaeget cellindex. Faunan är hundratal mot florans tiotusental, så indexet är litet och uppslaget behöver inte sålla bort växter en och en. Kandidaterna är typiskt noll eller ett par per grannskap, och de få som finns avgörs vektoriserat.

```
_step_sense_system     338,9 -> 53,1 us/djur     (0084 tog den till 246,3)
fauna-linjära pass     580,3 -> 290,0 us/djur
see_agent_first_hit    176,2 -> 0 anrop
```

Referenspassen låg still: `_integrate_motion` 25,0 → 24,6, `_perform_feeding` 67,8 → 70,0, `_step_world_and_flora` 112,3 → 111,0.

**Lokaliteten är oförändrad.** Kandidater hämtas ur celler inom synellipsens räckvidd, aldrig ur en global lista. Räckvidden är samma ellips som strålarna hade — `r(θ) = r_front (1-e) / (1 - e cos θ)` — så synfältet är fortsatt framåtriktat och den bakre blindzonen består. `m_eff` kapar räckvidden precis som `ray_depths` gjorde.

**Träffen är nu den verkligt närmaste inom räckhåll**, inte den första som råkade ligga på en stråle. Strålarna hade hål mellan sig som växte med avståndet, och en granne kunde vara osynlig för att den stod mellan två strålar.

**En bugg som ändringen blottade.** Födoreflexen läste `sensors._accB` och `_ang_base` direkt — strålsensorns buffertar. Efter 0084 fylls de aldrig, och reflexen läste oinitierat minne. Den läser nu `_acc_dir_B` och `_acc_dir_ang`, som sätts av den perceptionsväg som faktiskt kördes. Fångades av `-W error::RuntimeWarning`, inte av invariantsviten.

**Kvar:** riskvärderingen i `attack_risk` läser fortfarande `pheno.predation`, alltså motpartens arvsmassa. Och percepten bär ännu bara *en* artfrände till MLP:n — sektoraggregat av artfrändetäthet kräver att `OBS_DIM` växer, och det bör vara den enda ändringen i sin egen patch.

### Näringen sås vid sin jämvikt

*0086. Underlag: floraköring utan fauna, 60 000 tick, `runs/p85-flora`.*

`nutrient_init` är den **fria**, växttillgängliga poolen — inte näring bunden i marken. Bundet material är detritus, och det stod på noll. Världen startade alltså med hundra procent av sin näring omedelbart tillgänglig och steril mark. Verkliga ekosystem ligger på motsatt ytterlighet: under en procent tillgängligt, resten i markens organiska material.

Två följder, båda mätta.

**Inkörningen var mest bokföring.** Fri pool 4 142 kg vid tick 500, 50 kg vid jämvikt. De 45 000 ticken gick till största delen åt att flytta näring från den fria poolen in i växter och förna, inte till att floran hittade sin form. Antalet plantor planade ut redan kring tick 12 000.

**Stocken bar inte sig själv.** 5 243 kg såddes, 4 854 återstod efter hundra år, fortfarande fallande. Varje körning startade bördigare än världen kan hålla, så bärkraft mätt vid någon given tick var en mätning på ett förlopp.

Jämvikten följer av en identitet:

```
mineralisering_jämvikt = nutrient_input · n_cells / nutrient_loss_frac = 904 kg/år
uppmätt vid tick 60 000                                               = 1 293 kg/år
```

Nedbrytningstakten påverkar inte det flödet. Den bestämmer bara hur stor detrituspoolen måste vara för att bära det. Faktorn 0,699 ger jämvikten 1 475 kg näring i detritus, 1 885 i flora och 35 fri — totalt 3 395 mot sådda 5 243.

Talen sår den fördelningen direkt. Detritus läggs på sin jämviktsnivå; florans andel läggs i den fria poolen, eftersom det är därifrån växterna tar upp.

```
nutrient_init            0,32  ->  0,117    kg fri näring per cell
detritus_init            0,0   ->  21,16    kg förna per cell
detritus_structure_init         0,93        ny parameter
```

Strukturandelen behövs för att `detritus_structure` styr nedbrytningstakten. Sås förna utan den bryts allt ner som labilt material och hela poolen mineraliseras på nio månader. 0,93 är den uppmätta jämviktssammansättningen: strukturell förna bryts ner 6,7 gånger långsammare och anrikas därför i poolen oavsett vad floran fäller.

Utfall över 6 000 tick:

```
             före          efter
tillfört    6 147 kg      3 461 kg
förlorat    1 293 kg         90 kg
takt        -1 166 kg/h    +23 kg/h
tidskonstant     4 h        144 h
```

**Invariantsviten fångade ett verkligt fel.** Ledgern bokförde utgångstillståndets fria näring och den sådda florans vävnad som tillförd men inte förnan — posten var noll så länge `detritus_init` var det. 1 454 kg av 3 371 låg i marken och såg ut som en läcka på 43 procent. Sviten hittade det på tick 0.

**Kvar att verifiera:** var floran landar. Förutsägelsen är 121 000 kg biomassa och omkring 66 000 plantor, nått kring tick 12 000-15 000 i stället för 45 000. Vid tick 6 000 stod den på 87 500 kg och 68 869 plantor, vilket ligger i banan.

Talen skalar linjärt med `nutrient_input`. Det gör bördighetsstegen enkel att sätta upp: fördubblad bördighet fördubblar alla tre.

### Sådden får den täthet konkurrensen ändå ger

*0087. Fynd ur en faunakörning som visade sig mäta fel sak.*

Sextio djur i den nyseedade världen kollapsade på hundra månader: floran föll 130 170 → 20 268 kg och faunan till noll. Men körningen var ogiltig, och skälet stod i första raden:

```
[flora] sådd nådde 130 169 kg av begärda 191 390 — cellrymden (16384) räckte inte
```

Sådden drog celler **utan återläggning**. Varje cell kunde få högst en planta, så 130 170 kg blev 16 384 individer på 7,94 kg. Jämvikten är 89 319 plantor på 1,32 kg.

Djuren mötte alltså sextusen jättar där det borde stå nittiotusen småplantor. Betningshorisonten verkar per planta — betaren tar skottet och lämnar roten — så refug, återväxt och funktionell respons såg en värld som inte uppstår i drift. Det förklarar också varför floran föll så fort trots att totalmassan låg nära jämvikten.

**Hur många plantor en cell rymmer ska avgöras av rotkonkurrensen, inte av sådden.** Den mekanismen finns redan och är verksam: anspråket är rotarea mot cellarea 1, överskott spiller till de sex grannarna, och summan skalas ner där marken är slut. Jämvikten har 5,45 plantor per cell. Bara sådden kände inte till den.

Ändringen är att dra celler **med** återläggning och låta antalet följa av en rimlig plantstorlek i stället för av världens cellantal:

```
flora_init_plant_mass = 1,32 kg     (uppmätt jämviktsmassa per individ)
n_flora = måltotal / plantmassa
cells   = rng.choice(n_cells, size=n_flora, replace=True)
```

Utfall vid samma måltotal: 144 993 plantor, 8,85 per cell, 16 383 av 16 384 celler bebodda, median 9 och max 23 per cell. Mot tidigare 16 384 plantor och exakt en per cell.

**Tätheten är dessutom självbegränsande.** Sådden köps ur den fria näringspoolen sedan 0086, så en cell med `nutrient_init` = 0,117 kg räcker till omkring fem plantor av den storleken — samma storleksordning som jämviktens 5,45. Vid `--flora-ratio 2000` och `init_pop 60` levererade sådden 119 984 kg av begärda 191 390, alltså praktiskt taget exakt jämviktsmassan 118 104. Taket är nu världens verkliga bördighet i stället för dess cellantal, och diagnostikraden säger det.

Med tjugo djur i den nya sådden håller sig beståndet på 17–26 över 3 000 tick i stället för att kollapsa.

**Kvar:** sådd flora har fortfarande slumpade traits och ingen rumslig struktur — inga fläckar, ingen släktskapsstruktur, ingen selektionshistoria. En kort inkörning behövs fortfarande, men några tusen tick i stället för tjugotusen.

### Fördröjd insättning och grundarfläck

*0088. Två fel som båda gjorde faunakörningar ogiltiga.*

**Faunan mätte en halvfärdig värld.** Sådden ger 56 procent av jämviktens stående gröda, produktionen skalar med den, och ett bestånd som ligger under bärkraften i den färdiga världen ligger över den i den halvfärdiga. Tjugo djur betade ner floran, tappade kondition — medelmassa 1,75 → 1,05 kg — och när floran väl växte tillbaka var beståndet redan för litet. `--fauna-at N` håller tillbaka insättningen tills floran nått jämvikt.

**Jämn utspridning garanterar Allee-fällan.** Synellipsen med `r_front = 7` och excentricitet 0,7 täcker `π r² (1-e)² / (1-e²)^{3/2}` = **38 celler** av 16 384, alltså 0,23 procent:

```
 20 djur  ->  0,04 i synfältet  ->   4 % av tickarna med någon i sikte
100 djur  ->  0,23              ->  21 %
300 djur  ->  0,69              ->  50 %
```

Bärkraften är tjugo djur; mötesfrekvensen kräver trehundra. Vid jämn utspridning är talen femton gånger isär, och beståndet kan inte samtidigt vara stort nog att hitta varandra och litet nog att få mat. Under omkring tio individer upphör reproduktionen helt — uppmätt: 24 födslar mellan tick 1000 och 3000, en enda mellan 6000 och 12000.

`--fauna-spawn-radius` sätter djuren i en fläck. Tjugo djur i tusen celler ger sexton gånger den globala tätheten och ungefär femtio procents mötesfrekvens i stället för fyra. Fläcken köper ett etableringsfönster, inte en lösning: tusen celler producerar omkring 3 400 kg per år och bär ett par individer, så beståndet måste sprida ut sig och möter då samma geometri igen. Vad den gör är att **skilja de två felen åt** — dör beståndet trots parningar under fönstret är det spridningen och inte fodret.

Tre följdfel som ändringen blottade, alla fångade av invariantsviten eller av synliga tal:

- Med fördröjd fauna är `fauna_mass0` noll, och florasådden — som skalas mot den — blev noll. Floran uteblev helt. Skalan tas nu ur den fauna som kommer.
- Djur som sätts in efter tick noll får sin massa gratis och måste bokföras som tillförd näring, precis som utgångstillståndet. Utan det drev balansen 2,2e-4.
- Sådden allokerade 333 333 slots för en sådd som marken sänkte till 82 885 plantor, och store:n växte till en halv miljon. Antalet kapas nu av vad den fria näringspoolen kan betala.

### Utbredningsmåttet

*0089. Litet, men det skiljer två fel som ser likadana ut i populationskurvan.*

Ett bestånd som diffunderar fritt sprids som roten ur tiden och tappar mötesfrekvens tills reproduktionen upphör. Ett som håller ihop planar ut på ett avstånd satt av kohesionen. Båda kan sluta i noll djur, men bara det första är ett geometriproblem.

`--stats` rapporterar nu avstånd till närmaste artfrände — medel och median, toroidalt — och andelen djur som har någon inom `ray_len_front`. Den andelen är den storhet som direkt förklarar parningsfrekvensen.

Referenspunkten är jämn utspridning. Synellipsen täcker `π r² (1-e)² / (1-e²)^{3/2}` = 38 areaenheter av 16 384 vid r = 7 och e = 0,7, alltså 0,23 procent, och tjugo djur ser då varandra fyra procent av tickarna. Ligger andelen väsentligt högre håller beståndet ihop.

Första mätningen, 800 tick efter insättning i en fläck med radie 20: **19 procent har någon inom synhåll** mot fyra vid jämn utspridning. Fläcken finns alltså kvar men glesnar.

**Och kohesion kräver inte ett grannskap.** Reynolds tre regler använder ett, men det är alignment och formationens stabilitet som behöver det. Aggregering — attraktion på avstånd, repulsion på nära håll — fungerar med den närmaste grannen, och den informationen finns redan i percepten som `N_ag`, `pred_bearing` och `pred_dist`. Sektorutvidgningen är därför inte en förutsättning för flockning, vilket jag felaktigt hävdade. Den ger jämnare alignment och en känsla för lokal täthet.

Sektoraggregat ökar inte heller synvidden: sex sektorer över samma ellips ser samma 38 celler. Detektionsproblemet löses varken av en granne eller av ett grannskap — det är radien som sätter det. Vad som kan lösas är att hålla ihop när man väl är tillsammans, och för det räcker en granne.

### Sociability som startvillkor

*0090. Kontrollerat experiment, inte en mekanismändring.*

Reflexen finns redan: `soc_bias = 2·soc − 1` svänger mot artfränden när `sociability` är hög och bort när den är låg. Men **nollpunkten ligger vid 0,5**, så halva en uniformt slumpad startpopulation styr aktivt bort från artfränder. Och driften uttrycks bara när någon syns — i storleksordningen tre procent av agenttickarna — så selektionen på locus hinner inte verka innan beståndets öde avgjorts. Egenskapen driver i praktiken neutralt.

`--sociability-init` sätter grundarnas värde. Locus muteras normalt vidare, så avkomman kan avvika; det är startfördelningen som styrs, inte taket.

Utbredningsmåttet från 0089 fick samtidigt sin **Poissonreferens**. Utan den är andelen inte tolkbar: den stiger med beståndets storlek även vid helt slumpmässig fördelning. I p89-körningen såg 70 procent någon inom synhåll vid femtio djur — men Poissonförväntan vid samma täthet är 62, och medelavståndet 8,58 mot 9,05 vid slump. Beståndet var alltså bara marginellt klumpat, och de 70 procenten kom av täthet och inte av kohesion. Kvoten mot förväntan är det tal som säger om djuren håller ihop.

**Baslinjen att jämföra mot är p89**, som är den första körning där beståndet bar sig självt: insatt vid tick 15 000 i en fläck med radie 20, oscillerande mellan 22 och 50 över 25 000 tick, 341 födslar och 361 unika individer, med floran i motfas — en rovdjur–bytesdjur-cykel. Bärkraften ser ut att ligga på 35–50, högre än den skattning på 15–25 som gjordes ur kollapskörningen.

Den kvarvarande flaskhalsen är inte detektion utan vad som händer efter den: 5 276 tillfällen då en partner sågs gav 364 parningar, alltså sju procent.

### Alignment — den tredje regeln

*0091. Mekanismen som saknades för att flockning ska vara möjlig.*

0090 visade att `sociability` inte producerar aggregering: kohesionskvoten mot Poisson mättes till 1,02, 0,98 och 1,10 för base, 0,65 och 0,80 — alla ett. Höga värden skadade dessutom: vid 0,95 dog två av tre populationer ut och medelbeståndet föll till 14 mot baslinjens 30.

Djuren betalade alltså kostnaden utan att få nyttan, och koden visar varför.

```python
REP_ZONE = 0.35
if Nd_f < REP_ZONE:                     # separation
    turn -= 0.70 * rs * biasN
else:                                    # kohesion
    wdist = clamp(1.0 - Nd_f, 0.0, 1.0)
    turn += 0.70 * soc_bias * wdist * biasN
```

Två av Reynolds tre regler fanns. **Alignment saknades helt**, och det är den regel som omvandlar ett möte till ett sällskap: två djur som möts och sedan matchar kurs färdas tillsammans, två som möts utan att matcha divergerar omedelbart.

Viktningen var dessutom omvänd. `wdist = 1 − Nd` är störst på nära håll, alltså starkast precis där separationen borde ta över och svagast där gruppen behöver dras ihop. Hos Reynolds verkar kohesionen på lång räckvidd.

Ändringen ger de tre reglerna var sin räckvidd:

```
nära      separation     stöt bort
mellan    alignment      matcha kurs      w = 4·x·(1−x), topp vid halva vägen
långt     kohesion       dra ihop         w stiger med avståndet
```

Alignment läser motpartens **kurs**, inte dess arvsmassa — en observerbar storhet. Den hämtas i reflexen från den redan detekterade agenten, så `OBS_DIM` är oförändrad.

En tredje rättelse: utforskningsdriften dämpades med `abs(soc_bias)`, så även ett djur som *undviker* artfränder slutade söka föda. Dämpningen gäller nu bara den som söker sällskap.

**Mekanismen, inte beteendet.** Vad som kodas är att alignment är möjlig; `sociability` styr styrkan och är evolverbar. `--sociability-init` är fusket vi mäter med under utvecklingen, och locus släpps fritt när mekanismen visat sig fungera.

**Overifierad.** Sandlådan hinner inte en giltig körning: faunan måste sättas in i en flora vid jämvikt, alltså tidigast tick 15 000, och 40 000 tick tar tjugo minuter. Kohesionskvoten mot Poisson är måttet som avgör om alignment gör skillnad.

### Synfältet och minnet

*0092. Två fynd ur en visuell körning, båda räknebara i efterhand.*

0091 gav ingen mätbar flockning: kohesionskvoten mot Poisson låg kring ett i alla fyra nivåer och båda körningarna, med spridning inom en nivå från 0,79 till 1,88. Medelbeståndet var oförändrat, 24,2 mot 25,3. Vid visuell granskning syntes djuren upptäcka varandra frontalt, göra en ömsesidig riktningsändring — och sedan fortsätta i tangentens riktning.

**Synfältet är blint åt sidan.**

```
r(θ) = r_front · (1−e) / (1 − e·cos θ)     r_front = 10, e = 0,7

rakt fram    10,0        åt sidan   3,0
snett bak     2,1        rakt bak   1,8
```

En flockkamrat som färdas jämsides på fyra enheters avstånd är osynlig — och det är just där en flockkamrat befinner sig. Alignment kan inte fungera i ett sådant synfält oavsett hur regeln formuleras.

Sedan 0084 läser världskanalerna sitt grannskap isotropt via sektoraggregat, så ellipsen styr numera **bara** var en artfrände kan upptäckas. Att sänka excentriciteten till 0,3 påverkar därför ingenting annat: tio fram, sju åt sidan, 5,4 bak. Framåtriktningen är kvar men sidosynen räcker för att hålla sällskap. Det bör också höja parningsfrekvensen, som legat på sex till sju procent av alla tillfällen då en partner setts, orörd av allt vi gjort.

**Tidsskalan är knapp.** Vid 0,74 enheter per tick och sensing var tionde tick i vila förflyttar sig djuret 7,4 enheter mellan två sensingar — nästan hela synfältets längd. Alignmentfönstret är fyra och en halv enheter brett, vilket två djur på frontal kurs sluter på tre tick. De hinner ett enda samplingstillfälle per möte.

`social_memory_ticks` låter en sedd artfrände minnas efter att den lämnat synfältet. Positionen dödräknas framåt längs dess senast sedda kurs och tilltron avtar linjärt med åldern. Minnet styr **bara** — parning och predation kräver fortfarande en verkligt detekterad motpart.

Sätt `social_memory_ticks = 0` för att pröva excentriciteten ensam. De två ändringarna är avsiktligt separerbara.

### Rörelse- och perceptionsflaggor

*0093. Verktyg, inte beteende. Förvalen är oförändrade.*

Efter 0092 syntes fortfarande ingen flockning, varken vid ekologisk täthet eller i ett tätt visuellt test med `sociability = 0,9`. Mätning av reflexen visade att den **verkar**: den utlöses i 63 procent av anropen, flyttar styrkommandot med median 0,128 och klipps aldrig av mättnad. Men var tredje detektion gäller en **annan individ**, och var sjätte byter den sociala termen tecken. Nettoeffekten över ett möte är en slumpvandring i kurs.

Orsaken är dimensionslös och räknebar:

```
fart median 44 per månad  ->  0,88 enheter per tick
sensing var 10:e tick i vila
sträcka mellan två sensingar   8,80 enheter
synvidd                        7,00 enheter
kvot                           1,26
```

**Djuret förflyttar sig längre mellan två sensingar än synfältet är långt.** En granne som syns vid en sensing är ofta utanför räckhåll vid nästa — inte för att den flyttat sig, utan för att observatören gjort det. Vid det täta testet fanns bara två kandidater inom synvidd i median, så grannbytena beror på att den gamla grannen lämnat synfältet, inte på att kandidaterna är många.

Kvoten går att sänka från två håll, och båda ska kunna prövas utan kodändring:

```
--drag-lin      farten sätts av kraftbalansen F0·M^(2/3) mot drag_lin·v + drag_quad·v²
--drag-quad     vid v = 44 dominerar den linjära termen fyra mot ett
--v-max         binder inte vid förval: taket är 100, realiserad fart 44
--sense-idle    tick mellan sensingar i vila, förval 10
--sense-alert   tick mellan sensingar i beredskap, förval 3
```

Uppmätt:

```
drag_lin  220    fart 44,0    8,80 enh mellan sensingar    kvot 1,26
drag_lin 1350    fart  8,3    1,65 enh mellan sensingar    kvot 0,24
```

1350 ger alltså 81 procents fartsänkning och fyra observationer per gång en granne passerar synfältet.

Att räkna med vid sänkt fart: rörelse kostar energi, så åtta gånger långsammare djur betalar långt mindre för locomotion och energibudgeten ändras kraftigt. Betningsräckvidden `ceil(v·dt)` förblir 1 i båda fallen.

### Sådden mot marken, inte mot konsumenterna

*0094. Samma koppling har nu brustit fyra gånger.*

`target_mass = flora_init_mass_ratio · fauna_mass0` kräver att det finns fauna att skala mot. Bristerna i tur och ordning: vid ändrad världsstorlek gav samma ratio hundra gånger fel födotäthet; vid ändrad `init_pop` likaså; vid fördröjd fauna blev målet noll och floran uteblev; och senast vid `--init_pop 0` sådde en bördighetsstege med tolv körningar **en enda planta på 7,64 kg** och mätte därmed ingenting alls.

Rätt storhet fanns hela tiden. Sedan 0087 köps sådden ur den fria näringspoolen, så bördigheten var redan taket — nu blir den **målet** i stället för en spärr.

```
init_pop=0   nutrient_init=0,117   ->   82 201 plantor,  94 782 kg
init_pop=0   nutrient_init=0,936   ->  657 608 plantor, 851 541 kg
init_pop=20  nutrient_init=0,117   ->   51 588 plantor,  65 807 kg
```

Vid faktor 1 hamnar sådden på 94 782 kg mot jämviktens uppmätta 118 104, alltså i rätt storleksordning utan att någon parameter behöver ställas per körning. `flora_init_mass_ratio` används fortfarande när fauna finns, så befintliga körningar är oförändrade.

Åttafaldig bördighet ger 657 608 plantor, alltså **åtta gånger fler och inte större**. Det är en första indikation på att antalet skalar linjärt med bördigheten och att en femdubbling därför kostar omkring 160 ms/tick — men det är sådden och inte jämvikten, och bördighetsstegen ska avgöra saken.

### Mutationssteget och grundarspridningen

*0095. Två fel som tillsammans gjorde evolutionen omöjlig på våra tidsskalor.*

Analys av `life.jsonl` från p89 och p91 visade att fenotyperna är i praktiken kloner. I p89 hade 203 individer med kull och 158 utan **identisk median på tre decimaler** i `A_mature`, `M_target`, `sociability` och `mobility`. I p91 låg `median_M` fast på exakt 0,714 kg i varje mätpunkt från tick 540 och framåt.

**Den verksamma mutationsstorheten är `σ√p`, inte `σ`.**

```
traits_sigma = 0,02   traits_p = 0,05   ->   σ√p = 0,0045 logit/generation
                                        ->   0,0007 fenotypenheter
```

Uppmätt spridning bland 264 avkommor i p91: 0,0017 fenotypenheter. Vid mutation-drift-jämvikt `V = 2·Ne·σ_m²` svarar det mot **Ne ≈ 5** — i ett bestånd som räknade tjugo till femtio individer. Två grundarlinjer av tjugo tog 73 procent av avkommorna, och sex av tjugo lämnade ingen avkomma alls.

Drift raderar dessutom startvariationen snabbare än den kan användas:

```
Ne =  5    kvar efter 70 generationer   0,1 %    halveringstid    7 gen
Ne = 25                                24,3 %                    34 gen
Ne = 50                                49,5 %                    69 gen
```

**Kravet.** För att flytta en egenskap en halv fenotypenhet över 200 generationer vid måttlig selektionsintensitet (i = 0,2) krävs σ_A ≈ 0,078 i logit-rymden, vilket vid Ne = 20–50 motsvarar σ_m mellan 0,008 och 0,018.

```
traits_sigma  0,02 -> 0,05        traits_p  0,05 -> 0,15
σ√p           0,0045 -> 0,019     drygt fyra gånger
```

Högre än så är kontraproduktivt: mutationssteget börjar då konkurrera ut selektionen, och egenskaperna blir brus i stället för anpassningar.

**Grundarspridningen.** `--sociability-init` satte alla grundare till samma tal, så p90 och p91 jämförde tre konstanter snarare än tre startfördelningar. `sociability_init_sd` gör värdet till ett medelvärde: vid 0,5 i logit-rymden blir fenotypspridningen 0,077 kring 0,8, alltså p10 0,685 och p90 0,873. Det är samma storleksordning som den stående variation en verklig grundarpopulation bär, och drygt hundra gånger den effektiva mutations-sd:n per generation.

**Förbehåll.** Ne ≈ 20–50 är förutsättningen för att någotdera ska verka. Vid census 20–50 med skev reproduktion får vi Ne = 5, och då blir två linjer av tjugo två linjer av tjugo oavsett mutationstakt.

### Reproduktionen är inte energibegränsad

Ur `pop.jsonl` från p89, andel av energiintaget:

```
underhåll (basal)   77,8 %      termoreglering   60,1 %
rörelse              7,0 %      tillväxt          0,0 %
dräktighet           0,0 %
```

Dräktigheten är fem tiopotenser mindre än underhållet. Sjutton månader mellan kullar kan alltså inte förklaras av att spara ihop energi — kvar står dräktighetstiden och partnersökningen, och den senare är mätt till **fjorton månaders väntan efter mognad** med 42 procent som aldrig får någon kull.

Underhåll och termoreglering summerar till 138 procent av intaget: beståndet går med underskott och lever på reserven mellan måltider. Tillväxten är nära noll, vilket förklarar varför medianindividen aldrig når mer än hälften av `M_target`. Att termoregleringen ensam tar 60 procent av intaget är värt en egen undersökning.

`repro_cooldown_s = 8,0` binder aldrig: median kullintervall är 16,2–17,3 månader och **noll procent** av intervallen ligger under parametern.

### Köldaversionen får en läsare

*0096. En egenskap som fanns men inte gjorde något.*

`cold_aversion` hade **fem förekomster i hela repot**, alla i `phenotype.py`: deklaration, intervallgränser, härledning och export till loggen. Noll läsare. Djuren bar en ärftlig köldaversion som inte påverkade någonting.

Följden är mätbar i p89. Rumslig analys av 311 dödsfall mot temperaturfältet:

```
                dödsfall   andel   andel av världen
T < 5 °C            183    58,8 %          30,5 %
5-10                 66    21,2 %          14,1 %
25-30                 8     2,6 %          19,5 %

svält   n=233   T median  2,6 °C
skada   n= 76   T median 12,2 °C
```

**Svältdöden är en kylddöd.** Åttio procent av dödsfallen sker under tio grader. Och djuren drev dit: temperatur vid födsel median 11,5, vid död 3,8, med 68 procent som slutade kallare än de föddes. Vägsträcka 1 372 mot nettoförflyttning 175 — ren slumpvandring.

Kostnaden förklarar varför det dödar:

```
Tenv    termo/basal      fixkostnad relativt 2,6 °C
  0        1,23                 1,00
 12        0,84                 0,86
 25        0,40                 0,65
 30        0,23                 0,57
```

Vid noll grader kostar termoregleringen mer än hela basalmetabolismen. Underhåll plus termo låg på 138 procent av energiintaget; i de varmaste banden vore samma post 79 procent. Skillnaden mellan kroniskt underskott och överskott ligger i var djuret står.

**Ändringen** lägger temperaturen som tredje världskanal i sektorpercepten från 0084 — samma gather, ingen mättnad, eftersom temperatur är ett värde och inte en mängd. Reflexen styr mot den lokala gradienten, beräknad som viktad vektorsumma över sektorernas avvikelse från sitt medelvärde, skalad av `cold_aversion` och av hur mycket djuret faktiskt fryser.

Utfall efter 1 500 tick med 60 djur:

```
start                     T median 13,5
tick 1500                 T median 22,6     (världens median 12,2)
  hög cold_aversion                24,7
  låg cold_aversion                16,2
```

**Motverkande konsekvens finns i geometrin.** De varma banden är ungefär en fjärdedel av världen. Den som söker värme trängs med alla andra som gör det och betar hårdare lokalt; den som tål kyla får resten av världen. Det är en nisch-axel, inte en gratis vinst.

Och koncentrationen är en biprodukt värd att mäta: samlas djuren i banden fyrdubblas tätheten där, vilket mildrar mötesproblemet — fjorton månaders väntan efter mognad, 42 procent barnlösa — utan att någon flockningsmekanik behöver fungera.

### Parningsvillig som egen sökkanal

*0097. Defekten stod dokumenterad från början.*

Detektionen var odiskriminerande: `_acquire_neighbours` gav närmaste artfrände oavsett tillstånd, och filtret på parningsvillighet kom först efteråt i `_resolve_detected_agent`. Var den närmaste ovillig hade djuret **ingen** partner alls — även om en villig stod tio procent längre bort. Det är precis den defekt `docs/rorelsens-arkitektur.md` beskriver: fel granne närmast blockerar parningsdriften.

Med 32,7 procent grannbyten mellan på varandra följande detektioner byttes målet dessutom under approachen. En villig partner byter inte tillstånd varje tick, så en egen kanal ger något stabilt att hålla fast vid under de åtta till fyrtio tick det tar att sluta avståndet.

Ett parningsberett djur söker nu närmaste **villiga** inom synhåll; övriga söker närmaste artfrände. Sorteringsnyckeln gör det utan en andra gather. Lokaliteten är orörd — samma celler, samma synellips. Biologiskt är det en separat perceptuell kanal och inte en genväg: läten, dofter och uppvisning är evolverade just för att vara märkbara på avstånd.

**Bakgrund ur p96.** Sex körningar vid f4-bördighet med 80 startdjur. Perceptionen löstes — andelen blinda gick från 96,9 procent i p89 till 62–72 — och toppopulationerna nådde 159–193 mot tidigare högst 113, med upp till 942 unika individer. Men parningsandelen föll från 6,5 till 2,1 procent, och andelen reproduktionsberedda som ser en partner utanför parningsradien steg från 0,9 till 21,8.

`halv-s2` genomlevde fyra fullständiga rovdjur–bytesdjur-cykler över 60 000 tick, med återhämtning från tre individer till nittiotvå tre gånger om, innan den fjärde bottnen tog den. Långsammare djur ger grundare svängningar och överlevnad genom bottnarna.

**Vad som inte var problemet.** Reproduktionens kostnad är korrekt bokförd: fostervävnad byggs ur moderns reserv kilo för kilo, och räcker inte reserven kataboliserar hon egen vävnad. `E_build_gestation` är bara syntesarbetet ovanpå materialet — att jämföra den mot basalmetabolismen och dra slutsatsen att reproduktionen är gratis var en felläsning. Kullintervallet ligger på 15,6 månader i p96 mot 17,3 i p89, alltså praktiskt taget oförändrat mellan en explosiv och en stagnerande körning: takten är fysiologisk och redan emergent. `repro_cooldown_s = 8` binder i ingendera.

Explosionen kom från antalet, inte från takten. Median kullar per individ är fortfarande noll, medelvärdet 0,92 — populationen växer genom att fler individer lyckas alls.

**Nästa steg.** `mating_radius = 3,0` motsvarar nästan tre cellbredder och en träffyta på 28 celler, med grannavstånd 1,07. Parning bör kräva samma cell, alltså radie kring 0,6. Men att strama åt den innan approachen håller skulle sänka träffytan trettiofem gånger och stoppa reproduktionen helt — radien sätts därför efter att den här ändringen mätts.

### Säsongsbunden parning

*0098. Kvadreringen är det som dödar reproduktionen.*

Beredskapen var asynkron: varje individ blev redo på sin egen tidtabell. Uppmätt i p97 var **14,8 procent** av agenttickarna reproduktivt klara, och två djur måste vara det *samtidigt*:

```
båda redo   0,148²  =  2,2 %
```

Innan de ens ska hitta varandra och sluta avståndet. Det förklarar också varför 0097 inte hjälpte: att söka närmaste **villiga** i stället för närmaste artfrände gör målet längre bort när villiga är sällsynta, och andelen utanför parningsradien steg från 21,8 till 23,3 procent. Premissen var fel — problemet var inte fel granne närmast utan att villiga partners knappt finns.

Grindas beredskapen till en period blir alla redo samtidigt:

```
i dag                0,148² × 12 mån  =  0,26 månadsekvivalenter per år
tvåmånadersfönster   1,0²   ×  2 mån  =  2,0        ungefär åtta gånger fler
```

Två loci: `breed_phase` (när på året) och `breed_sync` (hur snävt). Grinden är von Mises-liknande, `exp(k·(cos Δ − 1)) ≥ 0,5`:

```
skärpa 0    fönster 12,0 mån    ingen säsong — dagens beteende
skärpa 1             4,9
skärpa 3             2,7
skärpa 6             1,9
```

Vid skärpa noll är grinden konstant ett, så det asynkrona beteendet är bevarat som en punkt i parameterrummet snarare än borttaget. Kuen är årscykeln som redan finns — `year_len` med sinusformad temperatur — så ingen ny perception behövs.

**Avvägningen är verklig och behöver inte uppfinnas.** En strikt säsongsparare som missar fönstret väntar ett år. Ju snävare synkronisering, desto större vinst i mötesfrekvens och desto större risk att individen inte är i kondition just då. Den adaptiva motiveringen är också verklig: säsongsparare synkroniserar för att avkomman ska födas när fodret är rikligt, och floran har en temperaturgrindad tillväxtcykel.

`n_traits` går från 38 till 40.

### Om sådden vid hög bördighet

p97 visade självgallring rent: sådden gav 328 804 plantor på 1,27 kg, jämvikten 92 370 på 3,42 kg. Antalet faller till 28 procent medan medelmassan nästan tredubblas — emergent ur rotkonkurrensen om cellarean och ljuskonkurrensen om höjd, inget i koden säger att beståndet ska gallra sig.

Praktisk följd: `flora_init_plant_mass = 1,32` sattes mot jämvikten vid f1, men vid f4 är den 3,42. Skalas plantmassan med bördigheten hamnar sådden nära jämvikt direkt, och inkörningen kortas från fjortontusen tick till några tusen.

### Följ samma individ

*0099. Nödvändigt men inte tillräckligt.*

Valet av artfrände var alltid "närmaste", och närmaste granne är en **instabil referens**: två djur på snarlikt avstånd byter plats i rangordningen vid minsta rörelse.

```
fart 37,5   sträcka/sensing 7,51   grannbyte 32,8 %
fart 23,2   sträcka/sensing 4,65   grannbyte 32,9 %
```

Samplingskvoten förbättrades en tredjedel och grannbytet rörde sig inte alls. Förklaringen att grannen hann lämna synfältet mellan sensingar var alltså fel — det är måttet som är instabilt, inte perceptionen som är för gles.

Följden var att den sociala reflexen svängde mot A, sedan B, sedan A. Amplituden var stor — median 0,128 i styrkommandot, aldrig mättad av klippning — men riktningen inkoherent. Kohesionskvoten mot Poisson har legat på 1,0 genom fyra patchar.

Ändringen gör den tidigare följda individen till förstahandsval så länge den är synlig, som ett halvt steg i sorteringsnyckeln — parningsvillighet väger fortfarande tyngre än vanan. Ingen global koordination: samma celler, samma synellips, samma gather. Bara valet inom grannskapet ändras. Ballerini m.fl. 2008 visade att starar följer bestämda grannar över tid snarare än de närmaste.

```
grannbyte        32,9 %  ->   8,6 %
kohesionskvot     1,00   ->   1,03
medelavstånd      4,67   ->   4,06     (Poisson-förväntan 4,67)
```

**Stabiliteten var nödvändig men inte tillräcklig.** Målvalet är åtgärdat och flockningen uppstår ändå inte. Medelavståndet ligger något under förväntan, vilket är en antydan men inte mer.

Nästa hypotes ska alltså sökas någon annanstans än i målvalet. Kvar står aggregat över grannskapet — tyngdpunkt för kohesion, medelkurs för alignment, närmaste avstånd för separation — som är vad Reynolds faktiskt gör. Medelvärden hoppar inte, men de bär också en riktning som en enskild granne inte kan ge: mot mitten av gruppen snarare än mot en individ.

### Reynolds tre regler på grannskapet

*0100. Kohesionen svarar för första gången. Alignment gör det inte.*

En enskild granne kan bara ge riktningen mot **den grannen**. Djur som svänger mot närmaste artfrände roterar runt varandra i stället för att konvergera. Ett aggregat ger riktningen mot gruppens **tyngdpunkt**, och det är en annan storhet.

0099 gjorde målvalet stabilt — grannbyte 32,9 → 8,6 procent — utan att kohesionskvoten rörde sig från 1,0. Stabiliteten var nödvändig men inte tillräcklig.

Artfränder byggs nu som täta per-cell-fält med `bincount` över `fauna_slots`, samma mönster som `flora_cell_mass`: antal och hastighetssumma. De går genom samma gather som flora, detritus och temperatur, och roteras till kroppsram — sektorindex med fraktionell förskjutning, vektorkomponenter med egen kurs avdragen.

**Vikterna delas, inte adderas.** De tre reglerna får dela på samma totala vikt som den enda kohesionstermen hade: 0,40 för tyngdpunkt, 0,20 för medelkurs, separationen oförändrad. Flockningen ska vara en drift bland flera — föda, värme, flykt och parning verkar oförändrat — inte ta över styrningen.

```
              inom synhåll   Poisson   kvot   nn/Poisson   kursordning
soc 0,1          84,7 %       82,1 %   1,03   4,34/4,73      0,067
soc 0,9          88,9 %       83,1 %   1,07   4,06/4,66      0,065
```

**Kohesionen svarar på `sociability` för första gången.** 1,07 mot 1,03, och medelavståndet 4,06 mot Poisson-förväntan 4,66. Effekten är liten men den är den första som alls beror på egenskapen.

**Alignment ger ingenting.** Kursordningen ligger på 0,066 i båda fallen, och slumpvärdet för 185 individer är 0,074 — kurserna är helt okorrelerade. Termen är 0,20 · soc_bias · (fel/π), alltså i storleksordningen 0,08 mot en total styrsignal som domineras av föda, värme och flykt. Att höja den skulle bryta mot att flockningen inte får ta över, så nästa steg är inte mer vikt utan att förstå varför riktningen inte hålls mellan tick.

### Ett synfält, inte två

*0101. Städning av en artefakt från den stegvisa migreringen.*

Individidentifieringen ärvde strålmodellens ellips från 0085, medan världskanalerna sedan 0084 läser en cirkel via `cells_within_many`. Samma djur hade därmed **två olika synfält för artfränder**:

```
aggregatet (sektorer)   cirkel, radie 10, 314 celler
identiteten             ellips: 10 framåt, 7 åt sidan, 5,4 bakåt
```

Ett djur kunde känna att det fanns artfränder rakt bakom sig utan att kunna identifiera någon av dem, och de två kanalerna gick därför inte att jämföra. Det var inte designat — det var en rest från 0085, där ellipsen bevarades för att inte ändra beteendet, med sektorkanalerna lagda ovanpå.

Den framåtriktade biasen var motiverad för jakt och födosök, men **födosöket använder den inte längre**. Ellipsen påverkade i praktiken ingenting utom vilka artfränder som kunde identifieras, och för flockning är den aktivt skadlig: en flockkamrat färdas jämsides, och där var räckvidden kortast.

`ray_eccentricity` har därmed noll läsare och kan tas bort när strålmodellens övriga rester städas — `RaySensors.sense` och `see_agent_first_hit` är också döda sedan 0084 och 0085.

```
                   före 0101    efter
kohesionskvot          1,07      1,03
medelavstånd      4,06/4,66   4,14/4,74
kursordning           0,065     0,097     (slumpvärde 0,074)
```

Kursordningen steg över slumpvärdet för första gången. Effekten är liten och ett enskilt mått på 182 individer, men det är rätt tecken och det kom av att synfältet blev konsistent.

### Kortsiktig plan för flockningen

Alignment kan inte verka i nuvarande brusregim. `tau_dir` interpolerar mot `explore_drive` och ligger vid `dir_tau_local = 0,35` som lägst:

```
riktningsbrus per tick              0,34 rad = 19 grader
ackumulerat över 10 tick            1,07 rad = 61 grader
alignmentkorrigering per sensing    0,04 rad =  2 grader
```

Bruset är trettio gånger starkare och verkar tio gånger oftare. Kursen randomiseras helt mellan två observationer. Att höja vikten till dominans är uteslutet — flockningen ska vara en drift bland flera.

Tre åtgärder, i ordning:

1. **Sensa oftare när grannar finns.** Artfrändedetektion utlöser beredskap, `sense_alert_steps = 3` mot `sense_idle_steps = 10`. Tre gånger tätare korrigering mot samma brus. Sensing kostar energi, så avvägningen finns redan.
2. **Sänk bruset när djuret följer någon.** `tau_dir` interpolerar redan mot `explore_drive`; att låta flockning räknas som "har ett mål" är en rad och är biologiskt riktig — ett djur som följer flocken irrar inte.
3. **Låt alignment verka mellan sensingar.** Grannarnas medelkurs finns i `_soc_sectors` och skulle kunna appliceras varje tick, precis som `_nb_mem` bär position mellan observationer.

### Avståndsberoende sensing och lugnare kurs i flock

*0102. Kohesionen svarar tydligare. Alignment fortfarande inte.*

**Sensingfrekvensen blir avståndsberoende**, som ekolokalisering: långsamma ping när ingenting är i närheten, tätare ju närmare grannen kommer. Intervallet interpolerar linjärt mellan `sense_alert_steps` och `sense_idle_steps` efter grannens avstånd i förhållande till synvidden.

Två fasta steg räckte inte. Riktningsbruset verkar varje tick och ackumulerar 61 grader över tio, mot en alignmentkorrigering på två grader per sensing — kursen randomiseras helt mellan två observationer. Sensing kostar energi, så avvägningen finns redan: ett djur som ständigt har grannar nära betalar mer.

**Riktningsbruset sänks när djuret följer flocken.** `tau_dir` interpolerade redan mot `explore_drive` — låg utforskning ger rakare färd — och att följa en granne är lika mycket "har ett mål" som att stå på föda. Ett djur som följer flocken irrar inte.

```
                 0100      0101      0102
kohesionskvot    1,07      1,03      1,11    (soc 0,9)
                 1,03      —         1,08    (soc 0,1)
medelavstånd  4,06/4,66  4,14/4,74  3,95/4,73
kursordning      0,065     0,097     0,025
```

Kohesionen har nu stigit tre mätningar i rad och medelavståndet ligger sexton procent under Poisson-förväntan. Djuren håller ihop mer än slumpen — svagt, men konsekvent.

**Alignment ger fortfarande ingenting**, och kursordningen svänger mellan 0,025 och 0,097 kring slumpvärdet 0,074 utan mönster.

**Måttet är dessutom fel.** Kursordningen beräknas globalt över hela populationen. Finns flera flockar som rör sig åt olika håll blir den låg även vid perfekt lokal samordning. Det som ska mätas är **lokal** ordning — medelkursen bland grannar inom synhåll, jämförd med samma mått i en slumpblandad population. Innan det är gjort vet vi inte om alignment är trasig eller osynlig.

**Kvar att göra:** låta alignment verka mellan sensingar, och städa strålmodellens rester. `RaySensors.sense`, `see_agent_first_hit` och `ray_eccentricity` har alla noll verkliga anropare sedan 0084, 0085 och 0101 — men reservgrenarna i `_run_full_sensing` gör att en borttagning behöver verifieras med omsorg snarare än göras blint.

### Strålmodellens reservgrenar bort

*0103. Verifierat dött innan borttagning.*

`_run_full_sensing` hade två reservgrenar kvar från migreringen: `if sectors is None` föll tillbaka på `RaySensors.sense`, och `if neighbour is False` på `see_agent_first_hit`. Båda är rester från 0084 och 0085, där den gamla vägen bevarades för säkerhets skull.

Mätt över två parameteruppsättningar, 400 tick vardera:

```
_run_full_sensing anrop              23 209
RaySensors.sense                          0
RaySensors.see_agent_first_hit            0
```

Grenarna togs aldrig. Att låta dem ligga kvar dolde dessutom att metoderna var döda — de såg ut att ha anropare.

Grenarna är borttagna. Metoddefinitionerna står kvar tills vidare men har nu bevisligen ingen väg in; de kan tas i en egen omgång tillsammans med `_rebuild_cache`, strålvikterna och `sample_flora_rays`, vilket är en större operation än den här. `ray_eccentricity` är satt till noll och märkt oanvänd sedan 0101.

### Flocken som relation

*0104. Alignment ger utslag för första gången.*

"Alla inom synhåll" ger en flock som byter medlemmar varje gång någon passerar: två flockar som möts smälter omedelbart samman, och en ensam vandrare fångas av den första grupp den korsar. Ingen identitet över tid.

Medlemskapet byggs nu upp som en **affinitet** — den stiger med `flock_gain` vid varje observation och avtar med `flock_decay` mellan dem. Vid 0,25 och 0,90 krävs ungefär fyra observationer för full medlem, och en granne som försvinner tappas efter ungefär trettio. En främling som passerar får låg vikt tills den varit där ett tag.

**Kohesion och alignment behandlas olika.** Kohesionen behåller det täta per-cell-fältet — att dras mot där det finns artfränder alls är rimligt även för icke-medlemmar — medan alignment blir medlemsviktad. Det ger en naturlig asymmetri: man dras mot främlingar men följer bara sina egna.

Aggregatet i `_build_sector_percept` kan inte viktas per observatör, eftersom alla läser samma fält. Alignment räknas därför i `_acquire_neighbours`, över varje djurs egna kandidater — listan finns redan efter filtreringen, så det är ingen ny gather.

**Och måttet var fel.** Kursordningen beräknades globalt över hela populationen och blir låg även vid perfekt lokal samordning om flera flockar rör sig åt olika håll. Rätt mått är lokal ordning — medelkursen bland grannar inom synhåll — jämförd med samma mått i en slumpblandad population.

```
                 lokal kursordning   blandad kontroll   kvot
soc 0,1                0,404              0,430         0,94
soc 0,9                0,494              0,407         1,21
```

Tjugoen procents överskott vid hög sociability och inget vid låg. Medianflocken har 25 medlemmar över affinitetsgolvet. Kohesionskvoten ligger kvar på 1,07–1,08 med medelavstånd 3,79 mot Poisson-förväntan 4,68, alltså nitton procent tätare än slumpen.

Flockning finns nu i mätbar form: djuren håller ihop tätare än slumpen och samordnar kurs med dem de känner.

### Ett varmt bälte, inte två

*0105. Ett räknefel som delade världen i två.*

```python
band_lat = -cos(2π · row / h)          # en cykel: -1 -> +1 -> -1
T        = T_eq - dT_pole · |lat|^1.5   # absolutbeloppet dubblar frekvensen
```

`band_lat` gjorde **en** cykel över världen, men klimatet läser `|lat|`, och absolutbeloppet dubblar frekvensen. Resultatet blev två varma och två kalla band i stället för ett av varje — uppmätt profil över 256 rader hade period 128.

Konsekvensen är värre än estetisk. **De två kalla banden delade den beboeliga världen i två frånskilda remsor.** Djur i den ena nådde inte den andra utan att korsa en zon där termokostnaden överstiger basalmetabolismen, och populationen fragmenterades i två delbestånd — i en modell där den effektiva populationsstorleken redan mätts till fem.

Latituden är nu `(1 − cos(2π·row/h)) / 2`: noll vid rad 0, ett vid rad h/2, tillbaka till noll vid rad h. Ett sammanhängande varmt bälte och ett sammanhängande kallt, båda hela vägen runt torusen — den närmaste analogin till jordens ekvator och poler som en torus tillåter.

```
                    före        efter
median              12,2 °C     19,4 °C
andel över 15 °C     ~35 %       58 %
andel över 20 °C     ~28 %       49 %
```

Medianen steg sju grader eftersom absolutbeloppet gav dubbelt så mycket kall yta. Termokostnaden i mediancellen faller därmed omkring tjugo procent.

**All temperaturkalibrering gjord före den här patchen är oanvändbar**, och det gäller även bärkraftsmätningarna: en varmare värld med sammanhängande beboelig zon bär fler djur vid samma bördighet.

### Scenariot som fil

*0106. Utgångsläget blir data i stället för kommandorad.*

Kommandoraden hade vuxit till fjorton flaggor varav sex bara beskrev scenariot, och tre av dem — `nutrient_input`, `nutrient_init`, `detritus_init` — skalar alltid tillsammans men räknades fram för hand vid varje körning. De hann gå isär två gånger.

**Bördigheten blir ett tal.** Jämvikten är linjär i näringsflödet, så faktor 4 betyder fyra gånger alla tre. Det går inte längre att sätta dem inkonsekvent.

**Insättningen uttrycks som en princip.** `fauna.insatts_vid: jamvikt` i stället för ett tickvärde valt på känsla — det felet gjorde både p87 och p97 ogiltiga, eftersom faunan mötte en halvfärdig flora och därmed låg över bärkraften just då fastän den legat under i den färdiga världen.

**Flera grundargrupper.** `flackar: 3` ger tre skilda fläckar i stället för en. En enda fläck ger en enda linje: i p91 tog två grundarlinjer av tjugo 73 procent av avkommorna, och den effektiva populationsstorleken mättes till fem.

```yaml
namn: f4-flock
varld:   {bredd: 64, hojd: 256, bordighet: 4.0}
fauna:   {antal: 80, insatts_vid: jamvikt, flackar: 3, flackradie: 20.0, max_antal: 8192}
fysiologi: {fartskala: 0.5, sociability: 0.8, sociability_sd: 0.5}
```

Uttryckliga flaggor vinner över filen, så ett scenario kan användas som utgångsläge och enskilda tal varieras ovanpå.

`kor.sh` tar scenario, utdatakatalog och ticks. Katalogen får scenariot, commit-hashen, en `dirty.txt` med eventuella oincheckade ändringar och hela konsolutskriften — så att en körning går att förstå långt efteråt utan terminalhistorik. Att blanda ihop vilken patch en körning gjordes mot har hänt flera gånger under arbetet.

```
./kor.sh scenarios/f4-flock.yaml runs/p106 60000
```

### Jämvikten upptäcks, inte gissas

*0107. Sista gissningen ur scenariot.*

`fauna.insatts_vid: jamvikt` översattes till en konstant på 20 000 tick. Det var mätt vid bördighet 1 och 4 och därmed en gissning utanför det intervallet — jämvikten infaller olika sent vid olika bördighet, eftersom sådden ligger närmare målet ju rikare marken är men gallringen tar längre tid när plantorna är fler.

Simuleringen upptäcker den nu själv. `fauna_at < 0` betyder jämvikt: floramassan mäts var `flora_eq_window` tick, och faunan sätts in när den relativa förändringen faller under `flora_eq_tol`.

```
flora_eq_window = 1000     flora_eq_tol = 0,01     flora_eq_max_ticks = 40000
```

En procent per tusen tick. Uppmätt låg floramassan på 2,5 procent per 5 000 tick vid tick 30 000, alltså långt under tröskeln, medan den under uppbyggnaden ändras tiotals procent per fönster. Takgränsen finns för att en värld som aldrig stabiliseras inte ska hänga för evigt.

Skälet att detektera i stället för att gissa är mätt: faunan mot en halvfärdig flora mäter fel sak. Sådden ger omkring hälften av jämviktens stående gröda, produktionen skalar med den, och ett bestånd som ligger under bärkraften i den färdiga världen ligger över den i den halvfärdiga. Det gjorde både p87 och p97 ogiltiga.

**Overifierad i drift.** Sandlådan hinner inte fram till jämvikt inom sin tidsgräns; mekaniken är prövad men inte själva utlösningen. Raden `[flora] jämvikt vid tick N` i utskriften visar när den slår till, och det talet bör jämföras med de 12 000–15 000 där floraantalet planade ut i p94.

### Jämvikten kräver att både massa och antal står still

*0108. Detektorn från 0107 mätte fel storhet.*

```
tick 3000   flora 105 019   M_flora 237 156
tick 4000   flora  84 720   M_flora 236 088   -> 0,45 %, utlöste
```

Massan låg still medan **antalet föll nitton procent**. Det är gallringsfasen: många små plantor dör bort medan de överlevande växer lika mycket som de döda tappar, så massan passerar en platå mitt i förloppet. Den verkliga jämvikten låg vid 316 000 kg i p97, alltså trettio procent högre.

Djuren sattes därmed in i en värld som fortfarande krympte, och p107 blev ogiltig av samma skäl som p87 och p97 — bara med en annan orsak till feldateringen.

Antalet är monotont under gallringen och avslöjar den direkt. Kriteriet kräver nu att **båda** står still: `max(rel_massa, rel_antal) <= flora_eq_tol`.

Utskriften visar båda, så det går att se vilken som band:

```
[flora] jämvikt vid tick N: M kg i C plantor, ändring X % massa / Y % antal per 1000 tick
```

### Betning öppnar inga luckor

*Observation från p107, ännu inte åtgärdad.*

När växtligheten betas ner är det de **stora** plantorna som återhämtar sig, inte små snabbväxare som återetablerar sig. Orsaken är mekanisk.

Betningen tar skott och lämnar rot, och `_release_flora_slot` anropas bara när massan går till noll — vilket kräver att plantan saknar rot helt. Betning dödar därför praktiskt taget aldrig. Det var med avsikt: rotrefugen var det som hindrade överbetningen i p87.

Men rotanspråket på cellarean är `a_root = sra · rot · (1−struktur)`, och **roten rörs inte av betningen**. En hårt betad cell har alltså exakt samma totala anspråk som före. Ingen yta frigörs, inga luckor öppnas, och ett frö som landar där får andel noll.

Återhämtningen kan därför bara ske genom att de sittande växer tillbaka ur sin reserv. **Rotrefugen som skyddar mot överbetning blockerar också successionen.**

Följden är att det inte finns någon r-strategi att evolvera mot. Utan luckor betyder `dispersal` och `seed_mass` ingenting, och Smith–Fretwell-optimum i fröstorleken saknar mening — det finns ingen anledning att satsa på små frön om det aldrig finns någonstans att landa.

**Åtgärden i naturen:** rotunderhåll kostar kol, och kolet kommer från bladen. En avlövad växt kan inte försörja sitt rotsystem och fäller rot. Det ger tre saker på en gång: luckor öppnas efter hård betning, snabbväxare får en verklig nisch, och strukturandelen får ännu en avvägning eftersom en vedartad planta har långsam återväxt och tappar mer när den betas hårt.

Mekaniken finns till hälften: `root_alloc` sätter målandelen och massan faller vid betning, så ett målvärde som följer aktuell massa skulle få överskottsroten att fällas av sig själv.

### Häckningsfasen mot årstiden

*0109. Rättar ett tankefel i 0098.*

`breed_phase` var en **absolut fas** i årscykeln, kodad av ett fritt locus som pekade var som helst. Två djur fick därmed oberoende fönster, och grinden blev ett filter i stället för en synkronisering: om var och en är i säsong en fjärdedel av året och faserna är oberoende är sannolikheten att två är det samtidigt en sextondel — **sämre** än den asynkrona utgångspunkten den skulle förbättra.

Regressionen är mätt. p108 mot p96, med allt annat lika eller bättre — varmare värld, sammanhängande bälte, tre fläckar, flockning:

```
                p96 (halv)     p108
födslar          211–862         75
unika            291–942        155
parningar        192–364         76
topp             159–193        116
överlevnad     60 000 tick   8 000 tick
```

I naturen är häckningstiden inte en godtycklig fas utan låst till att ungarna ska födas när fodret är rikligt, och djuret läser en signal — dagslängd eller temperatur — som är **gemensam för alla i samma trakt**. Synkroniseringen uppstår då gratis: alla läser samma värld, och ingen delad genetik behövs. Två djur på samma latitud får samma säsong även om de aldrig träffats; två på olika latitud får olika, vilket är biologiskt riktigt och ger rumslig struktur på köpet.

`breed_phase` är nu en **förskjutning** mot årets temperaturtopp, i intervallet ±0,25 år. Kvar att evolvera är förskjutningens storlek: rätt läge beror på dräktighetstid och på när floran faktiskt producerar, vilket djuret inte vet i förväg.

```
                          i säsong    båda samtidigt
absolut fas                 22 %          6,0 %
förskjutning mot toppen     22 %         10,9 %
```

Nästan fördubblad överlappning vid oförändrad andel av året per individ, mätt över 200 slumpade loci vid `breed_sync = 3,0`.

### På horisonten: band som riktade relationer

`_flock` bär i dag enbart positiv affinitet, byggd av närvaro. Ett band med **tecken**, byggt av utfall i stället, skulle göra samma mekanism till både flockbildning och predatorundvikande — och den skiljer då inte på art utan på erfarenhet.

Två skäl utöver realismen. Det är observerbart per definition och **kan slå fel**, vilket är vad startpromptens tredje mål efterlyser: riskvärderingen läser i dag `pheno.predation`, alltså motpartens arvsmassa.

Och aversionen bör kunna uppstå av att se **andra** angripas, inte bara av egen erfarenhet. Ett band byggt på egen erfarenhet lärs bara av dem som överlever mötet; den som dödades hann aldrig uppdatera något. Att observera flyttar informationen dit den behövs — den som ser attacken lever fortfarande. Bandet ska riktas mot angriparen, vilket kräver att observatören identifierar båda.

Det ger också ett selektionstryck mot att jaga i en flocks åsyn: ju fler som ser, desto fler undviker. Jaktbeteende som uppstår ur bandmekaniken i stället för att kodas.

### Social synkronisering av häckningsfasen

*0110. Flocken blir en reproduktiv enhet.*

Spridningen i `breed_phase` är ±0,19 år mot ett fönster på 2,7 månader vid skärpa 3 — **spridningen är större än fönstret**, och därför återstår bara halva överlappningen: 10,9 procent mot de 22 som identiska faser skulle ge. Att stänga glappet genom smalare spridning eller bredare fönster är kalibrering.

Social synkronisering av brunst är väldokumenterad hos får, getter, möss och flera flockdjur: kemisk signalering mellan individer som ofta är nära drar gruppens honor mot en gemensam cykel. **Affinitetsmatrisen är precis den viktning det ska ske över** — ett djur drar sin fas mot dem det verkligen umgås med, inte mot vem som helst som passerar.

`_T_BREED_PULL` kodar hur stor andel av avståndet till flockens fas som stängs per observation, 0 till 0,20. Locus behåller individens anlag; det som styr grinden är anlaget plus dragningen. Faser är cirkulära och medelvärdesbildas som vektorer, med kortaste vägen runt.

Konvergens vid 30 djur, full affinitet och dragning 0,10:

```
utgångsläge (±0,19 år)   R = 0,78
efter  1 observation     R = 0,79
efter  5                 R = 0,91
efter 10                 R = 0,97
efter 20                 R = 1,00
```

Tjugo observationer räcker för fullständig samling. Vid avståndsberoende sensing kring var åttonde tick är det omkring 160 tick, alltså drygt tre månader — snabbare än ett år, så synkroniseringen hinner verka innan första säsongen.

**Tre saker kalibrering inte kan ge.** Synkroniseringen blir **lokal**, så varje flock konvergerar mot sin egen fas — två flockar i olika delar av världen får därmed reproduktiv isolering, vilket är begynnande artbildning ur en mekanism som inte kodar den. Slingan är **självförstärkande**, eftersom de som synkroniserar får fler parningar och avkomman ärver dragningen. Och **avvägningen finns inbyggd**: stark dragning betyder att man följer flocken även när dess fas är dålig för den egna konditionen, och en ensam individ med hög dragning har ingen att dra mot.

Parbildningen faller ut som konsekvens i stället för som egen mekanism: flocken blir en reproduktiv enhet, inte bara en rumslig.

`n_traits` 40 → 41.

### Grundargrupper med egen tyngdpunkt

*0111. Raser vid introduktion.*

Varje grundares genom slumpades oberoende, så tre fläckar innehöll tre stickprov ur samma fördelning: geografiskt åtskilda men **genetiskt identiska**. Med en egen tyngdpunkt per grupp och liten spridning inom den får man verkliga raser redan vid introduktionen — en grupp kan ha starkt flockbeteende medan en annan saknar det.

```yaml
fauna:
  flackar: 3
  grupp_avstand: 1.0      # mellan tyngdpunkterna, i logit-enheter
  grupp_spridning: 0.4    # skala på spridningen inom gruppen
```

Uppmätt vid de värdena, 60 grundare i tre grupper:

```
avstånd inom grupp     3,96
avstånd mellan grupper 8,48
kvot                   2,14
```

**Kvoten avgör om det blir raser eller arter**, och det är just den som är intressant att variera: den säger när grupperna slutar utbyta gener.

Kombinerat med den sociala fassynkroniseringen i 0110 blir isoleringen **dubbel** — grupperna skiljs både genetiskt och reproduktivt, eftersom varje flock drar mot sin egen säsong. Det är ungefär så artbildning faktiskt börjar, och ingen av mekanismerna kodar den.

Scenariofilen beskriver därmed en **fauna** i stället för bara ett antal.

### Polerna vid kanterna

*0112. Kall zon överst och nederst, kontinent emellan.*

`(1 − cos(2π·row/h))/2` gav varmt vid rad 0 och rad h och kallt i mitten — alltså en kall zon tvärs över kartans mitt. Formen är nu `(1 + cos)/2`: kallt vid kanterna, varmast vid rad h/2.

Torusen gör rad 0 och rad h−1 till grannar, så de två kalla halvorna möts och bildar **ett** smalt polband. Det visuella intrycket blir två poler med en kontinent emellan.

```
rad     0   32   64   96  128  160  192  224
temp    0    6   19   28   30   28   19    6
```

Medianen är oförändrad 19,4 grader — det är samma profil, bara förskjuten en halv cykel.

### Profilering inför optimering

Flora ensam, f4-bördighet, 292 459 plantor:

```
                          ms/tick   andel   us/planta
totalt                      298,7             1,021
_step_world_and_flora       222,6   74,5 %    0,761
  _growth_system_flora      166,1   55,6 %    0,568
  _rebuild_flora_summary     45,2   15,1 %    0,155
  _dispersal_system_flora    28,8    9,7 %    0,099
```

Tillväxtpasset är över hälften av takten. Det är redan vektoriserat, så vinsten ligger i allokeringar snarare än algoritm: varje temporär mellanprodukt på 300 000 element kostar en full genomgång av minnet.

`_rebuild_flora_summary` bygger om per-cell-cachen varje tick och bör kunna använda samma glesningstrick som `detritus_active`.

### Profilering av florapasset

*0113. En liten säker vinst, och en tydlig karta över resten.*

```
                          ms/tick   andel   us/planta
totalt                      298,7             1,021
  _growth_system_flora      166,1   55,6 %    0,568
  _rebuild_flora_summary     45,2   15,1 %    0,155
  _dispersal_system_flora    28,8    9,7 %    0,099
```

cProfile inuti passet, 292 459 plantor:

```
_growth_system_flora   egen tid 125 ms/tick
astype                 2 854 anrop / 30 tick = 95 per tick
partition                240 anrop = 8 per tick
```

**Tillväxtpasset gör 29 `astype`-anrop.** Store:n är float32 och räkningen sker i float64, så varje läsning kopierar hela arrayen — 2,4 MB per anrop vid 300 000 plantor, alltså omkring 70 MB allokering per tick. Det är minnesbandbredd, inte aritmetik.

`stored_left.astype(np.float64)` förekom fyra gånger på samma array. Hissad till en variabel ger det identiskt resultat och sparar tre fulla genomgångar.

```
1,021 -> 0,983 us/planta      hissad konvertering
0,983 -> 0,823               kvantilerna gatade
                             tillsammans 20 procent
```

**Kvantilerna var sju procent för statistik ingen läste.** `_rebuild_flora_summary` beräknade sex percentiler varje tick — sex fulla partitioneringar av hela floravektorn — men de konsumeras bara av `_emit_world`, som skrivs vid loggintervall. De räknas nu bara när posten faktiskt skrivs. Summorna och medelvärdena är enkla genomgångar och kostar en bråkdel; det är sorteringen som är dyr.

**Tjugo procent, inte hälften.** De återstående vägarna är större och behöver var sin verifiering:

- **Räkna i float32 genomgående.** Alla 29 konverteringar försvinner och varje temporär halveras. Men float32 ger sju siffrors precision, och ledgerns drift ska hållas under 1e-9 relativt — det kräver att summeringarna hålls kvar i float64 och att felfortplantningen mäts, inte antas.
- **`_rebuild_flora_summary` på 15 procent** bygger om per-cell-cachen varje tick. Samma glesningstrick som `detritus_active` använder borde gå att tillämpa: bara de celler som faktiskt ändrats.
- **`np.partition`, åtta anrop per tick** — det är `median` eller `percentile` någonstans i passet. Åtta fulla partitioneringar av 300 000 element är 0,68 sekunder av 30 tick, alltså sju procent, för statistik som kanske bara behövs vid rapportering.

Den sista är troligen den billigaste riktiga vinsten: om partitioneringen bara behövs när diagnostik efterfrågas är det sju procent för ingenting.

### Tillväxtpasset som kompilerad kärna

*0114. Skal, kärna och efterspel.*

Avsnittsprofilen inuti passet visade att det inte finns någon enskild dyr del att angripa. Vid 310 240 plantor:

```
passet totalt   113,8 ms/tick   54,3 % av takten
  1 förnafall     16,5   14,5 %      5 allokering    5,2    4,6 %
  2 dödlighet     11,4   10,0 %      6 ljus         21,6   19,0 %
  3 anspråk       12,6   11,0 %      7 tillväxt     29,8   26,1 %
  4 inkomst       13,5   11,8 %      0 uppsättning   3,3    2,9 %
```

**Fördelningen är platt**, så en partiell konvertering hade gett lite. Hela passets aritmetik ligger nu i `flora_growth.growth_kernel`, en njit-kärna som bara tar arrayer och skalärer. `Population._growth_system_flora` är en dispatcher; `_growth_system_flora_numpy` är kvar som referens och väljs med `PopParams.flora_growth_backend` eller `--flora-growth numpy`.

**Två världsanrop måste ligga utanför, men bara det ena kunde skjutas upp.** `world.temperature_of_cells()` hissas till skalet. `world.excrete_cells()` flyttar till efterspelet, vilket är säkert eftersom passet aldrig läser `detritus` — bara `nutrient`. Näringsåterföringen från döende plantor kan däremot **inte** vänta: `_release_flora_slot` gör den i avsnitt 2 och avsnitt 3 läser `nutrient`, så grannarna ska hinna ta upp den samma tick. Kärnan gör den därför på plats, och `_release_flora_slot` har fått `return_nutrient=False` för bokföringen efteråt.

**Reduktionerna ackumuleras i radordning.** `np.bincount` summerar i indataordning, så en utströdd addition i samma ordning ger samma avrundning. Det är därför anspråk, upptag och bladarea kan flytta in i kärnan utan att sista biten flyttar med.

Två fällor som bara syns i en elementvis jämförelse: strukturandelen används **klippt** i näringsinnehåll, omsättning och livslängd men **oklippt** i areorna och energitätheten, och rotmassan går via en float32-rundtur i store:n mellan avsnitt 1 och 3.

**Verifieringen är två identiska världar med var sin väg**, jämförda fält för fält över hela slot- och cellrymden efter varje tick. `run_headless.py --verify-flora-growth N`. Att invariantsviten går igenom säger för lite: den skulle godkänna ett pass som fördelade upptaget fel mellan plantor så länge summan stämde.

```
400 tick, 64x64 med fauna, ingen strukturell avvikelse
  mass, energy, flora_root_mass, detritus, detritus_structure,
  flora_cell_claimed, flora_claim_share, alive, cell_idx      0
  flora_reserve, flora_repro_pool, flora_carbon_pool, nutrient
                                     ~1e-18 abs, ~2e-16 mot faltets skala
```

Massan är alltså bitidentisk. Kvar står float64-ackumulatorerna på sista bitens nivå — `exp` och `pow` skiljer sig mellan numpys vektoriserade och libms skalära variant, och bitidentitet var aldrig möjlig.

Uppmätt i samma process med interfolierade varv, 316 503 plantor:

```
                 passet          hel tick        us/planta
numpy           127,7 ms         261,1 ms          0,825
numba            60,4 ms         174,9 ms          0,552
kvot              2,12x            1,49x
```

**Kärnan är nu bara hälften av det som är kvar** i passet. Resten är skalets minnestrafik: gathers 8,4 ms, `excrete_cells` 3,6, scatters och `set_flora_claims` omkring 7, `rng.random` 0,8.

Två mätvärden inför `prange`: att allokera åtta n-långa arbetsvektorer i kärnan kostar mätbart noll, och att fylla dem 0,69 ms — bandbredden är alltså fortfarande inte flaskhalsen, vilket stämmer med float32-utfallet i 0113. Däremot är `40,0**s` i livslängden 4,7 ms av kärnans 30, eftersom `pow` är 2,4 gånger dyrare än `exp` utan SVML. Att skriva den som `exp(s·ln 40)` sparar omkring tre millisekunder men bryter den elementvisa jämförelsen, och är därför inte gjord.

### Efterhandsanalysen mot dagens loggar

*0115. `genopheno_analyze.py` skrivet om.*

Skriptet hade drivit ifrån både loggformatet och frågan. Tre fel gjorde det missvisande snarare än bara inaktuellt.

**Halva filen läste ett event som inte skrivs.** All per-agent-statistik byggdes ur `event: "step"`. `steps_life: 0` stod i rapporten och samtliga härledda fält var tysta NaN.

**Dödsorsaken lästes aldrig** — `cause` från Steg 5d, och `straightness` från Steg 5h som bär ett av Del E:s måltal.

**Selektionsdifferentialen rankades i blandade enheter.** `reserve_cap` överst med −78 021 och `mobility` med +0,037 säger bara att den ena mäts i joule. Den är nu i spridningsenheter, och bredvid står en permutationsbaserad brusnivå: med 41 loci och 211 individer är en topplista utan nollhypotes garanterat brus.

**Nollhypotesen finns dessutom i genomet.** Floraloci ärvs och muterar i faunans genom men har ingen läsare där, så de mäter vad urvalet ger av sig självt. Mot p114:

```
locus        differential   brusnivå   över brus
m_target           +0,383      0,168      ja
metab              -0,202      0,164      ja
hidden_2           -0,150      0,173
structure  flora   -0,115      0,164
uptake     flora   -0,107      0,168
```

Två faunaloci över bruset, noll av tretton floraloci. Det är första gången faunans selektion är avgränsad mot brus i stället för rapporterad som en topplista.

Rapporten är nu ett avsnitt per fråga: dödsorsaker över tid, energibudget per djur med kadaverandel, härstamning till grundargrupp, selektion per locus, rörelse över livstiden. `--run runs/pNNN` läser katalogen och skriver rapporten bredvid loggarna. 887 rader blir 426.

### Vägarna mätta mot varandra

*0116. `--bench-flora-growth`, och två rader i `kor.sh`.*

En körnings `ms/tick` går inte att jämföra med en annan. Beståndet skiljer sig, maskinen skiljer sig, och tickens fasta del — världspassen över alla celler — gör att `us/planta` stiger när floran krymper. p114 slutade på 0,31 us/planta mot 0,823 före patchen, men vid 91 209 plantor mot 292 459, så talen mäter inte samma sak.

Verktyget mäter i stället två vägar **i samma process med interfolierade varv**, under `pop.step()` och inte genom att anropa passet i en loop: upprepade anrop utan spridning låter floran krympa mellan mätpunkterna och mäter ett bestånd som inte finns. Passets egen tid tas med `perf_counter`, alltså samma oförvrängda metod som `--pass-timing`.

```
python run_headless.py --scenario scenarios/f4-flock.yaml \
    --bench-flora-growth 17000 --bench-rounds 5 --bench-ticks 8
```

Uppvärmningen bär två saker som inte hör till den stationära kostnaden: kompileringen vid första anropet, och att floran ska hinna till det bestånd man vill mäta vid. Den bokfördes först på den väg som råkade vara vald och gav 1 286 procents passandel innan nollställningen kom på plats.

Vägarna räknas upp ur `flora_growth.available_backends()`, så en tillkommande kärna syns i mätningen utan att verktyget rörs. `PopParams.flora_growth_backend` bär nu en sträng i stället för en boolesk flagga.

**Två rader i `kor.sh`.** `--world-log` skrivs — utan den går florans selektion inte att mäta i efterhand, och det är Steg 4:s hela fråga. Och `NUMBA_NUM_THREADS=1` är borttagen ur miljöprefixet: så länge kärnan bara fanns i njit-form var en tråd rätt, men en hårdkodad etta framför en parallelliserad kärna är samma sak som att mäta parallelliseringen med den avstängd. BLAS-trådarna står kvar på ett.

### Tillväxtkärnan parallelliserad

*0117. `prange` på samma källa.*

**En källa, två kompilat.** `_growth_kernel_impl` dekoreras två gånger, med och utan `parallel=True`. `prange` beter sig som `range` när parallelliseringen är av, så den seriella varianten är oförändrad i allt utom formen och de två kan inte glida isär. Vägen väljs med `--flora-growth parallel`.

**Ingen `prange` bär en reduktionsvariabel.** Varje parallell loop skriver bara per-planta-fält; allt som ackumulerar — summor, räknare och de utströdda additionerna till cellfälten — ligger i korta seriella svep emellan. Det kostar några extra genomlöpningar men ger två saker som är värda mer. Reduktionerna behåller sin ordning, så `np.bincount`-avrundningen följer med och den elementvisa jämförelsen står kvar. Och resultatet är bitidentiskt oavsett trådantal, vilket följer av konstruktionen och inte av tur: parallella loopar rör bara egna index, seriella svep har fast ordning.

Tre omflyttningar krävdes. Näringsåterföringen från döende plantor ligger nu i det seriella svepet efter dödlighetsloopen — den måste ändå ske före anspråken, och en utströdd addition till `nutrient` går inte att parallellisera. Bladarean per cell och den skuggvägda arean strös ut seriellt efter var sin parallell loop. Och tillväxten skriver `dm_out` och en flagga i stället för att summera på plats, så att `produced`, `taken` och andelen ljusbegränsade räknas i slotordning efteråt.

**Parallellt:** dödlighet med `exp` och areorna, inkomsten, allokeringen med skuggans `exp`, samt ljus och tillväxt. **Seriellt:** räknarna, radbygget, tre utströdda additioner och slutsummorna. Alla transcendentaler ligger i den parallella delen.

Verifierat elementvis mot numpy-vägen, 200 tick med fauna:

```
mass, energy, flora_root_mass, detritus, anspråken     0
flora_reserve, flora_repro_pool, nutrient              ~2e-16 mot fältets skala
```

Alltså samma nivå som den seriella kärnan, och ingen strukturell avvikelse.

**Farten är inte mätt.** Sandlådan har en kärna, och där kostar parallellversionen 8 procent extra i ren trådhantering — vilket är precis vad man ska se. Talet måste tas med `--bench-flora-growth` på tolvkärningsmaskinen. Räkna inte med tolv gånger: de seriella svepen är fortfarande O(n), och kärnan var redan bara hälften av passets tid efter 0114. Amdahl gäller på båda nivåerna.

### Mätordningen, och vad 0117 faktiskt gav

*0118. Alternerad ordning i `--bench-flora-growth`.*

Uppmätt på referensmaskinen vid 102 205 plantor, alltså vid det bestånd en riktig körning står i:

```
väg          passet   hel tick   kvot mot numpy
numpy        12,93      41,55
numba        10,20      37,71     1,27x
parallel     10,31      37,87     1,25x
```

**0117 gav ingenting.** Parallellversionen är en procent långsammare än den seriella på tolv kärnor. Orsaken är designvalet i 0117 självt: varje reduktion gjordes seriell för att bevara `bincount`-ordningen, och det lämnade fyra parallella loopar mot åtta seriella O(n)-svep. Amdahl med en seriell andel över hälften ger max omkring 1,6 gånger på kärnan, kärnan är halva passet, och trådstarten äter resten. Den räkningen gick att göra i förväg och gjordes inte.

**Och njit gav 1,27x, inte 2,12x som i utvecklingsmiljön.** Per planta: 0,127 mot 0,100 µs på referensmaskinen, 0,403 mot 0,191 i sandlådan. Maskinen är 3,2 gånger snabbare på numpy-vägen men bara 1,9 på kärnan, eftersom numpys elementvisa operationer är SIMD-vektoriserade medan kärnan är skalär — grenarna på `if holds[i]` hindrar autovektorisering. Ju bättre CPU, desto mindre vinst av att kompilera bort temporärerna. **Kvoter mellan njit och numpy bär alltså inte mellan maskiner**, bara mellan vägar på samma maskin.

Passet är efter allt detta 27 procent av takten i stället för 55, och 80 000 tick tar femtio minuter. Prestandan binder inte längre på 64x256.

**Mätfelet:** vägarna kördes i samma ordning varje varv medan floran drev monotont, tre procent under en mätning. Det gynnade den som kördes sist med ungefär en procent — samma storleksordning som skillnaden mellan njit och prange. Ordningen vänds nu varannat varv, och en drift över fem procent skrivs ut som varning.

### Parallelliseringen tillbakadragen

*0119. `prange` bort, kärnans form kvar.*

Två fel låg ovanpå varandra och tog tre mätomgångar att skala av.

**Kärnan kompilerades aldrig parallellt.** Numbas diskcache slår upp på funktionens `__qualname__` och signatur, inte på kompileringsflaggorna. `growth_kernel` och `growth_kernel_par` byggdes av samma funktionsobjekt med `cache=True`, hamnade i samma cachepost, och den som kompilerades sist läste in den förstas objektkod. Tyst, utan varning, med `parallel_diagnostics()` tom. Det förklarade varför `NUMBA_NUM_THREADS=1` och tolv trådar gav identiska tal.

**Med kompileringen rättad blev den tio gånger långsammare.** Vid 102 205 plantor:

```
trådar     passet    kvot mot njit
     1     10,2 ms          —
     2      8,8 ms       1,16x
    12    104,5 ms       0,10x
```

Orsaken är kärnans form. Den varvar tolv parallella regioner med åtta seriella O(n)-svep, eftersom reduktionerna hölls seriella för att bevara `bincount`-ordningen. Numbas trådlager parkerar inte trådarna mellan regionerna utan låter dem snurra, så elva trådar bränner CPU under varje seriellt svep medan den tolfte gör arbetet. Ju fler kärnor, desto värre.

**Bästa uppmätta utfall var 1,16x på passet vid två trådar**, alltså fyra procent av ticken — mot att bära en andra kodväg, en cachefiness som redan lurat oss en gång, och en patologi som gör körningen tre gånger långsammare om trådantalet råkar vara maskinens. Vägen är därför borta. `available_backends()` ger två vägar igen.

**Kärnans form är behållen.** Uppdelningen i räknande loopar och separata reduktionssvep kom till för parallelliseringen men står på egna meriter: reduktionerna har en fast ordning, så `bincount`-avrundningen följer med och den elementvisa jämförelsen mot numpy-vägen står kvar. Verifierad oförändrad efter reverten.

**Lärdomen inför nästa parallellisering:** en kärna som varvar parallellt och seriellt arbete är fel form för `prange`. Ska hydro, transporten eller sensingen parallelliseras ska passet vara parallellt hela vägen igenom — reduktioner via per-trådsackumulatorer, inte via seriella mellanled. Det är ett krav på passets utformning, inte ett val vid kompileringen. Och `cache=True` på två dispatchers ur samma funktionsobjekt är alltid fel.

### Världsposten byggdes varje tick och kastades

*0120. `hub.wants()` före nyttolasten.*

Passtidtagning vid 256x256 med noll djur visade `_finalize_store_and_emit` på 36 procent av takten. Nedbrytningen gav en post som inte borde funnits alls:

```
growth_flora      62,99 ms   34,0 %
rebuild_index     45,24 ms   24,4 %
emit_world        44,48 ms   24,0 %     <- med loggning avstängd
dispersal_flora   26,86 ms   14,5 %
world.step         5,58 ms    3,0 %
```

`_emit_world` byggde en full florasammanfattning **med kvantiler**, plus två svep över cell- och floravektorerna för näringskretsloppet, och lämnade den till `_emit` — som kastade allt när `self.hub is None`. Hela avgörandet låg i mottagaren. Och även med en logg kopplad skriver `WorldLogger` i sin egen kadens, som standard var annan simulerad sekund, alltså vart hundrade tick.

Värre: 0113 gatade percentilerna bakom `_flora_want_quantiles` för att de bara lästes vid loggning. `_emit_world` satte flaggan igen varje tick och tog tillbaka det mesta av den vinsten.

**Producenten frågar nu först.** `BaseObserver.wants(name, t)` är en sidoeffektfri variant av `allow()` — den flyttar inte fram nästa tillåtna tidpunkt, den svarar bara på om posten skulle skrivas. `EventHub.wants()` frågar alla observatörer, och `_emit_world` returnerar direkt om svaret är nej. Skillnaden mot `allow()` är hela poängen: `allow()` är fortfarande den som avgör när posten väl kommer.

Uppmätt:

```
256x256, noll djur, ingen loggning    188,3 -> 140,8 ms/tick   -25 %
64x64 med fauna och full loggning      13,9 ->  11,5 ms/tick   -17 %
```

Simuleringens utfall är oförändrat i båda fallen, och världsloggen skrivs med samma kadens och samma 56 fält som förut.

Mönstret gäller fler poster än den här. `_emit_population` är billig i dag men bygger också ovillkorligt, och samma fråga bör ställas där när den växer.

### Kadaver skilt från förna

*0121. Egen pool, egen strukturandel, egen nedbrytningstakt.*

Kroppar hälldes i `detritus`. Eftersom strukturandelen blandas massviktat drunknade ett kadaver vid 0,25 i cellens förna vid 0,83 och kom ut kring 0,80. Tre följder på en gång, alla uppmätta i p114.

**Asätarnischen kunde inte löna sig.** En ren asätare fick 0,087 ur detritus mot betarens 0,258 ur flora — en enkelriktad nackdel, inte en avvägning, och därmed ett brott mot regeln att varje trait ska ha motverkande konsekvenser. `_T_DIET` kunde straffa men aldrig belöna.

**Dödsspiralen hade ett skafferi.** Faunan åt 4 594 kg kadaver mot 3 248 kg flora över hela sin livstid, med andelen stigande monotont från 20 till 86 procent genom kollapsen. Kadaver är en stock: varje dödsfall matar de överlevande en gång till, vilket fördröjer kollapsen och gör den brantare när stocken tar slut.

**Och `M_detritus` mätte två saker.**

Efter delningen har dietaxeln en verklig brytpunkt:

```
diet    flora   kadaver   förna     bäst
0,00    0,258     —         —       flora
0,25    0,211   0,202     0,033     flora
0,50    0,159   0,329     0,053     kadaver
1,00      —     0,534     0,087     kadaver
```

En ren asätare får alltså **0,534 per kilo mot betarens 0,258** — högre kvalitet, men mot en resurs som är fläckvis, kortlivad och beror av andras död. Det är avvägningen som saknades.

**Tre designval.** Förnan är fortfarande ätbar; felet var aldrig att den var det utan att kadavret drunknade i den. Kadaver bryts ned åtta gånger snabbare, `carcass_decay = 0,62` mot förnans 0,077 — halveringstid knappt en månad i stället för nio, vilket är mekanismen som tar bort skafferiet. Och kadaver diffunderar inte; spridningen i `add_carcass` är deponering, en kropp som skingras där den faller.

Perceptionen skiljer inte på poolerna. Ett djur ser att här ligger något dött; vad det är värt visar sig när det äter. En egen sinneskanal vore en större ändring än defekten motiverar.

**Maskineriet är gemensamt, inte duplicerat.** `_pool_add`, `_pool_deactivate_if_empty` och `_decompose_pool` tar poolens arrayer som argument; `_detritus_*` och `_carcass_*` är tunna omslag. Poolerna ligger inte i en klass eftersom `detritus` och `detritus_structure` läses som attribut på `World` från ett dussin ställen.

Näringsledgern räknar båda poolerna. Missas den ena syns det som en läcka i takt med dödligheten, vilket är den storleksordning som är svårast att skilja från en verklig läcka. Uppmätt drift efter delningen: −7,7e-10 relativt över 1 500 tick.

### Förna och kadaver var för sig i loggen

*0122. Två pooler, två kurvor.*

`M_C` blev summan av båda dödpoolerna i 0121, vilket stänger ledgern men döljer just det som är intressant: poolerna har olika strukturandel, olika nedbrytningstakt och olika roll i födovalet. En gemensam kurva kan inte skilja ett bestånd som betar från ett som lever på sina egna döda.

Världsposten bär nu `M_detritus` och `M_carcass` var för sig, plus `nutrient_in_litter` och `nutrient_in_carcass` samt cellräknare för båda. `M_C` och `nutrient_in_detritus` är kvar som totalsiffror så att summan fri + flora + fauna + detritus stänger som förut för läsare som inte känner till delningen.

`live_world_plot.py` ritar kadavret som egen kurva i massapanelen och som eget skikt i näringsstacken. Skiktet är tunt — vid 1 200 tick står kadavret på 0,10 kg mot förnans 85 152 — men det är det som växer under en kollaps.

**Äldre loggar fungerar oförändrat.** Saknas de nya fälten blir kadaverserien NaN och ritas inte, medan förnan faller tillbaka på `M_C`, alltså exakt den kurva som ritades förut. Verifierat mot en logg med fälten borttagna.

### Spatialindexet gjorde allt två gånger

*0123. Delat efter kadensbehov, plus en smutsflagga.*

Efter 0120 var indexbygget den största posten vid 256x256: 45 ms av 141, alltså trettiotvå procent, och det byggdes två gånger per tick.

**Bygget gör tre saker med olika kadensbehov.** `id -> slot` och CSR-layouten ändras när någon föds, dör eller byter cell. De härledda florafälten `flora_cell_mass` och `flora_cell_structure` ändras varje tick, eftersom florans massa gör det. Båda anropen gjorde allt tre.

Ett tidigare försök var att bara ta bort det andra anropet. Simuleringens utfall blev bitidentiskt över 2 000 tick, men invariantsviten flaggade `[spatial_index] slotar i fel cellbucket`: bygget är redundant för simuleringen men inte för invarianten att indexet stämmer vid tickgränsen. Sviten gjorde sitt jobb.

En ren smutsflagga hjälpte inte heller, eftersom florans massa ändras varje tick och flaggan därmed alltid hade varit satt.

**Delningen är efter vad som faktiskt ändras.** Andra anropet, efter faunapassen, tar `with_flora_fields=False`: där kan bara medlemskapet ha ändrats, och betningen håller de härledda fälten uppdaterade inkrementellt medan den äter. Och `_index_dirty` sätts bara av `alloc_slot`, `clear_slot` och av rörelse som **verkligen byter cell** — ett djur som rör sig inom sin cell ändrar ingenting, och det är det vanliga fallet eftersom en tick sällan räcker över en cellbredd. Har ingen flyttat blir andra anropet en ren no-op.

```
256x256, noll djur      140,8 -> 94,6 ms/tick   -33 %
64x64 med fauna          10,7 ->  9,4 ms/tick   -12 %
```

Vid 256x256 utan fauna är `_finalize_store_and_emit` nu helt borta ur profilen: 99,7 procent av takten ligger i `_step_world_and_flora`.

**Utfallet är oförändrat.** Tio rapportpunkter över 2 000 tick med fauna ger identiska tal för bestånd, massa och detritus — bara millisekunderna skiljer.

Kvar i indexet är att andra anropet fortfarande sorterar om hela beståndet för att flytta hundra djur av trehundratusen. En inkrementell uppdatering av CSR är rätt operation men en riktig ändring i layouten, och den betalar sig först när fauna är många.

### Nedbetade plantor kunde aldrig växa igen

*0124. Meristemrefug och rotens återgång.*

Betningshorisonten tog `edible = m - rotmassa`. Eftersom tuggan normalt är större än medianplantans skott hamnade plantan på **exakt** skott = 0 — och då är bladarean noll, ljuset noll och `can_grow` falskt. Plantan kunde aldrig fotosyntetisera igen.

Uppmätt i en betad värld, 64x64:

```
tick 1500   87,5 % av floran hade skott = 0
tick 2500   82,9 %, och de höll 66 % av all rotmassa
```

2 693 sådana plantor följdes i 1 000 tick. **Inte en enda fick tillbaka ett skott.** De krympte monotont, 0,571 till 0,341 kg, och dog långsamt av svält medan de höll två tredjedelar av cellernas näringsanspråk och blockerade all nyrekrytering. Kodkommentaren vid betningen sa redan att plantan *"skjuter igen ur sin reserv"*. Koden gjorde inte det.

**Refugen måste mätas mot roten, inte mot skottet.** Första försöket lämnade en andel av skottet. Det räckte inte: andelen krymper geometriskt vid upprepade passager och rundas till noll i float32, och skott = 0 föll bara från 82,5 till 67,2 procent. Mot rotmassan är golvet stabilt och representerbart. `edible = m - rot·(1 + meristem)`.

**Rotens återgång** fäller överskjutande rot mot plantans egen `flora_root_alloc`. Utan den behåller en nedbetad planta hela sitt anspråk, eftersom anspråket räknas ur rotmassan och betningen inte rör roten. Den exakta återställningen är `(rot - rho·m)/(1 - rho)`, vilket för en hårt betad planta är merparten av den, så takten begränsar steget. Axeln får därmed en motverkande konsekvens: mer rot ger större anspråk men mer att fälla när betet kommer.

Effekterna separerade, 2 000 tick:

```
meristem  återgång   flora   massa   fauna   skott=0   rotandel
    0,00      0,00    6359    5769      22    82,5 %      0,847
    0,10      0,00    8920   11268      26     6,6 %      0,810
    0,00      0,50    1208    1761      12    12,5 %      0,548
    0,10      0,10    5265    3149      27     0,1 %      0,661
    0,10      0,50    1500    1177      20     0,0 %      0,587
```

Refugen ensam tar bort fällan och ger både mer flora och mer fauna. Återgången kostar floramassa — den fäller rot till förna — men sänker rotandelen från 0,85 till 0,66, vilket är det som faktiskt öppnar luckor, och ger flest djur. Vid 0,50 är den för hård. Valda värden: `flora_meristem_frac = 0,10`, `flora_root_dieback = 0,1`.

Elementvis jämförelse mellan numpy-vägen och kärnan är oförändrad: massa och rotmassa exakt lika, ackumulatorerna kring 1e-16.

### Rapportraden blandade massa och näring

*0125. Märkta grupper, kadaverpoolen synlig, rimlig precision.*

Raden såg ut så här:

```
detritus=325549.1440  fri_när=11423.54244
```

Två tal i samma enhet, bredvid varandra, som mäter olika saker: det första är kilo torrsubstans, det andra kilo näring. Att förnan väger hundra gånger mer än den bär i näring är dessutom hela poängen med strukturandelen — förnan är kol, inte kväve. Vid strukturandel 0,86 bär den 0,0065 kg näring per kilo.

Grupperna är nu märkta `M(kg)` och `N(kg)`, precisionen nedskuren från fyra decimaler på tal i tusental, och kadaverpoolen har fått plats fast den varit egen sedan 0121. Rotandelen står också med — den kom med 0124 och säger hur stor del av beståndet som är anspråk utan bladverk.

```
tick 2000  t=  40.0  fauna=  18  flora=  6481  rot=0.83
M(kg) fauna=  13.8 flora=5.925e+03 förna=7.375e+04 kadaver=  1.81
N(kg) fri=     253 flora=    326 förna=    263
föd=   9  död=  11  dräkt=  2/ 50%  19.23 ms/tick
```

Sammanfattningen delar näringen i förna och kadaver och skriver ut andelarna, eftersom fördelningen mellan fri, levande och död är det som säger något om systemets tillstånd — och det var den frågan p124 väckte.

`diagnostics()` får `carcass_mass_kg` och `flora_root_frac`; `nutrient_balance()` får `in_litter` och `in_carcass`. `in_detritus` står kvar som summa för läsare som inte känner till delningen.

### Ensamhet nådde aldrig rörelsens regimval

*0126. Söktillstånd när djuret är parningsberett och inte ser någon.*

`_integrate_motion` interpolerar redan `tau_dir` mellan kringgående sök och rak färd, styrt av `explore_drive`. Mekanismen fanns alltså. Det som saknades var att ensamhet aldrig nådde den.

Reflexkedjan hade fyra grenar — fly, jaga, gå till partner, flocka — och ingen för *parningsberedd utan någon i sikte*. Det fallet föll igenom till födostyrningen, som **sänker** utforskningen när organismen står på föda den vill ha:

```
explore_drive *= 1.0 - hunger_now * food_local
```

Ett mättat djur ensamt på full betesmark fick alltså minimal utforskning, kortaste persistenstid och en bana som slingrade på fläcken. Precis det djur som borde färdas stod stilla.

**p125 frö 2 är fallet renodlat.** Från tick 20 000 dog inte ett enda djur av svält på 8 000 tick. Floran låg på 92 000 plantor, mer än i någon av de överlevande körningarna. De tio som fanns kvar fick fyra ungar och dog av ålder med median-D på 0,999 — och medianavståndet till närmaste artfrände var 16 cellbredder mot en synradie på 7. De svalt inte. De hittade aldrig varandra.

Söktillståndet skyddas mot födodämpningen med `max(explore_drive, 1 - hunger_now)`, så hungern har fortfarande företräde: ett svältande djur letar mat, inte partner.

**Reflexen säger när, `_T_MOB` säger hur mycket.** Persistenstiden i sökläget är `pheno_dir_tau()`, som redan bär avvägningen — hög persistens ger effektiv förflyttning men dålig lokal genomsökning. Ingen ny parameter, och magnituden är evolverbar.

Uppmätt, samma frö och samma värld:

```
                        n   rakhet   nettoförflyttning
96x96, 8 djur    före   5    0,161         241
                 efter  5    0,318         496
96x96, 40 djur   före  65    0,167         154
                 efter 75    0,172         191
```

Effekten är stark där djuren är ensamma och nästan borta där de är många — vilket är hela poängen. Vid 40 djur rör medianen sig knappt medan p90 går från 0,398 till 0,469: det är svansen som färdas, alltså de individer som råkar vara ensamma.

Del E:s måltal för rakhet är att den ska ligga över 0,069 **och vara tillståndsberoende**. Den andra halvan är uppfylld först nu.

### Prefix i stället för gruppetiketter

*0127. `n_`, `M_`, `N_` på varje fält.*

0125 delade raden i grupper med rubrikerna `M(kg)` och `N(kg)`. Det löste enhetsförväxlingen men skapade en ny börda: läsaren måste hålla reda på vilken grupp hen befinner sig i, och ett fält mitt i raden säger ingenting om sig självt.

Prefixen gör varje fält självbeskrivande. `n_` är antal, `M_` massa i kilo torrsubstans, `N_` näring i kilo. `grep M_flora` fungerar oavsett var i raden det står.

```
tick 1500  t=  30.0  n_fauna=  28  n_flora=  6610  rotandel=0.67
M_fauna=   33.4  M_flora=4.711e+03  M_förna=8.506e+04  M_kadaver=   0.00
N_fri=     337  N_flora=    146  N_förna=    358
n_föd=   9  n_död=   1  dräkt=  2/ 94%  17.66 ms/tick
```

`M_`-prefixen är desamma som i världsloggen, så samma namn betyder samma sak i konsol och logg.

**Den korta raden fick samma delning.** `format_diagnostics`, som körs utan `--stats`, skrev fortfarande `M_detritus` för det som sedan 0121 bara är förna, och saknade kadaverpoolen helt. Två rapportformat som beskriver samma tillstånd med olika namn är värre än ett otydligt.

### Takten var kumulativ, inte aktuell

*0128. Glidande fönster över senaste rapportintervallet.*

`ms/tick` räknades som `elapsed / tick`, alltså ett medel över hela körningen från tick noll. Det gör talet trögt på ett sätt som är lätt att missta för information: i p124 stod det på 44,20 vid tick 1 000 och 22,17 vid 30 000, och fortsatte sjunka långt efter att takten planat ut. Marginaltakten i slutet var 26,8 medan raden skrev 27,17.

Fönstret är nu senaste rapportintervallet, alltså redan ett medel över `--report-every` tick — vid tusen tick är det gott om utjämning utan att lägga till eftersläpning. Det kumulativa medlet står kvar inom parentes, eftersom det är rätt tal för frågan *hur lång tid tar resten*.

```
17.11 ms/tick (medel 17.11)
16.84 ms/tick (medel 16.97)
18.13 ms/tick (medel 17.36)
19.03 ms/tick (medel 17.78)
15.89 ms/tick (medel 17.40)
```

Nu syns att takten faktiskt varierar med beståndet — de två sista raderna skiljer sig tjugo procent, vilket det kumulativa medlet döljer helt.

Fönstret nollställs per körning, eftersom `--seeds` kör flera världar i samma process. Sista raden, där tick inte hunnit öka sedan förra rapporten, faller tillbaka på det kumulativa medlet.

### Dräktigheten var en ögonblicksbild av en säsong

*0129. Tidsmedel och topp över rapportintervallet.*

`dräkt=` räknade antalet dräktiga i just den tick raden skrevs. Det är fel sorts mått på en säsongsbunden process.

Gestationen i p124 har uppmätt period **12,03 månader**, alltså exakt `WorldParams.year_len`, och pågår under en del av året. Rapportintervallet på tusen tick är 20 månader, alltså 1,67 år. Ett årligt förlopp samplat var 1,67 år ger en **svävning med period tre** — och det är precis den som ligger bakom att p124 växlade mellan `dräkt=0/0%` och `dräkt=39/54%` varannan rad. Talet mätte samplingsfasen, inte reproduktionen.

Måttet är nu tidsmedlet över rapportintervallet plus toppen: `dräkt=1.6/5`. Medlet svarar på hur mycket reproduktion som pågår, toppen på hur stor kullen kan bli. Ackumuleringen sker i samma genomgång av agenterna som `unika`, alltså utan extra kostnad.

Median-andelen av målmassan är borta. Den var meningsfull bara som ögonblicksbild och därmed aliasad på samma sätt.

Fönstret nollställs vid avläsning och per körning.

**Fallgropen är generell.** Varje mått som samplas vid rapporttillfället och beskriver något årstidsbundet — dräktighet, tillväxt, temperaturgrind — riskerar samma svävning så länge `--report-every · dt` inte är en multipel av `year_len`. Vid dt 0,02 och tolv månaders år är 600 tick ett år.

### Sådden hade två regler beroende på när faunan sattes in

*0130. Bördighetsmålet gäller alltid. Nytt scenario `f4-start20`.*

Sådden skalades mot `flora_init_mass_ratio · fauna_mass0` när det fanns agenter vid världens skapelse, och mot markens bördighet när det inte gjorde det. Kommentaren i koden listade redan fyra tillfällen då kvoten brustit och slog fast att bördighetsmålet var rätt storhet — men den låg kvar som huvudregel.

Femte gången: ett scenario med `insatts_vid: start` i stället för `jamvikt` gav **en fjärdedels värld**, eftersom grundarna hann bli till före sådden och den andra grenen därmed togs. 58 480 plantor mot 317 246.

Det värre fyndet ligger bakom det. **Samtliga körningar från p114 till p126 sätter in faunan vid jämvikt och hade alltså noll agenter vid sådden.** De använde redan bördighetsmålet. Kvoten gällde i praktiken bara körningar med fauna från start — alltså rökproven — så samma scenario byggde två olika världar beroende på en flagga som inte handlar om floran.

Nu gäller bördighetsmålet i båda fallen:

```
f4-flock   (80 djur, jämvikt)   317 246 plantor   4,1583e5 kg
f4-start20 (20 djur, start)     317 336 plantor   4,1577e5 kg
```

Skillnaden på 0,03 procent är att `seed_fauna` drar ur samma slumpström före sådden och förskjuter den. Att göra floran bitidentiskt oberoende av faunan kräver en egen ström för sådden, vilket skulle ändra utgångsläget i **alla** tidigare körningar och göra dem ojämförbara. Det är inte värt 0,03 procent.

`scenarios/f4-start20.yaml` är f4-flock med tjugo djur insatta vid tick 0. Ingen kompensation behövs.

**Kvar att veta om:** floran är inte i jämvikt vid tick 0. Sådden ligger över jämvikten och självgallrar — 204 640 plantor vid tick 1 000, botten 70 782 vid tick 6 000, jämvikt 102 205 vid 17 000. Faunan möter alltså en födobas som faller till en tredjedel under sina första sextusen tick, av skäl som inte har med betningen att göra. Det är en verklig skillnad mot p126, inte ett fel.

### Viewern förhandlar sitt utsnitt

*0132. Protokoll 4. Anspråksrader bara för synliga celler.*

Bildrutan bär en rad per planta och berörd cell — `claim_share`, `claim_fill`, `claim_dir` och `claim_trait`, tillsammans 28 byte per rad. Vid 256x256 med bördighet 4 blir det ohållbart:

```
                        plantor   MB/ruta   varav rader   vid 15 fps
p126  64x256            102 205      3,4         3,0        51 MB/s
p132 256x256 (t=1000)   818 128     25,6        24,0       384 MB/s
p132 vid sådd         1 315 216     40,2        38,6       604 MB/s
```

De cellindexerade fälten är 1,57 MB oavsett bestånd. **Nittiofyra procent av rutan var rader**, och de gick inte att se: en cell är då omkring fyra pixlar och innehåller 12,5 plantor, så servern packade tolv kilar per cell för att viewern skulle rita dem i fyra pixlar.

**Klienten säger vad den ser.** `{"cmd": "view", "cx", "cy", "hw", "hh"}` i världskoordinater, skickat när utsnittet ändrats — en stillastående viewer kostar ingen trafik. Under `DETAIL_MIN_PPU` pixlar per cellbredd skickar den `detail: false` i stället, eftersom en kil då inte är upplösbar. Beslutet ligger hos klienten därför att det är där pixlarna finns.

Kommandot går **före** styrgrinden: utsnittet ändrar inte simuleringen, bara vad servern bemödar sig att packa, och ska inte kräva `--serve-control`.

Servern tar unionen över klienterna, inte en ruta per klient. Rutan packas en gång och sänds till alla, och det vanliga fallet är en enda viewer. Två viewers på olika ställen får varandras celler också, vilket är billigare än två packningar.

`Grid.cells_in_rect()` gör världsrektangeln till celler. Geometrin bor där och inte hos viewern, så hexövergången ändrar utsnittsberäkningen på ett ställe. Rektangeln får wrappa; punkterna viks av `cell_of_many`.

Uppmätt på 128x128 med 73 521 plantor, komprimerad blob på tråden:

```
inga rader (utzoomat)   0,030 MB
40x40-utsnitt           0,070 MB
hela världen            0,403 MB
```

**Förvalet är inga rader**, inte alla. En klient som aldrig frågar får floran som celltäckning, vilket är allt som ändå syns utzoomat. Det gör att en klient från protokoll 3 skulle tappa floran tyst — så versionen är bumpad till 4 och handskakningen avvisar den i stället.

### `_Client` har `__slots__`

*0133. Ett fältnamn som saknades.*

`view` tilldelades i `__init__` men stod inte i `__slots__`, så första anslutningen gav `AttributeError`. Två saker gjorde felet obehagligare än det borde varit.

**Det syntes inte i rökprovet.** `_Client` skapas först när någon ansluter, och ingen av verifieringarna av 0132 öppnade en socket — de anropade `frame_from_pop` och `pack` direkt. Serverdelen var alltså aldrig körd.

**Och körningen avbröts inte.** Undantaget kastades i accept-tråden, efter att servern rapporterat att den lyssnar. Simuleringen fortsatte i huvudtråden medan viewern aldrig kunde ansluta.

Verifieringen är nu en riktig anslutning: handskakning, `cmd: "view"` med och utan `detail`, och kontroll av att `claim_cells()` svarar med respektive utan celler.

### Sökgrenen låg före flockningen

*0134. Ordningen i reflexkedjan, och vad den inte förklarar.*

0126 lade söktillståndet i kedjan **före** flockningsgrenen, trots att anteckningen till samma patch säger att den ska ligga efter "så den bara fångar ser ingen alls". Koden gjorde alltså något annat än beskrivningen.

Följden: ett parningsberett djur som ser en artfrände utan att den är en giltig partner — `best_mate` kräver att motparten också är parningsberedd — hoppade över flockningen helt. Värre är att flockningsgrenen bär minnet: `elif N > 0.5 or neighbour_memory is not None`. Sökgrenen före den föregriper alltså även det minne som håller ihop ett sällskap när sikten tillfälligt bryts, och den gör det för de 74,7 procent av agenttickarna där djuret är parningsberett.

Grenen ligger nu sist.

**Men det förklarar inte varför djuren inte slår följe**, och det ska sägas rakt ut. En A/B över 2 500 tick i en gles värld gav *identiskt* utfall — samma bestånd, samma medelavstånd, samma kvot mot slumpen. Uppmätt är fallet sällsynt:

```
agenttick          21 745
  parningsberedd   74,7 %
  ser någon alls    3,4 %
  berett OCH ser    2,4 %
  varav giltig partner  2,2 %   <- 90 % av mötena
  söktillstånd     70,8 %
```

Nio av tio möten sker mellan två parningsberedda djur, så partnergrenen tar dem ändå. Ordningsfelet biter i 0,2 procent av tickarna.

Det verkliga svaret ligger i de tre andra talen. Djuren är ensamma 96,6 procent av tiden, de **närmar sig** varandra när de möts, och sedan finns ingenting som håller dem kvar: söktillståndet återinträder så snart sikten bryts, och flockningen är svag med avsikt — nitton procent tätare än slumpen. Sensingen hjälper inte heller: i vila går tio tick mellan två avläsningar, alltså 5,6 celler av en synvidd på tolv.

### Kontaktens varaktighet

*0135. Mätning innan mekanik.*

Frågan *slår de följe när de möts* gick inte att svara på. Kvoten mot Poisson mäter momentan täthet och låg på 0,95 till 1,03 oavsett vad flockningen gjorde — och när en affinitetsmekanism prövades gav den 0,950 mot 0,973 med spridningen 0,936–0,987 inom respektive grupp. Måttet kunde inte skilja effekt från brus.

Det som saknades var **varaktigheten**. Ett möte som varar en tick är ingen flock.

Ett möte är en obruten följd av tick där samma artfrände ligger i `_cached_agent_hit`. Byter motparten identitet, eller försvinner den, stängs mötet och längden bokförs. Tre tal faller ut:

```
kontakt      977 avslutade möten, 1.450 per djur och månad
             längd median 0.14 mån, p90 0.38, max 1.6
             sällskap 29.1 % av agenttickarna
```

Medianmötet varar alltså **sju tick** — 0,14 månader — mot en synvidd på åtta cellbredder och en fart på 0,43 celler per tick. De passerar varandra. Det förklarar också varför Poisson-kvoten var okänslig: djuren har någon inom synhåll 29 procent av tiden, men nästan alltid en ny någon.

Tillståndet ligger i en dict med individens id som nyckel, inte på agenten, så instrumenteringen rör ingen produktionskod och är helt borta utan `--stats`.

**En fälla värd att minnas.** Nollställningen hamnade först i `gestation_window()`, som nollställer vid avläsning — alltså anropad av `format_stats` precis före utskriften. Räknarna var därför alltid noll när de skulle skrivas, och blocket hoppades tyst över. Två nollställningsmönster i samma fil, ett vid avläsning och ett per körning, och de ser likadana ut.

## Steg 6c — Rörelsemotorn

*Kräver Steg 5h för att vara mätbar och Steg 6a för sektorpercepten. Underlag: `docs/rorelsens-arkitektur.md`, Del 2–6.*

Den exklusiva reflexkedjan ersätts av viktade drifter. Tre döda beteendetraits får sina första läsare, predationen aktiveras, och flockning blir möjlig.

- Sex drifter — föda, flykt, jakt, parning, värme, social — ger var sin riktning och vikt. Vektorsumman styr; ingen drift skriver över någon annan.
- MLP:n viktar drifterna och får en sjunde utgång som är en fri riktning med egen vikt. Drifternas riktningar går in i observationsvektorn. Formen är beslutad: A med en utgång som är D, se dokumentets Del 4.
- `_T_RISK_AV`, `_T_COLD_AV` och `_T_SOC` får läsare som flyktens, värmens och avståndshållningens vikter. `_T_SOCIAL_DIST` och `_T_ALIGN` tillkommer. Genomet går 38 → 40.
- Predationen aktiveras: `attack_energy_gain` får en läsare så att predatorn tillgodogörs sitt byte. Förutsätter att kadaver skiljs från förna — beslutet står öppet sedan Steg 4 och blir här blockerande.
- Hungern skalar födodriften kontinuerligt i stället för att grinda den vid 0,4. Det är mekanismen bakom flockdelning: en mätt flock håller ihop, en hungrig följer var sin gradient.

**Klart när:** predationsdödsfall > 0 över 100 000 tick, inget beteendetrait saknar läsare, och `sociability` differentierar med predationen påslagen men driftar neutralt utan den.

### Två fynd ur styrningsmätningen (0136, 0138)

**Parningsgrenen är redan ett val.** Den skriver `turn = clamp(0,95 · biasN)`,
en tilldelning — inte `turn +=` som de åtta övriga. Den kastar alltså MLP:ns
och köldtermens anspråk och behåller ett. Att den samtidigt avviker mest av
grenarna från det slutliga `turn` — 30,3 % över trettio grader mot flockens
23,4 — är därför inte ett fel utan definitionen på ett val: den enda gren som
avviker mycket är den enda som faktiskt väljer, och det som lagts på efteråt
är födan.

Grenen ska alltså inte rättas utan **generaliseras**. Den är den befintliga
formen av det arbitreringen ska göra för alla nio bidrag. Det som saknas i den
är bara det arbitreringen tillför: en poäng som avgör *vilken* som vinner, i
stället för elif-kedjans fasta ordning.

**Predationens grind ligger i `attack_score`, inte i traitet.** Antagandet att
hotgrenen sällan avfyrar för att predatoriska genotyper är ovanliga stämmer
inte. Mätt över 79 000 agenttick i `f4-start20`-liknande värld:

```
hunt_eff                       median 0,124   p90 0,365   max 0,529
över predator_trait_min 0,20                  29,9 % av agenttickarna
över threat_predation_min 0,35                12,3 %

med byte i sikte och hunt_eff ≥ 0,20                     n = 9 736
  attack_score                 median −0,307  p90 −0,030  max +0,313
  över hunt_score_min 0,12                     1,2 %
  över attack_score_min 0,18                   0,4 %
  avstånd under attack_range 1,5               8,5 %
  båda villkoren samtidigt                     0,34 %
```

`attack_risk` har medianen 0,830 mot `attack_value` 0,578, alltså är
differensen negativ i det normala fallet. Termen `0,40 · target_pred` ensam
räcker för att göra en genomsnittlig artfrände olönsam att angripa för en
genomsnittlig predator. Utfallet blev 52 flykttick och 226 jakttick av 78 961,
och noll dödsfall i predationspasset.

Två följder. Steg 6c:s riskvärdering behöver kalibreras om, inte bara få nya
läsare — och det arbetet är skilt från att aktivera `attack_energy_gain`.
Och: flyktens mättnadsvärde i behovstrappans normering kan inte mätas i den
här världen, eftersom grenen knappt avfyrar. Det värdet blir en gissning som
ingen mätning motsäger, och det ska stå skrivet snarare än döljas av att det
finns en siffra. En scenariovariant med predatorisk diet i tillräcklig andel
av beståndet är eget arbete och hör inte till normeringssteget.

### Fyra beslut som p140 gör skarpare

**Jaktens beslutströskel ligger under attackens.** Båda använder `attack_score`,
men grenen avfyrar på `> hunt_score_min` 0,12 medan angreppet kräver
`> attack_score_min` 0,18. Ungefär två tredjedelar av alla jaktbeslut är alltså
åtaganden som per konstruktion inte kan sluta i ett angrepp — 41 173 jakttick i
p140 utan ett enda byte. Tröskeln ska inte flyttas för sig: i trappan är
jaktens *styrka* bytets värde mot risken, alltså `attack_score`, och den ska
normeras så att den är noll där angreppet blir omöjligt. Flykten mäter mot
samma 0,12 fast för motpartens poäng; två olika beslut delar i dag en konstant
utan skäl och ska ha var sin.

**λ måste prövas om när predationen slås på.** Trappans nivå 1 och 4 avfyrar i
dag i 0,18 respektive 0,51 procent av tickarna, och noll dödsfall följer. En
kalibrering av λ i den världen säger inget om hur trappan beter sig när flykten
blir vanlig. Ordningen är ändå rätt — λ är en ordning, och en gren utan styrka
stör ingen ordning — men omprövningen ska vara planerad, inte upptäckt.

**Styrkraften har redan en kanal, och det är slitage och inte energi.**
`effort = speed_n + 0,6 · activity` och `activity = 0,03 + 0,45 · speed_n +
0,10 · ate`. Ingen term för vridning, och `_integrate_motion` löser farten ur
kraftbalansen utan koppling till `d_steer`. Att svänga är alltså gratis i dag.
Men `dD_eff` är 33,7 procent av all skada i p140 och matar också
trötthetsintegratorn, medan lokomotionen bara är 3,8 procent av energiintaget.
Steg 5 behöver därför ingen ny mekanism utan en vinkelterm i ett uttryck som
redan finns — avvägningen uppstår genom slitage, inte genom joule.

**Normeringen i steg 2 måste bevara produkten vikt × styrka.** Att skala om
`fd` från 0–0,6 till 0–1 ändrar amplituden med faktorn 1,67 om inte vikten
följer med från 0,60 till 0,36. Görs det inte blir steg 2 en tyst
beteendeändring maskerad som en refaktor, och p140 upphör att vara jämförbar.
Kravet är starkare än så: eftersom simuleringen är kaotisk räcker det inte att
produkten stämmer analytiskt — även en omassociering på sista biten driver isär
banorna på några hundra tick. Steg 2 ska därför vara **bitidentisk**, inte bara
ekvivalent.

### Florans luckor saknar pionjärer

Uppmätt i p148, 80 000 tick på `f4-flock`:

```
  i   flora_n   celler  per cell    frön  etabl   frömassa  medianmassa  rotandel
  0    317719    16384     19,4    7683      0     0,0429      1,314       0,50
237     81773    16219      5,0     861     62     0,1814      0,297       0,58
790     55203    14043      3,9     104     39     0,1718      0,226       0,69
```

Fröproduktionen har fallit 200-faldigt, frömassan har fyrdubblats, och 2 300
celler — fjorton procent av världen — står tomma. Luckorna finns. Det saknas
frön att fylla dem med.

**Varför fröna blev stora.** `p = m²/(m² + h²)` med `h = 0,05 · e^(2·trängsel)`.
I det täta ungbeståndet var trängseln omkring 2,3 och `h ≈ 5 kg`; ett frö på
0,043 kg fick etableringssannolikheten sju hundratusendelar, och första
rapporten har noll etableringar på 7 683 frön. Selektionen gjorde det enda den
kunde. Sedan tunnade betet ut beståndet, trängseln föll till omkring 0,07 och
`h ≈ 0,057` — nu skulle ett frö på 0,05 kg vinna med runt sextio procent i
förväntad avkomma. Axeln vänder också: frömassan toppade på 0,187 och är nere
på 0,15–0,17. Selektionen är bara långsam jämfört med hur snabbt betet ändrar
världen.

**Två strukturella brister.**

Fröproduktionen kollapsar med plantstorleken: medianplantan väger 0,20 kg mot
1,31, och 69 procent är rot. Avsättningen går ur inkomsten, inkomsten ur
bladarean, och bladarean är det betet tar. Bete → små plantor → få frön →
luckor består → glesare bestånd → mindre plantor. Positiv återkoppling mot ett
glest tillstånd, inte en jämvikt.

Och **det finns ingen fröbank**. Etableringen avgörs i samma tick som fröet
landar; det som misslyckas blir detritus. Ett frö kan inte vänta på en lucka,
vilket är precis vad pionjärstrategin bygger på i verkligheten. Utan bank måste
den lilla genotypen råka producera ett frö som råkar landa i en av 2 300 tomma
celler — och hela världen gör 110 frön per rapportintervall.

**Förslag.** Frön som inte etablerar sig läggs i en per-cell-pool med en
överlevnadstakt och prövar igen varje tick. Förlusttakten är avvägningen som
gör det till en riktig axel enligt husregeln: en bank är dyr att bygga och
läcker. Den ger också frömassan en andra konsekvens i motsatt riktning — stora
frön etablerar bättre men lever kortare i banken.

### Aptitens massterm importerar florabristen

`hunger()` returnerar `max(reservunderskott, massunderskott)`, där
massatermen är `(M_exp − M)/(M_exp − M_min)`. Den kom till av ett riktigt
skäl: `Ecap = reserve_cap · M`, så när djuret magrar krymper både reserven och
taket och kvoten står nästan still — ett djur på väg mot `M_min` rapporterade
aptiten 0,33.

Men det är samma kategorifel som svälten hade en nivå upp. Att vara liten är
ett tillstånd, inte en aptit, och i p148 låg hela populationen stadigt på 69
procent av `expected_mass` **därför att floran inte levererar**. Massatermen
importerar alltså florabristen rakt in i födosökets styrka, och när dödzonen
nu är borta väger den tyngre än förut.

Rätt fix är troligen inte massatermen utan att mäta reserven mot *förväntad*
kapacitet: `Et / (reserve_cap · M_exp)` faller genuint när djuret tynar, utan
att importera ett kroniskt massunderskott. Då kan massatermen tas bort helt.

Bör göras efter fröbanken, inte före — annars kalibreras aptiten mot en värld
som ska ändras.

### Skademodellens svältterm

`dD_starve` drivs fortfarande av `massunderskott()`, alltså massan relativt
förväntad massa. Styrningens svältstyrka gör det inte längre — den bygger på
underskottets andel av underhållet, av det biologiska skälet att kronisk låg
tillgång ger mindre vuxna och inte sjuka vuxna.

Att låta skademodellen följa med är ett rimligt nästa steg men **inte samma
beslut**. Det ändrar dödligheten och därmed hela ekologin, och det är dessutom
sammanflätat med florabristen ovan: en del av dagens massunderskott är florans
fel, inte fysiologins. Rätt ordning är att laga floran, mäta om, och först då
avgöra om ett stadigt litet djur ska ta svältskada eller inte.

### Vad florabristen gör med styrningens beslut

Faunan i p148 ligger stadigt på 69 procent av `expected_mass` därför att floran
inte kan leverera, inte därför att målvikten är felkalibrerad. Rör alltså inte
`expected_mass`.

Följden för behovstrappan är att våra beslut ska delas i två högar. **Styrningen
byggs efter hur det bör fungera, inte för att kompensera en floramodell som ska
lagas.**

Står oberoende av floran, eftersom de kommer ur trösklar, fysiologi eller
struktur:

- flyktens mättnad = `attack_score_min`
- `styrka_angrepp` som kapacitet gånger närhet, kapaciteten utvärderad vid
  kontakt
- nedkylning ur `Tb` mot `Tb_min`
- parningsdriften som tid gånger den bindande av massöverskott och reserv
- `sig` som bäring, hungern som styrka
- regeln att ett nödläges styrka ska vara nära noll i normaldrift

Att svälten byggs på katabolism i stället för massunderskott hör också hit, men
**av det biologiska skälet, inte det uppmätta**: kronisk låg tillgång ger mindre
vuxna, inte sjuka vuxna, och ett litet djur i energibalans är friskt. Att
`mass_severity` låg på 0,52 i median i p148 är ett symptom på floran och får
inte ensamt bära beslutet.

Provisoriskt, och ska mätas om när floran är lagad:

- **λ.** Väljs ur den uttalade avsikten — ett maximalt socialt önskemål ska
  kunna slå en svag hunger men aldrig en verklig flykt — och inte ur p148:s
  fördelningar, som är fördelningar för ett överbetat system.
- nivåernas inbördes ordning i botten, särskilt födosök mot flock
- varje mättnadsvärde som skulle ha lästs ur en fördelning

### Synvidd, fart och ticklängd är samma fråga

Tre tal som var för sig ser rimliga ut och tillsammans inte går ihop:

```
tick        0,02 månader = 14,6 timmar
fart        0,52 cellbredder per tick = 100 m  ->  2,4 km/dygn, rimligt
syn         6 cellbredder = 600 m              ->  sex dygn dit, orimligt
```

Farten är den enda som stämmer. Att ett djur ser sexhundra meter men behöver
sex dygn för att nå dit betyder att synvidden inte är kalibrerad mot rörelsen.

Två utvägar som utesluter varandra. **Synvidden är för lång** — ett par
cellbredder är vad ett djur i den storleken urskiljer, och då nås målet på en
till två tick. Det skulle också göra sensing billigare, vilket är faunapassets
tyngsta post. **Eller farten är för låg**, vilket `varldens-skala.md` redan
hävdar: tvåhundra gånger under det realistiska. Då nås sexhundra meter på en
halv tick, och problemet blir det motsatta — djuret hoppar över det det ser.

Den andra vägen kräver kortare tidssteg, och där sitter låsningen: en timmes
tick ger fjorton gånger körtiden, alltså fyra timmar per körning i stället för
sjutton minuter. Utvägen är **delad kadens** — världen och floran på 0,02
månader, faunans rörelse och sensing i understeg inom samma tick. Faunapassen
är en bråkdel av tickens arbete (150 djur mot 300 000 plantor), så tio
understeg av bara dem kanske dubblar körtiden i stället för att fjortondubbla
den. `docs/varldens-kadensmodell.md` finns redan och tänker i de termerna.

Ingen av vägarna behöver väljas förrän farten ska höjas mot det realistiska.
Med omedelbar riktning hinner djuret sex tick genom sin egen synvidd och kan
väja under vilket som helst av dem, vilket räcker för nuvarande fart.

## Steg 6b — Fauna store-first

*Störst risk, störst utdelning.*

- `Body`:s skalära tillstånd flyttas fält för fält till store-arrayer med dokumenterat ägarskap. `Body.step()` behålls som sammanhållen fysiologikärna men opererar på store-slices.
- Gestationstillståndet byter ägare från `Body` till store när `gest_M`-ackumulationen flyttat med.
- Återstående kapacitetsfält kopplas till läsare: `mobility` styr rörelsekostnad, `attack_capacity` styr predation. Efter detta ska listan i B1 vara tom.
- Underhållskostnad per buren kapacitet generaliseras från sensing till övriga kapaciteter. Det är A2 fullt ut.
- Synkhjälparen och gestationscachen avvecklas när sista fältet bytt ägare.

**Klart när:** inget fauna-tillstånd har två skrivare, och inget kapacitetsfält saknar läsare.

## Steg 7 — Geologin och vattnet

*Underlag: `docs/geologin-och-vattnet.md`. Patchserien 7001 och uppåt.*

Steget hette tidigare bara "Hydro" och beskrev ett explicit grannflöde med tvåstegsmetod. Mätningen inför steget ändrade formen. Tidssteget är omkring femton timmar och vatten hinner på den tiden korsa hela världen, så ett CFL-begränsat schema låter en flod behöva tio simulerade månader på att nå havet. Uppmätt kostar det dessutom 11,4 ms per tick vid 512x512 mot jämviktslösningens 0,68. **Hydro löser därför stationärt tillstånd per tick längs ett förberäknat dräneringsnät.** Fysiken är oförändrat lokal, gradientdriven och kontinuitetsbevarande; det explicita schemat behålls som valideringsorakel.

Geologin kommer med i samma steg, eftersom hydro inte går att pröva utan höjdskillnader — och eftersom terrängen på köpet ger det modellen saknar mest: en miljöaxel som varierar finkornigt i **två** dimensioner. Latituden är i dag den enda rumsliga axeln, och korskorrelationen i födelsetakt mellan latitudband ligger på 0,45–0,67 mot ett mål under 0,3.

**Låsta beslut:** jämviktslösning, inte transient; 64x256 först och 512x512 efter florans GPU-väg; havet som ett sammanhängande bälte kring polerna; säsongsbunden och latitudberoende nederbörd med statisk orografisk modifierare; höjdgradient på temperaturen direkt.

| | innehåll | konsument | utfall |
|---|---|---|---|
| ~~7001~~ | dokumentet, kadensklasserna i manifestet, planen | — | **klart** |
| ~~7002~~ | terränggenerator via spektralsyntes över `grid.cell_center_*`; polarhavet; `varld.terrang` | ögat | **klart** |
| ~~7003~~ | dräneringsnätet: prioritetsflod, `flow_to`, `flow_order`, sjöhypsometri, `sea_mask` | 7004 | **klart** |
| ~~7004~~ | `hydro_pass()` som jämviktslösning; vattenbalans som hård invariant | 7005–7008 | **klart**, drift 1e-16 |
| ~~7005~~ | temperaturens höjdgradient, per cell | klimatet blir 2D | **klart** |
| ~~7006~~ | markvattnet som tredje term i tillväxtens `min()` | floran | **klart** |
| ~~7007~~ | vittring ur lutning, urlakning nedströms, havet som sänka | näringen | **klart**, 3,6x gradient |
| ~~7008~~ | partikulär transport av förna nedströms; sjöar som fällor | faunans asätande | **klart** |
| ~~7009~~ | täthet härledd ur strukturandel; `buoyancy` får skrivare och läsare; drag i vatten | faunans rörelse | **klart, men läsaren fyrar aldrig — se nedan** |
| ~~7009b~~ | lutningen som framkomlighet: uppför kostar, nedför är billigt | faunans rörelse | **klart**, fart 55 nedför mot 20 uppför |
| ~~7009c~~ | vadandet kostar, värmeledning vid nedsänkning, passiv drift ur kontinuitet | faunan möter vattnet | **klart** |
| ~~7023~~ | kroppen får ett djupmått; tre läsare delar det | draget, styrningen, driften | **klart**, se nedan |
| ~~7024~~ | driften blir uppehållstid längs nätet; `drift_max` utgår | styrningen | **klart**, se nedan — max 143 → 13 utan tak |
| ~~0169~~ | perceptets halvmättnad; `C_sense_K` var en halv gram | perceptet, 0170 | **klart**, se nedan — havet 0,60 → 0,003 |
| ~~0170~~ | betningen följer vägen i stället för en cirkel | florans fläckighet | **klart**, se nedan — **återtagen i 0171** |
| ~~0171~~ | betesgrannskapet är en rumslig räckvidd, inte en väg | betningen | **klart**, se nedan — 40/40/40 |
| ~~0172~~ | betesytan blir bansträcka gånger kroppens räckvidd | kadensbytet | **klart**, se nedan — neutral, men skalar med `dt` |
| ~~0174~~ | rörelsen kostar transport, inte dragdissipation | budgeten, kadensen | **klart**, se nedan — uppför var billigare än platt |
| ~~0175~~ | assimilationen straffade segheten två gånger | födobudgeten | **klart**, se nedan — 16 → 21 % |
| ~~0176~~ | betningen blir tidsstegsinvariant; `h` och `eat_rate` är samma tal | kadensbytet | **klart**, se nedan — intaget skalade som `dt²` |
| ~~0177~~ | parningen kunde inte vinna, och sökte i fel säsong | reproduktionen | **klart**, se nedan |
| ~~0178~~ | avsvalningen härleds ur dräktighet och laktation | reproduktionen | **klart**, se nedan |
| ~~0179~~ | kullstorleken som ärftlig axel; dödströskeln blir relativ | reproduktionen, r/K | **klart**, se nedan |
| ~~0180~~ | avmagring mäts mot kroppens egen topp, inte mot ett löfte | selektionen | **klart**, se nedan |
| ~~0181~~ | livshistoriens massor blir andelar av vuxenmassan | selektionen | **klart**, se nedan |
| ~~0182~~ | mognaden blir ett utfall, inte en ålder | selektionen | **klart**, se nedan |
| ~~0183~~ | betningen tar hela grannskapet i ett svep | prestanda | **klart**, se nedan — 3,4× |
| ~~0184~~ | kostnadsproven batchas; kustmätningen blir valfri | prestanda | **klart**, se nedan — `cell_of` 4,8× färre |
| ~~0185~~ | genomklämman höjs så att det deklarerade spannet blir nåbart | selektionen | **klart**, se nedan |
| ~~0186~~ | kustmätningen flyttar från modellen till instrumenteringen | mätningen | **klart**, se nedan |
| ~~0187~~ | bärarrollen blir en ärftlig allokering i stället för massa | selektionen | **klart**, se nedan — `M_target` stannar |
| ~~0189~~ | aptiten, reservtaket och fettets uttagstakt som en enhet | svälten | **klart**, se nedan — spillet 1249 → 91 kg |
| — | Fishers jämvikt nås inte; bärarandelen beror på `lactation_k` | reproduktionen | **öppen**, se 0187 |
| ~~—~~ | `f6-256-mager`: bär perceptet någon riktning i en fläckig värld? | 0169, kärnan | **scenariot finns**, kör det |
| — | per-agent-Python är kvar; store-first och Numba är nästa | prestanda | **öppen**, manifestets Fas 4–5 |
| — | `cell_of` anropas 65 gånger per djur och tick | prestanda | **öppen**, nästa |
| — | `M_repro_frac` 0,15–0,45 är bevarad storlek; verkligt är 0,6–0,8 | livshistorien | **öppen**, se 0181 |
| — | `M_waste_frac = 0,075` är bevarad storlek, inte fysiologisk | mortaliteten | **öppen**, se 0179 |
| — | `child_M` är absolut där den borde vara en andel av `M_target` | livshistorien | **öppen**, se 0179 |
| — | kullstorleken som ärftlig axel: antal mot storlek | reproduktionen | **öppen**, nästa |
| — | bäraren väljs på massa, vilket blir en väg ut ur reproduktionens kostnad | selektionen | **öppen**, se 0178 |
| — | `repro_cooldown_s` bär könsmognad och kullintervall i samma tal | reproduktionen | **öppen**, se 0177 |
| — | anspråkens styrkor spänner 0,044–0,955 i median på en deklarerad 0–1-skala | arbitreringen | **öppen**, se 0177 |
| — | hanteringstiden är åtta gånger för lång; aptiten blir enda taket | födobudgeten | **öppen**, se 0176 |
| — | reserven behöver en strukturandel innan magen får komma åt segt material | Steg 6 | **öppen**, se 0175 |
| ~~0173~~ | `docs/scenariers-anatomi.md` | scenarioformen | **klart**, se nedan |
| — | `dt` skrivs ut i varje scenario | kadensbytet | **öppen**, se dokumentet |
| — | kalibreringsblock i scenariofilen | svep utan kodredigering | **öppen**, se dokumentet |
| — | `f6-256-mager`: bär perceptet någon riktning i en fläckig värld? | 0169, kärnan | **öppen**, kräver ingen kod |
| 7010 | vattendjupet som fjärde världskanal och fri MLP-ingång | selektionen | öppen |
| ~~7010~~ | ~~vattenaxeln som nisch~~ | | **slås ihop med 7009** |
| ~~7011~~ | ~~glesning av hydro~~ | | **utgår**, se nedan |
| ~~7013~~ | klimatet som tidsprofil; latituden faller ur världsmodellen | hela världen | **klart**, se nedan |
| ~~7014~~ | världens position på en jordlik planet; klimatet härlett ur latitud och kontinentalitet | klimatet, fotoperioden | **klart**, se nedan |
| ~~7015~~ | lapse rate ur fysik: 6,5 °C/km, referensytan mot havsnivån | höjdklimatet | **klart**, se nedan |
| ~~7016~~ | `docs/regionen-och-omlandet.md`: den detaljerade världen som en region i ett grovt rutnät | planen | **klart**, se nedan |
| ~~7017~~ | landformerna som styrt brus; regionens form som romb i hexgitter | planen | **klart**, se nedan |
| ~~7018~~ | havet placeras efter höjd som bred bassäng; latituden ur terrängen | dräneringen | **klart**, se nedan |
| ~~7019~~ | bandet skalat mot världen; amplituden som Hurstexponent | höjdgradienten, sjöarna | **klart**, se nedan |
| ~~7020~~ | havet som nivå i stället för form; kustlinjen blir en nivåkurva | kusten | **klart**, se nedan |
| ~~7021~~ | knäckt spektrallutning: brant för långa vågor, flack för korta | kusten, sjöarna | **klart**, se nedan |
| 7022 | fotoperioden med florans ljusbudget som första läsare | floran, fenologin | öppen |
| ~~1010~~ | viewern visar terräng och vatten; världen klassar, viewern färgar | ögat | **klart** |
| ~~1011~~ | höjden skickas en gång vid handskakningen | bandbredd | **klart**, protokoll 6 |
| ~~1012~~ | höjdkurvor i viewern, ovanpå vattnet | ögat | **klart**, se nedan |
| ~~1013~~ | riktningsfördelningen fångas och mäts | 0169, 1014 | **klart**, se nedan — fördelningen är en cosinus |
| ~~1014~~ | sektorperceptet ritas som kilar; opaciteten är andelen | ögat, 0169 | **klart**, protokoll 7, se nedan — perceptet är nästan platt |
| ~~1015~~ | radien bär fördelningen, opaciteten mängden; kusten som kontroll | ögat, 0169 | **klart**, protokoll 8, se nedan — marken är sluten överallt |
| ~~1016~~ | kanalerna får var sitt läge; uppehållsfördelningen och visaren | ögat, 0169 | **klart**, protokoll 9, se nedan |
| 1017 | vattnet som kanal i bilden | ögat | **väntar på 7010** |

### Vad mätningarna ändrade i planen

**7008 blev något annat än planen sa.** Dränkning som mekanism visade sig onödig: markvattnet är noll i vatten, så en planta som hamnar där växer inte och svälter — uppmätt 14 individer i havets 3 275 celler av 30 917 totalt. Ingen regel behövdes. Och strandzonen rör sig knappt: 69 celler av 16 384 pendlar mellan blött och torrt över ett år, alltså 0,4 procent av världen. En mortalitetsregel för dem vore kod utan ekologi.

I stället gjordes 7008 till partikulär transport av förna, eftersom mätningen visade att **det inte fanns något att äta i vatten**: 22,6 procent av världen var vatten och innehöll 1,7 procent av växtligheten. Utan föda där får varje akvatisk trait bara en nedsida, och den kollapsar mot den terrestra änden precis som `digestibility` gjorde.

**Vattnet kan inte fragmentera den här världen.** Uppmätt: även med hav, sjöar och samtliga vattendrag oframkomliga är landet ett enda sammanhängande område. Det är en strukturell konsekvens av 7002:s kontinentallutning — vattendelaren ligger vid ekvatorn och alla floder rinner polerna till, så en flod går alltid att gå runt vid källan. Bara en flod som förbinder hav med hav skulle skära av, och lutningen gör det omöjligt per konstruktion. Motivet "floder sänker korskorrelationen mellan latitudband" faller därmed.

**`sediment_rate` är ett reglage mellan land och vatten, inte en fri parameter.** Näringsförlusten motsvarar exakt sedimentexporten: sjöarna behåller sitt, men floderna är ett avlopp till havet. Vid 0,5 kostar det 16 procent av näringsstocken och en tredjedel av floran att göra vattnet rikare per cell än landet.

**Glesning av hydro utgår.** Motivet var att ett tätt grannflöde skulle bli för dyrt. Jämviktslösningen kostar 0,216 ms per tick vid 16 384 celler och 4,1 vid 262 144 — mindre än den täta laplacianen i `transport_pass`. Görs bara om en mätning på en stor värld motiverar den.

### Perceptets halvmättnad låg fem tiopotenser fel (0169)

`_build_sector_percept` mättar världskanalerna med `v/(v+K)`. `K` är
halvmättnadsvärdet och måste ligga där fältet typiskt ligger.

Floraskanalen gjorde det: den läste `B_K = 11,0`, alltså en vuxen plantas
massa, mot en stock på 9,2 kg per landcell. Förnakanalen läste **`C_sense_K =
5e-4`, en halv gram**, mot 99 kg per landcell. Det är samma halva gram `B_K`
en gång hade — den höjdes till 11,0 när massaskalan rättades, och
syskonkonstanten lämnades kvar. **En kalibrering som flyttas på ett ställe och
inte på det andra ger ingen felsignal; den ger en kanal som alltid säger ja.**

Uppmätt i `liten6` efter 250 tick:

```
fält                land    sjö     hav      kg per cell
flora                 9,2    8,5    0,00
detritus+kadaver     99,0  187,4    0,38

kanal               land    sjö     hav
B  vid K = 11      0,415  0,410  0,0001     bär kontrasten
C  vid K = 5e-4    1,000  1,000  0,9705     mättad överallt
C  vid K = 80      0,524  0,677  0,0044
```

Dietviktat läste **havet 0,60 mot landets 0,87** trots 262 gångers skillnad i
verklig födomängd. Det var alltså förnakanalen som sa ja ute till havs, inte
floraskanalen — djuren såg inte växtlighet i vattnet, de såg en mättad
sedimentkanal.

Halvmättnaden läggs vid den uppmätta medianen, 88 kg, avrundat till 80.
`B_sense_K` får samtidigt ett eget namn vid samma värde som `B_K`; perceptet
läste massaskalan direkt, så en ändring av vad en planta väger ändrade tyst vad
ett djur ser.

Utfallet, `liten6` 400 tick, tre frön:

```
                toppandel median      vid kust
frö    7024    0169    0170     7024   0169   0170
 1     0,179   0,209   0,197    0,182  0,238  0,219
 2     0,179   0,200   0,190    0,183  0,227  0,212
 3     0,180   0,198   0,187    0,182  0,226  0,199
```

**Kustkontrollen går från att falla till att hålla.** Överskottet mot
jämnfördelningens 0,167 fyrdubblas vid kusten, och — avgörande — kustdjuren har
nu en *toppigare* profil än inlandsdjuren (0,238 mot 0,209) i stället för samma.
Tre sektorer mot hav kan inte ge en jämn profil, och nu gör de det inte heller.

Sjöarna hamnar över landet på förnakanalen, vilket de faktiskt är sedan 7008.
**Den akvatiska nischen fanns redan och var osynlig.**

Beståndet efter 400 tick: 32, 39, 39 mot 41, 39, 38. Frö 1 faller, de andra
står. **Detta invaliderar kalibreringar mot den mättade kanalen** — födostyrkans
skala och hungerns grindning sattes när `C` läste 1,0 i varje cell.

### Reserven fungerade inte som reserv (0189)

Frågan var varför faunan svälter när det finns hundra till sjuhundra gånger mer
växt per djur än i en verklig ekologi och Hollings tak ligger 6,6 gånger över
behovet. Svaret visade sig vara tre kopplade fel i reservens hantering, och
**inget av dem går att rätta ensamt** — det prövades och beståndet föll varje
gång.

**Ett: aptiten kände inte till utrymmet.** `eat_rate · dt · (0,25 + 0,75 ·
hunger)` är rent motivationsstyrd. Ett tugg vid full hunger ger 0,384 kg labil
massa mot ett reservtak på 0,319 — **ett mål kan fylla hela reserven 1,2 gånger**
— och överskottet töms i `Body.step` som `Et > Ecap`. Uppmätt fyrtiofem procent
av allt som togs upp, alltså dubbelt betestryck på floran utan nytta.

**Två: taket enforcerades bara en gång per tick.** Uppmätt reservfyllnad före
rättelsen:

```
E_total / E_cap    p10 1,000   median 1,000   p75 1,344   p90 3,677
```

Djuren bar alltså **1,3 till 3,7 gånger sitt nominella tak** mellan passen, upp
till 61 procent av kroppen som fett. Nominellt tak och verklig kapacitet var två
olika tal, och `Body.step` slängde skillnaden.

**Tre: fettet gick inte att ta ut.** `M_fast` är **noll i median** — den snabba
poolen är en genomströmning som fylls och töms varje tick — så hela reserven
ligger i `M_slow`. Med `slow_mobil_frac = 0,25` per månad blir uttaget 0,0013 kg
per tick mot ett underhåll på 0,026:

```
taket täckte fyra procent av behovet
```

**Organismen svalt med full tank.** Det märktes inte, därför att överätandet
skyfflade 0,29 kg per tick genom den snabba poolen — genomströmningen *var*
bufferten. Så snart aptiten begränsades föll den bort och taket blev synligt.
Konstantens motivering blandade dessutom ihop en uttagstakt med en varaktighet:
0,25 per månad betyder inte att fettet räcker fyra månader, utan att högst en
fjärdedel får tas ut per månad.

**Rättelserna, som en enhet.**

Aptiten blir återstående utrymme plus ticken förbrukning, omräknat med den
senast uppmätta assimilationsandelen; `eat_rate · dt` står kvar som tak — hur
fort munnen hinner — i stället för som drivkraft.

Reservtaket sätts ur verkligt fett. `M` innehåller inte reserven, så
`reserve_cap` är fett per kilo **stomme**:

```
reserve_cap    per kg stomme    andel av kroppen
0,5e6              0,054              5,1 %     mager gnagare
4,0e6              0,430             30,1 %     jordekorre — gamla taket
1,1e7              1,183             54,2 %     sjusovare före dvala
```

Taket vid femtiofyra procent gör en dvalande strategi nåbar utan att göra den
påbjuden. Golvet står kvar.

Uttagstakten blir 1,5 per månad, alltså full mobilisering på omkring tre veckor
— långsamt mot glykogenets tick, snabbt mot strukturens sista utväg. Vid det nya
taket täcker det 136 procent av underhållet; vid det gamla bara en fjärdedel,
vilket är varför de två inte går att flytta var för sig.

**Utfallet i `liten6`, 300 tick, tre frön:**

```
                bestånd        underhåll strypt   spill
baslinje        81  88 126     2,9 3,5 4,2 %      1249 kg
0189           101 103 114     4,4 4,0 4,1 %        91 kg
```

Spillet faller med en faktor fjorton och beståndet stiger. Reservfyllnaden blir
median 0,971 med p10 0,779 och fyra promille tomma — en reserv som används i
stället för en som svämmar över.

### Bärarrollen blir en ärftlig allokering (0187)

Rollen avgjordes av massa: `if best.body.M > agent.body.M` — den tyngre bär.
Regeln är rimlig i en modell utan kön men är **en runaway utan jämvikt**.
Fitness beror inte på att vara liten i absolut mening utan på att vara *lättare
än sin partner*: en genotyp under populationsmedelvärdet slipper alltid
dräktigheten, medelvärdet sjunker, och den nya undre halvan vinner igen.

Uppmätt i p185: `M_target` gick rakt genom det gamla golvet 0,653 och landade på
median 0,278 med **p10 0,269** mot det nya golvet 0,268 — hela spridningen
kollapsad mot väggen, i båda fungerande frön.

`bearer_p` är nu ett locus: sannolikheten att **den här organismens avkommor**
blir bärare. Barnet drar sin roll ur bärarförälderns benägenhet, en gång vid
födseln, och behåller den livet ut. Parning kräver komplementära roller, och
`_find_best_mate` släpper bara igenom sådana par — annars styr djuren mot
partner de inte kan para sig med, precis det 0177 rättade för säsongen.

**En första form prövades och föll.** Att låta benägenheten styra den *egna*
rollen ger en direkt individuell fördel: en icke-bärare slipper dräktighet,
laktation och avsvalning och betalar bara fem procent av energitaket. Uppmätt
gick bärarandelen till 0,333 efter 1 200 tick och `bearer_p` till 0,238 och
fortsatte falla — samma flyktväg som massregeln, bara ometiketterad. Fishers
argument gäller när egenskapen styr avkommornas roll, inte den egna.

**Huvudresultatet.** `liten6`, frö 2:

```
tick 1200        M_target  p10 / median / p90    djur   litter
0186 massregel     0,335 / 0,431 / 2,000          149    4,02
0187 bärarroll     0,669 / 0,711 / 1,484          586    5,08
```

**Undre kvartilen är exakt dubbel och står stilla** — 0,676 vid tick 600 och
0,669 vid 1 200 — medan massregelns p10 sjunker mot golvet. Populationen är
fyra gånger större. Runawayen är stängd.

**Men oraklet går inte igenom, och det ska sägas rakt ut.** Fishers princip
säger att bärarandelen måste gå mot 0,5 när de två rollerna kostar lika mycket
att producera, vilket de gör här — samma ungmassa. Uppmätt efter 700 tick, frö 2:

```
lactation_k = 0,0    bärarandel 0,705   bearer_p 0,888   356 djur
lactation_k = 1,0    bärarandel 0,44                     272 djur (vid 800)
lactation_k = 3,0    bärarandel 0,462   bearer_p 0,489   171 djur
```

**Andelen beror på bärarens kostnad, och det borde den inte göra.** Antingen är
körningarna för långt från demografisk jämvikt — populationen fyrdubblas under
mätningen, och Fishers ESS är ett jämviktspåstående — eller så finns en
asymmetri jag inte hittat. Talen är dessutom förvirrade av att tätheten skiljer
sig, vilket ändrar partnersökningen och därmed frekvensberoendet.

Andelen rapporteras i `--stats` tillsammans med `bearer_p`s kvartiler, så den går
att följa. **Skiljer sig genotypens fördelning från utfallet — en population av
0,3- och 0,7-genotyper ger också 0,5 i utfall — är det början till två kön.**

Patchen levereras trots att oraklet faller, därför att den löser det den byggdes
för och den gamla regeln bevisligen inte har någon jämvikt alls. Men
bärarandelen är inte en löst fråga.

### Kustmätningen hör inte hemma i modellen (0186)

`Agent._dir_vatten` — andelen sektorer som pekar mot vatten två cellbredder fram
— infördes i 1015 för att skilja kustdjur från inlandsdjur i kustkontrollen.
Den läses **bara** av `run_headless._mat_riktning`, alltså är den mätning i båda
ändar; bara beräkningen låg i `agent.py`.

0184 upptäckte att den kostade sex celluppslag per djur och tick, nio procent av
alla `cell_of`-anrop, för ett tal ingen mekanism läser — och stängde av den med
`AgentParams.mat_dir_vatten = False`.

**Det var fel form, och felet växte i tre steg.** En avstängd mätning kräver ett
sätt att slå på den; ett sätt att slå på den kräver antingen en ny flagga eller
ett generellt överstyrningsmaskineri; och ett sådant maskineri låter en körnings
tillstånd glida isär från dess `scenario.yaml` — vilket är precis det
`docs/scenariers-anatomi.md` säger att scenariosystemet finns för att förhindra.

Beräkningen flyttar därför till `_mat_riktning`, som är den enda läsaren.
Uppmätt: `--stats` av ger 45,8 ms/tick, `--stats` på ger 97,8. **Mätningen kostar
när man mäter och ingenting annars**, vilket är hela poängen med att den ligger i
instrumenteringen.

`AgentParams.mat_dir_vatten` och fältet `Agent._dir_vatten` tas bort.

### Genomklämman gjorde det deklarerade spannet onåbart (0185)

I p182 landade `M_target` på **0,653 i alla tre frön, ner till tredje
decimalen**. Det är ingen konvergens: `logit((0,653 − 0,20)/3,80) = −2,0000`
exakt, och `genetics.traits_clip` stod på 2,0.

Fenotypen fås som `_lerp(min, max, sigmoid(x))`, och `sigmoid(±2)` ger
`u ∈ [0,119, 0,881]` — alltså **76 procent av det deklarerade spannet**,
centrerat. Det gällde alla fyrtiotre loci:

```
axel          deklarerat      nåbart vid 2,0    nåbart vid 4,0
M_target      0,20–4,00       0,653–3,547       0,268–3,932
A_mature      5,0–20,0        6,79–18,21        5,27–19,73
child_frac    0,08–0,20       0,094–0,186       0,082–0,198
```

Tre axlar låg hårt mot den i p182: `M_target` i alla frön, `A_mature` på 6,81 i
ett, `child_frac` i två.

**Att en axel pinnar vid sin gräns ska betyda att biologin tagit slut.** Med
klämman på 2,0 betydde det i stället att representationen gjorde det, och de två
ser likadana ut i loggen. Vi kunde se riktningen men inte om 0,653 kg är ett
optimum eller en vägg.

Att bredda `M_target_min` flyttar bara klämman — med 0,05 blir nåbara golvet 0,52
i stället för 0,653. Det är klämman själv som måste flyttas.

Mappningens form är oförändrad: sigmoidens derivata vid `±clip` är fortfarande
omkring 2,4 gånger lägre än i mitten, så gränsen förblir mjuk och ett
mutationssteg nära kanten flyttar fenotypen mindre än ett i mitten.
`traits_sigma` och `traits_p` rörs inte.

Uppmätt i `liten6`, 400 tick, tre frön: 75/82/100 mot 0184:s 72/70/73. Det säger
ingenting om selektionen — fyrahundra tick räcker inte — bara att modellen bär
det bredare spannet.

**Nästa körning är ett experiment med ett svar.** Stannar `M_target` någonstans
över 0,268 är det ett optimum; går den till golvet finns ingen kostnad för att
vara liten, och det är nästa knut. Följ dess median över tid.

### Kostnadsproven batchas (0184)

`_kostnad_vag` provade sex punkter per bäring över upp till åtta kandidater, med
ett `grid.cell_of` per punkt. Det gjorde celluppslaget till modellens största
enskilda post: uppmätt **791 489 anrop på tjugofem tick med 482 djur**, alltså
sextiofem per djur och tick, och sjutton procent av tickens tid kumulativt.

Uttrycket är oförändrat — värsta provet, avståndsviktat — men punkterna slås upp
tillsammans med `cell_of_many`, som funnits hela tiden. Fyrtioåtta anrop blir
ett.

`Agent._dir_vatten` stängs av som förval. Den är ren diagnostik från 1015 —
andelen sektorer som pekar mot vatten, för att skilja kustdjur från inlandsdjur
i `--stats` — och kostade sex celluppslag per djur och tick för ett tal ingen
mekanism läser. `AgentParams.mat_dir_vatten` slår på den när kustkontrollen ska
mätas.

Uppmätt, 482 djur, 25 tick:

```
                      cell_of-anrop   tottime   ms/tick   µs/djur
före 0183                  791 489     2,679       931      1936
efter 0183                 791 489     2,679       862      1789
efter 0184                 164 585     0,722       785      1629
```

**Bitidentiskt tillstånd efter trettio tick** mot 0183: floramassa
83341,492188 kg, 64 672 plantor, 39 djur, summerad kroppsmassa 55,79001747.

`rebuild_spatial_index` lämnas orörd trots att den anropas två gånger per tick.
Det andra anropet är `with_flora_fields=False` och behövs: efter faunapassen kan
medlemskapet ha ändrats av rörelse, födslar och dödsfall. Anteckningen i den
gamla statusanalysen om ett dubbelarbete är alltså inaktuell.

**Vad som återstår.** 0183 och 0184 tillsammans ger sexton procent, och därefter
är kurvan platt: `Body.step`, betningen och kostnaden ligger nu på ungefär en
sekund var per tjugofem tick, och `getattr` ensam står för 1,67 miljoner anrop.
Det är per-agent-Python, och det finns ingen mer frukt på den nivån.

Nästa steg är manifestets Fas 4 och 5: flytta `Body`:s tillstånd till
store-arrayer och köra fysiologin som ett Numba-pass. Det bör inte påbörjas
förrän livshistorien slutat röra sig — sex av dess fält har bytt betydelse under
0174–0182 — annars migreras fält som ännu ändrar innebörd.

### Betningen tar hela grannskapet i ett svep (0183)

Uttaget var en nästlad Python-loop: för varje cell i grannskapet, för varje
planta i cellen, `min(edible, amt)` med ett tiotal `float()`-omvandlingar per
besök.

Så länge betet var glest räckte det — kommentaren i koden noterade 1,22 plantor
per anrop. **Den siffran gäller inte längre.** När betestrycket stiger ligger
plantorna vid sin meristemrefug, `edible` är noll, och loopen skannar hela
grannskapet utan att ta något. `cell_avail` kan inte fånga det, eftersom den bär
skottmassan `max(0, m − rot)` medan det ätbara är `m − rot · meristem_keep`: en
cell kan ha skott men inget ätbart.

Kostnaden växte alltså **med** betestrycket, vilket är precis fel håll.

Uppmätt på `liten6` med 482 djur, 25 tick:

```
                              tottime/anrop   kumulativt   ms/tick
före                              322 µs         472 µs      931
cellvis vektorisering             235 µs         459 µs        —
hela grannskapet i ett svep        94 µs         281 µs      854
```

**Cellvis vektorisering gav bara en tredjedel.** Med ett tjugotal plantor per
cell är numpy-anropens fasta kostnad — omkring trettiofem mikrosekunder per cell
— större än arbetet. Sju celler i ett svep gör den kostnaden till en sjundedel,
och det är den formen som gäller.

`OrganismStore.slots_in_cells` slår upp flera celler med ett `searchsorted` och
en konkatenering, i cellernas ordning. Ordningen är densamma som den gamla
nästlade loopens, eftersom `cells_within` ger ringarna inifrån och ut, så
girigheten fyller tuggan ur samma plantor i samma följd. Uttaget blir en
`cumsum` med ett `searchsorted`.

**Bitidentiskt tillstånd efter trettio tick**: floramassa 83341,492188 kg,
64 672 plantor, 39 djur, cellsumma 41346,750000 — samma tal före och efter.

Tickvinsten är åtta procent. Det låter lite och är rätt: betningen var
tjugotvå procent av tiden och är nu elva. **Det som blev störst i stället är
`cell_of` med 791 489 anrop på 25 tick — sextiofem per djur och tick**, alltså
tolv procent av tottime och sjutton kumulativt. Den står näst i tur.

### Mognaden blir ett utfall (0182)

Villkoret var `age >= A_mature`. Det gjorde **tidig mognad gratis**: den gav
tidigare reproduktion utan att kräva att kroppen faktiskt vuxit.

0180 tog bort *straffet* för ett obetalt tillväxtlöfte — svältskadan mättes mot
en utlovad kurva — men lät *belöningen* stå kvar. Uppmätt i p181 föll
`A_mature` därför mot golvet 5,0 i **två frön av tre igen**, precis som i p179.
Att rätta straffet utan att rätta belöningen räckte inte, och det borde ha stått
i 0180 i stället för förutsägelsen att axeln skulle plana ut.

Massan är kvar som grind och är sedan 0181 allometrisk, `M_repro_frac ·
M_target`. Att mogna tidigt kräver därmed att ha vuxit fort, och att växa fort
kräver mat. En kropp som svultit som ung mognar sent och får färre kullar —
**kostnaden är fysisk i stället för deklarerad, och den betalas när den
uppstår.**

`A_mature` styr efter detta bara tillväxtkurvans branthet via `growth_k`. En
brant kurva är fortfarande en fördel, men bara i den mån kroppen kan finansiera
den. Uppmätt i `liten6`:

```
faktisk mognadsålder   p10  1,2   median  7,4   p90 35,2 månader
deklarerad A_mature    p10  7,4   median 12,0   p90 17,9
korrelation                                     0,59
```

Mognaden följer alltså genotypen till hälften och världen till hälften, vilket
är vad ett utfall ska göra. Spannet 1,2–35,2 mot det deklarerade 7,4–17,9 visar
hur mycket av variationen som nu kommer från vad kroppen faktiskt fick i sig.

Den mjuka grinden läser samma villkor som den hårda, av samma skäl som i 0177:
ett djur ska inte vara motiverat till det som är omöjligt.

Uppmätt i `liten6`, 400 tick, tre frön: 64/73/92 mot 0181:s 49/81/59, med 142
födslar mot 100.

**Kvar att mäta:** finns någon kostnad för att vara liten? Underhållet skalar som
`M^0,75` och betesytan som `M^⅓`, så kvoten yta genom behov förbättras när
kroppen krymper. Motvikten borde vara termoregleringen — värmeförlust `M^⅔` mot
produktion `M^0,75` — men om den inte biter kan `M_target` rasa mot golvet 0,2.
I p181 gick frö 3 redan 1,54 → 0,66.

### Livshistoriens massor blir andelar av vuxenmassan (0181)

`M_target` spänner 0,2–4,0 kg, alltså tjugo gångers skillnad i kroppsstorlek.
`child_M` och `M_repro_min` var **absoluta** och oberoende av den:

  en kropp med `M_target = 0,20` investerade 0,16–0,40 kg i en kull, alltså upp
  till **tvåhundra procent av sin egen vuxenmassa**, och nådde
  fortplantningsmassan 0,3–0,9 kg först efter att ha passerat den. Den kunde
  aldrig reproducera sig.

  en kropp med `M_target = 4,00` investerade fyra till tio procent och mognade
  vid åtta till tjugotvå. Den hade det för lätt.

Uppmätt investerade **åtta procent av populationen mer än halva sin egen
vuxenmassa** i en kull. Kroppsstorleken var därmed inte en livshistoriestrategi
utan bara en underhållskostnad — den gav ingen billigare reproduktion tillbaka,
och det förklarar varför `M_target` kunde drivas uppåt i p179 utan att betala.

Andelarna är valda så att medianen ligger kvar där de absoluta talen låg vid
`M_target ≈ 2,0`: `child_frac` 0,08–0,20 ger 0,16–0,40 kg och `M_repro_frac`
0,15–0,45 ger 0,30–0,90 kg. **Bevarad storlek, rättad form** — samma mönster som
`E_move` i 0174 och betningen i 0176.

Uppmätt i `liten6`, 400 tick, tre frön: 49/81/59 mot 0180:s 78/54/47. Spridningen
mellan frön är stor och riktningen otydlig vid de talen; det som är verifierat är
att axeln blivit skalfri.

**En öppen post.** `M_repro_frac` 0,15–0,45 betyder att ett djur föder vid
femton till fyrtiofem procent av sin vuxenmassa. Ett verkligt däggdjur gör det
vid sextio till åttio. Talet är bevarat och inte härlett, av samma skäl som
`M_waste_frac`: hela mortalitets- och tillväxtkalibreringen vilar på de gamla
absoluta värdena och måste flyttas som en enhet.

### Avmagring mäts mot kroppens egen topp (0180)

`expected_mass(age)` är vad genomet har **lovat** att kroppen ska väga —
von Bertalanffy från `child_M` mot `M_target` vid `A_mature`. Svältskadan mättes
som `M / M_expected` mot 0,55, hungermåttet mot samma tal, och sedan 0179 även
dödströskeln.

**Det gjorde tillväxtkurvan till ett löfte som fenotypen straffades för att
bryta, medan vinsten av att lova mycket betalades ut först.** Sänkt `A_mature`
ger tidigare mognad och därmed tidigare reproduktion omedelbart; skadan kommer
efter att genen är vidarebefordrad.

Selektionen tog den affären. Uppmätt i p179, `f6-256` över 10 000 tick, tre frön:

```
             A_mature    M_target   litter   krävd tillväxt
s2 start       12,1        2,07      3,4      0,165 kg/mån
s2 slut         6,8        3,28      5,4      0,475
s3 start       13,2        1,54      3,9      0,111
s3 slut         6,8        2,00      4,9      0,285
```

`A_mature` föll mot golvet 5,0 medan `M_target` steg, så att den krävda
tillväxttakten nästan tredubblades utan att världens födotillgång ändrats.
**Alla tre frön dog ut** — nittio procent svält, med en dödsålder som sjönk mot
den nya mognadsåldern. Födelsetakten hade samtidigt fyrdubblats, från 0,010 till
0,040 per individmånad, så reproduktionen var löst; det var evolutionen som tog
beståndet.

Att magra ihjäl sig är att ha förlorat **mot sig själv**, inte att ha misslyckats
med en plan. Referensen blir därför kroppens högsta uppnådda massa, med en
avklingning på tolv månader — lång mot en säsong och kort mot ett liv, så att en
vinter inte skriver om vad kroppen är men ett år av tillbakagång gör det.

Tre följder:

  * **Ett löfte kan inte längre löna sig**, eftersom ingen mäter mot det.
  * **Straffet för långsam tillväxt blir det riktiga** — senare mognad och därmed
    färre kullar per liv, automatiskt och utan konstant.
  * **En juvenil straffas inte längre hårdast av alla** för att växa långsamt.
    Den är liten, inte svältande.

För en vuxen i jämvikt sammanfaller topp och förväntan, så
`starve_mass_ok_frac` och `starve_mass_crit_frac` behåller sin innebörd.
Skillnaden ligger nästan helt hos de unga.

`expected_mass` beräknas fortfarande — tillväxtens målkurva behöver den — men
den mäter inte längre kondition och delar inte längre ut skada.

Uppmätt i `liten6`, 400 tick, tre frön: 78/54/47 mot 0179:s 63/50/51, med 3/2/5
dödsfall mot 6/5/12. **Den evolutionära effekten syns inte där** — runawayen tog
10 000 tick i terrängvärlden — så det som är verifierat är mekanismen och
riktningen på dödligheten. Följ `A_mature`s median över en lång körning; den ska
sluta falla mot golvet.

### Kullstorleken blir en axel (0179)

Reproduktionen hade bara en sida: en förälder kunde välja hur mycket den
investerade, aldrig hur investeringen delades. En unge per parning och en
parning per avsvalning gav två till tre kullar per liv, alltså ett strukturellt
tak på **en till tre avkommor**, mot 1,0 som krävs bara för att hålla ett
bestånd.

**`child_M` blir kullens totala massa och `litter` delar den.** Massa per unge är
`child_M / litter`.

Den andra formen — `child_M` per unge och `litter` som multiplikator — byggdes
först och **prövades**: den gav en total fetal massa på 1,02 kg i en 1,2-kilos
kropp, dräktighet på tolv månader och **noll födslar i tre frön av tre**. Med
delningsformen är totalen oförändrad, så dräktighet, energikostnad och avsvalning
ligger kvar där 0178 satte dem.

Massan per unge blir 0,095 kg mot en vuxenmassa på 1,93, alltså **4,9 procent** —
mitt i vad en nyfödd hos ett litet däggdjur väger. Med en unge per kull var den
16 procent i median och över halva vuxenmassan för åtta procent av populationen.

**Och då dog varenda unge.** Trettionio av trettionio, av svält, vid ålder 0,0
månader. Orsaken var att `M_min = 0,14` inte bara är ett golv utan **själva
dödströskeln**: `if float(self.M) <= _M_min`. En unge på 0,095 kg föddes under
den.

Absolutformen var fel även för vuxna. En kropp med `M_target` 0,20 kg och en med
4,00 kg dog vid samma 0,14 — den ena vid sjuttio procent av sin vuxenmassa, den
andra vid tre och en halv.

Att magra ihjäl sig är att väga för lite **mot vad man borde väga**, och den
storheten finns redan: `_M_expected`, von Bertalanffy-kurvan från `child_M` mot
`M_target` vid `A_mature`. Tröskeln blir `M_waste_frac · M_expected(ålder)`. En
nyfödd väger per definition sin förväntade massa och klarar sig.

Uppmätt i `liten6`, 400 tick, tre frön:

```
          bestånd      födslar   dödsfall
0178      49 45 45     14  8 10   5  3  5
0179      63 50 51     29 15 23   6  5 12
```

**Födslarna fördubblades igen och beståndet växer i samtliga frön.**

**Två poster lämnas öppna, och båda är av samma sort.**

`M_waste_frac = 0,075` är valt så att den vuxna tröskeln blir 0,145 mot den gamla
0,140 — alltså **bevarad storlek, rättad form**. Sjuttiofem tusendelar av
förväntad massa är inte fysiologiskt; ett verkligt djur dör vid sextio till
sjuttio procent. Att sätta det värdet skulle döda beståndet, vilket prövades:
vid 0,50 föll det till 21/34/25. Hela mortalitetskalibreringen vilar på den gamla
absoluta 0,14 och måste flyttas som en enhet.

`child_M` är fortfarande **absolut** där den borde vara en andel av `M_target`:
0,16–0,40 kg oavsett om vuxenmassan är 0,20 eller 4,00. Åtta procent av
populationen investerade mer än halva sin egen vuxenmassa i en kull. Med
delningen är följden mildare än förut men axeln är fortfarande skev.

**Och en risk att mäta över lång tid.** Dödsfallen steg bara måttligt, alltså är
en unge på 0,095 kg fortfarande livskraftig. Motvikten mot stora kullar är därmed
svag, och `litter` kan pinna i taket. Följ dess median tillsammans med `child_M`
och `M_target` — tre axlar som frigjorts samtidigt kan driva åt håll ingen
förutser.

### Avsvalningen härleds ur dräktighet och laktation (0178)

`repro_cooldown_s = 8,0` månader bar **tre olika storheter**:

  *ålder vid könsmognad*, satt på den nyfödda. Den band mot `A_mature`, som
  spänner 5–20 månader, så **den nedre delen av den ärftliga mognadsaxeln var
  död** — arton procent av populationen bar ett värde under åtta och mognade
  vid åtta ändå.

  *intervall mellan kullar*, satt på den dräktiga föräldern vid födsel.

  *återhämtning efter parning*, satt på den andra föräldern — åtta månader för
  en part som inte bär fostret. **Den togs bort helt**, och det visade sig vara
  patchens tyngsta ändring; se nedan.

Dräktigheten är redan en härledd storhet, `child_M / gestation_growth_kg_per_s`,
så avsvalningen behöver ingen egen tidsskala. Den blir dräktighet plus laktation,
med laktationen som en multipel av dräktigheten (`lactation_k = 1,0`; för små
däggdjur ligger de i samma storleksordning).

```
child_M      dräktighet   avsvalning    (var 8,00 för alla)
0,16 kg          1,88        3,76
0,28 kg          3,29        6,59
0,40 kg          4,71        9,41
```

Uppmätt fördelning i populationen: p10 4,70, median 6,43, p90 8,91 månader.

**Och det är den intressanta följden.** Avsvalningen blir nu en **funktion av
ungens storlek**, som är ärftlig. En förälder som bygger en stor unge betalar
inte bara i massa utan också i tid — vid `child_M = 0,40` är avsvalningen längre
än den gamla konstanten. `child_M` går därmed från att vara en ren kostnad till
att vara en axel med två sidor.

Nyfödda får ingen avsvalning alls; `A_mature` är mognadsgrinden och är nu ensam
om det.

**Partnerns avsvalning togs bort, och det var den tyngsta ändringen.** Den bär
inget foster och ammar inte; dess kostnad är energin — fem procent av
energitaket — och den är redan tagen. Åtta månaders spärr hade ingen fysiologisk
motsvarighet.

Följden var värre än den såg ut. Parning kräver att **två** samtidigt är klara,
så att låsa båda parterna kvadrerar effekten på hur ofta det kan ske. Uppmätt i
`liten6`, 400 tick, tre frön:

```
                        bestånd      födslar   parningar
0177                    42 39 38      8  3  3   8  3  5
0178 med partnerlås     38 43 39      7  5  4   9  6  4
0178 utan partnerlås    49 45 45     14  8 10  15  9 10
```

**Födslarna mer än fördubblades — fjorton mot trettiotvå — och beståndet växte
över utgångsläget i samtliga frön.** Det är den första ändringen i hela serien
som flyttar reproduktionen, och kvoten födslar mot dödsfall gick från omkring
0,3 till över 2.

**Men taket är inte lyft, och en ny väg ut har öppnats.** Bäraren väljs på massa
— `if best.body.M > agent.body.M` — vilket är rimligt i en modell utan kön, men
blir en strategi nu när partnern inte längre låses: ett djur som håller sig lätt
hamnar alltid i partnerrollen, för sina gener vidare för fem procent av
energitaket, och blir aldrig dräktigt. **Selektionen mot `M_target` är därmed
något att mäta och inte att anta.**

Ett villkor som beror på **kondition** i stället för på ärftlig massa skulle
stänga vägen, eftersom kondition fluktuerar och inte går att ärva sig fri från.
Det är ett eget beslut och görs inte här.

Och fönstret är fortfarande två till tre kullar per liv à **en unge**.
Kullstorleken som ärftlig axel står näst i tur.

### Parningen kunde inte vinna, och sökte i fel säsong (0177)

Två fel som båda är samma sak: **ett tal som bär två storheter**.

**Motivationens tidskonstant lånade avsvalningstiden.** `parningsdrift` bygger
upp sig som `t/(t+τ)`, och `τ` var `repro_cooldown_s` — åtta månader. Ett djur
nådde alltså halv drivkraft först efter att ha gått oparat lika länge som hela
avsvalningen. Uppmätt blev tidsfaktorns median **0,188** medan kapacitetstermen
låg på **1,000**: anspråket hölls nere av tiden och av ingenting annat.

Följden är aritmetisk. Vid `λ = 2` och nivå 5 mot vardagens 6 måste parningen ha
styrkan **0,305** för att slå födosökets uppmätta 0,611. Medianen var **0,044**.
Anspråket kunde inte vinna, och det förklarar de 3 586 tillfällen i p176 då ett
djur såg en partner och betade i stället.

Avsvalningen är fysiologisk, motivationen beteendemässig och byggs upp över
brunstcykeln — ett par veckor för ett litet däggdjur. `repro_motivation_tau =
0,5 månader` är den storheten. Styrkan steg till 0,07–0,45 i median och
vinstandelen från 0,4 till 4–7 procent.

**Men det räckte inte, och mätningen sa varför.** Med starkare anspråk gick
"såg partner men parade inte" från 210 till 337 utan en enda extra parning.
Djuren närmade sig — *utanför parningsradie* föll från 48 till 36 procent — men
parade sig inte.

Skälet är att **styrningen och biologin läste olika villkor**.
`_mating_mode_slot` styr mot partner; `_ready_to_reproduce_slot` avgör om parning
sker, och den senare har tre villkor till. Uppmätt:

```
mjuk grind (styr mot partner)     23,4 % av agenttickarna
hård grind (tillåter parning)      7,7 %
båda i ett par samtidigt           0,59 %

av dem som styrde mot partner föll
  på säsongen                     65,4 %
  på massan                        8,9 %
  på energin                       0,0 %
```

**Två av tre djur som sökte partner gjorde det utanför sin parningssäsong.**
Säsongen är dessutom det enda av villkoren som är helt känt i förväg och som
inte kan ändras under en tick. Ett djur ska inte vara motiverat till det som är
omöjligt.

Säsongsgrinden bryts därför ut till `_i_parningssasong` och läses av båda. Efter
det faller noll procent på säsongen, och den mjuka grinden går på 7,7 procent i
stället för 23,4.

De två ändringarna hör ihop och kan inte göras var för sig: att höja styrkan utan
att smalna grinden ökade bara den bortkastade ansträngningen, vilket är mätt
ovan.

**Utfallet kan `liten6` inte upplösa.** Fyrahundra tick ger tre till åtta
parningar per körning, och skillnaden 42/39/38 mot 34/38/39 med 8/3/3 födslar mot
6/4/4 ligger under brusnivån. Det som är verifierat är mekanismen, inte effekten;
`f6-256` över tre frön och 3 000 tick krävs för det senare.

**Två poster lämnas öppna.** `repro_cooldown_s` bär fortfarande två storheter:
den sätts vid födsel, där den betyder *ålder vid könsmognad*, och efter parning,
där den betyder *intervall mellan kullar*. Åtta månader är rimligt för det
första och omkring fyra gånger för långt för det andra — dräktighet plus laktation
för en tvåkilos kropp är två till tre månader.

Och anspråkens styrkor spänner **0,044 till 0,955** i median på en deklarerad
0–1-skala: flykt och jakt ligger på 0,955 och graderar alltså inte alls, medan
parningen låg på 0,044. Modulens egen inledning förutsade precis det felet —
*"går en styrka 0–50 medan en annan går 0–1 gör trappan bara det skalskillnaden
redan gjorde"*. De deklarerade intervallen kontrollerades; de **realiserade**
fördelningarna mättes aldrig.

### Betningen skalade som `dt²` (0176)

Hollings typ II stod som `dt · a · B / (1 + a·h·B)` med `a` i enheten 1/månad
och `B` massan inom räckhåll. **Sedan 0172 är `B = täthet · svept yta` och ytan
växer med `dt`**, så uttrycket växer som `dt²`. Tiden räknades två gånger: en
gång i `dt` och en gång i den yta söktiden hunnit täcka.

Härledd ur tidsbudgeten med svept yta som sökmekanism blir formen

```
N = φ·B / (1 + h·φ·B/dt)
```

där `φ` är den **andel av beståndet inom den svepta ytan som betas av under en
passage** — dimensionslös, och därmed en egenskap hos djuret och inte hos
tidssteget. Vid liten `B` går intaget mot `φ·B`, vid stor mot `dt/h`, och båda
skalar som `dt¹`.

`φ = dt · a = 0,09` gör uttrycket **algebraiskt identiskt** med det gamla vid
`dt = 0,02`; verifierat till sista biten utom en avrundning på 2,2e-16 vid stora
`B`. Provet:

```
dt        intag normerat till dt = 0,02
        gammal     ny
0,020   1,1340   1,1340
0,010   0,8274   1,1340
0,005   0,5370   1,1340
```

Att halvera tidssteget tog tjugosju procent av intaget, att fjärdedela det
femtiotre. **Ett kadensbyte hade svultit populationen utan att någon mekanism
ändrats.**

Utfall i `liten6`, 400 tick, tre frön: 34, 38, 39 mot 0175:s 37, 40, 37, med
massa per individ 1,26, 1,34, 1,38 mot 1,36, 1,30, 1,49. Skillnaden är
flyttalsbrus som omfördelar banorna, inte en mekanismändring.

**Och ett dubblerat tal noterat.** `graze_handle_h` och `eat_rate` är samma
storhet: `1/90 = 0,0111` exakt, och kommentaren sade det själv — *h är satt så
att taket blir 1,8 kg per tick, vilket är den uppmätta maximala aptiten*.
Mättnadsasymptoten är alltså inte en hanteringstid utan aptiten skriven en andra
gång, och de två kan inte motsäga varandra eftersom den ena härleddes ur den
andra.

En verklig hanteringstid för ett litet betesdjur är omkring en timme per kilo —
bett på ungefär ett gram, ett bett i sekunden, plus tuggning. 0,0111 månader är
**8,1 timmar** per kilo. Rättas den blir hanteringstaket 14,6 kg per tick och
binder aldrig; djuret blir söktidsbegränsat över hela det relevanta intervallet
och aptiten blir det enda taket. Det är sannolikt rätt fysiologi — ett betesdjur
på god mark är magbegränsat, inte tuggbegränsat — men det är en beteendeändring
och hör ihop med `eat_rate` i en egen patch.

### Assimilationen straffade segheten två gånger (0175)

`assimilated_fraction` var `(1 − s) · digestion_efficiency(s) · d`. Den första
faktorn har redan tagit bort strukturmaterialet; att sedan sänka verkningsgraden
på det som återstår *därför att substratet var segt* är samma påstående en gång
till. De två sade dessutom oförenliga saker om samma material: `(1 − s)` att
strukturmaterial ger noll, `digestion_efficiency(s)` att rent strukturmaterial
ger 0,45.

```
s        (1−s)·dig(s)   (1−s)·0,80
0,000       0,800         0,800
0,500       0,313         0,400
0,567       0,260         0,346      <- florans uppmätta median
0,750       0,134         0,200
```

En herbivor tillgodogjorde sig **16 procent** av det den åt, mot 45–70 procent
för ett verkligt betesdjur på gräs. Det gjorde behovet till 0,54 kg per tick,
alltså sjuttiotre procent av kroppsvikten per dygn — och **hela födobudgeten är
kalibrerad mot det talet**: aptiten, den funktionella responsen, betesytan.

Rättat blir andelen 0,213 och behovet 0,40 kg per tick, alltså 55 procent av
kroppsvikten per dygn. Fortfarande högt, men det som återstår ligger i aptiten
och i den funktionella responsen, inte i magen.

Uppmätt i `liten6`, 400 tick, tre frön:

```
                bestånd        massa per individ     svält vinner
0174            34 37 41       1,28 1,24 1,30        16,7 12,2 12,9 %
0175            37 40 37       1,36 1,30 1,49        12,0  9,1  9,9 %
```

**Svälten faller med en fjärdedel till en tredjedel** och konditionen stiger.

**Och den fullständigare formen kunde inte tas i bruk.** `digestion_efficiency(s)`
är i själva verket redan tvåpoolsblandningen — `0,80·(1−s) + 0,45·s` är exakt
"av det labila tas 80 procent, av det strukturella 45" — och den ger 0,371, alltså
ett behov på 32 procent av kroppsvikten per dygn, mitt i det rimliga.

Den provades och föll på en premiss någon annanstans: **reserven är modellerad
som ren labil massa** och bokförs med `NUTRIENT_PER_KG_LABILE` per kilo. Så snart
upptaget innehåller strukturmaterial stämmer det inte, och näringsbalansen drev
2,1e-03 relativt mot invariantens 1e-06. Dessutom bygger `_excrete` på att allt
strukturmaterial passerar orört, vilket bara gäller när upptaget aldrig kan
överstiga den labila andelen — vid `s = 0,75` tar en generalist annars upp 0,331
av massan medan bara 0,25 är labil, och strukturmassa försvinner ur bokföringen.

Att låta magen komma åt segt material kräver alltså att reserven får en
strukturandel. Det hör till Steg 6, där matsmältningen blir en kapacitet med egen
underhållskostnad — och först då blir specialisering på segt material ett verkligt
val i stället för en universell nackdel.

### Rörelsen kostar transport, inte dragdissipation (0174)

`E_move` var `dt · F_prop · v / η`, alltså den effekt kraftbalansen dissiperar.
Det är rätt mekanik för en kropp i en vätska vid terminalfart och fel fysiologi
för ett djur som går: ett gående djurs kostnad domineras av inre arbete och är i
god approximation proportionell mot **massa gånger sträcka**, nästan oberoende av
farten.

**Formen hade fel tecken för allt som hindrar rörelse.** Höjt motstånd sänker
farten och därmed energin. Uppmätt vid full drivkraft:

```
fall               gammal J     ny J
platt, torrt        454 764     10 765
uppför (r = 1)      206 185     26 580     gammal: 55 % billigare än platt
vadande (w = 0,5)   134 057     42 372     gammal: 71 % billigare än platt
```

Vadandet hade fått en lapp för detta — ett additivt tillägg med kommentaren
*"den metabola kostnaden för rörelse sätts av muskelarbetet, inte av sträckan man
faktiskt tillryggalägger"*, alltså exakt den här principen tillämpad på
symptomet. **Lutningen hade samma fel och fick aldrig någon lapp.** Att gå uppför
var billigare än att gå på platt mark, och terrängens riktning för faunan pekade
åt fel håll sedan den infördes.

Formen är nu Taylor, Heglund och Maloiy (1982):

```
E = COT(M) · M · sträcka        COT = 10,7 · M^(-0,316)  J per kg och meter
```

och motståndet **höjer COT** i stället för att sänka farten. Kraftbalansen är
orörd och sätter fortfarande farten, så `fartskala`, `water_drag_gain` och
lutningstermerna fungerar som förut för rörelsen — de har bara slutat bestämma
energin.

Sträckan är **bansträckan**: födosökets vandring plus den riktade färden, samma
storhet som betesytan läser sedan 0172. Bägge skalar därmed med `dt` på samma
sätt.

Talen är metabola och inte mekaniska, så de divideras inte med en verkningsgrad
en gång till. `locomotion_eff` och `wade_cost` har därmed inga läsare kvar och är
borttagna.

Uppmätt föll `E_move` från 34 450 till 20 380 J per agenttick, alltså från
10,9 till 6,5 procent av underhållet — inom det spann en aktivt födosökande
organism lägger på rörelse. Fem meters gång för en tvåkilos kropp kostade 34 450 J
där fysiken säger omkring 60; nu betalas 873 meter.

Beståndet i `liten6`, 400 tick, tre frön: 34, 37, 41 mot 0172:s 37, 42, 37, med
massa per individ 1,28, 1,24, 1,30 mot 1,17, 1,17, 1,37. **Neutralt inom brus.**
Det var inte poängen: `E_move` låg i facket *fel form, rätt storlek*, och
patchen rättar formen utan att flytta storleken nämnvärt.

### Scenariers anatomi (0173)

`docs/scenariers-anatomi.md`. Vad ett scenario är, vilka sektioner det består
av, och vad som gör en körning ogiltig.

Den bärande regeln skrivs ut för första gången: **filen anger avsikter, koden
härleder tal.** Prövningen om något hör hemma i ett scenario eller i
parameterklasserna är om två olika värden kunde vara samtidigt riktiga i två
körningar — bördighet ja, cellarea nej.

Dokumentet bär också fyra planerade ändringar som alla följer av arbetet i
0169–0172:

- **`dt` ska stå i varje fil**, även när den är förvalet. Ticklängden är den
  enda parameter som ändrar innebörden av varje annan, och efter ett
  dygnstickbyte blir samma filnamn annars två olika världar.
- **Ett kalibreringsblock.** Fem kalibrerade tal flyttades under en enda
  arbetssession och inget av dem kan sättas per scenario. Det tvingar fram
  kodredigering för varje prov, alltså precis det kommandoradsflaggorna gjorde.
- **`fysiologi` delas.** Att skala en fysikkonstant och att låsa en ärftlig
  egenskap är olika ingrepp; `sociability` låser just den axel som skulle avgöra
  om flockningen är adaptiv, och det ska inte gå att göra av misstag.
- **`f6-256-mager`.** Vid `bordighet 4.0` är varje landcell sluten — `bar`
  20,1 % mot `hav` 20,0 % — och perceptets toppandel ligger på 0,185 mot
  jämnfördelningens 0,167. Bördigheten har ingen härledning i någon fil.

### Betesytan blir bansträcka gånger räckvidd (0172)

0171 satte betesgrannskapet till en cellradie. Det mätte bäst men var fortfarande
ett cellantal, alltså ett tal utan enhet som måste sättas om vid varje
geometri- eller kadensändring. 0172 ger det en enhet.

**Bansträcka och nettoförflyttning är olika storheter.** Ett betesdjur går ett
par kilometer på ett dygn och hamnar några hundra meter bort. Modellen känner
bara förflyttningen: `path_len` summerar per-tick-steg, så rakheten *inom* ett
tick är 1,0 per konstruktion medan den uppmätta över ett liv är 0,069. Allt som
skalar med sträcka var därför underprissatt — betesytan, mötesfrekvensen,
rörelsearbetet — och allt som skalar med förflyttning var rätt. Det är därför
felet kunnat sitta osynligt: halva modellen är korrekt.

Betesytan blir nu en kapsel:

```
A = 2 · r · L  +  π · r²
r = graze_reach_k · body_depth(M, struktur)      kroppsskalad, 40 cm vid 1,2 kg
L = forage_path_rate · dt                        868 m per tick = 1,4 km/dygn
```

och tillgången blir **täthet gånger yta** i stället för en summa över en
cellmängd. Grannskapet finns kvar men bara som fönstret tätheten skattas ur.

Talet är valt så att `A` blir 7,0 cellareor för den uppmätta medianmassan
1,2 kg, alltså 0171:s regim. Uppmätt i `liten6`, 400 tick, tre frön: 37, 42, 37
mot 40, 40, 40, med massa per individ 1,17, 1,17, 1,37 mot 1,30, 1,26, 1,29.
**Neutralt inom vad tre frön upplöser**, marginellt på den låga sidan.

Det som vunnits är inte beteende utan att ytan nu **följer av `dt`**: ett
dygnstick ger 1,64 gånger längre bana och därmed 1,64 gånger ytan, utan att någon
konstant sätts om. Dessutom en allometrisk axel som saknats — ytan växer som
`M^⅓` medan behovet växer som `M^¾`, vilket är varför stora betare behöver större
hemområden.

**Rörelsearbetet fick inte samma behandling, och det är patchens viktigaste
fynd.** `E_move = dt · F_prop · speed / η` är förflyttningens arbete. Uppmätt är
det redan **10,9 procent av underhållet** vid 0,476 cellbredders förflyttning per
tick. Skalat med bansträckans faktor 154 blir det tiodubbla hela budgeten och
allt dör.

Formen är alltså fel men storleken ligger rätt: fem till tjugofem procent på
rörelse är vad ett aktivt födosökande djur lägger. **Rätt storlek, fel form.**
Kommentaren om vadandet tio rader längre ner säger redan exakt detta — *"den
metabola kostnaden för rörelse sätts av muskelarbetet, inte av sträckan man
faktiskt tillryggalägger"* — men bara om korrektionstermen, inte om grundtermen.

Att rätta formen kräver att dragets konstanter prövas mot en verklig
transportkostnad; en tvåkilos kropp som går fem meter borde kosta omkring 60 J
och kostar 34 450. Det är ett eget arbete och står som öppen post.

### Betesgrannskapet är en räckvidd och inte en väg (0171)

0170 lät betningen följa organismens väg under ticket i stället för en cirkel
kring slutpunkten. Argumentet var att cirkeln var en heuristik och vägen en
fysisk storhet, och att den skärpta lokala utarmningen skulle göra världen
fläckig.

**Det höll inte, och två försök att rädda det höll inte heller.**

Uppmätt i `f6-256`, 3 000 tick, tre frön: beståndet gick från 22, 20, 21 till
**14, 9, 6**. Dödsorsakerna var oförändrade — omkring tre fjärdedelar svält —
men kroppen var kroniskt sämre: massa per individ 1,24 mot 1,05 och en reserv
som gick från 1,09 till 0,88 i stället för att plana ut. Ett tillförselunderskott,
inte en olyckshändelse.

Två hypoteser prövades i `liten6` över tre frön och föll båda:

```
variant                                bestånd      massa per individ
skiva, radie ceil(fart·dt)  (7024)     41 39 38     1,24 1,39 1,33
vägen                       (0170)     37 37 38     1,05 1,07 1,24
vägen + graze_search_a × 6             34 36 38     1,05 1,03 1,10
täthet × svept kapselyta               28 37 38     0,94 1,01 1,14
skiva, radie 1              (0171)     40 40 40     1,30 1,26 1,29
```

**Hollings tak var inte det bindande.** Att sexfaldiga `graze_search_a`, så att
halvmättnaden hamnade rätt i vägens geometri, ändrade ingenting i konditionen.
Aptiten är hungerstyrd, så en höjd gräns efterfrågas inte av den mätta medianen
— och den räddade inte heller svansen.

**Och den svepta ytan gjorde det sämre**, trots att den var räknad så att arean
motsvarade skivans sju cellareor. Skälet syns i utarmningsmätningen: den egna
cellen ligger på **0,54** av landets median och grannringen på **0,63**. En
täthet vägd mot vägen vägs mot den mest utarmade cellen av alla. Organismen betar
ner sig själv och lever på marginalen den når utanför.

Det skivan bar var alltså inte en tidsräckvidd utan en **rumslig**. Ett betande
djur sträcker sig efter födan; det äter inte bara det det trampar på. Radien blir
därför en egen deklarerad storhet, `graze_reach_cells`, i stället för
`ceil(fart · dt)` — den ska inte följa farten, och det var kopplingen till farten
som gjorde 0170 möjlig att tänka.

Att radien måste vara en hel cell när en tvåkilos kropps fysiska räckvidd är en
halv meter är **rörelsedeficitet uttryckt i födobudgeten**: organismen sveper
fem meter per tick där ett verkligt betesdjur går ett par kilometer per dygn, och
aptiten är kalibrerad mot den för höga ytan. Två fel som tar ut varandra, och de
hör hemma i kadensarbetet.

0169 står kvar orörd och mår bra av det: `toppandel` 0,205 och vid kust 0,232 med
det återställda grannskapet.

### Betningen följer vägen (0170)

`reach = ceil(fart · dt)` gav en cirkel med heltalsradie kring **slutpunkten**.
Den var fel åt två håll samtidigt: för stor, eftersom ett steg på en halv
cellbredd avrundades upp till radie ett och alltså sju celler, och osymmetrisk,
eftersom cirkeln låg kring slutpunkten och inte längs vägen.

Betningen läser nu organismens **uppehållsfördelning**: banan under ticket är en
sträcka från utgångspunkten till den skrivna positionen, och tiden i en cell är
sträckans andel i den. Vikterna summerar till ett av konstruktion — samma regel
som all annan transport i modellen, en uppdelning av ett lager.

Det är den första verkliga konsumenten av fördelningsrepresentationen, och det
gör betningen **mer** lokal, inte mindre. Farhågan var motsatt: en fördelning som
smetas ut betar medelfältet och förstör den lokala utarmning som driver rörelse.
Med ett halvt cellsteg täcker vägen en till två celler mot cirkelns sju, så
riktningen är den omvända.

**Men Hollings funktionella respons räknar sitt tak över den tillgängliga massan
inom räckhåll**, och räckhållet krympte sjufalt. `graze_search_a` och
`graze_handle_h` kalibrerades mot cirkeln. Uppmätt faller `toppandel` med
ungefär 0,011 i samtliga tre frön och `svält` vinner oftare — 18,7 mot 17,2
procent av tickarna i frö 1. Beståndet: 37, 37, 38 mot 0169:s 32, 39, 39.

**Geometrin är rättad, kalibreringen är det inte.** Talen ovan ska inte läsas
som att ändringen är dålig utan som att den frilade ett tal som var bundet till
en form som inte längre finns. Rätt plats att sätta det är mot betningens egen
mätning av intag per tick, och det hör till en egen patch.

### Driften blir en uppehållstid (7024)

Farten `Q/djup` hade tre fel som följde av varandra: den är obegränsad när
djupet går mot noll, den kräver därför ett tak, och taket är inte en parameter
utan en förbindelse. Uppmätt band taket i **varje** fårhändelse — formeln bidrog
med ingenting utom sin mättnad.

Värre var att steget lades ut som ett **rakt streck** mot nedströmscellens
centrum. Riktningen gäller en cell, längden gällde hundra. Det är den
kombinationen som slängde djuren rakt ut i havet flera celler från land: havet är
aldrig med i flödesgrafen — `flow_to` är -1 där, verifierat 13 107 av 13 107
havsceller — så de hamnade utanför nätet i en riktning nätet inte pekade. Den
tidigare diagnosen "havet behandlas som en fortsättning på fåran" var alltså
fel; havet var redan en sänka, och felet låg i geometrin hos steget.

Storheten är nu cellens **uppehållstid**, `τ = V/Q` med `V = djup · area` och
cellarean 1. `discharge` är en volym per tick, så τ är ett antal tick och
behöver ingen omräkning. Kroppen får en tidsbudget skalad med `buoyancy` och
vandrar längs `flow_to` medan budgeten räcker.

Uppmätt i `liten6`, 400 tick, tre frön, jämfört med samma instrument:

```
                   steg max      slutade
frö   före  7023   7024      hav  strand  lugn  budget    bestånd
 1    142,8 142,8  12,9     10,0   28,5   23,5   38,0     40 -> 41
 2    143,1 143,1  16,1      5,1   37,0   23,9   34,1     40 -> 39
 3    141,2 138,9   6,9      3,9   37,9   26,2   32,0     40 -> 38
```

**Maxsteget faller från 143 till 13 cellbredder utan att något tak finns kvar.**
`drift_max` är borttagen och `drift_gain` är en dimensionslös etta i stället för
en felaktig enhetsomräkning. Längsta vandring 11–18 celler mot skyddsnätets
4 096; nätet är acykliskt och gränsen nåddes aldrig.

**Tre fysiska utfall uppstår utan att kodas.** Ungefär en tredjedel av
vandringarna **strandar** — fåran blir grundare än kroppen nedströms och den
lämnas där, alltså sköljs i land utan en regel om det. En fjärdedel slutar i
**lugnvatten**, som är sjöarna: stort magasin mot måttligt genomflöde ger lång τ
och budgeten tar slut nästan direkt. Fyra till tio procent når **havet**, och då
vid mynningen, eftersom vandringen bryter där nätet gör det.

Beståndet gick 41, 39, 38 mot 7023:s 30, 38, 38 och utgångsläget 35, 36, 30. Tre
frön räcker inte för en slutsats, men riktningen är åtminstone inte fel, och
`liten6` kollapsar av egen kraft.

**Det som inte är gjort.** Vandringen är deterministisk och använder
uppehållstiden som väntevärde snarare än att dra ur `p = 1 − exp(−r·dt)`.
Skillnaden syns först om spridningen mellan individer i samma fåra ska betyda
något. Formen är den samma och bytet är lokalt.

### Kroppen får ett djupmått (7023)

`water_drag_depth_ref` stod på 0,20 längdenheter, alltså **två meter** — den
djuplek där en tvåkilos organism först räknades som helt nedsänkt. Fårorna i
modellen är sjutton centimeter djupa, så mättnaden nåddes aldrig. Samtidigt
grindade driften på `submerged_threshold`, 1e-6, alltså en mikrometer. **Samma
kropp mötte vattnet på två skalor som skilde tvåhundratusen gånger.**

Skalan är nu härledd och inte vald: `phenotype.body_depth` ger `L = (M/ρ)^(1/3)`
ur massa och täthet, som båda redan fanns. För modellens median — 2 kg vid
strukturandel 0,567, relativ täthet 1,177 — blir djupet 0,0119 längdenheter,
alltså **tolv centimeter**. Det skalar rätt: ett åtta gånger tyngre djur behöver
dubbelt så djupt vatten. `LENGTH_UNIT_M = 10.0` får samtidigt ett namn; talet har
funnits som text i `docs/varldens-skala.md` och som underförstådd omräkning i
kommentarer.

Tre läsare delar måttet, och det är patchens hela innehåll: draget och
värmeledningen i `_water_factor`, styrningens kostnadsterm i `_kostnad_vag`, och
driftens flytkraftsgrind i `_drift_system`.

**Sjökantens singularitet är borta vid källan.** En strandcell har per definition
djup mot noll, och `q/djup` gav där uppmätt 51 106 cellbredder per tick. En cell
grundare än kroppen kan nu inte drifta alls, och divisorns golv är en fysisk
storhet i stället för ett epsilon.

Uppmätt i `liten6`, 400 tick, samma frön före och efter:

```
frö   flytande        steg median      bestånd
      före  efter     före  efter      före      efter
 1    0,802 0,435     6,0   23,0       40 -> 35  40 -> 30
 2    0,914 0,381     9,6   22,9       40 -> 36  40 -> 38
 3    0,972 0,277     3,8   13,5       40 -> 30  40 -> 38
```

**Driften halveras till en tredjedel.** Andelen agenttick med drift faller från
0,80–0,97 procent till 0,28–0,44.

**Men medianen på steget stiger, och det är komposition och inte försämring.**
De händelser grinden tar bort är de grunda, och grunt vatten i den här världen är
sjöar där genomströmningen är noll — alltså små steg. Kvar står fårorna, där `q`
är stort. Maxsteget är oförändrat 143 cellbredder, eftersom taket binder både
före och efter. **7023 gör driften sällsyntare, inte mindre.** Magnituden hör till
uppehållstidspatchen och är orörd här.

Beståndet är inte avgjort: 35→30, 36→38, 30→38 över tre frön. Vattnet blir
genuint farligt när mättnaden nås sjutton gånger tidigare, och `liten6` kollapsar
dessutom av egen kraft. Talet ska tas i `f6-256` över flera frön innan något
påstås.

### 7009 fungerar men har inget att verka på

Tätheten härleds ur strukturandelen, `buoyancy` får sin första skrivare och läsare, och draget i vatten skalas med `(1 − buoyancy) · djup`. Kalibrerat mot faunans **uppmätta** strukturspann — p5 0,211, median 0,567, p95 0,684 — vilket ger flytförmåga från 0,885 till 0,255 över den fördelning som faktiskt finns. Ett första försök med ändpunkterna 0,97 och 1,90 gav flytförmåga exakt noll för varje strukturandel över 0,5, alltså en död halva av axeln.

**Men noll av fyra djur befinner sig i vatten efter 4 000 tick.** Vattnet är 22 procent av världen, men merparten är polarhav där klimatet ändå är obeboeligt; sjöar och fåror är omkring tre procent. Läsaren finns alltså men fyrar aldrig, och mekanismen går inte att döma.

**Och faunan bär inte terrängvärlden.** Uppmätt i `f5-terrang` vid 64x128: 20 grundare blir 1 till 4 individer efter 4 000 tick, mot 28 i den platta `f4-start20` vid tick 1 200. Det är en större sak än vattenaxeln och bör redas ut först — vattnet kan inte differentiera en population som inte finns.

### Terrängen fick en riktning för faunan

Höjdens gradient är statisk och förberäknas en gång, så den kostar två uppslag och en skalärprodukt per djur och tick. `Grid` exponerar grannarnas riktningar som enhetsvektorer — ett geometriskt faktum som hör hemma där — och gradienten tas som en viktad summa över grannringen, giltig för vilken cellform som helst.

Uppmätt medelfart mot lutning i färdriktningen:

```
lutning        n      fart
< -0,035     1 721    55,2
-0,035..-0,009  3 440    46,7
-0,009..+0,011  3 441    39,2
+0,011..+0,028  3 441    32,4
+0,028..+0,053  3 440    26,8
> +0,053     1 721    19,7
```

Korrelation −0,573. Asymmetrin mellan `climb_gain` 1,5 och `descend_gain` 0,5 är fysiologi och inte en avvägning: koncentriskt muskelarbete uppför kostar ungefär tre gånger excentriskt nedför.

**En mätfälla värd att veta om.** Ett första försök mätte andelen tick som djuren pekade nedför och fick 40,5 procent *med* kostnaden mot 51,6 utan — alltså till synes tvärtemot. Det är ett samplingsartefakt: ett långsamt djur bidrar med fler mätpunkter medan det pekar uppför. Tidsviktade och sträckviktade mått på rörelse går isär så snart farten beror på riktningen, och det gäller varje framtida mått på var djuren befinner sig.

### Vattnet fick två element med var sin kostnad

7009 gav bara halva asymmetrin. Tre fel rättade:

**Vatten var en fälla, inte en fara.** `E_move` är dragkraftens dissipation vid kraftbalans, alltså proportionell mot den *uppnådda* farten. Höjt drag sänkte farten och därmed energin — ett landdjur som vadade brände mindre per tick än ett som sprang. Den metabola kostnaden för rörelse sätts av muskelarbetet, inte av sträckan, så vadandet har nu ett eget tillägg proportionellt mot pådraget och nedsänkningen.

**`water_heatloss_gain` deklarerades i 7009 och kopplades aldrig.** En parameter utan läsare, införd av samma slags förbiseende som projektet gång på gång hittar hos sig självt. Nu multiplicerar den värmeledningen: vatten leder värme tjugofem gånger snabbare än luft, vilket är det verkliga skälet att nedsänkning dödar en jämnvarm organism.

**Passiv drift ger den flytande änden sin nedsida**, och den behövde inte uppfinnas: strömmen går till havet, och havet ligger vid noll grader.

**Strömhastigheten kommer ur kontinuitetsvillkoret**, `Q = A · v`. Cellarean är 1, så tvärsnittet är vattendjupet och hastigheten blir `Q / djup` i cellbredder per tick — `discharge` är redan en volym per tick, så ingen `dt` ska in en gång till. Ett mellanliggande försök skalade i stället med lutningen; det gav rätt kvalitativt utfall men ersatte en bevarandelag med en heuristik, och dolde att `dt` räknades två gånger.

Utfallet faller ut utan att kodas:

```
fåror   60 celler   hastighet median 2,00 (mättar mot taket)
sjöar  153 celler   hastighet median 0,34
faunans egen fart               0,48 cellbredder/tick
```

**Sjöar är simbara, floder är enkelriktade.** Och eftersom sjöarna är sedimentfällor sedan 7008 ligger födan just där vattnet står stilla. Den akvatiska nischen är alltså sjöar; floderna är rutschbanor till havet.

Taket på 2,0 binder i fåror och är inte ett numeriskt undantag utan en utsaga om det som ligger under cellskalan: en verklig flod har stränder och bakvatten där en organism kan hålla position, och en cellmedelhastighet överdriver hur obönhörligt strömmen sveper.

### Världens skala är fastställd — och latituden faller

Se `docs/varldens-skala.md`. **En cell har arean 100 m², längdenheten är 10 m.**

Tre oberoende mått på cellstorleken: florans biomassa säger 2,6–26 m, faunans täthet 4,1–18 m, faunans fart 1 250–6 250 m. De två första kommer från skilda delar av modellen och överlappar; farten är den ensamma avvikaren och är omkring tvåhundra gånger för låg.

Konsekvenserna, som alla är verkliga:

**Latituden är död.** Jordens meridionella gradient är 0,003 °C/km. Över 4096 rader, alltså 38 km, ger det 0,11 grader. Modellen har trettio. Klimatbanden, polarhavet och kontinentallutningen vilar alla på ett antagande som inte håller.

**Årstiden överlever oförändrad.** Den beror inte på skalan. Det som försvinner är att dess amplitud varierar med latitud.

**Höjdgradienten och kalluftsdräneringen överlever och är starkare.** Vid 6,5 °C/km ger 300–500 meters relief två till tre grader, alltså tjugo gånger latitudens bidrag. Kalluft som samlas i sänkor ger inversioner på 5–10 grader och har dessutom *motsatt* tecken mot lapse rate, vilket lapse rate inte kan ersätta.

**Havet överlever.** Det definieras av basnivån och inte av polen: allt annat dräneras till det, det dräneras ingenstans. Den definitionen är skalfri, och `_ocean_mask` klassar redan efter höjd. Bara *placeringen* var latitudberoende. Uppmätt med havet placerat som bred bassäng och ingen latitud alls: 27 % hav, 6,7 % sjö, floder på 2 012 celler, noll strandade celler. Bassängens mjuka kant är dessutom den regionala lutningen, så kontinentallutningen generaliseras från "mot polerna" till "mot basnivån" — vilket är vad den alltid var. Näringssänkan består; den döps om från export till sedimentation.

**Faunans fart är inte löst.** Att höja den tvåhundra gånger går inte — taket är omkring en cellbredd per tick, alltså femtio gånger, och även det är tio gånger under det realistiska. Antingen kortas tidssteget, eller så accepteras att faunan är långsammare än sin kropp motiverar, eller så byts kroppsstorleken. Det är den viktigaste öppna frågan i modellen.

**Lapse rate kan inte sättas förrän skalorna är ense.** Patch 7010 var påbörjad och lades åt sidan av det skälet: att kalibrera ett tal mot en enhet vi vet är trasig är samma fel som lutningen hade så länge höjden saknade enhet.

### Klimatet blev en tidsprofil (7013)

Latituden är borta ur världsmodellen. `T_band` med längd `n_bands` är en skalär `T_air`, och `T(t) = T_mean + T_amp · sin(2π t / year_len − season_phase0)`. Bandmaskineriet i klimatet, `g_band`, `_cell_bands` och soil-kärnans två arrayargument föll med den.

**Årstiden hade följt med i fallet om formen bara kollapsat.** Säsongstermen var `lat · sin(fas)`, alltså proportionell mot latituden, så en konstant latitud hade tagit årstiden med sig. Amplituden är därför en egen parameter nu, vilket den alltid borde ha varit.

**Grindvektorn per band hade noll läsare.** Den räknades varje tick och skrevs till `g_band`; varje faktisk läsare gick via `_gate_from_T()`, eftersom grinden behövs per cell så snart höjdmodifieraren finns. Samma sorts fynd som `water_heatloss_gain` och `cold_aversion`: en storhet som beräknas och aldrig konsumeras.

**Nivån är ett val, och den valdes till ett tempererat inlandsklimat**: årsmedel 11 grader, januari −1, juli 23. Alternativet var att bevara den gamla världens landmedel på 20,1 och därmed dess produktivitet. En värld där tillväxten aldrig stannar har ingen vinter att överleva, och årstiden blir en modulering i stället för en ekologisk kraft.

Priset är mätt och verkligt:

- **Termokostnaden stiger 53 procent i medel.** `P_need = K · (Tb_set − T_env)` är linjär i skillnaden, som går från 16,9 till 26,0 grader. På vintern är den 124 procent över dagens medel.
- **Fröbanken saknas fortfarande**, vilket gör en vinter farligare än den vore i en färdig modell. Vid amplitud 12 bottnar tillväxtgrinden på 0,42 och stannar aldrig helt, vilket är den försiktiga vägen tills fröbanken finns.

**Florans respons hänger inte på klimatets nivå.** Den är gaussisk kring individens `temp_opt`, en evolverande trait med spannet −5 till 35 grader, medan `T0/T1`-grinden i praktiken bara styr evapotranspirationen. Uppmätt i `f5-terrang`, seed 2, 3 000 tick: 64 025 plantor i det kalla klimatet mot 53 118 i det gamla. Floran klarar sig alltså **bättre**, tvärtemot vad en kalkyl på tillväxtgrinden skulle förutsagt.

**Nederbördens referens var en kvarleva.** `rain_T_ref = 25` låg nära den gamla världens beboeliga mitt; i en dalgång på 11 grader hade samma referens halverat regnet utan att någon valt det. Referensen är nu världens årsmedel, och `rain_base` satt så att **ariditeten bevaras**, alltså P/PET och inte P: ett kallare klimat har lägre potentiell avdunstning, och att hålla nederbörden konstant medan PET faller vore att smyga in en våtare värld i en klimatpatch. Talet är `0,494 · 0,532/0,821 = 0,320`.

Uppmätt vid 3 000 tick, `f5-terrang`, seed 2:

```
rain_base    fåror    sjömagasin   markfukt   flora    fauna
0,65 (före)  0,58 %      3,23        0,81     53 118     2
0,494        3,06 %      7,09        1,00     66 740     0
0,400        0,22 %      0,42        0,97     65 978     0
0,320        0,00 %      3,92        0,84     64 025     0
```

**Sjömagasinet är icke-monotont i nederbörden** — 0,42 vid 0,400 mot 3,92 vid 0,320 — så en ögonblicksbild av ett tröskelberoende mått skiljer inte kandidaterna. Talet står därför på principen och inte på mätningen, och räknas om när 7014 ger klimatet sin nivå.

**Faunan dog i alla fyra körningarna, referensen inräknad** — den gick från 20 grundare till 2 individer. Klimatsänkningen påskyndade det men orsakade det inte: `f5-terrang` bar inte faunan innan heller. Se den öppna frågan nedan.

**Bitidentiteten för `f4-start20` upphör här, och det är riktigt.** Kravet gällde att en platt värld inte ska påverkas av terrängburna mekanismer, och det står. Men klimatet är inte terrängburet: latituden var fel i varje värld, inte bara i dem med höjdskillnader. Att behålla den för platta scenarier vore att bevara just det maskineri som ska dö. Ersättningen är ett **ekvivalenstest**: med klimatet gjort rumsligt och tidsmässigt konstant i båda ändarna — `dT_pole = A_eq = A_pole = 0` före, `T_amp = 0` efter, samma medeltemperatur — är 1 500 tick av `f4-start20` bitidentiska, samma `md5` på `life.jsonl` och identisk konsolutskrift. Det visar att ingen annan väg än klimatets form har rubbats. `f4-start20` behöver en ny baslinje; de gamla talen i det här dokumentet är förklimatkollaps.

### Klimatet fick en position (7014)

`T_mean` och `T_amp` var valda tal. De är nu följder av **var världen ligger på en jordlik planet**: `latitud` och `kontinentalitet`, två tal i scenariot. Regeln ligger i `klimat.py`, som är fysiklager — den äger inget tillstånd och rör inga cellfält. Världslagret tillämpar den en gång vid världens tillkomst, och därefter är klimatets tre tal konstanter under hela körningen.

Det är inte en återkomst av latitudgradienten. Världen ligger *på* en breddgrad — den spänner inte över flera — så latituden är en `WorldParams`-skalär och inte ett cellfält. `Grid` skulle aldrig ha ägt `cell_lat`.

**Anpassningen.** Sjutton stationer från Singapore till Jakutsk, var och en höjdkorrigerad till havsnivå med 6,5 grader per kilometer innan anpassningen — utan den skulle Ulan Bators tolvhundra meter räknas två gånger när modellens egen lapse rate lägger på höjden.

```
T_mean = 25,93 - 49,20*(phi/100)^2 + k*(4,83 - 50,35*(phi/100)^2)    residual 1,96 °C
T_amp  = (4,51 + 21,73*k) * sin|phi|                                  residual 2,13 °C
eftersläpning = 2,0 - 1,0*k månader
```

Residualen är storleken på det en position inte kan veta: havsströmmar, molnighet, regnskuggor. Utfallet:

```
 lat      k=0,0            k=0,5            k=1,0        (T_mean / T_amp)
   0    25,9 /  0,0      28,3 /  0,0      30,8 /  0,0
  20    24,0 /  1,5      25,4 /  5,3      26,8 /  9,0
  40    18,1 /  2,9      16,4 /  9,9      14,8 / 16,9
  55    11,0 /  3,7       5,8 / 12,6       0,6 / 21,5
  66     4,5 /  4,1      -4,1 / 14,0     -12,6 / 24,0
```

**Kontinuiteten föll ut av sig själv.** 7013:s handvalda 11,0 / 12,0 motsvarar latitud 47,9 och kontinentalitet 0,54. Förvalet är därför 48,0 / 0,55, vilket ger 10,9 / 12,2 — patchen ändrar klimatets *härledning* utan att flytta dess *värde*. Med de tre talen tvingade till 7013:s och eftersläpningen till noll är 800 tick av `f4-start20` bitidentiska, samma `md5` på `life.jsonl`.

**Tre saker som faller ut utan att kodas:**

- **Södra halvklotet.** Negativ latitud inverterar årstiden, exakt sex månader förskjuten. Det följer av latitudens tecken och kostar en multiplikation.
- **Termisk eftersläpning.** Varmaste månaden ligger efter solståndet, en månad i inlandet och två vid kusten, eftersom underlaget lagrar värme. Den gör `t = 0` till astronomisk vårdagjämning i stället för till termiskt medel: startklimatet är nu 2,6 grader och stigande i stället för 10,9. Sådden sker alltså i sen vinter, vilket är rätt tid att så.
- **Nederbördens referens.** `rain_T_ref` var ett fast tal och är nu världens egen årsmedeltemperatur. Ett fast tal var det som gjorde 25 grader till en kvarleva; flyttas världen norrut utan att referensen följer med torkar den ut av ett tal ingen valt.

**Vad som medvetet inte gjordes.** Fotoperioden och dygnsamplituden hör hit fysiskt men saknar läsare, och en mekanism utan konsument är precis det fel projektet hittat hos sig självt fyra gånger — `water_heatloss_gain`, `cold_aversion`, `nutrient` som allokerades och aldrig rördes, `g_band`. Longitud ger dessutom ingenting utan en karta över planeten och infördes inte alls.

**Kända svagheter i formeln**, dokumenterade i `klimat.py` så att de inte upptäcks på nytt: maritim amplitud på hög latitud underskattas — Bergen har 6,8 där formeln ger 3,9, eftersom få stationer är både kustnära och nordliga — och kontinentalitet nära 1 vid ekvatorn extrapolerar utanför underlaget.

### Lapse rate blev fysik (7015)

`lapse_rate` var 16,0 grader per höjdenhet. Den är nu **0,065**, alltså jordens 6,5 grader per kilometer uttryckt i modellens längdenhet om tio meter. Skillnaden är tvåhundrafemtio gånger.

Det gick inte att veta förrän skalan fanns. Talet valdes mot vad det gjorde med tillväxtgrinden inom ett latitudband — alltså mot en storhet inuti modellen — eftersom höjden saknade enhet och det inte fanns något utanför att jämföra med. Det är precis den fällan manifestets nya avsnitt 7 beskriver: en storhet utan enhet går inte att sätta fel på ett sätt som märks.

**Referensytan mot bandets landmedel är borttagen.** Den var en lapp på att terrängens kontinentala lutning följde latituden och därmed åt upp klimatgradienten — mätt mot rå höjd tappade ekvatorn 7,3 grader medan rad 32 tappade 2,1, och klimatet inverterades. Med latituden borta finns varken gradienten att äta upp eller banden att mäta mot, och referensen är havsnivån. Det tar också bort `_build_T_offset`s sista beroende till `Grid`s bandmaskineri; kvar är bara terränggeneratorns `cell_lat`, som dör i 7017.

**Höjdgradienten är nu försumbar, och det är riktigt.** Uppmätt i `f5-terrang`: −0,1 grader på högsta punkten mot −7,0 före. Dagens relief är omkring två längdenheter, alltså tjugo meter, och tjugo meter *ska* inte kyla mätbart. Att den gjorde det förut var felet, inte att den slutat.

Verkan kommer med 7016. Ett kuperat landskap på tio till fyrtio kilometer har tre- till femhundra meters relief, och då blir bidraget två till tre grader — tjugo gånger mer än latituden någonsin kunde ge vid den här storleken.

### Regionen och omlandet (7016)

Se `docs/regionen-och-omlandet.md`. Frågan var vad världens rand ska vara när
den har en position på en planet och därmed inte gärna kan wrappa in i sig
själv. Svaret är att frågan var fel ställd: det som saknas är inte en rand utan
ett **omland**.

Den detaljerade världen blir en **region** i ett grovt rutnät av regioner utan
inre upplösning. Det är one-way nesting, som klimatmodeller gjort i decennier.
Tiotusen regioner med tio fält vardera är åttahundra kilobyte — lagret är
gratis, det är kopplingen som kostar.

Tre saker det löser:

**Hydrologin har inget uppströms.** Regionen är i dag hela sitt eget
avrinningsområde, och en flod kan därför aldrig bli större än regionens egen
nederbörd. För ett landskap på tio kilometer är det nästan alltid osant. Ett
tal per randcell rättar det.

**Latituden återuppstår på rätt skala.** Tre grader kräver tusen kilometer, och
hundra gånger hundra regioner à 1024² är elvahundra. Latituden dog som
cellfält, vilket var rätt, och kan komma tillbaka som regionfält, vilket också
är rätt.

**Genflödet.** Effektiv populationsstorlek fem är projektets tystaste problem.
En migrant per generation räcker enligt Wrights tumregel för att hindra
delpopulationer från att driva isär — flödet får alltså vara mycket glest och
ändå verka.

Fyra faror, alla dokumenterade: två biologier om grovregioner får evolvera
(gränsen dras vid *produktivitet, inte population*), tvåvägskoppling som ger
artefakter, massbalansen som kräver konton och inte bara egenskaper, och nya
skalfel — en region är 1 048 576 gånger en cell, och varje flöde över gränsen
måste bära den kvoten.

**Ordning:** statiskt omland envägs → konton → migration → tvåvägs endast om
mätning kräver det.

**Två följder för arbetet som pågår.** Torusen behålls tills nivå 1 — ett
periodiskt randvillkor är ett ärligare provisorium än en kant utan andra sida.
Och havet i 7018 blir delvis ett provisorium, eftersom basnivån med ett omland
ligger utanför regionen; det ska därför byggas som **en form bland andra** och
inte som ett antagande inbakat i dräneringskoden. Skillnaden avgör om flytten
senare kostar en parameter eller en ombyggnad.

### Landformerna och regionens form (7017)

Två riktningar fästa i dokumentation innan de hinner glida. Ingen kod.

**Sjöar styrs genom sin tröskel, inte genom sin grop.** En sjös yta bestäms av
var den spiller över, så att höja tröskelcellen breder ut sjön längs den
omgivande terrängens egna nivåkurvor. Mekanismen blir *hitta och förstärk*:
generera bruset, kör dräneringen som ändå körs, läs ut sänkorna, välj den som
bäst matchar och höj dess tröskel mjukt. Det är styrning av något emergent i
stället för en form kodad uppifrån, och det är billigt — några celler i stället
för hundratals.

Önskemålet uttrycks i **hektar**, som är fysiskt och skalfritt. Uppmätt vad
bruset gör av sig självt i `f5-terrang`: elva sänkor på 4,15, 2,39, 1,11 och
0,33 hektar plus sju små. Generatorn ska kunna **säga nej** när önskemålet
överstiger vad bandet kan bära, och fröval är en legitim del av mekanismen.

**Kusten genereras redan men syns inte.** Formerna adderas till bruset i stället
för att maskera det, så kustlinjen är en brusstörd kurva — men korrugeringen är
uppmätt till 1,09, alltså praktiskt taget en cirkel, eftersom bassängkantens
lutning är 0,150 mot brusets 0,019. Kustens karaktär är kvoten mellan de två:
under en femtedel ger cirkel, över ett ger vikar och öar. **En flack och bred
bassäng ger bättre kust än en djup och smal**, och talet ska i 7018 väljas mot
korrugeringen och inte bara mot havsandelen.

**Regionerna sitter i ett hexagonalt gitter men är romber.** En hexagon kaklar
inte av hexagoner — rot-7-hierarkin vrider drygt nitton grader per nivå. En
60-graders romb i axialkoordinater kaklar exakt och bildar ett triangulärt
gitter, som har sex likvärdiga grannar. Formen är en romb, mönstret är
hexagonalt, och det är mönstret som bär isotropin. Omlandet blir därmed
bokstavligen ett `Grid` till.

**Bedömt mot Numba och GPU innan beslutet**, eftersom en geometri som bryter
accelerationen inte är värd sin isotropi. Romben är sannolikt *snabbare*: med
`id = q + r*m` blir grannarna sex konstanta förskjutningar, radpariteten
försvinner, och `neighbor_idx` — 24 byte per cell, alltså 6,3 MB vid 262 144
celler och därmed större än L2 — behöver inte längre läsas i `transport_pass`.
Florans GPU-väg berörs inte alls. Påståendet ska mätas och inte antas, av samma
skäl som parallelliseringen av tillväxtkärnan drogs tillbaka.

**Reliefen fick sin fysik rättad på vägen.** Ett tidigare förslag var att låta
amplituden skala med världens storlek vid fast lutning. Det är fel: verkliga
landskap är självaffina och inte självsimilära. Skandinaviska halvön är 1 800 km
lång och 2,5 km hög, alltså 1:700 i medellutning, medan en enskild dalsida är
1:3. Anpassning mot relief-mot-skala ger **Hurstexponenten H = 0,64** och 37
meters relief per kilometer.

Modellen ligger i dag på motsatt ytterlighet: den analytiska normeringen tvingar
fältets standardavvikelse till `noise_sd` oavsett vilka våglängder som ingår,
alltså **H = 0**. En värld på hundra kilometer får samma tjugo meters relief som
en på en kilometer.

```
     värld    λ_max km   H=0 (idag)   H=0,65   H=1 (förkastat)
    64x128       0,52       10 m        10 m        10 m
   512x512       4,13       10 m        39 m        80 m
 1024x1024       8,25       10 m        61 m       160 m
 4096x4096      33,0        10 m       149 m       640 m
```

Följden är att **skaldokumentets löfte om två till tre graders höjdgradient inte
kan bäras av bruset** — inte ens 4096² ger mer än 0,97 grader. Det är riktigt:
femhundra meter på tio kilometer är en tektonisk struktur och inte en eroderad
yta. Höjdgradienten kommer därför från en placerad form med en höjd i meter, och
**placerade former skalar inte med världen** — ett berg blir inte högre för att
kartan blir större. Radien ska av samma skäl inte normeras; positionen är var på
kartan, radien är hur stor i meter.

### Havet placerat efter höjd (7018)

Polarsänkan och kontinentallutningen är borta. Båda var funktioner av
`grid.cell_lat`, och terrängen är därmed det sista stället i simuleringen som
läste latituden. Kvar står **en enda förvald form: havet**.

**Havsandelen styr, inte radien.** En radie är inte skalfri: som andel av
bredden gav den 21 procent hav vid 64x128 men bara 12 vid 64x256, eftersom en
cirkel inte når hörnen i en avlång värld. Radien löses därför fram med bisektion
mot det färdiga brusfältet så att den angivna andelen träffas — ett tjugotal
svep över `hypot` i världens engångsuppbygge.

**Bassängen är elliptisk och följer världens form.** Avståndet normeras mot
utsträckningen i varje led, så havet blir en cirkel i en kvadratisk värld och
ett avlångt hav i en avlång. Med en rund bassäng blev sjöandelen 14 procent vid
64x256 mot 1,5 vid 64x128, eftersom landet längst från mitten saknade väg till
havet. Formtypen `bassang` tar därför `radie_x` och `radie_y` vid sidan av
`radie`; en landform har ingen anledning att vara elliptisk och anger bara den
senare.

Uppmätt över tre frön och fyra världsformer, `hav_andel = 0,20` och djup 2,0:

```
  värld      hav %        sjö %         största sjö    största flod
 64x128    20,0–20,0    1,57–2,00        51–124        1 059–1 404
 64x256    18,8–20,0    3,41–5,87       155–332        3 261–4 319
128x128    19,3–20,0    3,30–4,38        99–265        1 830–3 634
256x256    19,3–19,7   10,57–14,00      465–854        9 137–10 704
```

Havsandelen träffas i varje form och storlek. Sjöandelen vid 256x256 är däremot
över målet på tio procent — och det är **inte havets fel utan bandets**:
`lambda_max` är fast 48 celler medan världen är fyra gånger bredare, så
landskapet blir en slätt med krusningar där varje sänka blir sin egen
ändstation. Det är precis vad 7019 ska rätta.

**Kusten lever nu.** Kusttalet — kustlinjens längd delad med omkretsen hos en
cirkel med samma havsarea — är 1,32 till 1,48 mot den tidigare kandidatens 1,05
till 1,09. Den flackare och bredare bassängen gör bruset synligt i kustlinjen,
precis som 7017 förutsade. Batymetrin bär samma brus, spridning 1,5–1,8 meter,
så botten är inte en slät skål.

Havsandelen 20 procent ligger nära vad polarhavet gav (19,3 %), så landytan är i
praktiken oförändrad — bytet flyttar havet, det krymper inte världen.

`f4-start20` är oförändrad. En platt värld har ingen terräng och berörs inte.

**Vad som står kvar.** `Grid.cell_lat`, `band_lat`, `n_bands`, `band_of_cell`
och `bands_of_cells` har inga läsare i simuleringen längre, men lever kvar i
viewerns backljus och i `calibrate.py`s gruppering. De rivs när de två flyttats,
som egen patch — det är en städning och inte en del av den här ändringen.

### Bandet och amplituden skalar med världen (7019)

Två fel med samma rot: terrängen visste inte hur stor världen var.

**Bandets övre ände var 48 cellbredder oavsett värld.** Det är 0,75 av sidan vid
bredd 64 men 0,09 vid 512, så en större värld blev *finkornigare* i stället för
större — en slätt med krusningar där varje sänka blir sin egen ändstation.
`lambda_max_frac` ersätter `lambda_max`; `lambda_min` står kvar i celler,
eftersom den är upplösningens gräns och inte världens.

**Amplituden var oberoende av bandet.** Den analytiska normeringen tvingade
fältets standardavvikelse till `noise_sd` oavsett vilka våglängder som ingick,
alltså Hurstexponent noll: en värld på hundra kilometer fick samma tjugo meters
relief som en på en kilometer. `hurst = 0,64` är anpassad mot relief-mot-skala
för verklig topografi, och den siffran är den som gör att Skandinavien är 1 800
km lång och 2,5 km hög i stället för tiotals kilometer.

**Amplitudskalan gäller hela basnivån**, alltså bruset, grundhöjden och
havsdjupet — men inte de placerade formerna, som är tektonik med storlek i
meter. Att skala bara bruset gav ett hav som dränktes i det: vid 512x512 föll
den sammanhängande havsmassan till 10,8 procent av begärda 20 medan sjöandelen
steg till 21,3 och största sjön till 21 486 celler. Bassängen upphörde att vara
en basnivå.

Uppmätt, tre frön per storlek:

```
   värld     λ_max   skala    relief      hav %        sjö %       största flod
 64x128         48    1,00      22 m    20,0–20,0    1,57–2,00     1 059–1 404
128x128         96    1,56      33 m    20,0–20,0    0,11–0,78     1 577–2 076
256x256        192    2,43      53 m    20,0–20,0    0,00–2,31     6 087–8 775
```

Sjöandelen vid 256x256 föll från 7018:s **10,6–14,0 procent till 0,0–2,3**, och
havsandelen träffas nu i varje storlek. Reliefen växer från 22 till 53 meter
medan lutningen p95 sjunker från 0,077 till 0,047 — landskapet blir större och
flackare på samma gång, vilket är precis vad en Hurstexponent under ett betyder.

**Vad som återstår: sjöarna blir för få i breda band.** Noll procent vid 256x256
och frö 1 är inte ett bättre utfall än fjorton, bara ett annat. Orsaken är att
bandet blir logaritmiskt bredare när `lambda_min` står stilla, så `beta` flyttar
mer vikt till de långa vågorna och landskapet blir väldränerat. `beta` är
sjöreglaget och behöver sänkas när bandet breddas — uppmätt vid 256x256, tre
frön:

```
beta 1,8   sjö 2,49 / 7,04 / 5,42 %     368 / 516 / 497 sjöar
beta 2,0   sjö 0,57 / 4,30 / 2,29 %      74 / 265 / 194
beta 2,2   sjö 0,07 / 2,82 / 0,90 %      13 /  95 /  37
beta 2,4   sjö 0,00 / 2,31 / 0,38 %       2 /  24 /   6
```

Det talet sätts när en stor värld faktiskt ska köras, inte nu — det vore
kalibrering mot en världsstorlek som väntar på florans GPU-väg.

`f5-terrang` är bitidentisk: 0,75 av 64 celler är just de 48 som gällde förut,
och amplitudskalan är då exakt ett. Verifierat med `md5` på renderad PNG före
och efter.

### Havet blev en nivå i stället för en form (7020)

Kustlinjen såg utklippt ut — en blå form lagd ovanpå landskapet — och mätningen
visade varför.

Kustlinjen är nivåkurvan `z = 0`, så dess krokighet är brusets amplitud delad
med formens lutning just där. Uppmätt vid 256x256: **bassängens lutning vid
kustlinjen 0,0248 mot landskapets typiska 0,0135**, alltså nästan dubbelt så
brant. Kusten följde formens kant och inte terrängen; den vandrade sjutton
celler av en våglängd på 192, alltså nio procent.

**Havet är därför inte längre en form utan en nivå.** Det som läggs är bara en
regional lutning som gör en del av världen systematiskt lägre, och havsnivån
sätts som den kvantil av det färdiga fältet som ger den begärda andelen.
Kustlinjen blir då per konstruktion en nivåkurva i hela höjden — brus, lutning
och placerade former tillsammans.

Tre följder:

- **Bisektionen på radien försvinner.** En kvantil är exakt och kostar
  ingenting, medan bisektionen dessutom blev icke-monoton så snart formen var
  flackare än bruset.
- **`base` faller som parameter.** Den satte var vattenlinjen hamnade, och det
  gör `hav_andel` nu direkt. Två rattar för samma sak låter den ena tyst vinna.
- **Lutningens styrka är dimensionslös**, uttryckt som multipel av landskapets
  egen typiska lutning, och därmed skalfri.

Avvägningen blir synlig i ett enda tal. Uppmätt vid 256x256:

```
kvot 0,5   kust 2,11   hav  9,7 %   sjö 14,70 %   största sjö 6 855
kvot 1,0   kust 1,91   hav 19,4 %   sjö  3,42 %                2 077
kvot 2,0   kust 1,27   hav 20,0 %   sjö  2,82 %                1 627
kvot 3,0   kust 1,19   hav 20,0 %   sjö  0,18 %                   37
```

Under ungefär ett ligger de lägsta punkterna utspridda i landskapets egna
sänkor i stället för samlade, och det som skulle vara hav blir insjöar. Över två
tar den regionala lutningen över och kusten blir en cirkel igen. 1,0 är den
svagaste lutning som fortfarande ger ett hav.

**Det räckte inte, och fröberoendet avslöjade varför.** Kusttalet blev 1,22 för
frö 1 men 2,15 för frö 2 med identiska parametrar. Orsaken är spektrallutningen
och inte havet: vid `beta = 2,4` har landskapet nästan ingen kortvågig energi,
alltså *är* ytan slät lokalt, och då kan ingen kustlinje bli fransig hur havet än
läggs. Se 7021.

### Spektret knäcker (7021)

7020 gjorde havet till en nivå men kusten blev ändå slät för hälften av frön.
Fröberoendet avslöjade orsaken: den låg i spektret och inte i havet. Vid
`beta = 2,4` har landskapet nästan ingen kortvågig energi — ytan **är** slät
lokalt, och då kan ingen kustlinje bli fransig hur havet än läggs.

Amplitudkurvan knäcker därför vid `lambda_bryt`: `beta` för de långa vågorna,
`beta_kort` för de korta, kontinuerlig i knäcken. Verklig topografi har en sådan
knäck, där tektonikens formgivning slutar och erosionens tar över.

**Det som gjorde uppdelningen värd att bygga:** största sjön är praktiskt taget
oförändrad genom hela svepet — 1039 mot 954, 2622 mot 2402 — oavsett hur flackt
kortvågsspektret är. De stora sjöarna styrs av de långa vågorna, alltså av
dräneringens organisation, och den rörs inte. Det som växer är antalet små
sjöar. Med en enda lutning måste ett tal göra båda jobben, och det kan det inte.

Uppmätt, tre frön:

```
β_kort    64x128 kust / sjö %        256x256 kust / sjö %     lutning p95
  1,0   2,86/4,10/3,61  15,3/13,6   1,55/5,03/1,81   7,6/15,8   0,164 / 0,082
  1,2   2,41/3,55/3,00  12,4/11,4   1,41/4,06/1,45   5,2/13,2   0,136 / 0,067
  1,4   2,01/3,12/2,52   9,9/ 9,5   1,32/3,42/1,36   3,7/10,8   0,114 / 0,057
  1,6   1,85/2,57/2,19   7,5/ 7,3   1,28/2,93/1,29   2,7/ 9,1   0,098 / 0,051
```

**1,4 valt**: kusttalet stiger från 7020:s 1,43–1,90 till 2,01–3,12 vid 64x128,
och sjöandelen landar på 8,6–9,9 procent, alltså omkring tio.

**Brytpunkten är absolut och skalar inte med världen.** Erosionens finstruktur
bestäms av jordart och nederbörd, inte av kartans storlek — samma argument som
gör placerade former skalfria. Följden är att den lilla världen blir strävare
än den stora, eftersom en större del av dess band ligger under knäcken: lutning
p95 0,114 vid 64x128 mot 0,057 vid 256x256.

**Ett för flackt kortvågsspektrum bryter dräneringen.** Vid `beta_kort = 0,6`
kraschade `drainage.build` med en cykel — den topologiska ordningen täckte
65 532 av 65 536 celler, sannolikt plana partier där grannar blir exakt lika
höga i float32. Under ungefär 0,8 existerar ingen dräneringsordning. Golvet bör
fångas med ett begripligt fel i stället för en RuntimeError om nätet.

**Faunans framkomlighet måste mätas om.** Lutningen p95 går från 0,077 till
0,114 vid 64x128. `climb_gain 1,5` mot `descend_gain 0,5` kalibrerades mot
lutningar kring ±0,05, och korrelationen −0,573 mättes där.

**Målvärdet för sjöandel skrivs om** från "median under tio procent" till
"median omkring tio". Det gamla sattes när sjöar var en artefakt av att
dräneringen inte samlade sig, inte när de var en nisch — och Sverige har omkring
nio procent sjöyta.

### Riktningsfördelningen är en cosinus, inte ett fält (1013)

Bakgrunden är att representationen av djuren ska bli en fördelning i stället för
en punkt — uppehållsplats plus sannolikhet för rörelse — eftersom ett tick på
fjorton timmar är längre än djurets perceptuella horisont och positionen därför
inte är en punkt utan ett revir. Se `docs/tidens-skalor.md` när det finns.

Argumentet för att det skulle bli billigt var att fördelningen redan finns:
`_valj_anspravk` utvärderar `styrka · cos(Δ) − vikt · kostnad(b)` i varje
kandidatbäring och kastar allt utom `argmax`. Patchen sparar uttrycket i
`Agent._dir_prof` — samma loop, samma aritmetik, bitidentiskt utfall verifierat
mot orörd HEAD — och mäter det i `--stats`.

**Argumentet höll inte.** Uppmätt på 64x64, 300 tick, frö 1, 5 998 agenttick:

```
spridning    p10 1,904   median 1,947   p90 1,989     (i styrkeenheter)
marginal     p10 0,010   median 0,051   p90 0,092
likvärdiga   1,05 sektorer av 6,0 inom marginalen från toppen
```

Spridningen ligger på 1,95 av teoretiskt högsta 2,0. **Det är cosinus, inte
världen.** Sex sektorer spänner hela varvet, så en pekar mot vinnarens bäring
och en rakt bort; spannet blir `2 · styrka` oavsett vad som finns där ute.
Kostnadstermen rör den med några procent.

Följden är strukturell och avgör hur 0169 ska byggas: **fördelningen är en
entoppig lob runt ett `argmax` som redan tagits.** Multimodaliteten förstördes
uppströms, i `_samla_anspravk`, där varje anspråk kollapsar sina sektorer till
en bäring innan arbitreringen ser dem. Två goda fläckar i olika riktningar kan
inte representeras här, hur mycket av uttrycket man än sparar. En
utnyttjandefördelning måste byggas ur perceptet, inte ur arbitreringen.

Och perceptet bär den informationen: `_W["tvekan"]` mäter att näst bästa
sektorn i födoperceptet är **97 procent** av bästa i median. Profilen före
kollapsen är alltså nästan platt — precis den flerkantighet arbitreringen sedan
kastar.

**Marginalen ger en andra sak.** Medianen 0,051 ligger under `baring_marginal`
0,15, alltså under hysteresens tröskel. Riktningen avgörs oftare av att den
redan var vald än av att den är bäst. Det är vad "Bäringen binds" installerade,
och nu finns talet.

`_dir_prof` har inga läsare i simuleringen och kan därför inte påverka utfallet.
Läsordningen är bindande: plats 0 är anspråkets egen bäring, plats 1 till och
med `n_sektor` är sektormitterna, en eventuell bunden bäring ligger sist.

### Kanalerna får var sitt läge, och uppehållsfördelningen finns redan (1016)

`D` stegar i stället för att slå av och på, och **kulören bär kanalen och inte
anspråket**. Alla sex kilarna på ett djur fick förut samma färg, vilket lät
bilden se ut som om den kodade riktning när den kodade individ. Anspråket flyttar
till visaren, där det är ett attribut hos valet.

**Objektet är återställt.** Samtalet började i att djuret ska bäras som en
fördelning över var det uppehåller sig. 1014 och 1015 ritade perceptet — en
indata — med en grammatik motiverad av utdatan. `UPPEHALL` ritar nu det som
faktiskt efterfrågades: övergångsfördelningen över celler.

**Och den var inte för liten att se.** Ett tidigare påstående i det här arbetet
var att fördelningen är degenererad vid dagens tick. Det gällde utsträckningen i
planet men inte övergången mellan celler. Steget är 0,5 cellbredder mot
hexagonens inradie 0,537 — djuret kommer knappt halvvägs till granncellens
centrum, vilket är långt från noll. Uppmätt i `liten6` efter 40 tick:

```
stannar i egen cell:     p10 0,00   median 0,21   p90 1,00
celler med massa:        median 2   max 5
brusets vidd:            median 0,24 rad
```

Varken "stannar alltid" eller "lämnar alltid" — den enda regim där en
cellfördelning bär information. Objektet finns alltså vid dagens tick, och
0169 behöver inte vänta på 7027 för att ha något att representera.

**Två kanaler visas för första gången, och den ena är tom.** `_temp_sectors` och
`_soc_sectors` har beräknats varje tick utan att någonsin ha lästs för visning.
Uppmätt spann i `liten6`:

```
fauna_dir_temp     min 0,5666   max 0,5767      över alla djur och sektorer
fauna_dir_flock    min 0,0000   max 0,1624
```

Temperaturkanalen spänner **en hundradel** av sin normerade skala över hela
populationen. Den bär ingen riktningsinformation alls, vilket är väntat i en värld
med försumbar höjdgradient — men det betyder att `nedkylning` får sin bäring ur
ett fält som är konstant, och att anspråkets riktning därmed är brus. Det hör
ihop med samma fynd som `toppandel`: världen har nästan inga rumsliga gradienter
för djuren att navigera efter.

Fördelningen räknas i viewern och inte i `viewframe`: talen skickas råa —
steglängd, medelriktning före brusdraget, brusets vidd — eftersom en omräkning i
sändaränden vore ett andra exemplar av rörelselagen. Kvadraturen är
deterministisk och inte dragen, så bilden står still mellan bildrutor.

Cellerna ritas som skivor och inte som hexagoner. **Viewern känner inte
hexagonens hörn och ska inte göra det** — `Grid` exponerar cellcentrum, inte
hörn, och att konstruera hörnen i ritkoden vore precis den geometriläcka
manifestet förbjuder. Arean bär sannolikheten, samma grammatik som kilarna.

Klienten fick en `erfinv` i stället för ett beroende till scipy;
`requirements-viewer.txt` är fyra filer och två paket och ska förbli det.

### Kilarna får form, och floran täcker allt (1015)

1014:s bild såg jämnt blek ut oavsett vad djuret stod på. Skälet var
aritmetiskt och mitt: med sex sektorer får en jämn profil en sjättedel av
alfaskalan var, alltså allt i den blekaste sjättedelen, och en måttligt toppig
profil skiljer sig med några få steg av 255. **Regeln som skulle bevara bläcket
komprimerade bort dynamiken.**

Nu bär **radien** fördelningen och **opaciteten** mängden. `r_k = R·sqrt(andel·S)`
ger arean proportionell mot andelen, så totalarean är `π R²` oavsett form —
bläcket är bevarat i geometrin i stället för i alfakanalen. Verifierat exakt:
summan av `r²/S` är 1,000000 per djur. Uppmätt radiespann i `liten6` 0,75–1,19
av synvidden mot jämnfördelningens 1,00, alltså ±20 procent i form där 1014 gav
±4 procent i alfa.

`R` är djurets **verkliga synvidd**, skickad per djur i protokoll 8 eftersom
sensingnivån gör den olika mellan individer (7, 8, 10 eller 12 cellbredder). En
halo som inte går att lägga mot terrängen säger bara att djuret ser något.

Kulörerna följer nu behovstrappans nivå: nödlägena i den röda bågen, vardagen i
grönt och blått. 1014 satte `nedkylning` i blått, vilket är semantiskt lockande
men bryter grammatiken — den ligger på nivå 3.

**Kustkontrollen faller, och det avgör frågan.** Ett djur med tre sektorer mot
land och tre mot hav kan inte ha en jämn profil. Uppmätt i `liten6`, 400 tick,
frö 1, med 36,7 procent av tickarna inom två cellbredder från vatten:

```
toppandel    median 0,181     (jämn fördelning = 0,167)
  vid kust   median 0,186
```

Kusten flyttar toppandelen med **en halv procentenhet**. Perceptet reagerar
alltså knappt ens på världens skarpaste kant.

Skärmbilderna visar varför, och det står i HUD:en: `bar=20,1 %` mot
`hav=20,0 %`. **Den bara marken är havet.** Varje landcell är sluten, 873 814
plantor på 52 429 landceller är sjutton per cell, och landet är en jämn matta
ända ut till strandlinjen. Femtioåtta djur kan inte göra hål i en gräsmatta som
växer igen fortare än de betar.

Min tidigare gissning — att synvidden är större än fläckstorleken — var alltså
fel. **Det finns inga fläckar.** Det är B4:s monotona floratillväxt igen, nu med
ett näringstak som finns men som vid `bördighet 4.0` ligger över vad som krävs
för sluten mark överallt.

Följden för 0169 är att frågan inte är hur rörelsekärnan byggs. Den är **om
världen har någon struktur att röra sig mot.** En kärna i en sluten matta ger
diffusion hur väl den än är byggd, och det är billigare att pröva en lägre
bördighet i ett scenario än att bygga kärnan och upptäcka det efteråt.

### Perceptet ritas, och det är nästan platt (1014)

`D` ritar födoperceptet per riktningssektor som kilar kring varje djur.
Opaciteten är sektorns **andel** av profilens summa och inte dess värde, så
bläcket är bevarat: två djur bär alltid lika mycket färg och det enda som
skiljer är hur samlad den är. Ett entoppigt djur får en ljus kil, ett platt får
sex bleka, och en flock blir sex överlagrade moln. Kulören är vinnande anspråk.

Att det är perceptet och inte arbitreringens uttryck följer av 1013.

Profilen produceras i `Agent` med samma dietviktning som `_food_local`, inte
omräknad i `viewframe` — två exemplar av samma uttryck är hur styrningen och
fysiken glider isär. Fältet har inga läsare i simuleringen; bitidentitet
verifierad mot orörd HEAD.

**Och den nya siffran är obekväm.** `toppandel` mäter bästa sektorns andel av
perceptets summa. Uppmätt på 64x64, 300 tick, frö 1:

```
toppandel    p10 0,155   median 0,176   p90 0,197   (jämn fördelning = 0,167)
```

Bästa riktningen bär **17,6 procent mot jämnfördelningens 16,7**. Perceptet är
alltså inte bara flerkantigt utan i praktiken *isotropt* — riktningen bär nästan
ingen information alls. Det är samma sak `tvekan` sa med 97 procent, nu som ett
tal med en teoretisk referens att jämföra mot.

Den troliga orsaken är en skala och inte en bugg: **synvidden är större än
födofältets fläckstorlek.** Djuret medelvärdesbildar över så stor yta att varje
sektor ser samma sak. Det är den rumsliga motsvarigheten till att ticket är
längre än den perceptuella horisonten — samma motsättning, andra axeln — och det
hör till `docs/tidens-skalor.md`.

Följden för 0169 är hård: en rörelsekärna byggd på det här perceptet blir nästan
isotrop, alltså diffusion och inte födosök. **Kärnan kan inte rätta det.**
Kvoten mellan synvidd och fläckstorlek måste mätas först, och det är den kvoten
som avgör om en fördelningsrepresentation ger något alls.

Talet ska mätas om i en terrängvärld innan slutsatsen dras: `f6-256` har hav,
sjöar och en näringsgradient på 12x mellan rygg och dal, och där kan fläckigheten
vara verklig.

### Höjdkurvor i viewern (1012)

`K` lägger bruna höjdkurvor ovanpå kartan, med jämn ekvidistans och var femte
som tjockare huvudkurva.

**De ritas ovanpå vattnet och inte under.** Det är avsiktligt och hela poängen:
en kurva som löper ut i en sjö visar vilken nivå sjöytan står på. Över vatten
dämpas linjen till knappt halva styrkan så att vattnet fortfarande läses som
vatten.

Kurvorna är **linjesegment i världskoordinater**, inte färgade celler. Ett
första försök målade celler som låg på en nivågräns, med motiveringen att cellen
är kartans upplösning. Det var fel: höjden är ett *sampel* vid cellcentrum och
inte ett styckvis konstant fält, och med den tolkningen är en kurva mellan
celler både meningsfull och tunnare än en cell.

**Hexgittrets dual är ett triangelnät**, vilket gör interpolationen entydig: tre
inbördes grannar spänner en triangel, inom vilken fältet är linjärt, och
nivåkurvan blir ett rakt segment. Marching squares tvetydiga fall uppstår inte.
Två trianglar per cell täcker planet — (i, öst, sydost) och (i, sydost,
sydväst). Trianglar som wrappar över sömmen hoppas över; ett segment ritat rakt
över världen skulle se ut som ett fel.

Segmenten räknas ur `Grid`s grannmatris och cellcentra, alltså utan att någon
kod vet vad en rad är, och cachas med terrängen. Vid 256x256 blir de 14 036
stycken vid 2,5 meters ekvidistans.

Ekvidistansen väljs ur världens relief så att kartan får ungefär tolv kurvor,
och avrundas till ett tal som får stå på en karta — 0,5, 1, 2, 2,5, 5, 10, 20,
25, 50, 100 eller 200 meter. En kurva var 3,7 meter är ingenting en människa
läser av. Den står i HUD:en, eftersom kurvor utan ekvidistans inte går att läsa,
och `render_frame.py --kurvsteg` sätter den explicit.

**Färgen fick mätas mot underlaget och inte mot papper.** Ett första försök
använde orienteringskartans tryckfärg, 0,55 / 0,33 / 0,13. Den syntes utmärkt i
havet och försvann fullständigt på land — terränglagret är redan brungrönt, och
11,3 procent av landcellerna låg på en kurva utan att en enda av dem gick att
se. Mörk sepia i stället.

**Ekvidistansen bör väljas grövre i det strävare landskapet efter 7021.**
Automatvalet siktar på tolv kurvor, vilket ger 2,5 meter vid 256x256 — och då
blir kurvorna korta virriga fragment som ögat inte läser som linjer. Vid 5 meter
framträder de som kurvor. Talet relief genom tolv sattes före 7021 och bör
sannolikt bli relief genom sex eller åtta.

### Fotoperioden som senare steg (7022)

Modellen har i dag **ingen säsong i ljuset alls**, bara i temperaturen, medan floran redan har ljuskonkurrens via Beer–Lambert. Vid 55 grader är vinterdagen 6,9 timmar och sommardagen 17,1 — en faktor 2,5 i ljustillgång, helt oberoende av temperaturen. Dagslängd och insolation är exakt astronomi ur latituden och kostar ingen kalibrering.

Dagslängd är dessutom en **brusfri kalender**: ett varmt februaridygn lurar inte en organism som räknar timmar. Faunans häckningsfas läser i dag temperaturen, vilket är den svaga signalen; med fotoperiod finns en ärlig, och vilken dagslängd som utlöser häckning blir en trait att evolvera.

**Dygnet kan däremot inte vara en cykel.** `dt` är 14,6 timmar, alltså längre än dygnet, och en dygnsvariation som tidsserie skulle aliasa — samma klass av fel som det CFL-begränsade hydroschemat. Dygnets *amplitud* är en legitim statisk parameter och hör ihop med kalluftsdräneringen, som är ett nattfenomen och skalar med den.

### Öppna frågor efter Steg 7

**Vad ska den akvatiska axeln heta och kosta?** `flood_tolerance` är ett symtomnamn — det beskriver ett utfall, inte en kroppsegenskap — och projektets egen erfarenhet säger att det spelar roll: `structure` blev modellens bästa trait därför att den är en materialegenskap med fem konsumenter. Kandidaterna är rörelseform (drag i vatten ned, drag på land upp, som säl och utter), täthet (fett och luft mot förtätat skelett, den enda med ett *inre* optimum), och värmeledning vid nedsänkning (vatten leder värme tjugofem gånger snabbare än luft, vilket är det verkliga skälet att nedsänkning är farligt för en jämnvarm organism). Beslut krävs innan 7009 byggs.

**`nutrient_loss_frac` gör i praktiken ingenting** sedan urlakningen finns — 11 kg mot 1 081 över 20 000 tick. Den representerar gasformig förlust, alltså något fysiskt skilt, så den togs inte bort. Bör antingen motiveras skriftligt eller nollställas i terrängvärldar.

**Inkörningen är nu omkring 15 000 tick** i en terrängvärld: näringsstocken faller från 3 395 till 2 824 kg medan systemet går från jämnt sådd till urlakad jämvikt. Det påverkar hur långa körningar som krävs innan differentieringsmått betyder något.

**Kausaliteten mellan fukt och biomassa är omvänd i mätningen.** Den torraste fuktkvartilen har högst biomassa, 7,68 kg per cell mot 1,88 till 2,90 i de blötare. Tät växtlighet transpirerar marken torr, så markfukten är ett utfall lika mycket som en drivkraft. Varje framtida mått som använder markfukt som förklarande variabel måste ta hänsyn till det.

**Terränggenereringen kostar 24 s vid 262 144 celler**, seriellt. Engångskostnad som parallelliserar, men vill man ner en storleksordning är modsumman separabel i x och y.

### Världen är för smal för orografi

Tre oberoende mätningar pekar åt samma håll.

Kontinentallutningen måste vara ungefär brusamplitud gånger radantal delat med `lambda_max` för att dominera bruset. Vid 64x256 ger det 2,7, vid 64x128 blir det 1,3 och vid 512x512 endast 0,67. Följden är att den lokala reliefen bara är ±0,25 höjdenheter mot lutningens tre — **kontinenten är en ramp med krusningar, inte ett landskap med berg.** Höjdgradienten i 7005 fick därför mätas mot bandets landmedel i stället för mot havsnivån, eftersom rå höjd lät lutningen äta upp latitudgradienten och invertera klimatet.

Samma sak gör att vattnet inte kan fragmentera landet, och samma sak begränsar hur mycket topografisk struktur ekologin kan få.

En bredare värld är alltså inte bara en skalfråga. Vid 512x512 räcker en fjärdedels lutning, och den lokala reliefen blir dominerande i stället för marginell. Det är förutsättningen både för verklig orografi och för att vatten ska bli geografi.

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
| 5h | Rakhet: nettoförflyttning ÷ bansträcka per livstid | högre än 0,069 och tillståndsberoende |
| 5h | Rakhet vid halverat `dt` | oförändrad |
| 5i | Dödsorsakernas fördelning | oförändrad regim; uppmätt 43/45 mot baslinjens 58/32 |
| 5i | Skadeinflödets termer | kända, inte gissade |
| 6a | Sensing-kostnad för organism med `sense_radius → 0` | 0, och koden aldrig berörd |
| 6a | Spridning i `sense_radius` med kostnad kontra utan | differentierar mot nisch kontra neutral drift |
| 6c | Beteendetraits utan läsare | 0; i dag 3 |
| 6c | Predationsdödsfall per 100 000 tick | > 0; i dag 0 i tre av tre |
| 6c | Spridning i `sociability` med predation kontra utan | differentierar kontra neutral drift |
| 6c | Korskorrelation i födelsetakt mellan latitudband | < 0,3; i dag 0,45–0,67 |
| 6c | Andel av styrvektorn ur MLP:ns fria kanal | följs över tid, inget målvärde |
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
