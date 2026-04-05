# nep-process — Arkitekturmanifest

*Version 0.6 — April 2026*

-----

## Vision

Det här projektet simulerar liv som process — emergent, evolverbar och ekologiskt sammankopplad.

Målet är en värld där:

- Primärproduktionen är levande, inte ett bakgrundsfält
- Ekologiska strukturer uppstår underifrån, utan att kodas uppifrån
- Komplexitet har en kostnad — biologiskt, beräkningsmässigt och evolutionärt
- Allt liv delar samma ontologi: organismer med genom, kapaciteter och en kropp i världen

Den nuvarande arkitekturen är ett prototypstadium. Den har lärt oss vad som krävs. Nu bygger vi det som faktiskt bär visionen.

-----

## Grundprinciper

### 1. Allt liv är organismer

Världen innehåller inte “agenter och resurser”. Den innehåller organismer och abiotisk materia.

En organism är en levande enhet med:

- position i världen
- massa och energi
- ett genom som kodar kapaciteter
- en härledd fenotyp som avgör vilka subsystem som uttrycks
- en livscykel: tillväxt, metabolism, reproduktion, död

Det som skiljer en växt från ett rovdjur är inte vilken klass den tillhör. Det är värdet på dess kapacitetsprofil.

En växt har hög `uptake_capacity`, hög `growth_capacity`, hög `dispersal_capacity` och noll eller minimal `mobility`, `sense_radius` och `attack_capacity`. En predator har hög `sense_radius`, hög `mobility`, hög `attack_capacity` och en annan reproduktionsstrategi.

Evolutionen utforskar kontinuerliga övergångar i detta rum. Vi definierar inga artgränser i koden.

### 2. Kapaciteter kostar

Varje subsystem bär tre typer av kostnad:

**Underhållskostnad** — bara att bära kapaciteten kostar energi varje tick.

**Aktiveringskostnad** — att använda kapaciteten kostar ytterligare energi vid varje aktivering.

**Strukturkostnad** — hög maxkapacitet kostar även när den inte utnyttjas fullt ut.

Dessa kostnader är symmetriska med beräkningskostnaden i simulatorn. En organism med `sense_radius = 0` kostar ingenting i sensing-passet. En organism med `sense_radius = 8` läser från ett stort grannskap och betalar för det — i energi och i CPU.

Det är inte bara en implementationsteknisk princip. Det är en del av modellens epistemologi: komplexitet uppstår inte gratis, och evolutionen formas av att det inte gör det.

### 3. Enkla organismer är billiga

En sessil organism med minimal kapacitet ska kosta en bråkdel av vad en avancerad agent kostar — i minne, i iterationstid och i energiförbrukning.

Det kräver att den universella kärnan är extremt liten, att subsystem verkligen inte körs när kapaciteten är noll, och att systempass arbetar mot aktiva delmängder — inte mot hela populationen. Det är inte en optimering som läggs till i efterhand. Det är ett designkrav från dag ett.

### 4. Fasbaserad exekvering

Varje simulationssteg är inte “varje organism gör allt”. Det är en sekvens av systempass där varje pass hanterar ett subsystem för alla relevanta organismer.

Passen körs i ordning:

1. **Världsprocesser** — diffusion, nedbrytning, abiotisk dynamik
1. **Uptake** — organismer tar upp näring från lokal cell
1. **Growth** — massa byggs från energi mot genetisk target
1. **Sensing** — organismer med `sense_radius > ε` samlar information
1. **Decision** — mål väljs av organismer med beteendelogik
1. **Locomotion** — rörelse för organismer med `mobility > ε`
1. **Interaction** — betning, predation, parning
1. **Metabolism** — underhållskostnad, skada, reparation, åldrande
1. **Reproduction** — reproduktion och spridning
1. **Death** — döda organismer omvandlas till detritus; deras index frigörs

Varje pass är en funktion som tar arrayer och returnerar eller muterar arrayer. Inga dolda sidoeffekter. Det gör varje pass testbart, profilbart och migrerbart oberoende av de andra.

### 5. Dataorienterad kärna

Alla organismer delar samma kärnstore och samma kapacitetsmodell. Kärnfälten lagras i täta parallella arrayer — en per fält — där index är organism-ID.

Tilläggsstate för specifika subsystem allokeras i separata tilläggsarrayer med tydlig ägarskap. Det bryter inte den gemensamma ontologin så länge frånvarande kapaciteter inte kostar något att bära.

### 6. Abstrakt geometri

Ingen del av biologin — inte ett enda systempass, inte sensing, inte spridning, inte rörelse — ska innehålla hårdkodade antaganden om världens geometri.

Allt rumsligt arbete sker via ett väldefinierat `Grid`-gränssnitt. Det är den enda plats där geometrin existerar. Det gör det möjligt att byta geometri utan att röra biologin.

-----

## Kärnrepresentation

### Minimifält i OrganismStore

Dessa fält finns för varje levande organism och för organismer som dött under innevarande tick och ännu inte omvandlats till detritus:

```
# Identitet
id[i]              # unikt heltal

# Position och rum
pos_x[i]           # float, toroidal kontinuerligt rum
pos_y[i]           # float, toroidal kontinuerligt rum
cell_idx[i]        # int, diskret cell-ID — cache av pos, se nedan

# Energistatus
energy[i]          # float, J
mass[i]            # float, kg

# Livsstatus
age[i]             # float, sekunder
alive[i]           # bool

# Genom och fenotyp
genome_idx[i]      # int, pekare till genomlagret
```

Kapacitetsprofilen härleds initialt från genomet vid birth och lagras som direkta fält för snabb åtkomst under systempass. Framtida utvidgningar kan tillåta plastisitet eller tillståndsberoende uttryck, men det avgörs av ekologisk nödvändighet — inte i förväg.

```
uptake_capacity[i]
growth_capacity[i]
dispersal_capacity[i]
sense_radius[i]        # geometrisk räckvidd i cellavstånd
sense_rate[i]          # frekvens: hur ofta sensing aktiveras
mobility[i]
attack_capacity[i]
repair_capacity[i]
repro_capacity[i]
```

Inget mer tillhör kärnan. Subsystemspecifikt state — sensorscache, rörelsemål, reproduktionsfas — placeras i separata tilläggsarrayer med dokumenterad ägarskap och allokeras bara för de organismer som faktiskt har kapaciteten. Kärnan ska inte växa för att rymma sådant.

### Kontinuerligt rum och diskret grid

Organismer rör sig i ett kontinuerligt toroidalt rum via `pos_x`, `pos_y`. `cell_idx` är en cache av organismens aktuella diskreta cell och hålls konsistent med `pos_x`, `pos_y` efter varje positionsuppdatering, oavsett vilket pass som orsakade den. Det är inte ett fält som uppdateras “vid rörelse” — det är en invariant som alltid ska hålla.

Det är ett medvetet hybridval: kontinuerlig position ger smidig rörelse och naturlig fysik; diskret cellindex möjliggör effektiv spatial indexering, lokal interaktion och resurstillgång. Biologin arbetar alltid via cell-ID:n och aldrig direkt mot råa koordinater — det är `Grid`s ansvar att hantera kopplingen.

### Aktiva delmängder och ticksemantik

Varje systempass arbetar mot en förberedd delmängd av organism-ID:n, inte mot hela populationen.

Delmängderna byggs en gång i början av varje tick, innan något pass körs, och betraktas som immutabla under ticken. Födslar och dödsfall under pågående tick registreras men påverkar inte delmängderna förrän nästa tick börjar. Det gör pass-ordningen deterministisk och förhindrar att ett pass ser halvanvändna tillstånd från ett annat.

En organism med `sense_radius < ε` ska aldrig beröra sensing-koden — inte ens som ett hopp. Det är den mekanism som faktiskt gör principen “enkla organismer är billiga” sann i koden.

### Indexlivscykel och slothantering

När en organism dör markeras dess index som ledigt. Lediga index återanvänds vid nästa födelseallokering. Nyfödd organism skriver över alla fält i det återanvända indexet innan det används i något pass.

En organism som dött under innevarande tick ligger kvar i `OrganismStore` med `alive = False` tills Death-passet har omvandlat dess massa till detritus i cellen. Därefter frigörs indexet. Inget annat pass efter Death ska läsa eller skriva till ett index med `alive = False`.

### Genomet

Genomet hålls initialt enkelt och lågdimensionellt för att stabilisera migrationen och hålla evolutionens sökrum hanterbart.

En organism bär ett fast antal genloci — initialt i storleksordningen 8–16 — kodade som kontinuerliga flyttal i intervallet [0, 1]. Kapacitetsprofilen härleds via enkla skalningsfunktioner, eventuellt med icke-linjäritet för att skapa trade-offs.

Genomet expanderas inte i omfång förrän ekologisk och evolutionär dynamik är stabil.

-----

## Världsmodell

### Grid-abstraktion

`Grid` är den enda plats i systemet där världens geometri är känd. Den exponerar ett gränssnitt som alla systempass, alla världsprocesser och all biologisk logik ska använda:

```python
grid.neighbors(cell)            # lista med granncell-ID:n
grid.distance(cell_a, cell_b)   # topologiskt cellavstånd under toroidal wrap
grid.cell_of(pos_x, pos_y)      # kontinuerlig position → cell-ID
grid.cells_within(cell, r)      # alla celler inom r steg
grid.wrap(cell)                 # toroidal periodisitet
```

`grid.distance()` definieras som topologiskt cellavstånd under toroidal wrap — det minsta antalet steg längs gridkanter mellan två celler. Det är den metrik alla systempass använder för räckvidd och spridning. Euklidisk distans i det inbäddade planet är inte detsamma och används inte i biologisk logik.

Inget utanför `Grid` ska referera till konkreta koordinattyper eller geometrispecifika operationer. Om det ändå sker är det ett fel att åtgärda, inte en detalj att lämna.

### Hexagongeometri

`Grid`-abstraktionen är geometriagnostisk. Projektets avsedda operativa geometri är hexagonalt grid.

Hex är inte nostalgisk estetik. Det är en välmotiverad geometrisk grund för just den typ av ekologi modellen syftar till:

- **Isotropi:** Alla sex grannar är likvärdiga — samma avstånd, samma vikt. Det eliminerar den axel- och diagonalbias som kvadratgrid skapar i diffusion, växtspridning och kortdistansrörelse.
- **Renare ekologisk struktur:** Vegetationsmönster, lokala territorier och spridningsfronter uppstår utan rutnätsartefakter.
- **Bättre sensing-geometri:** `sense_radius = 1` ger exakt 6 celler, `sense_radius = 2` ger exakt 18 — inga oklarheter om hur diagonaler ska räknas.

Implementationen använder axialkoordinater (q, r) internt i `Grid`. Cell-ID:n är heltal mappade från axialkoordinater. Allt utanför `Grid` ser bara cell-ID:n — aldrig råa koordinater.

Diffusion och nedbrytning implementeras via `grid.neighbors()` och är därmed geometriskt oberoende i kodstrukturen. Den resulterande dynamiken påverkas dock av vald geometri — hex och kvadrat ger inte identiskt beteende, och det är avsiktligt.

### Abiotiskt substrat

Varje cell i hexgridet bär:

- `nutrient` — löslig näring tillgänglig för uptake
- `detritus` — dött organiskt material under nedbrytning

Näring sprids via diffusion och regenereras från externa flöden. Detritus bryts ned och återför näring. Kretsloppen är slutna i princip men modelleras med approximationer — stabil dynamisk jämvikt är målet, inte exakt konservering.

### Flora

Flora är diskreta organismer vars kapacitetsprofil liknar autotrofa sessila livsformer.

Florans genomloci kodar initialt: `uptake_rate`, `growth_rate`, `dispersal_radius`, `repro_threshold`, `defense`, `digestibility`.

Flora saknar locomotion. Sensing är begränsad till lokal cell eller nollnivå. Reproduktion sker via spridning till grannrutor med mutation — spridningen sker via `grid.cells_within()` och är geometriskt agnostisk.

Flora representeras i samma SoA-arrayer som alla andra organismer. Det finns ingen separat flora-struktur, bara organismer med en viss kapacitetsprofil. Eventuell administrativ separation av floraindex under migrationen är en tillfällig teknisk konvention, inte en ontologisk kategori i systemlogiken.

-----

## Migrationsstrategi

Den befintliga koden är välkalibrerad och fungerande. Den kastas inte. Den ersätts gradvis inifrån, fas för fas, med tydlig ägarskap av varje datafält under hela övergången.

**Princip:** Varje fält har vid varje tidpunkt exakt en source of truth. Gammal kod läser därifrån. Ny kod skriver dit. Ingen dubbel uppdatering av samma fält.

-----

### Fas 0 — Grunden

Mål: Skapa kärnstrukturen utan att förändra beteendet.

- Inför `OrganismStore` som separat modul med SoA-arrayer och spatial cellindex
- Synka befintliga `Agent`-objekt mot `OrganismStore` efter varje tick (spegla, inte ersätt)
- Verifiera att simuleringen håller samma dynamiska regim — samma storleksordning i population, energi och livslängd
- Profilera synkens overhead — det ger baseline för kärnans faktiska kostnad

Inga beteendeförändringar i denna fas.

-----

### Fas 1 — Världen, näringsfälten och grid-abstraktionen

Mål: Flytta abiotisk dynamik till fasbaserade systempass och etablera `Grid` som enda geometrikälla.

**Grid-abstraktion:**

- Implementera `Grid`-klassen med hela gränssnittet: `neighbors`, `distance`, `cell_of`, `cells_within`, `wrap`
- Börja med kvadratgeometri som initial implementation — gränssnittet är abstrakt, geometrin är ett val som görs senare
- Granska befintlig kod och eliminera varje direkt geometriantagande utanför `Grid`: inga `(x±1, y)`, inga hårdkodade grannlistor, ingen toruslogik utspridd i systemen
- Geometriantaganden utanför `Grid` räknas som fel att åtgärda, inte detaljer att lämna

**Hexövergång:**

- När `Grid`-abstraktionen är ren och validerad: byt till hexgeometri genom att ersätta `Grid`-implementationen
- Inget utanför `Grid` ska behöva ändras — om det ändå krävs är abstraktionen ofullständig
- Tidpunkten avgörs av när world-lagret är tillräckligt rent för att bära bytet utan att blanda geometriombyggnad med biologisk kalibrering

**Världspass:**

- Extrahera `nutrient` och `detritus` till dedikerade arrayer i `world/fields.py`
- Skriv `diffusion_system()` och `decomposition_system()` som fristående funktioner via `grid.neighbors()`
- Låt befintlig `World.step()` delegera till dessa pass
- Verifiera att näringsdynamiken håller samma regim
- Mät prestanda — diffusion är primär GPU-kandidat längre fram

-----

### Fas 2 — Diskret flora

Mål: Ersätt biomassfältet med en levande florapopulation. Ekologisk hypotes testas här.

- Allokera flora direkt i `OrganismStore`-arrayerna — inga separata objekt, ingen wrapper-klass
- Skriv `uptake_system()`, `growth_system()`, `dispersal_system()` som pass — spridning sker via `grid.cells_within()`
- Konstruera aktiva delmängder för flora-pass i början av varje tick
- Låt agenternas konsumtion läsa florans celler via `OrganismStore` och cellindexet
- Ta bort det kontinuerliga biomassfältet när floran bär dess funktion

**Validering:** Uppstår stabila florapopulationer? Uppstår koevolution med konsumenter? Är prestanda acceptabel vid tusentals floraindivider?

Om den ekologiska hypotesen stämmer — fortsätt. Om dynamiken inte fungerar — revidera floramodellen, inte kärnan.

-----

### Fas 3 — Spatial integration

Mål: Alla organismer använder samma spatialindex.

- Flytta agenternas cellbaserade uppslag till `OrganismStore`-indexet
- Befintlig sensing och interaktionslogik läser via indexet och `grid.cells_within()`
- En agent som söker mat hittar floraindivider via samma mekanism som den hittar andra agenter

Nu är det rumsliga lagret gemensamt för allt liv.

-----

### Fas 4 — Subsystem som pass

Mål: Agenternas subsystem migreras ett i taget till fasmodellen.

Ordning, från lättast till svårast:

1. Basal metabolism och åldrande → `metabolism_system()`
1. Sensing → `sense_system()` — arbetar mot aktiv delmängd med `sense_radius > ε`, via `grid.cells_within()`
1. Locomotion → `move_system()` — arbetar mot aktiv delmängd med `mobility > ε`
1. Interaktion och predation → `interaction_system()`
1. Reproduktion → `reproduction_system()`
1. Intern fysiologi, M_target-logik, reparation — sist

Under denna fas överlämnar de gamla `Agent`- och `Body`-objekten gradvis source of truth till `OrganismStore`. Till slut är de tunna wrappers eller försvinner.

-----

### Fas 5 — Acceleration

Mål: Identifiera faktiska hotspots och accelerera dem.

- Profilera fasmodellen under realistisk belastning med blandad population
- Flytta diffusion och fältuppdateringar till GPU via CuPy om de dominerar
- Accelerera sensing-passet med Numba om det fortfarande är bottleneck
- Utvärdera om en Rust-kärna är motiverad baserat på verkliga mättal

Inget accelereras förrän det är mätt.

-----

## Tekniska riktlinjer

**Inga tunga Python-objekt i den heta loopen.** `OrganismStore` är arrayer, inte objektlistor.

**Varje systempass är en funktion.** Den tar arrayer och returnerar eller muterar arrayer. Inga dolda sidoeffekter.

**Aktiva delmängder byggs en gång per tick och är immutabla under ticken.** Födslar och dödsfall registreras men träder i kraft först vid nästa tick.

**`cell_idx` är en invariant, inte ett fält som uppdateras opportunistiskt.** Den hålls konsistent med `pos_x`, `pos_y` efter varje positionsuppdatering, oavsett orsak.

**Geometrin ägs av Grid, ingenting annat.** Direkta koordinatoperationer utanför `Grid` är fel. Det gäller även viewer — visualiseringen översätter cell-ID:n via `Grid`, inte tvärtom.

**Subsystemstate hålls i separata tilläggsarrayer.** De allokeras bara för organismer som har kapaciteten och har dokumenterad ägarskap. Kärnan ska inte växa för att rymma state som tillhör ett specifikt subsystem.

**Source of truth är explicit.** Under migration: vilket lager äger vilket fält ska framgå av kod och dokumentation. Aldrig dubbel uppdatering av samma fält.

**Biologisk validering före arkitekturinvestering.** Om en ekologisk mekanism inte ger önskad dynamik, revidera biologin — inte infrastrukturen.

**Genomet expanderas inte i förtid.** Ekologisk och evolutionär stabilitet verifieras med enkelt lågdimensionellt genom innan representationen kompliceras.

-----

## Vad vi bygger mot

En värld där primärproduktionen lever, konkurrerar och evolerar. Där konsumenter formar växtligheten och växtligheten formar konsumenterna. Där komplexitet uppstår i det evolutionära rummet snarare än i koden.

En motor där tiotusentals enkla organismer och hundratals komplexa kan samexistera utan att beräkningskostnaden kollapsar.

En arkitektur där nästa kapacitet — ett nytt subsystem, en ny livsform, ett nytt selektionstryck — kan läggas till som ett nytt pass, utan att röra det som redan fungerar.

En hexagonal värld, utan rutnätsartefakter, där lokal ekologisk struktur kan uppstå på sina egna villkor.

-----

*Version 0.6. Revideras när ny kunskap motiverar det. Kompass, inte kontrakt.*