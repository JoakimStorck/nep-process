# nep-process — Arkitekturmanifest

*Version 0.9 — April 2026*

---

## Vision

Det här projektet simulerar liv som process — emergent, evolverbar och ekologiskt sammankopplad.

Målet är en värld där:

- Primärproduktionen är levande, inte ett bakgrundsfält
- Ekologiska strukturer uppstår underifrån, utan att kodas uppifrån
- Komplexitet har en kostnad — biologiskt, beräkningsmässigt och evolutionärt
- Allt liv delar samma ontologi: organismer med genom, kapaciteter och en kropp i världen

Den nuvarande arkitekturen är ett prototypstadium. Den har lärt oss vad som krävs. Nu bygger vi det som faktiskt bär visionen.

---

## Grundprinciper

### 1. Allt liv är organismer

Världen innehåller inte "agenter och resurser". Den innehåller organismer och abiotisk materia.

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

Varje simulationssteg är inte "varje organism gör allt". Det är en sekvens av systempass där varje pass hanterar ett subsystem för alla relevanta organismer.

Passen körs i ordning:

1. **Hydro** — vatten flödar enligt potential och kontinuitet; flytande organismer transporteras passivt
2. **Transport** — diffusion av abiotiska lösta ämnen
3. **Decomposition** — nedbrytning och återföring av materia
4. **Uptake** — organismer tar upp näring från lokal cell
5. **Growth** — massa byggs från energi mot genetisk target
6. **Sensing** — organismer vars `sense_radius > ε` och sensingfrekvens är uppfylld samlar information
7. **Decision** — mål väljs av organismer med beteendelogik
8. **Locomotion** — aktiv rörelse för organismer med `mobility > ε`
9. **Interaction** — betning, predation, parning
10. **Metabolism** — underhållskostnad, skada, reparation, åldrande
11. **Reproduction** — reproduktion och spridning
12. **Death** — döda organismer omvandlas till detritus; deras slotindex frigörs

Världsprocesserna (Hydro, Transport, Decomposition) täcker alla celler. Biologiska pass (Uptake och nedåt) arbetar mot aktiva delmängder av organismer. Varje pass är en funktion som tar arrayer och returnerar eller muterar arrayer. Inga dolda sidoeffekter.

Passiv drift — förflyttning av flytande organismer med vattenflödet — sker i Hydro-passet, före biologisk sensing och decision. Det håller isär rörelse som fysik och rörelse som beteende.

### 5. Dataorienterad kärna

Alla organismer delar samma kärnstore och samma kapacitetsmodell. Kärnfälten lagras i täta parallella arrayer — en per fält — där slotindex är nyckel.

Tilläggsstate för specifika subsystem allokeras i separata tilläggsarrayer med tydlig ägarskap. Det bryter inte den gemensamma ontologin så länge frånvarande kapaciteter inte kostar något att bära.

### 6. Abstrakt geometri

Ingen del av biologin — inte ett enda systempass, inte sensing, inte spridning, inte rörelse — ska innehålla hårdkodade antaganden om världens geometri.

Allt rumsligt arbete sker via ett väldefinierat `Grid`-gränssnitt. Det är den enda plats där geometrin existerar. Det gör det möjligt att byta geometri utan att röra biologin.

### 7. Fysiklager

Simulatorn skiljer explicit mellan tre nivåer:

- **Fysiklagret** definierar lagar, storheter och regler
- **Världslagret** instansierar dessa lagar för konkreta cellfält
- **Biologin** läser härledda tillstånd ur världen

Fysiklagret definierar:
- grundstorheter: massa, energi, volym, tid
- bevarandeprinciper: kontinuitet för materia och energi; bevarade storheter uppdateras alltid via tvåstegsmetod
- generella transportlagar: diffusion och gradientdrivet flöde
- globala konstanter och skalor

Fysiklagret opererar inte direkt på världens tillstånd. Det definierar reglerna som världspass tillämpar.

**Energi och massa** är kopplade storheter i hela modellen. Biomassa representerar lagrad kemisk energi. Energi används för tillväxt, underhåll och arbete och överförs mellan organismer och till detritus vid dödsfall och konsumtion. Fysiklagret definierar konverteringsfaktorer och grundläggande kostnadsskalor. Biologiska systempass implementerar dessa relationer men bryter inte mot dem.

---

## Kärnrepresentation

### Minimifält i OrganismStore

Dessa fält finns för varje levande organism och för organismer som dött under innevarande tick och ännu inte omvandlats till detritus:

```
# Identitet
id[i]              # unikt heltal — stabil biologisk identitet, återanvänds aldrig
                   # i = slotindex i store, återanvänds vid ny allokering (se nedan)

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

Kapacitetsprofilen är indelad i subsystemkapaciteter (vad organismen kan göra) och mediumkapaciteter (hur organismen förhåller sig till världens fysiska tillstånd). Båda kategorier är genuina genotypiska egenskaper med underhållskostnader och evolutionär påverkan.

```
# Subsystemkapaciteter
uptake_capacity[i]
growth_capacity[i]
dispersal_capacity[i]
sense_radius[i]        # geometrisk räckvidd i cellavstånd
sense_rate[i]          # frekvens: andel av ticks då sensing aktiveras
mobility[i]
attack_capacity[i]
repair_capacity[i]
repro_capacity[i]

# Mediumkapaciteter
flood_tolerance[i]     # tålighet för vattendjup; lågt värde → ökad kostnad och skada i vatten
buoyancy[i]            # flytförmåga; styr passiv transport med vattenflöde
```

Inget mer tillhör kärnan. Subsystemspecifikt state — sensorscache, rörelsemål, reproduktionsfas — placeras i separata tilläggsarrayer med dokumenterad ägarskap och allokeras bara för de organismer som faktiskt har kapaciteten. Kärnan ska inte växa för att rymma sådant.

### Slotindex och organism-ID

`i` är ett slotindex i `OrganismStore` och återanvänds när en organism dör och en ny föds. `id[i]` är en stabil unik biologisk identitet — ett monotont ökande heltal som tilldelas vid birth och aldrig återanvänds.

Distinktionen är viktig: slotindex styr arrayåtkomst och prestanda; organism-ID styr biologisk identitet, loggning och spårning av livshistoria och släktskap. Kod som refererar till en individ över tid ska använda `id`, inte `i`.

### Kontinuerligt rum och diskret grid

Organismer rör sig i ett kontinuerligt toroidalt rum via `pos_x`, `pos_y`. `cell_idx` är en cache av organismens aktuella diskreta cell och hålls konsistent med `pos_x`, `pos_y` efter varje positionsuppdatering, oavsett vilket pass som orsakade den. Det är inte ett fält som uppdateras "vid rörelse" — det är en invariant som alltid ska hålla.

Det är ett medvetet hybridval: kontinuerlig position ger smidig rörelse och naturlig fysik; diskret cellindex möjliggör effektiv spatial indexering, lokal interaktion och resurstillgång. Biologin arbetar alltid via cell-ID:n och aldrig direkt mot råa koordinater — det är `Grid`s ansvar att hantera kopplingen.

### Aktiva delmängder och ticksemantik

Varje systempass arbetar mot en förberedd delmängd av slotindex, inte mot hela populationen.

Delmängderna byggs en gång i början av varje tick, innan något pass körs, och betraktas som immutabla under ticken. Födslar och dödsfall under pågående tick registreras men påverkar inte delmängderna förrän nästa tick börjar. Det gör pass-ordningen deterministisk och förhindrar att ett pass ser halvanvändna tillstånd från ett annat.

För sensing gäller ett dubbelt villkor: organismen inkluderas i sensing-delmängden bara om `sense_radius > ε` och om tickräknaren uppfyller organismens `sense_rate`. Det gör att sällan-sensande organismer inte berörs av sensing-koden de mellanliggande tickarna.

En organism med `sense_radius < ε` ska aldrig beröra sensing-koden — inte ens som ett hopp. Det är den mekanism som faktiskt gör principen "enkla organismer är billiga" sann i koden.

### Indexlivscykel och slothantering

När en organism dör markeras dess slotindex som ledigt. Lediga slotindex återanvänds vid nästa födelseallokering. Nyfödd organism skriver över alla fält i det återanvända slotindexet — inklusive ett nytt unikt `id` — innan slotindexet används i något pass.

En organism som dött under innevarande tick ligger kvar i `OrganismStore` med `alive = False` tills Death-passet har omvandlat dess massa till detritus i cellen. Därefter frigörs slotindexet. Inget annat pass efter Death ska läsa eller skriva till ett slotindex med `alive = False`.

### Genomet

Genomet hålls initialt enkelt och lågdimensionellt för att stabilisera migrationen och hålla evolutionens sökrum hanterbart.

En organism bär ett fast antal genloci — initialt i storleksordningen 8–16 — kodade som kontinuerliga flyttal i intervallet [0, 1]. Kapacitetsprofilen härleds via enkla skalningsfunktioner, eventuellt med icke-linjäritet för att skapa trade-offs.

Genomet expanderas inte i omfång förrän ekologisk och evolutionär dynamik är stabil.

---

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

### Världsfält och deras kategorier

Världens cellfält delas in i tre kategorier med tydliga roller:

**Primära tillståndsfält** — lagras och uppdateras av world-pass. De är bevarade storheter som lyder under fysiklagrets kontinuitetsprincip.

```
elevation     — fast substratnivå; förändras inte under normala simuleringstick
water         — vattendjup per cell
nutrient      — löslig näring tillgänglig för uptake
detritus      — dött organiskt material under nedbrytning
```

**Härledda fält** — beräknas av world-pass och exponeras för biologin. De lagras inte permanent utan räknas om varje tick.

```
surface_level     = elevation + water
flow_strength     — genomflöde under senaste hydro-pass
flow_direction    — dominerande nettoflödesriktning
submerged         — bool; sant om water > tröskel
```

**Forcing-fält** — extern input till systemet. De styrs inte av organismer och är i initiala modellen parametriska fält per cell, inte biologiska tillstånd.

```
rain_input        — vattentillförsel per cell per tick
spring_input      — lokala källflöden
infiltration      — vattenförlust till mark per cell
evaporation       — vattenförlust till atmosfär per cell
```

Denna indelning gör det explicit vad som bevaras, vad som härleds och vad som tillförs utifrån. Biologin läser primära och härledda fält. Den skriver aldrig till forcing-fält.

### Hydrologi

Vatten representeras som ett inkompressibelt medium diskretiserat per cell. Land och vatten är inte två ontologiskt skilda världar — de är två regimer i samma fältmodell, styrda av topografi och vattenmängd.

Fri yta definieras som:

```
surface = elevation + water
```

Flöde sker från cell till granncell baserat på skillnaden i fri yta. Flödet är lokalt och gradientdrivet. Alla flöden beräknas från cellernas tillstånd vid början av passet och appliceras simultant som nettoförändringar. Kontinuitet upprätthålls strikt: total utström från en cell får aldrig överstiga tillgängligt vatten.

Vattentillförsel (`rain_input`, `spring_input`) och förluster (`infiltration`, `evaporation`) tillämpas som källrespektive sänktermer per cell inom hydro-passet.

Celler som faller under `sea_level` utgör en hydrologisk randregim. Inflöde till dessa celler lämnar den explicita landvattenbudgeten — de absorberar vatten utan att ackumulera tryck mot omgivningen. Det ger naturliga kustlinjer och havsbassin utan en separat oceanmodell.

Hydro-passet producerar emergenta fenomen utan att koda dem direkt:
- strömmande vatten där gradient och tillförsel är stabila
- sjöar och hav i lågpunkter och bassänger
- våtmarker och strandzoner som ekologiska nischer
- passiv transport av flytande organismer via `flow_direction`

### Framkomlighet

En cells passerbarhet är inte en egenskap hos cellen utan en relation mellan cellens fysiska tillstånd och organismens kapaciteter. Världen lagrar inte "är passerbar för X" — den lagrar `water_depth` och `flow_strength`.

Locomotion- och decision-passen beräknar rörelsekostnad och framkomlighetsgräns från celldata och organismens `flood_tolerance`, `buoyancy` och `mass`. En landorganism med låg `flood_tolerance` möter ökad rörelsekostnad i grunt vatten och en hård gräns vid djupare. En flytande organism med hög `buoyancy` rör sig i vatten som i ett medium. En akvatisk organism kan föredra djupa och strömmande celler framför land.

Passiv drift — förflyttning driven av vattenflöde snarare än muskelkraft — hanteras i Hydro-passet, före biologisk decision. Det håller isär rörelse som fysik och rörelse som beteende.

### Flora

Flora är diskreta organismer vars kapacitetsprofil liknar autotrofa sessila livsformer.

Florans genomloci kodar initialt: `uptake_rate`, `growth_rate`, `dispersal_radius`, `repro_threshold`, `defense`, `digestibility`. Akvatiska floraformer kan ha hög `buoyancy` och hög `flood_tolerance`, vilket möjliggör passiv transport med vattenflöde och kolonisation via vattensystem.

Flora saknar aktiv locomotion. Sensing är begränsad till lokal cell eller nollnivå. Reproduktion sker via spridning till grannrutor med mutation — spridningen sker via `grid.cells_within()` och är geometriskt agnostisk.

Flora representeras i samma SoA-arrayer som alla andra organismer. Det finns ingen separat florastruktur, bara organismer med en viss kapacitetsprofil. Eventuell administrativ separation av floraindex under migrationen är en tillfällig teknisk konvention, inte en ontologisk kategori i systemlogiken.

---

## Migrationsstrategi

Den befintliga koden är välkalibrerad och fungerande. Den kastas inte. Den ersätts gradvis inifrån, fas för fas, med tydlig ägarskap av varje datafält under hela övergången.

**Princip:** Varje fält har vid varje tidpunkt exakt en source of truth. Gammal kod läser därifrån. Ny kod skriver dit. Ingen dubbel uppdatering av samma fält.

---

### Fas 0 — Grunden

Mål: Skapa kärnstrukturen utan att förändra beteendet.

- Inför `OrganismStore` som separat modul med SoA-arrayer och spatial cellindex
- Synka befintliga `Agent`-objekt mot `OrganismStore` efter varje tick (spegla, inte ersätt)
- Verifiera att simuleringen håller samma dynamiska regim — samma storleksordning i population, energi och livslängd
- Profilera synkens overhead — det ger baseline för kärnans faktiska kostnad

Inga beteendeförändringar i denna fas.

---

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
- Extrahera alla primära tillståndsfält (`elevation`, `water`, `nutrient`, `detritus`) till dedikerade arrayer i `world/fields.py`
- Inför forcing-fälten (`rain_input`, `spring_input`, `infiltration`, `evaporation`) som konfigurerbara per-cell-arrayer
- Skriv `hydro_pass()`, `transport_pass()` och `decomposition_pass()` som fristående funktioner via `grid.neighbors()`
- Inför fysiklagrets konstanter och tvåstegsmetod för alla bevarade storheter
- Låt befintlig `World.step()` delegera till dessa pass
- Verifiera att näringsdynamiken håller samma regim; validera att vattenfältet uppvisar stabil utjämning mot topografi
- Mät prestanda — hydro och diffusion är primära GPU-kandidater längre fram

---

### Fas 2 — Diskret flora

Mål: Ersätt biomassfältet med en levande florapopulation. Ekologisk hypotes testas här.

- Allokera flora direkt i `OrganismStore`-arrayerna — inga separata objekt, ingen wrapper-klass
- Skriv `uptake_system()`, `growth_system()`, `dispersal_system()` som pass — spridning sker via `grid.cells_within()`
- Konstruera aktiva delmängder för florapass i början av varje tick
- Låt agenternas konsumtion läsa florans celler via `OrganismStore` och cellindexet
- Ta bort det kontinuerliga biomassfältet när floran bär dess funktion

**Validering:** Uppstår stabila florapopulationer? Uppstår koevolution med konsumenter? Är prestanda acceptabel vid tusentals floraindivider?

Om den ekologiska hypotesen stämmer — fortsätt. Om dynamiken inte fungerar — revidera floramodellen, inte kärnan.

---

### Fas 3 — Spatial integration

Mål: Alla organismer använder samma spatialindex.

- Flytta agenternas cellbaserade uppslag till `OrganismStore`-indexet
- Befintlig sensing och interaktionslogik läser via indexet och `grid.cells_within()`
- En agent som söker mat hittar floraindivider via samma mekanism som den hittar andra agenter

Nu är det rumsliga lagret gemensamt för allt liv.

---

### Fas 4 — Subsystem som pass

Mål: Agenternas subsystem migreras ett i taget till fasmodellen.

Ordning, från lättast till svårast:

1. Basal metabolism och åldrande → `metabolism_system()`
2. Sensing → `sense_system()` — arbetar mot aktiv delmängd med `sense_radius > ε` och uppfyllt frekvensvillkor
3. Locomotion → `move_system()` — arbetar mot aktiv delmängd med `mobility > ε`
4. Interaktion och predation → `interaction_system()`
5. Reproduktion → `reproduction_system()`
6. Intern fysiologi, M_target-logik, reparation — sist

Under denna fas överlämnar de gamla `Agent`- och `Body`-objekten gradvis source of truth till `OrganismStore`. Till slut är de tunna wrappers eller försvinner.

---

### Fas 5 — Acceleration

Mål: Identifiera faktiska hotspots och accelerera dem.

- Profilera fasmodellen under realistisk belastning med blandad population
- Flytta hydro, diffusion och fältuppdateringar till GPU via CuPy om de dominerar
- Accelerera sensing-passet med Numba om det fortfarande är bottleneck
- Utvärdera om en Rust-kärna är motiverad baserat på verkliga mättal

Inget accelereras förrän det är mätt.

---

## Tekniska riktlinjer

**Inga tunga Python-objekt i den heta loopen.** `OrganismStore` är arrayer, inte objektlistor.

**Varje systempass är en funktion.** Den tar arrayer och returnerar eller muterar arrayer. Inga dolda sidoeffekter.

**Aktiva delmängder byggs en gång per tick och är immutabla under ticken.** Födslar och dödsfall registreras men träder i kraft först vid nästa tick.

**`cell_idx` är en invariant, inte ett fält som uppdateras opportunistiskt.** Den hålls konsistent med `pos_x`, `pos_y` efter varje positionsuppdatering, oavsett orsak.

**Slotindex och organism-ID är distinkta begrepp.** Slotindex återanvänds; organism-ID gör det aldrig. Kod som refererar till en individ över tid använder `id`, inte `i`.

**Alla bevarade storheter uppdateras via tvåstegsmetod.** Flöden beräknas från föregående tillstånd och appliceras simultant som nettoförändringar. Inga in-place uppdateringar under flödesberäkning. Det gäller vatten, näring, detritus och framtida transportabla fält.

**Framkomlighet är en relation, inte en celltyp.** Huruvida en cell är passerbar avgörs i locomotion- och decision-passen utifrån cellens fysiska tillstånd och organismens kapaciteter. Världen lagrar inte "är passerbar för X".

**Passiv drift hanteras i Hydro-passet, inte i Locomotion.** Rörelse driven av vattenflöde sker före biologisk sensing och decision. Det håller rörelse som fysik åtskild från rörelse som beteende.

**Biologin läser world-fält — den skriver inte till forcing-fält.** Forcing-fält (`rain_input` etc.) styrs av världskonfiguration, inte av organismer.

**Geometrin ägs av Grid, ingenting annat.** Direkta koordinatoperationer utanför `Grid` är fel. Det gäller även viewer — visualiseringen översätter cell-ID:n via `Grid`, inte tvärtom.

**Subsystemstate hålls i separata tilläggsarrayer.** De allokeras bara för organismer som har kapaciteten och har dokumenterad ägarskap. Kärnan ska inte växa för att rymma state som tillhör ett specifikt subsystem.

**Source of truth är explicit.** Under migration: vilket lager äger vilket fält ska framgå av kod och dokumentation. Aldrig dubbel uppdatering av samma fält.

**Biologisk validering före arkitekturinvestering.** Om en ekologisk mekanism inte ger önskad dynamik, revidera biologin — inte infrastrukturen.

**Genomet expanderas inte i förtid.** Ekologisk och evolutionär stabilitet verifieras med enkelt lågdimensionellt genom innan representationen kompliceras.

---

## Vad vi bygger mot

En värld där primärproduktionen lever, konkurrerar och evolerar. Där konsumenter formar växtligheten och växtligheten formar konsumenterna. Där komplexitet uppstår i det evolutionära rummet snarare än i koden.

En motor där tiotusentals enkla organismer och hundratals komplexa kan samexistera utan att beräkningskostnaden kollapsar.

En arkitektur där nästa kapacitet — ett nytt subsystem, en ny livsform, ett nytt selektionstryck — kan läggas till som ett nytt pass, utan att röra det som redan fungerar.

En hexagonal värld med levande topografi och strömmande vatten, där ekologiska nischer — stränder, flodfåror, lågland, höjder — uppstår ur samma enkla fysik, utan att kodas uppifrån.

---

*Version 0.9. Revideras när ny kunskap motiverar det. Kompass, inte kontrakt.*