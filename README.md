# nep-process — Arkitekturmanifest

*Version 1.0 — April 2026*

Senast uppdaterad 2026-08-07

---

## Köra

Simuleringen behöver `requirements.txt`. En viewerklient behöver bara
`requirements-viewer.txt` — numpy och pygame — eftersom `viewer_pygame.py`,
`viewframe.py` och `grid.py` inte importerar simuleringen.

**Lokalt, med fönster:**

```bash
pip install -r requirements.txt
python run_population.py --size 64 --T 2000
```

**Huvudlöst, utan fönster.** Det vanliga sättet att köra längre experiment.
`--check-every` kör invariantsviten och ger exitkod skild från noll om något
brister; `--snapshot-every` skriver PNG-bilder av världen.

```bash
mkdir -p runs/exp1
python run_headless.py --ticks 40000 --seed 1 --stats \
    --check-every 1000 --snapshot-every 5000 --snapshot-dir runs/exp1
```

Kör aldrig två samtidiga körningar mot samma katalog — de skriver över
varandras loggar.

### Viewer på en annan maskin

Simuleringen kan sända bildrutor till en eller flera viewers över nätet.
Servern packar en gång per bildruta oavsett hur många som tittar, väntar
aldrig på nätet och packar inte alls om ingen är ansluten. En långsam klient
hoppar över bildrutor i stället för att bromsa simuleringen.

På maskinen som räknar:

```bash
python run_headless.py --ticks 100000 --seed 1 --serve 8765 --serve-control
```

`--serve-control` låter anslutna viewers pausa körningen med mellanslag. Utan
flaggan är strömmen enkelriktad och tangenten svarar med ett nej i
serverloggen. Varje ansluten klient får styra när flaggan är på, och vem som
pausade visas i HUD:en — förtroendegränsen är redan tunneln.

På maskinen som tittar:

```bash
pip install -r requirements-viewer.txt
python viewer_client.py --host arbetsstationen
```

Servern binder till `127.0.0.1` som förval. Det är en säkerhetsposition, inte
en begränsning: nå den utifrån genom en SSH-tunnel, så sköter SSH
autentisering och kryptering och simuleringen behöver varken lösenord eller
certifikat.

```bash
ssh -N -L 8765:localhost:8765 arbetsstationen   # i ett eget skal
python viewer_client.py                         # förvalet är localhost
```

`--serve-host 0.0.0.0` binder brett i stället, men då kan vem som helst på
nätet se världen.

Klienten kan startas före simuleringen. Den öppnar sitt fönster direkt,
väntar, och ansluter av sig själv när servern dyker upp — och återansluter om
servern startas om. Läge, färgaxel, ytfördelning, gamma och zoom är lokala val
och kostar ingen rundtur.

Fönstret är ett fönster: världen ritas en gång, och där fönstret sträcker sig
utanför den fylls med tom bakgrund. Är världen mindre än fönstret centreras
den. Fönstret får världens form vid första bildrutan, nedskalad så att den
ryms på skärmen. En värld på 64×256 celler är nästan fyra gånger högre än den är bred,
och `--screen-frac` styr hur stor del av skärmen den får ta. `--window 400x900`
sätter måtten explicit i stället. Fönstret går att ändra storlek på under
körning.

`TERRANG` visar höjden med backljus, `VATTEN` markfukten som andel av
fältkapacitet. `K` lägger höjdkurvor ovanpå, med jämn ekvidistans och var femte
som tjockare huvudkurva. De ritas **ovanpå vattnet** och inte under: en kurva
som löper ut i en sjö visar vilken nivå sjöytan står på, och det är hela
poängen. Kurvorna är riktiga linjer och inte färgade celler: höjden tolkas som ett sampel
vid cellcentrum, och hexgittrets dual — ett triangelnät — gör interpolationen
entydig. Ekvidistansen väljs ur världens egen relief och står i HUD:en;
`render_frame.py --kurvor --kurvsteg 5` sätter den explicit, vilket ofta är
läsbarare i kuperad terräng. Båda kräver ett scenario med terräng; i en platt värld är de tomma
och bildrutan bär inte fälten alls. Hav, sjöar och vattendrag ritas ovanpå
*varje* läge, eftersom en karta i FLORA-läge annars ser ut som om havet vore torr
mark utan växtlighet.

Kör samma commit i båda ändar. Klient och server jämför protokollversion vid
anslutning och säger ifrån om de skiljer sig.

**Tangenter i viewern**

| Tangent | Verkan |
|---|---|
| `1`–`8` | läge: CB, B, C, FLORA, TEMP, CLAIM, TERRANG, VATTEN |
| `W` | vatten av och på (hav, sjöar och vattendrag ovanpå alla lägen) |
| `K` | höjdkurvor av och på; ekvidistansen visas i HUD:en |
| `F` | ytfördelning: vinkelkilar eller stippling |
| `T` | florans färgaxel |
| `A` / `H` | djur av och på, HUD av och på |
| hjul, `.` / `,` | zooma in och ut |
| dra, piltangenter | panorera |
| `0` | visa hela världen igen |
| `+` / `-` | gamma |
| mellanslag | paus (kräver `--serve-control` vid fjärrkörning) |
| `Q` / `Esc` | avsluta |

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

Den konceptuella passordningen är:

1. **Hydro** — vatten flödar enligt potential och kontinuitet; flytande organismer transporteras passivt
2. **Transport** — diffusion av abiotiska lösta ämnen
3. **Decomposition** — nedbrytning och återföring av materia
4. **Uptake** — organismer tar upp näring från lokal cell
5. **Growth** — massa byggs från energi mot genetisk target
6. **Sensing** — organismer vars `sense_radius > ε` och sensingfrekvens är uppfylld samlar information från lokala celler via `grid.cells_within()` och spatialindexet
7. **Decision** — mål väljs utifrån lokalt insamlad information
8. **Locomotion** — aktiv rörelse för organismer med `mobility > ε`
9. **Interaction** — betning, predation och parning mot lokalt upptäckta kandidater
10. **Metabolism** — underhållskostnad, skada, reparation, åldrande
11. **Reproduction** — reproduktion och spridning
12. **Death** — döda organismer omvandlas till detritus; deras slotindex frigörs

Världsprocesserna (Hydro, Transport, Decomposition) täcker alla celler. Biologiska pass (Uptake och nedåt) arbetar mot aktiva delmängder av organismer.

Varje pass ska arbeta direkt mot `OrganismStore`-arrayer och slotlistor, inte mot Python-objekt. In-place mutation av store-arrayer är att föredra framför att returnera kopior. Det som utesluts är dolda sidoeffekter och delad global state — inte effektiv arrayuppdatering.

Passordningen ovan är konceptuell, inte ett påbud om ett separat Python-anrop per punkt. Pass som delar tät dataåtkomst och är biologiskt sammanhängande — till exempel sensing och decision, eller locomotion och omedelbar interaktionsförberedelse — kan fusioneras pragmatiskt när Python-overhead annars dominerar. Separation of concerns är ett mål; onödig iteration är ett hinder. Komplex intern fysiologi med stark intern koherens, som energibudget, katabolism och reparation, behöver inte splittras sönder bara för att fasmodellen erbjuder separata namn. Den kan migrera samlad när strukturen är redo att bära det.

Passiv drift — förflyttning av flytande organismer med vattenflödet — sker i Hydro-passet, före biologisk sensing och decision. Det håller isär rörelse som fysik och rörelse som beteende.

### 5. Dataorienterad kärna

Alla organismer delar samma kärnstore och samma kapacitetsmodell. Kärnfälten lagras i täta parallella arrayer — en per fält — där slotindex är nyckel.

Tilläggsstate för specifika subsystem allokeras i separata tilläggsarrayer med tydlig ägarskap. Det bryter inte den gemensamma ontologin så länge frånvarande kapaciteter inte kostar något att bära.

### 6. Abstrakt geometri

Ingen del av biologin — inte ett enda systempass, inte sensing, inte spridning, inte rörelse — ska innehålla hårdkodade antaganden om världens geometri.

Allt rumsligt arbete sker via ett väldefinierat `Grid`-gränssnitt. Det är den enda plats där geometrin existerar. Det gör det möjligt att byta geometri utan att röra biologin.

**Geometrin äger både topologi och metrik.** Grannrelationer, avstånd och wrap är den ena halvan; hur stor en cell är den andra. Cellen har arean 100 m², vilket ger längdenheten 10 meter och centrumavståndet 10,746 — `d = sqrt(2A/sqrt(3))` för en regelbunden hexagon. Arean är det runda talet och avståndet dess härledda följd, eftersom arean är vad ekologin räknar med: biomassa, näring och ljus per cell.

**Metriken måste vara en enda.** Höjd, vattendjup och vågrätt avstånd delar samma enhet, eftersom fri yta är summan av de två första och lutning kvoten mellan den tredje och den första. En modell med två längdskalor kan inte ha lagbunden fysik: en lutning blir ett tal utan innebörd, och varje konstant som uttrycks per meter går inte att sätta. Projektet levde med det tills höjden fick sin enhet — och upptäckte då att faunans fart låg tvåhundra gånger fel. Se `docs/varldens-skala.md`.

**Vad geometrin inte äger.** Klimat, latitud och andra villkor som världen möter utifrån är inte geometriska egenskaper. `Grid` bar en gång `cell_lat`, och det var en felplacering: cellformen ska inte veta var på en planet världen ligger. Sådant hör till fysiklagret.

### 7. Fysiklager

Simulatorn skiljer explicit mellan tre nivåer:

- **Fysiklagret** definierar lagar, storheter och regler
- **Världslagret** instansierar dessa lagar för konkreta cellfält
- **Biologin** läser härledda tillstånd ur världen

Fysiklagret definierar:
- grundstorheter: massa, energi, volym, tid
- enheter och skalor: en längdenhet, en tidsenhet, en massenhet — namngivna, inte underförstådda
- bevarandeprinciper: kontinuitet för materia och energi; bevarade storheter uppdateras alltid via tvåstegsmetod
- generella transportlagar: diffusion och gradientdrivet flöde
- randvillkor: det som ligger utanför världens rand men bestämmer villkoren inom den

Fysiklagret opererar inte direkt på världens tillstånd. Det definierar reglerna som världspass tillämpar.

**Energi och massa** är kopplade storheter i hela modellen. Biomassa representerar lagrad kemisk energi. Energi används för tillväxt, underhåll och arbete och överförs mellan organismer och till detritus vid dödsfall och konsumtion. Fysiklagret definierar konverteringsfaktorer och grundläggande kostnadsskalor. Biologiska systempass implementerar dessa relationer men bryter inte mot dem.

**En storhet utan enhet är ingen fysisk storhet.** Den ser ut som ett tal och beter sig som ett tal, men den går inte att ställa mot något utanför modellen — och därmed går den inte heller att sätta fel på ett sätt som märks. Projektet har gjort det felet flera gånger: höjden var enhetslös och gjorde lutningen godtycklig, `water_extract_frac` var per tick där den skulle vara per månad, lapse rate valdes mot vad den gjorde med tillväxtgrinden i stället för mot 6,5 grader per kilometer. Varje ny konstant ska kunna uttryckas i modellens enheter och jämföras med ett värde från verkligheten. Kan den inte det är den en kalibrerad fri parameter, och det ska stå i klartext där den definieras.

**Randvillkor är fysik, inte biologi.** Världen är åtta tusen till en miljon celler, alltså ett landskap på någon kilometer till några mil. Planeten fortsätter utanför randen, och det som kommer därifrån — solens gång, årstidens djup, luftmassornas tröghet — kan inte simuleras inifrån. Modellen anger i stället **var världen ligger**: en latitud och en kontinentalitet. Ur latituden faller solhöjd, dagslängd och insolation som exakt astronomi utan en enda kalibrerad konstant; ur de två tillsammans faller årsmedeltemperatur, årstidens amplitud och dess eftersläpning som en uppslagsformel anpassad mot jordens klimatologi en gång. Se `klimat.py`.

Det är motsatsen till den latitudgradient modellen en gång hade. Världen ligger **på** en breddgrad — den spänner inte över flera — så positionen är en skalär i fysiklagret och inte ett fält över celler. Vad randvillkoret ger är att klimatet blir härlett i stället för valt, och att ett experiment uttrycks som en flyttning i stället för som fyra nya tal.

Vad som inte härleds ur positionen: nederbörden. Sahara och monsun-Indien ligger båda kring tjugo grader nord och skiljer tre tusen millimeter — spridningen inom en breddgrad är större än signalen mellan breddgrader, och nederbörden förblir därför världens egen parameter.

**Modellen har en känd inkonsistens.** Faunans fart är omkring tvåhundra gånger för låg för en tvåkilos organism vid tio meters celler, och den kan inte höjas mer än ungefär femtio gånger utan att djuret rör sig längre än sitt sensingavstånd på ett tidssteg. Antingen kortas tidssteget, eller så accepteras att faunan är långsammare än sin kropp motiverar, eller så byts kroppsstorleken. Att skriva ut det här är avsiktligt: en inkonsistens som står i manifestet är en öppen fråga, medan en som bara finns i koden är en dold felkälla.

### 8. Lokal upptäckt och situerad interaktion

Organismer har inte global tillgång till `OrganismStore` eller till andra organismer som abstrakta objekt i systemet. De kan bara upptäcka andra organismer genom lokal sensing i världen.

All biologisk upptäckt och interaktion sker via:
- organismens aktuella position och `cell_idx`
- `Grid` och lokala cellgrannskap via `grid.cells_within()`
- världens spatialindex — listor av slotindex per cell
- organismens egna kapaciteter: `sense_radius`, `sense_rate`, `attack_capacity`

Det är förbjudet för biologiska pass att iterera globalt över alla organismer för att hitta bytesdjur, partners eller hot. Kandidater måste hämtas från lokala celler inom organismens sensing- eller interaktionsradie.

Det är inte bara en prestandaprincip. Det är en del av modellens ontologi: organismer är i världen och kan bara veta det världen lokalt gör tillgängligt för dem. En predator hittar inte sitt byte för att den söker i en global lista — den hittar det för att de befinner sig i samma del av världen.

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

En organism bär ett fast antal genloci — initialt i storleksordningen 8–16 — kodade som kontinuerliga flyttal. Kapacitetsprofilen härleds via skalningsfunktioner i `phenotype.py`, som äger tolkningen av traitsemantiken. Mutation och arv ägs av `genetics.py`. Flora och djur kan ha distinkta men koordinerade locus-utrymmen inom samma representation.

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

**Forcing-fält** — extern input till systemet. De styrs inte av organismer och är i den initiala modellen parametriska fält per cell, inte biologiska tillstånd.

```
rain_input        — vattentillförsel per cell per tick
spring_input      — lokala källflöden
infiltration      — vattenförlust till mark per cell
evaporation       — vattenförlust till atmosfär per cell
```

Biologin läser primära och härledda fält. Den skriver aldrig till forcing-fält.

### Kadensklasser

Ett världsfält har både en ägare och en kadens. Ägaren säger vem som skriver; kadensen säger hur ofta fältet behöver röras. Att en tom cell ska vara billig är samma princip som att en organism utan en kapacitet inte ska kosta något för den.

**Statiska** — sveps aldrig. Är värdet dessutom rumsligt konstant lagras det som skalär, inte som array; fältet materialiseras först när något faktiskt varierar det. `elevation` och forcing-fälten så länge de är parametriska.

**Profilberoende** — fältet är en funktion av en enda koordinat och lagras som sådan, i formen *profil plus eventuella per-cell-modifierare*, där modifieraren kan vara frånvarande och då inte kostar något. Klimatet var klassens instans så länge det varierade med latitud och lagrades per band. Det gör det inte längre: världen är ett landskap och inte en planet, så klimatet är en skalär i tiden plus höjdens statiska modifierare — samma form med profillängd ett. Se `docs/varldens-skala.md`. Klassen står kvar som form; den väntar på nästa fält som faktiskt varierar med en enda koordinat.

**Glest dynamiska** — fältet bär en aktiv mängd och pass arbetar mot den. Kontraktet är att en cell utanför den aktiva mängden är exakt noll, vilket gör glesheten till en prövbar egenskap i stället för ett antagande. `detritus` och `carcass`.

**Tätt dynamiska** — fullt svep är genuint motiverat. `nutrient` när det diffunderar; näring som sprider sig har inget glest stöd.

**Nätverksdynamiska** — fältet är tätt men riktat. Varje cell rörs exakt en gång, i en ordning som geometrin och terrängen tillsammans bestämmer en gång för alla. Kostnaden är O(n) med perfekt lokalitet när ordningen lagras som en permutationsarray. `discharge` i hydro.

Ett pass ska skrivas mot fältets kadensklass från början. Skrivs det tätt när fältet är glest måste det skrivas om.

### Hydrologi

Vatten representeras som ett inkompressibelt medium diskretiserat per cell. Land och vatten är inte två ontologiskt skilda världar — de är två regimer i samma fältmodell, styrda av topografi och vattenmängd.

Fri yta definieras som:

```
surface = elevation + water
```

Flöde sker från cell till granncell baserat på skillnaden i fri yta. Flödet är lokalt och gradientdrivet. Alla flöden beräknas från cellernas tillstånd vid början av passet och appliceras simultant som nettoförändringar. Kontinuitet upprätthålls strikt: total utström från en cell får aldrig överstiga tillgängligt vatten.

**Flödet löses till jämvikt, inte som en transient.** Tidssteget är omkring femton timmar, och vatten hinner på den tiden korsa hela världen — flera storleksordningar snabbare än en organism rör sig. Ett explicit schema håller informationshastigheten under ungefär en cell per tick och skulle därför låta en flod behöva tio simulerade månader på att nå havet. Den fördröjningen är numerik utan biologisk motsvarighet. Hydro löser i stället stationärt tillstånd varje tick genom att dränera längs en förberäknad ordning: fysiken är oförändrat lokal, gradientdriven och kontinuitetsbevarande, det är lösningsmetoden som skiljer. Se `docs/geologin-och-vattnet.md`.

Vattentillförsel (`rain_input`, `spring_input`) och förluster (`infiltration`, `evaporation`) tillämpas som käll- respektive sänktermer per cell inom hydro-passet.

Celler som faller under `sea_level` utgör en hydrologisk randregim. Inflöde till dessa celler lämnar den explicita landvattenbudgeten — de absorberar vatten utan att ackumulera tryck mot omgivningen. Det ger naturliga kustlinjer och havsbassänger utan en separat oceanmodell.

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

Traitsemantiken för flora ägs av `phenotype.py`. Mutation och arv ägs av `genetics.py`. Flora reproducerar sig asexuellt — avkomman är en kopia av moderplantan med liten mutation — medan djur använder sexuell reproduktion med rekombination. Skillnaden ligger i reproduktionsmekaniken, inte i en hårdkodad ontologisk artgräns.

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
- Ge varje floraindivid ett ärftligt traitpaket via `phenotype.py`/`genetics.py`; tillväxt och spridning läser dessa traits, inte globala hårdkodade parametrar
- Skriv `uptake_system()`, `growth_system()`, `dispersal_system()` som pass — spridning sker via `grid.cells_within()`
- Konstruera aktiva delmängder för florapass i början av varje tick
- Låt agenternas konsumtion läsa florans celler via `OrganismStore` och cellindexet
- Ta bort det kontinuerliga biomassfältet när floran bär dess funktion

**Validering:** Uppstår stabila florapopulationer? Uppstår koevolution med konsumenter? Differentierar sig florastrategier via selektion? Är prestanda acceptabel vid tusentals floraindivider?

Om den ekologiska hypotesen stämmer — fortsätt. Om dynamiken inte fungerar — revidera floramodellen, inte kärnan.

---

### Fas 3 — Spatial integration

Mål: Alla organismer använder samma spatialindex.

- Flytta agenternas cellbaserade uppslag till `OrganismStore`-indexet
- Sensing och interaktionslogik läser via spatialindexet och `grid.cells_within()` — aldrig via global iteration
- En agent som söker mat hittar floraindivider via samma lokala mekanism som den hittar andra agenter

Nu är det rumsliga lagret gemensamt för allt liv, och all upptäckt är lokal.

---

### Fas 4 — Subsystem som pass

Mål: Agenternas subsystem migreras ett i taget till passmodellen, med start i det som tydligast vinner på separationen.

Ordning, från lättast till svårast:

1. Basal metabolism och åldrande → `metabolism_system()` — liten yttre koppling, naturlig startpunkt
2. Sensing → `sense_system()` — arbetar mot aktiv delmängd med `sense_radius > ε` och uppfyllt frekvensvillkor
3. Locomotion → `move_system()` — arbetar mot aktiv delmängd med `mobility > ε`
4. Interaktion och predation → `interaction_system()`
5. Reproduktion → `reproduction_system()`
6. Komplex intern fysiologi — energibudget, katabolism, reparation, termoreglering — migreras samlad sist, när strukturen är redo att bära dess interna koherens utan att spränga logiken

Under denna fas överlämnar de gamla `Agent`- och `Body`-objekten gradvis source of truth till `OrganismStore`. Till slut är de tunna wrappers eller försvinner. Dubbelrepresentation under övergången är det största riskmomentet: varje fält ska vid varje tidpunkt ha exakt en ägare.

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

**Inga tunga Python-objekt i den heta loopen.** `OrganismStore` är arrayer, inte objektlistor. Pass arbetar mot slotlistor och numpy-arrayer, inte mot `Agent`-instanser.

**Varje systempass arbetar in-place mot store-arrayer.** Det som utesluts är dolda sidoeffekter och delad global state — inte effektiv arrayuppdatering. Att returnera kopior av stora arrayer är ett antimönster.

**Aktiva delmängder byggs en gång per tick och är immutabla under ticken.** Födslar och dödsfall registreras men träder i kraft först vid nästa tick.

**`cell_idx` är en invariant, inte ett fält som uppdateras opportunistiskt.** Den hålls konsistent med `pos_x`, `pos_y` efter varje positionsuppdatering, oavsett orsak.

**Slotindex och organism-ID är distinkta begrepp.** Slotindex återanvänds; organism-ID gör det aldrig. Kod som refererar till en individ över tid använder `id`, inte `i`.

**Biologiska pass får inte göra globala sökningar i OrganismStore.** Upptäckt av andra organismer måste ske via lokala celluppslag i spatialindexet, begränsade av organismens `sense_radius` eller interaktionsradie. Det är förbjudet att iterera över hela populationen för att hitta bytesdjur, partners eller hot.

**Alla bevarade storheter uppdateras via tvåstegsmetod.** Flöden beräknas från föregående tillstånd och appliceras simultant som nettoförändringar. Inga in-place uppdateringar under flödesberäkning. Det gäller vatten, näring, detritus och framtida transportabla fält.

**Framkomlighet är en relation, inte en celltyp.** Huruvida en cell är passerbar avgörs i locomotion- och decision-passen utifrån cellens fysiska tillstånd och organismens kapaciteter. Världen lagrar inte "är passerbar för X".

**Passiv drift hanteras i Hydro-passet, inte i Locomotion.** Rörelse driven av vattenflöde sker före biologisk sensing och decision. Det håller rörelse som fysik åtskild från rörelse som beteende.

**Biologin läser world-fält — den skriver inte till forcing-fält.** Forcing-fält (`rain_input` etc.) styrs av världskonfiguration, inte av organismer.

**Traitsemantik ägs av `phenotype.py`. Mutation och arv ägs av `genetics.py`.** Ny biologisk logik byggs inte i `Population` eller i world-pass — den byggs i rätt modul och anropas därifrån.

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

*Version 1.0. Revideras när ny kunskap motiverar det. Kompass, inte kontrakt.*