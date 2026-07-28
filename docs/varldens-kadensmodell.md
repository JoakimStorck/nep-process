# Designskiss — världens kadensmodell

*Juli 2026. Underlag för Steg 3 i TODO.md. Status: förslag, inte beslut — uppdateras eller markeras som ersatt när steget byggs.*

---

## Problemet

Manifestet klassar världsfält efter **ägarskap**: primära tillståndsfält, härledda fält, forcing-fält. Den indelningen säger vem som skriver vad. Den säger ingenting om hur ofta ett fält behöver röras.

Vid nuvarande världsstorlek spelar det ingen roll. Vid den storlek terrängburen hydrologi kräver spelar det roll för allt.

Uppmätt vid en miljon celler, per tick:

```
hydro_pass            13,9 ms
decomposition_pass     7,6 ms
_update_temperature    5,0 ms
ledgersummor           2,6 ms
                      ~26,6 ms
```

Ungefär 15 av de 26 millisekunderna är inte beräkningskostnad utan representationsfel:

- `_update_temperature` beräknar en miljon värden av en funktion som bara beror på latitud. Det finns H distinkta värden, inte H·W.
- `decomposition_pass` sveper en miljon celler för att multiplicera nollor med en avklingningsfaktor. Detritus är nollskilt i storleksordningen 0,3 % av cellerna.
- Ledgern gör tre fulla summeringar per tick över storheter som kunde underhållas inkrementellt.

Dessutom är **samtliga världsfält i dag konstanta arrayer**. `rain_input`, `evaporation`, `elevation` och de andra lagras som miljonelementsarrayer med ett enda unikt värde.

Utanför världspasset finns en post till: spatialindexet gör `cumsum` över `n_cells` två gånger per tick. Vid en miljon celler och tiotusen organismer är det två miljoner operationer för att indexera tiotusen individer — mer än hela faunapipelinen kostar.

## Principen

Manifestet säger redan att en kapacitet som är noll varken ska kosta energi eller CPU, och att systempass ska arbeta mot aktiva delmängder i stället för hela populationen.

**Samma princip gäller celler.** En tom cell ska vara billig. Ett statiskt fält ska inte sopas. Ett fält som är en funktion av en koordinat ska inte lagras som en funktion av två.

Det här är arkitektur, inte optimering — av exakt samma skäl som lokal upptäckt är arkitektur och inte en prestandaåtgärd. Skrivs hydro tätt måste det skrivas om.

---

## Kadensklasser

Varje världsfält får en deklarerad kadensklass utöver sin ägarskapsklass.

### Statiska

`elevation`, och forcing-fälten så länge de är parametriska.

Sveps aldrig. Om värdet dessutom är rumsligt konstant lagras det som skalär, inte som array — en uniform `rain_input` behöver inte en miljon element. Fältet materialiseras först när något faktiskt varierar det.

Övergången skalär → array ska vara en engångshändelse som fältet självt hanterar, så att anropare inte behöver veta vilken form det har.

### Profilberoende

`T_cell`, `g_cell`.

Lagras som en profil med längd lika med antalet latitudband, uppdaterad en gång per tick. Läsning för en cell sker via en statisk bandindexering från `Grid`.

Det kräver en ny geometrisk egenskap: `Grid.cell_band`, ett heltal per cell som identifierar vilka celler som delar latitud. För kvadrat och för hex med radoffset är det raden. Begreppet är geometriskt och hör därför till `Grid`, inte till världen.

Läsning blir `T_band[grid.cell_band[c]]` — en gather i stället för en omberäkning. Vektoriserat för en mängd celler är det en indexering, inte ett svep över världen.

**Viktig begränsning:** klassen håller bara så länge klimatet enbart beror på latitud. Vill vi ha kontinentalitet — land som växlar temperatur snabbare än vatten — blir temperaturen genuint tvådimensionell. Designen ska därför vara *profil plus eventuella per-cell-modifierare*, inte hårdkodad O(H). Modifieraren kan vara frånvarande, och då kostar den ingenting.

### Glest dynamiska

`detritus`, och sannolikt `water` i en terrängvärld där merparten är torr.

Fältet bär en aktiv mängd: en lista över celler med nollskilt värde, plus en medlemskapsflagga per cell för konstant-tids test. Pass arbetar mot listan.

Kontrakt:
- att skriva ett nollskilt värde till en cell lägger till den i mängden
- ett värde som faller under epsilon nollställs exakt och tas bort ur mängden
- ett pass får aldrig läsa ett fält och anta att inaktiva celler kan vara nollskilda

Det tredje är det som gör klassen säker att lita på. Utan det blir glesheten en optimering som kan vara fel; med det är den en invariant som kan prövas.

### Tätt dynamiska

`nutrient` när det diffunderar.

Fullt svep är genuint motiverat. Men diffusion har en front: om näring bara uppstår där detritus bryts ner växer stödmängden utåt från källorna. Ett glest fält som får växa monotont är därför ett rimligt mellanläge under lång tid, och kan degradera till tätt svep när stödet passerar en tröskel.

Beslutet kan skjutas tills `transport_pass` faktiskt gör något.

---

## Spatialindexet

CSR-strukturen är i dag `cell_counts` med längd `n_cells`, `cell_offsets` med längd `n_cells + 1`, och `cell_slots` med längd `capacity`. Ombyggnaden gör `cumsum` över hela cellrymden.

Det är rätt struktur när celler är få och organismer många. Vid en miljon celler och tiotusen organismer är förhållandet omvänt: som mest tiotusen celler är bebodda.

**Förslag:** indexera bara bebodda celler. Sortera organismernas slots efter cell, plocka ut de unika cellerna och deras intervall, och slå upp via `searchsorted`. Kostnaden blir O(n·log n) i organismer utan någon term i `n_cells`.

Priset är att uppslagning går från konstant tid till O(log k) där k är antalet bebodda celler. För tiotusen organismer är det ungefär fjorton jämförelser, och `searchsorted` är vektoriserad för satsvisa uppslag — sensing gör redan sina uppslag i batch.

Det här hör ihop med Steg 4:s vektorisering av `rebuild_spatial_index` och bör göras i samma ändring, inte som två omskrivningar av samma kod.

---

## Hydro: det som måste vänta

Glesning av hydro är rätt teknik men fel tidpunkt.

Principen är en aktiv front: en cell är aktiv om dess fria yta skiljer sig från någon grannes med mer än epsilon, eller om den tar emot forcing, eller om en granne var aktiv förra ticken. En sjö i jämvikt är inaktiv. Torr högmark är inaktiv. Flodfåror och strandzoner är aktiva.

Två saker gör att den inte kan designas nu.

**Epsilon är inte känt.** Väljs det för stort stannar flöden som borde fortsätta; för litet och sjöar darrar för evigt utan att någonsin bli inaktiva. Värdet beror på hydrons faktiska numerik.

**Massbevarandet måste hålla över gränsen aktiv/inaktiv.** En inaktiv cell får varken läcka eller ackumulera. Det är precis den sortens fel som är osynligt tills det har körts länge.

Båda kräver en tät referensimplementation att validera mot. Metoden finns redan i projektet: kör tät och gles parallellt på samma seed och jämför fälten inom tolerans, på samma sätt som `check_body_store_mirror` jämför `Body` mot store.

Hydro byggs alltså tätt först, men **mot kadensstrukturen** — så att glesningen blir ett byte av exekveringsstrategi, inte en omskrivning av passet.

---

## Vad som ska in i manifestet

Kadensklassen bör stå bredvid ägarskapsklassen i avsnittet om världsfält, och principen bör formuleras uttryckligen:

> Ett världsfält har både en ägare och en kadens. Ägaren säger vem som skriver; kadensen säger hur ofta fältet behöver röras. Statiska fält sveps aldrig. Fält som är funktioner av en enda koordinat lagras som sådana. Glesa fält bär en aktiv mängd och pass arbetar mot den. Att en tom cell ska vara billig är samma princip som att en organism utan en kapacitet inte ska kosta något för den.

---

## Föreslagen ordning

1. **Kadensdeklaration och de tre riskfria rättningarna.** Latitudprofilen som O(antal band), glest detritus, inkrementell ledger, skalära konstantfält. Mätbart: världspasset vid en miljon celler ska gå från ~26 ms till storleksordningen 10.
2. **Spatialindexet över bebodda celler.** Tillsammans med Steg 4:s vektorisering.
3. **Hydro tätt, mot kadensstrukturen.**
4. **Hydro glest, validerat mot den täta.**

---

## Öppna frågor

**Var bor den aktiva mängden?** Antingen i fältet självt som en liten klass, eller i världen som parallella strukturer. Det första ger bättre inkapsling men inför objekt i ett lager som hittills varit rena arrayer.

**Ska glesa fält exponera samma gränssnitt som täta?** Om biologin läser `world.detritus[cell]` ska det fungera oavsett kadensklass. Det talar för att glesheten är en intern egenskap hos fältet och inte syns utåt — men då kan ett pass inte iterera över den aktiva mängden utan att bryta inkapslingen.

**Hur mäts det?** Förslag: en invariant som kontrollerar att inaktiva celler i ett glest fält verkligen är noll. Det gör glesheten till en prövbar egenskap i stället för ett antagande.
