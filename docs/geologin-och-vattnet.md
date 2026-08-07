# Geologin och vattnet

*Augusti 2026. Underlag för Steg 7 i TODO.md. Status: beslutad inriktning — de
fem punkterna under "Låsta beslut" är avgjorda och ska inte öppnas under
implementationen.*

---

## Mätningen som avgör designen

Frågan att ställa före allt annat är inte hur hydro ska implementeras utan
**vilken tidsskala vattnet rör sig på jämfört med tidssteget.**

`dt = 0,02` månader, alltså ungefär 14,6 timmar. Faunan rör sig uppmätt 37
cellbredder per månad, vilket är 0,74 cellbredder per tick — en cell är i
storleksordningen ett dygns förflyttning för ett djur. Vatten i en fåra rör sig
flera storleksordningar snabbare än så. **På ett tick hinner vatten korsa hela
världen.**

Det gör ett explicit grannflöde numeriskt olämpligt, inte bara långsamt.
Stabilitetsvillkoret håller informationshastigheten under ungefär en cell per
tick. En avrinningssignal från högland till hav i en 512 celler bred värld tar
då 512 tick, alltså tio simulerade månader. Årstiden är över innan vattnet
kommit fram. Den fördröjningen är ren numerik utan biologisk motsvarighet —
samma sorts fel som när headingen dekorrelerade på ett tick och rakheten föll
till 0,069.

Uppmätt i Numba, seriellt, på referensmaskinens sandlåda:

```
                              64x256    256x256   512x512
explicit grannflöde, tätt      0,73 ms   2,83 ms  11,40 ms
ren laplacian (transport)      0,07      0,30      1,22
routing över dräneringsnät     0,02      0,18      0,68
```

Det explicita schemat är alltså sexton gånger dyrare och samtidigt fel i
regimen.

**Slutsatsen är att vattnet är kvasistatiskt på tickskalan. Hydro löser jämvikt
i stället för att integrera en transient.**

Det är en avvikelse från manifestets formulering i *numerik*, inte i *fysik*.
Flödet är fortfarande lokalt, gradientdrivet och strikt kontinuitetsbevarande;
det löses till stationärt tillstånd per tick i stället för ett CFL-steg. Det
explicita schemat behålls som **valideringsorakel**: kör det till konvergens och
jämför fältvis mot routingen, på samma sätt som `check_body_store_mirror`
jämför `Body` mot store och som `docs/varldens-kadensmodell.md` föreskriver för
gles mot tät.

---

## Tvålagersdesignen

### Statisk geologi

Terrängen ändras inte under normala tick. Allt som *bara* beror på terrängen är
därför statiskt och förberäknas en gång — inklusive dräneringsnätet, vilket är
det grepp som gör hela konstruktionen billig.

```
elevation      f32[n]     höjd; promoveras från skalär
flow_to        i32[n]     brantaste fallets granne under fylld höjd; -1 i hav
flow_order     i32[n]     topologisk permutation, källa -> mynning
slope          f32[n]     höjdfall mot flow_to
upslope_area   f32[n]     uppströms cellantal — floderna som statiskt fält
lake_id        i32[n]     -1 utanför sjö
lake_hyps      per sjö    sorterade celltrösklar; ger nivå ur volym
sea_mask       bool[n]    manifestets hydrologiska randregim
```

Engångskostnad vid 262 144 celler, uppmätt: spektralsyntes 0,60 s,
prioritetsflod 1,26 s i Python-heapq (ska in i Numba), topologisk sortering
3 ms. Minne inklusive grannmatrisen omkring 15 MB.

### Dynamisk hydrologi

```
rain           profilberoende      rain_band[band] * oro[cell]
soil_water     tätt dyn., punktvis infiltration in, ET ut, avrinning ut
discharge      nätverksdynamiskt   ett svep över flow_order
lake_storage   per bassäng         f64[n_lakes] — skalärer, inte celler
water_depth    härlett             kanal ur discharge och slope, sjö ur nivå
submerged      härlett             water_depth över tröskel
flow_dir       härlett             riktning mot flow_to, styrka ur discharge
```

`soil_water` har ingen grannkoppling alls och kostar därför ett punktvis svep,
0,3 ms vid 262 k celler. `discharge` kostar ett svep över `flow_order`, 0,68 ms.

**Sjöarnas nivå är ett magasin per bassäng, inte ett djup per cell.**
Hypsometrin — cellerna i bassängen sorterade efter höjd — ger nivån ur volymen
med ett `searchsorted`. Det gör sjöar exakt bevarande och exakt stabila, och det
löser i förbigående det problem kadensdokumentet oroade sig för: *"för litet och
sjöar darrar för evigt utan att någonsin bli inaktiva."* En sjö kan inte darra
om den inte har ett per-cell-tillstånd att darra i. Epsilonfrågan uppstår aldrig.

### Nätverkskadensen

`discharge` passar inte i någon av de fyra befintliga kadensklasserna. Den är
varken statisk, profilberoende, gles eller tätt svepbar — den ackumuleras längs
en förberäknad topologisk ordning. Det är en femte klass och den hör in i
manifestet bredvid de andra fyra.

Kännetecknet är att fältet är tätt men **riktat**: varje cell rörs exakt en
gång, i en ordning som geometrin och terrängen tillsammans bestämmer en gång för
alla. Kostnaden är O(n) med perfekt lokalitet om ordningen lagras som en
permutationsarray, och den parallelliserar inte — vilket är rätt egenskap för
CPU-Numba och fel för GPU, och därmed förenlig med att floran läggs på GPU och
hydro inte gör det.

---

## Terränggeneratorn

Terrängen får inte läcka geometri. En FFT över ett `(h, w)`-rutnät är exakt det
kvadratantagande Steg 1 grävde ut. Rätt form är **spektralsyntes över
`grid.cell_center_x/y` och `grid.extent_x/y`**: en summa av periodiska plana
vågor med heltaliga vågtal över utsträckningen och amplitud som avtar med
vågtalet. Den är periodisk på torusen av konstruktion och geometriagnostisk.

Höjden byggs av tre termer:

```
elevation = (1 - r) * kontinent  +  r * strävhet  -  polarsänkan
```

**Kontinenten** är en brant spektralkomponent, `beta ~ 2,6` med få moder. Det är
den termen som avgör om floderna samlar sig. Uppmätt vid 256x256, havsandel
10 %, för olika spektrallutning:

```
beta = 1,8   sjöar 13,6 %   största flod    270 celler   inga vattendrag
beta = 2,2   sjöar  9,1 %   största flod  1 167
beta = 2,6   sjöar  4,3 %   största flod  4 152
beta = 3,0   sjöar  1,2 %   största flod  7 907          inga sjöar
```

`beta` är den enda knapp som spelar roll för landskapets karaktär, och den byter
floder mot sjöar. Under 2,0 blir landskapet så finkornigt att varje sänka är sin
egen ändstation och nätverket aldrig samlar sig.

**Strävheten** är en flack spektralkomponent med många moder och liten amplitud.
Utan den finns inga sänkor att fylla, och därmed inga sjöar. Andelen `r` styr
hur många.

**Polarsänkan** ger havet. Latituden är redan `|lat| = 1` vid projektionens
över- och underkant och `0` i mitten, så en smoothstep i `|lat|` sänker
terrängen till under havsnivån i ett bälte som omsluter båda polerna — vilka på
torusen är ett och samma bälte. Havet blir därmed en enda sammanhängande sänka
som all dränering mynnar i, och den ligger där klimatet ändå är obeboeligt.

Uppmätt vid 64x256 med kontinent `beta = 2,6`, strävhet `beta = 1,2`, andel
0,20, kustlatitud 0,93 och bredd 0,06:

```
seed 1   hav 20,2 %   sjö  3,6 %   största flod 778   kust vid rad 25
seed 2   hav 17,8 %   sjö 10,8 %   största flod 653   kust vid rad 25
seed 3   hav 14,2 %   sjö 51,7 %   största flod 722   kust vid rad 20
```

Inget innanhav uppstår i mittzonen i något av fallen — havet stannar där
polarsänkan lade det.

**Men seed 3 är ett verkligt utfall och inte en bugg.** Kontinenten fick där en
enda stor avloppslös bassäng, och prioritetsfloden fyller den till ett innanhav
som täcker halva landet. Det är vad en endorheisk bassäng ska bli. Frågan är om
vi vill ha den variansen mellan frön eller om terrängen ska villkoras — till
exempel genom att sänka strävheten, kräva att största sjön understiger en andel,
eller ge kontinenten en svag lutning mot polerna. Det avgörs i 7002 med
mätningen framför sig, inte här.

---

## Efterskrift: landformerna som styrt brus

*Augusti 2026, efter 7016. Avsnittet ovan beskriver terrängen som den byggdes.
Det här beskriver hur den ska styras.*

Terrängmodulen skiljer sedan de placerade formerna infördes mellan **tektonik
som ger struktur** och **erosion som ger detalj**. Delningen är riktig, men den
har en gräns: en form som placeras är kodad uppifrån, och manifestet vill att
strukturer uppstår underifrån. En sjö man ritar in ligger ovanpå landskapet i
stället för i det.

### Sjön styrs genom sin tröskel

En sjös yta bestäms inte av hur djup gropen är utan av **var den spiller över**.
Höjs tröskelcellen stiger vattenytan och sjön breder ut sig längs den
omgivande terrängens egna nivåkurvor. Att gräva ur en grop är ett ingrepp i
hundratals celler; att höja en tröskel är ett ingrepp i några få, och
strandlinjen blir en nivåkurva i det brus som redan fanns.

Mekanismen blir därför **hitta och förstärk**, inte placera:

1. generera bruset
2. kör dräneringen, som ändå körs
3. läs ut sänkorna med deras yta vid brädden
4. välj den som bäst matchar önskemålet
5. höj dess tröskel mjukt tills ytan stämmer

Vad bruset gör av sig självt, uppmätt i `f5-terrang` med havet som bassäng:

```
elva sänkor
i celler:  415  239  111   33   9   8   5   5   4   4   1
i hektar: 4,15 2,39 1,11 0,33
```

**Önskemålet uttrycks i hektar**, inte i celler. Hektar är fysiskt och skalfritt
— samma tal betyder samma sak i varje världsstorlek — medan ett antal celler
bara betyder något om man vet upplösningen.

```yaml
former:
  - typ: sjo
    yta: 4.0        # hektar
    tolerans: 0.5
```

Två saker hör till specifikationen och ska inte upptäckas senare.

**Generatorn ska kunna säga nej.** Ber man om femtio hektar i en värld vars
största sänka rymmer fyra är svaret inte att gräva en krater utan att rapportera
att önskemålet inte går att uppfylla i det bandet. En sjö som är stor i
förhållande till landskapets våglängder är inte en sjö utan ett hav, och det är
en annan form.

**Fröval är en legitim del av mekanismen.** Att generera ett tiotal frön och
välja det vars sänkfördelning bäst matchar önskelistan är billigare än att
förvränga ett dåligt frö. `describe()` förutsåg det redan med "gallring av
frön".

Ett finare alternativ finns: **villkorad simulering**, där fältet genereras
betingat på givna höjder i givna punkter så att spektrumet blir rätt överallt
och villkoren träffas exakt. Den bör inte införas förrän något kräver exakta
höjder i exakta punkter — en sjö gör inte det, eftersom en sjö är en egenskap
hos dräneringen och inte en höjd.

### Havet är undantaget, och det är principiellt

Brus ger inte en stor sammanhängande sänka på beställning. Ett hav är inte
heller en landform utan **världens basnivå**, och `docs/regionen-och-omlandet.md`
säger att basnivån egentligen hör hemma utanför regionen, som randvillkor från
omlandet. De två sakerna är samma sak sedd från två håll: *det bruset kan bära
är landformer, och det som inte är en landform ska inte ligga i bruset.*

Havet förblir därför placerat.

### Kusten genereras, men måste kalibreras

Formerna **adderas** till bruset i stället för att maskera det, så kustlinjen är
där summan korsar noll och alltså en brusstörd kurva — inte en cirkel. Det är
riktigt i princip. I praktiken syns det inte:

```
radie  djup  brus_sd   hav%   korrugering   kantlutning
   40   4,0     0,17   28,2      1,09         0,150
   40   1,0     0,17    1,5      1,31         0,037
   40   4,0     0,60   31,0      1,72         0,150
   55   1,5     0,60   23,8      2,75         0,041
   55   1,0     0,60   15,1      2,90         0,027
```

Korrugeringen är kustlinjens längd delad med omkretsen hos en cirkel med samma
havsarea. Vid dagens parametrar är den **1,09**, alltså praktiskt taget en
cirkel, eftersom bassängkantens lutning är 0,150 mot brusets typiska 0,019.

Det ger en styrande kvot i stället för en form att rita: **kustens karaktär är
brusets lutning delad med bassängkantens.** Under ungefär en femtedel blir
kusten en cirkel; över ett får man vikar, uddar och öar. Batymetrin bär samma
brus — uppmätt spridning 8,5 meter under havsytan — så berg som fortsätter ner i
vattnet följer utan egen mekanism.

För havspatchen betyder det att en **flack och bred bassäng ger bättre kust än
en djup och smal**, och att talet ska väljas mot korrugeringen och inte bara mot
havsandelen.

---

## Hydro-passets kontrakt

```
1. forcing        rain_band[band] * oro[cell] - ET(T, soil_water)
2. mark           soil_water punktvis; överskott blir avrinning
3. routing        avrinning längs flow_order; sjö fångar sitt inflöde i
                  magasinet och spiller överskottet vidare från utloppscellen
4. sjönivåer      ur hypsometrin
5. härledda fält  kanaldjup, sjödjup, havsmask, submerged, flow_dir
6. passiv drift   flytande organismer med strömmen — före sensing
```

Massbevarandet är exakt av strukturella skäl, inte approximativt: routing är en
ren överföring, precis som `transport_pass` redan resonerar om laplacianen.
Mätpunkten "drift < 1e-6 relativt" faller ut som avrundningsbrus.

Nederbörden är säsongsbunden och latitudberoende. `rain_band` är en profil per
band per tick och kostar därför ingenting extra — samma klass som `T_band` och
`g_band`. Den orografiska modifieraren `oro[cell]` är statisk och multiplikativ:
lovart blir vått, lä torrt, utan att regnskuggan kodas som en regel.

---

## Konsumenterna

`TODO.md` har ett återkommande fynd som här tas som hårt villkor: **produktion
utan konsumtion växer bara tomrummet.** `nutrient` låg allokerat och orört,
`transport_pass` returnerade noll, `flow_strength` nollställdes varje tick. Steg
7 får inte upprepa det. Varje fält nedan har en läsare i samma patchserie.

**Klimatet blir tvådimensionellt.** `T_cell = T_band[band] - lapse * elevation`.
Den billigaste konsument som finns — en statisk `T_offset[n]` och en addition
vid läsning — och exakt den form kadensdokumentet förutsåg: *"profil plus
eventuella per-cell-modifierare"*. Höglandet blir kallt, och terrängen är
ekologiskt verksam redan innan hydro är kalibrerad.

**Floran läser markvattnet.** Tillväxtpasset har redan formen
`min(room, dm_nutrient, light_growth)`. Vatten blir en tredje post i samma
`min()`, precis som ljuset lades till utan att passet skrevs om. Mät
`flora_water_limited` bredvid `flora_light_limited`.

**Näringen följer terrängen.** Kommentaren vid `nutrient_input` förutsäger det
redan: *"Konstant tills terrängen finns; då blir den vittring som funktion av
höjd, och utsköljning under havsnivå."* Vittring efter lutning, urlakning
nedströms längs `flow_to`.

Men **medelbevarande**. Identiteten
`mineralisering_jämvikt = nutrient_input * n_cells / nutrient_loss_frac` är
kalibrerad och verifierad mot 3 391 kg näring i systemet mot förutsagda 3 395.
Vittringsvikten normeras därför till medelvärde ett, så att bara den rumsliga
strukturen tillkommer och den globala jämvikten står still. Bördiga dalar och
magra ryggar utan att bördigheten som helhet ändras.

**Dränkning.** `submerged` grindar etablering och driver mortalitet. Strandzoner
uppstår utan att zonerna kodas.

**`flood_tolerance` och `buoyancy` får läsare.** Rörelsekostnad och skada ur
`water_depth`, passiv drift ur `flow_dir`.

### Och den egentliga vinsten

I dag är latituden världens **enda** miljöaxel som varierar i rummet, och den är
endimensionell. Korskorrelationen i födelsetakt mellan latitudband ligger på
0,45–0,67 mot ett mål under 0,3, och den effektiva populationsstorleken är mätt
till fem.

Vätheindexet är den första miljöaxel som varierar **finkornigt i två
dimensioner**. Uppmätt på land vid 64x256 spänner `ln(uppströmsarea / lutning)`
från 3,9 till 9,5 mellan femte och nittiofemte percentilen.

Det är en klassisk nischaxel, och den uppfyller kravet om motverkande
konsekvenser **genom miljön i stället för genom en påhittad kostnad**. En
torkanpassad planta på ryggen och en översvämningsanpassad i dalen straffas var
och en på den andras plats. Det är den renaste trait-form modellen kan få, och
det är hela skälet att bygga hydro nu snarare än senare.

---

## Låsta beslut

1. **Jämviktslösning, inte transient.** Hydro löser stationärt tillstånd per
   tick. Det explicita schemat behålls som orakel, inte som produktionsväg.
2. **64x256 först.** Geologin och hydrologin valideras mot den befintliga
   baslinjen där. 512x512 väntar på florans GPU-väg, som är det som faktiskt
   begränsar världsstorleken — hydro kostar 0,68 ms vid 262 144 celler mot
   florans dryga två sekunder vid samma celltal och nuvarande täthet.
3. **Havet omsluter polerna.** En enda sammanhängande sänka i det bälte där
   klimatet ändå är obeboeligt. All dränering mynnar där.
4. **Nederbörden är säsongsbunden och latitudberoende**, med statisk orografisk
   modifierare per cell.
5. **Höjdgradienten på temperaturen införs direkt**, inte efter att vattnet
   validerats.

---

## Vad som inte görs i Steg 7

- **Erosion och geomorfologi.** Terrängen är statisk. Att låta vattnet forma
  höjden gör `flow_order` dynamisk och river hela kostnadsargumentet.
- **Glesning av hydro.** Routingen kostar mindre än en millisekund vid en kvarts
  miljon celler. Glesning är sannolikt onödig och görs bara om en mätning
  motiverar den.
- **Transienta översvämningar.** Följer av beslut 1. Skulle vi någon gång vilja
  ha flodvågor som propagerar på tickskalan är det den punkt där designen måste
  öppnas igen.
- **Vattnet som bevarad storhet i ledgern på samma nivå som näringen.**
  Näringsbalansen är den hårda invarianten; vattenbalansen mäts men behöver inte
  samma tolerans, eftersom havet är en absorberande rand.

---

## Vad som ska mätas

| mätpunkt | målvärde |
|---|---|
| `flow_order` täcker alla celler, inga cykler | hård invariant |
| varje landcell når hav eller sjö | hård invariant |
| vattenbalans över 10 000 tick | drift < 1e-6 relativt |
| näringsjämvikt före och efter vittringens rumsliga fördelning | oförändrad ±2 % |
| hydro-passets kostnad vid 16 384 celler | < 0,1 ms/tick |
| hydro-passets kostnad vid 262 144 celler | < 2 ms/tick |
| andel land som är sjö, över tio frön | median under 10 % |
| `flora_water_limited` vid jämvikt | jämförbar med `flora_light_limited` |
| korskorrelation i födelsetakt mellan latitudband | under dagens 0,45–0,67 |
| kapacitetsfält utan läsare efter 7009 | `flood_tolerance` och `buoyancy` borta ur listan |
