# Designskiss — växternas livscykel

*Juli 2026. Underlag för Steg 4b i TODO.md. Status: förslag, inte beslut — uppdateras eller markeras som ersatt när steget byggs.*

---

## Frågan som ledde hit

Efter kalibreringen i Steg 4 ställdes tre enkla frågor till modellen: hur selekteras floran, varför blir en växande del av näringen fri, och har en planta någon livslängd?

Svaren visade sig kräva en mätning i stället för ett resonemang, och mätningen visade att flera av de mekanismer vi trodde var på plats inte är verksamma. Det här dokumentet samlar vad som mättes och föreslår vad växtligheten ska bli i stället.

Det ersätter inte `docs/substratets-struktur.md` utan bygger vidare på den. Ljusfrågan hör dit och upprepas inte här.

---

## Vad mätningen visade

*Körning: 3 000 respektive 6 000 tick, seed 1, 64×256, standardparametrar.*

### 1. Floran är fullständigt näringsbegränsad — men på fel ställe

```
celler totalt                16 384
bebodda celler                2 759
fri näring totalt             161,7 kg
  varav i bebodda celler        0,12 kg
  varav i tomma celler         161,6 kg
median näring i bebodd cell     3e-9 kg
```

De bebodda cellerna är rentvättade till maskinprecision varje tick. Att den fria näringen växer betyder inte att floran är mätt — det betyder att näringen frigörs där ingen står och att `transport_pass` inte hinner emellan.

### 2. Plantorna når inte sin vuxenmassa

Andel flora vid vuxenmassa: **1,4 %**. Medianplantan står på **26 %** av sin vuxenmassa. De slutar inte växa, de svälter.

### 3. `uptake_capacity` binder aldrig

Andel individer där kapacitetstaket är den begränsande termen: **0,0**. Median efterfrågan 0,22 kg mot ett tak på 3,6e4 kg — fem tiopotenser. Fördelningen inne i cellen sker proportionellt mot *tillväxtunderskottet*, alltså `m · (1 − m/M_vuxen) · gate`, och upptagsförmågan faller ur helt. Medelvärdet ligger kvar på sitt initieringsmitt, vilket är hur neutral drift ser ut.

En bieffekt av samma delningsregel: eftersom andelen följer `(1 − m/M_vuxen)` får en planta större del av poolen bara genom att deklarera en högre vuxenmassa. Ett mål den aldrig når ger ändå inkomst.

Det korrigerar en formulering i `docs/substratets-struktur.md`, som säger att näringskonkurrensen är symmetrisk och begränsas proportionellt av `uptake_capacity`. Den är symmetrisk, men på fel storhet.

### 4. Flera individer per cell inträffar inte

```
tick   flora   bebodda celler   1 planta   2   3
 500    4505           4487        4470   16   1
3000    2794           2759        2727   29   3
```

98,8 procent av cellerna har exakt en planta. Regeln togs bort i patch 0032; beteendet blev kvar. Skälet är att den första plantan tar hela cellens flöde oavsett sin storlek, så ett frö som landar där får en andel av noll.

Därmed föll också motivet i 0032: flera per cell infördes för att `uptake_capacity` skulle kunna selekteras, men eftersom samexistens aldrig inträffar tar ingen näring från någon.

### 5. Näringsekonomin är icke förnybar

```
tick      fri    flora  detritus   fauna   = summa   förlorat   tillfört
   0    0,000   535,77     0,00    0,32     536,09      0,00    536,088
1000   32,66    451,38    47,60    0,73     532,37      3,72    536,093
3000  161,75    254,45    95,48    0,54     512,21     23,90    536,103
6000  207,13    265,89    22,50    0,00     495,52     40,60    536,118
```

`nutrient_init = 0`. Världen börjar steril, och hela stocken på 536 kg kom in som vävnad i den sådda floran. `flora_init_mass_ratio` är alltså inte ett såddreglage utan **världens bördighetsreglage** — när kalibreringen i 0052 höjde det till 2000 var det egentligen världens näringsmängd som ändrades.

`nutrient_input = 1.5e-8` ger 0,029 kg på 6 000 tick, fem tusendels procent av stocken. Vittringstermen finns som namn men inte som flöde.

`nutrient_loss_frac = 0.10` tar däremot tio procent av allt som passerar nedbrytningen. 40,6 kg borta på 120 månader, monotont. **Utan tillförsel töms världen på 37 år simulerad tid** — innanför en normal körning.

### 6. Livslängd saknas

`flora_mortality = 2e-5` per månad ger en medellivslängd på 4 167 år. Talet behölls vid omskalningen till månader som klass B, men klass B motiveras av att den naturliga tidsskalan är livslängden — och floran saknar den åldrandemekanik som gör faunans skadetakter livslängdsbundna. Florans mortalitet skulle ha härletts, inte behållits.

En planta dör alltså bara av att bli uppäten. Utan generationsomsättning verkar selektionen bara på spridningen, och de gamla individerna lämnar aldrig plats.

---

## Grundprincipen: inkomst skild från allokering

Alla sex fynden har samma rot. En planta i modellen har ingen inkomst den äger. Den tar upp exakt den näring den i samma tick kan omsätta till massa, och inte ett gram mer. Den kan inte spara, inte betala underhåll, inte avsätta till frön. Hela livscykeln är därför en enda handling — växa — plus en avgift som dras direkt ur vävnaden när en tröskel nås.

Den reviderade modellen delar det i två:

**Inkomst.** Vad plantan tjänar bestäms av dess rotarea, dess upptagsförmåga och den lokala tillgången. Inte av vad den råkar vilja bygga. Näringen läggs i en reserv per individ, symmetriskt med faunans energireserv.

**Allokering.** Reserven fördelas på tillväxt, omsättning och reproduktion enligt genetiska andelar. Det är här livscykeln bor.

---

## Rotarean som anspråk

Hexcellen har area exakt 1 — det sattes i Steg 2 för att täthetsstorheter skulle bevaras vid geometribytet. Enheten finns alltså redan.

Rotsystemet är en **area**, härledd ur plantans **aktuella** massa:

```
A = m / B_K            härledd varje tick, inte lagrad
```

`B_K` får därmed en fysisk innebörd den saknar i dag: massan hos en planta vars rotsystem täcker exakt en cell. Världens massaskala definieras av världens geometri i stället för av en kalibrering.

Att arean följer aktuell massa och inte vuxenmassan är avgörande. Ett frö på 0,1 kg gör anspråk på en hundradel av en cell. Ett träd gror alltså in i en cell som redan är anspråkad av örter och får till en början nästan ingenting. När örterna dör sjunker den totala anspråkade arean, trädets andel stiger, det växer, och dess area växer med det. Ingen arealtilldelning behöver bokföras — **arean är massan**.

Detsamma gäller åt andra hållet: en betad planta tappar massa och därmed mark. Betning får en konkurrenskonsekvens utöver förlusten.

Fotavtrycket ersätter den tidigare tanken på diskreta hexringar. Ringarna är bara vad som händer när arean överstiger 1, och de flesta plantor ligger under.

---

## Upptagsregeln

```
andel_i  =  A_i / max(1, Σ A_j)          summerat över cellen
inkomst_i = min( uptake_capacity_i · u_max · A_i ,  andel_i · tillgänglig näring )
```

Tre saker faller ut av `max(1, ΣA)`:

**Outnyttjad näring blir verklig.** En planta med area 0,1 tar en tiondel av flödet; resten ligger kvar i marken. Obesatt mark är därmed en resurs, och kolonisation lönar sig. Det är motsatsen till dagens beteende, där en ensam planta suger cellen till noll oavsett storlek.

**Trängsel behöver ingen egen regel.** Överstiger anspråken 1 späds alla lika. Under 1 finns luft. Samma formel bär båda regimerna.

**Samexistens blir normalfallet.** En groddplanta får sin area i stället för en andel av noll.

En fjärde effekt gäller faunan: en betare som tar 1,8 kg ur en cell där fyra örter står tar en del av beståndet i stället för en hel individ i taget.

**Prestanda.** Eftersom de allra flesta plantor har area under 1 läser upptaget en enda cell för dem — samma vektoriserade gather som i dag. Bara de största spiller över i grannmatrisen. Arearegeln är billigare än ringmodellen, inte dyrare.

**Tillväxten begränsas av ett `min()` över resurser**, med näring som första och tills vidare enda post. Det är Liebigs minimumlag, och formen finns för att ljus ska kunna läggas till som andra post utan att tillväxtpasset skrivs om. Se avsnittet *Vad skissen inte gör*.

---

## Livscykeln

### Etablering

Fröets näringsförråd avgör om det etablerar sig. Formen är inte fri: fitness går som antal gånger etableringssannolikhet, alltså `f(m)/m`, och den har inget inre optimum om `f` är en vanlig mättande funktion — då vinner alltid det minsta fröet. Etableringen måste vara sigmoid med tröskel:

```
f(m) = m² / (m² + h²)        optimum hamnar exakt på m = h
```

Det är Smith och Fretwells klassiska villkor för fröstorlek mot fröantal.

**Referens:** Smith, C. C. & Fretwell, S. D. 1974. *The optimal balance between size and number of offspring.* The American Naturalist 108: 499–506.

Ska axeln differentiera i stället för att konvergera måste `h` bero på målcellens tillstånd — stort frö vinner där det är trångt, många små där det är öppet. Med arearegeln finns den storheten redan: `ΣA` i målcellen.

**Etableringsutfallet dras innan sloten allokeras.** Bara frön som etablerar sig blir individer; resten blir detritus i målcellen. Det är biologiskt rätt — fröregn föder marken — och ett hårt tak på allokeringstakten.

### Tillväxt

Som i dag, men betald ur reserven och begränsad av `min()` över resurser i stället för av en logistisk takt som ändå aldrig binder. Den nuvarande `flora_growth_rate` ger `regen · dt` mellan 260 och 2 600, alltså en term som saturerar mot vuxenmassetaket varje tick. Locuset har i praktiken ingen konsekvens och bör antingen få en bindande roll eller avvecklas.

### Omsättning

Det saknade flödet. En levande planta fäller förna och bygger om. Det ger tre saker på en gång:

- underhållskostnaden, alltså A2 för den enkla halvan av populationen
- näring som cirkulerar utan att någon behöver dö
- `structure` som verklig långsam–snabb-axel: tunna billiga blad ger hög omsättning och hög tillväxt, seg vävnad tvärtom

Omsättningstakten avtar med strukturandelen. Det är bladekonomispektrumet, och det är samma tal som redan styr nedbrytningstakt och betningsutbyte.

### Reproduktion

En genetisk allokeringsandel av inkomsten. **Semelpari faller ut som ytterligheten** i stället för att vara ett specialfall: den som lägger allt på frön tömmer sin reserv under omsättningsnivån och dör. Skillnaden mellan ettårig och flerårig blir därmed uttryckbar utan att någon artgräns kodas.

Den nuvarande grinden — spridningsklar vid 70 procent av *egen* vuxenmassa, fröet 10 procent av *egen* vuxenmassa — avvecklas. Den är skalfri, och därför kan storleksaxeln bara falla: en mindre planta når sin egen tröskel snabbare och fröet skalas ner med henne, så en stor planta har ingen fördel alls.

### Död

Livslängden **härleds** ur strukturandelen, inte ur ett eget locus. Ett fritt livslängdslocus skulle låta selektionen välja långt liv utan att betala i långsam tillväxt, och då försvinner avvägningen. Struktur är livslängdstraiten på samma sätt som den är tillväxttraiten.

Spannet bör täcka örter till björkar, ungefär sex månader till tjugo år. Ekänden ryms inte i ett simuleringsfönster på 120 månader och bör sägas rakt ut i stället för att finnas som en parameter som aldrig får verka.

Utöver det: **svältdöd** när reserven inte räcker till omsättningen. Det är den mekanism som gör att groddplantor i en full cell försvinner utan att en särskild groddplantsrisk behöver postuleras.

---

## Fröet och spridningen

**Propagulmassan blir en egen absolut axel**, inte en andel av moderns vuxenmassa. Verkliga kvoter frö mot vuxen spänner sex storleksordningar och nästan alltid åt andra hållet än vår: ett träd på 44 kg producerar i dag ett frö på 4,4 kg.

**Antalet frön** följer av avsatt massa dividerat med propagulmassan. Då blir "många små eller få stora" ett val i stället för en konstant.

**Spridningsapparaten** tar över det frigjorda `_T_DISPERSAL`-locuset. Locuset kodar i dag spridnings*takten*, men takten faller ut ur allokeringen i den nya modellen, så genomet behöver inte växa. Andelen `a` av propagulen ligger i vinge eller pappus i stället för i förrådet: den ökar avståndet och minskar etableringen.

**Avståndet** dras ur en tungsvansad kärna i kontinuerligt rum — vinkel likformig, avstånd ur kärnan, addera till moderns position, wrappa, slå upp cellen med `grid.cell_of()`. Det är O(1), vektoriserbart över alla frön i en tick och geometriagnostiskt, i stället för dagens BFS över en skiva med två diskreta radier.

Skalan, med frigöringshöjd ur vuxenmassan och en apparat vars massa växer brantare än sin area:

```
L  ∝  M_vuxen^(1/3) · a^(1/3) / m_frö^(1/6)
```

**En ärlighetsanmärkning.** Den sista termen är svag, och den är svag i verkligheten också — sjunkhastigheten varierar bara omkring tiofalt över hela fröstorleksspektrat. Att stora frön faller nära beror inte främst på aerodynamik utan på att de inte är vindspridda alls. Det är ett syndrom som selektionen byggt, inte en fysisk nödvändighet. Vi kodar därför inte in korrelationen: två oberoende axlar med ärlig kostnadsstruktur, och syndromet får uppstå om kostnaderna är rätt. Gör det inte det är kostnadsstrukturen fel, vilket i sig är ett resultat.

**Spridningens andra skäl.** Med en rotarea konkurrerar ett frö som landar nära modern med henne. Syskonkonkurrens är den klassiska anledningen till att spridning evolverar alls, och den finns inte i modellen i dag eftersom upptaget är cellprivat.

---

## Näringsekonomin

Tre reglage som i dag sitter på samma ratt ska skiljas åt.

**Bördighet.** `nutrient_init` sätter hur mycket näring världen innehåller, oberoende av hur mycket flora vi sår. Sådden ska vara en startpopulation och **betalas ur marken** — i dag myntar den näring, vilket ledgern visar som `added = 536,088` vid tick noll.

**Omsättning.** Tillförsel och förlust kalibreras som ett par mot den stock vi vill hålla. De är inte oberoende: i jämvikt är förlusten `λ` gånger hela primärproduktionen.

**Nivå.** Stocken sätts efter hur tätt bevuxen världen ska vara.

### Härledning

Med `M₁` som den vuxenmassa vars rotsystem är exakt en cell, `c` som näringsinnehåll per kilo vävnad och `T` som medeluppehållstid för levande vävnad:

```
levande näring   =  n_cells · M₁ · c
detritus         =  levande · τ_det / T
förlust per tid  =  λ · levande / T
```

`τ_det` är detritusets näringsviktade uppehållstid: det labila materialet bär 92,7 procent av näringen och ligger 13,0 månader, det strukturella bär 7,3 procent och ligger 86,6 — sammanvägt **18,3 månader**. Vid två års levandeomsättning håller förnan alltså tre fjärdedelar så mycket näring som växterna.

### Valet

```
  M₁    biomassa   levande  detritus   fri   TOTAL  init/cell |  plantor per cell        | individer
 kg                  näring   näring        näring            | minsta  median  största  | vid full täckning
 2,8      45 056       701      535     70    1 306   0,080   |   1,0     0,1     0,06   |   1 000 –  16 000
11,0     180 224     2 806    2 139    281    5 225   0,319   |   4,0     0,5     0,25   |   4 100 –  66 000
23,0     376 832     5 866    4 473    587   10 926   0,667   |   8,4     1,0     0,52   |   8 600 – 137 000
44,0     720 896    11 222    8 557  1 122   20 902   1,276   |  16,0     1,9     1,00   |  16 000 – 262 000
```

Nuläget är ~16 300 kg biomassa och 536 kg näring, så raderna är 3×, 11×, 23× och 44× dagens växtlighet.

**Beslut: `M₁ = B_K = 11`.** Inte för att elva är ett bra tal, utan för att det ger `B_K` sin fysiska innebörd. Örterna blir fyra per cell, medianplantan spänner två celler, de största fyra. Individantalet stannar under 66 000, vilket mot Steg 5:s uppmätta 0,44 µs per individ är omkring 30 ms per tick i värsta fall. `M₁ = 23` är också försvarbart men flyttar Steg 8 framåt i kön.

### Talen

```
nutrient_init          0.0      ->  0.32      kg fri näring per cell
nutrient_input         1.5e-8   ->  7.1e-5    per cell och månad
nutrient_loss_frac     0.10     ->  0.01
flora_init_mass_ratio  2000     ->  starttäthet, betald ur marken
uptake_rate_max        per individ  ->  per areaenhet, storleksordning 0,03 kg/mån
```

Tio procents förlust per varv är för mycket. Ostörda landekosystem är snåla återvinnare: förlusterna är någon procent av det interna flödet, och vittring och nedfall är små i motsvarande grad. Med `λ = 0,01` blir tömningstiden omkring 370 år och tillförseln en långsam forcing i stället för ett dropp som håller världen vid liv.

`uptake_rate_max` ligger i dag sju till åtta tiopotenser för högt, vilket är varför taket aldrig band. Med arearegeln får det dessutom rätt form: ett upptag per areaenhet, inte per individ. Storleksordningen följer av att en medianplanta ska hinna binda sina 0,36 kg näring på ungefär ett år; det exakta talet kan sättas först när mognadstiden är bestämd.

---

## Vad som blir tvåsidigt

Kravet från `docs/substratets-struktur.md` gäller: en trait med en enda konsekvens är ett reglage, inte en anpassning.

| Trait | Drar uppåt | Drar nedåt |
|---|---|---|
| `uptake_capacity` | mer inkomst per area | underhållskostnad (Steg 6) |
| `structure` | lång livslängd, låg omsättning, betningsmotstånd, billig i näring | låg tillväxt, låg energitäthet, dyr i kol när ljus finns |
| vuxenmassa | fler frön, större rotarea, betningsrefug | längre tid till mognad, fler konkurrenter |
| propagulmassa | högre etablering | färre frön |
| apparatandel `a` | längre spridning, undflyr syskonkonkurrens | lägre etablering |
| allokering till frön | fler avkommor nu | tömd reserv, svältdöd |

Två av dem har ännu ingen motkraft i koden: `uptake_capacity` får sin först i Steg 6, och `structure` får sin starkaste när ljuset kommer. Det ska stå i klartext hellre än att upptäckas som drift.

---

## Vad skissen inte gör

**Ljus som andra begränsande resurs.** Rotkonkurrens är storleksymmetrisk, ungefär vår arearegel; ljuskonkurrens är starkt asymmetrisk. Utan den asymmetrin växer alla i en cell med samma relativa takt — inkomsten är proportionell mot arean, arean mot massan, alltså `dm/dt ∝ m` för var och en, och kvoten mellan trädets och örtens massa ändras aldrig. Trädet växer inte långsamt i konkurrensen; det förblir en konstant bråkdel.

Värre: den enda term som bryter symmetrin drar åt fel håll. Byggkostnaden i näring sjunker med strukturandelen, från 0,0224 vid `s = 0,35` till 0,0036 vid `s = 0,95`. Under ren näringsbegränsning bygger den vedartade plantan alltså sex gånger fler kilo per kilo näring och växer sex gånger snabbare i relativa termer än den örtartade — precis omvänt mot bladekonomispektrumet. Skälet är att floran inte har någon bindande kolbudget; seg vävnad är billig i näring och dyr i kol, men kolet är gratis, så bara rabatten syns.

Ljus löser båda med samma grepp: det gör struktur dyrt och storlek lönsamt.

> **Falsifierat, 0061–0066.** Ljuset gav strukturandelen en *ny uppsida* i
> stället: höjden är strukturmassa, och ju knappare ljuset är desto mer är höjd
> värd. Axeln bromsades från 0,78 till 0,69 men vändes inte. Se
> `docs/statusanalys-vaxtcykeln.md`, Del B. Formen, exponenten och litteraturen finns redan i `docs/substratets-struktur.md`, liksom skälet att vänta — två begränsande resurser samtidigt gör kalibreringen oattribuerbar. **Storleksaxeln får därför ingen rättvis dom förrän ljuset finns, och det ska inte tolkas som att den är kalibrerad.**

**Humus.** Modellen har tre näringspooler — fri, förna, levande — men saknar den fjärde och i verkligheten största: en långsam markpool som binder mycket och släpper lite. Utan den svänger näringen lika snabbt som förnan bryts ner.

**Kopplingen till faunan.** Perception och betning läser cellaggregat medan floran är individer. Det är rimligt när en cell rymmer flera plantor och fel när en cell rymmer en fjärdedels träd. Frågan hör till betningen, inte till livscykeln.

---

## Mätpunkter

| Egenskap | Målvärde |
|---|---|
| Kapacitetsfält utan läsare bland floras | 0 — `uptake_capacity` binder mätbart |
| Andel bebodda celler med fler än en planta | betydligt över noll; i dag 1,2 % |
| Näring i bebodda celler | inte längre nollställd; skalar med obesatt area |
| Floras stationära antal vid dubblad `capacity` | oförändrat ±10 % |
| Näringsstockens drift över 20 000 tick | ingen monoton trend |
| Överlevnadsskillnad mellan hög och låg `uptake_capacity` | statistiskt skild |
| Spridning i propagulmassa | bimodal eller bred, inte kollapsad mot en ände |
| Spridning i vuxenmassa | **mäts men döms inte** förrän ljuset finns |
| Andel plantor som når vuxenmassa | väsentligt över dagens 1,4 % |
| Kostnad per floraindivid och tick | < 1 µs, oförändrat från Steg 5 |

---

*Skissen är ett förslag. Talen är härledda och står i dokumentet med sin härledning; besluten — `M₁ = B_K`, area ur aktuell massa, härledd livslängd, ljus uppskjutet — är låsta för implementationen och ska inte behandlas som öppna frågor under den.*
