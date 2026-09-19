# Designskiss — sammansättning och vatten

*September 2026. Underlag för serien efter revisionen av faunans balans
(`docs/revision-faunans-balans.md`, fynd F1 och M3). Status: genomgången
2026-09-19; besluten står under "Beslut vid genomgången" och ersätter de
öppna frågorna. Uppdateras eller markeras som ersatt när serien byggs.*

*Reviderar enaxelbeslutet i `docs/substratets-struktur.md` och utvidgar de
tre valutorna i `docs/metabolismen.md` med en fjärde.*

---

## Frågan som ledde hit

Revisionen fann att en betare i modellen behöver äta omkring fem gånger så
mycket som en verklig betare av samma storlek. Två av orsakerna ligger i hur
materia beskrivs, inte i hur den räknas:

- **Torr föda värderas som våt vävnad.** Floran räknas i torrsubstans, men
  energin per kilo, `E_labile = 9,3 MJ/kg`, är tillbakaräknad från våt
  kroppsvävnad. Strukturmaterialet ger noll. Ett kilo torrt bete ger 3,2 MJ
  mot verklighetens ~10 smältbara för en idisslare.
- **Fett har våt vävnads energitäthet.** `M_slow` håller 9,3 MJ/kg mot fettets
  37–39, och bär därför 3–4 gånger för mycket massa per lagrad joule.

Beslut som ligger fast inför skissen:

1. **Verklighetsnära fysik och kemi** är målet, inte en kalibrering som råkar
   ge rätt utfall.
2. **Djur och växter har våt massa, men bokföringen räknar på torrsubstansen.**
   Kemin — energi, näring, kol — ska inte bero på hur törstig eller vissen en
   organism råkar vara.
3. **Djuren ska dricka och upprätthålla en vätskebalans.** Modellen har vatten;
   det ska få en funktion för djuren.

---

## Principen: torrsubstans och vatten som två tillstånd

Varje organism bär två storheter:

```
torrsubstans   uppdelad i komponenter; bär all kemi (energi, näring, kol)
vatten         en egen, bevarad storhet
våt massa      = torrsubstans + vatten   (härledd, lagras inte)
```

Den våta massan är det fysiken verkar på: Kleiber, rörelsekostnad, värmeledning
och värmekapacitet, magfyllnad, tuggstorlek, flytkraft och kadaver. Den lagras
inte — den har då ingen egen skrivare, och "en ägare per fält" står kvar.

All bokföring sker på torrsubstansen. Näringsinvarianten behåller sin betydelse;
vattnet får en egen invariant (se *Vattnets bokföring*).

"Torr eller våt" är alltså inget val. Det är två delar av samma kropp, och det
som räknas är sammansättningen.

---

## Sammansättningen

### Växter

Torrsubstansen delas i tre, i stället för dagens två (`1 − s` labilt, `s`
strukturellt):

```
labilt          socker, stärkelse, protein, lipider — smälts direkt
jäsbar fiber    cellulosa, hemicellulosa — bara genom mikrobiell jäsning
lignin          smälts inte av någon
```

Strukturandelen `s` står kvar som summan av fiber och lignin. Det nya är
**ligninets andel av strukturen**, som skiljer ett segt gräs (lignin ~5 % av
torrsubstansen) från ved (~25 %). Den härleds ur `s`: graden av vedartad
struktur är lignininnehållet, så andelen växer med `s`.

### Djur

Kroppens torrsubstans delas i protein, fett och aska, plus kroppsvatten:

```
mager vävnad     protein + aska + vatten (hydratiseringen ~73 % av fettfri massa)
glykogen         M_fast: kolhydrat med ~3 kg vatten per kg
fettväv          M_slow: ~85 % lipid, lite vatten och protein
```

Dagens tre pooler (`M`, `M_fast`, `M_slow`) får en kemisk tolkning i stället
för en gemensam energitäthet.

**Aska** är en operationell storhet: det som återstår när organiskt material
förbränns fullständigt, alltså mineralämnena — kalcium, fosfor, kalium,
magnesium, svavel och spårämnen. Den är en av de fem posterna i
proximatanalys (vatten, aska, protein, fett, kolhydrat). Hos däggdjur är
askan mest skelettmineral: ben är ~60 % mineral, ~30 % kollagen — som räknas
som protein — och ~10 % vatten, och helkroppsaskan är 3–5 % av levande vikt,
alltså 12–20 % av den magra torrsubstansen. Skalbärare ligger mycket högre.

**Axeln `s` betyder samma sak i båda riken men har olika kemi.** Den är den
strukturella, icke omsättbara andelen av torrsubstansen: hos växter organisk
— cellulosa, hemicellulosa och lignin — och hos djur mineralisk. Växtaska är
bara 5–10 % av torrsubstansen och räknas inte särskilt.

---

## Energi och näring per komponent

Värden per kg torrsubstans. Bruttoenergi är förbränningsvärmet; metaboliserbar
är det kroppen kan använda.

```
komponent         brutto MJ/kg   metaboliserbar   näring (N) per kg   källa
kolhydrat           17,2            ~17              0                   Atwater
protein             23,6            ~17 (urea)       0,16                Rubner, Atwater
lipid               39,3            ~37              0                   Atwater
cellulosa           17,5            via jäsning      0                   —
lignin              ~26             0                0                   —
```

Två följder som rör nuvarande kod:

- **Fett bär ingen näring.** I dag bokförs reserven med samma näring per kg
  som labil vävnad. Med rätt kemi lämnar kvävet kroppen när fett lagras
  (proteinet i födan bryts ned, kvävet går ut som urea), och det kommer inte
  tillbaka när fettet mobiliseras. Reservens sammansättning går in i
  näringsbokföringen.
- **Protein kostar vatten och näring när det bränns.** Kväveöverskottet måste
  utsöndras som urea, och det kräver urin. Här möts näringen och vattnet.

Modellens enda näring **definieras som kväve** i djurens kemi: protein bär
16 %, fett och kolhydrat inget. Kväve och fosfor delas inte upp. I världen
beter sig näringen som en blandning — vittringen som fosfor, urlakningen och
denitrifikationen som kväve, sedimentet som fosfor — och det är en känd
förenkling, inte ett påstående.

### Kvävet i djurkroppen

Reserven — glykogen och fett — är kvävefri. Bara den magra vävnaden, proteinet,
bär kväve. Födans kväve får därför en egen väg:

```
assimilerat kväve  →  kvävepoolen (fria aminosyror, litet tak)
kvävepoolen        →  tillväxt och foster, som proteinets kväve
överskott          →  urea, utsöndrat till cellen
```

**Tillväxt och fosterbygge kräver kväve ur poolen.** Energin kan komma ur
kolhydrater och fiber, men vävnad och foster kan inte byggas utan kväve, och när
poolen är tom stannar de — Liebigs lag för djuren. Det ger den avvägning
`metabolismen.md` saknar: föda som är proteinrik men energifattig, och tvärtom.
Betare är i verkligheten ofta kvävebegränsade.

Urean gödslar cellen där djuret står. Den kostar vatten, men det bokförs först
när vätskebalansen finns (steg 4); fram till dess är urinens vatten gratis,
som allt vatten i dag.

---

## Matsmältningen

**Labilt material** tas upp med en fast verkningsgrad, ~0,8–0,9.

**Jäsbar fiber** bryts ned av symbionter, och det tar tid. Den andel som hinner
jäsas följer av uppehållstiden i tarmen:

```
f_jäst = 1 − exp(−k_jäs · τ)          k_jäs ~0,05–0,1 per timme för cellulosa
τ      ∝ tarmvolym / intagstakt ∝ M¹ / M^0,75 = M^0,25
```

Det är Jarman–Bell-principen som mekanism och inte som konstant. Små djur
hinner inte jäsa fiber och gynnas av att välja labilt bete; stora djur kan
leva på segt gräs. Jäsningen förlorar en del av energin som metan och
jäsningsvärme, ~10–20 %.

**Lignin** passerar orört.

Förmagsjäsning (idisslare) och baktarmsjäsning skiljer sig i hur mycket av
mikrobproteinet djuret får tillbaka. Modellen använder en gemensam form; att
skilja dem åt behövs inte.

**Omvandlingen från torrt till vått sker här**, på ett ställe: assimilerad
torrsubstans blir vävnad, och vävnadens vatten tas ur kroppens vattenpool.

---

## Vattnet

### Växternas vatten

Färsk växtmassa är 70–85 % vatten. Vattnet i plantan blir ett eget tillstånd
per planta — så att plantor som växer där det finns fukt växer bättre — med
markvattnet som källa och transpirationen som sänka (transpirationen finns
redan, bunden till tillväxten). En planta på torr mark har lägre vattenhalt:
torkan får en väg in i betets värde, och den väg in i magfyllnaden som
djurens tuggor behöver.

### Djurens vätskebalans

```
in     dricka (från vattenceller)
       vatten i födan
       metaboliskt vatten: 1,07 kg per kg fett, 0,41 per kg protein,
                           0,56 per kg kolhydrat som oxideras
ut     avdunstning via andning och hud — växer med ämnesomsättning och värme
       urin — bär urean från proteinnedbrytningen
       vatten i spillningen
```

**Avdunstningen kyler.** Ångbildningsvärmet är ~2,4 MJ per kg vatten. I värme
blir kylningen en vattenkostnad, inte en energikostnad, och det kopplar
vätskebalansen till termoregleringen (revisionen M1).

**Uttorkning skadar och dödar**: en förlust på 10–20 % av kroppsvattnet är
dödlig för däggdjur.

**Törst blir ett anspråk** bredvid hunger. Djuren måste söka sig till vatten.
Sjöar och vattendrag finns redan, men i dag är de bara hinder och kust.

### Vattnets bokföring

Världen har en vattenbalans med drift ~1e-16 (nederbörd, avrinning, sjöar,
transpiration). Den utvidgas med vattnet i organismerna: det som växter tar
upp och djur dricker lämnar världens pool, det som avdunstar går till
atmosfären som redan är sänka för transpirationen, och urin och spillning
återförs till cellen. Metaboliskt vatten är en källa som följer av
oxidationen — torrsubstans blir vatten och koldioxid. Kolet flödar redan
igenom modellen; vattnet måste bokföras.

Principen i `docs/geologin-och-vattnet.md` gäller: **varje nytt fält har en
läsare i samma patchserie.** Kroppens vatten läses av vätskebalansen och
törsten, växternas vatten av magfyllnaden och betets värde.

---

## Vad det ersätter

- **`substratets-struktur.md`:** en axel `s` blir tre komponenter för växter
  och en kemisk sammansättning för djur. Skissens egen förutsägelse — att
  strukturmaterial kräver "lång tarm, långsam passage, symbionter" — blir
  mekanismen för jäsningen. Skissen markeras som delvis ersatt när serien
  byggs.
- **`metabolismen.md`:** tre valutor blir fyra — massa (diagnostisk),
  energi (per organism och tick, och nu per population), näring (global
  invariant) och vatten (global invariant). Referensen uppdateras i samma
  commits som koden.
- **`E_labile_J_per_kg`** som gemensam energitäthet för allt organiskt
  material försvinner. Energin följer komponenterna.
- **0202:s jämvikt** står kvar. Floran räknas fortfarande i torrsubstans, och
  kalibreringen mot p201 gäller.

---

## Ordning

En ändring per commit. Varje dynamikändring får en körning med uppmätt utfall,
och 0210:s energibokföring mäter var energin tar vägen.

1. **Djurkroppens sammansättning och kväve.** Poolerna får kemisk tolkning;
   fettet sin täthet. Reserven blir kvävefri, kvävepoolen tillkommer,
   tillväxt och foster kräver kväve ur den, och överskottet utsöndras som urea.
   Näringsbokföringen följer sammansättningen.
2. **Assimilationen.** Torrt blir vått på ett ställe; labilt, fiber och lignin
   var för sig; jäsningen som funktion av uppehållstiden.
3. **Växternas vatten.** Vattenhalten som tillstånd; magfyllnaden i färsk massa.
4. **Djurens vätskebalans.** Kroppsvattnet, de fyra vägarna in och ut,
   uttorkningsskadan, vattenbokföringen utvidgad.
5. **Törsten som anspråk.** Beteendet att söka vatten.
6. **Avdunstningskylningen.** Kopplingen till termoregleringen.

Termoregleringens rättelse (M1, den metaboliska värmen) beror inte på serien
och kan göras före.

---

## Beslut vid genomgången

*2026-09-19.*

1. **Ligninets andel härleds ur `s`.** Graden av vedartad struktur är
   lignininnehållet; ingen ny axel.
2. **Växternas vatten är ett eget tillstånd per planta**, så att plantor som
   växer där det finns fukt växer bättre.
3. **Kväve och fosfor delas inte upp nu.** Näringen definieras som kväve i
   djurens kemi. Uppdelningen tas upp när en fråga kräver den — ben, eller en
   nisch för kvävefixerare — och den kostar en dubblerad invariant, en ny
   kvävekälla och en omkalibrering av 0202.
4. **En gemensam form för jäsningen**; förmag och baktarm skiljs inte åt.
5. **Reserven blir kvävefri redan i steg 1**, med kvävepoolen, kvävekravet för
   tillväxt och foster, och urean till cellen. Urinens vattenkostnad kommer med
   vätskebalansen i steg 4. Energi och kväve följer båda av sammansättningen och
   hör till samma steg; vattnet är en annan storhet. Alternativet — att vänta
   med kvävet till steg 4 — lämnade fettet kemiskt fel i tre steg och gjorde
   steg 4 för stort för att utfallen skulle gå att skilja åt.

Kvar att följa: **faunans livskraft** mäts efter varje steg. Rätt fysik
behöver inte ge livskraftiga djur — dör de ut med rätt fysik är det en annan
mekanism som saknas.

## Steg 1 i detalj: djurkroppens sammansättning och kväve

*Specificerat och genomgånget 2026-09-20. Fem delpatchar, en ändring per commit.*

### Tillstånd

Kroppens tillstånd byter från våt massa till torrsubstans per komponent:
`M` blir torr mager vävnad (protein och aska), `M_fast` glykogen, `M_slow`
lipid, `gest_M` torr fostervävnad, och `N_pool` tillkommer som fria
aminosyror mätta i proteinekvivalent torrsubstans. Den våta massan härleds
med fasta vattenhalter tills vattnet blir ett tillstånd i steg 4:

```
M_våt = M/0,27 + M_fast·4 + M_slow/0,85 + N_pool + gest_M/0,20
```

mager vävnad 73 % vatten (Pace & Rathbun), glykogen ~3 kg vatten per kg,
fettväv 85 % lipid, foster ~80 % vatten. Den våta massan används av allt som
är fysik: Kleiber, `M_carry`, värmeledning, rörelse, flytkraft, betets `M^0,5`
och predationen.

### Kemin

Per kg torrsubstans, metaboliserbart: glykogen 17,2 MJ, lipid 37,7 MJ
(39,3 brutto), protein 17,2 MJ (23,6 brutto; skillnaden är urea och värme)
med 0,16 kg kväve per kg. Djurens `s` är **askandelen** i den magra
torrsubstansen; kvävet per kg blir `0,16·(1 − s)`. Växtkonstanterna
`nutrient_content(s)` gäller inte längre djurvävnad.

### Kvävets flöden

Assimilerat protein går till `N_pool`; tillväxt och foster tar protein
därifrån; överskott deamineras, kolskelettets energi går till reserven och
kvävet ut som urea till cellen. Den obligatoriska kväveförlusten är Brodys
~2 mg N per kcal basalmetabolism (4,8e-10 kg N per J) och tas ur poolen
först, ur mager vävnad sedan. Utan den har ett vuxet djur inget kvävebehov,
och Liebigs lag kan inte binda.

### Bokföringen

`in_fauna = 0,16·(1−s)·M + 0,16·N_pool + 0,16·(1−s_f)·gest_M`; reserverna bär
inget kväve. `N_pool` får en egen energiterm i ledgern och i 0210:s poster.

### Delpatchar

| | innehåll | typ |
|---|---|---|
| 1a | `M_wet()` och den härledda våta massan, införd där fysiken läser; vattenhalterna är 1, så ändringen är bitidentisk | refaktor |
| 1b–1d | **slogs ihop i 0216:** materialets kemi binder ihop tätheterna och kvävet — vävnad kan inte byggas av fett, och en uppdelning hade mätt ett mellanläge som ändå skulle tas bort | dynamik |
| 1b | tillstånden blir torrsubstans med vattenhalterna ovan; reserverna får sina tätheter; `reserve_cap` och `E_cap_per_M` blir J per kg **våt** massa; `fast_frac` fördelar energi; isolering och flytkraft läser fettvävens våta andel | dynamik |
| 1c | `N_pool`, kvävefria reserver, deaminering, urea, endogen förlust, kvävebegränsad tillväxt och fosterbygge; invarianten och ledgern följer | dynamik |
| 1d | kadavrets kväveöverskott mineraliseras vid döden | dynamik |
| 1e | verklig inlagringseffektivitet: protein k≈0,5, fett k≈0,75, glykogen k≈0,95; byggvärmen räknas i termoregleringen; fostrets underhåll som egen dränering (Kleiber på fostrets massa) i stället för ARC:s sammanslagna konceptus-k | dynamik |

### Beslut i detalj (2026-09-20)

**Kadavret (1d).** Världens pooler beskriver material med `s` och
växtkonstanterna och kan bära högst 3,3 % kväve per kg. Djurens magra
torrsubstans har ~13 %. Vid döden läggs kadavret in med labil näringshalt och
kväveöverskottet mineraliseras direkt till cellens fria näring — verkliga
kadaver ger snabbt en kväverik fläck. Kadavrets energi för asätare
underskattas därmed, eftersom fettet räknas som labil vävnad: **känd
förenkling som tas bort i steg 2**, när födans sammansättning modelleras och
kadaver är föda. Storleken talar för det: faunan i `f6-256` bär ~2–3 kg kväve
mot världens ~10 000, och kadaver var 0,0 % av födan i 0197:s mätning.

**Strukturandelen (1b, 1c).** `s` tolkas som askandel, intervallet 0,05–0,85
står kvar: mjukkroppade djur i ena änden, skalbärare i den andra, däggdjur
vid 0,15–0,20. Startdjurens median 0,44–0,64 är i praktiken standardgenomets
mittvärde och inte ett selektionsutfall; **de sätts in med ~0,15–0,20 via
scenariot** och selektionen får avgöra därifrån. Med kvävebegränsningen i 1c
blir axeln en äkta avvägning: hög askandel sparar kväve — kropp av mineral i
stället för protein — och kostar energi vid svält, tyngre kropp och sämre
flytkraft. **Askan har ingen källa**: mineraler är ingen valuta i modellen,
bara kvävet, och askan byggs av reservmassa mot enbart energi. Känd
förenkling, av samma slag som att växternas kol kommer ur luften.

**Inlagringen (1e).** Dagens overhead är 0,43 av materialets energi för all
vävnad, och att lagra intaget i reserven är gratis. Verkliga tal ger ~24 MJ
per kg protein, ~13 per kg lipid och ~1 per kg glykogen. Det gör djuren
dyrare att driva — värmeökningen efter måltid är 10–30 % av intaget i
verkligheten — medan steg 2 gör födan tre gånger energirikare. Mellanläget
blir sämre; varje steg mäts för sig.

## Mätpunkter

- Intag per kg kroppsvikt och dygn, i färsk och torr massa, mot 5–6 % torrt
  för en betare på 2 kg.
- Smältbar energi per kg torrt bete mot ~10 MJ.
- Reservens räckvidd i dygn per kg kropp mot verkliga djur.
- Andelen fiber som jäses, per kroppsmassa.
- Kroppsvattnets andel, dricksfrekvens, och andelen vatten ur föda,
  dryck och metabolism.
- Vattenbalansens drift efter utvidgningen.
