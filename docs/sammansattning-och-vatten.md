# Designskiss — sammansättning och vatten

*September 2026. Underlag för serien efter revisionen av faunans balans
(`docs/revision-faunans-balans.md`, fynd F1 och M3). Status: förslag, inte
beslut — uppdateras eller markeras som ersatt när serien byggs.*

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
torrsubstansen) från ved (~25 %). Öppen fråga nedan: egen axel, eller härledd
ur `s`.

### Djur

Kroppens torrsubstans delas i protein, fett och aska, plus kroppsvatten:

```
mager vävnad     protein + aska + vatten (hydratiseringen ~73 % av fettfri massa)
glykogen         M_fast: kolhydrat med ~3 kg vatten per kg
fettväv          M_slow: ~85 % lipid, lite vatten och protein
```

Dagens tre pooler (`M`, `M_fast`, `M_slow`) får en kemisk tolkning i stället
för en gemensam energitäthet. Strukturandelen hos djur (ben, keratin, kitin)
blir den del av den magra vävnaden som inte är omsättbar.

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

Näringen i modellen är i dag ett gemensamt "N+P". Tabellen räknar kväve; om
fosfor ska skiljas ut är en öppen fråga.

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
mikrobproteinet djuret får tillbaka. Det kan bli en ärftlig axel bredvid
`diet`; den första versionen behöver det inte.

**Omvandlingen från torrt till vått sker här**, på ett ställe: assimilerad
torrsubstans blir vävnad, och vävnadens vatten tas ur kroppens vattenpool.

---

## Vattnet

### Växternas vatten

Färsk växtmassa är 70–85 % vatten. Vattnet i plantan blir ett tillstånd, med
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

1. **Djurkroppens sammansättning.** Poolerna får kemisk tolkning; fettet sin
   täthet och ingen näring. Näringsbokföringen följer reservens sammansättning.
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

## Öppna frågor

- **Ligninets andel**: egen ärftlig axel, eller härledd ur `s` (vedartat är
  mer lignifierat)?
- **Växtens vatten**: eget tillstånd per planta, eller härlett ur markvattnet
  i cellen? Ett eget tillstånd är dyrare men ger vissnande och återhämtning.
- **Kväve och fosfor**: ska näringen delas? Fettets kvävefrihet och ureans
  vattenkostnad talar för kväve; fosforn sitter i ben och nukleinsyror.
- **Förmag eller baktarm**: axel från början, eller en gemensam form först?
- **Faunans livskraft**: varje steg mäts mot frågan om faunan bär sig i
  `f6-256`. Rätt fysik behöver inte ge livskraftiga djur — om de dör ut med
  rätt fysik är det en annan mekanism som saknas.

---

## Mätpunkter

- Intag per kg kroppsvikt och dygn, i färsk och torr massa, mot 5–6 % torrt
  för en betare på 2 kg.
- Smältbar energi per kg torrt bete mot ~10 MJ.
- Reservens räckvidd i dygn per kg kropp mot verkliga djur.
- Andelen fiber som jäses, per kroppsmassa.
- Kroppsvattnets andel, dricksfrekvens, och andelen vatten ur föda,
  dryck och metabolism.
- Vattenbalansens drift efter utvidgningen.
