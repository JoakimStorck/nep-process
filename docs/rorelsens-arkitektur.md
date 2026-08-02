# Designskiss — rörelsens arkitektur

*Augusti 2026. Status: förslag, med två fastlagda beslut markerade i texten. Underlag: kodgranskning av `agent.py` och `population.py` vid `afe8a51`, samt de tre p75-körningarna (100 000 tick, seed 1–3).*

Systerdokument: `docs/synens-axlar.md` äger perceptionen — vad organismen kan uppfatta och vad det kostar. Det här dokumentet äger handlingen — hur uppfattning blir rörelse. `docs/metabolismen.md` äger rörelsens och termoregleringens energikostnader; ingenting här definierar om dem.

---

## Varför dokumentet finns

Manifestet anger emergent beteende som det primära evolutionära målet. Rörelsen är den kanal där beteende faktiskt uttrycks — allt annat en organism gör är beroende av var den befinner sig.

Den nuvarande rörelsemotorn kommer från en tidig version av modellen och har sedan dess fått lager ovanpå sig utan att grundformen omprövats. Tre mätningar visar att den inte bär det den ska:

- **Djuren kommer ingenstans.** Rakheten över livet — nettoförflyttning genom bansträcka — är 0,069. De rör sig fort, 37 cellbredder per månad, och lägger 1 563 cellbredders bana på 85 cellbredders förflyttning.
- **De uppfattar exakt en artfrände** — den närmaste — vilket gör flockning fysiskt omöjlig och blockerar parningsdriften när fel granne står närmast.
- **Predationen är död kod.** Noll predationsdödsfall i tre körningar om vardera 100 000 tick.

Följden är att faunans hela fenotyprymd är outnyttjad. Tre beteendetraits har noll läsare, och de beteenden som finns är hårdkodade reflexer som selektionen inte kan röra.

---

## Nuläget

### Kedjan

```
MLP (23 obs + rekurrent hidden → 5 utgångar)
  ↓ _decode_action_outputs   turn = tanh(y₀), thrust, allow_move, allow_eat, explore_drive
  ↓ _apply_reflex_drives     exklusiv if/elif: hot → byte → partner → social
  ↓ _apply_food_steering     additiv, grindad på flee_state < 0,5 och hunger > 0,4
  ↓ _integrate_motion        heading += dt · 300 · (0,85 · allow_move · turn + 0,25 · jitter)
                             F_prop = u · F₀ · M^(2/3), explicit Euler mot dragkraft
```

### Tio defekter

**1. Riktningen slumpas om varje tick.** `turn_rate · dt = 300 · 0,02 = 6,0 rad/tick` mot ett varv på 6,28. Enbart utforskarbruset ger en standardavvikelse på 0,97 rad/tick. Headingens dekorrelationstid är därmed ungefär ett tick.

**2. Fartintegrationen är divergent för merparten av populationen.** Explicit Euler mot dragkraften har förstärkningsfaktorn `|1 − dt·c₁/M|`, som passerar ett vid `M = dt·c₁/2 = 2,2 kg`. Uppmätt medianmassa är 1,3–1,7 kg. Schemat hålls ändligt bara av klampningen mot noll och `v_max`, och `F_prop` och `v` är därför aldrig i kraftbalans — vilket också gör att farten inte går att härleda ur `E_loss_loco` under antagande om stationaritet.

**3. Rotationsbruset är inte tidsstegsinvariant.** Steget skalar som `dt`, inte som `√dt`. Variansen per månad blir `c²σ²·dt` — halveras tidssteget halveras den effektiva rotationsdiffusionen. Modellens beteende beror alltså på tidsstegets storlek. Det är samma familj som den termiska relaxationen och den newtonska rörelseintegrationen, men en annan variant: stokastisk integration i stället för styv relaxation.

**4. Styrregulatorn slår över.** Förstärkningen är `0,85 · 300 · 0,95 / π = 1,54` korrektion per enhet fel och tick. Över ett betyder att felet byter tecken varje tick och dämpas med faktor 0,54. Det är vinglandet, och det är kalibrerat mot ett tick som är för långt.

**5. Driftkedjan är exklusiv.** Exakt en drift per tick, ingen viktning. Ett flyende djur kan inte samtidigt söka mat — food steering är dessutom separat grindad på `flee_state`. Verkliga djur avväger; de växlar inte läge.

**6. Parningsgrenen kastar policyns beslut.** `turn = clamp(0,95 · biasN, …)` — inte `turn +`. De övriga grenarna adderar. MLP:n är alltså helt frånkopplad i just det tillstånd där selektionen skulle ha mest att arbeta med.

**7. Agenten uppfattar exakt ett annat djur.** `see_agent_first_hit()` loopar avståndssteg utifrån och `break`ar vid första träffen. Observationsvektorn bär `pred_bearing` och `pred_dist` för den närmaste individen och ingen annan. Två följder: kohesion, separation och alignment saknar sina percept, och `best_mate` blir `None` när närmaste granne inte är parningsredo — även om en mottaglig partner står två cellbredder bakom.

**8. Riskbedömningen läser motpartens genom.** `_evaluate_local_agent_drives()` och `attack_risk()` läser `pheno.predation`, `pheno.diet`, `body.M`, `body.D` och `body.reserve_frac()` hos den upptäckta individen. De tre senare är i bästa fall delvis synliga; de två första är arvsmassa. Djuret vet att grannen är ett rovdjur genom att läsa dess gener. Det håller i en värld där hotgrenen aldrig avfyras och håller inte i den värld som byggs här.

**9. Predationen kan inte löna sig.** Tre skäl, alla verifierade:

- `attack_energy_gain = 0,5` har **noll läsare**. `_step_predation()` drar predatorns energi och lägger skada på bytet, men predatorn tillgodogörs ingenting. Kadavret hamnar i världen som allmän egendom.
- Kadavret poolas med förna i `detritus`, massviktat på strukturandel. En ren asätare får 0,087 ur detritus vid strukturandel 0,80 mot betarens 0,258 ur flora vid 0,57. Predatorn ser aldrig ett rikt kadaver. Frågan är sedan tidigare noterad som eget beslut i `TODO.md`; här blir den blockerande.
- `attack_range = 1,5` cellbredder mot ett medelavstånd till närmaste granne på 6,4 (N = 100) till 22,6 (N = 8). Med defekt 1 kan avståndet inte slutas.

Utfallet: `hunt_eff = predation · diet^1,5` når `predator_trait_min` 0,20 hos 4–11 % av födslarna och `threat_predation_min` 0,35 hos 1–3 %. Hotgrenen avfyras nästan aldrig.

**10. Sex parametrar och tre traits har noll läsare.** `mobility`, `risk_aversion` och `cold_aversion` i fenotypen; `flee_radius`, `mate_search_radius` och `attack_energy_gain` i `AgentParams`. `sociability` har en läsare, sist i en kedja som sällan når dit. Det är B1-glappet, på beteendesidan.

### Vad som ändå fungerar

Reflexernas *innehåll* är rimligt: flykt vänder från hotet och ökar farten, jakt vänder mot bytet, parning styr mot partnern. Riskvärderingen i `attack_value` / `attack_risk` väger massa, reserv och motpartens skada — vägningen är rimlig även om dess indata inte är det. Sensingens adaptiva frekvens sänker kostnaden när ingenting händer. Ingenting av det behöver kastas; det behöver bli bidrag i stället för lägen.

Och en observation värd att notera: `sociability` skiljer körningarna åt.

```
              median   början → slut     lutning
seed 1 (dog)   0,469   0,424 → 0,463   +0,02/1000 mån
seed 2 (dog)   0,120   0,438 → 0,123   −1,00/1000 mån
seed 3 (levde) 0,877   0,636 → 0,744   +0,07/1000 mån
```

Den som överlevde utvecklade attraktion, den som dog snabbast repulsion. Med n = 3 och en enda grundare per linje kan tecknet vara drift genom flaskhalsen. Men det är rätt tecken på rätt axel, genom en gren som knappt får verka.

---

## Del 1 — Rotationen

*Numerisk rättning plus en beteendemekanism. Oberoende av allt annat i dokumentet och bör göras först, eftersom varje beteendemätning dessförinnan mäter brus.*

### Rätt svar är inte rak rörelse

Den slingrande banan är inte i sig ett fel. Kringgående sök i ett område är rätt beteende när födan är riklig — det finns ingen anledning att färdas då, och den kurviga banan är just det som håller organismen kvar på fläcken. Områdesbegränsad sökning är ett av de bäst belagda rörelsemönstren i naturen.

Det som saknades var inte raka linjer utan **förmågan att välja**: att färdas när det finns skäl och söka lokalt när det inte finns. En organism vars riktning dekorrelerar på ett tick kan bara göra det ena.

### Ändringarna

**Riktningen får en persistenstid, och den är tillståndsberoende.** Headingen blir ett tillstånd med tröghet. `τ_dir` interpolerar mellan en kort söktid — kringgående sök på fläcken — och en lång färdtid. `explore_drive` väljer regim och byter därmed roll: den höjde tidigare bruset, alltså gav mer utforskning *sämre* förflyttning. Den är redan dämpad av `hunger · food_local` i födostyrningen och är därför rätt signal. Bruset skalar som `√dt` så att rotationsdiffusionen blir tidsstegsinvariant.

**Styrningen blir relaxation mot en önskad riktning**, med tak på vridhastigheten i stället för en proportionell förstärkning på 1,54 per tick. Formen är ovillkorligt stabil vid godtyckligt `dt`, av samma skäl som termoregleringen fick analytisk relaxation.

**Svängradien kopplas till farten** via centripetalvillkoret `ω ≤ a_lat / v`. Fart köper räckvidd och kostar manöverförmåga. Talet är satt så att en organism kan vända inom sitt eget synfält vid marschfart; annars går födostyrningen sönder.

`_T_MOB` (locus 11, utan läsare) får äga **färdregimens** persistenstid. Axeln är alltså inte "hur rakt djuret rör sig" utan "hur långt det förflyttar sig när det bestämt sig för att förflytta sig".

### Uppmätt

8 000 tick, seed 1, samma instrumentering i alla tre:

```
                      rakhet   p90     netto    fart    dödsfall   ålder
baslinje               0,069  0,134    85 cb   37,3       74       43,4
fast persistens        0,232  0,880   250 cb   36,6       57       47,1
områdessökning         0,175  0,387   252 cb   33,5       44       50,0
```

Områdessökningen når **samma nettoförflyttning som den fasta persistensen med hälften så hög topprakhet**. Den färdas alltså lika långt utan att sluta söka, vilket är hela poängen. Den ger också färst dödsfall och längst liv av de tre.

Tidsstegsberoendet, samma seed vid `dt = 0,01` mot `dt = 0,02`:

```
                      dt=0,02   dt=0,01   ändring
baslinje                0,069     0,051     −26 %
områdessökning          0,175     0,195     +11 %
```

Förbehåll: körningarna vid `dt = 0,01` avbröts efter 137–144 månader och ger n = 25 respektive 35, så talen är brusiga. Den gamla formens tidsstegsberoende är dessutom sammansatt — bruset skalar fel *och* styrförstärkningen faller från 1,54 till 0,77 och slutar slå över — så rakheten ensam separerar inte de två. Att brusskalningen är fel följer av formen och inte av mätningen: en slumpvandrings steg måste gå som `√dt`.

### Farten, och en rättelse

Fartintegrationen byter form i samma arbete men i en egen patch: `F_prop = c₁v + c₂v²` löses direkt i stället för att integreras explicit mot en relaxationstid på 0,45 tick.

**Rättelse.** Uppdelningen motiverades först med att en samtidig ändring flyttade dödsorsakerna "från 68 % svält till 63 % skada". Den jämförelsen var ogiltig — 68 % kom från p75:s hela 100 000-tickskörning och 63 % från en körning på 8 000 tick. Mätt mot rätt baslinje vid samma längd:

```
                     skada   svält
baslinje              58 %    32 %
5h, områdessökning    61 %    25 %
5i, kvasistatisk      43 %    45 %
```

Den kvasistatiska farten flyttar alltså fördelningen **tillbaka mot svält**, i motsatt riktning mot vad jag påstod. Uppdelningen i två patchar är ändå riktig — en ändring per körning är projektets metod — men den var inte motiverad av det jag angav.

### `effort` normeras mot ett arkitektoniskt tak

`speed_n = speed / v_max` med `v_max = 100`, som är en klampningsgräns och ingen biologisk fart. Uppmätt efter 5i: `speed_n` 0,32, `effort` 0,48, `rest` 0,50 i medeltal. Skadeinflödets termer, andel av totalt:

```
dD_met     49 %     drain_rate mot basalmetabolismen
dD_eff     29 %     effort, alltså fart och aktivitet
dD_age     15 %
dD_starve   6 %
dD_cold     1 %
```

Rörelsen står alltså för knappt en tredjedel av skadan, inte för merparten. Omnormeringen ska göras mot den mätningen och inte mot en gissning, och den ligger därför efter p78.

### Vad som fortfarande saknas

`_T_MOB`:s avvägning är tunn tills farten kopplas till rörelseregimen: att färdas ska vara snabbt och därmed dyrt, att söka lokalt ska vara långsamt och billigt. Det är en beteendeändring och inte en numerisk rättning, och den hör till ett eget steg.

## Del 2 — Grundprinciper

**Drifter viktas, de utesluter inte varandra.** Varje drift producerar en önskad riktning och en vikt. Summan är en vektor; dess riktning blir målriktningen och dess längd bidrar till farten. Ett hungrigt djur som ser ett hot får en kompromiss, inte ett läge.

**Ingen drift skriver över policyn.** MLP:ns utgång är ett bidrag bland de andra, inte något som kastas eller kastar.

**Driftens vikt är genetisk, dess innehåll är det inte.** Reflexernas geometri — vänd från hotet, styr mot partnern — är fysik och behöver inte evolveras. Hur mycket den ska väga mot allt annat är strategi och ska evolveras. Det låter selektionen stänga av en reflex helt utan att koden behöver ett specialfall.

**Varje beteendetrait ska ha motverkande konsekvenser.** Projektets hårda regel. En vikt som bara har uppsida är ett reglage.

**Perception är aggregerad och lokal.** Ett grannskap, aldrig en global lista och aldrig en enda träff. Aggregering är inte räckvidd: det som föreslås här ligger helt inom befintlig `sense_radius`. A8 gäller oförändrat.

---

## Del 3 — Perceptionen motorn kräver

Rörelsemotorn kan inte byggas ovanpå dagens percept. Den kräver tre saker av `docs/synens-axlar.md`, och ingen av dem ökar räckvidden.

**Sektorformatet.** `synens-axlar` föreslår fix antal riktningssektorer med akuiteten som blandningsgrad, så att MLP:ns indimension är oberoende av genotypen. Det är exakt det gränssnitt rörelsen behöver. Formatet ägs av `synens-axlar`; det här dokumentet konsumerar det.

**Artfrändekanalen blir ett aggregat.** Per sektor: antal, medelavstånd och — för alignment — medelriktning. `see_agent_first_hit()` avvecklas. Uppslaget går via `cells_within(sense_radius)` och grannmatrisen, alltså samma celler som redan läses. Skillnaden mot i dag är att dagens kod *kastar* allt utom den närmaste.

**Temperaturen blir en percept.** Ett skalärvärde per sektor ur `temperature_of_cells()`. Billigast av alla nya kanaler eftersom fältet redan finns per cell.

### Vad som är blockerande och vad som inte är det

> **Beslut, augusti 2026.** Det som blockerar rörelsemotorn är **genomuppslaget**, inte hela diskrimineringsarbetet.
>
> Sektorpercepten ska bära observerbara storheter — storlek och fart — och bedömningen "är detta ett hot" ska göras av organismen ur dem, med möjlighet att slå fel. Då får `attack_risk` en ärlig grund och defekt 8 försvinner.
>
> Den fulla formen av "sensorn detekterar, organismen tolkar" — att flora, kadaver och artfrände kan förväxlas — är fortsatt önskvärd men inte längre blockerande. Den behåller sin nedprioritering.

---

## Del 4 — Driftsmodellen

Sex drifter, var och en med en riktning ur perceptionen och en vikt ur genom och tillstånd.

| Drift | Riktning | Vikt skalar med |
|---|---|---|
| Föda | starkaste födosektorn, viktad med dietverkningsgrad | hunger |
| Flykt | bort från hotsektorn | `risk_aversion` × hotets närhet |
| Jakt | mot bytessektorn | `hunt_eff` × `attack_score` |
| Parning | mot partnersektorn | parningsberedskap |
| Värme | uppför temperaturgradienten | `cold_aversion` × köldunderskott |
| Social | mot eller från grannarna | `sociability` × avvikelse från önskat avstånd |

Vikterna normeras inte bort mot varandra. Summan får vara stark eller svag, och dess längd modulerar farten — ett djur utan drivkrafter står still, vilket är energiskt riktigt och beräkningsmässigt billigt.

Hungern är den variabel som gör modellen dynamisk: den skalar födodriften kontinuerligt i stället för dagens tröskel vid 0,4, och den kan därmed konkurrera ut den sociala driften när det är ont om mat. Det är mekanismen bakom flockdelning i Del 6.

### MLP:ns roll

> **Beslut, augusti 2026.** Nätverket **viktar** drifterna, och får därutöver en fri riktningskanal. Målet är att den fria kanalen på sikt ska kunna ta över.

Fyra alternativ övervägdes.

**A — nätverket viktar drifterna.** Litet, välkonditionerat sökrum; kort väg från genom till beteende. Men evolutionen kan bara välja bland de sex riktningar som är förkodade.

**B — nätverket bidrar vid sidan av reflexer med genetiska vikter.** Full riktningsfrihet, men två ärftlighetskanaler för samma beteende som muterar oberoende och kan ta ut varandra. Frågan "utvecklades flyktbeteende?" blir obesvarbar. Det är i praktiken dagens design med addition i stället för överskrivning, och därifrån kommer problemet.

**C — ren reflexmodell utan nätverk.** Inte övervägd på allvar; den avskaffar A10.

**D — drifterna blir percept och nätverket är enda aktuator.** Den rena formen av A10, och rätt mål. Men ett slumpinitierat nätverk har ingen flyktrespons, och flykt kan inte läras genom försök när ett misstag är dödligt. Med PC1 på 10,8 % av variansen och flaskhalsar på åtta individer finns ingen selektionskraft att bygga det med. Den tar dessutom bort de genetiska krokarna igen.

**Vald form: A med en utgång som är D.** Reflexerna ger riktningar, nätverket ger vikter, och en sjunde utgång är en fri riktning med egen vikt. Drifternas riktningar går dessutom in i observationsvektorn, så nätverket ser vad det viktar.

Systemet startar därmed i det nåbara läget och kan flytta vikt till den fria kanalen om det någonsin får selektionskraft. Och övergången blir **mätbar**: andelen av styrvektorns längd som kommer från den fria kanalen är ett tal som kan följas över körningar. "Hur mycket av beteendet är evolverat snarare än reflex" blir en skalär i stället för en trosfråga.

Kostnaden är två extra MLP-utgångar, sju i stället för fem. Formen motsvarar hur nervsystem faktiskt är byggda: reflexbågar som förkopplad överlevnad, med moduleringsvägar ovanpå som kan hämma dem.

---

## Del 5 — De fyra kontexterna

### Kyla

Världen har redan en verklig gradient. Med `T_eq = 30`, `dT_pole = 30` och säsongsamplitud 3 vid ekvatorn mot 15 vid polen:

```
latitud     medel   vinter   sommar   termokostnad vid medel
ekvator      30,0     27,0     33,0        0,2 × basal
0,5          19,4     12,2     26,6        0,5 ×
0,75         10,5     −0,3     21,3        0,8 ×
pol           0,0    −15,0     15,0        1,2 ×
```

Att leva vid polen kostar alltså ungefär två gånger så mycket som vid ekvatorn, och kostnaden svänger med årstiden. Djuren har i dag ingen beteendemässig respons på det alls — de betalar och stannar.

`_T_COLD_AV` (locus 15) får sin läsare som vikt på värmedriften. Avvägningen är dubbel och behöver inte konstrueras: den som flyr kylan lämnar sin födobas, och om alla söker sig mot ekvatorn stiger konkurrensen där. Båda kostnaderna finns redan i modellen.

Det intressanta utfallet ligger ett steg bort. Säsongsamplituden vid polen är ±15 °C mot ±3 vid ekvatorn, vilket betyder att den lönsamma latituden **flyttar sig över året**. Det är förutsättningen för att säsongsvandring ska kunna uppstå utan att kodas — och vandring är i sin tur den mekanism som skulle ge betestrycket rumslig och tidsmässig struktur i stället för dagens likformiga tryck.

### Matbrist

Födodriften blir kontinuerlig i hungern i stället för grindad, och den blir en av sex vikter i stället för ett tillägg efter reflexkedjan. Ett hungrigt djur ska kunna överge flocken; ett mätt ska inte.

Lokal utarmning bär redan sin egen signal genom betningens funktionella respons: `B` räknas över `r = 1`, så flera djur i samma grannskap tömmer det tillsammans. Det behöver ingen ny mekanism, bara att drifterna kan väga mot varandra.

### Parning

Parningsdriften slutar kasta policyns beslut och blir ett bidrag. Viktigare: den slutar vara beroende av att närmaste granne råkar vara parningsredo, eftersom sektoraggregatet kan bära antalet parningsredo grannar per sektor i stället för identiteten hos en.

`mate_search_radius = 5,0` får en läsare eller tas bort. `mating_radius = 3,0` behålls som den hårda grinden i `_try_mating`.

### Hot

Predationen aktiveras. Det kräver tre saker utöver rörelsemotorn:

**Predatorn måste tillgodogöra sig sitt byte.** `attack_energy_gain` får en läsare: en andel av bytets förlorade massa och energi går direkt till predatorn vid lyckad attack. Utan det är jakt ren kostnad med diffus allmännytta, och nischen kan aldrig evolveras fram.

**Kadaver måste skiljas från förna.** Antingen som egen pool eller genom att strukturandelen inte massviktas i samma fält. Så länge ett kadaver vid strukturandel 0,25 blandas ner till 0,80 av omgivande förna är asätarnischen olönsam, och eftersom `hunt_eff` kräver hög `diet` är predatorn i modellen en asätare som dödar sitt eget kadaver. Beslutet står sedan tidigare öppet i `TODO.md`.

**Räckvidden måste vara nåbar.** `attack_range = 1,5` är rimlig som greppavstånd men förutsätter att förföljelse fungerar. Den beror alltså på Del 1.

`_T_RISK_AV` (locus 9) får sin läsare som vikt på flyktdriften. Avvägningen är den klassiska och behöver inte uppfinnas: vaksamhet kostar födotid och flykt kostar energi, medan låg vaksamhet kostar liv. Att den blir mätbar först när predationen faktiskt biter är hela poängen med att aktivera den.

En anmärkning om ordning: predationen bör aktiveras **efter** att rörelsen fungerar, inte samtidigt. Ett rovdjur i en värld där ingen kan förfölja eller fly mäter ingenting.

---

## Del 6 — Flocken

### Två regler, inte tre

Reynolds tre regler brukar skrivas som kohesion, separation och alignment. Kohesion och separation är dock samma regel med olika tecken: styr så att avståndet till grannarna närmar sig ett önskat avstånd. Det är också så självdrivna partikelmodeller normalt formuleras — en repulsionszon innanför en attraktionszon, med ett föredraget avstånd emellan.

Det ger:

| Regel | Percept | Uppsida | Nedsida |
|---|---|---|---|
| Avståndshållning | grannarnas medelbäring och medelavstånd per sektor | parningsfrekvens, utspädning mot predation | lokal utarmning av betet |
| Alignment | grannarnas medelriktning | flocken rör sig samlat och håller ihop under förflyttning | följer fel individ |

Två loci i stället för tre: `social_distance` (önskat avstånd) och `flock_alignment` (vikt). `sociability` finns redan på `_T_SOC` och blir avståndshållningens vikt.

Alignment är den dyraste percepten — den kräver grannarnas riktning, inte bara deras position — och bör byggas i andra hand. Avståndshållningen ensam ger flockar; alignment ger flockar som färdas.

### Uppsidan måste vara verklig

Flockningens naturliga fitnessargument är utspädning och kollektiv vaksamhet. Med predationen aktiverad enligt Del 5 blir båda tillgängliga för första gången. Utan predation vore den enda uppsidan parningsfrekvens, och då hade axeln varit tunn.

Det ger ett skarpt mätkriterium: **flockningen ska vara en anpassning bara i den värld där predationen biter.** Kör man samma seeds med och utan aktiv predation ska `sociability` differentiera i den ena och drifta neutralt i den andra. Gör den inte det är uppsidan felskalad, eller så är utspädningen inte verklig i implementationen.

### Delning som emergens, inte som regel

Fission–fusion behöver inte kodas. Den faller ut ur att drifterna viktas mot varandra:

En mätt flock domineras av den sociala driften och håller ihop. När betet i grannskapet tar slut stiger hungern, födodriften växer, och individerna följer var sin lokala gradient — som pekar åt olika håll eftersom resursen är fläckvis. Flocken delar sig. Har delarna hittat föda faller hungern, den sociala driften tar över igen, och de som råkar mötas smälter samman.

Trängsel ger samma sak från andra hållet: överskrids det önskade avståndet av tillräckligt många grannar blir avståndshållningens bidrag repellerande för dem i mitten, och gruppen tänjs isär.

Det förutsätter tre saker som alla ligger i den här skissen: att drifterna summeras i stället för att utesluta varandra, att hungern skalar kontinuerligt, och att `social_distance` är ett föredraget avstånd snarare än en ren attraktion. Ingen av dem är en regel om flockdelning.

Om delningen faktiskt uppstår är det dessutom svaret på ett öppet problem i `TODO.md`: en population som svänger i flera osynkroniserade delbestånd har en helt annan utdöenderisk än en som svänger samlat. Två av tre p75-körningar dog vid Allee-tröskeln med hela världen i fas.

---

## Del 7 — Genomets kostnad

Genomet är 38 loci mot manifestets avsedda 8–16. Avvikelsen har noterats i tre statusanalyser utan att beslutas.

Den här skissen adderar **två** loci och ger **tre befintliga döda loci** sina första läsare:

```
_T_RISK_AV     (9)   död → flyktdriftens vikt
_T_SOC        (10)   svag → avståndshållningens vikt
_T_MOB        (11)   död → riktningspersistens och marschfart
_T_COLD_AV    (15)   död → värmedriftens vikt
_T_SOCIAL_DIST (ny)  → önskat grannavstånd
_T_ALIGN       (ny)  → alignmentens vikt
```

Netto 38 → 40. Det är en expansion, men den övervägande delen av arbetet består i att aktivera representation som redan bärs och betalas för utan att göra något. Om avvikelsen ska åtgärdas är fauna-loci utan tydlig konsekvens rätt ställe att skära, inte de här.

---

## Del 8 — Vad som ska mätas

| Egenskap | Målvärde | Nuläge |
|---|---|---|
| Rakhet: nettoförflyttning ÷ bansträcka per livstid | högre, och spridd mellan individer | 0,069 → **0,175** |
| Rakhet vid halverat `dt` | oförändrad | +11 % mot baslinjens −26 % |
| Beteendetraits utan läsare | 0 | **3** |
| Predationsdödsfall per 100 000 tick | > 0 | **0 i tre av tre** |
| Latitudinell fördelning, vinter mot sommar | mätbart skild | ingen skillnad |
| Spridning i `sociability` med predation på mot av | differentierar mot drift | ej mätbart |
| Andel parningsförsök som faller på "ser ingen" | faller under dagens 91 % | 91 % |
| Korskorrelation i födelsetakt mellan latitudband | < 0,3 | **0,45–0,67** |
| Andel av styrvektorn ur MLP:ns fria kanal | följs över tid | — |

Den första raden är den enda som måste vara sann innan någon av de andra betyder något. Målet är inte att maximera den — en organism som alltid färdas rakt söker inte — utan att den ska variera med tillståndet. Den sista är övergångsmåttet från A mot D och har inget målvärde — den ska följas, inte optimeras.

---

## Öppna frågor

**Ska drifterna summeras som vektorer eller som vinkelbidrag?** Vektorsumma är enklare och ger fartmodulering gratis, men två motsatta drifter tar ut varandra och lämnar djuret stillastående — vilket ibland är rätt och ibland är förlamning. Vinkelbidrag undviker det men förlorar fartkopplingen. Troligen vektorsumma med ett golv på farten under stark total drivkraft.

**Hur mycket ska utspädningen vara värd?** Predationens riktning mot en flock måste minska varje individs risk för att flockning ska löna sig. Görs det genom att predatorn bara kan angripa ett mål åt gången — vilket redan gäller — eller behövs en explicit förvirringseffekt? Det förra är billigare och mer ärligt; om det räcker är en empirisk fråga.

**Vad händer med `explore_drive`?** I dag är den MLP:ns femte utgång och multipliceras med bruset. Med riktningspersistens i Del 1 blir dess roll otydlig — den kan bli persistenstidens modulator i stället för brusets amplitud, vilket vore mer meningsfullt men gör `_T_MOB` delvis överlappande.

**Ska köldaversionen verka på gradienten eller på minnet?** En gradient inom sensingradien räcker för lokal termoregulering men inte för säsongsvandring, som kräver en riktning som hålls över månader. Det senare är antingen ett rekurrent tillstånd i MLP:n — som finns — eller ett fält att följa. Frågan bör avgöras av om vandring faktiskt uppstår med enbart gradienten.

---

*Skriven efter granskningen av p75. Ersätts när arbetet är byggt. Del 1 är oberoende av resten och bör göras först.*
