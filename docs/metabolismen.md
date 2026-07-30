# Metabolismen — referens

*Juli 2026. Beskriver hur materia, energi och näring rör sig genom en organism och genom världen, vilka storheter som är bevarade och vilka som inte är det, samt de fällor som visat sig när tidsskalan ändrades.*

---

## Tre valutor, olika lagar

Modellen räknar tre storheter. De blandas ofta ihop, och nästan varje fel vi hittat har berott på att någon behandlats som en annan.

**Massa är inte bevarad.** Flora bygger merparten av sin vävnad ur luft; faunans kol lämnar kroppen som koldioxid vid förbränning. Massbalansen är därför en diagnostisk storhet, aldrig en invariant. Den som söker en sluten massbudget kommer att hitta ett "läckage" som i själva verket är andning.

**Energi är bevarad inom organismen, per tick.** `Body.step()` för en ledger som ska stämma på maskinprecision: ingående reserv plus intag plus katabolism minus dräneringar, reparation, material och överskott ska vara lika med utgående reserv. Ledgern är hård — avvikelser räknas som fel, inte som brus.

**Näring är bevarad globalt.** Det är modellens egentliga valuta. Summan av fri näring, näring bunden i flora, i fauna och i detritus ska vara lika med extern tillförsel minus förlust. `check_nutrient_balance()` är hård invariant med toleransen 1e-6 relativt.

Skälet till att just näringen är den bevarade storheten står i `docs/naringens-ekonomi.md`: kol och väte finns i överflöd i luft och vatten, medan fosfor och kväve är det som faktiskt cirkulerar och begränsar.

---

## Substratet

All organisk materia beskrivs av **en** egenskap: strukturandelen `s` mellan 0 och 1. Allt annat härleds.

```
energiinnehåll     E_labile · (1 − s)              E_labile = 9,302e6 J/kg
näringsinnehåll    N_L·(1 − s) + N_S·s             N_L = 1/30, N_S = 1/500
matsmältning       0,80 − 0,35·s                   andel av det labila som tas upp
```

Strukturmaterial är energifattigt, näringsfattigt och svårsmält. Labilt material är motsatsen. En seg växt bär mindre av allt per kilo — den är sämre föda och billigare att bygga.

Att modellen har en enda substrataxel är ett medvetet val. Det betyder att energirikt och näringsrikt är nästan samma sak, så det finns ingen föda som är proteinrik men energifattig. Vill man ha den avvägningen räcker inte `structure` som beskrivning.

---

## Organismens pooler

En fauna-organism bär massa i tre pooler:

```
M                    committad vävnad, struktur och funktion
M_fast + M_slow      mobiliserbar reserv, labil
gest_M               foster, labilt tills födseln
```

Den bevarade storheten är summan. `M` bär organismens egen strukturandel; reserven och fostret är labila, alltså rent näringsrika. Kadavret bär hela summan med massviktad strukturandel — en välgödd kropp lämnar ett mjukare kadaver.

Reservens tak är `reserve_cap · M`, där `reserve_cap` är en genetisk axel som spänner ungefär 8 till 42 procent av kroppsmassan. Att bära reserv kostar, eftersom reservmassan räknas in i `M_carry` och därmed belastar basalmetabolism, rörelse och värmeförlust. Utan den kostnaden hade axeln ingen avvägning att selekteras på.

---

## Flödena genom en tick

### Intag

`consume_food()` rangordnar tillgänglig föda efter **värde per ingesterat kilo**:

```
värde = assimilated_fraction(s, verkningsgrad) · E_labile
assimilated_fraction(s, e) = (1 − s) · matsmältning(s) · e
```

Verkningsgraden kommer ur `diet`-traiten: `herb_eff = (1−d)^0.7` för levande flora, `scav_eff = d^0.7` för detritus. Bästa födan tas först, näst bästa rörs bara med det som återstår. Preferensen lagras inte — den härleds ur fysiologin.

Betningen når så långt organismen färdats under ticken. Ett djur äter medan det går.

Av det ingesterade passerar `assimilated_fraction` tarmväggen; resten exkreteras till cellen som detritus. Eftersom strukturmaterialet passerar i sin helhet medan bara en del av det labila gör det, är exkrementet **strukturrikare än födan** — betning koncentrerar segt material i detrituspoolen.

Den assimilerade massan är labil per konstruktion och går rakt in i reserven.

### Dräneringar

Obligatoriska poster, betalade ur reserven i ett svep:

```
basal        k_basal · M_carry^0.75 · metabolism_scale
hjärna       compute_cost · M_carry, skalad mot nätverkets storlek
sensing      per aktivitetsnivå
rörelse      F_prop · v / locomotion_eff
termo        värmeproduktion för att hålla Tb
gestation    overhead för att bära foster
```

Räcker reserven inte kataboliseras egen vävnad ner till `M_min`. Bara den labila fraktionen kan mobiliseras: `dM · (1 − s) · catabolism_eff` blir reserv, resten exkreteras. Vid faunans typiska strukturandel 0,25 ger det samma utbyte som den gamla konstanten `E_body_J_per_kg`, vilket inte är en slump — 7,0e6 ≈ 9,302e6 · (1 − 0,25).

### Tillväxt

Tillväxt är **diskretionär** och körs efter dräneringarna och katabolismen. Den får bara ta av det som återstår och kan därför aldrig utlösa katabolism.

Material tas ur reserven, ett kilo per byggt kilo, med `growth_E_per_kg` som syntesarbete ovanpå. De två är termer, inte alternativ: byggkostnaden är ATP för syntesen, materialet är det som blir vävnad. Eftersom reserven är labil och vävnaden bär struktur binder ett kilo vävnad mindre näring än ett kilo reserv — mellanskillnaden utsöndras som kvävehaltigt avfall.

### Förbränning och utsöndring

Varje kilo reservmassa som oxideras släpper `N_L` näring till cellen. Kolet lämnar modellen. `Body` har ingen världsreferens: den ackumulerar `out_excreta_kg`, `out_excreta_struct_kg` och `out_nutrient_kg`, som body-passet tömmer till rätt cell.

Överföring till avkomma bränns inte — `take_energy(burn=False)` finns för det.

### Skada och reparation

Skada byggs upp från ansträngning, metabolisk stress, ålder, svält och köld, och reduceras av reparation som kostar energi. `D_max` nås och organismen dör.

Reparationens tak och verkningsgrad avtar båda med **slitaget** `W`, och `W` är biologisk ålder: den tickar med `wear_a0` för varje individ oavsett hälsa, och ökar därutöver med skadetakten. Att ålderstermen är den bärande är avgörande. Byggs `W` enbart ur skada blir ett djur som aldrig skadas aldrig slitet, och ett djur som aldrig slits reparerar för evigt — åldrandet finns då som inflöde men aldrig som ackumulation.

Två saker är värda att veta. **Katabolism är inte skada.** `dD_starve` mäter redan utmärgling som massa relativt förväntad massa för åldern, alltså utfallet; att också straffa mekanismen är dubbelräkning. `k_cat_dmg` står därför lågt. **Reparationen är en takt**, multiplicerad med `dt` som allt annat, och `repair_capacity` är en genetisk axel med en kostnad som biter — livshistorieteorins avvägning mellan underhåll och reproduktion.

---

## Näringens väg genom världen

```
vittring  ->  fri näring  ->  flora  ->  fauna  ->  detritus  ->  fri näring
                  ^                         |                        |
                  +--- exkretion, förbränning ---+       nedbrytningsförlust
```

Flora tar upp fri näring ur sin cell, delad **proportionellt mot rotarea**: `A_i / max(1, ΣA)` över cellen, med överskjutande area fördelad på grannringen. Tidigare delades den mot tillväxtunderskottet, vilket gjorde att `uptake_capacity` band i noll av alla individer — se `docs/statusanalys-vaxtcykeln.md`. Kostnaden per byggt kilo är `nutrient_content(s)`, vilket sjunker med strukturandelen: strukturmaterial är kolrikt och näringsfattigt, och det är därför träd klarar mager mark där örter inte gör det.

Fauna får näring med födan och utsöndrar den vid förbränning och vävnadsomsättning. Detritus bryts ner till fri näring; en andel `nutrient_loss_frac` lämnar systemet.

Den sista termen är en platshållare. Den verkliga sänkan är begravning som sediment vid havsranden, alltså rumslig och inte proportionell mot nedbrytning överallt. När hydron finns bör den ersättas.

**Lokaliteten är bärande, inte bara en prestandaprincip.** Vatten flyttar näring strikt utför. Det enda som flyttar den uppför är kedjan förtäring, upptag, tillväxt, förflyttning, död, nedbrytning. Blir upptag, exkretion eller kadaverdeponering globalt eller instantant försvinner motgradienten och världen skiktas permanent efter hydron.

---

## Tidsskalan

Tidsenheten är **månader**. Ett tick är `dt = 0,02` månader, alltså ungefär 14,4 timmar. En livslängd på omkring 47 månader är 2 350 tick. Talet är en kalibreringsreferens, inte ett utfall: uppmätt når individer 157 månader, och åldersdödens tak ligger på 121 till 153 beroende på `repair_capacity`.

Enheten är inte godtycklig. Den valdes av att livstidsomsättningen — hur många gånger en organism omsätter sitt eget energiinnehåll under ett liv — ska hamna i det biologiska intervallet 30 till 100:

```
tidsenhet    k_basal      omsättning
   sekund     3,4e0            0,0
     dygn     2,9e5            2,3
    månad     9,0e6           71
       år     1,1e8          857
```

`k_basal = 9,0e6` J per månad och kg⁰·⁷⁵ är Kleibers 3,4 W uttryckt i modellens enhet. Inget fittat tal.

Och eftersom omsättningen beror på massan som M⁻⁰·²⁵ **väljer tidsenheten kroppsmasseskalan**: månader ger djur på ett par kilo, dygn skulle ge milligram. De två är inte oberoende val.

Med `year_len = 12` möter en individ fyra vintrar. Säsongen är därmed ett verkligt selektionstryck för första gången — tidigare levde ett djur arton procent av ett år.

### Tre klasser av parametrar

Vid ett enhetsbyte kan inte allt multipliceras med samma faktor. Storheterna delar sig i tre klasser, och att blanda ihop dem har orsakat de flesta felen vi hittat.

**Klass A — fysikaliska takter.** Massa- och energiflöden per tid: `k_basal`, `eat_rate`, tillväxt- och gestationstakter, termoreglering, hjärnkostnad, näringstillförsel, florans upptag. Skalar med tidsenheten, men **delkostnaderna ska ställas mot den nya `k_basal`, inte skalas från sina gamla värden** — de gamla var kalibrerade mot varandra i en trasig skala.

**Klass B — livshistorietakter.** Skada, åldrande, hazard, utmattning, nedbrytning, mortalitet, karenstider. Deras naturliga tidsskala *är* livslängden, och eftersom `D_max = 1` är en dimensionslös livstidsbudget behåller de sina tal när enheten byts.

Det är just förhållandet mellan klass A och klass B som var trasigt: ämnesomsättningen var komprimerad nio gånger medan livslängden var komprimerad tvåhundratusen gånger. Att skala alla takter lika hade varit en nolloperation — det byter bara namn på tidsenheten.

**Klass C — dimensionslösa, per massa eller per energi.** Verkningsgrader, strukturandelar, energitätheter, `growth_E_per_kg`, temperaturer, rumsliga radier. Rörs inte.

---

## Fällor

Åtta av dem har kostat oss mätbar tid. De är värda att känna igen.

**En stock är inte en takt.** `E_cap_per_M` är en lagringskapacitet och skalar inte med dräneringarna. När dräneringarna växte men taket stod still räckte reserven inte en tick. Samma fel i `drain_rate_n`, som normerade en takt mot en stock — kvoten har då enheten 1/tid och ändrar värde vid enhetsbyte. Normera mot basalmetabolismen i stället; då blir måttet skalfritt.

**Snabb dynamik får inte integreras explicit vid långa tick.** Den termiska tidskonstanten är en halv tick, och explicit Euler sköt kroppstemperaturen till −180 °C. Lösningen är analytisk relaxation, exakt för konstant tillförsel och ovillkorligt stabil. Rörelsens newtonska integration har samma problem — relaxationen mot terminalhastighet är sekunder mot ett tick på femton timmar — och bör förenklas till kvasistatisk form.

**Diskretionära utgifter får inte ligga bland de obligatoriska.** Låg tillväxten i dräneringssumman täckte katabolismen även den, så organismen bröt ner sin egen kropp för att bygga sin egen kropp. Materialet kom tillbaka som massa, så nettot syntes knappt, men varje varv kostade katabolismens utbyte och lade på skada.

**Ett överskott måste ha någonstans att ta vägen.** En vuxen individ vid `M_target` kunde varken växa eller lagra, så allt över reservtaket exkreterades — medan varje svacka kostade massa. Ett tak uppåt utan golv nedåt ger en långsam nedgång oavsett hur god försörjningen är. Därför är reservkapaciteten evolverbar.

**Aptit får inte mätas mot en kapacitet som krymper med underskottet.** `hunger = (Ecap − E)/Ecap` med `Ecap ∝ M` betyder att en utmärglad organism inte kan registrera sig som hungrig. Aptiten mäts nu även mot förväntad massa för åldern, och katabolism sätter den till fullt: den som bryter ner sin egen vävnad för att betala underhållet är hungrig, oavsett vad ögonblicksbilden säger.

**Ett inflöde som repareras bort i samma tick syns inte alls.** `dD_age` tickade för varje individ, men medianskadan i en matt population mätte exakt noll vid varje mätpunkt och 97 procent av alla dödsfall var svält. Klockan fanns, den nollställdes bara varje tick. Felet låg i att slitaget byggdes ur skadetakten i stället för ur tiden, så degraderingen av reparationen var villkorad av att skada redan skett. Talet `wear_a0` var dessutom kvar från sekundskalan. Se `k_age1` nedan för samma sorts fel.

**Ett klass B-tal som är 1/tid² är inte skalfritt.** Klassindelningen säger att livshistorietakter behåller sina tal vid enhetsbyte, eftersom `D_max` är en dimensionslös livstidsbudget. Det gäller storheter med dimensionen 1/tid. `k_age1` multiplicerar **åldern** och har därför dimensionen 1/tid², vilket gör att en livstidsintegral av den skalar med kvadraten på tidsenheten. Den bör härledas ur en målsatt livslängd i stället för att bäras över.

**Ett handlingsantagande som håller vid korta tick kan bli falskt vid långa.** Betningen samplade en punkt, vilket var riktigt när organismen färdades fyra hundradels cell per tick och blev fel när den korsar två. Djuret gick förbi föda det inte åt.

---

## Att mäta

```
run_headless.py --stats              bestånd, massakvot, omsättning, näringstermer, takt
run_headless.py --pop-log --world-log   loggar för live_pop_plot och live_world_plot
measure_leak.py                      faunans massaflöden och energiledgern
```

`--stats` rapporterar näringens flödestakter och tidskonstant, inte bara totalerna. En stock som ser stabil ut över några hundra tick kan ha en tidskonstant på timmar.

Två diagnoser är värda att göra rutinmässigt när något beter sig oväntat. **Masslagret** — assimilerat mot netto ΔM — visar om organismen tappar massa trots positiv energibalans, vilket energiledgern inte kan se. Och **balansen över hela livslängden**, inte bara i början, eftersom lokal betning kan sänka intaget över tid utan att något är fel i mekaniken.

---

*Uppdateras när mekaniken ändras. Talen i dokumentet är de som gällde vid skrivandet; koden är alltid källan.*
