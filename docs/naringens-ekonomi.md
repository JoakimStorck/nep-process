# Designskiss — näringens ekonomi

*Juli 2026. Underlag för Steg 4:s kalibrering och för hydron i Steg 7. Status: förslag, inte beslut — uppdateras eller markeras som ersatt när stegen byggs.*

---

## Frågan som ledde hit

`PopParams.flora_init_mass_ratio = 10.0` sår flora tills massan når tio gånger faunans. Det såg ut som ett ekologiskt villkor. Vid granskning visade det sig vara något annat.

`WorldParams.nutrient_init = 0`. Världen startar utan fri näring, så den sådda biomassan **är** hela systemets näringsbudget. Extern tillförsel över en fyrahundrasekunderskörning är 0,13 % av stocken. Parametern satte alltså inte en kvot utan bärkraften — och gjorde det via faunans massa, vilket vänder kausaliteten: primärproduktionen härleddes ur konsumenterna.

Uppmätt, 6 000 tick, seed 1:

| ratio | flora vid start | vid slut | kvar | näringsstock |
|---|---|---|---|---|
| 5 | 37,3 kg | 36,8 kg | 98,6 % | 0,73 kg |
| 10 | 74,1 kg | 73,4 kg | 99,0 % | 1,30 kg |
| 20 | 149,1 kg | 147,6 kg | 99,0 % | 2,46 kg |

Systemet håller vad det fick. Det finns ingen oberoende jämvikt att söka mot.

## Vad verkligheten säger

Frågan om hur mycket näring en värld ska starta med har ett svar i jordens historia, och det svaret är användbart.

**Bergartsbildande näringsämnen fanns från början.** Fosfor utgör omkring en promille av jordskorpan, mest bundet i apatit. Kalium, kalcium, magnesium, järn och svavel likaså, direkt från ackretionen. De behövde inget liv för att finnas.

**Kväve är undantaget.** Jordens kväve sitter till största delen i atmosfären som N₂ med en trippelbindning som gör det nästan inert. Prebiotiskt fixerat kväve var ont om. Biologisk kvävefixering var en innovation, inte en förutsättning.

Distinktionen är alltså **förekomst mot tillgänglighet**. Fosfor fanns i överflöd men apatit är svårlöslig, och löst fosfat adsorberas dessutom på järnoxider.

**En stock hann byggas upp.** Före livet vittrade silikater under en koldioxidrik atmosfär och levererade lösta ämnen till havet i hundratals miljoner år utan att något tog upp dem. Havet ackumulerade. Sedan kom livet och drog ner stocken och började återcirkulera den snabbt.

Att så en stor stock är därför inte en genväg. Det är en beskrivning av utgångsläget.

## Det tal som betyder mest

I dagens hav och jordar är den *stående* mängden tillgänglig näring liten jämfört med *flödet* genom organismerna. Ytvattnets nitrat är ofta nära noll — planktonet stripper det — och produktionen bärs av snabb återmineralisering, inte av lagret. En fosforatom cirkulerar i storleksordningen hundratals gånger genom levande vävnad innan den begravs som sediment.

`nutrient_loss_frac = 0.10` säger att en atom klarar tio varv. Det är ett par storleksordningar från verkligheten, och det är modellens största enskilda felkalibrering på den abiotiska sidan.

Uppmätt vid 12 fauna, 6 000 tick:

```
loss_frac = 0,100    tillfört 0,0118 kg/h   förlorat 0,1244 kg/h   netto -0,1126 kg/h
loss_frac = 0,002    tillfört 0,0118 kg/h   förlorat 0,0028 kg/h   netto +0,0090 kg/h
```

Hela stocken cirkulerar ungefär ett varv per simulerad timme. Vid 0,002 överlever en atom femhundra varv, alltså en uppehållstid kring femhundra timmar. Vid 0,100 dräneras världen med tidskonstant kring elva timmar — precis den horisont långkörningar har.

Det ändrar kalibreringen från att fitta ett tal till att sätta två som var för sig betyder något.

## Fyra påståenden

**1. Källan är en takt, inte en stock.** Vi kan inte simulera geologisk tid, så berggrunden är i praktiken outtömlig. Då ska den inte modelleras som en pool utan som `nutrient_input` per cell — ett forcing-fält i manifestets mening, parametriskt och inte biologiskt. Celler med förhöjd takt *är* källor, vilket ger geografin en roll utan ny mekanism. Parallellen finns redan i vattenmodellen: `spring_input` är samma sak för vatten, och de två ska på sikt kunna sammanfalla.

**2. Stocken sås, och representerar prebiotisk ackumulation.** Världen startar i ett läge där processerna redan pågått en okänd tid. Budgeten sätts explicit, floran sås ur budgeten och faunan ur floran — kvoten 1:10 behålls men kausaliteten vänds. Kravet är att det sådda tillståndet ska vara ett tillstånd processerna faktiskt skulle producera, annars är det inte en uppvärmning utan ett artificiellt startvärde med kort halveringstid.

**3. Den interna cirkulationen är nästan förlustfri.** `nutrient_loss_frac` sänks till storleksordningen 10⁻³. Det är inte en justering för att få siffrorna att gå ihop utan en korrigering mot vad ekosystem faktiskt gör.

**4. Sänkan är rumslig, inte proportionell.** Den verkliga slutstationen är begravning som sediment på havsbotten. `nutrient_loss_frac` är alltså en platshållare för begravning, och när hydron finns bör den ersättas av transport till havsceller. Fram till dess ska den vara liten, eftersom den i dag representerar något som i verkligheten sker sällan.

## Hur det möter hydron

Hydron ställer ett krav som avgör representationen: **bara det som är löst följer med vattnet.**

Det talar emot att lägga stocken i den fria poolen. Sätter vi `nutrient_init` högt börjar den vandra utför så fort hydron finns, samlas i bassänger och lämnar budgeten vid havsranden — manifestet säger att celler under `sea_level` absorberar inflöde utan mottryck. Vi skulle ha kalibrerat mot en stock som sedan rinner bort.

Med källan som takt per cell uppstår i stället rätt struktur av sig själv när `elevation` finns: vittringstakten kan göras beroende av lutning och genomströmning, höglandet utarmas, flodslätten berikas. Nischerna manifestet siktar på — stränder, flodfåror, lågland, höjder — faller ut ur samma fysik i stället för att kodas.

**Den biotiska omfördelningen är motriktningen.** Vatten flyttar näring strikt utför. Det enda som flyttar den uppför är kedjan förtäring, upptag, tillväxt, förflyttning, död, nedbrytning: ett djur betar i låglandet, går uppåt, exkreterar och dör där. Efter hydron är det den enda mekanism som håller höglandet beboeligt.

Det ger ett designkrav bakåt på det som redan är byggt. **Lokaliteten är det som gör motriktningen verklig:** upptag ur egen cell, exkretion där organismen står, kadaver där den dör. Skulle något av det bli globalt eller instantant försvinner motgradienten och världen skiktas permanent. Manifestets krav på lokal upptäckt har alltså en andra motivering utöver prestanda och ontologi.

## Vad kalibreringen då blir

Med `loss_frac` på fysiskt rimlig nivå går kalibreringen från gissning till räkning. Uppmätt marginalkostnad per djur vid ungefär oförändrad flora, `loss_frac = 0,002`:

```
12 fauna, 74 kg flora   ->  förlust 0,0027 kg/h
60 fauna, 83 kg flora   ->  förlust 0,0260 kg/h
```

Alltså omkring 4,9e-04 kg/h per djur på marginalen, och svagt överlinjärt i antal. Med tillförseln 0,0118 kg/h bär världen ungefär **trettio djur**.

Täthetssvepet i Steg 4 sa att mötesfrekvensen kräver omkring sex agenter per 1000 celler, alltså ungefär hundra djur i en värld på 16 384 celler. Extrapolerat blir tillförseln som krävs omkring fyra till fem gånger dagens.

Jämför med samma räkning före korrigeringen av `loss_frac`, som gav åttio gånger. Skillnaden är att den återstående faktorn nu är ett ekologiskt påstående — den här världen bär trettio djur, vi vill ha hundra — och inte kompensation för ett läckage.

## Vad som inte ska göras nu

Ingen lutningsberoende vittring, ingen advektion av lösta ämnen med flödet, ingen havsrandregim. Allt det kräver `elevation`, som kommer i Steg 7. Tillförselfältet är uniformt med eventuella punktkällor tills terrängen finns.

Ingen hård kalibrering mot `nutrient_loss_frac`. Talet styr direkt hur mycket tillförsel som krävs, och det ska ersättas av en fysisk mekanism. Storleksordningen är rätt att sätta nu; exakt värde hör till hydron.

## Öppna frågor

- Ska `nutrient_input` per cell vara en egen array eller härledas ur ett framtida bergartsfält? Det senare är renare när erosionen kommer men kräver ett fält som inte har någon annan användning ännu.
- Hur ska punktkällor initieras innan terrängen finns? Slumpvis utplacerade, eller inte alls förrän `elevation` kan motivera var de sitter?
- Uppehållstiden femhundra varv är vald mot marin fosfor. Terrestra ekosystem varierar. Ska talet vara ett per värld eller kunna variera rumsligt?
