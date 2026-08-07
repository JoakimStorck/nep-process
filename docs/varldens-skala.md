# Världens skala

*Augusti 2026. Fastställer modellens längdskala och räknar om det som därmed
inte längre stämmer.*

---

## Beslutet

**En cell har arean 100 m². Längdenheten är därmed 10 meter, och avståndet
mellan två cellcentrum 10,746 meter.**

Höjd, vattendjup och vågrätt avstånd delar den enheten sedan 7011. Beslutet ger
den ett namn och gör varje tal i modellen läsbart som en fysisk storhet.

Arean är det runda talet, inte centrumavståndet. Skälet är att arean är det
ekologin faktiskt räknar med — biomassa per cell, näring per cell, ljus per
cell — medan centrumavståndet är en härledd konsekvens av hexgeometrin,
`d = sqrt(2A/sqrt(3))`.

## Varför tio meter

Tre oberoende mått på cellstorleken, från `f4-start20` vid tick 1 200.

| mått | cellsida som gör det realistiskt |
|---|---|
| florans biomassa per cell, 13,5 kg | 2,6 – 26 m |
| faunans täthet, 28 djur på 16 384 celler | 4,1 – 18 m |
| faunans fart, 24 cellbredder per månad | 1 250 – 6 250 m |

De två första kommer från skilda delar av modellen — den ena ur
näringsekonomin, den andra ur faunaekologin — och de överlappar mellan 4 och
18 meter. **Farten är den ensamma avvikaren.**

Vid 10 meter blir utfallet:

```
flora         135 g/m²        gräsmark ligger på 100–400, skog över 2 000
faunatäthet   17 djur/km²     en 2 kg herbivor ligger på 10–100
faunafart     0,24 km/månad   ett 2 kg däggdjur rör sig 30–150
```

Två av tre stämmer. Faunans fart är omkring **tvåhundra gånger för låg**.

## Vad världen är

```
värld         celler        area        sida            hektar
  64x128        8 192      0,82 km²    0,7 x 1,2 km         82
 512x512      262 144      26,2 km²    5,5 x 4,8 km      2 621
1024x1024   1 048 576     104,9 km²   11,0 x 9,5 km     10 486
2048x2048   4 194 304     419,4 km²   22,0 x 19,1 km    41 943
4096x4096  16 777 216   1 677,7 km²   44,0 x 38,1 km   167 772
```

Även den största körbara världen är fyrtio kilometer tvärs över. **Det är ett
landskap, inte en kontinent**, och det gäller oavsett hur mycket hårdvara vi
kastar på problemet: en kontinent skulle kräva tio miljoner celler per sida.

---

## Vad som därmed inte stämmer

### Latituden är död

Jordens meridionella temperaturgradient är ungefär trettio grader över nittio
breddgrader, alltså 0,003 grader per kilometer. Över modellens
nord-sydliga utsträckning ger det:

```
 128 rader =  1,2 km  ->  0,004 °C
 512 rader =  4,8 km  ->  0,014 °C
1024 rader =  9,5 km  ->  0,029 °C
2048 rader = 19,1 km  ->  0,057 °C
4096 rader = 38,1 km  ->  0,114 °C
```

Vid den största världen är latitudens bidrag en tiondels grad. Modellen har i
dag trettio grader mellan pol och ekvator — **tvåhundrafemtio gånger för
mycket vid 4096 rader, och tiotusen gånger för mycket vid dagens 128.**

Det slår ut tre saker på en gång:

**Klimatbanden.** `T_band` som funktion av latitud har ingen fysisk grund. Med
en enda klimatzon blir temperaturen en profil i tiden och inte i rummet, vilket
kollapsar `T_band` från `n_bands` till en skalär.

**Polarhavet.** Det finns ingen pol. Havet är i dag placerat där latituden gör
klimatet obeboeligt, och det argumentet finns inte kvar.

**Kontinentallutningen.** Den stiger från kust mot inland för att organisera
dräneringen mot två hav. Utan hav har den inget att dränera mot.

### Årstiden överlever oförändrad

**Årstidsvariationen beror inte alls på skalan.** En dal på fyrtio kilometer
har fullständigt normala årstider — det är samma sol och samma axellutning.
Det som försvinner är att årstidens *amplitud* varierar med latitud; den blir
en enda siffra för hela världen.

Det är en förenkling och inte en förlust. Modellens tidsmässiga klimatvariation
är intakt; det är bara den rumsliga som var påhittad.

### Två rumsliga gradienter överlever, och de är starkare än latituden var

**Höjdgradienten.** Vid 6,5 grader per kilometer, som är jordens faktiska
lapse rate:

```
relief   40 m  ->  0,26 °C     dagens fyra längdenheter
relief  100 m  ->  0,65 °C
relief  300 m  ->  1,95 °C
relief  500 m  ->  3,25 °C
relief 1000 m  ->  6,50 °C
```

Dagens relief på fyrtio meter ger en fjärdedels grad, alltså nästan ingenting.
Men ett kuperat landskap på tio till fyrtio kilometer kan mycket väl ha
trehundra till femhundra meters relief — det är en genomsnittlig lutning på
någon procent, alltså mjukare än dagens terräng. **Med realistisk relief blir
höjdgradienten två till tre grader**, vilket är tjugo gånger mer än latituden
någonsin kunde ge vid den här storleken.

**Kalluftsdränering.** Kall luft är tyngre och rinner nedför. I ett kuperat
landskap samlas den i sänkorna, och inversioner på fem till tio grader mellan
dalbotten och sluttning är vanliga klara nätter. Det är den mekanism som lägger
frostlänta partier i dalarna och som gör att en dalbotten kan vara kallare än
kullen ovanför trots att den ligger lägre — alltså **motsatt tecken mot lapse
rate**, och därför inte något lapse rate kan ersätta.

Den verkar på hundratals meter, alltså på precis den skala modellen har. Den är
sannolikt den starkaste rumsliga temperatursignalen i en värld av den här
storleken.

En tredje kandidat är **sluttningsriktning** — nord- mot sydsluttning skiljer
flera grader i strålningsbalans. Den kräver en solvinkel och en normalvektor
per cell, alltså mer maskineri, men den är verklig på samma skala.

### Faunans fart

Tvåhundra gånger för låg för en tvåkilos organism. Två utvägar, och de utesluter
varandra:

**Djuren är mycket mindre än två kilo.** Vid tio meters celler är 0,24 km per
månad rimligt för en stor skalbagge eller en snigel. Men modellens fauna är
jämnvarm med termoreglering, och en jämnvarm organism på några gram är
fysiologiskt omöjlig — värmeförlusten skalar med ytan och ingen kropp så liten
kan hålla trettiosju grader.

**Farten är felkalibrerad.** Den sattes mot vad som ser levande ut i viewern,
inte mot något fysiskt, och den är den enda parameter som kopplar faunans kropp
till världens skala. Att höja den tvåhundra gånger går inte — vid dt = 0,02
månader skulle djuret röra sig hundra cellbredder per tick och all sensing bli
meningslös. Taket är omkring en cellbredd per tick, alltså ungefär femtio
gånger dagens värde, och även det är tio gånger under det realistiska.

**Det är alltså inte löst av att skalan fastställs.** Motsättningen är verklig:
ett däggdjur som rör sig realistiskt över ett landskap av den här upplösningen
korsar flera celler per tidssteg, och då kan modellen inte längre beskriva var
det befinner sig mellan besluten. Antingen kortas tidssteget, eller så accepteras
att faunan rör sig långsammare än sin kropp motiverar, eller så byts faunans
kroppsstorlek.

Det är den viktigaste öppna frågan i modellen efter det här dokumentet.

### Havet överlever — det definieras av basnivån, inte av polen

Ett första utkast drog slutsatsen att havet måste bort med latituden, och att
näringen därmed förlorar sin sänka. **Den slutsatsen var fel.**

Det som gör ett hav till ett hav är inte att det ligger vid polen utan att det
ligger vid **basnivån**: allt annat dräneras till det, och det dräneras
ingenstans. Det är en hydrologisk definition och inte en geografisk, och den är
skalfri — den fungerar lika bra i en dalgång på fyrtio kilometer som på en
planet. Kaspiska havet är ett hav i exakt den meningen.

Koden gör redan rätt. `_ocean_mask` från 7012 tar den största sammanhängande
mängden under havsnivån, alltså efter höjd och inte efter latitud. Det är bara
*placeringen* som var latitudberoende, genom polarsänkans form.

Placeras havet i stället som en bred bassäng gör den formen två jobb samtidigt:
den är havet, och dess mjuka kant är den regionala lutningen mot basnivån.
Kontinentallutningen behövs då inte som egen form — den generaliseras från "mot
polerna" till "mot basnivån", vilket är vad den alltid egentligen var.

Uppmätt vid 64x128, ett hav placerat som bassäng och ingen latitud alls:

```
grundhöjd  radie  djup   hav     sjö    största sjö  maxflod  relief  strandade
   1,0       40    4,0   27,3 %  6,71 %     183       2 012   47 m       0
   1,5       45    5,0   30,9 %  2,49 %     106       1 283   57 m       0
   2,0       50    6,0   35,3 %  1,35 %      64         936   67 m       0
```

Noll strandade celler i samtliga fall — all dränering terminerar, och
näringssänkan från 7007 fungerar oförändrat. Grundhöjden måste överstiga
bassängens bidrag på avstånd, annars sänker den mjuka kanten hela världen: vid
grundhöjd noll och radie 40 blev 76 procent av världen hav.

Kvar står ett val som inte är tvingande men värt att göra medvetet. Ett hav vid
basnivå i en sluten värld är endorheiskt, och i verkligheten gör en avloppslös
bassäng sig av med material genom att begrava det i sedimentet snarare än att
skicka det någon annanstans. Att låta näringen lämna modellen vid havet är
alltså rätt mekanism med fel namn: det är sedimentation, inte export.

---

## Vad detta inte är

Det är **ingen katastrof och ingen ny riktning.** Modellens ekologi, fysiologi,
näringsekonomi och hydrologi är oförändrade — de var kalibrerade mot varandra
och är det fortfarande. Det som ändras är vad världen *föreställer*.

Manifestet kräver en konsekvent fysikmotor. En sådan går inte att ha med två
längdskalor som skiljer tvåhundra gånger, och det var bara en tidsfråga innan
motsättningen blev tvingande. Att den blev det nu, när höjden fick en enhet, är
en följd av att geometrin och fysiken började hänga ihop — alltså av att
projektet gick åt rätt håll.

## Ordningen härnäst

1. **Klimatet blir en tidsprofil utan rumslig latitud.** `T_band` kollapsar till
   en skalär plus höjdmodifieraren. Billigare än i dag.
2. **Lapse rate sätts till 6,5 grader per kilometer**, alltså 0,065 per
   längdenhet — och blir därmed härledd ur fysik i stället för vald.
   Höjdgradienten blir försumbar tills reliefen växer, vilket är korrekt.
3. **Reliefen skalas upp** mot vad ett kuperat landskap av den storleken
   faktiskt har, tre- till femhundra meter.
4. **Havet placeras efter höjd i stället för efter latitud** — en bred bassäng
   som både är basnivån och den regionala lutningen mot den. Ingen ny mekanism
   krävs; `_ocean_mask` klassar redan efter höjd. Näringens sänka döps om från
   export till sedimentation.
5. **Faunans fart** utreds separat. Den är inte löst.

Ingen av de fem är stor i sig. Tillsammans är de den anpassning som gör att
fysiken och geometrin faktiskt går ihop.
