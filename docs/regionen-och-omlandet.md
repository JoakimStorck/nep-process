# Regionen och omlandet

*Augusti 2026. Fäster resonemanget om ett grovt lager runt den detaljerade
världen, innan det hinner glida.*

---

## Frågan som ledde hit

Torusen är inte en plats. När världen fick en position på en planet blev det
märkligt att den samtidigt wrappar in i sig själv: en dalgång på en kilometer
har inte en söm där öster möter väster.

Men en hård rand är också en fiktion, och en sämre. Vad *är* kanten? En vägg
som djur studsar mot finns inte i naturen. Det som finns är att landskapet
fortsätter.

**Det är den frågan det här dokumentet svarar på.** Inte "vad ska randen vara"
utan "vad ligger på andra sidan". Svaret är ett omland: ett grovt rutnät av
regioner utan inre upplösning men med aggregerade egenskaper, där den
detaljerade världen är en ruta bland många.

I litteraturen heter det **one-way nesting**, och klimat- och
hydrologimodeller har gjort det i decennier av exakt det skäl som gäller här:
man vill ha upplösning på ett ställe och sammanhang överallt, och man har inte
råd med båda.

## Vad en region är

Den detaljerade världen — det som i dag är hela modellen — blir **regionen**.
Den behåller allt: individer, evolution, hydrologi, näringskretslopp,
kadensklasser. Ingenting i den ändras av att den får grannar.

```
region          bredd x höjd         area
 128x128          1,4 x 1,2 km        2 km²
 512x512          5,5 x 4,8 km       26 km²
1024x1024        11,0 x  9,5 km      105 km²
2048x2048        22,0 x 19,1 km      419 km²
```

**Omlandet** är ett rutnät av regioner som saknar inre upplösning. Var och en
bär ett fåtal skalärer: höjd, latitud, nederbörd, temperatur, bördighet,
möjligen en faunatäthet och en genpoolssammanfattning. Ingen individ, ingen
cell, ingen evolution.

```
rutnät à 1024²      utsträckning      latitudspann   meridionell ΔT   poster
 16x16               176 x 152 km        1,4°           0,46 °C          256
 32x32               352 x 305 km        2,7°           0,91 °C        1 024
 64x64               704 x 610 km        5,5°           1,83 °C        4 096
100x100            1 100 x 953 km        8,6°           2,86 °C       10 000
256x256            2 817 x 2 440 km     22,0°           7,32 °C       65 536
```

Tiotusen regioner med tio fält vardera är åttahundra kilobyte. **Omlandet är
gratis.** Det som kostar är kopplingen, inte lagret.

## Vad det löser

### Hydrologin har inget uppströms, och det är ett verkligt fel

Regionen är i dag hela sitt eget avrinningsområde: allt vatten som finns kom
som regn inom rutan. För ett landskap på tio kilometer är det nästan alltid
osant. Riktiga floder kommer in från grannlandskapet och lämnar mot ett lägre,
och en region utan uppströms kan inte ha en flod som är större än sin egen
nederbörd.

Ett omland ger inflöde vid randceller från högre grannregioner och utflöde mot
lägre. Det är ett tal per randcell och gör dräneringsnätet fysiskt riktigt för
första gången.

Det är den starkaste posten i hela förslaget, och den billigaste.

### Latituden återuppstår, nu på rätt skala

`docs/varldens-skala.md` slog fast att latitudgradienten är död: jordens
meridionella gradient är 0,003 grader per kilometer, och även den största
körbara världen — 4096x4096 celler, 44x38 km — ger 0,11 grader mot modellens
trettio.

Räkningen ser annorlunda ut för ett omland. Tre grader kräver tusen kilometer,
och ett rutnät på hundra gånger hundra regioner är elvahundra. **Där blir
latitudgradienten meningsfull** — samma mekanism, rätt skala. Detsamma gäller
regnskugga, kontinentalitet som varierar med avståndet till havet, och
orografi i stor skala.

Latituden dog alltså inte som fysik. Den dog som *cellfält*, vilket var rätt,
och den kan komma tillbaka som *regionfält*, vilket också är rätt.

### Genflödet

Den effektiva populationsstorleken har mätts till fem. Ingen mekanism i
projektet skulle göra så mycket åt founder-flaskhalsen som ett omland som då
och då skickar in en invandrare.

Wrights tumregel — en migrant per generation räcker för att hindra att
delpopulationer driver isär — betyder att flödet får vara mycket glest och
ändå verka. Det kräver att grovregionerna bär en genpoolssammanfattning, medel
och varians per locus, och att immigration är ett sällsynt inflöde genom
randen.

### Randen blir en granne

Det som lämnar regionen tas emot av något, och det som kommer in kommer
någonstans ifrån. Det är skillnaden mellan att ta bort ett antagande och att
byta ut det mot ett sämre.

## Var faran ligger

### Två biologier

Om grovregionerna får flora som växer, konkurrerar och evolverar har projektet
plötsligt två ekologier med var sin kalibrering, och den grova blir aldrig
prövad. Det är samma fel som en gång gjorde primärproduktionen till ett
bakgrundsfält, fast utflyttat en nivå.

**Gränsen dras hårt: en grovregion bär produktivitet, inte population.** Den
har en bördighet, en fuktighet, en temperatur och möjligen en faunatäthet, men
ingen individ och ingen evolution utanför den detaljerade regionen.

### Tvåvägskoppling

Att låta regionen påverka omlandet tillbaka är där riktiga modeller får sina
artefakter, eftersom två upplösningar då ska hålla samma bevarandelag.
Kopplingen byggs strikt envägs och omprövas bara om en mätning kräver det.

### Massbalansen

Näringskretsloppet sluter sig i dag inom 1e-9 och prövas som hård invariant.
Med ett omland kan balansen bli sluten på **systemnivå** i stället, vilket är
starkare — men bara om grovregionerna har konton och inte bara egenskaper. Har
de bara egenskaper blir invarianten diagnostik igen, och det vore ett steg
bakåt.

Konton är alltså inte en förfining som kan skjutas upp obegränsat. De hör till
samma steg som det första verkliga flödet över randen.

### Skalfelen upprepas

Omlandet är ett nytt lager, och varje nytt lager är ett tillfälle att införa en
storhet utan enhet. En region är 105 km² och en cell 100 m²; kvoten är
1 048 576, och varje flöde över gränsen måste bära den. Det är samma klass av
fel som lapse rate hade, och det ska fångas av samma regel: en storhet som inte
går att uttrycka i modellens enheter är en kalibrerad fri parameter och ska
märkas som en.

## Ordningen

Nivåerna är oberoende och var och en betalar sig.

**1. Statiskt omland, envägs.** Ett rutnät av regioner med höjd, latitud,
nederbörd och temperatur, allt statiskt. Det levererar randvillkor till hydro:
inflöde vid randceller, dräneringsriktning ut. Ingen dynamik, inga konton.
Detta är det billiga steget och det som rättar hydrologins uppströms.

**2. Konton i omlandet** för vatten och näring, så att massbalansen sluter sig
på systemnivå och randflödena får en mottagare i ledgern.

**3. Migration.** Emigration ut, immigration in med genpool från
grannregionerna. Den ekologiska utdelningen.

**4. Tvåvägs** — bara om en mätning kräver det.

## Vad detta betyder för arbetet som pågår

**Torusen faller i nivå 1, men på rätt sätt.** Har regionen grannar ska den
inte wrappa in i sig själv. Slutsatsen är densamma som en hård rand skulle ha
gett, men med en mottagare på andra sidan i stället för en vägg. Fram till dess
behålls torusen: ett periodiskt randvillkor säger att landskapet fortsätter och
att det som lämnar liknar det som kommer in, vilket är ett ärligare provisorium
än en kant utan andra sida.

**Havets placering blir delvis ett provisorium.** Att lägga havet som basnivå
*inuti* regionen är rätt så länge regionen är hela världen. Med ett omland
ligger basnivån utanför, och regionens dränering ska gå ut genom randen i
stället för ner i ett eget hav. Havet byggs ändå nu — dräneringen behöver en
terminering i dag — men det ska byggas som **en form bland andra** och inte som
ett antagande inbakat i dräneringskoden. Skillnaden avgör om flytten senare
kostar en parameter eller en ombyggnad.

**Regionstorleken blir ett val med konsekvenser.** I dag är den en fråga om vad
maskinen orkar. Med ett omland blir den också en fråga om var gränsen mellan
upplöst och aggregerad ekologi ska gå — vilket är en modelleringsfråga och inte
en prestandafråga.
