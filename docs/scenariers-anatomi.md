# Scenariers anatomi

*Augusti 2026. Vad ett scenario är, vilka sektioner det består av, och vad som
gör en körning ogiltig.*

---

## Vad ett scenario är

Ett scenario är **en körnings utgångsläge som data**, och det skrivs till
körningens katalog så att varje utfall bär sitt eget utgångsläge.

Före `scenario.py` var utgångsläget fjorton kommandoradsflaggor, varav sex bara
beskrev scenariot. Tre av dem — `nutrient_input`, `nutrient_init`,
`detritus_init` — skalar alltid tillsammans men räknades fram för hand vid varje
körning, och de hann gå isär två gånger. Det är den sortens fel formen finns för
att omöjliggöra.

### Den bärande regeln

> **Filen anger avsikter. Koden härleder tal.**

`bordighet: 4.0` i stället för tre näringskonstanter. `sadd: bordighet` i
stället för ett plantantal. `insatts_vid: jamvikt` i stället för ett gissat
tick — det felet gjorde både p87 och p97 ogiltiga, eftersom faunan mötte en
halvfärdig flora. `fartskala: 0.5` i stället för `drag_lin: 440`.

Varje härledning ligger i `scenario.py` som en `@property` med sin motivering.
Ett tal i filen som inte är en avsikt är ett tal som kommer att gå isär från sitt
syskon.

### Vad som inte hör hemma här

Scenariot beskriver **detta körningsfall**, inte modellen. Konstanter som gäller
all fysik och all biologi hör i `WorldParams`, `AgentParams` och `PopParams` med
sin härledning i kommentaren. Skillnaden är prövbar: *skulle två olika värden
kunna vara samtidigt riktiga i två körningar?* Bördighet ja, cellarea nej.

---

## Sektionerna

### `namn`

Bär in i loggarna och i körningskatalogen. Ska räcka för att peka ut filen.

### `varld` — vad världen är

Storlek, kadens, bördighet, position på planeten, terräng.

| fält | betydelse |
|---|---|
| `bredd`, `hojd` | i celler. Cellarean är 1, alltså 100 m²; se `docs/varldens-skala.md`. Hexgeometrin ger centrumavståndet 1,0746, så en värld på 256×256 är 2,8 × 2,4 km och inte kvadratisk |
| `dt` | ticklängden i månader. **Skriv alltid ut den**, se nedan |
| `bordighet` | multiplikator på näringsflödet. Skalar `nutrient_input`, `nutrient_init` och `detritus_init` tillsammans, eftersom jämvikten är linjär i flödet |
| `latitud`, `kontinentalitet` | världens position *på* planeten. Klimatet härleds ur de två i `klimat.py`. Latituden är en skalär och inte ett fält: en dalgång på en kilometer ligger på **en** breddgrad, och den rumsliga gradienten föll i 7013 av det skälet. Negativ latitud lägger världen på södra halvklotet |
| `terrang` | utelämnad ger en platt värld. Nycklarna är fälten i `terrain.TerrainParams`; **okända nycklar är fel, inte tystnad** |

Terrängens fält i tur och ordning: `seed` (terrängens eget frö, skilt från
körningens), `relief` (amplitudens multiplikator), `lambda_min` och
`lambda_max_frac` (spektrets band, kortaste våglängd i celler och längsta som
andel av världen), `beta` och `beta_kort` med `lambda_bryt` (den brutna
potenslagens lutningar och brytpunkt), `hurst`, `noise_sd`, `hav_andel` (havet
placeras på en **höjdkvantil**, inte på en nivå) och `hav_lutning`.

### `flora` — vad marken bär vid tick 0

`sadd: bordighet` sår tills markens fria näring är förbrukad, alltså exakt så
mycket vävnad som bördigheten bär. Ett tal i stället för strängen är ett måltotal
i kg. `plantmassa` är medelmassan per sådd planta; **antalet faller ut** ur
måltotalen delat med den, i stället för ur cellantalet.

Sådden hade ingen plats i filen alls trots att den avgör hur världen ser ut vid
tick 0, och regeln bytte gren beroende på om faunan var insatt än — vilket gav en
fjärdedels värld för samma fil, se 0130.

### `fauna` — vilka som släpps in, var och när

`antal`, `flackar`, `flackradie` sätter grundarnas rumsliga fördelning;
`grupp_avstand` och `grupp_spridning` deras fördelning i traitrymden, och kvoten
mellan dem avgör om det blir raser eller arter.

`insatts_vid: jamvikt` låter simuleringen själv upptäcka när floran nått
stationärt tillstånd. `start` sätter in vid tick 0.

Att sprida grundarna över hela världen gör grundarproblemet värre, inte bättre:
den effektiva populationsstorleken har mätts till fem, och ett djur som inte
hittar en partner bidrar med ingenting.

### `fysiologi` — vad som skalas

`fartskala` är en multiplikator på jämviktsfarten, implementerad via `drag_lin =
220 / s` eftersom den linjära dragtermen dominerar den kvadratiska fyra mot ett.

**`sociability` och `sociability_sd` gör något annat och allvarligare: de låser
en ärftlig egenskap.** En låst trait kan inte selekteras, och just den axeln är
den som skulle avgöra om flockningen är adaptiv. Använd bara när frågan är
"vad gör en flock" och inte "uppstår flockar".

---

## Vad som gör en körning ogiltig

**Inkörningen.** I en terrängvärld är den lång: floran sås till omkring en
miljon plantor och självgallrar under de första tusentals ticken, och
näringsstocken faller mot sin urlakade jämvikt under samma tid. **Vänta med
slutsatser till efter tick 15 000.** Varje körning som utvärderats hittills har
gått 3 000, och de säger något om riktning men ingenting om jämvikt.

**Ett frö.** Enskilda frön passerar eller fastnar i grundarflaskhalsen oberoende
av varandra. Populationstrender kräver svep; tre frön skiljer en halvering från
brus men inte tio procent från noll.

**Aggregat över sneda fördelningar.** Näringspoolens medelvärde låg åtta gånger
över medianen i ett mätfall. Percentiler, inte medelvärden.

**Kadensbyte utan omkalibrering.** Se nedan.

---

## Planerade uppdateringar

### `dt` ska stå i filen, alltid

`VarldSpec.dt` finns som fält men står inte i något scenario, eftersom förvalet
0,02 räckt. Det kommer inte att räcka: kadensarbetet siktar på ett dygnstick,
och då blir samma filnamn två olika världar.

**Ticklängden är den enda parameter som ändrar innebörden av varje annan.** Den
ska skrivas ut även när den är förvalet, av samma skäl som en storhet utan enhet
inte är en fysisk storhet.

### Ett block för kalibreringskonstanter

Fem kalibrerade tal har flyttats under en enda arbetssession — `C_sense_K`,
`B_sense_K`, `water_drag_depth_ref`, `graze_reach_cells`, `forage_path_rate` —
och varje gång var problemet att talet var kalibrerat mot något som ändrades.
Inget av dem kan sättas per scenario.

Det gör kalibreringssvep omöjliga att uttrycka som filer och tvingar fram
kodredigering för varje prov, vilket är precis det kommandoradsflaggorna gjorde
och som formen skulle avskaffa. Ett bördighetssvep behöver i dag tre filer eller
tre flaggor; det borde behöva en fil med tre värden.

Formen bör följa terrängens: ett block vars nycklar är fält i `AgentParams` eller
`WorldParams`, med **okända nycklar som fel och inte som tystnad**. Överstyrningen
ska dessutom hamna i den dumpade filen, så att en körning som avvek från förvalen
bär det i sitt eget utgångsläge.

### `fysiologi` delas

Att skala en fysikkonstant och att låsa en ärftlig egenskap är två olika ingrepp
och hör inte under samma rubrik. Den senare hör hemma under ett namn som säger
vad den kostar — en låst axel är en axel som inte kan svara på selektion, och det
ska inte gå att göra av misstag.

### Ett magert scenario

`bordighet: 4.0` ger sluten mark överallt: `bar = 20,1 %` mot `hav = 20,0 %`,
alltså är varje landcell täckt. Uppmätt är perceptets toppandel då 0,185 mot en
jämnfördelnings 0,167 — riktningen bär nästan ingen information, och en
rörelsekärna byggd på det blir diffusion och inte födosök.

Bördigheten har ingen härledning i någon fil. Den är ärvd från mindre världar,
och `TODO.md` noterar redan att den inte är skalfri. Ett `f6-256-mager` med lägre
bördighet är den billigaste kvarvarande prövningen av om perceptet kan bära
någon riktning alls, och den kräver ingen kodändring.

### Kommentaren blir en beskrivning

Rubrikkommentaren i `f6-256.yaml` beskriver skillnaden mot `f5-terrang`, alltså
en historia snarare än ett tillstånd. Historien hör i `TODO.md`. Filen behöver
överst **vad scenariot är till för** och **vad som gör en körning ogiltig** — och
den sista raden i dagens kommentar, *vänta med slutsatser till efter tick
15 000*, är redan den viktigaste i filen och ska stå först.

---

## Vad detta inte är

Det här är inte en förteckning över giltiga värden. `scenario.py` äger
härledningarna och avvisar okända fält; dokumentet förklarar **varför formen ser
ut som den gör**, så att nästa parameter hamnar på rätt sida av gränsen mellan
avsikt och tal.
