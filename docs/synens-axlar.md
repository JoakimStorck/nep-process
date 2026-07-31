# Designskiss — synens axlar

*Juli 2026. Underlag för Steg 6a i TODO.md. Status: förslag, inte beslut — uppdateras eller markeras som ersatt när Steg 6a byggs.*

---

## Varför det här dokumentet finns

Manifestet anger emergent beteende som det primära evolutionära målet, och sensing är den kapacitet som mest direkt avgör vilket beteende som är möjligt. En organism kan bara agera på det den kan upptäcka.

I dag är synen evolverbar men i fyra diskreta steg längs en enda sammanslagen axel. Det räcker för att visa att mekanismen fungerar, men inte för att en meningsfull strategirymd ska uppstå. Den här skissen föreslår vilka axlar synen borde ha, vad var och en ska kosta, och vilka avvägningar som då blir möjliga.

## När det ska byggas

**Efter näringskretsloppet, inte före.**

Synens värde är en funktion av resursernas fläckvishet. I en värld där maten är jämnt utspridd och obegränsad hittar en långsynt organism ingenting som en närsynt inte också hittar — den betalar bara mer. Selektionen skulle då driva akuiteten mot minimum, vilket är ett korrekt svar på en felställd fråga.

Näringskretsloppet är det som skapar fläckvishet: konkurrens om cellnäring ger mättnadsgradienter, dödlighet ger luckor, och lokal utarmning ger driv att söka längre bort. Först i den världen betalar sig sikt.

Däremot kräver arbetet inte att hela faunamigreringen är gjord. Det behöver att `sense_radius` och `sense_rate` får läsare och kostnader — inte att `Body`:s samtliga fält har bytt ägare. Det motiverar en uppdelning: **Steg 6a — kapacitetsläsare och kostnader för sensing**, som kan ligga direkt efter delmängdsmaskineriet, och **Steg 6b — resten av fauna store-first**.

## Vad som redan är rättat

**Perceptionsfältet visar skottmassa.** `flora_cell_mass` summerade hela växtmassan, alltså även rötter, som varken kan ses eller ätas. Efter 0071 — betningshorisonten — innebar det att djuret navigerade mot en signal som överdrev måltiden två till tre gånger vid den uppmätta rotandelen 0,55–0,70. Rättat; se `docs/statusanalys-vaxtcykeln.md` för sammanhanget.

## Nuläget

`_T_SENSE` (locus 17) → `pheno.sense_strength` → `_sense_level()` ger fyra nivåer. Varje nivå sätter tre parametrar samtidigt på agentens egen AP-kopia:

```
nivå 0   12 strålar   räckvidd  7.0   brus 0.060   kostnad 0
nivå 1   16 strålar   räckvidd  8.0   brus 0.055   sense_cost_L1
nivå 2   24 strålar   räckvidd 10.0   brus 0.050   sense_cost_L2
nivå 3   32 strålar   räckvidd 12.0   brus 0.045   sense_cost_L3
```

Det som fungerar: avvägningen finns på riktigt, kostnaden är individuell, och färre strålar kostar faktiskt mindre CPU — manifestets symmetri mellan biologisk och beräkningsmässig kostnad håller redan här.

Det som saknas:

- **Axlarna är sammanslagna.** Räckvidd, vinkelupplösning och brus rör sig alltid tillsammans. Det gör att de tre inte kan handlas mot varandra, vilket är just vad som skulle skapa skilda strategier.
- **Frekvensen är inte genetisk.** `sense_idle_steps = 10` är samma för alla.
- **Fyra punkter är en trubbig fenotyprymd.** Selektion behöver gradienter att klättra i.
- **Kostnaden är platt per nivå.** Ingen strukturkostnad, alltså inget tryck mot enkelhet hos en organism som bär hög akuitet men sällan använder den.
- **`sense_radius` och `sense_rate` i store skrivs men läses inte.** Samma glapp som resten av kapacitetsmodellen.

---

## Föreslagna axlar

Fyra kontinuerliga loci i stället för ett. Det är en expansion av genomet, vilket manifestet varnar för — men det är den expansion som direkt tjänar det primära evolutionära målet, och den ersätter en axel som redan finns.

### 1. Räckvidd — `sense_radius`

Topologiskt cellavstånd, kontinuerligt. Avgör hur många celler som läses: `cells_within(r)`.

Det viktiga är att kostnaden faller ut ur geometrin utan att uppfinnas. På hex ger radie 1, 2, 3 … cellantalen 7, 19, 37 — alltså ungefär `3r²`. Att dubbla sikten fyrdubblar antalet lästa celler, i energi såväl som i CPU. Det är en stark och naturlig begränsning som inte behöver kalibreras fram.

### 2. Frekvens — `sense_rate`

Andel tick då sensing aktiveras, kontinuerlig i (0, 1].

Det här är den axel som saknas helt i dag, och den som mest direkt skapar skilda strategier. En organism som ser långt men sällan är en annan livsform än en som ser nära men ständigt. Manifestet anger också att en organism som inte sensar en viss tick inte ska beröra sensing-koden alls — frekvensen är alltså både en biologisk och en beräkningsmässig axel.

### 3. Vinkelupplösning — `sense_acuity`

Hur fint riktning kan urskiljas. Låg akuitet: organismen vet att det finns föda i närheten, inte varifrån. Hög akuitet: den kan skilja intilliggande riktningar åt.

Här ligger den intressanta designfrågan, som får ett eget avsnitt nedan.

### 4. Synfältets form — `sense_fov`

Från panoramisk till framåtriktad. Parametern finns redan som `ray_eccentricity = 0.7` men är fix.

Det här är den axel som mest direkt kopplar till verklig biologi: bytesdjur har laterala ögon och brett synfält för att upptäcka hot, rovdjur har framåtriktade för att fixera byte. Att låta den evolvera fritt bör ge samma differentiering utan att den kodas uppifrån — och den är billig, eftersom den omfördelar upplösning i stället för att lägga till den.

### Vad som inte bör bli en egen axel ännu

**Brus.** Det kan härledas ur akuitet i stället för att bära ett eget locus. Ett skarpare öga ger en renare signal; det behöver inte vara ett oberoende val.

**Diskriminering** — att skilja flora från fauna från kadaver.

> **Beslut, juli 2026.** Principen är fastlagd även om byggandet skjuts upp:
> **sensorn detekterar, organismen tolkar.** I dag levererar sensingen tre
> färdigsorterade kanaler — flora, kadaver, artfrände — vilket är en sensor som
> redan har dömt. Ingen organism betalar för att kunna skilja dem åt, och ingen
> kan förlora förmågan. Ett djur som ser en rörelse i gräset vet i verkligheten
> inte om det är en partner eller ett hot förrän signalen tolkats, och den
> tolkningen kan slå fel.
>
> Diskriminering ska därför vara en kapacitet med kostnad och med brus som gör
> misstag möjliga. Först då blir flykt, jakt och parning evolverade beteenden i
> stället för kanaler. Det är en förutsättning för att risk ska kunna vägas mot
> föda.
>
> Arbetet är stort och prioriteras inte nu. Men premissen är beslutad, så att
> Steg 6a inte behöver ta om frågan.

---

## Kostnadsmodellen

Manifestet anger tre sorters kostnad, och sensing bör bära alla tre.

**Strukturkostnad.** Att bära kapaciteten kostar även när den inte används. Bör skala med `sense_radius² × sense_acuity` — sensorapparatens storlek. Det är den kostnad som saknas helt i dag, och den som skapar tryck mot enkelhet hos en organism som bär hög akuitet men sällan använder den.

**Underhållskostnad.** Per tick, proportionell mot strukturen. Kan i praktiken slås ihop med strukturkostnaden i en första implementation.

**Aktiveringskostnad.** Per sensing-tillfälle, proportionell mot antalet faktiskt lästa celler gånger akuiteten. Förväntad kostnad per tick blir då `sense_rate × cells_within(sense_radius) × f(acuity)`.

Poängen med den uppdelningen är att den skapar en verklig avvägning mellan räckvidd och frekvens. Att se långt är dyrt per tillfälle; att se ofta är dyrt över tid. En organism som maximerar båda ska inte kunna betala för det.

---

## Representationsfrågan

Det här är den svåraste delen och den som avgör om axlarna går att implementera alls.

Om akuiteten styr hur många riktningssektorer organismen upplöser, så varierar antalet sensoringångar med akuiteten. Men MLP:ns indimension är fix, och en förändrad indimension bryter ärftligheten: avkomman kan inte ärva förälderns vikter om nätverket har en annan form.

**Förslag: fix inbreddd, variabel informationsmängd.**

Sensing producerar alltid samma antal riktningssektorer — förslagsvis sex, matchande hexgeometrin, eller åtta om vi vill ha finare upplösning än grannantalet. Akuiteten styr inte antalet sektorer utan hur mycket detalj som bevaras mellan dem:

- **Låg akuitet:** samtliga sektorer rapporterar samma värde — neighbourhood-medelvärdet. Organismen vet att det finns föda i närheten men inte varifrån.
- **Medel:** sektorvärdena blandas med varandra proportionellt mot `1 - acuity`. En gradvis oskärpa.
- **Hög akuitet:** varje sektor rapporterar sitt eget värde, oblandat.

Det ger en kontinuerlig akuitetsaxel med fix nätverksform, och oskärpan är en enkel viktad blandning som är billig att beräkna. Den generaliserar dessutom direkt till hex, eftersom sektorerna kan definieras topologiskt via grannmatrisen i stället för via vinklar i planet.

Samma mönster löser synfältsaxeln: `sense_fov` omfördelar sektorernas vikter framåt eller jämnt, utan att ändra antalet.

**Konsekvens:** strålarna kan avvecklas helt. Sensing blir en aggregering över `cells_within(sense_radius)` grupperad i sektorer via grannmatrisen — geometriagnostisk, vektoriserbar, och utan continuous-space-konstruktioner. Det är också vad manifestet beskriver.

---

## Strategirymden som bör uppstå

Om axlarna och kostnaderna är rätt satta ska följande differentiera sig utan att kodas:

| Strategi | radius | rate | acuity | fov |
|---|---|---|---|---|
| Betare i tät vegetation | låg | hög | låg | bred |
| Spanare i gles miljö | hög | låg | medel | bred |
| Sittande rovdjur | medel | låg | hög | smal |
| Aktiv jägare | hög | hög | hög | smal |
| Sessil eller nästan | ~0 | ~0 | — | — |

Den sista raden är den viktigaste testet: en organism som inte rör sig ska kunna evolvera bort synen helt och därmed sluta kosta något — både i energi och i CPU. Det är manifestets princip 3 gjord mätbar.

---

## Öppna frågor

**Hur många loci har vi råd med?** Genomet är redan 32 loci mot manifestets avsedda 8–16. Fyra sensingaxlar ersätter en befintlig, alltså netto +3. Det bör vägas mot att skära någon annanstans.

**Ska akuitet och brus verkligen kopplas?** Att härleda brus ur akuitet är enkelt, men det utesluter en organism som ser grovt men pålitligt. Oklart om den nischen är intressant.

**Vad är sektorupplösningen?** Sex matchar hexgrannarna och är billigast. Åtta eller tolv ger finare riktning men fler MLP-ingångar och därmed större nätverk för alla, även de närsynta.

**Hur mäter vi att det fungerar?** Förslag: kör identiska seeds med och utan kostnadsmodellen och jämför spridningen i `sense_radius` över tid. Utan kostnad ska den drifta neutralt; med kostnad ska den differentiera mot nisch. Om den inte gör det är antingen kostnaderna felskalade eller miljön för homogen — och det senare är i så fall ett svar om näringskretsloppet, inte om synen.
