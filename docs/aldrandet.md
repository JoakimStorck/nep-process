# Åldrandet: två flöden, en härledd klocka

*Konstruktionsskiss efter mätningen "150 månader i `f6-256`, och varför
skadesystemet aldrig biter". Ingen kod ändrad än. Underlag: `runs/p224-lang`
och läsning av skade- och reparationsvägen i `agent.py`.*

## Vad som är fel i dag

Mekanismen finns och gör vad den ska. Reparationsförmågan sjunker med slitaget
precis som avsett:

```
  ålder  2,9 mån  ->  W=0,34   reparationstak kvar  95,0 %   verkningsgrad  96,6 %
  ålder 10,5 mån  ->  W=1,26                        82,8 %                 88,2 %
  ålder 72,5 mån  ->  W=8,70                        27,1 %                 41,9 %
```

Ett djur på sex år har tappat tre fjärdedelar av sin reparationsförmåga.
Kurvan är rimlig. **Den spelar bara ingen roll**, eftersom reparationen har
sjuttiosju gångers marginal mot skadan — att förlora tre fjärdedelar lämnar
tjugo gångers marginal kvar.

Det verkliga felet ligger en nivå under:

```
  skadeinflöde                 0,0013 per månad
  reparationstak (locus)       0,10 – 1,50 per månad
  reparationens energikostnad  1,0 % av basalmetabolismen
  verklig proteinomsättning    15–25 % av BMR
```

**Systemets två halvor är kalibrerade mot olika fenomen.** `repair_capacity`
0,10–1,50 per månad är ett *omsättningstal* — en verklig kropp bygger om en
stor del av sitt protein varje månad. Skadeinflödet 0,0013 per månad är ett
*åldrandetal*. De ligger femhundra gånger isär, och därför kostar reparationen
en procent av basalen i stället för tjugo, och därför kan `D` aldrig
ackumuleras.

Utfallet, uppmätt över 150 månader: **svält 3 188 dödsfall, skada 4, hazard 0.**
`D` har medianen 0,0000 genom hela körningen.

## Konstruktionen

### Två flöden, inte ett

Biologin har två, och modellen har slagit ihop dem till ett `D`:

1. **Omsättningen.** Stort flöde, fullt reparerbart, och det är det som kostar
   en femtedel av BMR. Proteinet i en kropp byts ut i storleksordningen ett par
   procent per dygn.
2. **Den irreparabla resten.** Den lilla andel av flödet som inte går att laga
   — tvärbundna proteiner, mutationer, lipofuscin. Den ackumuleras, och den
   **är** åldrandet.

Att skilja dem åt löser tre saker samtidigt: underhållet får sin verkliga
kostnad, åldrandet får en långsam klocka ur ett snabbt flöde utan någon egen
konstant, och `repair_capacity` blir den axel den borde vara.

### Omsättningen ligger **inom** basalen, inte ovanpå

`k_basal` är Kleibers helkroppsmetabolism, och proteinomsättningen är en del av
den. Att dra omsättningen som en egen post ovanpå vore att ta betalt två
gånger — exakt felet 0212 rättade i termoregleringen, där kroppens eget
värmeöverskott inte fick räknas som en extra kostnad.

Basalen delas därför:

```
    out_basal = k_basal · M^0,75 · (1 − s_oms · (1 − u))
```

där `s_oms` är omsättningens andel av BMR och `u ∈ [0,1]` är hur stor del av
full omsättning individen faktiskt utför. Vid `u = 1` är basalen oförändrad mot
i dag. **Att snåla sänker basalen** — vilket är rätt biologi, sänkt
proteinomsättning är en verklig energisparstrategi — och priset är skada som
inte lagas. Det är *disposable soma*, och avvägningen blir verklig i båda
riktningarna utan att någon motkraft byggs.

### Flödets storlek härleds, och två ankare möts

Omsättningsflödet `Φ` i kroppar per månad följer ur energiandelen:

```
    Φ = s_oms · k_basal · M_våt^0,75 / (M · E_synt)
```

med `E_synt = E_labile · (1/anabolism_eff − 1) = 3,99e6 J/kg`, samma
syntesarbete som tillväxten och reparationen redan betalar sedan 0191.

```
   s_oms      Φ (kroppar/mån)    %/dygn      uppmätt FSR
    10 %           0,72            2,4        2–3 %/dygn
    15 %           1,08            3,6
    20 %           1,43            4,8
    25 %           1,79            6,0
```

**De två ankarna skär varandra vid omkring tio till tolv procent av BMR.**
Energiandelen och den uppmätta fraktionella syntestakten är oberoende
mätningar av samma sak, och de pekar på samma tal utan att något justeras.
Det är den sortens överbestämning en härledning ska ha.

### Allometrin faller ut gratis

Eftersom `Φ ∝ M^0,75 / M`:

```
  M= 0,011 ts  Φ= 2,79/mån
  M= 0,050 ts  Φ= 1,91
  M= 0,150 ts  Φ= 1,45
  M= 0,500 ts  Φ= 1,08
  M= 1,000 ts  Φ= 0,90
  exponent:    M^−0,250
```

Skadan ackumuleras alltså som `M^−0,25` och **livslängden som `M^0,25`** — utan
att någon exponent skrivs in någonstans. Det är fria-radikal-teorins kärna:
klockan går i takt med ämnesomsättningen per kilo.

**Därmed utgår rättelse 3 ur `docs/revision-storleksskalningen.md` som egen
patch.** Den var "`k_age1 ∝ M^−0,25`", alltså allometrin pådyvlad en konstant.
Här följer den i stället av mekanismen, vilket är vad grundprincip 1 begär.

Och eftersom flödet följer **faktisk** metabol effekt och inte bara basal
betyder det att **arbete åldrar**: ett djur som springer, fryser eller är
dräktigt slits fortare. Det är rätt biologi och en avvägning till.

### Den irreparabla andelen är den enda kalibrerade konstanten

En andel `f_irr` av omsättningsflödet kan inte lagas och lagras som `D`. Den
ankras mot maxlivslängd — en växtätare på två kilo blir som mest tio till tolv
år:

```
  D=0,5 vid 150 mån för referenskroppen  ->  f_irr = 0,31 % av omsättningen
  D=0,5 vid 120 mån                      ->  f_irr = 0,39 %
  D=1,0 vid 150 mån                      ->  f_irr = 0,62 %
```

**Markeras som kalibrerad**, med ankaret utskrivet: den är satt mot en
uppmätt maxlivslängd och inte mot ett önskat populationsutfall.

Storleksordningen är rimlig mot verkligheten — andelen oxiderat, icke
funktionellt protein i åldrad vävnad ligger på några procent — men det är ett
stödargument och inte härledningen.

### Åldrandet dödar inte. Det gör djuret sämre.

I naturen dör vilda djur nästan aldrig av ålderdom. De dör av svält eller
predation, **för att åldrandet gjorde dem sämre på att undvika det**.

Modellen har redan den kanalen: 3 188 av 3 192 dödsfall är svält. Vi behöver
alltså ingen ny dödsväg — vi behöver att `D` försämrar det som håller djuret
vid liv, så att det dör i den kanal som redan finns, bara tidigare. Då blir
Gompertz-kurvan ett **utfall** i stället för en inmatad formel.

**Första och enda konsekvensen i det här steget: rörelsehastigheten.** Ett
åldrat djur går långsammare, och i modellen betyder det direkt sämre
födosök — sedan 0223 sätter samma kroppsstorhet både marschfarten och
födosökets bansträcka, och därmed betesytan. En enda koppling ger alltså hela
kedjan från skada till svält:

```
    D upp  ->  fart ned  ->  bansträcka ned  ->  betesyta ned  ->  svält
```

Hooken finns redan men är kopplad till fel tillstånd: `Body.move_factor()`
läser `weakness()`, som mäter massa mot `M_crit`, alltså avmagring — inte
skada. Den ska läsa båda.

Fler konsekvenser — matsmältning, värmebalans, sinnen, reproduktion — är
biologiskt riktiga och ska komma, men **en i taget och med mätning emellan**,
annars går de inte att skilja åt.

### Vad som utgår

`dD_age = age_rate · ålder` är en **andra, parallell åldrandeklocka**. Kommer
skadan ur ämnesomsättningen dubblerar den termen samma fenomen, och åldrandet
får två ägare. Den utgår, tillsammans med `k_age0`, `k_age1` och `k_ageD`.

`death_h_age` står redan på noll och rörs inte: det ska inte finnas någon
programmerad åldersdöd.

`W` och dess nedgångskurva **behålls oförändrade**. De var aldrig felet — de
blir verksamma av sig själva så snart inflödet och taket ligger på samma skala.

## Ordning

En ändring per körning, med mätning emellan.

1. **Instrumentering, bitidentisk:** skriv `D`, `W`, omsättningens andel av
   basalen och skadeinflödets termer per massakvintil till pop-loggen. Utan den
   går inte de följande stegen att bedöma.
2. **De två flödena**, med `s_oms`, `Φ` och `f_irr` enligt ovan, och `dD_age`
   borttagen. Ingen konsekvens av `D` ännu — bara att `D` faktiskt ackumuleras
   och att omsättningen får sin andel av basalen. Mätning: ackumuleras `D` på
   rätt tidsskala, och vad händer med livslängdsfördelningen?
3. **`D` sänker farten**, via `move_factor()` som läser både avmagring och
   skada. Mätning: dör djuren tidigare, och dör de av svält?
4. Först därefter: flera frön, och frågan om `M_target` hittar ett optimum när
   storleken äntligen har både sin kostnad och sin vinst.

## Det som inte går att veta i förväg

Medianlivslängden är 2,86 månader och p90 är 10,5. Klockan ovan är satt för
tio till tolv år. **Det är fullt möjligt att åldrandet fortfarande inte hinner
bita** — de flesta djur i modellen dör som ungar, och det gör verkliga djur
också. Då är det inte ett fel: det betyder att åldrandet är rätt modellerat men
ekologiskt underordnat, och att storleksaxelns avvägning måste sökas någon
annanstans. Mätningen i steg 2 ska kunna skilja de två fallen åt, och den ska
skriva in vilket det blev.
