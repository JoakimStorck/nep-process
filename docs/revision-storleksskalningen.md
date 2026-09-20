# Revision: faunans storleksskalning

*Granskning efter 0222, när kroppsmassan drevs mot golvet i `f6-256`. Ingen
kodändring — en rad per massberoende term, med modellens exponent mot
biologins. Underlag: `runs/p219` och `runs/p222`, plus läsning av `agent.py`
och `phenotype.py`.*

## Frågan

Efter 0222 föll `M_target` från 1,99 till 0,53 och `M_repro_min` från 0,56 till
0,084, och populationens nedre tiondel hamnade på `M_birth_min = 0,02 kg`.
Frågan är **inte** hur vi håller djuren stora. Om litenhet är mest
konkurrenskraftigt ska evolutionen gå dit, och en motkraft som hindrar den vore
precis det grundprincip 1 förbjuder.

Frågan är om gradienten är verklig. Ett djur i modellen är en samling termer
som var och en beror på massan; driver några av dem åt fel håll är den uppmätta
selektionen delvis modellens egen konstruktion och inte världens.

**Svaret är att den delvis är det.** Mönstret är dessutom systematiskt:

> Varje term som **härletts** skalar rätt. Varje term som är en **konstant
> kalibrerad vid en kroppsstorlek** är massfri — och samtliga sådana gynnar
> det lilla djuret.

Det är inte en slump. En konstant sätts mot den kropp som råkade vara medianen
när den sattes, och den kroppen var alltid ungefär två kilo.

## Termerna

Exponenten avser massberoendet `∝ M^x` för den **våta magra massan**, som är
vad fysiken läser.

### Rätt skalade — härledda

| term | modellen | biologin | källa i koden |
|---|---|---|---|
| basalmetabolism | M^0,75 | M^0,75 (Kleiber) | `out_basal = dt·metab·M_eff^0,75·k_basal` |
| värmeledning | M^0,667 | M^0,667 (yta) | `thermo_mass_exp = 2/3` |
| transportkostnad per meter | M^−0,316 | M^−0,316 (Taylor 1982) | `cot_mass_exp` |
| marschfart | M^0,2 | M^0,17–0,24 | `v_mass_exp` (0219) |
| kroppens linjära skala | M^⅓ | M^⅓ | `body_depth` |
| reservtak | M^1 | M^1 | `reserve_cap` i J/kg |
| fastetid (reserv/basal) | M^0,25 | M^0,25 | följer av de två ovan |
| uppehållstid i tarmen | M^0,25 | M^0,25 (Jarman–Bell) | `retention_time_h` |
| mjölkens tak | M^0,75 | M^0,75 | `MILK_MAX_X_BASAL · basal` |
| endogen kväveförlust | M^0,75 | M^0,75 (Brody) | `ENDOGENOUS_N_PER_J · basal` |
| kvävepoolens tak | M^1 | M^1 | `N_POOL_CAP_FRAC · M` (0222) |

Elva termer, alla riktiga. Det är den delen av modellen som byggts med
härledning i stället för kalibrering, och den håller.

### Fel skalade — konstanter

| term | modellen | biologin | effekt |
|---|---|---|---|
| **födosökets bansträcka** | M^0 | M^0,25 (Garland 1983) | gynnar litet |
| **fosterbyggets takt** | M^0 | M^0,75 | gynnar litet, starkt |
| **åldrandets takt** | M^0 | M^−0,25 | gynnar litet |
| **mognadsåldern** | fritt locus, M^0 | M^0,25 | gynnar litet |
| **synvidden** | M^0 | växer med kroppen | gynnar litet |
| munkapaciteten | M^0 | M^0,71 (Shipley 1994) | binder inte i dag |

Ingen av dem pekar åt andra hållet. Det finns alltså **ingen term i modellen
som artificiellt gynnar stora kroppar** som kunde ha balanserat dem.

## De tre som betyder mest

### 1. Betesytan följer inte kroppen

Den svepta ytan är `2·r·L + π·r²` med `r = graze_reach_k · body_depth(M)`, alltså
`r ∝ M^⅓`. Men `L = forage_path_rate · dt`, och `forage_path_rate = 4 340`
längdenheter per månad är **samma tal för varje djur**. Konstantens egen
kommentar säger vad den är: *"valt så att den svepta ytan blir 7,0 cellareor
för den uppmätta medianmassan 1,2 kg"*.

```
   M (ts)   M (våt)   svept yta   basal/tick   yta per MJ basal
    0,011    0,041       2,35      0,016 MJ        144,1
    0,050    0,185       3,90      0,051 MJ         76,7
    0,150    0,556       5,62      0,116 MJ         48,5
    0,500    1,852       8,40      0,286 MJ         29,4
    1,000    3,704      10,58      0,481 MJ         22,0
```

Betesyta per enhet underhåll skalar som **M^−0,42**. Ett fyrtiogramsdjur får
4,9 gånger mer betesyta per joule underhåll än baslinjens medianindivid.
Uppmätt i körningen: massberoende intag 31,8 mot 10,6 kg per kg djur och månad,
alltså 3,0 gånger — samma storleksordning.

**Detta är dessutom internt motsägelsefullt sedan 0219.** Modellen har två mått
på hur långt ett djur rör sig, och bara det mindre av dem är härlett:

| storhet | skalning | ursprung |
|---|---|---|
| riktad färd | 1 200 lu/mån · M^0,2 | biomekaniskt härledd (0219) |
| födosökets bana | 4 340 lu/mån | kalibrerad vid 1,2 kg |

Den större storheten är den som aldrig fick behandlingen.

### 2. Dräktigheten tar tid proportionellt mot ungens massa

`gestation_growth_kg_per_s = 0,085 kg` per tidsenhet är en konstant, så
dräktighetens längd är `child_M / 0,085` — alltså **∝ M^1**. Hos verkliga
däggdjur går dräktigheten som ungefär M^0,25.

Halverad kroppsmassa halverar dräktigheten i modellen, mot sexton procents
förkortning i verkligheten. Efter 0222, när kvävet slutade strypa bygget, är
det den här konstanten som sätter kullintervallet — och den ger det lilla
djuret en reproduktionstakt som ingen biologi ger det.

Uppmätt föll intervallet mellan kullar från 10,72 till 1,44 månader samtidigt
som `M_target` föll till en fjärdedel. Det är vad `∝ M^1` förutsäger.

### 3. Åldrandet vet inte vad kroppen väger

`age_rate = k_age0 + k_age1 · ålder` med `k_age0 = 0` och `k_age1 = 3e−4`.
Ingen massberoende alls. Verklig livslängd går som M^0,25: en mus lever två år,
ett tvåkilosdjur åtta till tio.

I modellen lever alltså dvärgen lika länge som den stora kroppen **och**
förökar sig sju gånger snabbare. Det är den renaste formen av den saknade
avvägningen: litenhetens verkliga pris är ett kort liv, och modellen tar inte
ut det.

Mognadsåldern har samma brist från andra hållet: `A_mature` är ett fritt locus
mellan 5 och 20 månader, oberoende av `M_target`. Ett djur kan alltså
programmera vuxenmassa och mognadstid oberoende av varandra, medan de i
verkligheten är kopplade genom samma M^0,25.

## Vad mätningen också visade

Dvärgarna är **inte svultna**. Skadeinflödets sammansättning i `f6-256`:

```
                    p219 baslinje    p222 dvärgar
  metabol stress        52,6 %          81,7 %
  svält                  0,2 %           0,6 %
  kyla                   2,6 %           2,6 %
  ålder                 15,7 %           0,5 %
```

Svält 0,6 procent, kyla 2,6. De når sitt genetiska program exakt — programmet
är bara litet. Därför skulle en regel av typen *"dö om du inte når din
programmerade storlek"* inte röra dem alls. Den prövades dessutom en gång, som
svältskada mot `expected_mass(age)`, och gav den motsatta rusningen: i p179
sjönk `A_mature` mot golvet medan `M_target` steg, den krävda tillväxttakten
tredubblades, och alla tre frön dog ut. Se `Body._uppdatera_topp`.

Båda rusningarna har samma orsak. Löftet om storlek är gratis att sätta, och
det finns ingen fysisk storhet som binder det.

## Ordningsföljd för rättelserna

En ändring per körning, i fallande ordning efter förväntad verkan. Ingen av dem
är en motkraft: var och en tar en konstant som kalibrerats vid en kroppsstorlek
och låter den följa kroppen, som de elva redan riktiga termerna gör.

1. **Födosökets bansträcka skalas allometriskt.** `L ∝ M^0,25`, kalibrerad så
   att den uppmätta medianmassan behåller dagens bana — samma metod som 0219
   använde för marschfarten. Betesytan går då som M^0,58 och gradienten från
   M^−0,42 till M^−0,17.
2. **Fosterbyggets takt skalas med ämnesomsättningen.** `gest_rate ∝ M^0,75`,
   vilket ger dräktighet ∝ M^0,25. Kullintervallet slutar belöna litenhet
   linjärt.
3. **Åldrandet skalas med massan.** `k_age1 ∝ M^−0,25`, så att livslängden
   följer den allometri den har i verkligheten.
4. **Mognadsåldern kopplas till vuxenmassan.** `A_mature` blir en
   förskjutning kring `M_target^0,25` i stället för ett fritt tal — samma
   grepp som `breed_phase` fick när den blev en förskjutning mot årets topp i
   stället för en absolut fas.
5. Synvidden och munkapaciteten, om de visar sig binda efter de fyra ovan.

**Efter varje steg körs om, och evolutionen får svara.** Om massan fortfarande
drivs mot golvet när alla termer skalar rätt, då är litenhet svaret i den här
världen — och då ska det stå som ett resultat och inte rättas bort.
