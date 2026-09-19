# Revision: faunans närings-, energi- och massbalans

*2026-09-19, HEAD `8accae8`. Frågan: är balansen för de ätande djuren rimlig,
korrekt och sluten — finns fel, enhetsfel eller orimliga antaganden?*

Underlag: tre parallella kodgranskningar (föda och matsmältning, metabolism och
kroppsbudget, livshistoria och bokföring), var och en med fil:rad och räkning,
plus en mätkörning `runs/rev-fauna` (`f6-256` med 80 djur, 1 200 tick). De fynd
som bär slutsatserna är verifierade genom egen läsning av koden; det står vid
varje fynd vad som är läst, räknat eller mätt och vad som är misstanke.

## Sammanfattning

**Bokföringen är i huvudsak korrekt; storheterna är det inte.** Näringen
bevaras exakt genom ätande, exkretion, katabolism, tillväxt, födsel och död
(invariantsviten, drift 3e-10 i mätkörningen), energiledgern per individ och
tick stänger, `dt` förekommer en gång i varje takt, och basalmetabolismen är
Kleibers 3,4 W·kg^-0,75 omräknad till månader. Felen ligger i vad som räknas:

1. **Termoregleringen räknar inte kroppens egen värme** — den betalas helt
   ovanpå basal, rörelse och syntes, och lägger 0,6–1,2 × basal till
   underhållet vid 15 till −5 °C.
2. **Växtföda ger ungefär en tredjedel av sin verkliga energi**: 3,2 MJ per kg
   torrsubstans mot ~10 för smältbart gräs. Strukturmaterialet ger noll, och
   den labila delen värderas med våt vävnads 9,3 MJ/kg fast floran räknas i
   torrsubstans.
3. **Reserven har våt vävnads energitäthet även som fett**: `M_slow` håller
   9,3 MJ/kg mot fettets 32–39.

Tillsammans kräver de att ett djur äter omkring fem gånger så mycket växtmassa
som en verklig betare av samma storlek, medan det bär 3–4 gånger så tung reserv
per lagrad joule. Det är den troliga huvudorsaken till att faunan inte bär sig
(TODO, 0202): djuren svälter vid en flora som världen bär på lång sikt.

Utöver detta finns några buggar i hur kostnader prioriteras och takas, ett
utgångsläge som dödar en sjättedel av startdjuren inom en månad, och en
reproduktion som är nästan gratis energetiskt men kan äta upp modern.

## Mätkörningen

`runs/rev-fauna`: `f6-256`, frö 1, 1 200 tick, livs-, pop- och världslogg.
Faunan dör ut på ~24 månader (80 → 0; 94 döda, 93 av svält).

- **Startdjuren bär sig inte.** Reserven vid insättning är median 6 % av
  kroppsmassan (0,087 kg), åldern upp till 30 månader och skadan upp till
  D = 0,28. 14 av 80 dör av svält inom en månad och 36 inom tre, samtliga
  med tom reserv. Samma mönster som sådden av floran (23 % dör första ticken).
- **Poploggens energifält är ögonblicksbilder av en tick**, inte flöden över
  loggintervallet: de summerar varje djurs `last_flux`
  (`population.py:626–641`). En första läsning som månadsflöden gav en
  basalmetabolism femtio gånger under Kleiber — felaktigt, faktorn är tick per
  månad. Omräknat är basalen 20,4 MJ per tick för 118 kg, ~1 020 MJ/månad, i
  linje med Kleiber.
- **"Reservuttag"-tabellens `begärt kg`** är ett flöde: summan av begärda
  uttag över alla agenttick, i labila kg à 9,3 MJ. 1 400 kg på ~770 kg·månad
  djur är ~1,8 kg per kg och månad, alltså ~1,8 × basal — rimligt som
  fältmetabolism till storleken, men en tredjedel av det är termoposten i M1.

## Fynd

Beteckning: **F** = föda och matsmältning, **M** = metabolism och kroppsbudget,
**L** = livshistoria och bokföring, **I** = mätning och instrument. Status:
*verifierad* (läst kod och/eller räknat, av mig där det står så), *mätt*, eller
*misstänkt*.

### Energin i och ur kroppen — de stora felen

**M1. Termoregleringen räknar inte den metaboliska värmen.** *Verifierad (läst
`agent.py:1937–1951`).* `P_need = K·(Tb_set − Tenv)` betalas som ett tillägg,
och temperaturekvationen `T_inf = Tenv + P_gen/K` ser bara `P_gen`. Värmen från
basal, compute, rörelse och syntesarbete värmer aldrig kroppen. Uppmätt i en
isolerad `Body`: termo/basal 0,61–0,68 vid 15 °C, 0,87–0,96 vid 5 °C, 1,2 vid
−5 °C. Rätt bokfört är tillägget `max(0, K·ΔT − värmeproduktion)`; med
modellens K ≈ 0,18 W/°C vid 2 kg ligger den nedre kritiska temperaturen kring
5 °C, så termoposten vore nära noll i `liten6` och bara vintertid i `f6-256`.
K är dessutom ~0,55 × Herreid & Kessels allometri. *Dynamikändring;*
`isolering_max` och `isolering_halv` är härledda mot dagens termokostnad och
måste mätas om.

**F1. Växtföda ger en tredjedel av sin energi.** *Verifierad (räknat).*
`assimilated_fraction = (1 − s)·0,80·d` (`phenotype.py:929ff`) ger
strukturmaterialet noll, och det labila värderas med `E_labile_J_per_kg =
9,302e6` (`agent.py:218`), tillbakaräknat från den borttagna `E_body = 7,0e6 /
0,75` — alltså **våt** kroppsvävnad. Floramassan räknas i torrsubstans. Vid
florans medianstruktur s = 0,567 ger ett kg torr växt 0,433 · 0,80 · 9,3 =
3,2 MJ; smältbart gräs ger ~10 MJ/kg, eftersom labil torrsubstans bär 17–23
MJ/kg och en betares våm eller blindtarm bryter ned 40–50 % av cellulosan.
Basalen ensam kräver 7,7 % av kroppsvikten per dygn i växtmassa vid 2 kg; med
fältbehovet 2–2,5 × basal 15–19 %, mot 5–6 % hos en verklig betare. *Kräver ett
principbeslut om massbas:* torr eller våt, och om cellulosa ska kunna brytas ned
(en diet-axel, inte en konstant).

**M3. Reserven har samma energitäthet i båda poolerna.** *Verifierad.* `M_fast`
och `M_slow` omräknas båda med 9,3 MJ/kg (`agent.py:1491–1493`); `M_slow` kallas
fettet men fett håller 37–39 MJ/kg och fettväv ~32. Bärkostnaden per lagrad
joule — basal ∝ M_carry^0,75, värmeledning ∝ M^⅔, rörelse ∝ M — blir 3,5–4
gånger för hög för fett, `reserve_cap_max` = 1,18 kg per kg stomme motsvarar
energetiskt bara ~0,29 kg fett, och avvägningen snabbt/långsamt saknar fettets
verkliga fördel.

### Buggar i prioritering och tak

**M5/L1. Mobiliseringstaket gäller per anrop, inte per tick.** *Verifierad
(läst `agent.py:1185–1262`; isolerat mätt: fyra anrop gav 7,7 × basal per tick
mot taket 2 ×).* `tak = mobil_max_x_basal · P_basal · dt / E_labile` räknas om
i varje anrop till `_take_reserve_mass`; underhåll, efterbetalning, reparation,
byggarbete och predation får var sitt tak. Byggarbetet tas dessutom strypt medan
materialet tas ostrypt, och inget kontrollerar att arbetet blev fullt betalt —
vilande i dag (byggarbetet fick 100 % i rökprovet).

**L2. Dräktigheten går före moderns underhåll och kan katabolisera henne.**
*Verifierad (läst `agent.py:1980–2003`).* Fostret byggs i steg (2C), före
dräneringarna, och saknas reserven kataboliseras moderns vävnad ned till
`M_min` = 0,01 kg — inte till dödströskeln 0,65 · `M_peak` — utan
`k_cat_dmg`-skada och utan svältflagga. Varje kg foster kostar ~2,1 kg moderns
vävnad vid s = 0,25. Samma fälla som `docs/metabolismen.md` rättade för
tillväxten ("diskretionära utgifter får inte ligga bland de obligatoriska").

**L3. Kullen läcker massa och näring när store:n är full.** *Verifierad i kod,
sällsynt.* `population.py:1532` `except RuntimeError: break` efter att `gest_M`
nollats och startreserven betalats; en kull på upp till sex kan ta beståndet
till `max_pop` + 5.

**F3. Floran värderas ur den egna cellen men betas ur grannskapet.** *Läst,
frekvens omätt.* `population.py:3256` läser `flora_cell_structure[cell]`, som är
noll i en tom cell — floran värderas då som helt labil medan grannskapet som
faktiskt betas kan vara segt.

**F4. Betningen skriver faunans `store.energy`.** *Läst, latent.*
`population.py:2251–2254` skriver `mass` och `energy` för alla slotar i
grannskapet; masken nollar bara uttaget. Läks i dag av body-passet före alla
läsare, men bryter "en skrivare per fält" och slår igenom om passordningen
ändras.

**M4. Kroppstemperaturen startar på börvärdet och får då ingen uppvärmning.**
*Verifierad (isolerat mätt).* Villkoret `Tb < Tb_set` gör att första ticken
betalar ingen termo; nyfödda faller till 21 °C (vid 15 °C) eller 7 °C (vid
−5 °C) och tar köldskada 2–3 tick. Flyttalsavrundning återskapar läget senare.

### Orimliga antaganden

**F2. Intagstaket är ett fast tal per individ och blandar stock med takt.**
*Verifierad (läst `agent.py:4280–4296`).* `eat_rate · dt` = 1,8 kg per tick
oavsett massa — 90 % av kroppsvikten per tick vid 2 kg — och Hollings
asymptot dt/h är samma tal. `want_kg` lägger reservens hela tomma kapacitet
(en stock) till tickens förbrukning (en takt), så ett hungrigt djur vill fylla
hela reserven på en tick och påfyllnadstakten per månad beror på dt. Ingen
magkapacitet, ingen passagetid.

**L4/M7. Reproduktionen är nästan gratis energetiskt.** *Räknat.* En kull vid
2 kg kostar ~4,4 % av basalen över en cykel. Fostret saknar massbörda och
ämnesomsättning (`gestation_mass_burden` och `gestation_P_overhead_per_kg`
läses med `getattr`-standard 0 men finns inte i `AgentParams`), och laktationen
är ren väntetid — hos små däggdjur den dyraste fasen, 2–4 × BMR.

**L5. Dräktighetstakten är absolut, oberoende av moderns massa.**
`gestation_growth_kg_per_s = 0,085` kg/mån: dräktighet plus laktation ~100
dygn vid 2 kg och ~200 vid 4 kg, mot kanin 31 och hare 42. Bör skala ~M^0,75.

**L6. Nyfödda klarar sig själva från tick 0 med ~3 ticks reserv.** Gåvan är
högst 0,8 · 1,4e6 J/kg; en unge på 0,05 kg har reserv för ~2,9 ticks (~1,7
dygn), sedan katabolism ~6 % per tick. Utrustas dessutom mot `E_cap_per_M` och
inte mot sin ärvda `reserve_cap`, så en snål genotyp kastar upp till 55 % av
gåvan som överflöd.

**L7. Kullen kan väga mer än modern, och könsmognaden kommer vid 15 %.**
`M_repro_min` 0,15–0,45 och `child_M` 0,08–0,20 av `M_target` är oberoende
loci; i värsta fall bär modern 133 % av sin massa.

**L8. Kostnader skalar med lagringskapaciteten.** `repro_cost · E_cap()`,
parningen `0,05 · E_cap()` och attackkostnaden ∝ `E_cap()`; samma födsel kostar
22 gånger mer för den fetaste genotypen — en stock som används som takt.

**F5. Kadaver saknar funktionell respons**: tas ur egna cellen med upp till
1,8 kg per tick, utan söktid eller hanteringstid, medan floran har Holling.

**M2. Sensingkostnaden är en sekundkvarleva.** *Verifierad.* `sense_cost_L1..L3`
= 0,2/0,5/1,0 J/(kg·månad), 5–8e-8 × basal: sensingnivån är gratis.

**M6. Katabolismens förlust exkreteras i stället för att förbrännas**: 10 % av
den labila delen går till detritus som föda i stället för att bli värme. Näringen
bevaras; liten effekt.

**M8. Hjärnkostnaden skalar med fart**, ~675 J/(kg·m) marginellt mot Taylors
8,6 — men absolut 2–3 % av basal.

### Mätning och instrument

**I1. Poploggens energifält ser ut som flöden men är en ticks ögonblicksbild**
(se Mätkörningen). Populationens energiledger kan inte heller stänga: födslar,
död, `E_overflow`, startreserven till avkomman, `repro_cost`, parnings- och
attackkostnad och den döendes sista tick saknas. Per individ och tick stänger
den.

**I2. Enhetsetiketter från sekundskalan** på värden som redan är i månader:
`k_basal # [W]`, `eat_rate # kg/s`, `compute_cost # [W/kg]`,
`thermo_Pmax_per_kg (W/kg)`, `cold_damage_gain` "per second",
`gestation_growth_kg_per_s`, `starve_damage_gain` "D/s", `k_age1` "400–500s".
Värdena är rimliga; etiketterna ljuger.

**I3. Döda parametrar och inaktuell dokumentation.** Utan läsare:
`E_bio_J_per_kg`, `repro_cooldown_s`, `attack_energy_gain`, `birth_E0`,
`birth_k_E_per_M`, `birth_energy_eff`, `M_repro_soft`, `E_repro_soft`,
`median_age_s`, `PopParams.carcass_yield`, fenotypens `E_rep_min`;
`store.repro_capacity` skrivs men läses inte. `docs/metabolismen.md` anger
matsmältningen som 0,80 − 0,35·s, koden har 0,80. `E_labile_J_per_kg` finns i
både AgentParams och WorldParams. Fem öppna TODO-rader beskriver tillstånd som
inte längre gäller (`M_waste_frac`, absolut `child_M`, kullstorleken, bärarvalet,
`repro_cooldown_s`).

## Det som stämmer

- Näringen bevaras exakt i ätande (också blandad föda — näringshalten är linjär
  i s), exkretion, katabolism, tillväxt, födsel, död och överföringar till
  avkomman. Kadavret är M + reserv + foster med massviktad struktur.
- Energiledgern per individ och tick är komplett; drift 0 i isolerat test.
- `dt` finns en gång i varje takt; inga kvarvarande 3600 eller 86400.
- Basalen är Kleibers 3,42 W·kg^-0,75, uppmätt 0,5–4 kg.
- Rörelsekostnaden följer Taylors 10,7·M^-0,316 J/(kg·m), ~5 % av basal.
- Tillväxtens byggarbete/material 0,419 mot avsedda 0,4286.
- Katabolism av mager vävnad 3,8–6,3 MJ/kg, rimligt för blöt mager vävnad.
- Termiken: tidskonstant ~11 h vid 2 kg, värmekapacitet rätt, relaxationen
  stabil.

## Ordning

Mät före bygg, en ändring per commit, instrument och dynamik var för sig:

1. **I1 (instrument):** poploggens energifält som flöden över loggintervallet,
   plus poster för födsel och död, så att populationens energi stänger och
   varje följande ändring går att mäta.
2. **M1 (dynamik):** den metaboliska värmen räknas av mot termobehovet. Tydlig
   fysik, störst enskild kostnad.
3. **M5 och L2 (buggar):** mobiliseringstaket per tick; dräktigheten efter
   underhållet och med golv vid dödströskeln.
4. **F1 och M3 (principbeslut):** massbasen för föda och reserv — torr eller
   våt, fettets energitäthet, och om cellulosa ska kunna brytas ned.
5. **Utgångsläget:** startdjurens och de nyföddas reserv.
6. Därefter L4–L8, F2, F3, F5 och städningen i I2–I3.
