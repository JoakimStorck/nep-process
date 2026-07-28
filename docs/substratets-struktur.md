# Designskiss — substratets struktur

*Juli 2026. Underlag för Steg 4 i TODO.md. Status: förslag, inte beslut — uppdateras eller markeras som ersatt när steget byggs.*

---

## Frågan som ledde hit

När flora får mortalitet i Steg 4 hamnar dött växtmaterial i samma `detritus`-fält som kadaver. Hur skiljer vi då på en asätare och en organism som lever på förmultnande växter?

Två separata fält, `detritus_animal` och `detritus_plant`, vore fel svar. Det kodar in precis den sortens artgräns manifestet säger att vi inte definierar i koden, och det skalar illa: nästa distinktion — ved mot löv, färskt mot gammalt — kräver ett tredje fält.

Rätt mekanism är att detritus bär en **egenskap som ärvs från det som dog**, och att konsumenterna differentierar sig mot den.

## Varför inte `digestibility`

Manifestet nämner `digestibility` bland floras loci. Den traiten har ett problem: **den har ingen uppsida.** Ingen organism vinner på att vara lättare att äta. Selektionen driver den mot noll och sedan ligger den död — en axel som ser evolverbar ut men inte är det.

## Strukturandel

Det som faktiskt varierar är hur stor del av vävnaden som är segt bärande material — lignin och cellulosa hos växter, kitin, ben och keratin hos djur — jämfört med lättomsatt protein, socker och fett.

Den axeln har genuina avvägningar åt båda håll, och den beskriver **både ontologierna med samma tal**. Samma trait gäller en förvedad stam och ett benigt skelett, vilket är vad manifestet kräver: skillnaden mellan växt och djur ska ligga i kapacitetsprofilens värden, inte i klasstillhörighet.

---

## Principen: en trait, flera konsekvenser

En trait med en enda konsekvens är ett reglage, inte en anpassning. Selektionen hittar optimum och stannar där. Diversitet uppstår först när samma tal har **motverkande** konsekvenser, så att olika nischer gynnar olika värden.

Det bör vara ett krav på varje ny trait vi inför: kan den dras åt båda håll av olika selektionstryck? Kan den inte det är den antingen en konstant i förklädnad eller en axel som kommer att kollapsa.

`structure` klarar testet. Nedan är dess konsekvenser, och de pekar åt olika håll.

### Medan organismen lever

**Byggkostnad.** Strukturmaterial kostar mer energi per kilo massa att bygga. Hög struktur ger långsammare tillväxt för given energitillgång.

**Energitäthet.** Strukturmassa lagrar ingen användbar energi. En organism med hög strukturandel har mindre att katabolisera vid svält — dess effektiva reserv per kilo är lägre.

**Underhållskostnad.** Här vänder det: strukturvävnad är metaboliskt billig att underhålla. Ben och ved kräver lite. Hög struktur sänker alltså underhållskostnaden per kilo, vilket är en verklig löpande fördel som balanserar byggkostnaden.

**Betningsmotstånd.** Den som betar högstrukturerad vävnad får ut mindre energi per kilo. Passivt försvar genom seghet, till skillnad från `defense` som är aktivt avskräckande — tagg och gift. De två överlappar men är inte samma sak, och `defense` bör lämnas orörd tills vidare.

### Efter döden

**Nedbrytningstakt.** Hög strukturandel bryts ner långsammare. Det här är den konsekvens som gör att attributet inte är write-only från dag ett — `decomposition_pass` läser den omedelbart.

**Näringsfrisättning.** Långsam nedbrytning ger en utdragen näringskälla i stället för en puls. En högstrukturerad organism gödslar sin cell länge men svagt; en lågstrukturerad ger allt på en gång.

### Senare, på konsumentsidan

**Matsmältningsförmåga.** Att utvinna energi ur strukturmaterial kräver en kapacitet — lång tarm, långsam passage, symbionter — med egen underhållskostnad. Det hör till Steg 6, när kapacitetsfälten får sina läsare. Först då blir specialiseringen ett verkligt val.

---

## Vad det ger utan att kodas

Asätaren mot detritivoren faller ut av avvägningen i stället för att definieras.

**Kadaver** är lågstrukturerat. Mjukvävnad som vem som helst kan tillgodogöra sig, men sällsynt, kortlivat och omstritt — det bryts ner fort och andra hittar det också.

**Förna** är högstrukturerad. Kräver matsmältningskapacitet som kostar att bära, men finns överallt och försvinner inte.

Det är samma avvägning som i verkligheten skiljer en gam från en termit, och ingendera behöver finnas i koden.

---

## Representation

`detritus` bär massa per cell som i dag, plus ett andra glest fält med **massviktad medelstrukturandel** i samma aktiva mängd. Vid tillskott blandas den nya massans struktur in viktat; vid nedbrytning och konsumtion är den oförändrad.

Kadenklassen är densamma som för `detritus`: glest dynamisk, samma aktiva mängd, samma kontrakt att inaktiva celler är exakt noll.

**En ärlighetsanmärkning.** Med ett massviktat medelvärde per cell kan en färsk kadaver i en cell full av förna inte plockas ut separat — den späds ut. Det är en verklig förenkling. Men den är konsekvent med resten av modellen: cellen är miljöns enhet, vilket var hela grunden för beslutet att avveckla subcell-sampling. En stor kadaver dominerar medelvärdet, vilket är rätt; en liten i mycket förna drunknar, vilket är diskutabelt men försvarbart.

---

## Vad som hör till Steg 4 och vad som väntar

**I Steg 4**, eftersom de har konsumenter direkt:

- `structure` som locus, gemensamt för flora och fauna
- byggkostnad i tillväxt
- energitäthet i massa-till-energi-omvandlingen
- nedbrytningstakt som funktion av strukturandel
- näringsfrisättning som följer nedbrytningen
- betningsutbyte per kilo

**Till Steg 6**, tillsammans med övriga kapacitetskostnader:

- underhållskostnad som sjunker med strukturandel
- matsmältningskapacitet som konsumentsidans axel

---

## Öppna frågor

**Ska `structure` vara ett locus eller två?** Flora och fauna kan dela ett, eller ha var sitt inom samma representation. Ett gemensamt är mer i manifestets anda men antar att skalan betyder samma sak för ved och ben.

**Hur brant ska kostnadsfunktionerna vara?** Byggkostnad och underhållsbesparing måste balanseras så att båda ändarna av axeln är levbara. Om de inte är det kollapsar traiten, och då är det kalibreringen som är fel, inte idén.

**Ska betningsmotståndet vara utbyte eller åtkomst?** Antingen får betaren mindre energi per kilo, eller så går det långsammare att ta. Det första är enklare; det andra skapar tidskostnad som interagerar med predationsrisk.
