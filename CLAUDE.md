# CLAUDE.md — nep-process

Agentbaserad evolutionssimulator: flora och fauna som organismer i en
gemensam SoA-store (`organism_store.py`), världspass för näring, vatten och
temperatur (`world.py`), fasbaserad exekvering i `population.py`, numba-kärna
för floratillväxten i `flora_growth.py`. Allt arbete, alla kommentarer,
commitmeddelanden och dokument skrivs på svenska.

## Var planen finns

- `TODO.md` är projektets enda plan- och patchlogg. Filen börjar med
  utvecklingsplanen från juli (Del A–D, steg för steg). Patchloggen ligger
  längre ned, under Steg 7: en tabell med en rad per ändring i stigande
  ordning, där de öppna punkterna (`—` i första kolumnen) står sist efter
  de klara. Därefter följer en sektion per ändring, nyaste överst. Håll den
  uppdaterad — en ändring är inte klar förrän dess rad och sektion finns.
  Städcommits som bara rör arbetsordningen (CLAUDE.md, `.gitignore`) får
  ingen rad.
- Raden bär patchnumret i första kolumnen (överstruket när den är klar),
  och sektionsrubriken bär numret inom parentes: `### Kort rubrik (0201)`.
- `patches/` är arkivet med de gamla patch- och difffilerna. Den ignoreras
  av git; historiken finns i git-loggen.
- `docs/` bär de längre analyserna (livscykel, statusanalyser, revisioner).
  Mätprotokoll och revisionsdokument läggs där, inte i lösa filer.

## Grundprinciper

1. **Mekanism före heuristik.** Kod ska modellera en process, inte
   producera ett önskat utfall. Konstanter härleds fysiskt; en kalibrerad
   konstant markeras som sådan i kommentar, med vad den kalibrerades mot.
   Bygg aldrig en ny motkraft för att nå ett önskat utfall — hitta den
   saknade mekanismen i stället.
2. **Mät före bygg.** Ingen optimering och ingen mekanismändring utan
   mätning först. Misslyckanden dokumenteras lika noga som framgångar:
   en falsifierad hypotes skrivs in i TODO.md så att den inte prövas om.
   (Exempel som redan står där: parallelliseringen av tillväxtkärnan,
   0117, tillbakadragen 4 augusti — en procent långsammare på tolv kärnor,
   och Amdahl-räkningen hade gått att göra i förväg.)
3. **En ändring per commit.** Instrumentering och dynamikändring blandas
   aldrig i samma commit. Varje patch numreras löpande med fyra siffror i
   committiteln, följt av en kort svensk mening om vad som var fel eller
   vad som ändras: `0201: kort beskrivning`. Nästa nummer är det högsta
   0xxx-numret i `git log` plus ett. Numret är referensen i TODO.md och i
   löptext ("se 0190") — det tål omskrivning, vilket hashen inte gör.
   7xxx-serien var Steg 7:s geologi och vatten och fortsätts inte.
   Städcommits som bara rör arbetsordningen är onumrerade.
4. **Bitidentisk bana är måttstocken** för instrument- och
   prestandapatchar: kör referensen på ren HEAD och den patchade koden med
   samma frö, jämför alla loggrader utom tidtagning — och vid minsta tvivel
   även sluttillståndets arrayer bitvis. En dynamikändring får i stället en
   egen körning med uppmätt utfall i commitmeddelandet.
5. **Source of truth.** Varje fält har exakt en ägare; inget fält har två
   skrivare. Slotindex återanvänds, organism-id aldrig — kod som följer en
   individ över tid använder id.

## Verifiering före varje commit

Rökprov (invariantsviten ska godkännas):

    python run_headless.py --scenario scenarios/liten6.yaml \
        --ticks 400 --seed 1 --stats --report-every 400 --check-every 100

Bitprov för icke-dynamiska ändringar: samma kommando på ren HEAD respektive
arbetsträdet, diffa utskrifterna med tidtagningsrader bortfiltrerade
(`ms/tick`, `uppstart`, `på …s`). Noll skillnad krävs.

Tidsåtgång att räkna med innan ett kommando startas: rökprovet tar under en
minut; f6-256-skala kostar ~10–40 s i uppstart plus ~45 ms/tick vid jämvikt på
ledig maskin (p205, efter 0202 och 0204), så
långkörningar (tiotusentals tick) startas i bakgrunden med `nohup … &` mot
en egen katalog under `runs/`.

## Commitkonventioner

Claude committar direkt i trädet — ingen diffleverans, inga meddelandefiler
vid sidan av.

- Meddelandet dokumenterar **mekanism och uppmätt utfall**, inklusive
  förbehåll och det som inte gick att mäta.
- TODO.md får sin rad och sin sektion i samma commit.
- Rör aldrig omkringliggande kod "i förbifarten" — följdrättelser är egna
  patchar.

## Tekniska konventioner och kända fällor

- Heta pass är numpy-operationer över slotdelmängder — inga Python-loopar
  över `range(store.n)`. Per-organism-loopar tolereras bara för sällsynta
  händelser (födslar, dödsfall).
- `flora_growth.py` har två vägar: numpy-vägen **äger semantiken**,
  numba-kärnan ska ge samma resultat (dokumenterad divergens 1e-16). Ändras
  passet ändras båda vägarna.
- Traitsemantik ägs av `phenotype.py`, mutation och arv av `genetics.py` —
  ny biologisk logik byggs där och anropas från passen.
- Tidsenheten är **månader**: `dt = 0.02` → 50 tick per månad.
  `--world-every` räknar simulerade månader, inte tick.
- `_flora_slots()` cachas per tick; andra `rebuild_spatial_index`-anropet
  per tick är no-op via smutsflaggan — bryt inte de invarianterna.
- Mikrobenchfynd från utvecklingssandlådan, värda att minnas men billiga
  att ompröva på den här maskinen innan de får avgöra ett vägval:
  `np.empty` är i praktiken gratis (lat allokering), `np.take(out=)` var
  långsammare än fancy indexing, `np.bincount` används redan där det hör
  hemma. Prestandabeslut fattas på mätningar gjorda här, i realistisk
  skala.
- `store.energy` är för floran härledd diagnostik, inte tillstånd.
- Fysisk härledning i docstrings: enheter och källa för varje konstant.

## Vad som inte får göras

- Ändra dynamik i något som kallas prestanda- eller instrumentpatch.
- Ta bort en heuristik innan mekanismen som ersätter den finns och är mätt
  (`ESTABLISH_CROWD` är protes för saknad underhållsrespiration — den
  avvecklas sist, efter respirationen, en ändring per körning).
- Lägga till nya världsfält eller pooler utan konsument ("write-only-ytan").
- Optimera något som inte först profilerats i realistisk skala.
