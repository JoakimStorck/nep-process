# NEP Process – Utvecklingsplan
## Mot ett emergent, fysik-drivet forskningsverktyg (och senare konstmodus)

## 1. Syfte och målbild

NEP Process ska utvecklas från en parameterstyrd ALife-simulering till ett **lagbundet, kausalt läsbart och mätbart** system där emergenta fenomen kan:

- uppstå bottom-up ur lokala regler (agent–miljö-loopar)
- observeras i långa körningar utan numerisk instabilitet
- kvantifieras med enkla men robusta emergensmått
- senare (när dynamiken bär) exponeras som interaktiv konstinstallation

**Strategisk prioritet:**  
1) fysikaliskt konsekvent energibas → 2) kollektiv struktur via interaktioner → 3) evolutionär acceleration → 4) research/art-modus.

## 2. Övergripande designprinciper

1. Kausal läsbarhet först. Inför inte flera kraftiga mekanismer samtidigt.
2. All energi ska bokföras per tick (agentnivå och systemnivå).
3. Regler ska vara lag-/skalningsbaserade, inte “magiska tal”.
4. Lokala loopar är primära drivare. Undvik centrala “scripts” som skapar mönster.
5. Mätbarhet är ett krav. Varje fas ska ha minst 2–3 tydliga indikatorer.
6. Små steg, testbara förändringar. Varje delsteg ska kunna valideras med korta körningar.

## 3. Fasindelning (översikt)

- **FAS I** – Fysikalisk konsistens & energibokföring  
  Bygger en lagbunden grund och gör långkörningar stabila.

- **FAS II** – Agent–agent & agent–miljö-interaktioner  
  Skapar kollektiv struktur (kluster, territorier, flockar) utan ny evolutionär komplexitet.

- **FAS III** – Evolutionär acceleration  
  Först när FAS I–II är stabila: sexuell reproduktion, plastisitet och NEP-core.

- **FAS IV** – Forsknings- och konstmodus  
  När systemet redan producerar intressant dynamik: interaktiv input, sonifiering, presentationsläge.

---

# FAS I – Fysikalisk konsistens & energibokföring

## Mål
Etablera ett energikonsekvent dynamiskt system där all drift blir begriplig och mönster kan tolkas kausalt.

## Uteslutet i FAS I
- Sexuell reproduktion
- Plastisitet (lifetime learning)
- NEP-core (meta-evolution)
- Agent-agent kraftfält
- Interaktiv konstintegration
- Avancerade komplexitetsmått

### I.1 Energimodell (eliminera magiska tal)

#### I.1.1 Definiera massa/storlek (M)
- Varje agent ska ha entydig massa `M` (eller size som kan mappas till M).
- `M` ska dokumenteras i phenotype och användas i alla kostnadsfunktioner.

**TODO**
- [ ] Identifiera nuvarande definition av size/massa
- [ ] Om saknas: introducera `M` i phenotype
- [ ] Dokumentera hur `M` härleds och används

#### I.1.2 Allometrisk basal metabolism
Ersätt konstant basal med:

```python
basal_cost = k_basal * M**0.75 * dt

TODO
	•	Implementera basal_cost(M, dt)
	•	Ta bort hardkodade basalvärden
	•	Logga basal per tick

I.1.3 Rörelsekostnad (v²)
Ersätt konstant move_cost med:

move_cost = k_move * M * v**2 * dt

TODO
	•	Identifiera var rörelsekostnad tas ut
	•	Ersätt med M·v²
	•	Logga move_cost per agent

I.1.4 Energiintag (Eat) och fältkoppling
Energi som agenten får från fält ska:
	•	minska fältet motsvarande, eller
	•	explicit loggas som extern inflow (om fältet är “källa”).

Rekommendation: ΔE_agent = -ΔE_field_local.

TODO
	•	Identifiera nuvarande eat-implementation
	•	Säkerställ energiekvivalens eller loggad inflow
	•	Logga E_in_from_fields

I.2 Energy ledger (agentnivå och systemnivå)

I.2.1 Ledger per agent
Per tick bokförs:
	•	E_before, E_in, E_out (uppdelat), E_after

Invariant:

E_after = E_before + E_in - E_out

TODO
	•	Implementera agent-ledger (intern struktur)
	•	Assertion med tolerans (epsilon)
	•	Logga avvikelser och extrema värden

I.2.2 Ledger på systemnivå
Per tick loggas:
	•	Sum(E_agents)
	•	Sum(E_fields) (valfritt i början)
	•	E_total, dE_total

TODO
	•	Implementera systemtotal loggning
	•	Spara i steps.jsonl / world_log

I.3 Stabilitet för långkörningar

I.3.1 Numeriska skydd
	•	NaN-checks (E, pos, v, heading)
	•	Negativ energi → death eller clamp med tydlig regel
	•	Ev. maxhastighet om instabilitet kräver

TODO
	•	NaN-check i Body.step()
	•	Regel för E < 0
	•	Logga extrema värden

I.3.2 Testprotokoll
	•	3 seeds
	•	10× standard-T
	•	fast max_pop
	•	rapportera: överlevnad, drift, emergensindikatorer

TODO
	•	Skapa testscript
	•	Dokumentera resultat i /docs

I.4 Basala emergensindikatorer (minimikrav)

Syfte: kunna se kluster/struktur utan att introducera ny mekanik.

I.4.1 Spatial entropi
	•	Grid-binning av agenttäthet
	•	Shannon entropy per tick (eller var N tick)

TODO
	•	Implementera spatial entropy
	•	Logga tidsserie

I.4.2 Medelgrannavstånd
	•	k-NN avstånd (k=1..N) med numpy
	•	mean/median per tick

TODO
	•	Implementera med numpy
	•	Logga tidsserie

Definition: FAS I klar
	•	Inga hårdkodade energikonstanter kvar (för basal/move/eat)
	•	Ledger fungerar: agent + system
	•	Långkörningar stabila (inga NaN, begriplig drift)
	•	Minst två emergensindikatorer loggas

⸻

FAS II – Interaktioner (kollektiv struktur utan ny evolution)

Mål
Skapa genuin kollektiv dynamik (kluster, flockar, territorier) ur lokala regler via agent–agent och agent–miljö-loopar.

Uteslutet i FAS II
	•	Sexuell reproduktion
	•	Plastisitet
	•	NEP-core
	•	Konstmodus

II.1 Agent-agent perception (utan “social AI”)
Utöka sensing så att agenter kan detektera andra agenter på ett minimalt sätt:
	•	presence / density / nearest-agent vector
	•	ev. grov “tag” (art/typ) senare

TODO
	•	Lägg till agentdetektion i input-pipeline (kompakt signal)
	•	Logga hur ofta agentsensing aktiveras (diagnostik)

II.2 Enkel attraktion/repulsion (lokal potential)
Inför en minimal, kontrollerbar interaktionsregel:
	•	repulsion på kort distans (collision-avoid)
	•	optional attraction på längre distans (sociability)

Krav: ska gå att slå av/på och styras med få parametrar.

TODO
	•	Implementera repulsion (minsta stabila)
	•	Valfritt: attraction kopplad till trait
	•	Testa flockning utan central styrning

II.3 Miljömodifiering (feromonliknande spår)
Agenter får en enkel möjlighet att skriva till ett fält:
	•	deposit/consume i lokala celler
	•	diffusion skapar gradienter som agenter kan följa

Krav: energibokföring ska fortfarande fungera.

TODO
	•	Inför fält för deposition (eller reuse av befintligt fält med separat kanal)
	•	Koppla deposition till kostnad/energi (ledger)
	•	Testa självorganiserade stigar/territorier

II.4 Indirekt fitness via fält (ekologisk koppling)
Säkerställ att agenters handlingar påverkar andra via fälten:
	•	överkonsumtion → scarcity
	•	deposition → attraktorer/repellenter

TODO
	•	Mät korrelation mellan fältstruktur och agentdistribution
	•	Kontrollera att mönster består över tid (inte bara transient)

II.5 Emergensmått (utökning)
Behåll enkla mått, lägg till 1–2 robusta:
	•	klusterindex via grid-binning (andel massa i top-p cells)
	•	Moran’s I (spatial autocorrelation) om enkelt
	•	spektral proxy: variance över skalor (coarse-graining)

TODO
	•	Implementera 1–2 nya indikatorer utan externa lib
	•	Bygg liten “metrics summary” per körning

Definition: FAS II klar
	•	Reproducerbara kollektiva mönster (minst 2 seeds)
	•	Mönster kan kopplas till specifika lokala regler
	•	Metrics visar tydlig skillnad mot FAS I-baseline
	•	Inga stabilitetsproblem introducerade

⸻

FAS III – Evolutionär acceleration (när basen bär)

Mål
Öka adaptivitet och långsiktig diversitet utan att tappa kausal läsbarhet.
Krav: FAS I–II ska vara stabila och mätbara.

III.1 Sexuell reproduktion (crossover)
Inför parning baserat på:
	•	närhet
	•	enkel kompatibilitet (sociability / readiness)

Crossover: minimal först (t.ex. single-point eller blend för weights).

TODO
	•	Implementera parning + partner-val (lokalt)
	•	Implementera crossover i genetics
	•	Logga lineage/parent ids
	•	Starta med låg rate och tydliga begränsningar

III.2 Plastisitet (svag livstidsinlärning)
Inför mycket svag uppdatering av policy/weights:
	•	reward kopplat till energinetto / food_gain
	•	learning rate modulerad av trait

Krav: ska vara “små perturbationer” snarare än ny agentklass.

TODO
	•	Implementera liten weight update (valbar flagga)
	•	Logga learning activity per agent
	•	Utvärdera specialisering/robusthet

III.3 NEP-core (meta-evolution via agentoutput)
Inför en “evo-output” som modulerar mutation/crossover:
	•	stress → ökad variation
	•	stabilitet → minskad variation

Krav: mekanismen ska vara strikt begränsad och transparent i loggning.

TODO
	•	Implementera evo-output
	•	Koppla till mutation rate / noise scale
	•	Logga effektiv mutation per event
	•	Utvärdera diversitet över generationer

III.4 Diversitets- och evolutionsmått
Minimikrav:
	•	trait diversity (entropy över traitbins)
	•	lineage survival / branching
	•	phenotypic spread över tid

TODO
	•	Implementera diversitetsindex
	•	Exportera “run summary” (csv/json) per körning

Definition: FAS III klar
	•	Sexuell repro fungerar utan kaos/drift
	•	Plastisitet kan slås på/off och ger mätbara effekter
	•	NEP-core ger kontrollerbar meta-variation
	•	Diversitet och emergens ökar på ett begripligt sätt (metrics)

⸻

FAS IV – Forsknings- och konstmodus (när dynamiken redan är intressant)

Mål
Göra NEP till ett “verktyg + installation” genom att separera körlägen och lägga på interaktivitet/medieutgångar utan att påverka kärndynamiken.

IV.1 Körlägen: Research vs Art
Inför en enkel switch:
	•	Research: maximal logging, metrics, export
	•	Art: fokus på realtidsviz, input, sonifiering

Krav: kärnsimuleringen ska vara samma.

TODO
	•	Inför run-mode flagga i entrypoint
	•	Separera export/logik från rendering

IV.2 Interaktiv input (publik/pilot)
Exempel:
	•	klick för att lägga “perturbation” (hazard/heat/resource)
	•	sliders för globala parametrar (försiktigt)

Krav: input ska vara explicit loggad och reversibel.

TODO
	•	Implementera input events i viewer
	•	Logga input som “exogenous interventions”

IV.3 Multimedia outputs (sonifiering/video)
Bygg på befintliga loggar och metrics:
	•	ljud: population, energi, entropi, kluster → tonhöjd/rytmer
	•	video: export av frames eller summary clips

Krav: ska vara frikopplat från sim-tick.

TODO
	•	Implementera enkel sonifiering (optional)
	•	Frame export (optional)

IV.4 Demo- och dokumentationspaket
	•	minimal “one command demo”
	•	README med “what to look for”
	•	exempel runs + output

TODO
	•	Demo-skript
	•	Dokumentation och screenshots

Definition: FAS IV klar
	•	Research/Art-switch fungerar
	•	Interaktiv input påverkar sim på loggat, kontrollerbart sätt
	•	Multimedia output fungerar utan att störa sim
	•	Demo kan köras reproducerbart

⸻

4. Teststrategi (gäller alla faser)

För varje PR/ändring:
	•	Kör kort test (smoke test) + 1 längre baseline
	•	Minst 2 seeds för att undvika “one-off”
	•	Kontrollera:
	•	NaN/inf
	•	energiledger invariants
	•	metrics export
	•	regressions i runtime

TODO
	•	Standardisera “quick run” och “long run” presets
	•	Skapa en enkel “run summary” som alltid exporteras

5. Praktisk arbetsordning (rekommenderad)
	1.	FAS I: energi + ledger + stabilitet + 2 metrics
	2.	FAS II: agent sensing + repulsion + deposition + 1–2 metrics
	3.	FAS III: sexuell repro → plastisitet → NEP-core (en i taget)
	4.	FAS IV: research/art-switch + interaktivitet + multimedia

6. Sammanfattning

Planen bygger NEP som:
	1.	Energikonsekvent dynamik (FAS I)
	2.	Kollektiv struktur ur lokala regler (FAS II)
	3.	Evolutionär acceleration med bibehållen kausalitet (FAS III)
	4.	Forsknings- och konstmodus ovanpå ett stabilt system (FAS IV)

Fundamentet först; estetiken och interaktiviteten kommer när systemet redan “bär”.

