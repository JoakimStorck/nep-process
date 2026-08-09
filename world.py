# world.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _numba = None
    _NUMBA_AVAILABLE = False

from grid import Grid
from klimat import (
    arsamplitud,
    arsmedeltemperatur,
    faseftersläpning,
    halvklotstecken,
)
from terrain import TerrainParams, generate_elevation
from drainage import build as build_drainage
from hydro import (
    derive_water as hydro_derive_water,
    sediment_deposit as hydro_deposit,
    sediment_pass as hydro_sediment,
    leach_pass as hydro_leach,
    lake_levels as hydro_lake_levels,
    route_reservoirs as hydro_route,
    soil_pass as hydro_soil_pass,
)
from phenotype import (
    DECAY_SCALE_LABILE,
    DECAY_SCALE_STRUCT,
    NUTRIENT_PER_KG_LABILE,
    NUTRIENT_PER_KG_STRUCT,
)

# Under detta värde nollställs detritus exakt och cellen lämnar den aktiva
# mängden. Utan en tröskel skulle exponentiellt avklingande celler aldrig bli
# inaktiva, och glesheten vore verkningslös.
_DETRITUS_EPS = 1e-12

# Index i Grid.neighbor_idx. Ordningen är upp, ned, vänster, höger.
_NEIGHBOR_DOWN = 1

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

   
# -------------------------
# Parameters
# -------------------------
@dataclass
class WorldParams:
    # Världens form i celler. Höjden måste vara jämn: hexgeometrins radoffset
    # blir annars inkonsistent över sömmen. Bredd 64 och höjd 256 ger fyra
    # klimatband med 64 rader mellan pol och ekvator — se TODO.md, Steg 2.
    width: int = 64
    height: int = 256
    size: int = 0      # bekvämlighet för kvadratiska världar; sätter båda
    dt: float = 0.02

    # Florans massaskala. En vuxen planta väger B_K gånger sin genetiska
    # adult_mass_k, alltså 0,25–4 gånger detta tal.
    #
    # Var 5e-4, vilket gjorde en vuxen planta till en halv gram mot faunans
    # kilo. Följden var att konsumentbiomassan blev 138 gånger
    # primärproduktionen — förhållandet omvänt mot hur en ekologi ser ut.
    # Ingen kalibrering av näringstillförsel eller täthet rättar en sådan
    # inversion; det är massaskalan som är fel.
    B_K: float = 11.0

    # -------------------------
    # Temperature / seasons
    # -------------------------
    year_len: float = 12.0

    # Klimatet som **tidsprofil, härledd ur världens position**:
    #
    #     T(t) = T_mean + T_amp * sin(2π (t - lag) / year_len - season_phase0)
    #
    # där `T_mean`, `T_amp` och `lag` inte är parametrar utan följder av
    # `latitud` och `kontinentalitet`. Se `klimat.py`, som äger regeln.
    #
    # Den rumsliga latitudgradienten föll i 7013. Skälet står i
    # `docs/varldens-skala.md`: en cell har arean 100 m², och även den största
    # körbara världen — 4096x4096 celler, alltså 44x38 km — spänner 0,11 grader
    # av jordens meridionella gradient. Modellen hade trettio.
    #
    # **Det som kommer tillbaka här är inte gradienten utan positionen.**
    # Världen ligger *på* en breddgrad; den spänner inte över flera. Latituden
    # är därför en skalär i fysiklagret och inte ett fält över celler, och den
    # hör inte hemma i `Grid` — cellformen ska inte veta var på en planet
    # världen ligger.
    #
    # Vinsten är att klimatet blir härlett i stället för valt. 7013 satte
    # årsmedel 11 och amplitud 12 med motiveringen "en dalgång i södra Sverige"
    # — en analogi, inte en härledning. Förvalet nedan ger samma klimat
    # (11,0 / 12,0 inom en tiondel), men nu som en följd av var världen ligger:
    # 48 grader nord, halvvägs in i en landmassa. Ett experiment blir en
    # flyttning i stället för fyra nya tal.
    #
    # Årstiden inverteras söder om ekvatorn; det faller ur latitudens tecken.
    latitud: float = 48.0
    kontinentalitet: float = 0.55

    # Fasen i årscykeln vid t = 0. Den termiska eftersläpningen är **inte**
    # detta tal utan en följd av kontinentaliteten — havet lagrar värme och
    # släpper den sent, så varmaste månaden ligger en till två månader efter
    # solståndet. Den ändringen gör t = 0 till astronomisk vårdagjämning, där
    # temperaturen ännu ligger under årsmedlet. Före 7014 låg t = 0 vid
    # termiskt medel, alltså ungefär en och en halv månad senare.
    season_phase0: float = 0.0

    # Growth gating thresholds (degC): g(T) in [0,1]
    T0: float = 0.0
    T1: float = 20.0

    # Höjdgradient: grader per längdenhet, mätt mot havsnivån.
    #
    # **Talet är fysik och inte längre ett val.** Jordens lapse rate i fri luft
    # är 6,5 grader per kilometer. Längdenheten är tio meter sedan skalan
    # fastställdes, alltså 0,065 grader per enhet. Se `docs/varldens-skala.md`.
    #
    # Var 16,0, vilket är tvåhundrafemtio gånger för mycket — men det gick inte
    # att veta då, eftersom höjden saknade enhet. Talet valdes mot vad det
    # gjorde med tillväxtgrinden inom ett latitudband i stället för mot en
    # storhet utanför modellen, vilket är precis det manifestet nu förbjuder:
    # en storhet utan enhet går inte att sätta fel på ett sätt som märks.
    #
    # **Referensytan mot bandets landmedel är borttagen.** Den fanns därför att
    # terrängens kontinentala lutning följde latituden och därmed åt upp
    # latitudgradienten — mätt mot rå höjd tappade ekvatorn 7,3 grader medan
    # rad 32 tappade 2,1, och klimatet inverterades. Med latituden borta finns
    # varken gradienten att äta upp eller banden att mäta mot. Referensen är
    # havsnivån, vilket är vad en höjd mäts från.
    #
    # **Höjdgradienten är försumbar tills reliefen växer, och det är riktigt.**
    # Vid dagens relief på omkring två längdenheter, alltså tjugo meter, ger
    # den 0,14 grader. Ett kuperat landskap på tio till fyrtio kilometer har
    # tre- till femhundra meters relief, och då blir bidraget två till tre
    # grader — tjugo gånger mer än latituden någonsin kunde ge vid den här
    # storleken. Reliefen skalas upp som eget steg; det är den ändringen som
    # ger höjden dess verkan, inte den här.
    #
    # Sänkor kyls inte: bara upphöjning över havsnivån ger avdrag. Att en grop
    # skulle bli varmare än sitt omland är inte ett fenomen modellen har —
    # inversion har motsatt tecken och är kalluftsdränering, en egen mekanism
    # med egen orsak.
    lapse_rate: float = 0.065

    # -------------------------
    # Hydrology / terrain / world fields
    # -------------------------
    sea_level: float = 0.0
    submerged_threshold: float = 1e-6

    elevation_init: float = 0.0
    water_init: float = 0.0
    # Terrängen. None = platt värld, och `elevation` förblir en skalär i sin
    # statiska kadensklass. Sätts den promoveras fältet till en per-cell-array
    # en gång vid världens tillkomst och rörs aldrig därefter.
    #
    # Havsnivån är noll per definition, så en cell med negativ höjd ligger under
    # havet. Enheten delas med `water`, eftersom fri yta är summan av de två.
    terrain: TerrainParams | None = None
    # Sådden av näring, fördelad som vid jämvikt.
    #
    # `nutrient_init` är den **fria**, växttillgängliga poolen — inte näring
    # bunden i marken. Bundet material är detritus, som stod på noll. Världen
    # startade alltså med hundra procent av sin näring omedelbart tillgänglig
    # och en steril mark. Verkliga ekosystem ligger på motsatt ytterlighet:
    # under en procent tillgängligt, resten i markens organiska material.
    #
    # Följden var mätbar. En floraköring över 60 000 tick utan fauna ägnade
    # merparten av inkörningen åt omfördelning snarare än tillväxt: fri pool
    # 4 142 kg vid tick 500, 50 kg vid jämvikt. Och stocken bar inte sig själv
    # — 5 243 kg såddes, 4 854 återstod efter hundra år, fortfarande fallande.
    #
    # Jämvikten följer av en identitet. Förlusten är `nutrient_loss_frac`
    # gånger mineraliseringen, och vid jämvikt möter den tillförseln:
    #
    #     mineralisering_jämvikt = nutrient_input · n_cells / nutrient_loss_frac
    #
    # Nedbrytningstakten påverkar inte det flödet. Den bestämmer bara hur stor
    # detrituspoolen måste vara för att bära det. Vid uppmätt näringsviktad
    # takt 0,613/år och 16 384 celler ger identiteten 904 kg/år mot uppmätta
    # 1 293, alltså en faktor 0,699 kvar att falla. Jämvikten är då
    # 1 475 kg näring i detritus, 1 885 i flora och 35 fri — totalt 3 395.
    #
    # Talen nedan sår den fördelningen direkt. Detritus läggs på sin
    # jämviktsnivå; florans andel läggs i den fria poolen, eftersom det är
    # därifrån växterna tar upp. Under inkörningen dras den fria poolen ner
    # från 1 920 till 35 kg medan biomassan byggs, och systemet landar där det
    # hör hemma i stället för att först göra sig av med en gåva.
    #
    # Skalar linjärt med `nutrient_input`: fördubblad bördighet fördubblar alla
    # tre.
    nutrient_init: float = 0.117
    detritus_init: float = 21.16
    # Strukturandel i den sådda förnan. Behövs för att `detritus_structure`
    # styr nedbrytningstakten — sås detritus utan den bryts allt ner som labilt
    # material och hela poolen mineraliseras på nio månader. 0,93 är den
    # uppmätta jämviktssammansättningen: strukturell förna bryts ner 6,7 gånger
    # långsammare och anrikas därför i poolen, oavsett vad floran fäller.
    detritus_structure_init: float = 0.93

    rain_input_base: float = 0.0
    spring_input_base: float = 0.0
    infiltration_base: float = 0.0
    evaporation_base: float = 0.0

    # --- Terrängburen hydrologi -------------------------------------------
    #
    # Aktiv bara när `terrain` är satt. I en platt värld finns ingen riktning
    # och hydro faller tillbaka på det rumsligt konstanta forcing-uttrycket,
    # vilket gör varje scenario före Steg 7 bitidentiskt.
    #
    # Enheten för vatten är densamma som för höjd, eftersom fri yta är summan.

    # Nederbörd per månad vid världens årsmedeltemperatur, före orografi.
    #
    # Var 0,65 med referensen på 25 grader. Referensen låg då nära den gamla
    # latitudvärldens beboeliga mitt, så basen betydde ungefär "nederbörd på en
    # normalt bördig plats". När klimatet blev en dalgång på 11 grader hade
    # samma referens halverat nederbörden — inte som ett val utan som en
    # kvarleva av en världsbild.
    #
    # Referensen är därför världens årsmedel, som positionen sätter, och basen
    # den nederbörd som hör till det. Talet är satt så att **ariditeten är
    # oförändrad**: uppmätt landmedel av temperaturfaktorn var 0,897 i den
    # gamla `f5-terrang` och tidsmedlet vid amplitud 12 är 1,180, vilket ger
    # 0,494 för bevarad nederbörd — och gånger 0,532/0,821, kvoten mellan ny
    # och gammal potentiell avdunstning, blir det 0,320. Ett kallare klimat ska
    # ha mindre nederbörd; att hålla P konstant medan PET faller vore att smyga
    # in en våtare värld i en klimatändring.
    #
    # **Nederbörden härleds inte ur positionen.** Sahara och monsun-Indien
    # ligger båda kring tjugo grader nord och skiljer tre tusen millimeter —
    # spridningen inom en breddgrad är större än signalen mellan breddgrader.
    # Talet nedan hör därför till världen och inte till platsen, och bör ses
    # över när världen flyttas långt.
    rain_base: float = 0.320
    # Varm luft bär mer fukt — ungefär en fördubbling per tio grader. Med
    # klimatet som tidsprofil ger det en regnperiod på sommaren och en torr
    # vinter, vilket är den kontinentala nederbördsfördelningen. Den rumsliga
    # delen — våta tropiker och torra poler — föll med latituden; kvar rumsligt
    # är bara det orografiska lyftet.
    #
    # Referensen är inte en parameter utan **världens egen årsmedeltemperatur**,
    # och sätts därför av positionen. Ett fast tal var det som gjorde 25 grader
    # till en kvarleva: flyttas världen norrut utan att referensen följer med
    # torkar den ut av ett tal ingen valt.
    rain_T_doubling: float = 10.0
    # Orografiskt lyft: nederbörden växer med höjden. Ingen regnskugga — den
    # kräver en förhärskande vindriktning och hör till senare arbete.
    rain_oro_gain: float = 0.6

    # Markens fältkapacitet. Överskott blir avrinning samma tick.
    soil_capacity: float = 0.60
    # Potentiell avdunstning per månad vid full tillväxtgrind. Den faktiska
    # avtar linjärt med torrhet, så att markfuktigheten blir en gradient i
    # stället för en binär grind.
    et_max: float = 1.15
    # Andel av markvattnet som lämnar som basflöde per månad. Det är det som
    # håller fåror rinnande mellan regnen; utan det torkar varje vattendrag ut
    # samma tick som regnet slutar.
    baseflow_frac: float = 0.25
    # Infiltrationstak per cell och månad för vatten som passerar. Utan
    # återinfiltration är markvattnet rent punktvis och dalar blir inte blötare
    # än ryggar — den topografiska fuktgradienten uppstår aldrig. Taket är
    # absolut och inte en andel: som andel förlorade en flod hela sitt flöde på
    # tjugo celler och inga fåror kunde bildas.
    reinfiltration_max: float = 1.0
    # Öppet vatten avdunstar utan markens torrhetsbroms. Multiplikator på
    # `et_max` för sjöyta.
    lake_evap_mult: float = 1.0

    # Fårans djup ur genomströmningen: Mannings form förenklad till en
    # potenslag, eftersom bredden inte modelleras.
    channel_k: float = 0.010
    channel_exp: float = 0.40
    channel_slope_floor: float = 1e-3
    # Gränsen mellan fåra och sluttning, uttryckt i hur många cellers
    # medelavrinning som måste passera. Under den är vattnet ytavrinning och
    # markflöde, inte en vattenyta en organism kan möta. Skalfritt: talet
    # betyder samma sak oavsett nederbörd och världsstorlek.
    channel_min_upslope: float = 20.0

    # --- Vattnet som florans tredje resurs ---------------------------------
    #
    # Vattenbehov per kilo byggd vävnad. Sätter hur hårt vattnet binder mot
    # näring och ljus; kalibrerat mot att `flora_water_limited` och
    # `flora_light_limited` blir jämförbara, på samma sätt som ljuset
    # kalibrerades mot näringen. Uppmätt vid 64x128 över 5 000 tick:
    #
    #     0,03   vatten 0,004  ljus 0,272   bebodda 61,5 %   12 325 kg
    #     0,06          0,025       0,243            59,4    11 777
    #     0,12          0,196       0,251            48,8     7 067
    #     0,25          0,429       0,127            19,8     1 577
    water_per_kg: float = 0.12
    # Andel av cellens markvatten som är åtkomlig för upptag per **månad**.
    # Under ett betyder att marken inte kan tömmas på en gång — växten når det
    # lätt tillgängliga först, och resten binds hårdare i porerna.
    #
    # Var en andel per *tick*, vilket är fel skalningsklass: 0,25 per tick är
    # 12,5 gånger per månad, alltså tjugo gånger nederbörden, och floran
    # dränerade marken till 0,12 av fältkapacitet så att varje fåra torkade ut.
    # Värre än så var att utfallet berodde på `dt` — halverat tidssteg hade gett
    # halverad månadstakt. Samma sorts fel som bruset som skalade med `dt` i
    # stället för med roten ur det.
    water_extract_rate: float = 0.25

    # -------------------------
    # Detritus / decay
    # -------------------------
    detritus_decay: float = 0.077
    # Kadaver bryts ned åtta gånger snabbare än förna. Kött är inte lignin, och
    # takten är det som avgör om en kollaps kan livnära sig på sina egna döda:
    # med förnans takt låg kadavret kvar i månader och blev ett skafferi. Vid
    # 0,62 är halveringstiden knappt en månad, alltså en resurs man måste hinna
    # fram till.
    carcass_decay: float = 0.62
    # Andel av frisatt näring som lämnar systemet (urlakning, denitrifikation).
    #
    # Var 0,10, vilket töms världen på 37 år simulerad tid: uppmätt förlust var
    # 40,6 kg av en stock på 536 under 6 000 tick, monotont, medan tillförseln
    # bidrog med 0,029 kg. Ostörda landekosystem är snåla återvinnare —
    # förlusterna är någon procent av det interna flödet, och vittring och
    # nedfall är små i motsvarande grad. Vid 0,01 blir tömningstiden omkring
    # 370 år och tillförseln en långsam forcing i stället för ett dropp som
    # håller världen vid liv.
    nutrient_loss_frac: float = 0.01
    # Näringstillförsel per cell och månad. Konstant tills terrängen finns; då
    # blir den vittring som funktion av höjd, och utsköljning under havsnivå.
    #
    # Tillförsel och förlust är inte oberoende: i jämvikt är förlusten
    # `nutrient_loss_frac` gånger hela primärproduktionen. Med stocken från
    # `nutrient_init`, två års omsättning av levande vävnad och 16 384 celler
    # ger det 7,1e-5. Se docs/vaxternas-livscykel.md.
    #
    # Mätt över 40 000 tick: tillförseln gav 930 kg medan förlusten tog 600, en
    # kvot på 1,55, och stocken växte med 331 kg. Talet härleddes ur en antagen
    # levandeomsättning på 24 månader; den verkliga blev längre. 4,6e-5 är
    # samma härledning med den uppmätta omsättningen.
    nutrient_input: float = 4.6e-5
    # --- Näringen följer terrängen ----------------------------------------
    #
    # Vittringen är källan och urlakningen sänkan. Ingenting återförs från
    # havet: det som lakas ut är borta, precis som i en verklig landekologi,
    # där budgeten är vittring in mot urlakning ut och havet är en sänka på
    # ekologisk tidsskala.
    #
    # Vittringsvikten normeras till medelvärde ett över land, så att den totala
    # tillförseln är exakt densamma som med ett konstant fält. Bara den
    # rumsliga strukturen tillkommer; identiteten som `nutrient_init` och
    # `detritus_init` härleddes ur står orörd.
    #
    # Exponenten styr hur skarpt vittringen följer lutningen. Färsk berggrund
    # exponeras där materialet rör sig, alltså i brant terräng.
    weathering_slope_exp: float = 0.5
    # Andel av den lösta näringen som faktiskt följer med vattnet. Under ett
    # svarar mot att en del är bunden till partiklar. Takten i övrigt kommer ur
    # vattenbudgeten och inte ur en parameter — se hydro.leach_pass.
    leach_efficiency: float = 0.6

    # --- Partikulär transport av förna --------------------------------------
    #
    # Löst näring följer vattnet ovillkorligt; partiklar måste ryckas med och
    # sjunker när flödet saktar. Takten skalas därför både med vattnets andel
    # och med lutningen, normerad mot `sediment_slope_ref`. En sjöcell har
    # lutning noll och blir en fälla utan att det kodas som regel.
    # Takten är en avvägning och inte ett fritt val. Uppmätt vid 64x128 över
    # 5 000 tick:
    #
    #     takt   flora kg   näring total   förna land/vatten per cell
    #      0,0    16 128        1 581        15,19 /  1,50
    #      0,5    10 922        1 321         8,78 / 10,60
    #      1,5     8 379        1 135         4,99 / 14,13
    #      3,0     7 908        1 027         3,27 / 15,32
    #
    # Vattnet blir en resurs på landets bekostnad, och näringsförlusten
    # motsvarar exakt sedimentexporten: allt som når en fåra hamnar till slut i
    # havet. Sjöarna behåller sitt — lutning noll gör dem till fällor — medan
    # floderna är ett rent avlopp. Vid 0,5 är vattnet redan rikare per cell än
    # landet, till en kostnad av 16 procent av näringsstocken.
    sediment_rate: float = 0.5
    sediment_slope_ref: float = 0.05
    # Strukturmaterialets rörlighet relativt det labila.
    #
    # Avsikten var sortering: fint labilt material skulle transporteras lättare
    # och ge vattnet lägre strukturandel och därmed högre betningsutbyte.
    # **Det sker inte.** Uppmätt strukturandel i vatten 0,965 mot landets
    # 0,956, alltså högre. Det labila bryts ned långt snabbare än det
    # transporteras, så material som blir liggande åldras till struktur oavsett
    # var det ligger. Vattnet får mer föda, inte bättre.
    #
    # Parametern behålls eftersom mekanismen är fysiskt riktig och skulle bita
    # om transporten någon gång blir snabb mot nedbrytningen, men den ska inte
    # läsas som att sorteringen är verksam.
    sediment_struct_mobility: float = 0.25
    # Näringsupptag per månad och **areaenhet** vid uptake_capacity = 1.
    #
    # Var ett tak per individ på 2,86e6, alltså sju till åtta tiopotenser för
    # högt: uppmätt band det i noll av alla individer, med median efterfrågan
    # 0,22 kg mot ett tak på 3,6e4. Kapacitetsmodellens första läsare läste
    # ingenting. Med rotarean som anspråk får taket dessutom rätt form — ett
    # upptag per areaenhet, inte per individ. Storleksordningen följer av att
    # en medianplanta på 23 kg (area 2,1) ska hinna binda sina 0,36 kg näring
    # på ungefär ett år.
    uptake_rate_per_area: float = 0.03

    # --- Ljus som andra begränsande resurs ---------------------------------
    #
    # Tillförsel per cell och månad, uttryckt direkt i **kg vävnad** som
    # cellens instrålning räcker till att bygga. Att välja den enheten i
    # stället för joule sparar en omvandlingskonstant utan att förlora något:
    # ingenting annat i modellen läser ljus.
    #
    # Nivån är satt så att ljus och näring är jämförbart begränsande vid den
    # uppmätta jämvikten. Vid 254 000 kg stående biomassa och ett förnafall
    # kring tre procent per månad är primärproduktionen omkring 0,46 kg per
    # cell och månad. Är talet mycket högre blir ljuset inert, mycket lägre
    # och näringen blir det.
    #
    # 0,60 gav en ljusbegränsad värld: `flora_light_limited` låg på 0,886 vid
    # mättnad och biomassan på 136 914 kg mot baslinjens 254 566. Och ju
    # knappare ljuset är, desto mer är höjd värd — vilket ger strukturandelen
    # en uppsida som förstärker just det ljuset skulle motverka. 1,5 är satt
    # för att kväve och kol ska binda jämförbart; utfallet mäts som att
    # `flora_light_limited` faller mot 0,5.
    light_input: float = 1.5

    # Extinktionskoefficient i Beer–Lambert. Ljuset som når en planta avtar
    # exponentiellt med bladarean ovanför den. Λ i modellen är ett
    # markarealindex och inte ett bladarealindex, så koefficienten bär även
    # omräkningen mellan dem. Vid Λ = 1,4 får en helt undertryckt planta
    # omkring elva procent av full instrålning.
    light_extinction: float = 0.5

    # Bladarea per kilo labil vävnad, i cellareor per B_K. Utan den blir
    # bladarean lika med rotarean, och då täcker ett kilo blad en elftedels
    # cell — en groddplanta hinner då inte växa snabbare än sitt eget
    # förnafall och varje bestånd dör ut. Specifik bladarea är i verkligheten
    # den variabel som skiljer växtstrategier mest, och den är stor: ett kilo
    # blad täcker långt mer mark än ett kilo rot försörjer.
    # Specifik bladarea, per kilo **skott**. Fördubblad från 10 när kroppen
    # delades i rot och skott: vid rotandel 0,5 är bara halva massan skott, och
    # talet är satt så att bladinkomsten vid ρ = 0,5 och strukturandel 0,70
    # blir densamma som före uppdelningen. Den nya axeln startar därmed
    # neutralt i stället för att smyga in en omkalibrering.
    leaf_area_per_kg: float = 20.0

    # Specifik rotarea, per kilo rot. Härledd på samma sätt: vid ρ = 0,5 och
    # s = 0,70 ger 6,7 samma upptagsarea som den gamla regeln A = m / B_K.
    root_area_per_kg: float = 6.7

    # Asymmetrins skärpa. Höjden är m · s — strukturmassan, alltså det som
    # håller organismen uppe, enligt docs/substratets-struktur.md. En planta
    # som är dubbelt så hög som cellens medelhöjd får r = 0,67 och undgår två
    # tredjedelar av skuggan.
    light_height_ref: float = 1.0

    @property
    def uptake_rate_max(self) -> float:
        """Upptagstak i kg näring per månad och areaenhet."""
        return float(self.uptake_rate_per_area)
    # Diffusionstakt för löst näring. Explicit schema: D*dt måste hållas under
    # ett för stabilitet, och laplacianen är normerad med grannantalet.
    nutrient_diffusion: float = 0.20

    # --- Perceptionens mättnad -------------------------------------------
    #
    # Världskanalerna mättas med `v / (v + K)`. `K` är halvmättnadsvärdet och
    # måste ligga där fältet **typiskt** ligger; annars mappas hela den
    # observerade fördelningen på en ände av skalan och kanalen slutar bära
    # information.
    #
    # `C_sense_K` stod på 5e-4, alltså **en halv gram**, mot en förnastock på
    # åttioåtta kilo per landcell. Talet är samma halva gram som `B_K` en gång
    # hade: den höjdes till 11,0 när massaskalan rättades, och syskonkonstanten
    # lämnades kvar. En kalibrering som flyttas på ett ställe och inte på det
    # andra ger ingen felsignal — den ger en kanal som alltid säger ja.
    #
    # Uppmätt i `liten6` efter 250 tick, per cell och regim:
    #
    #     fält                land    sjö     hav     (kg per cell)
    #     flora                 9,2    8,5     0,00
    #     detritus+kadaver     99,0  187,4     0,38
    #
    #     kanal               land    sjö     hav
    #     B  vid K = 11      0,415  0,410   0,0001    <- bär kontrasten
    #     C  vid K = 5e-4    1,000  1,000   0,970     <- mättad överallt
    #     C  vid K = 80      0,524  0,677   0,0044
    #
    # Med det gamla värdet läste **havet 0,60 mot landets 0,87** i den
    # dietviktade blandningen, trots 262 gångers skillnad i verklig födomängd.
    # Det är hela förklaringen till att djuren uppfattar föda ute till havs: det
    # är förnakanalen som säger ja, inte floraskanalen.
    #
    # Halvmättnaden läggs vid den uppmätta medianen, 88 kg, avrundat till 80.
    # **Det är ett kalibrerat värde och inte ett härlett** — förnastocken är
    # nedbrytningstidens produkt och har ingen egen naturlig enhet, till skillnad
    # från floran där `B_K` är en vuxen plantas massa.
    #
    # Sjöarna hamnar då **över** landet, vilket de faktiskt är: sedan 7008 är de
    # sedimentfällor. Den akvatiska nischen fanns redan och var osynlig.
    #
    # `B_sense_K` får samtidigt ett eget namn vid samma värde som `B_K`.
    # Perceptet läste massaskalan direkt, så en ändring av hur mycket en planta
    # väger ändrade tyst vad ett djur ser. Två storheter, ett tal — nu två namn.
    B_sense_K: float = 11.0
    C_sense_K: float = 80.0

    # Energy densities for open-system ledger diagnostics (J/kg)
    # Energi per kilo *labil* vävnad, gemensam för allt organiskt material.
    # Den användbara energitätheten är detta gånger (1 - strukturandel);
    # strukturmaterial lagrar ingen användbar energi.
    #
    # De tidigare skilda konstanterna för växt och kadaver, 4,0e6 och 7,0e6,
    # kodade som typskillnad det som är en materialegenskap. Med initierad
    # medelstruktur 0,57 för flora och 0,25 för fauna ger den här enda
    # konstanten 4,00e6 respektive 6,98e6 — samma kvot, men härledd.
    E_labile_J_per_kg: float = 9.302e6


# -------------------------
# World
# -------------------------
@dataclass
class World:
    WP: WorldParams
    grid: Grid = field(init=False)    

    def __post_init__(self) -> None:
        self.grid = Grid(size=int(self.WP.size),
                         width=int(self.WP.width),
                         height=int(self.WP.height))

        self.sample_flora_local_hook = None
        self.sample_flora_rays_hook = None        
        self.consume_food_hook = None

        # -------------------------
        # Primary world fields
        # -------------------------
        # Världsfälten har både en ägare och en kadens. Ägaren säger vem som
        # skriver; kadensen säger hur ofta fältet behöver röras.
        #
        #   statiska      lagras som skalär tills något varierar dem rumsligt
        #   dynamiska     platta per-cell-arrayer indexerade med cell_idx
        #   härledda      beräknas vid läsning, inte varje tick
        #
        # En tom cell ska vara billig, av samma skäl som en organism utan en
        # kapacitet inte ska kosta något för den. Se docs/varldens-kadensmodell.md.
        nc = int(self.grid.n_cells)

        # --- statiska: skalära tills terräng eller väder gör dem rumsliga ---
        #
        # Promoveringen skalär -> array är en engångshändelse som fältet självt
        # hanterar; anropare behöver inte veta vilken form det har.
        # `check_world_field_domains` prövar båda formerna.
        if self.WP.terrain is not None:
            self.elevation = generate_elevation(self.grid, self.WP.terrain)
        else:
            self.elevation = float(self.WP.elevation_init)

        # Dräneringsnätet är terrängens statiska konsekvens och byggs en gång.
        # Det är inte ett världsfält utan en struktur över dem — därför en egen
        # attribut och en egen invariant, inte en post i WORLD_CELL_FIELDS.
        # None i en platt värld: utan höjdskillnader finns ingen riktning.
        self.drainage = (
            build_drainage(self.grid, self.elevation, float(self.WP.sea_level))
            if self.WP.terrain is not None
            else None
        )
        # Höjdens statiska temperaturmodifierare. Byggs före hydrolagren,
        # eftersom markens avdunstning läser tillväxtgrinden per cell.
        self._T_offset = self._build_T_offset()
        self.rain_input = float(self.WP.rain_input_base)
        self.spring_input = float(self.WP.spring_input_base)
        self.infiltration = float(self.WP.infiltration_base)
        self.evaporation = float(self.WP.evaporation_base)

        # --- dynamiska: per cell ---
        self.water = np.full(nc, np.float32(self.WP.water_init), dtype=np.float32)
        # Havet fylls till nivå noll vid världens tillkomst. Det är en
        # begynnelsevillkor och inte hydrologi: en cell under havsnivån har
        # vatten från första ticken, och `surface_level` är därmed konsistent
        # innan något pass har kört. Hydro tar över underhållet i 7004.
        if self.WP.terrain is not None:
            deep = np.maximum(-np.asarray(self.elevation, dtype=np.float32), np.float32(0.0))
            np.maximum(self.water, deep, out=self.water)
        # nutrient lagras i float64 till skillnad från övriga fält. Det är den
        # bevarade storhet vi vill kunna påstå balans om: näring cirkulerar
        # medan kol flödar igenom, så näringsbalansen är den enda hårda
        # invarianten som är möjlig. I float32 ackumulerar diffusionens
        # laplacian ett fel kring 5e-7 per tusen pass; i float64 är det 1e-15.
        self.nutrient = np.full(nc, float(self.WP.nutrient_init), dtype=np.float64)
        # Sådden läggs på land. Med havet som ren sänka lakas dess andel ut
        # under inkörningen och är borta för alltid — uppmätt 21,4 procent av
        # den fria näringen i havsceller, alltså en värld som startar med en
        # femtedel för lite bördighet och en inkörning som mäter omfördelning i
        # stället för ekologi. Totalen är oförändrad, så jämviktsidentiteten
        # bakom `nutrient_init` står orörd.
        if self.WP.terrain is not None:
            _sea = np.asarray(self.elevation, dtype=np.float64) < float(self.WP.sea_level)
            _nland = int((~_sea).sum())
            if _nland > 0:
                self.nutrient[:] = 0.0
                self.nutrient[~_sea] = (
                    float(self.WP.nutrient_init) * nc / float(_nland)
                )
        # detritus är glest dynamiskt: nollskilt i en bråkdel av cellerna, och
        # ett fullt svep skulle mest multiplicera nollor. Fältet bär därför en
        # aktiv mängd, och kontraktet är att inaktiva celler är exakt noll.
        # detritus och detritus_structure lagras i float64 av samma skäl som
        # nutrient: de ingår i näringsbalansen, som är hård invariant. I float32
        # låg balansens drift kring 1e-7 relativt och golvet sattes av just
        # dessa två fält. Kostnaden är minne, inte tid — fälten är glest
        # dynamiska och sveps bara över den aktiva mängden.
        self.detritus = np.full(nc, float(self.WP.detritus_init), dtype=np.float64)
        if self.WP.terrain is not None:
            _sea2 = np.asarray(self.elevation, dtype=np.float64) < float(self.WP.sea_level)
            _nl2 = int((~_sea2).sum())
            if _nl2 > 0:
                self.detritus[:] = 0.0
                self.detritus[~_sea2] = (
                    float(self.WP.detritus_init) * nc / float(_nl2)
                )
        self._detritus_member = self.detritus > _DETRITUS_EPS
        self._detritus_active = np.flatnonzero(self._detritus_member).astype(np.int32, copy=False)

        # Massviktad medelstrukturandel i cellens detritus, ärvd från det som
        # dog. Styr nedbrytningstakten: högstrukturerat material bryts ner
        # långsammare. Samma kadensklass och samma aktiva mängd som detritus.
        self.detritus_structure = np.full(
            nc, float(self.WP.detritus_structure_init), dtype=np.float64
        )
        # Tom förna har ingen sammansättning. Att lämna andelen satt i celler
        # utan detritus vore ett tillstånd utan bärare, och `_detritus_add`
        # blandar massviktat och skulle då blanda mot ett spöke.
        self.detritus_structure[~self._detritus_member] = 0.0

        # Ackumulerad exkreterad massa sedan senaste ledgeruppdatering.
        self._dM_excreted = 0.0

        # Näringens externa flöden, ackumulerade sedan start. Näringen är den
        # enda storhet som kan cirkulera slutet — kol flödar igenom — så
        # balansen mäts mot dessa och inte mot total massa.
        self._nutrient_added_total = 0.0
        self._nutrient_lost_total = 0.0

        # Aktiveringar och avaktiveringar samlas och slås ihop en gång per tick.
        # Att göra dem direkt mot arrayen kostade O(n) per händelse, vilket dög
        # för sällsynta kadaver men inte för exkretion i varje betningshändelse.
        self._detritus_pending: list[int] = []
        self._detritus_dirty = False

        # Kadaver är en egen pool. Tidigare hälldes kroppar i `detritus`, och
        # eftersom strukturandelen blandas massviktat drunknade ett kadaver vid
        # 0,25 i cellens förna vid 0,83 och kom ut kring 0,80. Följden var att
        # asätarnischen aldrig kunde löna sig: en ren asätare fick 0,087 ur
        # detritus mot betarens 0,258 ur flora, alltså en enkelriktad nackdel
        # och ingen avvägning. Samma glesa maskineri som förnan, egen takt.
        self.carcass = np.zeros(nc, dtype=np.float64)
        self.carcass_structure = np.zeros(nc, dtype=np.float64)
        self._carcass_member = np.zeros(nc, dtype=bool)
        self._carcass_active = np.zeros(0, dtype=np.int32)
        self._carcass_pending: list[int] = []
        self._carcass_dirty = False

        # --- härledda: flödesstyrkan är noll tills hydro räknar grannflöde ---
        self.flow_strength = 0.0

        # --- hydrologins tre lager -------------------------------------------
        #
        # Allokeras bara i en terrängvärld. En platt värld har ingen riktning
        # och hydro faller tillbaka på det gamla forcing-uttrycket.
        # soil_water i float64 av samma skäl som nutrient och detritus: det är
        # den bevarade storhet vattenbalansen påstås om. I float32 ackumulerade
        # avrundningen systematiskt och balansen drev 5,4e-8 relativt, linjärt
        # i tid; i float64 ligger den vid 1e-16.
        self.soil_water = np.zeros(nc, dtype=np.float64)
        # discharge i float64: sjöarnas magasin och havets utflöde summeras ur
        # den, och vattenbalansen ska kunna påstås på 1e-6.
        self.discharge = np.zeros(nc, dtype=np.float64)
        self._runoff = np.zeros(nc, dtype=np.float64)
        self._hydro_acc = np.zeros(2, dtype=np.float64)
        if self.drainage is not None:
            dr = self.drainage
            nl = int(dr.n_lakes)
            # Magasinet är per bassäng, inte per cell. En sjö kan därmed inte
            # darra kring en tröskel, eftersom den inte har något
            # per-cell-tillstånd att darra i.
            self.lake_storage = np.zeros(nl, dtype=np.float64)
            self._lake_level = np.zeros(max(nl, 1), dtype=np.float64)
            self._lake_area = np.zeros(max(nl, 1), dtype=np.float64)
            # Sjöarna börjar fulla. En tom sänka som fylls under inkörningen
            # vore samma sorts övergående omfördelning som `nutrient_init`
            # rättade för näringen.
            self.lake_storage[:] = dr.lake_cap
            # Den orografiska modifieraren är statisk. Höjden normeras mot
            # landets högsta punkt, så att talet betyder samma sak oavsett
            # `relief`.
            z = np.asarray(self.elevation, dtype=np.float64)
            zmax = float(z.max())
            self._oro = (
                1.0 + float(self.WP.rain_oro_gain) * np.maximum(z, 0.0) / max(zmax, 1e-9)
            ).astype(np.float64, copy=False)
            self.soil_water[:] = float(self.WP.soil_capacity)
            self.soil_water[dr.sea] = 0.0
            self._update_lake_levels()
            self._n_land = int((~dr.sea).sum())
            # Numba-kärnan tar en array, inte en union av skalär och array. I
            # en terrängvärld finns alltid en modifierare; är lapse_rate noll
            # är den nollfylld och kostar bara sin minnestrafik.
            off = self._T_offset
            self._T_offset_arr = (
                np.asarray(off, dtype=np.float64) if isinstance(off, np.ndarray)
                else np.full(nc, float(off), dtype=np.float64)
            )
        else:
            self.lake_storage = np.zeros(0, dtype=np.float64)
            self._lake_level = np.zeros(1, dtype=np.float64)
            self._lake_area = np.zeros(1, dtype=np.float64)
            self._oro = None

        # Vattnets externa flöden, ackumulerade sedan start. Stocken är
        # markvatten plus sjömagasin; kanalvatten är i transit och lagras inte,
        # vilket följer av jämviktsantagandet.
        self._water_added_total = 0.0
        self._water_lost_total = 0.0
        self._water_stock_init = self.water_stock()
        self._leach_acc = np.zeros(2, dtype=np.float64)
        self._sed_acc = np.zeros(3, dtype=np.float64)
        self._sed_in_lab = np.zeros(nc, dtype=np.float64)
        self._sed_in_str = np.zeros(nc, dtype=np.float64)
        self._sed_changed = np.zeros(nc, dtype=bool)
        self._weathering = self._build_weathering()
        self._grad_x, self._grad_y = self._build_elevation_gradient()

        # time
        self.t = 0.0
        self.last_flux = {
            "dM_growth": 0.0,
            "dM_wither": 0.0,
            "dM_decay": 0.0,
            "dM_detritus_decay": 0.0,
            "dM_nutrient_from_detritus": 0.0,
            "dM_water_added": 0.0,
            "dM_water_removed": 0.0,
            "dM_transport": 0.0,
            "E_in_growth": 0.0,
            "E_loss_wither": 0.0,
            "E_loss_decay": 0.0,
        }

        # Klimatet är en **skalär i tiden plus en statisk modifierare i
        # rummet**. Kadensdokumentet beskrev formen som *profil plus eventuella
        # per-cell-modifierare*; profilen har nu längd ett.
        #
        # Det som föll är latituden, inte årstiden. Bandmaskineriet var därmed
        # inte en optimering som togs bort utan en representation vars enda
        # innehåll visade sig vara påhittat: att lagra `n_bands` tal som alla
        # är samma tal är dyrare än att lagra ett.
        #
        # `T_air` är luftens temperatur i världen som helhet. Cellens
        # temperatur är `T_air + _T_offset[cell]`, där modifieraren i dag bär
        # höjden och senare kan bära kalluftsdränering.
        self.T_air = np.float32(0.0)

        # Klimatets tal härleds **en gång** ur positionen. Fysiklagret äger
        # regeln (`klimat.py`), världslagret instansierar den, och därefter är
        # de tre konstanter under hela körningen — ingen tick rör dem.
        lat = float(self.WP.latitud)
        kont = float(self.WP.kontinentalitet)
        self.T_mean = float(arsmedeltemperatur(lat, kont))
        self.T_amp = float(arsamplitud(lat, kont)) * float(halvklotstecken(lat))
        self.season_lag = float(faseftersläpning(kont))

        self._update_temperature()

    def _build_T_offset(self):
        """
        Statisk temperaturmodifierare per cell: höjdgradienten.

        Kadensdokumentet förutsåg formen — *profil plus eventuella
        per-cell-modifierare* — och att modifieraren kan vara frånvarande och
        då inte kosta något. I en platt värld returneras skalären 0,0.

        Referensytan är **havsnivån**, vilket är vad en höjd mäts från. Den var
        en gång landets medelhöjd i samma latitudband, som en lapp på att
        terrängens kontinentala lutning följde latituden och åt upp
        klimatgradienten. Med latituden borta finns varken gradienten eller
        banden, och lappen kan tas bort. Se `lapse_rate`.

        Bara upphöjning kyler. Att en sänka blir varmare än sitt omland är inte
        ett fenomen modellen har — inversion har motsatt tecken och är
        kalluftsdränering, en egen mekanism med egen orsak.
        """
        if self.WP.terrain is None or float(self.WP.lapse_rate) == 0.0:
            return 0.0
        z = np.asarray(self.elevation, dtype=np.float64)
        rel = np.maximum(z - float(self.WP.sea_level), 0.0)
        return (-float(self.WP.lapse_rate) * rel).astype(np.float32, copy=False)


    # -------------------------
    # Temperature / season
    # -------------------------
    def _update_temperature(self) -> None:
        """
        Årstiden. Ett sinusvarv per år kring `T_mean`, förskjutet med den
        termiska eftersläpningen.

        Eftersläpningen är skälet att den varmaste månaden inte är den
        soligaste: underlaget lagrar värme och släpper den sent, en månad i
        inlandet och två vid kusten. `t = 0` är därmed astronomisk
        vårdagjämning, där temperaturen ännu ligger under årsmedlet.

        Tillväxtgrinden räknas inte här. Den var en bandvektor som ingen läste
        — grinden behövs alltid per cell, eftersom höjdmodifieraren gör den
        olika inom vad som en gång var ett band, och `_gate_from_T()` är den
        enda vägen dit.
        """
        WP = self.WP
        year_len = float(WP.year_len) if float(WP.year_len) > 1e-9 else 1.0

        phase = 2.0 * math.pi * (((self.t - self.season_lag) % year_len) / year_len)
        phase -= float(WP.season_phase0)

        self.T_air = np.float32(self.T_mean + self.T_amp * math.sin(phase))

    def _gate_from_T(self, T):
        """
        Tillväxtgrinden ur temperaturen. Formen ligger på ett ställe, eftersom
        den nu räknas från fyra håll: cell, cellmängd, helt fält och hydro.
        """
        T0 = float(self.WP.T0)
        T1 = float(self.WP.T1)
        if T1 <= T0 + 1e-9:
            return (np.asarray(T) >= T0).astype(np.float32)
        g = (np.asarray(T, dtype=np.float32) - np.float32(T0)) / np.float32(T1 - T0)
        return np.clip(g, 0.0, 1.0).astype(np.float32, copy=False)

    def temperature_of_cell(self, cell: int) -> float:
        """Temperatur i en cell. Den form biologin ska använda."""
        T = float(self.T_air)
        off = self._T_offset
        return T + (float(off[int(cell)]) if isinstance(off, np.ndarray) else float(off))

    def temperature_of_cells(self, cells: np.ndarray) -> np.ndarray:
        """
        Temperatur för en mängd celler, utan att materialisera hela fältet.

        Skalären bär årstiden; den statiska modifieraren bär höjden. Summan är
        en per-cell-storhet, men den kostar bara en gather över den
        efterfrågade delmängden — inget svep över världen, och inget
        bandindexuppslag.
        """
        cells = np.asarray(cells)
        off = self._T_offset
        if isinstance(off, np.ndarray):
            return (np.float32(self.T_air) + off[cells]).astype(np.float32, copy=False)
        return np.full(cells.shape, np.float32(self.T_air) + np.float32(off),
                       dtype=np.float32)

    def growth_gate_of_cell(self, cell: int) -> float:
        return float(self._gate_from_T(self.temperature_of_cell(cell)))

    def growth_gate_of_cells(self, cells: np.ndarray) -> np.ndarray:
        return self._gate_from_T(self.temperature_of_cells(cells))

    def temperature_field(self) -> np.ndarray:
        """
        Hela temperaturfältet per cell.

        Materialiserar en array med längd n_cells och kostar därmed O(n_cells).
        Avsedd för visning och diagnostik, inte för systempass — dessa ska
        använda temperature_of_cell() eller temperature_of_cells().
        """
        return self.temperature_of_cells(np.arange(int(self.grid.n_cells)))

    def growth_gate_field(self) -> np.ndarray:
        """Hela tillväxtgrindsfältet per cell. Samma kostnadsanmärkning som ovan."""
        return self._gate_from_T(self.temperature_field())

    def temperature_at(self, x: float, y: float) -> float:
        """
        Temperatur vid en kontinuerlig position.

        Bekvämlighetsomslag kring temperature_of_cell(). Ingen interpolation
        mellan celler: cellen är miljöns enhet, och alla organismer i samma
        cell möter samma temperatur — vilket floran redan gjorde. Kroppens
        temperatur integrerar över tid och jämnar ut steget mellan celler.

        Anroparen i agent.py bör på sikt slå upp cellen själv; det hör till
        Steg 5b när fauna blir store-first.
        """
        return self.temperature_of_cell(self.grid.cell_of(float(x), float(y)))


    # -------------------------
    # Abiotiska världspass
    # -------------------------
    def temperature_pass(self) -> None:
        """
        Uppdatera klimatet för aktuell tid.

        Returnerar inget fält. Att materialisera temperaturen per cell vore
        O(n_cells) arbete varje tick för information som ryms i **ett** tal,
        och ingen anropare behövde det. Läsare använder temperature_of_cell()
        eller temperature_of_cells().
        """
        self._update_temperature()

    # ---- glesa dödpooler ------------------------------------------------
    #
    # Förna och kadaver har samma form: massa och massviktad strukturandel per
    # cell, nollskild i en bråkdel av cellerna. Maskineriet nedan är gemensamt
    # och tar poolens arrayer som argument; `_detritus_*` och `_carcass_*` är
    # tunna omslag. Skälet att inte lägga poolerna i en klass är att `detritus`
    # och `detritus_structure` läses som attribut på `World` från ett dussin
    # ställen — viewer, records, invariantsvit, kalibrering.

    def _pool_activate(self, member, pending, cell: int) -> None:
        c = int(cell)
        if not member[c]:
            member[c] = True
            pending.append(c)

    def _pool_add(self, v, st, member, pending, cell: int,
                  amount: float, structure: float) -> None:
        """
        Lägg till massa i en cell och blanda in strukturandelen massviktat.
        """
        add = float(amount)
        if not (math.isfinite(add) and add > 0.0):
            return
        c = int(cell)
        old = float(v[c])
        tot = old + add
        if tot <= 0.0:
            return
        s_old = float(st[c])
        st[c] = (s_old * old + float(structure) * add) / tot
        v[c] = tot
        self._pool_activate(member, pending, c)

    def _pool_deactivate_if_empty(self, v, st, member, cell: int) -> bool:
        c = int(cell)
        if member[c] and float(v[c]) <= _DETRITUS_EPS:
            v[c] = 0.0
            st[c] = 0.0
            member[c] = False
            return True
        return False

    def _carcass_flush(self) -> None:
        if self._carcass_pending:
            add = np.asarray(self._carcass_pending, dtype=np.int32)
            self._carcass_active = np.concatenate((self._carcass_active, add))
            self._carcass_pending = []
            self._carcass_dirty = True
        if self._carcass_dirty:
            act = self._carcass_active
            if act.size:
                act = act[self._carcass_member[act]]
                self._carcass_active = np.unique(act).astype(np.int32, copy=False)
            self._carcass_dirty = False

    def _carcass_add(self, cell: int, amount: float, structure: float) -> None:
        self._pool_add(self.carcass, self.carcass_structure,
                       self._carcass_member, self._carcass_pending,
                       cell, amount, structure)

    def _carcass_deactivate_if_empty(self, cell: int) -> None:
        if self._pool_deactivate_if_empty(self.carcass, self.carcass_structure,
                                          self._carcass_member, cell):
            self._carcass_dirty = True

    @property
    def carcass_active_cells(self) -> np.ndarray:
        """Celler med nollskilt kadaver. Läsvy för pass och diagnostik."""
        self._carcass_flush()
        return self._carcass_active

    def _detritus_activate(self, cell: int) -> None:
        """Markera en cell som nollskild. Idempotent, O(1)."""
        c = int(cell)
        if not self._detritus_member[c]:
            self._detritus_member[c] = True
            self._detritus_pending.append(c)

    def _detritus_flush(self) -> None:
        """
        Slå ihop väntande aktiveringar och släpp avaktiverade celler.

        Dedupliceringen är nödvändig och inte defensiv: en cell som töms och
        fylls igen innan flush hinner köra hamnar annars två gånger i mängden,
        eftersom medlemsflaggan är sann både före och efter.
        """
        if self._detritus_pending:
            add = np.asarray(self._detritus_pending, dtype=np.int32)
            self._detritus_active = np.concatenate((self._detritus_active, add))
            self._detritus_pending = []
            self._detritus_dirty = True
        if self._detritus_dirty:
            act = self._detritus_active
            if act.size:
                act = act[self._detritus_member[act]]
                self._detritus_active = np.unique(act).astype(np.int32, copy=False)
            self._detritus_dirty = False

    def _detritus_add(self, cell: int, amount: float, structure: float) -> None:
        """
        Lägg till massa i en cells detritus och blanda in dess strukturandel
        massviktat. Se `_pool_add`.
        """
        self._pool_add(self.detritus, self.detritus_structure,
                       self._detritus_member, self._detritus_pending,
                       cell, amount, structure)

    def _detritus_deactivate_if_empty(self, cell: int) -> None:
        if self._pool_deactivate_if_empty(self.detritus, self.detritus_structure,
                                          self._detritus_member, cell):
            self._detritus_dirty = True

    @property
    def detritus_active_cells(self) -> np.ndarray:
        """Celler med nollskilt detritus. Läsvy för pass och diagnostik."""
        self._detritus_flush()
        return self._detritus_active

    @property
    def surface_level(self) -> np.ndarray:
        """
        Fri yta, elevation + water. Härledd: beräknas vid läsning i stället för
        varje tick, eftersom inget systempass läser den ännu. Kostar O(n_cells)
        per anrop — hydro ska räkna på `water` direkt.
        """
        return (self.water + np.float32(self.elevation)).astype(np.float32, copy=False)

    @property
    def submerged(self) -> np.ndarray:
        """Bool per cell, water över tröskeln. Härledd, se surface_level."""
        return self.water > np.float32(self.WP.submerged_threshold)

    def _update_lake_levels(self) -> None:
        dr = self.drainage
        if dr is None or dr.n_lakes == 0:
            return
        hydro_lake_levels(
            self.lake_storage, dr.lake_start, dr.lake_cells, dr.lake_vol,
            np.asarray(self.elevation, dtype=np.float64),
            self._lake_level, self._lake_area,
        )

    def _rain_now(self) -> float:
        """
        Nederbörd per månad, före orografi.

        Clausius–Clapeyron i förenklad form: mängden fördubblas per
        `rain_T_doubling` grader. Den är därmed en funktion av `T_air`, som bär
        årstiden — en regnperiod på sommaren och en torr vinter faller ut utan
        att någon av dem kodas. Den rumsliga delen föll med latituden; kvar
        rumsligt är `_oro`.
        """
        WP = self.WP
        dT = float(self.T_air) - self.T_mean
        return float(WP.rain_base) * 2.0 ** (dT / max(1e-9, float(WP.rain_T_doubling)))

    def hydro_pass(self) -> tuple[float, float]:
        """
        Vattnet i jämvikt över terrängen, eller det gamla forcing-uttrycket i
        en platt värld.

        Returnerar (tillfört, borttaget) för ledgern.
        """
        if self.drainage is not None:
            return self._hydro_pass_terrain()
        return self._hydro_pass_uniform()

    def _hydro_pass_terrain(self) -> tuple[float, float]:
        WP = self.WP
        dt = float(WP.dt)
        dr = self.drainage

        # 1. Marken, punktvis. Forcingen räknas inne i kärnan ur bandprofilerna
        #    och den statiska orografin — se hydro.soil_pass.
        T0 = float(WP.T0)
        T_span = max(1e-9, float(WP.T1) - T0)
        hydro_soil_pass(
            self.soil_water,
            self._rain_now() * dt,
            float(self.T_air), self._T_offset_arr,
            T0, T_span, self._oro,
            float(WP.et_max) * dt,
            float(WP.soil_capacity),
            float(WP.baseflow_frac) * dt,
            dr.sea, dr.lake_id, self.water,
            float(WP.submerged_threshold),
            self._runoff, self._hydro_acc,
        )
        added = float(self._hydro_acc[0])
        removed = float(self._hydro_acc[1])

        # 2. Sjöytan avdunstar utan markens torrhetsbroms. Per bassäng, alltså
        #    några hundra tal.
        if dr.n_lakes:
            et_lake = (self.growth_gate_of_cells(dr.lake_outlet).astype(np.float64)
                       * float(WP.et_max) * float(WP.lake_evap_mult) * dt)
            loss = np.minimum(self.lake_storage, et_lake * self._lake_area[:dr.n_lakes])
            self.lake_storage -= loss
            removed += float(loss.sum())

        # 3. Routing med sjöarna som magasin. Ett svep, varje cell en gång.
        hydro_route(
            self._runoff, dr.flow_to, dr.flow_order, dr.outlet_lake, dr.lake_id,
            self.lake_storage, dr.lake_cap, dr.sea,
            self.soil_water, float(WP.soil_capacity),
            float(WP.reinfiltration_max) * dt,
            self.discharge, self._hydro_acc,
        )
        # Det som når havet lämnar landvattenbudgeten. Havet är en absorberande
        # rand och ackumulerar inget tryck mot omgivningen — manifestets
        # hydrologiska randregim.
        removed += float(self._hydro_acc[0])

        # 4. Nivåer och härledda fält.
        self._update_lake_levels()
        # Fåra eller sluttning: tröskeln uttrycks i cellers medelavrinning, så
        # att den betyder samma sak oavsett nederbörd och världsstorlek.
        q_min = float(WP.channel_min_upslope) * added / max(1.0, float(self._n_land))
        hydro_derive_water(
            np.asarray(self.elevation, dtype=np.float32), dr.sea, dr.lake_id,
            self._lake_level, self.discharge, dr.slope,
            float(WP.channel_k), float(WP.channel_exp),
            float(WP.channel_slope_floor), q_min, self.water,
        )
        self.flow_strength = float(np.mean(self.discharge, dtype=np.float64))

        self._water_added_total += added
        self._water_lost_total += removed
        return added, removed

    def _hydro_pass_uniform(self) -> tuple[float, float]:
        """
        Minimalt forcing-uttryck för en platt värld. Oförändrat sedan fas 1.5.

        Forcing-termerna är rumsligt konstanta, så nettotillskottet per tick är
        ett tal och inte ett fält.
        """
        dt = float(self.WP.dt)
        dwater = dt * (
            float(self.rain_input)
            + float(self.spring_input)
            - float(self.infiltration)
            - float(self.evaporation)
        )

        if dwater == 0.0:
            return 0.0, 0.0

        w = self.water
        if dwater > 0.0:
            # Inget klipps bort: varje cell ökar lika mycket.
            w += np.float32(dwater)
            n = float(w.shape[0])
            return dwater * n, 0.0

        # Negativt tillskott: celler med mindre vatten än uttaget klipps mot noll,
        # så det faktiska uttaget är summan av min(water, -dwater).
        removed = float(np.sum(np.minimum(w, np.float32(-dwater)), dtype=np.float64))
        np.maximum(w + np.float32(dwater), np.float32(0.0), out=w)
        return 0.0, removed

    def water_stock(self) -> float:
        """
        Vattnet som lagras: markvatten plus sjömagasin. Kanalvatten är i
        transit och lagras inte — det följer av att hydro löser jämvikt.
        """
        if self.drainage is None:
            return float(np.sum(self.water, dtype=np.float64))
        return (float(np.sum(self.soil_water, dtype=np.float64))
                + float(np.sum(self.lake_storage, dtype=np.float64)))

    def _build_elevation_gradient(self):
        """
        Höjdens gradient per cell, som två statiska fält.

        Terrängen ändras inte, så gradienten gör det inte heller: den byggs en
        gång och kostar noll per tick. Det är samma kadensresonemang som
        `elevation` självt — ett fält som inte ändras ska inte räknas om.

        Riktningarna kommer från `Grid.neighbor_dx/dy`. Ingen kod här vet vad
        en rad eller kolumn är; gradienten är en viktad summa över
        grannringen, och den är därmed giltig för vilken cellform som helst.
        """
        if self.WP.terrain is None:
            return 0.0, 0.0
        z = np.asarray(self.elevation, dtype=np.float64)
        idx = np.asarray(self.grid.neighbor_idx, dtype=np.int64)
        dx = np.asarray(self.grid.neighbor_dx, dtype=np.float64)
        dy = np.asarray(self.grid.neighbor_dy, dtype=np.float64)
        dist = np.asarray(self.grid.neighbor_dist, dtype=np.float64)
        k = idx.shape[1]

        gx = np.zeros(z.shape[0], dtype=np.float64)
        gy = np.zeros(z.shape[0], dtype=np.float64)
        for j in range(k):
            dz = (z[idx[:, j]] - z) / dist[j]
            gx += dz * dx[j]
            gy += dz * dy[j]
        # Minsta-kvadratlösningen för en plan yta över en isotrop grannring är
        # summan gånger 2/k; faktorn gör gradienten till en verklig lutning och
        # inte bara en riktningsindikator.
        gx *= 2.0 / float(k)
        gy *= 2.0 / float(k)
        return (gx.astype(np.float32, copy=False), gy.astype(np.float32, copy=False))

    def slope_along(self, cell: int, heading: float) -> float:
        """
        Lutningen i en given riktning: positiv uppför, negativ nedför.

        Bekvämlighetsomslag kring de statiska gradientfälten. Kostar två
        uppslag och en skalärprodukt.
        """
        gx = self._grad_x
        if not isinstance(gx, np.ndarray):
            return 0.0
        c = int(cell)
        return float(gx[c] * math.cos(heading) + self._grad_y[c] * math.sin(heading))

    def _build_weathering(self) -> np.ndarray | None:
        """
        Vittringsvikt per cell, normerad till medelvärde ett över hela världen.

        Havet får noll: berggrund under vatten vittrar inte till markvatten som
        någon växt når. Vikten flyttas till land, så den totala tillförseln är
        exakt densamma som med ett konstant fält och jämviktsidentiteten står
        orörd. Det är samma disciplin som ljuset fick — inför strukturen, rör
        inte nivån.
        """
        dr = self.drainage
        if dr is None:
            return None
        land = ~dr.sea
        if not land.any():
            return None
        s = np.asarray(dr.slope, dtype=np.float64)
        # Lutningen är noll på plana ytor och inne i sjöar. Ett golv gör att
        # även en flack cell vittrar något, annars vore en slätt steril.
        base = (np.maximum(s, 1e-3) ** float(self.WP.weathering_slope_exp))
        w = np.where(land, base, 0.0)
        tot = float(w.sum())
        if tot <= 0.0:
            return None
        return (w * (float(self.grid.n_cells) / tot)).astype(np.float64, copy=False)

    def nutrient_input_pass(self) -> float:
        """
        Vittring. Rumsligt konstant i en platt värld, lutningsstyrd med terräng.
        """
        add = float(self.WP.dt) * float(self.WP.nutrient_input)
        if add == 0.0:
            return 0.0
        if self._weathering is None:
            self.nutrient += add
        else:
            self.nutrient += add * self._weathering
        total = add * float(self.grid.n_cells)
        self._nutrient_added_total += total
        return total

    def sediment_pass(self) -> tuple[float, float]:
        """
        Partikulär transport av förna nedströms. Returnerar (till havet, flyttat).

        Kontraktet för det glesa fältet hålls: celler som får material
        aktiveras, celler som töms under epsilon nollställs exakt och lämnar
        mängden. Näringen som når havet bokförs som förlust, eftersom havet är
        sänka för varje väg dit.
        """
        dr = self.drainage
        if dr is None or float(self.WP.sediment_rate) <= 0.0:
            return 0.0, 0.0
        self._detritus_flush()

        hydro_sediment(
            self.detritus, self.detritus_structure, dr.slope,
            dr.flow_to, dr.flow_order, dr.sea,
            float(self.WP.sediment_rate) * float(self.WP.dt),
            max(1e-9, float(self.WP.sediment_slope_ref)),
            float(self.WP.sediment_struct_mobility),
            float(_DETRITUS_EPS),
            self._sed_in_lab, self._sed_in_str, self._sed_acc,
        )
        hydro_deposit(self.detritus, self.detritus_structure,
                      self._sed_in_lab, self._sed_in_str,
                      float(_DETRITUS_EPS), self._sed_changed)

        # Medlemskapet: celler som tagit emot material blir aktiva, celler som
        # tömts lämnar mängden. Utan det bryts kontraktet att en inaktiv cell
        # är exakt noll, och `check_sparse_fields` fångar det direkt.
        gained = np.flatnonzero(self._sed_changed & ~self._detritus_member)
        if gained.size:
            self._detritus_member[gained] = True
            self._detritus_pending.extend(int(c) for c in gained)
            self._detritus_dirty = True
        empt = self._detritus_active[self.detritus[self._detritus_active] <= _DETRITUS_EPS]
        if empt.size:
            self.detritus[empt] = 0.0
            self.detritus_structure[empt] = 0.0
            self._detritus_member[empt] = False
            self._detritus_dirty = True

        to_sea = float(self._sed_acc[0])
        to_sea_str = float(self._sed_acc[1])
        if to_sea > 0.0:
            lab = to_sea - to_sea_str
            self._nutrient_lost_total += (lab * NUTRIENT_PER_KG_LABILE
                                          + to_sea_str * NUTRIENT_PER_KG_STRUCT)
        return to_sea, float(self._sed_acc[2])

    def leaching_pass(self) -> tuple[float, float]:
        """
        Urlakning nedströms. Returnerar (nådde havet, flyttad mängd).

        Körs efter hydro, eftersom den läser den avrinning hydro just räknat.
        """
        dr = self.drainage
        if dr is None:
            return 0.0, 0.0
        hydro_leach(
            self.nutrient, self.soil_water, self._runoff,
            dr.flow_to, dr.flow_order, dr.lake_id, dr.sea,
            float(self.WP.leach_efficiency), self._leach_acc,
        )
        lost = float(self._leach_acc[0])
        self._nutrient_lost_total += lost
        return lost, float(self._leach_acc[1])

    def add_nutrient(self, cell: int, amount: float) -> float:
        """
        Återför näring till en cell. Returnerar tillförd mängd.

        Detta är recirkulation, inte tillförsel: kvävehaltigt avfall från
        organismernas förbränning och vävnadsomsättning. Den räknas därför
        inte in i `_nutrient_added_total`, som bara bokför extern forcing.
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0
        c = int(cell)
        if c < 0 or c >= int(self.grid.n_cells):
            return 0.0
        self.nutrient[c] += amt
        return amt

    def take_soil_water(self, cells, amounts) -> float:
        """
        Dra transpirerat vatten ur marken och bokför det som förlust.

        Symmetriskt med `take_nutrient`: biologin äger inte fysiken, men den
        får konsumera världens fält genom världen. Skillnaden är att näringen
        cirkulerar medan vattnet lämnar systemet — transpirerat vatten går till
        atmosfären, precis som avdunstningen i hydro-passet.
        """
        if self.drainage is None:
            return 0.0
        c = np.asarray(cells, dtype=np.int64)
        a = np.asarray(amounts, dtype=np.float64)
        if c.size == 0:
            return 0.0
        np.subtract.at(self.soil_water, c, a)
        total = float(a.sum())
        self._water_lost_total += total
        return total

    def book_transpiration(self, amount: float) -> None:
        """
        Bokför vatten som redan dragits ur `soil_water` av tillväxtkärnan.

        Kärnan får inte göra världsanrop, så den muterar fältet på plats precis
        som den gör med `nutrient`, och ledgern uppdateras här. Transpirerat
        vatten lämnar systemet till atmosfären, som avdunstningen.
        """
        a = float(amount)
        if a > 0.0:
            self._water_lost_total += a

    def take_nutrient(self, cell: int, amount: float) -> float:
        """Ta upp till `amount` kg näring ur en cell. Returnerar faktiskt uttag."""
        c = int(cell)
        avail = float(self.nutrient[c])
        got = amount if amount < avail else avail
        if got <= 0.0:
            return 0.0
        self.nutrient[c] = avail - got
        return float(got)

    def transport_pass(self) -> float:
        """
        Diffusion av lösta ämnen över topologiska grannar.

        Tvåstegsmetod: flödena beräknas ur tillståndet vid passets början och
        appliceras simultant som nettoförändringar. Diskret laplacian via
        grannmatrisen, alltså geometriagnostisk — samma kod gäller för fyra
        eller sex grannar.

        Massbevarandet är exakt och inte approximativt. Grannrelationen är
        ömsesidig och graden konstant, så varje cell förekommer i exakt k
        grannlistor och summan av laplacianen över hela världen är noll.

        Kadensmässigt gör det här `nutrient` tätt dynamiskt — den fjärde
        klassen i docs/varldens-kadensmodell.md, och den enda där fullt svep
        är genuint motiverat. Näring som diffunderar har inget glest stöd.

        Returnerar summan av absolut omfördelad mängd, som diagnostik.
        """
        D = float(self.WP.nutrient_diffusion)
        dt = float(self.WP.dt)
        if D <= 0.0 or dt <= 0.0:
            return 0.0

        n = self.nutrient
        idx = self.grid.neighbor_idx
        k = int(self.grid.neighbor_count)

        # Laplacian: grannarnas summa minus k gånger egen halt.
        lap = n[idx].sum(axis=1, dtype=np.float64) - float(k) * n

        delta = lap * (D * dt / k)
        n += delta

        return float(np.sum(np.abs(delta))) * 0.5

    def _decompose_pool(self, v, st, member, active, rate: float) -> tuple:
        """
        Nedbrytning av en dödpool. Returnerar (nedbruten massa, frisatt näring).

        Nedbrytning per fraktion: labilt och strukturellt material bryts ner med
        var sin takt, räknade ur massa och strukturandel utan ett andra fält.
        Att sakta ner hela massan i stället lät strukturandelen skena — bara det
        labila försvann, och kvarvarande material blev asymptotiskt ren struktur.
        """
        if active.size == 0:
            return 0.0, 0.0, False
        dt = float(self.WP.dt)
        d = v[active]
        sf = st[active]

        lab = d * (1.0 - sf)
        stru = d * sf
        k_lab = dt * float(rate) * float(DECAY_SCALE_LABILE)
        k_str = dt * float(rate) * float(DECAY_SCALE_STRUCT)
        d_lab = np.minimum(lab, lab * k_lab)
        d_str = np.minimum(stru, stru * k_str)

        lab_new = lab - d_lab
        stru_new = stru - d_str
        new = lab_new + stru_new

        decayed = float(np.sum(d_lab + d_str, dtype=np.float64))
        released = float(
            np.sum(d_lab, dtype=np.float64) * NUTRIENT_PER_KG_LABILE
            + np.sum(d_str, dtype=np.float64) * NUTRIENT_PER_KG_STRUCT
        )
        retained = 1.0 - float(self.WP.nutrient_loss_frac)
        np.add.at(self.nutrient, active,
                  (d_lab * NUTRIENT_PER_KG_LABILE
                   + d_str * NUTRIENT_PER_KG_STRUCT) * retained)

        with np.errstate(invalid="ignore", divide="ignore"):
            st_new = np.where(new > 0.0, stru_new / np.maximum(new, 1e-300), 0.0)
        st[active] = np.clip(st_new, 0.0, 1.0)

        # Celler under tröskeln nollställs exakt och lämnar den aktiva mängden,
        # så att kontraktet "inaktiv cell är noll" håller.
        emptied = False
        empty = new <= float(_DETRITUS_EPS)
        if empty.any():
            new[empty] = 0.0
            st[active[empty]] = 0.0
            member[active[empty]] = False
            emptied = True
        v[active] = new
        return decayed, released, emptied

    def decomposition_pass(self) -> tuple[float, float]:
        """
        Nedbrytning av förna och kadaver, var och en med sin egen takt.

        Kadaver bryts ned åtta gånger snabbare. Det är inte en detalj: så länge
        de två låg i samma pool och samma takt kunde ett bestånd i kollaps leva
        på sina egna döda i månader — i p114 åt faunan 4 594 kg kadaver mot
        3 248 kg flora, med andelen stigande från 20 till 86 procent genom
        förloppet. En stock som ruttnar bort kan inte bära den återkopplingen.
        """
        self._detritus_flush()
        self._carcass_flush()

        d_dec, d_rel, d_empty = self._decompose_pool(
            self.detritus, self.detritus_structure, self._detritus_member,
            self._detritus_active, float(self.WP.detritus_decay))
        if d_empty:
            self._detritus_dirty = True

        c_dec, c_rel, c_empty = self._decompose_pool(
            self.carcass, self.carcass_structure, self._carcass_member,
            self._carcass_active, float(self.WP.carcass_decay))
        if c_empty:
            self._carcass_dirty = True

        released = d_rel + c_rel
        retained = 1.0 - float(self.WP.nutrient_loss_frac)
        self._nutrient_lost_total += released * (1.0 - float(retained))
        return d_dec + c_dec, released * float(retained)

    def update_flux(
        self,
        *,
        dM_growth: float = 0.0,
        dM_wither: float = 0.0,
        dM_detritus_decay: float = 0.0,
        dM_nutrient_from_detritus: float = 0.0,
        dM_water_added: float = 0.0,
        dM_water_removed: float = 0.0,
        dM_transport: float = 0.0,
    ) -> None:
        """
        Uppdatera världens öppna-system-ledger för senaste tick.
        """
        P = self.WP
        # Nominella energiskalor för ledgern; se anmärkningen i Population.
        e_lab = float(getattr(P, "E_labile_J_per_kg", 9.302e6))
        e_plant = e_lab * (1.0 - 0.57)
        e_carc = e_lab * (1.0 - 0.25)

        self.last_flux = {
            "dM_growth": max(0.0, dM_growth),
            "dM_wither": max(0.0, dM_wither),
            "dM_decay": max(0.0, dM_detritus_decay),
            "dM_detritus_decay": max(0.0, dM_detritus_decay),
            "dM_nutrient_from_detritus": max(0.0, dM_nutrient_from_detritus),
            "dM_water_added": max(0.0, dM_water_added),
            "dM_water_removed": max(0.0, dM_water_removed),
            "dM_transport": float(dM_transport),
            "E_in_growth": max(0.0, dM_growth) * e_plant,
            "E_loss_wither": max(0.0, dM_wither) * e_plant,
            "E_loss_decay": max(0.0, dM_detritus_decay) * e_carc,
        }

    def step(self) -> None:
        dt = float(self.WP.dt)

        self.temperature_pass()
        self.nutrient_input_pass()
        dM_water_added, dM_water_removed = self.hydro_pass()
        dM_transport = self.transport_pass()
        # Urlakningen efter hydro och före nedbrytningen: den läser den
        # avrinning hydro just räknat, och näring som frigörs i den här ticken
        # ska ligga kvar till nästa i stället för att sköljas ut samma tick den
        # mineraliserades.
        self.leaching_pass()
        # Partiklarna efter det lösta: båda drivs av samma avrinning, och
        # ordningen mellan dem spelar ingen roll för bevarandet — men förnan
        # ska flyttas innan den bryts ned, annars mineraliserar den där den låg
        # i stället för dit vattnet tog den.
        self.sediment_pass()
        dM_detritus_decay, dM_nutrient_from_detritus = self.decomposition_pass()
        self.update_flux(
            dM_growth=0.0,
            dM_wither=0.0,
            dM_detritus_decay=dM_detritus_decay,
            dM_nutrient_from_detritus=dM_nutrient_from_detritus,
            dM_water_added=dM_water_added,
            dM_water_removed=dM_water_removed,
            dM_transport=dM_transport,
        )

        self.t += dt

    # -------------------------
    # Sampling (renodlad)
    # -------------------------
    def sample_carcass(self, x: float, y: float) -> float:
        """
        All död massa i cellen, förna plus kadaver.

        Perceptionen skiljer inte på de två. Ett djur ser att här ligger något
        dött; vad det är värt visar sig när det äter. Att ge kadavret en egen
        sinneskanal vore en större ändring än den här defekten motiverar.
        """
        c = self.grid.cell_of(float(x), float(y))
        return float(self.detritus[c]) + float(self.carcass[c])

    def sample_many_carcass(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        outC: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vektoriserad sampling av detritus i de celler punkterna faller i.

        Ingen interpolation: fältet är styckvis konstant per cell, så en
        organism läser cellens värde. `tmp` togs bort med den bilinjära
        mellanlagringen.
        """
        cells = self.grid.cell_of_many(xs, ys)
        if outC is None:
            return (self.detritus[cells] + self.carcass[cells]).astype(
                np.float32, copy=False)
        outC[...] = (self.detritus[cells] + self.carcass[cells]).reshape(
            np.shape(outC))
        return outC

    def sample_flora_local(self, x: float, y: float) -> float:
        hook = getattr(self, "sample_flora_local_hook", None)
        if hook is None:
            return 0.0
        return float(hook(x, y))
    
    def sample_food_local(self, x: float, y: float) -> tuple[float, float]:
        """
        Returns (B_kg, detritus_kg) from current world interfaces:
          - B via flora provider
          - detritus via world detritus field
        """
        B = float(self.sample_flora_local(x, y))
        C = float(self.sample_carcass(x, y))
        return B, C

    def sample_flora_rays(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        hook = getattr(self, "sample_flora_rays_hook", None)
        if hook is None:
            return np.zeros_like(xs, dtype=np.float32)
        return hook(xs, ys)
        
    # -------------------------
    # Consumption + carcass
    # -------------------------
    def _consume_from_field(self, field: np.ndarray, x: float, y: float, amount: float) -> tuple[float, float]:
        """
        Konsumera upp till `amount` kg ur `field` i den cell (x, y) faller i.

        Returnerar (kg, energi_J). Energin följer av materialets strukturandel:
        strukturmaterial lagrar ingen användbar energi, så ett kilo segt
        substrat är värt mindre än ett kilo mjukt. Konsumentens
        matsmältningsverkningsgrad tillämpas hos konsumenten, inte här.

        Organismen befinner sig i en cell och äter ur den. Vill den åt en
        granncells innehåll måste den flytta sig dit — position har betydelse.
        `field` är en platt per-cell-array; funktionen är därmed generell och
        kan bära nutrient-upptaget i Steg 3.
        """
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0

        xf = float(x)
        yf = float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            return 0.0, 0.0

        cell = int(self.grid.cell_of(xf, yf))
        avail = float(field[cell])
        if not math.isfinite(avail) or avail <= 1e-12:
            return 0.0, 0.0

        got = amt if amt < avail else avail
        field[cell] = (avail - got) if field.dtype == np.float64 else np.float32(avail - got)

        struct = 0.0
        if field is self.detritus:
            struct = float(self.detritus_structure[cell])
            self._detritus_deactivate_if_empty(cell)
        elif field is self.carcass:
            struct = float(self.carcass_structure[cell])
            self._carcass_deactivate_if_empty(cell)

        energy = got * float(self.WP.E_labile_J_per_kg) * (1.0 - struct)
        return float(got), float(energy)

    def consume_food(self, x: float, y: float, amount: float,
                     diet: float = 0.5,
                     reach: int = 1) -> Tuple[float, float, float, float]:
        """
        Reservkonsumtion i World: bara detritus. Levande föda hanteras av
        Population via consume_food_hook.

        Returnerar (kg_levande, kg_detritus, energi_levande_J, energi_detritus_J).
        """
        hook = getattr(self, "consume_food_hook", None)
        if hook is not None:
            return hook(x, y, amount, diet, reach)
    
        amt = float(amount)
        if not math.isfinite(amt) or amt <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
    
        got_d, e_d = self._consume_from_field(self.detritus, x, y, amt)
        return 0.0, float(got_d), 0.0, float(e_d)

    def excrete_at(self, x: float, y: float, amount_kg: float, structure: float) -> float:
        """
        Återför icke assimilerad massa till cellen som detritus.

        Till skillnad från add_carcass sprids inget: exkrementet hamnar där
        organismen står. Returnerar tillförd massa för ledgern.
        """
        amt = float(amount_kg)
        if not (math.isfinite(amt) and amt > 0.0):
            return 0.0
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            return 0.0
        cell = int(self.grid.cell_of(float(x), float(y)))
        self._detritus_add(cell, amt, float(structure))
        self._dM_excreted += amt
        return amt

    def excrete_cells(self, cells, amounts, structures) -> float:
        """
        Vektoriserad exkretion: massa till detritus i angivna celler.

        Samma semantik som `excrete_at`, men för många deponeringar på en gång.
        Florans förnafall berör varje levande planta varje tick, och ett
        Python-anrop per individ skulle kosta mer än hela tillväxtpasset.
        Strukturandelen blandas in massviktat per cell, precis som i
        `_detritus_add`.
        """
        c = np.asarray(cells, dtype=np.int64)
        a = np.asarray(amounts, dtype=np.float64)
        s = np.asarray(structures, dtype=np.float64)
        keep = np.isfinite(a) & (a > 0.0) & (c >= 0) & (c < int(self.grid.n_cells))
        if not np.any(keep):
            return 0.0
        c = c[keep]; a = a[keep]; s = s[keep]

        # Aggregering via bincount över n_cells i stället för np.unique med
        # return_inverse. Unique sorterar; bincount gör samma jobb med ett svep.
        # Uppmätt på 300 000 rader: 19,9 ms mot 1,66 ms, tolv gånger.
        nc = int(self.grid.n_cells)
        add_all = np.bincount(c, weights=a, minlength=nc)[:nc]
        addw_all = np.bincount(c, weights=a * s, minlength=nc)[:nc]
        uniq = np.flatnonzero(add_all > 0.0)
        if uniq.size == 0:
            return 0.0
        add = add_all[uniq]
        addw = addw_all[uniq]

        old = np.asarray(self.detritus)[uniq].astype(np.float64)
        s_old = np.asarray(self.detritus_structure)[uniq].astype(np.float64)
        tot = old + add
        safe = tot > 0.0
        s_new = s_old.copy()
        s_new[safe] = (s_old[safe] * old[safe] + addw[safe]) / tot[safe]
        self.detritus[uniq] = tot.astype(self.detritus.dtype, copy=False)
        self.detritus_structure[uniq] = s_new.astype(self.detritus_structure.dtype, copy=False)

        fresh = uniq[~self._detritus_member[uniq]]
        if fresh.size:
            self._detritus_member[fresh] = True
            self._detritus_pending.extend(int(v) for v in fresh)

        total = float(add.sum())
        self._dM_excreted += total
        return total

    def add_carcass(self, x: float, y: float, amount_kg: float, rad: int = 3,
                    structure: float = 0.45) -> None:
        """
        Deponera kadavermassa i kadaverpoolen (kg/cell).

        Spridningen över `rad` celler är deponering och inte transport: en kropp
        skingras där den faller. Kadaver diffunderar inte efteråt.
        """
        amt = float(amount_kg)
        if not math.isfinite(amt) or amt <= 0.0:
            return
    
        r = int(rad)
        if r < 1:
            r = 1
    
        center = int(self.grid.cell_of(float(x), float(y)))
    
        # Topologisk spridning: cellerna inom r steg, viktade med topologiskt
        # avstånd. Ersätter den kvadratiska dx/dy-kärnan med euklidiskt avstånd,
        # som var det sista geometriantagandet i world.py.
        cells = self.grid.cells_within(center, r)
        if not cells:
            self._carcass_add(center, amt, structure)
            return
    
        sigma = max(0.75, 0.5 * r)
        inv2sig2 = 1.0 / (2.0 * sigma * sigma)
    
        weights = []
        wsum = 0.0
        for cell in cells:
            d = float(self.grid.distance(center, int(cell)))
            w = math.exp(-(d * d) * inv2sig2)
            weights.append((int(cell), w))
            wsum += w
    
        if wsum <= 1e-12:
            self._carcass_add(center, amt, structure)
            return
    
        scale = amt / wsum
        for cell, w in weights:
            self._carcass_add(cell, scale * w, structure)


