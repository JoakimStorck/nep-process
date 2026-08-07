"""
Klimatet ur världens position på planeten.

**Det här är fysiklager, inte världslager.** Modulen äger inget tillstånd och
rör inga cellfält. Den definierar en regel — givet var världen ligger, vilket
klimat råder där — och världslagret tillämpar den en gång vid världens
tillkomst.

**Varför en position och inte fyra tal.** Världen är åtta tusen till en miljon
celler, alltså ett landskap på någon kilometer till några mil. Planeten
fortsätter utanför randen, och det som kommer därifrån — solens gång,
årstidens djup, luftmassornas tröghet — kan inte simuleras inifrån. Att ange
`latitud` och `kontinentalitet` är att beskriva randvillkoret i stället för
att gissa dess följder. Ett experiment blir då en flyttning, inte fyra nya
parametrar.

**Det är motsatsen till den latitudgradient modellen en gång hade.** Världen
ligger *på* en breddgrad — den spänner inte över flera. Vid tio meters celler
ger även 4096 rader 0,11 graders meridionell gradient, mot de trettio modellen
kodade. Latituden är därför en skalär i fysiklagret och inte ett fält över
celler. Se `docs/varldens-skala.md`.

**Vad som är exakt och vad som är anpassat.**

Dagslängd och insolation följer ur latitud och solens deklination som ren
astronomi, utan en enda kalibrerad konstant. De hör hemma här men införs
först när de har en läsare — fotoperioden kommer med florans ljusbudget.

Temperaturen går inte att härleda ur latitud allena. Köpenhamn och Moskva
ligger båda på 55–56 grader nord och skiljer 9 mot 5,8 grader i årsmedel och
8 mot 14,5 i årsamplitud; skillnaden är havets termiska tröghet och inte
solen. Därför två variabler, och därför en **anpassning mot mätdata i stället
för en härledning**. Det står här i klartext eftersom manifestet kräver att en
kalibrerad fri parameter ska vara utmärkt som sådan.

**Underlaget** är sjutton stationer med känd klimatnormal, spridda från
Singapore till Jakutsk och från Azorerna till Ulan Bator. Varje station är
höjdkorrigerad till havsnivå med 6,5 grader per kilometer innan anpassningen,
eftersom modellens egen lapse rate lägger på höjden separat — utan den
korrigeringen skulle Ulan Bators tolvhundra meter räknas två gånger.
Residualen är 1,96 grader i årsmedel och 2,13 i amplitud, vilket är storleken
på det en position inte kan veta: havsströmmar, molnighet och regnskuggor.

**Kända svagheter**, som inte ska upptäckas på nytt:

- Maritim amplitud på hög latitud underskattas. Bergen har 6,8 där formeln ger
  3,9. Det är anpassningens tunnaste hörn — få stationer är både kustnära och
  nordliga.
- `kontinentalitet` nära 1 vid ekvatorn extrapolerar utanför underlaget.
  Manaus och Kinshasa håller emot men är bara två punkter.
- Nederbörden härleds **inte**. Sahara och monsun-Indien ligger båda kring
  tjugo grader nord och skiljer tre tusen millimeter — spridningen inom en
  breddgrad är större än signalen mellan breddgrader, och nederbörden förblir
  därför världens egen parameter.
"""

from __future__ import annotations

import math

# Jordens axellutning. Används inte ännu — deklinationen kommer med
# fotoperioden — men står här eftersom den är planetens egenskap och inte
# klimatets.
OBLIQUITY_DEG: float = 23.44

# --- Anpassade koefficienter -------------------------------------------------
#
# T_mean = T0 + T1*(phi/100)^2 + k*(T2 + T3*(phi/100)^2)
#
# Kvadraten i latituden är den vanliga formen för jordens meridionella
# temperaturprofil. Kontinentalitetens bidrag är också kvadratiskt, och med
# motsatt tecken vid hög latitud: inlandet är varmare än kusten vid ekvatorn
# och kallare vid polen, vilket är samma sak sett från två håll — havet dämpar,
# och det som dämpas är avvikelsen från medlet.
_T_MEAN_C0: float = 25.93
_T_MEAN_C1: float = -49.20
_T_MEAN_K0: float = 4.83
_T_MEAN_K1: float = -50.35

# T_amp = (A0 + A1*k) * sin|phi|
#
# Sinusformen framför kvadraten: amplituden växer snabbt redan vid låga
# latituder och planar av mot polen, vilket är vad stationerna visar.
_T_AMP_A0: float = 4.51
_T_AMP_A1: float = 21.73

# Termisk eftersläpning i månader. Havet lagrar värme och släpper den sent, så
# den varmaste månaden ligger efter solståndet — två månader vid kusten, en i
# inlandet. Det är samma tröghet som styr amplituden, sedd i tidsdomänen.
_LAG_MARITIME: float = 2.0
_LAG_CONTINENTAL: float = 1.0


def arsmedeltemperatur(latitud: float, kontinentalitet: float) -> float:
    """Årsmedeltemperatur vid havsnivå, grader Celsius."""
    p2 = (abs(float(latitud)) / 100.0) ** 2
    k = _klamp01(kontinentalitet)
    return _T_MEAN_C0 + _T_MEAN_C1 * p2 + k * (_T_MEAN_K0 + _T_MEAN_K1 * p2)


def arsamplitud(latitud: float, kontinentalitet: float) -> float:
    """
    Halva skillnaden mellan varmaste och kallaste månad, grader.

    Noll vid ekvatorn oavsett kontinentalitet, vilket är riktigt: utan
    årstidsväxling i solhöjd finns ingenting för trögheten att dämpa.
    """
    k = _klamp01(kontinentalitet)
    return (_T_AMP_A0 + _T_AMP_A1 * k) * math.sin(math.radians(abs(float(latitud))))


def faseftersläpning(kontinentalitet: float) -> float:
    """Termisk eftersläpning i månader, mellan sol och temperatur."""
    k = _klamp01(kontinentalitet)
    return _LAG_MARITIME + (_LAG_CONTINENTAL - _LAG_MARITIME) * k


def halvklotstecken(latitud: float) -> float:
    """
    +1 på norra halvklotet, -1 på det södra.

    Årstiden inverteras söder om ekvatorn — det är hela skillnaden, eftersom
    både medel och amplitud beror på latitudens belopp. Vid exakt noll är
    amplituden ändå noll, så tecknet saknar verkan där.
    """
    return -1.0 if float(latitud) < 0.0 else 1.0


def beskriv(latitud: float, kontinentalitet: float) -> str:
    """En rad för startutskriften. Positionen och vad den ger."""
    Tm = arsmedeltemperatur(latitud, kontinentalitet)
    Ta = arsamplitud(latitud, kontinentalitet)
    return (
        f"latitud {float(latitud):+.1f}° kontinentalitet {_klamp01(kontinentalitet):.2f}"
        f"  ->  årsmedel {Tm:.1f} °C  amplitud ±{Ta:.1f}"
        f"  (kallast {Tm - Ta:.1f}, varmast {Tm + Ta:.1f})"
        f"  eftersläpning {faseftersläpning(kontinentalitet):.1f} mån"
    )


def _klamp01(x: float) -> float:
    v = float(x)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
