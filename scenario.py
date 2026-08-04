"""
Scenario: en körnings utgångsläge som data i stället för som kommandorad.

Kommandoraden hade vuxit till fjorton flaggor varav sex bara beskrev
scenariot, och tre av dem — `nutrient_input`, `nutrient_init`,
`detritus_init` — skalar alltid tillsammans men räknades fram för hand vid
varje körning. De hann gå isär två gånger.

Två saker den här formen ger utöver kortare kommandorad:

**Bördigheten blir ett tal.** Jämvikten skalar linjärt med näringsflödet, så
faktor 4 betyder fyra gånger alla tre. Det går inte längre att sätta dem
inkonsekvent.

**Insättningen kan uttryckas som en princip.** `fauna.insatts_vid:
"jamvikt"` i stället för ett tickvärde jag gissat fram — det felet gjorde
både p87 och p97 ogiltiga, eftersom faunan mötte en halvfärdig flora.

Filen skrivs till körningens katalog, så att varje utfall bär sitt eget
utgångsläge.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Bördighetens basvärden vid faktor 1. Härledda i 0086 ur identiteten
# mineralisering = nutrient_input · n_cells / nutrient_loss_frac, och verifierade
# mot en floraköring: 3 391 kg näring i systemet mot förutsagda 3 395.
_NUTRIENT_INPUT_BASE = 4.6e-5
_NUTRIENT_INIT_BASE = 0.117
_DETRITUS_INIT_BASE = 21.16

# Sentinel: låt simuleringen själv upptäcka när floran nått jämvikt, i stället
# för att gissa ett tickvärde. Jämvikten infaller olika sent vid olika
# bördighet, och ett tal mätt vid faktor 1 och 4 är en gissning utanför det
# intervallet. Se `Population._fauna_release_now`.
_EQUILIBRIUM_TICKS = -1


@dataclass
class VarldSpec:
    bredd: int = 64
    hojd: int = 256
    dt: float = 0.02
    # En multiplikator på näringsflödet. Skalar nutrient_input, nutrient_init
    # och detritus_init tillsammans, eftersom jämvikten är linjär i flödet.
    bordighet: float = 1.0


@dataclass
class FaunaSpec:
    antal: int = 20
    # Heltal = tick. "jamvikt" = när floran nått stationärt tillstånd.
    insatts_vid: Any = 0
    # Antal grundargrupper. Flera fläckar ger genetisk struktur från start i
    # stället för en enda linje — mätt i p91 tog två grundarlinjer av tjugo 73
    # procent av avkommorna.
    flackar: int = 1
    flackradie: float = 0.0
    # Avstånd mellan gruppernas tyngdpunkter i traitrymden (logit-enheter), och
    # skala på spridningen inom varje grupp. Kvoten avgör om det blir raser
    # eller arter. 0 = alla grundare ur samma fördelning.
    grupp_avstand: float = 0.0
    grupp_spridning: float = 1.0
    max_antal: int = 4096


@dataclass
class FysiologiSpec:
    # Multiplikator på farten. Implementeras via drag_lin, som sätter
    # jämviktsfarten: kraftbalansen F0·M^(2/3) mot drag_lin·v + drag_quad·v².
    # 1,0 ger uppmätt 37–44; 0,5 ger omkring 23.
    fartskala: float = 1.0
    sociability: float | None = None
    sociability_sd: float = 0.5


@dataclass
class Scenario:
    namn: str = "standard"
    varld: VarldSpec = field(default_factory=VarldSpec)
    fauna: FaunaSpec = field(default_factory=FaunaSpec)
    fysiologi: FysiologiSpec = field(default_factory=FysiologiSpec)

    # -- härledda världsvärden ------------------------------------------

    @property
    def nutrient_input(self) -> float:
        return _NUTRIENT_INPUT_BASE * float(self.varld.bordighet)

    @property
    def nutrient_init(self) -> float:
        return _NUTRIENT_INIT_BASE * float(self.varld.bordighet)

    @property
    def detritus_init(self) -> float:
        return _DETRITUS_INIT_BASE * float(self.varld.bordighet)

    @property
    def fauna_at_tick(self) -> int:
        v = self.fauna.insatts_vid
        if isinstance(v, str):
            key = v.strip().lower()
            if key in ("jamvikt", "jämvikt", "equilibrium"):
                return _EQUILIBRIUM_TICKS
            if key in ("start", "genast", "0"):
                return 0
            raise ValueError(f"okänt värde för fauna.insatts_vid: {v!r}")
        return int(v)

    @property
    def drag_lin(self) -> float:
        # Farten skalar i praktiken omvänt mot drag_lin: den linjära termen
        # dominerar den kvadratiska fyra mot ett vid uppmätt fart.
        s = max(1e-6, float(self.fysiologi.fartskala))
        return 220.0 / s

    # -- serialisering ---------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        def sub(cls, key):
            raw = dict(d.get(key) or {})
            known = {f for f in cls.__dataclass_fields__}
            okänt = set(raw) - known
            if okänt:
                raise ValueError(f"okända fält i {key}: {sorted(okänt)}")
            return cls(**raw)

        known_top = {"namn", "varld", "fauna", "fysiologi"}
        okänt = set(d) - known_top
        if okänt:
            raise ValueError(f"okända fält i scenariot: {sorted(okänt)}")
        return Scenario(
            namn=str(d.get("namn", "namnlöst")),
            varld=sub(VarldSpec, "varld"),
            fauna=sub(FaunaSpec, "fauna"),
            fysiologi=sub(FysiologiSpec, "fysiologi"),
        )

    @staticmethod
    def load(path: str) -> "Scenario":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return Scenario.from_dict(yaml.safe_load(fh) or {})

    def dump(self, path: str) -> None:
        """Skriv scenariot till körningens katalog, för spårbarhet."""
        import yaml

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)

    def summary(self) -> str:
        return (
            f"scenario '{self.namn}': {self.varld.bredd}x{self.varld.hojd}, "
            f"bördighet {self.varld.bordighet:g} "
            f"(nutrient_input {self.nutrient_input:.3e}), "
            f"{self.fauna.antal} djur i {self.fauna.flackar} fläck(ar) "
            f"radie {self.fauna.flackradie:g} vid tick {self.fauna_at_tick}, "
            f"gruppavstånd {self.fauna.grupp_avstand:g}/spridning "
            f"{self.fauna.grupp_spridning:g}, "
            f"fartskala {self.fysiologi.fartskala:g}"
        )
