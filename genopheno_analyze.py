"""
Efterhandsanalys av en körnings `life.jsonl` och `pop.jsonl`.

Skriptet fanns sedan tidigare men hade drivit ifrån både loggformatet och
frågeställningen. Tre saker gjorde det missvisande snarare än bara inaktuellt:

**Halva filen läste ett event som inte längre skrivs.** All per-agent-statistik
byggdes ur `event: "step"`, som inte finns i loggen. `steps_life: 0` stod i
rapporten, och samtliga härledda fält — fart, hunger, energi över livet — var
tysta NaN. De är borta här.

**Dödsorsaken lästes aldrig.** `cause` tillkom i Steg 5d och är det enskilt mest
informativa fältet i `life.jsonl`. Likaså `straightness`, `path_len` och
`net_disp` från Steg 5h, som bär ett av Del E:s måltal.

**Selektionsdifferentialen rankades i blandade enheter.** `reserve_cap` låg
överst med -78 021 och `mobility` långt ned med +0,037, vilket bara säger att
den ena mäts i joule och den andra i en enhetsskala. Rankningen var alltså en
rankning av måttenheter. Differentialen standardiseras nu i spridningsenheter,
och en permutationsbaserad brusnivå skrivs ut bredvid: med fyrtioen loci och
tvåhundra individer är en topplista utan nollhypotes garanterat full av brus.

Det som tillkommit är i stället riktat mot det körningarna faktiskt frågar om
just nu: kollapsens anatomi. Dödsorsaker över tid, energibudgeten per djur,
kadavrets andel av födan, och härstamningen tillbaka till grundargrupp.

Anropas som:

    python genopheno_analyze.py --run runs/p114

vilket läser `life.jsonl` och `pop.jsonl` i katalogen och skriver
`genopheno_report.md` bredvid dem.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Locusnamnen ägs av phenotype.py. Att läsa dem därifrån i stället för att
# duplicera en lista är samma princip som gäller i koden: ändras locuskartan
# ska analysen följa med i stället för att gå sönder tyst.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import phenotype as _ph

    TRAIT_NAMES = {
        v: k[3:].lower()
        for k, v in vars(_ph).items()
        if k.startswith("_T_") and isinstance(v, int)
    }
except Exception:  # pragma: no cover - analysen ska gå att köra fristående
    TRAIT_NAMES = {}

# Loci som bara flora uttrycker. De ärvs och muterar i faunans genom också, men
# har ingen läsare där, så de fungerar som inbyggd nollhypotes: en selektion som
# slår lika hårt på dem som på faunaloci är brus.
FLORA_LOCI = frozenset(range(25, 38))


# ---------------------------------------------------------------------------
# inläsning
# ---------------------------------------------------------------------------

def load_jsonl(fp: Path) -> list[dict]:
    out = []
    with fp.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def f(x, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def q(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |",
           "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# life.jsonl
# ---------------------------------------------------------------------------

def read_life(rows: list[dict]) -> dict:
    births: dict[int, dict] = {}
    deaths: dict[int, dict] = {}
    for r in rows:
        aid = r.get("agent_id")
        if aid is None:
            continue
        aid = int(aid)
        if r.get("event") == "birth":
            births[aid] = r
        elif r.get("event") == "death":
            deaths[aid] = r

    parent: dict[int, int | None] = {}
    for aid, b in births.items():
        p = b.get("parent_id")
        parent[aid] = None if p in (None, "None") else int(p)

    founders = [a for a, p in parent.items() if p is None]

    # Rot i härstamningen, iterativt och med besöksmängd: en cykel i loggen ska
    # ge ett obestämt svar, inte en spräckt stack.
    root: dict[int, int] = {}
    for a in list(parent):
        chain, cur, seen = [], a, set()
        while cur is not None and cur in parent and cur not in root and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = parent[cur]
        base = root.get(cur, cur if cur is not None else a)
        for c in chain:
            root[c] = base

    offspring: Counter = Counter()
    for aid, p in parent.items():
        if p is not None:
            offspring[p] += 1

    return {"births": births, "deaths": deaths, "parent": parent,
            "founders": founders, "root": root, "offspring": offspring}


def founder_groups(life: dict, k: int = 3, height: float = 256.0) -> dict[int, int]:
    """
    Gruppera grundarna efter födelseposition.

    Scenariot sätter ut faunan i `flackar` rumsligt åtskilda fläckar med var sin
    tyngdpunkt i traitrymden, men grupptillhörigheten loggas inte. Den går att
    återvinna ur positionen, eftersom fläckarna är åtskilda i höjdled och
    radien är liten mot världens höjd. Enkel k-means i y med toruskorrigering.
    """
    fs = life["founders"]
    if len(fs) < 2 * k or k < 2:
        return {a: 0 for a in fs}
    ys = np.array([f(life["births"][a]["pos"]["y"]) for a in fs])
    c = np.percentile(ys, np.linspace(100 / (2 * k), 100 - 100 / (2 * k), k))
    lab = np.zeros(ys.size, dtype=int)
    for _ in range(60):
        d = np.abs(ys[:, None] - c[None, :])
        d = np.minimum(d, height - d)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = ys[lab == j].mean()
    return {a: int(lab[i]) for i, a in enumerate(fs)}


# ---------------------------------------------------------------------------
# avsnitt
# ---------------------------------------------------------------------------

def sec_deaths(life: dict, dt: float) -> str:
    D = list(life["deaths"].values())
    if not D:
        return ""
    md = ["## Dödsorsaker\n"]
    rows = []
    for c, n in Counter(d.get("cause", "okänd") for d in D).most_common():
        sel = [d for d in D if d.get("cause") == c]
        g = lambda k: np.array([f(d.get(k)) for d in sel])
        rows.append([c, str(n), f"{n / len(D) * 100:.1f} %",
                     f"{q(g('age'), 50):.1f}", f"{q(g('D'), 50):.3f}",
                     f"{q(g('M'), 50):.3f}", f"{q(g('straightness'), 50):.3f}"])
    md.append(table(["orsak", "n", "andel", "median ålder", "median D",
                     "median M", "median rakhet"], rows))

    # Fördelning över tid. Ett kollapsförlopp har en signatur här som
    # totalsiffran döljer: svälten kommer i en puls, skadan ligger jämnt.
    td = [f(d.get("t")) for d in D]
    tb = [f(life["births"][a].get("t")) for a in life["parent"]
          if life["parent"][a] is not None]
    lo, hi = min(td + tb), max(td + tb)
    if hi > lo:
        nb = 12
        edges = np.linspace(lo, hi + 1e-9, nb + 1)
        rows = []
        for i in range(nb):
            a0, a1 = edges[i], edges[i + 1]
            sel = [d for d in D if a0 <= f(d.get("t")) < a1]
            c = Counter(d.get("cause") for d in sel)
            rows.append([f"{a0 / dt:.0f}-{a1 / dt:.0f}",
                         str(sum(1 for t in tb if a0 <= t < a1)), str(len(sel)),
                         str(c.get("starvation", 0)), str(c.get("damage", 0)),
                         str(c.get("hazard", 0)), str(c.get("predation", 0))])
        md.append("\n### Över tid, i tick\n")
        md.append(table(["fönster", "födslar", "döda", "svält", "skada",
                         "hazard", "predation"], rows))
    return "\n".join(md)


def sec_budget(pop: list[dict], dt: float, n_rows: int = 12) -> str:
    """
    Energibudget per djur ur `pop.jsonl`, och kadavrets andel av födan.

    Det är den här tabellen som skiljer en svält orsakad av att födan tar slut
    från en svält orsakad av att den inte hittas: den första syns som fallande
    intag per djur, den andra som ett intag som räcker i medel medan den nedre
    svansen står på noll.
    """
    R = [r for r in pop if r.get("event") == "population"
         and (r.get("summary") or {}).get("pop", 0) > 0]
    if not R:
        return ""
    keys_out = ("E_loss_basal", "E_loss_thermo", "E_loss_loco",
                "E_loss_compute", "E_loss_sense", "E_loss_repair")
    rows = []
    for i in np.linspace(0, len(R) - 1, min(n_rows, len(R))).astype(int):
        r = R[i]
        s = r["summary"]
        p = max(1, int(s["pop"]))
        ein = f(r.get("E_in_total", 0.0)) / p / 1e6
        out = sum(f(r.get(k, 0.0)) for k in keys_out) / p / 1e6
        bio, car = f(r.get("food_bio_kg", 0.0)), f(r.get("food_carcass_kg", 0.0))
        rows.append([f"{f(s['t']) / dt:.0f}", str(int(s["pop"])),
                     f"{f(s.get('mean_M')):.3f}", f"{f(s.get('p10_E')):.2e}",
                     f"{ein:.3f}", f"{out:.3f}", f"{ein - out:+.3f}",
                     f"{car / max(1e-12, bio + car) * 100:.0f} %",
                     f"{f(s.get('mean_D')):.3f}"])
    md = ["## Energibudget per djur\n",
          "Intag och förbrukning i MJ per djur och loggsteg. `p10_E` är tionde "
          "percentilen av energilagret — står den på noll medan nettot är "
          "positivt är svälten fördelningsbunden, inte en fråga om total "
          "tillgång.\n",
          table(["tick", "pop", "medel M", "p10 E", "intag", "förbrukn.",
                 "netto", "kadaver", "medel D"], rows)]

    tot_bio = sum(f(r.get("food_bio_kg", 0.0)) for r in R)
    tot_car = sum(f(r.get("food_carcass_kg", 0.0)) for r in R)
    md.append(f"\nÖver hela faunaperioden: {tot_bio:.0f} kg flora och "
              f"{tot_car:.0f} kg kadaver. ")
    md.append("**Kadavret är större än floran** — beståndet levde till större "
              "delen på sina egna döda, vilket är en stock och inte ett flöde.\n"
              if tot_car > tot_bio else "\n")

    dsum = {k: sum(f(r.get(k, 0.0)) for r in R)
            for k in ("dD_eff", "dD_met", "dD_age", "dD_starve", "dD_cold")}
    tt = sum(dsum.values())
    if tt > 0:
        md.append("\n### Skadeinflödets termer\n")
        md.append(table(["term", "summa", "andel"],
                        [[k, f"{v:.2f}", f"{v / tt * 100:.1f} %"]
                         for k, v in sorted(dsum.items(), key=lambda x: -x[1])]))

    g = sum(f(r.get("dM_growth", 0.0)) for r in R)
    c = sum(f(r.get("dM_catabolism", 0.0)) for r in R)
    md.append(f"\nTillväxt {g:.2f} kg mot katabolism {c:.2f} kg. ")
    md.append("Beståndet byggde nettomassa.\n" if g > c
              else "**Beståndet brände nettomassa** över hela perioden.\n")
    return "\n".join(md)


def sec_lineage(life: dict, groups: dict[int, int]) -> str:
    off, fs = life["offspring"], life["founders"]
    kids = [a for a, p in life["parent"].items() if p is not None]
    md = ["## Härstamning\n"]

    n_grp = (max(groups.values()) + 1) if groups else 1
    if n_grp > 1:
        rows = []
        for g in range(n_grp):
            gf = [a for a in fs if groups.get(a) == g]
            desc = sum(1 for a in kids if groups.get(life["root"].get(a, -1)) == g)
            rows.append([str(g), str(len(gf)), str(desc),
                         f"{desc / max(1, len(gf)):.2f}"])
        md.append("Grundargrupperna återvunna ur födelseposition. Skiljer sig "
                  "avkomma per grundare mellan grupperna har startgenetiken "
                  "haft betydelse.\n")
        md.append(table(["grupp", "grundare", "ättlingar totalt",
                         "per grundare"], rows))

    r0 = np.array([off.get(a, 0) for a in life["parent"]], dtype=float)
    if r0.size:
        sr = np.sort(r0)[::-1]
        top = sr[: max(1, int(0.1 * sr.size))].sum() / max(1e-9, sr.sum())
        md.append(f"\n{int((r0 == 0).sum())} av {r0.size} lämnade ingen "
                  f"avkomma. Den mest produktiva tiondelen står för "
                  f"{top * 100:.0f} procent av alla födslar.\n")
    return "\n".join(md)


def sec_traits(life: dict, rng: np.random.Generator, n_perm: int = 2000) -> str:
    """
    Selektionsdifferential per locus, standardiserad och med brusnivå.

    Differentialen är skillnaden i medelvärde mellan dem som lämnade avkomma
    och alla, uttryckt i loci-fördelningens egen spridning. Brusnivån är 95:e
    percentilen av samma statistik under slumpad tillhörighet, alltså vad
    urvalet ger av sig självt vid det här antalet individer.
    """
    ids = [a for a in life["parent"] if life["births"].get(a, {}).get("traits")]
    if len(ids) < 20:
        return ""
    T = np.array([[f(x) for x in life["births"][a]["traits"]] for a in ids])
    r0 = np.array([life["offspring"].get(a, 0) for a in ids], dtype=float)
    sel = r0 > 0
    k = int(sel.sum())
    if k < 5 or (T.shape[0] - k) < 5:
        return ""

    sd = T.std(axis=0)
    sd = np.where(sd < 1e-12, np.nan, sd)
    diff = (T[sel].mean(axis=0) - T.mean(axis=0)) / sd

    null = np.empty((n_perm, T.shape[1]))
    for i in range(n_perm):
        idx = rng.choice(T.shape[0], size=k, replace=False)
        null[i] = (T[idx].mean(axis=0) - T.mean(axis=0)) / sd
    floor = np.nanpercentile(np.abs(null), 95, axis=0)

    rows, n_over = [], 0
    for j in np.argsort(-np.abs(np.nan_to_num(diff)))[:14]:
        j = int(j)
        over = bool(np.isfinite(diff[j]) and abs(diff[j]) > floor[j])
        n_over += int(over and j not in FLORA_LOCI)
        rows.append([str(j), TRAIT_NAMES.get(j, f"locus {j}"),
                     "flora" if j in FLORA_LOCI else "",
                     f"{diff[j]:+.3f}", f"{floor[j]:.3f}", "ja" if over else ""])
    n_flora_over = sum(1 for j in range(T.shape[1]) if j in FLORA_LOCI
                       and np.isfinite(diff[j]) and abs(diff[j]) > floor[j])
    md = ["## Selektion per locus\n",
          f"n = {len(ids)}, varav {k} lämnade avkomma. Differentialen är i "
          "spridningsenheter; brusnivån är 95:e percentilen av |differential| "
          f"under {n_perm} slumpade urval av samma storlek.\n",
          table(["locus", "namn", "", "differential", "brusnivå", "över brus"],
                rows),
          f"\nÖver brusnivån: {n_over} faunaloci i topplistan, {n_flora_over} "
          f"av {len(FLORA_LOCI)} floraloci totalt. Floraloci har ingen läsare i "
          "faunans fysiologi, så de anger hur många träffar som uppstår utan "
          "selektion.\n"]
    return "\n".join(md)


def sec_movement(life: dict) -> str:
    D = list(life["deaths"].values())
    g = lambda k: np.array([f(d.get(k)) for d in D])
    st = g("straightness")
    if not np.isfinite(st).any():
        return ""
    md = ["## Rörelsen över en livstid\n",
          "Rakhet är nettoförflyttning genom bansträcka. Del E:s måltal är att "
          "den ska ligga över 0,069 och vara tillståndsberoende.\n",
          table(["mått", "p10", "median", "medel", "p90"],
                [[n, f"{q(v,10):.3f}", f"{q(v,50):.3f}",
                  f"{np.nanmean(v):.3f}", f"{q(v,90):.3f}"]
                 for n, v in (("rakhet", st), ("bansträcka", g("path_len")),
                              ("nettoförflyttning", g("net_disp")))])]
    return "\n".join(md)


# ---------------------------------------------------------------------------

def build_report(run: Path, dt: float, seed: int) -> str:
    life_fp, pop_fp = run / "life.jsonl", run / "pop.jsonl"
    if not life_fp.exists():
        raise SystemExit(f"saknas: {life_fp}")
    life = read_life(load_jsonl(life_fp))
    pop = load_jsonl(pop_fp) if pop_fp.exists() else []
    rng = np.random.default_rng(seed)

    ages = np.array([f(d.get("age")) for d in life["deaths"].values()])
    real = [a for a, p in life["parent"].items() if p is not None]
    name = run.resolve().name or str(run)
    md = [f"# Genopheno - {name}\n",
          f"- {len(life['founders'])} grundare, {len(real)} födslar, "
          f"{len(life['deaths'])} dödsfall\n"]
    if ages.size:
        md.append(f"- livslängd p10/median/p90: {q(ages,10):.1f} / "
                  f"{q(ages,50):.1f} / {q(ages,90):.1f} månader\n")
    md.append("\n")
    for part in (sec_deaths(life, dt), sec_budget(pop, dt),
                 sec_lineage(life, founder_groups(life)),
                 sec_traits(life, rng), sec_movement(life)):
        if part:
            md.append(part)
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description="Efterhandsanalys av en körning.")
    ap.add_argument("--run", type=str, default=".",
                    help="körningens katalog; läser life.jsonl och pop.jsonl")
    ap.add_argument("--out", type=str, default=None,
                    help="rapportfil; standard är <run>/genopheno_report.md")
    ap.add_argument("--dt", type=float, default=0.02,
                    help="tidssteg, för att räkna om simulerad tid till tick")
    ap.add_argument("--seed", type=int, default=1,
                    help="frö för permutationstestets nollfördelning")
    a = ap.parse_args()

    run = Path(a.run)
    md = build_report(run, float(a.dt), int(a.seed))
    out = Path(a.out) if a.out else run / "genopheno_report.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"skrev {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
