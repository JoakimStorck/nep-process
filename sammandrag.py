#!/usr/bin/env python3
"""
Destillerar en körningsserie till ett sammandrag som går att skicka.

Körs från repo-roten, eftersom den importerar `phenotype` för att härleda
egenskaper ur genomet:

    python sammandrag.py runs/p182 > p182_sammandrag.json

Det som behålls är fördelningar över tid, inte individer: kvartiler per
tidsfönster för de livshistorieaxlar som är under selektion, dödsorsaker med
ålder och årstid, och populationsbanan glesad till några hundra punkter.
Genomen själva behövs inte för att se vart selektionen går.
"""
import json
import sys
import os
import math

sys.path.insert(0, os.getcwd())
import numpy as np
import phenotype as ph

# Axlar som är under selektion efter 0178–0182. Läggs till här om fler
# frigörs; namnen måste finnas som fält på Phenotype.
AXLAR = ("A_mature", "M_target", "litter", "child_M", "M_repro_min",
         "structure", "diet", "breed_sync", "breed_phase", "sociability")

N_FONSTER = 40      # tidsfönster för fördelningarna
N_BANA = 300        # punkter i populationsbanan


def kvantiler(v):
    if len(v) == 0:
        return None
    a = np.asarray(v, dtype=float)
    return [round(float(x), 4) for x in np.percentile(a, [10, 25, 50, 75, 90])]


def las_seed(kat):
    ut = {}
    for f in ("console.log", "commit.txt", "scenario.yaml"):
        p = os.path.join(kat, f)
        if os.path.exists(p):
            ut[f] = open(p, encoding="utf-8", errors="replace").read()

    # Populationsbanan, glesad.
    bana = []
    p = os.path.join(kat, "pop.jsonl")
    if os.path.exists(p):
        rader = []
        for L in open(p, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(L)["summary"]
            except Exception:
                continue
            rader.append([round(float(d.get("t", 0.0)), 3),
                          int(d.get("pop", 0)),
                          round(float(d.get("mean_M", 0.0) or 0.0), 4),
                          round(float(d.get("mean_R", 0.0) or 0.0), 4),
                          round(float(d.get("mean_E", 0.0) or 0.0), 1)])
        steg = max(1, len(rader) // N_BANA)
        bana = rader[::steg]
    ut["bana"] = {"kolumner": ["t", "pop", "mean_M", "mean_R", "mean_E"],
                  "rader": bana}

    # Liv och död.
    fodslar, dodsfall = [], []
    p = os.path.join(kat, "life.jsonl")
    if os.path.exists(p):
        for L in open(p, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(L)
            except Exception:
                continue
            if d.get("event") == "birth":
                tr = d.get("traits")
                if tr is not None and len(tr) > 8:
                    fodslar.append((float(d.get("t", 0.0)),
                                    np.asarray(tr, dtype=float)))
            elif d.get("cause"):
                dodsfall.append((float(d.get("t", 0.0)),
                                 str(d.get("cause")),
                                 float(d.get("age", 0.0) or 0.0)))

    t_max = max([r[0] for r in bana] + [f[0] for f in fodslar] + [1.0])
    kant = np.linspace(0.0, t_max, N_FONSTER + 1)

    # Egenskapernas fördelning per tidsfönster. Härleds ur genomet med
    # modellens egen `derive_pheno`, så att kvantilerna är fenotypiska och inte
    # locus-värden.
    axlar = {a: [] for a in AXLAR}
    n_per = []
    for i in range(N_FONSTER):
        blk = [tr for t, tr in fodslar if kant[i] <= t < kant[i + 1]]
        n_per.append(len(blk))
        if not blk:
            for a in AXLAR:
                axlar[a].append(None)
            continue
        P = [ph.derive_pheno(tr) for tr in blk]
        for a in AXLAR:
            axlar[a].append(kvantiler([getattr(p_, a, float("nan")) for p_ in P]))

    # Dödsorsaker: antal per fönster, samt ålder och årstid.
    orsaker = sorted({c for _, c, _ in dodsfall})
    dod_per = {c: [0] * N_FONSTER for c in orsaker}
    alder_per = {c: [None] * N_FONSTER for c in orsaker}
    for i in range(N_FONSTER):
        blk = [(c, a) for t, c, a in dodsfall if kant[i] <= t < kant[i + 1]]
        for c in orsaker:
            al = [a for cc, a in blk if cc == c]
            dod_per[c][i] = len(al)
            alder_per[c][i] = kvantiler(al)

    manad = {c: [0] * 12 for c in orsaker}
    for t, c, _ in dodsfall:
        manad[c][int(t % 12.0)] += 1

    ut["fonster"] = [round(float(x), 3) for x in kant]
    ut["fodslar_per_fonster"] = n_per
    ut["axlar"] = axlar
    ut["dod_per_fonster"] = dod_per
    ut["dodsalder_per_fonster"] = alder_per
    ut["dod_per_manad"] = manad
    ut["n_fodslar"] = len(fodslar)
    ut["n_dodsfall"] = len(dodsfall)
    return ut


def main():
    rot = sys.argv[1] if len(sys.argv) > 1 else "runs/p182"
    fron = sorted(d for d in os.listdir(rot)
                  if os.path.isdir(os.path.join(rot, d)))
    allt = {"rot": rot,
            "kvantiler": ["p10", "p25", "median", "p75", "p90"],
            "fron": {}}
    for s in fron:
        allt["fron"][s] = las_seed(os.path.join(rot, s))
        sys.stderr.write(
            "%s: %d födslar, %d dödsfall\n"
            % (s, allt["fron"][s]["n_fodslar"], allt["fron"][s]["n_dodsfall"]))
    json.dump(allt, sys.stdout, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
