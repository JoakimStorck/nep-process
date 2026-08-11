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
                # Massa, reserv och skada bär verkliga tal sedan 0190; dessförinnan
                # nollställdes kroppen innan posten skrevs. `M_peak` finns inte i
                # posten, men `M_waste_frac` binder i 99 procent av svältdödsfallen,
                # så toppen är `M / M_waste_frac` för dem.
                dodsfall.append((float(d.get("t", 0.0)),
                                 str(d.get("cause")),
                                 float(d.get("age", 0.0) or 0.0),
                                 float(d.get("M", 0.0) or 0.0),
                                 float(d.get("M_reserve", 0.0) or 0.0),
                                 float(d.get("D", 0.0) or 0.0)))

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
    orsaker = sorted({r[1] for r in dodsfall})
    dod_per = {c: [0] * N_FONSTER for c in orsaker}
    alder_per = {c: [None] * N_FONSTER for c in orsaker}
    massa_per = {c: [None] * N_FONSTER for c in orsaker}
    reserv_per = {c: [None] * N_FONSTER for c in orsaker}
    skada_per = {c: [None] * N_FONSTER for c in orsaker}
    for i in range(N_FONSTER):
        blk = [r for r in dodsfall if kant[i] <= r[0] < kant[i + 1]]
        for c in orsaker:
            rr = [r for r in blk if r[1] == c]
            dod_per[c][i] = len(rr)
            alder_per[c][i] = kvantiler([r[2] for r in rr])
            massa_per[c][i] = kvantiler([r[3] for r in rr])
            reserv_per[c][i] = kvantiler([r[4] for r in rr])
            skada_per[c][i] = kvantiler([r[5] for r in rr])

    manad = {c: [0] * 12 for c in orsaker}
    for r in dodsfall:
        manad[r[1]][int(r[0] % 12.0)] += 1

    # Årstid mot ålder och mot fettreserv. Två frågor som bara går att ställa
    # här: om en vinterkrympning blir en dödsdom **nästa** år skrivs den som en
    # topp i andra levnadsårets dödlighet, och om mobiliseringstaket binder i
    # kylan dör djuren med fett kvar just då och inte jämnt över året.
    #
    # `andel_med_fett` räknar dödsfall där reserven översteg en procent av
    # kroppsmassan. Tröskeln är godtycklig men skiljer "tom" från "hade kvar".
    manad_alder = {c: [None] * 12 for c in orsaker}
    manad_fett = {c: [0.0] * 12 for c in orsaker}
    for c in orsaker:
        for m in range(12):
            rr = [r for r in dodsfall if r[1] == c and int(r[0] % 12.0) == m]
            manad_alder[c][m] = kvantiler([r[2] for r in rr])
            if rr:
                manad_fett[c][m] = round(
                    sum(1 for r in rr if r[4] > 0.01 * max(r[3], 1e-12)) / len(rr), 4)

    # Åldersband per månad, för att skilja juvenil svält från vuxen. Banden är
    # satta mot `A_mature`, som ligger på 6–12 månader i körningarna.
    band = ((0.0, 1.0), (1.0, 6.0), (6.0, 12.0), (12.0, 24.0), (24.0, 1e9))
    manad_band = {c: [[0] * 12 for _ in band] for c in orsaker}
    for r in dodsfall:
        m = int(r[0] % 12.0)
        for bi, (lo, hi) in enumerate(band):
            if lo <= r[2] < hi:
                manad_band[r[1]][bi][m] += 1
                break

    ut["fonster"] = [round(float(x), 3) for x in kant]
    ut["fodslar_per_fonster"] = n_per
    ut["axlar"] = axlar
    ut["dod_per_fonster"] = dod_per
    ut["dodsalder_per_fonster"] = alder_per
    ut["dodsmassa_per_fonster"] = massa_per
    ut["dodsreserv_per_fonster"] = reserv_per
    ut["dodsskada_per_fonster"] = skada_per
    ut["dod_per_manad"] = manad
    ut["dodsalder_per_manad"] = manad_alder
    ut["andel_med_fett_per_manad"] = manad_fett
    ut["aldersband"] = [list(b) for b in band]
    ut["dod_per_manad_och_aldersband"] = manad_band
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
