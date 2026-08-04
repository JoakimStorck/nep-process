#!/usr/bin/env bash
#
# Standardiserad uppstart. Ett scenario, en utdatakatalog, full spårbarhet.
#
#   ./kor.sh scenarios/f4-flock.yaml runs/p106 60000 [extra flaggor...]
#
# Katalogen får scenariot, commit-hashen och hela konsolutskriften, så att en
# körning går att förstå långt efteråt utan terminalhistorik. Att blanda ihop
# vilken patch en körning gjordes mot har hänt flera gånger.
#
# Världsloggen skrivs också. Utan den går florans selektion inte att mäta i
# efterhand — genopheno_analyze.py kan bara se faunans genom ur life.jsonl, och
# frågan om floran differentierar sig är Steg 4:s hela poäng.
#
set -eu
cd "$(dirname "$0")"

SCEN="${1:?ange scenariofil}"
OUT="${2:?ange utdatakatalog}"
TICKS="${3:-60000}"
shift 3 || true

exec 9>"/tmp/$(basename "$OUT").lock"
flock -n 9 || { echo "kör redan: $OUT"; exit 1; }

mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "okänd" > "$OUT/commit.txt"
git status --porcelain > "$OUT/dirty.txt" 2>/dev/null || true

# BLAS-trådarna hålls på en. De ger ingenting här och stör mätningen.
#
# NUMBA_NUM_THREADS sätts däremot inte längre. Så länge tillväxtkärnan bara
# fanns i njit-form var en tråd rätt, men med en parallelliserad kärna är en
# hårdkodad etta samma sak som att mäta parallelliseringen med den avstängd.
# Sätt variabeln i miljön om du vill begränsa den.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -W error::RuntimeWarning run_headless.py \
  --scenario "$SCEN" \
  --scenario-out "$OUT/scenario.yaml" \
  --ticks "$TICKS" --stats \
  --pop-log "$OUT/pop.jsonl" --pop-every 1 \
  --life-log "$OUT/life.jsonl" \
  --world-log "$OUT/world.jsonl" --world-every 2 \
  --check-every 10000 --report-every 1000 \
  "$@" 2>&1 | tee "$OUT/console.log"

echo "KLART $(date)" > "$OUT/DONE"
