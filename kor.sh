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

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -W error::RuntimeWarning run_headless.py \
  --scenario "$SCEN" \
  --scenario-out "$OUT/scenario.yaml" \
  --ticks "$TICKS" --stats \
  --pop-log "$OUT/pop.jsonl" --pop-every 1 \
  --life-log "$OUT/life.jsonl" \
  --check-every 10000 --report-every 1000 \
  "$@" 2>&1 | tee "$OUT/console.log"

echo "KLART $(date)" > "$OUT/DONE"
