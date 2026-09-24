#!/bin/bash
# Run the E6 teacher-student ladder locally: all rungs x {dfa, bp} x 3 seeds, at most $2 concurrent processes.
set -u
CMC="$(cd "$(dirname "$0")/.." && pwd)"
PY=${CMC_PYTHON:-python3}
NPROC=${2:-6}
LOGS="$CMC/logs/E6_ladder"; mkdir -p "$LOGS"
mapfile -t RUNGS < <(cd "$CMC" && PYTHONPATH=src $PY -m cmc.ladder --list | awk '{print $1}')
for r in "${RUNGS[@]}"; do
  for s in 0 1 2; do
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n; done
    ( cd "$CMC" && OMP_NUM_THREADS=1 PYTHONPATH=src $PY -m cmc.ladder --rung "$r" --seed "$s" > "$LOGS/${r}_s${s}.log" 2>&1 )&
  done
done
wait
echo "ladder done"
