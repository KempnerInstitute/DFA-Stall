#!/usr/bin/env bash
# Run every line of a job file locally, N at a time. Usage: scripts/run_local.sh configs/jobs.txt [N]
set -uo pipefail
JOBS=$1; N=${2:-6}; ROOT=$(cd "$(dirname "$0")/.." && pwd); PY=${CMC_PYTHON:-python3}
mkdir -p "$ROOT/logs"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | xargs -P "$N" -I{} bash -c "cd $ROOT && $PY -m cmc.run {} --threads 2 >> logs/\$(echo '{}' | tr ' /=-' '____' | cut -c1-120).log 2>&1 || echo 'FAILED: {}' >> logs/failures.txt"
