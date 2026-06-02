#!/bin/bash
#SBATCH --job-name=dfa_stall
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=250G
#SBATCH --time=0-04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

SCRIPT_DIR=/n/home00/varunreddy/DFA_evolution/DFA-STALL
cd "$SCRIPT_DIR"

echo "hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

PYTHON=/n/home00/varunreddy/dynamics/venv/bin/python

$PYTHON "$SCRIPT_DIR/train.py" \
    --device cuda \
    --total-steps 3000 \
    --batch-size 128 \
    --lr 1e-3 \
    --seed 42

echo "Training done. Generating paper figures …"
$PYTHON "$SCRIPT_DIR/make_paper_figures.py"

echo "Done. Figures in $SCRIPT_DIR/figures/"
