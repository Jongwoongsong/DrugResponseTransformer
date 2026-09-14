#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GDSC_CSV:-}" || -z "${BASAL_CSV:-}" ]]; then
    echo "Set GDSC_CSV and BASAL_CSV before running." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="$REPO_ROOT/baselines/classical/cell_blind"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

python "$BASE/train_exact_cellblind_classical.py" \
    --csv "$GDSC_CSV" \
    --basal-csv "$BASAL_CSV" \
    --split-dir "$BASE/manifests" \
    --split-report "$BASE/canonical970_split_report.json" \
    --fit-cells-csv "$BASE/manifests/cell_blind_train_cells.csv" \
    --output-dir "${OUTPUT_DIR:-$REPO_ROOT/outputs/classical_cellblind}" \
    --models "${MODELS:-ridge,xgb}" \
    --seed 42 \
    --threads 12 \
    --fp-size 1024 \
    --fp-radius 2 \
    --expected-cell-features 1954 \
    --ridge-alpha 1.0 \
    --ridge-tol 0.0001 \
    --ridge-max-iter 1000 \
    --xgb-device auto \
    --xgb-n-estimators 1000 \
    --xgb-max-depth 6 \
    --xgb-learning-rate 0.03 \
    --xgb-subsample 0.8 \
    --xgb-colsample 0.8 \
    --xgb-reg-lambda 1.0 \
    --xgb-reg-alpha 0.0 \
    --xgb-min-child-weight 1.0 \
    --xgb-max-bin 256 \
    --xgb-early-stopping 50
