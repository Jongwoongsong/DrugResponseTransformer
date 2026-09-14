#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${GDSC_CSV:?Set GDSC_CSV to the canonical GDSC970 CSV}"
: "${BASAL_CSV:?Set BASAL_CSV to the harmonized basal-expression CSV}"

OUT_ROOT="${1:-$PWD/mixed_classical_reproduction}"

TRAIN_MANIFEST="$HERE/manifests/train.csv"
VAL_MANIFEST="$HERE/manifests/validation.csv"
TEST_MANIFEST="$HERE/manifests/test.csv"

RIDGE_XGB="$HERE/train_exact_mixed_ridge_xgb.py"
RF="$HERE/train_exact_mixed_random_forest.py"

mkdir -p \
  "$OUT_ROOT/ridge" \
  "$OUT_ROOT/xgboost" \
  "$OUT_ROOT/random_forest"

echo "======================================================================"
echo "MIXED-SPLIT CLASSICAL BASELINE REPRODUCTION"
echo "======================================================================"
echo "[GDSC]  $GDSC_CSV"
echo "[BASAL] $BASAL_CSV"
echo "[OUT]   $OUT_ROOT"
echo "======================================================================"

echo
echo "===== RIDGE alpha=1.0 ====="

OMP_NUM_THREADS=12 \
MKL_NUM_THREADS=12 \
OPENBLAS_NUM_THREADS=12 \
NUMEXPR_NUM_THREADS=12 \
python "$RIDGE_XGB" \
  --csv_path "$GDSC_CSV" \
  --train_manifest "$TRAIN_MANIFEST" \
  --validation_manifest "$VAL_MANIFEST" \
  --test_manifest "$TEST_MANIFEST" \
  --output_dir "$OUT_ROOT/ridge" \
  --models ridge \
  --fp_bits 1024 \
  --seed 42 \
  --threads 12

echo
echo "===== XGBOOST final CPU-hist configuration ====="

OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
python "$RIDGE_XGB" \
  --csv_path "$GDSC_CSV" \
  --train_manifest "$TRAIN_MANIFEST" \
  --validation_manifest "$VAL_MANIFEST" \
  --test_manifest "$TEST_MANIFEST" \
  --output_dir "$OUT_ROOT/xgboost" \
  --models xgb \
  --fp_bits 1024 \
  --seed 42 \
  --threads 8 \
  --xgb_device cpu \
  --xgb_n_estimators 5000 \
  --xgb_max_depth 6 \
  --xgb_learning_rate 0.03 \
  --xgb_subsample 0.8 \
  --xgb_colsample 0.8 \
  --xgb_early_stopping 50

echo
echo "===== RANDOM FOREST six-config validation tuning ====="

python "$RF" \
  --csv "$GDSC_CSV" \
  --basal-csv "$BASAL_CSV" \
  --train-manifest "$TRAIN_MANIFEST" \
  --validation-manifest "$VAL_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --output-dir "$OUT_ROOT/random_forest" \
  --n-jobs 16 \
  --seed 42

echo
echo "======================================================================"
echo "DONE"
echo "Outputs: $OUT_ROOT"
echo "======================================================================"
