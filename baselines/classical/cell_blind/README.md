# Exact cell-blind classical baselines

This directory contains the portable release of the canonical classical
cell-blind baselines used in the manuscript.

The models use the same exact cell-blind split as the main evaluation:

- 776 training cells
- 97 validation cells
- 97 test cells
- zero overlap between cell sets

The public manifests in `manifests/` are exact copies, including row order, of
the canonical split manifests used in the reported experiment.

## Features

Drug features:
- Morgan fingerprint
- radius = 2
- 1024 bits

Cell features:
- 1,954 harmonized basal-expression features
- normalization statistics fitted using the 776 training cells only

## Models

Ridge:
- alpha = 1.0
- solver = lsqr
- tolerance = 1e-4
- max iterations = 1000

XGBoost:
- 1000 estimators
- max depth = 6
- learning rate = 0.03
- subsample = 0.8
- column subsample = 0.8
- L2 regularization = 1.0
- L1 regularization = 0.0
- min child weight = 1.0
- max bin = 256
- early stopping patience = 50

## Usage

The canonical GDSC response table and harmonized basal-expression table are not
redistributed in this repository. See `docs/DATA_PREPARATION.md` for input
preparation and expected schemas.

Example:

    python baselines/classical/cell_blind/train_exact_cellblind_classical.py \
      --csv "$GDSC_CSV" \
      --basal-csv "$BASAL_CSV" \
      --split-dir baselines/classical/cell_blind/manifests \
      --split-report baselines/classical/cell_blind/canonical970_split_report.json \
      --fit-cells-csv baselines/classical/cell_blind/manifests/cell_blind_train_cells.csv \
      --output-dir outputs/classical_cellblind \
      --models ridge,xgb \
      --seed 42

The bundled `cellblind_shared_helpers.py` contains the canonical preprocessing,
manifest-audit, feature-construction, and metric functions required by this
pipeline.

The manuscript-reported outputs are preserved in:

- `reproducibility/reports/cell_blind/ridge_exact_cellblind.report.json`
- `reproducibility/reports/cell_blind/xgboost_exact_cellblind.report.json`

Those public reports were verified to be semantically identical to the
canonical run reports.

## Reproduction notes

For exact reproduction of the archived Ridge result, the original threading
environment must be established before the Python process starts:

    OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    OPENBLAS_NUM_THREADS=12 \
    NUMEXPR_NUM_THREADS=12 \
    python baselines/classical/cell_blind/train_exact_cellblind_classical.py ...

With this pre-launch environment, the public Ridge pipeline reproduced the
archived canonical validation and test metrics exactly, including the original
123 LSQR iterations.

For XGBoost, two repeated current runs were identical to each other and used
the same audited data, split, feature construction, XGBoost 1.6.2 `gpu_hist`
backend, hyperparameters, A100 GPU, and driver as the archived run. They showed
small numerical differences from the archived canonical GPU-hist result
(test PCC 0.87725 versus 0.87634; test RMSE 1.33286 versus 1.33759).
The source of this historical numerical divergence was not further resolved.
The archived manuscript-reported report remains available under
`reproducibility/reports/cell_blind/`.
