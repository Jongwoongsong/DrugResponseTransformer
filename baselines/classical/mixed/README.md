# Mixed-split classical baselines

This directory contains the reproducibility package for the classical
mixed-split comparators reported in Table 1 of the manuscript.

The final reported Ridge, Random Forest, and XGBoost results came from
separate finalized model-specific runs. They nevertheless use the same
exact downstream mixed-split manifests distributed in `manifests/`.

## Final reported results

| Model | RMSE | MAE | Pooled PCC | Macro within-structure PCC |
|---|---:|---:|---:|---:|
| Ridge regression | 1.2934 | 0.9706 | 0.8833 | 0.4938 |
| Random Forest | 1.2525 | 0.9340 | 0.8910 | 0.5235 |
| XGBoost | 1.1882 | 0.8919 | 0.9026 | 0.5734 |

Macro within-structure PCC is the arithmetic mean of defined Pearson
correlations computed after grouping observations by canonical molecular
structure.

## Exact mixed split

The public manifests contain `source_row`, cell, drug, canonical-SMILES,
and label information needed to recover the exact downstream rows.

- train: 332,537 rows
- validation: 41,567 rows
- test: 41,567 rows

SHA256:

- train: `8212927b7a56d4719269aee689f377e611512b303ff220713c638cd0154e9b31`
- validation: `c42e94721626cb739638b17f1e18d941d09ca3608e2566ab0ab7e1c38578ca77`
- test: `3c38320b122103c4ca72bc3c91dfec41cde11e88f858dd77fd1fe16f660f00c3`

## Input representation

All three models use the same global feature representation:

- Morgan fingerprint, radius 2, 1024 bits
- 1954-gene harmonized basal-expression vector
- total feature dimension: 2978

Expected input SHA256 values:

- canonical GDSC970 CSV:
  `8f34aab89dee99f1e9c08ab0c4b4777905cd2b5a977f1b0fede21cd77aea5652`
- harmonized basal-expression CSV:
  `1ca0859617db4960cc643c1822035a91e79a8d3b4043fbc08a9fcdfd887f45f1`

The input datasets are not redistributed here.

## Ridge regression

The final mixed-split Ridge result used alpha = 1.0.

The surviving historical exact-split implementation is distributed as
`train_exact_mixed_ridge_xgb.py`.

For the historical numerical result, the thread environment was fixed
before Python startup to 12 threads for OMP, MKL, OpenBLAS, and NumExpr.

The retained final result is stored in
`reports/ridge_final_report.json`.

## Random Forest

Random Forest was selected by validation RMSE from six prespecified
configurations, all with 200 trees. The selected configuration was:

- `max_depth=None`
- `min_samples_leaf=5`
- `max_features=0.5`
- `random_state=42`
- `n_jobs=16`

The test split was evaluated only after validation-based configuration
selection.

The historical source script was retained and its scientific training
logic is preserved in `train_exact_mixed_random_forest.py`; only
machine-specific paths and filesystem discovery were replaced by
explicit command-line inputs.

The retained final result is stored in
`reports/random_forest_final_report.json`.

## XGBoost

The final manuscript XGBoost result is the finalized 2026-08-18 CPU
`hist` run, not the earlier 1000-tree GPU run.

Final settings:

- XGBoost 1.6.2
- `tree_method=hist`
- CPU
- 8 threads
- seed 42
- 5000 maximum estimators
- depth 6
- learning rate 0.03
- subsample 0.8
- column subsample 0.8
- L2 regularization 1.0
- early stopping patience 50
- selected best iteration 3634

The final run retained its arguments, trained model, feature audit, and
report, but did not retain a local source-script snapshot or launch
command. Repository-wide provenance tracing identified the surviving
2026-08-02 exact-split classical pipeline as the only matching training
implementation with the same artifact schema and CLI.

Accordingly, `train_exact_mixed_ridge_xgb.py` is the public reconstructed
implementation for this final XGBoost result. It is not claimed to be a
byte-identical snapshot of the historical 2026-08-18 source file.

The retained final result is stored in
`reports/xgboost_final_report.json`.

## Reproduction

Set the two required input paths and run the launcher:

```bash
export GDSC_CSV=/path/to/GDSC_with_SMILES_BGE_entrez_KEGG_canonical970.csv
export BASAL_CSV=/path/to/GDSC970_basal_log2p1_genematch_lincs_clipz3_all970stats.csv

bash baselines/classical/mixed/run_exact_mixed_classical.sh \
  /path/to/output_directory
```
The launcher runs Ridge, final-configuration XGBoost, and Random Forest
separately so that their historical compute settings are not conflated.

See provenance.json for the full lineage, exact split hashes, and
model-specific final metrics.
