# Drug-blind main-effect decomposition

This analysis quantifies how much held-out drug-response variation is
explained by additive cell-line and drug main effects.

Release-specific duplicate observations are collapsed to one
cell-line × canonical-structure pair before the two-way additive
decomposition.

The additive model contains:

- one intercept
- `n_cell - 1` cell-line indicators
- `n_drug - 1` drug indicators
- no interaction term

The model is fitted with sparse LSQR. The non-additive residual is the
difference between each response and its fitted additive value.

The manuscript-reported Table S6 summarizes the fixed drug-blind split across
optimization seeds 42, 123, and 2026.

Canonical derived outputs are provided in:

`results/main_effect_decomposition/`

Reported values include:

- observed additive R2: 0.7858
- observed residual fraction: 0.2142
- scratch additive R2: 0.9529 ± 0.0143
- scratch residual PCC: 0.0545 ± 0.0319
- pretrained additive R2: 0.9218 ± 0.0426
- pretrained residual PCC: 0.0480 ± 0.0180


## Usage

The input prediction CSV must contain:

- `cell_id`
- `canonical_smiles`
- `y_true`
- `y_pred`

For example:

```bash
python analysis/main_effect_decomposition/run_main_effect_decomposition.py \
  --input_csv test_predictions.csv \
  --output_json main_effect_decomposition.json
The script first collapses release-specific duplicate observations to one
cell-line × canonical-structure pair and then performs the two-way additive
decomposition. In the canonical drug-blind test set, 42,448 response records
collapse to 35,115 unique cell-line × structure pairs.

The standalone implementation was verified against the canonical seed-42
scratch and pretrained outputs used for Supplementary Table S6, with agreement
to floating-point precision.
