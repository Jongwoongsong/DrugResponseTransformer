# DrugResponseTransformer

Reproducibility repository for DrugResponseTransformer:

**Interpretable Prediction of Drug Sensitivity via Transfer Learning Based on Biological Graphs**

## Overview

DrugResponseTransformer (DRT) integrates molecular graphs, pathway-grounded cell-line representations, biological graph structure, and Transformer-based cross-modal contextualization.

The analysis evaluates predictive performance, generalization, architectural contributions, and limitations of model attribution.

## Repository structure

- `Model/` — core DRT modules
- `final_union_da_ic50_finetune.py` — final GDSC fine-tuning/evaluation
- `pge_pretrain_da_bidir_union.py` — final LINCS pretraining
- `pge_pretrain_faithful_token_dedup.py` — shared token-preserving utilities
- `build_lincs218_union_da_cache.py` — pathway-graph cache construction
- `baselines/` — comparator wrappers
- `analysis/` — evaluation and figure scripts
- `configs/` — final configuration
- `environment/` — software environment
- `splits/` — public-safe split assignments
- `results/` — evaluation results
- `figures/` — supplementary figures
- `reproducibility/` — run reports and provenance
- `resources/` — external resource identifiers

## Data and response variable

Detailed input schemas and preprocessing steps are documented in [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md).

Raw LINCS, GDSC, Cell Model Passports, and KEGG source data are not redistributed.

The KEGG pathway identifiers used in the study are listed in `resources/kegg_pathway_ids.txt`.

The downstream model uses the `LN_IC50` field supplied by GDSC directly. No additional logarithmic transformation is applied.

## Evaluation

### Mixed split
Record-level interpolation; drug and cell-line identities may occur across partitions.

### Drug-blind split
Canonical molecular structures are partitioned before assigning downstream GDSC response records.

This evaluates structures excluded from downstream GDSC supervision and does not necessarily imply exclusion from LINCS pretraining.

### Cell-blind split
Cell-line identities are held out from downstream GDSC supervision.

For the primary drug-blind robustness analysis, split seed 42 is fixed while optimization seeds 42, 123, and 2026 are varied. This evaluates optimization stochasticity and is not k-fold cross-validation.

## Results

- `results/final_drugblind_summary.csv`
- `reproducibility/drt_drugblind_run_index.tsv`
- `reproducibility/reports/`
- `results/per_structure/`
- `figures/per_structure/`

Reported metrics are RMSE, MAE, pooled Pearson correlation, and macro within-structure Pearson correlation.

## Environment

Create the Conda environment with:

    conda env create -f environment/environment.yml
    conda activate drug-response-transformer

Additional package snapshots are provided in:

- `environment/requirements_full.txt`
- `environment/core_versions.txt`

## Final configuration

Final architecture and training settings are recorded in `configs/final_experiment_config.json`.

The architecture uses 31 KEGG pathways, 712 pathway-specific gene occurrences aggregated into 268 unique gene tokens, typed/direction-aware biological graph processing, and a two-layer eight-head Transformer.

## Interpretation

Pathway-, gene-, and atom-level attribution outputs describe model behavior and are intended for structured hypothesis generation rather than causal inference.

## Third-party resources

External datasets, KEGG resources, and comparator implementations remain subject to the licenses and usage terms of their original providers.

## Data-root configuration

Raw input datasets are not bundled with this repository.

If source data and derived caches are stored outside the repository, set `DRT_ROOT` to the project/data root before running the pipelines:

    export DRT_ROOT=/path/to/your/project

If `DRT_ROOT` is not defined, the repository directory is used as the default root.

<!-- CELLBLIND_FINAL_START -->
## Cell-blind evaluation

A complementary held-out-cell evaluation uses a fixed split of 776 training,
97 validation, and 97 test cell lines. Cell identities are partitioned before
response-row assignment, yielding 332,414 / 42,000 / 41,257 retained
train/validation/test measurements. Basal-expression harmonization statistics
are fitted using training cells only.

Final-aligned pretrained DRT was repeated with optimization seeds 42, 123, and
2026 and achieved RMSE 1.4110 ± 0.0094, MAE 1.0552 ± 0.0095, pooled PCC
0.8630 ± 0.0026, and macro per-drug PCC 0.3097 ± 0.0203. The final-aligned
scratch seed-42 run achieved RMSE 1.3962, MAE 1.0443, pooled PCC 0.8639, and
macro per-drug PCC 0.3245. These results do not provide evidence of a
consistent downstream benefit from perturbation pretraining in the held-out-cell
setting.

Public release files:

- `splits/cell_blind/`: cell-identity-only manifests.
- `results/final_cellblind_summary.csv`: final comparator and DRT results.
- `reproducibility/cellblind_drt_run_index.tsv`: final DRT run registry.
- `reproducibility/reports/cell_blind/`: sanitized final DRT and classical reports.
- `reproducibility/provenance/cell_blind/`: cell-blind result provenance.

The public split files do not redistribute GDSC response labels.
<!-- CELLBLIND_FINAL_END -->

## LINCS pretraining exposure audit

Exact RDKit canonical-structure matching against the final 591,912-profile LINCS pretraining corpus showed that 23 of 41 (56.1%) downstream drug-blind test structures were present as exact canonical structures in LINCS, whereas 18 of 41 (43.9%) were absent.

The exact-unexposed subset was evaluated separately across optimization seeds 42, 123, and 2026. Pretrained DRT did not show a consistent advantage over scratch training in this subset (RMSE 3.1797 ± 0.1881 vs. 3.1077 ± 0.0418; pooled PCC 0.1777 ± 0.1734 vs. 0.2855 ± 0.0715). These results distinguish exact pretraining exposure from downstream GDSC structure holdout; they do not constitute a scaffold-novel or pretraining-excluded retraining experiment.

Public derived outputs are provided in `results/lincs_exposure/`. The exact-overlap audit can be recomputed from locally obtained LINCS data using `analysis/audit_lincs_exact_exposure.py`.

## Baseline reproduction

### GAT-Cross

The GAT-Cross implementation used for the reported mixed-split comparison is
included under `baselines/gat_cross_reference/kci_model_gat.py`.
The preserved module is the provenance-audited implementation used in the
reported experiment.

Example command:

    python baselines/train_exact_gat_cross.py \
      --feature_cache /path/to/prepared_949_feature_cache \
      --drug_graph_cache /path/to/drug_graph_cache.pkl \
      --output_dir /path/to/output/gat_cross \
      --device cuda:0 \
      --seed 42 \
      --epochs 50 \
      --batch_size 64 \
      --hidden_dim 128 \
      --embed_dim 256 \
      --num_heads 8 \
      --gat_heads 4 \
      --dropout 0.2 \
      --lr 1e-4 \
      --weight_decay 0.01 \
      --patience 10

The required feature cache can be prepared from locally obtained source data
using `baselines/prepare_exact_949_cache.py`. Raw GDSC and expression data are
not redistributed.

### CSG2A

`baselines/train_exact_csg2a.py` is a wrapper around the external CSG2A
reference implementation. The original CSG2A source tree and LINCS-pretrained
checkpoint must be obtained separately and are not redistributed in this
repository.

Example command:

    python baselines/train_exact_csg2a.py \
      --feature_cache /path/to/prepared_949_feature_cache \
      --csg2a_root /path/to/CSG2A/source \
      --pretrained_checkpoint /path/to/CSG2A/pretrained_checkpoint.pth \
      --output_dir /path/to/output/csg2a \
      --device cuda:0 \
      --seed 42 \
      --batch_size 128 \
      --max_epochs 200 \
      --patience 20 \
      --lr_init 1e-4 \
      --lr_final 1e-5 \
      --weight_decay 1e-5 \
      --dropout 0.1 \
      --gene_hdim 64 \
      --finetune_hdim1 512 \
      --finetune_hdim2 64

The wrapper freezes the pretrained CSG2A backbone and uses validation MSE as
the primary checkpoint-selection criterion, matching the reported comparison.

## Reproducing the reported drug-blind DRT run

The exact arguments and provenance of the reported runs are preserved in
`reproducibility/reports/`. The following reproduces the reported pretrained
DRT configuration for split seed 42 and optimization seed 42, given locally
prepared input files and the selected LINCS checkpoint.

```bash
python final_union_da_ic50_finetune.py \
  --csv_path /path/to/gdsc_processed.csv \
  --basal_csv /path/to/gdsc_basal_expression.csv \
  --landmark_csv /path/to/kegg_lincs_landmark_overlap.csv \
  --drug_graph_cache /path/to/drug_graph_cache.pt \
  --cell_graph_cache /path/to/cell_graph_cache.pt \
  --pretrained_checkpoint /path/to/da_union_pge_full_ep6_bs32_s42_checkpoint.pth \
  --save_dir /path/to/output \
  --run_name drt_drugblind_pretrained_s42 \
  --split_mode drug_blind \
  --split_seed 42 \
  --seed 42 \
  --epochs 18 \
  --batch_size 32 \
  --num_workers 0 \
  --fixed_dose 1.0 \
  --fixed_time 72.0 \
  --body_lr 2e-4 \
  --head_lr 7e-4 \
  --weight_decay 5e-4 \
  --warmup_epochs 2 \
  --alpha_corr 0.2 \
  --corr_start_epoch 3 \
  --patience 5 \
  --min_delta 0.001 \
  --grad_clip 1.0 \
  --max_num_nodes 96 \
  --num_pathways 31 \
  --mode pretrained \
  --edge_encoding scalar \
  --edge_direction direction_aware_bidirectional \
  --amp
```

The exact drug-blind response-row assignments used in the manuscript are
provided under `splits/drug_blind/`. Checkpoint hashes, selected epochs, and
test metrics are recorded in `reproducibility/drt_drugblind_run_index.tsv`.

## Reproducing LINCS pretraining

The final reported backbone was trained with split seed 42 and optimization
seed 42 for six epochs. The command below reproduces the reported training
configuration given locally obtained source data and prepared graph caches.

Raw LINCS and KEGG resources are not redistributed and must be obtained from
their original providers under the applicable terms.

```bash
python pge_pretrain_da_bidir_union.py \
  --mode train \
  --perturbed_csv /path/to/pretraining_profiles.csv \
  --basal_csv /path/to/lincs_basal_expression.csv \
  --kegg_pathway_dir /path/to/kegg_pathway_files \
  --landmark_csv /path/to/kegg_lincs_landmark_overlap.csv \
  --cache_dir /path/to/runtime_cache \
  --drug_graph_cache /path/to/drug_graph_cache.pt \
  --cell_graph_cache /path/to/cell_graph_cache.pt \
  --cache_scope all \
  --save_dir /path/to/output/checkpoints \
  --run_name da_union_pge_full_ep6_bs32_s42 \
  --device cuda:0 \
  --num_workers 0 \
  --valid_ratio 0.1 \
  --test_ratio 0.1 \
  --split_seed 42 \
  --split_mode mixed \
  --epochs 6 \
  --batch_size 32 \
  --grad_clip 1.0 \
  --warmup_epochs 1 \
  --accum_steps 1 \
  --body_lr 2e-4 \
  --head_lr 7e-4 \
  --weight_decay 5e-4 \
  --dim_node 64 \
  --dim_drug 64 \
  --dim_cell 64 \
  --transformer_layers 2 \
  --transformer_heads 8 \
  --dropout_ratio 0.1 \
  --max_num_nodes 96 \
  --ffn_dim 256 \
  --pe_dim 1 \
  --num_pathways 31 \
  --use_pathway_batching \
  --seed 42 \
  --zscore_by_gene \
  --amp
```

The selected backbone was the epoch-6 checkpoint. Its SHA256 digest and
training provenance are provided in
`reproducibility/provenance/final_pretraining.tsv`. The checkpoint binary
itself is not redistributed in the current public release. A sanitized report
for the selected run is provided in
`reproducibility/reports/final_lincs_pretraining.report.json`.

## Integrated Gradients analysis

The representation-level Integrated Gradients (IG) analysis reported in the
manuscript used the mixed-split pretrained DRT checkpoint (split seed 42,
optimization seed 42).

One test observation was selected for each of 413 canonical molecular
structures. IG was evaluated at the 268 pathway-aggregated gene-token
positions using the learned gene-identity embeddings as the baseline while
drug atom tokens, dose/time context, token masks, and fitted parameters were
held fixed.

The manuscript-reported agreement statistics correspond to a 128-point
Gauss-Legendre quadrature run. A separate 64-point convergence run produced
nearly identical agreement estimates.

Example:

```bash
IG_BASELINE_MODE=gene_embedding \
IG_QUADRATURE=gauss_legendre \
python analysis/integrated_gradients/run_final_mixed_ig_v2.py \
  --checkpoint /path/to/final_mixed_pretrained_seed42_checkpoint.pth \
  --device cuda:0 \
  --steps 128 \
  --per_structure 1 \
  --max_structures 0 \
  --selection_seed 42 \
  --equivalence_tol 5e-6 \
  --out /path/to/ig_output
```

The v2 script replaces only the numerical IG integration routine while
retaining the audited model loading, sample selection, attention extraction,
and evaluation workflow implemented in `analysis/integrated_gradients/run_final_mixed_ig.py`.

Public derived outputs are provided under `results/integrated_gradients/`.
`manuscript_reported_summary.json` records the reported run configuration,
checkpoint SHA256, and agreement statistics.

## Global representation and residual-hybrid analyses

Reproducibility code and manuscript-canonical results for the standalone
Global MLP and frozen-core residual hybrid are provided under:

- `analysis/global_representation/`
- `results/global_representation/`

Both analyses use a 1,024-bit radius-2 Morgan fingerprint and the 1,954-gene
harmonized basal-expression representation.

The exact mixed-split equivalence audit confirms identical source-row order
and labels across the standalone Global MLP and final residual-hybrid
train/validation/test feature caches.

See `analysis/global_representation/README.md` for the reported run settings
and metrics.

## Drug-blind main-effect decomposition

Code and canonical derived outputs for the two-way cell-line/drug
main-effect decomposition reported in Supplementary Table S6 are provided in:

- `analysis/main_effect_decomposition/`
- `results/main_effect_decomposition/`

See `analysis/main_effect_decomposition/README.md` for the analysis definition
and reported summary values.

## Classical cell-blind baselines

The exact Ridge and XGBoost cell-blind baseline pipeline, canonical split
manifests, and provenance are available under:

`baselines/classical/cell_blind/`

Reported Ridge/XGBoost metrics are retained under
`reproducibility/reports/cell_blind/`.

## Classical mixed-split baselines

The exact mixed-split Ridge, Random Forest, and XGBoost reproducibility package, including canonical split manifests, model-specific provenance, retained final reports, and portable launch scripts, is available under:

`baselines/classical/mixed/`

See `baselines/classical/mixed/README.md` for exact final settings, provenance caveats, and reproduction commands.
