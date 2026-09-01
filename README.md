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
0.8630 ± 0.0026, and macro per-drug PCC 0.3098 ± 0.0203. The final-aligned
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
