# Data preparation

DrugResponseTransformer (DRT) uses public LINCS L1000, GDSC, Cell Model
Passports, and KEGG resources. Raw source datasets and KEGG pathway files are
not redistributed in this repository and must be obtained from their original
providers under the applicable terms.

The public repository contains identifiers, split manifests, configuration
metadata, and derived reproducibility materials required to reconstruct the
reported experiments.

## 1. LINCS L1000 pretraining data

The reported pretraining experiment used LINCS L1000 beta-release Level 5
perturbation profiles.

The final retained corpus contained 591,912 small-molecule perturbation
profiles after molecular-graph filtering.

Expected pretraining table schema:

- `cell_iname`: LINCS cell-line identifier
- `canonical_smiles`: RDKit-canonicalized molecular structure
- `converted_dose`: log10 drug concentration in micromolar units
- `pert_itime`: exposure time in hours
- 268 Entrez-gene columns: perturbational-expression targets

The 268 target genes are the unique landmark genes shared between the LINCS
landmark set and the selected KEGG pathway collection.

LINCS source releases include GEO accessions GSE92742 and GSE70138.

## 2. LINCS basal expression

The LINCS basal-expression input contains:

- `cell_iname`
- Entrez-gene expression columns

Basal profiles were derived from available DMSO controls and are used as
cell-line-specific node features for pathway graph construction.

## 3. Landmark gene mapping

`kegg_lincs_landmark_overlap.csv` contains a single column:

- `landmark_overlap_genes`

This file defines the ordered landmark-gene subset used by the graph/token
pipeline. The final downstream token and pretraining-target space contains
268 unique genes.

## 4. GDSC drug-response data

GDSC1 and GDSC2 response records were combined using the provided `LN_IC50`
values directly. No additional logarithmic transformation is applied.

The processed response table contains at minimum:

- `CELL_LINE_NAME`
- `DRUG_NAME`
- `MIN_CONC`
- `MAX_CONC`
- `LN_IC50`
- `canonical_smiles`
- Entrez-gene expression columns

Distinct GDSC1/GDSC2 measurements of the same drug-cell combination are
retained as separate response records.

After graph and cell-representation coverage filtering, the final dataset
contains 415,671 measurements, 413 canonical molecular structures, 417
drug-name entries, and 970 cell lines.

## 5. GDSC basal-expression harmonization

Basal expression is obtained from Cell Model Passports.

The final harmonized feature space contains 1,954 matched genes.

For each gene, expression is processed as:

1. `x' = log2(x + 1)`
2. gene-wise standardization
3. clipping to `[-3, 3]`
4. rescaling to the corresponding LINCS basal-expression domain

For mixed and drug-blind evaluation, GDSC-domain normalization statistics are
estimated using all 970 represented cell lines.

For cell-blind evaluation, statistics are estimated using only the 776
training cell lines and then applied unchanged to validation and test cells.

The final GDSC basal-expression table contains:

- `CELL_LINE_NAME`
- 1,954 Entrez-gene expression columns

## 6. KEGG pathway resources

The model uses 31 selected human KEGG pathways.

The pathway identifiers are listed in:

`resources/kegg_pathway_ids.txt`

KEGG XML/KGML files are not redistributed. Users must obtain pathway files
directly from KEGG subject to KEGG terms and place them in a local directory
provided through `--kegg_pathway_dir`.

## 7. Derived graph caches

The training scripts use locally generated PyTorch graph caches.

Typical inputs include:

- molecular graph cache (`--drug_graph_cache`)
- pathway/cell graph cache (`--cell_graph_cache`)
- optional runtime cache (`--cache_dir`)

These binary caches are generated from the locally obtained source datasets
and are not included in the current public release.

Molecular graphs use RDKit-canonicalized SMILES, a maximum of 96 atoms,
57-dimensional atom features, four bond-type channels, and a one-dimensional
Laplacian positional encoding.

Cell graphs use basal expression as node features and the selected KEGG
pathway topology.

## 8. Public split manifests

Exact public-safe split assignments used by the reported experiments are
provided under `splits/`.

Drug-blind manifests contain:

- `source_row`
- `cell_id`
- `drug_id`
- `canonical_smiles`

The reported drug-blind split contains 334,320 / 38,903 / 42,448
training/validation/test response records.

Cell-blind manifests contain only:

- `cell_id`

The reported cell-blind split contains 776 / 97 / 97
training/validation/test cell lines.

Response labels are not redistributed through the split manifests.

## 9. Reproducibility metadata

Exact training settings and sanitized run reports are provided under:

- `configs/`
- `reproducibility/reports/`
- `reproducibility/provenance/`
- `results/`

The final LINCS-pretraining checkpoint provenance is recorded in:

`reproducibility/provenance/final_pretraining.tsv`

The checkpoint binary itself is not redistributed in the current public
release.
