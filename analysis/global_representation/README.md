# Global representation and residual-hybrid analyses

This directory contains the code used for the mixed-split standalone Global
MLP and frozen-core residual-hybrid analyses.

## Inputs

Both models use:

- 1,024-bit radius-2 Morgan fingerprint
- 1,954-gene harmonized basal-expression vector

The drug and cell branches each produce 128-dimensional representations.
Fusion concatenates the drug representation, cell representation,
element-wise product, and absolute difference, yielding 512 dimensions before
the `512 -> 256 -> 64 -> 1` prediction head.

## Standalone Global MLP

The manuscript-reported standalone Global MLP is the preserved mixed-split
seed-42 run with:

- learning rate: 5e-4
- batch size: 2048
- dropout: 0.15
- weight decay: 1e-4
- patience: 5
- best epoch: 15

Reported test metrics:

- RMSE: 1.0509412474
- MAE: 0.7738260755
- pooled PCC: 0.9246131782
- macro within-structure PCC: 0.6803970042

## Frozen-core residual hybrid

The manuscript-reported residual hybrid uses predictions from the final mixed
core DRT as a frozen input and trains a separate global branch to predict the
residual correction.

The manuscript-canonical residual run uses seed 42, batch size 2048,
learning rate 3e-4, dropout 0.15, weight decay 1e-4, and patience 5.
The selected checkpoint was epoch 9.

Reported test metrics:

- RMSE: 1.0509395340
- MAE: 0.7692860702
- pooled PCC: 0.9247458298
- macro within-structure PCC: 0.6873853966

The corresponding frozen core DRT has RMSE 1.2394000386 and pooled PCC
0.8940325877 on the identical test rows.

## Split equivalence

`results/global_representation/mixed_split_equivalence.json` verifies that
the standalone Global MLP and final residual-hybrid feature caches contain
identical source-row order and labels for all train, validation, and test
partitions.

Binary model checkpoints and source datasets are not redistributed.
