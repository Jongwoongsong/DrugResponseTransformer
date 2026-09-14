"""
Canonical helper functions used by the exact cell-blind classical baselines.

Extracted from the audited shared cell-blind baseline implementation.
Only helpers required by the Ridge/XGBoost pipeline are retained.
"""

from __future__ import print_function


import argparse


import csv


import hashlib


import json


import math


import os


import random


import shutil


import sys


import time


from pathlib import Path


import numpy as np


import pandas as pd


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit("PyTorch is required: {}".format(exc))

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError as exc:
    raise SystemExit("SciPy is required: {}".format(exc))

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    try:
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        rdFingerprintGenerator = None
except ImportError as exc:
    raise SystemExit("RDKit is required: {}".format(exc))

RDLogger.DisableLog("rdApp.*")



CELL_COLUMN_CANDIDATES = [
    "CELL_LINE_NAME",
    "CELL_LINE",
    "cell_line_name",
    "cell_line",
    "CellLine",
    "MODEL_NAME",
    "model_name",
    "CELL",
    "cell",
]


SMILES_COLUMN_CANDIDATES = [
    "SMILES",
    "smiles",
    "CANONICAL_SMILES",
    "Canonical_SMILES",
    "canonical_smiles",
    "PUBCHEM_SMILES",
]


LABEL_COLUMN_CANDIDATES = [
    "LN_IC50",
    "ln_ic50",
    "LOG_IC50",
    "log_ic50",
    "IC50",
    "ic50",
]


DRUG_COLUMN_CANDIDATES = [
    "DRUG_ID",
    "drug_id",
    "DRUG_NAME",
    "drug_name",
    "Drug",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_file(path, label):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("{} not found: {}".format(label, p))
    return p.resolve()


def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_string_series(series):
    return series.fillna("").astype(str).str.strip()


def resolve_column(df, explicit, candidates, kind):
    if explicit:
        if explicit not in df.columns:
            raise KeyError(
                "Requested {} column '{}' does not exist. Available columns: {}".format(
                    kind, explicit, list(df.columns)
                )
            )
        return explicit

    lower_to_original = {}
    for col in df.columns:
        lower_to_original.setdefault(str(col).lower(), []).append(col)

    for candidate in candidates:
        hits = lower_to_original.get(candidate.lower(), [])
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RuntimeError(
                "Ambiguous {} column for candidate '{}': {}".format(
                    kind, candidate, hits
                )
            )

    raise RuntimeError(
        "Could not identify {} column. Use --{}-col. Available columns: {}".format(
            kind, kind.replace("_", "-"), list(df.columns)
        )
    )


def resolve_optional_column(df, explicit, candidates):
    if explicit:
        if explicit not in df.columns:
            raise KeyError(
                "Requested drug column '{}' does not exist".format(explicit)
            )
        return explicit

    lower_to_original = {}
    for col in df.columns:
        lower_to_original.setdefault(str(col).lower(), []).append(col)

    for candidate in candidates:
        hits = lower_to_original.get(candidate.lower(), [])
        if len(hits) == 1:
            return hits[0]
    return None


def safe_corr(func, y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float("nan")
    result = func(y_true, y_pred)
    if isinstance(result, tuple):
        return float(result[0])
    if hasattr(result, "statistic"):
        return float(result.statistic)
    return float(result)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    centered = y_true - np.mean(y_true)
    ss_tot = float(np.sum(centered ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return {
        "n": int(len(y_true)),
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "pearson": safe_corr(pearsonr, y_true, y_pred),
        "spearman": safe_corr(spearmanr, y_true, y_pred),
        "r2": r2,
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
    }


# -----------------------------------------------------------------------------
# Exact split manifests
# -----------------------------------------------------------------------------


def load_single_cell_manifest(path):
    df = pd.read_csv(str(path), low_memory=False)
    visible_cols = [
        c for c in df.columns if not str(c).lower().startswith("unnamed")
    ]
    if not visible_cols:
        raise RuntimeError("No usable columns in manifest: {}".format(path))

    preferred = [
        c
        for c in visible_cols
        if "cell" in str(c).lower() or "model" in str(c).lower()
    ]
    if len(visible_cols) == 1:
        col = visible_cols[0]
    elif len(preferred) == 1:
        col = preferred[0]
    else:
        raise RuntimeError(
            "Cannot uniquely identify cell column in {}: {}".format(
                path, visible_cols
            )
        )

    values = normalize_string_series(df[col])
    values = values[values != ""]
    if values.duplicated().any():
        duplicates = values[values.duplicated()].unique().tolist()[:10]
        raise RuntimeError(
            "Duplicate cells in {}: {}".format(path, duplicates)
        )
    return set(values.tolist()), col


def load_and_audit_manifests(split_dir, fit_cells_path=None):
    paths = {
        "train": split_dir / "cell_blind_train_cells.csv",
        "validation": split_dir / "cell_blind_validation_cells.csv",
        "test": split_dir / "cell_blind_test_cells.csv",
    }
    for split_name, path in paths.items():
        ensure_file(path, "{} cell manifest".format(split_name))

    sets = {}
    columns = {}
    for split_name, path in paths.items():
        sets[split_name], columns[split_name] = load_single_cell_manifest(path)

    pair_overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }
    if any(v != 0 for v in pair_overlaps.values()):
        raise RuntimeError("Cell manifest overlap detected: {}".format(pair_overlaps))

    union = sets["train"] | sets["validation"] | sets["test"]
    expected_counts = {"train": 776, "validation": 97, "test": 97}
    observed_counts = {k: len(v) for k, v in sets.items()}
    if observed_counts != expected_counts:
        raise RuntimeError(
            "Unexpected cell counts. observed={} expected={}".format(
                observed_counts, expected_counts
            )
        )
    if len(union) != 970:
        raise RuntimeError("Expected 970 total cells, found {}".format(len(union)))

    fit_audit = None
    if fit_cells_path is not None:
        fit_set, fit_col = load_single_cell_manifest(fit_cells_path)
        fit_audit = {
            "path": str(fit_cells_path),
            "column": str(fit_col),
            "n": len(fit_set),
            "identical_to_train": fit_set == sets["train"],
            "manifest_only": len(sets["train"] - fit_set),
            "fit_only": len(fit_set - sets["train"]),
        }
        if not fit_audit["identical_to_train"]:
            raise RuntimeError(
                "Normalization fit cells do not exactly match train manifest: {}".format(
                    fit_audit
                )
            )

    audit = {
        "paths": {k: str(v) for k, v in paths.items()},
        "columns": columns,
        "cell_counts": observed_counts,
        "total_unique_cells": len(union),
        "pairwise_overlaps": pair_overlaps,
        "fit_cells_audit": fit_audit,
        "status": "PASS",
    }
    return sets, paths, audit


# -----------------------------------------------------------------------------
# Filtering and feature construction
# -----------------------------------------------------------------------------


def canonicalize_smiles(raw_smiles):
    text = "" if raw_smiles is None else str(raw_smiles).strip()
    if text == "" or text.lower() in ("nan", "none", "null") or text.isdigit():
        return None, "blank_or_digits"
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None, "rdkit_fail"
    return Chem.MolToSmiles(mol, canonical=True), "valid"


def assign_exact_split(valid_df, manifest_sets, split_report_path):
    cell_to_split = {}
    for split_name, cell_set in manifest_sets.items():
        for cell in cell_set:
            if cell in cell_to_split:
                raise RuntimeError("Cell occurs in multiple splits: {}".format(cell))
            cell_to_split[cell] = split_name

    valid_df = valid_df.copy()
    valid_df["_split"] = valid_df["_cell"].map(cell_to_split)
    unassigned = valid_df[valid_df["_split"].isna()]["_cell"].unique().tolist()
    if unassigned:
        raise RuntimeError(
            "Valid rows contain cells absent from manifests (first 20): {}".format(
                unassigned[:20]
            )
        )

    with open(str(split_report_path), "r", encoding="utf-8") as handle:
        report = json.load(handle)

    try:
        expected = {
            "train": int(report["splits"]["cell_blind"]["train"]["rows"]),
            "validation": int(
                report["splits"]["cell_blind"]["validation"]["rows"]
            ),
            "test": int(report["splits"]["cell_blind"]["test"]["rows"]),
        }
    except Exception as exc:
        raise RuntimeError(
            "Could not read expected cell-blind row counts from {}: {}".format(
                split_report_path, exc
            )
        )

    observed = {
        split_name: int((valid_df["_split"] == split_name).sum())
        for split_name in ("train", "validation", "test")
    }
    if observed != expected:
        per_cell = (
            valid_df.groupby(["_split", "_cell"])
            .size()
            .reset_index(name="rows")
            .sort_values(["_split", "_cell"])
        )
        raise RuntimeError(
            "Exact row-count audit failed. observed={} expected={}\n"
            "Per-cell summary head:\n{}".format(
                observed, expected, per_cell.head(20).to_string(index=False)
            )
        )

    audit = {
        "expected_rows": expected,
        "observed_rows": observed,
        "exact_match": True,
        "cell_counts_from_rows": {
            split_name: int(
                valid_df.loc[valid_df["_split"] == split_name, "_cell"].nunique()
            )
            for split_name in ("train", "validation", "test")
        },
        "status": "PASS",
    }
    print("[SPLIT AUDIT] {}".format(json.dumps(audit, sort_keys=True)), flush=True)
    return valid_df, audit


def detect_basal_cell_column(basal_df, expected_cells):
    scores = []
    expected_n = len(expected_cells)
    for col in basal_df.columns:
        values = normalize_string_series(basal_df[col])
        values = set(values[values != ""].tolist())
        overlap = len(values & expected_cells)
        scores.append((overlap, col))
    scores.sort(key=lambda x: x[0], reverse=True)
    best_overlap, best_col = scores[0]
    if best_overlap < int(0.95 * expected_n):
        raise RuntimeError(
            "Could not detect basal cell column. Best overlap={}/{} at '{}'. Top: {}".format(
                best_overlap, expected_n, best_col, scores[:5]
            )
        )
    return best_col, best_overlap


def load_cell_feature_bank(basal_path, all_cells, expected_feature_count):
    print("[LOAD] basal-expression CSV -> {}".format(basal_path), flush=True)
    basal = pd.read_csv(str(basal_path), low_memory=False)
    cell_col, overlap = detect_basal_cell_column(basal, all_cells)
    basal[cell_col] = normalize_string_series(basal[cell_col])

    if basal[cell_col].duplicated().any():
        duplicates = basal.loc[basal[cell_col].duplicated(), cell_col].unique().tolist()
        raise RuntimeError("Duplicate basal cells: {}".format(duplicates[:20]))

    basal = basal.set_index(cell_col, drop=True)
    missing_cells = sorted(all_cells - set(basal.index.tolist()))
    if missing_cells:
        raise RuntimeError(
            "Basal CSV is missing manifest cells (first 20): {}".format(
                missing_cells[:20]
            )
        )

    numeric_columns = []
    skipped_columns = []
    converted = {}
    for col in basal.columns:
        series = pd.to_numeric(basal[col], errors="coerce")
        selected = series.loc[list(all_cells)]
        if selected.notna().all():
            numeric_columns.append(col)
            converted[col] = series
        else:
            skipped_columns.append(col)

    if len(numeric_columns) != expected_feature_count:
        raise RuntimeError(
            "Expected {} numeric cell features, found {}. "
            "Skipped columns (first 20): {}".format(
                expected_feature_count,
                len(numeric_columns),
                [str(x) for x in skipped_columns[:20]],
            )
        )

    cell_names = sorted(all_cells)
    matrix_df = pd.DataFrame(
        {col: converted[col].loc[cell_names] for col in numeric_columns},
        index=cell_names,
    )
    matrix = matrix_df.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite values in cell feature matrix")

    cell_to_idx = {cell: idx for idx, cell in enumerate(cell_names)}
    metadata = {
        "path": str(basal_path),
        "cell_column": str(cell_col),
        "cell_overlap_for_detection": int(overlap),
        "n_cells": int(len(cell_names)),
        "n_features": int(matrix.shape[1]),
        "feature_columns": [str(x) for x in numeric_columns],
        "matrix_mean": float(matrix.mean()),
        "matrix_std": float(matrix.std()),
        "matrix_min": float(matrix.min()),
        "matrix_max": float(matrix.max()),
    }
    print(
        "[CELL FEATURES] cells={} features={} range=[{:.4f}, {:.4f}]".format(
            matrix.shape[0], matrix.shape[1], matrix.min(), matrix.max()
        ),
        flush=True,
    )
    return matrix, cell_names, cell_to_idx, metadata


def morgan_fp(smiles, radius, fp_size, generator=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Canonical SMILES unexpectedly failed: {}".format(smiles))
    if generator is not None:
        bitvect = generator.GetFingerprint(mol)
    else:
        bitvect = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=fp_size
        )
    arr = np.zeros((fp_size,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr.astype(np.float32, copy=False)


def build_drug_feature_bank(valid_df, radius, fp_size):
    smiles_list = sorted(valid_df["_canonical_smiles"].unique().tolist())
    generator = None
    if rdFingerprintGenerator is not None:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fp_size
        )
    matrix = np.stack(
        [morgan_fp(s, radius, fp_size, generator) for s in smiles_list], axis=0
    ).astype(np.float32, copy=False)
    smiles_to_idx = {smiles: idx for idx, smiles in enumerate(smiles_list)}
    metadata = {
        "n_smiles": int(len(smiles_list)),
        "fp_size": int(fp_size),
        "radius": int(radius),
        "mean_bits_on": float(matrix.sum(axis=1).mean()),
        "min_bits_on": int(matrix.sum(axis=1).min()),
        "max_bits_on": int(matrix.sum(axis=1).max()),
    }
    print(
        "[DRUG FEATURES] smiles={} fp={} mean_bits_on={:.2f}".format(
            len(smiles_list), fp_size, metadata["mean_bits_on"]
        ),
        flush=True,
    )
    return matrix, smiles_list, smiles_to_idx, metadata


# -----------------------------------------------------------------------------
# Dataset and model
# -----------------------------------------------------------------------------
