#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact-manifest classical global baselines for canonical GDSC970 cell-blind evaluation.

Fair input representation (identical to the exact cell-blind MLP baseline)
-------------------------------------------------------------------------
Drug : Morgan fingerprint (radius=2, 1024 bits by default)
Cell : harmonized basal-expression vector fitted using the 776 training cells
Input: concatenation [drug_fp, cell_expression]
Models: Ridge and XGBoost
Target: LN_IC50 used directly

The script imports the audit/feature helpers from the exact MLP script so that
cell manifests, SMILES canonicalization, basal feature detection, and metric
implementations remain aligned. The canonical CSV itself is loaded minimally
(cell, SMILES, label, drug only) to avoid retaining ~1,954 unused raw columns
while constructing the ~5 GB concatenated feature matrices.

Compatible with Python 3.7+.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError as exc:
    raise SystemExit("SciPy is required: {}".format(exc))

try:
    from sklearn.linear_model import Ridge
except ImportError as exc:
    raise SystemExit("scikit-learn is required: {}".format(exc))


# -----------------------------------------------------------------------------
# CLI and utilities
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Exact cell-blind Ridge/XGBoost baselines for canonical GDSC970"
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--basal-csv", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--split-report", required=True)
    p.add_argument("--fit-cells-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mlp-metrics", default="")

    p.add_argument("--cell-col", default="")
    p.add_argument("--smiles-col", default="")
    p.add_argument("--label-col", default="")
    p.add_argument("--drug-col", default="")

    p.add_argument("--models", default="ridge,xgb")
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--expected-cell-features", type=int, default=1954)
    p.add_argument("--feature-chunk-size", type=int, default=8192)
    p.add_argument("--reuse-feature-cache", action="store_true")
    p.add_argument("--keep-feature-cache", action="store_true")

    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--ridge-tol", type=float, default=1.0e-4)
    p.add_argument("--ridge-max-iter", type=int, default=1000)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument(
        "--xgb-device", choices=["auto", "gpu", "cpu"], default="auto"
    )
    p.add_argument("--xgb-n-estimators", type=int, default=1000)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-learning-rate", type=float, default=0.03)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample", type=float, default=0.8)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    p.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    p.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    p.add_argument("--xgb-max-bin", type=int, default=256)
    p.add_argument("--xgb-early-stopping", type=int, default=50)

    p.add_argument("--skip-hashes", action="store_true")
    return p.parse_args()


def ensure_file(path, label):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("{} not found: {}".format(label, p))
    return p.resolve()


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def json_dump(obj, path):
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, sort_keys=True)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_shared_module(path):
    spec = importlib.util.spec_from_file_location("exact_cellblind_mlp_shared", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import shared MLP script: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_thread_env(threads):
    value = str(int(threads))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = value


# -----------------------------------------------------------------------------
# Canonical filtering, exactly matching the MLP logic but loading fewer columns
# -----------------------------------------------------------------------------


def filter_canonical_dataframe_minimal(
    shared, csv_path, cell_col_arg, smiles_col_arg, label_col_arg, drug_col_arg
):
    print("[LOAD] canonical CSV header -> {}".format(csv_path), flush=True)
    header = pd.read_csv(str(csv_path), nrows=0)
    cell_col = shared.resolve_column(
        header, cell_col_arg, shared.CELL_COLUMN_CANDIDATES, "cell"
    )
    smiles_col = shared.resolve_column(
        header, smiles_col_arg, shared.SMILES_COLUMN_CANDIDATES, "smiles"
    )
    label_col = shared.resolve_column(
        header, label_col_arg, shared.LABEL_COLUMN_CANDIDATES, "label"
    )
    drug_col = shared.resolve_optional_column(
        header, drug_col_arg, shared.DRUG_COLUMN_CANDIDATES
    )

    usecols = []
    for col in (cell_col, smiles_col, label_col, drug_col):
        if col is not None and col not in usecols:
            usecols.append(col)

    print("[LOAD] minimal columns={}".format(usecols), flush=True)
    df = pd.read_csv(str(csv_path), usecols=usecols, low_memory=False)
    df["_source_row_index"] = np.arange(len(df), dtype=np.int64)

    input_rows = int(len(df))
    df[cell_col] = shared.normalize_string_series(df[cell_col])
    raw_smiles = shared.normalize_string_series(df[smiles_col])
    y_numeric = pd.to_numeric(df[label_col], errors="coerce")
    label_mask = np.isfinite(y_numeric.to_numpy(dtype=np.float64))
    label_valid_rows = int(label_mask.sum())

    work = df.loc[label_mask].copy()
    work[label_col] = y_numeric.loc[label_mask].astype(np.float32)
    work[smiles_col] = raw_smiles.loc[label_mask]

    unique_raw = work[smiles_col].drop_duplicates().tolist()
    canonical_map = {}
    reason_map = {}
    for raw in unique_raw:
        canonical, reason = shared.canonicalize_smiles(raw)
        canonical_map[raw] = canonical
        reason_map[raw] = reason

    reasons = work[smiles_col].map(reason_map)
    blank_or_digits = int((reasons == "blank_or_digits").sum())
    rdkit_fail = int((reasons == "rdkit_fail").sum())

    work["_canonical_smiles"] = work[smiles_col].map(canonical_map)
    valid = work[work["_canonical_smiles"].notna()].copy()
    valid["_canonical_smiles"] = valid["_canonical_smiles"].astype(str)
    valid["_label"] = valid[label_col].astype(np.float32)
    valid["_cell"] = shared.normalize_string_series(valid[cell_col])

    filtering = {
        "path": str(csv_path),
        "input_rows": input_rows,
        "label_valid_rows": label_valid_rows,
        "blank_or_digits": blank_or_digits,
        "rdkit_fail": rdkit_fail,
        "valid_rows": int(len(valid)),
        "unique_cells": int(valid["_cell"].nunique()),
        "unique_smiles": int(valid["_canonical_smiles"].nunique()),
        "drugs": int(valid[drug_col].nunique()) if drug_col is not None else None,
        "resolved_columns": {
            "cell": str(cell_col),
            "smiles": str(smiles_col),
            "label": str(label_col),
            "drug": str(drug_col) if drug_col is not None else None,
        },
        "minimal_column_load": True,
    }
    print("[FILTER] {}".format(json.dumps(filtering, sort_keys=True)), flush=True)
    return valid, filtering, drug_col


def audit_filtering_against_report(filtering, split_report_path):
    with open(str(split_report_path), "r", encoding="utf-8") as handle:
        report = json.load(handle)
    expected = report.get("filtering", {})
    keys = [
        "input_rows",
        "label_valid_rows",
        "blank_or_digits",
        "rdkit_fail",
        "valid_rows",
        "unique_cells",
        "unique_smiles",
    ]
    mismatches = {}
    for key in keys:
        observed = filtering.get(key)
        wanted = expected.get(key)
        if observed != wanted:
            mismatches[key] = {"observed": observed, "expected": wanted}
    if mismatches:
        raise RuntimeError(
            "Filtering does not match canonical split report: {}".format(mismatches)
        )
    audit = {
        "status": "PASS",
        "checked_keys": keys,
        "mismatches": {},
    }
    print("[FILTER AUDIT] PASS exact canonical filtering", flush=True)
    return audit


# -----------------------------------------------------------------------------
# Disk-backed feature matrices
# -----------------------------------------------------------------------------


def cache_paths(cache_dir, split_name):
    return {
        "x": cache_dir / "{}_X.npy".format(split_name),
        "y": cache_dir / "{}_y.npy".format(split_name),
        "source": cache_dir / "{}_source_row.npy".format(split_name),
        "cell_idx": cache_dir / "{}_cell_idx.npy".format(split_name),
        "drug_idx": cache_dir / "{}_drug_idx.npy".format(split_name),
    }


def build_or_load_split_cache(
    frame,
    split_name,
    cache_dir,
    cell_matrix,
    drug_matrix,
    cell_to_idx,
    smiles_to_idx,
    chunk_size,
    reuse,
):
    paths = cache_paths(cache_dir, split_name)
    n_rows = int(len(frame))
    n_features = int(drug_matrix.shape[1] + cell_matrix.shape[1])

    if reuse and all(path.is_file() for path in paths.values()):
        x = np.load(str(paths["x"]), mmap_mode="r")
        y = np.load(str(paths["y"]), mmap_mode="r")
        source = np.load(str(paths["source"]), mmap_mode="r")
        cell_idx = np.load(str(paths["cell_idx"]), mmap_mode="r")
        drug_idx = np.load(str(paths["drug_idx"]), mmap_mode="r")
        if x.shape != (n_rows, n_features) or y.shape != (n_rows,):
            raise RuntimeError(
                "Cached {} shape mismatch: X={} y={} expected=({}, {})".format(
                    split_name, x.shape, y.shape, n_rows, n_features
                )
            )
        print(
            "[FEATURE CACHE][{}] REUSE shape={} mem={:.2f}GB".format(
                split_name, x.shape, x.nbytes / 1.0e9
            ),
            flush=True,
        )
        return x, y, source, cell_idx, drug_idx, paths

    for path in paths.values():
        if path.exists():
            path.unlink()

    mapped_cells = frame["_cell"].map(cell_to_idx)
    mapped_drugs = frame["_canonical_smiles"].map(smiles_to_idx)
    if mapped_cells.isna().any() or mapped_drugs.isna().any():
        raise RuntimeError("Feature-bank index mapping failed for {}".format(split_name))

    cell_idx_arr = mapped_cells.to_numpy(dtype=np.int32, copy=True)
    drug_idx_arr = mapped_drugs.to_numpy(dtype=np.int32, copy=True)
    y_arr = frame["_label"].to_numpy(dtype=np.float32, copy=True)
    source_arr = frame["_source_row_index"].to_numpy(dtype=np.int64, copy=True)

    np.save(str(paths["y"]), y_arr)
    np.save(str(paths["source"]), source_arr)
    np.save(str(paths["cell_idx"]), cell_idx_arr)
    np.save(str(paths["drug_idx"]), drug_idx_arr)

    print(
        "[FEATURE CACHE][{}] BUILD rows={} features={} estimated={:.2f}GB".format(
            split_name, n_rows, n_features, n_rows * n_features * 4 / 1.0e9
        ),
        flush=True,
    )
    started = time.time()
    xmap = np.lib.format.open_memmap(
        str(paths["x"]), mode="w+", dtype=np.float32, shape=(n_rows, n_features)
    )
    drug_dim = int(drug_matrix.shape[1])
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        xmap[start:end, :drug_dim] = drug_matrix[drug_idx_arr[start:end]]
        xmap[start:end, drug_dim:] = cell_matrix[cell_idx_arr[start:end]]
        if start == 0 or end == n_rows or end % (chunk_size * 10) == 0:
            print(
                "[FEATURE CACHE][{}] {}/{}".format(split_name, end, n_rows),
                flush=True,
            )
    xmap.flush()
    del xmap

    x = np.load(str(paths["x"]), mmap_mode="r")
    y = np.load(str(paths["y"]), mmap_mode="r")
    source = np.load(str(paths["source"]), mmap_mode="r")
    cell_idx = np.load(str(paths["cell_idx"]), mmap_mode="r")
    drug_idx = np.load(str(paths["drug_idx"]), mmap_mode="r")

    if not np.isfinite(y).all():
        raise RuntimeError("Non-finite labels in {}".format(split_name))
    # Feature banks were already checked for finite values. Verify edge rows here.
    probe_rows = sorted(set([0, max(0, n_rows // 2), max(0, n_rows - 1)]))
    if n_rows and not np.isfinite(np.asarray(x[probe_rows])).all():
        raise RuntimeError("Non-finite feature probe in {}".format(split_name))

    print(
        "[FEATURE CACHE][{}] DONE shape={} elapsed={:.1f}s".format(
            split_name, x.shape, time.time() - started
        ),
        flush=True,
    )
    return x, y, source, cell_idx, drug_idx, paths


# -----------------------------------------------------------------------------
# Metrics and outputs
# -----------------------------------------------------------------------------


def safe_corr(func, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float("nan")
    result = func(y_true, y_pred)
    if hasattr(result, "statistic"):
        return float(result.statistic)
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)


def per_drug_metrics(frame, y_true, y_pred):
    work = pd.DataFrame(
        {
            "canonical_smiles": frame["_canonical_smiles"].astype(str).to_numpy(),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
        }
    )
    rows = []
    for smiles, group in work.groupby("canonical_smiles", sort=True):
        yy = group["y_true"].to_numpy(dtype=np.float64)
        pp = group["y_pred"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "canonical_smiles": smiles,
                "n": int(len(group)),
                "rmse": float(np.sqrt(np.mean((yy - pp) ** 2))),
                "mae": float(np.mean(np.abs(yy - pp))),
                "pearson": safe_corr(pearsonr, yy, pp),
                "spearman": safe_corr(spearmanr, yy, pp),
            }
        )
    return pd.DataFrame(rows)


def save_predictions(path, split_name, frame, y_true, y_pred, drug_col):
    out = pd.DataFrame(
        {
            "source_row_index": frame["_source_row_index"].to_numpy(dtype=np.int64),
            "split": split_name,
            "CELL_LINE_NAME": frame["_cell"].astype(str).to_numpy(),
            "canonical_smiles": frame["_canonical_smiles"].astype(str).to_numpy(),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
        }
    )
    out["residual"] = out["y_pred"] - out["y_true"]
    out["absolute_error"] = np.abs(out["residual"])
    if drug_col is not None and drug_col in frame.columns:
        out[str(drug_col)] = frame[drug_col].to_numpy()
    out.to_csv(str(path), index=False)


def summarize_macro(per_drug):
    return {
        "n_drugs": int(len(per_drug)),
        "mean_pearson": float(per_drug["pearson"].dropna().mean()),
        "median_pearson": float(per_drug["pearson"].dropna().median()),
        "mean_spearman": float(per_drug["spearman"].dropna().mean()),
        "median_spearman": float(per_drug["spearman"].dropna().median()),
        "mean_rmse": float(per_drug["rmse"].mean()),
    }


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


def run_ridge(
    args, shared, out, arrays, split_frames, drug_col
):
    print("\n" + "=" * 80)
    print("RIDGE exact cell-blind baseline")
    print("=" * 80, flush=True)
    started = time.time()
    model = Ridge(
        alpha=args.ridge_alpha,
        fit_intercept=True,
        solver="lsqr",
        tol=args.ridge_tol,
        max_iter=args.ridge_max_iter,
        copy_X=True,
    )
    model.fit(arrays["train"][0], arrays["train"][1])

    metrics = {}
    macro = {}
    for split_name in ("validation", "test"):
        x, y = arrays[split_name][0], arrays[split_name][1]
        pred = model.predict(x).astype(np.float64)
        metrics[split_name] = shared.regression_metrics(y, pred)
        pdm = per_drug_metrics(split_frames[split_name], y, pred)
        macro[split_name] = summarize_macro(pdm)
        save_predictions(
            out / "ridge_{}.predictions.csv".format(split_name),
            split_name,
            split_frames[split_name],
            y,
            pred,
            drug_col,
        )
        pdm.to_csv(
            str(out / "ridge_{}.per_drug.csv".format(split_name)), index=False
        )

    report = {
        "model": "Ridge",
        "parameters": {
            "alpha": args.ridge_alpha,
            "solver": "lsqr",
            "tol": args.ridge_tol,
            "max_iter": args.ridge_max_iter,
            "fit_intercept": True,
            "copy_X": True,
        },
        "n_iter": np.asarray(getattr(model, "n_iter_", [])).tolist(),
        "elapsed_seconds": float(time.time() - started),
        "metrics": metrics,
        "per_drug_macro": macro,
    }
    joblib.dump(model, str(out / "ridge_alpha1.joblib"), compress=3)
    json_dump(report, out / "ridge_report.json")
    print("[RIDGE] {}".format(json.dumps(report["metrics"], sort_keys=True)), flush=True)
    return report


def xgb_fit_with_compat(model, xtr, ytr, xva, yva, early_stopping):
    try:
        model.fit(
            xtr,
            ytr,
            eval_set=[(xva, yva)],
            verbose=50,
            early_stopping_rounds=early_stopping,
        )
        return model
    except TypeError as exc:
        message = str(exc)
        if "early_stopping_rounds" not in message:
            raise
        params = model.get_params()
        params["early_stopping_rounds"] = early_stopping
        model = model.__class__(**params)
        model.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=50)
        return model


def fit_xgboost(args, xtr, ytr, xva, yva):
    try:
        import xgboost as xgb
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("XGBoost is required for --models xgb: {}".format(exc))

    try:
        major = int(str(xgb.__version__).split(".")[0])
    except Exception:
        major = 1

    common = {
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_learning_rate,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
        "reg_lambda": args.xgb_reg_lambda,
        "reg_alpha": args.xgb_reg_alpha,
        "min_child_weight": args.xgb_min_child_weight,
        "max_bin": args.xgb_max_bin,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": args.seed,
        "n_jobs": args.threads,
        "verbosity": 1,
    }

    attempts = []
    if args.xgb_device in ("auto", "gpu"):
        if major >= 2:
            attempts.append(("hist_cuda", {"tree_method": "hist", "device": "cuda"}))
        attempts.append(
            (
                "gpu_hist",
                {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": 0},
            )
        )
    if args.xgb_device in ("auto", "cpu"):
        attempts.append(("hist_cpu", {"tree_method": "hist"}))

    last_error = None
    for backend, extra in attempts:
        print(
            "[XGBOOST] version={} backend={}".format(xgb.__version__, backend),
            flush=True,
        )
        model = XGBRegressor(**dict(common, **extra))
        try:
            fitted = xgb_fit_with_compat(
                model, xtr, ytr, xva, yva, args.xgb_early_stopping
            )
            return fitted, backend, str(xgb.__version__)
        except Exception as exc:
            last_error = exc
            print(
                "[XGBOOST][WARN] {} failed: {}: {}".format(
                    backend, type(exc).__name__, exc
                ),
                flush=True,
            )
            if args.xgb_device == "gpu":
                raise
    raise RuntimeError("All XGBoost backends failed: {}".format(last_error))


def run_xgboost(args, shared, out, arrays, split_frames, drug_col):
    print("\n" + "=" * 80)
    print("XGBOOST exact cell-blind baseline")
    print("=" * 80, flush=True)
    started = time.time()
    model, backend, version = fit_xgboost(
        args,
        arrays["train"][0],
        arrays["train"][1],
        arrays["validation"][0],
        arrays["validation"][1],
    )

    metrics = {}
    macro = {}
    for split_name in ("validation", "test"):
        x, y = arrays[split_name][0], arrays[split_name][1]
        pred = model.predict(x).astype(np.float64)
        metrics[split_name] = shared.regression_metrics(y, pred)
        pdm = per_drug_metrics(split_frames[split_name], y, pred)
        macro[split_name] = summarize_macro(pdm)
        save_predictions(
            out / "xgboost_{}.predictions.csv".format(split_name),
            split_name,
            split_frames[split_name],
            y,
            pred,
            drug_col,
        )
        pdm.to_csv(
            str(out / "xgboost_{}.per_drug.csv".format(split_name)), index=False
        )

    report = {
        "model": "XGBoost",
        "xgboost_version": version,
        "backend": backend,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", float("nan"))),
        "elapsed_seconds": float(time.time() - started),
        "parameters": model.get_params(),
        "metrics": metrics,
        "per_drug_macro": macro,
    }
    model.save_model(str(out / "xgboost_exact_cellblind.json"))
    json_dump(report, out / "xgboost_report.json")
    print(
        "[XGBOOST] {}".format(json.dumps(report["metrics"], sort_keys=True)),
        flush=True,
    )
    return report


# -----------------------------------------------------------------------------
# Final comparison
# -----------------------------------------------------------------------------


def append_metric_row(rows, model_name, metrics, extra=None):
    extra = extra or {}
    row = {"model": model_name}
    for split_name in ("validation", "test"):
        sm = metrics.get(split_name, {})
        row["{}_rmse".format(split_name)] = sm.get("rmse")
        row["{}_mae".format(split_name)] = sm.get("mae")
        row["{}_pcc".format(split_name)] = sm.get("pearson")
        row["{}_scc".format(split_name)] = sm.get("spearman")
        row["{}_r2".format(split_name)] = sm.get("r2")
    row.update(extra)
    rows.append(row)


def make_comparison(out, reports, mlp_metrics_path=None):
    rows = []
    if mlp_metrics_path is not None and mlp_metrics_path.is_file():
        with open(str(mlp_metrics_path), "r", encoding="utf-8") as handle:
            mlp_metrics = json.load(handle)
        append_metric_row(rows, "Global MLP", mlp_metrics, {"source": str(mlp_metrics_path)})
    if "ridge" in reports:
        append_metric_row(rows, "Ridge", reports["ridge"]["metrics"])
    if "xgboost" in reports:
        append_metric_row(rows, "XGBoost", reports["xgboost"]["metrics"])
    comparison = pd.DataFrame(rows)
    comparison.to_csv(str(out / "baseline_comparison.csv"), index=False)
    return comparison


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    set_thread_env(args.threads)

    shared_script = ensure_file(
        Path(__file__).resolve().with_name("cellblind_shared_helpers.py"),
        "bundled cell-blind shared helper module",
    )
    csv_path = ensure_file(args.csv, "canonical CSV")
    basal_path = ensure_file(args.basal_csv, "harmonized basal CSV")
    split_dir = Path(args.split_dir).resolve()
    if not split_dir.is_dir():
        raise FileNotFoundError("split directory not found: {}".format(split_dir))
    split_report_path = ensure_file(args.split_report, "split report")
    fit_cells_path = ensure_file(args.fit_cells_csv, "fit-cells CSV")
    output_dir = ensure_dir(args.output_dir)
    provenance_dir = ensure_dir(output_dir / "provenance")
    cache_dir = ensure_dir(output_dir / "feature_cache")
    mlp_metrics_path = (
        ensure_file(args.mlp_metrics, "MLP metrics") if args.mlp_metrics else None
    )

    if (output_dir / "baseline_report.json").exists():
        raise RuntimeError(
            "Completed output already exists: {}".format(output_dir / "baseline_report.json")
        )

    shared = load_shared_module(shared_script)
    shared.set_all_seeds(args.seed)

    print("=" * 92)
    print("Canonical970 exact cell-blind classical global baselines")
    print("models={} seed={} threads={}".format(args.models, args.seed, args.threads))
    print("XGBoost device request={}".format(args.xgb_device))
    print("=" * 92, flush=True)

    manifest_sets, manifest_paths, manifest_audit = shared.load_and_audit_manifests(
        split_dir, fit_cells_path
    )
    all_cells = (
        manifest_sets["train"]
        | manifest_sets["validation"]
        | manifest_sets["test"]
    )
    print("[MANIFEST AUDIT] PASS: 776 / 97 / 97, overlaps=0", flush=True)

    valid_df, filtering, drug_col = filter_canonical_dataframe_minimal(
        shared,
        csv_path,
        args.cell_col,
        args.smiles_col,
        args.label_col,
        args.drug_col,
    )
    filtering_report_audit = audit_filtering_against_report(
        filtering, split_report_path
    )
    valid_df, row_split_audit = shared.assign_exact_split(
        valid_df, manifest_sets, split_report_path
    )

    cell_matrix, cell_names, cell_to_idx, cell_metadata = shared.load_cell_feature_bank(
        basal_path, all_cells, args.expected_cell_features
    )
    drug_matrix, smiles_list, smiles_to_idx, drug_metadata = shared.build_drug_feature_bank(
        valid_df, args.fp_radius, args.fp_size
    )

    split_frames = {
        name: valid_df.loc[valid_df["_split"] == name].copy()
        for name in ("train", "validation", "test")
    }
    arrays = {}
    feature_paths = {}
    for split_name in ("train", "validation", "test"):
        bundle = build_or_load_split_cache(
            split_frames[split_name],
            split_name,
            cache_dir,
            cell_matrix,
            drug_matrix,
            cell_to_idx,
            smiles_to_idx,
            args.feature_chunk_size,
            args.reuse_feature_cache,
        )
        arrays[split_name] = bundle[:5]
        feature_paths[split_name] = {k: str(v) for k, v in bundle[5].items()}

    feature_audit = {
        "status": "PASS",
        "drug_features": drug_metadata,
        "cell_features": {
            k: v for k, v in cell_metadata.items() if k != "feature_columns"
        },
        "total_features": int(drug_matrix.shape[1] + cell_matrix.shape[1]),
        "split_shapes": {
            name: list(arrays[name][0].shape)
            for name in ("train", "validation", "test")
        },
        "split_feature_gb": {
            name: float(arrays[name][0].nbytes / 1.0e9)
            for name in ("train", "validation", "test")
        },
        "feature_cache_paths": feature_paths,
        "label": "LN_IC50 used directly; no additional log transform",
        "same_representation_as_global_mlp": True,
    }

    config = {
        "argv": sys.argv,
        "python": sys.version,
        "seed": args.seed,
        "threads": args.threads,
        "models": args.models,
        "inputs": {
            "shared_mlp_script": str(shared_script),
            "canonical_csv": str(csv_path),
            "basal_csv": str(basal_path),
            "split_dir": str(split_dir),
            "split_report": str(split_report_path),
            "fit_cells_csv": str(fit_cells_path),
            "mlp_metrics": str(mlp_metrics_path) if mlp_metrics_path else None,
        },
        "feature_configuration": {
            "fp_size": args.fp_size,
            "fp_radius": args.fp_radius,
            "expected_cell_features": args.expected_cell_features,
            "feature_chunk_size": args.feature_chunk_size,
        },
        "ridge": {
            "alpha": args.ridge_alpha,
            "solver": "lsqr",
            "tol": args.ridge_tol,
            "max_iter": args.ridge_max_iter,
        },
        "xgboost": {
            "device_request": args.xgb_device,
            "n_estimators": args.xgb_n_estimators,
            "max_depth": args.xgb_max_depth,
            "learning_rate": args.xgb_learning_rate,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample,
            "reg_lambda": args.xgb_reg_lambda,
            "reg_alpha": args.xgb_reg_alpha,
            "min_child_weight": args.xgb_min_child_weight,
            "max_bin": args.xgb_max_bin,
            "early_stopping": args.xgb_early_stopping,
        },
    }

    json_dump(config, output_dir / "config.json")
    json_dump(manifest_audit, output_dir / "manifest_audit.json")
    json_dump(filtering, output_dir / "filtering_audit.json")
    json_dump(filtering_report_audit, output_dir / "filtering_report_audit.json")
    json_dump(row_split_audit, output_dir / "row_split_audit.json")
    json_dump(feature_audit, output_dir / "feature_audit.json")
    json_dump(
        cell_metadata["feature_columns"],
        provenance_dir / "cell_feature_columns.json",
    )

    try:
        shutil.copy2(str(Path(__file__).resolve()), str(provenance_dir / Path(__file__).name))
        shutil.copy2(str(shared_script), str(provenance_dir / "shared_mlp_script_snapshot.py"))
    except Exception as exc:
        print("[WARN] Script snapshot failed: {}".format(exc), flush=True)

    hashes = {}
    if not args.skip_hashes:
        hash_targets = {
            "classical_script": Path(__file__).resolve(),
            "shared_mlp_script": shared_script,
            "canonical_csv": csv_path,
            "basal_csv": basal_path,
            "split_report": split_report_path,
            "fit_cells": fit_cells_path,
            "train_cells": manifest_paths["train"],
            "validation_cells": manifest_paths["validation"],
            "test_cells": manifest_paths["test"],
        }
        if mlp_metrics_path is not None:
            hash_targets["mlp_metrics"] = mlp_metrics_path
        print("[HASH] Computing SHA256 provenance...", flush=True)
        for name, path in hash_targets.items():
            hashes[name] = {"path": str(path), "sha256": sha256_file(path)}
        json_dump(hashes, provenance_dir / "input_sha256.json")

    requested = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    allowed = {"ridge", "xgb", "xgboost"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError("Unknown --models entries: {}".format(unknown))

    reports = {}
    if "ridge" in requested:
        reports["ridge"] = run_ridge(
            args, shared, output_dir, arrays, split_frames, drug_col
        )
    if "xgb" in requested or "xgboost" in requested:
        reports["xgboost"] = run_xgboost(
            args, shared, output_dir, arrays, split_frames, drug_col
        )

    comparison = make_comparison(output_dir, reports, mlp_metrics_path)
    final_report = {
        "status": "PASS_EXACT_CELL_BLIND_CLASSICAL",
        "models": reports,
        "manifest_audit": manifest_audit,
        "filtering_audit": filtering,
        "filtering_report_audit": filtering_report_audit,
        "row_split_audit": row_split_audit,
        "feature_audit": feature_audit,
        "input_hashes": hashes,
    }
    json_dump(final_report, output_dir / "baseline_report.json")

    print("\n" + "=" * 92)
    print("EXACT CELL-BLIND BASELINE SUMMARY")
    print("=" * 92)
    if len(comparison):
        print(comparison.to_string(index=False))
    print("[DONE] PASS_EXACT_CELL_BLIND_CLASSICAL")
    print("[REPORT] {}".format(output_dir / "baseline_report.json"))
    print("[COMPARISON] {}".format(output_dir / "baseline_comparison.csv"))
    print("=" * 92, flush=True)

    # Delete only the giant, reproducible feature arrays after a successful run.
    if not args.keep_feature_cache:
        print("[CLEANUP] Removing disk-backed feature cache...", flush=True)
        for split_name in ("train", "validation", "test"):
            for path in cache_paths(cache_dir, split_name).values():
                try:
                    if path.exists():
                        path.unlink()
                except Exception as exc:
                    print("[WARN] Could not delete {}: {}".format(path, exc), flush=True)
        try:
            cache_dir.rmdir()
        except Exception:
            pass
        print("[CLEANUP] Feature cache removed", flush=True)


if __name__ == "__main__":
    main()
