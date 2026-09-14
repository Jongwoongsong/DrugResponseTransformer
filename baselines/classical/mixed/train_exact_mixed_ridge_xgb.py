#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

META = [
    "CELL_LINE_NAME", "DRUG_NAME", "MIN_CONC", "MAX_CONC",
    "LN_IC50", "canonical_smiles",
]


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def calc_pcc(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(pearsonr(y, pred)[0])


def calc_metrics(y, pred):
    mse = float(mean_squared_error(y, pred))
    return {
        "n": int(len(y)),
        "pcc": calc_pcc(y, pred),
        "mse_ln_ic50": mse,
        "rmse_ln_ic50": float(math.sqrt(mse)),
        "mae_ln_ic50": float(mean_absolute_error(y, pred)),
    }


def per_drug_metrics(meta, y, pred):
    df = meta[["DRUG_NAME", "canonical_smiles"]].copy()
    df["y_true"] = np.asarray(y, dtype=np.float64)
    df["y_pred"] = np.asarray(pred, dtype=np.float64)
    rows = []
    for smiles, g in df.groupby("canonical_smiles", sort=True):
        yy = g["y_true"].to_numpy()
        pp = g["y_pred"].to_numpy()
        rows.append({
            "canonical_smiles": smiles,
            "drug_name": str(g["DRUG_NAME"].iloc[0]),
            "n": int(len(g)),
            "pcc": calc_pcc(yy, pp),
            "rmse": float(np.sqrt(np.mean((yy - pp) ** 2))),
            "mae": float(np.mean(np.abs(yy - pp))),
        })
    return pd.DataFrame(rows)


def load_manifest(path):
    df = pd.read_csv(path, low_memory=False)
    required = {
        "source_row", "cell_id", "drug_id",
        "canonical_smiles", "LN_IC50",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if df["source_row"].duplicated().any():
        raise ValueError(f"Duplicate source_row in {path}")
    return df


def load_canonical(csv_path):
    header = pd.read_csv(csv_path, nrows=0)
    cols = [str(c) for c in header.columns]
    genes = [c for c in cols if re.fullmatch(r"\d+", c)]
    if len(genes) != 1954:
        raise RuntimeError(f"Expected 1954 numeric Entrez columns, got {len(genes)}")
    missing = [c for c in META if c not in cols]
    if missing:
        raise RuntimeError(f"Missing metadata columns: {missing}")
    dtype = {g: np.float32 for g in genes}
    print(f"[LOAD] genes={len(genes)} total_columns={len(cols)}", flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(
        csv_path,
        usecols=META + genes,
        dtype=dtype,
        low_memory=False,
    )
    df["_source_row"] = np.arange(len(df), dtype=np.int64)
    print(f"[LOAD] rows={len(df)} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return df, genes


def select_exact(canonical, manifest, split_name):
    rows = manifest["source_row"].to_numpy(dtype=np.int64)
    if rows.min() < 0 or rows.max() >= len(canonical):
        raise IndexError(f"{split_name}: source_row out of range")
    selected = canonical.iloc[rows].copy()
    label_delta = float(np.max(np.abs(
        selected["LN_IC50"].to_numpy(np.float64)
        - manifest["LN_IC50"].to_numpy(np.float64)
    )))
    smiles_equal = bool(np.all(
        selected["canonical_smiles"].astype(str).to_numpy()
        == manifest["canonical_smiles"].astype(str).to_numpy()
    ))
    cell_equal = bool(np.all(
        selected["CELL_LINE_NAME"].astype(str).to_numpy()
        == manifest["cell_id"].astype(str).to_numpy()
    ))
    if label_delta > 1e-6 or not smiles_equal or not cell_equal:
        raise RuntimeError(
            f"{split_name} manifest audit failed: "
            f"label_delta={label_delta}, smiles={smiles_equal}, cell={cell_equal}"
        )
    print(
        f"[MANIFEST AUDIT][{split_name}] n={len(selected)} "
        f"label_delta={label_delta:.3e} "
        f"smiles_equal={smiles_equal} cell_equal={cell_equal}",
        flush=True,
    )
    return selected


def smiles_to_fp(smiles, nbits):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit failed: {smiles}")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    onbits = np.asarray(list(bv.GetOnBits()), dtype=np.int64)
    if onbits.size:
        arr[onbits] = 1.0
    return arr


def build_fp_cache(frames, nbits):
    smiles = sorted(set().union(*[
        set(df["canonical_smiles"].astype(str)) for df in frames
    ]))
    print(f"[FP] unique_smiles={len(smiles)} bits={nbits}", flush=True)
    cache = {}
    for i, smi in enumerate(smiles, 1):
        cache[smi] = smiles_to_fp(smi, nbits)
        if i % 100 == 0 or i == len(smiles):
            print(f"[FP] {i}/{len(smiles)}", flush=True)
    return cache


def build_features(frame, genes, fp_cache, nbits, split_name):
    t0 = time.perf_counter()
    n = len(frame)
    x = np.empty((n, nbits + len(genes)), dtype=np.float32)
    x[:, :nbits] = np.stack([
        fp_cache[s] for s in frame["canonical_smiles"].astype(str)
    ])
    x[:, nbits:] = frame[genes].to_numpy(dtype=np.float32, copy=False)
    y = frame["LN_IC50"].to_numpy(dtype=np.float32, copy=True)
    meta = frame[
        ["_source_row", "CELL_LINE_NAME", "DRUG_NAME", "canonical_smiles"]
    ].copy()
    print(
        f"[FEATURES][{split_name}] shape={x.shape} "
        f"mem={x.nbytes/1e9:.2f}GB elapsed={time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError(f"{split_name}: non-finite data")
    return x, y, meta


def save_predictions(out, model_name, split_name, meta, y, pred):
    df = meta.copy()
    df["y_true"] = np.asarray(y, dtype=np.float64)
    df["y_pred"] = np.asarray(pred, dtype=np.float64)
    df["residual"] = df["y_pred"] - df["y_true"]
    path = out / f"{model_name}.{split_name}_predictions.csv"
    df.to_csv(path, index=False)
    return path


def run_ridge(out, xtr, ytr, xva, yva, xte, yte, mva, mte):
    print("\n" + "=" * 70)
    print("RIDGE alpha=1.0")
    print("=" * 70, flush=True)
    t0 = time.perf_counter()
    model = Ridge(alpha=1.0)
    model.fit(xtr, ytr)
    pva = model.predict(xva)
    pte = model.predict(xte)
    val = calc_metrics(yva, pva)
    test = calc_metrics(yte, pte)
    vpd = per_drug_metrics(mva, yva, pva)
    tpd = per_drug_metrics(mte, yte, pte)
    report = {
        "model": "Ridge",
        "alpha": 1.0,
        "elapsed_seconds": time.perf_counter() - t0,
        "validation": val,
        "test": test,
        "per_drug_macro_pcc": {
            "validation": float(vpd["pcc"].dropna().mean()),
            "test": float(tpd["pcc"].dropna().mean()),
        },
    }
    joblib.dump(model, out / "ridge_alpha1.joblib", compress=3)
    save_predictions(out, "ridge_alpha1", "validation", mva, yva, pva)
    save_predictions(out, "ridge_alpha1", "test", mte, yte, pte)
    vpd.to_csv(out / "ridge_alpha1.validation_per_drug.csv", index=False)
    tpd.to_csv(out / "ridge_alpha1.test_per_drug.csv", index=False)
    print("[RIDGE] " + json.dumps(report, sort_keys=True), flush=True)
    return report


def fit_xgb(args, xtr, ytr, xva, yva):
    import xgboost as xgb
    from xgboost import XGBRegressor

    common = dict(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=args.seed,
        n_jobs=args.threads,
        verbosity=1,
    )
    attempts = []
    if args.xgb_device in {"auto", "gpu"}:
        attempts.append(("gpu_hist", {
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "gpu_id": 0,
        }))
    if args.xgb_device in {"auto", "cpu"}:
        attempts.append(("hist", {"tree_method": "hist"}))
    last = None
    for label, extra in attempts:
        print(f"[XGBOOST] version={xgb.__version__} backend={label}", flush=True)
        model = XGBRegressor(**common, **extra)
        try:
            model.fit(
                xtr, ytr,
                eval_set=[(xva, yva)],
                verbose=50,
                early_stopping_rounds=args.xgb_early_stopping,
            )
            return model, label, xgb.__version__
        except Exception as exc:
            last = exc
            print(
                f"[XGBOOST][WARN] {label} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.xgb_device == "gpu":
                raise
    raise RuntimeError(f"All XGBoost backends failed: {last}")


def run_xgb(args, out, xtr, ytr, xva, yva, xte, yte, mva, mte):
    print("\n" + "=" * 70)
    print("XGBOOST exact-split global baseline")
    print("=" * 70, flush=True)
    t0 = time.perf_counter()
    model, backend, version = fit_xgb(args, xtr, ytr, xva, yva)
    pva = model.predict(xva)
    pte = model.predict(xte)
    val = calc_metrics(yva, pva)
    test = calc_metrics(yte, pte)
    vpd = per_drug_metrics(mva, yva, pva)
    tpd = per_drug_metrics(mte, yte, pte)
    report = {
        "model": "XGBoost",
        "xgboost_version": version,
        "backend": backend,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "elapsed_seconds": time.perf_counter() - t0,
        "parameters": model.get_params(),
        "validation": val,
        "test": test,
        "per_drug_macro_pcc": {
            "validation": float(vpd["pcc"].dropna().mean()),
            "test": float(tpd["pcc"].dropna().mean()),
        },
    }
    model.save_model(str(out / "xgboost_exact_split.json"))
    save_predictions(out, "xgboost", "validation", mva, yva, pva)
    save_predictions(out, "xgboost", "test", mte, yte, pte)
    vpd.to_csv(out / "xgboost.validation_per_drug.csv", index=False)
    tpd.to_csv(out / "xgboost.test_per_drug.csv", index=False)
    print("[XGBOOST] " + json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True)
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--validation_manifest", required=True)
    p.add_argument("--test_manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--models", default="ridge,xgb")
    p.add_argument("--fp_bits", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--xgb_device", choices=["auto", "gpu", "cpu"], default="gpu")
    p.add_argument("--xgb_n_estimators", type=int, default=1000)
    p.add_argument("--xgb_max_depth", type=int, default=6)
    p.add_argument("--xgb_learning_rate", type=float, default=0.03)
    p.add_argument("--xgb_subsample", type=float, default=0.8)
    p.add_argument("--xgb_colsample", type=float, default=0.8)
    p.add_argument("--xgb_early_stopping", type=int, default=50)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "arguments.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True)
    )

    manifests = {
        "train": load_manifest(args.train_manifest),
        "validation": load_manifest(args.validation_manifest),
        "test": load_manifest(args.test_manifest),
    }
    for name, df in manifests.items():
        print(f"[MANIFEST][{name}] n={len(df)}", flush=True)

    canonical, genes = load_canonical(args.csv_path)
    selected = {
        name: select_exact(canonical, df, name)
        for name, df in manifests.items()
    }
    fp_cache = build_fp_cache(selected.values(), args.fp_bits)

    xtr, ytr, mtr = build_features(
        selected["train"], genes, fp_cache, args.fp_bits, "train"
    )
    xva, yva, mva = build_features(
        selected["validation"], genes, fp_cache, args.fp_bits, "validation"
    )
    xte, yte, mte = build_features(
        selected["test"], genes, fp_cache, args.fp_bits, "test"
    )

    audit = {
        "canonical_csv_sha256": sha256_file(args.csv_path),
        "manifest_sha256": {
            "train": sha256_file(args.train_manifest),
            "validation": sha256_file(args.validation_manifest),
            "test": sha256_file(args.test_manifest),
        },
        "n_entrez_expression_features": len(genes),
        "morgan_radius": 2,
        "morgan_bits": args.fp_bits,
        "total_features": int(xtr.shape[1]),
        "split_counts": {
            "train": len(ytr),
            "validation": len(yva),
            "test": len(yte),
        },
        "label": "LN_IC50 used directly; no additional log",
    }
    (out / "feature_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )
    print("[FEATURE AUDIT] " + json.dumps(audit, sort_keys=True), flush=True)

    requested = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    reports = {}
    if "ridge" in requested:
        reports["ridge"] = run_ridge(
            out, xtr, ytr, xva, yva, xte, yte, mva, mte
        )
    if "xgb" in requested or "xgboost" in requested:
        reports["xgboost"] = run_xgb(
            args, out, xtr, ytr, xva, yva, xte, yte, mva, mte
        )

    final = {"feature_audit": audit, "models": reports}
    (out / "baseline_report.json").write_text(
        json.dumps(final, indent=2, sort_keys=True)
    )

    print("\n" + "=" * 70)
    print("EXACT-SPLIT BASELINE SUMMARY")
    print("=" * 70)
    for name, report in reports.items():
        print(
            f"{name:12s} "
            f"val_pcc={report['validation']['pcc']:.6f} "
            f"test_pcc={report['test']['pcc']:.6f} "
            f"test_rmse={report['test']['rmse_ln_ic50']:.6f} "
            f"test_mae={report['test']['mae_ln_ic50']:.6f} "
            f"test_macro_drug_pcc="
            f"{report['per_drug_macro_pcc']['test']:.6f}"
        )
    print("[PASS] exact-split global baselines completed")
    print("[REPORT]", out / "baseline_report.json")


if __name__ == "__main__":
    main()

