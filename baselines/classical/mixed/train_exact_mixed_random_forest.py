#!/usr/bin/env python3
import argparse
import ast
import gc
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


# ============================================================
# CONFIG
# ============================================================

parser = argparse.ArgumentParser(
    description="Exact mixed-split Random Forest baseline reproduction."
)
parser.add_argument("--csv", required=True)
parser.add_argument("--basal-csv", required=True)
parser.add_argument("--train-manifest", required=True)
parser.add_argument("--validation-manifest", required=True)
parser.add_argument("--test-manifest", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--n-jobs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

CANONICAL_CSV = Path(args.csv)
BASAL_CSV = Path(args.basal_csv)
OUT = Path(args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = int(args.n_jobs)
RF_SEED = int(args.seed)
FP_BITS = 1024
FP_RADIUS = 2

EXPECTED_COUNTS = {
    "train": 332537,
    "validation": 41567,
    "test": 41567,
}

EXPECTED_MANIFEST_SHA256 = {
    "train": "8212927b7a56d4719269aee689f377e611512b303ff220713c638cd0154e9b31",
    "validation": "c42e94721626cb739b17f1e18d941d09ca3608e2566ab0ab7e1c38578ca77",
    "test": "3c38320b122103c4ca72bc3c91dfec41cde11e88f858dd77fd1fe16f660f00c3",
}

# Prespecified validation-tuning grid used for the finalized mixed RF result.
RF_CONFIGS = [
    {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 1,
        "max_features": 0.3,
    },
    {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 1,
        "max_features": 0.3,
    },
    {
        "n_estimators": 200,
        "max_depth": 30,
        "min_samples_leaf": 2,
        "max_features": 0.3,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.3,
    },
    {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": 0.5,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": 0.5,
    },
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def detect_col(df, candidates, label):
    lower = {str(c).lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in lower:
            return lower[x.lower()]
    raise RuntimeError(
        f"Could not identify {label}. "
        f"Candidates={candidates}; columns={list(df.columns)[:100]}"
    )


def metric_dict(y, pred, smiles):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    smiles = np.asarray(smiles, dtype=str)

    mse = float(np.mean((y - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y - pred)))

    if len(y) >= 2 and np.std(y) > 0 and np.std(pred) > 0:
        pooled_pcc = float(np.corrcoef(y, pred)[0, 1])
    else:
        pooled_pcc = float("nan")

    per_drug = []
    for smi in np.unique(smiles):
        mask = smiles == smi
        yy = y[mask]
        pp = pred[mask]

        if len(yy) >= 2 and np.std(yy) > 0 and np.std(pp) > 0:
            pcc = float(np.corrcoef(yy, pp)[0, 1])
        else:
            pcc = float("nan")

        per_drug.append({
            "canonical_smiles": smi,
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean((yy - pp) ** 2))),
            "mae": float(np.mean(np.abs(yy - pp))),
            "pcc": pcc,
        })

    per_drug_df = pd.DataFrame(per_drug)
    defined = per_drug_df["pcc"].dropna()

    macro_pcc = (
        float(defined.mean())
        if len(defined)
        else float("nan")
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "pooled_pcc": pooled_pcc,
        "macro_pcc": macro_pcc,
        "n": int(len(y)),
        "n_structures": int(len(per_drug_df)),
        "n_defined_structure_pcc": int(len(defined)),
    }, per_drug_df


# ============================================================
# 1. EXACT MIXED MANIFESTS AND RF GRID
# ============================================================


# ============================================================
# 2. LOAD DATA
# ============================================================

manifest_paths = {
    "train": Path(args.train_manifest),
    "validation": Path(args.validation_manifest),
    "test": Path(args.test_manifest),
}

for split_name, path in manifest_paths.items():
    if not path.is_file():
        raise FileNotFoundError(
            "{} manifest not found: {}".format(split_name, path)
        )

manifest_hashes = {
    k: sha256_file(v)
    for k, v in manifest_paths.items()
}

if manifest_hashes != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Mixed-split manifest SHA256 mismatch. "
        "Expected {}, got {}".format(
            EXPECTED_MANIFEST_SHA256,
            manifest_hashes,
        )
    )

print("[MANIFEST] exact canonical mixed manifests verified", flush=True)
for split_name in ("train", "validation", "test"):
    print(
        "[MANIFEST] {} sha256={}".format(
            split_name,
            manifest_hashes[split_name],
        ),
        flush=True,
    )

configs = [dict(x) for x in RF_CONFIGS]
print("[RF] using {} prespecified configurations".format(len(configs)), flush=True)

print("[DATA] loading canonical CSV...", flush=True)
df = pd.read_csv(CANONICAL_CSV, low_memory=False)

smiles_col = detect_col(
    df,
    [
        "canonical_smiles",
        "CANONICAL_SMILES",
        "canonical_SMILES",
        "SMILES",
        "smiles",
    ],
    "canonical SMILES",
)

cell_col = detect_col(
    df,
    [
        "CELL_LINE_NAME",
        "cell_line_name",
        "CELL_LINE",
        "cell_line",
    ],
    "cell line",
)

label_col = detect_col(
    df,
    [
        "LN_IC50",
        "ln_ic50",
        "LN_IC50_VALUE",
    ],
    "LN_IC50",
)

print(
    f"[COLUMNS] smiles={smiles_col} cell={cell_col} label={label_col}",
    flush=True,
)

print("[DATA] loading exact manifests...", flush=True)

split_frames = {}

for split_name, p in manifest_paths.items():
    m = pd.read_csv(p)

    if "source_row" not in m.columns:
        raise RuntimeError(
            f"{p} has no source_row column: {list(m.columns)}"
        )

    rows = pd.to_numeric(
        m["source_row"],
        errors="raise",
    ).astype(np.int64).to_numpy()

    if len(rows) != EXPECTED_COUNTS[split_name]:
        raise RuntimeError(
            f"{split_name}: expected {EXPECTED_COUNTS[split_name]}, "
            f"got {len(rows)}"
        )

    if rows.min() < 0 or rows.max() >= len(df):
        raise RuntimeError(
            f"{split_name}: invalid source_row range "
            f"{rows.min()}..{rows.max()} for n={len(df)}"
        )

    s = df.iloc[rows].copy()
    s["_source_row"] = rows

    s[smiles_col] = s[smiles_col].astype(str).str.strip()
    s[cell_col] = s[cell_col].astype(str).str.strip()
    s[label_col] = pd.to_numeric(
        s[label_col],
        errors="coerce",
    )

    if not np.isfinite(s[label_col]).all():
        raise RuntimeError(
            f"{split_name}: non-finite labels found"
        )

    split_frames[split_name] = s.reset_index(drop=True)

    print(
        f"[SPLIT] {split_name}: n={len(s)} "
        f"structures={s[smiles_col].nunique()} "
        f"cells={s[cell_col].nunique()}",
        flush=True,
    )


# ============================================================
# 4. BASAL EXPRESSION
# ============================================================

print("[FEATURE] loading 1954-gene basal expression...", flush=True)

basal = pd.read_csv(BASAL_CSV, low_memory=False)

basal_cell_col = detect_col(
    basal,
    ["CELL_LINE_NAME", "cell_line_name", "CELL_LINE", "cell_line"],
    "basal cell line",
)

basal[basal_cell_col] = (
    basal[basal_cell_col]
    .astype(str)
    .str.strip()
)

gene_cols = [
    c for c in basal.columns
    if c != basal_cell_col
]

if len(gene_cols) != 1954:
    raise RuntimeError(
        f"Expected 1954 basal genes, got {len(gene_cols)}"
    )

basal_matrix = (
    basal[gene_cols]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=np.float32)
)

if not np.isfinite(basal_matrix).all():
    raise RuntimeError("Basal matrix contains non-finite values")

basal_lookup = {
    cell: i
    for i, cell in enumerate(
        basal[basal_cell_col].tolist()
    )
}

print(
    f"[FEATURE] basal shape={basal_matrix.shape}",
    flush=True,
)


# ============================================================
# 5. MORGAN 1024 FP
# ============================================================

all_smiles = sorted(
    set(
        pd.concat(
            [
                split_frames[k][smiles_col]
                for k in ("train", "validation", "test")
            ],
            ignore_index=True,
        ).tolist()
    )
)

print(
    f"[FEATURE] building Morgan radius={FP_RADIUS} "
    f"{FP_BITS}-bit fingerprints for {len(all_smiles)} structures...",
    flush=True,
)

fp_matrix = np.zeros(
    (len(all_smiles), FP_BITS),
    dtype=np.float32,
)

for i, smi in enumerate(all_smiles):
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        raise RuntimeError(f"RDKit failed: {smi}")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=FP_RADIUS,
        nBits=FP_BITS,
    )

    arr = np.zeros((FP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    fp_matrix[i] = arr.astype(np.float32)

smiles_lookup = {
    smi: i
    for i, smi in enumerate(all_smiles)
}


# ============================================================
# 6. MATERIALIZE SPLIT MATRICES
# ============================================================

def build_X(s, split_name):
    smi_idx = np.asarray(
        [smiles_lookup[x] for x in s[smiles_col]],
        dtype=np.int64,
    )

    missing_cells = sorted(
        set(s[cell_col])
        - set(basal_lookup)
    )

    if missing_cells:
        raise RuntimeError(
            f"{split_name}: cells missing from basal: "
            f"{missing_cells[:20]}"
        )

    cell_idx = np.asarray(
        [basal_lookup[x] for x in s[cell_col]],
        dtype=np.int64,
    )

    n = len(s)

    X = np.empty(
        (n, FP_BITS + len(gene_cols)),
        dtype=np.float32,
    )

    X[:, :FP_BITS] = fp_matrix[smi_idx]
    X[:, FP_BITS:] = basal_matrix[cell_idx]

    y = s[label_col].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise RuntimeError(f"{split_name}: non-finite X")

    print(
        f"[FEATURE] {split_name}: X={X.shape} "
        f"RAM≈{X.nbytes / 1024**3:.2f} GiB",
        flush=True,
    )

    return X, y


X_train, y_train = build_X(
    split_frames["train"],
    "train",
)

X_val, y_val = build_X(
    split_frames["validation"],
    "validation",
)

X_test, y_test = build_X(
    split_frames["test"],
    "test",
)


# ============================================================
# 7. VALIDATION TUNING
# ============================================================

print("\n" + "=" * 80, flush=True)
print("[RF] VALIDATION TUNING START", flush=True)
print("=" * 80, flush=True)

validation_rows = []

best_model = None
best_cfg = None
best_val_rmse = float("inf")

for i, cfg in enumerate(configs, 1):
    print("\n" + "-" * 80, flush=True)
    print(
        f"[RF CONFIG {i}/{len(configs)}] {cfg}",
        flush=True,
    )

    t0 = time.time()

    model = RandomForestRegressor(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=cfg["max_depth"],
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        max_features=cfg["max_features"],
        criterion="squared_error",
        random_state=RF_SEED,
        n_jobs=N_JOBS,
        verbose=1,
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    val_metrics, val_per_drug = metric_dict(
        y_val,
        val_pred,
        split_frames["validation"][smiles_col].to_numpy(),
    )

    elapsed = time.time() - t0

    row = {
        "config_index": i,
        **cfg,
        **{
            f"val_{k}": v
            for k, v in val_metrics.items()
        },
        "elapsed_seconds": elapsed,
    }

    validation_rows.append(row)

    print(
        "[VAL] "
        f"RMSE={val_metrics['rmse']:.6f} "
        f"MAE={val_metrics['mae']:.6f} "
        f"PCC={val_metrics['pooled_pcc']:.6f} "
        f"macro={val_metrics['macro_pcc']:.6f} "
        f"time={elapsed/60:.1f} min",
        flush=True,
    )

    pd.DataFrame(validation_rows).to_csv(
        OUT / "mixed_rf_validation_grid.csv",
        index=False,
    )

    if val_metrics["rmse"] < best_val_rmse:
        print(
            f"[BEST UPDATE] {best_val_rmse:.6f} "
            f"-> {val_metrics['rmse']:.6f}",
            flush=True,
        )

        if best_model is not None:
            del best_model
            gc.collect()

        best_model = model
        best_cfg = dict(cfg)
        best_val_rmse = val_metrics["rmse"]

        val_per_drug.to_csv(
            OUT / "selected_validation_per_structure.csv",
            index=False,
        )

        pd.DataFrame({
            "source_row":
                split_frames["validation"]["_source_row"],
            "canonical_smiles":
                split_frames["validation"][smiles_col],
            "y_true": y_val,
            "y_pred": val_pred,
        }).to_csv(
            OUT / "selected_validation_predictions.csv",
            index=False,
        )

    else:
        del model
        gc.collect()

    del val_pred
    gc.collect()


if best_model is None:
    raise RuntimeError("No RF model selected")


# ============================================================
# 8. TEST — SELECTED CONFIG ONLY
# ============================================================

print("\n" + "=" * 80, flush=True)
print("[RF] SELECTED CONFIG", flush=True)
print(best_cfg, flush=True)
print(
    f"[RF] best validation RMSE={best_val_rmse:.6f}",
    flush=True,
)
print("=" * 80, flush=True)

# Important: test touched only after validation selection.
test_pred = best_model.predict(X_test)

test_metrics, test_per_drug = metric_dict(
    y_test,
    test_pred,
    split_frames["test"][smiles_col].to_numpy(),
)

print(
    "[TEST] "
    f"RMSE={test_metrics['rmse']:.6f} "
    f"MAE={test_metrics['mae']:.6f} "
    f"Pooled PCC={test_metrics['pooled_pcc']:.6f} "
    f"Macro PCC={test_metrics['macro_pcc']:.6f}",
    flush=True,
)

test_per_drug.to_csv(
    OUT / "selected_test_per_structure.csv",
    index=False,
)

pd.DataFrame({
    "source_row":
        split_frames["test"]["_source_row"],
    "canonical_smiles":
        split_frames["test"][smiles_col],
    "y_true": y_test,
    "y_pred": test_pred,
}).to_csv(
    OUT / "selected_test_predictions.csv",
    index=False,
)


# ============================================================
# 9. FINAL PROVENANCE
# ============================================================

report = {
    "analysis": "mixed Random Forest validation tuning",
    "selection_metric": "validation RMSE",
    "test_evaluated_after_selection": True,

    "input": {
        "canonical_csv": str(CANONICAL_CSV),
        "canonical_csv_sha256": sha256_file(CANONICAL_CSV),
        "basal_csv": str(BASAL_CSV),
        "basal_csv_sha256": sha256_file(BASAL_CSV),
        "feature_representation": (
            "Morgan radius-2 1024-bit fingerprint + "
            "1954-gene harmonized basal expression"
        ),
    },

    "manifests": {
        k: {
            "path": str(manifest_paths[k]),
            "sha256": manifest_hashes[k],
            "n": EXPECTED_COUNTS[k],
        }
        for k in manifest_paths
    },

    "rf": {
        "random_state": RF_SEED,
        "n_jobs": N_JOBS,
        "criterion": "squared_error",
        "configs": configs,
        "selected_config": best_cfg,
        "best_validation_rmse": best_val_rmse,
    },

    "test": test_metrics,
}

with open(
    OUT / "mixed_rf_final_report.json",
    "w",
) as f:
    json.dump(
        report,
        f,
        indent=2,
        default=str,
    )

print(
    f"\n[DONE] report -> "
    f"{OUT / 'mixed_rf_final_report.json'}",
    flush=True,
)

print("\n=== FINAL TABLE ROW ===", flush=True)
print(
    "Random Forest | "
    f"{test_metrics['rmse']:.4f} | "
    f"{test_metrics['mae']:.4f} | "
    f"{test_metrics['pooled_pcc']:.4f} | "
    f"{test_metrics['macro_pcc']:.4f}",
    flush=True,
)
