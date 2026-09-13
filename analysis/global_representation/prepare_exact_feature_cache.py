#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare exact-split feature arrays for global MLP and residual hybrid."""
from __future__ import print_function

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


META = [
    "CELL_LINE_NAME",
    "DRUG_NAME",
    "MIN_CONC",
    "MAX_CONC",
    "LN_IC50",
    "canonical_smiles",
]


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    df = pd.read_csv(path, low_memory=False)
    required = {
        "source_row", "cell_id", "drug_id",
        "canonical_smiles", "LN_IC50",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("%s missing columns: %r" % (path, missing))
    if df["source_row"].duplicated().any():
        raise ValueError("Duplicate source_row in %s" % path)
    return df


def smiles_to_fp(smiles, nbits):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("RDKit failed for SMILES: %s" % smiles)
    bv = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=nbits
    )
    arr = np.zeros((nbits,), dtype=np.uint8)
    onbits = np.asarray(list(bv.GetOnBits()), dtype=np.int64)
    if onbits.size:
        arr[onbits] = 1
    return arr


def chunk_mean_std(array, chunk_size=8192):
    n = int(array.shape[0])
    d = int(array.shape[1])
    total = np.zeros((d,), dtype=np.float64)
    total_sq = np.zeros((d,), dtype=np.float64)
    seen = 0
    for start in range(0, n, chunk_size):
        chunk = np.asarray(
            array[start:start + chunk_size],
            dtype=np.float64,
        )
        total += chunk.sum(axis=0)
        total_sq += np.square(chunk).sum(axis=0)
        seen += len(chunk)
    mean = total / float(seen)
    var = np.maximum(total_sq / float(seen) - np.square(mean), 1e-12)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--validation_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fp_bits", type=int, default=1024)
    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        raise RuntimeError("Output already exists: %s" % out)
    out.mkdir(parents=True)

    manifests = {
        "train": load_manifest(args.train_manifest),
        "validation": load_manifest(args.validation_manifest),
        "test": load_manifest(args.test_manifest),
    }

    header = pd.read_csv(args.csv_path, nrows=0)
    columns = [str(c) for c in header.columns]
    genes = [c for c in columns if re.fullmatch(r"\d+", c)]
    if len(genes) != 1954:
        raise RuntimeError(
            "Expected 1,954 Entrez columns, got %d" % len(genes)
        )
    missing_meta = [c for c in META if c not in columns]
    if missing_meta:
        raise RuntimeError("Missing metadata: %r" % missing_meta)

    print(
        "[LOAD] rows/genes from canonical CSV; genes=%d" % len(genes),
        flush=True,
    )
    start = time.perf_counter()
    dtype = {gene: np.float32 for gene in genes}
    canonical = pd.read_csv(
        args.csv_path,
        usecols=META + genes,
        dtype=dtype,
        low_memory=False,
    )
    canonical["_source_row"] = np.arange(len(canonical), dtype=np.int64)
    print(
        "[LOAD] rows=%d elapsed=%.1fs"
        % (len(canonical), time.perf_counter() - start),
        flush=True,
    )

    selected = {}
    for split_name, manifest in manifests.items():
        rows = manifest["source_row"].to_numpy(dtype=np.int64)
        frame = canonical.iloc[rows].copy()
        label_delta = float(np.max(np.abs(
            frame["LN_IC50"].to_numpy(dtype=np.float64)
            - manifest["LN_IC50"].to_numpy(dtype=np.float64)
        )))
        smiles_ok = bool(np.all(
            frame["canonical_smiles"].astype(str).to_numpy()
            == manifest["canonical_smiles"].astype(str).to_numpy()
        ))
        cell_ok = bool(np.all(
            frame["CELL_LINE_NAME"].astype(str).to_numpy()
            == manifest["cell_id"].astype(str).to_numpy()
        ))
        if label_delta > 1e-6 or not smiles_ok or not cell_ok:
            raise RuntimeError(
                "Manifest audit failed for %s: delta=%g smiles=%s cell=%s"
                % (split_name, label_delta, smiles_ok, cell_ok)
            )
        selected[split_name] = frame
        print(
            "[MANIFEST AUDIT][%s] n=%d label_delta=%.3e"
            % (split_name, len(frame), label_delta),
            flush=True,
        )

    unique_smiles = sorted(set().union(*[
        set(frame["canonical_smiles"].astype(str).tolist())
        for frame in selected.values()
    ]))
    fp_cache = {}
    for index, smiles in enumerate(unique_smiles, 1):
        fp_cache[smiles] = smiles_to_fp(smiles, args.fp_bits)
        if index % 100 == 0 or index == len(unique_smiles):
            print(
                "[FP] %d/%d" % (index, len(unique_smiles)),
                flush=True,
            )

    for split_name, frame in selected.items():
        split_dir = out / split_name
        split_dir.mkdir()
        expr = frame[genes].to_numpy(dtype=np.float32, copy=True)
        fp = np.stack([
            fp_cache[s]
            for s in frame["canonical_smiles"].astype(str)
        ]).astype(np.uint8, copy=False)
        y = frame["LN_IC50"].to_numpy(dtype=np.float32, copy=True)
        source_row = frame["_source_row"].to_numpy(
            dtype=np.int64, copy=True
        )
        meta = frame[
            ["_source_row", "CELL_LINE_NAME", "DRUG_NAME",
             "canonical_smiles", "LN_IC50"]
        ].copy()

        np.save(str(split_dir / "expression.npy"), expr)
        np.save(str(split_dir / "drug_fp.npy"), fp)
        np.save(str(split_dir / "label.npy"), y)
        np.save(str(split_dir / "source_row.npy"), source_row)
        meta.to_csv(split_dir / "meta.csv", index=False)
        print(
            "[SAVE][%s] expr=%s %.2fGB fp=%s %.2fGB"
            % (
                split_name,
                tuple(expr.shape),
                expr.nbytes / 1e9,
                tuple(fp.shape),
                fp.nbytes / 1e9,
            ),
            flush=True,
        )

    train_expr = np.load(
        str(out / "train/expression.npy"),
        mmap_mode="r",
    )
    expr_mean, expr_std = chunk_mean_std(train_expr)
    np.save(str(out / "expression_mean.npy"), expr_mean)
    np.save(str(out / "expression_std.npy"), expr_std)

    train_y = np.load(str(out / "train/label.npy"), mmap_mode="r")
    label_mean = float(np.mean(train_y, dtype=np.float64))
    label_std = float(np.std(train_y, dtype=np.float64) + 1e-8)

    audit = {
        "canonical_csv": args.csv_path,
        "canonical_csv_sha256": sha256_file(args.csv_path),
        "manifest_sha256": {
            "train": sha256_file(args.train_manifest),
            "validation": sha256_file(args.validation_manifest),
            "test": sha256_file(args.test_manifest),
        },
        "split_counts": {
            key: int(len(value))
            for key, value in manifests.items()
        },
        "n_expression_features": len(genes),
        "expression_columns": genes,
        "morgan_radius": 2,
        "morgan_bits": args.fp_bits,
        "label": "LN_IC50 used directly; no additional log",
        "label_mean": label_mean,
        "label_std": label_std,
    }
    (out / "feature_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )
    print(
        "[PASS] exact feature cache prepared -> %s" % out,
        flush=True,
    )


if __name__ == "__main__":
    main()

