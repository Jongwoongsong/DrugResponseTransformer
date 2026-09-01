#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare canonical970 exact-split 949-gene inputs for GAT-Cross and CSG2A."""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


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
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "source_row",
        "cell_id",
        "drug_id",
        "canonical_smiles",
        "LN_IC50",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("%s missing columns: %r" % (path, missing))
    if frame["source_row"].duplicated().any():
        raise ValueError("Duplicate source_row values in %s" % path)
    return frame


def running_mean_std(array, chunk_size=8192):
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
    variance = np.maximum(
        total_sq / float(seen) - np.square(mean),
        1e-12,
    )
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def save_standardized(raw_path, output_path, mean, std, chunk_size=8192):
    raw = np.load(str(raw_path), mmap_mode="r")
    out = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=raw.shape,
    )
    for start in range(0, len(raw), chunk_size):
        stop = min(len(raw), start + chunk_size)
        out[start:stop] = (
            np.asarray(raw[start:stop], dtype=np.float32) - mean
        ) / std
    out.flush()
    del out


def load_legacy_csg2a_features(label_csv, feature_pickle, required_smiles):
    print("[CSG2A CHEM] loading legacy feature cache", flush=True)
    labels = pd.read_csv(
        label_csv,
        usecols=["canonical_smiles"],
        low_memory=False,
    )
    with open(feature_pickle, "rb") as handle:
        features = pickle.load(handle)

    if len(labels) != len(features):
        raise RuntimeError(
            "Legacy label/feature length mismatch: %d vs %d"
            % (len(labels), len(features))
        )

    required = set(str(x) for x in required_smiles)
    cache = {}
    for smiles, feat in zip(
        labels["canonical_smiles"].astype(str).tolist(),
        features,
    ):
        if smiles in required and smiles not in cache:
            if not isinstance(feat, (list, tuple)) or len(feat) != 3:
                raise RuntimeError(
                    "Unexpected CSG2A feature object for %s" % smiles
                )
            node, adj, dist = feat
            node = np.asarray(node, dtype=np.float32)
            adj = np.asarray(adj, dtype=np.float32)
            dist = np.asarray(dist, dtype=np.float32)
            if node.ndim != 2 or node.shape[1] != 28:
                raise RuntimeError(
                    "CSG2A atom feature mismatch for %s: %r"
                    % (smiles, node.shape)
                )
            if adj.shape != (len(node), len(node)):
                raise RuntimeError(
                    "CSG2A adjacency mismatch for %s" % smiles
                )
            if dist.shape != adj.shape:
                raise RuntimeError(
                    "CSG2A distance mismatch for %s" % smiles
                )
            cache[smiles] = (node, adj, dist)
            if len(cache) == len(required):
                break

    missing = sorted(required - set(cache))
    if missing:
        raise RuntimeError(
            "Legacy CSG2A feature cache misses %d current SMILES. "
            "Examples: %r" % (len(missing), missing[:10])
        )
    max_nodes = max(len(value[0]) for value in cache.values())
    print(
        "[CSG2A CHEM] coverage=%d/%d max_nodes=%d"
        % (len(cache), len(required), max_nodes),
        flush=True,
    )
    return cache, max_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical_csv", required=True)
    parser.add_argument("--cell_expression_949", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--validation_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--landmark_ids", required=True)
    parser.add_argument("--symbol_entrez_mapping", required=True)
    parser.add_argument("--string_edges", required=True)
    parser.add_argument("--legacy_csg2a_gex", required=True)
    parser.add_argument("--legacy_csg2a_label", required=True)
    parser.add_argument("--legacy_csg2a_features", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        raise RuntimeError("Output already exists: %s" % out)
    out.mkdir(parents=True)

    landmark = pd.read_csv(
        args.landmark_ids,
        dtype={"entrez_id": str},
    )
    if list(landmark.columns) != ["entrez_id"]:
        raise RuntimeError(
            "landmark_949_ids.csv must contain only entrez_id"
        )
    gene_ids = landmark["entrez_id"].astype(str).tolist()
    if len(gene_ids) != 949 or len(set(gene_ids)) != 949:
        raise RuntimeError(
            "Expected 949 unique ordered Entrez IDs, got %d/%d"
            % (len(gene_ids), len(set(gene_ids)))
        )

    mapping = pd.read_csv(
        args.symbol_entrez_mapping,
        dtype={"gene_symbol": str, "entrez_id": str},
    )
    mapping = mapping[
        mapping["entrez_id"].isin(gene_ids)
    ].copy()
    if (
        len(mapping) != 949
        or mapping["entrez_id"].nunique() != 949
        or mapping["gene_symbol"].nunique() != 949
    ):
        raise RuntimeError(
            "The ordered 949 genes do not map one-to-one to symbols"
        )
    id_to_symbol = (
        mapping.drop_duplicates("entrez_id")
        .set_index("entrez_id")["gene_symbol"]
        .to_dict()
    )
    gene_symbols = [id_to_symbol[x] for x in gene_ids]

    legacy_header = pd.read_csv(
        args.legacy_csg2a_gex,
        nrows=0,
    )
    legacy_symbols = [str(x) for x in legacy_header.columns]

    def normalize_gene_symbol(value):
        return re.sub(
            r"[^A-Z0-9]",
            "",
            str(value).upper(),
        )

    exact_mismatches = [
        (index, observed, expected, gene_ids[index])
        for index, (observed, expected) in enumerate(
            zip(legacy_symbols, gene_symbols)
        )
        if observed != expected
    ]
    substantive_mismatches = [
        item
        for item in exact_mismatches
        if normalize_gene_symbol(item[1])
        != normalize_gene_symbol(item[2])
    ]

    if (
        len(legacy_symbols) != len(gene_symbols)
        or substantive_mismatches
    ):
        raise RuntimeError(
            "Legacy CSG2A order has substantive symbol mismatches. "
            "legacy_n=%d expected_n=%d mismatches=%r"
            % (
                len(legacy_symbols),
                len(gene_symbols),
                substantive_mismatches[:20],
            )
        )

    if exact_mismatches:
        print(
            "[GENE ORDER][ALIAS] accepted %d punctuation/case "
            "differences; examples=%r"
            % (
                len(exact_mismatches),
                exact_mismatches[:10],
            ),
            flush=True,
        )

    print(
        "[GENE ORDER] ordered 949-gene mapping PASS; "
        "first=%s/%s last=%s/%s"
        % (
            legacy_symbols[0], gene_ids[0],
            legacy_symbols[-1], gene_ids[-1],
        ),
        flush=True,
    )

    canonical_header = pd.read_csv(
        args.canonical_csv,
        nrows=0,
    )
    canonical_cols = [str(x) for x in canonical_header.columns]
    missing_meta = [x for x in META if x not in canonical_cols]
    if missing_meta:
        raise RuntimeError(
            "Canonical CSV misses metadata: %r" % missing_meta
        )

    print("[LOAD] canonical970 metadata/labels", flush=True)
    start = time.perf_counter()
    canonical = pd.read_csv(
        args.canonical_csv,
        usecols=META,
        low_memory=False,
    )
    canonical["CELL_LINE_NAME"] = canonical[
        "CELL_LINE_NAME"
    ].astype(str)
    canonical["_source_row"] = np.arange(
        len(canonical), dtype=np.int64
    )
    print(
        "[LOAD] canonical rows=%d elapsed=%.1fs"
        % (len(canonical), time.perf_counter() - start),
        flush=True,
    )

    expected_cell_columns = ["CELL_LINE_NAME"] + gene_ids
    cell_header = pd.read_csv(
        args.cell_expression_949,
        nrows=0,
    )
    cell_columns = [str(x) for x in cell_header.columns]
    missing_cell_columns = [
        x for x in expected_cell_columns
        if x not in cell_columns
    ]
    if missing_cell_columns:
        raise RuntimeError(
            "Cell-level 949 matrix misses columns: %r"
            % missing_cell_columns[:20]
        )

    cell_dtype = {gene: np.float32 for gene in gene_ids}
    cell_expression = pd.read_csv(
        args.cell_expression_949,
        usecols=expected_cell_columns,
        dtype=cell_dtype,
        low_memory=False,
    )
    cell_expression["CELL_LINE_NAME"] = cell_expression[
        "CELL_LINE_NAME"
    ].astype(str)

    if cell_expression["CELL_LINE_NAME"].duplicated().any():
        raise RuntimeError(
            "Duplicate CELL_LINE_NAME rows in cell-level 949 matrix"
        )

    canonical_cells = set(
        canonical["CELL_LINE_NAME"].unique()
    )
    expression_cells = set(
        cell_expression["CELL_LINE_NAME"].unique()
    )
    missing_cells = sorted(canonical_cells - expression_cells)
    extra_cells = sorted(expression_cells - canonical_cells)
    if missing_cells:
        raise RuntimeError(
            "Cell-level 949 matrix misses canonical cells: %r"
            % missing_cells[:20]
        )

    cell_expression = cell_expression.set_index(
        "CELL_LINE_NAME",
        drop=True,
    )
    expression_values = cell_expression[
        gene_ids
    ].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(expression_values).all():
        raise RuntimeError(
            "Cell-level 949 matrix contains non-finite values"
        )

    print(
        "[CELL949 AUDIT] genes=%d canonical_cells=%d "
        "matched_cells=%d extra_cells=%d nonfinite=0"
        % (
            len(gene_ids),
            len(canonical_cells),
            len(canonical_cells),
            len(extra_cells),
        ),
        flush=True,
    )

    manifests = {
        "train": load_manifest(args.train_manifest),
        "validation": load_manifest(args.validation_manifest),
        "test": load_manifest(args.test_manifest),
    }
    selected = {}
    for split_name, manifest in manifests.items():
        rows = manifest["source_row"].to_numpy(dtype=np.int64)
        if rows.min() < 0 or rows.max() >= len(canonical):
            raise IndexError(
                "%s source_row outside canonical range" % split_name
            )
        frame = canonical.iloc[rows].copy()

        label_delta = float(np.max(np.abs(
            frame["LN_IC50"].to_numpy(dtype=np.float64)
            - manifest["LN_IC50"].to_numpy(dtype=np.float64)
        )))
        smiles_equal = bool(np.all(
            frame["canonical_smiles"].astype(str).to_numpy()
            == manifest["canonical_smiles"].astype(str).to_numpy()
        ))
        cell_equal = bool(np.all(
            frame["CELL_LINE_NAME"].astype(str).to_numpy()
            == manifest["cell_id"].astype(str).to_numpy()
        ))
        if label_delta > 1e-6 or not smiles_equal or not cell_equal:
            raise RuntimeError(
                "%s manifest audit failed: delta=%g smiles=%s cell=%s"
                % (
                    split_name,
                    label_delta,
                    smiles_equal,
                    cell_equal,
                )
            )
        selected[split_name] = frame
        print(
            "[MANIFEST AUDIT][%s] n=%d label_delta=%.3e"
            % (split_name, len(frame), label_delta),
            flush=True,
        )

        split_dir = out / split_name
        split_dir.mkdir()
        cell_order = frame[
            "CELL_LINE_NAME"
        ].astype(str).tolist()
        raw = cell_expression.loc[
            cell_order,
            gene_ids,
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        label = frame["LN_IC50"].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        source_row = frame["_source_row"].to_numpy(
            dtype=np.int64,
            copy=True,
        )
        meta = frame[
            [
                "_source_row",
                "CELL_LINE_NAME",
                "DRUG_NAME",
                "canonical_smiles",
                "LN_IC50",
            ]
        ].copy()

        if not np.isfinite(raw).all():
            raise RuntimeError(
                "%s has non-finite expression values" % split_name
            )
        if not np.isfinite(label).all():
            raise RuntimeError(
                "%s has non-finite labels" % split_name
            )

        np.save(str(split_dir / "expression_raw.npy"), raw)
        np.save(str(split_dir / "label.npy"), label)
        np.save(str(split_dir / "source_row.npy"), source_row)
        meta.to_csv(split_dir / "meta.csv", index=False)
        print(
            "[SAVE][%s] raw=%r %.2fGB"
            % (
                split_name,
                raw.shape,
                raw.nbytes / 1e9,
            ),
            flush=True,
        )

    train_raw = np.load(
        str(out / "train/expression_raw.npy"),
        mmap_mode="r",
    )
    mean, std = running_mean_std(train_raw)
    np.save(str(out / "gat_expression_mean.npy"), mean)
    np.save(str(out / "gat_expression_std.npy"), std)

    for split_name in ["train", "validation", "test"]:
        save_standardized(
            out / split_name / "expression_raw.npy",
            out / split_name / "expression_gat_z.npy",
            mean,
            std,
        )
        print(
            "[SAVE][%s] GAT train-standardized expression"
            % split_name,
            flush=True,
        )

    order = pd.DataFrame({
        "gene_index": np.arange(949, dtype=np.int64),
        "gene_symbol": gene_symbols,
        "entrez_id": gene_ids,
    })
    order.to_csv(out / "gene_order_949.csv", index=False)

    string_df = pd.read_csv(
        args.string_edges,
        dtype={"source": str, "target": str},
    )
    symbol_to_index = {
        symbol: index
        for index, symbol in enumerate(gene_symbols)
    }
    ppi = np.eye(949, dtype=np.float32)
    edge_set = set()
    for source, target in string_df[
        ["source", "target"]
    ].itertuples(index=False, name=None):
        if (
            source in symbol_to_index
            and target in symbol_to_index
        ):
            left = symbol_to_index[source]
            right = symbol_to_index[target]
            ppi[left, right] = 1.0
            ppi[right, left] = 1.0
            if left != right:
                edge_set.add(tuple(sorted((left, right))))
    np.save(str(out / "csg2a_ppi_adj_949.npy"), ppi)
    print(
        "[PPI] shape=%r undirected_edges=%d self_loops=%d"
        % (ppi.shape, len(edge_set), int(np.trace(ppi))),
        flush=True,
    )

    required_smiles = sorted(set().union(*[
        set(
            frame["canonical_smiles"]
            .astype(str)
            .tolist()
        )
        for frame in selected.values()
    ]))
    chem_cache, max_nodes = load_legacy_csg2a_features(
        args.legacy_csg2a_label,
        args.legacy_csg2a_features,
        required_smiles,
    )
    with open(out / "csg2a_chemical_features.pkl", "wb") as handle:
        pickle.dump(chem_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    audit = {
        "canonical_csv": args.canonical_csv,
        "canonical_csv_sha256": sha256_file(args.canonical_csv),
        "cell_expression_949": args.cell_expression_949,
        "cell_expression_949_sha256": sha256_file(
            args.cell_expression_949
        ),
        "manifest_sha256": {
            "train": sha256_file(args.train_manifest),
            "validation": sha256_file(args.validation_manifest),
            "test": sha256_file(args.test_manifest),
        },
        "landmark_ids_sha256": sha256_file(args.landmark_ids),
        "mapping_sha256": sha256_file(
            args.symbol_entrez_mapping
        ),
        "string_edges_sha256": sha256_file(args.string_edges),
        "legacy_csg2a_gex_header_match": True,
        "n_genes": 949,
        "n_unique_smiles": len(required_smiles),
        "csg2a_max_nodes_including_dummy": int(max_nodes),
        "ppi_undirected_edges": int(len(edge_set)),
        "split_counts": {
            key: int(len(value))
            for key, value in selected.items()
        },
        "target": "LN_IC50 used directly; no additional log",
        "csg2a_conditions": {
            "dose": 0.1,
            "time": 1.0,
            "meaning": "original CSG2A normalized constants for 10 uM / 72 h",
        },
    }
    (out / "feature_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )
    print(
        "[PASS] exact 949-gene cache prepared -> %s" % out,
        flush=True,
    )


if __name__ == "__main__":
    main()
