#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import final_union_da_ic50_finetune as fin  # noqa


def pcc(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    return {
        "n": int(len(y)),
        "rmse_ln_ic50": float(
            np.sqrt(np.mean((p - y) ** 2))
        ),
        "mae_ln_ic50": float(
            np.mean(np.abs(p - y))
        ),
        "pcc": pcc(y, p),
    }


@torch.no_grad()
def infer_split(
    model,
    dataframe,
    colmap,
    drug_cache,
    cell_cache,
    device,
    batch_size,
    fixed_time,
    fixed_dose,
    amp,
    split_name,
):
    original_order = dataframe[["_source_row"]].copy()

    original_order["_order"] = np.arange(
        len(original_order),
        dtype=np.int64,
    )

    # Put identical cells together so final token-preserving
    # cell dedup can be effective during inference.
    sorted_df = dataframe.sort_values(
        [
            colmap["cell"],
            colmap["smiles"],
            "_source_row",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    ds = fin.IC50Dataset(
        sorted_df,
        colmap,
        drug_cache,
        cell_cache,
        fixed_time,
        fixed_dose,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=fin.collate_ic50,
        drop_last=False,
    )

    rows = []
    seen = 0
    t0 = time.time()

    model.eval()

    for step, batch in enumerate(loader, 1):

        with torch.cuda.amp.autocast(
            enabled=(
                amp
                and device.type == "cuda"
            )
        ):
            pred = fin.forward_batch(
                model,
                batch,
                device,
            )

        pred = (
            pred.detach()
            .float()
            .cpu()
            .numpy()
        )

        y = (
            batch["y"]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        for i in range(len(pred)):
            rows.append({
                "source_row": int(
                    batch["meta"]["source_row"][i]
                ),
                "cell_id": str(
                    batch["meta"]["cell_id"][i]
                ),
                "drug_id": str(
                    batch["meta"]["drug_id"][i]
                ),
                "canonical_smiles": str(
                    batch["meta"]["canonical_smiles"][i]
                ),
                "y_true": float(y[i]),
                "y_bio": float(pred[i]),
            })

        seen += len(pred)

        if step % 500 == 0:
            rate = seen / max(
                1e-9,
                time.time() - t0,
            )

            print(
                "[INFER][%s] step=%d n=%d/%d rate=%.1f/s"
                % (
                    split_name,
                    step,
                    seen,
                    len(ds),
                    rate,
                ),
                flush=True,
            )

    result = pd.DataFrame(rows)

    result = (
        result
        .merge(
            original_order,
            left_on="source_row",
            right_on="_source_row",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("_order")
        .drop(
            columns=[
                "_source_row",
                "_order",
            ]
        )
        .reset_index(drop=True)
    )

    expected = dataframe[
        "_source_row"
    ].to_numpy(dtype=np.int64)

    observed = result[
        "source_row"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        expected,
        observed,
    ):
        raise RuntimeError(
            "Prediction-order audit failed: %s"
            % split_name
        )

    label_delta = float(
        np.max(
            np.abs(
                result["y_true"].to_numpy(
                    dtype=np.float64
                )
                -
                dataframe[
                    colmap["label"]
                ].to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    if label_delta > 1e-5:
        raise RuntimeError(
            "Prediction label audit failed: "
            "%s delta=%g"
            % (
                split_name,
                label_delta,
            )
        )

    m = metrics(
        result["y_true"],
        result["y_bio"],
    )

    print(
        "[CORE][%s] n=%d RMSE=%.6f MAE=%.6f PCC=%.6f"
        % (
            split_name,
            m["n"],
            m["rmse_ln_ic50"],
            m["mae_ln_ic50"],
            m["pcc"],
        ),
        flush=True,
    )

    return result, m


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--csv_path",
        required=True,
    )
    ap.add_argument(
        "--landmark_csv",
        required=True,
    )
    ap.add_argument(
        "--drug_graph_cache",
        required=True,
    )
    ap.add_argument(
        "--cell_graph_cache",
        required=True,
    )

    ap.add_argument(
        "--train_manifest",
        required=True,
    )
    ap.add_argument(
        "--validation_manifest",
        required=True,
    )
    ap.add_argument(
        "--test_manifest",
        required=True,
    )

    ap.add_argument(
        "--checkpoint",
        required=True,
    )
    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    ap.add_argument(
        "--fixed_time",
        type=float,
        default=72.0,
    )
    ap.add_argument(
        "--fixed_dose",
        type=float,
        default=1.0,
    )
    ap.add_argument(
        "--amp",
        action="store_true",
    )

    args = ap.parse_args()

    out = Path(args.output_dir)

    if out.exists():
        raise RuntimeError(
            "Output already exists: %s"
            % out
        )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    device = torch.device(
        args.device
    )

    print(
        "[LOAD] canonical GDSC",
        flush=True,
    )

    dataframe, colmap = (
        fin.load_compact_gdsc(
            args.csv_path
        )
    )

    drug_cache, drug_fdim = (
        fin.load_drug_cache(
            args.drug_graph_cache
        )
    )

    cell_cache = (
        fin.load_cell_cache(
            args.cell_graph_cache,
            31,
        )
    )

    landmarks = fin.load_landmarks(
        args.landmark_csv
    )

    if len(landmarks) != 268:
        raise RuntimeError(
            "Expected 268 landmarks, got %d"
            % len(landmarks)
        )

    model_args = SimpleNamespace(
        max_num_nodes=96,
        num_pathways=31,
        edge_encoding="scalar",
        edge_direction=(
            "direction_aware_bidirectional"
        ),
    )

    model, _ = fin.build_model(
        landmarks,
        drug_fdim,
        model_args,
        device,
    )

    pkg = fin.safe_torch_load(
        args.checkpoint
    )

    state = fin.extract_state(pkg)

    incompatible = model.load_state_dict(
        state,
        strict=True,
    )

    print(
        "[STRICT LOAD] missing=%r unexpected=%r"
        % (
            list(
                incompatible.missing_keys
            ),
            list(
                incompatible.unexpected_keys
            ),
        ),
        flush=True,
    )

    model.to(device)
    model.eval()

    print(
        "[CHECKPOINT] %s epoch=%s"
        % (
            args.checkpoint,
            pkg.get("epoch"),
        ),
        flush=True,
    )

    manifests = {
        "train": pd.read_csv(
            args.train_manifest,
            low_memory=False,
        ),
        "validation": pd.read_csv(
            args.validation_manifest,
            low_memory=False,
        ),
        "test": pd.read_csv(
            args.test_manifest,
            low_memory=False,
        ),
    }

    reports = {}

    os.environ[
        "DRT_FAITHFUL_AUDIT"
    ] = "1"

    for split_name, manifest in (
        manifests.items()
    ):

        source_rows = manifest[
            "source_row"
        ].to_numpy(dtype=np.int64)

        split_df = (
            dataframe
            .iloc[source_rows]
            .copy()
        )

        # Exact-manifest audit.
        y_delta = float(
            np.max(
                np.abs(
                    split_df[
                        colmap["label"]
                    ].to_numpy(
                        dtype=np.float64
                    )
                    -
                    manifest[
                        "LN_IC50"
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            )
        )

        cell_ok = bool(
            np.array_equal(
                split_df[
                    colmap["cell"]
                ].astype(str)
                .str.strip()
                .to_numpy(),
                manifest[
                    "cell_id"
                ].astype(str)
                .str.strip()
                .to_numpy(),
            )
        )

        smiles_ok = bool(
            np.array_equal(
                split_df[
                    colmap["smiles"]
                ].astype(str)
                .str.strip()
                .to_numpy(),
                manifest[
                    "canonical_smiles"
                ].astype(str)
                .str.strip()
                .to_numpy(),
            )
        )

        print(
            "[MANIFEST AUDIT][%s] "
            "n=%d label_delta=%.3e "
            "cell=%s smiles=%s"
            % (
                split_name,
                len(split_df),
                y_delta,
                cell_ok,
                smiles_ok,
            ),
            flush=True,
        )

        if (
            y_delta > 1e-6
            or not cell_ok
            or not smiles_ok
        ):
            raise RuntimeError(
                "Manifest audit failed: %s"
                % split_name
            )

        result, rep = infer_split(
            model,
            split_df,
            colmap,
            drug_cache,
            cell_cache,
            device,
            args.batch_size,
            args.fixed_time,
            args.fixed_dose,
            args.amp,
            split_name,
        )

        result.to_csv(
            out
            / (
                "%s_bio_predictions.csv"
                % split_name
            ),
            index=False,
        )

        reports[
            split_name
        ] = rep

    # ==========================================================
    # HARD AUDIT:
    # This must reproduce our frozen final mixed core result.
    # ==========================================================

    observed = reports["test"]

    expected = {
        "rmse_ln_ic50": 1.2394,
        "mae_ln_ic50": 0.9205,
        "pcc": 0.8940,
    }

    tolerance = {
        "rmse_ln_ic50": 0.0020,
        "mae_ln_ic50": 0.0020,
        "pcc": 0.0020,
    }

    audit = {}

    for key in expected:

        delta = abs(
            observed[key]
            - expected[key]
        )

        audit[key] = {
            "observed": observed[key],
            "expected": expected[key],
            "abs_delta": delta,
            "tolerance": tolerance[key],
            "pass": bool(
                delta
                <= tolerance[key]
            ),
        }

    print(
        "[FINAL MIXED CORE AUDIT] %s"
        % json.dumps(
            audit,
            sort_keys=True,
        ),
        flush=True,
    )

    if not all(
        x["pass"]
        for x in audit.values()
    ):
        raise RuntimeError(
            "FINAL MIXED CORE REPRODUCTION FAILED. "
            "Do not train residual hybrid."
        )

    report = {
        "checkpoint": (
            args.checkpoint
        ),
        "checkpoint_epoch": (
            pkg.get("epoch")
        ),
        "splits": reports,
        "frozen_expected_test": expected,
        "reproduction_audit": audit,
        "status": "PASS",
    }

    (
        out
        / "final_core_prediction_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "[PASS] FINAL mixed core "
        "predictions reproduced and saved",
        flush=True,
    )


if __name__ == "__main__":
    main()
