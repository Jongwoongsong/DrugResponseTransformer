#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact-source-row CSG2A finetuning using the original frozen backbone."""
from __future__ import print_function

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_pcc(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(pearsonr(y, pred)[0])


def calc_metrics(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mse = float(np.mean(np.square(y - pred)))
    return {
        "n": int(len(y)),
        "pcc": calc_pcc(y, pred),
        "mse_ln_ic50": mse,
        "rmse_ln_ic50": float(math.sqrt(mse)),
        "mae_ln_ic50": float(np.mean(np.abs(y - pred))),
    }


def per_drug_metrics(meta, y, pred):
    frame = meta[["DRUG_NAME", "canonical_smiles"]].copy()
    frame["y_true"] = np.asarray(y, dtype=np.float64)
    frame["y_pred"] = np.asarray(pred, dtype=np.float64)
    rows = []
    for smiles, group in frame.groupby(
        "canonical_smiles",
        sort=True,
    ):
        yy = group["y_true"].to_numpy()
        pp = group["y_pred"].to_numpy()
        rows.append({
            "canonical_smiles": smiles,
            "drug_name": str(group["DRUG_NAME"].iloc[0]),
            "n": int(len(group)),
            "pcc": calc_pcc(yy, pp),
            "rmse": float(np.sqrt(np.mean(np.square(yy - pp)))),
            "mae": float(np.mean(np.abs(yy - pp))),
        })
    return pd.DataFrame(rows)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ExactCSG2ADataset(Dataset):
    def __init__(self, cache_dir, split_name, chemical_cache):
        split = Path(cache_dir) / split_name
        self.expression = np.load(
            str(split / "expression_raw.npy"),
            mmap_mode="r",
        )
        self.labels = np.load(
            str(split / "label.npy"),
            mmap_mode="r",
        )
        self.source_rows = np.load(
            str(split / "source_row.npy"),
            mmap_mode="r",
        )
        self.meta = pd.read_csv(
            split / "meta.csv",
            low_memory=False,
        )
        self.smiles = (
            self.meta["canonical_smiles"]
            .astype(str)
            .tolist()
        )
        self.drug_names = (
            self.meta["DRUG_NAME"]
            .astype(str)
            .tolist()
        )
        self.cell_names = (
            self.meta["CELL_LINE_NAME"]
            .astype(str)
            .tolist()
        )
        self.chemical_cache = chemical_cache
        missing = sorted(
            set(self.smiles) - set(chemical_cache)
        )
        if missing:
            raise RuntimeError(
                "%s misses chemical features: %r"
                % (split_name, missing[:10])
            )

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, index):
        node, adjacency, distance = self.chemical_cache[
            self.smiles[index]
        ]
        return {
            "node": node,
            "adjacency": adjacency,
            "distance": distance,
            "expression": np.asarray(
                self.expression[index],
                dtype=np.float32,
            ),
            "label": float(self.labels[index]),
            "source_row": int(self.source_rows[index]),
            "drug_name": self.drug_names[index],
            "cell_name": self.cell_names[index],
            "canonical_smiles": self.smiles[index],
        }


def pad_array(array, shape):
    padded = np.zeros(shape, dtype=np.float32)
    padded[:array.shape[0], :array.shape[1]] = array
    return padded


def collate_fn(batch):
    max_size = max(len(item["node"]) for item in batch)
    adjacency = []
    node = []
    distance = []
    mask = []
    expression = []
    label = []
    for item in batch:
        n = len(item["node"])
        adjacency.append(
            pad_array(
                item["adjacency"],
                (max_size, max_size),
            )
        )
        distance.append(
            pad_array(
                item["distance"],
                (max_size, max_size),
            )
        )
        node.append(
            pad_array(
                item["node"],
                (max_size, item["node"].shape[1]),
            )
        )
        row_mask = np.zeros(
            (max_size,),
            dtype=np.float32,
        )
        row_mask[:n] = 1.0
        mask.append(row_mask)
        expression.append(item["expression"])
        label.append(item["label"])

    batch_size = len(batch)
    return {
        "adjacency": torch.from_numpy(
            np.asarray(adjacency, dtype=np.float32)
        ),
        "node": torch.from_numpy(
            np.asarray(node, dtype=np.float32)
        ),
        "mask": torch.from_numpy(
            np.asarray(mask, dtype=np.float32)
        ),
        "distance": torch.from_numpy(
            np.asarray(distance, dtype=np.float32)
        ),
        "expression": torch.from_numpy(
            np.asarray(expression, dtype=np.float32)
        ),
        "label": torch.from_numpy(
            np.asarray(label, dtype=np.float32)
        ),
        "dose": torch.full(
            (batch_size, 1),
            0.1,
            dtype=torch.float32,
        ),
        "time": torch.full(
            (batch_size, 1),
            1.0,
            dtype=torch.float32,
        ),
        "source_rows": np.asarray([
            item["source_row"] for item in batch
        ], dtype=np.int64),
        "drug_names": [
            item["drug_name"] for item in batch
        ],
        "cell_names": [
            item["cell_name"] for item in batch
        ],
        "canonical_smiles": [
            item["canonical_smiles"] for item in batch
        ],
    }


def forward_batch(model, batch, device):
    expression = batch["expression"].to(device)
    node = batch["node"].to(device)
    mask = batch["mask"].to(device).bool()
    adjacency = batch["adjacency"].to(device)
    distance = batch["distance"].to(device)
    dose = batch["dose"].to(device)
    time_value = batch["time"].to(device)
    pred, _ = model(
        expression,
        node,
        mask,
        adjacency,
        distance,
        dose,
        time_value,
    )
    return pred.reshape(-1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_all = []
    pred_all = []
    source_all = []
    for batch in loader:
        pred = forward_batch(model, batch, device)
        y_all.append(batch["label"].numpy())
        pred_all.append(
            pred.detach().float().cpu().numpy()
        )
        source_all.append(batch["source_rows"])
    y = np.concatenate(y_all)
    pred = np.concatenate(pred_all)
    source = np.concatenate(source_all)
    return {
        "metrics": calc_metrics(y, pred),
        "y": y,
        "pred": pred,
        "source_row": source,
    }


def save_outputs(output, checkpoint_name, split_name, meta, result):
    order = pd.DataFrame({
        "_source_row": result["source_row"].astype(np.int64),
        "y_true": result["y"].astype(np.float64),
        "y_pred": result["pred"].astype(np.float64),
    })
    merged = meta.merge(
        order,
        on="_source_row",
        how="inner",
        validate="one_to_one",
    )
    merged["residual"] = (
        merged["y_pred"] - merged["y_true"]
    )
    pred_path = output / (
        "csg2a_%s.%s_predictions.csv"
        % (checkpoint_name, split_name)
    )
    merged.to_csv(pred_path, index=False)
    per_drug = per_drug_metrics(
        merged,
        merged["y_true"].to_numpy(),
        merged["y_pred"].to_numpy(),
    )
    per_path = output / (
        "csg2a_%s.%s_per_drug.csv"
        % (checkpoint_name, split_name)
    )
    per_drug.to_csv(per_path, index=False)
    return {
        "predictions": str(pred_path),
        "per_drug": str(per_path),
        "per_drug_macro_pcc": float(
            per_drug["pcc"].dropna().mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--csg2a_root", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr_init", type=float, default=1e-4)
    parser.add_argument("--lr_final", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gene_hdim", type=int, default=64)
    parser.add_argument("--finetune_hdim1", type=int, default=512)
    parser.add_argument("--finetune_hdim2", type=int, default=64)
    parser.add_argument("--log_interval", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError("Output already exists: %s" % output)
    output.mkdir(parents=True)
    (output / "arguments.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True)
    )

    sys.path.insert(0, args.csg2a_root)
    from models.CSG2A_net import CSG2A_finetune

    cache = Path(args.feature_cache)
    with open(
        cache / "csg2a_chemical_features.pkl",
        "rb",
    ) as handle:
        chemical_cache = pickle.load(handle)
    ppi = torch.from_numpy(
        np.load(
            str(cache / "csg2a_ppi_adj_949.npy")
        ).astype(np.float32)
    )

    datasets = {}
    loaders = {}
    for split_name in ["train", "validation", "test"]:
        datasets[split_name] = ExactCSG2ADataset(
            cache,
            split_name,
            chemical_cache,
        )
        loaders[split_name] = DataLoader(
            datasets[split_name],
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

    device = torch.device(args.device)
    csg2a_params = {
        "gex_dim": 949,
        "hdim": args.gene_hdim,
        "dropout": args.dropout,
        "ppi_adj": ppi.to(device),
        "bias": False,
    }
    model = CSG2A_finetune(
        gex_dim=949,
        CSG2A_params=csg2a_params,
        finetune_hdim1=args.finetune_hdim1,
        finetune_hdim2=args.finetune_hdim2,
        dropout=args.dropout,
    ).to(device)

    pretrained = safe_torch_load(
        args.pretrained_checkpoint
    )
    if not isinstance(pretrained, dict):
        raise TypeError(
            "Unexpected CSG2A checkpoint: %s"
            % type(pretrained)
        )
    model.CSG2A.load_state_dict(
        pretrained,
        strict=True,
    )
    for parameter in model.CSG2A.parameters():
        parameter.requires_grad = False

    frozen = sum(
        parameter.numel()
        for parameter in model.CSG2A.parameters()
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable <= 0:
        raise RuntimeError("No trainable CSG2A finetune parameters")
    print(
        "[MODEL] CSG2A frozen_backbone=%d trainable_head=%d"
        % (frozen, trainable),
        flush=True,
    )

    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.lr_init,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.patience,
        eta_min=args.lr_final,
    )
    criterion = nn.MSELoss()

    # One-batch finite-gradient audit.
    audit_batch = next(iter(loaders["train"]))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    audit_pred = forward_batch(
        model,
        audit_batch,
        device,
    )
    audit_y = audit_batch["label"].to(device)
    audit_loss = criterion(audit_pred, audit_y)
    audit_loss.backward()
    finite_grad = True
    positive_grad = False
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            finite_grad = finite_grad and bool(
                torch.isfinite(parameter.grad).all().item()
            )
            positive_grad = positive_grad or bool(
                torch.any(parameter.grad != 0).item()
            )
    if not finite_grad or not positive_grad:
        raise RuntimeError("CSG2A gradient audit failed")
    optimizer.zero_grad(set_to_none=True)
    print(
        "[AUDIT] one-batch loss=%.6f gradients finite/nonzero PASS"
        % float(audit_loss.detach()),
        flush=True,
    )

    best_mse = float("inf")
    best_mse_epoch = 0
    best_pcc = -float("inf")
    best_pcc_epoch = 0
    stale = 0
    checkpoint_mse = output / "csg2a_best_val_mse.pth"
    checkpoint_pcc = output / "csg2a_best_val_pcc.pth"
    history = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        start = time.perf_counter()
        for step, batch in enumerate(loaders["train"], 1):
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = forward_batch(model, batch, device)
            loss = criterion(pred, y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    "Non-finite CSG2A loss epoch=%d step=%d"
                    % (epoch, step)
                )
            loss.backward()
            optimizer.step()
            batch_n = int(len(y))
            total_loss += float(loss.detach()) * batch_n
            seen += batch_n
            if step % args.log_interval == 0:
                rate = seen / max(
                    1e-9,
                    time.perf_counter() - start,
                )
                print(
                    "  [CSG2A epoch=%d step=%d] loss=%.4f %.1f samples/s"
                    % (
                        epoch,
                        step,
                        total_loss / max(1, seen),
                        rate,
                    ),
                    flush=True,
                )

        if epoch <= args.patience:
            scheduler.step()

        validation = evaluate(
            model,
            loaders["validation"],
            device,
        )
        metrics = validation["metrics"]
        train_loss = total_loss / max(1, seen)
        print(
            "[CSG2A Epoch %03d] train_mse=%.6f "
            "val_mse=%.6f val_pcc=%.6f "
            "val_rmse=%.6f val_mae=%.6f"
            % (
                epoch,
                train_loss,
                metrics["mse_ln_ic50"],
                metrics["pcc"],
                metrics["rmse_ln_ic50"],
                metrics["mae_ln_ic50"],
            ),
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "validation": metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if metrics["mse_ln_ic50"] < best_mse:
            best_mse = float(metrics["mse_ln_ic50"])
            best_mse_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "selection": "validation_mse",
                "validation": metrics,
                "args": vars(args),
            }, checkpoint_mse)
            print(
                "[SAVE][CSG2A MSE] epoch=%d val_mse=%.6f val_pcc=%.6f"
                % (epoch, best_mse, metrics["pcc"]),
                flush=True,
            )
        else:
            stale += 1

        if metrics["pcc"] > best_pcc:
            best_pcc = float(metrics["pcc"])
            best_pcc_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "selection": "validation_pcc",
                "validation": metrics,
                "args": vars(args),
            }, checkpoint_pcc)
            print(
                "[SAVE][CSG2A PCC] epoch=%d val_pcc=%.6f val_mse=%.6f"
                % (
                    epoch,
                    best_pcc,
                    metrics["mse_ln_ic50"],
                ),
                flush=True,
            )

        if stale >= args.patience:
            print(
                "[EARLY STOP][CSG2A] original val-MSE criterion; "
                "best_epoch=%d best_val_mse=%.6f"
                % (best_mse_epoch, best_mse),
                flush=True,
            )
            break

    meta = {
        split_name: pd.read_csv(
            cache / split_name / "meta.csv",
            low_memory=False,
        )
        for split_name in ["validation", "test"]
    }
    report_checkpoints = {}
    for checkpoint_name, checkpoint_path in [
        ("original_val_mse", checkpoint_mse),
        ("sensitivity_val_pcc", checkpoint_pcc),
    ]:
        package = safe_torch_load(checkpoint_path)
        model.load_state_dict(
            package["model_state_dict"],
            strict=True,
        )
        model.to(device)
        final = {}
        paths = {}
        for split_name in ["validation", "test"]:
            result = evaluate(
                model,
                loaders[split_name],
                device,
            )
            paths[split_name] = save_outputs(
                output,
                checkpoint_name,
                split_name,
                meta[split_name],
                result,
            )
            final[split_name] = {
                "metrics": result["metrics"],
                "per_drug_macro_pcc": paths[
                    split_name
                ]["per_drug_macro_pcc"],
            }
        report_checkpoints[checkpoint_name] = {
            "epoch": int(package["epoch"]),
            "selection": package["selection"],
            "saved_validation": package["validation"],
            "final": final,
            "paths": paths,
            "checkpoint": str(checkpoint_path),
        }
        test = final["test"]
        print(
            "[TEST][CSG2A %s] pcc=%.6f rmse=%.6f "
            "mae=%.6f macro_drug_pcc=%.6f"
            % (
                checkpoint_name,
                test["metrics"]["pcc"],
                test["metrics"]["rmse_ln_ic50"],
                test["metrics"]["mae_ln_ic50"],
                test["per_drug_macro_pcc"],
            ),
            flush=True,
        )

    report = {
        "model": "CSG2A",
        "protocol": {
            "backbone": "250911 LINCS-pretrained CSG2A, frozen",
            "gene_order": "exact legacy 949-symbol order audited against landmark_949_ids",
            "ppi": "STRING edges plus self loops, bidirectional",
            "dose": 0.1,
            "time": 1.0,
            "primary_selection": "validation MSE, matching original notebook",
            "secondary_selection": "validation PCC sensitivity analysis",
        },
        "checkpoints": report_checkpoints,
        "history": history,
    }
    (output / "csg2a_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(
        "[PASS] exact-split CSG2A completed -> %s" % output,
        flush=True,
    )


if __name__ == "__main__":
    main()
