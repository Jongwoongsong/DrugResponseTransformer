#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact-source-row GAT-Cross comparator on canonical970."""
from __future__ import print_function

import argparse
import json
import math
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
from torch_geometric.data import Batch, Data


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


def load_drug_cache(path):
    package = safe_torch_load(path)
    if isinstance(package, dict) and "cache" in package:
        cache = package["cache"]
        fdim = int(package.get(
            "fdim",
            next(iter(cache.values())).x.shape[1],
        ))
    elif isinstance(package, dict):
        cache = package
        fdim = int(next(iter(cache.values())).x.shape[1])
    else:
        raise TypeError(
            "Unexpected drug graph cache: %s" % type(package)
        )
    clean = {}
    for raw_key, graph in cache.items():
        if not isinstance(graph, Data):
            continue
        graph = graph.clone()
        graph.x = graph.x.detach().cpu().float().contiguous()
        graph.edge_index = (
            graph.edge_index.detach().cpu().long().contiguous()
        )
        clean[str(raw_key).strip()] = graph
    if not clean:
        raise RuntimeError("No valid graphs in drug cache")
    if fdim != 57:
        raise RuntimeError(
            "GAT-Cross expects atom fdim 57, got %d" % fdim
        )
    print(
        "[DRUG CACHE] graphs=%d fdim=%d"
        % (len(clean), fdim),
        flush=True,
    )
    return clean


class ExactGATDataset(Dataset):
    def __init__(self, cache_dir, split_name, drug_cache):
        split = Path(cache_dir) / split_name
        self.expression = np.load(
            str(split / "expression_gat_z.npy"),
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
        self.drug_cache = drug_cache

        missing = sorted(set(self.smiles) - set(drug_cache))
        if missing:
            raise RuntimeError(
                "%s misses %d graph-cache SMILES: %r"
                % (split_name, len(missing), missing[:10])
            )

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, index):
        return {
            "drug_graph": self.drug_cache[self.smiles[index]],
            "gene_expr": torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(
                        self.expression[index],
                        dtype=np.float32,
                    )
                )
            ),
            "label": torch.tensor(
                float(self.labels[index]),
                dtype=torch.float32,
            ),
            "source_row": int(self.source_rows[index]),
            "drug_name": self.drug_names[index],
            "cell_name": self.cell_names[index],
            "canonical_smiles": self.smiles[index],
        }


def collate_fn(batch):
    return {
        "drug_graphs": Batch.from_data_list([
            item["drug_graph"] for item in batch
        ]),
        "gene_expr": torch.stack([
            item["gene_expr"] for item in batch
        ]),
        "labels": torch.stack([
            item["label"] for item in batch
        ]),
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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_all = []
    pred_all = []
    source_all = []
    for batch in loader:
        graphs = batch["drug_graphs"].to(device)
        expression = batch["gene_expr"].to(
            device,
            non_blocking=True,
        )
        pred = model(graphs, expression)
        y_all.append(batch["labels"].numpy())
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


def save_outputs(output, split_name, meta, result):
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
        "gat_cross.%s_predictions.csv" % split_name
    )
    merged.to_csv(pred_path, index=False)
    per_drug = per_drug_metrics(
        merged,
        merged["y_true"].to_numpy(),
        merged["y_pred"].to_numpy(),
    )
    per_path = output / (
        "gat_cross.%s_per_drug.csv" % split_name
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
    parser.add_argument("--drug_graph_cache", required=True)
    parser.add_argument(
        "--model_dir",
        default=str(Path(__file__).resolve().parent / "gat_cross_reference"),
        help="Directory containing kci_model_gat.py",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
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

    sys.path.insert(0, args.model_dir)
    from kci_model_gat import GATCrossAttentionModel

    device = torch.device(args.device)
    drug_cache = load_drug_cache(args.drug_graph_cache)
    datasets = {}
    loaders = {}
    for split_name in ["train", "validation", "test"]:
        datasets[split_name] = ExactGATDataset(
            args.feature_cache,
            split_name,
            drug_cache,
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

    model = GATCrossAttentionModel(
        num_genes=949,
        atom_dim=57,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
    ).to(device)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        "[MODEL] GAT-Cross trainable=%d" % trainable,
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    criterion = nn.MSELoss()

    # One-batch finite-gradient audit.
    audit_batch = next(iter(loaders["train"]))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    audit_pred = model(
        audit_batch["drug_graphs"].to(device),
        audit_batch["gene_expr"].to(device),
    )
    audit_y = audit_batch["labels"].to(device)
    audit_loss = criterion(audit_pred, audit_y)
    audit_loss.backward()
    finite_grad = True
    positive_grad = False
    for parameter in model.parameters():
        if parameter.grad is not None:
            finite_grad = finite_grad and bool(
                torch.isfinite(parameter.grad).all().item()
            )
            positive_grad = positive_grad or bool(
                torch.any(parameter.grad != 0).item()
            )
    if not finite_grad or not positive_grad:
        raise RuntimeError(
            "GAT-Cross gradient audit failed"
        )
    optimizer.zero_grad(set_to_none=True)
    print(
        "[AUDIT] one-batch loss=%.6f gradients finite/nonzero PASS"
        % float(audit_loss.detach()),
        flush=True,
    )

    best_val = -float("inf")
    best_epoch = 0
    stale = 0
    checkpoint = output / "gat_cross_best.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        y_train = []
        pred_train = []
        start = time.perf_counter()

        for step, batch in enumerate(loaders["train"], 1):
            graphs = batch["drug_graphs"].to(device)
            expression = batch["gene_expr"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(graphs, expression)
            loss = criterion(pred, y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    "Non-finite GAT loss at epoch=%d step=%d"
                    % (epoch, step)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            batch_n = int(len(y))
            total_loss += float(loss.detach()) * batch_n
            seen += batch_n
            y_train.append(y.detach().cpu().numpy())
            pred_train.append(
                pred.detach().cpu().numpy()
            )
            if step % args.log_interval == 0:
                rate = seen / max(
                    1e-9,
                    time.perf_counter() - start,
                )
                print(
                    "  [GAT epoch=%d step=%d] loss=%.4f %.1f samples/s"
                    % (
                        epoch,
                        step,
                        total_loss / max(1, seen),
                        rate,
                    ),
                    flush=True,
                )

        scheduler.step()
        train_y = np.concatenate(y_train)
        train_pred = np.concatenate(pred_train)
        train_metrics = calc_metrics(
            train_y,
            train_pred,
        )
        validation = evaluate(
            model,
            loaders["validation"],
            device,
        )
        val_metrics = validation["metrics"]
        print(
            "[GAT Epoch %03d] train_loss=%.6f train_pcc=%.6f "
            "val_pcc=%.6f val_rmse=%.6f val_mae=%.6f"
            % (
                epoch,
                total_loss / max(1, seen),
                train_metrics["pcc"],
                val_metrics["pcc"],
                val_metrics["rmse_ln_ic50"],
                val_metrics["mae_ln_ic50"],
            ),
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            "train_metrics": train_metrics,
            "validation": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if val_metrics["pcc"] > best_val:
            best_val = float(val_metrics["pcc"])
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation": val_metrics,
                "args": vars(args),
            }, checkpoint)
            print(
                "[SAVE][GAT] epoch=%d val_pcc=%.6f"
                % (epoch, best_val),
                flush=True,
            )
        else:
            stale += 1

        if stale >= args.patience:
            print(
                "[EARLY STOP][GAT] best_epoch=%d best_val_pcc=%.6f"
                % (best_epoch, best_val),
                flush=True,
            )
            break

    package = safe_torch_load(checkpoint)
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
        meta = pd.read_csv(
            Path(args.feature_cache)
            / split_name
            / "meta.csv",
            low_memory=False,
        )
        paths[split_name] = save_outputs(
            output,
            split_name,
            meta,
            result,
        )
        final[split_name] = {
            "metrics": result["metrics"],
            "per_drug_macro_pcc": paths[
                split_name
            ]["per_drug_macro_pcc"],
        }

    report = {
        "model": "GAT-Cross",
        "important_note": (
            "This code-backed comparator corresponds to the uploaded "
            "gat_cross_fixed reference (old test PCC 0.902453), not yet "
            "to the manuscript Table-1 GAT value 0.9193."
        ),
        "best_epoch": int(package["epoch"]),
        "best_validation": package["validation"],
        "final": final,
        "history": history,
        "paths": paths,
        "checkpoint": str(checkpoint),
    }
    (output / "gat_cross_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    test = final["test"]
    print(
        "[TEST][GAT] pcc=%.6f rmse=%.6f mae=%.6f macro_drug_pcc=%.6f"
        % (
            test["metrics"]["pcc"],
            test["metrics"]["rmse_ln_ic50"],
            test["metrics"]["mae_ln_ic50"],
            test["per_drug_macro_pcc"],
        ),
        flush=True,
    )
    print(
        "[PASS] exact-split GAT-Cross completed -> %s" % output,
        flush=True,
    )


if __name__ == "__main__":
    main()
