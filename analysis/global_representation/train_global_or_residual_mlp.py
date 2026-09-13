#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train exact-split global MLP or frozen faithful residual hybrid."""
from __future__ import print_function

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    for smiles, group in frame.groupby("canonical_smiles", sort=True):
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


class ExactFeatureDataset(Dataset):
    def __init__(
        self,
        cache_dir,
        split_name,
        expression_mean,
        expression_std,
        bio_predictions_path=None,
    ):
        split_dir = Path(cache_dir) / split_name
        self.expression = np.load(
            str(split_dir / "expression.npy"),
            mmap_mode="r",
        )
        self.drug_fp = np.load(
            str(split_dir / "drug_fp.npy"),
            mmap_mode="r",
        )
        self.label = np.load(
            str(split_dir / "label.npy"),
            mmap_mode="r",
        )
        self.source_row = np.load(
            str(split_dir / "source_row.npy"),
            mmap_mode="r",
        )
        self.expression_mean = expression_mean
        self.expression_std = expression_std
        self.bio = None

        if bio_predictions_path:
            pred = pd.read_csv(bio_predictions_path, low_memory=False)
            required = {"source_row", "y_bio"}
            missing = sorted(required - set(pred.columns))
            if missing:
                raise ValueError(
                    "Bio predictions missing columns: %r" % missing
                )
            mapping = pred.set_index("source_row")["y_bio"]
            aligned = mapping.reindex(
                np.asarray(self.source_row, dtype=np.int64)
            )
            if aligned.isna().any():
                raise RuntimeError(
                    "Missing bio predictions for %s" % split_name
                )
            self.bio = aligned.to_numpy(dtype=np.float32)

    def __len__(self):
        return int(len(self.label))

    def __getitem__(self, index):
        expression = (
            np.asarray(self.expression[index], dtype=np.float32)
            - self.expression_mean
        ) / self.expression_std
        fp = np.asarray(
            self.drug_fp[index],
            dtype=np.float32,
        )
        item = {
            "expression": torch.from_numpy(
                np.ascontiguousarray(expression)
            ),
            "drug_fp": torch.from_numpy(
                np.ascontiguousarray(fp)
            ),
            "label": torch.tensor(
                float(self.label[index]),
                dtype=torch.float32,
            ),
            "source_row": torch.tensor(
                int(self.source_row[index]),
                dtype=torch.long,
            ),
        }
        if self.bio is not None:
            item["bio"] = torch.tensor(
                float(self.bio[index]),
                dtype=torch.float32,
            )
        return item


class GlobalBranch(nn.Module):
    def __init__(self, fp_dim=1024, expression_dim=1954, dropout=0.15):
        super(GlobalBranch, self).__init__()
        self.drug_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(expression_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(64, 1)

    def forward(self, drug_fp, expression):
        drug = self.drug_encoder(drug_fp)
        cell = self.cell_encoder(expression)
        fusion_input = torch.cat([
            drug,
            cell,
            drug * cell,
            torch.abs(drug - cell),
        ], dim=-1)
        hidden = self.fusion(fusion_input)
        return self.output(hidden).reshape(-1)


def corr_tensor(pred, target):
    pred = pred.float()
    target = target.float()
    px = pred - pred.mean()
    py = target - target.mean()
    denom = torch.sqrt(
        (px.square().sum() + 1e-8)
        * (py.square().sum() + 1e-8)
    )
    return (px * py).sum() / denom


def load_initial_global_weights(model, checkpoint_path, zero_output):
    package = torch.load(checkpoint_path, map_location="cpu")
    source = package["model_state_dict"]
    target = model.state_dict()
    loaded = {}
    for key, tensor in source.items():
        if key in target and tuple(tensor.shape) == tuple(target[key].shape):
            loaded[key] = tensor
    merged = dict(target)
    merged.update(loaded)
    model.load_state_dict(merged, strict=True)
    if zero_output:
        nn.init.zeros_(model.output.weight)
        nn.init.zeros_(model.output.bias)
    print(
        "[INIT] loaded %d/%d tensors from %s; zero_output=%s"
        % (len(loaded), len(target), checkpoint_path, zero_output),
        flush=True,
    )


@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval()
    y_all = []
    pred_all = []
    bio_all = []
    delta_all = []
    source_all = []

    for batch in loader:
        fp = batch["drug_fp"].to(device, non_blocking=True)
        expr = batch["expression"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        source = batch["source_row"]

        raw = model(fp, expr)
        if task == "residual":
            bio = batch["bio"].to(device, non_blocking=True)
            final = bio + raw
            bio_all.append(bio.cpu().numpy())
            delta_all.append(raw.cpu().numpy())
        else:
            final = raw

        y_all.append(y.cpu().numpy())
        pred_all.append(final.cpu().numpy())
        source_all.append(source.numpy())

    y = np.concatenate(y_all)
    pred = np.concatenate(pred_all)
    source = np.concatenate(source_all)
    result = {
        "metrics": calc_metrics(y, pred),
        "y": y,
        "pred": pred,
        "source_row": source,
    }
    if task == "residual":
        result["bio"] = np.concatenate(bio_all)
        result["delta"] = np.concatenate(delta_all)
        result["bio_metrics"] = calc_metrics(y, result["bio"])
    return result


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    task,
    label_mean,
    label_std,
    epoch,
    corr_start_epoch,
    alpha_corr,
    lambda_delta,
    grad_clip,
    log_interval,
):
    model.train()
    total = 0.0
    seen = 0
    start = time.perf_counter()

    for step, batch in enumerate(loader, 1):
        fp = batch["drug_fp"].to(device, non_blocking=True)
        expr = batch["expression"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=(scaler.is_enabled())
        ):
            raw = model(fp, expr)
            if task == "residual":
                bio = batch["bio"].to(device, non_blocking=True)
                pred = bio + raw
                delta_penalty = torch.mean(
                    torch.square(raw / label_std)
                )
            else:
                pred = raw
                delta_penalty = torch.zeros(
                    (), device=device, dtype=pred.dtype
                )

            loss = F.mse_loss(
                (pred - label_mean) / label_std,
                (y - label_mean) / label_std,
            )
            if epoch >= corr_start_epoch and len(y) > 1:
                loss = loss + alpha_corr * (
                    1.0 - corr_tensor(pred, y)
                )
            if task == "residual":
                loss = loss + lambda_delta * delta_penalty

        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(
                "Non-finite training loss at step %d" % step
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        batch_n = int(len(y))
        total += float(loss.detach()) * batch_n
        seen += batch_n
        if step % log_interval == 0:
            rate = seen / max(
                1e-9, time.perf_counter() - start
            )
            alloc = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            print(
                "  [step %d] loss=%.4f | %.1f samples/s "
                "| GPU=%.1f/%.1fGB"
                % (
                    step,
                    total / max(1, seen),
                    rate,
                    alloc,
                    reserved,
                ),
                flush=True,
            )

    return total / max(1, seen)


def save_eval_outputs(
    output_dir,
    prefix,
    split_name,
    meta,
    result,
    task,
):
    order = pd.DataFrame({
        "source_row": result["source_row"].astype(np.int64),
        "y_true": result["y"].astype(np.float64),
        "y_pred": result["pred"].astype(np.float64),
    })
    if task == "residual":
        order["y_bio"] = result["bio"].astype(np.float64)
        order["delta_global"] = result["delta"].astype(np.float64)

    merged = meta.merge(
        order,
        left_on="_source_row",
        right_on="source_row",
        how="inner",
        validate="one_to_one",
    )
    merged["residual"] = merged["y_pred"] - merged["y_true"]
    pred_path = output_dir / (
        "%s.%s_predictions.csv" % (prefix, split_name)
    )
    merged.to_csv(pred_path, index=False)

    per_drug = per_drug_metrics(
        merged,
        merged["y_true"].to_numpy(),
        merged["y_pred"].to_numpy(),
    )
    per_drug_path = output_dir / (
        "%s.%s_per_drug.csv" % (prefix, split_name)
    )
    per_drug.to_csv(per_drug_path, index=False)
    return {
        "predictions": str(pred_path),
        "per_drug": str(per_drug_path),
        "per_drug_macro_pcc": float(
            per_drug["pcc"].dropna().mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["global", "residual"],
        required=True,
    )
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bio_prediction_dir", default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=5e-4)
    parser.add_argument("--corr_start_epoch", type=int, default=2)
    parser.add_argument("--alpha_corr", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    if out.exists():
        raise RuntimeError("Output already exists: %s" % out)
    out.mkdir(parents=True)

    cache = Path(args.feature_cache)
    expression_mean = np.load(
        str(cache / "expression_mean.npy")
    ).astype(np.float32)
    expression_std = np.load(
        str(cache / "expression_std.npy")
    ).astype(np.float32)
    audit = json.loads(
        (cache / "feature_audit.json").read_text()
    )
    label_mean = float(audit["label_mean"])
    label_std = float(audit["label_std"])

    bio_paths = {}
    if args.task == "residual":
        if not args.bio_prediction_dir:
            raise ValueError(
                "Residual task requires --bio_prediction_dir"
            )
        bio_root = Path(args.bio_prediction_dir)
        for split_name in ["train", "validation", "test"]:
            path = bio_root / (
                "%s_bio_predictions.csv" % split_name
            )
            if not path.is_file():
                raise FileNotFoundError(str(path))
            bio_paths[split_name] = str(path)

    datasets = {}
    loaders = {}
    for split_name in ["train", "validation", "test"]:
        dataset = ExactFeatureDataset(
            cache,
            split_name,
            expression_mean,
            expression_std,
            bio_paths.get(split_name),
        )
        datasets[split_name] = dataset
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        )

    device = torch.device(args.device)
    model = GlobalBranch(
        fp_dim=1024,
        expression_dim=1954,
        dropout=args.dropout,
    ).to(device)
    if args.init_checkpoint:
        load_initial_global_weights(
            model,
            args.init_checkpoint,
            zero_output=(args.task == "residual"),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.05,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.amp and device.type == "cuda"),
        init_scale=1024.0,
        growth_interval=2000,
    )

    checkpoint = out / (
        "%s_best.pth" % args.task
    )
    best_val = -float("inf")
    best_epoch = 0
    patience_reference = -float("inf")
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            args.task,
            label_mean,
            label_std,
            epoch,
            args.corr_start_epoch,
            args.alpha_corr,
            args.lambda_delta,
            args.grad_clip,
            args.log_interval,
        )
        validation = evaluate(
            model,
            loaders["validation"],
            device,
            args.task,
        )
        val_metrics = validation["metrics"]
        print(
            "[Epoch %03d] train_loss=%.4f "
            "| val_pcc=%.4f "
            "| val_rmse=%.4f "
            "| val_mae=%.4f"
            % (
                epoch,
                train_loss,
                val_metrics["pcc"],
                val_metrics["rmse_ln_ic50"],
                val_metrics["mae_ln_ic50"],
            ),
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        current_pcc = float(val_metrics["pcc"])
        if current_pcc > best_val:
            best_val = current_pcc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "validation": val_metrics,
                "feature_audit": audit,
            }, checkpoint)
            print(
                "[SAVE] %s epoch=%d val_pcc=%.6f"
                % (checkpoint, epoch, best_val),
                flush=True,
            )
        if current_pcc > patience_reference + args.min_delta:
            patience_reference = current_pcc
            stale = 0
        else:
            stale += 1

        scheduler.step()
        if stale >= args.patience:
            print(
                "[EARLY STOP] best_epoch=%d best_val_pcc=%.6f"
                % (best_epoch, best_val),
                flush=True,
            )
            break

    package = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.to(device)
    print(
        "[BEST CHECKPOINT] reloaded %s epoch=%d val_pcc=%.6f"
        % (
            checkpoint,
            int(package["epoch"]),
            float(package["validation"]["pcc"]),
        ),
        flush=True,
    )

    meta = {
        split_name: pd.read_csv(
            cache / split_name / "meta.csv",
            low_memory=False,
        )
        for split_name in ["validation", "test"]
    }
    final_results = {}
    paths = {}
    for split_name in ["validation", "test"]:
        result = evaluate(
            model,
            loaders[split_name],
            device,
            args.task,
        )
        final_results[split_name] = {
            "metrics": result["metrics"],
        }
        if args.task == "residual":
            final_results[split_name]["bio_metrics"] = (
                result["bio_metrics"]
            )
            delta = np.asarray(
                result["delta"],
                dtype=np.float64,
            )
            final_results[split_name]["correction"] = {
                "mean": float(np.mean(delta)),
                "mean_abs": float(np.mean(np.abs(delta))),
                "median_abs": float(np.median(np.abs(delta))),
                "p90_abs": float(np.quantile(np.abs(delta), 0.90)),
                "p95_abs": float(np.quantile(np.abs(delta), 0.95)),
            }
        paths[split_name] = save_eval_outputs(
            out,
            args.task,
            split_name,
            meta[split_name],
            result,
            args.task,
        )
        final_results[split_name][
            "per_drug_macro_pcc"
        ] = paths[split_name]["per_drug_macro_pcc"]

    report = {
        "task": args.task,
        "args": vars(args),
        "best_epoch": int(package["epoch"]),
        "best_validation": package["validation"],
        "final": final_results,
        "history": history,
        "paths": paths,
        "checkpoint": str(checkpoint),
    }
    (out / ("%s_report.json" % args.task)).write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(
        "[TEST] pcc=%.6f rmse=%.6f mae=%.6f macro_drug_pcc=%.6f"
        % (
            final_results["test"]["metrics"]["pcc"],
            final_results["test"]["metrics"]["rmse_ln_ic50"],
            final_results["test"]["metrics"]["mae_ln_ic50"],
            final_results["test"]["per_drug_macro_pcc"],
        ),
        flush=True,
    )
    if args.task == "residual":
        print(
            "[BIO ONLY][TEST] pcc=%.6f rmse=%.6f mae=%.6f"
            % (
                final_results["test"]["bio_metrics"]["pcc"],
                final_results["test"]["bio_metrics"][
                    "rmse_ln_ic50"
                ],
                final_results["test"]["bio_metrics"][
                    "mae_ln_ic50"
                ],
            ),
            flush=True,
        )
        print(
            "[CORRECTION][TEST] %s"
            % json.dumps(
                final_results["test"]["correction"],
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        "[PASS] %s model completed -> %s"
        % (args.task, out),
        flush=True,
    )


if __name__ == "__main__":
    main()

