#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful 268-gene-token IC50 fine-tuning on canonical GDSC970."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

ROOT = Path(os.environ.get("DRT_ROOT", Path(__file__).resolve().parent)).resolve()
V2 = ROOT / "revision_work/20260730/faithful_gene_token_model_v2_token_dedup"
sys.path.insert(0, str(V2))
sys.path.insert(1, str(ROOT))

from Model.DrugResponseTransformer import ModelConfig
from Model.DrugResponseTransformerFaithfulDedup import (
    FaithfulTokenDedupDrugResponseTransformer,
)
from pge_pretrain_faithful_token_dedup import load_landmarks, sanitize_pyg_data
from Model.CellLine_graph import NUM_EDGE_TYPES


@dataclass
class Args:
    csv_path: str
    basal_csv: str
    landmark_csv: str
    drug_graph_cache: str
    cell_graph_cache: str
    pretrained_checkpoint: Optional[str]
    save_dir: str
    run_name: str
    split_mode: str = "mixed"
    split_seed: int = 42
    seed: int = 42
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    subset_ratio: float = 1.0
    epochs: int = 12
    batch_size: int = 32
    num_workers: int = 0
    device: str = "cuda:0"
    fixed_dose: float = 1.0
    fixed_time: float = 72.0
    body_lr: float = 2e-4
    head_lr: float = 7e-4
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    alpha_corr: float = 0.2
    corr_start_epoch: int = 3
    amp: bool = False
    log_interval: int = 200
    patience: int = 5
    min_delta: float = 1e-3
    max_num_nodes: int = 96
    num_pathways: int = 31
    mode: str = "pretrained"
    edge_encoding: str = "scalar"
    edge_direction: str = "directed"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_col(columns: Sequence[str], candidates: Sequence[str], required=True):
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find column among {list(candidates)}")
    return None


def load_compact_gdsc(csv_path):
    header = pd.read_csv(csv_path, nrows=0)
    cols = list(header.columns)
    cell = find_col(cols, ["CELL_LINE_NAME", "cell_line_name", "cell", "cell_line"])
    label = find_col(cols, ["LN_IC50", "ln_ic50"])
    smiles = find_col(cols, ["canonical_smiles", "smiles", "SMILES"])
    drug = find_col(cols, ["DRUG_NAME", "drug_name", "drug"], required=False)
    usecols = [cell, label, smiles] + ([] if drug is None else [drug])
    df = pd.read_csv(csv_path, usecols=list(dict.fromkeys(usecols)), low_memory=False)
    df["_source_row"] = np.arange(len(df), dtype=np.int64)
    df[cell] = df[cell].astype(str).str.strip()
    df[smiles] = df[smiles].astype(str).str.strip()
    df[label] = pd.to_numeric(df[label], errors="coerce")
    if drug is None:
        drug = "_drug_id"
        df[drug] = df[smiles]
    else:
        df[drug] = df[drug].astype(str).str.strip()
    before = len(df)
    df = df[np.isfinite(df[label].to_numpy(dtype=float))].copy()
    df = df[(df[smiles] != "") & (df[smiles].str.lower() != "nan")].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Loaded canonical rows={before}; finite/nonblank={len(df)}")
    print(
        f"[INFO] Label stats mean={df[label].mean():.6f} "
        f"std={df[label].std():.6f} min={df[label].min():.6f} "
        f"max={df[label].max():.6f}"
    )
    return df, {"cell": cell, "label": label, "smiles": smiles, "drug": drug}


def load_drug_cache(path):
    pkg = safe_torch_load(path)
    if isinstance(pkg, dict) and "cache" in pkg:
        cache = pkg["cache"]
        fdim = int(pkg.get("fdim", next(iter(cache.values())).x.shape[1]))
    elif isinstance(pkg, dict):
        cache = pkg
        fdim = int(next(iter(cache.values())).x.shape[1])
    else:
        raise TypeError(f"Unexpected drug cache type: {type(pkg)}")
    out = {str(k).strip(): v for k, v in cache.items() if isinstance(v, Data)}
    if not out:
        raise RuntimeError("No valid drug graphs")
    print(f"[INFO] Drug cache graphs={len(out)} fdim={fdim}")
    return out, fdim


def load_cell_cache(path, num_pathways):
    pkg = safe_torch_load(path)
    cache = pkg.get("cache", pkg) if isinstance(pkg, dict) else pkg
    if not isinstance(cache, dict):
        raise TypeError(f"Unexpected cell cache type: {type(cache)}")
    out = {}
    for raw_cell, raw_seq in cache.items():
        if not isinstance(raw_seq, (list, tuple)):
            continue
        seq = []
        for graph in raw_seq[:num_pathways]:
            if not isinstance(graph, Data):
                continue
            clean = sanitize_pyg_data(graph)
            if not hasattr(clean, "node_ids"):
                raise RuntimeError(f"Cell graph lacks node_ids: {raw_cell}")
            clean.node_ids = [str(x) for x in clean.node_ids]
            seq.append(clean)
        if len(seq) == num_pathways:
            out[str(raw_cell).strip()] = seq
    if not out:
        raise RuntimeError("No valid cell graph sequences")
    print(f"[INFO] Cell graph cache cells={len(out)} pathways={num_pathways}")
    return out


def split_dataframe(df, mode, seed, valid_ratio, test_ratio, c):
    rng = np.random.RandomState(seed)
    if mode in {"drug_blind", "cell_blind"}:
        key_col = c["smiles"] if mode == "drug_blind" else c["cell"]
        keys = df[key_col].astype(str).unique().copy()
        rng.shuffle(keys)
        n_test = int(len(keys) * test_ratio)
        n_val = int(len(keys) * valid_ratio)
        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test:n_test + n_val])
        train = df[~df[key_col].isin(test_keys | val_keys)]
        val = df[df[key_col].isin(val_keys)]
        test = df[df[key_col].isin(test_keys)]
        if set(train[key_col]) & set(val[key_col]) or set(train[key_col]) & set(test[key_col]):
            raise RuntimeError(f"{mode} leakage")
    elif mode == "mixed":
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = int(len(idx) * test_ratio)
        n_val = int(len(idx) * valid_ratio)
        test = df.iloc[idx[:n_test]]
        val = df.iloc[idx[n_test:n_test + n_val]]
        train = df.iloc[idx[n_test + n_val:]]
    else:
        raise ValueError(mode)
    return tuple(x.reset_index(drop=True) for x in (train, val, test))


def subset_df(df, ratio, seed):
    if ratio >= 0.999999:
        return df.reset_index(drop=True)
    return df.sample(n=max(1, int(len(df) * ratio)), random_state=seed).reset_index(drop=True)


class IC50Dataset(Dataset):
    def __init__(self, df, c, drug_cache, cell_cache, fixed_time, fixed_dose):
        self.df = df.reset_index(drop=True)
        self.c = c
        self.drug_cache = drug_cache
        self.cell_cache = cell_cache
        self.fixed_time = float(fixed_time)
        self.fixed_dose = float(fixed_dose)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        smi = str(row[self.c["smiles"]]).strip()
        cell = str(row[self.c["cell"]]).strip()
        return {
            "drug_graph": self.drug_cache[smi],
            "cell_seq": self.cell_cache[cell],
            "cell_id": cell,
            "drug_id": str(row[self.c["drug"]]),
            "smiles": smi,
            "y": float(row[self.c["label"]]),
            "time": self.fixed_time,
            "dose": self.fixed_dose,
            "source_row": int(row["_source_row"]),
        }


def collate_ic50(items):
    return {
        "drug_graph": Batch.from_data_list([x["drug_graph"] for x in items]),
        "cell_seq": [x["cell_seq"] for x in items],
        "cell_ids": [x["cell_id"] for x in items],
        "time": torch.tensor([x["time"] for x in items], dtype=torch.float32),
        "dose": torch.tensor([x["dose"] for x in items], dtype=torch.float32),
        "y": torch.tensor([x["y"] for x in items], dtype=torch.float32),
        "meta": {
            "cell_id": [x["cell_id"] for x in items],
            "drug_id": [x["drug_id"] for x in items],
            "canonical_smiles": [x["smiles"] for x in items],
            "source_row": [x["source_row"] for x in items],
        },
    }


class EncArgs:
    pass


def _direct_multihot_process_edge_attr(self, edge_attr, device):
    if edge_attr is None:
        return None
    edge_attr = edge_attr.to(device)
    if edge_attr.dim() != 2 or edge_attr.size(1) != NUM_EDGE_TYPES:
        raise RuntimeError(
            f"Direct multi-hot edge encoding requires [E,{NUM_EDGE_TYPES}], "
            f"got {tuple(edge_attr.shape)}"
        )
    return self.edge_emb(edge_attr)



def _direction_aware_process_edge_attr(
    self,
    edge_attr,
    device,
):
    if edge_attr is None:
        return None

    edge_attr = edge_attr.to(device)

    expected_dim = NUM_EDGE_TYPES + 1

    if (
        edge_attr.dim() != 2
        or edge_attr.size(1) != expected_dim
    ):
        raise RuntimeError(
            "Direction-aware edge_attr must be "
            f"[E,{expected_dim}], got "
            f"{tuple(edge_attr.shape)}"
        )

    relation_attr = edge_attr[
        :,
        :NUM_EDGE_TYPES,
    ]

    direction_raw = edge_attr[
        :,
        NUM_EDGE_TYPES,
    ]

    direction_ids = (
        direction_raw
        .round()
        .long()
    )

    if not bool(
        (
            (direction_ids == 0)
            | (direction_ids == 1)
        ).all()
    ):
        raise RuntimeError(
            "Direction IDs must be 0/1"
        )

    # EXISTING pretrained relation mechanism.
    relation_scalar = self.compute_edge_weight(
        relation_attr
    )

    relation_emb = self.edge_emb(
        relation_scalar
    )

    # NEW direction identity.
    direction_emb = self.direction_embedding(
        direction_ids
    )

    return relation_emb + direction_emb


def configure_direction_aware(
    model,
    edge_direction,
    device,
):
    if (
        edge_direction
        != "direction_aware_bidirectional"
    ):
        return

    cell_encoder = model.CellEncoder

    hidden = int(
        cell_encoder.hidden_channels
    )

    dtype = next(
        cell_encoder.edge_emb.parameters()
    ).dtype

    direction_embedding = nn.Embedding(
        2,
        hidden,
    ).to(
        device=device,
        dtype=dtype,
    )

    # Critical controlled initialization:
    #
    # original and reverse are initially identical.
    # Therefore before learning the direction embedding,
    # this is equivalent to naive bidirectional message flow.
    nn.init.zeros_(
        direction_embedding.weight
    )

    cell_encoder.direction_embedding = (
        direction_embedding
    )

    cell_encoder._process_edge_attr = (
        types.MethodType(
            _direction_aware_process_edge_attr,
            cell_encoder,
        )
    )

    cell_encoder.edge_direction_mode = (
        "direction_aware_bidirectional"
    )

    max_abs = float(
        cell_encoder.direction_embedding
        .weight.detach()
        .abs()
        .max()
        .item()
    )

    print(
        "[DIRECTION-AWARE INIT AUDIT] "
        + json.dumps(
            {
                "relation_encoding":
                    "pretrained_scalar",
                "direction_states": 2,
                "direction_semantics": {
                    "0": "original_KEGG",
                    "1": "computational_reverse",
                },
                "direction_embedding_shape":
                    list(
                        cell_encoder
                        .direction_embedding
                        .weight.shape
                    ),
                "zero_init_max_abs":
                    max_abs,
                "initial_equivalence":
                    "naive_bidirectional_before_direction_learning",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if max_abs != 0.0:
        raise RuntimeError(
            "Direction embedding did not zero-initialize"
        )


def configure_edge_encoding(model, edge_encoding, device):
    edge_encoding = str(edge_encoding).lower()
    if edge_encoding == "scalar":
        model.CellEncoder.edge_encoding = "scalar"
        return
    if edge_encoding != "multihot":
        raise ValueError(f"Unknown edge encoding: {edge_encoding}")

    old_linear = model.CellEncoder.edge_emb[0]
    if not isinstance(old_linear, nn.Linear) or old_linear.in_features != 1:
        raise RuntimeError(
            "Expected scalar edge projection Linear(1,H), got "
            f"{old_linear}"
        )
    direct_linear = nn.Linear(
        NUM_EDGE_TYPES,
        old_linear.out_features,
        bias=old_linear.bias is not None,
    ).to(device=device, dtype=old_linear.weight.dtype)
    model.CellEncoder.edge_emb[0] = direct_linear
    model.CellEncoder.edge_encoding = "multihot"
    model.CellEncoder._process_edge_attr = types.MethodType(
        _direct_multihot_process_edge_attr,
        model.CellEncoder,
    )
    # Kept in the state_dict for checkpoint compatibility, but unused here.
    model.CellEncoder.edge_type_weight.requires_grad_(False)


def build_model(landmarks, drug_fdim, args, device):
    enc = EncArgs()
    enc.num_feature_drug = drug_fdim
    enc.dim_drug = enc.dim_cell = enc.dim_node = 64
    enc.num_feature_cell = 1
    enc.dropout_ratio = 0.1
    enc.transformer_heads = 8
    enc.transformer_layers = 2
    enc.ffn_dim = 256
    enc.max_num_nodes = args.max_num_nodes
    enc.pe_dim = 1
    enc.edge_encoding = args.edge_encoding
    cfg = ModelConfig(
        dim_node=64, out_drug=64, out_cell=64, pe_dim=1,
        num_pathways=args.num_pathways, transformer_heads=8, ffn_dim=256,
        transformer_layers=2, dropout_ratio=0.1,
        max_num_nodes=args.max_num_nodes, freeze_encoders=False,
        last_n_layers=2, unfreeze_pool_query=True,
        unfreeze_time_dose=True, unfreeze_type_embed=True,
        use_pathway_batching=True, use_gradient_checkpointing=False,
    )
    model = FaithfulTokenDedupDrugResponseTransformer(
        args=enc, landmark_set=landmarks, config=cfg, task="ic50",
        strict_gene_coverage=True,
    ).to(device)
    model.setup_finetune(cfg)
    configure_edge_encoding(model, args.edge_encoding, device)
    configure_direction_aware(
        model,
        args.edge_direction,
        device,
    )

    # The PGE reconstruction head is not used for IC50 and must remain frozen.
    if hasattr(model, "regressor"):
        for parameter in model.regressor.parameters():
            parameter.requires_grad = False

    return model, cfg


def extract_state(pkg):
    if isinstance(pkg, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            value = pkg.get(key)
            if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
                return value
        if pkg and all(isinstance(v, torch.Tensor) for v in pkg.values()):
            return pkg
    raise TypeError("Could not extract model state_dict")


def is_head_key(name):
    plain = name[7:] if name.startswith("module.") else name
    return (
        plain.startswith("regressor.")
        or plain.startswith("ic50_head.")
        or plain in {"pool_query", "pool_query_drug", "pool_query_cell"}
    )


def transfer_backbone(model, checkpoint_path, edge_encoding):
    raw = extract_state(safe_torch_load(checkpoint_path))
    source = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}
    target = model.state_dict()
    eligible = {k: v for k, v in target.items() if not is_head_key(k)}
    loaded, mismatch, missing, adapted = {}, {}, [], {}
    for key, tensor in eligible.items():
        if key not in source:
            missing.append(key)
            continue
        if tuple(source[key].shape) == tuple(tensor.shape):
            loaded[key] = source[key]
            continue

        can_adapt = (
            edge_encoding == "multihot"
            and key.endswith("CellEncoder.edge_emb.0.weight")
            and source[key].dim() == 2
            and source[key].shape[1] == 1
            and tensor.dim() == 2
            and tensor.shape[1] == NUM_EDGE_TYPES
        )
        if can_adapt:
            prefix = key[: -len("edge_emb.0.weight")]
            relation_key = prefix + "edge_type_weight"
            if relation_key not in source:
                raise RuntimeError(
                    f"Missing relation weights needed to adapt {key}: {relation_key}"
                )
            relation = source[relation_key].reshape(1, -1).to(
                dtype=source[key].dtype
            )
            converted = source[key] * relation
            if tuple(converted.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"Adapted edge projection shape mismatch: "
                    f"{tuple(converted.shape)} vs {tuple(tensor.shape)}"
                )
            loaded[key] = converted
            adapted[key] = {
                "source_shape": list(source[key].shape),
                "target_shape": list(tensor.shape),
                "relation_key": relation_key,
                "initialization": (
                    "W_direct[:,j] = W_scalar[:,0] * relation_weight[j]"
                ),
            }
            continue

        mismatch[key] = [list(source[key].shape), list(tensor.shape)]

    merged = dict(target)
    merged.update(loaded)
    model.load_state_dict(merged, strict=True)
    eligible_numel = sum(v.numel() for v in eligible.values())
    loaded_numel = sum(v.numel() for v in loaded.values())
    coverage = loaded_numel / max(1, eligible_numel)
    modules = {}
    for key, tensor in loaded.items():
        prefix = key.split(".", 1)[0]
        modules.setdefault(prefix, {"keys": 0, "numel": 0})
        modules[prefix]["keys"] += 1
        modules[prefix]["numel"] += tensor.numel()
    audit = {
        "checkpoint": checkpoint_path,
        "loaded_keys": len(loaded),
        "eligible_keys": len(eligible),
        "loaded_numel": loaded_numel,
        "eligible_numel": eligible_numel,
        "coverage": coverage,
        "missing_first20": missing[:20],
        "shape_mismatch": mismatch,
        "adapted_keys": adapted,
        "loaded_modules": modules,
        "excluded_task_head_keys": [k for k in target if is_head_key(k)],
    }
    if edge_encoding == "multihot":
        edge_key = next(
            (k for k in source if k.endswith("CellEncoder.edge_emb.0.weight")),
            None,
        )
        if edge_key is None:
            raise RuntimeError("Could not find pretrained scalar edge projection")
        prefix = edge_key[: -len("edge_emb.0.weight")]
        relation_key = prefix + "edge_type_weight"
        bias_key = prefix + "edge_emb.0.bias"
        patterns = torch.cat(
            [
                torch.eye(NUM_EDGE_TYPES),
                torch.randint(0, 2, (128, NUM_EDGE_TYPES)).float(),
            ],
            dim=0,
        )
        old_scalar = patterns @ source[relation_key].float()
        old_out = F.linear(
            old_scalar[:, None],
            source[edge_key].float(),
            source.get(bias_key, None).float() if bias_key in source else None,
        )
        direct = model.CellEncoder.edge_emb[0]
        new_out = F.linear(
            patterns.to(direct.weight.device),
            direct.weight.float(),
            direct.bias.float() if direct.bias is not None else None,
        ).cpu()
        audit["multihot_initialization_max_abs_delta"] = float(
            (old_out - new_out).abs().max().item()
        )
        print(
            "[MULTIHOT INIT EQUIVALENCE AUDIT] "
            + json.dumps(
                {
                    "max_abs_delta": audit[
                        "multihot_initialization_max_abs_delta"
                    ],
                    "patterns": int(patterns.size(0)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if audit["multihot_initialization_max_abs_delta"] > 1e-5:
            raise RuntimeError("Direct multi-hot initialization is not equivalent")

    print("[TRANSFER AUDIT] " + json.dumps(audit, sort_keys=True), flush=True)
    if coverage < 0.98:
        raise RuntimeError(f"Backbone transfer coverage too low: {coverage:.6f}")
    for required in ["DrugEncoder", "CellEncoder", "Transformer", "time_proj",
                     "dose_proj", "gene_embedding", "occurrence_score"]:
        if not any(k == required or k.startswith(required + ".") for k in loaded):
            raise RuntimeError(f"Required module not loaded: {required}")
    return audit


def optimizer_groups(model, args):
    body, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("ic50_head.") or name in {"pool_query", "pool_query_drug", "pool_query_cell"}:
            head.append(p)
        else:
            body.append(p)
    if not body or not head:
        raise RuntimeError(f"Invalid optimizer groups body={len(body)} head={len(head)}")
    return [
        {"params": head, "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": body, "lr": args.body_lr, "weight_decay": args.weight_decay},
    ]


def corr_tensor(pred, y):
    px, py = pred.float() - pred.float().mean(), y.float() - y.float().mean()
    return (px * py).sum() / torch.sqrt((px.square().sum() + 1e-8) * (py.square().sum() + 1e-8))


def gradient_audit(model):
    prefixes = ["DrugEncoder", "CellEncoder", "Transformer", "occurrence_score",
                "gene_embedding", "time_proj", "dose_proj", "ic50_head"]
    out = {k: 0.0 for k in prefixes}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        ss = float(p.grad.detach().float().square().sum())
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                out[prefix] += ss
    return {k: math.sqrt(v) for k, v in out.items()}


def edge_parameter_gradient_audit(model):
    wanted = {
        "CellEncoder.edge_type_weight",
        "CellEncoder.edge_emb.0.weight",
        "CellEncoder.edge_emb.0.bias",
        "CellEncoder.direction_embedding.weight",
    }
    output = {}
    for name, parameter in model.named_parameters():
        if name not in wanted:
            continue
        output[name] = {
            "requires_grad": bool(parameter.requires_grad),
            "grad_l2": (
                None
                if parameter.grad is None
                else float(parameter.grad.detach().float().norm().item())
            ),
        }
    return output


def forward_batch(model, batch, device):
    pred = model(
        drug_graph=batch["drug_graph"], cell_graph_seq=batch["cell_seq"],
        cell_ids=batch["cell_ids"], time=batch["time"].to(device).reshape(-1),
        dose=batch["dose"].to(device).reshape(-1), return_attn=False, bge=None,
    )
    return (pred[0] if isinstance(pred, tuple) else pred).reshape(-1)


def train_epoch(model, loader, optimizer, scaler, device, y_mu, y_sd, epoch, args):
    model.train()
    total = seen = 0
    start = time.perf_counter()
    first_audit = None
    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        y = batch["y"].to(device)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            pred = forward_batch(model, batch, device)
            base = F.mse_loss((pred - y_mu) / y_sd, (y - y_mu) / y_sd)
            loss = base
            if epoch >= args.corr_start_epoch and y.numel() > 1:
                loss = base + args.alpha_corr * (1.0 - corr_tensor(pred, y))

        if (
            not bool(torch.isfinite(pred).all().item())
            or not bool(torch.isfinite(loss).item())
        ):
            raise RuntimeError(
                "[NONFINITE FORWARD] "
                f"step={step} "
                f"pred_finite={bool(torch.isfinite(pred).all().item())} "
                f"loss={float(loss.detach())}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)



        if first_audit is None:
            nonfinite_grad_names = [
                name
                for name, parameter in model.named_parameters()
                if (
                    parameter.grad is not None
                    and not bool(
                        torch.isfinite(parameter.grad).all().item()
                    )
                )
            ]

            if nonfinite_grad_names:
                current_scale = (
                    float(scaler.get_scale())
                    if scaler.is_enabled()
                    else 1.0
                )
                print(
                    "[AMP OVERFLOW RECOVERY] "
                    f"step={step} "
                    f"scale={current_scale:g} "
                    f"nonfinite_count={len(nonfinite_grad_names)} "
                    f"first={nonfinite_grad_names[:8]}",
                    flush=True,
                )

                if not scaler.is_enabled():
                    raise RuntimeError(
                        "Non-finite gradients occurred without AMP: "
                        + repr(nonfinite_grad_names[:20])
                    )


                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        if first_audit is None:
            first_audit = gradient_audit(model)
            print("[FAITHFUL IC50 GRADIENT AUDIT] " + json.dumps(first_audit, sort_keys=True), flush=True)
            print(
                "[EDGE PARAM GRADIENT AUDIT] "
                + json.dumps(edge_parameter_gradient_audit(model), sort_keys=True),
                flush=True,
            )
            bad = [k for k in ["DrugEncoder", "CellEncoder", "Transformer", "occurrence_score", "ic50_head"]
                   if first_audit.get(k, 0.0) <= 0]
            if bad:
                raise RuntimeError(f"Missing gradients: {bad}")
        scaler.step(optimizer)
        scaler.update()
        n = y.numel()
        total += float(loss.detach()) * n
        seen += n
        if step % args.log_interval == 0:
            rate = seen / max(1e-9, time.perf_counter() - start)
            alloc = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0
            reserv = torch.cuda.memory_reserved(device) / 1e9 if device.type == "cuda" else 0
            print(f"  [step {step}] loss={total/seen:.4f} | {rate:.1f} samples/s | GPU={alloc:.1f}/{reserv:.1f}GB", flush=True)
    return total / max(1, seen), first_audit


@torch.no_grad()
def evaluate(model, loader, device, y_mu, y_sd):
    model.eval()
    preds, ys, rows = [], [], []
    std_sum = seen = 0
    for batch in loader:
        y = batch["y"].to(device)
        pred = forward_batch(model, batch, device)
        std_loss = F.mse_loss((pred - y_mu) / y_sd, (y - y_mu) / y_sd)
        p = pred.float().cpu().numpy()
        t = y.float().cpu().numpy()
        preds.append(p); ys.append(t)
        std_sum += float(std_loss) * len(t); seen += len(t)
        for i in range(len(t)):
            rows.append({
                "source_row": int(batch["meta"]["source_row"][i]),
                "cell_id": batch["meta"]["cell_id"][i],
                "drug_id": batch["meta"]["drug_id"][i],
                "canonical_smiles": batch["meta"]["canonical_smiles"][i],
                "y_true": float(t[i]), "y_pred": float(p[i]),
                "residual": float(p[i] - t[i]),
            })
    y, p = np.concatenate(ys).astype(float), np.concatenate(preds).astype(float)
    pcc = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(y) and np.std(p) else float("nan")
    mse = float(np.mean((y - p) ** 2))
    return {
        "std_mse": std_sum / max(1, seen),
        "mse_ln_ic50": mse,
        "rmse_ln_ic50": math.sqrt(mse),
        "mae_ln_ic50": float(np.mean(np.abs(y - p))),
        "pcc": pcc, "n": int(seen),
    }, pd.DataFrame(rows)


def per_drug_metrics(pred_df):
    rows = []
    for smi, g in pred_df.groupby("canonical_smiles", sort=True):
        y, p = g.y_true.to_numpy(float), g.y_pred.to_numpy(float)
        pcc = float(np.corrcoef(y, p)[0, 1]) if len(g) >= 2 and np.std(y) and np.std(p) else float("nan")
        rows.append({
            "canonical_smiles": smi, "drug_id": str(g.drug_id.iloc[0]), "n": len(g),
            "rmse": float(np.sqrt(np.mean((y-p)**2))),
            "mae": float(np.mean(np.abs(y-p))), "pcc": pcc,
        })
    return pd.DataFrame(rows)


def save_manifest(df, c, path):
    out = df[["_source_row", c["cell"], c["drug"], c["smiles"], c["label"]]].copy()
    out.columns = ["source_row", "cell_id", "drug_id", "canonical_smiles", "LN_IC50"]
    out.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    for name in ["csv_path", "basal_csv", "landmark_csv", "drug_graph_cache",
                 "cell_graph_cache", "save_dir", "run_name"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--pretrained_checkpoint")
    p.add_argument("--split_mode", choices=["mixed", "drug_blind", "cell_blind"], default="mixed")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid_ratio", type=float, default=.1)
    p.add_argument("--test_ratio", type=float, default=.1)
    p.add_argument("--subset_ratio", type=float, default=1.)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fixed_dose", type=float, default=1.)
    p.add_argument("--fixed_time", type=float, default=72.)
    p.add_argument("--body_lr", type=float, default=2e-4)
    p.add_argument("--head_lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=1.)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--alpha_corr", type=float, default=.2)
    p.add_argument("--corr_start_epoch", type=int, default=3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--max_num_nodes", type=int, default=96)
    p.add_argument("--num_pathways", type=int, default=31)
    p.add_argument("--mode", choices=["pretrained", "scratch"], default="pretrained")
    p.add_argument(
        "--edge_encoding",
        choices=["scalar", "multihot"],
        default="scalar",
    )
    p.add_argument(
        "--edge_direction",
        choices=["directed", "bidirectional", "direction_aware_bidirectional"],
        default="directed",
    )
    args = Args(**vars(p.parse_args()))

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)
    (save_dir / "provenance").mkdir()
    device = torch.device(args.device)
    print("[ENTRY] faithful IC50 fine-tuning")
    print("[ARGS] " + json.dumps(asdict(args), sort_keys=True), flush=True)
    if device.type == "cuda":
        print("[DEVICE] " + torch.cuda.get_device_name(device))

    df, c = load_compact_gdsc(args.csv_path)
    drug_cache, drug_fdim = load_drug_cache(args.drug_graph_cache)
    cell_cache = load_cell_cache(args.cell_graph_cache, args.num_pathways)
    before = len(df)
    df = df[df[c["smiles"]].isin(drug_cache) & df[c["cell"]].isin(cell_cache)].reset_index(drop=True)
    print(f"[INFO] Cache coverage retained={len(df)}/{before}")
    if len(df) != 415671:
        print(f"[WARN] Expected 415671 covered rows, observed {len(df)}")

    train_full, val_full, test_full = split_dataframe(
        df, args.split_mode, args.split_seed, args.valid_ratio, args.test_ratio, c
    )
    print(f"[SPLIT FULL] train={len(train_full)} val={len(val_full)} test={len(test_full)}")
    manifest_dir = save_dir / "split_manifests"; manifest_dir.mkdir()
    for name, frame in [("train", train_full), ("validation", val_full), ("test", test_full)]:
        save_manifest(frame, c, manifest_dir / f"{name}.csv")

    train_df = subset_df(train_full, args.subset_ratio, args.seed)
    val_df = subset_df(val_full, args.subset_ratio, args.seed)
    test_df = subset_df(test_full, args.subset_ratio, args.seed)
    print(f"[SPLIT ACTIVE] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    y_mu = float(train_df[c["label"]].mean())
    y_sd = float(train_df[c["label"]].std() + 1e-8)
    print(f"[LABEL STANDARDIZATION] mu={y_mu:.8f} sd={y_sd:.8f}")

    datasets = [IC50Dataset(x, c, drug_cache, cell_cache, args.fixed_time, args.fixed_dose)
                for x in (train_df, val_df, test_df)]
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=False, persistent_workers=False,
                  collate_fn=collate_ic50, drop_last=False)
    gen = torch.Generator(); gen.manual_seed(args.seed)
    tr_loader = DataLoader(datasets[0], shuffle=True, generator=gen, **common)
    va_loader = DataLoader(datasets[1], shuffle=False, **common)
    te_loader = DataLoader(datasets[2], shuffle=False, **common)

    landmarks = load_landmarks(args.landmark_csv)
    if len(landmarks) != 268:
        raise RuntimeError(f"Expected 268 landmarks, got {len(landmarks)}")
    model, cfg = build_model(landmarks, drug_fdim, args, device)
    print(
        "[EDGE ABLATION AUDIT] "
        + json.dumps(
            {
                "edge_encoding": args.edge_encoding,
                "edge_direction": args.edge_direction,
                "edge_projection_in_features": int(
                    model.CellEncoder.edge_emb[0].in_features
                ),
                "edge_type_weight_trainable": bool(
                    model.CellEncoder.edge_type_weight.requires_grad
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    transfer = None
    if args.mode == "pretrained":
        if not args.pretrained_checkpoint:
            raise ValueError("Missing pretrained checkpoint")
        transfer = transfer_backbone(
            model, args.pretrained_checkpoint, args.edge_encoding
        )
    else:
        print("[TRANSFER AUDIT] scratch initialization; no checkpoint loaded")
    print(f"[TRAINABLE] numel={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.AdamW(optimizer_groups(model, args))
    def lr_factor(e):
        if args.warmup_epochs and e < args.warmup_epochs:
            return (e + 1) / args.warmup_epochs
        remain = max(1, args.epochs - args.warmup_epochs)
        progress = min(max((e - args.warmup_epochs + 1) / remain, 0), 1)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == "cuda",
        init_scale=1024.0,
        growth_interval=2000,
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt = save_dir / f"{args.run_name}_{stamp}.pth"
    best_pcc, best_epoch, no_improve = -float("inf"), 0, 0
    history = []
    hashes_before = {
        "drug_graph_cache": sha256_file(args.drug_graph_cache),
        "cell_graph_cache": sha256_file(args.cell_graph_cache),
    }
    os.environ["DRT_FAITHFUL_AUDIT"] = "1"

    for epoch in range(1, args.epochs + 1):
        train_loss, grad = train_epoch(model, tr_loader, optimizer, scaler, device, y_mu, y_sd, epoch, args)
        val_metrics, _ = evaluate(model, va_loader, device, y_mu, y_sd)
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"| val_std_mse={val_metrics['std_mse']:.4f} "
            f"| val_pcc={val_metrics['pcc']:.4f} "
            f"| val_rmse_ln_ic50={val_metrics['rmse_ln_ic50']:.4f} "
            f"| val_mae_ln_ic50={val_metrics['mae_ln_ic50']:.4f}",
            flush=True,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics,
                        "lrs": [g["lr"] for g in optimizer.param_groups],
                        "gradient_audit": grad})
        if val_metrics["pcc"] > best_pcc + args.min_delta:
            best_pcc, best_epoch, no_improve = val_metrics["pcc"], epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": asdict(args), "model_config": asdict(cfg),
                "epoch": epoch, "val_metrics": val_metrics,
                "label_mu": y_mu, "label_sd": y_sd,
                "transfer_audit": transfer,
            }, ckpt)
            print(f"[SAVE] {ckpt} (epoch={epoch}, val_pcc={best_pcc:.6f})")
        else:
            no_improve += 1
        scheduler.step()
        print(f"[SCHEDULER AUDIT] epoch={epoch} next_lrs={[g['lr'] for g in optimizer.param_groups]}")
        if args.patience > 0 and no_improve >= args.patience:
            print(f"[EARLY STOP] best_epoch={best_epoch} best_val_pcc={best_pcc:.6f}")
            break

    if not ckpt.exists():
        raise RuntimeError("No checkpoint saved")
    best = safe_torch_load(ckpt)
    model.load_state_dict(best["model_state_dict"], strict=True)
    print(f"[BEST CHECKPOINT] reloaded from disk: {ckpt} "
          f"(epoch={best['epoch']}, val_pcc={best['val_metrics']['pcc']:.6f})")

    val_metrics, val_pred = evaluate(model, va_loader, device, y_mu, y_sd)
    test_metrics, test_pred = evaluate(model, te_loader, device, y_mu, y_sd)
    val_expected = val_df.set_index("_source_row")[c["label"]]
    test_expected = test_df.set_index("_source_row")[c["label"]]
    val_delta = float(np.max(np.abs(val_pred.y_true.to_numpy() - val_pred.source_row.map(val_expected).to_numpy())))
    test_delta = float(np.max(np.abs(test_pred.y_true.to_numpy() - test_pred.source_row.map(test_expected).to_numpy())))
    print(f"[PREDICTION ORDER AUDIT][validation] n={len(val_pred)} label_max_abs_delta={val_delta:.8g}")
    print(f"[PREDICTION ORDER AUDIT][test] n={len(test_pred)} label_max_abs_delta={test_delta:.8g}")
    if val_delta > 1e-5 or test_delta > 1e-5:
        raise RuntimeError("Prediction-order audit failed")

    print(f"[VAL BEST] std_mse={val_metrics['std_mse']:.4f} pcc={val_metrics['pcc']:.4f} "
          f"rmse_ln_ic50={val_metrics['rmse_ln_ic50']:.4f} mae_ln_ic50={val_metrics['mae_ln_ic50']:.4f}")
    print(f"[TEST] std_mse={test_metrics['std_mse']:.4f} pcc={test_metrics['pcc']:.4f} "
          f"rmse_ln_ic50={test_metrics['rmse_ln_ic50']:.4f} mae_ln_ic50={test_metrics['mae_ln_ic50']:.4f}")

    val_pd, test_pd = per_drug_metrics(val_pred), per_drug_metrics(test_pred)
    val_macro = float(val_pd.pcc.dropna().mean()); test_macro = float(test_pd.pcc.dropna().mean())
    print(f"[PER-DRUG][VALIDATION] n_drugs={len(val_pd)} defined_pcc={val_pd.pcc.notna().sum()} macro_pcc_mean={val_macro}")
    print(f"[PER-DRUG][TEST] n_drugs={len(test_pd)} defined_pcc={test_pd.pcc.notna().sum()} macro_pcc_mean={test_macro}")

    prefix = ckpt.with_suffix("")
    paths = {
        "checkpoint": str(ckpt),
        "val_predictions": str(prefix) + ".val_predictions.csv",
        "test_predictions": str(prefix) + ".test_predictions.csv",
        "val_per_drug": str(prefix) + ".val_per_drug_metrics.csv",
        "test_per_drug": str(prefix) + ".test_per_drug_metrics.csv",
        "report": str(prefix) + ".report.json",
    }
    val_pred.to_csv(paths["val_predictions"], index=False)
    test_pred.to_csv(paths["test_predictions"], index=False)
    val_pd.to_csv(paths["val_per_drug"], index=False)
    test_pd.to_csv(paths["test_per_drug"], index=False)

    hashes_after = {
        "drug_graph_cache": sha256_file(args.drug_graph_cache),
        "cell_graph_cache": sha256_file(args.cell_graph_cache),
    }
    if hashes_before != hashes_after:
        raise RuntimeError("Cache hash changed")
    print("[CACHE INTEGRITY AUDIT] passed")

    report = {
        "args": asdict(args), "model_config": asdict(cfg),
        "split_full": {"train": len(train_full), "validation": len(val_full), "test": len(test_full)},
        "split_active": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "label_mu": y_mu, "label_sd": y_sd,
        "best_epoch": int(best["epoch"]), "best_validation": val_metrics,
        "test": test_metrics,
        "per_drug_macro_pcc": {"validation": val_macro, "test": test_macro},
        "transfer_audit": transfer, "history": history,
        "token_audit": getattr(model, "last_token_audit", {}),
        "cache_hashes": hashes_after,
        "prediction_order_max_abs_delta": {"validation": val_delta, "test": test_delta},
        "paths": paths,
    }
    Path(paths["report"]).write_text(json.dumps(report, indent=2, sort_keys=True))
    print("[INFO] Report saved -> " + paths["report"])
    print("[PASS] Faithful canonical970 IC50 run completed")


if __name__ == "__main__":
    main()
