# train_ic50_with_cell_graph_resume.py (A-setting: DrugEncoder trainable, CellEncoder frozen)
# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from torch_geometric.data import Data, Batch

# ── Silence RDKit logs ────────────────────────────────────────────────────────
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.debug')
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
except Exception:
    pass

# ── Flexible imports ───────────────────────────────────────────────────────────
try:
    from Model.DrugResponseTransformer import DrugResponseTransformer, ModelConfig
except Exception:
    from DrugResponseTransformer import DrugResponseTransformer, ModelConfig  # fallback

try:
    from Model.drug_graph import smiles_to_graph
except Exception:
    from drug_graph import smiles_to_graph  # fallback

try:
    from Model.CellLine_graph import create_cell_line_graph
except Exception:
    from CellLine_graph import create_cell_line_graph  # fallback


# ───────────────────────────────────────────────────────────────────────────────
# 0) Reproducibility & MP strategy
# ───────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ───────────────────────────────────────────────────────────────────────────────
# 1) Args
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class Args:
    # Paths
    csv_path: str = "GDSC_with_SMILES_BGE_entrez_KEGG_fixed.csv"
    save_dir: str = "checkpoints_ic50_graphcache"
    run_name: Optional[str] = None

    # Graph caches
    drug_graph_cache: Optional[str] = "graph_cache/drug_graph_cache.pt"
    cell_graph_cache: Optional[str] = "graph_cache/cell_graph_cache.gdsc_unknown_z.fixed.pt"
    cache_scope: str = "all"  # {"all", "train_only"}

    # Fast cell-embed cache ([P,D]) — if not using node-graph cache
    cell_embed_cache_path: Optional[str] = "graph_cache/cell_embed_cache.pth"
    rebuild_cell_embed_cache: bool = False

    # KEGG graph options (fallback)
    kegg_pathway_dir: Optional[str] = "pathwayxml"
    num_pathways: int = 31

    # Hardware
    # ▶ 여기 cuda:1 로 변경
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0              # RDKit 문제 때문에 0 유지
    pin_memory: bool = False
    persistent_workers: bool = False

    # Data
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    split_mode: str = "mixed"  # ["mixed", "drug_blind", "cell_blind"]
    drop_na_label: bool = True
    subset_ratio: float = 1.0

    # Training (fresh run 기준)
    epochs: int = 60
    batch_size: int = 64
    grad_clip: float = 1.0
    amp: bool = True
    log_interval: int = 200
    warmup_epochs: int = 2

    # Optim
    body_lr: float = 2e-4
    head_lr: float = 7e-4
    weight_decay: float = 5e-4

    # Model dims
    dim_node: int = 64
    dim_drug: int = 64
    dim_cell: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 8
    dropout_ratio: float = 0.1
    max_num_nodes: int = 44
    ffn_dim: Optional[int] = None
    pe_dim: int = 1

    # Fine-tune policy
    freeze_encoders: bool = False
    last_n_layers: int = 2
    unfreeze_pool_query: bool = True
    unfreeze_time_dose: bool = True
    unfreeze_type_embed: bool = False

    # Staged unfreeze
    staged_unfreeze: bool = False
    unfreeze_drug_epoch: int = 3
    unfreeze_cell_epoch: int = 999  # kept for compatibility

    # Misc
    seed: int = 42

    # Fixed condition
    fixed_time: float = 72.0
    fixed_dose: float = 10.0

    # Auto batch-size & accumulation
    auto_batch_tune: bool = True
    target_effective_bs: int = 256
    max_probe_bs: int = 128

    # Loss / z-norm
    use_label_standardize: bool = True
    use_corr_loss: bool = True
    alpha_corr: float = 0.2
    use_smoothl1: bool = False

    # Node-graph cache 스위치
    use_cell_graph_cache: bool = False

    # ── Resume 옵션 ────────────────────────────────────────────────────────────
    resume_from: Optional[str] = None
    epochs_more: int = 12
    resume_strict: bool = False
    resume_lower_lr: bool = True

    # ── Early Stopping 옵션 ────────────────────────────────────────────────────
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-3

    # ── BGE branch 옵션 ────────────────────────────────────────────────────────
    # baseline GAT+BGE와 fair하게 비교하기 위해 BGE 벡터도 추가 입력으로 사용
    use_bge_branch: bool = True      # BGE를 모델에 전달할지 여부
    bge_standardize: bool = False    # 필요하면 z-score로 스케일링 (지금은 False로 시작)


# ───────────────────────────────────────────────────────────────────────────────
# 2) CSV parsing helpers
# ───────────────────────────────────────────────────────────────────────────────
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def build_colmap(df: pd.DataFrame) -> Dict[str, object]:
    cell_col  = _find_col(df, ["CELL_LINE_NAME", "cell_line_name", "cell", "cell_line"])
    drug_col  = _find_col(df, ["DRUG_NAME", "drug_name", "drug"])
    minc_col  = _find_col(df, ["MIN_CONC", "min_conc", "minc"])
    maxc_col  = _find_col(df, ["MAX_CONC", "maxc", "maxc"])
    label_col = _find_col(df, ["LN_IC50", "ln_ic50", "IC50", "ic50"])
    smi_col   = _find_col(df, ["canonical_smiles", "smiles", "SMILES"])

    required = {
        "CELL_LINE_NAME": cell_col,
        "DRUG_NAME": drug_col,
        "MIN_CONC": minc_col,
        "MAX_CONC": maxc_col,
        "LN_IC50": label_col,
        "canonical_smiles": smi_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Required column(s) missing or not recognized: {missing}")

    gene_cols = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]
    if len(gene_cols) == 0:
        raise ValueError("No gene-expression columns detected (digits-only EntrezID column names).")

    return {
        "cell": cell_col,
        "drug": drug_col,
        "minc": minc_col,
        "maxc": maxc_col,
        "label": label_col,
        "smiles": smi_col,
        "genes": gene_cols,
    }


# ───────────────────────────────────────────────────────────────────────────────
# 3) Split helpers + subset
# ───────────────────────────────────────────────────────────────────────────────
def split_dataframe(df: pd.DataFrame, valid_ratio: float, test_ratio: float, seed: int,
                    mode: str, colmap: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert 0 < valid_ratio < 0.5 and 0 < test_ratio < 0.5 and valid_ratio + test_ratio < 0.9
    rng = np.random.RandomState(seed)

    if mode == "drug_blind":
        keys = df[colmap["smiles"]].astype(str).unique()
        rng.shuffle(keys)
        n = len(keys)
        n_test = int(n * test_ratio); n_val = int(n * valid_ratio)
        test_keys = set(keys[:n_test]); val_keys = set(keys[n_test:n_test + n_val])
        train = df[~df[colmap["smiles"]].isin(test_keys | val_keys)]
        valid = df[df[colmap["smiles"]].isin(val_keys)]
        test  = df[df[colmap["smiles"]].isin(test_keys)]
    elif mode == "cell_blind":
        keys = df[colmap["cell"]].astype(str).unique()
        rng.shuffle(keys)
        n = len(keys)
        n_test = int(n * test_ratio); n_val = int(n * valid_ratio)
        test_keys = set(keys[:n_test]); val_keys = set(keys[n_test:n_test + n_val])
        train = df[~df[colmap["cell"]].isin(test_keys | val_keys)]
        valid = df[df[colmap["cell"]].isin(val_keys)]
        test  = df[df[colmap["cell"]].isin(test_keys)]
    else:
        idx = np.arange(len(df)); rng.shuffle(idx)
        n_test = int(len(idx) * test_ratio); n_val  = int(len(idx) * valid_ratio)
        test_idx = idx[:n_test]; val_idx  = idx[n_test:n_test + n_val]
        train_idx= idx[n_test + n_val:]
        train = df.iloc[train_idx]; valid = df.iloc[val_idx]; test  = df.iloc[test_idx]

    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)

def subset_df(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    if ratio >= 0.999:
        return df
    n = max(1, int(len(df) * ratio))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ───────────────────────────────────────────────────────────────────────────────
# 4) PyG graph sanitizers
# ───────────────────────────────────────────────────────────────────────────────
def _to_resizable_cpu(t: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        arr = t.detach().to("cpu").contiguous().numpy().copy()
        out = torch.from_numpy(arr)
    else:
        out = torch.tensor(t)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out

def sanitize_pyg_data(g: Data) -> Data:
    if not isinstance(g, Data):
        raise TypeError(f"Expected Data, got {type(g)}")

    out = Data()

    if hasattr(g, "x") and g.x is not None:
        out.x = _to_resizable_cpu(g.x, torch.float32)
    else:
        raise ValueError("graph has no 'x'")

    if hasattr(g, "edge_index") and g.edge_index is not None:
        ei = _to_resizable_cpu(g.edge_index, torch.long)
        if ei.dim() != 2 or ei.size(0) != 2:
            raise ValueError(f"edge_index must be [2, E], got {tuple(ei.size())}")
        out.edge_index = ei
    else:
        raise ValueError("graph has no 'edge_index'")

    if hasattr(g, "edge_attr") and g.edge_attr is not None:
        out.edge_attr = _to_resizable_cpu(g.edge_attr, torch.float32)
    if hasattr(g, "pos") and g.pos is not None:
        out.pos = _to_resizable_cpu(g.pos, torch.float32)
    if hasattr(g, "y") and g.y is not None:
        if isinstance(g.y, torch.Tensor):
            out.y = _to_resizable_cpu(g.y, torch.float32)
        else:
            out.y = torch.tensor(float(g.y), dtype=torch.float32)

    out.num_nodes = int(getattr(g, "num_nodes", out.x.size(0)))

    skip_keys = {
        "x", "edge_index", "edge_attr", "pos", "y", "num_nodes",
        "__num_nodes__", "__slices__", "__cat_dim__", "__cumsum__", "__inc__"
    }
    for k, v in g.__dict__.items():
        if k in skip_keys:
            continue
        try:
            if torch.is_tensor(v):
                setattr(out, k, _to_resizable_cpu(v))
            else:
                setattr(out, k, v)
        except Exception:
            pass

    def _coerce_to_int_list(v, n_expected):
        if torch.is_tensor(v):
            v = v.detach().cpu().tolist()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        v = list(map(int, v))
        if len(v) != n_expected:
            raise ValueError(f"[sanitize] candidate node_ids length {len(v)} != num_nodes {n_expected}")
        return v

    if hasattr(out, "node_ids"):
        out.node_ids = _coerce_to_int_list(getattr(out, "node_ids"), out.num_nodes)
    else:
        candidates = []
        for name in ["gene_ids", "entrez_ids", "node_id_list", "nodes", "ids"]:
            if hasattr(out, name):
                candidates.append(name)
        assigned = False
        for name in candidates:
            try:
                vals = getattr(out, name)
                out.node_ids = _coerce_to_int_list(vals, out.num_nodes)
                assigned = True
                break
            except Exception:
                continue
        if not assigned:
            out.node_ids = list(range(out.num_nodes))
    return out

def sanitize_graph_list(graphs: List[Data]) -> List[Data]:
    return [sanitize_pyg_data(g) for g in graphs]

def enforce_drug_feature_dim(drug_cache: Optional[Dict[str, Any]], expected_dim: int) -> Optional[Dict[str, Any]]:
    if not isinstance(drug_cache, dict):
        return drug_cache
    for k in list(drug_cache.keys()):
        v = drug_cache[k]
        ok = isinstance(v, Data) and hasattr(v, "x") and isinstance(v.x, torch.Tensor) and v.x.ndim == 2
        if not ok or v.x.shape[1] != expected_dim:
            del drug_cache[k]
    return drug_cache

def enforce_cell_feature_dim(cell_cache: Optional[Dict[str, Any]], expected_dim: int = 1) -> Optional[Dict[str, Any]]:
    if not isinstance(cell_cache, dict):
        return cell_cache
    for k in list(cell_cache.keys()):
        v = cell_cache[k]
        ok = isinstance(v, list) and len(v) > 0
        if not ok:
            del cell_cache[k]
            continue
        good = True
        for g in v:
            if not (
                isinstance(g, Data)
                and hasattr(g, "x")
                and isinstance(g.x, torch.Tensor)
                and g.x.ndim == 2
                and g.x.shape[1] == expected_dim
            ):
                good = False
                break
        if not good:
            del cell_cache[k]
    return cell_cache


# ───────────────────────────────────────────────────────────────────────────────
# 5) Dataset
# ───────────────────────────────────────────────────────────────────────────────
class IC50Dataset(Dataset):
    """
    __getitem__은
      - drug_graph
      - cell_id
      - label
      - fixed time/dose
      - (옵션) bge 벡터
    를 반환하고, collate에서 cell_graph_cache 또는 cell_embed_cache를 사용.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        colmap: Dict[str, object],
        gene_order: List[str],
        fixed_time: float,
        fixed_dose: float,
        drug_cache: Optional[Dict[str, Data]] = None,
        expected_drug_fdim: Optional[int] = None,
        use_bge_branch: bool = False,
        bge_mu: Optional[np.ndarray] = None,
        bge_sd: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.colmap = colmap
        # gene_order = Entrez ID 문자열 컬럼 이름 리스트
        self.gene_order = [str(g) for g in gene_order]
        self.fixed_time = float(fixed_time)
        self.fixed_dose = float(fixed_dose)
        self.expected_drug_fdim = expected_drug_fdim
        self.drug_cache = drug_cache or None  # 이미 warmup 시 sanitize했다고 가정

        # BGE branch 관련
        self.use_bge_branch = use_bge_branch
        self.bge_mu = bge_mu
        self.bge_sd = bge_sd

    def _get_drug_graph(self, smiles: str, drug_id: str) -> Data:
        g = None
        if self.drug_cache is not None:
            for key in (
                smiles,
                smiles.lower(),
                drug_id,
                drug_id.lower(),
            ):
                if key in self.drug_cache and isinstance(
                    self.drug_cache[key], Data
                ):
                    g = self.drug_cache[key]
                    break

        if g is None:
            g = _safe_smiles_to_graph_or_none(smiles)
            if g is None:
                raise ValueError(
                    f"Invalid SMILES after filtering: '{smiles}' (drug_id='{drug_id}')"
                )
            g = sanitize_pyg_data(g)

        if (
            self.expected_drug_fdim is not None
            and hasattr(g, "x")
            and g.x is not None
            and g.x.ndim == 2
            and int(g.x.shape[1]) != int(self.expected_drug_fdim)
        ):
            g2 = _safe_smiles_to_graph_or_none(smiles)
            if g2 is not None:
                g2 = sanitize_pyg_data(g2)
                if (
                    hasattr(g2, "x")
                    and g2.x is not None
                    and g2.x.ndim == 2
                    and int(g2.x.shape[1]) == int(self.expected_drug_fdim)
                ):
                    g = g2
            if int(g.x.shape[1]) != int(self.expected_drug_fdim):
                raise ValueError(
                    f"[FDIM mismatch] drug_id={drug_id} smiles={smiles} "
                    f"got={g.x.shape[1]} expected={self.expected_drug_fdim}"
                )

        if self.drug_cache is not None:
            for key in (
                smiles,
                smiles.lower(),
                drug_id,
                drug_id.lower(),
            ):
                self.drug_cache[key] = g
        return g

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        smiles = str(row[self.colmap["smiles"]])
        drug_id = str(row[self.colmap["drug"]])
        cell_id = str(row[self.colmap["cell"]])

        drug_graph = self._get_drug_graph(smiles, drug_id)
        y = torch.tensor(float(row[self.colmap["label"]]), dtype=torch.float32)
        t = torch.tensor(self.fixed_time, dtype=torch.float32)
        d = torch.tensor(self.fixed_dose, dtype=torch.float32)

        # ── BGE 벡터 구성 ─────────────────────────────────────────────────────
        if self.use_bge_branch:
            # gene_order 순서대로 값 뽑기
            vals = row[self.gene_order].astype(float).to_numpy()
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            if self.bge_mu is not None and self.bge_sd is not None:
                vals = (vals - self.bge_mu) / (self.bge_sd + 1e-6)
            bge = torch.from_numpy(vals.astype(np.float32))
        else:
            bge = None

        return {
            "drug_graph": drug_graph,
            "time": t,
            "dose": d,
            "y": y,
            "cell_id": cell_id,
            "drug_id": drug_id,
            "bge": bge,
        }


# ───────────────────────────────────────────────────────────────────────────────
# 6) Collate: graph-cache(노드 단위) & fast-cache([P,D])
# ───────────────────────────────────────────────────────────────────────────────
def pad_cell_embed_sequence(
    seq_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    B = len(seq_list)
    Pmax = max(x.size(0) for x in seq_list)
    D = seq_list[0].size(1)
    device = seq_list[0].device
    out = torch.zeros(B, Pmax, D, device=device, dtype=seq_list[0].dtype)
    mask = torch.ones(B, Pmax, device=device, dtype=torch.bool)  # True=PAD
    for i, x in enumerate(seq_list):
        P = x.size(0)
        out[i, :P, :] = x
        mask[i, :P] = False
    return out, mask


def make_collate_fn_with_cell_embed_cache(
    cell_embed_cache: Dict[str, torch.Tensor]
):
    def _collate(batch: List[Dict]):
        drug_batch = Batch.from_data_list(
            [b["drug_graph"] for b in batch]
        )
        time_t = torch.stack([b["time"] for b in batch], 0)
        dose_t = torch.stack([b["dose"] for b in batch], 0)
        y_t = torch.stack([b["y"] for b in batch], 0)
        cell_ids = [b["cell_id"] for b in batch]
        seq_list = [cell_embed_cache[cid] for cid in cell_ids]
        cell_embed, cell_pad_mask = pad_cell_embed_sequence(seq_list)

        # ── BGE 스택 ──────────────────────────────────────────────────────
        if batch[0].get("bge") is not None:
            bge_t = torch.stack(
                [b["bge"] for b in batch], dim=0
            )
        else:
            bge_t = None

        meta = {
            "cell_id": cell_ids,
            "drug_id": [b["drug_id"] for b in batch],
        }
        return {
            "drug_graph": drug_batch,
            "cell_graph_seq": None,
            "cell_embed": cell_embed,
            "cell_pad_mask": cell_pad_mask,
            "time": time_t,
            "dose": dose_t,
            "y": y_t,
            "bge": bge_t,
            "meta": meta,
        }

    return _collate


def make_collate_fn_with_cell_graph_cache(
    cell_graph_cache: Dict[str, List[Data]],
    max_pathways: int,
):
    """
    cell_graph_cache는 미리 sanitize된 상태라고 가정.
    collate에서는 slice/index만 수행.
    """
    def _collate(batch: List[Dict]):
        drug_batch = Batch.from_data_list(
            [b["drug_graph"] for b in batch]
        )
        time_t = torch.stack([b["time"] for b in batch], 0)
        dose_t = torch.stack([b["dose"] for b in batch], 0)
        y_t = torch.stack([b["y"] for b in batch], 0)

        cell_ids = [b["cell_id"] for b in batch]
        seq_list = []
        for cid in cell_ids:
            seq = cell_graph_cache[cid]
            if max_pathways is not None and max_pathways > 0:
                seq = seq[:max_pathways]
            seq_list.append(seq)

        # ── BGE 스택 ──────────────────────────────────────────────────────
        if batch[0].get("bge") is not None:
            bge_t = torch.stack(
                [b["bge"] for b in batch], dim=0
            )
        else:
            bge_t = None

        meta = {
            "cell_id": cell_ids,
            "drug_id": [b["drug_id"] for b in batch],
        }
        return {
            "drug_graph": drug_batch,
            "cell_graph_seq": seq_list,  # List[List[Data]]
            "cell_embed": None,
            "cell_pad_mask": None,
            "time": time_t,
            "dose": dose_t,
            "y": y_t,
            "bge": bge_t,
            "meta": meta,
        }

    return _collate


# ───────────────────────────────────────────────────────────────────────────────
# 7) Metrics / Loss
# ───────────────────────────────────────────────────────────────────────────────
def pcc_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu()
    y = y.detach().cpu()
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float((x * y).mean().item())


def corr_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = (pred - pred.mean()) / (pred.std() + 1e-8)
    t = (y - y.mean()) / (y.std() + 1e-8)
    return 1.0 - (x * t).mean()


# ───────────────────────────────────────────────────────────────────────────────
# 8) Cell 임베딩 fast-cache 사전계산([P,D])
# ───────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def precompute_cell_embeds_or_load(
    args: Args,
    model: DrugResponseTransformer,
    all_cells: List[str],
    cell_graph_cache: Optional[Dict[str, List[Data]]],
    df_all: pd.DataFrame,
    colmap: Dict[str, str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    path = args.cell_embed_cache_path
    if (not args.rebuild_cell_embed_cache) and path and os.path.isfile(
        path
    ):
        print(f"[INFO] Loading cell_embed_cache from {path}")
        obj = torch.load(path, map_location="cpu")
        for k in list(obj.keys()):
            obj[k] = obj[k].to(device, non_blocking=True)
        return obj

    if (cell_graph_cache is None or len(cell_graph_cache) == 0) and (
        args.kegg_pathway_dir is None
        or not os.path.isdir(args.kegg_pathway_dir)
    ):
        raise RuntimeError(
            "No source to build cell_embed_cache. Provide cell_graph_cache or KEGG pathway dir."
        )

    if not hasattr(model, "CellEncoder"):
        raise RuntimeError(
            "Model must have CellEncoder to precompute cell embeddings."
        )
    model.CellEncoder.eval()

    basal_df = (
        df_all[[colmap["cell"]] + [str(c) for c in colmap["genes"]]]
        .groupby(colmap["cell"], as_index=True)
        .first()
        if (args.kegg_pathway_dir and os.path.isdir(args.kegg_pathway_dir))
        else None
    )

    out: Dict[str, torch.Tensor] = {}
    uniq_cells = sorted(set(all_cells))
    for i, cid in enumerate(uniq_cells):
        if i % 50 == 0:
            print(f"[EMB] {i}/{len(uniq_cells)}")
        if cell_graph_cache is not None and cid in cell_graph_cache:
            seq = cell_graph_cache[cid][: args.num_pathways]
        elif basal_df is not None:
            graphs, _ = create_cell_line_graph(
                basal_df,
                args.kegg_pathway_dir,
                cid,
                max_pathways=args.num_pathways,
            )
            seq = sanitize_graph_list(graphs)
        else:
            raise KeyError(
                f"cell_id '{cid}' not found in cell_graph_cache and no KEGG fallback."
            )

        path_embs = []
        for pi, g in enumerate(seq):
            gd = g.to(device, non_blocking=True)
            node_feats = model.CellEncoder(gd, pi)
            if node_feats.dim() != 2:
                raise RuntimeError("CellEncoder must return (N, D)")
            path_embs.append(node_feats.mean(dim=0))
        if len(path_embs) == 0:
            raise RuntimeError(f"No pathway graphs for cell '{cid}'")
        cell_pd = torch.stack(path_embs, dim=0)
        out[cid] = cell_pd.detach().clone().to(device)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({k: v.cpu() for k, v in out.items()}, path)
        print(
            f"[INFO] Saved cell_embed_cache to {path} (cells={len(out)})"
        )

    return out


# ───────────────────────────────────────────────────────────────────────────────
# 9) 공통: 모델 호출 헬퍼
# ───────────────────────────────────────────────────────────────────────────────
def model_forward_from_batch(model, batch, device, time_, dose_):
    drug_graph = batch["drug_graph"].to(device, non_blocking=True)
    bge = batch.get("bge", None)
    if isinstance(bge, torch.Tensor):
        bge = bge.to(device, non_blocking=True)

    if batch.get("cell_graph_seq", None) is not None:
        return model(
            drug_graph=drug_graph,
            cell_graph_seq=batch["cell_graph_seq"],
            cell_embed=None,
            cell_pad_mask=None,
            time=time_,
            dose=dose_,
            bge=bge,
            return_attn=False,
        )
    else:
        return model(
            drug_graph=drug_graph,
            cell_graph_seq=None,
            cell_embed=batch["cell_embed"],
            cell_pad_mask=batch["cell_pad_mask"],
            time=time_,
            dose=dose_,
            bge=bge,
            return_attn=False,
        )


# ───────────────────────────────────────────────────────────────────────────────
# 10) Train / Eval
# ───────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
    grad_clip=1.0,
    log_interval: int = 200,
    accum_steps: int = 1,
    y_mu: Optional[float] = None,
    y_sd: Optional[float] = None,
    use_label_standardize: bool = True,
    use_corr_loss: bool = True,
    alpha_corr: float = 0.2,
    use_smoothl1: bool = False,
):
    model.train()
    base_loss = nn.SmoothL1Loss(beta=0.5) if use_smoothl1 else nn.MSELoss()
    total_loss, n = 0.0, 0
    running = 0.0
    step_in_accum = 0
    epoch_t0 = time.time()
    last_log_t = epoch_t0
    n_since_log = 0

    for step, batch in enumerate(loader, 1):
        time_ = batch["time"].to(device, non_blocking=True)
        dose_ = batch["dose"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        if step_in_accum == 0:
            optimizer.zero_grad(set_to_none=True)

        def _loss_fn(pred, yy):
            if use_label_standardize and (y_mu is not None) and (y_sd is not None):
                core = base_loss((pred - y_mu) / y_sd, (yy - y_mu) / y_sd)
            else:
                core = base_loss(pred, yy)
            if use_corr_loss and alpha_corr > 0:
                core = core + alpha_corr * corr_loss(pred, yy)
            return core

        if scaler is not None and scaler.is_enabled():
            with torch.cuda.amp.autocast():
                pred = model_forward_from_batch(
                    model, batch, device, time_, dose_
                )
                loss = _loss_fn(pred, y) / max(1, accum_steps)
            scaler.scale(loss).backward()
        else:
            pred = model_forward_from_batch(
                model, batch, device, time_, dose_
            )
            loss = _loss_fn(pred, y) / max(1, accum_steps)
            loss.backward()

        step_in_accum += 1
        do_step = (step_in_accum == accum_steps) or (step == len(loader))
        if do_step:
            if scaler is not None and scaler.is_enabled():
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip
                    )
                optimizer.step()
            step_in_accum = 0

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs * max(1, accum_steps)
        running += float(loss.item()) * bs * max(1, accum_steps)
        n += bs
        n_since_log += bs

        if log_interval and (step % log_interval == 0):
            now = time.time()
            interval_dt = max(1e-6, now - last_log_t)
            samples_per_sec = n_since_log / interval_dt
            avg_step_sec = interval_dt / log_interval
            denom = max(1, (log_interval * loader.batch_size))
            print(
                f"  [step {step}] running_loss={(running/denom):.4f} | "
                f"avg_step={avg_step_sec:.3f}s | "
                f"throughput≈{samples_per_sec:.1f} samples/s"
            )
            last_log_t = now
            n_since_log = 0
            running = 0.0

    epoch_train_sec = time.time() - epoch_t0
    avg_step_sec = epoch_train_sec / max(1, len(loader))
    throughput = n / max(1e-6, epoch_train_sec)
    print(
        f"[Time] train_epoch: {epoch_train_sec:.1f}s | "
        f"avg_step={avg_step_sec:.3f}s | "
        f"throughput≈{throughput:.1f} samples/s"
    )
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    device,
    y_mu: Optional[float] = None,
    y_sd: Optional[float] = None,
    use_label_standardize: bool = True,
):
    model.eval()
    base_loss = nn.MSELoss()
    total_loss, n = 0.0, 0
    preds, gts = [], []
    for batch in loader:
        time_ = batch["time"].to(device, non_blocking=True)
        dose_ = batch["dose"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        pred = model_forward_from_batch(model, batch, device, time_, dose_)

        if use_label_standardize and (y_mu is not None) and (y_sd is not None):
            loss = base_loss((pred - y_mu) / y_sd, (y - y_mu) / y_sd)
        else:
            loss = base_loss(pred, y)

        total_loss += float(loss.item()) * y.shape[0]
        n += y.shape[0]
        preds.append(pred.detach())
        gts.append(y.detach())

    if n == 0:
        return {"mse": float("nan"), "pcc": float("nan")}
    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    return {"mse": total_loss / n, "pcc": pcc_torch(preds, gts)}


# ───────────────────────────────────────────────────────────────────────────────
# 11) Utilities
# ───────────────────────────────────────────────────────────────────────────────
def _maybe_load_cache(path: Optional[str], name: str):
    if path and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            return obj
        except Exception as e:
            print(f"[WARN] Failed to load {name} {path}: {e}")
    return None


def _safe_smiles_to_graph_or_none(smiles: str) -> Optional[Data]:
    try:
        return smiles_to_graph(smiles)
    except Exception:
        return None


def filter_invalid_smiles(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    smi_col = colmap["smiles"]
    vals = df[smi_col].astype(str).fillna("").str.strip()
    mask_blank_or_digits = (vals == "") | vals.str.fullmatch(r"\d+")
    df1 = df[~mask_blank_or_digits].copy()

    rdkit_fail_idx = []
    for i, s in df1[smi_col].astype(str).items():
        if _safe_smiles_to_graph_or_none(s) is None:
            rdkit_fail_idx.append(i)

    df2 = df1.drop(index=rdkit_fail_idx).reset_index(drop=True)
    print(
        f"[INFO] SMILES filter: removed "
        f"{int(mask_blank_or_digits.sum()) + len(rdkit_fail_idx)} rows "
        f"(blank/digits={int(mask_blank_or_digits.sum())}, "
        f"rdkit_fail={len(rdkit_fail_idx)}). Remain: {len(df2)}"
    )
    return df2


def warmup_drug_cache(
    all_rows: pd.DataFrame,
    colmap: Dict[str, str],
    drug_cache: Optional[Dict[str, Data]],
) -> Dict[str, Data]:
    if not isinstance(drug_cache, dict):
        drug_cache = {} if drug_cache is None else dict(drug_cache)
    for _, row in (
        all_rows[[colmap["drug"], colmap["smiles"]]].drop_duplicates().iterrows()
    ):
        drug_id = str(row[colmap["drug"]])
        smiles = str(row[colmap["smiles"]])
        if not any(
            k in drug_cache
            for k in (
                smiles,
                smiles.lower(),
                drug_id,
                drug_id.lower(),
            )
        ):
            g = _safe_smiles_to_graph_or_none(smiles)
            if g is None:
                continue
            drug_cache[smiles] = sanitize_pyg_data(g)
    return drug_cache


# ───────────────────────────────────────────────────────────────────────────────
# 12) Resume helper
# ───────────────────────────────────────────────────────────────────────────────
def load_checkpoint_and_resume(
    args: Args,
    model: DrugResponseTransformer,
    build_param_groups_fn,
    fresh_body_lr: float,
    fresh_head_lr: float,
):
    """
    A-setting: resume_from이 주어지면
    - ckpt에서 가중치 로드
    - saved_epoch 이후로 epochs_more 만큼 이어서 학습
    - body/head LR는 fresh_* 에 resume_lower_lr 적용한 값 사용
    """
    from math import cos, pi

    def _build_scheduler(optimizer, total_epochs, warmup_epochs):
        def lr_lambda(epoch):
            if epoch <= warmup_epochs:
                return (epoch) / max(1, warmup_epochs)
            t = (epoch - warmup_epochs) / max(
                1, (total_epochs - warmup_epochs)
            )
            return 0.1 + 0.9 * (1 + cos(pi * t)) / 2

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1
        )

    best_state = None
    start_epoch = 0

    if not args.resume_from:
        # 호출 안 하는 방향으로 main에서 처리하지만,
        # 안전하게 fresh optimizer도 만들어 둘 수는 있음.
        param_groups = build_param_groups_fn(
            body_lr=fresh_body_lr,
            head_lr=fresh_head_lr,
            weight_decay=args.weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = _build_scheduler(
            optimizer,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
        )
        return None, 0, optimizer, scheduler

    ck = torch.load(args.resume_from, map_location="cpu")
    saved_epoch = int(ck.get("epoch", 0))
    saved_state = ck.get("model_state_dict", ck)
    saved_args = ck.get("args", None)
    saved_cfg = ck.get("model_config", None)
    best_val_from_ckpt = ck.get("val_metrics", None)
    print(
        f"[RESUME] Loading checkpoint: {args.resume_from} "
        f"(epoch={saved_epoch})"
    )

    # Manual partial load: skip size-mismatched and missing keys
    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in saved_state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_sd, strict=True)
    missing = [k for k in model_sd if k not in saved_state]
    unexpected = [k for k in saved_state if k not in model_sd]
    print(f"[RESUME] Loaded {loaded} params, skipped {skipped} (shape mismatch or missing)")
    if not args.resume_strict:
        if missing:
            print(f"[RESUME] missing keys: {len(missing)}")
        if unexpected:
            print(f"[RESUME] unexpected keys: {len(unexpected)}")

    total_epochs = saved_epoch + max(0, int(args.epochs_more))
    if total_epochs <= saved_epoch:
        raise ValueError(
            f"[RESUME] epochs_more={args.epochs_more} → "
            f"total_epochs({total_epochs}) <= saved_epoch({saved_epoch})."
        )

    body_lr = fresh_body_lr * (0.5 if args.resume_lower_lr else 1.0)
    head_lr = fresh_head_lr * (0.5 if args.resume_lower_lr else 1.0)
    optimizer = torch.optim.AdamW(
        build_param_groups_fn(
            body_lr=body_lr,
            head_lr=head_lr,
            weight_decay=args.weight_decay,
        )
    )
    print(
        f"[RESUME] Resume with LR(body/head)={body_lr}/{head_lr}, "
        f"total_epochs={total_epochs}"
    )

    scheduler = _build_scheduler(
        optimizer,
        total_epochs=total_epochs,
        warmup_epochs=args.warmup_epochs,
    )
    scheduler.last_epoch = saved_epoch

    best_state = {
        "model_state_dict": saved_state,
        "args": saved_args if saved_args is not None else asdict(args),
        "model_config": saved_cfg if saved_cfg is not None else {},
        "epoch": saved_epoch,
        "val_metrics": best_val_from_ckpt
        if best_val_from_ckpt is not None
        else {},
    }
    start_epoch = saved_epoch
    print(
        f"[RESUME] Resumed from epoch {saved_epoch} → "
        f"training to epoch {total_epochs} (+= {args.epochs_more})"
    )
    return best_state, start_epoch, optimizer, scheduler


# ───────────────────────────────────────────────────────────────────────────────
# 13) Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    try:
        if mp.get_sharing_strategy() != "file_descriptor":
            mp.set_sharing_strategy("file_descriptor")
    except Exception as e:
        print(f"[WARN] set_sharing_strategy failed: {e}")

    parser = argparse.ArgumentParser()
    for field in Args.__dataclass_fields__.values():
        name = field.name
        default = field.default
        ftype = type(default) if default is not None else str
        if isinstance(default, bool):
            parser.add_argument(
                f"--{name}",
                action="store_true" if not default else "store_false",
            )
        else:
            parser.add_argument(
                f"--{name}", type=ftype, default=default
            )
    args_ns, _ = parser.parse_known_args()
    args = Args(**vars(args_ns))

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    # CSV
    df = pd.read_csv(args.csv_path, low_memory=False)
    colmap = build_colmap(df)

    # Label
    label_col = colmap["label"]
    print(f"[INFO] Using label column: {label_col}")
    if str(label_col).upper() == "IC50":
        df[label_col] = np.log(
            df[label_col].astype(float) + 1e-12
        )
        print(
            "[INFO] Applied log transform to IC50 -> using ln(IC50)."
        )
    print(
        "[INFO] Label stats: mean=",
        float(df[label_col].mean()),
        "std=",
        float(df[label_col].std()),
        "min=",
        float(df[label_col].min()),
        "max=",
        float(df[label_col].max()),
    )

    if args.drop_na_label:
        before = len(df)
        df = df[~df[colmap["label"]].isna()].reset_index(drop=True)
        print(
            f"[INFO] Dropped {before - len(df)} rows with NaN label. "
            f"Remain: {len(df)}"
        )

    # SMILES 정제
    df = filter_invalid_smiles(df, colmap)
    gene_order = [str(c) for c in colmap["genes"]]
    print(f"[INFO] #genes (BGE dim) = {len(gene_order)}")

    # split + subset
    train_df, val_df, test_df = split_dataframe(
        df,
        args.valid_ratio,
        args.test_ratio,
        args.split_seed,
        args.split_mode,
        colmap,
    )
    print(
        f"[INFO] Split -> train:{len(train_df)} "
        f"val:{len(val_df)} test:{len(test_df)}"
    )
    if args.subset_ratio < 0.999:
        train_df = subset_df(
            train_df, args.subset_ratio, args.split_seed
        )
        val_df = subset_df(
            val_df, args.subset_ratio, args.split_seed
        )
        test_df = subset_df(
            test_df, args.subset_ratio, args.split_seed
        )
        print(
            f"[INFO] After subset -> train:{len(train_df)} "
            f"val:{len(val_df)} test:{len(test_df)}"
        )

    if args.use_bge_branch:
        bge_cols = [str(c) for c in colmap["genes"]]
        bge_train = train_df[bge_cols].astype(np.float32)
        # inf → nan → 0
        bge_train = bge_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bge_mu = bge_train.mean(axis=0).to_numpy(dtype=np.float32)
        bge_sd = bge_train.std(axis=0).to_numpy(dtype=np.float32)
        print(f"[INFO] BGE mu/std computed (dim={len(bge_mu)})")
    else:
        bge_mu = None
        bge_sd = None

    # DRUG_FDIM
    DRUG_FDIM = None
    for s in df[colmap["smiles"]].dropna().astype(str).values[:2000]:
        g = _safe_smiles_to_graph_or_none(s)
        if (
            g is not None
            and hasattr(g, "x")
            and isinstance(g.x, torch.Tensor)
            and g.x.ndim == 2
        ):
            DRUG_FDIM = int(g.x.shape[1])
            break
    if DRUG_FDIM is None:
        cache_probe = _maybe_load_cache(
            args.drug_graph_cache, "drug_graph_cache"
        )
        if isinstance(cache_probe, dict):
            for v in cache_probe.values():
                if (
                    isinstance(v, Data)
                    and hasattr(v, "x")
                    and isinstance(v.x, torch.Tensor)
                    and v.x.ndim == 2
                ):
                    DRUG_FDIM = int(v.x.shape[1])
                    break
    if DRUG_FDIM is None:
        raise RuntimeError("Failed to infer DRUG_FDIM.")
    print(f"[INFO] DRUG_FDIM: {DRUG_FDIM}")

    # drug_graph_cache warmup
    drug_cache = _maybe_load_cache(
        args.drug_graph_cache, "drug_graph_cache"
    ) or {}
    all_rows = (
        pd.concat(
            [train_df, val_df, test_df],
            axis=0,
            ignore_index=True,
        )
        if args.cache_scope == "all"
        else train_df
    )
    drug_cache = warmup_drug_cache(all_rows, colmap, drug_cache)
    drug_cache = enforce_drug_feature_dim(drug_cache, DRUG_FDIM)
    if args.drug_graph_cache:
        try:
            os.makedirs(
                os.path.dirname(args.drug_graph_cache),
                exist_ok=True,
            )
            torch.save(drug_cache, args.drug_graph_cache)
            print(
                f"[INFO] drug_graph_cache saved -> "
                f"{args.drug_graph_cache}"
            )
        except Exception as e:
            print(
                f"[WARN] failed to save drug_graph_cache: {e}"
            )

    # cell_graph_cache 로드(노드 단위 학습에 사용 가능)
    cell_graph_cache = _maybe_load_cache(
        args.cell_graph_cache, "cell_graph_cache"
    )
    if cell_graph_cache is not None:
        cell_graph_cache = enforce_cell_feature_dim(
            cell_graph_cache, expected_dim=1
        )

        # sanitize 한 번만 수행
        sanitized = {}
        for cid, seq in cell_graph_cache.items():
            try:
                if not isinstance(seq, (list, tuple)) or len(seq) == 0:
                    continue
                graphs = [
                    g for g in seq if isinstance(g, Data)
                ]
                if not graphs:
                    continue
                sanitized[str(cid)] = sanitize_graph_list(
                    graphs
                )
            except Exception as e:
                print(
                    f"[WARN] drop bad cell '{cid}': {e}"
                )
        cell_graph_cache = {
            cid: seq
            for cid, seq in sanitized.items()
            if len(seq) > 0
        }
        print(
            f"[INFO] sanitized cell_graph_cache: "
            f"{len(cell_graph_cache)} cells"
        )
    elif args.use_cell_graph_cache:
        raise RuntimeError(
            "use_cell_graph_cache=True 인데 cell_graph_cache가 없습니다."
        )

    # 없는 셀 row 1회 패치 (train/val/test 전체에 대해 필터) — 그래프 캐시 사용 시에만
    if args.use_cell_graph_cache and cell_graph_cache is not None:
        valid_cells = set(cell_graph_cache.keys())

        def _filter_by_cell_cache(df_split, name: str):
            col = colmap["cell"]
            mask = df_split[col].astype(str).isin(
                valid_cells
            )
            dropped = int((~mask).sum())
            if dropped > 0:
                print(
                    f"[FILTER][{name}] drop {dropped} rows "
                    f"(cell_graph_cache miss)"
                )
            return df_split[mask].reset_index(
                drop=True
            )

        train_df = _filter_by_cell_cache(train_df, "train")
        val_df = _filter_by_cell_cache(val_df, "val")
        test_df = _filter_by_cell_cache(test_df, "test")
        print(
            f"[INFO] After cell_cache filter -> "
            f"train:{len(train_df)} "
            f"val:{len(val_df)} test:{len(test_df)}"
        )

    # ModelConfig / EncArgs
    mcfg = ModelConfig(
        dim_node=args.dim_node,
        out_drug=args.dim_drug,
        out_cell=args.dim_cell,
        pe_dim=args.pe_dim,
        num_pathways=args.num_pathways,
        transformer_heads=args.transformer_heads,
        ffn_dim=(
            args.ffn_dim
            if args.ffn_dim is not None
            else args.dim_node * 4
        ),
        transformer_layers=args.transformer_layers,
        dropout_ratio=args.dropout_ratio,
        max_num_nodes=args.max_num_nodes,
        freeze_encoders=args.freeze_encoders,   # A-setting: False
        last_n_layers=args.last_n_layers,
        unfreeze_pool_query=args.unfreeze_pool_query,
        unfreeze_time_dose=args.unfreeze_time_dose,
        unfreeze_type_embed=args.unfreeze_type_embed,
    )

    class _EncArgs:
        pass

    enc_args = _EncArgs()
    enc_args.num_feature_drug = DRUG_FDIM
    enc_args.dim_drug = args.dim_drug
    enc_args.num_feature_cell = 1
    enc_args.dim_cell = args.dim_cell
    enc_args.dim_node = args.dim_node
    enc_args.dropout_ratio = args.dropout_ratio
    enc_args.transformer_heads = args.transformer_heads
    enc_args.transformer_layers = args.transformer_layers
    enc_args.ffn_dim = mcfg.ffn_dim
    enc_args.max_num_nodes = args.max_num_nodes
    enc_args.pe_dim = args.pe_dim

    model = DrugResponseTransformer(
        args=enc_args,
        landmark_set=gene_order,
        config=mcfg,
        task="ic50",
    )
    model.setup_finetune(mcfg)
    device = torch.device(args.device)
    model.to(device)

    # freeze helper
    def set_requires_grad(module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    # A-setting 초기 freeze 정책:
    # - DrugEncoder: 학습 (requires_grad=True)
    # - CellEncoder: 끝까지 freeze (requires_grad=False)
    if hasattr(model, "DrugEncoder"):
        set_requires_grad(model.DrugEncoder, True)
    if hasattr(model, "CellEncoder"):
        set_requires_grad(model.CellEncoder, False)

    # param groups getter
    def build_param_groups(
        body_lr: float, head_lr: float, weight_decay: float
    ):
        return model.get_finetune_param_groups(
            body_lr=body_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
        )

    # Resume or Fresh Optim/Scheduler
    if args.resume_from:
        best_state, start_epoch, optimizer, scheduler = load_checkpoint_and_resume(
            args=args,
            model=model,
            build_param_groups_fn=build_param_groups,
            fresh_body_lr=args.body_lr,
            fresh_head_lr=args.head_lr,
        )
        # 이어학습이면 파일명에 resume epoch 반영
        total_epochs = start_epoch + args.epochs_more
    else:
        best_state = None
        start_epoch = 0
        optimizer = torch.optim.AdamW(
            build_param_groups(
                body_lr=args.body_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay,
            )
        )

        from math import cos, pi

        def lr_lambda(epoch):
            if epoch <= args.warmup_epochs:
                return (epoch) / max(1, args.warmup_epochs)
            t = (epoch - args.warmup_epochs) / max(
                1, (args.epochs - args.warmup_epochs)
            )
            return 0.1 + 0.9 * (1 + cos(pi * t)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1
        )
        total_epochs = args.epochs

    # collate & (필요시) fast-cache 준비
    use_graph = bool(args.use_cell_graph_cache)
    if use_graph:
        if cell_graph_cache is None:
            raise RuntimeError(
                "use_cell_graph_cache=True 이지만 cell_graph_cache가 없습니다."
            )
        collate_fn = make_collate_fn_with_cell_graph_cache(
            cell_graph_cache=cell_graph_cache,
            max_pathways=args.num_pathways,
        )
        cell_embed_cache = None
    else:
        all_cells = (
            df[colmap["cell"]]
            .astype(str)
            .unique()
            .tolist()
            if args.cache_scope == "all"
            else train_df[colmap["cell"]]
            .astype(str)
            .unique()
            .tolist()
        )
        cell_embed_cache = precompute_cell_embeds_or_load(
            args=args,
            model=model,
            all_cells=all_cells,
            cell_graph_cache=cell_graph_cache,
            df_all=df,
            colmap=colmap,
            device=device,
        )
        print(
            f"[INFO] cell_embed_cache ready: "
            f"{len(cell_embed_cache)} cells"
        )
        collate_fn = make_collate_fn_with_cell_embed_cache(
            cell_embed_cache
        )

    # Dataset/Loader
    ds_kwargs = dict(
        colmap=colmap,
        gene_order=gene_order,
        fixed_time=args.fixed_time,
        fixed_dose=args.fixed_dose,
        drug_cache=drug_cache,
        expected_drug_fdim=DRUG_FDIM,
        use_bge_branch=args.use_bge_branch,
        bge_mu=bge_mu,
        bge_sd=bge_sd,
    )
    tr_ds = IC50Dataset(train_df, **ds_kwargs)
    va_ds = IC50Dataset(val_df, **ds_kwargs)
    te_ds = IC50Dataset(test_df, **ds_kwargs)

    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.amp and device.type == "cuda")
    )

    # 저장 경로/파일명
    ts = time.strftime("%Y%m%d-%H%M%S")
    dev_idx = getattr(device, "index", None)
    dev_tag = (
        f"{device.type}{dev_idx}"
        if (device.type == "cuda" and dev_idx is not None)
        else device.type
    )
    ptag = f"p{args.num_pathways}"
    mode_tag = "graph" if use_graph else "fast"
    prefix = (
        args.run_name
        if args.run_name
        else f"ic50_{mode_tag}_{args.split_mode}_seed{args.seed}_{dev_tag}_{ptag}"
    )
    run_id = f"{prefix}_{ts}"
    model_save_path = os.path.join(
        args.save_dir, f"{run_id}.pth"
    )
    report_save_path = os.path.join(
        args.save_dir, f"{run_id}.report.json"
    )

    # Auto batch-size & accumulation
    auto_bs = args.batch_size
    accum_steps = 1
    if args.auto_batch_tune:
        try:
            start_bs = min(
                max(4, args.batch_size), args.max_probe_bs
            )
            auto_bs = start_bs
        except Exception as e:
            print(
                f"[BS-TUNER][WARN] auto-tune failed: {e} → "
                f"fallback to args.batch_size={args.batch_size}"
            )
            auto_bs = args.batch_size

        if (
            args.target_effective_bs
            and auto_bs < args.target_effective_bs
        ):
            import math

            accum_steps = math.ceil(
                args.target_effective_bs / auto_bs
            )
        print(
            f"[BS-TUNER] final batch_size={auto_bs}, "
            f"accum_steps={accum_steps}, "
            f"effective_bs≈{auto_bs*accum_steps}"
        )
    else:
        auto_bs = args.batch_size
        accum_steps = 1

    def build_loaders(bs):
        tr_loader = DataLoader(
            tr_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=(
                args.persistent_workers
                and args.num_workers > 0
            ),
        )
        va_loader = DataLoader(
            va_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=(
                args.persistent_workers
                and args.num_workers > 0
            ),
        )
        te_loader = DataLoader(
            te_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=(
                args.persistent_workers
                and args.num_workers > 0
            ),
        )
        return tr_loader, va_loader, te_loader

    tr_loader, va_loader, te_loader = build_loaders(auto_bs)

    # best_val 초기화 (resume 시 ckpt 값 있으면 가져올 수도 있음)
    if best_state is not None and isinstance(best_state.get("val_metrics", None), dict):
        best_val = dict(best_state["val_metrics"])
        # pcc가 없을 수도 있으니 기본값 보정
        best_val.setdefault("mse", float("inf"))
        best_val.setdefault("pcc", -1.0)
    else:
        best_val = {"mse": float("inf"), "pcc": -1.0}

    no_improve_epochs = 0  # early stopping 카운터

    # 학습 루프
    for epoch in range(start_epoch + 1, total_epochs + 1):
        # A-setting: staged_unfreeze는 사용하지 않으므로 epoch별 unfreeze 없음.
        alpha_corr_now = (
            args.alpha_corr if epoch >= 3 else 0.0
        )

        y_mu_train = (
            float(train_df[colmap["label"]].mean())
            if args.use_label_standardize
            else None
        )
        y_sd_train = (
            float(
                train_df[colmap["label"]].std()
                + 1e-8
            )
            if args.use_label_standardize
            else None
        )

        tr_loss = train_one_epoch(
            model,
            tr_loader,
            optimizer,
            device,
            scaler,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            accum_steps=accum_steps,
            y_mu=y_mu_train,
            y_sd=y_sd_train,
            use_label_standardize=args.use_label_standardize,
            use_corr_loss=args.use_corr_loss,
            alpha_corr=alpha_corr_now,
            use_smoothl1=args.use_smoothl1,
        )
        val_metrics = evaluate_epoch(
            model,
            va_loader,
            device,
            y_mu=y_mu_train,
            y_sd=y_sd_train,
            use_label_standardize=args.use_label_standardize,
        )
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={tr_loss:.4f} | "
            f"val_mse={val_metrics['mse']:.4f} | "
            f"val_pcc={val_metrics['pcc']:.4f}"
        )

        # ── Best 갱신 및 checkpoint 저장 (val_pcc 기준, min_delta 적용) ──
        if args.early_stopping:
            improved = val_metrics["pcc"] > best_val["pcc"] + args.early_stopping_min_delta
        else:
            improved = val_metrics["pcc"] > best_val["pcc"]

        if improved:
            best_val = val_metrics
            best_state = {
                "model_state_dict": model.state_dict(),
                "args": asdict(args),
                "model_config": asdict(mcfg),
                "epoch": epoch,
                "val_metrics": best_val,
            }
            torch.save(best_state, model_save_path)
            print(
                f"[SAVE] {model_save_path} "
                f"(val_pcc={best_val['pcc']:.4f})"
            )
            no_improve_epochs = 0
        else:
            if args.early_stopping:
                no_improve_epochs += 1

        # 스케줄러 업데이트
        if getattr(optimizer, "_step_count", 0) > 0:
            scheduler.step()

        # ── Early stopping 체크 ───────────────────────────────────────────────
        if args.early_stopping and no_improve_epochs >= args.early_stopping_patience:
            print(
                f"[EARLY STOP] val_pcc 개선 없음 (>{args.early_stopping_min_delta:g}) "
                f"{no_improve_epochs} epochs 연속 (patience={args.early_stopping_patience}). "
                f"best_val_pcc={best_val['pcc']:.4f} @ epoch {best_state['epoch']}"
            )
            break

    # best 로드 후 테스트
    if best_state is not None:
        model.load_state_dict(
            best_state["model_state_dict"]
        )

    y_mu_train = (
        float(train_df[colmap["label"]].mean())
        if args.use_label_standardize
        else None
    )
    y_sd_train = (
        float(train_df[colmap["label"]].std() + 1e-8)
        if args.use_label_standardize
        else None
    )
    test_metrics = evaluate_epoch(
        model,
        te_loader,
        device,
        y_mu=y_mu_train,
        y_sd=y_sd_train,
        use_label_standardize=args.use_label_standardize,
    )
    print(
        f"[TEST] mse={test_metrics['mse']:.4f} "
        f"pcc={test_metrics['pcc']:.4f}"
    )

    report = {
        "best_val": best_val,
        "test": test_metrics,
        "args": asdict(args),
        "model_path": model_save_path,
        "resumed_from": args.resume_from,
        "start_epoch": start_epoch,
        "total_epochs": total_epochs,
    }
    with open(report_save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"[INFO] Report saved to {report_save_path}"
    )


if __name__ == "__main__":
    main()