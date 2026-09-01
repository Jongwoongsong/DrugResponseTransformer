import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import os
import re
import json
import argparse
import time
import gc
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from torch.utils.data import Dataset, DataLoader
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
    from Model.DrugResponseTransformerFaithfulDedupDirectionAware import DrugResponseTransformer, ModelConfig
except Exception:
    from Model.DrugResponseTransformerFaithful import DrugResponseTransformer, ModelConfig

try:
    from Model.drug_graph import smiles_to_graph
except Exception:
    from drug_graph import smiles_to_graph

try:
    from Model.CellLine_graph import create_cell_line_graph
except Exception:
    from CellLine_graph import create_cell_line_graph


# ═══════════════════════════════════════════════════════════════════════════════
# LRU Cache for Cell Embeddings (Memory-safe)
# ═══════════════════════════════════════════════════════════════════════════════
class LRUCellCache:
    """
    LRU Cache for cell embeddings with:
    - Size limit to prevent OOM
    - Always stores detached tensors
    - Automatic eviction of least recently used items
    """
    def __init__(self, max_size: int = 100, device: str = "cpu"):
        self.max_size = max_size
        self.device = device
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: torch.Tensor):
        """Store detached tensor in cache"""
        # Always detach and move to CPU to save GPU memory
        detached = value.detach().cpu()

        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = detached
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest (first) item
                self.cache.popitem(last=False)
            self.cache[key] = detached

    def get_to_device(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        """Get and move to specified device"""
        val = self.get(key)
        if val is not None:
            return val.to(device, non_blocking=True)
        return None

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / max(1, total)
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
        }


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def _safe_smiles_to_graph_or_none(smiles: str) -> Optional[Data]:
    try:
        return smiles_to_graph(smiles)
    except Exception:
        return None


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

    out.num_nodes = int(getattr(g, "num_nodes", out.x.size(0)))
    n = out.num_nodes

    node_ids = None
    candidates = []
    if hasattr(g, "node_ids"):
        candidates.append("node_ids")
    for name in ["gene_ids", "entrez_ids", "node_id_list", "nodes", "ids"]:
        if hasattr(g, name):
            candidates.append(name)

    for name in candidates:
        try:
            v = getattr(g, name)
            if torch.is_tensor(v):
                v = v.detach().cpu().flatten().tolist()
            elif isinstance(v, np.ndarray):
                v = v.flatten().tolist()
            elif isinstance(v, (list, tuple)):
                v = list(v)
            else:
                continue
            if len(v) == n:
                node_ids = [str(x) for x in v]
                break
        except Exception:
            continue

    if node_ids is None:
        node_ids = [str(i) for i in range(n)]
    out.node_ids = node_ids

    if hasattr(g, "pathway_name"):
        try:
            out.pathway_name = str(getattr(g, "pathway_name"))
        except Exception:
            pass

    return out


def sanitize_graph_list(graphs: List[Data]) -> List[Data]:
    return [sanitize_pyg_data(g) for g in graphs]


def _norm_key(s: str) -> str:
    return str(s).strip().lower()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def filter_invalid_smiles(df: pd.DataFrame, smi_col: str) -> pd.DataFrame:
    vals = df[smi_col].astype(str).fillna("").str.strip()
    mask_blank_or_digits = (vals == "") | vals.str.fullmatch(r"\d+")
    df1 = df[~mask_blank_or_digits].copy()

    rdkit_fail_idx = []
    for i, s in df1[smi_col].astype(str).items():
        if _safe_smiles_to_graph_or_none(s) is None:
            rdkit_fail_idx.append(i)

    df2 = df1.drop(index=rdkit_fail_idx).reset_index(drop=True)
    print(
        f"[INFO] SMILES filter (PGE): removed "
        f"{int(mask_blank_or_digits.sum()) + len(rdkit_fail_idx)} rows "
        f"(blank/digits={int(mask_blank_or_digits.sum())}, rdkit_fail={len(rdkit_fail_idx)}). "
        f"Remain: {len(df2)}"
    )
    return df2


def build_drug_cache_from_scratch(
    df: pd.DataFrame,
    smiles_col: str,
) -> Tuple[Dict[str, Data], int]:
    uniq_smiles = sorted(set(df[smiles_col].astype(str)))
    cache: Dict[str, Data] = {}
    DRUG_FDIM = None
    n_ok, n_fail = 0, 0

    print(f"[DRUG_CACHE] building from scratch for {len(uniq_smiles)} unique SMILES")

    for i, s in enumerate(uniq_smiles, start=1):
        s_norm = str(s).strip()
        if s_norm == "" or s_norm.isdigit():
            n_fail += 1
            continue

        g = _safe_smiles_to_graph_or_none(s_norm)
        if g is None:
            n_fail += 1
            continue

        try:
            g = sanitize_pyg_data(g)
        except Exception:
            n_fail += 1
            continue

        if not (hasattr(g, "x") and isinstance(g.x, torch.Tensor) and g.x.ndim == 2):
            n_fail += 1
            continue

        fdim = int(g.x.shape[1])
        if DRUG_FDIM is None:
            DRUG_FDIM = fdim
        elif fdim != DRUG_FDIM:
            n_fail += 1
            continue

        key_norm = _norm_key(s_norm)
        cache[key_norm] = g
        cache[s_norm] = g
        n_ok += 1

        if i % 1000 == 0:
            print(
                f"[DRUG_CACHE] {i}/{len(uniq_smiles)} processed "
                f"| ok={n_ok} fail={n_fail} cache_keys={len(cache)}"
            )

    if DRUG_FDIM is None or n_ok == 0:
        raise RuntimeError(
            f"[DRUG_CACHE] failed to build any valid drug graphs "
            f"(ok={n_ok}, fail={n_fail})"
        )

    print(
        f"[DRUG_CACHE] done. unique_ok={n_ok}, fail={n_fail}, "
        f"DRUG_FDIM={DRUG_FDIM}, cache_keys={len(cache)}"
    )
    return cache, DRUG_FDIM


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Args:
    mode: str = "train"
    # Files
    perturbed_csv: str = "pretraining_label_landmark268_smiles.csv"
    basal_csv: str = "basal_final.csv"
    kegg_pathway_dir: str = "pathwayxml"
    landmark_csv: str = "kegg_lincs_landmark_overlap.csv"

    # caches
    cache_dir: str = "graph_cache"
    drug_graph_cache: str = "graph_cache/drug_graph_cache.pt"
    cell_graph_cache: Optional[str] = "graph_cache/cell_graph_cache_union.pt"
    cache_scope: str = "all"

    # Save
    save_dir: str = "checkpoints_pge_pretrain"
    run_name: Optional[str] = None

    # Hardware
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_gpu: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False

    # Data split
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    split_mode: str = "mixed"
    subset_ratio: float = 1.0

    # Training
    epochs: int = 20
    batch_size: int = 48
    grad_clip: float = 1.0
    amp: bool = True
    log_interval: int = 200
    warmup_epochs: int = 2
    accum_steps: int = 1

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
    num_pathways: int = 31
    # PATCH_20260720_EXPOSE_PATHWAY_BATCHING
    use_pathway_batching: bool = True

    # Fine-tune policy
    freeze_encoders: bool = False
    last_n_layers: int = 2
    unfreeze_pool_query: bool = True
    unfreeze_time_dose: bool = True
    unfreeze_type_embed: bool = True

    # Misc
    seed: int = 42
    zscore_by_gene: bool = True
    clamp_label: Optional[float] = 10.0

    # Resume
    resume_from: Optional[str] = None
    epochs_more: int = 15
    resume_strict: bool = False
    resume_lower_lr: bool = True

    # Bad sample handling
    on_bad_cell: str = "error"
    on_bad_drug: str = "error"

    # Diagnostics
    first_batch_probe: bool = False

    # pathway sampling
    num_pathways_sample: int = 0
    sample_strategy: str = "random"

    # Speed options
    use_batch_dedup: bool = False
    epoch_embed_cache: bool = False
    refresh_every_k_steps: int = 0
    precomputed_cell_embed: Optional[str] = None
    try_torch_compile: bool = False
    precompute_cell_embed_only: bool = False

    # ═══ NEW: Memory optimization options ═══
    live_cell_update_freq: int = 500
    cell_cache_max_size: int = 100      # LRU cache size limit
    empty_cache_freq: int = 100         # torch.cuda.empty_cache() frequency
    gradient_checkpointing: bool = False  # Enable gradient checkpointing


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_landmarks(landmark_csv: str) -> List[str]:
    lf = pd.read_csv(landmark_csv)
    cand = [c for c in lf.columns if re.fullmatch(r"\d+", str(c))] or \
           [c for c in lf.columns if str(c).lower() in ("entrez_id", "entrez", "gene", "gene_id")]
    if len(cand) == 0:
        if lf.shape[1] == 1:
            genes = lf.iloc[:, 0].astype(str).tolist()
        else:
            raise ValueError("LANDMARK_FILE has no gene column")
    else:
        genes = lf[cand[0]].astype(str).tolist()
    genes = [g for g in genes if re.fullmatch(r"\d+", g)]
    genes = sorted(set(genes))
    if len(genes) == 0:
        raise ValueError("Empty landmark list")
    return genes


def build_colmap_perturb(df: pd.DataFrame, landmark_genes: List[str],
                         override: Dict[str, Optional[str]] = None) -> Dict[str, object]:
    override = override or {}
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    def pick(primary: List[str], rx: List[str] = None, key: str = None):
        if key and override.get(key) and override[key] in df.columns:
            return override[key]
        for cand in primary:
            if cand in df.columns:
                return cand
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        if rx:
            import re as _re
            for c in cols:
                lc = c.lower()
                if any(_re.search(r, lc) for r in rx):
                    return c
        return None

    cell_col = pick(
        ["cell_iname", "CELL_LINE_NAME", "cell_line_name", "cell_id", "cell", "cell_line", "line", "cl_id"],
        [r"\bcell\b", r"cell.*name", r"\bcell[_ ]?id\b"], key="cell"
    )
    drug_col = pick(
        ["DRUG_NAME", "drug_name", "drug", "compound", "pert_iname", "treatment", "agent"],
        [r"\bdrug\b", r"pert.*name", r"\bcompound\b", r"\bagent\b", r"treat"], key="drug"
    )
    smi_col = pick(
        ["canonical_smiles", "smiles", "SMILES"],
        [r"smiles"], key="smiles"
    )
    time_col = pick(
        ["pert_itime", "time", "TIME_H", "time_h", "hour", "HOUR"],
        [r"\btime\b", r"hour"], key="time"
    )
    dose_col = pick(
        ["converted_dose", "dose", "DOSE", "conc", "CONC", "dose_um", "dose_uM", "concentration"],
        [r"dose", r"conc"], key="dose"
    )

    if cell_col is None:
        raise ValueError(f"Missing CELL column. Got: {cols[:30]}")
    if smi_col is None:
        raise ValueError(f"Missing SMILES column. Got: {cols[:30]}")
    if drug_col is None:
        drug_col = smi_col

    gene_set = set(map(str, landmark_genes))
    gene_cols = [c for c in df.columns if str(c) in gene_set]
    if len(gene_cols) == 0:
        gene_cols = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]
    if len(gene_cols) == 0:
        raise ValueError("No landmark gene columns found in perturbed CSV")

    return {
        "cell": cell_col,
        "drug": drug_col,
        "smiles": smi_col,
        "time": time_col,
        "dose": dose_col,
        "genes": [str(c) for c in gene_cols],
    }


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
class PGEDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 colmap: Dict[str, object],
                 landmark_order: List[str],
                 drug_cache: Dict[str, Data],
                 expected_drug_fdim: int,
                 cell_graph_cache: Dict[str, List[Data]],
                 max_pathways: int = 31,
                 on_bad_cell: str = "error",
                 on_bad_drug: str = "error",
                 num_pathways_sample: int = 0,
                 sample_strategy: str = "random",
                 rng: Optional[np.random.RandomState] = None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.colmap = colmap
        self.landmark_order = [str(g) for g in landmark_order]
        self.expected_drug_fdim = expected_drug_fdim
        self.drug_cache = drug_cache
        self.cell_cache = cell_graph_cache
        self.max_pathways = max_pathways
        self.on_bad_cell = on_bad_cell
        self.on_bad_drug = on_bad_drug
        self.num_pathways_sample = int(num_pathways_sample) if num_pathways_sample else 0
        self.sample_strategy = str(sample_strategy).lower().strip()
        self.rng = rng or np.random.RandomState(1234)

    def __len__(self):
        return len(self.df)

    def _get_drug_graph(self, smiles: str, drug_id: str) -> Optional[Data]:
        key_s = _norm_key(smiles)
        key_d = _norm_key(drug_id)
        for k in (key_s, key_d, smiles, drug_id):
            v = self.drug_cache.get(k, None)
            if isinstance(v, Data):
                g = v
                if (
                    self.expected_drug_fdim is not None
                    and hasattr(g, "x")
                    and isinstance(g.x, torch.Tensor)
                    and g.x.ndim == 2
                    and int(g.x.shape[1]) != int(self.expected_drug_fdim)
                ):
                    return None
                return g
        return None

    def _get_cell_graph_seq_full(self, cell_id: str) -> Optional[List[Data]]:
        if self.cell_cache is None or cell_id not in self.cell_cache:
            return None
        seq = self.cell_cache[cell_id]
        if self.max_pathways and self.max_pathways > 0:
            seq = seq[: self.max_pathways]
        return seq

    def _maybe_sample_pathways(self, seq: List[Data]) -> List[Data]:
        if not isinstance(seq, list) or len(seq) == 0:
            return seq
        if self.num_pathways_sample and self.num_pathways_sample > 0 and len(seq) > self.num_pathways_sample:
            k = self.num_pathways_sample
            if self.sample_strategy == "random":
                idx = self.rng.choice(len(seq), size=k, replace=False)
                return [seq[i] for i in idx.tolist()]
            elif self.sample_strategy == "head":
                return seq[:k]
            elif self.sample_strategy == "tail":
                return seq[-k:]
        return seq

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        smiles = str(row[self.colmap["smiles"]])
        drug_id = str(row[self.colmap["drug"]])
        cell_id = str(row[self.colmap["cell"]])

        g = self._get_drug_graph(smiles, drug_id)
        if g is None:
            msg = f"[BAD_DRUG] cache miss or fdim mismatch for drug_id={drug_id}, smiles={smiles}"
            if self.on_bad_drug == "skip":
                return {"_invalid": True, "reason": msg}
            raise RuntimeError(msg)

        cell_seq_full = self._get_cell_graph_seq_full(cell_id)
        if cell_seq_full is None:
            msg = f"[BAD_CELL] cell_graph_cache MISS for cell_id={cell_id}"
            if self.on_bad_cell == "skip":
                return {"_invalid": True, "reason": msg}
            raise KeyError(msg)

        cell_seq = self._maybe_sample_pathways(cell_seq_full)
        bad = (
            not isinstance(cell_seq, list)
            or len(cell_seq) == 0
            or any((cg is None) for cg in cell_seq)
        )
        if bad:
            msg = f"[BAD_CELL] invalid cell_seq for cell_id={cell_id}"
            if self.on_bad_cell == "skip":
                return {"_invalid": True, "reason": msg}
            raise ValueError(msg)

        if self.colmap["time"] is not None:
            t = torch.tensor(float(row[self.colmap["time"]]), dtype=torch.float32)
        else:
            t = torch.tensor(0.0, dtype=torch.float32)
        if self.colmap["dose"] is not None:
            d = torch.tensor(float(row[self.colmap["dose"]]), dtype=torch.float32)
        else:
            d = torch.tensor(0.0, dtype=torch.float32)

        y = torch.tensor(
            [float(row.get(gid, np.nan)) for gid in self.landmark_order],
            dtype=torch.float32,
        )

        return {
            "drug_graph": g,
            "cell_seq": cell_seq,
            "time": t,
            "dose": d,
            "y": y,
            "cell_id": cell_id,
            "drug_id": drug_id,
            "_invalid": False,
        }


def pge_collate(batch: List[Dict]):
    valid = [b for b in batch if not b.get("_invalid", False)]
    if len(valid) == 0:
        raise ValueError("All samples invalid in batch. Check on_bad_cell/on_bad_drug policy.")
    drug_batch = Batch.from_data_list([b["drug_graph"] for b in valid])
    cell_seq_batch = [b["cell_seq"] for b in valid]
    time_t = torch.stack([b["time"] for b in valid], 0)
    dose_t = torch.stack([b["dose"] for b in valid], 0)
    y_t = torch.stack([b["y"] for b in valid], 0)
    meta = {
        "cell_id": [b["cell_id"] for b in valid],
        "drug_id": [b["drug_id"] for b in valid],
    }
    return {
        "drug_graph": drug_batch,
        "cell_seq": cell_seq_batch,
        "time": time_t,
        "dose": dose_t,
        "y": y_t,
        "meta": meta,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics / Loss
# ═══════════════════════════════════════════════════════════════════════════════
def pcc_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu().view(-1)
    y = y.detach().cpu().view(-1)
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float((x * y).mean().item())


def _masked_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(y) & torch.isfinite(pred)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    diff2 = (pred[mask] - y[mask]) ** 2
    return diff2.mean()


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
class ModelAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model

        self.has_split = all([
            hasattr(unwrap_model(model), name)
            for name in ["encode_drug", "encode_cell_seq", "fuse_and_predict"]
        ])

        core = unwrap_model(model)
        self.has_modules = hasattr(core, "drug_encoder") and hasattr(core, "cell_encoder")

    @torch.no_grad()
    def try_compile(self):
        try:
            core = unwrap_model(self.m)
            if hasattr(core, "drug_encoder"):
                core.drug_encoder = torch.compile(core.drug_encoder)
            if hasattr(core, "cell_encoder"):
                core.cell_encoder = torch.compile(core.cell_encoder)
            print("[INFO] torch.compile applied to encoders")
        except Exception as e:
            print(f"[WARN] torch.compile skipped: {e}")

    def encode_drug(self, drug_batch):
        core = unwrap_model(self.m)
        if self.has_split:
            return core.encode_drug(drug_batch)
        if self.has_modules and hasattr(core.drug_encoder, "encode"):
            return core.drug_encoder.encode(drug_batch)
        if self.has_modules and callable(getattr(core.drug_encoder, "forward", None)):
            return core.drug_encoder(drug_batch)
        return None

    def encode_cell_seq(self, cell_seq: List[Data]):
        core = unwrap_model(self.m)
        if self.has_split:
            return core.encode_cell_seq(cell_seq)
        if self.has_modules and hasattr(core.cell_encoder, "encode_seq"):
            return core.cell_encoder.encode_seq(cell_seq)
        if self.has_modules and callable(getattr(core.cell_encoder, "forward", None)):
            return core.cell_encoder(cell_seq)
        return None

    def fuse_and_predict(self, drug_repr, cell_repr_list, time, dose, return_attn=False):
        core = unwrap_model(self.m)
        if self.has_split:
            return core.fuse_and_predict(
                drug_repr=drug_repr,
                cell_repr_list=cell_repr_list,
                time=time,
                dose=dose,
                return_attn=return_attn,
            )
        return core(
            drug_graph=drug_repr if isinstance(drug_repr, Batch) else drug_repr,
            cell_graph_seq=cell_repr_list if isinstance(cell_repr_list, list) else cell_repr_list,
            time=time,
            dose=dose,
            return_attn=return_attn,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ★★★ FIXED: Train function with memory optimization ★★★
# ═══════════════════════════════════════════════════════════════════════════════
# PATCH_20260720_PGE_BENCHMARK_INSTRUMENTATION
LAST_TRAIN_AUDIT = {}


def train_one_epoch_pge(
    model_adapter: ModelAdapter,
    loader,
    optimizer,
    device,
    scaler=None,
    log_interval: int = 200,
    accum_steps: int = 1,
    zscore_by_gene: bool = True,
    y_mu=None,
    y_sd=None,
    grad_clip: float = 1.0,
    use_batch_dedup: bool = True,
    epoch_cache: Optional[dict] = None,
    refresh_every_k_steps: int = 0,
    precomputed_cell: Optional[Dict[str, torch.Tensor]] = None,
    cell_graph_cache: Optional[Dict[str, List[Data]]] = None,
    live_cell_update_freq: int = 500,
    # ═══ NEW: Memory optimization args ═══
    cell_lru_cache: Optional[LRUCellCache] = None,
    empty_cache_freq: int = 100,
):
    """
    Memory-optimized training loop.

    Key fixes:
    1. Cell embeddings are stored detached in LRU cache
    2. Periodic torch.cuda.empty_cache() calls
    3. Clear separation between gradient-flow tensors and cached tensors
    """
    global LAST_TRAIN_AUDIT

    if int(live_cell_update_freq) <= 0:
        raise ValueError(
            "live_cell_update_freq must be positive"
        )

    model = model_adapter.m
    model.train()

    audit_start_time = time.time()

    if (
        torch.cuda.is_available()
        and device.type == "cuda"
    ):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    core_for_audit = unwrap_model(model)

    cell_module_for_audit = getattr(
        core_for_audit,
        "CellEncoder",
        None,
    )

    cell_initial_state = {}
    cell_encoder_trainable_parameters = 0

    if isinstance(
        cell_module_for_audit,
        nn.Module,
    ):
        for name, parameter in (
            cell_module_for_audit.named_parameters()
        ):
            if not parameter.requires_grad:
                continue

            cell_encoder_trainable_parameters += int(
                parameter.numel()
            )

            cell_initial_state[name] = (
                parameter.detach()
                .cpu()
                .float()
                .clone()
            )

    def _cell_gradient_l2():
        if not isinstance(
            cell_module_for_audit,
            nn.Module,
        ):
            return 0.0

        total_squared = 0.0

        for parameter in (
            cell_module_for_audit.parameters()
        ):
            if (
                not parameter.requires_grad
                or parameter.grad is None
            ):
                continue

            gradient = (
                parameter.grad.detach().float()
            )

            if not torch.isfinite(
                gradient
            ).all():
                return float("nan")

            total_squared += float(
                torch.sum(
                    gradient * gradient
                ).item()
            )

        return total_squared ** 0.5

    running_examples, running_loss_sum = 0, 0.0
    total_loss_sum, total_examples = 0.0, 0
    last_log_t = time.time()

    train_step_count = 0
    live_step_count = 0
    cached_step_count = 0
    live_sample_count = 0
    cached_sample_count = 0
    live_unique_cell_encodes = 0

    optimizer_step_count = 0
    cell_gradient_optimizer_steps = 0
    cell_gradient_norms = []
    faithful_grad_audit_printed = False

    observed_batch_sizes = []

    # Use LRU cache if provided, else create a simple dict (but detached!)
    # PATCH_20260720_EXACT_SAFE_OPTIMIZATIONS_V2
    # Exact joint training never consumes detached cell embeddings.
    # Disable LRU traffic entirely when every step is live.
    use_lru = (
        cell_lru_cache is not None
        and int(live_cell_update_freq) > 1
    )

    step_global = 0

    for step, batch in enumerate(loader, 1):
        step_global += 1

        drug_graph = batch["drug_graph"].to(device, non_blocking=True)
        cell_seq = batch["cell_seq"]
        time_ = batch["time"].to(device, non_blocking=True)
        dose = batch["dose"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        cell_ids = batch["meta"]["cell_id"]

        if (step - 1) % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        can_split = hasattr(unwrap_model(model), "encode_drug") or model_adapter.has_modules

        def _loss_fn(pred, y_):
            if zscore_by_gene and (y_mu is not None) and (y_sd is not None):
                pred_n = (pred - y_mu) / y_sd
                y_n = (y_ - y_mu) / y_sd
                return _masked_mse(pred_n, y_n)
            return _masked_mse(pred, y_)


        is_live_update_step = (
            step_global
            % int(live_cell_update_freq)
            == 0
        ) or (
            step_global == 1
        )

        current_batch_size = int(
            y.shape[0]
        )

        unique_cells_this_batch = len(
            set(
                str(cell_id)
                for cell_id in cell_ids
            )
        )

        train_step_count += 1
        observed_batch_sizes.append(
            current_batch_size
        )

        if is_live_update_step:
            live_step_count += 1
            live_sample_count += (
                current_batch_size
            )
            live_unique_cell_encodes += (
                unique_cells_this_batch
            )
        else:
            cached_step_count += 1
            cached_sample_count += (
                current_batch_size
            )

        if scaler is not None and scaler.is_enabled():
            with torch.cuda.amp.autocast():
                if can_split and use_batch_dedup:

                    drug_repr = model_adapter.encode_drug(drug_graph)

                    # ═══ FIXED: Cell encoding with proper gradient handling ═══
                    cell_repr_list: List[torch.Tensor] = [None] * len(cell_seq)
                    uniq_cells: Dict[str, int] = {}
                    for i, cid in enumerate(cell_ids):
                        if cid not in uniq_cells:
                            uniq_cells[cid] = i

                    if is_live_update_step:


                        batch_cell_enc: Dict[str, torch.Tensor] = {}

                        for cid in uniq_cells.keys():
                            idx0 = uniq_cells[cid]
                            cseq = cell_seq[idx0]
                            enc = model_adapter.encode_cell_seq(cseq)


                            batch_cell_enc[cid] = enc


                            if use_lru:
                                cell_lru_cache.put(cid, enc)


                        for i, cid in enumerate(cell_ids):
                            cell_repr_list[i] = batch_cell_enc[cid]


                        del batch_cell_enc

                    else:

                        for i, cid in enumerate(cell_ids):
                            cached = None
                            if use_lru:
                                cached = cell_lru_cache.get_to_device(cid, device)

                            if cached is not None:
                                cell_repr_list[i] = cached
                            else:

                                idx0 = uniq_cells.get(cid, 0)
                                cseq = cell_seq[idx0] if idx0 < len(cell_seq) else cell_seq[0]
                                enc = model_adapter.encode_cell_seq(cseq)


                                if use_lru:
                                    cell_lru_cache.put(cid, enc)


                                cell_repr_list[i] = enc.detach()

                    pred = model_adapter.fuse_and_predict(
                        drug_repr=drug_repr,
                        cell_repr_list=cell_repr_list,
                        time=time_,
                        dose=dose,
                        return_attn=False,
                    )
                else:
                    pred = model(
                        drug_graph=drug_graph,
                        cell_graph_seq=cell_seq,
                        cell_ids=cell_ids,
                        time=time_,
                        dose=dose,
                        return_attn=False,
                    )
                loss = _loss_fn(pred, y) / max(1, accum_steps)
            scaler.scale(loss).backward()
        else:

            if can_split and use_batch_dedup:
                drug_repr = model_adapter.encode_drug(drug_graph)

                cell_repr_list: List[torch.Tensor] = [None] * len(cell_seq)
                uniq_cells: Dict[str, int] = {}
                for i, cid in enumerate(cell_ids):
                    if cid not in uniq_cells:
                        uniq_cells[cid] = i

                if is_live_update_step:
                    batch_cell_enc: Dict[str, torch.Tensor] = {}

                    for cid in uniq_cells.keys():
                        idx0 = uniq_cells[cid]
                        cseq = cell_seq[idx0]
                        enc = model_adapter.encode_cell_seq(cseq)
                        batch_cell_enc[cid] = enc

                        if use_lru:
                            cell_lru_cache.put(cid, enc)

                    for i, cid in enumerate(cell_ids):
                        cell_repr_list[i] = batch_cell_enc[cid]

                    del batch_cell_enc
                else:
                    for i, cid in enumerate(cell_ids):
                        cached = None
                        if use_lru:
                            cached = cell_lru_cache.get_to_device(cid, device)

                        if cached is not None:
                            cell_repr_list[i] = cached
                        else:
                            idx0 = uniq_cells.get(cid, 0)
                            cseq = cell_seq[idx0] if idx0 < len(cell_seq) else cell_seq[0]
                            enc = model_adapter.encode_cell_seq(cseq)

                            if use_lru:
                                cell_lru_cache.put(cid, enc)

                            cell_repr_list[i] = enc.detach()

                pred = model_adapter.fuse_and_predict(
                    drug_repr=drug_repr,
                    cell_repr_list=cell_repr_list,
                    time=time_,
                    dose=dose,
                    return_attn=False,
                )
            else:
                pred = model(
                    drug_graph=drug_graph,
                    cell_graph_seq=cell_seq,
                    cell_ids=cell_ids,
                    time=time_,
                    dose=dose,
                    return_attn=False,
                )
            loss = _loss_fn(pred, y) / max(1, accum_steps)
            loss.backward()

        # Gradient step
        if (
            step % accum_steps == 0
        ) or (
            step == len(loader)
        ):
            optimizer_step_count += 1

            if (
                scaler is not None
                and scaler.is_enabled()
            ):
                # Required before inspecting or clipping
                # AMP-scaled gradients.
                scaler.unscale_(optimizer)

                cell_gradient_norm = (
                    _cell_gradient_l2()
                )

                cell_gradient_norms.append(
                    cell_gradient_norm
                )

                if (
                    not faithful_grad_audit_printed
                    and os.environ.get(
                        "DRT_FAITHFUL_GRAD_AUDIT", "0"
                    ) == "1"
                ):
                    core_grad = unwrap_model(model)

                    def module_grad_l2(module):
                        if module is None:
                            return None
                        total = 0.0
                        found = False
                        for parameter in module.parameters():
                            if parameter.grad is None:
                                continue
                            grad = parameter.grad.detach().float()
                            total += float((grad * grad).sum().item())
                            found = True
                        return total ** 0.5 if found else 0.0

                    grad_payload = {
                        "DrugEncoder": module_grad_l2(
                            getattr(core_grad, "DrugEncoder", None)
                        ),
                        "CellEncoder": module_grad_l2(
                            getattr(core_grad, "CellEncoder", None)
                        ),
                        "direction_embedding": module_grad_l2(
                            getattr(
                                getattr(core_grad, "CellEncoder", None),
                                "direction_embedding",
                                None,
                            )
                        ),
                        "Transformer": module_grad_l2(
                            getattr(core_grad, "Transformer", None)
                        ),
                        "dose_proj": module_grad_l2(
                            getattr(core_grad, "dose_proj", None)
                        ),
                        "time_proj": module_grad_l2(
                            getattr(core_grad, "time_proj", None)
                        ),
                        "occurrence_score": module_grad_l2(
                            getattr(core_grad, "occurrence_score", None)
                        ),
                        "PGE_regressor": module_grad_l2(
                            getattr(core_grad, "regressor", None)
                        ),
                    }
                    print(
                        "[FAITHFUL GRADIENT AUDIT] "
                        + json.dumps(grad_payload, sort_keys=True),
                        flush=True,
                    )
                    faithful_grad_audit_printed = True

                if (
                    np.isfinite(
                        cell_gradient_norm
                    )
                    and cell_gradient_norm > 0.0
                ):
                    cell_gradient_optimizer_steps += 1

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )

                scaler.step(optimizer)
                scaler.update()

            else:
                cell_gradient_norm = (
                    _cell_gradient_l2()
                )

                cell_gradient_norms.append(
                    cell_gradient_norm
                )

                if (
                    not faithful_grad_audit_printed
                    and os.environ.get(
                        "DRT_FAITHFUL_GRAD_AUDIT", "0"
                    ) == "1"
                ):
                    core_grad = unwrap_model(model)

                    def module_grad_l2(module):
                        if module is None:
                            return None
                        total = 0.0
                        found = False
                        for parameter in module.parameters():
                            if parameter.grad is None:
                                continue
                            grad = parameter.grad.detach().float()
                            total += float((grad * grad).sum().item())
                            found = True
                        return total ** 0.5 if found else 0.0

                    grad_payload = {
                        "DrugEncoder": module_grad_l2(
                            getattr(core_grad, "DrugEncoder", None)
                        ),
                        "CellEncoder": module_grad_l2(
                            getattr(core_grad, "CellEncoder", None)
                        ),
                        "Transformer": module_grad_l2(
                            getattr(core_grad, "Transformer", None)
                        ),
                        "dose_proj": module_grad_l2(
                            getattr(core_grad, "dose_proj", None)
                        ),
                        "time_proj": module_grad_l2(
                            getattr(core_grad, "time_proj", None)
                        ),
                        "occurrence_score": module_grad_l2(
                            getattr(core_grad, "occurrence_score", None)
                        ),
                        "PGE_regressor": module_grad_l2(
                            getattr(core_grad, "regressor", None)
                        ),
                    }
                    print(
                        "[FAITHFUL GRADIENT AUDIT] "
                        + json.dumps(grad_payload, sort_keys=True),
                        flush=True,
                    )
                    faithful_grad_audit_printed = True

                if (
                    np.isfinite(
                        cell_gradient_norm
                    )
                    and cell_gradient_norm > 0.0
                ):
                    cell_gradient_optimizer_steps += 1

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )

                optimizer.step()

        # ★ Periodic memory cleanup
        if empty_cache_freq > 0 and step % empty_cache_freq == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        bs = y.shape[0]
        val = float(loss.item()) * bs * max(1, accum_steps)
        total_loss_sum += val
        total_examples += bs
        running_loss_sum += val
        running_examples += bs

        if log_interval and (step % log_interval == 0):
            now = time.time()
            dt = max(1e-6, now - last_log_t)
            sps = running_examples / dt

            live_tag = "[LIVE]" if is_live_update_step else "[CACHED]"
            cache_info = cell_lru_cache.stats() if use_lru else {"size": "N/A"}


            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                mem_str = f"GPU: {mem_alloc:.1f}GB/{mem_reserved:.1f}GB"
            else:
                mem_str = "CPU"

            print(
                f"  [step {step}] {live_tag} avg_loss={running_loss_sum/max(1,running_examples):.4f} "
                f"| {sps:.1f} samples/s | cache={cache_info['size']} | {mem_str}"
            )
            running_examples, running_loss_sum = 0, 0.0
            last_log_t = now

    if (
        torch.cuda.is_available()
        and device.type == "cuda"
    ):
        torch.cuda.synchronize(device)

    train_elapsed_seconds = float(
        time.time() - audit_start_time
    )

    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    cuda_device_name = None

    if (
        torch.cuda.is_available()
        and device.type == "cuda"
    ):
        peak_allocated_bytes = int(
            torch.cuda.max_memory_allocated(
                device
            )
        )

        peak_reserved_bytes = int(
            torch.cuda.max_memory_reserved(
                device
            )
        )

        cuda_device_name = (
            torch.cuda.get_device_name(
                device
            )
        )

    cell_parameter_delta_squared = 0.0
    cell_parameter_initial_squared = 0.0

    if isinstance(
        cell_module_for_audit,
        nn.Module,
    ):
        for name, parameter in (
            cell_module_for_audit.named_parameters()
        ):
            if name not in cell_initial_state:
                continue

            current_value = (
                parameter.detach()
                .cpu()
                .float()
            )

            initial_value = (
                cell_initial_state[name]
            )

            difference = (
                current_value
                - initial_value
            )

            cell_parameter_delta_squared += float(
                torch.sum(
                    difference * difference
                ).item()
            )

            cell_parameter_initial_squared += float(
                torch.sum(
                    initial_value
                    * initial_value
                ).item()
            )

    cell_parameter_delta_l2 = (
        cell_parameter_delta_squared ** 0.5
    )

    cell_parameter_relative_delta = (
        cell_parameter_delta_l2
        / max(
            cell_parameter_initial_squared ** 0.5,
            1e-12,
        )
    )

    finite_gradient_norms = [
        float(value)
        for value in cell_gradient_norms
        if np.isfinite(value)
    ]

    cache_stats_for_audit = (
        cell_lru_cache.stats()
        if use_lru
        else {}
    )

    LAST_TRAIN_AUDIT = {
        "train_examples": int(
            total_examples
        ),
        "train_steps": int(
            train_step_count
        ),
        "optimizer_steps": int(
            optimizer_step_count
        ),
        "live_cell_update_freq": int(
            live_cell_update_freq
        ),
        "live_steps": int(
            live_step_count
        ),
        "cached_steps": int(
            cached_step_count
        ),
        "live_step_fraction": float(
            live_step_count
            / max(1, train_step_count)
        ),
        "live_samples": int(
            live_sample_count
        ),
        "cached_samples": int(
            cached_sample_count
        ),
        "live_sample_fraction": float(
            live_sample_count
            / max(1, total_examples)
        ),
        "live_unique_cell_encodes": int(
            live_unique_cell_encodes
        ),
        "optimizer_steps_with_cell_gradient": int(
            cell_gradient_optimizer_steps
        ),
        "cell_gradient_optimizer_step_fraction": float(
            cell_gradient_optimizer_steps
            / max(1, optimizer_step_count)
        ),
        "cell_gradient_l2_mean": (
            float(
                np.mean(
                    finite_gradient_norms
                )
            )
            if finite_gradient_norms
            else 0.0
        ),
        "cell_gradient_l2_max": (
            float(
                np.max(
                    finite_gradient_norms
                )
            )
            if finite_gradient_norms
            else 0.0
        ),
        "cell_encoder_trainable_parameters": int(
            cell_encoder_trainable_parameters
        ),
        "cell_encoder_parameter_delta_l2": float(
            cell_parameter_delta_l2
        ),
        "cell_encoder_parameter_relative_delta": float(
            cell_parameter_relative_delta
        ),
        "minimum_batch_size": (
            int(min(observed_batch_sizes))
            if observed_batch_sizes
            else 0
        ),
        "maximum_batch_size": (
            int(max(observed_batch_sizes))
            if observed_batch_sizes
            else 0
        ),
        "train_wall_seconds": float(
            train_elapsed_seconds
        ),
        "train_samples_per_second": float(
            total_examples
            / max(
                train_elapsed_seconds,
                1e-12,
            )
        ),
        "peak_cuda_allocated_bytes": int(
            peak_allocated_bytes
        ),
        "peak_cuda_reserved_bytes": int(
            peak_reserved_bytes
        ),
        "peak_cuda_allocated_gb": float(
            peak_allocated_bytes / 1e9
        ),
        "peak_cuda_reserved_gb": float(
            peak_reserved_bytes / 1e9
        ),
        "cuda_device_name": (
            cuda_device_name
        ),
        "cache_stats": (
            cache_stats_for_audit
        ),
    }

    print(
        "[TRAIN AUDIT JSON] {}".format(
            json.dumps(
                LAST_TRAIN_AUDIT,
                sort_keys=True,
            )
        )
    )

    return total_loss_sum / max(1, total_examples)


@torch.no_grad()
def evaluate_pge(
    model_adapter: ModelAdapter,
    loader,
    device,
    zscore_by_gene: bool = True,
    y_mu=None,
    y_sd=None,
    use_batch_dedup: bool = True,
    precomputed_cell: Optional[Dict[str, torch.Tensor]] = None,
):
    model = model_adapter.m
    model.eval()
    total_loss, n = 0.0, 0
    preds, gts = [], []
    can_split = hasattr(unwrap_model(model), "encode_drug") or model_adapter.has_modules
    # Exact evaluation cache: no gradients and fixed model weights.
    evaluation_cell_cache: Dict[str, torch.Tensor] = {}
    evaluation_cache_hits = 0
    evaluation_cache_misses = 0

    for batch in loader:
        drug_graph = batch["drug_graph"].to(device, non_blocking=True)
        cell_seq = batch["cell_seq"]
        time_ = batch["time"].to(device, non_blocking=True)
        dose = batch["dose"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        cell_ids = batch["meta"]["cell_id"]

        if can_split and use_batch_dedup:
            drug_repr = model_adapter.encode_drug(drug_graph)

            cell_repr_list: List[torch.Tensor] = [None] * len(cell_seq)
            uniq_cells: Dict[str, int] = {}
            for i, cid in enumerate(cell_ids):
                if cid not in uniq_cells:
                    uniq_cells[cid] = i
            enc_map: Dict[str, torch.Tensor] = {}

            for cid in uniq_cells.keys():
                cache_key = str(cid)

                if (
                    precomputed_cell is not None
                    and cid in precomputed_cell
                ):
                    enc_map[cid] = (
                        precomputed_cell[cid].to(
                            device,
                            non_blocking=True,
                        )
                    )

                elif cache_key in evaluation_cell_cache:
                    enc_map[cid] = (
                        evaluation_cell_cache[
                            cache_key
                        ]
                    )

                    evaluation_cache_hits += 1

                else:
                    idx0 = uniq_cells[cid]
                    cseq = cell_seq[idx0]

                    encoded_cell = (
                        model_adapter.encode_cell_seq(
                            cseq
                        )
                    )

                    evaluation_cell_cache[
                        cache_key
                    ] = encoded_cell

                    enc_map[cid] = encoded_cell
                    evaluation_cache_misses += 1
            for i, cid in enumerate(cell_ids):
                cell_repr_list[i] = enc_map[cid]

            pred = model_adapter.fuse_and_predict(
                drug_repr=drug_repr,
                cell_repr_list=cell_repr_list,
                time=time_,
                dose=dose,
                return_attn=False,
            )
        else:
            pred = model(
                drug_graph=drug_graph,
                cell_graph_seq=cell_seq,
                cell_ids=cell_ids,
                time=time_,
                dose=dose,
                return_attn=False,
            )

        if zscore_by_gene and (y_mu is not None) and (y_sd is not None):
            loss = _masked_mse((pred - y_mu) / y_sd, (y - y_mu) / y_sd)
        else:
            loss = _masked_mse(pred, y)
        total_loss += float(loss.item()) * y.shape[0]
        n += y.shape[0]
        preds.append(pred.detach())
        gts.append(y.detach())

    print(
        "[EVAL CELL CACHE AUDIT] "
        "size={} hits={} misses={}".format(
            len(evaluation_cell_cache),
            evaluation_cache_hits,
            evaluation_cache_misses,
        )
    )

    if n == 0:
        return {"mse": float("nan"), "pcc": float("nan")}
    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    return {"mse": total_loss / n, "pcc": pcc_torch(preds, gts)}


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
def split_dataframe(df: pd.DataFrame, valid_ratio: float, test_ratio: float, seed: int,
                    mode: str, cell_col: str, drug_col: str):
    rng = np.random.RandomState(seed)
    if mode == "drug_blind":
        keys = df[drug_col].astype(str).unique()
        rng.shuffle(keys)
        n = len(keys)
        n_test = int(n * test_ratio)
        n_val = int(n * valid_ratio)
        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test:n_test + n_val])
        train = df[~df[drug_col].isin(test_keys | val_keys)]
        valid = df[df[drug_col].isin(val_keys)]
        test = df[df[drug_col].isin(test_keys)]
    elif mode == "cell_blind":
        keys = df[cell_col].astype(str).unique()
        rng.shuffle(keys)
        n = len(keys)
        n_test = int(n * test_ratio)
        n_val = int(n * valid_ratio)
        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test:n_test + n_val])
        train = df[~df[cell_col].isin(test_keys | val_keys)]
        valid = df[df[cell_col].isin(val_keys)]
        test = df[df[cell_col].isin(test_keys)]
    else:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = int(len(idx) * test_ratio)
        n_val = int(len(idx) * valid_ratio)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        train = df.iloc[train_idx]
        valid = df.iloc[val_idx]
        test = df.iloc[test_idx]
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def build_cell_graph_cache_union(
    cells: List[str],
    existing_cache: Optional[Dict[str, List[Data]]],
    kegg_pathway_dir: str,
    basal_df: pd.DataFrame,
    max_pathways: int,
) -> Dict[str, List[Data]]:
    cache = {} if existing_cache is None else dict(existing_cache)
    uniq_cells = sorted(set(cells))
    for i, cid in enumerate(uniq_cells):
        if cid in cache:
            continue
        graphs, _ = create_cell_line_graph(
            basal_df, kegg_pathway_dir, cid, max_pathways=max_pathways
        )
        graphs = sanitize_graph_list(graphs)
        if len(graphs) == 0:
            continue
        cache[cid] = graphs
        if (i + 1) % 50 == 0:
            print(f"[CELL] built {i+1}/{len(uniq_cells)} new cell graphs")
    return cache


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
def run_precompute_graph_cache(args: Args):
    os.makedirs(args.cache_dir, exist_ok=True)
    set_seed(args.seed)

    print("[PRECOMP] ===== Step 1. Load landmarks & CSV =====")
    landmark_order = load_landmarks(args.landmark_csv)
    print(f"[PRECOMP] #landmarks: {len(landmark_order)} head={landmark_order[:5]}")

    df = pd.read_csv(args.perturbed_csv, low_memory=False)
    colmap = build_colmap_perturb(df, landmark_order)
    print(
        f"[PRECOMP] Using columns: cell={colmap['cell']} drug={colmap['drug']} "
        f"smiles={colmap['smiles']} time={colmap['time']} dose={colmap['dose']}"
    )

    df = filter_invalid_smiles(df, smi_col=colmap["smiles"])

    basal_df_raw = pd.read_csv(args.basal_csv, low_memory=False)
    if colmap["cell"] not in basal_df_raw.columns:
        raise ValueError(
            f"[PRECOMP] Basal CSV must contain '{colmap['cell']}'. "
            f"Got: {list(basal_df_raw.columns)[:20]}"
        )
    basal_gene_cols = [c for c in basal_df_raw.columns if re.fullmatch(r"\d+", str(c))]
    basal_df = (
        basal_df_raw[[colmap["cell"]] + basal_gene_cols]
        .groupby(colmap["cell"], as_index=True)
        .first()
    )

    all_cells = df[colmap["cell"]].astype(str).tolist()
    uniq_cells = sorted(set(all_cells))
    print(f"[PRECOMP] #uniq cells = {len(uniq_cells)}")

    print("[PRECOMP] ===== Step 2. Build drug_graph_cache =====")
    drug_cache, DRUG_FDIM = build_drug_cache_from_scratch(
        df,
        smiles_col=colmap["smiles"],
    )
    print(f"[PRECOMP] DRUG_FDIM = {DRUG_FDIM}")

    drug_cache_pkg = {
        "cache": drug_cache,
        "fdim": int(DRUG_FDIM),
    }
    out_path = args.drug_graph_cache or os.path.join(args.cache_dir, "drug_graph_cache.pt")
    torch.save(drug_cache_pkg, out_path)
    print(f"[PRECOMP] drug_graph_cache saved -> {out_path}")

    print("[PRECOMP] ===== Step 3. Build cell_graph_cache_union =====")
    cell_graph_cache = build_cell_graph_cache_union(
        cells=uniq_cells,
        existing_cache=None,
        kegg_pathway_dir=args.kegg_pathway_dir,
        basal_df=basal_df,
        max_pathways=args.num_pathways,
    )
    print(
        f"[PRECOMP] cell_graph_cache ready: {len(cell_graph_cache)} cells "
        f"(needed={len(uniq_cells)})"
    )

    cell_out = args.cell_graph_cache or os.path.join(args.cache_dir, "cell_graph_cache_union.pt")
    torch.save(cell_graph_cache, cell_out)
    print(f"[PRECOMP] cell_graph_cache saved -> {cell_out}")

    print("[PRECOMP] ===== DONE =====")


# ═══════════════════════════════════════════════════════════════════════════════
# ★★★ FIXED: Main training function with memory optimization ★★★
# ═══════════════════════════════════════════════════════════════════════════════
def run_pge_pretrain_with_precomputed(args: Args):
    try:
        if mp.get_sharing_strategy() != "file_descriptor":
            mp.set_sharing_strategy("file_descriptor")
    except Exception as e:
        print(f"[WARN] set_sharing_strategy failed: {e}")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Landmarks
    landmark_order = load_landmarks(args.landmark_csv)
    print(f"[INFO] #landmarks: {len(landmark_order)} head={landmark_order[:5]}")

    # 2) Perturbed CSV + colmap
    df = pd.read_csv(args.perturbed_csv, low_memory=False)
    colmap = build_colmap_perturb(df, landmark_order)
    print(
        f"[INFO] Using columns: cell={colmap['cell']} drug={colmap['drug']} "
        f"smiles={colmap['smiles']} time={colmap['time']} dose={colmap['dose']} "
        f"#genes_in_csv={len(colmap['genes'])}"
    )

    df = filter_invalid_smiles(df, smi_col=colmap["smiles"])

    if args.clamp_label is not None:
        cap = float(args.clamp_label)
        for g in colmap["genes"]:
            if g in df.columns:
                df[g] = df[g].clip(-cap, cap)

    # 3) Basal
    basal_df_raw = pd.read_csv(args.basal_csv, low_memory=False)
    if colmap["cell"] not in basal_df_raw.columns:
        raise ValueError(
            f"Basal CSV must contain '{colmap['cell']}'. Got: {list(basal_df_raw.columns)[:20]}"
        )
    basal_gene_cols = [c for c in basal_df_raw.columns if re.fullmatch(r"\d+", str(c))]
    basal_df = (
        basal_df_raw[[colmap["cell"]] + basal_gene_cols]
        .groupby(colmap["cell"], as_index=True)
        .first()
    )
    del basal_df_raw

    # 4) Split & subset
    train_df, val_df, test_df = split_dataframe(
        df,
        args.valid_ratio,
        args.test_ratio,
        args.split_seed,
        args.split_mode,
        cell_col=colmap["cell"],
        drug_col=colmap["drug"],
    )
    print(f"[INFO] Split -> train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    if args.subset_ratio < 0.999:
        train_df = train_df.sample(frac=args.subset_ratio, random_state=args.split_seed).reset_index(drop=True)
        val_df = val_df.sample(frac=args.subset_ratio, random_state=args.split_seed).reset_index(drop=True)
        test_df = test_df.sample(frac=args.subset_ratio, random_state=args.split_seed).reset_index(drop=True)
        print(f"[INFO] After subset -> train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    # 5) Drug graph cache
    from torch_geometric.data import Data as GeoData
    DRUG_FDIM = None
    drug_cache = {}

    if args.drug_graph_cache and os.path.isfile(args.drug_graph_cache):
        try:
            pkg = torch.load(args.drug_graph_cache, map_location="cpu")
            if isinstance(pkg, dict) and "cache" in pkg and "fdim" in pkg:
                drug_cache = pkg["cache"]
                DRUG_FDIM = int(pkg["fdim"])
                print(f"[INFO] loaded drug_graph_cache from {args.drug_graph_cache} ({len(drug_cache)} drugs, fdim={DRUG_FDIM})")
            else:
                if isinstance(pkg, dict):
                    drug_cache = pkg
                    any_g = next(iter(drug_cache.values()))
                    DRUG_FDIM = int(any_g.x.shape[1])
                    print(f"[INFO] loaded legacy drug_graph_cache ({len(drug_cache)} drugs, fdim={DRUG_FDIM})")
        except Exception as e:
            print(f"[WARN] failed to load drug_graph_cache: {e}")

    if (not drug_cache) or (DRUG_FDIM is None):
        print("[WARN] Rebuilding drug_graph_cache from scratch")
        base_rows = pd.concat([train_df, val_df, test_df], ignore_index=True) if args.cache_scope == "all" else train_df
        drug_cache, DRUG_FDIM = build_drug_cache_from_scratch(base_rows, smiles_col=colmap["smiles"])

    # 6) Filter by drug cache
    def _has_valid_drug(smiles: str, drug_id: str) -> bool:
        for k in (_norm_key(smiles), _norm_key(drug_id), smiles, drug_id):
            v = drug_cache.get(k)
            if isinstance(v, GeoData) and hasattr(v, "x") and v.x.ndim == 2 and int(v.x.shape[1]) == DRUG_FDIM:
                return True
        return False

    def _filter_by_drug(df_split: pd.DataFrame, name: str) -> pd.DataFrame:
        if df_split.empty:
            return df_split
        mask = [_has_valid_drug(str(row[colmap["smiles"]]), str(row[colmap["drug"]])) for _, row in df_split.iterrows()]
        mask = np.asarray(mask, dtype=bool)
        dropped = int((~mask).sum())
        if dropped > 0:
            print(f"[FILTER][{name}] drop {dropped} rows without valid drug graph")
        return df_split[mask].reset_index(drop=True)

    train_df = _filter_by_drug(train_df, "train")
    val_df = _filter_by_drug(val_df, "val")
    test_df = _filter_by_drug(test_df, "test")
    print(f"[INFO] After drug_cache filter -> train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    # 7) Cell graph cache
    raw_cell_cache = None
    if args.cell_graph_cache and os.path.isfile(args.cell_graph_cache):
        try:
            raw_cell_cache = torch.load(args.cell_graph_cache, map_location="cpu")
        except Exception as e:
            print(f"[WARN] Failed to load cell_graph_cache: {e}")

    cell_graph_cache = {}
    if isinstance(raw_cell_cache, dict):
        print(f"[INFO] loaded raw cell_graph_cache from {args.cell_graph_cache}")
        for cid, seq in raw_cell_cache.items():
            try:
                if not isinstance(seq, (list, tuple)):
                    continue
                g_list = [sanitize_pyg_data(g) for g in seq if isinstance(g, Data)]
                if len(g_list) > 0:
                    cell_graph_cache[cid] = g_list
            except Exception as e:
                print(f"[WARN] drop bad cell '{cid}': {e}")
        print(f"[INFO] sanitized cell_graph_cache: {len(cell_graph_cache)} cells")

    # Check coverage
    all_cells = pd.concat([train_df[[colmap["cell"]]], val_df[[colmap["cell"]]], test_df[[colmap["cell"]]]], ignore_index=True)[colmap["cell"]].astype(str).tolist()
    uniq_cells = sorted(set(all_cells))
    missing = [c for c in uniq_cells if c not in cell_graph_cache]
    if missing:
        print(f"[WARN] {len(missing)} cells missing in cell_graph_cache")
    else:
        print("[INFO] All cells covered by cell_graph_cache")

    # Faithful token-preserving pathway batching is enabled only after
    # passing the stored forward/gradient equivalence benchmark.
    benchmark_json = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "equivalence_benchmark_s42",
        "equivalence_benchmark.json",
    )

    print(
        "[PATHWAY BATCHING SAFETY] enabled={} | validation={}".format(
            bool(args.use_pathway_batching),
            benchmark_json,
        )
    )

    if bool(args.use_pathway_batching):
        if not os.path.isfile(benchmark_json):
            raise RuntimeError(
                "Pathway batching equivalence report is missing: "
                + benchmark_json
            )

        with open(benchmark_json, "r") as handle:
            pathway_audit = json.load(handle)

        required_metrics = [
            "vectorized_vs_v1_eval_max_abs",
            "optimized_vs_vectorized_eval_max_abs",
            "train_prediction_max_abs",
            "train_loss_delta",
            "gradient_relative_l2",
            "optimized_token_audit",
        ]
        missing_metrics = [
            key for key in required_metrics
            if key not in pathway_audit
        ]
        if missing_metrics:
            raise RuntimeError(
                "Incomplete pathway batching equivalence report: "
                + repr(missing_metrics)
            )

        failures = {}

        if float(
            pathway_audit["vectorized_vs_v1_eval_max_abs"]
        ) > 2.0e-4:
            failures["vectorized_vs_v1_eval_max_abs"] = (
                pathway_audit["vectorized_vs_v1_eval_max_abs"]
            )

        if float(
            pathway_audit["optimized_vs_vectorized_eval_max_abs"]
        ) > 2.0e-4:
            failures["optimized_vs_vectorized_eval_max_abs"] = (
                pathway_audit[
                    "optimized_vs_vectorized_eval_max_abs"
                ]
            )

        if float(
            pathway_audit["train_prediction_max_abs"]
        ) > 2.0e-4:
            failures["train_prediction_max_abs"] = (
                pathway_audit["train_prediction_max_abs"]
            )

        if float(pathway_audit["train_loss_delta"]) > 1.0e-7:
            failures["train_loss_delta"] = (
                pathway_audit["train_loss_delta"]
            )

        if float(
            pathway_audit["gradient_relative_l2"]
        ) > 5.0e-3:
            failures["gradient_relative_l2"] = (
                pathway_audit["gradient_relative_l2"]
            )

        token_audit = pathway_audit["optimized_token_audit"]

        if not bool(token_audit.get("pathway_batching")):
            failures["pathway_batching"] = token_audit.get(
                "pathway_batching"
            )

        if int(
            token_audit.get("unique_gene_coverage_min", -1)
        ) != 268:
            failures["unique_gene_coverage_min"] = (
                token_audit.get("unique_gene_coverage_min")
            )

        if failures:
            raise RuntimeError(
                "Pathway batching equivalence validation failed: "
                + json.dumps(failures, sort_keys=True)
            )

        print(
            "[PATHWAY BATCHING VALIDATION] passed | "
            "eval_max_abs={:.3e} | train_max_abs={:.3e} | "
            "gradient_relative_l2={:.3e} | genes={}".format(
                float(
                    pathway_audit[
                        "optimized_vs_vectorized_eval_max_abs"
                    ]
                ),
                float(
                    pathway_audit["train_prediction_max_abs"]
                ),
                float(
                    pathway_audit["gradient_relative_l2"]
                ),
                int(
                    token_audit["unique_gene_coverage_min"]
                ),
            )
        )

    # 8) Model
    mcfg = ModelConfig(
        dim_node=args.dim_node,
        out_drug=args.dim_drug,
        out_cell=args.dim_cell,
        pe_dim=args.pe_dim,
        num_pathways=args.num_pathways,
        transformer_heads=args.transformer_heads,
        ffn_dim=args.ffn_dim if args.ffn_dim else args.dim_node * 4,
        transformer_layers=args.transformer_layers,
        dropout_ratio=args.dropout_ratio,
        max_num_nodes=args.max_num_nodes,
        freeze_encoders=args.freeze_encoders,
        last_n_layers=args.last_n_layers,
        unfreeze_pool_query=args.unfreeze_pool_query,
        unfreeze_time_dose=args.unfreeze_time_dose,
        unfreeze_type_embed=args.unfreeze_type_embed,
        use_pathway_batching=args.use_pathway_batching,
        use_gradient_checkpointing=args.gradient_checkpointing,
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

    device = torch.device(args.device)
    model = DrugResponseTransformer(
        args=enc_args,
        landmark_set=landmark_order,
        config=mcfg,
        task="pge",
    )
    print(
        "[PATHWAY BATCHING AUDIT] enabled={}".format(
            bool(args.use_pathway_batching)
        )
    )
    model.setup_finetune(mcfg)
    model.to(device)

    adapter = ModelAdapter(model)
    if args.try_torch_compile:
        adapter.try_compile()

    # ★ NEW: Create LRU cache for cell embeddings
    cell_lru_cache = LRUCellCache(
        max_size=args.cell_cache_max_size,
        device="cpu"  # Store on CPU to save GPU memory
    )
    print(f"[INFO] Created LRU cell cache with max_size={args.cell_cache_max_size}")

    # Precompute cell embed only mode
    if args.precompute_cell_embed_only:
        model.eval()
        precomputed = {}
        with torch.no_grad():
            print(f"[PRECOMP_CELL] Precomputing cell embeddings for {len(uniq_cells)} cells")
            for i, cid in enumerate(uniq_cells, start=1):
                seq = cell_graph_cache.get(cid)
                if seq is None:
                    continue
                emb = adapter.encode_cell_seq(seq)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu()
                precomputed[cid] = emb
                if i % 50 == 0:
                    print(f"[PRECOMP_CELL] {i}/{len(uniq_cells)} done")

        out_path = args.precomputed_cell_embed or os.path.join(args.cache_dir, "precomputed_cell_embed.pth")
        torch.save(precomputed, out_path)
        print(f"[PRECOMP_CELL] saved {len(precomputed)} cell embeddings -> {out_path}")
        return

    # 9) Datasets/Loaders
    rng_train = np.random.RandomState(args.seed + 123)

    def build_ds(d, rng=None):
        return PGEDataset(
            df=d,
            colmap=colmap,
            landmark_order=landmark_order,
            drug_cache=drug_cache,
            expected_drug_fdim=DRUG_FDIM,
            cell_graph_cache=cell_graph_cache,
            max_pathways=args.num_pathways,
            on_bad_cell=args.on_bad_cell,
            on_bad_drug=args.on_bad_drug,
            num_pathways_sample=args.num_pathways_sample,
            sample_strategy=args.sample_strategy,
            rng=rng,
        )

    tr_ds = build_ds(train_df, rng=rng_train)
    va_ds = build_ds(val_df)
    te_ds = build_ds(test_df)

    common_loader = dict(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=pge_collate,
        drop_last=False,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, generator=g, **common_loader)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, generator=g, **common_loader)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, generator=g, **common_loader)

    if args.first_batch_probe:
        t0 = time.time()
        _ = next(iter(tr_loader))
        print(f"[PROBE] first batch ready in {time.time() - t0:.2f}s")

    # 10) Optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    core = unwrap_model(model)
    param_groups = core.get_finetune_param_groups(
        body_lr=args.body_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    from math import cos, pi

    # PATCH_20260719_PGE_SCHEDULER_AND_DISK_BEST
    def make_scheduler(total_epochs: int, warmup_epochs: int):
        """
        Epoch-level linear warmup followed by cosine decay.

        LambdaLR evaluates lr_lambda(0) during construction, before
        epoch 1. Therefore epoch_index=0 must return a meaningful
        first-epoch factor rather than nearly zero.
        """
        total_epochs = max(1, int(total_epochs))
        warmup_epochs = max(
            0,
            min(
                int(warmup_epochs),
                total_epochs,
            ),
        )

        def lr_lambda(epoch_index):
            epoch_index = int(epoch_index)

            if warmup_epochs > 0 and epoch_index < warmup_epochs:
                return float(
                    epoch_index + 1
                ) / float(warmup_epochs)

            decay_epochs = max(
                1,
                total_epochs - warmup_epochs,
            )

            decay_index = max(
                0,
                epoch_index - warmup_epochs,
            )

            progress = min(
                1.0,
                float(decay_index)
                / float(decay_epochs),
            )

            return (
                0.1
                + 0.9
                * (
                    1.0
                    + cos(pi * progress)
                )
                / 2.0
            )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )

    # 11) Z-norm
    print("[INFO] Z-norm: building matrix ...")
    y_train_df = train_df.reindex(columns=landmark_order)
    y_train = y_train_df.to_numpy(dtype=np.float32, copy=True)
    print(f"[INFO] Z-norm: shape={y_train.shape}, mem≈{y_train.size*4/1e9:.2f} GB")
    mu_np = np.nanmean(y_train, axis=0)
    sd_np = np.nanstd(y_train, axis=0)
    sd_np[~np.isfinite(sd_np)] = 1.0
    sd_np[sd_np < 1e-8] = 1.0
    del y_train

    y_mu = torch.tensor(mu_np, dtype=torch.float32, device=device).view(1, -1)
    y_sd = torch.tensor(sd_np, dtype=torch.float32, device=device).view(1, -1)
    print("[INFO] Z-norm ready.")

    # 12) Save paths
    ts = time.strftime("%Y%m%d-%H%M%S")
    dev_idx = getattr(device, "index", None)
    dev_tag = f"{device.type}{dev_idx}" if (device.type == "cuda" and dev_idx is not None) else device.type
    prefix = args.run_name or f"pge_pretrain_v3_{args.split_mode}_seed{args.seed}_{dev_tag}_p{args.num_pathways}"
    run_id = f"{prefix}_{ts}"
    model_save_path = os.path.join(args.save_dir, f"{run_id}.pth")
    report_save_path = os.path.join(args.save_dir, f"{run_id}.report.json")

    best_val = {"mse": float("inf"), "pcc": -1.0}
    best_state = None
    best_checkpoint_path = None
    best_checkpoint_reloaded_from_disk = False
    resume_ckpt = None
    resume_lower_lr_requested = bool(args.resume_lower_lr)

    # Resume
    start_epoch = 1
    total_epochs = args.epochs
    if args.resume_from:
        print(f"[RESUME] loading checkpoint: {args.resume_from}")
        resume_ckpt = torch.load(
            args.resume_from,
            map_location="cpu",
        )
        ckpt = resume_ckpt
        best_checkpoint_path = args.resume_from
        core.load_state_dict(
            ckpt["model_state_dict"],
            strict=args.resume_strict,
        )
        if "val_metrics" in ckpt:
            best_val = ckpt["val_metrics"]
            print(f"[RESUME] prev best val: {best_val}")
        if "y_mu" in ckpt and "y_sd" in ckpt:
            try:
                y_mu = ckpt["y_mu"].to(device).view(1, -1)
                y_sd = ckpt["y_sd"].to(device).view(1, -1)
                print("[RESUME] restored y_mu/y_sd from ckpt")
            except Exception:
                pass
        saved_epoch = int(ckpt.get("epoch", 0))
        start_epoch = saved_epoch + 1

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("[RESUME] optimizer state restored")
            except Exception as e:
                print(f"[WARN] optimizer state restore failed: {e}")

        if args.resume_lower_lr:
            print(
                "[RESUME] lower LR requested; "
                "will apply after scheduler restoration"
            )

        if args.epochs_more > 0:
            total_epochs = saved_epoch + args.epochs_more
            print(f"[RESUME] will train from epoch {start_epoch} to {total_epochs}")
        else:
            total_epochs = saved_epoch

    scheduler = make_scheduler(
        total_epochs=total_epochs,
        warmup_epochs=int(args.warmup_epochs),
    )

    if (
        resume_ckpt is not None
        and "scheduler_state_dict" in resume_ckpt
    ):
        try:
            scheduler.load_state_dict(
                resume_ckpt[
                    "scheduler_state_dict"
                ]
            )

            restored_last_lrs = (
                resume_ckpt[
                    "scheduler_state_dict"
                ].get(
                    "_last_lr"
                )
            )

            if (
                isinstance(
                    restored_last_lrs,
                    (list, tuple),
                )
                and len(restored_last_lrs)
                == len(optimizer.param_groups)
            ):
                for group, restored_lr in zip(
                    optimizer.param_groups,
                    restored_last_lrs,
                ):
                    group["lr"] = float(
                        restored_lr
                    )

            print(
                "[RESUME] scheduler state restored "
                "| last_epoch={} | lrs={}".format(
                    scheduler.last_epoch,
                    [
                        float(group["lr"])
                        for group
                        in optimizer.param_groups
                    ],
                )
            )

        except Exception as error:
            print(
                "[WARN] scheduler state restore failed: "
                "{}".format(error)
            )

    if (
        resume_ckpt is not None
        and resume_lower_lr_requested
    ):
        for group in optimizer.param_groups:
            group["lr"] *= 0.5

        print(
            "[RESUME] lowered restored LR x0.5 "
            "| lrs={}".format(
                [
                    float(group["lr"])
                    for group
                    in optimizer.param_groups
                ]
            )
        )

    print(
        "[SCHEDULER AUDIT] initialization "
        "| total_epochs={} | warmup_epochs={} "
        "| last_epoch={} | base_lrs={} | current_lrs={}".format(
            total_epochs,
            int(args.warmup_epochs),
            scheduler.last_epoch,
            [
                float(value)
                for value
                in scheduler.base_lrs
            ],
            [
                float(group["lr"])
                for group
                in optimizer.param_groups
            ],
        )
    )

    # Multi-GPU
    if args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        adapter.m = model
        core = unwrap_model(model)

    # Precomputed cell embed
    precomputed_cell = None
    if args.precomputed_cell_embed and os.path.isfile(args.precomputed_cell_embed):
        try:
            precomputed_cell = torch.load(args.precomputed_cell_embed, map_location="cpu")
            if isinstance(precomputed_cell, dict):
                print(f"[INFO] loaded precomputed cell embeddings: {len(precomputed_cell)} cells")
            else:
                precomputed_cell = None
        except Exception as e:
            print(f"[WARN] failed to load precomputed_cell_embed: {e}")

    epoch_cache = {} if args.epoch_embed_cache else None

    # 13) Training loop
    did_train = False
    epoch_train_audits = []

    for epoch in range(start_epoch, total_epochs + 1):
        did_train = True
        if epoch_cache is not None:
            epoch_cache.clear()

        # ★ Clear LRU cache at epoch start (optional, can comment out to keep across epochs)
        # cell_lru_cache.clear()

        tr_mse = train_one_epoch_pge(
            adapter,
            tr_loader,
            optimizer,
            device,
            scaler=scaler,
            log_interval=args.log_interval,
            accum_steps=args.accum_steps,
            zscore_by_gene=args.zscore_by_gene,
            y_mu=y_mu,
            y_sd=y_sd,
            grad_clip=args.grad_clip,
            use_batch_dedup=args.use_batch_dedup,
            epoch_cache=epoch_cache,
            refresh_every_k_steps=int(args.refresh_every_k_steps),
            precomputed_cell=precomputed_cell,
            live_cell_update_freq=args.live_cell_update_freq,
            # ★ NEW: Memory optimization args
            cell_lru_cache=cell_lru_cache,
            empty_cache_freq=args.empty_cache_freq,
        )

        epoch_train_audit = dict(
            LAST_TRAIN_AUDIT
        )

        epoch_train_audit[
            "epoch"
        ] = int(epoch)

        epoch_train_audits.append(
            epoch_train_audit
        )

        # Print cache stats
        cache_stats = cell_lru_cache.stats()
        print(f"[Epoch {epoch}] Cache stats: {cache_stats}")

        val_metrics = evaluate_pge(
            adapter,
            va_loader,
            device,
            zscore_by_gene=args.zscore_by_gene,
            y_mu=y_mu,
            y_sd=y_sd,
            use_batch_dedup=args.use_batch_dedup,
            precomputed_cell=precomputed_cell,
        )
        print(f"[Epoch {epoch:03d}] train_mse={tr_mse:.4f} | val_mse={val_metrics['mse']:.4f} | val_pcc={val_metrics['pcc']:.4f}")

        val_pcc = float(
            val_metrics["pcc"]
        )

        improved = bool(
            np.isfinite(val_pcc)
        ) and (
            best_checkpoint_path is None
            or val_pcc > float(
                best_val["pcc"]
            )
        )

        if improved:
            best_val = dict(
                val_metrics
            )

        # Step at the end of the completed epoch. The checkpoint
        # therefore stores the LR state needed for the next epoch.
        scheduler.step()

        print(
            "[SCHEDULER AUDIT] after epoch {} "
            "| last_epoch={} | next_epoch_lrs={}".format(
                epoch,
                scheduler.last_epoch,
                [
                    float(group["lr"])
                    for group
                    in optimizer.param_groups
                ],
            )
        )

        if improved:
            core = unwrap_model(model)

            best_state = {
                "model_state_dict": core.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_lrs": [
                    float(group["lr"])
                    for group
                    in optimizer.param_groups
                ],
                "args": asdict(args),
                "model_config": asdict(mcfg),
                "epoch": epoch,
                "val_metrics": dict(best_val),
                "landmarks": [
                    str(g)
                    for g in landmark_order
                ],
                "y_mu": y_mu.detach().cpu(),
                "y_sd": y_sd.detach().cpu(),
            }

            torch.save(
                best_state,
                model_save_path,
            )

            best_checkpoint_path = (
                model_save_path
            )

            print(
                "[SAVE] {} "
                "(epoch={}, val_pcc={:.4f})".format(
                    model_save_path,
                    epoch,
                    best_val["pcc"],
                )
            )

        # ★ Memory cleanup after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if (
        best_checkpoint_path is None
        or not os.path.isfile(
            best_checkpoint_path
        )
    ):
        raise RuntimeError(
            "No valid best checkpoint is available "
            "before test evaluation."
        )

    disk_best_state = torch.load(
        best_checkpoint_path,
        map_location="cpu",
    )

    unwrap_model(model).load_state_dict(
        disk_best_state[
            "model_state_dict"
        ],
        strict=True,
    )

    best_state = disk_best_state
    best_val = dict(
        disk_best_state.get(
            "val_metrics",
            best_val,
        )
    )

    best_checkpoint_reloaded_from_disk = True

    print(
        "[BEST CHECKPOINT] reloaded from disk: {} "
        "(epoch={}, val_pcc={:.6f})".format(
            best_checkpoint_path,
            int(
                disk_best_state.get(
                    "epoch",
                    -1,
                )
            ),
            float(
                best_val["pcc"]
            ),
        )
    )

    # 14) Test
    test_metrics = evaluate_pge(
        adapter,
        te_loader,
        device,
        zscore_by_gene=args.zscore_by_gene,
        y_mu=y_mu,
        y_sd=y_sd,
        use_batch_dedup=args.use_batch_dedup,
        precomputed_cell=precomputed_cell,
    )
    print(f"[TEST] mse={test_metrics['mse']:.4f} pcc={test_metrics['pcc']:.4f}")

    # 15) Report
    report = {
        "best_val": best_val,
        "test": test_metrics,
        "args": asdict(args),
        "model_path": best_checkpoint_path,
        "best_epoch": int(
            best_state.get(
                "epoch",
                -1,
            )
        ),
        "best_checkpoint_reloaded_from_disk": (
            best_checkpoint_reloaded_from_disk
        ),
        "tested_checkpoint_path": (
            best_checkpoint_path
        ),
        "trained": did_train,
        "start_epoch": start_epoch,
        "total_epochs": total_epochs,
        "landmark_order": [str(g) for g in landmark_order],
        "colmap": colmap,
        "final_cache_stats": cell_lru_cache.stats(),
        "epoch_train_audits": epoch_train_audits,
    }
    with open(report_save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Report saved -> {report_save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["precompute", "train"])

    for field in Args.__dataclass_fields__.values():
        name = field.name
        if name == "mode":
            continue
        default = field.default
        ftype = type(default) if default is not None else str

        # PATCH_20260719_OPTIONAL_INT_FFN_DIM
        # Args.ffn_dim is Optional[int] with default None.
        # The generic parser would otherwise parse --ffn_dim 256
        # as the string "256", which is invalid for nn.Linear.
        if name == "ffn_dim":
            ftype = int

        if isinstance(default, bool):
            parser.add_argument(f"--{name}", action="store_true" if not default else "store_false")
        else:
            parser.add_argument(f"--{name}", type=ftype, default=default)

    args_ns = parser.parse_args()
    args = Args(**vars(args_ns))


    if args.use_batch_dedup:
        raise RuntimeError(
            "Faithful gene-token pretraining forbids use_batch_dedup=True "
            "because that path pools atoms/genes before the Transformer."
        )

    if args.mode == "precompute":
        args.num_workers = 0
        print("[ENTRY] run_precompute_graph_cache() with num_workers=0")
        run_precompute_graph_cache(args)
    elif args.mode == "train":
        print(f"[ENTRY] run_pge_pretrain_with_precomputed() on {args.device} with num_workers={args.num_workers}")
        run_pge_pretrain_with_precomputed(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()