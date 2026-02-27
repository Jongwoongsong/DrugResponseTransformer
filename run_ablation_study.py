"""
═══════════════════════════════════════════════════════════════════════════════
Ablation Study for DrugResponseTransformer (CSBJ Paper)
═══════════════════════════════════════════════════════════════════════════════

4가지 Ablation Variant:
  1. full          : Full model (baseline, pretrained)
  2. no_pretrain   : Full model without pretraining
  3. no_kegg       : w/o KEGG pathway graphs → Flat MLP cell encoder
  4. no_condition   : w/o dose/time condition embeddings (zeroed out)
  5. no_transformer : w/o Transformer → Concat + MLP

Usage:
    # 단일 실험
    python run_ablation_study.py --variant no_kegg --split mixed --device cuda:0

    # 전체 ablation (GPU 1개)
    python run_ablation_study.py --variant all --split mixed --device cuda:0

    # 특정 variant들만
    python run_ablation_study.py --variant no_kegg,no_condition --split mixed --device cuda:0

Note:
    - 'full' 과 'no_pretrain'은 이미 결과가 있으므로 기본적으로 skip
    - 새로 돌려야 하는 건 no_kegg, no_condition, no_transformer
    - 동일 split, seed, hyperparameter 사용으로 fair comparison 보장
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Silence RDKit ──
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.debug')
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
except Exception:
    pass

# ── Imports ──
try:
    from Model.drug_graph import smiles_to_graph, calculate_position_encoding
except Exception:
    from drug_graph import smiles_to_graph, calculate_position_encoding

try:
    from Model.DrugEncoder import DrugEncoder, DrugGAT
except Exception:
    from DrugEncoder import DrugEncoder, DrugGAT


# ═══════════════════════════════════════════════════════════════════════════════
# 0) Config
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class AblationConfig:
    # Variant
    variant: str = "no_kegg"  # full, no_pretrain, no_kegg, no_condition, no_transformer

    # Paths
    csv_path: str = "GDSC_with_SMILES_BGE_entrez_KEGG_fixed.csv"
    save_dir: str = "ablation_results"
    drug_graph_cache: Optional[str] = "graph_cache/drug_graph_cache.pt"
    cell_graph_cache: Optional[str] = "graph_cache/cell_graph_cache.gdsc_unknown_z.fixed.pt"

    # Hardware
    device: str = "cuda:0"
    num_workers: int = 0
    pin_memory: bool = True

    # Data split (동일해야 fair comparison!)
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    split_mode: str = "mixed"  # mixed or drug_blind
    drop_na_label: bool = True
    subset_ratio: float = 1.0

    # Training (기존 모델과 동일한 hyperparameters)
    epochs: int = 50
    batch_size: int = 64
    lr: float = 6e-4
    weight_decay: float = 2e-3
    grad_clip: float = 1.0
    amp: bool = True
    log_interval: int = 100

    # Loss
    use_label_standardize: bool = True
    use_corr_loss: bool = True
    alpha_corr: float = 0.2
    use_smoothl1: bool = False

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-3

    # Model dims (기존 모델과 동일!)
    dim_node: int = 64
    pe_dim: int = 1
    max_num_nodes: int = 44
    num_pathways: int = 31
    drug_hidden_dim: int = 64
    drug_out_dim: int = 64
    drug_heads: int = 4
    cell_hidden_dim: int = 64
    cell_out_dim: int = 30

    # Transformer (for no_transformer variant: 이 부분을 MLP로 대체)
    transformer_heads: int = 8
    transformer_layers: int = 2
    ffn_dim: int = 256  # dim_node * 4

    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════════
# 1) Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# 2) Column / Split helpers (기존과 동일)
# ═══════════════════════════════════════════════════════════════════════════════
def _find_col(df, candidates):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def build_colmap(df):
    cell_col  = _find_col(df, ["CELL_LINE_NAME", "cell_line_name", "cell", "cell_line"])
    drug_col  = _find_col(df, ["DRUG_NAME", "drug_name", "drug"])
    minc_col  = _find_col(df, ["MIN_CONC", "min_conc", "minc"])
    maxc_col  = _find_col(df, ["MAX_CONC", "maxc", "maxc"])
    label_col = _find_col(df, ["LN_IC50", "ln_ic50", "IC50", "ic50"])
    smi_col   = _find_col(df, ["canonical_smiles", "smiles", "SMILES"])
    gene_cols = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]

    required = {"cell": cell_col, "drug": drug_col, "minc": minc_col, "maxc": maxc_col,
                "label": label_col, "smiles": smi_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not gene_cols:
        raise ValueError("No gene-expression columns")

    return {**required, "genes": gene_cols}

def split_dataframe(df, valid_ratio, test_ratio, seed, mode, colmap):
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
        n_test = int(len(idx) * test_ratio); n_val = int(len(idx) * valid_ratio)
        train = df.iloc[idx[n_test + n_val:]]
        valid = df.iloc[idx[n_test:n_test + n_val]]
        test  = df.iloc[idx[:n_test]]
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3) Graph utils
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_smiles_to_graph(smiles):
    try:
        return smiles_to_graph(smiles)
    except:
        return None

def _to_cpu(t, dtype=None):
    if isinstance(t, torch.Tensor):
        out = torch.from_numpy(t.detach().cpu().numpy().copy())
    else:
        out = torch.tensor(t)
    if dtype:
        out = out.to(dtype)
    return out

def sanitize_pyg(g):
    out = Data()
    out.x = _to_cpu(g.x, torch.float32)
    out.edge_index = _to_cpu(g.edge_index, torch.long)
    if hasattr(g, "edge_attr") and g.edge_attr is not None:
        out.edge_attr = _to_cpu(g.edge_attr, torch.float32)
    if hasattr(g, "pe") and g.pe is not None:
        out.pe = _to_cpu(g.pe, torch.float32)
    out.num_nodes = int(out.x.size(0))
    return out

def warmup_drug_cache(df, colmap, drug_cache):
    if drug_cache is None:
        drug_cache = {}
    for _, row in df[[colmap["drug"], colmap["smiles"]]].drop_duplicates().iterrows():
        smiles = str(row[colmap["smiles"]])
        if smiles not in drug_cache:
            g = _safe_smiles_to_graph(smiles)
            if g is not None:
                drug_cache[smiles] = sanitize_pyg(g)
    return drug_cache

def filter_invalid_smiles(df, colmap):
    smi_col = colmap["smiles"]
    vals = df[smi_col].astype(str).fillna("").str.strip()
    mask = (vals == "") | vals.str.fullmatch(r"\d+")
    rdkit_fail = []
    df1 = df[~mask].copy()
    for i, s in df1[smi_col].astype(str).items():
        if _safe_smiles_to_graph(s) is None:
            rdkit_fail.append(i)
    df2 = df1.drop(index=rdkit_fail).reset_index(drop=True)
    logger.info(f"SMILES filter: removed {int(mask.sum()) + len(rdkit_fail)}, remain {len(df2)}")
    return df2


# ═══════════════════════════════════════════════════════════════════════════════
# 4) Cell Embedding 사전계산
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def precompute_cell_embeddings_kegg(cell_graph_cache, cfg, device):
    """
    KEGG pathway graph → CellEncoder → mean pool → [D] per cell
    (기존 방식, Full model / no_condition / no_transformer에 사용)
    """
    from CellEncoder import CellEncoder

    D = cfg.dim_node
    num_token_types = 2 + cfg.num_pathways
    token_type_embed = nn.Embedding(num_token_types, D).to(device)
    pathway_type_ids = list(range(2, 2 + cfg.num_pathways))

    class _A: pass
    enc_args = _A()
    enc_args.dim_node = cfg.dim_node
    enc_args.pe_dim = cfg.pe_dim
    enc_args.max_num_nodes = cfg.max_num_nodes
    enc_args.dropout_ratio = 0.2
    enc_args.dropout = 0.2

    cell_encoder = CellEncoder(
        in_channels=1, hidden_channels=cfg.cell_hidden_dim,
        out_channels=cfg.cell_out_dim, edge_attr_dim=1, dropout=0.5,
        args=enc_args, token_type_embed=token_type_embed,
        pathway_type_ids=pathway_type_ids,
    ).to(device)
    cell_encoder.eval()

    result = {}
    cell_ids = list(cell_graph_cache.keys())
    t0 = time.time()

    for i, cid in enumerate(cell_ids):
        graphs = cell_graph_cache[cid][:cfg.num_pathways]
        per_path = []
        for p_idx, g in enumerate(graphs):
            if g is None:
                continue
            g = g.to(device)
            emb = cell_encoder(g, pathway_idx=p_idx)
            per_path.append(emb.mean(dim=0))
        if per_path:
            result[cid] = torch.stack(per_path).mean(0).cpu()
        else:
            result[cid] = torch.zeros(D).cpu()
        if (i + 1) % 200 == 0:
            logger.info(f"  KEGG cell embed: [{i+1}/{len(cell_ids)}] ({time.time()-t0:.1f}s)")

    logger.info(f"Precomputed {len(result)} KEGG cell embeddings")
    return result


@torch.no_grad()
def precompute_cell_embeddings_flat(df, colmap, cfg, device):
    """
    ★ ABLATION: w/o KEGG pathways
    Gene expression → Flat MLP → [D] per cell

    KEGG pathway graph 구조를 사용하지 않고,
    raw gene expression을 MLP로 인코딩
    """
    D = cfg.dim_node
    gene_cols = colmap["genes"]
    num_genes = len(gene_cols)

    # Simple MLP encoder (pathway graph 대신)
    flat_encoder = nn.Sequential(
        nn.Linear(num_genes, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, D),
    ).to(device)
    flat_encoder.eval()

    # Cell별 unique gene expression vector
    cell_col = colmap["cell"]
    cell_expr = df.groupby(cell_col)[gene_cols].first()

    result = {}
    for cid in cell_expr.index:
        expr = torch.tensor(cell_expr.loc[cid].values.astype(np.float32)).to(device)
        emb = flat_encoder(expr.unsqueeze(0)).squeeze(0)
        result[str(cid)] = emb.cpu()

    logger.info(f"Precomputed {len(result)} FLAT cell embeddings (no KEGG)")
    return result, flat_encoder


# ═══════════════════════════════════════════════════════════════════════════════
# 5) Dataset
# ═══════════════════════════════════════════════════════════════════════════════
class AblationDataset(Dataset):
    """모든 ablation variant에 공통으로 사용되는 Dataset"""
    def __init__(self, df, colmap, drug_cache, cell_embed_cache):
        self.df = df.reset_index(drop=True)
        self.colmap = colmap
        self.drug_cache = drug_cache
        self.cell_embed_cache = cell_embed_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = str(row[self.colmap["smiles"]])
        cell_id = str(row[self.colmap["cell"]])

        drug_graph = self.drug_cache.get(smiles)
        if drug_graph is None:
            drug_graph = _safe_smiles_to_graph(smiles)
            if drug_graph is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            drug_graph = sanitize_pyg(drug_graph)

        cell_emb = self.cell_embed_cache.get(cell_id)
        if cell_emb is None:
            raise ValueError(f"Cell '{cell_id}' not in cache")

        y = torch.tensor(float(row[self.colmap["label"]]), dtype=torch.float32)

        # dose/time (condition 정보)
        minc = float(row[self.colmap["minc"]]) if self.colmap["minc"] else 0.0
        maxc = float(row[self.colmap["maxc"]]) if self.colmap["maxc"] else 0.0
        dose = torch.tensor(np.log1p(maxc), dtype=torch.float32)
        time_val = torch.tensor(np.log1p(72.0), dtype=torch.float32)  # GDSC default: 72h

        return {
            "drug_graph": drug_graph,
            "cell_emb": cell_emb,
            "y": y,
            "dose": dose,
            "time": time_val,
        }


def ablation_collate(batch):
    return {
        "drug_graph": Batch.from_data_list([b["drug_graph"] for b in batch]),
        "cell_emb": torch.stack([b["cell_emb"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "dose": torch.stack([b["dose"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6) Ablation Models
# ═══════════════════════════════════════════════════════════════════════════════

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AblationModel_NoKEGG(nn.Module):
    """
    ★ Ablation: w/o KEGG pathways
    
    Cell encoding: Flat MLP (trainable, 학습 중에도 업데이트됨)
    Drug encoding: DrugEncoder (GAT) → graph-level pooling
    Integration: Concat + MLP → IC50
    
    KEGG pathway graph 구조를 완전히 제거하고,
    raw gene expression을 MLP로 인코딩
    """
    def __init__(self, cfg, drug_in_dim, num_genes):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim_node

        # Token type embedding (DrugEncoder용)
        num_types = 2 + cfg.num_pathways
        self.token_type_embed = nn.Embedding(num_types, D)

        class _A: pass
        enc_args = _A()
        enc_args.dim_node = D
        enc_args.pe_dim = cfg.pe_dim
        enc_args.max_num_nodes = cfg.max_num_nodes
        enc_args.dropout_ratio = 0.2
        enc_args.dropout = 0.2

        # Drug Encoder (동일)
        self.drug_encoder = DrugEncoder(
            in_channels=drug_in_dim, hidden_channels=cfg.drug_hidden_dim,
            out_channels=cfg.drug_out_dim, token_type_embed=self.token_type_embed,
            drug_type_id=1, args=enc_args, heads=cfg.drug_heads, dropout=0.2,
        )

        # ★ Flat Cell Encoder (KEGG 대체)
        self.cell_encoder = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, D),
        )

        # Condition projections
        self.dose_proj = nn.Linear(1, D)
        self.time_proj = nn.Linear(1, D)

        # IC50 head: drug(D) + cell(D) + dose(D) + time(D) = 4D
        self.ic50_head = MLPHead(D * 4, hidden_dim=256, num_layers=3, dropout=0.2)

    def forward(self, drug_graph, cell_expr, dose, time_val):
        """
        cell_expr: [B, num_genes] raw gene expression (사전계산 안 됨 - 직접 encoding)
        """
        device = next(self.parameters()).device
        drug_graph = drug_graph.to(device)
        cell_expr = cell_expr.to(device)

        drug_emb = self.drug_encoder(drug_graph, return_graph=True)  # [B, D]
        cell_emb = self.cell_encoder(cell_expr)  # [B, D]

        dose_emb = self.dose_proj(dose.to(device).view(-1, 1))
        time_emb = self.time_proj(time_val.to(device).view(-1, 1))

        feat = torch.cat([drug_emb, cell_emb, dose_emb, time_emb], dim=-1)
        return self.ic50_head(feat)


class AblationModel_NoKEGG_Frozen(nn.Module):
    """
    ★ Ablation: w/o KEGG pathways (Frozen cell encoder version)
    
    Cell: 사전계산된 flat embedding 사용 (frozen)
    Drug: DrugEncoder (trainable)
    Integration: Concat + MLP → IC50
    
    이 버전은 cell embedding이 사전계산되어 있어 학습이 빠름
    """
    def __init__(self, cfg, drug_in_dim):
        super().__init__()
        D = cfg.dim_node

        num_types = 2 + cfg.num_pathways
        self.token_type_embed = nn.Embedding(num_types, D)

        class _A: pass
        enc_args = _A()
        enc_args.dim_node = D
        enc_args.pe_dim = cfg.pe_dim
        enc_args.max_num_nodes = cfg.max_num_nodes
        enc_args.dropout_ratio = 0.2
        enc_args.dropout = 0.2

        self.drug_encoder = DrugEncoder(
            in_channels=drug_in_dim, hidden_channels=cfg.drug_hidden_dim,
            out_channels=cfg.drug_out_dim, token_type_embed=self.token_type_embed,
            drug_type_id=1, args=enc_args, heads=cfg.drug_heads, dropout=0.2,
        )

        # dose/time
        self.dose_proj = nn.Linear(1, D)
        self.time_proj = nn.Linear(1, D)

        # IC50: drug(D) + cell(D) + dose(D) + time(D)
        self.ic50_head = MLPHead(D * 4, hidden_dim=256, num_layers=3, dropout=0.2)

    def forward(self, drug_graph, cell_emb, dose, time_val):
        device = next(self.parameters()).device
        drug_graph = drug_graph.to(device)
        cell_emb = cell_emb.to(device)

        drug_emb = self.drug_encoder(drug_graph, return_graph=True)
        dose_emb = self.dose_proj(dose.to(device).view(-1, 1))
        time_emb = self.time_proj(time_val.to(device).view(-1, 1))

        feat = torch.cat([drug_emb, cell_emb, dose_emb, time_emb], dim=-1)
        return self.ic50_head(feat)


class AblationModel_NoCondition(nn.Module):
    """
    ★ Ablation: w/o dose/time condition embeddings
    
    Full model 구조와 동일하되, dose/time embedding을 zero로 마스킹
    Cell: KEGG pathway (frozen, 사전계산)
    Drug: DrugEncoder (trainable)
    Integration: Concat + MLP (dose/time = 0) → IC50
    """
    def __init__(self, cfg, drug_in_dim):
        super().__init__()
        D = cfg.dim_node

        num_types = 2 + cfg.num_pathways
        self.token_type_embed = nn.Embedding(num_types, D)

        class _A: pass
        enc_args = _A()
        enc_args.dim_node = D
        enc_args.pe_dim = cfg.pe_dim
        enc_args.max_num_nodes = cfg.max_num_nodes
        enc_args.dropout_ratio = 0.2
        enc_args.dropout = 0.2

        self.drug_encoder = DrugEncoder(
            in_channels=drug_in_dim, hidden_channels=cfg.drug_hidden_dim,
            out_channels=cfg.drug_out_dim, token_type_embed=self.token_type_embed,
            drug_type_id=1, args=enc_args, heads=cfg.drug_heads, dropout=0.2,
        )

        # ★ dose/time projection은 있지만 forward에서 ZERO로 대체
        # IC50: drug(D) + cell(D) + zero(D) + zero(D)
        self.ic50_head = MLPHead(D * 4, hidden_dim=256, num_layers=3, dropout=0.2)

    def forward(self, drug_graph, cell_emb, dose, time_val):
        device = next(self.parameters()).device
        drug_graph = drug_graph.to(device)
        cell_emb = cell_emb.to(device)

        drug_emb = self.drug_encoder(drug_graph, return_graph=True)

        B = drug_emb.size(0)
        D = drug_emb.size(1)
        # ★ dose/time를 zero로!
        zero_dose = torch.zeros(B, D, device=device)
        zero_time = torch.zeros(B, D, device=device)

        feat = torch.cat([drug_emb, cell_emb, zero_dose, zero_time], dim=-1)
        return self.ic50_head(feat)


class AblationModel_NoTransformer(nn.Module):
    """
    ★ Ablation: w/o Transformer (concat + MLP)
    
    Full model에서 Transformer를 제거하고 단순 concat + MLP로 대체
    Cell: KEGG pathway (frozen, 사전계산)
    Drug: DrugEncoder (trainable, graph-level pooling)
    Integration: Concat + MLP → IC50 (Transformer cross-attention 없음)
    """
    def __init__(self, cfg, drug_in_dim):
        super().__init__()
        D = cfg.dim_node

        num_types = 2 + cfg.num_pathways
        self.token_type_embed = nn.Embedding(num_types, D)

        class _A: pass
        enc_args = _A()
        enc_args.dim_node = D
        enc_args.pe_dim = cfg.pe_dim
        enc_args.max_num_nodes = cfg.max_num_nodes
        enc_args.dropout_ratio = 0.2
        enc_args.dropout = 0.2

        self.drug_encoder = DrugEncoder(
            in_channels=drug_in_dim, hidden_channels=cfg.drug_hidden_dim,
            out_channels=cfg.drug_out_dim, token_type_embed=self.token_type_embed,
            drug_type_id=1, args=enc_args, heads=cfg.drug_heads, dropout=0.2,
        )

        # dose/time
        self.dose_proj = nn.Linear(1, D)
        self.time_proj = nn.Linear(1, D)

        # ★ MLP (Transformer 대체) — 더 깊게
        self.ic50_head = nn.Sequential(
            nn.Linear(D * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, drug_graph, cell_emb, dose, time_val):
        device = next(self.parameters()).device
        drug_graph = drug_graph.to(device)
        cell_emb = cell_emb.to(device)

        drug_emb = self.drug_encoder(drug_graph, return_graph=True)
        dose_emb = self.dose_proj(dose.to(device).view(-1, 1))
        time_emb = self.time_proj(time_val.to(device).view(-1, 1))

        feat = torch.cat([drug_emb, cell_emb, dose_emb, time_emb], dim=-1)
        return self.ic50_head(feat).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 7) Metrics / Loss
# ═══════════════════════════════════════════════════════════════════════════════
def pcc_torch(x, y):
    x, y = x.detach().cpu(), y.detach().cpu()
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float((x * y).mean().item())

def rmse_torch(pred, y):
    return float(torch.sqrt(F.mse_loss(pred.detach().cpu(), y.detach().cpu())).item())

def corr_loss(pred, y):
    x = (pred - pred.mean()) / (pred.std() + 1e-8)
    t = (y - y.mean()) / (y.std() + 1e-8)
    return 1.0 - (x * t).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 8) Training / Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, device, scaler, cfg,
                    y_mu=None, y_sd=None):
    model.train()
    base_loss_fn = nn.SmoothL1Loss(beta=0.5) if cfg.use_smoothl1 else nn.MSELoss()
    total_loss, n = 0.0, 0
    t0 = time.time()
    alpha = cfg.alpha_corr

    for step, batch in enumerate(loader, 1):
        drug_graph = batch["drug_graph"].to(device)
        cell_emb = batch["cell_emb"].to(device)
        y = batch["y"].to(device)
        dose = batch["dose"].to(device)
        time_val = batch["time"].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred = model(drug_graph, cell_emb, dose, time_val)

            if cfg.use_label_standardize and y_mu is not None and y_sd is not None:
                loss = base_loss_fn((pred - y_mu) / y_sd, (y - y_mu) / y_sd)
            else:
                loss = base_loss_fn(pred, y)

            if cfg.use_corr_loss and alpha > 0:
                loss = loss + alpha * corr_loss(pred, y)

        # ★ nan safety: skip batch if loss is nan
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            if step <= 3:
                logger.warning(f"  [step {step}] loss=nan/inf → skipping batch (pred range: {pred.min():.4f}~{pred.max():.4f})")
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        if cfg.log_interval and step % cfg.log_interval == 0:
            logger.info(f"  [step {step}/{len(loader)}] loss={total_loss/n:.4f}")

    dt = time.time() - t0
    logger.info(f"[Train] {dt:.1f}s, throughput={n/dt:.0f} samples/s")
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, cfg, y_mu=None, y_sd=None):
    model.eval()
    preds, gts = [], []

    for batch in loader:
        drug_graph = batch["drug_graph"].to(device)
        cell_emb = batch["cell_emb"].to(device)
        y = batch["y"].to(device)
        dose = batch["dose"].to(device)
        time_val = batch["time"].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred = model(drug_graph, cell_emb, dose, time_val)

        preds.append(pred.float().detach())
        gts.append(y.float().detach())

    preds = torch.cat(preds)
    gts = torch.cat(gts)
    return {
        "pcc": pcc_torch(preds, gts),
        "rmse": rmse_torch(preds, gts),
        "mse": float(F.mse_loss(preds.cpu(), gts.cpu()).item()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9) Main Run Function
# ═══════════════════════════════════════════════════════════════════════════════
def run_single_ablation(cfg: AblationConfig):
    variant = cfg.variant
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    logger.info("=" * 70)
    logger.info(f"★ Ablation: {variant} | Split: {cfg.split_mode}")
    logger.info("=" * 70)

    # ── Load Data ──
    df = pd.read_csv(cfg.csv_path, low_memory=False)
    colmap = build_colmap(df)
    label_col = colmap["label"]

    if str(label_col).upper() == "IC50":
        df[label_col] = np.log(df[label_col].astype(float) + 1e-12)

    if cfg.drop_na_label:
        df = df[~df[label_col].isna()].reset_index(drop=True)

    df = filter_invalid_smiles(df, colmap)

    train_df, val_df, test_df = split_dataframe(
        df, cfg.valid_ratio, cfg.test_ratio, cfg.split_seed, cfg.split_mode, colmap
    )
    logger.info(f"Split → train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    if cfg.subset_ratio < 0.999:
        for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
            n = max(1, int(len(d) * cfg.subset_ratio))
            d = d.sample(n=n, random_state=cfg.seed).reset_index(drop=True)
            if name == "train": train_df = d
            elif name == "val": val_df = d
            else: test_df = d
        logger.info(f"Subset → train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    # ── Drug cache ──
    drug_cache = None
    if cfg.drug_graph_cache and os.path.isfile(cfg.drug_graph_cache):
        drug_cache = torch.load(cfg.drug_graph_cache, map_location="cpu")
        logger.info(f"Loaded drug cache: {len(drug_cache)} entries")

    all_rows = pd.concat([train_df, val_df, test_df], ignore_index=True)
    drug_cache = warmup_drug_cache(all_rows, colmap, drug_cache)

    # Infer DRUG_FDIM
    DRUG_FDIM = None
    for s in df[colmap["smiles"]].dropna().astype(str).values[:500]:
        g = _safe_smiles_to_graph(s)
        if g is not None and hasattr(g, "x") and g.x.ndim == 2:
            DRUG_FDIM = int(g.x.shape[1])
            break
    if DRUG_FDIM is None:
        raise RuntimeError("Failed to infer DRUG_FDIM")
    logger.info(f"DRUG_FDIM = {DRUG_FDIM}")

    # ── Cell embeddings ──
    if variant == "no_kegg":
        # Flat MLP: raw gene expression 사용
        # 사전계산이 아닌, gene expression vector를 직접 cache
        gene_cols = colmap["genes"]
        num_genes = len(gene_cols)
        cell_col = colmap["cell"]

        # Cell별 unique gene expression
        cell_expr_dict = {}
        for cid in df[cell_col].astype(str).unique():
            mask = df[cell_col].astype(str) == cid
            expr = df.loc[mask, gene_cols].iloc[0].values.astype(np.float32)
            cell_expr_dict[cid] = torch.tensor(expr, dtype=torch.float32)

        # ★ Gene expression standardization (per-gene z-score)
        # Raw gene expression 값이 매우 크면 MLP 출력이 overflow → nan
        # Filter out NaN cells before computing stats
        nan_cell_ids = [cid for cid, expr in cell_expr_dict.items() if torch.isnan(expr).any()]
        if nan_cell_ids:
            logger.warning(f"⚠️ {len(nan_cell_ids)} cells have NaN gene expr → filling with 0: {nan_cell_ids[:5]}")
            for cid in nan_cell_ids:
                cell_expr_dict[cid] = torch.zeros_like(cell_expr_dict[cid])
        valid_exprs = [expr for expr in cell_expr_dict.values() if not torch.isnan(expr).any()]
        all_expr = torch.stack(valid_exprs) if valid_exprs else torch.stack(list(cell_expr_dict.values()))
        gene_mean = all_expr.mean(dim=0)  # [num_genes]
        gene_std = all_expr.std(dim=0) + 1e-8  # [num_genes]
        logger.info(f"Gene expr stats BEFORE standardize: mean={all_expr.mean():.4f}, std={all_expr.std():.4f}, "
                     f"min={all_expr.min():.4f}, max={all_expr.max():.4f}")

        # Standardize
        for cid in cell_expr_dict:
            cell_expr_dict[cid] = (cell_expr_dict[cid] - gene_mean) / gene_std

        all_expr_z = torch.stack(list(cell_expr_dict.values()))
        logger.info(f"Gene expr stats AFTER standardize: mean={all_expr_z.mean():.4f}, std={all_expr_z.std():.4f}, "
                     f"min={all_expr_z.min():.4f}, max={all_expr_z.max():.4f}")

        # 사전계산 (random init이므로 일관성을 위해 seed 고정 후)
        set_seed(cfg.seed)
        flat_encoder = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, cfg.dim_node),
        ).to(device)
        flat_encoder.eval()

        cell_embed_cache = {}
        nan_count = 0
        with torch.no_grad():
            for cid, expr in cell_expr_dict.items():
                # ★ float32로 강제 (AMP 우회)
                emb = flat_encoder(expr.unsqueeze(0).to(device).float()).squeeze(0)
                # nan 체크
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    emb = torch.zeros(cfg.dim_node)
                    nan_count += 1
                cell_embed_cache[cid] = emb.cpu()

        if nan_count > 0:
            logger.warning(f"⚠️ {nan_count} cells had nan/inf embeddings → replaced with zeros")
        logger.info(f"Flat cell embeddings: {len(cell_embed_cache)} cells, {num_genes} genes → {cfg.dim_node}D")

        # ★ Embedding 통계 확인
        all_emb = torch.stack(list(cell_embed_cache.values()))
        logger.info(f"Cell embedding stats: mean={all_emb.mean():.4f}, std={all_emb.std():.4f}, "
                     f"nan_count={torch.isnan(all_emb).sum().item()}")

    else:
        # KEGG pathway 기반 (no_condition, no_transformer 포함)
        cell_embed_cache_path = os.path.join(cfg.save_dir, f"cell_embed_kegg_{cfg.split_mode}.pth")

        if os.path.isfile(cell_embed_cache_path):
            cell_embed_cache = torch.load(cell_embed_cache_path, map_location="cpu")
            logger.info(f"Loaded KEGG cell embeddings: {len(cell_embed_cache)}")
        else:
            if cfg.cell_graph_cache and os.path.isfile(cfg.cell_graph_cache):
                cell_graph_cache = torch.load(cfg.cell_graph_cache, map_location="cpu")
            else:
                raise RuntimeError(f"cell_graph_cache not found: {cfg.cell_graph_cache}")

            # Sanitize
            sanitized = {}
            for cid, seq in cell_graph_cache.items():
                if isinstance(seq, (list, tuple)) and len(seq) > 0:
                    graphs = [g for g in seq if isinstance(g, Data)]
                    if graphs:
                        sanitized[str(cid)] = [sanitize_pyg(g) for g in graphs]
            cell_graph_cache = sanitized
            logger.info(f"Sanitized cell graphs: {len(cell_graph_cache)} cells")

            cell_embed_cache = precompute_cell_embeddings_kegg(cell_graph_cache, cfg, device)

            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save(cell_embed_cache, cell_embed_cache_path)
            logger.info(f"Saved KEGG cell embeddings to {cell_embed_cache_path}")

    # Filter cells
    valid_cells = set(cell_embed_cache.keys())
    cell_col = colmap["cell"]
    train_df = train_df[train_df[cell_col].astype(str).isin(valid_cells)].reset_index(drop=True)
    val_df = val_df[val_df[cell_col].astype(str).isin(valid_cells)].reset_index(drop=True)
    test_df = test_df[test_df[cell_col].astype(str).isin(valid_cells)].reset_index(drop=True)
    logger.info(f"After cell filter → train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    # ── Model ──
    if variant == "no_kegg":
        model = AblationModel_NoKEGG_Frozen(cfg, DRUG_FDIM).to(device)
    elif variant == "no_condition":
        model = AblationModel_NoCondition(cfg, DRUG_FDIM).to(device)
    elif variant == "no_transformer":
        model = AblationModel_NoTransformer(cfg, DRUG_FDIM).to(device)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {variant} | Total: {total_params:,} | Trainable: {trainable_params:,}")

    # ── Dataloaders ──
    ds_kwargs = dict(colmap=colmap, drug_cache=drug_cache, cell_embed_cache=cell_embed_cache)
    tr_ds = AblationDataset(train_df, **ds_kwargs)
    va_ds = AblationDataset(val_df, **ds_kwargs)
    te_ds = AblationDataset(test_df, **ds_kwargs)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           collate_fn=ablation_collate)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           collate_fn=ablation_collate)
    te_loader = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           collate_fn=ablation_collate)

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # Label stats
    y_mu = float(train_df[colmap["label"]].mean()) if cfg.use_label_standardize else None
    y_sd = float(train_df[colmap["label"]].std() + 1e-8) if cfg.use_label_standardize else None

    # ── Training Loop ──
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"ablation_{variant}_{cfg.split_mode}_seed{cfg.seed}_{ts}"
    ckpt_path = os.path.join(cfg.save_dir, f"{run_id}.pth")

    best_val_pcc = -1.0
    best_state = None
    no_improve = 0

    logger.info(f"\n{'='*70}")
    logger.info(f"Training [{variant}] ...")
    logger.info(f"{'='*70}")

    for epoch in range(1, cfg.epochs + 1):
        alpha_now = cfg.alpha_corr if epoch >= 3 else 0.0
        cfg_epoch = AblationConfig(**{**asdict(cfg), 'alpha_corr': alpha_now})

        tr_loss = train_one_epoch(model, tr_loader, optimizer, device, scaler, cfg_epoch,
                                   y_mu=y_mu, y_sd=y_sd)
        val_metrics = evaluate(model, va_loader, device, cfg, y_mu=y_mu, y_sd=y_sd)
        scheduler.step()

        logger.info(
            f"[Ep {epoch:02d}] loss={tr_loss:.4f} | "
            f"val_pcc={val_metrics['pcc']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f}"
        )

        improved = val_metrics["pcc"] > best_val_pcc + cfg.early_stopping_min_delta
        if improved:
            best_val_pcc = val_metrics["pcc"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "config": asdict(cfg),
            }
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save(best_state, ckpt_path)
            logger.info(f"  ✅ SAVE (val_pcc={best_val_pcc:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if cfg.early_stopping and no_improve >= cfg.early_stopping_patience:
            logger.info(f"⚠️ Early stop at epoch {epoch}")
            break

    # ── Test ──
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    test_metrics = evaluate(model, te_loader, device, cfg, y_mu=y_mu, y_sd=y_sd)

    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 [{variant}] Final Results ({cfg.split_mode} split):")
    logger.info(f"   Best Val PCC : {best_val_pcc:.4f}")
    logger.info(f"   Test PCC     : {test_metrics['pcc']:.4f}")
    logger.info(f"   Test RMSE    : {test_metrics['rmse']:.4f}")
    logger.info(f"{'='*70}")

    # Save report
    report = {
        "variant": variant,
        "split_mode": cfg.split_mode,
        "seed": cfg.seed,
        "best_val_pcc": best_val_pcc,
        "test_pcc": test_metrics["pcc"],
        "test_rmse": test_metrics["rmse"],
        "test_mse": test_metrics["mse"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "ckpt_path": ckpt_path,
    }
    report_path = os.path.join(cfg.save_dir, f"{run_id}.report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# 10) Compare All Variants
# ═══════════════════════════════════════════════════════════════════════════════
def run_all_ablations(cfg: AblationConfig, variants=None):
    if variants is None:
        variants = ["no_kegg", "no_condition", "no_transformer"]

    results = []
    for v in variants:
        try:
            vcfg = AblationConfig(**{**asdict(cfg), 'variant': v})
            r = run_single_ablation(vcfg)
            results.append(r)
        except Exception as e:
            logger.error(f"[{v}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"variant": v, "error": str(e)})

    # ── Summary Table ──
    # 기존 결과 추가 (졸업논문)
    known_results = [
        {"variant": "full (w/ pretrain)", "test_pcc": 0.9108, "test_rmse": 1.0872, "note": "졸업논문"},
        {"variant": "w/o pretrain", "test_pcc": 0.8916, "test_rmse": 1.0915, "note": "졸업논문"},
    ]

    print("\n" + "=" * 80)
    print("★★★ ABLATION STUDY RESULTS ★★★")
    print(f"Split: {cfg.split_mode} | Seed: {cfg.seed}")
    print("=" * 80)
    print(f"{'Variant':<30} {'Test PCC':>10} {'Test RMSE':>10} {'△PCC':>10} {'Note':<15}")
    print("-" * 80)

    full_pcc = 0.9108  # baseline

    for r in known_results:
        delta = r["test_pcc"] - full_pcc
        print(f"{r['variant']:<30} {r['test_pcc']:>10.4f} {r['test_rmse']:>10.4f} {delta:>+10.4f} {r.get('note',''):<15}")

    for r in results:
        if "error" in r:
            print(f"{r['variant']:<30} {'ERROR':>10} {'':>10} {'':>10} {r['error'][:30]}")
        else:
            delta = r["test_pcc"] - full_pcc
            print(f"{r['variant']:<30} {r['test_pcc']:>10.4f} {r['test_rmse']:>10.4f} {delta:>+10.4f}")

    print("=" * 80)

    # Save combined report
    all_results = known_results + results
    combined_path = os.path.join(cfg.save_dir, f"ablation_combined_{cfg.split_mode}.json")
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Combined report saved to {combined_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 11) Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study for DrugResponseTransformer")
    parser.add_argument("--variant", type=str, default="all",
                        help="Variant: no_kegg, no_condition, no_transformer, all, or comma-separated")
    parser.add_argument("--split", type=str, default="mixed", help="mixed or drug_blind")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=float, default=1.0, help="Data subset ratio (for quick testing)")
    parser.add_argument("--csv_path", type=str, default="GDSC_with_SMILES_BGE_entrez_KEGG_fixed.csv")
    parser.add_argument("--save_dir", type=str, default="ablation_results")
    args = parser.parse_args()

    cfg = AblationConfig(
        variant=args.variant,
        csv_path=args.csv_path,
        save_dir=args.save_dir,
        split_mode=args.split,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        subset_ratio=args.subset,
    )

    if args.variant == "all":
        run_all_ablations(cfg)
    elif "," in args.variant:
        variants = [v.strip() for v in args.variant.split(",")]
        run_all_ablations(cfg, variants=variants)
    else:
        run_single_ablation(cfg)
