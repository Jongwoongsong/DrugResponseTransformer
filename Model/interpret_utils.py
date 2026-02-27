# Model/interpret_utils.py
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from Model.CellLine_graph import create_cell_line_graph   # KEGG 그래프 생성용


# ─────────────────────────────────────────────────────────
# 0) PyG Batch → 각 drug 그래프 atom 수 복원
# ─────────────────────────────────────────────────────────
def _count_atoms_per_graph(drug_batch: Batch) -> List[int]:
    """
    PyG Batch에서 각 그래프(=drug)의 atom 수를 복원.
    drug_batch.batch: [N_total] 에서 0..B-1 인덱스.
    """
    if not hasattr(drug_batch, "batch"):
        # 단일 graph가 들어오는 경우 대비
        return [int(getattr(drug_batch, "num_nodes", 0))]

    bvec = drug_batch.batch
    if bvec.numel() == 0:
        return []

    B = int(bvec.max().item()) + 1
    counts = []
    for b in range(B):
        counts.append(int((bvec == b).sum().item()))
    return counts


# ─────────────────────────────────────────────────────────
# 1) node-cache용 cell_id → gene_id 리스트 매핑 만들기
#    (KEGG 그래프 + landmark_order 기준)
# ─────────────────────────────────────────────────────────
def build_cell_gene_map_for_nodecache(
    basal_df,
    kegg_dir: str,
    all_cells: List[str],
    landmark_order: List[str],
    num_pathways: int = 31,
) -> Dict[str, List[str]]:
    """
    node-cache에서는 cell_embed[cid]가 [N_gene, D] 형태이고,
    각 토큰이 어떤 gene에 해당하는지 정보가 사라져 있음.

    precompute 시점 로직을 그대로 재현해:
      - create_cell_line_graph(basal_df, kegg_dir, cid, max_pathways)
        로 해당 셀의 KEGG 그래프를 만들고
      - 그래프들의 node_ids를 모두 모아 gene_set 구성
      - landmark_order를 순서대로 훑으면서 gene_set에 있는 것만 남김
      - max_nodes_for_cache에서 잘렸던 부분은, 해석 시에는
        실제 토큰 길이(= (~cell_pad_mask).sum())에 맞춰 앞에서부터 사용

    반환:
      cell_to_genes[cid] = [gene_id_0, gene_id_1, ..., gene_id_{N-1}]
    """
    cell_to_genes: Dict[str, List[str]] = {}
    uniq_cells = sorted(set(map(str, all_cells)))
    print(f"[INTERP] build_cell_gene_map_for_nodecache: {len(uniq_cells)} cells")

    for i, cid in enumerate(uniq_cells, start=1):
        if i % 50 == 0:
            print(f"[INTERP]   cell {i}/{len(uniq_cells)}: {cid}")

        try:
            graphs, _ = create_cell_line_graph(
                basal_df,
                kegg_dir,
                cid,
                max_pathways=num_pathways,
            )
        except Exception as e:
            print(f"[INTERP][WARN] create_cell_line_graph failed for cell '{cid}': {e}")
            continue

        if not graphs:
            continue

        gene_set = set()
        for g in graphs:
            node_ids = getattr(g, "node_ids", None)
            if node_ids is None:
                continue
            gene_set.update(map(str, node_ids))

        if not gene_set:
            continue

        # precompute 시점과 동일한 순서:
        # landmark_order 중에서 실제 그래프에 존재하는 gene만 순서대로
        genes_for_cell = [str(g) for g in landmark_order if str(g) in gene_set]
        if not genes_for_cell:
            continue

        cell_to_genes[str(cid)] = genes_for_cell

    if not cell_to_genes:
        print("[INTERP][WARN] cell_to_genes is empty. Check KEGG / basal_df / landmark_order.")
    else:
        print(f"[INTERP] cell_to_genes built: {len(cell_to_genes)} cells")

    return cell_to_genes


# ─────────────────────────────────────────────────────────
# 2) node-cache 기반 IC50 해석
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def explain_ic50_batch(
    model,
    batch: Dict[str, Any],
    cell_to_gene_ids: Dict[str, List[str]],
    layer_idx: int = -1,
    top_k_genes: int = 20,
    top_k_atoms: int = 10,
) -> List[Dict[str, Any]]:
    """
    node-cache 기반 fastcache IC50 모델에 대한 한 배치 해석.

    입력:
      - model: DrugResponseTransformer (task="ic50")
      - batch: train_ic50_fastcache.py의 collate_fn_with_node_cache가 만든 dict
          {
            "drug_graph": Batch,
            "cell_embed": [B, Nmax, D],
            "cell_pad_mask": [B, Nmax] (True=PAD),
            "time": [B],
            "dose": [B],
            "y": [B],
            "meta": { "cell_id": [...], "drug_id": [...] }
          }
      - cell_to_gene_ids: cell_id → [gene_id,...] 매핑
      - layer_idx: attention 해석할 Transformer layer index (기본 -1 = 마지막)
      - top_k_genes: gene 중요도 Top-K
      - top_k_atoms: drug atom 중요도 Top-K

    반환 (배치 길이만큼 list):
      [{
        "drug_id": str,
        "cell_id": str,
        "true_ln_ic50": float,
        "pred_ln_ic50": float,
        "top_atoms_by_alpha": List[{"atom_idx": int, "alpha": float}],
        "top_atoms_by_cond": List[{"atom_idx": int, "score": float}],
        "top_genes_by_alpha": List[{"gene": str, "alpha": float}],
        "top_genes_by_cond": List[{"gene": str, "score": float}],
      }, ...]
    """
    device = next(model.parameters()).device
    model.eval()

    drug_graph = batch["drug_graph"]
    time = batch["time"].to(device)
    dose = batch["dose"].to(device)
    cell_embed = batch.get("cell_embed", None)
    cell_pad_mask = batch.get("cell_pad_mask", None)
    cell_ids = [str(c) for c in batch["meta"]["cell_id"]]
    drug_ids = [str(d) for d in batch["meta"]["drug_id"]]
    y_true = batch["y"].detach().cpu()

    # ---- 1) 토큰 배치 구성 (모델 내부 헬퍼 그대로 사용) ----
    token_batch = model._encode_batch(
        drug_graph=drug_graph,
        cell_graph_seq=None,
        cell_embed=cell_embed,
        cell_pad_mask=cell_pad_mask,
    )
    token_batch = model._add_condition_tokens(token_batch, time, dose)
    token_batch = model._align_mask_with_tokens(token_batch)

    token_emb = token_batch.tokens       # [B, T, D]
    token_mask = token_batch.mask        # [B, T]

    # ---- 2) Transformer forward (attention 반환) ----
    out, attn_maps = model.Transformer(
        token_emb=token_emb,
        token_mask=token_mask,
        return_attn=True,
    )
    # out: [B, T, D]
    # attn_maps: List[ Tensor[B, H, T, T] ]  (레이어별)

    # ---- 3) IC50 예측 재계산 (forward와 동일 로직) ----
    body_out = out[:, 2:, :]               # [B, T_body, D] (dose/time 제외)
    body_mask = token_mask[:, 2:]          # [B, T_body]

    # pooling α
    scores = torch.matmul(body_out, model.pool_query)  # [B, T_body]
    scores = scores.masked_fill(~body_mask, float("-inf"))
    alpha = torch.softmax(scores, dim=1)               # [B, T_body]

    # summary 토큰
    summary = torch.einsum("bt,btd->bd", alpha, body_out)  # [B, D]

    # dose/time 임베딩
    dose_emb = model.dose_proj(dose.view(-1, 1))       # [B, D]
    time_emb = model.time_proj(time.view(-1, 1))       # [B, D]

    feat = torch.cat([summary, dose_emb, time_emb], dim=1)  # [B, 3D]
    ic50_pred = model.ic50_head(feat).squeeze(-1).detach().cpu()  # [B]

    # ---- 4) cond-attn (dose/time → body 토큰) ----
    if isinstance(attn_maps, (list, tuple)):
        attn_last = attn_maps[layer_idx]   # [B, H, T, T]
    else:
        attn_last = attn_maps              # [B, H, T, T]

    attn_last = attn_last.to(device)
    # head 평균: [B, T, T]
    attn_mean = attn_last.mean(dim=1)

    # cond(0,1) → 전체 토큰 attention 합산 후 body 부분만
    cond_to_all = attn_mean[:, 0, :] + attn_mean[:, 1, :]   # [B, T]
    cond_to_body = cond_to_all[:, 2:]                       # [B, T_body]
    cond_to_body = cond_to_body * body_mask.float()

    # ---- 5) drug atom 수, cell gene 토큰 수 복원 ----
    B, T_body, _ = body_out.shape
    atom_counts = _count_atoms_per_graph(drug_graph)  # 길이 B

    if cell_embed is not None:
        if cell_pad_mask is not None:
            # cell_pad_mask: True=PAD → ~pad가 유효 토큰
            cell_valid_counts = (~cell_pad_mask).sum(dim=1).tolist()
        else:
            cell_valid_counts = [cell_embed.size(1)] * B
    else:
        cell_valid_counts = [0] * B

    results: List[Dict[str, Any]] = []

    for b in range(B):
        Ta = int(atom_counts[b]) if b < len(atom_counts) else 0
        P_candidate = int(cell_valid_counts[b]) if b < len(cell_valid_counts) else 0

        # 실제 body_mask 상 유효 토큰 수
        valid_body = int(body_mask[b].sum().item())
        if valid_body < Ta + P_candidate:
            P_candidate = max(0, valid_body - Ta)
        P = P_candidate

        alpha_b = alpha[b]          # [T_body]
        cond_b  = cond_to_body[b]   # [T_body]

        # body 토큰 순서:
        #   0..Ta-1: drug atoms
        #   Ta..Ta+P-1: cell gene tokens
        alpha_drug = alpha_b[:Ta]
        cond_drug  = cond_b[:Ta]
        alpha_gene = alpha_b[Ta:Ta+P]
        cond_gene  = cond_b[Ta:Ta+P]

        # gene 이름 복원 (node-cache에서 사라졌던 부분)
        cid = cell_ids[b]
        genes_all = cell_to_gene_ids.get(cid, [])
        # precompute에서도 max_nodes_for_cache로 잘랐으므로, 앞에서부터 P개만 사용
        genes_for_sample = genes_all[:P]

        # --- drug atoms Top-K ---
        k_atom = min(top_k_atoms, Ta)
        top_atoms_by_alpha, top_atoms_by_cond = [], []
        if k_atom > 0 and Ta > 0:
            a_scores, a_indices = torch.topk(alpha_drug, k=k_atom)
            c_scores, c_indices = torch.topk(cond_drug, k=k_atom)
            top_atoms_by_alpha = [
                {"atom_idx": int(i.item()), "alpha": float(s.item())}
                for s, i in zip(a_scores, a_indices)
            ]
            top_atoms_by_cond = [
                {"atom_idx": int(i.item()), "score": float(s.item())}
                for s, i in zip(c_scores, c_indices)
            ]

        # --- gene tokens Top-K ---
        k_gene = min(top_k_genes, P)
        top_genes_by_alpha, top_genes_by_cond = [], []
        if k_gene > 0 and P > 0 and len(genes_for_sample) == P:
            # alpha 기준
            g_a_scores, g_a_indices = torch.topk(alpha_gene, k=k_gene)
            for s, idx in zip(g_a_scores, g_a_indices):
                gi = int(idx.item())
                if gi < P:
                    gname = genes_for_sample[gi]
                    top_genes_by_alpha.append(
                        {"gene": str(gname), "alpha": float(s.item())}
                    )
            # cond-attn 기준
            g_c_scores, g_c_indices = torch.topk(cond_gene, k=k_gene)
            for s, idx in zip(g_c_scores, g_c_indices):
                gi = int(idx.item())
                if gi < P:
                    gname = genes_for_sample[gi]
                    top_genes_by_cond.append(
                        {"gene": str(gname), "score": float(s.item())}
                    )

        results.append(
            {
                "drug_id": drug_ids[b],
                "cell_id": cid,
                "true_ln_ic50": float(y_true[b].item()),
                "pred_ln_ic50": float(ic50_pred[b].item()),
                "top_atoms_by_alpha": top_atoms_by_alpha,
                "top_atoms_by_cond": top_atoms_by_cond,
                "top_genes_by_alpha": top_genes_by_alpha,
                "top_genes_by_cond": top_genes_by_cond,
            }
        )

    return results