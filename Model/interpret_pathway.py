# Model/interpret_utils.py

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


def _count_atoms_per_graph(drug_batch: Batch) -> List[int]:
    """
    PyG Batch에서 각 그래프(=drug)의 atom 수를 복원.
    drug_batch.batch: [N_total] 에서 0..B-1 인덱스.
    """
    if not hasattr(drug_batch, "batch"):
        # 혹시 단일 graph가 들어오는 경우 대비
        return [int(getattr(drug_batch, "num_nodes", 0))]

    bvec = drug_batch.batch
    if bvec.numel() == 0:
        return []

    B = int(bvec.max().item()) + 1
    counts = []
    for b in range(B):
        counts.append(int((bvec == b).sum().item()))
    return counts


@torch.no_grad()
def explain_ic50_batch(
    model,
    drug_graph: Batch,
    cell_graph_seq,
    time: torch.Tensor,
    dose: torch.Tensor,
    cell_embed: Optional[torch.Tensor] = None,
    cell_pad_mask: Optional[torch.Tensor] = None,
    layer_idx: int = -1,
    top_k_genes: int = 20,   # 이름만 남겨둔 파라미터, pathway에도 그대로 사용
) -> List[Dict[str, Any]]:
    """
    IC50 파인튜닝 모델에 대해 한 배치 단위 해석을 수행.

    - fast-cache 모드 (cell_embed != None)를 기준으로 작성:
      * dose / time 토큰 + drug atom 토큰 + pathway 토큰들로 구성된
        Transformer 입력에서,
      * pooling α, cond-attn을 이용해 pathway 중요도를 계산한다.

    반환: 각 샘플에 대해
      {
        "ic50_pred": float,
        "top_pathways_by_alpha": List[(path_idx, score)],
        "top_pathways_by_cond_attn": List[(path_idx, score)],
      }
    """
    device = next(model.parameters()).device
    model.eval()

    # ---- 1) 토큰 배치 구성 (모델 내부 헬퍼 재사용) ----
    token_batch = model._encode_batch(
        drug_graph=drug_graph,
        cell_graph_seq=cell_graph_seq,
        cell_embed=cell_embed,
        cell_pad_mask=cell_pad_mask,
    )
    token_batch = model._add_condition_tokens(token_batch, time, dose)
    token_batch = model._align_mask_with_tokens(token_batch)

    # token_emb: [B, T, D], token_mask: [B, T]
    token_emb = token_batch.tokens
    token_mask = token_batch.mask

    # ---- 2) Transformer forward (attention 반환) ----
    out, attn_maps = model.Transformer(
        token_emb=token_emb,
        token_mask=token_mask,
        return_attn=True,
    )
    # out: [B, T, D]
    # attn_maps: List[ Tensor[B, H, T, T] ] (레이어별)

    # ---- 3) IC50 예측 재계산 (forward와 동일 로직) ----
    #   body_out: dose/time 제거한 토큰들만
    body_out = out[:, 2:, :]               # [B, T_body, D]
    body_mask = token_mask[:, 2:]          # [B, T_body]

    # pooling α (pool_query 기반)
    scores = torch.matmul(body_out, model.pool_query)  # [B, T_body]
    scores = scores.masked_fill(~body_mask, float("-inf"))
    alpha = torch.softmax(scores, dim=1)               # [B, T_body]

    # summary 토큰
    summary = torch.einsum("bt,btd->bd", alpha, body_out)  # [B, D]

    # dose/time 임베딩
    dose_emb = model.dose_proj(dose.view(-1, 1))       # [B, D]
    time_emb = model.time_proj(time.view(-1, 1))       # [B, D]

    feat = torch.cat([summary, dose_emb, time_emb], dim=1)  # [B, 3D]
    ic50 = model.ic50_head(feat).squeeze(-1)                # [B]

    # ---- 4) cond-attn (dose/time → body 토큰) 계산 ----
    if isinstance(attn_maps, (list, tuple)):
        attn_last = attn_maps[layer_idx]   # [B, H, T, T]
    else:
        # 혹시 한 레이어만 반환하는 구현이라면 직접 사용
        attn_last = attn_maps              # [B, H, T, T]

    attn_last = attn_last.to(device)
    # head 평균: [B, T, T]
    attn_mean = attn_last.mean(dim=1)

    # cond(0,1) → 전체 토큰 attention 을 합산 후, body 부분만 취함
    cond_to_all = attn_mean[:, 0, :] + attn_mean[:, 1, :]   # [B, T]
    cond_to_body = cond_to_all[:, 2:]                       # [B, T_body]

    # mask 적용
    cond_to_body = cond_to_body * body_mask.float()         # invalid는 0으로

    # ---- 5) fast-cache 모드: pathway 단위 중요도 계산 ----
    B, T_body, _ = body_out.shape
    results: List[Dict[str, Any]] = []

    # 각 sample별 drug atom 개수 (Ta_b)
    atom_counts = _count_atoms_per_graph(drug_graph)  # 길이 B 리스트

    # cell_embed / cell_pad_mask로 pathway 개수(P_b) 계산
    if cell_embed is not None:
        # cell_embed: [B, P_max, D], cell_pad_mask: [B, P_max] (True=PAD)
        if cell_pad_mask is not None:
            cell_valid_counts = (~cell_pad_mask).sum(dim=1).tolist()  # 길이 B
        else:
            # pad mask가 없으면 전부 유효 경로라고 가정
            cell_valid_counts = [cell_embed.size(1)] * B
    else:
        # 이 경우는 gene-level 토큰이 있는 모드일 수 있지만,
        # fast-cache IC50에서만 사용할 거라면 0경로로 처리
        cell_valid_counts = [0] * B

    for b in range(B):
        Ta = int(atom_counts[b]) if b < len(atom_counts) else 0
        P  = int(cell_valid_counts[b]) if b < len(cell_valid_counts) else 0

        # body_mask 기준 실제 유효 토큰 수 (검증용)
        valid_body = int(body_mask[b].sum().item())
        # 이론상 valid_body = Ta + P 여야 함 (패딩 제외)
        # 혹시 mismatch가 있으면 가능한 범위 내에서 잘라줌
        if valid_body < Ta + P:
            # 경로 개수를 줄이되 음수가 되지는 않도록
            P = max(0, valid_body - Ta)

        # 아무 pathway도 없으면 skip
        if P <= 0:
            results.append(
                {
                    "ic50_pred": float(ic50[b].item()),
                    "top_pathways_by_alpha": [],
                    "top_pathways_by_cond_attn": [],
                }
            )
            continue

        # body 토큰들에서 drug 부분 / pathway 부분 분리
        alpha_b = alpha[b]          # [T_body]
        cond_b  = cond_to_body[b]   # [T_body]

        # 유효 body 토큰 인덱스 순서에서:
        #   0..Ta-1: drug, Ta..Ta+P-1: pathway
        start = Ta
        end   = Ta + P

        alpha_path = alpha_b[start:end]  # [P]
        cond_path  = cond_b[start:end]   # [P]

        # 상위 top_k_genes (여기선 pathway 개수에 대해 적용)
        k = min(top_k_genes, P)

        # alpha 기준 top-k
        if k > 0:
            alpha_scores, alpha_indices = torch.topk(alpha_path, k=k)
            top_by_alpha = [
                (int(idx.item()), float(score.item()))
                for score, idx in zip(alpha_scores, alpha_indices)
            ]
        else:
            top_by_alpha = []

        # cond-attn 기준 top-k
        if k > 0:
            cond_scores, cond_indices = torch.topk(cond_path, k=k)
            top_by_cond = [
                (int(idx.item()), float(score.item()))
                for score, idx in zip(cond_scores, cond_indices)
            ]
        else:
            top_by_cond = []

        results.append(
            {
                "ic50_pred": float(ic50[b].item()),
                "top_pathways_by_alpha": top_by_alpha,
                "top_pathways_by_cond_attn": top_by_cond,
            }
        )

    return results