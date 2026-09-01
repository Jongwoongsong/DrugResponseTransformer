"""Token-preserving faithful model with unique-cell deduplication.

Scientific contract
-------------------
* Drug branch remains atom-token based. No drug pooling shortcut is used.
* Cell branch performs full KEGG message passing, keeps 712 landmark
  occurrences, and aggregates them into the same ordered 268 gene tokens.
* Repeated cell lines in a mini-batch are encoded once, then expanded by a
  differentiable index_select. No cell representation is pooled globally.
* Optional pathway batching changes only GPU execution, not token identity.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch_geometric.utils import softmax as pyg_segment_softmax

from Model.DrugResponseTransformer import ModelConfig, TokenBatch
from Model.DrugResponseTransformerFaithful import (
    FaithfulDrugResponseTransformer,
)


class FaithfulTokenDedupDrugResponseTransformer(
    FaithfulDrugResponseTransformer
):
    """Faithful atom–gene model with token-preserving unique-cell encoding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._occurrence_gene_ids_reference: Optional[Tuple] = None
        self._occurrence_gene_index_cpu: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Vectorized 712 occurrence -> 268 unique-gene aggregation.
    # ------------------------------------------------------------------
    def _get_occurrence_gene_index(
        self,
        occurrence_gene_ids: List[List[Optional[str]]],
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not occurrence_gene_ids:
            raise RuntimeError("Empty occurrence_gene_ids batch")

        reference = tuple(occurrence_gene_ids[0][:token_count])
        same_layout = all(
            tuple(row[:token_count]) == reference
            for row in occurrence_gene_ids
        )

        if same_layout:
            if self._occurrence_gene_ids_reference != reference:
                indices = [
                    self.gene2idx.get(str(gene_id), -1)
                    if gene_id is not None
                    else -1
                    for gene_id in reference
                ]
                self._occurrence_gene_ids_reference = reference
                self._occurrence_gene_index_cpu = torch.tensor(
                    indices,
                    dtype=torch.long,
                    device="cpu",
                )

            if self._occurrence_gene_index_cpu is None:
                raise RuntimeError("Occurrence-index cache was not initialized")

            return self._occurrence_gene_index_cpu.to(
                device=device,
                non_blocking=True,
            ).unsqueeze(0).expand(len(occurrence_gene_ids), -1)

        # Safe fallback for variable layouts or padding.
        rows = []
        for row in occurrence_gene_ids:
            row_indices = [
                self.gene2idx.get(str(gene_id), -1)
                if gene_id is not None
                else -1
                for gene_id in row[:token_count]
            ]
            if len(row_indices) < token_count:
                row_indices.extend([-1] * (token_count - len(row_indices)))
            rows.append(row_indices)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def _aggregate_occurrences_to_unique(
        self,
        occurrence_tokens: torch.Tensor,
        occurrence_mask: torch.Tensor,
        occurrence_gene_ids: List[List[Optional[str]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        if occurrence_tokens.dim() != 3:
            raise ValueError(
                "occurrence_tokens must be [B,T,D], got "
                f"{tuple(occurrence_tokens.shape)}"
            )
        if occurrence_mask.shape != occurrence_tokens.shape[:2]:
            raise ValueError(
                "occurrence_mask shape mismatch: "
                f"{tuple(occurrence_mask.shape)} vs "
                f"{tuple(occurrence_tokens.shape[:2])}"
            )

        batch_size, token_count, dim = occurrence_tokens.shape
        num_genes = self.num_genes
        device = occurrence_tokens.device
        dtype = occurrence_tokens.dtype

        gene_index = self._get_occurrence_gene_index(
            occurrence_gene_ids,
            token_count=token_count,
            device=device,
        )
        valid = occurrence_mask.to(torch.bool) & (gene_index >= 0)
        if not bool(valid.any()):
            raise RuntimeError(
                "No landmark-gene occurrences were produced. "
                "Check pathway node_ids and landmark identifiers."
            )

        batch_index = torch.arange(
            batch_size,
            device=device,
            dtype=torch.long,
        ).unsqueeze(1).expand(batch_size, token_count)

        flat_tokens = occurrence_tokens.reshape(-1, dim)
        flat_valid = valid.reshape(-1)
        selected = flat_tokens[flat_valid]

        flat_gene_index = gene_index.reshape(-1)[flat_valid]
        flat_batch_index = batch_index.reshape(-1)[flat_valid]
        segment_index = flat_batch_index * num_genes + flat_gene_index

        scores = self.occurrence_score(selected).squeeze(-1)
        alpha = pyg_segment_softmax(
            scores,
            segment_index,
            num_nodes=batch_size * num_genes,
        )

        unique_flat = torch.zeros(
            batch_size * num_genes,
            dim,
            device=device,
            dtype=dtype,
        )
        unique_flat.index_add_(
            0,
            segment_index,
            selected * alpha.unsqueeze(-1),
        )
        unique_tokens = unique_flat.view(batch_size, num_genes, dim)

        counts_flat = torch.zeros(
            batch_size * num_genes,
            device=device,
            dtype=dtype,
        )
        counts_flat.index_add_(
            0,
            segment_index,
            torch.ones_like(alpha, dtype=dtype),
        )
        counts = counts_flat.view(batch_size, num_genes)
        present = counts > 0

        unique_tokens = unique_tokens + self.gene_embedding.to(dtype=dtype)

        if self.strict_gene_coverage and not bool(present.all()):
            first_bad = int(torch.where(~present.all(dim=1))[0][0].item())
            missing_indices = torch.where(~present[first_bad])[0].tolist()
            missing_genes = [
                self.idx2gene[index]
                for index in missing_indices[:20]
            ]
            raise RuntimeError(
                "Incomplete landmark coverage after KEGG message passing: "
                f"sample={first_bad}; "
                f"present={int(present[first_bad].sum().item())}/{num_genes}; "
                f"first_missing={missing_genes}"
            )

        positive_counts = counts[counts > 0]
        self.last_token_audit.update({
            "aggregation_impl": "vectorized_segment_softmax",
            "cell_occurrence_tokens": list(occurrence_tokens.shape),
            "cell_unique_gene_tokens": list(unique_tokens.shape),
            "unique_gene_coverage_min": int(
                present.sum(dim=1).min().item()
            ),
            "unique_gene_coverage_max": int(
                present.sum(dim=1).max().item()
            ),
            "occurrences_per_gene_min": float(
                positive_counts.min().item()
            ) if positive_counts.numel() else 0.0,
            "occurrences_per_gene_max": float(
                positive_counts.max().item()
            ) if positive_counts.numel() else 0.0,
            "occurrences_per_gene_mean": float(
                positive_counts.float().mean().item()
            ) if positive_counts.numel() else 0.0,
        })

        gene_ids = [
            list(self.landmark_gene_order)
            for _ in range(batch_size)
        ]
        return unique_tokens, present, gene_ids

    # ------------------------------------------------------------------
    # Unique-cell token encoding and differentiable expansion.
    # ------------------------------------------------------------------
    @staticmethod
    def _unique_first_inverse(keys: Sequence[str]):
        first_position = {}
        unique_positions = []
        inverse = []
        for position, raw_key in enumerate(keys):
            key = str(raw_key)
            if key not in first_position:
                first_position[key] = len(unique_positions)
                unique_positions.append(position)
            inverse.append(first_position[key])
        return unique_positions, inverse

    def _encode_cell_tokens_deduplicated(
        self,
        cell_graph_seq,
        cell_ids: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        if len(cell_graph_seq) != len(cell_ids):
            raise ValueError(
                "cell_graph_seq/cell_ids length mismatch: "
                f"{len(cell_graph_seq)} vs {len(cell_ids)}"
            )

        unique_positions, inverse = self._unique_first_inverse(cell_ids)
        unique_sequences = [cell_graph_seq[index] for index in unique_positions]

        unique_tokens, unique_masks, unique_gene_ids = (
            self._encode_cell_tokens(unique_sequences)
        )

        inverse_tensor = torch.tensor(
            inverse,
            dtype=torch.long,
            device=unique_tokens.device,
        )
        expanded_tokens = unique_tokens.index_select(0, inverse_tensor)
        expanded_masks = unique_masks.index_select(0, inverse_tensor)
        expanded_gene_ids = [
            list(unique_gene_ids[index])
            for index in inverse
        ]

        batch_size = len(cell_ids)
        unique_count = len(unique_positions)
        self.last_token_audit.update({
            "cell_batch_size": batch_size,
            "cell_unique_count": unique_count,
            "cell_duplicate_count": batch_size - unique_count,
            "cell_dedup_fraction": (
                1.0 - unique_count / max(1, batch_size)
            ),
            "cell_unique_gene_tokens_encoded": list(unique_tokens.shape),
            "cell_unique_gene_tokens_expanded": list(expanded_tokens.shape),
            "pathway_batching": bool(self.config.use_pathway_batching),
        })
        return expanded_tokens, expanded_masks, expanded_gene_ids

    def forward(
        self,
        drug_graph,
        cell_graph_seq,
        time: torch.Tensor,
        dose: torch.Tensor,
        return_attn: bool = False,
        cell_embed: Optional[torch.Tensor] = None,
        cell_pad_mask: Optional[torch.Tensor] = None,
        bge: Optional[torch.Tensor] = None,
        cell_ids: Optional[Sequence[str]] = None,
    ):
        self._validate_forward_inputs(time, dose)

        drug_tokens, drug_masks, drug_gene_ids = (
            self._encode_drug_tokens(drug_graph)
        )

        if cell_embed is not None:
            cell_tokens, cell_masks, cell_gene_ids = self._handle_cell_embed(
                cell_embed,
                cell_pad_mask=cell_pad_mask,
            )
        elif cell_ids is not None:
            cell_tokens, cell_masks, cell_gene_ids = (
                self._encode_cell_tokens_deduplicated(
                    cell_graph_seq,
                    cell_ids,
                )
            )
        else:
            cell_tokens, cell_masks, cell_gene_ids = self._encode_cell_tokens(
                cell_graph_seq
            )

        tokens, masks, gene_ids = self._combine_tokens(
            drug_tokens,
            drug_masks,
            drug_gene_ids,
            cell_tokens,
            cell_masks,
            cell_gene_ids,
        )
        token_batch = TokenBatch(tokens, masks, gene_ids)
        token_batch = self._add_condition_tokens(token_batch, time, dose)
        token_batch = self._align_mask_with_tokens(token_batch)

        self.last_token_audit.update({
            "batch_size": token_batch.batch_size,
            "transformer_input": list(token_batch.tokens.shape),
            "valid_tokens_min": int(
                token_batch.mask.sum(dim=1).min().item()
            ),
            "valid_tokens_max": int(
                token_batch.mask.sum(dim=1).max().item()
            ),
            "num_landmark_genes": self.num_genes,
            "architecture": (
                "dose+time+atom+268_unique_gene_tokens+"
                "token_preserving_cell_dedup"
            ),
        })

        transformer_output = self._process_transformer(
            token_batch,
            return_attn,
        )
        if self.task == "pge":
            result = self._process_pge_output(
                transformer_output,
                token_batch,
            )
        else:
            result = self._process_ic50_output(
                transformer_output,
                token_batch,
                time,
                dose,
                bge=bge,
            )

        if (
            os.environ.get("DRT_FAITHFUL_AUDIT", "0") == "1"
            and not self._faithful_audit_printed
        ):
            print(
                "[FAITHFUL TOKEN AUDIT] "
                + json.dumps(self.last_token_audit, sort_keys=True),
                flush=True,
            )
            self._faithful_audit_printed = True
        return result


DrugResponseTransformer = FaithfulTokenDedupDrugResponseTransformer

__all__ = [
    "FaithfulTokenDedupDrugResponseTransformer",
    "DrugResponseTransformer",
    "ModelConfig",
]
