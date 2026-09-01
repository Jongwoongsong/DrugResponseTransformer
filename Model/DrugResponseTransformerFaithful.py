"""Faithful atom–gene-token DrugResponseTransformer."""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_segment_softmax

from Model.DrugResponseTransformer import (
    DrugResponseTransformer as BaseDrugResponseTransformer,
    ModelConfig,
    TokenBatch,
)


class FaithfulDrugResponseTransformer(BaseDrugResponseTransformer):
    """Transformer input: dose, time, atom tokens, 268 unique gene tokens."""

    def __init__(
        self,
        args,
        landmark_set: List[str],
        config: Optional[ModelConfig] = None,
        task: str = "pge",
        include_cls: bool = False,
        strict_gene_coverage: bool = True,
    ):
        super().__init__(
            args=args,
            landmark_set=landmark_set,
            config=config,
            task=task,
            include_cls=include_cls,
        )
        self.strict_gene_coverage = bool(strict_gene_coverage)
        self.occurrence_score = nn.Sequential(
            nn.Linear(self.config.dim_node, self.config.dim_node),
            nn.Tanh(),
            nn.Linear(self.config.dim_node, 1, bias=False),
        )
        self.last_token_audit = {}
        self._faithful_audit_printed = False

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

        batch_size, _, dim = occurrence_tokens.shape
        num_genes = self.num_genes
        device = occurrence_tokens.device
        dtype = occurrence_tokens.dtype

        unique_batch = []
        mask_batch = []
        occurrence_counts = []

        for batch_index in range(batch_size):
            positions = []
            gene_indices = []

            for token_index, gene_id in enumerate(occurrence_gene_ids[batch_index]):
                if token_index >= occurrence_mask.size(1):
                    break
                if not bool(occurrence_mask[batch_index, token_index].item()):
                    continue
                if gene_id is None:
                    continue
                gene_index = self.gene2idx.get(str(gene_id))
                if gene_index is None:
                    continue
                positions.append(token_index)
                gene_indices.append(gene_index)

            if not positions:
                raise RuntimeError(
                    "No landmark-gene occurrences were produced. "
                    "Check pathway node_ids and landmark identifiers."
                )

            position_tensor = torch.tensor(
                positions, dtype=torch.long, device=device
            )
            gene_index_tensor = torch.tensor(
                gene_indices, dtype=torch.long, device=device
            )
            selected = occurrence_tokens[batch_index].index_select(
                0, position_tensor
            )

            scores = self.occurrence_score(selected).squeeze(-1)
            alpha = pyg_segment_softmax(
                scores, gene_index_tensor, num_nodes=num_genes
            )

            unique = torch.zeros(
                num_genes, dim, device=device, dtype=dtype
            )
            unique.index_add_(
                0,
                gene_index_tensor,
                selected * alpha.unsqueeze(-1),
            )

            counts = torch.zeros(
                num_genes, device=device, dtype=dtype
            )
            counts.index_add_(
                0,
                gene_index_tensor,
                torch.ones_like(alpha, dtype=dtype),
            )
            present = counts > 0

            # Stable identity/alignment shared by LINCS and GDSC.
            unique = unique + self.gene_embedding.to(dtype=dtype)

            if self.strict_gene_coverage and not bool(present.all().item()):
                missing_indices = torch.where(~present)[0].tolist()
                missing_genes = [
                    self.idx2gene[index]
                    for index in missing_indices[:20]
                ]
                raise RuntimeError(
                    "Incomplete landmark coverage after KEGG message passing: "
                    f"present={int(present.sum().item())}/{num_genes}; "
                    f"first_missing={missing_genes}"
                )

            unique_batch.append(unique)
            mask_batch.append(present)
            occurrence_counts.append(counts.detach().cpu())

        unique_tokens = torch.stack(unique_batch, dim=0)
        unique_mask = torch.stack(mask_batch, dim=0)
        unique_gene_ids = [
            list(self.landmark_gene_order)
            for _ in range(batch_size)
        ]

        stacked_counts = torch.stack(occurrence_counts, dim=0)
        positive_counts = stacked_counts[stacked_counts > 0]
        self.last_token_audit.update({
            "cell_occurrence_tokens": list(occurrence_tokens.shape),
            "cell_unique_gene_tokens": list(unique_tokens.shape),
            "unique_gene_coverage_min": int(
                unique_mask.sum(dim=1).min().item()
            ),
            "unique_gene_coverage_max": int(
                unique_mask.sum(dim=1).max().item()
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
        return unique_tokens, unique_mask, unique_gene_ids

    def _encode_cell_tokens(
        self,
        cell_graph_seq,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        occurrence_tokens, occurrence_mask, occurrence_gene_ids = (
            super()._encode_cell_tokens(cell_graph_seq)
        )
        return self._aggregate_occurrences_to_unique(
            occurrence_tokens,
            occurrence_mask,
            occurrence_gene_ids,
        )

    def _handle_cell_embed(
        self,
        cell_embed: Union[torch.Tensor, List[torch.Tensor]],
        cell_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        tokens, masks, _ = super()._handle_cell_embed(
            cell_embed,
            cell_pad_mask=cell_pad_mask,
        )
        if tokens.size(1) != self.num_genes:
            raise ValueError(
                "Faithful cell cache must be [B,G,D]: "
                f"expected G={self.num_genes}, got {tokens.size(1)}"
            )
        gene_ids = [
            list(self.landmark_gene_order)
            for _ in range(tokens.size(0))
        ]
        return tokens, masks, gene_ids

    def _process_pge_output(
        self,
        transformer_output,
        token_batch: TokenBatch,
    ):
        if isinstance(transformer_output, tuple):
            output, attn_maps = transformer_output
            return_attn = True
        else:
            output, attn_maps = transformer_output, None
            return_attn = False

        body_out = output[:, 2:, :]
        body_gene_ids = [
            gene_ids[2:] for gene_ids in token_batch.gene_ids
        ]
        gene_out = body_out[:, -self.num_genes:, :]
        gene_ids_tail = [
            gene_ids[-self.num_genes:]
            for gene_ids in body_gene_ids
        ]
        expected = list(self.landmark_gene_order)
        for batch_index, observed in enumerate(gene_ids_tail):
            if list(observed) != expected:
                raise RuntimeError(
                    "Gene-token order changed before PGE readout at "
                    f"batch index {batch_index}."
                )

        # Position-wise shared head: gene i token -> PGE target i.
        pred = self.regressor(gene_out).squeeze(-1)
        self.last_token_audit.update({
            "transformer_output": list(output.shape),
            "pge_gene_output": list(gene_out.shape),
            "pge_prediction": list(pred.shape),
        })
        if return_attn:
            return pred, attn_maps
        return pred

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
    ):
        self._validate_forward_inputs(time, dose)
        token_batch = self._encode_batch(
            drug_graph=drug_graph,
            cell_graph_seq=cell_graph_seq,
            cell_embed=cell_embed,
            cell_pad_mask=cell_pad_mask,
        )
        token_batch = self._add_condition_tokens(
            token_batch, time, dose
        )
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
            "architecture": "dose+time+atom+268_unique_gene_tokens",
        })

        transformer_output = self._process_transformer(
            token_batch, return_attn
        )
        if self.task == "pge":
            result = self._process_pge_output(
                transformer_output, token_batch
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

    # Fail-fast guard against the shortcut that pools before Transformer.
    def encode_drug(self, *args, **kwargs):
        raise RuntimeError(
            "Pooled encode_drug is disabled. Set use_batch_dedup=False."
        )

    def encode_cell_seq(self, *args, **kwargs):
        raise RuntimeError(
            "Pooled encode_cell_seq is disabled. Set use_batch_dedup=False."
        )

    def fuse_and_predict(self, *args, **kwargs):
        raise RuntimeError(
            "Pooled fuse_and_predict is disabled. Set use_batch_dedup=False."
        )


DrugResponseTransformer = FaithfulDrugResponseTransformer

__all__ = [
    "FaithfulDrugResponseTransformer",
    "DrugResponseTransformer",
    "ModelConfig",
]
