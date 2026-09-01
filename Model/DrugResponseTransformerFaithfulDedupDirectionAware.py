import json
import types

import torch
import torch.nn as nn

from Model.CellLine_graph import NUM_EDGE_TYPES
from Model.DrugResponseTransformer import ModelConfig
from Model.DrugResponseTransformerFaithfulDedup import (
    FaithfulTokenDedupDrugResponseTransformer,
)


def _direction_aware_process_edge_attr(self, edge_attr, device):
    if edge_attr is None:
        return None

    edge_attr = edge_attr.to(device)

    expected = NUM_EDGE_TYPES + 1
    if edge_attr.dim() != 2 or edge_attr.size(1) != expected:
        raise RuntimeError(
            f"DA edge_attr must be [E,{expected}], got {tuple(edge_attr.shape)}"
        )

    relation = edge_attr[:, :NUM_EDGE_TYPES]
    direction = edge_attr[:, NUM_EDGE_TYPES].round().long()

    if not bool(((direction == 0) | (direction == 1)).all()):
        raise RuntimeError("Direction flag must be 0/1")

    # Existing pretrained/scalar relation mechanism
    relation_scalar = self.compute_edge_weight(relation)
    relation_emb = self.edge_emb(relation_scalar)

    # New direction identity
    direction_emb = self.direction_embedding(direction)

    return relation_emb + direction_emb


class DirectionAwareFaithfulTokenDedupDrugResponseTransformer(
    FaithfulTokenDedupDrugResponseTransformer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hidden = int(self.CellEncoder.hidden_channels)

        self.CellEncoder.direction_embedding = nn.Embedding(
            2, hidden
        )

        # original/reverse initially identical
        nn.init.zeros_(
            self.CellEncoder.direction_embedding.weight
        )

        self.CellEncoder._process_edge_attr = types.MethodType(
            _direction_aware_process_edge_attr,
            self.CellEncoder,
        )

        self.CellEncoder.edge_direction_mode = (
            "direction_aware_bidirectional"
        )

        self.last_token_audit.update({
            "edge_direction_mode": "direction_aware_bidirectional",
            "edge_relation_encoding": "scalar",
            "direction_states": 2,
            "direction_zero_initialized": True,
        })

        print(
            "[DIRECTION-AWARE INIT AUDIT] "
            + json.dumps({
                "relation_encoding": "scalar",
                "direction_states": 2,
                "0": "original_KEGG",
                "1": "computational_reverse",
                "direction_embedding_shape": list(
                    self.CellEncoder.direction_embedding.weight.shape
                ),
                "zero_init_max_abs": float(
                    self.CellEncoder.direction_embedding.weight
                    .detach().abs().max()
                ),
            }, sort_keys=True),
            flush=True,
        )


DrugResponseTransformer = (
    DirectionAwareFaithfulTokenDedupDrugResponseTransformer
)

__all__ = [
    "DirectionAwareFaithfulTokenDedupDrugResponseTransformer",
    "DrugResponseTransformer",
    "ModelConfig",
]
