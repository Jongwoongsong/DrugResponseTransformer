import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, List


try:
    from Model.CellLine_graph import NUM_EDGE_TYPES, SUBTYPE_INIT_VALUES
except ImportError:
    from CellLine_graph import NUM_EDGE_TYPES, SUBTYPE_INIT_VALUES


class CellEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        out_channels: int = 30,
        edge_attr_dim: int = 1,
        dropout: float = 0.5,
        args=None,
        token_type_embed: nn.Embedding = None,
        pathway_type_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        assert token_type_embed is not None
        assert pathway_type_ids is not None

        self.target_dim = getattr(args, "dim_node", 32)
        self.pe_dim = getattr(args, "pe_dim", 1)
        self.dropout = float(getattr(args, "dropout_ratio", getattr(args, "dropout", dropout)))
        self.hidden_channels = hidden_channels

        # ═══════════════════════════════════════════════════════════════════
        # ✅ Learnable Edge Type Weights

        # ═══════════════════════════════════════════════════════════════════
        self.edge_type_weight = nn.Parameter(
            torch.tensor(SUBTYPE_INIT_VALUES, dtype=torch.float32)
        )

        # Edge embedding: weighted scalar → hidden_channels
        self.edge_emb = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )


        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=hidden_channels)
        self.res1 = nn.Linear(in_channels, hidden_channels * 8)
        self.norm1 = nn.LayerNorm(hidden_channels * 8)

        self.gat2 = GATv2Conv(hidden_channels * 8, hidden_channels, heads=4, concat=True, edge_dim=hidden_channels)
        self.res2 = nn.Linear(hidden_channels * 8, hidden_channels * 4)
        self.norm2 = nn.LayerNorm(hidden_channels * 4)

        self.gat3 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=True, edge_dim=hidden_channels)
        self.res3 = nn.Linear(hidden_channels * 4, hidden_channels * 2)
        self.norm3 = nn.LayerNorm(hidden_channels * 2)

        self.gat4 = GATv2Conv(hidden_channels * 2, out_channels, heads=1, concat=False, edge_dim=hidden_channels)

        self.to_feat = nn.Linear(out_channels, self.target_dim - self.pe_dim)

        self.token_type_embed = token_type_embed
        self.register_buffer("pathway_type_ids_tensor", torch.tensor(pathway_type_ids, dtype=torch.long))

    def compute_edge_weight(self, edge_attr):
        """
        ✅ Multi-hot edge_attr → learnable weighted scalar
        
        edge_attr: [E, NUM_EDGE_TYPES] (multi-hot encoding)
        return: [E, 1] (learnable weighted sum)
        """
        # edge_attr @ edge_type_weight → [E]

        weights = torch.matmul(edge_attr, self.edge_type_weight)
        return weights.unsqueeze(-1)  # [E, 1]

    # ═══════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    def _process_edge_attr(self, edge_attr, device):
        """Shared edge-attribute processing for forward and forward_batched."""
        if edge_attr is None:
            return None

        edge_attr = edge_attr.to(device)


        if edge_attr.dim() == 2 and edge_attr.size(1) == NUM_EDGE_TYPES:
            edge_scalar = self.compute_edge_weight(edge_attr)  # [E, 1]

        elif edge_attr.dim() == 1:
            edge_scalar = edge_attr.unsqueeze(-1)  # [E, 1]
        elif edge_attr.dim() == 2 and edge_attr.size(1) == 1:
            edge_scalar = edge_attr  # [E, 1]
        else:

            edge_scalar = edge_attr[:, :1] if edge_attr.dim() == 2 else edge_attr.unsqueeze(-1)

        return self.edge_emb(edge_scalar)  # [E, hidden_channels]

    # ═══════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    def _forward_gat_layers(self, x, edge_index, edge_emb):
        """Shared GAT-layer logic for forward and forward_batched."""
        h1 = self.gat1(x, edge_index, edge_emb)
        h1 = h1 + self.res1(x)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.gat2(h1, edge_index, edge_emb)
        h2 = h2 + self.res2(h1)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        h3 = self.gat3(h2, edge_index, edge_emb)
        h3 = h3 + self.res3(h2)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)

        h4 = self.gat4(h3, edge_index, edge_emb)

        return h4

    def forward(self, graph_data, pathway_idx: int):
        """Forward pass for a single pathway graph."""
        device = next(self.parameters()).device
        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr = getattr(graph_data, "edge_attr", None)


        edge_emb = self._process_edge_attr(edge_attr, device)


        h4 = self._forward_gat_layers(x, edge_index, edge_emb)

        node_feats = self.to_feat(h4)  # (N, target_dim - pe_dim)
        pe = torch.zeros((node_feats.size(0), self.pe_dim), device=device)  # (N, pe_dim)
        combined = torch.cat([node_feats, pe], dim=1)  # (N, target_dim)

        num_types = int(self.pathway_type_ids_tensor.numel())
        if num_types == 0:
            raise RuntimeError("pathway_type_ids is empty")
        safe_idx = int(pathway_idx) % num_types
        type_id = self.pathway_type_ids_tensor[safe_idx]
        type_vec = self.token_type_embed(type_id.unsqueeze(0)).squeeze(0)  # (target_dim,)
        combined = combined + type_vec

        return combined

    # ═══════════════════════════════════════════════════════════════════════════
    # ★★★ NEW: Batched Forward for Pathway Batching Optimization ★★★
    # ═══════════════════════════════════════════════════════════════════════════
    def forward_batched(
        self,
        batched_graphs,
        pathway_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        ★ PATHWAY BATCHING OPTIMIZED ★
        
        Process multiple pathway graphs in one batch for improved throughput.
        
        Original: 31 pathways x B samples = 31B individual forward passes.
        Optimized: a single batched forward pass.
        
        Args:
            batched_graphs: PyG Batch containing all pathway graphs
                - batched_graphs.x: [total_nodes, in_channels]
                - batched_graphs.edge_index: [2, total_edges]
                - batched_graphs.batch: [total_nodes] - graph index for each node
                - batched_graphs.ptr: [num_graphs + 1] - starting node offset for each graph
            pathway_indices: [num_graphs] - pathway index for each graph (0-30)
        
        Returns:
            combined: [total_nodes, target_dim] - all node embeddings
        
        Important: this function does not "merge" the graphs.
           Each pathway is processed independently, and edges remain within each graph.
           Batching is used only for GPU-parallel forward computation.
        """
        device = next(self.parameters()).device

        # Move to device
        x = batched_graphs.x.to(device)
        edge_index = batched_graphs.edge_index.to(device)
        edge_attr = getattr(batched_graphs, "edge_attr", None)
        batch = batched_graphs.batch.to(device)


        edge_emb = self._process_edge_attr(edge_attr, device)




        h4 = self._forward_gat_layers(x, edge_index, edge_emb)

        # Feature projection
        node_feats = self.to_feat(h4)  # [total_nodes, target_dim - pe_dim]
        pe = torch.zeros((node_feats.size(0), self.pe_dim), device=device)
        combined = torch.cat([node_feats, pe], dim=1)  # [total_nodes, target_dim]




        pathway_indices = pathway_indices.to(device)
        pw_idx_per_node = pathway_indices[batch]  # [total_nodes]


        num_types = int(self.pathway_type_ids_tensor.numel())
        safe_pw_idx = pw_idx_per_node % num_types
        type_ids = self.pathway_type_ids_tensor[safe_pw_idx]  # [total_nodes]
        type_vec = self.token_type_embed(type_ids)  # [total_nodes, target_dim]

        combined = combined + type_vec

        return combined  # [total_nodes, target_dim]

    def get_edge_type_weights(self):
        """
        Return learned edge-type weights for interpretability.
        """
        try:
            from CellLine_graph import SUBTYPE_LIST
        except ImportError:
            from Model.CellLine_graph import SUBTYPE_LIST

        weights = self.edge_type_weight.detach().cpu().numpy()
        return {name: float(weights[i]) for i, name in enumerate(SUBTYPE_LIST)}