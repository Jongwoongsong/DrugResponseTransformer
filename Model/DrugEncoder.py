import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool

try:
    from Model.drug_graph import calculate_position_encoding
except Exception:
    from drug_graph import calculate_position_encoding


class DrugGAT(nn.Module):
    """Bond-aware Drug GAT using GATv2Conv with edge features."""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2, edge_dim=4):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout, edge_dim=edge_dim)
        self.gat3 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, edge_dim=edge_dim)

        self.res1 = nn.Linear(in_channels, hidden_channels * heads)
        self.res2 = nn.Linear(hidden_channels * heads, hidden_channels * heads)

        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h1 = self.gat1(x, edge_index, edge_attr=edge_attr)
        h1 = h1 + self.res1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        h2 = self.gat2(h1, edge_index, edge_attr=edge_attr)
        h2 = h2 + self.res2(h1)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)

        h3 = self.gat3(h2, edge_index, edge_attr=edge_attr)
        return h3


class DrugEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        token_type_embed: nn.Embedding,
        drug_type_id: int,
        args,   # args에서 dim_node, pe_dim, max_num_nodes 사용
        heads: int = 4,
        dropout: float = 0.2,
        edge_dim: int = 4,  # bond feature dimension
    ):
        super().__init__()
        self.gat = DrugGAT(in_channels, hidden_channels, out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)

        # 항상 args.dim_node 사용
        self.target_dim = int(getattr(args, "dim_node", out_channels))
        self.pe_dim = int(getattr(args, "pe_dim", 1))
        self.max_num_nodes = int(getattr(args, "max_num_nodes", 44))
        assert self.target_dim > self.pe_dim, f"dim_node({self.target_dim}) must be > pe_dim({self.pe_dim})"

        self.to_feat = nn.Linear(out_channels, self.target_dim - self.pe_dim)

        self.token_type_embed = token_type_embed
        self.register_buffer("drug_type_id_tensor", torch.tensor(drug_type_id, dtype=torch.long))

    def _add_type(self, feats: torch.Tensor) -> torch.Tensor:
        type_vec = self.token_type_embed(self.drug_type_id_tensor)  # [D]
        # broadcast add
        return feats + type_vec.view(*([1] * (feats.dim() - 1)), -1)

    def forward(self, drug_graph, return_graph: bool = False):
        device = next(self.parameters()).device
        drug_graph = drug_graph.to(device)

        x = drug_graph.x
        edge_index = drug_graph.edge_index
        # ---- NEW: extract bond features ----
        edge_attr = getattr(drug_graph, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        N = x.size(0)

        batch = getattr(drug_graph, "batch", None)
        if batch is None:
            # 배치 정보가 없으면 단일 그래프로 취급
            batch = torch.zeros(N, dtype=torch.long, device=device)
        else:
            batch = batch.to(device)
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # 1) GAT 인코딩 (batched) — now with edge_attr
        node_embeddings = self.gat(x, edge_index, edge_attr=edge_attr)  # [N, out_channels]
        node_feats_full = self.to_feat(node_embeddings)          # [N, target_dim - pe_dim]

        # 2) Position Encoding: 캐시에 있으면 data.pe 재사용, 없으면 계산
        pe_full = getattr(drug_graph, "pe", None)
        if pe_full is not None:
            pe_full = pe_full.to(device)
            # 모양 확인 (N, pe_dim) 아니면 다시 계산
            if pe_full.dim() != 2 or pe_full.size(0) != node_embeddings.size(0) or pe_full.size(1) != self.pe_dim:
                pe_full = calculate_position_encoding(drug_graph, k=self.pe_dim).to(device)
        else:
            pe_full = calculate_position_encoding(drug_graph, k=self.pe_dim).to(device)

        # return_graph=True: 그래프 임베딩 풀링 경로
        if return_graph:
            combined = torch.cat([node_feats_full, pe_full], dim=1)  # [N, target_dim]
            combined = self._add_type(combined)
            graph_emb = global_mean_pool(combined, batch)            # [B, target_dim]
            return graph_emb

        # 3) per-graph 토큰 시퀀스 생성 (top-k → pad)
        target_dim = self.target_dim
        max_k = self.max_num_nodes

        tokens = node_feats_full.new_zeros((B, max_k, target_dim), dtype=torch.float32)
        mask   = torch.zeros((B, max_k), dtype=torch.bool, device=device)

        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            nf = node_feats_full.index_select(0, idx)  # [n_b, D-pe]
            pe = pe_full.index_select(0, idx)          # [n_b, pe]
            n_b = nf.size(0)

            # 중요도 top-k (L2 norm)
            if n_b > max_k:
                importance = nf.norm(p=2, dim=1)       # [n_b]
                topk = torch.topk(importance, k=max_k, largest=True).indices
                nf = nf.index_select(0, topk)
                pe = pe.index_select(0, topk)
                n_keep = max_k
            else:
                n_keep = n_b

            combined = torch.cat([nf, pe], dim=1)      # [n_keep, target_dim]
            combined = self._add_type(combined)         # type embedding add

            tokens[b, :n_keep, :] = combined
            mask[b, :n_keep] = True

        return tokens, mask  # [B, max_k, D], [B, max_k]


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    import random

    # 더미 배치 3개
    datas = []
    for _ in range(3):
        n = random.randint(8, 20)
        x = torch.randn(n, 57)
        ei = torch.randint(0, n, (2, n*2))
        ea = torch.randn(n*2, 4)  # bond features
        datas.append(Data(x=x, edge_index=ei, edge_attr=ea))
    batch = Batch.from_data_list(datas)

    class DummyArgs: pass
    args = DummyArgs()
    args.dim_node = 64
    args.pe_dim = 1
    args.max_num_nodes = 16

    tok = nn.Embedding(8, args.dim_node)
    enc = DrugEncoder(in_channels=57, hidden_channels=64, out_channels=64,
                      token_type_embed=tok, drug_type_id=1, args=args,
                      heads=4, dropout=0.1, edge_dim=4)

    out, m = enc(batch, return_graph=False)
    print("tokens:", out.shape, "mask:", m.shape, "per-graph valid:", m.sum(dim=1))
    ge = enc(batch, return_graph=True)
    print("graph_emb:", ge.shape)
