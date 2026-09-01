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
        args,
        heads: int = 4,
        dropout: float = 0.2,
        edge_dim: int = 4,  # bond feature dimension
    ):
        super().__init__()
        self.gat = DrugGAT(in_channels, hidden_channels, out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)


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

    # PATCH_20260715_PER_GRAPH_LAPLACIAN_PE
    def _calculate_position_encoding_per_graph(
        self,
        drug_graph,
        device,
    ):
        """
        Calculate Laplacian positional encoding independently
        for each molecular graph.

        A PyG Batch represents multiple disconnected molecular
        graphs in one block-diagonal graph. Computing a single
        Laplacian eigendecomposition over that full Batch creates
        a degenerate eigenspace whose eigenvectors can depend on
        batch size, composition, and sample position.

        The order returned by Batch.to_data_list() is the same as
        the node order represented by the PyG batch vector, so the
        per-graph encodings can be concatenated safely.
        """

        # PATCH_20260719_USE_CACHED_PER_GRAPH_PE
        # PyG Batch concatenates node-level attributes in graph order,
        # so a precomputed [total_nodes, pe_dim] lap_pe tensor can be
        # used directly without another eigendecomposition.
        cached_pe = getattr(
            drug_graph,
            "lap_pe",
            None,
        )

        if isinstance(
            cached_pe,
            torch.Tensor,
        ):
            pe_full = cached_pe

            if pe_full.dim() == 1:
                pe_full = pe_full.unsqueeze(-1)

            expected_nodes = int(
                drug_graph.x.size(0)
            )

            if int(pe_full.size(0)) != expected_nodes:
                raise RuntimeError(
                    "Cached lap_pe node mismatch: "
                    "lap_pe_nodes={} graph_nodes={}".format(
                        int(pe_full.size(0)),
                        expected_nodes,
                    )
                )

            if int(pe_full.size(1)) < int(self.pe_dim):
                missing_dim = (
                    int(self.pe_dim)
                    - int(pe_full.size(1))
                )

                pe_full = torch.cat(
                    [
                        pe_full,
                        torch.zeros(
                            pe_full.size(0),
                            missing_dim,
                            dtype=pe_full.dtype,
                            device=pe_full.device,
                        ),
                    ],
                    dim=1,
                )

            elif int(pe_full.size(1)) > int(self.pe_dim):
                pe_full = pe_full[
                    :,
                    : int(self.pe_dim),
                ]

            if not torch.isfinite(
                pe_full
            ).all():
                raise RuntimeError(
                    "Cached lap_pe contains NaN or Inf."
                )

            # PATCH_20260719_CACHED_PE_RUNTIME_AUDIT
            if not getattr(
                self,
                "_cached_pe_audit_printed",
                False,
            ):
                num_graphs = int(
                    getattr(
                        drug_graph,
                        "num_graphs",
                        1,
                    )
                )

                print(
                    "[DRUG PE AUDIT] "
                    "source=cached_lap_pe "
                    "graphs={} total_nodes={} "
                    "pe_shape={}".format(
                        num_graphs,
                        int(drug_graph.x.size(0)),
                        tuple(pe_full.shape),
                    ),
                    flush=True,
                )

                self._cached_pe_audit_printed = True

            return pe_full.to(
                device=device,
                dtype=torch.float32,
            )

        if not getattr(
            self,
            "_runtime_pe_fallback_audit_printed",
            False,
        ):
            print(
                "[DRUG PE AUDIT] "
                "source=runtime_per_graph_fallback "
                "total_nodes={}".format(
                    int(drug_graph.x.size(0))
                ),
                flush=True,
            )

            self._runtime_pe_fallback_audit_printed = True

        if (
            hasattr(drug_graph, "to_data_list")
            and callable(
                getattr(
                    drug_graph,
                    "to_data_list",
                )
            )
        ):
            graph_list = (
                drug_graph.to_data_list()
            )
        else:
            graph_list = [
                drug_graph
            ]

        if len(graph_list) == 0:
            raise RuntimeError(
                "No molecular graph was found "
                "for positional encoding."
            )

        pe_parts = []
        expected_total_nodes = 0

        for graph_index, graph in enumerate(
            graph_list
        ):
            if not hasattr(graph, "x"):
                raise AttributeError(
                    "Molecular graph {} has no x "
                    "attribute.".format(
                        graph_index
                    )
                )

            expected_nodes = int(
                graph.x.size(0)
            )

            # NetworkX / SciPy eigendecomposition is performed
            # on CPU for each individual molecule.
            graph_cpu = graph.cpu()

            pe = calculate_position_encoding(
                graph_cpu,
                k=self.pe_dim,
            )

            if not isinstance(
                pe,
                torch.Tensor,
            ):
                pe = torch.as_tensor(
                    pe,
                    dtype=torch.float32,
                )

            pe = pe.to(
                dtype=torch.float32
            )

            if pe.dim() == 1:
                pe = pe.unsqueeze(-1)

            if int(pe.size(0)) != expected_nodes:
                raise RuntimeError(
                    "PE node mismatch for graph {}: "
                    "pe_nodes={} graph_nodes={}".format(
                        graph_index,
                        int(pe.size(0)),
                        expected_nodes,
                    )
                )

            # Defensive dimensional alignment.
            if int(pe.size(1)) < int(self.pe_dim):
                pad_width = (
                    int(self.pe_dim)
                    - int(pe.size(1))
                )

                pe = torch.cat(
                    [
                        pe,
                        torch.zeros(
                            pe.size(0),
                            pad_width,
                            dtype=pe.dtype,
                        ),
                    ],
                    dim=1,
                )

            elif int(pe.size(1)) > int(self.pe_dim):
                pe = pe[
                    :,
                    : int(self.pe_dim),
                ]

            pe_parts.append(pe)
            expected_total_nodes += (
                expected_nodes
            )

        pe_full = torch.cat(
            pe_parts,
            dim=0,
        ).to(device)

        actual_total_nodes = int(
            drug_graph.x.size(0)
        )

        if (
            expected_total_nodes
            != actual_total_nodes
        ):
            raise RuntimeError(
                "Batched graph node mismatch: "
                "per_graph_total={} batch_total={}".format(
                    expected_total_nodes,
                    actual_total_nodes,
                )
            )

        if int(pe_full.size(0)) != (
            actual_total_nodes
        ):
            raise RuntimeError(
                "Final positional-encoding node "
                "count mismatch."
            )

        return pe_full

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

            batch = torch.zeros(N, dtype=torch.long, device=device)
        else:
            batch = batch.to(device)
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1


        node_embeddings = self.gat(x, edge_index, edge_attr=edge_attr)  # [N, out_channels]
        node_feats_full = self.to_feat(node_embeddings)          # [N, target_dim - pe_dim]


        pe_full = getattr(drug_graph, "pe", None)
        if pe_full is not None:
            pe_full = pe_full.to(device)

            if pe_full.dim() != 2 or pe_full.size(0) != node_embeddings.size(0) or pe_full.size(1) != self.pe_dim:
                pe_full = self._calculate_position_encoding_per_graph(drug_graph, device)
        else:
            pe_full = self._calculate_position_encoding_per_graph(drug_graph, device)


        if return_graph:
            combined = torch.cat([node_feats_full, pe_full], dim=1)  # [N, target_dim]
            combined = self._add_type(combined)
            graph_emb = global_mean_pool(combined, batch)            # [B, target_dim]
            return graph_emb


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
