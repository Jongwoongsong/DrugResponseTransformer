import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch


class DrugGAT(nn.Module):
    def __init__(self, in_channels=57, hidden_channels=128, out_channels=256, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

        self.res1 = nn.Linear(in_channels, hidden_channels * heads)
        self.res2 = nn.Linear(hidden_channels * heads, hidden_channels * heads)

        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h1 = self.gat1(x, edge_index)
        h1 = h1 + self.res1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        h2 = self.gat2(h1, edge_index)
        h2 = h2 + self.res2(h1)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)

        h3 = self.gat3(h2, edge_index)
        
        # Global pooling
        graph_emb = global_mean_pool(h3, batch)  # [B, out_channels]
        return graph_emb


# ─────────────────────────────────────────────────────────────────────────────
# Gene Encoder (Per-gene embedding with expression weighting)
# ─────────────────────────────────────────────────────────────────────────────
class GeneEncoder(nn.Module):
    """Gene expression을 직접 embedding으로 변환"""
    def __init__(self, num_genes=949, embed_dim=256, dropout=0.1):
        super().__init__()
        self.num_genes = num_genes
        
        # Gene expression value → embedding
        self.gene_proj = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Learnable gene-specific embedding
        self.gene_embed = nn.Embedding(num_genes, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, gene_expr):
        """
        gene_expr: [B, num_genes]
        returns: [B, num_genes, embed_dim]
        """
        batch_size = gene_expr.shape[0]
        device = gene_expr.device
        
        # Expression value → embedding
        expr_emb = gene_expr.unsqueeze(-1)  # [B, 949, 1]
        expr_emb = self.gene_proj(expr_emb)  # [B, 949, embed_dim]
        
        # Gene identity embedding
        gene_idx = torch.arange(self.num_genes, device=device)
        gene_id_emb = self.gene_embed(gene_idx)  # [949, embed_dim]
        gene_id_emb = gene_id_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine: expression info + gene identity
        gene_emb = expr_emb + gene_id_emb  # [B, 949, embed_dim]
        
        gene_emb = self.layer_norm(gene_emb)
        gene_emb = self.dropout(gene_emb)
        
        return gene_emb


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Attention (Drug queries, Gene keys/values)
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value, return_attn=False):
        """
        query: [B, embed_dim] - drug embedding
        key_value: [B, num_genes, embed_dim] - gene embeddings
        """
        B, N, D = key_value.shape
        
        # Expand query: [B, embed_dim] -> [B, 1, embed_dim]
        query = query.unsqueeze(1)
        
        q = self.q_proj(query)  # [B, 1, D]
        k = self.k_proj(key_value)  # [B, N, D]
        v = self.v_proj(key_value)  # [B, N, D]
        
        # Multi-head reshape
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, 1, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v)  # [B, H, 1, d]
        out = out.transpose(1, 2).reshape(B, -1)  # [B, D]
        out = self.out_proj(out)
        
        if return_attn:
            # Average attention across heads: [B, N]
            attn_weights = attn.mean(dim=1).squeeze(1)
            return out, attn_weights
        return out, None


# ─────────────────────────────────────────────────────────────────────────────
# Full Model: GAT + Cross-Attention
# ─────────────────────────────────────────────────────────────────────────────
class GATCrossAttentionModel(nn.Module):
    def __init__(
        self,
        num_genes=949,
        atom_dim=57,
        hidden_dim=128,
        embed_dim=256,
        num_heads=8,
        gat_heads=4,
        dropout=0.2,
    ):
        super().__init__()
        
        # Drug encoder (GAT)
        self.drug_encoder = DrugGAT(
            in_channels=atom_dim,
            hidden_channels=hidden_dim,
            out_channels=embed_dim,
            heads=gat_heads,
            dropout=dropout
        )
        self.drug_ln = nn.LayerNorm(embed_dim)
        
        # Gene encoder
        self.gene_encoder = GeneEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Cross-attention
        self.cross_attn = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.attn_ln = nn.LayerNorm(embed_dim)
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, drug_graphs, gene_expr, return_attn=False):
        """
        drug_graphs: Batch of PyG Data objects
        gene_expr: [B, num_genes]
        """
        # Drug encoding
        drug_emb = self.drug_encoder(
            drug_graphs.x, 
            drug_graphs.edge_index, 
            drug_graphs.batch
        )  # [B, embed_dim]
        drug_emb = self.drug_ln(drug_emb)
        
        # Gene encoding
        gene_emb = self.gene_encoder(gene_expr)  # [B, num_genes, embed_dim]
        
        # Cross-attention
        attended_emb, attn_weights = self.cross_attn(
            query=drug_emb,
            key_value=gene_emb,
            return_attn=return_attn
        )
        attended_emb = self.attn_ln(attended_emb)
        
        # Prediction
        combined = torch.cat([drug_emb, attended_emb], dim=-1)
        pred = self.pred_head(combined).squeeze(-1)
        
        if return_attn:
            return pred, drug_emb, attn_weights
        return pred


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    
    # Test
    B = 4
    num_genes = 949
    
    # Dummy drug graphs
    graphs = []
    for _ in range(B):
        n = torch.randint(10, 30, (1,)).item()
        x = torch.randn(n, 57)
        edge_index = torch.randint(0, n, (2, n * 2))
        graphs.append(Data(x=x, edge_index=edge_index))
    drug_batch = Batch.from_data_list(graphs)
    
    # Dummy gene expression
    gene_expr = torch.randn(B, num_genes)
    
    # Model
    model = GATCrossAttentionModel()
    pred = model(drug_batch, gene_expr)
    print(f"Prediction shape: {pred.shape}")  # [4]
    
    pred, drug_emb, attn = model(drug_batch, gene_expr, return_attn=True)
    print(f"Attention shape: {attn.shape}")  # [4, 949]