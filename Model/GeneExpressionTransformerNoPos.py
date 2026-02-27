import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayerNoPos(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor, return_attn: bool = False):
        token_mask = token_mask.bool()
        key_padding_mask = ~token_mask  # [B, T], True=무시(패딩)

        y = self.norm1(x)
        attn_out, attn_w = self.mha(
            y, y, y,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False
        )
        x = x + self.dropout1(attn_out)

        y = self.norm2(x)
        y = self.ffn(y)
        x = x + self.dropout2(y)

        if return_attn:
            return x, attn_w
        else:
            return x


class GeneExpressionTransformerNoPos(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_layers: int, dropout: float, num_genes: int):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            TransformerEncoderLayerNoPos(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)

        self.num_genes = num_genes

    def forward(self, token_emb: torch.Tensor, token_mask: torch.Tensor, return_attn: bool = False):
        x = token_emb
        attn_maps = [] if return_attn else None

        for layer in self.layers:
            if return_attn:
                x, attn_w = layer(x, token_mask, return_attn=True)  # [B,T,D], [B,H,T,T]
                attn_maps.append(attn_w)
            else:
                x = layer(x, token_mask, return_attn=False)

        x = self.out_ln(x)

        if return_attn:
            return x, attn_maps
        else:
            return x