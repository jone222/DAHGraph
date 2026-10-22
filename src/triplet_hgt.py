from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv

from .triplet_features import compose_triplet_features


class TripletHGT(nn.Module):
    def __init__(self, metadata, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.2, use_residual: bool = True, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.convs = nn.ModuleList([HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in metadata[0]}) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict) -> dict[str, torch.Tensor]:
        h = x_dict
        for i, conv in enumerate(self.convs):
            out = conv(h, edge_index_dict)
            next_h = {}
            for nt, x in h.items():
                y = out.get(nt, x)
                if self.use_residual and y.shape == x.shape:
                    y = y + x
                if self.use_layer_norm:
                    y = self.norms[i][nt](y)
                next_h[nt] = self.dropout(torch.nn.functional.gelu(y))
            h = next_h
        return h

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict, drug_index: torch.Tensor, microbe_index: torch.Tensor, disease_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x_dict, edge_index_dict)
        z = compose_triplet_features(h["drug"][drug_index], h["microbe"][microbe_index], h["disease"][disease_index])
        return self.decoder(z)

