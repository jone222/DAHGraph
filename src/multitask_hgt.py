from __future__ import annotations

import torch
from torch import nn

from .triplet_features import compose_triplet_features
from .triplet_hgt import TripletHGT


class PairDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, b, a * b, torch.abs(a - b)], dim=-1))


class MultiTaskTripletHGT(nn.Module):
    def __init__(self, metadata, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.2, use_residual: bool = True, use_layer_norm: bool = True, use_deim: bool = True, use_mbda: bool = True, use_pppa: bool = False, use_mppa: bool = False) -> None:
        super().__init__()
        base = TripletHGT(metadata, hidden_dim, num_layers, num_heads, dropout, use_residual, use_layer_norm)
        self.convs = base.convs
        self.norms = base.norms
        self.dropout = base.dropout
        self.use_residual = base.use_residual
        self.use_layer_norm = base.use_layer_norm
        self.triplet_decoder = base.decoder
        self.use_deim = use_deim
        self.use_mbda = use_mbda
        self.use_pppa = use_pppa
        self.use_mppa = use_mppa
        self.deim_decoder = PairDecoder(hidden_dim, dropout) if use_deim else None
        self.mbda_decoder = PairDecoder(hidden_dim, dropout) if use_mbda else None
        self.pppa_decoder = PairDecoder(hidden_dim, dropout) if use_pppa else None
        self.mppa_decoder = PairDecoder(hidden_dim, dropout) if use_mppa else None

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

    def main_logits(self, h: dict[str, torch.Tensor], drug_index: torch.Tensor, microbe_index: torch.Tensor, disease_index: torch.Tensor) -> torch.Tensor:
        z = compose_triplet_features(h["drug"][drug_index], h["microbe"][microbe_index], h["disease"][disease_index])
        return self.triplet_decoder(z)

    def deim_logits(self, h: dict[str, torch.Tensor], drug_index: torch.Tensor, microbe_index: torch.Tensor) -> torch.Tensor:
        if self.deim_decoder is None:
            raise RuntimeError("DEIM decoder disabled")
        return self.deim_decoder(h["drug"][drug_index], h["microbe"][microbe_index])

    def mbda_logits(self, h: dict[str, torch.Tensor], microbe_index: torch.Tensor, disease_index: torch.Tensor) -> torch.Tensor:
        if self.mbda_decoder is None:
            raise RuntimeError("MBDA decoder disabled")
        return self.mbda_decoder(h["microbe"][microbe_index], h["disease"][disease_index])

    def pppa_logits(self, h: dict[str, torch.Tensor], protein_index: torch.Tensor, pathway_index: torch.Tensor) -> torch.Tensor:
        if self.pppa_decoder is None:
            raise RuntimeError("PPPA decoder disabled")
        return self.pppa_decoder(h["protein"][protein_index], h["pathway"][pathway_index])

    def mppa_logits(self, h: dict[str, torch.Tensor], microbe_index: torch.Tensor, pathway_index: torch.Tensor) -> torch.Tensor:
        if self.mppa_decoder is None:
            raise RuntimeError("MPPA decoder disabled")
        return self.mppa_decoder(h["microbe"][microbe_index], h["pathway"][pathway_index])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
