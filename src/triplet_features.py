from __future__ import annotations

import torch


def compose_triplet_features(hd: torch.Tensor, hm: torch.Tensor, hy: torch.Tensor) -> torch.Tensor:
    return torch.cat([hd, hm, hy, hd * hm, hm * hy, hd * hy, torch.abs(hd - hm), torch.abs(hm - hy)], dim=-1)

