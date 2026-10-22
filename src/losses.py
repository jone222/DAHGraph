from __future__ import annotations

import torch


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    return (per * weights).sum() / weights.sum().clamp_min(1e-8)

