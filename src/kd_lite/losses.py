"""Regularization losses used during base training and pruning."""

from __future__ import annotations

import torch
from torch import nn


class InstanceFeatureSparseLoss(nn.Module):
    """Instance-wise L2,1-norm feature sparsity regularizer.

    For a feature map of shape ``(B, C, T)``, computes per-instance
    L2 norm along time, then L1 across channels. Encourages each
    sample to activate only a sparse subset of filters.

    Args:
        lambda_reg: Regularization weight applied to the final scalar.
    """

    def __init__(self, lambda_reg: float = 1e-5) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        # L2 norm along time -> (B, C)
        norm_t = torch.norm(feature_maps, p=2, dim=-1)
        # L1 norm across channels -> (B,)
        norm_c = torch.norm(norm_t, p=1, dim=-1)
        return self.lambda_reg * norm_c.sum()


def feature_sparsity_loss(
    features: list[torch.Tensor],
    weights: list[float] | None = None,
    lambda_reg: float = 1e-5,
    inception_first_layer_filters: int | None = 96,
) -> torch.Tensor:
    """Weighted sum of per-layer sparsity penalties.

    Args:
        features: Per-stage feature maps from the model's forward pass.
        weights: Optional per-layer multipliers (e.g. ``[4, 2, 1]`` to
            penalize early-layer redundancy more aggressively).
        lambda_reg: Base regularization weight.
        inception_first_layer_filters: For LITE, restrict the first-layer
            penalty to the learnable filters (excludes hybrid filters).
            Pass ``None`` to penalize the whole tensor.
    """
    if weights is None:
        weights = [1.0] * len(features)
    if len(weights) != len(features):
        raise ValueError(
            f"weights length ({len(weights)}) must match features length ({len(features)})"
        )

    loss_fn = InstanceFeatureSparseLoss(lambda_reg=lambda_reg)
    total = features[0].new_zeros(())
    for i, (feat, w) in enumerate(zip(features, weights)):
        slice_ = feat
        if i == 0 and inception_first_layer_filters is not None:
            slice_ = feat[:, :inception_first_layer_filters]
        total = total + w * loss_fn(slice_)
    return total
