"""Loss functions for multitask perception training."""

import math
from typing import TypedDict

import torch
from torch import Tensor
from torch.nn import functional as functional


class LossOutput(TypedDict):
    """Total loss and individually reportable task components."""

    total: Tensor
    semantic: Tensor
    depth_nll: Tensor
    depth_gradient: Tensor


def semantic_cross_entropy(
    semantic_logits: Tensor,
    semantic_target: Tensor,
    *,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Compute dense semantic cross-entropy."""
    if semantic_logits.ndim != 4:
        raise ValueError("Semantic logits must have shape [B, C, H, W]")

    if semantic_target.ndim != 3:
        raise ValueError("Semantic target must have shape [B, H, W]")

    return functional.cross_entropy(
        semantic_logits,
        semantic_target,
        weight=class_weights,
    )


def uncertainty_aware_log_depth_loss(
    predicted_log_depth: Tensor,
    predicted_log_scale: Tensor,
    target_depth_m: Tensor,
    valid_mask: Tensor,
    *,
    minimum_log_scale: float = -6.0,
    maximum_log_scale: float = 6.0,
) -> Tensor:
    """Compute masked heteroscedastic Laplace loss in log-depth space."""
    if predicted_log_depth.shape != target_depth_m.shape:
        raise ValueError("Predicted and target depth shapes must match")

    if predicted_log_scale.shape != target_depth_m.shape:
        raise ValueError("Depth and uncertainty shapes must match")

    if valid_mask.shape != target_depth_m.shape:
        raise ValueError("Depth validity shape must match target depth")

    valid_mask = valid_mask.to(torch.bool)

    if not torch.any(valid_mask):
        return predicted_log_depth.sum() * 0.0

    target_log_depth = torch.log(target_depth_m[valid_mask])
    prediction = predicted_log_depth[valid_mask]
    log_scale = predicted_log_scale[valid_mask].clamp(
        minimum_log_scale,
        maximum_log_scale,
    )

    absolute_residual = torch.abs(prediction - target_log_depth)
    negative_log_likelihood = torch.exp(-log_scale) * absolute_residual + log_scale + math.log(2.0)

    return negative_log_likelihood.mean()


def log_depth_gradient_loss(
    predicted_log_depth: Tensor,
    target_depth_m: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Preserve local depth structure using masked horizontal/vertical gradients."""
    if predicted_log_depth.shape != target_depth_m.shape:
        raise ValueError("Predicted and target depth shapes must match")

    if valid_mask.shape != target_depth_m.shape:
        raise ValueError("Depth validity shape must match target depth")

    valid_mask = valid_mask.to(torch.bool)
    safe_target = torch.where(
        valid_mask,
        target_depth_m,
        torch.ones_like(target_depth_m),
    )
    target_log_depth = torch.log(safe_target)

    predicted_dx = predicted_log_depth[..., :, 1:] - predicted_log_depth[..., :, :-1]
    target_dx = target_log_depth[..., :, 1:] - target_log_depth[..., :, :-1]
    valid_dx = valid_mask[..., :, 1:] & valid_mask[..., :, :-1]

    predicted_dy = predicted_log_depth[..., 1:, :] - predicted_log_depth[..., :-1, :]
    target_dy = target_log_depth[..., 1:, :] - target_log_depth[..., :-1, :]
    valid_dy = valid_mask[..., 1:, :] & valid_mask[..., :-1, :]

    losses = []

    if torch.any(valid_dx):
        losses.append(torch.abs(predicted_dx[valid_dx] - target_dx[valid_dx]).mean())

    if torch.any(valid_dy):
        losses.append(torch.abs(predicted_dy[valid_dy] - target_dy[valid_dy]).mean())

    if not losses:
        return predicted_log_depth.sum() * 0.0

    return torch.stack(losses).mean()


def multitask_loss(
    semantic_logits: Tensor,
    predicted_log_depth: Tensor,
    predicted_log_scale: Tensor,
    semantic_target: Tensor,
    target_depth_m: Tensor,
    depth_valid: Tensor,
    *,
    class_weights: Tensor | None = None,
    semantic_weight: float = 1.0,
    depth_weight: float = 1.0,
    gradient_weight: float = 0.5,
) -> LossOutput:
    """Combine semantic, uncertainty-aware depth and structural losses."""
    semantic = semantic_cross_entropy(
        semantic_logits,
        semantic_target,
        class_weights=class_weights,
    )
    depth_nll = uncertainty_aware_log_depth_loss(
        predicted_log_depth,
        predicted_log_scale,
        target_depth_m,
        depth_valid,
    )
    depth_gradient = log_depth_gradient_loss(
        predicted_log_depth,
        target_depth_m,
        depth_valid,
    )

    total = semantic_weight * semantic + depth_weight * depth_nll + gradient_weight * depth_gradient

    return LossOutput(
        total=total,
        semantic=semantic,
        depth_nll=depth_nll,
        depth_gradient=depth_gradient,
    )
