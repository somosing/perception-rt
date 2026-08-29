import pytest
import torch

from perception_rt.training.metrics import (
    DenseMetricAccumulator,
)


def test_perfect_predictions_produce_perfect_metrics() -> None:
    semantic_target = torch.tensor([[[0, 1], [1, 0]]])
    semantic_logits = torch.full((1, 2, 2, 2), -10.0)
    semantic_logits.scatter_(
        1,
        semantic_target.unsqueeze(1),
        10.0,
    )

    target_depth = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
    valid = torch.ones_like(
        target_depth,
        dtype=torch.bool,
    )

    metrics = DenseMetricAccumulator(
        2,
        maximum_depth_m=200.0,
    )
    metrics.update(
        semantic_logits,
        torch.log(target_depth),
        torch.zeros_like(target_depth),
        semantic_target,
        target_depth,
        valid,
    )
    result = metrics.compute()

    assert result["mean_iou"] == 1.0
    assert result["depth_abs_rel"] == pytest.approx(
        0.0,
        abs=1e-7,
    )
    assert result["depth_rmse_m"] == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert result["depth_delta1"] == 1.0


def test_uncertainty_correlates_with_log_depth_error() -> None:
    target_depth = torch.ones(1, 1, 1, 3)
    log_errors = torch.tensor([[[[0.1, 0.2, 0.3]]]])
    uncertainties = torch.tensor([[[[0.1, 0.2, 0.3]]]])

    semantic_target = torch.zeros(
        1,
        1,
        3,
        dtype=torch.int64,
    )
    semantic_logits = torch.zeros(1, 2, 1, 3)
    semantic_logits[:, 0] = 1.0

    metrics = DenseMetricAccumulator(
        2,
        maximum_depth_m=200.0,
    )
    metrics.update(
        semantic_logits,
        log_errors,
        torch.log(uncertainties),
        semantic_target,
        target_depth,
        torch.ones_like(target_depth, dtype=torch.bool),
    )

    assert metrics.uncertainty_error_pearson() > 0.999
