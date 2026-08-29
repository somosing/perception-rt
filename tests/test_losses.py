import torch

from perception_rt.training.losses import (
    log_depth_gradient_loss,
    multitask_loss,
    uncertainty_aware_log_depth_loss,
)


def test_multitask_loss_is_finite_and_backpropagates() -> None:
    semantic_logits = torch.randn(
        2,
        15,
        8,
        16,
        requires_grad=True,
    )
    predicted_log_depth = torch.randn(
        2,
        1,
        8,
        16,
        requires_grad=True,
    )
    predicted_log_scale = torch.zeros(
        2,
        1,
        8,
        16,
        requires_grad=True,
    )
    semantic_target = torch.randint(0, 15, (2, 8, 16))
    target_depth_m = torch.rand(2, 1, 8, 16) * 199.0 + 1.0
    valid_mask = torch.rand(2, 1, 8, 16) > 0.2

    losses = multitask_loss(
        semantic_logits,
        predicted_log_depth,
        predicted_log_scale,
        semantic_target,
        target_depth_m,
        valid_mask,
    )

    assert set(losses) == {
        "total",
        "semantic",
        "depth_nll",
        "depth_gradient",
    }

    for loss in losses.values():
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    losses["total"].backward()

    assert semantic_logits.grad is not None
    assert predicted_log_depth.grad is not None
    assert predicted_log_scale.grad is not None


def test_invalid_depth_pixels_do_not_affect_nll() -> None:
    target = torch.tensor([[[[10.0, 0.0]]]])
    valid = torch.tensor([[[[True, False]]]])
    log_scale = torch.zeros_like(target)

    prediction_a = torch.tensor([[[[torch.log(torch.tensor(10.0)), 0.0]]]])
    prediction_b = torch.tensor([[[[torch.log(torch.tensor(10.0)), 999.0]]]])

    loss_a = uncertainty_aware_log_depth_loss(
        prediction_a,
        log_scale,
        target,
        valid,
    )
    loss_b = uncertainty_aware_log_depth_loss(
        prediction_b,
        log_scale,
        target,
        valid,
    )

    torch.testing.assert_close(loss_a, loss_b)


def test_perfect_log_depth_has_zero_gradient_loss() -> None:
    target = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
    prediction = torch.log(target)
    valid = torch.ones_like(target, dtype=torch.bool)

    loss = log_depth_gradient_loss(
        prediction,
        target,
        valid,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_depth_losses_handle_no_valid_pixels() -> None:
    prediction = torch.randn(1, 1, 2, 2, requires_grad=True)
    log_scale = torch.randn(1, 1, 2, 2, requires_grad=True)
    target = torch.zeros(1, 1, 2, 2)
    valid = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    nll = uncertainty_aware_log_depth_loss(
        prediction,
        log_scale,
        target,
        valid,
    )
    gradient = log_depth_gradient_loss(
        prediction,
        target,
        valid,
    )
    total = nll + gradient

    torch.testing.assert_close(total, torch.tensor(0.0))
    total.backward()

    assert prediction.grad is not None
