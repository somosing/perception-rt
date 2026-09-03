"""Tests for TensorRT parity validation."""

import math

import pytest
import torch

from perception_rt.validate_onnx import (
    DEFAULT_ABSOLUTE_TOLERANCE,
)
from perception_rt.validate_tensorrt import (
    DEFAULT_FP16_MINIMUM_WITHIN_TOLERANCE_FRACTION,
    DEFAULT_UNCERTAINTY_RELATIVE_TOLERANCE,
    run_tensorrt_sample_parity,
    validate_sample_indices,
)


def test_fp16_requires_ninety_nine_percent_pixel_coverage() -> None:
    assert DEFAULT_FP16_MINIMUM_WITHIN_TOLERANCE_FRACTION == 0.99


def test_uncertainty_tolerance_is_log_equivalent() -> None:
    assert DEFAULT_UNCERTAINTY_RELATIVE_TOLERANCE == (
        pytest.approx(math.expm1(DEFAULT_ABSOLUTE_TOLERANCE))
    )


def test_validate_sample_indices_accepts_valid_indices() -> None:
    validate_sample_indices((0, 2, 4), 5)


@pytest.mark.parametrize("index", [-1, 5])
def test_validate_sample_indices_rejects_invalid_index(
    index: int,
) -> None:
    with pytest.raises(IndexError, match="outside"):
        validate_sample_indices((index,), 5)


def test_run_tensorrt_sample_parity_uses_same_input() -> None:
    class TinyModel(torch.nn.Module):
        def forward(
            self,
            image: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {
                "semantic_logits": image,
                "log_depth": image[:, :1],
                "depth_log_scale": torch.zeros_like(image[:, :1]),
            }

    class MatchingRunner:
        dtype = torch.float16

        def infer(
            self,
            image: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            assert image.is_contiguous()
            assert image.dtype == self.dtype
            return {
                "semantic_logits": image,
                "log_depth": image[:, :1],
                "depth_log_scale": torch.zeros_like(image[:, :1]),
            }

    sample = {
        "image": torch.zeros(
            (3, 2, 2),
            dtype=torch.float32,
        ),
        "scene": "Scene18",
        "variation": "clone",
        "frame": 0,
    }

    result = run_tensorrt_sample_parity(
        model=TinyModel(),
        runner=MatchingRunner(),
        sample=sample,
        device=torch.device("cpu"),
        maximum_depth_m=200.0,
        absolute_tolerance=0.02,
        relative_tolerance=0.01,
        uncertainty_relative_tolerance=math.expm1(0.02),
        minimum_semantic_agreement=0.999,
    )

    assert result["passed"] is True
    assert result["scene"] == "Scene18"
    assert result["semantic_argmax_agreement"] == 1.0
