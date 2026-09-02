"""Tests for PyTorch–ONNX Runtime parity calculations."""

import numpy as np
import pytest
import torch

from perception_rt.validate_onnx import (
    compare_arrays,
    compare_output_sets,
    run_sample_parity,
    semantic_argmax_agreement,
    summarize_parity_results,
)


def test_compare_arrays_reports_numerical_parity() -> None:
    reference = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    candidate = np.array(
        [0.00005, 1.0005, 2.001, 3.01],
        dtype=np.float32,
    )

    result = compare_arrays(
        reference,
        candidate,
        absolute_tolerance=1e-4,
        relative_tolerance=1e-3,
    )

    assert result["maximum_absolute_error"] == pytest.approx(0.01)
    assert result["mean_absolute_error"] == pytest.approx(0.0028875, abs=1e-7)
    assert result["within_tolerance_fraction"] == pytest.approx(0.75)
    assert result["all_within_tolerance"] is False


def test_compare_arrays_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_arrays(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
        )


def test_semantic_argmax_agreement() -> None:
    reference = np.array(
        [[[[3.0, 0.0]], [[1.0, 2.0]], [[0.0, 1.0]]]],
        dtype=np.float32,
    )
    candidate = np.array(
        [[[[2.0, 0.0]], [[1.0, 1.0]], [[0.0, 3.0]]]],
        dtype=np.float32,
    )

    assert semantic_argmax_agreement(reference, candidate) == 0.5


def test_compare_output_sets_includes_deployment_outputs() -> None:
    reference = {
        "semantic_logits": np.array(
            [[[[2.0]], [[1.0]], [[0.0]]]],
            dtype=np.float32,
        ),
        "log_depth": np.log(np.array([[[[2.0]]]], dtype=np.float32)),
        "depth_log_scale": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }
    candidate = {name: value.copy() for name, value in reference.items()}
    candidate["semantic_logits"] += 1e-5
    candidate["log_depth"] += 1e-5

    result = compare_output_sets(
        reference,
        candidate,
        maximum_depth_m=200.0,
    )

    assert result["passed"] is True
    assert result["semantic_argmax_agreement"] == 1.0
    assert set(result["raw_outputs"]) == {
        "semantic_logits",
        "log_depth",
        "depth_log_scale",
    }
    assert set(result["postprocessed_outputs"]) == {
        "depth_m",
        "uncertainty_scale",
    }


def test_run_sample_parity_uses_identical_model_input() -> None:
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

    class MatchingSession:
        def run(
            self,
            output_names: list[str],
            inputs: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            image = inputs["image"]

            assert output_names == [
                "semantic_logits",
                "log_depth",
                "depth_log_scale",
            ]

            return [
                image,
                image[:, :1],
                np.zeros_like(image[:, :1]),
            ]

    sample = {
        "image": torch.zeros((3, 2, 2), dtype=torch.float32),
        "scene": "Scene18",
        "variation": "clone",
        "frame": 0,
    }

    result = run_sample_parity(
        model=TinyModel(),
        session=MatchingSession(),
        sample=sample,
        device=torch.device("cpu"),
        maximum_depth_m=200.0,
    )

    assert result["passed"] is True
    assert result["scene"] == "Scene18"
    assert result["semantic_argmax_agreement"] == 1.0


def test_summarize_parity_results_reports_worst_case() -> None:
    def make_result(
        maximum_error: float,
        agreement: float,
        passed: bool,
    ) -> dict[str, object]:
        metric = {
            "maximum_absolute_error": maximum_error,
            "mean_absolute_error": maximum_error / 2.0,
            "within_tolerance_fraction": 1.0 if passed else 0.99,
            "all_within_tolerance": passed,
        }

        return {
            "raw_outputs": {
                "semantic_logits": metric,
                "log_depth": metric,
                "depth_log_scale": metric,
            },
            "postprocessed_outputs": {
                "depth_m": metric,
                "uncertainty_scale": metric,
            },
            "semantic_argmax_agreement": agreement,
            "passed": passed,
        }

    summary = summarize_parity_results(
        [
            make_result(0.01, 1.0, True),
            make_result(0.02, 0.999, False),
        ]
    )

    assert summary["sample_count"] == 2
    assert summary["passed"] is False
    assert summary["minimum_semantic_argmax_agreement"] == 0.999
    assert summary["raw_outputs"]["semantic_logits"]["maximum_absolute_error"] == 0.02


def test_compare_output_sets_enforces_semantic_gate() -> None:
    reference = {
        "semantic_logits": np.array(
            [[[[0.005]], [[0.0]]]],
            dtype=np.float32,
        ),
        "log_depth": np.zeros((1, 1, 1, 1), dtype=np.float32),
        "depth_log_scale": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }
    candidate = {
        "semantic_logits": np.array(
            [[[[0.0]], [[0.005]]]],
            dtype=np.float32,
        ),
        "log_depth": np.zeros((1, 1, 1, 1), dtype=np.float32),
        "depth_log_scale": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }

    result = compare_output_sets(
        reference,
        candidate,
        maximum_depth_m=200.0,
    )

    assert result["outputs_within_tolerance"] is True
    assert result["semantic_argmax_agreement"] == 0.0
    assert result["passed"] is False


def test_uncertainty_scale_supports_log_equivalent_tolerance() -> None:
    reference = {
        "semantic_logits": np.array(
            [[[[2.0]], [[1.0]]]],
            dtype=np.float32,
        ),
        "log_depth": np.zeros(
            (1, 1, 1, 1),
            dtype=np.float32,
        ),
        "depth_log_scale": np.full(
            (1, 1, 1, 1),
            5.0,
            dtype=np.float32,
        ),
    }
    candidate = {name: value.copy() for name, value in reference.items()}
    candidate["depth_log_scale"] += 0.015

    strict = compare_output_sets(
        reference,
        candidate,
        maximum_depth_m=200.0,
        absolute_tolerance=0.02,
        relative_tolerance=0.0,
    )
    log_equivalent = compare_output_sets(
        reference,
        candidate,
        maximum_depth_m=200.0,
        absolute_tolerance=0.02,
        relative_tolerance=0.0,
        uncertainty_relative_tolerance=np.expm1(0.02),
    )

    assert strict["passed"] is False
    assert log_equivalent["passed"] is True
