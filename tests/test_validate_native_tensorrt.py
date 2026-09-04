"""Tests for native C++ TensorRT parity calculations."""

import numpy as np
import pytest

from perception_rt.validate_native_tensorrt import (
    compare_native_arrays,
    semantic_argmax_agreement,
    summarize_native_results,
    validate_sample_indices,
)


def test_compare_native_arrays_reports_exact_parity() -> None:
    reference = np.array([0.0, 1.0, 2.0], dtype=np.float16)

    result = compare_native_arrays(reference, reference.copy())

    assert result == {
        "all_within_tolerance": True,
        "exact_match_fraction": 1.0,
        "maximum_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
    }


def test_compare_native_arrays_applies_tolerances() -> None:
    reference = np.array([1.0, 2.0], dtype=np.float16)
    candidate = np.array([1.001, 2.004], dtype=np.float16)

    result = compare_native_arrays(
        reference,
        candidate,
        absolute_tolerance=0.005,
        relative_tolerance=0.0,
    )

    assert result["all_within_tolerance"] is True
    assert result["exact_match_fraction"] == 0.0
    assert result["maximum_absolute_error"] == pytest.approx(0.00390625)


def test_compare_native_arrays_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_native_arrays(
            np.zeros((2,), dtype=np.float16),
            np.zeros((1, 2), dtype=np.float16),
        )


def test_semantic_argmax_agreement() -> None:
    reference = np.array([[[[2.0]], [[1.0]]]], dtype=np.float16)
    candidate = np.array([[[[3.0]], [[0.0]]]], dtype=np.float16)

    assert semantic_argmax_agreement(reference, candidate) == 1.0


def test_summarize_native_results_uses_worst_sample() -> None:
    def result(maximum: float, exact: float, passed: bool) -> dict[str, object]:
        metric = {
            "all_within_tolerance": passed,
            "exact_match_fraction": exact,
            "maximum_absolute_error": maximum,
            "mean_absolute_error": maximum / 2.0,
        }
        return {
            "outputs": {
                name: dict(metric)
                for name in (
                    "semantic_logits",
                    "log_depth",
                    "depth_log_scale",
                )
            },
            "semantic_argmax_agreement": exact,
        }

    summary = summarize_native_results([result(0.0, 1.0, True), result(0.01, 0.9, False)])

    assert summary["sample_count"] == 2
    assert summary["passed"] is False
    assert summary["minimum_semantic_argmax_agreement"] == 0.9
    assert summary["outputs"]["log_depth"] == {
        "all_within_tolerance": False,
        "maximum_absolute_error": 0.01,
        "maximum_mean_absolute_error": 0.005,
        "minimum_exact_match_fraction": 0.9,
    }


def test_validate_sample_indices() -> None:
    assert validate_sample_indices([0, 5, 9], 10) == (0, 5, 9)

    with pytest.raises(ValueError, match="unique"):
        validate_sample_indices([0, 0], 10)

    with pytest.raises(ValueError, match=r"within \[0, 10\)"):
        validate_sample_indices([10], 10)
