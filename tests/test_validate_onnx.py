"""Tests for PyTorch–ONNX Runtime parity calculations."""

import numpy as np
import pytest

from perception_rt.validate_onnx import (
    compare_arrays,
    semantic_argmax_agreement,
)


def test_compare_arrays_reports_numerical_parity() -> None:
    reference = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    candidate = np.array(
        [0.00005, 1.0005, 2.001, 3.01],
        dtype=np.float32,
    )

    result = compare_arrays(reference, candidate)

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
