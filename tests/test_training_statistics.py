import numpy as np
import pytest

from perception_rt.training_statistics import (
    compute_class_weights,
)


def test_class_weights_emphasize_rare_observed_classes() -> None:
    counts = np.array([900, 100, 0], dtype=np.int64)

    weights = compute_class_weights(counts)

    assert weights[1] > weights[0]
    assert weights[2] == 0.0
    np.testing.assert_allclose(
        weights[:2].mean(),
        1.0,
    )


def test_class_weights_reject_empty_counts() -> None:
    with pytest.raises(
        ValueError,
        match="At least one class pixel",
    ):
        compute_class_weights(np.zeros(3, dtype=np.int64))
