import numpy as np
import pytest
import torch

from perception_rt.data.torch_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from perception_rt.qualitative import (
    denormalize_image,
    validate_indices,
)


def test_denormalize_image_restores_rgb_values() -> None:
    original = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.3, 0.4], [0.5, 0.6]],
        ],
        dtype=torch.float32,
    )
    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=torch.float32,
    ).view(3, 1, 1)
    standard_deviation = torch.tensor(
        IMAGENET_STD,
        dtype=torch.float32,
    ).view(3, 1, 1)
    normalized = (original - mean) / standard_deviation

    restored = denormalize_image(normalized)

    np.testing.assert_allclose(
        restored,
        original.permute(1, 2, 0).numpy(),
        atol=1e-6,
    )


def test_validate_indices_rejects_out_of_range() -> None:
    assert validate_indices((0, 4), 5) == (0, 4)

    with pytest.raises(
        IndexError,
        match="outside",
    ):
        validate_indices((5,), 5)
