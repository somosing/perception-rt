from pathlib import Path

import pytest
import torch

from perception_rt.data.vkitti2 import SemanticClass
from perception_rt.training.weights import (
    load_semantic_class_weights,
)

CLASSES = (
    SemanticClass(0, "Road", (100, 60, 100)),
    SemanticClass(1, "Sky", (90, 200, 255)),
)


def test_load_weights_orders_values_by_class_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weights.yaml"
    path.write_text(
        "Sky: 2.0\nRoad: 0.5\n",
        encoding="utf-8",
    )

    weights = load_semantic_class_weights(
        path,
        CLASSES,
    )

    torch.testing.assert_close(
        weights,
        torch.tensor([0.5, 2.0]),
    )


def test_load_weights_rejects_missing_class(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weights.yaml"
    path.write_text(
        "Road: 0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing class weights.*Sky",
    ):
        load_semantic_class_weights(path, CLASSES)
