from pathlib import Path

import pytest
import torch
from torch import nn

from perception_rt.data.vkitti2 import SemanticClass
from perception_rt.evaluate import (
    load_checkpoint_model,
    named_per_class_iou,
)


def test_load_checkpoint_model_restores_weights(
    tmp_path: Path,
) -> None:
    source = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "model": source.state_dict(),
            "epoch": 7,
            "global_step": 123,
            "best_validation_loss": -0.25,
        },
        path,
    )

    metadata = load_checkpoint_model(target, path)

    torch.testing.assert_close(
        target.weight,
        source.weight,
    )
    torch.testing.assert_close(
        target.bias,
        source.bias,
    )
    assert metadata == {
        "epoch": 7,
        "global_step": 123,
        "best_validation_loss": -0.25,
    }


def test_named_per_class_iou_handles_absent_class() -> None:
    classes = (
        SemanticClass(0, "Road", (100, 60, 100)),
        SemanticClass(1, "Sky", (90, 200, 255)),
    )
    iou = torch.tensor(
        [0.75, float("nan")],
        dtype=torch.float64,
    )

    result = named_per_class_iou(iou, classes)

    assert result["Road"] == pytest.approx(0.75)
    assert result["Sky"] is None
