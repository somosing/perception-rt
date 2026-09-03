"""Tests for selective INT8 ONNX quantization helpers."""

import numpy as np
import pytest
import torch
from onnx import helper

from perception_rt.export_onnx import ONNX_INPUT_NAME
from perception_rt.quantize_onnx import (
    Scene06CalibrationReader,
    select_calibration_indices,
    select_sr_conv_node_names,
)


class DummyDataset:
    """Return small deterministic calibration tensors."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": torch.full((3, 2, 4), float(index))}


def test_select_calibration_indices_is_even_and_deterministic() -> None:
    assert select_calibration_indices(10, 4) == (0, 3, 6, 9)


@pytest.mark.parametrize(
    ("dataset_size", "sample_count"),
    [(0, 1), (10, 0), (10, 11)],
)
def test_select_calibration_indices_rejects_invalid_counts(
    dataset_size: int,
    sample_count: int,
) -> None:
    with pytest.raises(ValueError):
        select_calibration_indices(dataset_size, sample_count)


def test_select_sr_conv_node_names_uses_operator_and_component() -> None:
    nodes = [
        helper.make_node(
            "Conv",
            ["image", "model.encoder.block.attention.self.sr.weight"],
            ["sr_output"],
            name="sr_conv",
        ),
        helper.make_node(
            "Conv",
            ["image", "model.encoder.patch_embeddings.weight"],
            ["patch_output"],
            name="patch_conv",
        ),
        helper.make_node(
            "Gemm",
            ["image", "model.encoder.block.attention.self.sr.weight"],
            ["gemm_output"],
            name="sr_gemm",
        ),
    ]
    graph = helper.make_graph(nodes, "test", [], [])
    model = helper.make_model(graph)

    assert select_sr_conv_node_names(model) == ("sr_conv",)


def test_calibration_reader_returns_fp32_batches_and_rewinds() -> None:
    reader = Scene06CalibrationReader(DummyDataset(), (1, 3))

    first = reader.get_next()
    second = reader.get_next()

    assert first is not None
    assert second is not None
    assert first[ONNX_INPUT_NAME].shape == (1, 3, 2, 4)
    assert first[ONNX_INPUT_NAME].dtype == np.float32
    assert float(second[ONNX_INPUT_NAME][0, 0, 0, 0]) == 3.0
    assert reader.get_next() is None

    reader.rewind()
    assert reader.get_next() is not None
