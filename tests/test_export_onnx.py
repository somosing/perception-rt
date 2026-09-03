from pathlib import Path

import onnx
import pytest
import torch
from torch import Tensor, nn

from perception_rt.export_onnx import (
    DEFAULT_ONNX_OPSET,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAMES,
    ExportablePerceptionRTModel,
    export_model_to_onnx,
)


class FakeMultitaskModel(nn.Module):
    def forward(self, image: Tensor) -> dict[str, Tensor]:
        batch_size, _, height, width = image.shape

        return {
            "semantic_logits": torch.full(
                (batch_size, 15, height, width),
                1.0,
            ),
            "log_depth": torch.full(
                (batch_size, 1, height, width),
                2.0,
            ),
            "depth_log_scale": torch.full(
                (batch_size, 1, height, width),
                3.0,
            ),
        }


class SimpleMultitaskModel(nn.Module):
    def forward(self, image: Tensor) -> dict[str, Tensor]:
        base = image.mean(
            dim=1,
            keepdim=True,
        )

        return {
            "semantic_logits": base.repeat(1, 15, 1, 1),
            "log_depth": base,
            "depth_log_scale": base + 0.5,
        }


def test_onnx_contract_names_and_opset() -> None:
    assert ONNX_INPUT_NAME == "image"
    assert ONNX_OUTPUT_NAMES == (
        "semantic_logits",
        "log_depth",
        "depth_log_scale",
    )
    assert DEFAULT_ONNX_OPSET == 18


def test_exportable_model_returns_stable_output_order() -> None:
    image = torch.zeros(1, 3, 32, 64)
    model = ExportablePerceptionRTModel(
        FakeMultitaskModel(),
    )

    semantic, log_depth, depth_log_scale = model(image)

    assert semantic.shape == (1, 15, 32, 64)
    assert log_depth.shape == (1, 1, 32, 64)
    assert depth_log_scale.shape == (1, 1, 32, 64)
    assert torch.all(semantic == 1.0)
    assert torch.all(log_depth == 2.0)
    assert torch.all(depth_log_scale == 3.0)


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("fp32", onnx.TensorProto.FLOAT),
        ("fp16", onnx.TensorProto.FLOAT16),
    ],
)
def test_export_writes_valid_static_contract(
    tmp_path: Path,
    precision: str,
    expected_dtype: int,
) -> None:
    output_path = tmp_path / f"model_{precision}.onnx"

    result = export_model_to_onnx(
        SimpleMultitaskModel(),
        output_path,
        input_size=(32, 64),
        precision=precision,
    )

    graph = onnx.load(str(result)).graph
    input_shape = tuple(
        dimension.dim_value for dimension in graph.input[0].type.tensor_type.shape.dim
    )

    assert result == output_path
    assert input_shape == (1, 3, 32, 64)
    assert [value.name for value in graph.input] == [
        ONNX_INPUT_NAME,
    ]
    assert [value.name for value in graph.output] == list(
        ONNX_OUTPUT_NAMES,
    )
    assert graph.input[0].type.tensor_type.elem_type == expected_dtype
    assert {value.type.tensor_type.elem_type for value in graph.output} == {expected_dtype}


def test_export_rejects_unknown_precision(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported ONNX precision",
    ):
        export_model_to_onnx(
            SimpleMultitaskModel(),
            tmp_path / "invalid.onnx",
            input_size=(32, 64),
            precision="int4",
        )
