"""Tests for native TensorRT runtime helpers."""

from pathlib import Path

import pytest
import torch

from perception_rt.build_tensorrt import (
    EXPECTED_INPUT_SHAPE,
    EXPECTED_OUTPUT_SHAPES,
)
from perception_rt.export_onnx import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAMES,
)
from perception_rt.tensorrt_runtime import (
    deserialize_engine,
    resolve_cuda_device,
    validate_engine_contract,
    validate_input_tensor,
)


class FakeTensorIOMode:
    INPUT = "input"
    OUTPUT = "output"


class FakeTensorRT:
    TensorIOMode = FakeTensorIOMode
    float32 = "float32"


class FakeEngine:
    def __init__(self) -> None:
        self.names = (ONNX_INPUT_NAME, *ONNX_OUTPUT_NAMES)
        self.shapes = {
            ONNX_INPUT_NAME: EXPECTED_INPUT_SHAPE,
            **EXPECTED_OUTPUT_SHAPES,
        }
        self.dtypes = {name: FakeTensorRT.float32 for name in self.names}
        self.modes = {
            name: (FakeTensorIOMode.INPUT if name == ONNX_INPUT_NAME else FakeTensorIOMode.OUTPUT)
            for name in self.names
        }

    @property
    def num_io_tensors(self) -> int:
        return len(self.names)

    def get_tensor_name(self, index: int) -> str:
        return self.names[index]

    def get_tensor_shape(
        self,
        name: str,
    ) -> tuple[int, ...]:
        return self.shapes[name]

    def get_tensor_dtype(self, name: str) -> str:
        return self.dtypes[name]

    def get_tensor_mode(self, name: str) -> str:
        return self.modes[name]


def test_validate_engine_contract_accepts_expected_engine() -> None:
    validate_engine_contract(
        FakeEngine(),
        FakeTensorRT,
    )


def test_validate_engine_contract_rejects_wrong_names() -> None:
    engine = FakeEngine()
    engine.names = tuple(reversed(engine.names))

    with pytest.raises(ValueError, match="Expected engine tensors"):
        validate_engine_contract(engine, FakeTensorRT)


def test_validate_engine_contract_rejects_wrong_shape() -> None:
    engine = FakeEngine()
    engine.shapes["log_depth"] = (1, 1, 160, 320)

    with pytest.raises(ValueError, match="log_depth.*shape"):
        validate_engine_contract(engine, FakeTensorRT)


def test_validate_engine_contract_rejects_wrong_dtype() -> None:
    engine = FakeEngine()
    engine.dtypes["semantic_logits"] = "float16"

    with pytest.raises(ValueError, match="semantic_logits.*FP32"):
        validate_engine_contract(engine, FakeTensorRT)


def test_validate_engine_contract_rejects_wrong_mode() -> None:
    engine = FakeEngine()
    engine.modes["log_depth"] = FakeTensorIOMode.INPUT

    with pytest.raises(ValueError, match="invalid I/O mode"):
        validate_engine_contract(engine, FakeTensorRT)


def test_validate_input_tensor_rejects_wrong_shape() -> None:
    image = torch.zeros(1, 3, 160, 320)

    with pytest.raises(ValueError, match="Expected image shape"):
        validate_input_tensor(image, image.device)


def test_validate_input_tensor_rejects_wrong_dtype() -> None:
    image = torch.zeros(
        EXPECTED_INPUT_SHAPE,
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="torch.float32"):
        validate_input_tensor(image, image.device)


def test_validate_input_tensor_rejects_wrong_device() -> None:
    image = torch.zeros(EXPECTED_INPUT_SHAPE)

    with pytest.raises(ValueError, match="Expected input on"):
        validate_input_tensor(
            image,
            torch.device("cuda:0"),
        )


def test_validate_input_tensor_rejects_non_contiguous() -> None:
    image = torch.zeros(1, 3, 640, 320).transpose(2, 3)

    assert tuple(image.shape) == EXPECTED_INPUT_SHAPE
    assert not image.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        validate_input_tensor(image, image.device)


def test_resolve_cuda_device_rejects_cpu() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        resolve_cuda_device("cpu")


def test_deserialize_engine_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        deserialize_engine(
            tmp_path / "missing.engine",
            FakeTensorRT,
            object(),
        )
