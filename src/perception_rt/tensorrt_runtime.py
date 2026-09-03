"""Native TensorRT inference for the static PerceptionRT engine."""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from perception_rt.build_tensorrt import (
    DEFAULT_ENGINE_PATH,
    EXPECTED_INPUT_SHAPE,
    EXPECTED_OUTPUT_SHAPES,
    load_tensorrt,
)
from perception_rt.export_onnx import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAMES,
)


def validate_engine_contract(
    engine: Any,
    trt: Any,
) -> torch.dtype:
    """Validate the engine contract and return its PyTorch dtype."""
    expected_names = (ONNX_INPUT_NAME, *ONNX_OUTPUT_NAMES)

    if engine.num_io_tensors != len(expected_names):
        raise ValueError(
            f"Expected {len(expected_names)} engine tensors, found {engine.num_io_tensors}"
        )

    names = tuple(engine.get_tensor_name(index) for index in range(engine.num_io_tensors))
    if names != expected_names:
        raise ValueError(f"Expected engine tensors {expected_names}, found {names}")

    expected_shapes = {
        ONNX_INPUT_NAME: EXPECTED_INPUT_SHAPE,
        **EXPECTED_OUTPUT_SHAPES,
    }

    for name in expected_names:
        shape = tuple(engine.get_tensor_shape(name))
        expected_shape = expected_shapes[name]

        if shape != expected_shape:
            raise ValueError(f"Expected {name!r} shape {expected_shape}, found {shape}")

        expected_mode = (
            trt.TensorIOMode.INPUT if name == ONNX_INPUT_NAME else trt.TensorIOMode.OUTPUT
        )
        if engine.get_tensor_mode(name) != expected_mode:
            raise ValueError(f"Tensor {name!r} has an invalid I/O mode")

    engine_dtypes = {name: engine.get_tensor_dtype(name) for name in expected_names}
    unique_dtypes = set(engine_dtypes.values())

    if len(unique_dtypes) != 1:
        details = ", ".join(f"{name}={dtype}" for name, dtype in engine_dtypes.items())
        raise ValueError(f"Expected all engine tensors to use one dtype; found {details}")

    engine_dtype = next(iter(unique_dtypes))
    dtype_mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
    }

    try:
        return dtype_mapping[engine_dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported TensorRT engine dtype: {engine_dtype}") from error


def deserialize_engine(
    engine_path: Path,
    trt: Any,
    logger: Any,
) -> tuple[Any, Any]:
    """Load a serialized TensorRT engine while retaining its runtime."""
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine does not exist: {engine_path}")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())

    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

    return runtime, engine


def resolve_cuda_device(
    device: str | torch.device,
) -> torch.device:
    """Resolve and validate the requested CUDA device."""
    requested = torch.device(device)

    if requested.type != "cuda":
        raise ValueError("TensorRT inference requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device index: {index}")

    return torch.device("cuda", index)


def validate_input_tensor(
    image: Tensor,
    device: torch.device,
    *,
    expected_dtype: torch.dtype = torch.float32,
) -> None:
    """Validate the static TensorRT input contract."""
    if tuple(image.shape) != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"Expected image shape {EXPECTED_INPUT_SHAPE}, found {tuple(image.shape)}")
    if image.dtype != expected_dtype:
        raise ValueError(f"TensorRT input must use {expected_dtype}")
    if image.device != device:
        raise ValueError(f"Expected input on {device}, found {image.device}")
    if not image.is_contiguous():
        raise ValueError("TensorRT input must be contiguous")


class TensorRTRunner:
    """Execute a static FP32 or FP16 engine with reusable CUDA outputs."""

    def __init__(
        self,
        engine_path: Path = DEFAULT_ENGINE_PATH,
        *,
        device: str | torch.device = "cuda",
        logger: Any | None = None,
    ) -> None:
        self.device = resolve_cuda_device(device)
        self.trt = load_tensorrt()
        self.logger = logger if logger is not None else self.trt.Logger(self.trt.Logger.WARNING)

        with torch.cuda.device(self.device):
            self.runtime, self.engine = deserialize_engine(
                engine_path,
                self.trt,
                self.logger,
            )
            self.dtype = validate_engine_contract(
                self.engine,
                self.trt,
            )

            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")

            self.stream = torch.cuda.Stream(device=self.device)
            self._outputs = {
                name: torch.empty(
                    EXPECTED_OUTPUT_SHAPES[name],
                    dtype=self.dtype,
                    device=self.device,
                )
                for name in ONNX_OUTPUT_NAMES
            }

            for name, tensor in self._outputs.items():
                if not self.context.set_tensor_address(
                    name,
                    tensor.data_ptr(),
                ):
                    raise RuntimeError(f"Failed to bind TensorRT output {name!r}")

    def infer(self, image: Tensor) -> dict[str, Tensor]:
        """Run synchronous inference; outputs are reused next call."""
        validate_input_tensor(
            image,
            self.device,
            expected_dtype=self.dtype,
        )

        with torch.cuda.device(self.device):
            caller_stream = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(caller_stream)

            if not self.context.set_tensor_address(
                ONNX_INPUT_NAME,
                image.data_ptr(),
            ):
                raise RuntimeError("Failed to bind TensorRT input")

            executed = self.context.execute_async_v3(self.stream.cuda_stream)
            if not executed:
                raise RuntimeError("TensorRT execution failed")

            self.stream.synchronize()

        return dict(self._outputs)
