"""Build a static FP32 TensorRT engine from the PerceptionRT ONNX model."""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

from perception_rt.export_onnx import ONNX_INPUT_NAME, ONNX_OUTPUT_NAMES

DEFAULT_ONNX_PATH = Path("outputs/onnx/perception_rt_mit_b2_fp32.onnx")
DEFAULT_ENGINE_PATH = Path("outputs/tensorrt/perception_rt_mit_b2_fp32.engine")
DEFAULT_WORKSPACE_MIB = 1024
EXPECTED_INPUT_SHAPE = (1, 3, 320, 640)
EXPECTED_OUTPUT_SHAPES = {
    "semantic_logits": (1, 15, 320, 640),
    "log_depth": (1, 1, 320, 640),
    "depth_log_scale": (1, 1, 320, 640),
}


def load_tensorrt() -> Any:
    """Import the optional TensorRT API with a useful error."""
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT is unavailable; install the tensorrt extra") from error

    return trt


def collect_parser_errors(parser: Any) -> tuple[str, ...]:
    """Return all TensorRT ONNX parser diagnostics."""
    return tuple(str(parser.get_error(index)) for index in range(parser.num_errors))


def parse_onnx_network(parser: Any, onnx_path: Path) -> None:
    """Parse an ONNX model and report every parser error."""
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {onnx_path}")

    if parser.parse_from_file(str(onnx_path)):
        return

    errors = collect_parser_errors(parser)
    details = "\n".join(errors) if errors else "No diagnostics reported."
    raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{details}")


def validate_network_contract(
    network: Any,
    *,
    expected_dtype: Any | None = None,
) -> None:
    """Validate TensorRT tensor names, shapes and dtypes."""
    if network.num_inputs != 1:
        raise ValueError(f"Expected one input, found {network.num_inputs}")

    input_tensor = network.get_input(0)
    input_shape = tuple(input_tensor.shape)

    if input_tensor.name != ONNX_INPUT_NAME:
        raise ValueError(f"Expected input {ONNX_INPUT_NAME!r}, found {input_tensor.name!r}")
    if input_shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"Expected input shape {EXPECTED_INPUT_SHAPE}, found {input_shape}")

    outputs = [network.get_output(index) for index in range(network.num_outputs)]
    output_names = tuple(tensor.name for tensor in outputs)

    if output_names != ONNX_OUTPUT_NAMES:
        raise ValueError(f"Expected outputs {ONNX_OUTPUT_NAMES}, found {output_names}")

    tensors = [input_tensor]
    for tensor in outputs:
        shape = tuple(tensor.shape)
        expected_shape = EXPECTED_OUTPUT_SHAPES[tensor.name]

        if shape != expected_shape:
            raise ValueError(f"Expected {tensor.name!r} shape {expected_shape}, found {shape}")

        tensors.append(tensor)

    if expected_dtype is not None:
        invalid = [
            f"{tensor.name}={tensor.dtype}" for tensor in tensors if tensor.dtype != expected_dtype
        ]
        if invalid:
            raise ValueError("Expected all tensors to be FP32; found " + ", ".join(invalid))


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    workspace_mib: int = DEFAULT_WORKSPACE_MIB,
    overwrite: bool = False,
) -> Path:
    """Build and save a strongly typed static FP32 engine."""
    if workspace_mib <= 0:
        raise ValueError("Workspace size must be positive")

    if engine_path.exists() and not overwrite:
        raise FileExistsError(f"Engine already exists: {engine_path}; pass --force to replace it")

    if onnx_path.resolve() == engine_path.resolve():
        raise ValueError("Input and output paths must differ")

    trt = load_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    parse_onnx_network(parser, onnx_path)
    validate_network_contract(
        network,
        expected_dtype=trt.float32,
    )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_mib * 1024**2,
    )

    serialized_engine = builder.build_serialized_network(
        network,
        config,
    )
    if serialized_engine is None:
        raise RuntimeError("TensorRT returned no serialized engine")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized_engine))
    return engine_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a static FP32 TensorRT engine.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ENGINE_PATH,
    )
    parser.add_argument(
        "--workspace-mib",
        type=int,
        default=DEFAULT_WORKSPACE_MIB,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    """Build the configured TensorRT engine."""
    arguments = parse_arguments()
    started = perf_counter()

    engine_path = build_tensorrt_engine(
        arguments.onnx,
        arguments.output,
        workspace_mib=arguments.workspace_mib,
        overwrite=arguments.force,
    )

    duration = perf_counter() - started
    size_mib = engine_path.stat().st_size / 1024**2

    print(f"Built FP32 TensorRT engine: {engine_path}")
    print(f"Size: {size_mib:.2f} MiB")
    print(f"Build time: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
