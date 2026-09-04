"""Benchmark FP32, FP16 and selective INT8 inference backends."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from perception_rt.build_tensorrt import (
    DEFAULT_ENGINE_PATH,
    DEFAULT_FP16_ENGINE_PATH,
    DEFAULT_INT8_ENGINE_PATH,
    EXPECTED_OUTPUT_SHAPES,
)
from perception_rt.evaluate import build_test_loader
from perception_rt.export_onnx import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_PATH,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAMES,
    build_checkpoint_model,
)
from perception_rt.tensorrt_runtime import TensorRTRunner
from perception_rt.training.config import load_training_config
from perception_rt.training.engine import seed_everything
from perception_rt.validate_onnx import create_onnx_session

DEFAULT_WARMUP_ITERATIONS = 30
DEFAULT_MEASURED_ITERATIONS = 100
DEFAULT_BENCHMARK_SAMPLE_INDEX = 0
DEFAULT_BENCHMARK_REPORT_PATH = Path("outputs/tensorrt/benchmark_int8.json")


def validate_benchmark_parameters(
    warmup_iterations: int,
    measured_iterations: int,
) -> None:
    """Validate benchmark iteration counts."""
    if warmup_iterations < 0:
        raise ValueError("Warmup iterations must be nonnegative")
    if measured_iterations <= 0:
        raise ValueError("Measured iterations must be positive")


def calculate_latency_statistics(
    latencies_ms: list[float],
) -> dict[str, float]:
    """Summarize positive per-inference latency measurements."""
    if not latencies_ms:
        raise ValueError("At least one latency measurement is required")

    values = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Latency measurements must be finite")
    if np.any(values <= 0.0):
        raise ValueError("Latency measurements must be positive")

    mean_ms = float(values.mean())

    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "throughput_fps": 1000.0 / mean_ms,
    }


def calculate_speedup(
    reference_mean_ms: float,
    optimized_mean_ms: float,
) -> float:
    """Return latency speedup from positive mean measurements."""
    values = np.asarray(
        [reference_mean_ms, optimized_mean_ms],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Mean latency measurements must be finite and positive")

    return reference_mean_ms / optimized_mean_ms


def benchmark_backend(
    function: Callable[[], None],
    *,
    synchronize: Callable[[], None],
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, float]:
    """Measure synchronous wall-clock GPU inference latency."""
    validate_benchmark_parameters(
        warmup_iterations,
        measured_iterations,
    )

    for _ in range(warmup_iterations):
        function()

    synchronize()
    latencies_ms = []

    for _ in range(measured_iterations):
        started = perf_counter()
        function()
        synchronize()
        latencies_ms.append((perf_counter() - started) * 1000.0)

    return calculate_latency_statistics(latencies_ms)


def bind_onnx_cuda_buffers(
    session: Any,
    image: torch.Tensor,
    device: torch.device,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Bind PyTorch CUDA buffers directly to ONNX Runtime."""
    binding = session.io_binding()
    device_id = device.index or 0

    binding.bind_input(
        name=ONNX_INPUT_NAME,
        device_type="cuda",
        device_id=device_id,
        element_type=np.float32,
        shape=tuple(image.shape),
        buffer_ptr=image.data_ptr(),
    )

    outputs = {}
    for name in ONNX_OUTPUT_NAMES:
        tensor = torch.empty(
            EXPECTED_OUTPUT_SHAPES[name],
            dtype=torch.float32,
            device=device,
        )
        outputs[name] = tensor
        binding.bind_output(
            name=name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    return binding, outputs


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark device-resident synchronous FP32, TensorRT FP16 "
            "and selective TensorRT INT8 inference."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=DEFAULT_ENGINE_PATH,
    )
    parser.add_argument(
        "--fp16-engine",
        type=Path,
        default=DEFAULT_FP16_ENGINE_PATH,
    )
    parser.add_argument(
        "--int8-engine",
        type=Path,
        default=DEFAULT_INT8_ENGINE_PATH,
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=DEFAULT_BENCHMARK_SAMPLE_INDEX,
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
    )
    parser.add_argument(
        "--measured-iterations",
        type=int,
        default=DEFAULT_MEASURED_ITERATIONS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BENCHMARK_REPORT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    """Run the five-backend FP32, FP16 and selective INT8 benchmark."""
    arguments = parse_arguments()
    validate_benchmark_parameters(
        arguments.warmup_iterations,
        arguments.measured_iterations,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference benchmarking")

    import onnxruntime as ort

    config = load_training_config(arguments.config)
    seed_everything(config.seed)

    loader, _, _ = build_test_loader(config)
    dataset = loader.dataset

    if arguments.sample_index < 0 or arguments.sample_index >= len(dataset):
        raise IndexError(f"Sample index {arguments.sample_index} outside [0, {len(dataset) - 1}]")

    sample = dataset[arguments.sample_index]
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    image = sample["image"].unsqueeze(0).contiguous().to(device)
    image_fp16 = image.to(dtype=torch.float16).contiguous()
    torch.cuda.synchronize(device)

    model, metadata = build_checkpoint_model(
        config,
        arguments.checkpoint,
    )
    model = model.to(device).eval()

    session = create_onnx_session(arguments.onnx_model)
    io_binding, _onnx_output_buffers = bind_onnx_cuda_buffers(
        session,
        image,
        device,
    )
    fp32_runner = TensorRTRunner(
        arguments.engine,
        device=device,
    )
    fp16_runner = TensorRTRunner(
        arguments.fp16_engine,
        device=device,
        logger=fp32_runner.logger,
    )
    int8_runner = TensorRTRunner(
        arguments.int8_engine,
        device=device,
        logger=fp32_runner.logger,
    )
    if fp32_runner.dtype != torch.float32:
        raise ValueError("FP32 engine path does not contain an FP32 engine")
    if fp16_runner.dtype != torch.float16:
        raise ValueError("FP16 engine path does not contain an FP16 engine")
    if int8_runner.dtype != torch.float32:
        raise ValueError("INT8 engine path must expose FP32 input and outputs")

    def run_pytorch() -> None:
        with torch.inference_mode():
            model(image)

    def run_onnx_runtime() -> None:
        session.run_with_iobinding(io_binding)

    def run_tensorrt_fp32() -> None:
        fp32_runner.infer(image)

    def run_tensorrt_fp16() -> None:
        fp16_runner.infer(image_fp16)

    def run_tensorrt_int8() -> None:
        int8_runner.infer(image)

    def synchronize() -> None:
        torch.cuda.synchronize(device)

    backend_functions = (
        ("pytorch_fp32", run_pytorch),
        (
            "onnx_runtime_cuda_fp32",
            run_onnx_runtime,
        ),
        ("tensorrt_fp32", run_tensorrt_fp32),
        ("tensorrt_fp16", run_tensorrt_fp16),
        ("tensorrt_int8", run_tensorrt_int8),
    )

    results = {
        name: benchmark_backend(
            function,
            synchronize=synchronize,
            warmup_iterations=(arguments.warmup_iterations),
            measured_iterations=(arguments.measured_iterations),
        )
        for name, function in backend_functions
    }

    tensorrt_fp32 = results["tensorrt_fp32"]
    tensorrt_fp16 = results["tensorrt_fp16"]
    tensorrt_int8 = results["tensorrt_int8"]

    tensorrt_fp32["speedup_vs_pytorch_fp32"] = calculate_speedup(
        results["pytorch_fp32"]["mean_ms"],
        tensorrt_fp32["mean_ms"],
    )
    tensorrt_fp32["speedup_vs_onnx_runtime_cuda_fp32"] = calculate_speedup(
        results["onnx_runtime_cuda_fp32"]["mean_ms"],
        tensorrt_fp32["mean_ms"],
    )

    tensorrt_fp16["speedup_vs_pytorch_fp32"] = calculate_speedup(
        results["pytorch_fp32"]["mean_ms"],
        tensorrt_fp16["mean_ms"],
    )
    tensorrt_fp16["speedup_vs_onnx_runtime_cuda_fp32"] = calculate_speedup(
        results["onnx_runtime_cuda_fp32"]["mean_ms"],
        tensorrt_fp16["mean_ms"],
    )
    tensorrt_fp16["speedup_vs_tensorrt_fp32"] = calculate_speedup(
        tensorrt_fp32["mean_ms"],
        tensorrt_fp16["mean_ms"],
    )

    for reference_name in (
        "pytorch_fp32",
        "onnx_runtime_cuda_fp32",
        "tensorrt_fp32",
        "tensorrt_fp16",
    ):
        tensorrt_int8[f"speedup_vs_{reference_name}"] = calculate_speedup(
            results[reference_name]["mean_ms"],
            tensorrt_int8["mean_ms"],
        )

    report = {
        "sample": {
            "index": arguments.sample_index,
            "scene": sample["scene"],
            "variation": sample["variation"],
            "frame": int(sample["frame"]),
        },
        "input_shape": list(image.shape),
        "precisions": ["FP32", "FP16", "selective INT8"],
        "checkpoint_epoch": metadata["epoch"],
        "warmup_iterations": (arguments.warmup_iterations),
        "measured_iterations": (arguments.measured_iterations),
        "methodology": {
            "timing": (
                "synchronous wall-clock latency with CUDA synchronization after every inference"
            ),
            "device_resident_io": True,
            "model_loading_excluded": True,
            "preprocessing_excluded": True,
            "fp16_input_conversion_excluded": True,
            "backend_order": [name for name, _ in backend_functions],
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "software": {
            "pytorch": torch.__version__,
            "onnx_runtime": ort.__version__,
            "tensorrt": fp16_runner.trt.__version__,
        },
        "results": results,
    }

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    arguments.output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for name, result in results.items():
        print(
            f"{name}: "
            f"mean={result['mean_ms']:.3f} ms, "
            f"p95={result['p95_ms']:.3f} ms, "
            f"throughput={result['throughput_fps']:.2f} FPS"
        )

    print(
        "TensorRT FP16 speedup: "
        f"{tensorrt_fp16['speedup_vs_tensorrt_fp32']:.3f}x "
        "vs TensorRT FP32, "
        f"{tensorrt_fp16['speedup_vs_pytorch_fp32']:.3f}x "
        "vs PyTorch FP32, "
        f"{tensorrt_fp16['speedup_vs_onnx_runtime_cuda_fp32']:.3f}x "
        "vs ONNX Runtime CUDA FP32"
    )
    print(
        "TensorRT selective INT8 speedup: "
        f"{tensorrt_int8['speedup_vs_tensorrt_fp32']:.3f}x "
        "vs TensorRT FP32, "
        f"{tensorrt_int8['speedup_vs_tensorrt_fp16']:.3f}x "
        "vs TensorRT FP16"
    )
    print(f"Saved benchmark report: {arguments.output}")


if __name__ == "__main__":
    main()
