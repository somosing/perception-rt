"""Run and record the native C++ TensorRT FP16 benchmark."""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from perception_rt.build_tensorrt import DEFAULT_FP16_ENGINE_PATH
from perception_rt.evaluate import build_test_loader
from perception_rt.export_onnx import DEFAULT_CONFIG_PATH
from perception_rt.training.config import load_training_config
from perception_rt.training.engine import seed_everything
from perception_rt.validate_native_tensorrt import DEFAULT_NATIVE_EXECUTABLE

DEFAULT_NATIVE_WARMUP_ITERATIONS = 30
DEFAULT_NATIVE_MEASURED_ITERATIONS = 100
DEFAULT_NATIVE_BENCHMARK_SAMPLE_INDEX = 0
DEFAULT_NATIVE_BENCHMARK_REPORT_PATH = Path("outputs/native_cpp/benchmark.json")


def validate_benchmark_parameters(
    *,
    warmup_iterations: int,
    measured_iterations: int,
    sample_index: int,
) -> None:
    """Validate native benchmark parameters."""
    if warmup_iterations < 0:
        raise ValueError("Warmup iterations must be nonnegative")
    if measured_iterations <= 0:
        raise ValueError("Measured iterations must be positive")
    if sample_index < 0:
        raise ValueError("Sample index must be nonnegative")


def build_native_command(
    *,
    executable: Path,
    engine: Path,
    input_path: Path,
    output_directory: Path,
    warmup_iterations: int,
    measured_iterations: int,
) -> list[str]:
    """Construct the native benchmark command."""
    return [
        str(executable),
        "--engine",
        str(engine),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_directory),
        "--warmup",
        str(warmup_iterations),
        "--iterations",
        str(measured_iterations),
    ]


def parse_native_benchmark_output(output: str) -> dict[str, str | int | float]:
    """Parse stable metrics emitted by the native executable."""
    patterns = {
        "tensorrt_runtime": r"^TensorRT runtime: (\d+)$",
        "gpu": r"^GPU: (.+)$",
        "precision": r"^Precision: (.+)$",
        "mean_ms": r"^Mean latency: ([0-9.]+) ms$",
        "p95_ms": r"^P95 latency: ([0-9.]+) ms$",
        "throughput_fps": r"^Throughput: ([0-9.]+) FPS$",
    }
    values: dict[str, str | int | float] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, output, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"Native benchmark output is missing {name!r}")
        value = match.group(1)
        if name == "tensorrt_runtime":
            values[name] = int(value)
        elif name in {"mean_ms", "p95_ms", "throughput_fps"}:
            values[name] = float(value)
        else:
            values[name] = value
    return values


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark native C++ TensorRT FP16 inference.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--engine", type=Path, default=DEFAULT_FP16_ENGINE_PATH)
    parser.add_argument(
        "--executable",
        type=Path,
        default=DEFAULT_NATIVE_EXECUTABLE,
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=DEFAULT_NATIVE_BENCHMARK_SAMPLE_INDEX,
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=DEFAULT_NATIVE_WARMUP_ITERATIONS,
    )
    parser.add_argument(
        "--measured-iterations",
        type=int,
        default=DEFAULT_NATIVE_MEASURED_ITERATIONS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NATIVE_BENCHMARK_REPORT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    """Benchmark native inference and save a JSON report."""
    arguments = parse_arguments()
    validate_benchmark_parameters(
        warmup_iterations=arguments.warmup_iterations,
        measured_iterations=arguments.measured_iterations,
        sample_index=arguments.sample_index,
    )
    if not arguments.executable.is_file():
        raise FileNotFoundError(f"Native executable does not exist: {arguments.executable}")
    if not arguments.engine.is_file():
        raise FileNotFoundError(f"TensorRT engine does not exist: {arguments.engine}")

    config = load_training_config(arguments.config)
    seed_everything(config.seed)
    loader, _, _ = build_test_loader(config)
    dataset = loader.dataset
    if arguments.sample_index >= len(dataset):
        raise ValueError(f"Sample index must be within [0, {len(dataset)})")
    sample = dataset[arguments.sample_index]
    image = sample["image"].unsqueeze(0).numpy().astype(np.float16, copy=False)

    with tempfile.TemporaryDirectory(prefix="perception_rt_benchmark_") as temporary:
        temporary_root = Path(temporary)
        input_path = temporary_root / "image.fp16.bin"
        output_directory = temporary_root / "outputs"
        image.tofile(input_path)
        command = build_native_command(
            executable=arguments.executable,
            engine=arguments.engine,
            input_path=input_path,
            output_directory=output_directory,
            warmup_iterations=arguments.warmup_iterations,
            measured_iterations=arguments.measured_iterations,
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Native TensorRT benchmark failed:\n{details}")

    print(completed.stdout, end="")
    metrics = parse_native_benchmark_output(completed.stdout)
    if metrics["precision"] != "FP16":
        raise RuntimeError(f"Expected native FP16 benchmark, found {metrics['precision']}")

    report = {
        "device_resident_io": True,
        "engine": str(arguments.engine),
        "hardware": {"gpu": metrics["gpu"]},
        "input_shape": list(image.shape),
        "measured_iterations": arguments.measured_iterations,
        "precision": metrics["precision"],
        "results": {
            "native_cpp_tensorrt_fp16": {
                "mean_ms": metrics["mean_ms"],
                "p95_ms": metrics["p95_ms"],
                "throughput_fps": metrics["throughput_fps"],
            }
        },
        "sample": {
            "frame": int(sample["frame"]),
            "index": arguments.sample_index,
            "scene": str(sample["scene"]),
            "variation": str(sample["variation"]),
        },
        "software": {
            "native_standard": "C++17",
            "tensorrt_runtime": metrics["tensorrt_runtime"],
        },
        "synchronous_inference": True,
        "warmup_iterations": arguments.warmup_iterations,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved native benchmark report: {arguments.output}")


if __name__ == "__main__":
    main()
