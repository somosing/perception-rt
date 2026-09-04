"""Tests for the native C++ TensorRT benchmark wrapper."""

from pathlib import Path

import pytest

from perception_rt.benchmark_native_tensorrt import (
    build_native_command,
    parse_native_benchmark_output,
    validate_benchmark_parameters,
)


def test_parse_native_benchmark_output() -> None:
    output = """TensorRT runtime: 110201
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
Precision: FP16
Mean latency: 6.842 ms
P95 latency: 6.959 ms
Throughput: 146.164 FPS
"""

    assert parse_native_benchmark_output(output) == {
        "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
        "mean_ms": 6.842,
        "p95_ms": 6.959,
        "precision": "FP16",
        "tensorrt_runtime": 110201,
        "throughput_fps": 146.164,
    }


def test_parse_native_benchmark_output_rejects_missing_metric() -> None:
    output = """TensorRT runtime: 110201
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
Precision: FP16
P95 latency: 6.959 ms
Throughput: 146.164 FPS
"""

    with pytest.raises(ValueError, match="mean_ms"):
        parse_native_benchmark_output(output)


def test_build_native_command() -> None:
    command = build_native_command(
        executable=Path("build/native/perception_rt_native"),
        engine=Path("model.engine"),
        input_path=Path("image.bin"),
        output_directory=Path("outputs"),
        warmup_iterations=3,
        measured_iterations=10,
    )

    assert command == [
        "build/native/perception_rt_native",
        "--engine",
        "model.engine",
        "--input",
        "image.bin",
        "--output-dir",
        "outputs",
        "--warmup",
        "3",
        "--iterations",
        "10",
    ]


@pytest.mark.parametrize(
    ("warmup", "measured", "sample", "message"),
    [
        (-1, 1, 0, "Warmup"),
        (0, 0, 0, "Measured"),
        (0, 1, -1, "Sample"),
    ],
)
def test_validate_benchmark_parameters_rejects_invalid_values(
    warmup: int,
    measured: int,
    sample: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_benchmark_parameters(
            warmup_iterations=warmup,
            measured_iterations=measured,
            sample_index=sample,
        )
