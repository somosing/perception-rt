"""Tests for inference benchmark calculations."""

import pytest

from perception_rt.benchmark_inference import (
    calculate_latency_statistics,
    validate_benchmark_parameters,
)


def test_calculate_latency_statistics() -> None:
    result = calculate_latency_statistics([1.0, 2.0, 3.0, 4.0])

    assert result["mean_ms"] == 2.5
    assert result["median_ms"] == 2.5
    assert result["minimum_ms"] == 1.0
    assert result["maximum_ms"] == 4.0
    assert result["p90_ms"] == pytest.approx(3.7)
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["p99_ms"] == pytest.approx(3.97)
    assert result["throughput_fps"] == 400.0


@pytest.mark.parametrize(
    "latencies",
    [
        [],
        [0.0, 1.0],
        [float("inf")],
    ],
)
def test_calculate_latency_statistics_rejects_invalid_values(
    latencies: list[float],
) -> None:
    with pytest.raises(ValueError):
        calculate_latency_statistics(latencies)


def test_validate_benchmark_parameters_accepts_zero_warmup() -> None:
    validate_benchmark_parameters(0, 1)


@pytest.mark.parametrize(
    ("warmup", "measured"),
    [
        (-1, 1),
        (0, 0),
    ],
)
def test_validate_benchmark_parameters_rejects_invalid_counts(
    warmup: int,
    measured: int,
) -> None:
    with pytest.raises(ValueError):
        validate_benchmark_parameters(
            warmup,
            measured,
        )
