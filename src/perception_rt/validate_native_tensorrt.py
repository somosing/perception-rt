"""Validate native C++ TensorRT inference against the Python runtime."""

import argparse
import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from perception_rt.build_tensorrt import (
    DEFAULT_FP16_ENGINE_PATH,
    EXPECTED_OUTPUT_SHAPES,
)
from perception_rt.evaluate import build_test_loader
from perception_rt.export_onnx import DEFAULT_CONFIG_PATH, ONNX_OUTPUT_NAMES
from perception_rt.tensorrt_runtime import TensorRTRunner
from perception_rt.training.config import load_training_config
from perception_rt.training.engine import seed_everything
from perception_rt.validate_onnx import DEFAULT_SAMPLE_INDICES

DEFAULT_NATIVE_EXECUTABLE = Path("build/native/perception_rt_native")
DEFAULT_NATIVE_PARITY_REPORT_PATH = Path("outputs/native_cpp/parity.json")
DEFAULT_NATIVE_ABSOLUTE_TOLERANCE = 1e-3
DEFAULT_NATIVE_RELATIVE_TOLERANCE = 1e-3


def compare_native_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    absolute_tolerance: float = DEFAULT_NATIVE_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_NATIVE_RELATIVE_TOLERANCE,
) -> dict[str, float | bool]:
    """Calculate exact-match coverage and numerical parity."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("Tolerances must be nonnegative")

    reference_fp32 = reference.astype(np.float32, copy=False)
    candidate_fp32 = candidate.astype(np.float32, copy=False)
    difference = np.abs(reference_fp32 - candidate_fp32)

    return {
        "all_within_tolerance": bool(
            np.allclose(
                reference_fp32,
                candidate_fp32,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        ),
        "exact_match_fraction": float(np.mean(reference == candidate)),
        "maximum_absolute_error": float(np.max(difference)),
        "mean_absolute_error": float(np.mean(difference, dtype=np.float64)),
    }


def semantic_argmax_agreement(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    """Return the fraction of pixels with identical semantic predictions."""
    if reference.shape != candidate.shape:
        raise ValueError("Semantic output shapes do not match")
    if reference.ndim != 4:
        raise ValueError("Semantic logits must be a four-dimensional tensor")
    return float(np.mean(np.argmax(reference, axis=1) == np.argmax(candidate, axis=1)))


def summarize_native_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate native parity metrics across held-out samples."""
    if not results:
        raise ValueError("At least one native parity result is required")

    output_summary: dict[str, dict[str, float | bool]] = {}
    for name in ONNX_OUTPUT_NAMES:
        metrics = [result["outputs"][name] for result in results]
        output_summary[name] = {
            "all_within_tolerance": all(bool(metric["all_within_tolerance"]) for metric in metrics),
            "maximum_absolute_error": max(
                float(metric["maximum_absolute_error"]) for metric in metrics
            ),
            "maximum_mean_absolute_error": max(
                float(metric["mean_absolute_error"]) for metric in metrics
            ),
            "minimum_exact_match_fraction": min(
                float(metric["exact_match_fraction"]) for metric in metrics
            ),
        }

    return {
        "sample_count": len(results),
        "minimum_semantic_argmax_agreement": min(
            float(result["semantic_argmax_agreement"]) for result in results
        ),
        "outputs": output_summary,
        "passed": all(bool(metric["all_within_tolerance"]) for metric in output_summary.values()),
    }


def validate_sample_indices(indices: Sequence[int], dataset_size: int) -> tuple[int, ...]:
    """Validate and normalize deterministic dataset indices."""
    normalized = tuple(indices)
    if not normalized:
        raise ValueError("At least one sample index is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Sample indices must be unique")
    if any(index < 0 or index >= dataset_size for index in normalized):
        raise ValueError(f"Sample indices must be within [0, {dataset_size})")
    return normalized


def run_native_process(
    executable: Path,
    engine: Path,
    input_path: Path,
    output_directory: Path,
) -> None:
    """Execute one native inference process."""
    completed = subprocess.run(
        [
            str(executable),
            "--engine",
            str(engine),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_directory),
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Native TensorRT process failed:\n{details}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate native C++ TensorRT FP16 inference.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--engine", type=Path, default=DEFAULT_FP16_ENGINE_PATH)
    parser.add_argument(
        "--executable",
        type=Path,
        default=DEFAULT_NATIVE_EXECUTABLE,
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_INDICES),
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=DEFAULT_NATIVE_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=DEFAULT_NATIVE_RELATIVE_TOLERANCE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NATIVE_PARITY_REPORT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    """Run deterministic held-out native parity validation."""
    arguments = parse_arguments()
    if not arguments.executable.is_file():
        raise FileNotFoundError(f"Native executable does not exist: {arguments.executable}")
    if not arguments.engine.is_file():
        raise FileNotFoundError(f"TensorRT engine does not exist: {arguments.engine}")

    config = load_training_config(arguments.config)
    seed_everything(config.seed)
    loader, _, _ = build_test_loader(config)
    dataset = loader.dataset
    indices = validate_sample_indices(arguments.sample_indices, len(dataset))

    device = torch.device("cuda:0")
    runner = TensorRTRunner(arguments.engine, device=device)
    results: list[dict[str, Any]] = []

    print("Comparing native C++ with Python TensorRT FP16 inference")
    with tempfile.TemporaryDirectory(prefix="perception_rt_native_") as temporary:
        temporary_root = Path(temporary)
        for index in indices:
            sample = dataset[index]
            host_image = sample["image"].unsqueeze(0).to(dtype=torch.float16).contiguous()
            input_path = temporary_root / f"input_{index}.fp16.bin"
            output_directory = temporary_root / f"outputs_{index}"
            host_image.numpy().tofile(input_path)

            reference_tensors = runner.infer(host_image.to(device))
            run_native_process(
                arguments.executable,
                arguments.engine,
                input_path,
                output_directory,
            )

            output_metrics: dict[str, dict[str, float | bool]] = {}
            native_outputs: dict[str, np.ndarray] = {}
            reference_outputs: dict[str, np.ndarray] = {}
            for name in ONNX_OUTPUT_NAMES:
                shape = EXPECTED_OUTPUT_SHAPES[name]
                reference = reference_tensors[name].cpu().numpy().copy()
                candidate = np.fromfile(
                    output_directory / f"{name}.fp16.bin",
                    dtype=np.float16,
                ).reshape(shape)
                reference_outputs[name] = reference
                native_outputs[name] = candidate
                output_metrics[name] = compare_native_arrays(
                    reference,
                    candidate,
                    absolute_tolerance=arguments.absolute_tolerance,
                    relative_tolerance=arguments.relative_tolerance,
                )

            agreement = semantic_argmax_agreement(
                reference_outputs["semantic_logits"],
                native_outputs["semantic_logits"],
            )
            passed = all(bool(metric["all_within_tolerance"]) for metric in output_metrics.values())
            result = {
                "frame": int(sample["frame"]),
                "index": index,
                "outputs": output_metrics,
                "passed": passed,
                "scene": str(sample["scene"]),
                "semantic_argmax_agreement": agreement,
                "variation": str(sample["variation"]),
            }
            results.append(result)
            print(
                f"index={index}, {result['scene']}/{result['variation']}/"
                f"frame={result['frame']}, exact_semantics={agreement:.8f}, "
                f"passed={passed}"
            )

    summary = summarize_native_results(results)
    report = {
        "absolute_tolerance": arguments.absolute_tolerance,
        "engine": str(arguments.engine),
        "executable": str(arguments.executable),
        "hardware": {"gpu": torch.cuda.get_device_name(device)},
        "precision": "FP16",
        "relative_tolerance": arguments.relative_tolerance,
        "results": results,
        "software": {
            "pytorch": torch.__version__,
            "tensorrt": runner.trt.__version__,
        },
        "summary": summary,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved native parity report: {arguments.output}")
    print("Native TensorRT parity validation: " + ("PASSED" if summary["passed"] else "FAILED"))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
