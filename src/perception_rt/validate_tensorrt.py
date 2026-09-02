"""Validate FP32 TensorRT parity against the PyTorch checkpoint."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from perception_rt.build_tensorrt import DEFAULT_ENGINE_PATH
from perception_rt.data.torch_dataset import TrainingSample
from perception_rt.evaluate import build_test_loader
from perception_rt.export_onnx import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    ONNX_OUTPUT_NAMES,
    build_checkpoint_model,
)
from perception_rt.tensorrt_runtime import TensorRTRunner
from perception_rt.training.config import load_training_config
from perception_rt.training.engine import seed_everything
from perception_rt.validate_onnx import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_MINIMUM_SEMANTIC_AGREEMENT,
    DEFAULT_RELATIVE_TOLERANCE,
    DEFAULT_SAMPLE_INDICES,
    compare_output_sets,
    summarize_parity_results,
)

DEFAULT_UNCERTAINTY_RELATIVE_TOLERANCE = math.expm1(DEFAULT_ABSOLUTE_TOLERANCE)
DEFAULT_TENSORRT_PARITY_REPORT_PATH = Path("outputs/tensorrt/parity.json")


def validate_sample_indices(
    indices: tuple[int, ...],
    dataset_size: int,
) -> None:
    """Validate deterministic parity sample indices."""
    if dataset_size <= 0:
        raise ValueError("Dataset must not be empty")
    if not indices:
        raise ValueError("At least one sample index is required")

    for index in indices:
        if index < 0 or index >= dataset_size:
            raise IndexError(f"Sample index {index} outside [0, {dataset_size - 1}]")


def run_tensorrt_sample_parity(
    *,
    model: torch.nn.Module,
    runner: Any,
    sample: TrainingSample,
    device: torch.device,
    maximum_depth_m: float,
    absolute_tolerance: float,
    relative_tolerance: float,
    uncertainty_relative_tolerance: float,
    minimum_semantic_agreement: float,
) -> dict[str, object]:
    """Compare PyTorch and TensorRT on one transformed sample."""
    image = sample["image"].unsqueeze(0).contiguous().to(device)

    with torch.inference_mode():
        torch_outputs = model(image)
        tensorrt_outputs = runner.infer(image)

    reference_outputs = {
        name: torch_outputs[name].float().cpu().numpy() for name in ONNX_OUTPUT_NAMES
    }
    candidate_outputs = {
        name: (tensorrt_outputs[name].float().cpu().numpy().copy()) for name in ONNX_OUTPUT_NAMES
    }

    parity = compare_output_sets(
        reference_outputs,
        candidate_outputs,
        maximum_depth_m=maximum_depth_m,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        uncertainty_relative_tolerance=(uncertainty_relative_tolerance),
        minimum_semantic_agreement=(minimum_semantic_agreement),
    )

    return {
        "scene": sample["scene"],
        "variation": sample["variation"],
        "frame": int(sample["frame"]),
        **parity,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse TensorRT parity-validation arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare FP32 PyTorch and native TensorRT outputs on held-out Virtual KITTI 2 samples."
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
        "--engine",
        type=Path,
        default=DEFAULT_ENGINE_PATH,
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_INDICES,
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=DEFAULT_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=DEFAULT_RELATIVE_TOLERANCE,
    )
    parser.add_argument(
        "--uncertainty-relative-tolerance",
        type=float,
        default=DEFAULT_UNCERTAINTY_RELATIVE_TOLERANCE,
    )
    parser.add_argument(
        "--minimum-semantic-agreement",
        type=float,
        default=DEFAULT_MINIMUM_SEMANTIC_AGREEMENT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TENSORRT_PARITY_REPORT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    """Run held-out PyTorch–TensorRT parity validation."""
    arguments = parse_arguments()

    for name, value in (
        ("Absolute", arguments.absolute_tolerance),
        ("Relative", arguments.relative_tolerance),
        (
            "Uncertainty relative",
            arguments.uncertainty_relative_tolerance,
        ),
    ):
        if value < 0.0:
            raise ValueError(f"{name} tolerance must be nonnegative")

    if not 0.0 <= arguments.minimum_semantic_agreement <= 1.0:
        raise ValueError("Semantic agreement threshold must be within [0, 1]")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT parity validation")

    config = load_training_config(arguments.config)
    seed_everything(config.seed)

    loader, _, scenes = build_test_loader(config)
    dataset = loader.dataset
    indices = tuple(arguments.indices)
    validate_sample_indices(indices, len(dataset))

    device = torch.device("cuda")
    model, metadata = build_checkpoint_model(
        config,
        arguments.checkpoint,
    )
    model = model.to(device).eval()
    runner = TensorRTRunner(
        arguments.engine,
        device=device,
    )

    print(f"Comparing checkpoint epoch {metadata['epoch']} with {arguments.engine}")
    print(f"TensorRT {runner.trt.__version__}, GPU: {torch.cuda.get_device_name(device)}")

    results: list[dict[str, Any]] = []

    for index in indices:
        sample = dataset[index]
        result = run_tensorrt_sample_parity(
            model=model,
            runner=runner,
            sample=sample,
            device=device,
            maximum_depth_m=config.maximum_depth_m,
            absolute_tolerance=arguments.absolute_tolerance,
            relative_tolerance=arguments.relative_tolerance,
            uncertainty_relative_tolerance=(arguments.uncertainty_relative_tolerance),
            minimum_semantic_agreement=(arguments.minimum_semantic_agreement),
        )
        result["index"] = index
        results.append(result)

        print(
            f"index={index}, "
            f"{result['scene']}/{result['variation']}/"
            f"frame={result['frame']}, "
            f"semantic_agreement="
            f"{float(result['semantic_argmax_agreement']):.8f}, "
            f"passed={result['passed']}"
        )

    summary = summarize_parity_results(results)
    report = {
        "dataset": "Virtual KITTI 2",
        "split": "held-out test",
        "scenes": list(scenes),
        "indices": list(indices),
        "checkpoint": {
            "path": str(arguments.checkpoint),
            **metadata,
        },
        "engine": {
            "path": str(arguments.engine),
            "precision": "FP32",
            "size_mib": (arguments.engine.stat().st_size / 1024**2),
            "tensorrt_version": runner.trt.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "tolerances": {
            "absolute": arguments.absolute_tolerance,
            "relative": arguments.relative_tolerance,
            "uncertainty_relative": (arguments.uncertainty_relative_tolerance),
            "minimum_semantic_agreement": (arguments.minimum_semantic_agreement),
        },
        "summary": summary,
        "samples": results,
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

    print(f"Saved parity report: {arguments.output}")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not summary["passed"]:
        raise SystemExit("TensorRT parity validation: FAILED")

    print("TensorRT parity validation: PASSED")


if __name__ == "__main__":
    main()
