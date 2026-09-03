"""Validate numerical parity between PyTorch and ONNX Runtime."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from perception_rt.data.torch_dataset import TrainingSample
from perception_rt.evaluate import build_test_loader
from perception_rt.export_onnx import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_PATH,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAMES,
    build_checkpoint_model,
)
from perception_rt.training.config import load_training_config
from perception_rt.training.engine import seed_everything

DEFAULT_ABSOLUTE_TOLERANCE = 2e-2
DEFAULT_RELATIVE_TOLERANCE = 1e-2
DEFAULT_MINIMUM_SEMANTIC_AGREEMENT = 0.999
DEFAULT_MINIMUM_WITHIN_TOLERANCE_FRACTION = 1.0
DEFAULT_SAMPLE_INDICES = (0, 847, 1695, 2542, 3389)
DEFAULT_PARITY_REPORT_PATH = Path("outputs/onnx/parity.json")


def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> dict[str, float | bool]:
    """Measure numerical agreement between identically shaped arrays."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )

    if reference.size == 0:
        raise ValueError("Arrays must not be empty")

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("Tolerances must be nonnegative")

    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Arrays must contain only finite values")

    reference_float = reference.astype(np.float64, copy=False)
    candidate_float = candidate.astype(np.float64, copy=False)
    absolute_error = np.abs(reference_float - candidate_float)
    allowed_error = absolute_tolerance + relative_tolerance * np.abs(reference_float)
    within_tolerance = absolute_error <= allowed_error

    return {
        "maximum_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "within_tolerance_fraction": float(within_tolerance.mean()),
        "all_within_tolerance": bool(within_tolerance.all()),
    }


def semantic_argmax_agreement(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
) -> float:
    """Return the fraction of pixels with matching semantic predictions."""
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("Semantic-logit shapes do not match")

    if reference_logits.ndim != 4 or reference_logits.shape[1] == 0:
        raise ValueError("Semantic logits must have nonempty NCHW shape")

    reference_classes = reference_logits.argmax(axis=1)
    candidate_classes = candidate_logits.argmax(axis=1)

    return float((reference_classes == candidate_classes).mean())


def compare_output_sets(
    reference_outputs: Mapping[str, np.ndarray],
    candidate_outputs: Mapping[str, np.ndarray],
    *,
    maximum_depth_m: float,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    uncertainty_relative_tolerance: float | None = None,
    minimum_semantic_agreement: float = DEFAULT_MINIMUM_SEMANTIC_AGREEMENT,
    minimum_within_tolerance_fraction: float = (DEFAULT_MINIMUM_WITHIN_TOLERANCE_FRACTION),
) -> dict[str, object]:
    """Compare every raw and post-processed deployment output."""
    output_names = (
        "semantic_logits",
        "log_depth",
        "depth_log_scale",
    )
    expected_names = set(output_names)

    if set(reference_outputs) != expected_names:
        raise ValueError("Reference outputs do not match the deployment contract")

    if set(candidate_outputs) != expected_names:
        raise ValueError("Candidate outputs do not match the deployment contract")

    if maximum_depth_m <= 0.0:
        raise ValueError("Maximum depth must be positive")

    if uncertainty_relative_tolerance is None:
        uncertainty_relative_tolerance = relative_tolerance

    if uncertainty_relative_tolerance < 0.0:
        raise ValueError("Uncertainty relative tolerance must be nonnegative")

    if not 0.0 <= minimum_semantic_agreement <= 1.0:
        raise ValueError("Semantic agreement threshold must be within [0, 1]")

    if not 0.0 <= minimum_within_tolerance_fraction <= 1.0:
        raise ValueError("Minimum tolerance fraction must be within [0, 1]")

    raw_parity = {
        name: compare_arrays(
            reference_outputs[name],
            candidate_outputs[name],
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for name in output_names
    }

    minimum_log_depth = np.log(1e-3)
    maximum_log_depth = np.log(maximum_depth_m)

    reference_depth = np.exp(
        np.clip(
            reference_outputs["log_depth"],
            minimum_log_depth,
            maximum_log_depth,
        )
    )
    candidate_depth = np.exp(
        np.clip(
            candidate_outputs["log_depth"],
            minimum_log_depth,
            maximum_log_depth,
        )
    )
    reference_uncertainty = np.exp(np.clip(reference_outputs["depth_log_scale"], -6.0, 6.0))
    candidate_uncertainty = np.exp(np.clip(candidate_outputs["depth_log_scale"], -6.0, 6.0))

    postprocessed_parity = {
        "depth_m": compare_arrays(
            reference_depth,
            candidate_depth,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ),
        "uncertainty_scale": compare_arrays(
            reference_uncertainty,
            candidate_uncertainty,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=uncertainty_relative_tolerance,
        ),
    }

    parity_results = (
        *raw_parity.values(),
        *postprocessed_parity.values(),
    )

    semantic_agreement = semantic_argmax_agreement(
        reference_outputs["semantic_logits"],
        candidate_outputs["semantic_logits"],
    )
    for result in parity_results:
        result["meets_minimum_fraction"] = (
            float(result["within_tolerance_fraction"]) >= minimum_within_tolerance_fraction
        )

    outputs_within_tolerance = all(
        bool(result["meets_minimum_fraction"]) for result in parity_results
    )

    return {
        "raw_outputs": raw_parity,
        "postprocessed_outputs": postprocessed_parity,
        "semantic_argmax_agreement": semantic_agreement,
        "outputs_within_tolerance": outputs_within_tolerance,
        "passed": (outputs_within_tolerance and semantic_agreement >= minimum_semantic_agreement),
    }


def create_onnx_session(onnx_path: Path) -> Any:
    """Create an ONNX Runtime session that must use CUDA."""
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    import onnxruntime as ort

    available_providers = ort.get_available_providers()

    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    active_providers = session.get_providers()

    if not active_providers or active_providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ONNX Runtime did not activate CUDA: {active_providers}")

    return session


def run_sample_parity(
    *,
    model: torch.nn.Module,
    session: Any,
    sample: TrainingSample,
    device: torch.device,
    maximum_depth_m: float,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    minimum_semantic_agreement: float = DEFAULT_MINIMUM_SEMANTIC_AGREEMENT,
) -> dict[str, object]:
    """Compare PyTorch and ONNX Runtime for one transformed sample."""
    image = sample["image"].unsqueeze(0).contiguous()

    with torch.inference_mode():
        torch_outputs = model(image.to(device))

    reference_outputs = {
        name: torch_outputs[name].float().cpu().numpy() for name in ONNX_OUTPUT_NAMES
    }
    candidate_values = session.run(
        list(ONNX_OUTPUT_NAMES),
        {ONNX_INPUT_NAME: image.numpy()},
    )
    candidate_outputs = dict(
        zip(
            ONNX_OUTPUT_NAMES,
            candidate_values,
            strict=True,
        )
    )
    parity = compare_output_sets(
        reference_outputs,
        candidate_outputs,
        maximum_depth_m=maximum_depth_m,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        minimum_semantic_agreement=minimum_semantic_agreement,
    )

    return {
        "scene": sample["scene"],
        "variation": sample["variation"],
        "frame": int(sample["frame"]),
        **parity,
    }


def _summarize_metric_group(
    results: list[dict[str, Any]],
    group_name: str,
    metric_names: tuple[str, ...],
) -> dict[str, dict[str, float | bool]]:
    """Summarize one metric group across samples."""
    summary: dict[str, dict[str, float | bool]] = {}

    for metric_name in metric_names:
        metrics = [result[group_name][metric_name] for result in results]
        summary[metric_name] = {
            "maximum_absolute_error": max(
                float(metric["maximum_absolute_error"]) for metric in metrics
            ),
            "maximum_mean_absolute_error": max(
                float(metric["mean_absolute_error"]) for metric in metrics
            ),
            "minimum_within_tolerance_fraction": min(
                float(metric["within_tolerance_fraction"]) for metric in metrics
            ),
            "all_within_tolerance": all(bool(metric["all_within_tolerance"]) for metric in metrics),
            "meets_minimum_fraction": all(
                bool(metric["meets_minimum_fraction"]) for metric in metrics
            ),
        }

    return summary


def summarize_parity_results(
    results: list[dict[str, Any]],
) -> dict[str, object]:
    """Summarize parity across all selected samples."""
    if not results:
        raise ValueError("At least one parity result is required")

    return {
        "sample_count": len(results),
        "raw_outputs": _summarize_metric_group(
            results,
            "raw_outputs",
            ONNX_OUTPUT_NAMES,
        ),
        "postprocessed_outputs": _summarize_metric_group(
            results,
            "postprocessed_outputs",
            ("depth_m", "uncertainty_scale"),
        ),
        "minimum_semantic_argmax_agreement": min(
            float(result["semantic_argmax_agreement"]) for result in results
        ),
        "passed": all(bool(result["passed"]) for result in results),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse ONNX parity-validation arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare FP32 PyTorch and ONNX Runtime CUDA outputs "
            "on held-out Virtual KITTI 2 samples."
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
        "--minimum-semantic-agreement",
        type=float,
        default=DEFAULT_MINIMUM_SEMANTIC_AGREEMENT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PARITY_REPORT_PATH,
    )

    return parser.parse_args()


def main() -> None:
    """Run held-out PyTorch–ONNX Runtime CUDA parity validation."""
    arguments = parse_arguments()

    if arguments.absolute_tolerance < 0.0:
        raise ValueError("Absolute tolerance must be nonnegative")

    if arguments.relative_tolerance < 0.0:
        raise ValueError("Relative tolerance must be nonnegative")

    if not 0.0 <= arguments.minimum_semantic_agreement <= 1.0:
        raise ValueError("Semantic agreement threshold must be within [0, 1]")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for parity validation")

    config = load_training_config(arguments.config)
    seed_everything(config.seed)

    loader, _, scenes = build_test_loader(config)
    dataset = loader.dataset
    indices = tuple(arguments.indices)

    for index in indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"Sample index {index} outside [0, {len(dataset) - 1}]")

    device = torch.device("cuda")
    model, metadata = build_checkpoint_model(
        config,
        arguments.checkpoint,
    )
    model = model.to(device).eval()
    session = create_onnx_session(arguments.onnx_model)

    print(f"Comparing checkpoint epoch {metadata['epoch']} with {arguments.onnx_model}")
    print(f"ONNX Runtime providers: {session.get_providers()}")

    results: list[dict[str, Any]] = []

    for index in indices:
        sample = dataset[index]
        result = run_sample_parity(
            model=model,
            session=session,
            sample=sample,
            device=device,
            maximum_depth_m=config.maximum_depth_m,
            absolute_tolerance=arguments.absolute_tolerance,
            relative_tolerance=arguments.relative_tolerance,
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
        "onnx_model": str(arguments.onnx_model),
        "providers": session.get_providers(),
        "absolute_tolerance": arguments.absolute_tolerance,
        "relative_tolerance": arguments.relative_tolerance,
        "minimum_semantic_agreement": arguments.minimum_semantic_agreement,
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

    if not summary["passed"]:
        print("ONNX parity validation: FAILED")
        raise SystemExit(1)

    print("ONNX parity validation: PASSED")


if __name__ == "__main__":
    main()
