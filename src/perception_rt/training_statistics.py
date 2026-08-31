"""Calculate training-split semantic and depth statistics."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from perception_rt.data.vkitti2 import (
    discover_samples,
    load_color_table,
    load_depth,
    load_semantic_mask,
    split_samples_by_scene,
)

DEPTH_THRESHOLDS_M = (50.0, 80.0, 100.0, 150.0, 200.0, 300.0, 500.0)


def compute_class_weights(
    class_counts: np.ndarray,
    *,
    frequency_offset: float = 1.02,
) -> np.ndarray:
    """Compute normalized ENet-style logarithmic class weights."""
    if class_counts.ndim != 1:
        raise ValueError("Class counts must be one-dimensional")

    if np.any(class_counts < 0):
        raise ValueError("Class counts cannot be negative")

    total = int(class_counts.sum())

    if total == 0:
        raise ValueError("At least one class pixel is required")

    observed = class_counts > 0
    frequencies = class_counts.astype(np.float64) / total
    weights = np.zeros_like(frequencies)
    weights[observed] = 1.0 / np.log(frequency_offset + frequencies[observed])
    weights[observed] /= weights[observed].mean()

    return weights


def analyze_training_split(
    root: Path,
    *,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Scan only training scenes and calculate target statistics."""
    samples = discover_samples(root, camera_id=0)
    train_samples = split_samples_by_scene(samples).train

    first = train_samples[0]
    classes = load_color_table(root / first.scene / first.variation / "colors.txt")

    class_counts = np.zeros(len(classes), dtype=np.int64)
    depth_threshold_counts = {threshold: 0 for threshold in DEPTH_THRESHOLDS_M}

    valid_depth_count = 0
    minimum_depth_m = float("inf")
    maximum_depth_m = float("-inf")

    for index, sample in enumerate(train_samples, start=1):
        semantic = load_semantic_mask(
            sample.semantic_path,
            classes,
        )
        depth = load_depth(sample.depth_path)
        valid_depth = depth.values_m[depth.valid_mask]

        class_counts += np.bincount(
            semantic.reshape(-1),
            minlength=len(classes),
        )

        valid_depth_count += int(valid_depth.size)
        minimum_depth_m = min(
            minimum_depth_m,
            float(valid_depth.min()),
        )
        maximum_depth_m = max(
            maximum_depth_m,
            float(valid_depth.max()),
        )

        for threshold in DEPTH_THRESHOLDS_M:
            depth_threshold_counts[threshold] += int(np.count_nonzero(valid_depth <= threshold))

        if progress_every > 0 and (index % progress_every == 0 or index == len(train_samples)):
            print(f"Analyzed {index}/{len(train_samples)} training samples")

    weights = compute_class_weights(class_counts)

    return {
        "training_samples": len(train_samples),
        "valid_depth_pixels": valid_depth_count,
        "minimum_valid_depth_m": minimum_depth_m,
        "maximum_valid_depth_m": maximum_depth_m,
        "depth_coverage": {
            f"at_or_below_{threshold:g}m": {
                "pixels": depth_threshold_counts[threshold],
                "fraction_of_valid": (depth_threshold_counts[threshold] / valid_depth_count),
            }
            for threshold in DEPTH_THRESHOLDS_M
        },
        "class_pixel_counts": {
            semantic_class.name: int(class_counts[semantic_class.class_id])
            for semantic_class in classes
        },
        "class_weights": {
            semantic_class.name: float(weights[semantic_class.class_id])
            for semantic_class in classes
        },
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze Virtual KITTI 2 training targets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/vkitti2/raw"),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vkitti2_training_statistics.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Analyze training targets and save the JSON report."""
    arguments = parse_arguments()
    report = analyze_training_split(
        arguments.root,
        progress_every=arguments.progress_every,
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved statistics: {arguments.output}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
