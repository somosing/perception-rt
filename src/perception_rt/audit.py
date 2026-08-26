"""Audit Virtual KITTI 2 samples and report dataset statistics."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from perception_rt.data.vkitti2 import (
    SemanticClass,
    discover_samples,
    load_color_table,
    load_sample,
)


def audit_dataset(
    root: Path,
    *,
    limit: int | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Decode samples and collect integrity and distribution statistics."""
    if limit is not None and limit <= 0:
        raise ValueError("Audit limit must be positive")

    samples = discover_samples(root, camera_id=0)
    total_discovered = len(samples)

    if limit is not None:
        samples = samples[:limit]

    class_table: tuple[SemanticClass, ...] | None = None
    class_counts: np.ndarray | None = None
    class_cache: dict[tuple[str, str], tuple[SemanticClass, ...]] = {}

    valid_fraction_sum = 0.0
    minimum_valid_fraction = 1.0
    maximum_valid_fraction = 0.0
    minimum_depth = float("inf")
    maximum_depth = float("-inf")
    shape_counts: dict[str, int] = {}

    for index, paths in enumerate(samples, start=1):
        cache_key = (paths.scene, paths.variation)

        if cache_key not in class_cache:
            class_cache[cache_key] = load_color_table(
                root / paths.scene / paths.variation / "colors.txt"
            )

        classes = class_cache[cache_key]

        if class_table is None:
            class_table = classes
            class_counts = np.zeros(len(classes), dtype=np.int64)
        elif classes != class_table:
            raise ValueError(
                f"Inconsistent semantic class table in {paths.scene}/{paths.variation}"
            )

        loaded = load_sample(paths, classes)
        height, width = loaded.semantic_mask.shape
        shape_key = f"{height}x{width}"
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1

        valid_values = loaded.depth.values_m[loaded.depth.valid_mask]

        if valid_values.size == 0:
            raise ValueError(f"Sample has no valid depth: {paths.depth_path}")

        valid_fraction = float(loaded.depth.valid_mask.mean())
        valid_fraction_sum += valid_fraction
        minimum_valid_fraction = min(
            minimum_valid_fraction,
            valid_fraction,
        )
        maximum_valid_fraction = max(
            maximum_valid_fraction,
            valid_fraction,
        )
        minimum_depth = min(minimum_depth, float(valid_values.min()))
        maximum_depth = max(maximum_depth, float(valid_values.max()))

        assert class_counts is not None
        class_counts += np.bincount(
            loaded.semantic_mask.reshape(-1),
            minlength=len(classes),
        )

        if progress_every > 0 and (index % progress_every == 0 or index == len(samples)):
            print(f"Audited {index}/{len(samples)} samples")

    if class_table is None or class_counts is None:
        raise ValueError("No samples were audited")

    return {
        "total_discovered_samples": total_discovered,
        "audited_samples": len(samples),
        "shape_counts": shape_counts,
        "mean_valid_depth_fraction": valid_fraction_sum / len(samples),
        "minimum_valid_depth_fraction": minimum_valid_fraction,
        "maximum_valid_depth_fraction": maximum_valid_fraction,
        "minimum_valid_depth_m": minimum_depth,
        "maximum_valid_depth_m": maximum_depth,
        "class_pixel_counts": {
            semantic_class.name: int(class_counts[semantic_class.class_id])
            for semantic_class in class_table
        },
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audit the extracted Virtual KITTI 2 dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/vkitti2/raw"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit only the first N samples.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vkitti2_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset audit and write its JSON report."""
    arguments = parse_arguments()
    summary = audit_dataset(
        arguments.root,
        limit=arguments.limit,
        progress_every=arguments.progress_every,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved audit report: {arguments.output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
