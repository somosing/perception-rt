"""Visualize aligned Virtual KITTI 2 modalities."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perception_rt.data.vkitti2 import (
    LoadedSample,
    SemanticClass,
    discover_samples,
    load_color_table,
    load_sample,
)


def colorize_semantic_mask(
    class_ids: np.ndarray,
    classes: tuple[SemanticClass, ...],
) -> np.ndarray:
    """Convert integer semantic IDs back into an RGB visualization."""
    colorized = np.zeros((*class_ids.shape, 3), dtype=np.uint8)

    for semantic_class in classes:
        colorized[class_ids == semantic_class.class_id] = semantic_class.color

    return colorized


def save_sample_visualization(
    sample: LoadedSample,
    classes: tuple[SemanticClass, ...],
    output_path: Path,
) -> None:
    """Save a four-panel visualization of an aligned sample."""
    valid_depth = sample.depth.values_m[sample.depth.valid_mask]

    if valid_depth.size == 0:
        raise ValueError("Cannot visualize a sample with no valid depth")

    depth_limit = float(np.percentile(valid_depth, 99.0))
    depth_display = np.ma.masked_where(
        ~sample.depth.valid_mask,
        sample.depth.values_m,
    )
    semantic_rgb = colorize_semantic_mask(
        sample.semantic_mask,
        classes,
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 8))

    axes[0, 0].imshow(sample.rgb)
    axes[0, 0].set_title("RGB input")

    depth_image = axes[0, 1].imshow(
        depth_display,
        cmap="turbo",
        vmin=0.0,
        vmax=depth_limit,
    )
    axes[0, 1].set_title(f"Metric depth (0–{depth_limit:.1f} m)")
    figure.colorbar(
        depth_image,
        ax=axes[0, 1],
        label="Depth [m]",
        fraction=0.046,
        pad=0.04,
    )

    axes[1, 0].imshow(sample.depth.valid_mask, cmap="gray")
    axes[1, 0].set_title("Valid depth mask")

    axes[1, 1].imshow(semantic_rgb)
    axes[1, 1].set_title("Semantic ground truth")

    for axis in axes.flat:
        axis.axis("off")

    identity = (
        f"{sample.paths.scene} / {sample.paths.variation} / "
        f"frame {sample.paths.frame:05d} / Camera_{sample.paths.camera_id}"
    )
    figure.suptitle(identity, fontsize=14)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize one aligned Virtual KITTI 2 sample.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/vkitti2/raw"),
        help="Extracted Virtual KITTI 2 root.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index in the sorted left-camera sample list.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vkitti2_sample.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    """Load and visualize one sample."""
    arguments = parse_arguments()
    samples = discover_samples(arguments.root, camera_id=0)

    if arguments.index < 0 or arguments.index >= len(samples):
        raise IndexError(f"Sample index {arguments.index} outside [0, {len(samples) - 1}]")

    paths = samples[arguments.index]
    classes = load_color_table(arguments.root / paths.scene / paths.variation / "colors.txt")
    sample = load_sample(paths, classes)

    save_sample_visualization(sample, classes, arguments.output)
    print(f"Saved visualization: {arguments.output}")


if __name__ == "__main__":
    main()
