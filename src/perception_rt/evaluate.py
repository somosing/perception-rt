"""Evaluate trained checkpoints on held-out data."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from perception_rt.data.torch_dataset import VirtualKitti2Dataset
from perception_rt.data.vkitti2 import (
    SemanticClass,
    discover_samples,
    load_color_table,
    split_samples_by_scene,
)
from perception_rt.models.multitask import PerceptionRTModel
from perception_rt.training.config import (
    TrainingConfig,
    load_training_config,
)
from perception_rt.training.engine import (
    evaluate as evaluate_loader,
)
from perception_rt.training.engine import seed_everything
from perception_rt.training.metrics import DenseMetricAccumulator
from perception_rt.training.weights import (
    load_semantic_class_weights,
)


def load_checkpoint_model(
    model: nn.Module,
    path: Path,
) -> dict[str, int | float]:
    """Load model weights and return checkpoint metadata."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])

    return {
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
    }


def named_per_class_iou(
    iou: Tensor,
    classes: tuple[SemanticClass, ...],
) -> dict[str, float | None]:
    """Associate class IoUs with semantic class names."""
    if iou.numel() != len(classes):
        raise ValueError("IoU count does not match semantic classes")

    result: dict[str, float | None] = {}

    for semantic_class, value in zip(
        classes,
        iou,
        strict=True,
    ):
        numeric_value = float(value)
        result[semantic_class.name] = numeric_value if math.isfinite(numeric_value) else None

    return result


def build_test_loader(
    config: TrainingConfig,
    *,
    sample_limit: int | None = None,
) -> tuple[
    DataLoader,
    tuple[SemanticClass, ...],
    tuple[str, ...],
]:
    """Build a deterministic loader for the held-out scene."""
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("Test sample limit must be positive")

    samples = discover_samples(
        config.dataset_root,
        camera_id=0,
    )
    test_samples = split_samples_by_scene(samples).test

    if sample_limit is not None:
        test_samples = test_samples[:sample_limit]

    if not test_samples:
        raise ValueError("No test samples were selected")

    first = test_samples[0]
    classes = load_color_table(config.dataset_root / first.scene / first.variation / "colors.txt")
    dataset = VirtualKitti2Dataset(
        test_samples,
        classes,
        crop_size=config.crop_size,
        maximum_depth_m=config.maximum_depth_m,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.number_of_workers,
        pin_memory=True,
        persistent_workers=(config.number_of_workers > 0),
    )
    scenes = tuple(sorted({sample.scene for sample in test_samples}))

    return loader, classes, scenes


def evaluate_checkpoints(
    config: TrainingConfig,
    checkpoint_paths: tuple[Path, ...],
    *,
    test_sample_limit: int | None = None,
    progress_every_batches: int = 100,
) -> dict[str, Any]:
    """Evaluate checkpoints on the untouched test split."""
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint evaluation")

    config.validate()
    seed_everything(config.seed)
    device = torch.device("cuda")

    loader, classes, scenes = build_test_loader(
        config,
        sample_limit=test_sample_limit,
    )
    class_weights = None

    if config.semantic_class_weights_path is not None:
        class_weights = load_semantic_class_weights(
            config.semantic_class_weights_path,
            classes,
            device=device,
        )

    model = PerceptionRTModel.from_pretrained(
        config.encoder_checkpoint,
        number_of_classes=config.number_of_classes,
        decoder_channels=config.decoder_channels,
    ).to(device)

    checkpoint_results: dict[str, Any] = {}

    for checkpoint_path in checkpoint_paths:
        result_name = checkpoint_path.name

        if result_name in checkpoint_results:
            raise ValueError(f"Duplicate checkpoint name: {result_name}")

        print(f"Evaluating checkpoint: {checkpoint_path}")
        metadata = load_checkpoint_model(
            model,
            checkpoint_path,
        )
        accumulator = DenseMetricAccumulator(
            config.number_of_classes,
            maximum_depth_m=config.maximum_depth_m,
        )
        aggregate = evaluate_loader(
            model,
            loader,
            device,
            config,
            class_weights=class_weights,
            metric_accumulator=accumulator,
            progress_every_batches=(progress_every_batches),
        )
        per_class_iou = named_per_class_iou(
            accumulator.semantic_iou(),
            classes,
        )

        checkpoint_results[result_name] = {
            "path": str(checkpoint_path),
            **metadata,
            **aggregate,
            "per_class_iou": per_class_iou,
        }

        print(
            f"{result_name}: "
            f"epoch={metadata['epoch']}, "
            f"mIoU={aggregate['mean_iou']:.4f}, "
            f"AbsRel={aggregate['depth_abs_rel']:.4f}, "
            f"RMSE={aggregate['depth_rmse_m']:.2f}m, "
            f"delta1={aggregate['depth_delta1']:.4f}, "
            f"uncertainty_r="
            f"{aggregate['uncertainty_error_pearson']:.4f}"
        )

    return {
        "dataset": "Virtual KITTI 2",
        "split": "test",
        "sample_count": len(loader.dataset),
        "scenes": list(scenes),
        "checkpoints": checkpoint_results,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PerceptionRT checkpoints on the held-out Virtual KITTI 2 test split."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_vkitti2.yaml"),
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/vkitti2_test.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate checkpoints and save a JSON report."""
    arguments = parse_arguments()
    config = load_training_config(arguments.config)
    report = evaluate_checkpoints(
        config,
        tuple(arguments.checkpoints),
        test_sample_limit=arguments.test_samples,
        progress_every_batches=(arguments.progress_every_batches),
    )

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

    print(f"Saved evaluation report: {arguments.output}")


if __name__ == "__main__":
    main()
