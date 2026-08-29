"""Reusable data, validation and checkpoint utilities for training."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from perception_rt.data.torch_dataset import VirtualKitti2Dataset
from perception_rt.data.vkitti2 import (
    SemanticClass,
    discover_samples,
    load_color_table,
    split_samples_by_scene,
)
from perception_rt.models.multitask import PerceptionRTModel
from perception_rt.training.config import TrainingConfig
from perception_rt.training.losses import multitask_loss

LOSS_NAMES = ("total", "semantic", "depth_nll", "depth_gradient")


@dataclass(frozen=True)
class LoaderBundle:
    """Training data loaders and semantic metadata."""

    train: DataLoader
    validation: DataLoader
    classes: tuple[SemanticClass, ...]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _limit_samples(
    samples: tuple,
    limit: int | None,
    name: str,
) -> tuple:
    if limit is None:
        return samples

    if limit <= 0:
        raise ValueError(f"{name} sample limit must be positive")

    return samples[:limit]


def build_loaders(
    config: TrainingConfig,
    *,
    train_sample_limit: int | None = None,
    validation_sample_limit: int | None = None,
    overfit_sample_count: int | None = None,
) -> LoaderBundle:
    """Build leakage-safe training and validation loaders."""
    samples = discover_samples(config.dataset_root, camera_id=0)
    splits = split_samples_by_scene(samples)

    if overfit_sample_count is not None:
        if train_sample_limit is not None or validation_sample_limit is not None:
            raise ValueError("Overfit mode cannot be combined with other sample limits")

        train_samples = _limit_samples(
            splits.train,
            overfit_sample_count,
            "Overfit",
        )
        validation_samples = train_samples
        use_training_augmentation = False
    else:
        train_samples = _limit_samples(
            splits.train,
            train_sample_limit,
            "Training",
        )
        validation_samples = _limit_samples(
            splits.validation,
            validation_sample_limit,
            "Validation",
        )
        use_training_augmentation = True

    first = train_samples[0]
    classes = load_color_table(config.dataset_root / first.scene / first.variation / "colors.txt")

    if len(classes) != config.number_of_classes:
        raise ValueError(
            f"Configuration expects {config.number_of_classes} classes, "
            f"but the dataset defines {len(classes)}"
        )

    train_dataset = VirtualKitti2Dataset(
        train_samples,
        classes,
        crop_size=config.crop_size,
        maximum_depth_m=config.maximum_depth_m,
        training=use_training_augmentation,
    )
    validation_dataset = VirtualKitti2Dataset(
        validation_samples,
        classes,
        crop_size=config.crop_size,
        maximum_depth_m=config.maximum_depth_m,
        training=False,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    common_arguments = {
        "batch_size": config.batch_size,
        "num_workers": config.number_of_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.number_of_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common_arguments,
    )

    return LoaderBundle(
        train=train_loader,
        validation=validation_loader,
        classes=classes,
    )


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move model inputs and targets to the training device."""
    names = ("image", "depth_m", "depth_valid", "semantic")

    return {name: batch[name].to(device, non_blocking=True) for name in names}


def evaluate(
    model: PerceptionRTModel,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    """Evaluate mean loss components without gradient tracking."""
    model.eval()
    sums = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0

    with torch.inference_mode():
        for batch in loader:
            tensors = move_batch_to_device(batch, device)
            batch_size = tensors["image"].shape[0]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.use_amp,
            ):
                output = model(tensors["image"])
                losses = multitask_loss(
                    output["semantic_logits"],
                    output["log_depth"],
                    output["depth_log_scale"],
                    tensors["semantic"],
                    tensors["depth_m"],
                    tensors["depth_valid"],
                    semantic_weight=config.semantic_loss_weight,
                    depth_weight=config.depth_loss_weight,
                    gradient_weight=config.gradient_loss_weight,
                )

            for name in LOSS_NAMES:
                sums[name] += float(losses[name].detach()) * batch_size

            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Validation loader produced no samples")

    return {name: value / sample_count for name, value in sums.items()}


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
) -> None:
    """Save all state required to continue training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
        },
        path,
    )


def restore_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    """Restore training state and return next epoch, step and best loss."""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        float(checkpoint["best_validation_loss"]),
    )


def append_history(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    split: str,
    losses: dict[str, float],
) -> None:
    """Append one epoch summary to a CSV history file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "epoch",
        "global_step",
        "split",
        *LOSS_NAMES,
    ]
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "epoch": epoch,
                "global_step": global_step,
                "split": split,
                **losses,
            }
        )
