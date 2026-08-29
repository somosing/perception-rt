"""Validated configuration for PerceptionRT training."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    """Complete reproducible training configuration."""

    seed: int
    dataset_root: Path
    output_directory: Path

    crop_height: int
    crop_width: int
    maximum_depth_m: float

    encoder_checkpoint: str
    decoder_channels: int
    number_of_classes: int
    semantic_class_weights_path: Path | None

    batch_size: int
    number_of_workers: int
    epochs: int
    maximum_steps: int | None

    learning_rate: float
    weight_decay: float
    warmup_steps: int
    minimum_learning_rate_ratio: float
    gradient_accumulation_steps: int
    gradient_clip_norm: float

    use_amp: bool
    amp_initial_scale: float

    semantic_loss_weight: float
    depth_loss_weight: float
    gradient_loss_weight: float

    log_every_steps: int
    validate_every_epochs: int

    @property
    def crop_size(self) -> tuple[int, int]:
        """Return crop height and width."""
        return self.crop_height, self.crop_width

    def validate(self) -> None:
        """Reject invalid training settings before expensive work starts."""
        positive_integers = {
            "crop_height": self.crop_height,
            "crop_width": self.crop_width,
            "decoder_channels": self.decoder_channels,
            "number_of_classes": self.number_of_classes,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "log_every_steps": self.log_every_steps,
            "validate_every_epochs": self.validate_every_epochs,
        }

        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.number_of_workers < 0:
            raise ValueError("number_of_workers cannot be negative")

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")

        if not 0.0 <= self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must be between 0 and 1")

        if self.maximum_steps is not None and self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive when provided")

        positive_floats = {
            "maximum_depth_m": self.maximum_depth_m,
            "learning_rate": self.learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "amp_initial_scale": self.amp_initial_scale,
        }

        for name, value in positive_floats.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        nonnegative_floats = {
            "weight_decay": self.weight_decay,
            "semantic_loss_weight": self.semantic_loss_weight,
            "depth_loss_weight": self.depth_loss_weight,
            "gradient_loss_weight": self.gradient_loss_weight,
        }

        for name, value in nonnegative_floats.items():
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")

        if not self.encoder_checkpoint:
            raise ValueError("encoder_checkpoint cannot be empty")


def load_training_config(path: Path) -> TrainingConfig:
    """Load and validate a YAML training configuration."""
    contents = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(contents, dict):
        raise ValueError("Training configuration must be a YAML mapping")

    expected_keys = {field.name for field in fields(TrainingConfig)}
    observed_keys = set(contents)
    missing_keys = expected_keys - observed_keys
    unknown_keys = observed_keys - expected_keys

    if missing_keys:
        raise ValueError(f"Missing configuration keys: {', '.join(sorted(missing_keys))}")

    if unknown_keys:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown_keys))}")

    converted: dict[str, Any] = dict(contents)
    converted["dataset_root"] = Path(converted["dataset_root"])
    converted["output_directory"] = Path(converted["output_directory"])
    weights_path = converted["semantic_class_weights_path"]

    if weights_path is not None:
        converted["semantic_class_weights_path"] = Path(weights_path)

    config = TrainingConfig(**converted)
    config.validate()
    return config
