from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from perception_rt.training.config import (
    TrainingConfig,
    load_training_config,
)


def valid_config() -> TrainingConfig:
    return TrainingConfig(
        seed=42,
        dataset_root=Path("datasets/vkitti2/raw"),
        output_directory=Path("outputs/training/test"),
        crop_height=320,
        crop_width=640,
        maximum_depth_m=200.0,
        photometric_jitter_strength=0.2,
        encoder_checkpoint="nvidia/mit-b2",
        decoder_channels=128,
        number_of_classes=15,
        semantic_class_weights_path=None,
        batch_size=2,
        number_of_workers=2,
        epochs=30,
        maximum_steps=None,
        learning_rate=2e-4,
        weight_decay=1e-2,
        warmup_steps=500,
        minimum_learning_rate_ratio=0.05,
        gradient_accumulation_steps=4,
        gradient_clip_norm=1.0,
        use_amp=True,
        amp_initial_scale=1024.0,
        semantic_loss_weight=1.0,
        depth_loss_weight=1.0,
        gradient_loss_weight=0.5,
        log_every_steps=10,
        validate_every_epochs=1,
    )


def write_config(path: Path, config: dict[str, object]) -> None:
    serializable = {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()
    }
    path.write_text(
        yaml.safe_dump(serializable, sort_keys=False),
        encoding="utf-8",
    )


def test_load_training_config_converts_paths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training.yaml"
    write_config(path, asdict(valid_config()))

    loaded = load_training_config(path)

    assert loaded.dataset_root == Path("datasets/vkitti2/raw")
    assert loaded.output_directory == Path("outputs/training/test")
    assert loaded.crop_size == (320, 640)
    assert loaded.maximum_steps is None


def test_load_training_config_rejects_unknown_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training.yaml"
    contents = asdict(valid_config())
    contents["mystery_setting"] = 1
    write_config(path, contents)

    with pytest.raises(ValueError, match="Unknown configuration keys"):
        load_training_config(path)


def test_training_config_rejects_invalid_batch_size() -> None:
    contents = asdict(valid_config())
    contents["batch_size"] = 0
    config = TrainingConfig(**contents)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        config.validate()
