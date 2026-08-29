from pathlib import Path

from perception_rt.train import (
    accumulation_window_size,
    serialize_config,
)
from perception_rt.training.config import TrainingConfig


def test_accumulation_window_handles_short_final_window() -> None:
    assert accumulation_window_size(1, 9, 4) == 4
    assert accumulation_window_size(4, 9, 4) == 4
    assert accumulation_window_size(5, 9, 4) == 4
    assert accumulation_window_size(8, 9, 4) == 4
    assert accumulation_window_size(9, 9, 4) == 1


def test_serialize_config_converts_paths() -> None:
    config = TrainingConfig(
        seed=42,
        dataset_root=Path("dataset"),
        output_directory=Path("output"),
        crop_height=32,
        crop_width=64,
        maximum_depth_m=200.0,
        encoder_checkpoint="test",
        decoder_channels=16,
        number_of_classes=15,
        semantic_class_weights_path=None,
        batch_size=2,
        number_of_workers=0,
        epochs=1,
        maximum_steps=1,
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
        log_every_steps=1,
        validate_every_epochs=1,
    )

    serialized = serialize_config(config)

    assert serialized["dataset_root"] == "dataset"
    assert serialized["output_directory"] == "output"
