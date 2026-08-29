"""Train the PerceptionRT multitask network."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml

from perception_rt.models.multitask import PerceptionRTModel
from perception_rt.training.config import (
    TrainingConfig,
    load_training_config,
)
from perception_rt.training.engine import (
    LOSS_NAMES,
    append_history,
    build_loaders,
    evaluate,
    move_batch_to_device,
    restore_checkpoint,
    save_checkpoint,
    seed_everything,
)
from perception_rt.training.losses import multitask_loss
from perception_rt.training.weights import (
    load_semantic_class_weights,
)


def accumulation_window_size(
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
) -> int:
    """Return the size of the current gradient-accumulation window."""
    window_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
    window_end = min(
        window_start + accumulation_steps - 1,
        total_batches,
    )
    return window_end - window_start + 1


def serialize_config(config: TrainingConfig) -> dict[str, object]:
    """Convert a training configuration into YAML-safe values."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def train(
    config: TrainingConfig,
    *,
    resume_path: Path | None = None,
    train_sample_limit: int | None = None,
    validation_sample_limit: int | None = None,
    overfit_sample_count: int | None = None,
) -> None:
    """Run training, validation, logging and checkpointing."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PerceptionRT training")

    seed_everything(config.seed)
    device = torch.device("cuda")

    loaders = build_loaders(
        config,
        train_sample_limit=train_sample_limit,
        validation_sample_limit=validation_sample_limit,
        overfit_sample_count=overfit_sample_count,
    )

    class_weights = None

    if config.semantic_class_weights_path is not None:
        class_weights = load_semantic_class_weights(
            config.semantic_class_weights_path,
            loaders.classes,
            device=device,
        )
        print(
            "Loaded semantic class weights:",
            config.semantic_class_weights_path,
        )

    model = PerceptionRTModel.from_pretrained(
        config.encoder_checkpoint,
        number_of_classes=config.number_of_classes,
        decoder_channels=config.decoder_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp,
        init_scale=config.amp_initial_scale,
        growth_interval=2000,
    )

    output_directory = config.output_directory
    history_path = output_directory / "history.csv"
    latest_checkpoint = output_directory / "latest.pt"
    best_checkpoint = output_directory / "best.pt"

    if resume_path is None and history_path.exists():
        raise FileExistsError(
            f"Training history already exists: {history_path}. "
            "Use --resume or choose another output directory."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            serialize_config(config),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    start_epoch = 1
    global_step = 0
    best_validation_loss = float("inf")

    if resume_path is not None:
        (
            start_epoch,
            global_step,
            best_validation_loss,
        ) = restore_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        print(f"Resumed {resume_path} at epoch {start_epoch}, global step {global_step}")

    print(f"Training samples: {len(loaders.train.dataset)}")
    print(f"Validation samples: {len(loaders.validation.dataset)}")
    print(
        "Effective batch size:",
        config.batch_size * config.gradient_accumulation_steps,
    )

    stop_training = False

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_sums = {name: 0.0 for name in LOSS_NAMES}
        epoch_sample_count = 0
        total_batches = len(loaders.train)

        for batch_index, batch in enumerate(
            loaders.train,
            start=1,
        ):
            tensors = move_batch_to_device(batch, device)
            batch_size = tensors["image"].shape[0]
            window_size = accumulation_window_size(
                batch_index,
                total_batches,
                config.gradient_accumulation_steps,
            )

            with torch.autocast(
                device_type="cuda",
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
                    class_weights=class_weights,
                )
                backward_loss = losses["total"] / window_size

            scaler.scale(backward_loss).backward()

            for name in LOSS_NAMES:
                epoch_sums[name] += float(losses[name].detach()) * batch_size

            epoch_sample_count += batch_size

            complete_window = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == total_batches
            )

            if not complete_window:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
            )

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)

            if scale_after < scale_before:
                print(f"Epoch {epoch}, batch {batch_index}: FP16 overflow; optimizer step skipped")
                continue

            global_step += 1

            if global_step % config.log_every_steps == 0:
                print(
                    f"Epoch {epoch}/{config.epochs} "
                    f"step {global_step}: "
                    f"total={float(losses['total'].detach()):.4f}, "
                    f"semantic={float(losses['semantic'].detach()):.4f}, "
                    f"depth={float(losses['depth_nll'].detach()):.4f}, "
                    f"gradient="
                    f"{float(losses['depth_gradient'].detach()):.4f}, "
                    f"scale={scale_after:.0f}"
                )

            if config.maximum_steps is not None and global_step >= config.maximum_steps:
                stop_training = True
                break

        if epoch_sample_count == 0:
            raise RuntimeError("Training loader produced no samples")

        train_losses = {name: value / epoch_sample_count for name, value in epoch_sums.items()}
        append_history(
            history_path,
            epoch=epoch,
            global_step=global_step,
            split="train",
            losses=train_losses,
        )

        should_validate = epoch % config.validate_every_epochs == 0 or stop_training

        if should_validate:
            validation_losses = evaluate(
                model,
                loaders.validation,
                device,
                config,
                class_weights=class_weights,
            )
            append_history(
                history_path,
                epoch=epoch,
                global_step=global_step,
                split="validation",
                losses=validation_losses,
            )

            print(
                f"Epoch {epoch} validation: "
                f"total={validation_losses['total']:.4f}, "
                f"semantic={validation_losses['semantic']:.4f}, "
                f"depth={validation_losses['depth_nll']:.4f}, "
                f"gradient="
                f"{validation_losses['depth_gradient']:.4f}"
            )

            is_best = validation_losses["total"] < best_validation_loss

            if is_best:
                best_validation_loss = validation_losses["total"]
        else:
            is_best = False

        save_checkpoint(
            latest_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
        )

        if is_best:
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
            )
            print(f"Saved new best checkpoint: {best_checkpoint}")

        if stop_training:
            break

    print(
        f"Training finished at global step {global_step}. "
        f"Best validation loss: {best_validation_loss:.4f}"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the PerceptionRT multitask network.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_vkitti2.yaml"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--overfit-samples", type=int)
    return parser.parse_args()


def main() -> None:
    """Load configuration and launch training."""
    arguments = parse_arguments()
    config = load_training_config(arguments.config)

    if arguments.maximum_steps is not None:
        config = replace(
            config,
            maximum_steps=arguments.maximum_steps,
        )

    if arguments.output_directory is not None:
        config = replace(
            config,
            output_directory=arguments.output_directory,
        )

    config.validate()

    train(
        config,
        resume_path=arguments.resume,
        train_sample_limit=arguments.train_samples,
        validation_sample_limit=arguments.validation_samples,
        overfit_sample_count=arguments.overfit_samples,
    )


if __name__ == "__main__":
    main()
