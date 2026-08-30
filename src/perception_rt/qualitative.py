"""Visualize trained predictions on held-out samples."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from perception_rt.data.torch_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TrainingSample,
    VirtualKitti2Dataset,
)
from perception_rt.data.vkitti2 import (
    SamplePaths,
    SemanticClass,
    discover_samples,
    load_color_table,
    split_samples_by_scene,
)
from perception_rt.evaluate import load_checkpoint_model
from perception_rt.models.multitask import PerceptionRTModel
from perception_rt.training.config import load_training_config
from perception_rt.training.engine import seed_everything
from perception_rt.visualize import colorize_semantic_mask


def denormalize_image(image: Tensor) -> np.ndarray:
    """Convert a normalized CHW image into displayable HWC RGB."""
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    standard_deviation = image.new_tensor(IMAGENET_STD).view(3, 1, 1)

    return (image * standard_deviation + mean).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()


def validate_indices(
    indices: tuple[int, ...],
    sample_count: int,
) -> tuple[int, ...]:
    """Validate requested dataset indices."""
    if not indices:
        raise ValueError("At least one index is required")

    for index in indices:
        if index < 0 or index >= sample_count:
            raise IndexError(f"Sample index {index} outside [0, {sample_count - 1}]")

    return indices


def _masked(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ma.MaskedArray:
    return np.ma.masked_where(~valid, values)


def save_prediction_figure(
    *,
    sample: TrainingSample,
    paths: SamplePaths,
    classes: tuple[SemanticClass, ...],
    semantic_prediction: np.ndarray,
    depth_prediction_m: np.ndarray,
    uncertainty_scale: np.ndarray,
    output_path: Path,
) -> None:
    """Save an eight-panel qualitative prediction figure."""
    rgb = denormalize_image(sample["image"])
    semantic_target = sample["semantic"].cpu().numpy()
    depth_target_m = sample["depth_m"][0].cpu().numpy()
    depth_valid = sample["depth_valid"][0].cpu().numpy()

    semantic_target_rgb = colorize_semantic_mask(
        semantic_target,
        classes,
    )
    semantic_prediction_rgb = colorize_semantic_mask(
        semantic_prediction,
        classes,
    )
    semantic_disagreement = semantic_prediction != semantic_target

    absolute_error_m = np.abs(depth_prediction_m - depth_target_m)
    valid_target = depth_target_m[depth_valid]
    valid_error = absolute_error_m[depth_valid]
    valid_uncertainty = uncertainty_scale[depth_valid]

    if valid_target.size == 0:
        raise ValueError("Sample contains no valid depth")

    depth_limit = max(
        float(np.percentile(valid_target, 99.0)),
        1e-3,
    )
    error_limit = max(
        float(np.percentile(valid_error, 95.0)),
        1e-3,
    )
    uncertainty_limit = max(
        float(np.percentile(valid_uncertainty, 99.0)),
        1e-3,
    )

    depth_colormap = plt.get_cmap("turbo").copy()
    depth_colormap.set_bad("black")
    error_colormap = plt.get_cmap("magma").copy()
    error_colormap.set_bad("black")
    uncertainty_colormap = plt.get_cmap("viridis").copy()
    uncertainty_colormap.set_bad("black")

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(20, 9),
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB input")

    axes[0, 1].imshow(semantic_target_rgb)
    axes[0, 1].set_title("Semantic ground truth")

    axes[0, 2].imshow(semantic_prediction_rgb)
    axes[0, 2].set_title("Semantic prediction")

    axes[0, 3].imshow(
        semantic_disagreement,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0, 3].set_title("Semantic disagreement")

    target_image = axes[1, 0].imshow(
        _masked(depth_target_m, depth_valid),
        cmap=depth_colormap,
        vmin=0.0,
        vmax=depth_limit,
    )
    axes[1, 0].set_title("Metric depth ground truth")
    figure.colorbar(
        target_image,
        ax=axes[1, 0],
        label="Depth [m]",
        fraction=0.046,
        pad=0.04,
    )

    prediction_image = axes[1, 1].imshow(
        _masked(depth_prediction_m, depth_valid),
        cmap=depth_colormap,
        vmin=0.0,
        vmax=depth_limit,
    )
    axes[1, 1].set_title("Metric depth prediction")
    figure.colorbar(
        prediction_image,
        ax=axes[1, 1],
        label="Depth [m]",
        fraction=0.046,
        pad=0.04,
    )

    error_image = axes[1, 2].imshow(
        _masked(absolute_error_m, depth_valid),
        cmap=error_colormap,
        vmin=0.0,
        vmax=error_limit,
    )
    axes[1, 2].set_title(f"Absolute depth error (95th={error_limit:.1f} m)")
    figure.colorbar(
        error_image,
        ax=axes[1, 2],
        label="Absolute error [m]",
        fraction=0.046,
        pad=0.04,
    )

    uncertainty_image = axes[1, 3].imshow(
        _masked(uncertainty_scale, depth_valid),
        cmap=uncertainty_colormap,
        vmin=0.0,
        vmax=uncertainty_limit,
    )
    axes[1, 3].set_title("Predicted uncertainty scale (log-depth)")
    figure.colorbar(
        uncertainty_image,
        ax=axes[1, 3],
        label="Predicted scale",
        fraction=0.046,
        pad=0.04,
    )

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(
        f"{paths.scene} / {paths.variation} / frame {paths.frame:05d} / Camera_0",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Visualize best-checkpoint predictions on held-out Virtual KITTI 2 samples.")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_vkitti2.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 847, 1695, 2542, 3389],
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/qualitative/vkitti2_test"),
    )
    return parser.parse_args()


def main() -> None:
    """Generate qualitative predictions."""
    arguments = parse_arguments()
    config = load_training_config(arguments.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prediction")

    seed_everything(config.seed)
    device = torch.device("cuda")
    samples = split_samples_by_scene(
        discover_samples(
            config.dataset_root,
            camera_id=0,
        )
    ).test
    indices = validate_indices(
        tuple(arguments.indices),
        len(samples),
    )

    first = samples[0]
    classes = load_color_table(config.dataset_root / first.scene / first.variation / "colors.txt")
    dataset = VirtualKitti2Dataset(
        samples,
        classes,
        crop_size=config.crop_size,
        maximum_depth_m=config.maximum_depth_m,
        training=False,
    )
    model = PerceptionRTModel.from_pretrained(
        config.encoder_checkpoint,
        number_of_classes=config.number_of_classes,
        decoder_channels=config.decoder_channels,
    ).to(device)
    metadata = load_checkpoint_model(
        model,
        arguments.checkpoint,
    )
    model.eval()

    print(f"Loaded checkpoint epoch {metadata['epoch']}: {arguments.checkpoint}")

    for index in indices:
        sample = dataset[index]
        paths = samples[index]
        image = sample["image"].unsqueeze(0).to(device)

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=config.use_amp,
            ):
                output = model(image)

        semantic_prediction = output["semantic_logits"].argmax(dim=1)[0].cpu().numpy()
        depth_prediction_m = (
            output["log_depth"][0, 0]
            .float()
            .exp()
            .clamp(
                min=1e-3,
                max=config.maximum_depth_m,
            )
            .cpu()
            .numpy()
        )
        uncertainty_scale = (
            output["depth_log_scale"][0, 0].float().clamp(-6.0, 6.0).exp().cpu().numpy()
        )

        filename = f"{paths.scene.lower()}_{paths.variation}_frame_{paths.frame:05d}.png"
        output_path = arguments.output_directory / filename
        save_prediction_figure(
            sample=sample,
            paths=paths,
            classes=classes,
            semantic_prediction=semantic_prediction,
            depth_prediction_m=depth_prediction_m,
            uncertainty_scale=uncertainty_scale,
            output_path=output_path,
        )
        print(f"Saved prediction figure: {output_path}")


if __name__ == "__main__":
    main()
