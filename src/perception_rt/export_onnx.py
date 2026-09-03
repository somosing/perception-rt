"""Export the PerceptionRT multitask model to ONNX."""

import argparse
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

from perception_rt.evaluate import load_checkpoint_model
from perception_rt.models.multitask import PerceptionRTModel
from perception_rt.training.config import (
    TrainingConfig,
    load_training_config,
)

ONNX_INPUT_NAME = "image"
ONNX_OUTPUT_NAMES = (
    "semantic_logits",
    "log_depth",
    "depth_log_scale",
)
DEFAULT_ONNX_OPSET = 18
DEFAULT_ONNX_PRECISION = "fp32"
ONNX_PRECISIONS = ("fp32", "fp16")
DEFAULT_CONFIG_PATH = Path("configs/train_vkitti2.yaml")
DEFAULT_CHECKPOINT_PATH = Path("outputs/training/vkitti2_multitask/best.pt")
DEFAULT_OUTPUT_PATH = Path("outputs/onnx/perception_rt_mit_b2_fp32.onnx")
DEFAULT_FP16_OUTPUT_PATH = Path("outputs/onnx/perception_rt_mit_b2_fp16.onnx")
ONNX_TORCH_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


class ExportablePerceptionRTModel(nn.Module):
    """Expose dictionary model outputs as an ordered ONNX tuple."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        image: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return outputs in the stable deployment contract order."""
        output = self.model(image)

        return (
            output["semantic_logits"],
            output["log_depth"],
            output["depth_log_scale"],
        )


def export_model_to_onnx(
    model: nn.Module,
    output_path: Path,
    *,
    input_size: tuple[int, int],
    opset_version: int = DEFAULT_ONNX_OPSET,
    precision: str = DEFAULT_ONNX_PRECISION,
) -> Path:
    """Export and validate a static batch-one ONNX graph."""
    height, width = input_size

    if height <= 0 or width <= 0:
        raise ValueError("ONNX input dimensions must be positive")

    if opset_version <= 0:
        raise ValueError("ONNX opset version must be positive")

    try:
        dtype = ONNX_TORCH_DTYPES[precision]
    except KeyError as error:
        raise ValueError(
            f"Unsupported ONNX precision {precision!r}; choose from {ONNX_PRECISIONS}"
        ) from error

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exportable_model = ExportablePerceptionRTModel(
        model.cpu().to(dtype=dtype).eval(),
    ).eval()
    example_image = torch.zeros(
        1,
        3,
        height,
        width,
        dtype=dtype,
    )

    torch.onnx.export(
        exportable_model,
        (example_image,),
        f=output_path,
        input_names=[ONNX_INPUT_NAME],
        output_names=list(ONNX_OUTPUT_NAMES),
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
    )

    onnx.checker.check_model(str(output_path))

    return output_path


def build_checkpoint_model(
    config: TrainingConfig,
    checkpoint_path: Path,
) -> tuple[PerceptionRTModel, dict[str, int | float]]:
    """Construct the trained model and load checkpoint weights."""
    config.validate()
    model = PerceptionRTModel.from_pretrained(
        config.encoder_checkpoint,
        number_of_classes=config.number_of_classes,
        decoder_channels=config.decoder_channels,
    )
    metadata = load_checkpoint_model(
        model,
        checkpoint_path,
    )

    return model.eval(), metadata


def parse_arguments() -> argparse.Namespace:
    """Parse ONNX export command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Export a trained PerceptionRT checkpoint to a static batch-one ONNX graph.")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--precision",
        choices=ONNX_PRECISIONS,
        default=DEFAULT_ONNX_PRECISION,
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=DEFAULT_ONNX_OPSET,
    )
    return parser.parse_args()


def main() -> None:
    """Load the checkpoint and export its ONNX graph."""
    arguments = parse_arguments()
    config = load_training_config(arguments.config)
    output_path = arguments.output
    if output_path is None:
        output_path = (
            DEFAULT_FP16_OUTPUT_PATH if arguments.precision == "fp16" else DEFAULT_OUTPUT_PATH
        )

    model, metadata = build_checkpoint_model(
        config,
        arguments.checkpoint,
    )
    output_path = export_model_to_onnx(
        model,
        output_path,
        input_size=config.crop_size,
        opset_version=arguments.opset_version,
        precision=arguments.precision,
    )
    size_mib = output_path.stat().st_size / (1024**2)

    print(
        f"Exported {arguments.precision.upper()} checkpoint "
        f"epoch {metadata['epoch']} to {output_path}"
    )
    print(
        f"ONNX contract: input=[1, 3, "
        f"{config.crop_height}, {config.crop_width}], "
        f"outputs={ONNX_OUTPUT_NAMES}"
    )
    print(f"Opset: {arguments.opset_version}, size: {size_mib:.2f} MiB")


if __name__ == "__main__":
    main()
