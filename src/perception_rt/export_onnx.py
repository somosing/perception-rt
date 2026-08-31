"""Export the PerceptionRT multitask model to ONNX."""

from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

ONNX_INPUT_NAME = "image"
ONNX_OUTPUT_NAMES = (
    "semantic_logits",
    "log_depth",
    "depth_log_scale",
)
DEFAULT_ONNX_OPSET = 18


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
) -> Path:
    """Export and validate a static batch-one FP32 ONNX graph."""
    height, width = input_size

    if height <= 0 or width <= 0:
        raise ValueError("ONNX input dimensions must be positive")

    if opset_version <= 0:
        raise ValueError("ONNX opset version must be positive")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exportable_model = ExportablePerceptionRTModel(
        model.cpu().eval(),
    ).eval()
    example_image = torch.zeros(
        1,
        3,
        height,
        width,
        dtype=torch.float32,
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
