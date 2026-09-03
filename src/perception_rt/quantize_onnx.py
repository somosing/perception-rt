"""Create the selective INT8 ONNX deployment model."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from perception_rt.export_onnx import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_PATH,
    ONNX_INPUT_NAME,
)
from perception_rt.training.config import load_training_config
from perception_rt.training.engine import build_loaders, seed_everything

DEFAULT_PREPROCESSED_ONNX_PATH = Path("outputs/onnx/perception_rt_mit_b2_fp32_preprocessed.onnx")
DEFAULT_INT8_ONNX_PATH = Path("outputs/onnx/perception_rt_mit_b2_int8.onnx")
DEFAULT_CALIBRATION_SAMPLE_COUNT = 20
EXPECTED_SR_CONV_COUNT = 13
SR_CONV_PATTERN = "attention.self.sr"


def select_calibration_indices(
    dataset_size: int,
    sample_count: int = DEFAULT_CALIBRATION_SAMPLE_COUNT,
) -> tuple[int, ...]:
    """Select deterministic, evenly spaced calibration samples."""
    if dataset_size <= 0:
        raise ValueError("Calibration dataset must not be empty")
    if not 0 < sample_count <= dataset_size:
        raise ValueError("Calibration sample count must be within the dataset")

    indices = tuple(
        int(index)
        for index in np.linspace(
            0,
            dataset_size - 1,
            sample_count,
            dtype=np.int64,
        )
    )
    if len(set(indices)) != sample_count:
        raise RuntimeError("Calibration selection contains duplicate indices")
    return indices


def select_sr_conv_node_names(model: onnx.ModelProto) -> tuple[str, ...]:
    """Select MiT spatial-reduction convolutions for INT8 execution."""
    return tuple(
        node.name
        for node in model.graph.node
        if node.op_type == "Conv"
        and SR_CONV_PATTERN in " ".join((node.name, *node.input, *node.output))
    )


class Scene06CalibrationReader(CalibrationDataReader):
    """Provide deterministic Scene06 images to ONNX Runtime calibration."""

    def __init__(self, dataset: Any, indices: tuple[int, ...]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.rewind()

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next batch-one FP32 image."""
        try:
            index = next(self._iterator)
        except StopIteration:
            return None

        image = self.dataset[index]["image"].unsqueeze(0).numpy()
        return {ONNX_INPUT_NAME: image.astype(np.float32, copy=False)}

    def rewind(self) -> None:
        """Restart calibration from the first selected sample."""
        self._iterator = iter(self.indices)


def quantize_int8_onnx(
    *,
    input_path: Path,
    preprocessed_path: Path,
    output_path: Path,
    dataset: Any,
    sample_count: int = DEFAULT_CALIBRATION_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Quantize only accuracy-stable encoder spatial-reduction convolutions."""
    if not input_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {input_path}")

    indices = select_calibration_indices(len(dataset), sample_count)
    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quant_pre_process(
        input_model=input_path,
        output_model_path=preprocessed_path,
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )

    preprocessed_model = onnx.load(preprocessed_path)
    node_names = select_sr_conv_node_names(preprocessed_model)
    if len(node_names) != EXPECTED_SR_CONV_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SR_CONV_COUNT} spatial-reduction Conv nodes, "
            f"found {len(node_names)}"
        )

    quantize_static(
        model_input=preprocessed_path,
        model_output=output_path,
        calibration_data_reader=Scene06CalibrationReader(dataset, indices),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"],
        nodes_to_quantize=list(node_names),
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "DedicatedQDQPair": True,
            "QuantizeBias": False,
        },
    )

    result = onnx.load(output_path)
    onnx.checker.check_model(result)
    operators = Counter(node.op_type for node in result.graph.node)

    return {
        "calibration_indices": indices,
        "quantized_node_count": len(node_names),
        "quantize_linear_count": operators["QuantizeLinear"],
        "dequantize_linear_count": operators["DequantizeLinear"],
        "size_mib": output_path.stat().st_size / 1024**2,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create the selective INT8 PerceptionRT ONNX model."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--preprocessed",
        type=Path,
        default=DEFAULT_PREPROCESSED_ONNX_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_INT8_ONNX_PATH)
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLE_COUNT,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run leakage-safe Scene06 post-training quantization."""
    arguments = parse_arguments()
    generated_paths = (arguments.preprocessed, arguments.output)
    if not arguments.force:
        existing = [path for path in generated_paths if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing[0]}")

    config = load_training_config(arguments.config)
    seed_everything(config.seed)
    dataset = build_loaders(config, train_sample_limit=1).validation.dataset

    scenes = {sample.scene for sample in dataset.samples}
    if scenes != {"Scene06"}:
        raise RuntimeError(f"INT8 calibration must use only Scene06, found {sorted(scenes)}")

    indices = select_calibration_indices(
        len(dataset),
        arguments.calibration_samples,
    )
    distribution = Counter(dataset.samples[index].variation for index in indices)
    print("Calibration distribution:", dict(sorted(distribution.items())))

    result = quantize_int8_onnx(
        input_path=arguments.input,
        preprocessed_path=arguments.preprocessed,
        output_path=arguments.output,
        dataset=dataset,
        sample_count=arguments.calibration_samples,
    )

    print(f"Quantized SR Conv nodes: {result['quantized_node_count']}")
    print(f"Q/DQ nodes: {result['quantize_linear_count']}/{result['dequantize_linear_count']}")
    print(f"Created INT8 ONNX model: {arguments.output}")
    print(f"Size: {result['size_mib']:.2f} MiB")


if __name__ == "__main__":
    main()
