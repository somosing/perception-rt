"""Tests for TensorRT engine construction helpers."""

from pathlib import Path

import pytest

from perception_rt.build_tensorrt import (
    DEFAULT_ENGINE_PATH,
    DEFAULT_FP16_ENGINE_PATH,
    DEFAULT_FP16_ONNX_PATH,
    DEFAULT_ONNX_PATH,
    EXPECTED_INPUT_SHAPE,
    EXPECTED_OUTPUT_SHAPES,
    collect_parser_errors,
    parse_onnx_network,
    resolve_default_paths,
    validate_network_contract,
)
from perception_rt.export_onnx import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAMES,
)


class FakeTensor:
    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: object,
    ) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype


class FakeNetwork:
    def __init__(
        self,
        *,
        output_names: tuple[str, ...] = ONNX_OUTPUT_NAMES,
        dtype: object | None = None,
    ) -> None:
        if dtype is None:
            dtype = object()
        self.inputs = [
            FakeTensor(
                ONNX_INPUT_NAME,
                EXPECTED_INPUT_SHAPE,
                dtype,
            )
        ]
        self.outputs = [
            FakeTensor(
                name,
                EXPECTED_OUTPUT_SHAPES[name],
                dtype,
            )
            for name in output_names
        ]

    @property
    def num_inputs(self) -> int:
        return len(self.inputs)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    def get_input(self, index: int) -> FakeTensor:
        return self.inputs[index]

    def get_output(self, index: int) -> FakeTensor:
        return self.outputs[index]


class FakeParser:
    def __init__(
        self,
        *,
        parsed: bool,
        errors: tuple[str, ...] = (),
    ) -> None:
        self.parsed = parsed
        self.errors = errors
        self.parsed_path: str | None = None

    @property
    def num_errors(self) -> int:
        return len(self.errors)

    def get_error(self, index: int) -> str:
        return self.errors[index]

    def parse_from_file(self, path: str) -> bool:
        self.parsed_path = path
        return self.parsed


def test_validate_network_contract_accepts_expected_contract() -> None:
    validate_network_contract(FakeNetwork())


def test_validate_network_contract_rejects_wrong_output_order() -> None:
    network = FakeNetwork(output_names=tuple(reversed(ONNX_OUTPUT_NAMES)))

    with pytest.raises(ValueError, match="Expected outputs"):
        validate_network_contract(network)


def test_collect_parser_errors_returns_every_error() -> None:
    parser = FakeParser(
        parsed=False,
        errors=("first", "second"),
    )

    assert collect_parser_errors(parser) == ("first", "second")


def test_parse_onnx_network_accepts_valid_model(
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"model")
    parser = FakeParser(parsed=True)

    parse_onnx_network(parser, onnx_path)

    assert parser.parsed_path == str(onnx_path)


def test_parse_onnx_network_reports_all_errors(
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"invalid")
    parser = FakeParser(
        parsed=False,
        errors=("first", "second"),
    )

    with pytest.raises(RuntimeError, match=r"first\nsecond"):
        parse_onnx_network(parser, onnx_path)


def test_parse_onnx_network_rejects_missing_file(
    tmp_path: Path,
) -> None:
    parser = FakeParser(parsed=True)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        parse_onnx_network(
            parser,
            tmp_path / "missing.onnx",
        )


@pytest.mark.parametrize(
    ("precision", "expected_paths"),
    [
        ("fp32", (DEFAULT_ONNX_PATH, DEFAULT_ENGINE_PATH)),
        (
            "fp16",
            (DEFAULT_FP16_ONNX_PATH, DEFAULT_FP16_ENGINE_PATH),
        ),
    ],
)
def test_resolve_default_paths(
    precision: str,
    expected_paths: tuple[Path, Path],
) -> None:
    assert resolve_default_paths(precision) == expected_paths


def test_resolve_default_paths_rejects_unknown_precision() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported TensorRT precision",
    ):
        resolve_default_paths("int4")


def test_validate_network_contract_enforces_dtype() -> None:
    expected_dtype = object()
    network = FakeNetwork(dtype=expected_dtype)

    validate_network_contract(
        network,
        expected_dtype=expected_dtype,
    )

    with pytest.raises(ValueError, match="Expected all tensors"):
        validate_network_contract(
            network,
            expected_dtype=object(),
        )
