"""Load semantic class weights by class name."""

from pathlib import Path

import torch
import yaml
from torch import Tensor

from perception_rt.data.vkitti2 import SemanticClass


def load_semantic_class_weights(
    path: Path,
    classes: tuple[SemanticClass, ...],
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Load class weights and order them by semantic class ID."""
    contents = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(contents, dict):
        raise ValueError("Class weights must be a YAML mapping")

    expected_names = {semantic_class.name for semantic_class in classes}
    observed_names = set(contents)

    missing_names = expected_names - observed_names
    unknown_names = observed_names - expected_names

    if missing_names:
        raise ValueError("Missing class weights: " + ", ".join(sorted(missing_names)))

    if unknown_names:
        raise ValueError("Unknown class weights: " + ", ".join(sorted(unknown_names)))

    ordered_weights = []

    for semantic_class in classes:
        value = contents[semantic_class.name]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Weight for {semantic_class.name} must be numeric")

        weight = float(value)

        if weight < 0.0:
            raise ValueError(f"Weight for {semantic_class.name} cannot be negative")

        ordered_weights.append(weight)

    if not any(weight > 0.0 for weight in ordered_weights):
        raise ValueError("At least one class weight must be positive")

    return torch.tensor(
        ordered_weights,
        dtype=torch.float32,
        device=device,
    )
