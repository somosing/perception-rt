"""Validate numerical parity between PyTorch and ONNX Runtime."""

import numpy as np

DEFAULT_ABSOLUTE_TOLERANCE = 1e-4
DEFAULT_RELATIVE_TOLERANCE = 1e-3


def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> dict[str, float | bool]:
    """Measure numerical agreement between identically shaped arrays."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )

    if reference.size == 0:
        raise ValueError("Arrays must not be empty")

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("Tolerances must be nonnegative")

    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Arrays must contain only finite values")

    reference_float = reference.astype(np.float64, copy=False)
    candidate_float = candidate.astype(np.float64, copy=False)
    absolute_error = np.abs(reference_float - candidate_float)
    allowed_error = absolute_tolerance + relative_tolerance * np.abs(reference_float)
    within_tolerance = absolute_error <= allowed_error

    return {
        "maximum_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "within_tolerance_fraction": float(within_tolerance.mean()),
        "all_within_tolerance": bool(within_tolerance.all()),
    }


def semantic_argmax_agreement(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
) -> float:
    """Return the fraction of pixels with matching semantic predictions."""
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("Semantic-logit shapes do not match")

    if reference_logits.ndim != 4 or reference_logits.shape[1] == 0:
        raise ValueError("Semantic logits must have nonempty NCHW shape")

    reference_classes = reference_logits.argmax(axis=1)
    candidate_classes = candidate_logits.argmax(axis=1)

    return float((reference_classes == candidate_classes).mean())
