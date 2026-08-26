"""Utilities for reading Virtual KITTI 2 data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SemanticClass:
    """One semantic class defined by Virtual KITTI 2."""

    class_id: int
    name: str
    color: tuple[int, int, int]

@dataclass(frozen=True)
class CameraIntrinsics:
    """Intrinsic parameters for one camera at one frame."""

    frame: int
    camera_id: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        """Return the 3x3 camera intrinsic matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def load_color_table(path: Path) -> tuple[SemanticClass, ...]:
    """Load a Virtual KITTI 2 colors.txt file.

    Class IDs are assigned according to the row order after the header.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    if not lines:
        raise ValueError(f"Empty color table: {path}")

    expected_header = ["Category", "r", "g", "b"]
    if lines[0].split() != expected_header:
        raise ValueError(f"Invalid color-table header in {path}")

    classes: list[SemanticClass] = []
    observed_colors: set[tuple[int, int, int]] = set()

    for class_id, line in enumerate(lines[1:]):
        columns = line.split()

        if len(columns) != 4:
            raise ValueError(f"Invalid color-table row: {line!r}")

        name = columns[0]

        try:
            color = (
                int(columns[1]),
                int(columns[2]),
                int(columns[3]),
            )
        except ValueError as error:
            raise ValueError(f"Non-integer RGB value in row: {line!r}") from error

        if any(channel < 0 or channel > 255 for channel in color):
            raise ValueError(f"RGB value outside [0, 255]: {color}")

        if color in observed_colors:
            raise ValueError(f"Duplicate semantic color: {color}")

        observed_colors.add(color)
        classes.append(
            SemanticClass(
                class_id=class_id,
                name=name,
                color=color,
            )
        )

    if not classes:
        raise ValueError(f"No semantic classes found in {path}")

    return tuple(classes)

def load_intrinsics(path: Path) -> dict[tuple[int, int], CameraIntrinsics]:
    """Load camera intrinsics indexed by (frame, camera_id)."""
    lines = path.read_text(encoding="utf-8").splitlines()

    if not lines:
        raise ValueError(f"Empty intrinsic table: {path}")

    expected_header = [
        "frame",
        "cameraID",
        "K[0,0]",
        "K[1,1]",
        "K[0,2]",
        "K[1,2]",
    ]
    if lines[0].split() != expected_header:
        raise ValueError(f"Invalid intrinsic-table header in {path}")

    intrinsics: dict[tuple[int, int], CameraIntrinsics] = {}

    for line in lines[1:]:
        columns = line.split()

        if len(columns) != 6:
            raise ValueError(f"Invalid intrinsic-table row: {line!r}")

        try:
            frame = int(columns[0])
            camera_id = int(columns[1])
            fx, fy, cx, cy = map(float, columns[2:])
        except ValueError as error:
            raise ValueError(f"Invalid numeric value in row: {line!r}") from error

        if frame < 0 or camera_id < 0:
            raise ValueError(f"Negative frame or camera ID in row: {line!r}")

        if fx <= 0 or fy <= 0:
            raise ValueError(f"Focal lengths must be positive: {line!r}")

        key = (frame, camera_id)

        if key in intrinsics:
            raise ValueError(f"Duplicate intrinsic entry: {key}")

        intrinsics[key] = CameraIntrinsics(
            frame=frame,
            camera_id=camera_id,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    if not intrinsics:
        raise ValueError(f"No camera intrinsics found in {path}")

    return intrinsics
