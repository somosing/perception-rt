"""Utilities for reading Virtual KITTI 2 data."""

from dataclasses import dataclass
from pathlib import Path

import cv2
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


@dataclass(frozen=True)
class DepthMap:
    """Decoded metric depth and its validity mask."""

    values_m: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class SamplePaths:
    """Paths and identifiers for one aligned Virtual KITTI 2 sample."""

    scene: str
    variation: str
    frame: int
    camera_id: int
    rgb_path: Path
    depth_path: Path
    semantic_path: Path


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


def load_depth(path: Path) -> DepthMap:
    """Load a Virtual KITTI 2 depth PNG as metric float32 depth.

    Raw values are centimetres. Zero and the uint16 maximum value are treated
    as invalid. Invalid metric-depth values are stored as zero and must be
    ignored using valid_mask.
    """
    raw_depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if raw_depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")

    if raw_depth.ndim != 2:
        raise ValueError(f"Expected single-channel depth image, got shape {raw_depth.shape}")

    if raw_depth.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth image, got {raw_depth.dtype}")

    invalid_far_plane = np.iinfo(np.uint16).max
    valid_mask = (raw_depth > 0) & (raw_depth < invalid_far_plane)

    values_m = raw_depth.astype(np.float32) / 100.0
    values_m[~valid_mask] = 0.0

    return DepthMap(
        values_m=values_m,
        valid_mask=valid_mask,
    )


def load_rgb(path: Path) -> np.ndarray:
    """Load an RGB image as an HxWx3 uint8 array."""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected three-channel RGB image, got {image_bgr.shape}")

    if image_bgr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got {image_bgr.dtype}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_semantic_mask(
    path: Path,
    classes: tuple[SemanticClass, ...],
) -> np.ndarray:
    """Load an RGB semantic mask and convert its colors to int64 class IDs."""
    semantic_rgb = load_rgb(path)
    class_ids = np.full(semantic_rgb.shape[:2], -1, dtype=np.int64)

    for semantic_class in classes:
        color = np.asarray(semantic_class.color, dtype=np.uint8)
        matches = np.all(semantic_rgb == color, axis=2)
        class_ids[matches] = semantic_class.class_id

    unknown_mask = class_ids == -1

    if np.any(unknown_mask):
        unknown_colors = np.unique(
            semantic_rgb[unknown_mask].reshape(-1, 3),
            axis=0,
        )
        preview = unknown_colors[:5].tolist()
        raise ValueError(
            f"Found {len(unknown_colors)} unknown semantic colors in {path}: {preview}"
        )

    return class_ids


def discover_samples(
    root: Path,
    camera_id: int = 0,
) -> tuple[SamplePaths, ...]:
    """Discover aligned RGB, depth, and semantic samples."""
    if camera_id not in (0, 1):
        raise ValueError(f"Camera ID must be 0 or 1, got {camera_id}")

    pattern = f"Scene*/*/frames/rgb/Camera_{camera_id}/rgb_*.jpg"
    rgb_paths = sorted(root.glob(pattern))

    if not rgb_paths:
        raise FileNotFoundError(f"No Camera_{camera_id} RGB images found below {root}")

    samples: list[SamplePaths] = []

    for rgb_path in rgb_paths:
        variation_directory = rgb_path.parents[3]
        scene_directory = rgb_path.parents[4]

        frame_text = rgb_path.stem.removeprefix("rgb_")

        try:
            frame = int(frame_text)
        except ValueError as error:
            raise ValueError(f"Invalid RGB frame name: {rgb_path.name}") from error

        depth_path = (
            variation_directory
            / "frames"
            / "depth"
            / f"Camera_{camera_id}"
            / f"depth_{frame:05d}.png"
        )
        semantic_path = (
            variation_directory
            / "frames"
            / "classSegmentation"
            / f"Camera_{camera_id}"
            / f"classgt_{frame:05d}.png"
        )

        missing_paths = [path for path in (depth_path, semantic_path) if not path.is_file()]

        if missing_paths:
            missing_text = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Missing paired modality for {rgb_path}: {missing_text}")

        samples.append(
            SamplePaths(
                scene=scene_directory.name,
                variation=variation_directory.name,
                frame=frame,
                camera_id=camera_id,
                rgb_path=rgb_path,
                depth_path=depth_path,
                semantic_path=semantic_path,
            )
        )

    return tuple(samples)
