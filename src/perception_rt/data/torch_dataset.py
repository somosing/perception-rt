"""PyTorch dataset for aligned Virtual KITTI 2 training samples."""

from collections.abc import Sequence
from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from perception_rt.data.vkitti2 import (
    SamplePaths,
    SemanticClass,
    load_sample,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TrainingSample(TypedDict):
    """Tensors and identifiers for one transformed training sample."""

    image: Tensor
    depth_m: Tensor
    depth_valid: Tensor
    semantic: Tensor
    scene: str
    variation: str
    frame: int


class VirtualKitti2Dataset(Dataset[TrainingSample]):
    """Load aligned Virtual KITTI 2 samples as training tensors."""

    def __init__(
        self,
        samples: Sequence[SamplePaths],
        classes: tuple[SemanticClass, ...],
        *,
        crop_size: tuple[int, int] = (320, 640),
        maximum_depth_m: float = 200.0,
        training: bool = False,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        if not samples:
            raise ValueError("Dataset requires at least one sample")

        crop_height, crop_width = crop_size

        if crop_height <= 0 or crop_width <= 0:
            raise ValueError("Crop dimensions must be positive")

        if maximum_depth_m <= 0.0:
            raise ValueError("Maximum depth must be positive")

        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("Flip probability must be within [0, 1]")

        self.samples = tuple(samples)
        self.classes = classes
        self.crop_size = crop_size
        self.maximum_depth_m = maximum_depth_m
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability

        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        """Return the number of indexed samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingSample:
        """Load and jointly transform one RGB-depth-semantic sample."""
        paths = self.samples[index]
        loaded = load_sample(paths, self.classes)

        image = torch.from_numpy(loaded.rgb.copy()).permute(2, 0, 1)
        depth_m = torch.from_numpy(loaded.depth.values_m.copy()).unsqueeze(0)
        depth_valid = torch.from_numpy(loaded.depth.valid_mask.copy()).unsqueeze(0)
        semantic = torch.from_numpy(loaded.semantic_mask.copy())

        image, depth_m, depth_valid, semantic = self._crop(
            image,
            depth_m,
            depth_valid,
            semantic,
        )

        if self.training and torch.rand(()) < self.horizontal_flip_probability:
            image = torch.flip(image, dims=(-1,))
            depth_m = torch.flip(depth_m, dims=(-1,))
            depth_valid = torch.flip(depth_valid, dims=(-1,))
            semantic = torch.flip(semantic, dims=(-1,))

        image = image.to(torch.float32).div(255.0)
        image = (image - self._mean) / self._std

        depth_m = depth_m.to(torch.float32)
        depth_valid = depth_valid.to(torch.bool)
        semantic = semantic.to(torch.int64)

        depth_valid &= depth_m > 0.0
        depth_valid &= depth_m <= self.maximum_depth_m
        depth_m = torch.where(depth_valid, depth_m, 0.0)

        return TrainingSample(
            image=image.contiguous(),
            depth_m=depth_m.contiguous(),
            depth_valid=depth_valid.contiguous(),
            semantic=semantic.contiguous(),
            scene=paths.scene,
            variation=paths.variation,
            frame=paths.frame,
        )

    def _crop(
        self,
        image: Tensor,
        depth_m: Tensor,
        depth_valid: Tensor,
        semantic: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply one spatial crop consistently to every modality."""
        image_height, image_width = semantic.shape
        crop_height, crop_width = self.crop_size

        if crop_height > image_height or crop_width > image_width:
            raise ValueError(
                f"Crop {self.crop_size} exceeds sample shape {(image_height, image_width)}"
            )

        maximum_top = image_height - crop_height
        maximum_left = image_width - crop_width

        if self.training:
            top = int(torch.randint(maximum_top + 1, ()).item())
            left = int(torch.randint(maximum_left + 1, ()).item())
        else:
            top = maximum_top // 2
            left = maximum_left // 2

        rows = slice(top, top + crop_height)
        columns = slice(left, left + crop_width)

        return (
            image[:, rows, columns],
            depth_m[:, rows, columns],
            depth_valid[:, rows, columns],
            semantic[rows, columns],
        )
