from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from perception_rt.data.torch_dataset import VirtualKitti2Dataset
from perception_rt.data.vkitti2 import (
    DepthMap,
    LoadedSample,
    SamplePaths,
    SemanticClass,
)


def make_loaded_sample() -> LoadedSample:
    paths = SamplePaths(
        scene="Scene01",
        variation="clone",
        frame=7,
        camera_id=0,
        rgb_path=Path("rgb.jpg"),
        depth_path=Path("depth.png"),
        semantic_path=Path("semantic.png"),
    )

    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    depth_m = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [10.0, 20.0, 30.0, 250.0, 50.0, 60.0],
            [70.0, 80.0, 0.0, 100.0, 200.0, 201.0],
            [110.0, 120.0, 130.0, 140.0, 150.0, 160.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.ones((4, 6), dtype=np.bool_)
    semantic = np.arange(4 * 6, dtype=np.int64).reshape(4, 6)

    return LoadedSample(
        paths=paths,
        rgb=rgb,
        depth=DepthMap(
            values_m=depth_m,
            valid_mask=valid_mask,
        ),
        semantic_mask=semantic,
    )


def make_dataset(
    monkeypatch: pytest.MonkeyPatch,
    *,
    training: bool = False,
    crop_size: tuple[int, int] = (2, 4),
    flip_probability: float = 0.5,
    jitter_strength: float = 0.0,
) -> VirtualKitti2Dataset:
    loaded = make_loaded_sample()
    monkeypatch.setattr(
        "perception_rt.data.torch_dataset.load_sample",
        lambda _paths, _classes: loaded,
    )

    classes = (SemanticClass(0, "Road", (100, 60, 100)),)

    return VirtualKitti2Dataset(
        [loaded.paths],
        classes,
        crop_size=crop_size,
        maximum_depth_m=200.0,
        training=training,
        horizontal_flip_probability=flip_probability,
        photometric_jitter_strength=jitter_strength,
    )


def test_validation_sample_uses_aligned_center_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(monkeypatch)

    sample = dataset[0]

    assert sample["image"].shape == (3, 2, 4)
    assert sample["image"].dtype == torch.float32
    assert sample["depth_m"].shape == (1, 2, 4)
    assert sample["depth_m"].dtype == torch.float32
    assert sample["depth_valid"].dtype == torch.bool
    assert sample["semantic"].dtype == torch.int64

    torch.testing.assert_close(
        sample["semantic"],
        torch.tensor(
            [
                [7, 8, 9, 10],
                [13, 14, 15, 16],
            ]
        ),
    )
    torch.testing.assert_close(
        sample["depth_m"],
        torch.tensor(
            [
                [
                    [20.0, 30.0, 0.0, 50.0],
                    [80.0, 0.0, 100.0, 200.0],
                ]
            ]
        ),
    )
    torch.testing.assert_close(
        sample["depth_valid"],
        torch.tensor(
            [
                [
                    [True, True, False, True],
                    [True, False, True, True],
                ]
            ]
        ),
    )

    assert sample["scene"] == "Scene01"
    assert sample["variation"] == "clone"
    assert sample["frame"] == 7


def test_training_flip_is_synchronized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        monkeypatch,
        training=True,
        crop_size=(4, 6),
        flip_probability=1.0,
    )

    sample = dataset[0]

    expected_semantic = torch.from_numpy(np.fliplr(make_loaded_sample().semantic_mask).copy())
    torch.testing.assert_close(sample["semantic"], expected_semantic)


def test_dataloader_batches_training_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(monkeypatch, crop_size=(4, 6))
    dataset.samples = dataset.samples * 2

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["image"].shape == (2, 3, 4, 6)
    assert batch["depth_m"].shape == (2, 1, 4, 6)
    assert batch["depth_valid"].shape == (2, 1, 4, 6)
    assert batch["semantic"].shape == (2, 4, 6)
    assert batch["scene"] == ["Scene01", "Scene01"]
    torch.testing.assert_close(batch["frame"], torch.tensor([7, 7]))


def test_dataset_rejects_crop_larger_than_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(monkeypatch, crop_size=(5, 6))

    with pytest.raises(ValueError, match="exceeds sample shape"):
        dataset[0]


def test_photometric_jitter_is_reproducible_and_target_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = make_dataset(
        monkeypatch,
        crop_size=(4, 6),
    )[0]
    dataset = make_dataset(
        monkeypatch,
        training=True,
        crop_size=(4, 6),
        flip_probability=0.0,
        jitter_strength=0.4,
    )

    torch.manual_seed(123)
    first = dataset[0]
    torch.manual_seed(123)
    second = dataset[0]

    assert not torch.allclose(
        first["image"],
        baseline["image"],
    )
    torch.testing.assert_close(
        first["image"],
        second["image"],
    )
    torch.testing.assert_close(
        first["depth_m"],
        baseline["depth_m"],
    )
    torch.testing.assert_close(
        first["depth_valid"],
        baseline["depth_valid"],
    )
    torch.testing.assert_close(
        first["semantic"],
        baseline["semantic"],
    )
    assert torch.isfinite(first["image"]).all()
