from pathlib import Path

import cv2
import numpy as np
import pytest

from perception_rt.data.vkitti2 import (
    discover_samples,
    load_color_table,
    load_depth,
    load_intrinsics,
    load_rgb,
    load_sample,
    load_semantic_mask,
    split_samples_by_scene,
)


def write_color_table(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_color_table_assigns_ordered_class_ids(tmp_path: Path) -> None:
    table_path = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\nRoad 100 60 100\nSky 90 200 255\nUndefined 0 0 0\n",
    )

    classes = load_color_table(table_path)

    assert [item.class_id for item in classes] == [0, 1, 2]
    assert [item.name for item in classes] == ["Road", "Sky", "Undefined"]
    assert classes[0].color == (100, 60, 100)


def test_load_color_table_rejects_invalid_rgb_value(tmp_path: Path) -> None:
    table_path = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\nRoad 300 60 100\n",
    )

    with pytest.raises(ValueError, match=r"outside \[0, 255\]"):
        load_color_table(table_path)


def test_load_color_table_rejects_duplicate_color(tmp_path: Path) -> None:
    table_path = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\nRoad 100 60 100\nAnotherClass 100 60 100\n",
    )

    with pytest.raises(ValueError, match="Duplicate semantic color"):
        load_color_table(table_path)


def test_load_intrinsics_builds_camera_matrix(tmp_path: Path) -> None:
    table_path = tmp_path / "intrinsic.txt"
    table_path.write_text(
        "frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n"
        "0 0 725.0087 725.0087 620.5 187\n"
        "0 1 725.0087 725.0087 620.5 187\n",
        encoding="utf-8",
    )

    intrinsics = load_intrinsics(table_path)
    camera = intrinsics[(0, 0)]

    assert len(intrinsics) == 2
    assert camera.frame == 0
    assert camera.camera_id == 0

    np.testing.assert_allclose(
        camera.matrix(),
        [
            [725.0087, 0.0, 620.5],
            [0.0, 725.0087, 187.0],
            [0.0, 0.0, 1.0],
        ],
    )


def test_load_intrinsics_rejects_duplicate_entry(tmp_path: Path) -> None:
    table_path = tmp_path / "intrinsic.txt"
    table_path.write_text(
        "frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n"
        "0 0 725.0 725.0 620.5 187\n"
        "0 0 725.0 725.0 620.5 187\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate intrinsic entry"):
        load_intrinsics(table_path)


def test_load_depth_converts_centimetres_to_metres(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    raw_depth = np.array(
        [
            [0, 394, 2500],
            [65535, 10000, 1],
        ],
        dtype=np.uint16,
    )
    assert cv2.imwrite(str(depth_path), raw_depth)

    depth = load_depth(depth_path)

    assert depth.values_m.dtype == np.float32
    assert depth.valid_mask.dtype == np.bool_

    np.testing.assert_allclose(
        depth.values_m,
        [
            [0.0, 3.94, 25.0],
            [0.0, 100.0, 0.01],
        ],
    )
    np.testing.assert_array_equal(
        depth.valid_mask,
        [
            [False, True, True],
            [False, True, True],
        ],
    )


def test_load_depth_rejects_eight_bit_image(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    raw_depth = np.array([[10, 20]], dtype=np.uint8)
    assert cv2.imwrite(str(depth_path), raw_depth)

    with pytest.raises(ValueError, match="Expected uint16"):
        load_depth(depth_path)


def test_load_rgb_converts_opencv_bgr_to_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    expected_rgb = np.array(
        [
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ]
        ],
        dtype=np.uint8,
    )
    image_bgr = cv2.cvtColor(expected_rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(image_path), image_bgr)

    loaded_rgb = load_rgb(image_path)

    assert loaded_rgb.dtype == np.uint8
    np.testing.assert_array_equal(loaded_rgb, expected_rgb)


def test_load_rgb_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError, match="Failed to read RGB"):
        load_rgb(missing_path)


def test_load_semantic_mask_converts_colors_to_class_ids(
    tmp_path: Path,
) -> None:
    color_table_path = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\nRoad 100 60 100\nSky 90 200 255\nUndefined 0 0 0\n",
    )
    classes = load_color_table(color_table_path)

    mask_path = tmp_path / "semantic.png"
    semantic_rgb = np.array(
        [
            [
                [100, 60, 100],
                [90, 200, 255],
            ],
            [
                [0, 0, 0],
                [100, 60, 100],
            ],
        ],
        dtype=np.uint8,
    )
    semantic_bgr = cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(mask_path), semantic_bgr)

    class_ids = load_semantic_mask(mask_path, classes)

    assert class_ids.dtype == np.int64
    np.testing.assert_array_equal(
        class_ids,
        [
            [0, 1],
            [2, 0],
        ],
    )


def test_load_semantic_mask_rejects_unknown_color(
    tmp_path: Path,
) -> None:
    color_table_path = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\nRoad 100 60 100\n",
    )
    classes = load_color_table(color_table_path)

    mask_path = tmp_path / "semantic.png"
    unknown_rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
    unknown_bgr = cv2.cvtColor(unknown_rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(mask_path), unknown_bgr)

    with pytest.raises(ValueError, match="unknown semantic colors"):
        load_semantic_mask(mask_path, classes)


def create_sample_files(
    root: Path,
    *,
    scene: str = "Scene01",
    variation: str = "clone",
    frame: int = 0,
    camera_id: int = 0,
) -> None:
    variation_root = root / scene / variation / "frames"

    rgb_path = variation_root / "rgb" / f"Camera_{camera_id}" / f"rgb_{frame:05d}.jpg"
    depth_path = variation_root / "depth" / f"Camera_{camera_id}" / f"depth_{frame:05d}.png"
    semantic_path = (
        variation_root / "classSegmentation" / f"Camera_{camera_id}" / f"classgt_{frame:05d}.png"
    )

    for path in (rgb_path, depth_path, semantic_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_discover_samples_builds_aligned_records(tmp_path: Path) -> None:
    create_sample_files(tmp_path, frame=1)
    create_sample_files(tmp_path, frame=0)

    samples = discover_samples(tmp_path, camera_id=0)

    assert len(samples) == 2
    assert [sample.frame for sample in samples] == [0, 1]
    assert samples[0].scene == "Scene01"
    assert samples[0].variation == "clone"
    assert samples[0].camera_id == 0
    assert samples[0].depth_path.name == "depth_00000.png"
    assert samples[0].semantic_path.name == "classgt_00000.png"


def test_discover_samples_rejects_missing_pair(tmp_path: Path) -> None:
    rgb_path = tmp_path / "Scene01" / "clone" / "frames" / "rgb" / "Camera_0" / "rgb_00000.jpg"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.touch()

    with pytest.raises(FileNotFoundError, match="Missing paired modality"):
        discover_samples(tmp_path)


def test_split_samples_by_scene_keeps_variations_together(
    tmp_path: Path,
) -> None:
    create_sample_files(
        tmp_path,
        scene="Scene01",
        variation="clone",
    )
    create_sample_files(
        tmp_path,
        scene="Scene01",
        variation="rain",
    )
    create_sample_files(
        tmp_path,
        scene="Scene06",
        variation="clone",
    )
    create_sample_files(
        tmp_path,
        scene="Scene18",
        variation="clone",
    )

    samples = discover_samples(tmp_path)
    splits = split_samples_by_scene(samples)

    assert {sample.scene for sample in splits.train} == {"Scene01"}
    assert {sample.variation for sample in splits.train} == {"clone", "rain"}
    assert {sample.scene for sample in splits.validation} == {"Scene06"}
    assert {sample.scene for sample in splits.test} == {"Scene18"}


def test_split_samples_by_scene_rejects_unknown_scene(
    tmp_path: Path,
) -> None:
    create_sample_files(
        tmp_path,
        scene="Scene99",
        variation="clone",
    )
    create_sample_files(tmp_path, scene="Scene06")
    create_sample_files(tmp_path, scene="Scene18")

    samples = discover_samples(tmp_path)

    with pytest.raises(ValueError, match="Unconfigured scenes.*Scene99"):
        split_samples_by_scene(samples)


def test_load_sample_returns_aligned_modalities(tmp_path: Path) -> None:
    create_sample_files(tmp_path)
    sample = discover_samples(tmp_path)[0]

    expected_rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    assert cv2.imwrite(
        str(sample.rgb_path),
        cv2.cvtColor(expected_rgb, cv2.COLOR_RGB2BGR),
    )

    raw_depth = np.array(
        [
            [100, 200],
            [300, 65535],
        ],
        dtype=np.uint16,
    )
    assert cv2.imwrite(str(sample.depth_path), raw_depth)

    semantic_rgb = np.array(
        [
            [[100, 60, 100], [90, 200, 255]],
            [[100, 60, 100], [90, 200, 255]],
        ],
        dtype=np.uint8,
    )
    assert cv2.imwrite(
        str(sample.semantic_path),
        cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR),
    )

    color_table = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\nRoad 100 60 100\nSky 90 200 255\n",
    )
    classes = load_color_table(color_table)

    loaded = load_sample(sample, classes)

    assert loaded.rgb.shape == (2, 2, 3)
    assert loaded.depth.values_m.shape == (2, 2)
    assert loaded.depth.valid_mask.shape == (2, 2)
    assert loaded.semantic_mask.shape == (2, 2)

    assert loaded.rgb.dtype == np.uint8
    np.testing.assert_allclose(
        loaded.depth.values_m,
        [[1.0, 2.0], [3.0, 0.0]],
    )
    np.testing.assert_array_equal(
        loaded.semantic_mask,
        [[0, 1], [0, 1]],
    )


def test_load_sample_rejects_spatial_mismatch(tmp_path: Path) -> None:
    create_sample_files(tmp_path)
    sample = discover_samples(tmp_path)[0]

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((3, 2), dtype=np.uint16)
    semantic = np.zeros((2, 2, 3), dtype=np.uint8)

    assert cv2.imwrite(str(sample.rgb_path), rgb)
    assert cv2.imwrite(str(sample.depth_path), depth)
    assert cv2.imwrite(str(sample.semantic_path), semantic)

    color_table = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\nUndefined 0 0 0\n",
    )
    classes = load_color_table(color_table)

    with pytest.raises(ValueError, match="RGB/depth shape mismatch"):
        load_sample(sample, classes)
