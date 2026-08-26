from pathlib import Path

import numpy as np
import pytest

from perception_rt.data.vkitti2 import load_color_table, load_intrinsics


def write_color_table(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_color_table_assigns_ordered_class_ids(tmp_path: Path) -> None:
    table_path = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\n"
        "Road 100 60 100\n"
        "Sky 90 200 255\n"
        "Undefined 0 0 0\n",
    )

    classes = load_color_table(table_path)

    assert [item.class_id for item in classes] == [0, 1, 2]
    assert [item.name for item in classes] == ["Road", "Sky", "Undefined"]
    assert classes[0].color == (100, 60, 100)


def test_load_color_table_rejects_invalid_rgb_value(tmp_path: Path) -> None:
    table_path = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\n"
        "Road 300 60 100\n",
    )

    with pytest.raises(ValueError, match=r"outside \[0, 255\]"):
        load_color_table(table_path)


def test_load_color_table_rejects_duplicate_color(tmp_path: Path) -> None:
    table_path = write_color_table(
        tmp_path / "colors.txt",
        "Category r g b\n"
        "Road 100 60 100\n"
        "AnotherClass 100 60 100\n",
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