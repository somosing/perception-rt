import numpy as np

from perception_rt.data.vkitti2 import SemanticClass
from perception_rt.visualize import colorize_semantic_mask


def test_colorize_semantic_mask_restores_class_colors() -> None:
    classes = (
        SemanticClass(0, "Road", (100, 60, 100)),
        SemanticClass(1, "Sky", (90, 200, 255)),
    )
    class_ids = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )

    colorized = colorize_semantic_mask(class_ids, classes)

    np.testing.assert_array_equal(
        colorized,
        [
            [[100, 60, 100], [90, 200, 255]],
            [[90, 200, 255], [100, 60, 100]],
        ],
    )
