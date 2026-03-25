import cv2
import pytest
from util import (
    files_exist,
    header_image_path,
    header_left_anno_path,
    header_left_image_path,
    header_right_anno_path,
    header_right_image_path,
    table_filtered_path,
    table_image_path,
)

from taulu import img_util


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(header_image_path(0), header_left_anno_path(0)),
    reason="Files needed for test are missing",
)
def test_non_split():
    return
    from taulu import Taulu

    tl = Taulu(
        template_path=header_image_path(0),
        annotation_path=header_left_anno_path(0),
        row_height_factor=[0.85],
        binarization_sensitivity=0.05,
        search_radius=40,
        position_weight=0.8,
        intersection_kernel_size=31,
        line_thickness=8,
        line_gap_fill=4,
        min_rows=10,
        detection_threshold=0.5,
        extrapolation_distance=3,
    )

    im = cv2.imread(table_image_path(0))
    table = tl.segment_table(im, debug_view=True)

    table.visualize_points(im)
    table.show_cells(im)


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(
        table_image_path(0),
        header_left_image_path(0),
        header_right_image_path(0),
        header_left_anno_path(0),
        header_right_anno_path(0),
    ),
    reason="Files needed for test are missing",
)
def test_split():
    return
    from taulu import Taulu
    from taulu.split import Split

    tl = Taulu(
        template_path=Split(header_left_image_path(0), header_right_image_path(0)),
        annotation_path=Split(header_left_anno_path(0), header_right_anno_path(0)),
        row_height_factor=[0.85],
        binarization_sensitivity=0.05,
        search_radius=40,
        position_weight=0.8,
        intersection_kernel_size=31,
        line_thickness=8,
        line_gap_fill=4,
        min_rows=10,
        detection_threshold=0.5,
        extrapolation_distance=2,
    )

    im = cv2.imread(table_image_path(0))
    table = tl.segment_table(im, debug_view=True)

    table.visualize_points(im)
    table.show_cells(im)


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(
        table_image_path(1),
        table_filtered_path(1),
        header_left_image_path(1),
        header_right_image_path(1),
        header_left_anno_path(1),
        header_right_anno_path(1),
    ),
    reason="Files needed for test are missing",
)
def test_already_filtered():
    from taulu import Taulu
    from taulu.split import Split

    tl = Taulu(
        template_path=Split(header_left_image_path(1), header_right_image_path(1)),
        annotation_path=Split(header_left_anno_path(1), header_right_anno_path(1)),
        row_height_factor=Split([0.35, 0.23], [0.37, 0.24]),
        matching_scale=0.5,
        detection_scale=0.5,
        binarization_sensitivity=0.05,
        search_radius=40,
        position_weight=0.8,
        intersection_kernel_size=31,
        line_thickness=8,
        line_gap_fill=4,
        min_rows=20,
        detection_threshold=0.5,
        extrapolation_distance=5,
        pathfinding_threshold=0.1,
        growing_resets=10,
        reset_fraction=0.8,
    )

    im = cv2.imread(table_image_path(1))
    assert im is not None, f"Image {table_image_path(1)} couldn't be read"
    filtered = table_filtered_path(1)
    table = tl.segment_table(im, filtered=filtered, debug_view=False)
    table.save("points.json")

    cells = table.highlight_all_cells(im)
    img_util.show(cells)

    table.show_cells(im)
