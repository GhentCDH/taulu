import pytest
from util import (
    files_exist,
    header_image_path,
    header_left_anno_path,
    header_left_image_path,
    header_right_anno_path,
    header_right_image_path,
    table_image_path,
)
import cv2


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(header_image_path(0), header_left_anno_path(0)),
    reason="Files needed for test are missing",
)
def test_non_split():
    from taulu import Taulu

    tl = Taulu(
        header_image_path=header_image_path(0),
        header_anno_path=header_left_anno_path(0),
        cell_height_factor=[0.85],
        sauvola_k=0.05,
        search_region=40,
        distance_penalty=0.8,
        kernel_size=31,
        cross_width=8,
        morph_size=4,
        min_rows=10,
        grow_threshold=0.5,
        look_distance=3,
    )

    im = cv2.imread(table_image_path(0))
    table = tl.segment_table(im, debug_view=True)

    table.visualize_points(im)
    table.show_cells(im)


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(
        header_left_image_path(0),
        header_right_image_path(0),
        header_left_anno_path(0),
        header_right_anno_path(0),
    ),
    reason="Files needed for test are missing",
)
def test_split():
    from taulu import Taulu
    from taulu.split import Split

    tl = Taulu(
        header_image_path=Split(header_left_image_path(0), header_right_image_path(0)),
        header_anno_path=Split(header_left_anno_path(0), header_right_anno_path(0)),
        cell_height_factor=[0.85],
        sauvola_k=0.05,
        search_region=40,
        distance_penalty=0.8,
        kernel_size=31,
        cross_width=8,
        morph_size=4,
        min_rows=10,
        grow_threshold=0.5,
        look_distance=2,
    )

    im = cv2.imread(table_image_path(0))
    table = tl.segment_table(im, debug_view=True)

    table.visualize_points(im)
    table.show_cells(im)
