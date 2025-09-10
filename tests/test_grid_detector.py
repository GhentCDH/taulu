import pytest
from taulu.img_util import show
from taulu import HeaderTemplate, GridDetector
from util import (
    table_left_image_path,
    header_anno_path,
    table_image_path,
    header_right_anno_path,
    header_left_anno_path,
    files_exist,
)
import cv2


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(header_left_anno_path(0), table_image_path(3)),
    reason="Files needed for test are missing",
)
def test_thumb():
    filter = GridDetector(
        kernel_size=41,
        cross_width=10,
        morph_size=7,
        search_region=30,
        sauvola_k=0.05,
        distance_penalty=0.5,
    )
    im = cv2.imread(table_image_path(3))

    template = HeaderTemplate.from_saved(header_left_anno_path(0))

    # known start point (should be retrieved from template alignment)
    start = (834, 1222)

    points = filter.find_table_points(
        im, start, template.cell_widths(0), template.cell_height(), visual=True
    )

    points.visualize_points(im)
    points.show_cells(im)


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(header_right_anno_path(1), table_image_path(1)),
    reason="Files needed for test are missing",
)
def test_filter():
    filter = GridDetector(
        kernel_size=41,
        cross_width=10,
        morph_size=7,
        search_region=40,
        sauvola_k=0.05,
        distance_penalty=0.8,
    )
    im = cv2.imread(table_image_path(1))

    template = HeaderTemplate.from_saved(header_right_anno_path(1))

    # known start point (should be retrieved from template alignment)
    start = (2937, 1531)
    # start = (838, 1585)

    points = filter.find_table_points(
        im, start, template.cell_widths(0), template.cell_height(0.44), visual=True
    )

    points.visualize_points(im)
    points.show_cells(im)


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(header_left_anno_path(0), table_image_path(0)),
    reason="Files needed for test are missing",
)
def test_text_regions():
    filter = GridDetector(
        kernel_size=41, cross_width=6, morph_size=4, search_region=60, sauvola_k=0.05
    )
    im = cv2.imread(table_left_image_path(0))

    template = HeaderTemplate.from_saved(header_anno_path(0))

    # known start point (should be retrieved from template alignment)
    start = (240, 419)

    points = filter.find_table_points(
        im, start, template.cell_widths(0), template.cell_height(), visual=True
    )

    points.show_cells(im)

    regions: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for row in range(points.rows):
        for region in points.text_regions(im, row):
            regions.add(region)

    for region in regions:
        crop = points.crop_region(im, region[0], region[1])
        show(crop, title="region crop")
