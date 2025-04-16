import pytest
from taulu.img_util import show
from taulu.corner_filter import CornerFilter
from taulu.header_template import HeaderTemplate
from util import table_left_image_path, header_anno_path
import cv2


@pytest.mark.visual
def test_filter():
    filter = CornerFilter(
        kernel_size=41, cross_width=6, morph_size=4, region=60, k=0.05
    )
    im = cv2.imread(table_left_image_path(0))

    template = HeaderTemplate.from_saved(header_anno_path(0))

    filtered = filter.apply(im, True)

    show(filtered)

    # known start point (should be retrieved from template alignment)
    start = (240, 426)

    points = filter.find_table_points(
        im, start, template.cell_widths(0), template.cell_height()
    )

    points.visualize_points(im)
    points.show_cells(im)


@pytest.mark.visual
def test_text_regions():
    filter = CornerFilter(
        kernel_size=41, cross_width=6, morph_size=4, region=60, k=0.05
    )
    im = cv2.imread(table_left_image_path(0))

    template = HeaderTemplate.from_saved(header_anno_path(0))

    # known start point (should be retrieved from template alignment)
    start = (240, 426)

    points = filter.find_table_points(
        im, start, template.cell_widths(0), template.cell_height()
    )

    regions = set()
    for row in range(points.rows):
        for region in points.text_regions(im, row):
            regions.add(region)

    print(f"regions: {regions}")

    exit(1)
