from pathlib import Path
import pytest
from tabular.img_util import show
from tabular.corner_filter import CornerFilter
from tabular.header_template import HeaderTemplate
from util import table_left_image_path, header_anno_path
import cv2


@pytest.mark.visual
def test_filter():
    filter = CornerFilter(
        kernel_size=41, cross_width=6, morph_size=4, region=60, k=0.05
    )
    im = cv2.imread(table_left_image_path(0))

    template = HeaderTemplate.from_saved(Path(header_anno_path(0)))

    filtered = filter.apply(im, True)

    show(filtered)

    # known start point, for now
    start = (300, 426)

    points = filter.find_table_points(
        im, start, template.cell_widths(1), template.cell_height()
    )

    points.show_cells(im)
