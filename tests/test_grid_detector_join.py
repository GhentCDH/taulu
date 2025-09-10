import pytest
from taulu import (
    HeaderAligner,
    PageCropper,
    GridDetector,
    HeaderTemplate,
    Split,
    TableGrid,
)
from util import (
    table_image_path,
    header_left_anno_path,
    header_right_anno_path,
    header_left_image_path,
    header_right_image_path,
    files_exist,
)
import cv2


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(
        header_left_anno_path(0),
        header_right_anno_path(0),
        header_left_image_path(0),
        header_right_image_path(0),
        table_image_path(0),
    ),
    reason="Files needed for test are missing",
)
def test_full():
    filter = GridDetector(
        kernel_size=41, cross_width=6, morph_size=4, region=60, k=0.05
    )
    cropper = PageCropper()

    templates = Split(
        HeaderTemplate.from_saved(header_left_anno_path(0)),
        HeaderTemplate.from_saved(header_right_anno_path(0)),
    )

    aligners = Split(
        HeaderAligner(header_left_image_path(0)),
        HeaderAligner(header_right_image_path(0)),
    )

    im = cv2.imread(table_image_path(0))

    cropped, offsets = cropper.crop_split(im)

    h = aligners.align(cropped)

    s = templates.intersection((1, 1))
    for i in s:
        i = (int(i[0]), int(i[1]))
    s: Split[tuple[int, int]] = aligners.template_to_img(h, s)

    crosses: Split[TableGrid] = s.apply(
        lambda start, img, widths, height: filter.find_table_points(  # type:ignore
            img, start, widths, height
        ),
        cropped,
        templates.cell_widths(1),
        templates.cell_height(0.8),
    )
    crosses.add_left_col(templates.cell_width(0))

    segmenter = TableGrid.from_split(crosses, offsets)

    segmenter.show_cells(im)
