import cv2
import pytest
from util import files_exist, header_image_path, table_left_image_path


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(header_image_path(0), table_left_image_path(0)),
    reason="Files needed for test are missing",
)
def test_aligner():
    from taulu.template_matcher import TemplateMatcher

    im = cv2.imread(table_left_image_path(0))
    assert im is not None, f"Unable to read image {table_left_image_path(0)}"
    header = cv2.imread(header_image_path(0))

    aligner = TemplateMatcher(header)
    h = aligner.align(im)

    aligner.view_alignment(im, h)


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(header_image_path(0), table_left_image_path(0)),
    reason="Files needed for test are missing",
)
def test_aligner_thresholded():
    from taulu.template_matcher import TemplateMatcher

    im = cv2.imread(table_left_image_path(0))
    assert im is not None, f"Unable to read image {table_left_image_path(0)}"
    header = cv2.imread(header_image_path(0))
    assert header is not None, f"Unable to read image {header_image_path(0)}"

    aligner = TemplateMatcher(header, k=0.10)
    h = aligner.align(im, visual=True)

    aligner.view_alignment(im, h)


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(header_image_path(0), table_left_image_path(0)),
    reason="Files needed for test are missing",
)
def test_aligner_transform():
    from taulu.img_util import show
    from taulu.template_matcher import TemplateMatcher

    im = cv2.imread(table_left_image_path(0))
    assert im is not None, f"Unable to read image {table_left_image_path(0)}"
    header = cv2.imread(header_image_path(0))
    assert header is not None, f"Unable to read image {header_image_path(0)}"

    aligner = TemplateMatcher(header)
    h = aligner.align(im)

    lt = aligner.template_to_img(h, (0, 0))
    rt = aligner.template_to_img(h, (header.shape[1], 0))
    rb = aligner.template_to_img(h, (header.shape[1], header.shape[0]))
    lb = aligner.template_to_img(h, (0, header.shape[0]))

    cv2.circle(im, lt, 5, (0, 255, 0), 2)
    cv2.circle(im, rt, 5, (0, 255, 0), 2)
    cv2.circle(im, rb, 5, (0, 255, 0), 2)
    cv2.circle(im, lb, 5, (0, 255, 0), 2)

    show(im)
