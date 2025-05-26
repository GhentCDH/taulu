import pytest
from util import table_image_path, files_exist
import cv2 as cv


@pytest.mark.visual
@pytest.mark.skipif(
    not files_exist(
        table_image_path(0),
    ),
    reason="Files needed for test are missing",
)
def test_threshold():
    from taulu.img_util import sauvola, show

    im = cv.imread(table_image_path(0))
    result = sauvola(im, k=0.04, window_size=15)
    show(result, title="sauvola thresholded")
