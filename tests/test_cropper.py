import pytest
from taulu.img_util import show
from util import table_image_path


@pytest.mark.visual
def test_cropper_full():
    from taulu.page_cropper import PageCropper

    cropper = PageCropper(
        target_hue=12,
        target_s=26,
        target_v=230,
        tolerance=40,
        margin=140,
        split=0.5,
        split_margin=0.06,
    )

    cropped, _ = cropper.crop(table_image_path(0))

    show(cropped, title="full crop")


@pytest.mark.visual
def test_cropper_split():
    from taulu.page_cropper import PageCropper

    cropper = PageCropper(
        target_hue=12,
        target_s=26,
        target_v=230,
        tolerance=40,
        margin=140,
        split=0.5,
        split_margin=0.06,
    )

    cropped, _ = cropper.crop_split(table_image_path(0))

    show(cropped.left, title="left crop")
    show(cropped.right, title="right crop")
