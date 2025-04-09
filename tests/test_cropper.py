import pytest
from tabular.img_util import show
from util import image_path


@pytest.mark.visual
def test_cropper_full():
    from tabular.page_cropper import PageCropper

    cropper = PageCropper(
        target_hue=12,
        target_s=26,
        target_v=230,
        tolerance=40,
        margin=140,
        split=0.5,
        split_margin=0.06,
    )

    cropped = cropper.crop(image_path(0))

    show(cropped, title="full crop")


@pytest.mark.visual
def test_cropper_split():
    from tabular.page_cropper import PageCropper

    cropper = PageCropper(
        target_hue=12,
        target_s=26,
        target_v=230,
        tolerance=40,
        margin=140,
        split=0.5,
        split_margin=0.06,
    )

    cropped = cropper.crop_split(image_path(0))

    show(cropped.left, title="left crop")
    show(cropped.right, title="right crop")
