from taulu import HeaderTemplate
from taulu.img_util import show
from util import header_image_path, table_image_path


def test_annotation():
    _ = HeaderTemplate.annotate_image(header_image_path(0))


def test_annotation_crop():
    template = HeaderTemplate.annotate_image(table_image_path(0))

    cropped = template.crop_to_annotation(table_image_path(0))

    show(cropped)
