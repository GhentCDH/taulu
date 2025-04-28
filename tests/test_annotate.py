from taulu import HeaderTemplate
from util import header_image_path, table_image_path


def test_annotation():
    _ = HeaderTemplate.annotate_image(header_image_path(0))


