from taulu import HeaderTemplate
import pytest
from util import header_image_path, table_image_path, files_exist


@pytest.mark.skipif(
    not files_exist(header_image_path(0)),
    reason="Files needed for test are missing",
)
def test_annotation():
    _ = HeaderTemplate.annotate_image(header_image_path(0))


@pytest.mark.skipif(
    not files_exist(table_image_path(0)),
    reason="Files needed for test are missing",
)
def test_annotation_crop():
    _ = HeaderTemplate.annotate_image(table_image_path(0), crop="/tmp/header.png")
