# Taulu
_segments tables from images_

## Installation

### pip
```sh
pip install git+https://github.com/ghentcdh/taulu.git
```

### uv
```sh
uv add git+https://github.com/ghentcdh/taulu.git
```

## Usage

This package has a modular structure, defining multiple classes that can work together

example usage:

```py
from pathlib import Path

from taulu.page_cropper import PageCropper
from taulu.header_aligner import HeaderAligner
from taulu.header_template import HeaderTemplate
from taulu.corner_filter import CornerFilter

cropper = PageCropper(
    target_hue=12,
    target_s=26,
    target_v=230,
    tolerance=40,
    margin=140,
    split=0.5,
    split_margin=0.06,
)
aligner = HeaderAligner("header.png")
filter = CornerFilter(
    kernel_size=41, cross_width=6, morph_size=4, region=60, k=0.05
)
template = HeaderTemplate.from_saved("header.json")

cropped = cropper.crop("full_table.png")
h = aligner.align(cropped)

filtered = filter.apply(im, True)

start_point_template = template.intersection((0, 0))
start_point_cropped = aligner.template_to_img(h, start_point_template)

table_structure = filter.find_table_points(
    im, 
    start_point_template, 
    template.cell_widths(), 
    template.cell_height()
)

table_structure.show_cells(im)
```

using this setup, `table_structure`'s methods can be used to segments the image into its cells
