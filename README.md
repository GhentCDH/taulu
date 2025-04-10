# Taulu
_Segmentation of tables from images_

## Data Requirements 

This package assumes that you are working with images of tables that have **clearly visible rules** (the lines that divide the table into cells).

To fully utilize the automated workflow, your tables should include a recognizable header. This header will be used to identify the position of the first cell in the input image and determine the expected widths of the table's cells.

For optimal segmentation, ensure that the tables are rotated so the borders are approximately vertical and horizontal. Minor page warping is acceptable.


## Installation

### Using pip
```sh
pip install git+https://github.com/ghentcdh/taulu.git
```

### Using uv
```sh
uv add git+https://github.com/ghentcdh/taulu.git
```


## Workflow

This package is structured in a modular way, with several components that work together.

The algorithm identifies the header's location in the input image, which provides a starting point. From there, it scans the image to find intersections of the rules (borders) and segments the image into cells accordingly.

The output is a `TableCrosses` object that contains the detected intersections, enabling you to segment the image into rows, columns, and cells.

Here is a visualization of the workflow and the components:

```mermaid
flowchart LR
    h(header.png) --> A[HeaderAligner]
    t(table.png) --> C[PageCropper]
    j(header.json) --> T[HeaderTemplate]
    C --> F[CornerFilter]
    A --> H((h))
    C --> H
    T --> S((s))
    H --> S
    F --> R
    S --> R(result)
    T --> R
```

The components are:

- `HeaderAligner`: Uses template matching to identify the header's location in the input images.
- `PageCropper`: An optional component that crops the image to a region containing a given color. This is useful if your image contains a lot of background, but can be skipped if the table occupies most of the image. Only works if your table has a distinct color from the background.
- `HeaderTemplate`: Stores table template information by reading an annotation JSON file. You can create this file by running `HeaderTemplate.annotate_image` on a cropped image of your table’s header.
You can make such a file by running `HeaderTemplate.annotate_image` on a cropped image of your table's header.
- `CornerFilter`: Processes the image to identify intersections of horizontal and vertical lines (borders).
- `h`: A transformation matrix that maps points from the header template to the input image.
- `s`: The starting point of the segmentation algorithm (typically the top-left intersection, just below the header).

example code:

```py
from pathlib import Path
from cv2 import imshow

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

# crop the input image (this step is only necessary if the image contains more than just the table)
cropped = cropper.crop("table.png")
h = aligner.align(cropped)

# find the intersections of rules in the image
# the `True` parameter means that intermediate results are shown too, for debugging and parameter tuning
filtered = filter.apply(im, True)

# define the start point as the intersection of the first 
start_point_template = template.intersection((1, 0))
start_point_cropped = aligner.template_to_img(h, start_point_template)

table_structure = filter.find_table_points(
    im, 
    start_point_template, 
    template.cell_widths(), 
    template.cell_height()
)

table_structure.show_cells(im)

# get
cell_3_4 = table_structure.crop_cell((3,4))
imshow("cell (row: 3, col: 4)", cell_3_4)
```

