<p align="center">
  <img src="./data/banner.svg" alt="Banner" width="400"/>
  <br>
  <i>Segmentation of tables from images</i>
  <br>
  <br>
  <a href="https://pypi.org/project/taulu/">
    <img src="https://img.shields.io/pypi/v/taulu" alt="PyPi version of taulu" />
  </a>
  <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/ghentcdh/taulu/maturin.yml">
</p>

## Data Requirements 

This package assumes that you are working with images of tables that have **clearly visible rules** (the lines that divide the table into cells).

To fully utilize the automated workflow, your tables should include a recognizable header. This header will be used to identify the position of the first cell in the input image and determine the expected widths of the table's cells.

For optimal segmentation, ensure that the tables are rotated so the borders are approximately vertical and horizontal. Minor page warping is acceptable.


## Installation

### Using pip
```sh
pip install taulu
```

### Using uv
```sh
uv add taulu
```

## Example

```bash
git clone https://github.com/GhentCDH/taulu.git
cd taulu/examples
bash run.bash
```

During this example, you will need to annotate the header image. You do this by simply clicking twice per line, once for each endpoint. It does not matter in which order you annotate the lines. Example:

![Table Header Annotation Example](./data/header_annotation.png)

Below is an example of table cell identification using the `Taulu` package:

![Table Cell Identification Example](./data/example_segmentation.gif)


## Workflow

This package is structured in a modular way, with several components that work together.

The algorithm identifies the header's location in the input image, which provides a starting point. From there, it scans the image to find intersections of the rules (borders) and segments the image into cells accordingly.

The output is a `TableGrid` object that contains the detected intersections, enabling you to segment the image into rows, columns, and cells.

Here is a visualization of the workflow and the components:

```mermaid
flowchart LR
    h(header.png) --> A[HeaderAligner]
    t(table.png) --> C[PageCropper]
    j(header.json) --> T[HeaderTemplate]
    C --> F[GridDetector]
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
- `GridDetector`: Processes the image to identify intersections of horizontal and vertical lines (borders).
- `h`: A transformation matrix that maps points from the header template to the input image.
- `s`: The starting point of the segmentation algorithm (typically the top-left intersection, just below the header).

## Parameters

The taulu algorithm has a few parameters which you might need to tune in order for it to fit your data's characteristics.
The following is a summary of the most important parameters and how you could tune them to your data.

### `GridDetector`

- `kernel_size`, `cross_width`, `cross_height`: The GridDetector uses a kernel to detect intersections of rules in the image. By default, `cross_height` follows the value of `cross_width`. The kernel looks like this:

  ![kernel diagram](./data/kernel.svg)

  The goal is to make this kernel look like the actual corners in your images after thresholding and dilation. The example script shows the dilated result, which you can use to estimate the `cross_width` and `cross_height` values that fit your image.
  Note that the optimal values will depend on the `morph_size` parameter too.
- `morph_size`: The GridDetector uses a dilation step in order to _connect lines_ in the image that might be broken up after thresholding. With a larger `morph_size`, larger gaps in the lines will be connected, but it will also lead to much thicker lines. As such, this parameter affects the optimal `cross_width` and `cross_height`.
- `region`: This parameter influences the search algorithm. The algorithm starts at an already-detected intersection, and jumps right with a distance that is derived from the annotated header template. At the new location, the algorithm then finds the best corner-match that is within a square of size `region` around that point, and selects that as the detected corner. Visualized:

  ![search algorithm region](./data/search.svg)

  A larger region will be more forgiving for warping or other artefacts, but could lead to false positives too.
- `k`, `w`: These parameters affect the thresholding algorithm that's used in the `GridDetector`. `k` adjusts the threshold. Larger values of `k` correspond with a larger threshold, meaning more pixels will be mapped to zero. You should increase this parameter until most of the noise is gone in your image, without removing too many pixels from the actual lines of the table. `w` is less important, but adjusts the window size of the sauvola thresholding algorithm that is used under the hood.

### `HeaderTemplate`

- `intersection((row, height))`: this method calculates the intersection of a horizontal and vertical line in the annotated header template. For example, running `template.intersection((1, 1))` corresponds with this intersection:

  ![intersection diagram](./data/intersect.svg)

  This point can then be transformed to the image using the aligner, and this can serve as the starting point of the search algorithm. Note that in this case, the first column is skipped. This can often be useful since the `GridDetector` kernel looks for crosses, and the left-most intersection often only has a T shape (the left leg of the cross might be missing).
  If that is the case with your data too, it is a good idea to set the starting point to the (1, 1) intersection, and add in the first row later using the `add_left_col(width)` function. When doing this, you also need to set the parameter of the `cell_widths` function to `1`. See [this example](./examples/example.py).
- `cell_height(fraction: float)`: this method defines a single cell height for all of the rows. The fraction is multiplied with the height of the annotated header template to get the cell height relative to it.
