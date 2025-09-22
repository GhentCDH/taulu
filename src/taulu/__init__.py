"""
Taulu - *segment tables from images*

Taulu is a Python package designed to segment images of tables into their constituent rows and columns (and cells).

To use this package, you first need to make an annotation of the headers in your table images.
The idea is that these headers will be similar across your full set of images, and they will be
used as a starting point for the search algorithm that finds the table grid.

Here is an example python script of how to use Taulu:
```python
from taulu import Taulu
import os


def setup():
    # create an Annotation file of the headers in the image
    # (one for the left header, one for the right)
    # and store them in the examples directory
    print("Annotating the LEFT header...")
    Taulu.annotate("../data/table_00.png", "table_00_header_left.png")

    print("Annotating the RIGHT header...")
    Taulu.annotate("../data/table_00.png", "table_00_header_right.png")


def main():
    taulu = Taulu(("table_00_header_left.png", "table_00_header_right.png"))
    table = taulu.segment_table("../data/table_00.png",  cell_height_factor=0.8, debug_view=True)

    table.show_cells("../data/table_00.png")


if __name__ == "__main__":
    if os.path.exists("table_00_header_left.png") and os.path.exists(
        "table_00_header_right.png"
    ):
        main()
    else:
        setup()
        main()

```

If you want a high-level overview of how to use Taulu, see [the Taulu class](./taulu.html#taulu.taulu.Taulu)
"""

from .grid import GridDetector, TableGrid
from .header_aligner import HeaderAligner
from .header_template import HeaderTemplate
from .table_indexer import TableIndexer
from .split import Split
from .taulu import Taulu

__pdoc__ = {}
__pdoc__["constants"] = False
__pdoc__["main"] = False
__pdoc__["decorators"] = False
__pdoc__["error"] = False
__pdoc__["types"] = False
__pdoc__["img_util"] = False

__all__ = [
    "GridDetector",
    "TableGrid",
    "HeaderAligner",
    "HeaderTemplate",
    "TableIndexer",
    "Split",
    "Taulu",
]
