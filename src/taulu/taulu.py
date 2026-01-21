"""
The Taulu class is a convenience class that hides the inner workings
of taulu as much as possible.
"""

import logging
import os
from os import PathLike
from os.path import exists
from pathlib import Path
from time import perf_counter
from typing import Optional, cast

import cv2
from cv2.typing import MatLike
from numpy.typing import NDArray

from taulu.header_template import HeaderTemplate

from .error import TauluException
from .grid import GridDetector, TableGrid
from .header_aligner import HeaderAligner
from .split import Split

# needed: header images, header templates, parameters

logger = logging.getLogger(__name__)


# Helper function to get parameter value for a side
def get_param(param, side: str):
    if isinstance(param, Split):
        return getattr(param, side)
    return param


type Splittable[T] = Split[T] | T


class Taulu:
    """
    High-level API for table segmentation from images.

    Taulu orchestrates header alignment, grid detection, and table segmentation
    into a single workflow.

    Workflow:
        1. Create annotated header images via `Taulu.annotate()`
        2. Initialize Taulu with header(s) and parameters
        3. Call `segment_table()` to get a `TableGrid` with cell boundaries

    For two-page tables, use `Split[T]` to provide different parameters for
    left and right sides.

    Example:
        >>> from taulu import Taulu
        >>> Taulu.annotate("table_image.png", "header.png")
        >>> taulu = Taulu("header.png")
        >>> grid = taulu.segment_table("table_page_01.png")
        >>> cell_image = grid.crop_cell(cv2.imread("table_page_01.png"), (0, 0))
    """

    def __init__(
        self,
        header_image_path: Splittable[PathLike[str]] | Splittable[str],
        cell_height_factor: Splittable[float] | Splittable[list[float]] = [1.0],
        header_anno_path: Splittable[PathLike[str]] | Splittable[str] | None = None,
        sauvola_k: Splittable[float] = 0.25,
        search_region: Splittable[int] = 60,
        distance_penalty: Splittable[float] = 0.4,
        cross_width: Splittable[int] = 10,
        morph_size: Splittable[int] = 4,
        kernel_size: Splittable[int] = 41,
        processing_scale: Splittable[float] = 1.0,
        skip_astar_threshold: Splittable[float] = 0.2,
        min_rows: Splittable[int] = 5,
        look_distance: Splittable[int] = 3,
        grow_threshold: Splittable[float] = 0.3,
        smooth_grid: bool = False,
        cuts: Splittable[int] = 3,
        cut_fraction: Splittable[float] = 0.5,
    ):
        """
        Args:
            header_image_path: Path to header template image(s). Use `Split` for two-page tables.
            cell_height_factor: Row height relative to header (e.g., 0.8 for 80%). Default: [1.0]
            header_anno_path: Explicit annotation JSON path. Default: inferred from image path.
            sauvola_k: Binarization threshold (0.0-1.0). Higher = less noise. Default: 0.25
            search_region: Corner search area in pixels. Default: 60
            distance_penalty: Position penalty weight [0, 1]. Default: 0.4
            cross_width: Cross-kernel width matching line thickness. Default: 10
            morph_size: Morphological dilation size. Default: 4
            kernel_size: Cross-kernel size (odd). Default: 41
            processing_scale: Image downscale factor (0, 1]. Default: 1.0
            skip_astar_threshold: Confidence to skip A* pathfinding. Default: 0.2
            min_rows: Minimum rows before completion. Default: 5
            look_distance: Rows to examine for extrapolation. Default: 3
            grow_threshold: Corner acceptance confidence [0, 1]. Default: 0.3
            smooth_grid: Apply grid smoothing after detection. Default: False
            cuts: Number of grid cuts during growing. Default: 3
            cut_fraction: Fraction of points to delete per cut. Default: 0.5
        """
        self._processing_scale = processing_scale
        self._cell_height_factor = cell_height_factor
        self._smooth = smooth_grid

        if isinstance(header_image_path, Split) or isinstance(header_anno_path, Split):
            header = Split(Path(header_image_path.left), Path(header_image_path.right))

            if not exists(header.left.with_suffix(".png")) or not exists(
                header.right.with_suffix(".png")
            ):
                raise TauluException(
                    "The header images you provided do not exist (or they aren't .png files)"
                )

            if header_anno_path is None:
                if not exists(header.left.with_suffix(".json")) or not exists(
                    header.right.with_suffix(".json")
                ):
                    raise TauluException(
                        "You need to annotate the headers of your table first\n\nsee the Taulu.annotate method"
                    )

                template_left = HeaderTemplate.from_saved(
                    header.left.with_suffix(".json")
                )
                template_right = HeaderTemplate.from_saved(
                    header.right.with_suffix(".json")
                )

            else:
                if not exists(header_anno_path.left) or not exists(
                    header_anno_path.right
                ):
                    raise TauluException(
                        "The header annotation files you provided do not exist (or they aren't .json files)"
                    )

                template_left = HeaderTemplate.from_saved(header_anno_path.left)
                template_right = HeaderTemplate.from_saved(header_anno_path.right)

            self._header = Split(
                cv2.imread(os.fspath(header.left)), cv2.imread(os.fspath(header.right))
            )

            self._aligner = Split(
                HeaderAligner(
                    self._header.left, scale=get_param(self._processing_scale, "left")
                ),
                HeaderAligner(
                    self._header.right, scale=get_param(self._processing_scale, "right")
                ),
            )

            self._template = Split(template_left, template_right)

            self._cell_heights = Split(
                self._template.left.cell_heights(get_param(cell_height_factor, "left")),
                self._template.right.cell_heights(
                    get_param(cell_height_factor, "right")
                ),
            )

            # Create GridDetector for left and right with potentially different parameters
            self._grid_detector = Split(
                GridDetector(
                    kernel_size=get_param(kernel_size, "left"),
                    cross_width=get_param(cross_width, "left"),
                    morph_size=get_param(morph_size, "left"),
                    search_region=get_param(search_region, "left"),
                    sauvola_k=get_param(sauvola_k, "left"),
                    distance_penalty=get_param(distance_penalty, "left"),
                    scale=get_param(self._processing_scale, "left"),
                    skip_astar_threshold=get_param(skip_astar_threshold, "left"),
                    min_rows=get_param(min_rows, "left"),
                    look_distance=get_param(look_distance, "left"),
                    grow_threshold=get_param(grow_threshold, "left"),
                    cuts=get_param(cuts, "left"),
                    cut_fraction=get_param(cut_fraction, "left"),
                ),
                GridDetector(
                    kernel_size=get_param(kernel_size, "right"),
                    cross_width=get_param(cross_width, "right"),
                    morph_size=get_param(morph_size, "right"),
                    search_region=get_param(search_region, "right"),
                    sauvola_k=get_param(sauvola_k, "right"),
                    distance_penalty=get_param(distance_penalty, "right"),
                    scale=get_param(self._processing_scale, "right"),
                    skip_astar_threshold=get_param(skip_astar_threshold, "right"),
                    min_rows=get_param(min_rows, "right"),
                    look_distance=get_param(look_distance, "right"),
                    grow_threshold=get_param(grow_threshold, "right"),
                    cuts=get_param(cuts, "right"),
                    cut_fraction=get_param(cut_fraction, "right"),
                ),
            )

        else:
            header_image_path = Path(header_image_path)
            self._header = cv2.imread(os.fspath(header_image_path))
            self._aligner = HeaderAligner(self._header)
            self._template = HeaderTemplate.from_saved(
                header_image_path.with_suffix(".json")
            )

            # For single header, parameters should not be Split objects
            if any(
                isinstance(param, Split)
                for param in [
                    sauvola_k,
                    search_region,
                    distance_penalty,
                    cross_width,
                    morph_size,
                    kernel_size,
                    processing_scale,
                    min_rows,
                    look_distance,
                    grow_threshold,
                    cell_height_factor,
                    cuts,
                    cut_fraction,
                ]
            ):
                raise TauluException(
                    "Split parameters can only be used with split headers (tuple header_path)"
                )

            self._cell_heights = self._template.cell_heights(self._cell_height_factor)

            self._grid_detector = GridDetector(
                kernel_size=kernel_size,  # ty: ignore
                cross_width=cross_width,  # ty: ignore
                morph_size=morph_size,  # ty: ignore
                search_region=search_region,  # ty: ignore
                sauvola_k=sauvola_k,  # ty: ignore
                distance_penalty=distance_penalty,  # ty: ignore
                scale=self._processing_scale,  # ty: ignore
                skip_astar_threshold=skip_astar_threshold,  # ty: ignore
                min_rows=min_rows,  # ty: ignore
                look_distance=look_distance,  # ty: ignore
                grow_threshold=grow_threshold,  # ty: ignore
                cuts=cuts,
                cut_fraction=cut_fraction,
            )

    @staticmethod
    def annotate(image_path: PathLike[str] | str, output_path: PathLike[str] | str):
        """
        Interactive tool to create header annotations for table segmentation.

        This method guides you through a two-step annotation process:

        1. **Crop the header**: Click four corners to define the header region
        2. **Annotate lines**: Click pairs of points to define each vertical and
           horizontal line in the header

        The annotations are saved as:
        - A cropped header image (.png) at `output_path`
        - A JSON file (.json) containing line coordinates

        ## Annotation Guidelines

        **Which lines to annotate:**
        - All vertical lines that extend into the table body (column separators)
        - The top horizontal line of the header
        - The bottom horizontal line of the header (top of data rows)

        **Order doesn't matter** - annotate lines in any order that's convenient.

        **To annotate a line:**
        1. Click once at one endpoint
        2. Click again at the other endpoint
        3. A green line appears showing your annotation

        **To undo:**
        - Right-click anywhere to remove the last line you drew

        **When finished:**
        - Press 'n' to save and exit
        - Press 'q' to quit without saving

        Args:
            image_path (PathLike[str] | str): Path to a table image containing
                a clear view of the header. This can be a full table image.
            output_path (PathLike[str] | str): Where to save the cropped header
                image. The annotation JSON will be saved with the same name but
                .json extension.

        Raises:
            TauluException: If image_path doesn't exist or output_path is a directory

        Examples:
            Annotate a single header:

            >>> from taulu import Taulu
            >>> Taulu.annotate("scan_page_01.png", "header.png")
            # Interactive window opens
            # After annotation: creates header.png and header.json

            Annotate left and right headers for a split table:

            >>> Taulu.annotate("scan_page_01.png", "header_left.png")
            >>> Taulu.annotate("scan_page_01.png", "header_right.png")
            # Creates header_left.{png,json} and header_right.{png,json}

        Notes:
            - The header image doesn't need to be perfectly cropped initially -
              the tool will help you crop it precisely
            - Annotation accuracy is important: misaligned lines will cause
              segmentation errors
            - You can re-run this method to update annotations if needed
        """

        if not exists(image_path):
            raise TauluException(f"Image path {image_path} does not exist")

        if os.path.isdir(output_path):
            raise TauluException("Output path should be a file")

        output_path = Path(output_path)

        template = HeaderTemplate.annotate_image(
            os.fspath(image_path), crop=output_path.with_suffix(".png")
        )

        template.save(output_path.with_suffix(".json"))

    def segment_table(
        self,
        image: MatLike | PathLike[str] | str,
        filtered: Optional[MatLike | PathLike[str] | str] = None,
        debug_view: bool = False,
    ) -> TableGrid:
        """
        Segment a table image into a grid of cells.

        Orchestrates header alignment, grid detection, corner growing, and
        extrapolation to produce a complete grid structure.

        Args:
            image: Table image to segment (file path or numpy array).
            filtered: Optional pre-filtered binary image for corner detection.
                If provided, binarization parameters are ignored.
            debug_view: Show intermediate processing steps. Press 'n' to advance,
                'q' to quit. Default: False

        Returns:
            TableGrid: Grid structure with methods for cell access (`crop_cell`,
                `cell_polygon`), visualization (`show_cells`), and persistence
                (`save`, `from_saved`).

        Raises:
            TauluException: If image cannot be loaded or grid detection fails.
        """

        if not isinstance(image, MatLike):
            image = cast(str | PathLike[str], image)
            image = cv2.imread(os.fspath(image))

        now = perf_counter()
        h = self._aligner.align(image, visual=debug_view)
        align_time = perf_counter() - now
        logger.info(f"Header alignment took {align_time:.2f} seconds")

        # find the starting point for the table grid algorithm

        def make_top_row(template: HeaderTemplate, aligner: HeaderAligner, h: NDArray):
            top_row = []
            for x in range(template.cols + 1):
                on_template = template.intersection((1, x))
                on_template = (int(on_template[0]), int(on_template[1]))

                on_img = aligner.template_to_img(h, on_template)

                top_row.append(on_img)

            return top_row

        if isinstance(self._aligner, Split):
            top_row = Split(
                make_top_row(self._template.left, self._aligner.left, h.left),  # ty:ignore
                make_top_row(self._template.right, self._aligner.right, h.right),  # ty:ignore
            )
        else:
            top_row = make_top_row(self._template, self._aligner, h)  # ty:ignore

        now = perf_counter()
        table = self._grid_detector.find_table_points(
            image,  # ty:ignore
            top_row,  # ty:ignore
            self._template.cell_widths(0),
            self._cell_heights,  # ty:ignore
            visual=debug_view,
            filtered=filtered,  # ty:ignore
            smooth=self._smooth,
        )
        grid_time = perf_counter() - now
        logger.info(f"Grid detection took {grid_time:.2f} seconds")

        if isinstance(table, Split):
            table = TableGrid.from_split(table, (0, 0))  # ty: ignore

        return table
