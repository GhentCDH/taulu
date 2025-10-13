"""
The Taulu class is a convenience class that hides the inner workings
of taulu as much as possible.
"""

from time import perf_counter
import os
from os import PathLike
from os.path import exists
import cv2
from cv2.typing import MatLike
from pathlib import Path
import logging

from taulu.header_template import HeaderTemplate

from .split import Split
from .header_aligner import HeaderAligner
from .grid import GridDetector, TableGrid
from .error import TauluException

# needed: header images, header templates, parameters

logger = logging.getLogger(__name__)


# Helper function to get parameter value for a side
def get_param(param, side: str):
    if isinstance(param, Split):
        return getattr(param, side)
    return param


class Taulu:
    """
    High-level API for table segmentation from images.

    Taulu provides a simplified interface that orchestrates header alignment,
    grid detection, and table segmentation into a single workflow. It's designed
    to hide complexity while still allowing fine-tuned control through parameters.

    ## Workflow Overview

    1. **Header Template Creation**: Use `Taulu.annotate()` to create annotated
       header images that define your table structure
    2. **Initialization**: Create a Taulu instance with your header(s) and parameters
    3. **Segmentation**: Call `segment_table()` on your table images to get a
       `TableGrid` object containing all detected cell boundaries

    ## Single vs Split Tables

    Taulu supports two modes:

    - **Single header**: For tables that fit on one page or have consistent structure
    - **Split header**: For tables that span two pages (left/right) with potentially
      different parameters for each side

    Use `Split[T]` objects to provide different parameters for left and right sides.

    ## Parameter Tuning Strategy

    If segmentation fails or is inaccurate:

    1. **Visual debugging**: Set `debug_view=True` in `segment_table()` to see
       intermediate results
    2. **Adjust thresholding**: Modify `sauvola_k` to change binarization sensitivity
       - Increase to remove more noise (more aggressive)
       - Decrease to preserve faint lines
    3. **Tune cross-kernel**: Adjust `cross_width`, `cross_height`, `kernel_size`
       to match your rule thickness after morphology
    4. **Morphology**: Increase `morph_size` to connect broken lines, but be aware
       this also thickens lines (requiring larger cross_width)
    5. **Search parameters**: Increase `search_region` for warped documents,
       adjust `distance_penalty` to control how strictly positions are enforced
    6. **Growth parameters**: Lower `grow_threshold` if the algorithm stops too early,
       increase `look_distance` for better extrapolation

    Examples:
        Basic usage with a single header:

        >>> from taulu import Taulu
        >>>
        >>> # First, create annotated header (one-time setup)
        >>> Taulu.annotate("table_image.png", "header.png")
        >>> # This creates header.png and header.json
        >>>
        >>> # Initialize Taulu with the header
        >>> taulu = Taulu(
        ...     header_image_path="header.png",
        ...     cell_height_factor=0.8,  # Rows are 80% of header height
        ...     sauvola_k=0.25,
        ...     search_region=60,
        ...     cross_width=10
        ... )
        >>>
        >>> # Segment a table image
        >>> grid = taulu.segment_table("table_page_01.png")
        >>>
        >>> # Use the grid to extract cells
        >>> import cv2
        >>> img = cv2.imread("table_page_01.png")
        >>> cell_image = grid.crop_cell(img, (0, 0))  # First cell

        Using split headers for two-page tables:

        >>> from taulu import Taulu, Split
        >>>
        >>> # Annotate both headers
        >>> Taulu.annotate("scan_01.png", "header_left.png")
        >>> Taulu.annotate("scan_01.png", "header_right.png")
        >>>
        >>> # Use different parameters for each side
        >>> taulu = Taulu(
        ...     header_image_path=Split("header_left.png", "header_right.png"),
        ...     cell_height_factor=Split([0.8, 0.9], [0.75]),  # Different row heights
        ...     sauvola_k=Split(0.25, 0.30),  # Different thresholds
        ...     cross_width=10  # Same for both sides
        ... )
        >>>
        >>> # Segment returns a unified grid
        >>> grid = taulu.segment_table("scan_01.png")

        Debug visualization to tune parameters:

        >>> taulu = Taulu("header.png", sauvola_k=0.15)
        >>>
        >>> # Opens windows showing each processing step
        >>> # Press 'n' to advance, 'q' to quit
        >>> grid = taulu.segment_table("table.png", debug_view=True)
        >>>
        >>> # Adjust parameters based on what you see:
        >>> # - If binarization is too noisy: increase sauvola_k
        >>> # - If lines are broken after morphology: increase morph_size
        >>> # - If filtered image has "undefined" corners: adjust cross_width to match line thickness (after morphology)
        >>> # - If corners are missed during search: decrease grow_threshold or increase search_region


    Attributes:
        _header (MatLike | Split[MatLike]): Loaded header image(s)
        _aligner (HeaderAligner | Split[HeaderAligner]): Header alignment engine(s)
        _template (HeaderTemplate | Split[HeaderTemplate]): Parsed header structure(s)
        _grid_detector (GridDetector | Split[GridDetector]): Grid detection engine(s)
        _cell_heights (list[int] | Split[list[int]]): Computed cell heights in pixels

    Raises:
        TauluException: If header files don't exist, annotation is missing, or
            Split parameters are used incorrectly with single headers

    See Also:
        - `TableGrid`: The result object with methods for accessing cells
        - `Split`: Container for paired left/right parameters
        - `GridDetector`: Lower-level grid detection (for advanced usage)
        - `HeaderAligner`: Lower-level header alignment (for advanced usage)
    """

    def __init__(
        self,
        header_image_path: PathLike[str] | str | Split[PathLike[str] | str],
        cell_height_factor: float | list[float] | Split[float | list[float]] = [1.0],
        header_anno_path: PathLike[str]
        | str
        | Split[PathLike[str] | str]
        | None = None,
        sauvola_k: float | Split[float] = 0.25,
        search_region: int | Split[int] = 60,
        distance_penalty: float | Split[float] = 0.4,
        cross_width: int | Split[int] = 10,
        morph_size: int | Split[int] = 4,
        kernel_size: int | Split[int] = 41,
        processing_scale: float | Split[float] = 1.0,
        min_rows: int | Split[int] = 5,
        look_distance: int | Split[int] = 3,
        grow_threshold: float | Split[float] = 0.3,
    ):
        """
        Args:
            header_image_path:
                Path to the header template image(s). The header should be a cropped
                image showing a clear view of the table's first row. An annotation
                file (.json) must exist alongside the image, created via `Taulu.annotate()`.
                For split tables, provide a `Split` containing left and right header paths.

            cell_height_factor:
                Height of data rows relative to header height. For example, if your
                header is 100px tall and data rows are 80px tall, use 0.8.

                - **float**: All rows have the same height
                - **list[float]**: Different heights for different rows. The last value
                  is repeated for any additional rows beyond the list length. Useful when
                  the first data row is taller than subsequent rows.
                - **Split**: Different height factors for left and right sides

                Default: [1.0]

            header_anno_path (PathLike[str] | str | Split[PathLike[str] | str] | None):
                Optional explicit path to header annotation JSON file(s). If None,
                looks for a .json file with the same name as `header_image_path`.
                Default: None

            sauvola_k (float | Split[float]):
                Threshold sensitivity for Sauvola adaptive binarization (0.0-1.0).
                Controls how aggressively the algorithm converts the image to binary.

                - **Lower values** (0.04-0.15): Preserve faint lines, more noise
                - **Higher values** (0.20-0.35): Remove noise, may lose faint lines

                Start with 0.25 and adjust based on your image quality.
                Default: 0.25

            search_region (int | Split[int]):
                Size in pixels of the square region to search for the next corner point.
                The algorithm estimates where a corner should be, then searches within
                this region for the best match.

                - **Smaller values** (20-40): Faster, requires well-aligned tables
                - **Larger values** (60-100): More robust to warping and distortion

                Default: 60

            distance_penalty (float | Split[float]):
                Weight factor [0, 1] for penalizing corners far from expected position.
                Uses Gaussian weighting within the search region.

                - **0.0**: No penalty, any position in search region is equally valid
                - **0.5**: Moderate preference for positions near the expected location
                - **1.0**: Strong preference, only accepts positions very close to expected

                Default: 0.4

            cross_width (int | Split[int]):
                Width in pixels of the cross-shaped kernel used to detect intersections.
                Should approximately match the thickness of your table rules AFTER
                morphological dilation.

                **Tuning**: Look at the dilated image in debug_view. The cross_width
                should match the thickness of the black lines you see.
                Default: 10

            morph_size (int | Split[int]):
                Size of morphological structuring element for dilation. Controls how
                much gap-bridging occurs to connect broken line segments.

                - **Smaller values** (2-4): Minimal connection, preserves thin lines
                - **Larger values** (6-10): Connects larger gaps, but thickens lines

                Note: Increasing this requires increasing `cross_width` proportionally.
                Default: 4

            kernel_size (int | Split[int]):
                Size of the cross-shaped kernel (must be odd). Larger kernels are more
                selective, reducing false positives but potentially missing valid corners.

                - **Smaller values** (21-31): More sensitive, finds more candidates
                - **Larger values** (41-61): More selective, fewer false positives

                Default: 41

            processing_scale (float | Split[float]):
                Image downscaling factor (0, 1] for processing speed. Processing is done
                on scaled images, then results are scaled back to original size.

                - **1.0**: Full resolution (slowest, most accurate)
                - **0.5-0.75**: Good balance for high-res scans (2x-4x speedup)
                - **0.25-0.5**: Fast processing for very large images

                Default: 1.0

            min_rows (int | Split[int]):
                Minimum number of rows required before the algorithm considers the
                table complete. Prevents stopping too early on tables with initial
                low-confidence detections.
                Default: 5

            look_distance (int | Split[int]):
                Number of adjacent rows/columns to examine when extrapolating missing
                corners using polynomial regression. Higher values provide more context
                but may smooth over legitimate variations.

                - **2-3**: Good for consistent grids
                - **4-6**: Better for grids with some irregularity

                Default: 3

            grow_threshold (float | Split[float]):
                Initial minimum confidence [0, 1] required to accept a detected corner
                during the growing phase. The algorithm may adaptively lower this
                threshold if growth stalls.

                - **Higher values** (0.5-0.8): Stricter, fewer errors but may miss valid corners
                - **Lower values** (0.2-0.4): More permissive, finds more corners but more errors

                Default: 0.3

        """
        self._processing_scale = processing_scale
        self._cell_height_factor = cell_height_factor

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
                    min_rows=get_param(min_rows, "left"),
                    look_distance=get_param(look_distance, "left"),
                    grow_threshold=get_param(grow_threshold, "left"),
                ),
                GridDetector(
                    kernel_size=get_param(kernel_size, "right"),
                    cross_width=get_param(cross_width, "right"),
                    morph_size=get_param(morph_size, "right"),
                    search_region=get_param(search_region, "right"),
                    sauvola_k=get_param(sauvola_k, "right"),
                    distance_penalty=get_param(distance_penalty, "right"),
                    scale=get_param(self._processing_scale, "right"),
                    min_rows=get_param(min_rows, "right"),
                    look_distance=get_param(look_distance, "right"),
                    grow_threshold=get_param(grow_threshold, "right"),
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
                ]
            ):
                raise TauluException(
                    "Split parameters can only be used with split headers (tuple header_path)"
                )

            self._cell_heights = self._template.cell_heights(self._cell_height_factor)

            self._grid_detector = GridDetector(
                kernel_size=kernel_size,
                cross_width=cross_width,
                morph_size=morph_size,
                search_region=search_region,
                sauvola_k=sauvola_k,
                distance_penalty=distance_penalty,
                scale=self._processing_scale,
                min_rows=min_rows,
                look_distance=look_distance,
                grow_threshold=grow_threshold,
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
        debug_view: bool = False,
    ) -> TableGrid:
        """
        Segment a table image into a grid of cells.

        This is the main entry point for the taulu package. It orchestrates:

        1. **Header alignment**: Locates the table by matching the header template
           to the image using feature-based registration (ORB features + homography)
        2. **Grid detection**: Applies morphological filtering and cross-correlation
           to find corner intersections
        3. **Grid growing**: Iteratively detects corners row-by-row and column-by-column,
           starting from the aligned header position
        4. **Extrapolation**: Fills in any missing corners using polynomial regression
           based on neighboring detected points
        5. **Smoothing**: Refines corner positions for consistency

        ## Performance Notes

        Processing time depends on:
        - Image resolution (use `processing_scale < 1.0` for large images)
        - Table complexity (more rows/columns = longer processing)
        - Parameter settings

        ## Troubleshooting

        **If segmentation fails (returns incomplete grid):**
        1. Enable `debug_view=True` to see where it stops
        2. Check if header alignment is correct (first debug image)
        3. Verify cross-correlation shows bright spots at corners
        4. Adjust `grow_threshold` (lower if stopping too early)
        5. Increase `search_region` if corners are far from expected positions

        **If segmentation is inaccurate (corners in wrong positions):**
        1. Check binarization quality (adjust `sauvola_k`)
        2. Verify cross-kernel size matches line thickness (adjust `cross_width`)
        3. Ensure morphology isn't over-connecting (reduce `morph_size`)
        4. Increase `distance_penalty` to enforce expected positions more strictly

        Args:
            image (MatLike | PathLike[str] | str): Table image to segment.
                Can be a file path or a numpy array (BGR or grayscale).

            debug_view (bool): If True, opens OpenCV windows showing intermediate
                processing steps:
                - Header alignment overlay
                - Binarized image
                - After morphological operations
                - Cross-correlation result
                - Growing progress (corner-by-corner)

                **Controls:**
                - Press 'n' to advance to next step
                - Press 'q' to quit immediately

                Useful for parameter tuning and understanding failures.
                Default: False

        Returns:
            TableGrid: A grid structure containing detected corner positions with
                methods for:

                **Position queries:**
                - `cell(point)`: Get (row, col) at pixel coordinates (x, y)
                - `cell_polygon(cell)`: Get 4 corners of a cell as (lt, rt, rb, lb)
                - `region(start, end)`: Get bounding box for a cell range

                **Image extraction:**
                - `crop_cell(img, cell, margin=0)`: Extract single cell with optional margin
                - `crop_region(img, start, end, margin=0)`: Extract rectangular region

                **Visualization:**
                - `show_cells(img)`: Interactive cell viewer (click to highlight)
                - `highlight_all_cells(img)`: Draw all cell boundaries
                - `visualize_points(img)`: Show detected corner points

                **Analysis:**
                - `text_regions(img, row)`: Find continuous text regions in a row
                - `cells()`: Generator yielding all (row, col) indices

                **Persistence:**
                - `save(path)`: Save grid to JSON file
                - `TableGrid.from_saved(path)`: Load grid from JSON

                **Properties:**
                - `rows`: Number of data rows (header not included)
                - `cols`: Number of columns
                - `points`: Raw list of detected corner coordinates

        Raises:
            TauluException: If image cannot be loaded, header alignment fails,
                or grid detection produces no results

        Examples:
            Basic segmentation:

            >>> from taulu import Taulu
            >>> import cv2
            >>>
            >>> taulu = Taulu("header.png")
            >>> grid = taulu.segment_table("table_page_01.png")
            >>>
            >>> print(f"Detected {grid.rows} rows and {grid.cols} columns")
            >>>
            >>> # Extract first cell
            >>> img = cv2.imread("table_page_01.png")
            >>> cell_img = grid.crop_cell(img, (0, 0))
            >>> cv2.imwrite("cell_0_0.png", cell_img)

            Debug mode for parameter tuning:

            >>> grid = taulu.segment_table("table_page_01.png", debug_view=True)
            # Windows open showing each step
            # Adjust parameters based on what you see

            Process multiple images with the same header:

            >>> taulu = Taulu("header.png", sauvola_k=0.25)
            >>>
            >>> for i in range(1, 11):
            ...     img_path = f"table_page_{i:02d}.png"
            ...     grid = taulu.segment_table(img_path)
            ...     grid.save(f"grid_{i:02d}.json")
            ...     print(f"Page {i}: {grid.rows} rows detected")

            Extract all cells from a table:

            >>> img = cv2.imread("table.png")
            >>> grid = taulu.segment_table("table.png")
            >>>
            >>> for row, col in grid.cells():
            ...     cell_img = grid.crop_cell(img, (row, col), margin=5)
            ...     cv2.imwrite(f"cell_{row}_{col}.png", cell_img)

            Find text regions for OCR:

            >>> for row in range(grid.rows):
            ...     text_regions = grid.text_regions(img, row)
            ...     for start_cell, end_cell in text_regions:
            ...         # Extract region spanning multiple cells
            ...         region_img = grid.crop_region(img, start_cell, end_cell)
            ...         # Run OCR on region_img...

        See Also:
            - `TableGrid`: Complete documentation of the returned object
            - `GridDetector.find_table_points()`: Lower-level grid detection
            - `HeaderAligner.align()`: Lower-level header alignment
        """

        if not isinstance(image, MatLike):
            image = cv2.imread(os.fspath(image))

        now = perf_counter()
        h = self._aligner.align(image, visual=debug_view)
        align_time = perf_counter() - now
        logger.info(f"Header alignment took {align_time:.2f} seconds")

        # find the starting point for the table grid algorithm
        left_top_template = self._template.intersection((1, 0))
        if isinstance(left_top_template, Split):
            left_top_template = Split(
                (int(left_top_template.left[0]), int(left_top_template.left[1])),
                (int(left_top_template.right[0]), int(left_top_template.right[1])),
            )
        else:
            left_top_template = (int(left_top_template[0]), int(left_top_template[1]))

        left_top_table = self._aligner.template_to_img(h, left_top_template)

        now = perf_counter()
        table = self._grid_detector.find_table_points(
            image,
            left_top_table,
            self._template.cell_widths(0),
            self._cell_heights,
            visual=debug_view,
        )
        grid_time = perf_counter() - now
        logger.info(f"Grid detection took {grid_time:.2f} seconds")

        if isinstance(table, Split):
            table = TableGrid.from_split(table, (0, 0))

        return table
