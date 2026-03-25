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
from typing import Literal, cast

import cv2
from cv2.typing import MatLike
from numpy.typing import NDArray

from taulu.table_template import TableTemplate

from .config import TauluConfig
from .error import TauluException
from .grid import SegmentedTable, TableDetector
from .split import Split
from .template_matcher import FeatureDetector, TemplateMatcher

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
        3. Call `segment_table()` to get a `SegmentedTable` with cell boundaries

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
        template_path: Splittable[PathLike[str]] | Splittable[str],
        row_height_factor: Splittable[float] | Splittable[list[float]] | None = None,
        annotation_path: Splittable[PathLike[str]] | Splittable[str] | None = None,
        binarization_sensitivity: Splittable[float] = 0.25,
        search_radius: Splittable[int] = 60,
        position_weight: Splittable[float] = 0.4,
        line_thickness: Splittable[int] = 10,
        line_gap_fill: Splittable[int] = 4,
        intersection_kernel_size: Splittable[int] = 41,
        detection_scale: Splittable[float] = 1.0,
        pathfinding_threshold: Splittable[float] = 0.2,
        min_rows: Splittable[int] = 5,
        extrapolation_distance: Splittable[int] = 3,
        detection_threshold: Splittable[float] = 0.3,
        smooth: bool = False,
        smooth_strength: float = 0.5,
        smooth_iterations: int = 1,
        smooth_degree: int = 1,
        growing_resets: Splittable[int] = 0,
        reset_fraction: Splittable[float] = 0.5,
        feature_detector: Splittable[FeatureDetector] = "akaze",
        matching_scale: float = 1.0,
    ):
        """
        Args:
            template_path: Path to header template image(s). Use `Split` for two-page tables.
            row_height_factor: Row height relative to header (e.g., 0.8 for 80%). Default: [1.0]
            annotation_path: Explicit annotation JSON path. Default: inferred from image path.
            binarization_sensitivity: Binarization threshold (0.0-1.0). Higher = less noise. Default: 0.25
            search_radius: Corner search area in pixels. Default: 60
            position_weight: Position penalty weight [0, 1]. Default: 0.4
            line_thickness: Cross-kernel width matching line thickness. Default: 10
            line_gap_fill: Morphological dilation size. Default: 4
            intersection_kernel_size: Cross-kernel size (odd). Default: 41
            detection_scale: Image downscale factor (0, 1]. Default: 1.0
            pathfinding_threshold: Confidence to skip A* pathfinding. Default: 0.2
            min_rows: Minimum rows before completion. Default: 5
            extrapolation_distance: Rows to examine for extrapolation. Default: 3
            detection_threshold: Corner acceptance confidence [0, 1]. Default: 0.3
            smooth: Apply grid smoothing after detection. Default: False
            smooth_strength: Blend factor per smoothing iteration (0.0-1.0). Default: 0.5
            smooth_iterations: Number of smoothing passes. Default: 1
            smooth_degree: Polynomial degree for smoothing regression (1 or 2). Default: 1
            growing_resets: Number of grid cuts during growing. Default: 0
            reset_fraction: Fraction of points to delete per cut. Default: 0.5
            feature_detector: Feature matching method for header alignment. One of "orb"
                (fast, patent-free), "sift" (robust, uses FLANN), or "akaze" (robust,
                patent-free). Default: "akaze"
            matching_scale: Downscale factor (0, 1] for header alignment only. Lower
                values speed up feature matching. Default: 1.0
        """
        self._detection_scale = detection_scale
        self._smooth = smooth
        self._smooth_strength = smooth_strength
        self._smooth_iterations = smooth_iterations
        self._smooth_degree = smooth_degree

        if row_height_factor is None:
            row_height_factor = [1.0]

        self._row_height_factor = row_height_factor

        if isinstance(template_path, Split) or isinstance(annotation_path, Split):
            header = Split(Path(template_path.left), Path(template_path.right))

            if not exists(header.left.with_suffix(".png")) or not exists(
                header.right.with_suffix(".png")
            ):
                raise TauluException(
                    "The header images you provided do not exist (or they aren't .png files)"
                )

            if annotation_path is None:
                if not exists(header.left.with_suffix(".json")) or not exists(
                    header.right.with_suffix(".json")
                ):
                    raise TauluException(
                        "You need to annotate the headers of your table first\n\nsee the Taulu.annotate method"
                    )

                template_left = TableTemplate.from_saved(
                    header.left.with_suffix(".json")
                )
                template_right = TableTemplate.from_saved(
                    header.right.with_suffix(".json")
                )

            else:
                if not exists(annotation_path.left) or not exists(  # ty: ignore[unresolved-attribute]
                    annotation_path.right  # ty: ignore[unresolved-attribute]
                ):
                    raise TauluException(
                        "The header annotation files you provided do not exist (or they aren't .json files)"
                    )

                template_left = TableTemplate.from_saved(annotation_path.left)  # ty: ignore[unresolved-attribute]
                template_right = TableTemplate.from_saved(annotation_path.right)  # ty: ignore[unresolved-attribute]

            self._header = Split(
                cv2.imread(os.fspath(header.left)), cv2.imread(os.fspath(header.right))
            )

            self._aligner = Split(
                TemplateMatcher(
                    self._header.left,
                    method=get_param(feature_detector, "left"),
                    scale=matching_scale,
                ),
                TemplateMatcher(
                    self._header.right,
                    method=get_param(feature_detector, "right"),
                    scale=matching_scale,
                ),
            )

            self._template = Split(template_left, template_right)

            self._cell_heights = Split(
                self._template.left.cell_heights(get_param(row_height_factor, "left")),
                self._template.right.cell_heights(
                    get_param(row_height_factor, "right")
                ),
            )

            # Create TableDetector for left and right with potentially different parameters
            self._grid_detector = Split(
                TableDetector(
                    intersection_kernel_size=get_param(
                        intersection_kernel_size, "left"
                    ),
                    line_thickness=get_param(line_thickness, "left"),
                    line_gap_fill=get_param(line_gap_fill, "left"),
                    search_radius=get_param(search_radius, "left"),
                    binarization_sensitivity=get_param(
                        binarization_sensitivity, "left"
                    ),
                    position_weight=get_param(position_weight, "left"),
                    detection_scale=get_param(self._detection_scale, "left"),
                    pathfinding_threshold=get_param(pathfinding_threshold, "left"),
                    min_rows=get_param(min_rows, "left"),
                    extrapolation_distance=get_param(extrapolation_distance, "left"),
                    detection_threshold=get_param(detection_threshold, "left"),
                    growing_resets=get_param(growing_resets, "left"),
                    reset_fraction=get_param(reset_fraction, "left"),
                ),
                TableDetector(
                    intersection_kernel_size=get_param(
                        intersection_kernel_size, "right"
                    ),
                    line_thickness=get_param(line_thickness, "right"),
                    line_gap_fill=get_param(line_gap_fill, "right"),
                    search_radius=get_param(search_radius, "right"),
                    binarization_sensitivity=get_param(
                        binarization_sensitivity, "right"
                    ),
                    position_weight=get_param(position_weight, "right"),
                    detection_scale=get_param(self._detection_scale, "right"),
                    pathfinding_threshold=get_param(pathfinding_threshold, "right"),
                    min_rows=get_param(min_rows, "right"),
                    extrapolation_distance=get_param(extrapolation_distance, "right"),
                    detection_threshold=get_param(detection_threshold, "right"),
                    growing_resets=get_param(growing_resets, "right"),
                    reset_fraction=get_param(reset_fraction, "right"),
                ),
            )

        else:
            template_path = Path(template_path)
            self._header = cv2.imread(os.fspath(template_path))
            self._aligner = TemplateMatcher(
                self._header,
                method=cast(FeatureDetector, feature_detector),
                scale=matching_scale,
            )
            self._template = TableTemplate.from_saved(
                template_path.with_suffix(".json")
            )

            # For single header, parameters should not be Split objects
            if any(
                isinstance(param, Split)
                for param in [
                    binarization_sensitivity,
                    search_radius,
                    position_weight,
                    line_thickness,
                    line_gap_fill,
                    intersection_kernel_size,
                    detection_scale,
                    min_rows,
                    extrapolation_distance,
                    detection_threshold,
                    row_height_factor,
                    growing_resets,
                    reset_fraction,
                    feature_detector,
                ]
            ):
                raise TauluException(
                    "Split parameters can only be used with split headers (tuple header_path)"
                )

            self._cell_heights = self._template.cell_heights(
                cast(list[float] | float, self._row_height_factor)
            )

            self._grid_detector = TableDetector(
                intersection_kernel_size=intersection_kernel_size,  # ty: ignore
                line_thickness=line_thickness,  # ty: ignore
                line_gap_fill=line_gap_fill,  # ty: ignore
                search_radius=search_radius,  # ty: ignore
                binarization_sensitivity=binarization_sensitivity,  # ty: ignore
                position_weight=position_weight,  # ty: ignore
                detection_scale=self._detection_scale,  # ty: ignore
                pathfinding_threshold=pathfinding_threshold,  # ty: ignore
                min_rows=min_rows,  # ty: ignore
                extrapolation_distance=extrapolation_distance,  # ty: ignore
                detection_threshold=detection_threshold,  # ty: ignore
                growing_resets=growing_resets,  # ty:ignore
                reset_fraction=reset_fraction,  # ty:ignore
            )

    @classmethod
    def from_config(cls, config: TauluConfig) -> "Taulu":
        """
        Create a :class:`Taulu` instance from a :class:`~taulu.config.TauluConfig`.

        Args:
            config: A populated :class:`~taulu.config.TauluConfig` instance.

        Returns:
            A :class:`Taulu` instance configured according to ``config``.

        Example::

            from taulu import Taulu
            from taulu.config import TauluConfig

            config = TauluConfig.from_toml("my_table.toml")
            taulu = Taulu.from_config(config)
        """
        import dataclasses

        return cls(
            **{f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
        )

    @staticmethod
    def annotate(
        image_path: PathLike[str] | str,
        output_path: PathLike[str] | str,
        *,
        backend: Literal["auto", "gui", "notebook"] = "auto",
    ):
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

        def running_in_notebook() -> bool:
            try:
                from IPython import get_ipython

                ip = get_ipython()
                return ip is not None and "IPKernelApp" in ip.config
            except Exception:
                return False

        # Decide backend
        if backend not in ("auto", "gui", "notebook"):
            raise TauluException("backend must be one of: 'auto', 'gui', 'notebook'")
        if backend == "auto":
            use_notebook = running_in_notebook()
        else:
            use_notebook = backend == "notebook"

        if use_notebook:
            # Notebook way
            logger.info(
                "\x1b[32mNotebook environment detected/selected. Using notebook annotation backend."
            )
            session = TableTemplate.annotate_image_notebook(
                os.fspath(image_path), crop=output_path.with_suffix(".png")
            )
            session._save_path = output_path.with_suffix(".json")  # ty: ignore[unresolved-attribute]
            return session

        else:
            # GUI way
            template = TableTemplate.annotate_image(
                os.fspath(image_path), crop=output_path.with_suffix(".png")
            )
            template.save(output_path.with_suffix(".json"))

    def segment_table(
        self,
        image: MatLike | PathLike[str] | str,
        filtered: MatLike | PathLike[str] | str | None = None,
        debug_view: bool = False,
        debug_view_notebook: bool = False,
    ) -> SegmentedTable:
        """
        Segment a table image into a grid of cells.

        Orchestrates header alignment, grid detection, corner growing, and
        extrapolation to produce a complete grid structure.

        Args:
            image: Table image to segment (file path or numpy array).
            filtered: Optional pre-filtered binary image for corner detection.
                If provided, binarization parameters are ignored.
            debug_view: Show intermediate processing steps via OpenCV windows. Press 'n' to advance,
                'q' to quit. Default: False
            debug_view_notebook: Show intermediate processing steps inline in a Jupyter notebook
                using matplotlib. Default: False

        Returns:
            SegmentedTable: Grid structure with methods for cell access (`crop_cell`,
                `cell_polygon`), visualization (`show_cells`), and persistence
                (`save`, `from_saved`).

        Raises:
            TauluException: If image cannot be loaded or grid detection fails.
        """

        if not isinstance(image, MatLike):
            image = cast(str | PathLike[str], image)
            tmp_image = cv2.imread(os.fspath(image))
            assert tmp_image is not None
            image = tmp_image

        now = perf_counter()
        h = self._aligner.align(
            image,  # ty: ignore[invalid-argument-type]
            visual=debug_view,
            visual_notebook=debug_view_notebook,
        )
        align_time = perf_counter() - now
        logger.info(f"Header alignment took {align_time:.2f} seconds")

        # find the starting point for the table grid algorithm

        def make_top_row(template: TableTemplate, aligner: TemplateMatcher, h: NDArray):
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
            visual_notebook=debug_view_notebook,
            filtered=filtered,  # ty:ignore
            smooth=self._smooth,
            smooth_strength=self._smooth_strength,
            smooth_iterations=self._smooth_iterations,
            smooth_degree=self._smooth_degree,
        )
        grid_time = perf_counter() - now
        logger.info(f"Grid detection took {grid_time:.2f} seconds")

        if debug_view_notebook:
            self._aligner.show_matches_notebook()

        if isinstance(table, Split):
            table = SegmentedTable.from_split(table, (0, 0))  # ty: ignore

        return table
