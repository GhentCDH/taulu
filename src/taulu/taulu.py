"""
The Taulu class is a convenience class that hides the inner workings
of taulu as much as possible.
"""

from time import perf_counter
import os
from os import PathLike
from os.path import exists
from typing import Dict, List, Tuple
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


class Taulu:
    """
    The Taulu class is a convenience class that hides the inner workings of taulu as much as possible.

    For more advanced use cases, it might be useful to implement the workflow directly yourself,
    in order to have control over the intermediate steps.
    """

    def __init__(
        self,
        header_path: PathLike[str]
        | str
        | Tuple[PathLike[str] | str, PathLike[str] | str],
        sauvola_k: float = 0.25,
        search_region: int = 60,
        distance_penalty: float = 0.4,
        cross_width: int = 10,
        morph_size: int = 4,
        kernel_size: int = 41,
        processing_scale: float = 1.0,
        min_rows: int = 5,
        look_distance: int = 3,
        grow_threshold: float = 0.3,
    ):
        self._processing_scale = processing_scale

        if isinstance(header_path, Tuple):
            header = Split(Path(header_path[0]), Path(header_path[1]))

            if not exists(header.left.with_suffix(".png")) or not exists(
                header.right.with_suffix(".png")
            ):
                raise TauluException("The header images you provided do not exist")
            if not exists(header.left.with_suffix(".json")) or not exists(
                header.right.with_suffix(".json")
            ):
                raise TauluException(
                    "You need to annotate the headers of your table first\n\nsee the Taulu.annotate method"
                )

            template_left = HeaderTemplate.from_saved(header.left.with_suffix(".json"))
            template_right = HeaderTemplate.from_saved(
                header.right.with_suffix(".json")
            )

            self._header = Split(
                cv2.imread(os.fspath(header.left)), cv2.imread(os.fspath(header.right))
            )

            self._aligner = Split(
                HeaderAligner(self._header.left, scale=self._processing_scale),
                HeaderAligner(self._header.right, scale=self._processing_scale),
            )

            self._template = Split(template_left, template_right)

        else:
            header_path = Path(header_path)
            self._header = cv2.imread(os.fspath(header_path))
            self._aligner = HeaderAligner(self._header)
            self._template = HeaderTemplate.from_saved(header_path.with_suffix(".json"))

        # TODO: currently, these parameters are fixed and optimized for the example
        #       image specifically (which is probably a good starting point,
        #       espeicially after normalizing the image size)
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

        if isinstance(self._template, Split):
            self._grid_detector = Split(self._grid_detector, self._grid_detector)

    @staticmethod
    def annotate(image_path: PathLike[str] | str, output_path: PathLike[str] | str):
        """
        Annotate the header of a table image.

        Saves the annotated header image and a json file containing the
        header template to the output path.

        Args:
            image_path (PathLike[str]): the path of the image which you want to annotate
            output_path (PathLike[str]): the path where the output files should go (image files and json files)
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

    # TODO: check if PathLike works like this
    # TODO: get rid of cell_height and make this part of the header template
    def segment_table(
        self,
        image: MatLike | PathLike[str] | str,
        cell_height_factor: float | List[float] | Dict[str, float | List[float]],
        debug_view: bool = False,
    ) -> TableGrid:
        """
        Main function of the class, segmenting the input image into cells.

        Returns a TableGrid object, which has methods with which you can find
        the location of cells in the table

        Args:
            image (MatLike | PathLike[str]): The image to segment (path or np.ndarray)

            cell_height_factor (float | list[float] | dict[str, float | list[float]]): The height factor of a row. This factor is the fraction of the header height each row is.
                If your header has height 12 and your rows are of height 8, you should pass 8/12 as this argument.
                Also accepts a list of heights, useful if your row heights are not constant (often, the first row is
                higher than the others). The last entry in the list is used repeatedly when there are more
                rows in the image than there are entries in your list.

                By passing a dictionary with keys "left" and "right", you can specify a different cell_height_factor
                for the different sides of your table.

            debug_view (bool): By setting this setting to True, an OpenCV window will open and show the results of intermediate steps.
                Press `n` for advancing to the next image, and `q` to quit.
        """

        if not isinstance(image, MatLike):
            image = cv2.imread(os.fspath(image))

        # TODO: perform checks on the image

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

        if isinstance(cell_height_factor, dict):
            if not isinstance(self._template, Split):
                raise TauluException(
                    "You provided a cell_height_factor dictionary, but the header is not a Split"
                )
            if "left" not in cell_height_factor or "right" not in cell_height_factor:
                raise TauluException(
                    "When providing a cell_height_factor dictionary, it should contain both 'left' and 'right' keys"
                )
            cell_heights = Split(
                self._template.left.cell_heights(cell_height_factor.get("left", 1.0)),
                self._template.right.cell_heights(cell_height_factor.get("right", 1.0)),
            )
        else:
            cell_heights = self._template.cell_heights(cell_height_factor)

        now = perf_counter()
        table = self._grid_detector.find_table_points(
            image,
            left_top_table,
            self._template.cell_widths(0),
            cell_heights,
            visual=debug_view,
        )
        grid_time = perf_counter() - now
        logger.info(f"Grid detection took {grid_time:.2f} seconds")

        if isinstance(table, Split):
            table = TableGrid.from_split(table, (0, 0))

        return table
