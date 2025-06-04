from abc import ABC, abstractmethod
from typing import Generator, Tuple

import cv2 as cv
from cv2.typing import MatLike
import numpy as np

from . import img_util as imu
from .constants import WINDOW
from .error import TauluException

Point = Tuple[int, int]


def _add(left: Point, right: Point) -> Point:
    return (left[0] + right[0], left[1] + right[1])


def _apply_margin(
    lt: Point,
    rt: Point,
    rb: Point,
    lb: Point,
    margin: int = 0,
    margin_top: int | None = None,
    margin_bottom: int | None = None,
    margin_left: int | None = None,
    margin_right: int | None = None,
    margin_y: int | None = None,
    margin_x: int | None = None,
) -> tuple[Point, Point, Point, Point]:
    """
    Apply margins to the bounding box, with priority:
        top/bottom/left/right > x/y > margin
    """

    top = (
        margin_top
        if margin_top is not None
        else (margin_y if margin_y is not None else margin)
    )
    bottom = (
        margin_bottom
        if margin_bottom is not None
        else (margin_y if margin_y is not None else margin)
    )
    left = (
        margin_left
        if margin_left is not None
        else (margin_x if margin_x is not None else margin)
    )
    right = (
        margin_right
        if margin_right is not None
        else (margin_x if margin_x is not None else margin)
    )

    lt_out = _add(lt, (-left, -top))
    rt_out = _add(rt, (right, -top))
    rb_out = _add(rb, (right, bottom))
    lb_out = _add(lb, (-left, bottom))

    return lt_out, rt_out, rb_out, lb_out


class TableIndexer(ABC):
    """
    Subclasses implement methods for going from a pixel in the input image to a table cell index,
    and cropping an image to the given table cell index.
    """

    def __init__(self):
        self._col_offset = 0

    @property
    def col_offset(self) -> int:
        return self._col_offset

    @col_offset.setter
    def col_offset(self, value: int):
        assert value >= 0
        self._col_offset = value

    @property
    @abstractmethod
    def cols(self) -> int:
        pass

    @property
    @abstractmethod
    def rows(self) -> int:
        pass

    def cells(self) -> Generator[tuple[int, int], None, None]:
        for row in range(self.rows):
            for col in range(self.cols):
                yield (row, col)

    def _check_row_idx(self, row: int):
        if row < 0:
            raise TauluException("row number needs to be positive or zero")
        if row >= self.rows:
            raise TauluException(f"row number too high: {row} >= {self.rows}")

    def _check_col_idx(self, col: int):
        if col < 0:
            raise TauluException("col number needs to be positive or zero")
        if col >= self.cols:
            raise TauluException(f"col number too high: {col} >= {self.cols}")

    @abstractmethod
    def cell(self, point: tuple[float, float]) -> tuple[int, int]:
        """
        Returns the coordinate (row, col) of the cell that contains the given position

        Args:
            point (tuple[float, float]): a location in the input image

        Returns:
            tuple[int, int]: the cell index (row, col) that contains the given point
        """
        pass

    @abstractmethod
    def cell_polygon(
        self, cell: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """returns the polygon (used in e.g. opencv) that enscribes the cell at the given cell position"""
        pass

    def _highlight_cell(
        self,
        image: MatLike,
        cell: tuple[int, int],
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ):
        polygon = self.cell_polygon(cell)
        points = np.int32(list(polygon))  # type:ignore
        cv.polylines(image, [points], True, color, thickness, cv.LINE_AA)  # type:ignore
        cv.putText(
            image,
            str(cell),
            (int(polygon[3][0] + 10), int(polygon[3][1] - 10)),
            cv.FONT_HERSHEY_PLAIN,
            2.0,
            (255, 255, 255),
            2,
        )

    def highlight_all_cells(
        self,
        image: MatLike,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 1,
    ) -> MatLike:
        img = np.copy(image)

        for cell in self.cells():
            self._highlight_cell(img, cell, color, thickness)

        return img

    def select_one_cell(
        self,
        image: MatLike,
        window: str = WINDOW,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> tuple[int, int] | None:
        clicked = None

        def click_event(event, x, y, flags, params):
            nonlocal clicked

            img = np.copy(image)
            _ = flags
            _ = params
            if event == cv.EVENT_LBUTTONDOWN:
                cell = self.cell((x, y))
                if cell[0] >= 0:
                    clicked = cell
                else:
                    return
                self._highlight_cell(img, cell, color, thickness)
                cv.imshow(window, img)

        imu.show(image, click_event=click_event, title="select one cell", window=window)

        return clicked

    def show_cells(self, image: MatLike, window: str = WINDOW) -> list[tuple[int, int]]:
        img = np.copy(image)

        cells = []

        def click_event(event, x, y, flags, params):
            _ = flags
            _ = params
            if event == cv.EVENT_LBUTTONDOWN:
                cell = self.cell((x, y))
                if cell[0] >= 0:
                    cells.append(cell)
                else:
                    return
                self._highlight_cell(img, cell)
                cv.imshow(window, img)

        imu.show(
            img,
            click_event=click_event,
            title="click to highlight cells",
            window=window,
        )

        return cells

    @abstractmethod
    def region(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> tuple[Point, Point, Point, Point]:
        """
        Get the bounding box for the rectangular region that goes from start to end

        Returns:
            4 points: lt, rt, rb, lb, in format (x, y)
        """
        pass

    def crop_region(
        self,
        image: MatLike,
        start: tuple[int, int],
        end: tuple[int, int],
        margin: int = 0,
        margin_top: int | None = None,
        margin_bottom: int | None = None,
        margin_left: int | None = None,
        margin_right: int | None = None,
        margin_y: int | None = None,
        margin_x: int | None = None,
    ) -> MatLike:
        """Crop the input image to a rectangular region with the start and end cells as extremes"""

        region = self.region(start, end)

        lt, rt, rb, lb = _apply_margin(
            *region,
            margin=margin,
            margin_top=margin_top,
            margin_bottom=margin_bottom,
            margin_left=margin_left,
            margin_right=margin_right,
            margin_y=margin_y,
            margin_x=margin_x,
        )

        # apply margins according to priority:
        # margin_top > margin_y > margin (etc.)

        w = (rt[0] - lt[0] + rb[0] - lb[0]) / 2
        h = (rb[1] - rt[1] + lb[1] - lt[1]) / 2

        # crop by doing a perspective transform to the desired quad
        src_pts = np.array([lt, rt, rb, lb], dtype="float32")
        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv.warpPerspective(image, M, (int(w), int(h)))  # type:ignore

        return warped

    @abstractmethod
    def text_regions(
        self, img: MatLike, row: int, margin_x: int = 0, margin_y: int = 0
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Split the row into regions of continuous text

        Returns
            list[tuple[int, int]]: a list of spans (start col, end col)
        """

        pass

    def crop_cell(self, image, cell: tuple[int, int], margin: int = 0) -> MatLike:
        return self.crop_region(image, cell, cell, margin)
