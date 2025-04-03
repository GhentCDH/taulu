from abc import ABC, abstractmethod

import cv2 as cv
from cv2.typing import MatLike
import numpy as np

from . import img_util as imu
from .constants import WINDOW
from .error import TabularException

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

    def _check_row_idx(self, row: int):
        if row < 0:
            raise TabularException("row number needs to be positive or zero")
        if row >= self.rows:
            raise TabularException(
                f"row number too high: {row} >= {self.rows}")

    def _check_col_idx(self, col: int):
        if col < 0:
            raise TabularException("col number needs to be positive or zero")
        if col >= self.cols:
            raise TabularException(
                f"col number too high: {col} >= {self.cols}")

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
    def cell_polygon(self, cell: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """returns the polygon (used in e.g. opencv) that enscribes the cell at the given cell position"""
        pass

    def _highlight_cell(self, image: MatLike, cell: tuple[int, int]):
        polygon = self.cell_polygon(cell)
        points = np.int32(list(polygon)) #type:ignore
        cv.polylines(image, [points], True, (0,0,255), 2, cv.LINE_AA) #type:ignore
        cv.putText(image, str(cell), (int(polygon[3][0] + 10), int(polygon[3][1] - 10)), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)

    def show_cells(self, image: MatLike) -> list[tuple[int, int]]:
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
                print(f"clicked cell: {cell}")
                self._highlight_cell(img, cell)
                cv.imshow(WINDOW, img)

        imu.show(img, click_event, "click to highlight cells")

        return cells

    @abstractmethod
    def crop_region(self, image, start: tuple[int, int], end: tuple[int, int], margin: int = 0) -> MatLike:
        """Crop the input image to a rectangular region with the start and end cells as extremes"""
        pass

    
    @abstractmethod
    def text_regions(self, img: MatLike, row: int, margin_x: int = 0, margin_y: int = 0) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Split the row into regions of continuous text

        Returns
            list[tuple[int, int]]: a list of spans (start col, end col)
        """

        pass
