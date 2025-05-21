import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from pathlib import Path
import json

from . import img_util as imu
from .constants import WINDOW
from .table_indexer import Point, TableIndexer
from .header_template import _Rule
from .split import Split
from .error import TauluException


class BTreeNode:
    def __init__(self, value: float, point: Point):
        self.value = value
        self.point = point
        self.naive: None | BTreeNode = None
        self.match: None | BTreeNode = None

    def score(self) -> float:
        """Get the score of this node (the maximum sum of all of its paths)"""
        naive_score = self.naive.score() if self.naive is not None else 0
        match_score = self.match.score() if self.match is not None else 0

        return max(naive_score, match_score) + self.value

    # def

    def leaves(self) -> list["BTreeNode"]:
        if self.naive is None or self.match is None:
            return [self]
        else:
            return self.naive.leaves() + self.match.leaves()

    def best(self) -> Point:
        if self.naive is None or self.match is None:
            raise TauluException("shouldn't call best on an uninitialised tree node")

        if self.naive.score() > self.match.score():
            return self.naive.point
        else:
            return self.match.point

    def print(self, indent: int = 0):
        print(
            "  " * indent
            + f"Value: {self.value}, Point: {self.point}, Score: {self.score()}"
        )
        if self.naive:
            print("  " * (indent + 1) + "Naive:")
            self.naive.print(indent + 2)
        if self.match:
            print("  " * (indent + 1) + "Match:")
            self.match.print(indent + 2)


class GridDetector:
    """
    Implements filters that show high activation where the image has an intersection of a vertical
    and horizontal rule, useful for finding the bounding boxes of cells
    """

    def __init__(
        self,
        kernel_size: int = 21,
        cross_width: int = 6,
        cross_height: int | None = None,
        morph_size: int | None = None,
        region: int = 40,
        k: float = 0.04,
        w: int = 15,
        distance_penalty: float = 0.4,
    ):
        """
        Args:
            kernel_size (int): the size of the cross kernel
                a larger kernel size often means that more penalty is applied, often leading
                to more sparse results
            cross_width (int): the width of one of the edges in the cross filter, should be
                roughly equal to the width of the rules in the image after morphology is applied
            cross_height (int | None (default)): useful if the horizontal rules and vertical rules
                have different sizes
            morph_size (int | None (default)): the size of the morphology operators that are applied before
                the cross kernel. 'bridges the gaps' of broken-up lines
            region (int): area in which to search for a new max value in `find_nearest` etc.
            k (float): threshold parameter for sauvola thresholding
            w (int): window_size parameter for sauvola thresholding
            distance_penalty (float): how much the point finding algorithm penalizes points that are further in the region [0, 1]
        """
        assert (
            kernel_size % 2 == 1
        ), "GridDetector: kernel size (k) needs to be ann odd number"

        self._k = kernel_size
        self._w = cross_width
        self._h = cross_width if cross_height is None else cross_height
        self._m = morph_size if morph_size is not None else cross_width
        self._region = region
        self._k_thresh = k
        self._w_thresh = w
        self._cross_kernel = self._cross_kernel_uint8()
        self._distance_penalty = distance_penalty

    def _gaussian_weights(self, region: int):
        """
        Create a 2D Gaussian weight mask.

        Args:
            shape (tuple[int, int]): Shape of the region (height, width)
            p (float): Minimum value at the edge = 1 - p

        Returns:
            NDArray: Gaussian weight mask
        """
        h, w = region, region
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        xv, yv = np.meshgrid(x, y)
        dist_squared = xv**2 + yv**2

        # Scale so that center is 1 and edges are 1 - p
        sigma = np.sqrt(
            -1 / (2 * np.log(1 - self._distance_penalty))
        )  # solves exp(-r^2 / (2 * sigma^2)) = 1 - p for r=1
        weights = np.exp(-dist_squared / (2 * sigma**2))

        return weights

    def _cross_kernel_uint8(self) -> NDArray:
        kernel = np.zeros((self._k, self._k), dtype=np.uint8)

        # Define the center
        center = self._k // 2

        # Create horizontal and vertical bars of width y
        kernel[center - self._h // 2 : center + (self._h + 1) // 2, :] = (
            255  # Horizontal line
        )
        kernel[:, center - self._w // 2 : center + (self._w + 1) // 2] = (
            255  # Vertical line
        )

        return kernel

    def apply(self, img: MatLike, visual: bool = False) -> MatLike:
        binary = imu.sauvola(img, k=self._k_thresh, window_size=self._w_thresh)

        if visual:
            imu.show(binary, title="thresholded")

        # Define a horizontal kernel (adjust width as needed)
        kernel_hor = cv.getStructuringElement(cv.MORPH_RECT, (self._m, 1))
        kernel_ver = cv.getStructuringElement(cv.MORPH_RECT, (1, self._m))

        # Apply dilation
        dilated = cv.dilate(binary, kernel_hor, iterations=1)
        dilated = cv.dilate(dilated, kernel_ver, iterations=1)

        if visual:
            imu.show(dilated, title="dilated")

        pad_y = self._cross_kernel.shape[0] // 2
        pad_x = self._cross_kernel.shape[1] // 2

        padded = cv.copyMakeBorder(
            dilated,
            pad_y,
            pad_y,  # top, bottom
            pad_x,
            pad_x,  # left, right
            borderType=cv.BORDER_CONSTANT,
            value=[0, 0, 0],  # black padding (BGR)
        )

        filtered = cv.matchTemplate(padded, self._cross_kernel, cv.TM_SQDIFF_NORMED)
        filtered = 255 - cv.normalize(
            filtered, None, 0, 255, cv.NORM_MINMAX
        ).astype(  # type:ignore
            np.uint8
        )

        return filtered

    def find_nearest(
        self, filtered: MatLike, point: Point, region: None | int = None
    ) -> tuple[Point, float]:
        """
        Find the nearest 'corner match' in the image, along with its score [0,1]

        Args:
            filtered (MatLike): the filtered image (obtained through `apply`)
            point (tuple[int, int]): the approximate target point (x, y)
            region (None (default) | int): alternative value for search region,
                overwriting the `__init__` parameter `region`
        """

        if region is None:
            region = self._region

        x = point[0] - region // 2
        y = point[1] - region // 2

        cropped = imu.safe_crop(filtered, x, y, region, region)

        if cropped.shape != (region, region):
            return point, 1.0

        weighted = cropped * self._gaussian_weights(region)

        best_match = np.argmax(weighted)
        best_match = np.unravel_index(best_match, cropped.shape)

        result = (
            int(x + best_match[1]),
            int(y + best_match[0]),
        )

        return result, weighted[best_match]

    def find_table_points(
        self,
        img: MatLike,
        left_top: Point,
        cell_widths: list[int],
        cell_height: int,
        visual: bool = False,
        window: str = WINDOW,
    ) -> "TableGrid":
        """
        Parse the image to a `TableGrid` structure that holds all of the
        intersections between horizontal and vertical rules, starting near the `left_top` point

        Args:
            img (MatLike): the input image of a table
            left_top (tuple[int, int]): the starting point of the algorithm
            cell_widths (list[int]): the expected widths of the cells (based on a header template)
            cell_height (int): the expected height of one row of data

        Returns:
            a TableGrid object
        """

        filtered = self.apply(img, visual)
        if visual:
            imu.show(filtered, window=window)

        left_top, _ = self.find_nearest(filtered, left_top, int(self._region * 3 / 2))

        points: list[list[Point]] = []
        current = left_top
        row = [current]

        N = 3

        while True:
            while len(row) <= len(cell_widths):
                jumps = cell_widths[len(row) - 1 : len(row) - 1 + N]

                if len(points) == 0:
                    previous_row_x = [current[0] + j for j in jumps]
                else:
                    idx = len(row)
                    previous_row_x = [p[0] for p in points[-1][idx : idx + N]]

                tree = self._grow_tree(filtered, current, jumps, previous_row_x)
                current = tree.best()
                row.append(current)

            points.append(row)

            current = (row[0][0], row[0][1] + cell_height)

            if current[1] > filtered.shape[0]:
                break

            current, _ = self.find_nearest(filtered, current)
            row = [current]

        return TableGrid(points)

    def _grow_tree(
        self,
        filtered: MatLike,
        start_point: Point,
        cell_widths: list[int],
        previous_row_x: list[int],
    ) -> BTreeNode:
        """
        Grow a search tree from the starting point and with given depth
        """

        tree = BTreeNode(0, start_point)

        def grow_right(tree: BTreeNode, jump: int, previous_x: int):
            for leaf in tree.leaves():
                naive_target = (leaf.point[0] + jump, leaf.point[1])

                x, y = naive_target

                if y >= len(filtered):
                    y = len(filtered) - 1

                if x >= len(filtered[y]):
                    x = len(filtered[y]) - 1

                naive_target = (x, y)

                naive_target = (previous_x, leaf.point[1])

                match, match_score = self.find_nearest(filtered, naive_target)

                naive_node = BTreeNode(filtered[y][x], naive_target)
                match_node = BTreeNode(match_score, match)

                leaf.naive = naive_node
                leaf.match = match_node

        for jump, previous_x in zip(cell_widths, previous_row_x):
            grow_right(tree, jump, previous_x)

        return tree


class TableGrid(TableIndexer):
    """
    A data class that allows segmenting the image into cells
    """

    _right_offset: int | None = None

    def __init__(self, points: list[list[Point]]):
        """
        Args:
            points: a 2D list of intersections between hor. and vert. rules
        """
        self._points = points

    @property
    def points(self) -> list[list[Point]]:
        return self._points

    def row(self, i: int) -> list[Point]:
        assert 0 <= i and i < len(self._points)
        return self._points[i]

    @property
    def cols(self) -> int:
        if self._right_offset is not None:
            return len(self.row(0)) - 2
        else:
            return len(self.row(0)) - 1

    @property
    def rows(self) -> int:
        return len(self._points) - 1

    @staticmethod
    def from_split(
        split_grids: Split["TableGrid"], offsets: Split[Point]
    ) -> "TableGrid":
        """
        Convert two ``TableGrid`` objects into one, that is able to segment the original (non-cropped) image

        Args:
            split_grids (Split[TableGrid]): a Split of TableGrid objects of the left and right part of the table
            offsets (Split[tuple[int, int]]): a Split of the offsets in the image where the crop happened
        """

        def offset_points(points, offset):
            return [
                [(p[0] + offset[0], p[1] + offset[1]) for p in row] for row in points
            ]

        split_points = split_grids.apply(
            lambda grid, offset: offset_points(grid.points, offset), offsets
        )

        points = []

        rows = min(split_grids.left.rows, split_grids.right.rows)

        for row in range(rows + 1):
            row_points = []

            row_points.extend(split_points.left[row])
            row_points.extend(split_points.right[row])

            points.append(row_points)

        table_grid = TableGrid(points)
        table_grid._right_offset = split_grids.left.cols

        return table_grid

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump({"points": self.points}, f)

    @staticmethod
    def from_saved(path: str | Path) -> "TableGrid":
        with open(path, "r") as f:
            points = json.load(f)
            points = [[(p[0], p[1]) for p in pointes] for pointes in points["points"]]
            return TableGrid(points)

    def add_left_col(self, width: int):
        for row in self._points:
            first = row[0]
            new_first = (first[0] - width, first[1])
            row.insert(0, new_first)

    def add_top_row(self, height: int):
        new_row = []
        for point in self._points[0]:
            new_row.append((point[0], point[1] - height))

        self.points.insert(0, new_row)

    def _surrounds(self, rect: list[Point], point: tuple[float, float]) -> bool:
        """point: x, y"""
        lt, rt, rb, lb = rect
        x, y = point

        top = _Rule(*lt, *rt)
        if top._y_at_x(x) > y:
            return False

        right = _Rule(*rt, *rb)
        if right._x_at_y(y) < x:
            return False

        bottom = _Rule(*lb, *rb)
        if bottom._y_at_x(x) < y:
            return False

        left = _Rule(*lb, *lt)
        if left._x_at_y(y) > x:
            return False

        return True

    def cell(self, point: tuple[float, float]) -> tuple[int, int]:
        for r in range(len(self._points) - 1):
            offset = 0
            for c in range(len(self.row(0)) - 1):
                if self._right_offset is not None and c == self._right_offset:
                    offset = -1
                    continue

                if self._surrounds(
                    [
                        self._points[r][c],
                        self._points[r][c + 1],
                        self._points[r + 1][c + 1],
                        self._points[r + 1][c],
                    ],
                    point,
                ):
                    return (r, c + offset)

        return (-1, -1)

    def cell_polygon(self, cell: tuple[int, int]) -> tuple[Point, Point, Point, Point]:
        r, c = cell

        self._check_row_idx(r)
        self._check_col_idx(c)

        if self._right_offset is not None and c >= self._right_offset:
            c = c + 1

        return (
            self._points[r][c],
            self._points[r][c + 1],
            self._points[r + 1][c + 1],
            self._points[r + 1][c],
        )

    def region(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[Point, Point, Point, Point]:
        r0, c0 = start
        r1, c1 = end

        self._check_row_idx(r0)
        self._check_row_idx(r1)
        self._check_col_idx(c0)
        self._check_col_idx(c1)

        if self._right_offset is not None and c0 >= self._right_offset:
            c0 = c0 + 1

        if self._right_offset is not None and c1 >= self._right_offset:
            c1 = c1 + 1

        lt = self._points[r0][c0]
        rt = self._points[r0][c1 + 1]
        rb = self._points[r1 + 1][c1 + 1]
        lb = self._points[r1 + 1][c0]

        return lt, rt, rb, lb

    def visualize_points(self, img: MatLike):
        """
        Draw the detected table points on the image for visual verification
        """
        import colorsys

        def clr(index, total_steps):
            hue = index / total_steps  # Normalized hue between 0 and 1
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            return int(r * 255), int(g * 255), int(b * 255)

        for i, row in enumerate(self._points):
            for p in row:
                cv.circle(img, p, 4, clr(i, len(self._points)), -1)

        imu.show(img)

    def text_regions(
        self, img: MatLike, row: int, margin_x: int = 10, margin_y: int = -3
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        def vertical_rule_crop(row: int, col: int):
            self._check_col_idx(col)
            self._check_row_idx(row)

            if self._right_offset is not None and col >= self._right_offset:
                col = col + 1

            top = self._points[row][col]
            bottom = self._points[row + 1][col]

            left = int(min(top[0], bottom[0]))
            right = int(max(top[0], bottom[0]))

            return img[
                int(top[1]) - margin_y : int(bottom[1]) + margin_y,
                left - margin_x : right + margin_x,
            ]

        result = []

        start = None
        for col in range(self.cols):
            crop = vertical_rule_crop(row, col)
            text_over_score = imu.text_presence_score(crop)
            text_over = text_over_score > -0.10

            if not text_over:
                if start is not None:
                    result.append(((row, start), (row, col - 1)))
                start = col

        if start is not None:
            result.append(((row, start), (row, self.cols - 1)))

        return result
