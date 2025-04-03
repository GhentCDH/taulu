import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from . import img_util as imu
from .table_indexer import TableIndexer
from .header_template import Rule

REGION = 40

class CornerFilter:
    def __init__(self, kernel_size: int = 21, cross_width: int = 6, cross_height: int | None = None, morph_size: int | None = None):
        self._k = kernel_size
        self._w = cross_width
        self._h = cross_width if cross_height is None else cross_height
        self._m = morph_size if morph_size is not None else cross_width

    @property
    def _cross_kernel(self) -> NDArray:
        return self._cross_kernel_penalty(-1)

    def _cross_kernel_penalty(self, penalty: float = -1) -> NDArray:
        kernel = np.full((self._k, self._k), penalty, dtype=np.float32)

        # Define the center
        center = self._k // 2

        # Create horizontal and vertical bars of width y
        kernel[center - self._h // 2 : center + (self._h + 1) // 2, :] = 1  # Horizontal line
        kernel[:, center - self._w // 2 : center + (self._w + 1) // 2] = 1  # Vertical line
        
        return kernel

    @property
    def _dot_kernel(self) -> NDArray:
        penalty = -0.5
        dot = cv.filter2D(self._cross_kernel_penalty(penalty), -1, self._cross_kernel_penalty(penalty), borderType=cv.BORDER_CONSTANT, delta=0)
        dot = cv.normalize(dot, None, -0.3, 1, cv.NORM_MINMAX) #type:ignore
        
        return dot / abs(dot.sum())

    def apply(self, img: MatLike) -> MatLike:
        binary = imu.sauvola(img)

        # Define a horizontal kernel (adjust width as needed)
        kernel_hor = cv.getStructuringElement(cv.MORPH_RECT, (self._m, 1))
        kernel_ver = cv.getStructuringElement(cv.MORPH_RECT, (1, self._m))
        
        # Apply dilation
        dilated = cv.dilate(binary, kernel_hor, iterations=1)
        dilated = cv.dilate(dilated, kernel_ver, iterations=1)

        # apply cross kernel to find intersections
        closed = dilated.astype(np.float32)
        filtered = cv.filter2D(closed, -1, self._cross_kernel)
        filtered[filtered < 0] = 0 #type:ignore
        filtered *= 255 / filtered.max()

        # find the best matches to the cross kernel
        filtered = cv.filter2D(filtered, -1, self._dot_kernel)
        filtered *= 255 / filtered.max()
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)
        filtered[filtered < 90] = 0

        return filtered

    def find_nearest(self, filtered: MatLike, point: tuple[int, int], area: int = REGION, area_y: int | None = None) -> tuple[int, int]:

        if area_y is None:
            area_y = area
        # crop the filtered image around the given point

        y = point[0] - area_y // 2
        x = point[1] - area // 2

        cropped = filtered[y:y+area_y,x:x+area]

        best_match = np.argmax(cropped)
        best_match = np.unravel_index(best_match, cropped.shape)

        if cropped[best_match] < 80:
            return point

        result = (int(y + best_match[0]), int(x + best_match[1]))

        return result
        
    def find_table_points(self, img: MatLike, left_top: tuple[int, int], cell_widths: list[int], cell_height: int, filtered = None) -> "TableCrosses":
        """left_top is given in (y, x)"""
        if filtered is None:
            filtered = self.apply(img)

        left_top = self.find_nearest(filtered, left_top, 100)

        points: list[list[tuple[int, int]]] = []
        current = left_top
        row = [current]

        while True:
            while len(row) <= len(cell_widths):
                jump = cell_widths[len(row) - 1]

                match = self.find_nearest(filtered, (current[0], current[1] + jump), 60, 40)

                current = match
                row.append(current)
            
            points.append(row)

            current = (row[0][0]+cell_height, row[0][1])

            if current[0] > filtered.shape[0]:
                break

            current = self.find_nearest(filtered, current)
            row = [current]

        # reverse the order of the points
        points = [[(p[1], p[0]) for p in row] for row in points]

        return TableCrosses(points)

class TableCrosses(TableIndexer):
    """
    points are in (x, y) order
    """
    def __init__(self, points: list[list[tuple[int,int]]]):
        self._points = points

    @property
    def points(self) -> list[list[tuple[int, int]]]:
        return self._points

    def row(self, i: int) -> list[tuple[int, int]]:
        assert 0 <= i and i < len(self._points)
        return self._points[i]

    @property
    def cols(self) -> int:
        return len(self.row(0)) - 1

    @property
    def rows(self) -> int:
        return len(self._points) - 1

    def add_left_col(self, width: int):
        for row in self._points:
            first = row[0]
            new_first = (first[0] - width, first[1])
            row.insert(0, new_first)

    def _surrounds(self, rect: list[tuple[int, int]], point: tuple[float, float]) -> bool:
        """point: x, y"""
        lt, rt, rb, lb = rect
        x, y = point

        top = Rule(*lt, *rt)
        if top._y_at_x(x) > y:
            return False

        right = Rule(*rt, *rb)
        if right._x_at_y(y) < x:
            return False

        bottom = Rule(*lb, *rb)
        if bottom._y_at_x(x) < y:
            return False

        left = Rule(*lb, *lt)
        if left._x_at_y(y) > x:
            return False

        return True

    def cell(self, point: tuple[float, float]) -> tuple[int, int]:
        for r in range(len(self._points) - 1):
            for c in range(len(self.row(0)) - 1):
                if self._surrounds([self._points[r][c],self._points[r][c+1],self._points[r+1][c+1],self._points[r+1][c]], point):
                    return (r, c)

        return (-1, -1)

    def cell_polygon(self, cell: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        r, c = cell

        self._check_row_idx(r)
        self._check_col_idx(c)

        return self._points[r][c], self._points[r][c+1], self._points[r+1][c+1], self._points[r+1][c]

    def crop_region(self, image, start: tuple[int, int], end: tuple[int, int], margin: int = 0) -> MatLike:
        _ = margin
        r0, c0 = start
        r1, c1 = end
                
        self._check_row_idx(r0)
        self._check_row_idx(r1)
        self._check_col_idx(c0)
        self._check_col_idx(c1)

        lt = self._points[r0][c0]
        rt = self._points[r0][c1+1]
        rb = self._points[r1+1][c1+1]
        lb = self._points[r1+1][c0]

        w = (rt[0] - lt[0] + rb[0] - lb[0]) / 2 
        h = (rb[1] - rt[1] + lb[1] - lt[1]) / 2

        # crop by doing a perspective transform to the desired quad
        src_pts = np.array(
            [lt, rt, rb, lb], dtype="float32")
        dst_pts = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype="float32")
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv.warpPerspective(image, M, (int(w), int(h))) #type:ignore

        return warped

    def text_regions(self, img: MatLike, row: int, margin_x: int = 10, margin_y: int = -3) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        def vertical_rule_crop(row: int, col: int): 
            self._check_col_idx(col)
            self._check_row_idx(row)

            top = self._points[row][col]
            bottom = self._points[row+1][col]

            left = int(min(top[0], bottom[0]))
            right = int(max(top[0], bottom[0]))

            return img[int(top[1]) - margin_y : int(bottom[1]) + margin_y, left - margin_x : right + margin_x]

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
