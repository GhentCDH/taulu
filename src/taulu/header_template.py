from pathlib import Path
from typing import Iterable, Optional, cast
import csv
import json
import numpy as np

import math

import cv2 as cv
from cv2.typing import MatLike
from .table_indexer import TableIndexer

from .error import TauluException
from . import img_util as imu
from . import constants

# angle tolerance for horizontal or vertical clasification (radians)
TOLERANCE = math.pi / 6


class _Rule:
    def __init__(
        self, x0: int, y0: int, x1: int, y1: int, tolerance: float = TOLERANCE
    ):
        """
        Two points define a rule in a table

        Args:
            x0, y0, x1, y1 (int): the two points that make define this rule
            tolerance (float, optional): tolerance for defining lines as "horizontal or vertical"
                a rule is horizontal when its angle is between -tolerance and +tolerance, and
                similar for vertical
        """

        self._p0: tuple[int, int] = (x0, y0)
        self._p1: tuple[int, int] = (x1, y1)
        self._tolerance = tolerance

        y_diff: float = self._p1[1] - self._p0[1]
        x_diff = self._p1[0] - self._p0[0]
        if x_diff == 0:
            self._slope = 1000000000.0
        else:
            self._slope = y_diff / x_diff

    def to_dict(self) -> dict[str, int]:
        return {
            "x0": self._p0[0],
            "y0": self._p0[1],
            "x1": self._p1[0],
            "y1": self._p1[1],
        }

    @staticmethod
    def from_dict(value: dict, tolerance: float = TOLERANCE) -> "_Rule":
        return _Rule(value["x0"], value["y0"], value["x1"], value["y1"], tolerance)

    @property
    def _angle(self) -> float:
        """
        angle of the line in radians, -pi/2 <= angle <= pi/2
        """

        y_diff: float = self._p1[1] - self._p0[1]
        x_diff = self._p1[0] - self._p0[0]

        if x_diff == 0:
            return (1 if y_diff >= 0 else -1) * math.pi / 2

        return math.atan(y_diff / x_diff)

    @property
    def _x(self) -> float:
        """the x value of the center of the line"""
        return (self._p0[0] + self._p1[0]) / 2

    @property
    def _y(self) -> float:
        """the y value of the center of the line"""
        return (self._p0[1] + self._p1[1]) / 2

    def _is_horizontal(self) -> bool:
        angle = self._angle
        return -self._tolerance <= angle and angle <= self._tolerance

    def _is_vertical(self) -> bool:
        angle = self._angle
        return (
            angle <= -math.pi / 2 + self._tolerance
            or angle >= math.pi / 2 - self._tolerance
        )

    def _y_at_x(self, x: float) -> float:
        """Calculates y value at given x."""
        return self._p0[1] + self._slope * (x - self._p0[0])

    def _x_at_y(self, y: float) -> float:
        """Calculates x value at given y."""
        if self._slope == 0:
            # not accurate but doesn't matter for this usecase
            return self._p0[0]
        return self._p0[0] + (y - self._p0[1]) / self._slope

    def intersection(self, other: "_Rule") -> Optional[tuple[float, float]]:
        """Calculates the intersection point of two lines."""
        if self._slope == other._slope:
            return None  # Parallel lines

        x = (
            other._p0[1]
            - self._p0[1]
            + self._slope * self._p0[0]
            - other._slope * other._p0[0]
        ) / (self._slope - other._slope)
        y = self._y_at_x(x)

        return (x, y)


class HeaderTemplate(TableIndexer):
    def __init__(self, rules: Iterable[Iterable[int]]):
        """
        A TableTemplate is a collection of rules of a table. This class implements methods
        for finding cell positions in a table image, given the template the image adheres to.

        Args:
            rules: 2D array of lines, where each line is represented as [x0, y0, x1, y1]
        """

        super().__init__()
        self._rules = [_Rule(*rule) for rule in rules]
        self._h_rules = sorted(
            [rule for rule in self._rules if rule._is_horizontal()], key=lambda r: r._y
        )
        self._v_rules = sorted(
            [rule for rule in self._rules if rule._is_vertical()], key=lambda r: r._x
        )

    def save(self, path: Path):
        """
        Save the HeaderTemplate to the given path, as a json
        """

        data = {"rules": [r.to_dict() for r in self._rules]}

        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def from_saved(path: str) -> "HeaderTemplate":
        with open(path, "r") as f:
            data = json.load(f)
            rules = data["rules"]
            rules = [[r["x0"], r["y0"], r["x1"], r["y1"]] for r in rules]

            return HeaderTemplate(rules)

    @property
    def cols(self) -> int:
        return len(self._v_rules) - 1

    @property
    def rows(self) -> int:
        return len(self._h_rules) - 1

    @staticmethod
    def annotate_image(template: MatLike | str) -> "HeaderTemplate":
        """
        Utility method that allows users to create a template form a template image.

        The user is asked to click to annotate lines (two clicks per line).
        """

        if type(template) is str:
            value = cv.imread(template)
            template = value
        template = cast(MatLike, template)

        start_point = None
        lines: list[list[int]] = []

        def get_point(event, x, y, flags, params):
            nonlocal lines, start_point
            _ = flags
            _ = params
            if event == cv.EVENT_LBUTTONDOWN:
                if start_point is not None:
                    line: list[int] = [start_point[1], start_point[0], x, y]

                    cv.line(
                        template,
                        (start_point[1], start_point[0]),
                        (x, y),
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )
                    cv.imshow(constants.WINDOW, template)

                    lines.append(line)
                    start_point = None
                else:
                    start_point = (y, x)

        imu.show(template, get_point, title="annotate the lines on the template")

        return HeaderTemplate(lines)

    @staticmethod
    def from_vgg_annotation(annotation: str) -> "HeaderTemplate":
        """
        Create a TableTemplate from annotations made in [vgg](https://annotate.officialstatistics.org/), using the polylines tool.

        Args:
            annotation (str): the path of the annotation csv file
        """

        rules = []
        with open(annotation, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                shape_attributes = json.loads(row["region_shape_attributes"])
                if shape_attributes["name"] == "polyline":
                    x_points = shape_attributes["all_points_x"]
                    y_points = shape_attributes["all_points_y"]
                    if len(x_points) == 2 and len(y_points) == 2:
                        rules.append(
                            [x_points[0], y_points[0], x_points[1], y_points[1]]
                        )

        return HeaderTemplate(rules)

    def cell_width(self, i: int) -> int:
        self._check_col_idx(i)
        return int(self._v_rules[i + 1]._x - self._v_rules[i]._x)

    def cell_widths(self, start: int = 0) -> list[int]:
        return [self.cell_width(i) for i in range(start, self.cols)]

    def cell_height(self, header_factor: float = 0.8) -> int:
        return int((self._h_rules[1]._y - self._h_rules[0]._y) * header_factor)

    def intersection(self, index: tuple[int, int]) -> tuple[float, float]:
        """
        Returns the interaction of the index[0]th horizontal rule and the
        index[1]th vertical rule
        """

        ints = self._h_rules[index[0]].intersection(self._v_rules[index[1]])
        assert ints is not None
        return ints

    def cell(self, point: tuple[float, float]) -> tuple[int, int]:
        """
        Get the cell index (row, col) that corresponds with the point (x, y) in the template image

        Args:
            point (tuple[float, float]): the coordinates in the template image

        Returns:
            tuple[int, int]: (row, col)
        """

        x, y = point

        row = -1
        col = -1

        for i in range(self.rows):
            y0 = self._h_rules[i]._y_at_x(x)
            y1 = self._h_rules[i + 1]._y_at_x(x)
            if min(y0, y1) <= y <= max(y0, y1):
                row = i
                break

        for i in range(self.cols):
            x0 = self._v_rules[i]._x_at_y(y)
            x1 = self._v_rules[i + 1]._x_at_y(y)
            if min(x0, x1) <= x <= max(x0, x1):
                col = i
                break

        if row == -1 or col == -1:
            return (-1, -1)

        return (row, col)

    def cell_polygon(
        self, cell: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        Return points (x,y) that make up a polygon around the requested cell
        (top left, top right, bottom right, bottom left)
        """

        row, col = cell

        self._check_col_idx(col)
        self._check_row_idx(row)

        top_rule = self._h_rules[row]
        bottom_rule = self._h_rules[row + 1]
        left_rule = self._v_rules[col]
        right_rule = self._v_rules[col + 1]

        # Calculate corner points using intersections
        top_left = top_rule.intersection(left_rule)
        top_right = top_rule.intersection(right_rule)
        bottom_left = bottom_rule.intersection(left_rule)
        bottom_right = bottom_rule.intersection(right_rule)

        if not all(
            [
                point is not None
                for point in [top_left, top_right, bottom_left, bottom_right]
            ]
        ):
            raise TauluException("the lines around this cell do not intersect")

        return top_left, top_right, bottom_right, bottom_left  # type:ignore

    def crop_region(
        self, image, start: tuple[int, int], end: tuple[int, int], margin: int = 0
    ) -> MatLike:
        self._check_row_idx(start[0])
        self._check_row_idx(end[0])
        self._check_col_idx(start[1])
        self._check_col_idx(end[1])

        # the rules that surround this row
        top_rule = self._h_rules[start[0]]
        bottom_rule = self._h_rules[end[0] + 1]
        left_rule = self._v_rules[start[1]]
        right_rule = self._v_rules[end[1] + 1]

        # four points that will be the bounding polygon of the result,
        # which needs to be rectified
        top_left = top_rule.intersection(left_rule)
        top_right = top_rule.intersection(right_rule)
        bottom_left = bottom_rule.intersection(left_rule)
        bottom_right = bottom_rule.intersection(right_rule)

        if (
            top_left is None
            or top_right is None
            or bottom_left is None
            or bottom_right is None
        ):
            raise TauluException("the lines around this row do not intersect properly")

        top_left = cast(tuple[float, float], top_left)
        bottom_left = cast(tuple[float, float], bottom_left)
        top_right = cast(tuple[float, float], top_right)
        bottom_right = cast(tuple[float, float], bottom_right)

        top_left = (top_left[0] - margin, top_left[1] - margin)
        bottom_left = (bottom_left[0] - margin, bottom_left[1] + margin)
        top_right = (top_right[0] + margin, top_right[1] - margin)
        bottom_right = (bottom_right[0] + margin, bottom_right[1] + margin)

        w = (top_right[0] - top_left[0] + bottom_right[0] - bottom_left[0]) / 2
        h = -(top_right[1] - bottom_right[1] + top_left[1] - bottom_left[1]) / 2

        # crop by doing a perspective transform to the desired quad
        src_pts = np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype="float32"
        )
        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv.warpPerspective(image, M, (int(w), int(h)))  # type:ignore

        return warped

    def text_regions(
        self, img: MatLike, row: int, margin_x: int = 10, margin_y: int = -20
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        raise TauluException("text_regions should not be called on a HeaderTemplate")
