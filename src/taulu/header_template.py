"""
A HeaderTemplate defines the structure of a table header.
"""

import csv
import json
import logging
import math
import os
from collections.abc import Iterable
from os import PathLike
from typing import Optional, cast

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike

from taulu.decorators import log_calls

from . import constants
from . import img_util as imu
from .error import TauluException
from .table_indexer import Point, TableIndexer

logger = logging.getLogger(__name__)

# angle tolerance for horizontal or vertical clasification (radians)
TOLERANCE = math.pi / 6

ANNO_HELP = """Start annotating the header

\x1b[1;32mTo draw a line, click on the image twice (the line will be drawn between your two clicks).

To undo the last line you drew, right-click anywhere on the image (you cannot redo the undo).

You should annotate all of the vertical lines that extend into the table body, as well as the top and bottom horizontal lines.

\x1b[31mPress 'n' when you are done.
\x1b[0m
"""

CROP_HELP = """Crop the template to just the header

\x1b[1;32mClick just outside the four corners of the header, such that the entire header is contained within the rectangle you define.
\x1b[31mPress 'n' when you are done.
\x1b[0m
"""


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

    def intersection(self, other: "_Rule") -> tuple[float, float] | None:
        """Calculates the intersection point of two lines."""
        if self._slope == other._slope:
            logger.warning("trying to intersect parallel lines")
            return None  # Parallel lines

        x = (
            other._p0[1]
            - self._p0[1]
            + self._slope * self._p0[0]
            - other._slope * other._p0[0]
        ) / (self._slope - other._slope)
        y = self._y_at_x(x)

        return (x, y)


class AnnotationSession:
    """
    Session object for notebook-based annotation.

    In Jupyter notebooks with %matplotlib widget, plt.show() is non-blocking.
    This session object holds the result once the user clicks "Done".

    Usage:
        session = HeaderTemplate.annotate_image_notebook("image.png")
        # ... interact with the plot, click Done ...
        template = session.result  # Access the HeaderTemplate after clicking Done
    """

    def __init__(self):
        self._result: HeaderTemplate | None = None
        self._save_path: PathLike[str] | None = None
        self._crop_path: PathLike[str] | None = None
        self._margin: int = 10
        self._original_template: MatLike | None = None

    @property
    def result(self) -> Optional["HeaderTemplate"]:
        """Returns the HeaderTemplate once Done is clicked, or None if not yet done."""
        return self._result

    @property
    def is_done(self) -> bool:
        """Returns True if the user has clicked Done."""
        return self._result is not None

    def save(self, path: PathLike[str]):
        """Save the result to a JSON file. Raises if not done yet."""
        if self._result is None:
            raise TauluException(
                "Cannot save: annotation not complete. Click 'Done' first."
            )
        self._result.save(path)


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

    @log_calls(level=logging.DEBUG)
    def save(self, path: PathLike[str]):
        """
        Save the HeaderTemplate to the given path, as a json
        """

        data = {"rules": [r.to_dict() for r in self._rules]}

        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    @log_calls(level=logging.DEBUG)
    def from_saved(path: PathLike[str]) -> "HeaderTemplate":
        with open(path) as f:
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
    @log_calls(level=logging.DEBUG)
    def annotate_image(
        template: MatLike | str, crop: PathLike[str] | None = None, margin: int = 10
    ) -> "HeaderTemplate":
        """
        Utility method that allows users to create a template form a template image.

        The user is asked to click to annotate lines (two clicks per line).

        Args:
            template: the image on which to annotate the header lines
            crop (str | None): if str, crop the template image first, then do the annotation.
                The cropped image will be stored at the supplied path
            margin (int): margin to add around the cropping of the header
        """

        if type(template) is str:
            value = cv.imread(template)
            template = value
        template = cast(MatLike, template)

        if crop is not None:
            cropped = HeaderTemplate._crop(template, margin)
            cv.imwrite(os.fspath(crop), cropped)
            template = cropped

        start_point = None
        lines: list[list[int]] = []

        anno_template = np.copy(template)

        def get_point(event, x, y, flags, params):
            nonlocal lines, start_point, anno_template
            _ = flags
            _ = params
            if event == cv.EVENT_LBUTTONDOWN:
                if start_point is not None:
                    line: list[int] = [start_point[1], start_point[0], x, y]

                    cv.line(  # type:ignore
                        anno_template,  # type:ignore
                        (start_point[1], start_point[0]),
                        (x, y),
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )
                    cv.imshow(constants.WINDOW, anno_template)  # type:ignore

                    lines.append(line)
                    start_point = None
                else:
                    start_point = (y, x)
            elif event == cv.EVENT_RBUTTONDOWN:
                start_point = None

                # remove the last annotation
                lines = lines[:-1]

                anno_template = np.copy(anno_template)

                for line in lines:
                    cv.line(
                        template,
                        (line[0], line[1]),
                        (line[2], line[3]),
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )

                cv.imshow(constants.WINDOW, template)

        print(ANNO_HELP)

        imu.show(anno_template, get_point, title="annotate the header")

        return HeaderTemplate(lines)

    @staticmethod
    @log_calls(level=logging.DEBUG)
    def annotate_image_notebook(
        template: MatLike | str, crop: PathLike[str] | None = None, margin: int = 10
    ) -> "AnnotationSession":
        """
        Notebook-compatible version of annotate_image. Returns an AnnotationSession immediately.
        Interact with the widget and click Done to finalize.
        Access the result via session.result after clicking Done.

        Args:
            template: the image on which to annotate the header lines
            crop (str | None): if str, crop the template image first, then do the annotation.
                The cropped image will be stored at the supplied path
            margin (int): margin to add around the cropping of the header

        Returns:
            AnnotationSession: access .result after clicking Done to get the HeaderTemplate.
        """
        if isinstance(template, str):
            tmp = cv.imread(template)
            assert tmp is not None
            template = tmp

        session = AnnotationSession()
        session._crop_path = crop
        session._margin = margin
        session._original_template = template

        if crop is not None:
            # First show crop UI, then annotation UI
            HeaderTemplate._crop_notebook(template, margin, session)
        else:
            # Go directly to annotation
            HeaderTemplate._show_annotation_ui(template, session)

        return session

    @staticmethod
    def _crop_notebook(template: MatLike, margin: int, session: "AnnotationSession"):
        """Notebook-compatible crop UI using matplotlib + ipywidgets."""
        import ipywidgets as widgets
        from IPython.display import display

        display_img = cv.cvtColor(template, cv.COLOR_BGR2RGB)

        points: list[tuple[int, int]] = []
        drawn_points: list = []

        # Create output widget to contain everything
        _out = widgets.Output(
            layout=widgets.Layout(width="800px", height="600px", overflow="hidden")
        )

        fig, ax = plt.subplots(figsize=(15, 15))

        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        ax.imshow(display_img)
        ax.set_title(
            "Annotate the header: \nClick 4 corners of the header region such that the entire header is contained within the rectangle."
        )
        ax.set_axis_off()

        # Create ipywidgets buttons
        done_button = widgets.Button(
            description="Done Cropping",
            button_style="success",
            layout=widgets.Layout(width="200px", height="50px"),
        )

        undo_button = widgets.Button(
            description="Undo Last Point",
            button_style="warning",
            layout=widgets.Layout(width="200px", height="50px"),
        )

        done_button.style.font_size = "18px"
        undo_button.style.font_size = "18px"

        status_label = widgets.Label(
            value="Press 'Done' when finished. Press 'Undo Last Point' to remove the last point.",
            style={"font_size": "18px"},
        )

        def on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return

            x, y = int(event.xdata), int(event.ydata)

            if event.button == 1:  # Left click - add point
                points.append((x, y))
                (point_marker,) = ax.plot(x, y, "go", markersize=10)
                drawn_points.append(point_marker)
                status_label.value = f"Points: {len(points)}/4"
                fig.canvas.draw_idle()

        def on_undo(_):
            if points:
                points.pop()
                drawn_points.pop().remove()
                status_label.value = f"Points: {len(points)}/4"
                fig.canvas.draw_idle()

        def on_done(_):
            nonlocal cid

            if len(points) != 4:
                status_label.value = (
                    f"Error: Need exactly 4 points! Currently have {len(points)}"
                )
                return

            fig.canvas.mpl_disconnect(cid)

            # Crop the image
            points_np = np.array(points)
            img_h, img_w = template.shape[:2]
            x_min = max(int(np.min(points_np[:, 0])) - margin, 0)
            y_min = max(int(np.min(points_np[:, 1])) - margin, 0)
            x_max = min(int(np.max(points_np[:, 0])) + margin, img_w)
            y_max = min(int(np.max(points_np[:, 1])) + margin, img_h)

            cropped = template[y_min:y_max, x_min:x_max]

            # Save cropped image if path provided
            if session._crop_path is not None:
                cv.imwrite(os.fspath(session._crop_path), cropped)

            # Close current figure and show annotation UI
            plt.close(fig)
            done_button.close()
            undo_button.close()
            status_label.close()

            # Show annotation UI
            HeaderTemplate._show_annotation_ui(cropped, session)

        done_button.on_click(on_done)
        undo_button.on_click(on_undo)

        cid = fig.canvas.mpl_connect("button_press_event", on_click)

        # Display figure first, then buttons below
        plt.tight_layout(pad=0)
        plt.show()
        display(widgets.HBox([done_button, undo_button, status_label]))

    @staticmethod
    def _show_annotation_ui(template: MatLike, session: "AnnotationSession"):
        """Show the line annotation UI using matplotlib + ipywidgets."""
        import ipywidgets as widgets
        from IPython.display import display

        print(
            "\x1b[32m[Taulu]: Don't forget to save annotations with annotation.save()!"
        )

        display_img = cv.cvtColor(template, cv.COLOR_BGR2RGB)

        lines: list[list[int]] = []
        start_point: list[tuple[int, int] | None] = [None]
        drawn_lines: list = []

        fig, ax = plt.subplots(figsize=(15, 12))
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        ax.imshow(display_img)
        ax.set_title("Click pairs of points to draw lines. Lines: 0")
        ax.set_axis_off()

        # Create ipywidgets buttons
        done_button = widgets.Button(
            description="Done Annotating",
            button_style="success",
            layout=widgets.Layout(width="200px", height="50px"),
        )
        undo_button = widgets.Button(
            description="Undo Last Line",
            button_style="warning",
            layout=widgets.Layout(width="200px", height="50px"),
        )
        status_label = widgets.Label(
            value="Click to start a line, click again to end it",
            style={"font_size": "18px"},
        )

        done_button.style.font_size = "18px"
        undo_button.style.font_size = "18px"

        def on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return

            x, y = int(event.xdata), int(event.ydata)

            if event.button == 1:  # Left click
                if start_point[0] is not None:
                    x0, y0 = start_point[0]
                    lines.append([x0, y0, x, y])
                    (ln,) = ax.plot([x0, x], [y0, y], color="lime", linewidth=2)
                    drawn_lines.append(ln)
                    ax.set_title(
                        f"Click pairs of points to draw lines. Lines: {len(lines)}"
                    )
                    status_label.value = (
                        f"Line {len(lines)} added. Click to start next line."
                    )
                    fig.canvas.draw_idle()
                    start_point[0] = None
                else:
                    start_point[0] = (x, y)
                    status_label.value = (
                        f"Start point set at ({x}, {y}). Click end point."
                    )
                    # Draw a temporary marker
                    ax.plot(x, y, "ro", markersize=5)
                    fig.canvas.draw_idle()

        def on_undo(_):
            start_point[0] = None
            if lines:
                lines.pop()
                drawn_lines.pop().remove()
                ax.set_title(
                    f"Click pairs of points to draw lines. Lines: {len(lines)}"
                )
                status_label.value = f"Undone. Lines: {len(lines)}"
                fig.canvas.draw_idle()

        def on_done(_):
            session._result = HeaderTemplate(lines)
            fig.canvas.mpl_disconnect(cid)
            done_button.disabled = True
            undo_button.disabled = True
            ax.set_title(
                f"Done! {len(lines)} lines annotated. Call session.save() to save."
            )
            status_label.value = (
                "Annotation complete! Run session.save('filename.json') to save."
            )
            fig.canvas.draw_idle()

        done_button.on_click(on_done)
        undo_button.on_click(on_undo)

        cid = fig.canvas.mpl_connect("button_press_event", on_click)

        # Display figure first, then buttons below
        plt.tight_layout(pad=0)
        plt.show()
        display(widgets.HBox([done_button, undo_button, status_label]))

    @staticmethod
    @log_calls(level=logging.DEBUG, include_return=True)
    def _crop(template: MatLike, margin: int = 10) -> MatLike:
        """
        Crop the image to contain only the annotations, such that it can be used as the header image in the taulu workflow.
        """

        points = []
        anno_template = np.copy(template)

        def get_point(event, x, y, flags, params):
            nonlocal points, anno_template
            _ = flags
            _ = params
            if event == cv.EVENT_LBUTTONDOWN:
                point = (x, y)

                cv.circle(  # type:ignore
                    anno_template,  # type:ignore
                    (x, y),
                    4,
                    (0, 255, 0),
                    2,
                )
                cv.imshow(constants.WINDOW, anno_template)  # type:ignore

                points.append(point)
            elif event == cv.EVENT_RBUTTONDOWN:
                # remove the last annotation
                points = points[:-1]

                anno_template = np.copy(anno_template)

                for p in points:
                    cv.circle(
                        anno_template,
                        p,
                        4,
                        (0, 255, 0),
                        2,
                    )

                cv.imshow(constants.WINDOW, anno_template)

        print(CROP_HELP)

        imu.show(anno_template, get_point, title="crop the header")

        assert len(points) == 4, (
            "you need to annotate the four corners of the table in order to crop it"
        )

        # crop the image to contain all of the points (just crop rectangularly, x, y, w, h)
        # Convert points to numpy array
        points_np = np.array(points)

        # Find bounding box
        x_min = np.min(points_np[:, 0])
        y_min = np.min(points_np[:, 1])
        x_max = np.max(points_np[:, 0])
        y_max = np.max(points_np[:, 1])

        # Compute width and height
        width = x_max - x_min
        height = y_max - y_min

        # Ensure integers and within image boundaries
        x_min = max(int(x_min), 0)
        y_min = max(int(y_min), 0)
        width = int(width)
        height = int(height)

        # Crop the image
        cropped = template[
            y_min - margin : y_min + height + margin,
            x_min - margin : x_min + width + margin,
        ]

        return cropped

    @staticmethod
    def from_vgg_annotation(annotation: str) -> "HeaderTemplate":
        """
        Create a TableTemplate from annotations made in [vgg](https://annotate.officialstatistics.org/), using the polylines tool.

        Args:
            annotation (str): the path of the annotation csv file
        """

        rules = []
        with open(annotation) as csvfile:
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

    def cell_heights(self, header_factors: list[float] | float) -> list[int]:
        if isinstance(header_factors, float):
            header_factors = [header_factors]
        header_factors = cast(list, header_factors)
        return [
            int((self._h_rules[1]._y - self._h_rules[0]._y) * f) for f in header_factors
        ]

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

    def region(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[Point, Point, Point, Point]:
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

        def to_point(pnt) -> Point:
            return (int(pnt[0]), int(pnt[1]))

        return (
            to_point(top_left),
            to_point(top_right),
            to_point(bottom_right),
            to_point(bottom_left),
        )

    def text_regions(
        self, img: MatLike, row: int, margin_x: int = 10, margin_y: int = -20
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        raise TauluException("text_regions should not be called on a HeaderTemplate")
