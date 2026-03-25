"""
Defines an abstract class TableIndexer, which provides methods for mapping pixel coordinates
in an image to table cell indices and for cropping images to specific table cells or regions.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Generator

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from . import img_util as imu
from .constants import WINDOW
from .error import TauluException
from .types import Point


class ShowCellsSession:
    """
    Session object for notebook-based cell visualization.

    In Jupyter notebooks with %matplotlib widget, this allows interactive
    cell highlighting without blocking. Access the clicked cells via the
    .cells property.

    Usage:
        session = grid.show_cells_notebook(image)
        # ... click on cells in the plot ...
        clicked_cells = session.cells  # Access the list of clicked cells
    """

    def __init__(self):
        self._cells: list[tuple[int, int]] = []

    @property
    def cells(self) -> list[tuple[int, int]]:
        """Returns the list of cells that have been clicked."""
        return self._cells.copy()


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
    Abstract base class for table cell indexing and cropping.

    Subclasses (`SegmentedTable`, `TableTemplate`) implement the `cols`, `rows`,
    and `cell_polygon` interface. This base provides shared methods for
    mapping pixel coordinates to cell indices and cropping cells/regions.
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

    def cells(self) -> Generator[tuple[int, int]]:
        """
        Generate all cell indices in row-major order.

        Yields (row, col) tuples for every cell in the table, iterating
        through each row from left to right, top to bottom.

        Yields:
            tuple[int, int]: Cell indices as (row, col).

        Example:
            >>> for row, col in grid.cells():
            ...     cell_img = grid.crop_cell(image, (row, col))
            ...     process(cell_img)
        """
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
        cv.polylines(image, [points], True, color, thickness, cv.LINE_AA)
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
        image: MatLike | os.PathLike[str] | str,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 1,
    ) -> MatLike:
        if not isinstance(image, np.ndarray):
            image = cv.imread(os.fspath(image))  # ty:ignore
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

    def show_cells(
        self, image: MatLike | os.PathLike[str] | str, window: str = WINDOW
    ) -> list[tuple[int, int]] | ShowCellsSession:
        """
        Interactively display and highlight table cells.

        In standard environments, shows an OpenCV window where clicking highlights cells.
        In Jupyter notebooks, returns a ShowCellsSession and displays using matplotlib.

        Args:
            image: Source image (path or array).
            window: OpenCV window name (ignored in notebooks).

        Returns:
            list[tuple[int, int]]: Clicked cell indices (non-notebook).
            ShowCellsSession: Session object with .cells property (notebook).

        Example:
            >>> # Standard Python
            >>> cells = grid.show_cells("table.png")
            >>>
            >>> # Jupyter Notebook
            >>> session = grid.show_cells("table.png")
            >>> # ... click cells ...
            >>> cells = session.cells
        """
        if not isinstance(image, np.ndarray):
            image = cv.imread(os.fspath(image))  # ty:ignore

        def running_in_notebook() -> bool:
            try:
                from IPython import get_ipython

                ip = get_ipython()
                return ip is not None and "IPKernelApp" in ip.config
            except Exception:
                return False

        use_notebook = running_in_notebook()

        if use_notebook:
            return self.show_cells_notebook(image)
        else:
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

    def show_cells_notebook(
        self, image: MatLike | os.PathLike[str] | str
    ) -> ShowCellsSession:
        """
        Notebook-compatible version of show_cells using matplotlib.

        Returns a ShowCellsSession immediately. Click on cells to highlight them.
        Access clicked cells via session.cells.

        Args:
            image: Source image (path or array).

        Returns:
            ShowCellsSession: Access .cells to get list of clicked cell indices.

        Example:
            >>> session = grid.show_cells_notebook("table.png")
            >>> # Click cells in the interactive plot
            >>> print(session.cells)  # [(0, 0), (1, 2), ...]
        """
        if not isinstance(image, np.ndarray):
            tmp_image = cv.imread(os.fspath(image))
            assert tmp_image is not None
            image = tmp_image

        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display

        session = ShowCellsSession()

        # Convert BGR to RGB for matplotlib
        display_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img_with_highlights = np.copy(display_img)

        fig, ax = plt.subplots(figsize=(15, 12))
        fig.canvas.toolbar_visible = False  # ty:ignore[unresolved-attribute]
        fig.canvas.header_visible = False  # ty:ignore[unresolved-attribute]

        im_display = ax.imshow(img_with_highlights)
        ax.set_title("Click cells to highlight them. Cells clicked: 0")
        ax.set_axis_off()

        # Create buttons
        done_button = widgets.Button(
            description="Done",
            button_style="success",
            layout=widgets.Layout(width="150px", height="50px"),
        )
        clear_button = widgets.Button(
            description="Clear All",
            button_style="warning",
            layout=widgets.Layout(width="150px", height="50px"),
        )
        undo_button = widgets.Button(
            description="Undo Last",
            button_style="info",
            layout=widgets.Layout(width="150px", height="50px"),
        )

        done_button.style.font_size = "18px"
        clear_button.style.font_size = "18px"
        undo_button.style.font_size = "18px"

        status_label = widgets.Label(
            value="Click on cells to highlight them", style={"font_size": "18px"}
        )

        def draw_highlight(cell_idx: tuple[int, int]):
            """Draw a highlighted cell on the image."""
            polygon = self.cell_polygon(cell_idx)
            points = np.array(list(polygon), dtype=np.int32)

            # Draw polyline on the RGB image
            cv.polylines(
                img_with_highlights,
                [points],
                True,
                (255, 0, 0),  # Red in RGB
                2,
                cv.LINE_AA,
            )

            # Draw cell index text
            cv.putText(
                img_with_highlights,
                str(cell_idx),
                (int(polygon[3][0] + 10), int(polygon[3][1] - 10)),
                cv.FONT_HERSHEY_PLAIN,
                2.0,
                (255, 255, 255),  # White in RGB
                2,
            )

        def redraw_all():
            """Redraw the image with all current highlights."""
            nonlocal img_with_highlights
            img_with_highlights = np.copy(display_img)

            for cell_idx in session._cells:
                draw_highlight(cell_idx)

            im_display.set_data(img_with_highlights)
            ax.set_title(
                f"Click cells to highlight them. Cells clicked: {len(session._cells)}"
            )
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return

            x, y = int(event.xdata), int(event.ydata)

            if event.button == 1:  # Left click
                cell_idx = self.cell((x, y))
                if cell_idx[0] >= 0:
                    session._cells.append(cell_idx)
                    draw_highlight(cell_idx)
                    im_display.set_data(img_with_highlights)
                    ax.set_title(
                        f"Click cells to highlight them. Cells clicked: {len(session._cells)}"
                    )
                    status_label.value = (
                        f"Cell {cell_idx} highlighted. Total: {len(session._cells)}"
                    )
                    fig.canvas.draw_idle()
                else:
                    status_label.value = f"Click at ({x}, {y}) is outside table bounds"

        def on_clear(_):
            session._cells.clear()
            redraw_all()
            status_label.value = "All highlights cleared"

        def on_undo(_):
            if session._cells:
                removed = session._cells.pop()
                redraw_all()
                status_label.value = (
                    f"Removed cell {removed}. Remaining: {len(session._cells)}"
                )
            else:
                status_label.value = "No cells to undo"

        def on_done(_):
            fig.canvas.mpl_disconnect(cid)
            done_button.disabled = True
            clear_button.disabled = True
            undo_button.disabled = True
            ax.set_title(f"Done! {len(session._cells)} cells highlighted.")
            status_label.value = "Complete! Access clicked cells via session.cells"
            fig.canvas.draw_idle()

        done_button.on_click(on_done)
        clear_button.on_click(on_clear)
        undo_button.on_click(on_undo)

        cid = fig.canvas.mpl_connect("button_press_event", on_click)

        plt.tight_layout(pad=0)
        plt.show()
        display(widgets.HBox([done_button, clear_button, undo_button, status_label]))

        return session

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
        """
        Extract a multi-cell region from the image with perspective correction.

        Crops the image to include all cells from start to end (inclusive),
        applying a perspective transform to produce a rectangular output.

        Args:
            image: Source image (BGR or grayscale).
            start: Top-left cell as (row, col).
            end: Bottom-right cell as (row, col).
            margin: Uniform margin in pixels (default 0).
            margin_top: Override top margin.
            margin_bottom: Override bottom margin.
            margin_left: Override left margin.
            margin_right: Override right margin.
            margin_y: Override vertical margins (top and bottom).
            margin_x: Override horizontal margins (left and right).

        Returns:
            Cropped and perspective-corrected image.

        Example:
            >>> # Extract a 3x2 region starting at cell (1, 0)
            >>> region_img = grid.crop_region(image, (1, 0), (3, 1))
        """

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
        m = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv.warpPerspective(image, m, (int(w), int(h)))

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

    def crop_cell(
        self,
        image,
        cell: tuple[int, int],
        margin: int = 0,
        margin_top: int | None = None,
        margin_bottom: int | None = None,
        margin_left: int | None = None,
        margin_right: int | None = None,
        margin_y: int | None = None,
        margin_x: int | None = None,
    ) -> MatLike:
        """
        Extract a single cell from the image with perspective correction.

        Convenience method equivalent to `crop_region(image, cell, cell, margin)`.

        Args:
            image: Source image (BGR or grayscale).
            cell: Cell indices as (row, col).
            margin: Padding in pixels around the cell (default 0).

        Returns:
            Cropped and perspective-corrected cell image.

        Example:
            >>> cell_img = grid.crop_cell(image, (0, 0))
            >>> cv2.imwrite("cell_0_0.png", cell_img)
        """
        return self.crop_region(
            image,
            cell,
            cell,
            margin,
            margin_top,
            margin_bottom,
            margin_left,
            margin_right,
            margin_y,
            margin_x,
        )
