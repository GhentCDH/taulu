from typing import Optional
import numpy as np
from .types import Point, PointFloat

def astar(
    img: np.ndarray,
    start: tuple[int, int],
    goals: list[tuple[int, int]],
    direction: str,
) -> list[tuple[int, int]] | None: ...
def median_slope(
    lines: list[tuple[PointFloat, PointFloat]],
) -> float: ...

class TableGrower:
    """
    Grow a table using this omni directional method
    """

    def __init__(
        self,
        table_image: np.ndarray,
        cross_correlation: np.ndarray,
        column_widths: list[int],
        row_heights: list[int],
        start_point: tuple[int, int],
        search_region: int,
        distance_penalty: float,
        look_distance: int,
        grow_threshold: float,
        min_row_count: int,
    ): ...
    def get_corner(self, coord: tuple[int, int]) -> Optional[Point]: ...
    def all_rows_complete(self) -> bool: ...
    def get_all_corners(self) -> list[list[Optional[Point]]]: ...
    def get_edge_points(self) -> list[tuple[Point, float]]: ...
    def grow_point(
        self,
        table_image: np.ndarray,
        cross_correlation: np.ndarray,
    ) -> Optional[float]: ...
    def grow_points(
        self,
        table_image: np.ndarray,
        cross_correlation: np.ndarray,
    ): ...
    def extrapolate_one(
        self, table_image: np.ndarray, cross_correlation: np.ndarray
    ) -> Optional[Point]: ...
    def is_table_complete(self) -> bool: ...
    def grow_table(
        self,
        table_image: np.ndarray,
        cross_correlation: np.ndarray,
    ): ...
    def set_threshold(self, value: float): ...
