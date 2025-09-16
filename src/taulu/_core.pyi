from typing import Optional
import numpy as np
from .types import Point

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
    ): ...
    def get_corner(self, coord: tuple[int, int]) -> Optional[Point]: ...
    def all_rows_complete(self) -> bool: ...
    def get_all_corners(self) -> list[list[Optional[Point]]]: ...
    def get_edge_points(self) -> list[tuple[Point, float]]: ...
    def grow_point(
        self, table_image: np.ndarray, cross_correlation: np.ndarray
    ) -> Optional[float]: ...
    def grow_points(self, table_image: np.ndarray, cross_correlation: np.ndarray): ...
