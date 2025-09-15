import numpy as np
from .types import PointFloat

def astar(
    img: np.ndarray,
    start: tuple[int, int],
    goals: list[tuple[int, int]],
    direction: str,
) -> list[tuple[int, int]] | None: ...
def median_slope(
    lines: list[tuple[PointFloat, PointFloat]],
) -> float: ...
