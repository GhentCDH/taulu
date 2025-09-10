import numpy as np

def astar(
    img: np.ndarray,
    start: tuple[int, int],
    goals: list[tuple[int, int]],
    direction: str,
) -> list[tuple[int, int]] | None: ...
