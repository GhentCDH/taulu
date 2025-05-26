import os
from pathlib import Path
import pytest

this_dir = Path(__file__).parent


def table_image_path(index: int) -> str:
    return os.fspath((this_dir / f"../data/table_{index:02}.png").resolve())


def table_left_image_path(index: int) -> str:
    return os.fspath((this_dir / f"../data/table_left_{index:02}.png").resolve())


def header_image_path(index: int) -> str:
    return os.fspath((this_dir / f"../data/header_{index:02}.png").resolve())


def header_anno_path(index: int) -> str:
    return os.fspath((this_dir / f"../data/header_{index:02}.json").resolve())


def header_left_image_path(index: int) -> str:
    return os.fspath((this_dir / f"../data/header_left_{index:02}.png").resolve())


def header_left_anno_path(index: int) -> str:
    return os.fspath((this_dir / f"../data/header_left_{index:02}.json").resolve())


def header_right_image_path(index: int) -> str:
    return os.fspath((this_dir / f"../data/header_right_{index:02}.png").resolve())


def header_right_anno_path(index: int) -> str:
    return os.fspath((this_dir / f"../data/header_right_{index:02}.json").resolve())


def files_exist(*paths):
    return all(os.path.exists(p) for p in paths)
