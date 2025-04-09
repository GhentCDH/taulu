import os
from pathlib import Path

this_dir = Path(__file__).parent

def image_path(index: int) -> str:
    return os.fspath((this_dir / f"../data/table_{index:02}.png").resolve())
