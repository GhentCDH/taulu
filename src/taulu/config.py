"""
TauluConfig: a dataclass representation of all Taulu constructor parameters.

Can be loaded from a TOML file with TauluConfig.from_toml().

TOML format
-----------
Scalar values map directly to parameters::

    header_image_path = "header.png"
    sauvola_k = 0.25
    kernel_size = 41

For split (two-page) tables, use a table with ``left`` and ``right`` keys for
any parameter that differs between sides::

    [header_image_path]
    left = "header_left.png"
    right = "header_right.png"

    [kernel_size]
    left = 41
    right = 35

    sauvola_k = 0.25  # same for both sides — scalar is fine
"""

import tomllib
from dataclasses import dataclass
from os import PathLike
from typing import Any

from .split import Split

type Splittable[T] = Split[T] | T


def _parse_value(value: Any) -> Any:
    """Convert a dict with 'left'/'right' keys into a Split; leave others as-is."""
    if isinstance(value, dict) and "left" in value and "right" in value:
        return Split(value["left"], value["right"])
    return value


@dataclass
class TauluConfig:
    """
    Configuration for :class:`~taulu.Taulu`.

    All parameters mirror the ``Taulu.__init__`` signature. Any parameter that
    accepts a ``Split[T]`` can be given as a ``Split`` instance or as a plain
    scalar (applied to both sides).

    Use :meth:`from_toml` to load from a ``.toml`` file, then pass to
    :meth:`Taulu.from_config <taulu.Taulu.from_config>`.
    """

    header_image_path: Splittable[str]
    cell_height_factor: Splittable[float] | Splittable[list[float]] | None = None
    header_anno_path: Splittable[str] | None = None
    sauvola_k: Splittable[float] = 0.25
    search_region: Splittable[int] = 60
    distance_penalty: Splittable[float] = 0.4
    cross_width: Splittable[int] = 10
    morph_size: Splittable[int] = 4
    kernel_size: Splittable[int] = 41
    processing_scale: Splittable[float] = 1.0
    skip_astar_threshold: Splittable[float] = 0.2
    min_rows: Splittable[int] = 5
    look_distance: Splittable[int] = 3
    grow_threshold: Splittable[float] = 0.3
    smooth_grid: bool = False
    smooth_strength: float = 0.5
    smooth_iterations: int = 1
    smooth_degree: int = 1
    cuts: Splittable[int] = 0
    cut_fraction: Splittable[float] = 0.5
    match_method: Splittable[str] = "akaze"
    alignment_scale: float = 1.0

    @classmethod
    def from_toml(cls, *paths: PathLike[str] | str) -> "TauluConfig":
        """
        Load a :class:`TauluConfig` from one or more TOML files.

        When multiple paths are given, files are merged in order: later files
        override keys from earlier ones. Use this to share a common base config
        and override only the fields that differ::

            config = TauluConfig.from_toml("common.toml", "left.toml")

        Args:
            *paths: One or more paths to ``.toml`` configuration files.

        Returns:
            A fully populated :class:`TauluConfig` instance.

        Raises:
            KeyError: If a required field (``header_image_path``) is missing.
            TypeError: If a field value has an unexpected type.
        """
        merged: dict = {}
        for path in paths:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            merged.update(data)

        parsed = {key: _parse_value(value) for key, value in merged.items() if not key.startswith("$")}
        return cls(**parsed)
