"""
TauluConfig: a Pydantic model representation of all Taulu constructor parameters.

Can be loaded from a TOML file with TauluConfig.from_toml().

TOML format
-----------
Scalar values map directly to parameters::

    template_path = "header.png"
    binarization_sensitivity = 0.25
    intersection_kernel_size = 41

For split (two-page) tables, use a table with ``left`` and ``right`` keys for
any parameter that differs between sides::

    [template_path]
    left = "header_left.png"
    right = "header_right.png"

    [intersection_kernel_size]
    left = 41
    right = 35

    binarization_sensitivity = 0.25  # same for both sides — scalar is fine
"""

import tomllib
from os import PathLike
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .split import Split

type Splittable[T] = Split[T] | T


def _parse_value(value: Any) -> Any:
    """Convert a dict with 'left'/'right' keys into a Split; leave others as-is."""
    if isinstance(value, dict) and "left" in value and "right" in value:
        return Split(value["left"], value["right"])
    return value


class TauluConfig(BaseModel):
    """
    Configuration for :class:`~taulu.Taulu`.

    All parameters mirror the ``Taulu.__init__`` signature. Any parameter that
    accepts a ``Split[T]`` can be given as a ``Split`` instance or as a plain
    scalar (applied to both sides).

    Use :meth:`from_toml` to load from a ``.toml`` file, then pass to
    :meth:`Taulu.from_config <taulu.Taulu.from_config>`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    template_path: Splittable[str] = Field(
        description="Path to header template image(s). Use left/right split for two-page tables.",
    )
    row_height_factor: Splittable[float] | Splittable[list[float]] | None = Field(
        default=None,
        description="Row height relative to header (e.g. 0.8 for 80%). Default: [1.0]",
    )
    annotation_path: Splittable[str] | None = Field(
        default=None,
        description="Explicit annotation JSON path. Default: inferred from template_path.",
    )
    binarization_sensitivity: Splittable[float] = Field(
        default=0.25,
        description="Binarization threshold (0.0-1.0). Higher = less noise.",
    )
    search_radius: Splittable[int] = Field(
        default=60,
        description="Corner search area in pixels.",
    )
    position_weight: Splittable[float] = Field(
        default=0.4,
        description="Position penalty weight [0, 1].",
    )
    line_thickness: Splittable[int] = Field(
        default=10,
        description="Cross-kernel width matching line thickness.",
    )
    line_gap_fill: Splittable[int] = Field(
        default=4,
        description="Morphological dilation size for gap filling.",
    )
    intersection_kernel_size: Splittable[int] = Field(
        default=41,
        description="Cross-kernel size (must be odd).",
    )
    detection_scale: Splittable[float] = Field(
        default=1.0,
        description="Image downscale factor (0, 1].",
    )
    pathfinding_threshold: Splittable[float] = Field(
        default=0.2,
        description="Confidence threshold to skip A* pathfinding.",
    )
    min_rows: Splittable[int] = Field(
        default=5,
        description="Minimum rows before completion.",
    )
    extrapolation_distance: Splittable[int] = Field(
        default=3,
        description="Rows to examine for extrapolation.",
    )
    detection_threshold: Splittable[float] = Field(
        default=0.3,
        description="Corner acceptance confidence [0, 1].",
    )
    smooth: bool = Field(
        default=False,
        description="Apply grid smoothing after detection.",
    )
    smooth_strength: float = Field(
        default=0.5,
        description="Blend factor per smoothing iteration (0.0-1.0).",
    )
    smooth_iterations: int = Field(
        default=1,
        description="Number of smoothing passes.",
    )
    smooth_degree: int = Field(
        default=1,
        description="Polynomial degree for smoothing regression (1 or 2).",
    )
    growing_resets: Splittable[int] = Field(
        default=0,
        description="Number of grid resets during growing.",
    )
    reset_fraction: Splittable[float] = Field(
        default=0.5,
        description="Fraction of points to delete per reset.",
    )
    feature_detector: Splittable[Literal["orb", "sift", "akaze"]] = Field(
        default="akaze",
        description="Feature matching method: 'orb' (fast), 'sift' (robust), 'akaze'.",
    )
    matching_scale: float = Field(
        default=1.0,
        description="Downscale factor (0, 1] for header alignment only.",
    )
    auto_row_heights: bool = Field(
        default=False,
        description="If True, detect variable per-row heights from the cross-correlation map (overrides row_height_factor).",
    )
    min_row_height_factor: Splittable[float] = Field(
        default=0.5,
        description="Minimum row height as a fraction of header height when auto_row_heights is enabled.",
    )
    max_row_height_factor: Splittable[float] = Field(
        default=1.5,
        description="Maximum row height as a fraction of header height when auto_row_heights is enabled.",
    )
    row_detection_path_scale: float = Field(
        default=0.25,
        description="Downscale factor (0, 1] for the A* path following used by auto row height detection.",
    )

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
            KeyError: If a required field (``template_path``) is missing.
            TypeError: If a field value has an unexpected type.
        """
        merged: dict = {}
        for path in paths:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            merged.update(data)

        parsed = {
            key: _parse_value(value)
            for key, value in merged.items()
            if not key.startswith("$")
        }
        return cls(**parsed)
