"""
JSON Schema generation for TauluConfig TOML files.

Run ``taulu-schema > taulu-config.schema.json`` to export the schema to your
project directory, then reference it in your TOML file::

    "$schema" = "./taulu-config.schema.json"

Editors with taplo support (VS Code "Even Better TOML", Neovim) will use this
for autocompletion and validation.

Multiple TOML files can be merged, with later files overriding earlier ones::

    "$schema" = "./taulu-config.schema.json"
    # common.toml — shared parameters
    binarization_sensitivity = 0.05

    # left.toml — overrides only what differs
    "$schema" = "./taulu-config.schema.json"
    template_path = "header_left.png"
"""

import json
import sys


def _splittable(inner: dict) -> dict:
    """JSON Schema oneOf: scalar value or {left, right} split table."""
    return {
        "oneOf": [
            inner,
            {
                "type": "object",
                "properties": {"left": inner, "right": inner},
                "required": ["left", "right"],
                "additionalProperties": False,
            },
        ]
    }


_INT = {"type": "integer"}
_FLOAT = {"type": "number"}
_STR = {"type": "string"}
_BOOL = {"type": "boolean"}
_FEATURE_DETECTOR = {"type": "string", "enum": ["orb", "sift", "akaze"]}
_CELL_HEIGHT_FACTOR = {
    "oneOf": [
        {"type": "number"},
        {"type": "array", "items": {"type": "number"}},
    ]
}


def generate_schema() -> dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "TauluConfig",
        "description": (
            "Configuration file for the Taulu table segmentation library. "
            "Multiple files can be merged via TauluConfig.from_toml('common.toml', 'left.toml'), "
            "with later files overriding earlier ones. template_path is required across "
            "the merged set but may live in any of the files."
        ),
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "$schema": {
                "type": "string",
                "description": "Path or URL to this JSON Schema file.",
            },
            "template_path": {
                **_splittable(_STR),
                "description": "Path to header template image(s). Use left/right split for two-page tables.",
            },
            "annotation_path": {
                "oneOf": [
                    _STR,
                    {
                        "type": "object",
                        "properties": {"left": _STR, "right": _STR},
                        "required": ["left", "right"],
                        "additionalProperties": False,
                    },
                ],
                "description": "Explicit annotation JSON path. Default: inferred from template_path.",
            },
            "row_height_factor": {
                "oneOf": [
                    _CELL_HEIGHT_FACTOR,
                    {
                        "type": "object",
                        "properties": {
                            "left": _CELL_HEIGHT_FACTOR,
                            "right": _CELL_HEIGHT_FACTOR,
                        },
                        "required": ["left", "right"],
                        "additionalProperties": False,
                    },
                ],
                "description": "Row height relative to header (e.g. 0.8 for 80%). Default: [1.0]",
            },
            "binarization_sensitivity": {
                **_splittable(_FLOAT),
                "description": "Binarization threshold (0.0-1.0). Higher = less noise. Default: 0.25",
                "default": 0.25,
            },
            "search_radius": {
                **_splittable(_INT),
                "description": "Corner search area in pixels. Default: 60",
                "default": 60,
            },
            "position_weight": {
                **_splittable(_FLOAT),
                "description": "Position penalty weight [0, 1]. Default: 0.4",
                "default": 0.4,
            },
            "line_thickness": {
                **_splittable(_INT),
                "description": "Cross-kernel width matching line thickness. Default: 10",
                "default": 10,
            },
            "line_gap_fill": {
                **_splittable(_INT),
                "description": "Morphological dilation size for gap filling. Default: 4",
                "default": 4,
            },
            "intersection_kernel_size": {
                **_splittable(_INT),
                "description": "Cross-kernel size (must be odd). Default: 41",
                "default": 41,
            },
            "detection_scale": {
                **_splittable(_FLOAT),
                "description": "Image downscale factor (0, 1]. Default: 1.0",
                "default": 1.0,
            },
            "pathfinding_threshold": {
                **_splittable(_FLOAT),
                "description": "Confidence threshold to skip A* pathfinding. Default: 0.2",
                "default": 0.2,
            },
            "min_rows": {
                **_splittable(_INT),
                "description": "Minimum rows before completion. Default: 5",
                "default": 5,
            },
            "extrapolation_distance": {
                **_splittable(_INT),
                "description": "Rows to examine for extrapolation. Default: 3",
                "default": 3,
            },
            "detection_threshold": {
                **_splittable(_FLOAT),
                "description": "Corner acceptance confidence [0, 1]. Default: 0.3",
                "default": 0.3,
            },
            "smooth": {
                **_BOOL,
                "description": "Apply grid smoothing after detection. Default: false",
                "default": False,
            },
            "smooth_strength": {
                **_FLOAT,
                "description": "Blend factor per smoothing iteration (0.0-1.0). Default: 0.5",
                "default": 0.5,
            },
            "smooth_iterations": {
                **_INT,
                "description": "Number of smoothing passes. Default: 1",
                "default": 1,
            },
            "smooth_degree": {
                **_INT,
                "description": "Polynomial degree for smoothing regression (1 or 2). Default: 1",
                "default": 1,
                "enum": [1, 2],
            },
            "growing_resets": {
                **_splittable(_INT),
                "description": "Number of grid resets during growing. Default: 0",
                "default": 0,
            },
            "reset_fraction": {
                **_splittable(_FLOAT),
                "description": "Fraction of points to delete per reset. Default: 0.5",
                "default": 0.5,
            },
            "feature_detector": {
                **_splittable(_FEATURE_DETECTOR),
                "description": "Feature matching method: 'orb' (fast), 'sift' (robust), 'akaze'. Default: 'akaze'",
                "default": "akaze",
            },
            "matching_scale": {
                **_FLOAT,
                "description": "Downscale factor (0, 1] for header alignment only. Default: 1.0",
                "default": 1.0,
            },
        },
    }


def main():
    print(json.dumps(generate_schema(), indent=2))


if __name__ == "__main__":
    sys.exit(main())
