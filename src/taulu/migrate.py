"""
Migrate a TauluConfig TOML file from old parameter names to new ones.

Relevant for Taulu upgrades to 3.0.0

Usage::

    uv run -m taulu.migrate config.toml           # print migrated config to stdout
    uv run -m taulu.migrate config.toml --inplace  # overwrite file (creates .bak backup)
"""

import re
import sys
import tomllib
from pathlib import Path

RENAMES: dict[str, str] = {
    "header_image_path": "template_path",
    "header_anno_path": "annotation_path",
    "cell_height_factor": "row_height_factor",
    "sauvola_k": "binarization_sensitivity",
    "search_region": "search_radius",
    "distance_penalty": "position_weight",
    "cross_width": "line_thickness",
    "morph_size": "line_gap_fill",
    "kernel_size": "intersection_kernel_size",
    "processing_scale": "detection_scale",
    "skip_astar_threshold": "pathfinding_threshold",
    "grow_threshold": "detection_threshold",
    "look_distance": "extrapolation_distance",
    "smooth_grid": "smooth",
    "cuts": "growing_resets",
    "cut_fraction": "reset_fraction",
    "match_method": "feature_detector",
    "alignment_scale": "matching_scale",
}


def migrate(text: str) -> tuple[str, list[str]]:
    """
    Rename old parameter keys in a TOML string.

    Returns the migrated text and a list of human-readable changes made.
    """
    lines = text.splitlines(keepends=True)
    result = []
    changes = []

    # Matches `key =` at start of line (with optional leading whitespace)
    key_assign = re.compile(r"^(\s*)(\w+)(\s*=)")
    # Matches `[key]` section headers used for Split values
    key_section = re.compile(r"^\[(\w+)\]")

    for line in lines:
        m = key_assign.match(line)
        if m:
            key = m.group(2)
            if key in RENAMES:
                new_key = RENAMES[key]
                line = line[: m.start(2)] + new_key + line[m.end(2) :]
                changes.append(f"  {key} → {new_key}")

        m = key_section.match(line)
        if m:
            key = m.group(1)
            if key in RENAMES:
                new_key = RENAMES[key]
                line = f"[{new_key}]\n"
                changes.append(f"  [{key}] → [{new_key}]")

        result.append(line)

    return "".join(result), changes


def main():
    args = sys.argv[1:]
    inplace = "--inplace" in args
    paths = [a for a in args if not a.startswith("-")]

    if not paths:
        print("Usage: uv run -m taulu.migrate <file.toml> [--inplace]", file=sys.stderr)
        sys.exit(1)

    path = Path(paths[0])

    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    text = path.read_text()

    # Validate input is valid TOML
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        print(f"Error: {path} is not valid TOML: {e}", file=sys.stderr)
        sys.exit(1)

    migrated, changes = migrate(text)

    if not changes:
        print("No changes needed.", file=sys.stderr)
        if not inplace:
            print(migrated, end="")
        sys.exit(0)

    print(f"Renamed {len(changes)} key(s):", file=sys.stderr)
    for change in changes:
        print(change, file=sys.stderr)

    if inplace:
        backup = path.with_suffix(".toml.bak")
        backup.write_text(text)
        path.write_text(migrated)
        print(f"Written to {path} (backup: {backup})", file=sys.stderr)
    else:
        print(migrated, end="")


if __name__ == "__main__":
    main()
