#!/usr/bin/env python3
"""Check horizontal/vertical rule classification in annotation files."""

import json
import math

TOLERANCE = math.pi / 6  # Same as in header_template.py


def check_file(filename):
    with open(filename) as f:
        data = json.load(f)
        rules = data["rules"]

    print(f"\n{filename}:")
    print(f"  Total rules: {len(rules)}")

    h_count = 0
    v_count = 0

    for i, rule in enumerate(rules):
        x0, y0, x1, y1 = rule["x0"], rule["y0"], rule["x1"], rule["y1"]

        # Calculate angle
        y_diff = y1 - y0
        x_diff = x1 - x0

        if x_diff == 0:
            angle = (1 if y_diff >= 0 else -1) * math.pi / 2
        else:
            angle = math.atan(y_diff / x_diff)

        # Check if horizontal
        is_horizontal = -TOLERANCE <= angle <= TOLERANCE

        # Check if vertical
        is_vertical = (
            angle <= -math.pi / 2 + TOLERANCE or angle >= math.pi / 2 - TOLERANCE
        )

        angle_deg = math.degrees(angle)

        if is_horizontal:
            h_count += 1
            print(f"    Rule {i}: HORIZONTAL (angle: {angle_deg:.1f}°)")
        elif is_vertical:
            v_count += 1
            print(f"    Rule {i}: VERTICAL (angle: {angle_deg:.1f}°)")
        else:
            print(f"    Rule {i}: NEITHER (angle: {angle_deg:.1f}°)")

    print(f"  Horizontal: {h_count}, Vertical: {v_count}")

    if h_count < 2:
        print(f"  ⚠️  ERROR: Need at least 2 horizontal rules, only have {h_count}!")
    if v_count < 2:
        print(f"  ⚠️  WARNING: Need at least 2 vertical rules, only have {v_count}!")

    return h_count, v_count


# Check all files
files = [
    "header.json",
    "table_00_left.json",
    "table_00_right.json",
    "table_01_left.json",
    "table_01_right.json",
]

for f in files:
    try:
        check_file(f)
    except FileNotFoundError:
        print(f"\n{f}: NOT FOUND")
