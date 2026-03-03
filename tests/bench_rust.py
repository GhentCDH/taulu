from time import perf_counter

import cv2
from util import (
    header_left_anno_path,
    header_left_image_path,
    header_right_anno_path,
    header_right_image_path,
    table_filtered_path,
    table_image_path,
)

import taulu._core as c
from taulu import Taulu
from taulu.img_util import draw_points, ensure_gray, show
from taulu.split import Split


def simple_bench_astar():
    img = ensure_gray(cv2.imread(table_image_path(0)))

    start = (856, 1057)

    goals = [(2000 + i, 2200) for i in range(400)]

    strt = perf_counter()
    path = c.astar(img, start, goals, "any")
    print(f"Astar took {(perf_counter() - strt) * 1000} ms")

    drawn = draw_points(img, path)
    show(drawn)


def taulu_bench():
    tl = Taulu(
        header_image_path=Split(header_left_image_path(1), header_right_image_path(1)),
        header_anno_path=Split(header_left_anno_path(1), header_right_anno_path(1)),
        cell_height_factor=Split([0.35, 0.23], [0.37, 0.24]),
        sauvola_k=0.05,
        search_region=40,
        distance_penalty=0.8,
        kernel_size=31,
        cross_width=8,
        morph_size=4,
        min_rows=10,
        grow_threshold=0.5,
        look_distance=2,
    )

    im = cv2.imread(table_image_path(1))
    filtered = table_filtered_path(1)
    print(tl.segment_table(im, filtered=filtered, debug_view=False))


if __name__ == "__main__":
    # simple_bench_astar()
    taulu_bench()
