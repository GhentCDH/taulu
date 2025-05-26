import taulu._core as c
from taulu.img_util import draw_points, ensure_gray, show
from util import table_image_path
import cv2

from time import perf_counter


def simple_bench_astar():
    img = ensure_gray(cv2.imread(table_image_path(0)))

    start = (856, 1057)

    goals = [(2000 + i, 2200) for i in range(400)]

    strt = perf_counter()
    path = c.astar(img, start, goals, "any")
    print(f"Astar took {(perf_counter() - strt) * 1000} ms")

    drawn = draw_points(img, path)
    show(drawn)


if __name__ == "__main__":
    simple_bench_astar()
