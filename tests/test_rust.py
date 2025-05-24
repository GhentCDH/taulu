import taulu._core as c
from taulu.img_util import draw_points, ensure_gray, show
from util import table_image_path
import cv2

from time import perf_counter


def test_hello():
    assert c.hello_from_bin() == "Hello from rust!"


def test_astar():
    img = ensure_gray(cv2.imread(table_image_path(0)))

    start = (856, 1057)

    goals = [(1000 + i, 1200) for i in range(400)]

    strt = perf_counter()
    path = c.astar(img, start, goals, "any")
    print(f"took {(perf_counter() - strt) * 1000} ms")

    drawn = draw_points(img, path)
    show(drawn)


def test_astar_small_img():
    img = ensure_gray(cv2.imread(table_image_path(0)))

    offset = (850, 1000)
    start = (856 - offset[0], 1057 - offset[1])
    goals = [(1000 + i - offset[0], 1200 - offset[1]) for i in range(100)]

    strt = perf_counter()
    path = c.astar(img[offset[1] : 1300, offset[0] : 1100], start, goals, "any")
    print(f"took {(perf_counter() - strt) * 1000} ms")

    path = [(p[0] + offset[0], p[1] + offset[1]) for p in path]

    drawn = draw_points(img, path)
    show(drawn)


if __name__ == "__main__":
    test_astar_small_img()
    test_astar()
