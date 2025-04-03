from tabular.img_util import sauvola, show
import cv2 as cv

def test_threshold():
    im = cv.imread("/home/mielpeeters/code/ancestors-tale/data/imgs/format_2/1890/IMG_2024_01_24_09_10_50S.png")
    result = sauvola(im)
    show(result)

