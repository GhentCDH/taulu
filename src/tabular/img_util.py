import cv2 as cv
from skimage.filters import threshold_sauvola
import numpy as np

from .constants import WINDOW

def show(image, click_event = None, title: str | None = None):
    """
    shows an image to the user, who then gets the option to press q or n, 
    determining the output of the function

    pressing 'q' exits with code 0, 'n' returns from this function
    """

    try: 
        cv.namedWindow(WINDOW, cv.WINDOW_NORMAL)
    except cv.error:
        # window already exists
        pass

    image = np.copy(image)

    text = "<q> quit <n> next"
    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN, 2.0, 2)[0]
    image_height, image_width = image.shape[:2]
    text_x = (image_width - text_size[0]) // 2
    text_y = image_height - 10  # 10 pixels from the bottom
    position = (text_x, text_y)
    cv.putText(image, text, position, cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)

    if title:
        text_size = cv.getTextSize(title, cv.FONT_HERSHEY_PLAIN, 2.0, 2)[0]
        position = ((image.shape[1] - text_size[0]) // 2, 10 + text_size[1])
        cv.putText(image, title, position, cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)

    cv.imshow(WINDOW, image)
    if click_event:
        cv.setMouseCallback(WINDOW, click_event)

    while True:
        key = cv.waitKey(10)
        if key == ord("q"):
            exit(0)
        if key == ord("n"):
            break
        elif key == -1 and cv.getWindowProperty(WINDOW, cv.WND_PROP_VISIBLE) < 1: # Check if the window is actually closed
            break

def vertical_edges(img, x = 6):
    canny = cv.Canny(img, 50, 255, None)

    vertical = cv.morphologyEx(canny, cv.MORPH_DILATE, np.ones((1, x)))
    vertical = cv.morphologyEx(vertical, cv.MORPH_ERODE, np.ones((x, 1)))
    vertical = cv.morphologyEx(vertical, cv.MORPH_ERODE, np.ones((x, x)))

    return vertical

def horizontal_edges(img):
    canny = cv.Canny(img, 50, 255, None)

    y = 5
    horizontal = cv.morphologyEx(canny, cv.MORPH_DILATE, np.ones((y, 1)))
    horizontal = cv.morphologyEx(horizontal, cv.MORPH_ERODE, np.ones((1, y)))
    horizontal = cv.morphologyEx(horizontal, cv.MORPH_ERODE, np.ones((y, y)))

    return horizontal

def ensure_gray(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img

def clahe(img):
    img = ensure_gray(img)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(15,15))
    return clahe.apply(img)

def blur(img, blur_size: int = 7):
    return cv.GaussianBlur(img,(blur_size, blur_size),0)

def clahe_otsu(img, size: int = 12, limit: float = 8.0):
    gray = ensure_gray(img)
    clahe = cv.createCLAHE(clipLimit=limit, tileGridSize=(size, size))
    enhanced = clahe.apply(gray)
    _, binary = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return cv.bitwise_not(binary)

def sauvola(img, k: float = 0.04, window_size: int = 15):
    gray = ensure_gray(img)
    sauvola_thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary_sauvola = (gray > sauvola_thresh).astype(np.uint8) * 255
    binary_sauvola = cv.bitwise_not(binary_sauvola)

    return binary_sauvola

def _rm_v_edges(img):
    edges = vertical_edges(img, x = 8)
    edges_n = cv.bitwise_not(edges)

    img = cv.bitwise_not(img)
    masked = cv.bitwise_and(img, img, mask=edges_n, dst=None)
    masked = cv.bitwise_not(masked)

    masked = cv.dilate(masked, np.ones((2, 2), dtype=np.uint8))

    return masked 

def text_presence_score(img) -> float:
    img = sauvola(img)
    masked = _rm_v_edges(img)

    black_pixels_fraction = np.sum(masked == 0) / masked.size

    return min((black_pixels_fraction - 0.020) / 0.1, 1)
    
