from typing import cast
import numpy as np
import cv2 as cv
from cv2.typing import MatLike
from .split import Split
from .error import TauluException


class PageCropper:
    """
    Very simple utility that crops an image to the smallest rectangular region that
    contains approximately the given colour (hsv).

    This only works if the region of interest actually has a distinct colour from
    the background, but it is very fast thanks.

    The `crop` method returns a single image, the `crop_split` method returns a `Split`
    of the left and right side (useful for when there are two pages next to each other,
    and they need to be processed separately).
    """

    def __init__(
        self,
        target_hue: int = 12,
        target_s: int = 26,
        target_v: int = 230,
        tolerance: int = 40,
        margin: int = 140,
        split: float = 0.5,
        split_margin: float = 0.06,
    ):
        """
        Simple object that crops an input image to the rectangular region that contains the target hsv colour,
        with some tolerance and margin

        Args:
            target_hue, target_s, target_v (int): the hsv colour to match
            tolerance (int): the amount with which h, s, v can differ from the target in order to
                be considered a colour match
            margin (int): margin to add to the rectangle
        """

        self._target_hue = target_hue
        self._target_s = target_s
        self._target_v = target_v
        self._tolerance = tolerance
        self._margin = margin
        self._split = split
        self._split_margin = split_margin

    def _create_hue_mask(self, image: MatLike):
        """
        Creates a mask of an image where pixels are close to a given hue.

        Args:
            image: The input image (NumPy array).
            target_hue: The target hue value (0-180 in OpenCV's HSV).
            tolerance: The tolerance range around the target hue.

        Returns:
            A binary mask (NumPy array) where white pixels (255) represent pixels
            within the hue range, and black pixels (0) represent pixels outside.
        """

        # Convert the image to the HSV color space
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Calculate the lower and upper hue bounds
        lower_hue = np.array(
            [
                max(0, self._target_hue - self._tolerance),
                max(0, self._target_s - self._tolerance),
                max(0, self._target_v - self._tolerance),
            ]
        )
        upper_hue = np.array(
            [
                min(180, self._target_hue + self._tolerance),
                min(255, self._target_s + self._tolerance),
                min(255, self._target_v + self._tolerance),
            ]
        )

        # Create the mask
        mask = cv.inRange(hsv, lower_hue, upper_hue)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((9, 9)))

        return mask

    def _find_bounding_box_with_margin(self, mask: MatLike):
        """
        Finds the smallest bounding box containing all white pixels in a mask,
        with an optional margin.

        Args:
            mask: The binary mask (NumPy array).
            margin: The margin to add around the bounding box (in pixels).

        Returns:
            A tuple (x, y, w, h) representing the bounding box, or None if no white pixels.
        """

        # Find the coordinates of white pixels
        coords = cv.findNonZero(mask)

        if coords is None:
            return None  # No white pixels found

        # Get the bounding rectangle
        x, y, w, h = cv.boundingRect(coords)

        # Apply margin
        x = max(0, x - self._margin)
        y = max(0, y - self._margin)
        w = min(mask.shape[1] - x, w + 2 * self._margin)
        h = min(mask.shape[0] - y, h + 2 * self._margin)

        return x, y, w, h

    def crop_split(
        self, img: MatLike | str
    ) -> tuple[Split[MatLike], Split[tuple[int, int]]]:
        """
        Crops the given image with margin into two,
        one containing the left page, one containing the right page
        (with margin)
        """
        cropped, offset = self.crop(img)
        w = cropped.shape[1]

        cropped_left = cropped[:, : int(w * (self._split + self._split_margin))]
        cropped_right = cropped[:, int(w * (self._split - self._split_margin)) :]

        return Split(cropped_left, cropped_right), Split(
            offset, (offset[0] + int(w * (self._split - self._split_margin)), offset[1])
        )

    def crop(self, img: MatLike | str) -> tuple[MatLike, tuple[int, int]]:
        """
        Crops the given image to the smallest region that contains the target colour

        Returns:
            The cropped image
            The offset in the original image where the crop starts (x, y)
        """

        if type(img) is str:
            img = cv.imread(img)
        img = cast(MatLike, img)

        mask = self._create_hue_mask(img)

        bb = self._find_bounding_box_with_margin(mask)
        if bb is None:
            raise TauluException("couldn't create bounding box")

        x, y, w, h = bb

        if self._split is not None:
            pass

        return img[y : y + h, x : x + w], (x, y)
