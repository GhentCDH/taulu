import cv2 as cv
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Iterable, cast
from cv2.typing import MatLike
from pathlib import Path
import logging

from .decorators import log_calls
from .constants import WINDOW
from .error import TauluException
from . import img_util as imu


class HeaderAligner:
    """
    Calculates a transformation matrix to transform points from header-template-image-space to
    subject-image-space.
    """

    def __init__(
        self,
        template: None | MatLike | str | Path = None,
        max_features: int = 25_000,
        patch_size: int = 31,
        match_fraction: float = 0.6,
        scale: float = 1.0,
        max_dist: float = 1.00,
        k: float | None = None,
    ):
        """
        Args:
            template (MatLike | str): (path of) template image, with the table template clearly visible
            max_features (int): maximal number of features that will be extracted by ORB
            patch_size (int): for ORB feature extractor
            match_fraction (float): best fraction of matches that are kept
            scale (float): image scale factor to do calculations on (useful for increasing calculation speed mostly)
            max_dist (float): maximum distance (relative to image size) of matched features.
                Increase this value if the warping between image and template needs to be more agressive
            k (float | None): sauvola thresholding threshold value. If None, no sauvola thresholding is done
        """

        if type(template) is str or type(template) is Path:
            value = cv.imread(template)
            template = value

        self._k = k
        self._template: MatLike = cast(MatLike, template)
        self._template_orig: None | MatLike = None
        self._preprocess_template()
        self._max_features = max_features
        self._patch_size = patch_size
        self._match_fraction = match_fraction
        self._scale = scale
        self._max_dist = max_dist

    @property
    def template(self):
        """The template image that subject images are aligned to"""
        return self._template

    @template.setter
    def template(self, value: MatLike | str):
        """Set the template image as a path or an image"""

        if type(value) is str:
            value = cv.imread(value)
            self._template = value

        # TODO: check if the image has the right properties (dimensions etc.)
        self._template = cast(MatLike, value)

        self._preprocess_template()

    def _preprocess_template(self):
        self._template_orig = cv.cvtColor(self._template, cv.COLOR_BGR2GRAY)
        if self._k is not None:
            self._template = imu.sauvola(self._template, self._k)
            self._template = cv.bitwise_not(self._template)
        else:
            _, _, self._template = cv.split(self._template)

    def _preprocess_image(self, img: MatLike):
        if self._template_orig is None:
            raise TauluException("process the template first")

        if self._k is not None:
            img = imu.sauvola(img, self._k)
            img = cv.bitwise_not(img)
        else:
            _, _, img = cv.split(img)

        return img

    @log_calls(logging.DEBUG, include_return=True)
    def _find_transform_of_template_on(
        self, im: MatLike, visual: bool = False, window: str = WINDOW
    ):
        # Detect ORB features and compute descriptors.
        orb = cv.ORB_create(
            self._max_features,  # type:ignore
            patchSize=self._patch_size,
        )
        keypoints_im, descriptors_im = orb.detectAndCompute(im, None)
        keypoints_tg, descriptors_tg = orb.detectAndCompute(self._template, None)

        # Match features
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_im, descriptors_tg)

        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self._match_fraction)
        matches = matches[:numGoodMatches]

        if visual:
            final_img_filtered = cv.drawMatches(
                im,
                keypoints_im,
                self._template,
                keypoints_tg,
                matches[:10],
                None,  # type:ignore
                cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            imu.show(final_img_filtered, title="matches", window=window)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints_tg[match.trainIdx].pt
            points2[i, :] = keypoints_im[match.queryIdx].pt

        # Prune reference points based upon distance between
        # key points. This assumes a fairly good alignment to start with
        # due to the protocol used (location of the sheets)
        p1 = pd.DataFrame(data=points1)
        p2 = pd.DataFrame(data=points2)
        refdist = abs(p1 - p2)

        mask_x = refdist.loc[:, 0] < (im.shape[0] * self._max_dist)
        mask_y = refdist.loc[:, 1] < (im.shape[1] * self._max_dist)
        mask = mask_x & mask_y
        points1 = points1[mask.to_numpy()]
        points2 = points2[mask.to_numpy()]

        # Find homography
        h, _ = cv.findHomography(points1, points2, cv.RANSAC)

        return h

    def view_alignment(self, img: MatLike, h: NDArray):
        """
        Show the alignment of the template on the given image
        by transforming it using the supplied transformation matrix `h`
        and visualising both on different channels

        Args:
            img (MatLike): the image on which the template is transformed
            h (NDArray): the transformation matrix
        """

        im = imu.ensure_gray(img)
        header = imu.ensure_gray(self._template)
        height, width = im.shape

        header_warped = cv.warpPerspective(header, h, (width, height))

        merged = np.full((height, width, 3), 255, dtype=np.uint8)

        merged[..., 1] = im
        merged[..., 2] = header_warped

        return imu.show(merged)

    @log_calls(logging.DEBUG, include_return=True)
    def align(
        self, img: MatLike | str, visual: bool = False, window: str = WINDOW
    ) -> NDArray:
        """
        Calculates a homogeneous transformation matrix that maps pixels of
        the template to the given image
        """

        if type(img) is str:
            img = cv.imread(img)
        img = cast(MatLike, img)

        img = self._preprocess_image(img)

        return self._find_transform_of_template_on(img, visual, window)

    def template_to_img(self, h: NDArray, point: Iterable[int]) -> tuple[int, int]:
        """
        Transform the given point (in template-space) using the transformation h
        (obtained through the `align` method)

        Args:
            h (NDArray): transformation matrix of shape (3, 3)
            point (Iterable[int]): the to-be-transformed point, should conform to (x, y)
        """

        point = np.array([[point[0], point[1], 1]])  # type:ignore
        transformed = np.dot(h, point.T)  # type:ignore

        transformed /= transformed[2]

        return int(transformed[0][0]), int(transformed[1][0])
