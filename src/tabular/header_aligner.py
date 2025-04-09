import cv2 as cv
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import cast
from cv2.typing import MatLike

from .error import TabularException
from . import img_util as imu


class HeaderAligner:
    def __init__(
        self,
        template: None | MatLike | str = None,
        max_features: int = 25_000,
        patch_size: int = 31,
        match_fraction: float = 0.6,
        scale: float = 1.0,
        max_dist: float = 1.00,
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
        """

        if type(template) is str:
            value = cv.imread(template)
            template = value

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
        _, _, self._template = cv.split(self._template)

    def _preprocess_image(self, img: MatLike):
        if self._template_orig is None:
            raise TabularException("process the template first")

        img_orig = np.copy(img)
        _, _, img = cv.split(img)
        # img = imu.threshold(img)

        # img = cv.resize(
        #     img, (self._template.shape[1], self._template.shape[0]))
        # img_orig = cv.resize(
        #     img_orig, (self._template_orig.shape[1], self._template_orig.shape[0]))

        return img, img_orig

    def _align_images(self, im: MatLike, im_original: MatLike):
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

        final_img_filtered = cv.drawMatches(
            im,
            keypoints_im,
            self._template,
            keypoints_tg,
            matches[:10],
            None,
            cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )  # type:ignore
        imu.show(final_img_filtered, title="matches")

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

    def view_alignment(self, warped: MatLike):
        # create an alignment preview
        sz = warped.shape
        im_preview = np.full((sz[0], sz[1], 3), 255, dtype=np.uint8)
        im_preview[:, :, 1] = warped[:, :, 2]
        im_preview[:, :, 2] = self._template_orig

        return imu.show(im_preview, title="template alignment")

    def align(self, img: MatLike | str) -> NDArray:
        if type(img) is str:
            img = cv.imread(img)
        img = cast(MatLike, img)

        img, img_orig = self._preprocess_image(img)
        matrix = self._align_images(img, img_orig)

        return matrix

    def find_point_of_template_in_img(
        self, matrix: NDArray, point: tuple[int, int]
    ) -> tuple[int, int]:
        point = np.array([[point[0], point[1], 1]])
        transformed = np.dot(matrix, point.T)

        transformed /= transformed[2]

        return int(transformed[0]), int(transformed[1])
