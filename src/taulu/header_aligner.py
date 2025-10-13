"""
Header alignment functionality
"""

from os import PathLike, fspath
import cv2 as cv
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Iterable, cast
from cv2.typing import MatLike
import logging

from .decorators import log_calls
from .constants import WINDOW
from .error import TauluException
from . import img_util as imu

logger = logging.getLogger(__name__)


class HeaderAligner:
    """
    Aligns table header templates to subject images using feature-based registration.
    
    This class uses ORB (Oriented FAST and Rotated BRIEF) feature detection and
    matching to compute a homography transformation that maps points from a header
    template image to their corresponding locations in full table images.
    
    ## How it Works
    
    1. **Feature Detection**: Extracts ORB keypoints from both template and subject
    2. **Feature Matching**: Finds correspondences using Hamming distance
    3. **Filtering**: Keeps top matches and prunes based on spatial consistency
    4. **Homography Estimation**: Computes perspective transform using RANSAC
    
    The computed homography can then transform any point from template space to
    image space, allowing you to locate table structures based on your annotation.
    
    ## Preprocessing Options
    
    - Set `k` parameter to apply Sauvola thresholding before feature detection.
      This can improve matching on documents with variable lighting.
    - Set `k=None` to use raw images (just extract blue channel for BGR images)
    
    ## Tuning Guidelines
    
    - **max_features**: Increase if matching fails on complex templates
    - **match_fraction**: Decrease if you get many incorrect matches
    - **max_dist**: Increase for documents with more warping/distortion
    - **scale**: Decrease (<1.0) to speed up on high-resolution images
    
    Args:
        template (MatLike | PathLike[str] | str | None): Header template image or path.
            This should contain a clear, representative view of the table header.
        max_features (int): Maximum ORB features to detect. More features = slower
            but potentially more robust matching.
        patch_size (int): ORB patch size for feature extraction.
        match_fraction (float): Fraction [0, 1] of matches to keep after sorting by
            quality. Higher = more matches but potentially more outliers.
        scale (float): Image downscaling factor (0, 1] for processing speed.
        max_dist (float): Maximum allowed distance (relative to image size) between
            matched keypoints. Filters out spatially inconsistent matches.
        k (float | None): Sauvola threshold parameter for preprocessing. If None,
            no thresholding is applied. Typical range: 0.03-0.15.
    """

    def __init__(
        self,
        template: None | MatLike | PathLike[str] | str = None,
        max_features: int = 25_000,
        patch_size: int = 31,
        match_fraction: float = 0.6,
        scale: float = 1.0,
        max_dist: float = 1.00,
        k: float | None = 0.05,
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

        if type(template) is str or type(template) is PathLike:
            value = cv.imread(fspath(template))
            template = value

        self._k = k
        if scale > 1.0:
            raise TauluException(
                "Scaling up the image for header alignment is useless. Use 0 < scale <= 1.0"
            )
        if scale == 0:
            raise TauluException("Use 0 < scale <= 1.0")

        self._scale = scale
        self._template = self._scale_img(cast(MatLike, template))
        self._template_orig: None | MatLike = None
        self._preprocess_template()
        self._max_features = max_features
        self._patch_size = patch_size
        self._match_fraction = match_fraction
        self._max_dist = max_dist

    def _scale_img(self, img: MatLike) -> MatLike:
        if self._scale == 1.0:
            return img

        return cv.resize(img, None, fx=self._scale, fy=self._scale)

    def _unscale_img(self, img: MatLike) -> MatLike:
        if self._scale == 1.0:
            return img

        return cv.resize(img, None, fx=1 / self._scale, fy=1 / self._scale)

    def _unscale_homography(self, h: np.ndarray) -> np.ndarray:
        if self._scale == 1.0:
            return h

        scale_matrix = np.diag([self._scale, self._scale, 1.0])
        # inv_scale_matrix = np.linalg.inv(scale_matrix)
        inv_scale_matrix = np.diag([1.0 / self._scale, 1.0 / self._scale, 1.0])
        # return inv_scale_matrix @ h @ scale_matrix
        return inv_scale_matrix @ h @ scale_matrix

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

    @log_calls(level=logging.DEBUG, include_return=True)
    def _find_transform_of_template_on(
        self, im: MatLike, visual: bool = False, window: str = WINDOW
    ):
        im = self._scale_img(im)
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

        return self._unscale_homography(h)

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
        header = imu.ensure_gray(self._unscale_img(self._template))
        height, width = im.shape

        header_warped = cv.warpPerspective(header, h, (width, height))

        merged = np.full((height, width, 3), 255, dtype=np.uint8)

        merged[..., 1] = im
        merged[..., 2] = header_warped

        return imu.show(merged)

    @log_calls(level=logging.DEBUG, include_return=True)
    def align(
        self, img: MatLike | str, visual: bool = False, window: str = WINDOW
    ) -> NDArray:
        """
        Calculates a homogeneous transformation matrix that maps pixels of
        the template to the given image
        """

        logger.info("Aligning header with supplied table image")

        if type(img) is str:
            img = cv.imread(img)
        img = cast(MatLike, img)

        img = self._preprocess_image(img)

        h = self._find_transform_of_template_on(img, visual, window)

        if visual:
            self.view_alignment(img, h)

        return h

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
