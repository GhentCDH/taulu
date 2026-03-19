"""
Header alignment functionality
"""

import logging
from collections.abc import Iterable
from os import PathLike, fspath
from typing import Literal, cast

import cv2 as cv
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from numpy.typing import NDArray

from . import img_util as imu
from .constants import WINDOW
from .decorators import log_calls
from .error import TauluException

logger = logging.getLogger(__name__)

MatchMethod = Literal["orb", "sift", "surf", "akaze"]


class HeaderAligner:
    """
    Aligns table header templates to subject images using feature-based registration.

    This class supports multiple feature detection and matching methods to compute
    a homography transformation that maps points from a header template image to
    their corresponding locations in full table images.

    ## How it Works

    1. **Feature Detection**: Extracts keypoints from both template and subject
    2. **Feature Matching**: Finds correspondences using the selected matcher
    3. **Filtering**: Keeps top matches and prunes based on spatial consistency
    4. **Homography Estimation**: Computes perspective transform using RANSAC

    The computed homography can then transform any point from template space to
    image space, allowing you to locate table structures based on your annotation.

    ## Available Methods

    - **orb** (default): ORB features with BFMatcher (Hamming distance). Fast and
      patent-free. Good for most use cases.
    - **sift**: SIFT features with FLANN-based matcher. More robust to scale and
      rotation changes. Slower but often more accurate.
    - **surf**: SURF features with BFMatcher (L2 norm). Requires opencv-contrib-python
      with non-free modules enabled. Fast and robust.
    - **akaze**: AKAZE features with BFMatcher (Hamming distance). Patent-free,
      handles scale/rotation well, and often more robust than ORB on documents.

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
        method (MatchMethod): Feature detection/matching method. One of "orb", "sift",
            or "surf". Default is "orb".
        max_features (int): Maximum features to detect. More features = slower
            but potentially more robust matching.
        patch_size (int): ORB patch size for feature extraction (only used with "orb").
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
        method: MatchMethod = "orb",
        max_features: int = 100_000,
        patch_size: int = 31,
        match_fraction: float = 0.3,
        scale: float = 1.0,
        max_dist: float = 1.00,
        k: float | None = None,
    ):
        """
        Args:
            template (MatLike | str): (path of) template image, with the table template clearly visible
            method (MatchMethod): feature detection/matching method ("orb", "sift", or "surf")
            max_features (int): maximal number of features that will be extracted
            patch_size (int): for ORB feature extractor (only used with method="orb")
            match_fraction (float): best fraction of matches that are kept
            scale (float): image scale factor to do calculations on (useful for increasing calculation speed mostly)
            max_dist (float): maximum distance (relative to image size) of matched features.
                Increase this value if the warping between image and template needs to be more agressive
            k (float | None): sauvola thresholding threshold value. If None, no sauvola thresholding is done
        """

        if type(template) is str or type(template) is PathLike:
            value = cv.imread(fspath(template))
            template = value

        self._method = method
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
        self._validate_method()
        self._matches_notebook_img = None

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
    def method(self) -> MatchMethod:
        """The feature detection/matching method being used."""
        return self._method

    @property
    def template(self):
        """The template image that subject images are aligned to"""
        return self._template

    @template.setter
    def template(self, value: MatLike | str):
        """Set the template image as a path or an image"""

        if type(value) is str:
            tmp_value = cv.imread(value)
            assert tmp_value is not None
            value = tmp_value
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

    def _validate_method(self):
        """Validate that the selected method is available."""
        if self._method == "surf":
            if not hasattr(cv, "xfeatures2d"):
                raise TauluException(
                    "SURF requires opencv-contrib-python with non-free modules. "
                    "Install with: pip install opencv-contrib-python"
                )

    def _create_detector(self):
        """Create the feature detector based on the selected method."""
        if self._method == "orb":
            return cv.ORB_create(  # type:ignore
                self._max_features,
                patchSize=self._patch_size,
            )
        elif self._method == "sift":
            return cv.SIFT_create(  # type:ignore
                nfeatures=self._max_features, sigma=2.5, edgeThreshold=10
            )
        elif self._method == "akaze":
            return cv.AKAZE_create()  # type:ignore
        elif self._method == "surf":
            # SURF is in xfeatures2d (requires opencv-contrib-python)
            return cv.xfeatures2d.SURF_create(hessianThreshold=400)  # type:ignore
        else:
            raise TauluException(f"Unknown method: {self._method}")

    def _create_matcher(self):
        """Create the feature matcher based on the selected method."""
        if self._method == "orb":
            # ORB uses binary descriptors -> Hamming distance
            return cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        elif self._method == "sift":
            # SIFT uses float descriptors -> L2 norm with crossCheck
            return cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        elif self._method == "akaze":
            # AKAZE uses binary descriptors -> Hamming distance
            return cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        elif self._method == "surf":
            # SURF uses float descriptors -> L2 norm
            return cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        else:
            raise TauluException(f"Unknown method: {self._method}")

    def _match_features(self, matcher, descriptors_im, descriptors_tg):
        """Match features using BFMatcher with crossCheck for all methods."""
        return list(matcher.match(descriptors_im, descriptors_tg))

    @log_calls(level=logging.DEBUG, include_return=True)
    def _find_transform_of_template_on(
        self,
        im: MatLike,
        visual: bool = False,
        visual_notebook: bool = False,
        window: str = WINDOW,
    ):
        im = self._scale_img(im)

        # Create detector and matcher based on selected method
        detector = self._create_detector()
        matcher = self._create_matcher()

        # Detect features and compute descriptors
        keypoints_im, descriptors_im = detector.detectAndCompute(im, None)
        keypoints_tg, descriptors_tg = detector.detectAndCompute(self._template, None)

        if descriptors_im is None or descriptors_tg is None:
            raise TauluException("No features detected in one or both images")

        # Match features
        matches = self._match_features(matcher, descriptors_im, descriptors_tg)

        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance)

        # Remove not so good matches
        num_good_matches = int(len(matches) * self._match_fraction)
        matches = matches[:num_good_matches]

        if visual or visual_notebook:
            final_img_filtered = cv.drawMatches(
                im,
                keypoints_im,
                self._template,
                keypoints_tg,
                matches[:10],
                None,
                cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            if visual:
                imu.show(final_img_filtered, title="matches", window=window)
            if visual_notebook:
                self._matches_notebook_img = final_img_filtered

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
        mask_array = mask.to_numpy()
        points1 = points1[mask_array]
        points2 = points2[mask_array]

        # Filter matches for visualization
        filtered_matches = [
            m for m, keep in zip(matches, mask_array, strict=False) if keep
        ]

        if visual:
            final_img_filtered = cv.drawMatches(
                im,
                keypoints_im,
                self._template,
                keypoints_tg,
                filtered_matches[:100],
                None,
                cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            imu.show(final_img_filtered, title="matches", window=window)

        # Find homography
        h, _ = cv.findHomography(points1, points2, cv.RANSAC)

        return self._unscale_homography(h)

    def show_matches_notebook(self):
        """Display the stored feature matches image in the notebook (call after grid detection)."""
        if self._matches_notebook_img is not None:
            imu.show_notebook(self._matches_notebook_img, title="matches")
            self._matches_notebook_img = None

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
        self,
        img: MatLike | str,
        visual: bool = False,
        visual_notebook: bool = False,
        window: str = WINDOW,
    ) -> NDArray:
        """
        Calculates a homogeneous transformation matrix that maps pixels of
        the template to the given image
        """

        logger.info("Aligning header with supplied table image")

        if type(img) is str:
            tmp_img = cv.imread(img)
            assert tmp_img is not None
            img = tmp_img
        img = cast(MatLike, img)

        img = self._preprocess_image(img)

        h = self._find_transform_of_template_on(img, visual, visual_notebook, window)

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
        transformed = np.dot(h, point.T)

        transformed /= transformed[2]

        return int(transformed[0][0]), int(transformed[1][0])
