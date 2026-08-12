from dataclasses import dataclass
import cv2
import numpy as np
from drone_alignment.config.schema import FeatureDetectionConfig, DetectorType


class InsufficientMatchesError(ValueError):
    """Raised when feature matching produces fewer matches than the required minimum threshold."""
    pass


@dataclass
class MatchResult:
    pts_rgb: np.ndarray          # (N, 2) float32 - (x, y) coordinates in downsampled RGB image
    pts_ms: np.ndarray           # (N, 2) float32 - (x, y) coordinates in downsampled MS image
    num_raw_matches: int
    num_good_matches: int
    match_quality_score: float   # ratio of good_matches / raw_matches


def match_features(
    rgb_det: DetectionResult,
    ms_det: DetectionResult,
    config: FeatureDetectionConfig,
) -> MatchResult:
    """
    Match descriptors using KNN + Lowe's ratio test.
    """
    if len(rgb_det.keypoints) == 0 or len(ms_det.keypoints) == 0:
        raise InsufficientMatchesError(
            f"Keypoint detection yielded 0 keypoints (RGB: {len(rgb_det.keypoints)}, MS: {len(ms_det.keypoints)})."
        )

    if rgb_det.descriptors is None or ms_det.descriptors is None or len(rgb_det.descriptors) == 0 or len(ms_det.descriptors) == 0:
        raise InsufficientMatchesError("Empty descriptor arrays provided to matcher.")

    if config.detector == DetectorType.ORB:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:  # SIFT
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    try:
        raw_matches = matcher.knnMatch(rgb_det.descriptors, ms_det.descriptors, k=2)
    except Exception as e:
        raise InsufficientMatchesError(f"KNN descriptor matching failed: {e}") from e

    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance == 0.0 or m.distance < config.lowe_ratio * n.distance:
                good_matches.append(m)

    num_raw = len(raw_matches)
    num_good = len(good_matches)
    match_quality = num_good / max(1, num_raw)

    if num_good < config.min_good_matches:
        raise InsufficientMatchesError(
            f"Only {num_good} good matches found after Lowe ratio test "
            f"(configured minimum required: {config.min_good_matches})."
        )

    pts_rgb = np.float32([rgb_det.keypoints[m.queryIdx].pt for m in good_matches])
    pts_ms = np.float32([ms_det.keypoints[m.trainIdx].pt for m in good_matches])

    return MatchResult(
        pts_rgb=pts_rgb,
        pts_ms=pts_ms,
        num_raw_matches=num_raw,
        num_good_matches=num_good,
        match_quality_score=match_quality,
    )
