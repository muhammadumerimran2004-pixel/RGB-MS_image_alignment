import numpy as np
import pytest

from drone_alignment.config.schema import FeatureDetectionConfig, DetectorType
from drone_alignment.alignment.feature_detector import detect_features, normalize_to_uint8
from drone_alignment.alignment.feature_matcher import match_features, InsufficientMatchesError


def test_normalize_to_uint8():
    arr = np.array([0.0, 1.0, 2.0, 10.0, 100.0], dtype=np.float32)
    norm = normalize_to_uint8(arr)
    assert norm.dtype == np.uint8
    assert norm.shape == arr.shape


def test_detect_and_match_synthetic():
    # Create 2D images with distinct grid features
    h, w = 256, 256
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    pattern = (((xx // 16) + (yy // 16)) % 2) * 200.0 + 30.0
    pattern = pattern.astype(np.float32)

    cfg = FeatureDetectionConfig(detector=DetectorType.ORB, max_keypoints=2000, min_good_matches=10)
    rgb_det, ms_det = detect_features(pattern, pattern, cfg)

    assert len(rgb_det.keypoints) > 0
    assert len(ms_det.keypoints) > 0

    match_res = match_features(rgb_det, ms_det, cfg)
    assert match_res.num_good_matches >= 10


def test_normalization_excludes_alpha_invalid_uint32_sentinel():
    image = np.full((20, 20), 0.05, dtype=np.float64)
    image[:, :2] = 2**32
    image[5:15, 5:15] = np.linspace(0.02, 0.12, 100).reshape(10, 10)
    valid = np.ones(image.shape, dtype=bool)
    valid[:, :2] = False

    normalized = normalize_to_uint8(image, valid)

    assert np.all(normalized[:, :2] == 0)
    assert normalized[valid].max() > 200
    assert np.unique(normalized[valid]).size > 20
