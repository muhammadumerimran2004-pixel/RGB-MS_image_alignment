import numpy as np
import pytest

from drone_alignment.config.schema import TransformValidationConfig, FeatureDetectionConfig, TransformType
from drone_alignment.alignment.feature_matcher import MatchResult
from drone_alignment.alignment.transform_estimator import (
    estimate_transform, TransformUnreliableError, _fallback_phase_correlation, validate_phase_transform,
)


def test_estimate_transform_known_shift():
    # Known shift: dx = 5px, dy = -3px
    pts_ms = np.array([[10, 10], [50, 10], [50, 50], [10, 50], [30, 30]], dtype=np.float32)
    pts_rgb = pts_ms + np.array([5.0, -3.0], dtype=np.float32)

    match_res = MatchResult(
        pts_rgb=pts_rgb,
        pts_ms=pts_ms,
        num_raw_matches=5,
        num_good_matches=5,
        match_quality_score=1.0,
    )

    t_cfg = TransformValidationConfig(transform_type=TransformType.AFFINE, max_translation_m=10.0)
    f_cfg = FeatureDetectionConfig()

    result = estimate_transform(
        match_res,
        scale_factor=1.0,
        target_gsd=0.06,
        config=t_cfg,
        feature_config=f_cfg,
    )

    assert pytest.approx(result.translation_px[0], abs=0.1) == 5.0
    assert pytest.approx(result.translation_px[1], abs=0.1) == -3.0
    assert pytest.approx(result.rotation_deg, abs=0.1) == 0.0


def test_estimate_transform_violates_max_translation():
    pts_ms = np.array([[10, 10], [50, 10], [50, 50], [10, 50], [30, 30]], dtype=np.float32)
    # Huge shift of 1000px
    pts_rgb = pts_ms + np.array([1000.0, 1000.0], dtype=np.float32)

    match_res = MatchResult(
        pts_rgb=pts_rgb,
        pts_ms=pts_ms,
        num_raw_matches=5,
        num_good_matches=5,
        match_quality_score=1.0,
    )

    t_cfg = TransformValidationConfig(max_translation_m=5.0)
    f_cfg = FeatureDetectionConfig()

    with pytest.raises(TransformUnreliableError, match="exceeds max allowed"):
        estimate_transform(
            match_res,
            scale_factor=1.0,
            target_gsd=0.06,
            config=t_cfg,
            feature_config=f_cfg,
        )


def test_fallback_phase_correlation():
    img_rgb = np.zeros((100, 100), dtype=np.float32)
    img_rgb[40:60, 40:60] = 100.0

    img_ms = np.zeros((100, 100), dtype=np.float32)
    img_ms[42:62, 38:58] = 100.0  # Shifted dy=+2, dx=-2

    matrix = _fallback_phase_correlation(img_rgb, img_ms)
    assert matrix.shape == (2, 3)
    assert pytest.approx(matrix[0, 2], abs=0.25) == -2.0
    assert pytest.approx(matrix[1, 2], abs=0.25) == 2.0


def test_phase_candidate_cannot_bypass_translation_limit():
    matrix = np.array([[1.0, 0.0, 1000.0], [0.0, 1.0, 1000.0]])
    cfg = TransformValidationConfig(max_translation_m=5.0)
    with pytest.raises(TransformUnreliableError, match="exceeds max allowed"):
        validate_phase_transform(matrix, response=0.9, target_gsd=0.06, config=cfg)


def test_phase_candidate_requires_response():
    matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    cfg = TransformValidationConfig(min_phase_correlation_response=0.5)
    with pytest.raises(TransformUnreliableError, match="response"):
        validate_phase_transform(matrix, response=0.1, target_gsd=0.06, config=cfg)
