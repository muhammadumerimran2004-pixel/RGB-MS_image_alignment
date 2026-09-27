import numpy as np
import pytest

from drone_alignment.config.schema import QualityConfig, TransformValidationConfig
from drone_alignment.quality.metrics import evaluate_spatial_residuals, evaluate_footprint, footprint_gate_failures


def test_evaluate_spatial_residuals_perfect():
    pts_ref = np.array([[10, 10], [50, 50], [90, 90]], dtype=np.float32)
    pts_warped = pts_ref.copy()

    cfg = QualityConfig(max_acceptable_rmse_px=2.0)
    report = evaluate_spatial_residuals(pts_ref, pts_warped, (100, 100), cfg)

    assert report.global_rmse_px == 0.0
    assert report.status == "PASS"
    assert report.is_spatial_drift_acceptable is True


def test_evaluate_spatial_residuals_edge_drift():
    # Center points have low error (0.1px), edge points have large error (4.0px)
    pts_ref = np.array([[50, 50], [52, 52], [5, 5], [95, 95]], dtype=np.float32)
    pts_warped = np.array([[50.1, 50.0], [52.0, 52.1], [9.0, 5.0], [95.0, 99.0]], dtype=np.float32)

    cfg = QualityConfig(max_acceptable_rmse_px=5.0, max_acceptable_edge_rmse_px=3.0)
    report = evaluate_spatial_residuals(pts_ref, pts_warped, (100, 100), cfg)

    assert report.edge_corner_rmse_px > 3.0
    assert report.is_spatial_drift_acceptable is False


def _grid_points(n_side: int = 10, size: float = 1000.0) -> np.ndarray:
    coords = np.linspace(50, size - 50, n_side)
    return np.array([[x, y] for x in coords for y in coords], dtype=np.float64)


def test_one_false_match_no_longer_fails_a_good_transform():
    """Regression for real Farm B data: a transform fitting 310 of 311 held-out
    matches within ~2px was rejected at 96px RMSE because of one false match
    1653px away. Screening excludes it from the RMSE and counts it against agreement."""
    pts_ref = _grid_points()
    pts_warped = pts_ref + 1.0
    pts_warped[37] += 1650.0

    unscreened = evaluate_spatial_residuals(pts_ref, pts_warped, (1000, 1000), QualityConfig())
    assert unscreened.status == "FAIL"
    assert unscreened.global_rmse_px > 100

    screened = evaluate_spatial_residuals(pts_ref, pts_warped, (1000, 1000), QualityConfig(), screen_false_matches=True)
    assert screened.status == "PASS"
    assert screened.global_rmse_px == pytest.approx(np.sqrt(2.0))
    assert screened.gross_mismatch_count == 1
    assert screened.holdout_agreement_ratio == pytest.approx(0.99)


def test_screening_still_fails_a_wrong_transform_on_agreement():
    """A wrong transform can't pass by having its disagreeing points screened out:
    the fraction that still agrees must meet min_holdout_agreement."""
    pts_ref = _grid_points()
    pts_warped = pts_ref + 40.0
    pts_warped[:20] = pts_ref[:20] + 0.5  # 20% happen to agree

    report = evaluate_spatial_residuals(pts_ref, pts_warped, (1000, 1000), QualityConfig(), screen_false_matches=True)
    assert report.status == "FAIL"
    assert report.holdout_agreement_ratio == pytest.approx(0.20)
    assert report.gross_mismatch_count == 80


def test_screening_still_applies_the_rmse_limits_to_agreeing_points():
    """Screening only removes gross mismatches; a transform that is consistently
    ~5px off (within the gross cutoff) must still fail the 2px limit."""
    pts_ref = _grid_points()
    pts_warped = pts_ref + np.array([3.0, 4.0])

    report = evaluate_spatial_residuals(pts_ref, pts_warped, (1000, 1000), QualityConfig(), screen_false_matches=True)
    assert report.gross_mismatch_count == 0
    assert report.global_rmse_px == pytest.approx(5.0)
    assert report.status == "FAIL"


def test_unscreened_default_keeps_large_residuals_for_manual_control_points():
    """Manual GCPs use the default (no screening): a 50px error is a real mistake."""
    pts_ref = _grid_points(4)
    pts_warped = pts_ref.copy()
    pts_warped[0] += 50.0
    report = evaluate_spatial_residuals(pts_ref, pts_warped, (1000, 1000), QualityConfig())
    assert report.status == "FAIL"
    assert report.gross_mismatch_count == 0
    assert report.holdout_agreement_ratio is None


def test_footprint_rejects_catastrophic_translation_shape():
    rgb = np.ones((100, 100), dtype=bool)
    ms = np.ones((100, 100), dtype=bool)
    matrix = np.array([[1, 0, 60], [0, 1, 60]], dtype=np.float64)
    metrics = evaluate_footprint(rgb, ms, matrix, None)
    assert metrics.retained_source_valid_ratio < 0.20
    assert metrics.reference_overlap_ratio < 0.20
    assert footprint_gate_failures(metrics, TransformValidationConfig())


def test_uncovered_rgb_does_not_fail_a_correct_transform():
    """Regression for Farm B A007: the RGB covers ground the MS flight never did.
    A correct transform covered only ~79% of the RGB and was rejected; the gate now
    asks whether the aligned MS lands on real RGB, which it fully does."""
    rgb = np.ones((100, 100), dtype=bool)
    ms = np.zeros((100, 100), dtype=bool)
    ms[20:70, 10:80] = True
    matrix = np.array([[1, 0, 5], [0, 1, 3]], dtype=np.float64)
    metrics = evaluate_footprint(rgb, ms, matrix, None)

    assert metrics.reference_overlap_ratio < 0.40
    assert metrics.target_overlap_ratio == pytest.approx(1.0)
    assert footprint_gate_failures(metrics, TransformValidationConfig()) == []


def test_ms_extending_past_rgb_coverage_does_not_fail():
    """Regression for FAI-AAA0106: the MS flight covers ground the RGB never did (only
    ~72% of the MS lands on valid RGB). Coverage is reported, not gated; residual QA
    judges accuracy on the shared ground."""
    rgb = np.zeros((100, 100), dtype=bool)
    rgb[:, :50] = True
    ms = np.zeros((100, 100), dtype=bool)
    ms[20:70, 10:40] = True
    matrix = np.array([[1, 0, 25], [0, 1, 0]], dtype=np.float64)
    metrics = evaluate_footprint(rgb, ms, matrix, None)

    assert metrics.retained_source_valid_ratio == pytest.approx(1.0)
    assert metrics.target_overlap_ratio < 0.80
    assert footprint_gate_failures(metrics, TransformValidationConfig()) == []
