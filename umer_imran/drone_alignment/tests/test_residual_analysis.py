import numpy as np
import pytest

from drone_alignment.config.schema import QualityConfig
from drone_alignment.quality.metrics import evaluate_spatial_residuals, evaluate_footprint


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


def test_footprint_rejects_catastrophic_translation_shape():
    rgb = np.ones((100, 100), dtype=bool)
    ms = np.ones((100, 100), dtype=bool)
    matrix = np.array([[1, 0, 60], [0, 1, 60]], dtype=np.float64)
    metrics = evaluate_footprint(rgb, ms, matrix, None)
    assert metrics.retained_source_valid_ratio < 0.20
    assert metrics.reference_overlap_ratio < 0.20
