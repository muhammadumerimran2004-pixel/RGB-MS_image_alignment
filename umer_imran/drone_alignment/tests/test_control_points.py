import numpy as np
import pytest

from drone_alignment.alignment.control_points import (
    ControlPointSet, GeometricModel,
    _apply_affine, apply_outlier_policy, build_control_point_set, check_bounds,
    evaluate_jacobian, fit_linear, linear_parameters_at_centroid, loo_rmse,
    loo_residual_vectors, robust_residuals, select_model, validate_geometry,
)
from drone_alignment.alignment.transform_estimator import TransformUnreliableError


def _cps(ref, tgt, is_check=None, ids=None):
    ref = np.asarray(ref, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    ids = ids or [f"P{i}" for i in range(len(ref))]
    return build_control_point_set(ids, ref, tgt, is_check)


class _FakeManualConfig:
    def __init__(self, model="auto", min_tps_points=8, min_hull_coverage=0.35, min_tps_gain=0.15):
        self.model = model
        self.min_tps_points = min_tps_points
        self.min_hull_coverage = min_hull_coverage
        self.min_tps_gain = min_tps_gain


class _FakeBounds:
    def __init__(self, max_translation_m=50.0, max_rotation_deg=5.0, max_scale_deviation=0.1):
        self.max_translation_m = max_translation_m
        self.max_rotation_deg = max_rotation_deg
        self.max_scale_deviation = max_scale_deviation


# --- fit_linear -----------------------------------------------------------

def test_fit_linear_translation_is_mean_difference():
    ref = np.array([(10.0, 20.0), (30.0, 40.0), (12.0, 22.0)])
    tgt = ref - np.array([5.0, 3.0])
    matrix = fit_linear(_cps(ref, tgt), GeometricModel.TRANSLATION)
    np.testing.assert_allclose(matrix, [[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]])


def test_fit_linear_similarity_recovers_umeyama_rotation_scale():
    tgt = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])
    theta = np.radians(15.0)
    scale = 1.4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    translation = np.array([3.0, -2.0])
    ref = (scale * (rot @ tgt.T)).T + translation
    matrix = fit_linear(_cps(ref, tgt), GeometricModel.SIMILARITY)
    predicted = _apply_affine(matrix, tgt)
    np.testing.assert_allclose(predicted, ref, atol=1e-8)
    recovered_scale = np.hypot(matrix[0, 0], matrix[1, 0])
    assert recovered_scale == pytest.approx(scale, abs=1e-8)


def test_fit_linear_affine_ordinary_least_squares_exact_fit():
    tgt = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)])
    true_matrix = np.array([[1.1, 0.05, 2.0], [-0.03, 0.95, -1.0]])
    ref = _apply_affine(true_matrix, tgt)
    matrix = fit_linear(_cps(ref, tgt), GeometricModel.AFFINE)
    np.testing.assert_allclose(matrix, true_matrix, atol=1e-8)


def test_fit_linear_insufficient_points_raises():
    with pytest.raises(TransformUnreliableError):
        fit_linear(_cps([(0, 0), (1, 1)], [(0, 0), (1, 1)]), GeometricModel.AFFINE)
    with pytest.raises(TransformUnreliableError):
        fit_linear(_cps([(0, 0)], [(0, 0)]), GeometricModel.SIMILARITY)


# --- robust_residuals -------------------------------------------------------

def test_robust_residuals_flags_injected_outlier():
    ref = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0), (5.0, 5.0)])
    tgt = ref.copy()
    tgt[2] += np.array([20.0, 20.0])  # gross outlier
    cps = _cps(ref, tgt)
    matrix = fit_linear(cps, GeometricModel.TRANSLATION)
    residuals = robust_residuals(cps, matrix, threshold_mad=3.5)
    assert residuals.outlier[2]
    assert not residuals.outlier[[0, 1, 3, 4]].any()


def test_robust_residuals_mad_floor_avoids_flagging_tiny_residuals():
    ref = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)])
    tgt = ref + np.array([[0.0, 0.0], [0.001, 0.0], [0.0, -0.001], [0.001, 0.001]])
    cps = _cps(ref, tgt)
    matrix = fit_linear(cps, GeometricModel.TRANSLATION)
    residuals = robust_residuals(cps, matrix, threshold_mad=3.5)
    assert not residuals.outlier.any()


def test_apply_outlier_policy_reject_raises_and_warn_drops():
    ref = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0), (5.0, 5.0)])
    tgt = ref.copy()
    tgt[2] += np.array([20.0, 20.0])
    cps = _cps(ref, tgt)
    matrix = fit_linear(cps, GeometricModel.TRANSLATION)
    residuals = robust_residuals(cps, matrix)

    with pytest.raises(TransformUnreliableError, match="outliers"):
        apply_outlier_policy(cps, residuals, "reject")

    kept = apply_outlier_policy(cps, residuals, "warn")
    assert len(kept) == 4
    assert "P2" not in kept.ids


# --- linear_parameters_at_centroid / check_bounds --------------------------

def test_linear_parameters_at_centroid_identity():
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    params = linear_parameters_at_centroid(matrix, np.array([50.0, 50.0]))
    assert params.rotation_deg == pytest.approx(0.0)
    assert params.scale_x == pytest.approx(1.0)
    assert params.scale_y == pytest.approx(1.0)
    assert params.translation_at_centroid == pytest.approx((5.0, -3.0))


def test_check_bounds_flags_each_violation_independently():
    bounds = _FakeBounds(max_translation_m=1.0, max_rotation_deg=5.0, max_scale_deviation=0.1)
    violations = check_bounds(rotation_deg=20.0, scale_x=1.5, scale_y=1.5, translation_m=(5.0, 0.0), bounds=bounds)
    assert len(violations) == 3
    ok = check_bounds(rotation_deg=1.0, scale_x=1.02, scale_y=0.98, translation_m=(0.1, 0.1), bounds=bounds)
    assert ok == []


# --- leave-one-out -----------------------------------------------------------

def test_loo_rmse_affine_matches_manual_leave_one_out():
    tgt = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0), (5.0, 3.0)])
    true_matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    ref = _apply_affine(true_matrix, tgt)
    cps = _cps(ref, tgt)

    errors = []
    n = len(cps)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        subset = ControlPointSet(cps.ref_xy[keep], cps.tgt_xy[keep], tuple(np.array(cps.ids)[keep]), cps.is_check[keep])
        matrix = fit_linear(subset, GeometricModel.AFFINE)
        predicted = _apply_affine(matrix, cps.tgt_xy[i:i + 1])[0]
        errors.append(np.linalg.norm(cps.ref_xy[i] - predicted))
    expected_rmse = float(np.sqrt(np.mean(np.square(errors))))

    assert loo_rmse(cps, GeometricModel.AFFINE) == pytest.approx(expected_rmse, abs=1e-8)
    # An exact affine fit (no noise) leaves zero leave-one-out error.
    assert expected_rmse == pytest.approx(0.0, abs=1e-8)


def test_loo_rmse_none_below_minimum_points():
    cps = _cps([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
    assert loo_residual_vectors(cps, GeometricModel.AFFINE) is None
    assert loo_rmse(cps, GeometricModel.AFFINE) is None


# --- select_model decision table --------------------------------------------

def test_select_model_forced():
    cfg = _FakeManualConfig(model="affine")
    cps = _cps([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
    decision = select_model(cps, cfg, scipy_available=True)
    assert decision.model == GeometricModel.AFFINE


def test_select_model_n1_translation():
    cfg = _FakeManualConfig()
    cps = _cps([(10.0, 10.0)], [(8.0, 9.0)])
    assert select_model(cps, cfg, scipy_available=True).model == GeometricModel.TRANSLATION


def test_select_model_n2_similarity():
    cfg = _FakeManualConfig()
    cps = _cps([(0.0, 0.0), (10.0, 0.0)], [(0.0, 0.0), (9.0, 1.0)])
    assert select_model(cps, cfg, scipy_available=True).model == GeometricModel.SIMILARITY


def test_select_model_below_min_tps_points_uses_affine():
    cfg = _FakeManualConfig(min_tps_points=8)
    tgt = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0), (5.0, 5.0)])
    ref = tgt + np.array([1.0, 1.0])
    decision = select_model(_cps(ref, tgt), cfg, scipy_available=True, hull_coverage=1.0)
    assert decision.model == GeometricModel.AFFINE
    assert "min_tps_points" in decision.reason


def test_select_model_no_scipy_uses_affine_even_with_enough_points():
    cfg = _FakeManualConfig(min_tps_points=4)
    rng = np.random.default_rng(0)
    tgt = rng.uniform(0, 100, size=(8, 2))
    ref = tgt + np.array([1.0, 1.0])
    decision = select_model(_cps(ref, tgt), cfg, scipy_available=False, hull_coverage=1.0)
    assert decision.model == GeometricModel.AFFINE
    assert "SciPy" in decision.reason


def test_select_model_low_hull_coverage_uses_affine():
    cfg = _FakeManualConfig(min_tps_points=4, min_hull_coverage=0.5)
    rng = np.random.default_rng(1)
    tgt = rng.uniform(0, 100, size=(8, 2))
    ref = tgt + np.array([1.0, 1.0])
    decision = select_model(_cps(ref, tgt), cfg, scipy_available=True, hull_coverage=0.1)
    assert decision.model == GeometricModel.AFFINE
    assert "hull" in decision.reason


def test_select_model_8_well_spread_points_with_genuine_bend_selects_tps():
    cfg = _FakeManualConfig(min_tps_points=8, min_hull_coverage=0.0, min_tps_gain=0.15)
    tgt = np.array([
        (10.0, 10.0), (490.0, 10.0), (10.0, 490.0), (490.0, 490.0),
        (250.0, 10.0), (10.0, 250.0), (490.0, 250.0), (250.0, 490.0),
    ])
    baseline_matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    ref_linear = _apply_affine(baseline_matrix, tgt)
    bend = 4.0 * np.sin(tgt[:, 0] / 100.0)[:, None] * np.array([1.0, 0.4]) + \
        4.0 * np.cos(tgt[:, 1] / 100.0)[:, None] * np.array([0.4, 1.0])
    ref = ref_linear + bend
    decision = select_model(_cps(ref, tgt), cfg, scipy_available=True, hull_coverage=1.0)
    assert decision.model == GeometricModel.TPS
    assert decision.loo_rmse_tps < decision.loo_rmse_affine


def test_select_model_8_points_without_genuine_bend_stays_affine():
    cfg = _FakeManualConfig(min_tps_points=8, min_hull_coverage=0.0, min_tps_gain=0.15)
    tgt = np.array([
        (10.0, 10.0), (490.0, 10.0), (10.0, 490.0), (490.0, 490.0),
        (250.0, 10.0), (10.0, 250.0), (490.0, 250.0), (250.0, 490.0),
    ])
    baseline_matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    # Small per-point click noise (no genuine non-linear bend): with a purely
    # affine truth, TPS has more free parameters to overfit that noise and
    # should not win the leave-one-out comparison by the configured margin.
    rng = np.random.default_rng(42)
    noise = rng.normal(scale=0.05, size=tgt.shape)
    ref = _apply_affine(baseline_matrix, tgt) + noise
    decision = select_model(_cps(ref, tgt), cfg, scipy_available=True, hull_coverage=1.0)
    assert decision.model == GeometricModel.AFFINE
    assert "did not beat affine" in decision.reason


# --- geometry validation -----------------------------------------------------

def test_validate_geometry_flags_near_duplicate_points():
    cps = _cps([(10.0, 20.0), (80.0, 25.0), (10.3, 20.2)], [(12.0, 19.0), (81.0, 27.0), (12.2, 19.1)])
    report = validate_geometry(cps, min_separation_px=1.0)
    assert not report.ok
    assert any("closer than" in m or "apart" in m for m in report.messages)


def test_validate_geometry_flags_collinear_points():
    cps = _cps([(10.0, 10.0), (20.0, 20.001), (30.0, 30.0)], [(12.0, 12.0), (22.0, 22.001), (32.0, 32.0)])
    report = validate_geometry(cps, min_conditioning=1e-3)
    assert not report.ok
    assert any("collinear" in m for m in report.messages)


def test_validate_geometry_well_spread_points_pass():
    cps = _cps([(10.0, 10.0), (400.0, 20.0), (60.0, 400.0)], [(12.0, 11.0), (402.0, 21.0), (62.0, 401.0)])
    report = validate_geometry(cps)
    assert report.ok


# --- Jacobian gate -----------------------------------------------------------

class _IdentityField:
    name = "identity"

    def evaluate(self, xy):
        return np.zeros((len(np.asarray(xy).reshape(-1, 2)), 2))


class _FoldingField:
    """A residual field whose gradient folds the mapping (det(I+grad) < 0)."""
    name = "folding"

    def evaluate(self, xy):
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        # residual_x = -2 * x makes d(residual_x)/dx = -2, so 1 + (-2) = -1 < 0.
        return np.column_stack([-2.0 * points[:, 0], np.zeros(len(points))])


def test_evaluate_jacobian_identity_field_passes():
    report = evaluate_jacobian(_IdentityField(), (100, 100), min_det=0.5, max_gradient=0.15)
    assert report.ok
    assert report.min_det == pytest.approx(1.0, abs=1e-6)
    assert report.max_gradient == pytest.approx(0.0, abs=1e-6)


def test_evaluate_jacobian_folding_field_fails():
    report = evaluate_jacobian(_FoldingField(), (100, 100), min_det=0.5, max_gradient=0.15)
    assert not report.ok
    assert report.min_det < 0.5
