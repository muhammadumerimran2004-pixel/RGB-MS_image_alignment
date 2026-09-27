import numpy as np
import pytest

from drone_alignment.alignment.displacement_field import (
    compare_displacement_fields,
    fit_selected_displacement_field,
    fit_regularized_mesh_field,
    fit_tapered_thin_plate_spline_field,
    nearest_control_support,
    renormalized_taper,
    warp_registration_band_with_field,
)
from drone_alignment.alignment.local_evidence import CellMatchStatus, LocalMatchSample, SparseDisplacementResult
from drone_alignment.alignment.local_mesh_aligner import compute_sparse_local_displacements
from drone_alignment.config.schema import LocalMeshConfig, RoadGridConfig


def _evidence_result():
    h, w = 120, 120
    rgb = np.full((h, w), 0.2, dtype=np.float32)
    rgb[:, 20:25] = 0.9
    rgb[:, 84:89] = 0.9
    rgb[50:55, :] = 0.9
    ms = np.full_like(rgb, 0.2)
    ms[:-2, :-3] = rgb[2:, 3:]
    mask = np.ones_like(rgb, dtype=bool)
    config = LocalMeshConfig(grid_rows=3, grid_cols=3, max_search_radius_px=8,
                             min_road_points=5, min_tree_points=5,
                             min_match_confidence=0.02, ambiguity_margin=0.01,
                             tree_only_min_confidence=0.02, max_field_gradient=1.0,
                             field_support_radius_px=80)
    result = compute_sparse_local_displacements(
        rgb, ms, mask, mask, (1, 1), RoadGridConfig(blur_kernel_size=3), config,
    )
    return result, config, (h, w)


def test_regularized_mesh_evaluates_finite_residuals():
    result, config, shape = _evidence_result()
    field = fit_regularized_mesh_field(result.accepted, shape, config.grid_rows, config.grid_cols, config)
    values = field.evaluate(np.array([[0.0, 0.0], [60.0, 60.0], [119.0, 119.0]]))
    assert values.shape == (3, 2)
    assert np.isfinite(values).all()


def test_field_comparison_reports_gradient_safe_candidate():
    result, config, shape = _evidence_result()
    comparison = compare_displacement_fields(result, shape, config.grid_rows, config.grid_cols, config)
    assert comparison.metrics
    assert any(metric.is_gradient_safe for metric in comparison.metrics)


def test_absolute_leave_one_out_gate_rejects_poor_but_finite_field():
    samples = tuple(
        LocalMatchSample(
            row=row, col=col, center_xy=center, source_ms_xy=center, predicted_rgb_xy=center,
            residual_dx_dy=residual, displacement_dx_dy=residual, confidence=0.9,
            road_score=None, second_road_score=None, tree_residual_dx_dy=None,
            status=CellMatchStatus.ACCEPTED,
        )
        for row, col, center, residual in (
            (0, 0, (0.0, 0.0), (0.0, 0.0)),
            (0, 1, (100.0, 0.0), (12.0, 0.0)),
            (1, 0, (0.0, 100.0), (0.0, 12.0)),
            (1, 1, (100.0, 100.0), (20.0, -10.0)),
        )
    )
    result = SparseDisplacementResult(samples, None, None, (0.0, 0.0), len(samples), 1.0)
    config = LocalMeshConfig(
        grid_rows=2, grid_cols=2, field_support_radius_px=100,
        max_field_gradient=5.0, max_field_loo_rmse_px=0.001,
    )

    comparison = compare_displacement_fields(result, (101, 101), 2, 2, config)

    assert comparison.metrics
    assert comparison.selected_name is None
    assert "absolute leave-one-out" in comparison.reason


def test_selected_field_helper_constructs_requested_mesh():
    result, config, shape = _evidence_result()

    field = fit_selected_displacement_field(
        "regularized_bilinear_mesh", result.accepted, shape, config.grid_rows, config.grid_cols, config,
    )

    assert field.name == "regularized_bilinear_mesh"
    assert np.isfinite(field.evaluate(np.array([[60.0, 60.0]]))).all()


def test_registration_grid_warp_uses_forward_residual_as_inverse_sample_shift():
    image = np.tile(np.arange(32, dtype=np.float32), (32, 1))
    mask = np.ones_like(image, dtype=bool)
    # Use a constant field here so the expected pull direction is unambiguous.
    from drone_alignment.alignment.displacement_field import RegularizedMeshField
    constant = RegularizedMeshField(np.array([0., 31.]), np.array([0., 31.]), np.full((2, 2, 2), [2., 0.]))
    warped, valid = warp_registration_band_with_field(image, mask, np.array([[1., 0., 0.], [0., 1., 0.]]), constant)
    assert valid[12, 15]
    assert warped[12, 15] == 13.0


def test_registration_grid_warp_applies_residual_before_inverse_affine():
    image = np.tile(np.arange(32, dtype=np.float32), (32, 1))
    mask = np.ones_like(image, dtype=bool)
    from drone_alignment.alignment.displacement_field import RegularizedMeshField
    constant = RegularizedMeshField(np.array([0., 31.]), np.array([0., 31.]), np.full((2, 2, 2), [2., 0.]))

    warped, valid = warp_registration_band_with_field(
        image, mask, np.array([[2., 0., 0.], [0., 1., 0.]]), constant,
    )

    # p = inverse(M)(q - residual): at q=10, source x=(10-2)/2=4.
    assert valid[12, 10]
    assert warped[12, 10] == 4.0


def test_taper_is_one_at_control_points():
    """A control point's own support must be exactly 1 regardless of fallback_weight,
    so the taper there is always exactly 1 (the field must still pass through its own
    control points, not just approximately)."""
    control_xy = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [10.0, 10.0]])  # last two are close together
    support = nearest_control_support(control_xy, control_xy, sigma_px=20.0)
    np.testing.assert_allclose(support, 1.0)

    for fallback_weight in (0.0, 0.2, 5.0):
        taper = renormalized_taper(support, fallback_weight)
        np.testing.assert_allclose(taper, 1.0)


def test_taper_decays_to_zero_far_away():
    control_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    far_points = np.array([[10_000.0, 10_000.0]])
    support = nearest_control_support(far_points, control_xy, sigma_px=10.0)
    taper = renormalized_taper(support, fallback_weight=0.5)
    assert taper[0] < 1e-6


def test_taper_independent_of_point_density():
    """Nearest-control support must depend only on distance to the nearest control,
    not on how many other controls happen to be clustered nearby (the bug this
    replaces: a density-summed support exceeds 1, and decreasing fallback_weight's
    effect, wherever many controls happen to cluster)."""
    query = np.array([[5.0, 0.0]])
    sparse_controls = np.array([[0.0, 0.0], [100.0, 100.0]])
    # 50 exactly-coincident duplicates of the same nearest control: a
    # density-summed support would sum contributions from all 50 and end up
    # far larger than the sparse case's single contribution.
    dense_controls = np.array([[0.0, 0.0]] * 50)

    sparse_support = nearest_control_support(query, sparse_controls, sigma_px=10.0)
    dense_support = nearest_control_support(query, dense_controls, sigma_px=10.0)

    np.testing.assert_allclose(sparse_support, dense_support)


def test_taper_memory_bounded():
    """Support/taper evaluation over a large point set with many controls must not
    allocate an (n_points, n_controls) dense array, which would be hundreds of MB
    to several GB at native-tile resolution."""
    rng = np.random.default_rng(0)
    controls = rng.uniform(0.0, 500.0, size=(50, 2))
    points = rng.uniform(0.0, 500.0, size=(4_200_000, 2))  # ~ one 2048x2048 tile

    import tracemalloc
    tracemalloc.start()
    support = nearest_control_support(points, controls, sigma_px=50.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert support.shape == (len(points),)
    assert peak < 256 * 1024 * 1024


def test_tapered_tps_field_passes_through_its_own_control_points():
    samples = tuple(
        LocalMatchSample(
            row=0, col=0, center_xy=center, source_ms_xy=center, predicted_rgb_xy=center,
            residual_dx_dy=residual, displacement_dx_dy=residual, confidence=0.9,
            road_score=None, second_road_score=None, tree_residual_dx_dy=None,
            status=CellMatchStatus.ACCEPTED,
        )
        for center, residual in (
            ((10.0, 10.0), (2.0, -1.0)),
            ((90.0, 15.0), (-3.0, 4.0)),
            ((20.0, 90.0), (1.0, 1.0)),
            ((80.0, 80.0), (-1.0, -2.0)),
        )
    )
    config = LocalMeshConfig(field_support_radius_px=30.0, field_fallback_weight=0.2, rbf_smoothing=0.0)
    field = fit_tapered_thin_plate_spline_field(samples, config)

    control_xy = np.array([sample.center_xy for sample in samples])
    expected = np.array([sample.residual_dx_dy for sample in samples])
    evaluated = field.evaluate(control_xy)

    np.testing.assert_allclose(evaluated, expected, atol=1e-6)
