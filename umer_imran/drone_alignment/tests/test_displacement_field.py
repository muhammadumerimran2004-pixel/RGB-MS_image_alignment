import numpy as np

from drone_alignment.alignment.displacement_field import (
    compare_displacement_fields,
    fit_selected_displacement_field,
    fit_regularized_mesh_field,
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
