import cv2
import numpy as np
import pytest
from dataclasses import replace

from drone_alignment.alignment.cell_correlator import (
    _apply_spatial_filter,
    _hann_window,
    _phase_correlate_bounded,
    _prepare_patch,
    compute_cell_displacements,
    evaluate_heldout_field_improvement,
)
from drone_alignment.alignment.local_evidence import CellMatchStatus, LocalMatchSample, SparseDisplacementResult
from drone_alignment.config.schema import CellCorrelationConfig


def _config(**overrides) -> CellCorrelationConfig:
    values = dict(
        grid_rows=3,
        grid_cols=3,
        cell_halo_px=6,
        max_search_radius_px=4,
        interpolation_margin_px=2,
        min_valid_fraction=0.30,
        min_texture_std=0.01,
        min_phase_response=0.01,
        min_peak_sharpness=1.01,
        full_confidence_peak_sharpness=1.5,
        min_match_confidence=0.01,
        max_neighbor_difference_px=4.0,
        min_trusted_cells=4,
    )
    values.update(overrides)
    return CellCorrelationConfig(**values)


def _textured_reference(shape=(120, 120), seed=9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.normal(size=shape).astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), 1.1)


def _target_for_forward_shift(reference: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return a target that must move (+dx, +dy) to align with reference."""
    matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
    return cv2.warpAffine(reference, matrix, (reference.shape[1], reference.shape[0]))


def _sample(row: int, col: int, residual: tuple[float, float]) -> LocalMatchSample:
    return LocalMatchSample(
        row=row,
        col=col,
        center_xy=(float(col * 10), float(row * 10)),
        source_ms_xy=(0.0, 0.0),
        predicted_rgb_xy=(0.0, 0.0),
        residual_dx_dy=residual,
        displacement_dx_dy=residual,
        confidence=0.9,
        road_score=None,
        second_road_score=None,
        tree_residual_dx_dy=None,
        status=CellMatchStatus.ACCEPTED,
    )


def test_hann_window_has_correct_shape_and_zero_edges():
    window = _hann_window((17, 29))

    assert window.shape == (17, 29)
    assert np.allclose(window[0], 0.0)
    assert np.allclose(window[-1], 0.0)
    assert np.allclose(window[:, 0], 0.0)
    assert np.allclose(window[:, -1], 0.0)
    assert window[8, 14] > 0.9


def test_bounded_phase_correlation_reports_target_to_reference_sign():
    config = _config()
    reference = _textured_reference()
    target = _target_for_forward_shift(reference, 4, 3)
    valid = np.ones(reference.shape, dtype=bool)
    prepared_reference, status_ref = _prepare_patch(reference, valid, config)
    prepared_target, status_target = _prepare_patch(target, valid, config)

    assert status_ref is None and status_target is None
    estimate = _phase_correlate_bounded(prepared_reference, prepared_target, radius=4, exclusion_radius=2)

    assert estimate.shift_dx_dy[0] == pytest.approx(4.0, abs=0.2)
    assert estimate.shift_dx_dy[1] == pytest.approx(3.0, abs=0.2)
    assert estimate.is_on_search_boundary


def test_phase_peak_on_search_boundary_is_marked_for_rejection():
    reference = _textured_reference()
    target = _target_for_forward_shift(reference, 4, 0)
    result = compute_cell_displacements(
        {"green": reference}, {"green": target},
        np.ones(reference.shape, bool), np.ones(reference.shape, bool),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), ["green"], _config(),
    )

    assert any(sample.status == CellMatchStatus.SEARCH_BOUNDARY for sample in result.samples)


def test_flat_patch_is_rejected_as_low_texture():
    image = np.full((90, 90), 0.5, dtype=np.float32)
    result = compute_cell_displacements(
        {"green": image}, {"green": image.copy()},
        np.ones(image.shape, bool), np.ones(image.shape, bool),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), ["green"], _config(),
    )

    assert all(sample.status == CellMatchStatus.LOW_TEXTURE for sample in result.samples)


def test_core_valid_fraction_is_not_inflated_by_halo():
    image = _textured_reference((120, 120))
    mask = np.ones(image.shape, dtype=bool)
    mask[:40, :40] = False
    mask[:12, :12] = True
    result = compute_cell_displacements(
        {"green": image}, {"green": image.copy()}, mask, mask,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), ["green"],
        _config(min_valid_fraction=0.50),
    )

    top_left = next(sample for sample in result.samples if sample.row == 0 and sample.col == 0)
    assert top_left.valid_fraction < 0.50
    assert top_left.status == CellMatchStatus.INSUFFICIENT_VALID_DATA


def test_conflicting_channels_are_not_averaged_into_a_field_control():
    green = _textured_reference(seed=1)
    red = _textured_reference(seed=2)
    result = compute_cell_displacements(
        {"green": green, "red": red},
        {"green": _target_for_forward_shift(green, 2, 0), "red": _target_for_forward_shift(red, -2, 0)},
        np.ones(green.shape, bool), np.ones(green.shape, bool),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), ["green", "red"],
        _config(max_channel_disagreement_px=0.5),
    )

    assert any(sample.status == CellMatchStatus.CHANNEL_CONFLICT for sample in result.samples)
    assert not result.accepted


def test_spatial_filter_rejects_anomalous_center_vector():
    samples = [_sample(row, col, (0.0, 0.0)) for row in range(3) for col in range(3)]
    samples[4] = _sample(1, 1, (12.0, 0.0))

    filtered = _apply_spatial_filter(samples, _config(max_neighbor_difference_px=3.0))

    assert filtered[4].status == CellMatchStatus.SPATIAL_OUTLIER
    assert filtered[4].confidence == 0.0


def test_smooth_non_rigid_synthetic_field_recovers_larger_right_side_residuals():
    reference = _textured_reference((180, 180), seed=33)
    yy, xx = np.indices(reference.shape, dtype=np.float32)
    residual_x = 0.5 + 2.0 * xx / (reference.shape[1] - 1)
    target = cv2.remap(reference, xx + residual_x, yy, cv2.INTER_LINEAR)
    config = _config(grid_rows=3, grid_cols=3, cell_halo_px=8, max_search_radius_px=5)
    result = compute_cell_displacements(
        {"green": reference}, {"green": target},
        np.ones(reference.shape, bool), np.ones(reference.shape, bool),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), ["green"], config,
    )

    left = [sample.residual_dx_dy[0] for sample in result.accepted if sample.col == 0]
    right = [sample.residual_dx_dy[0] for sample in result.accepted if sample.col == 2]
    assert len(result.accepted) >= 5
    assert left and right
    assert float(np.median(right)) > float(np.median(left)) + 0.7


def test_heldout_field_validation_accepts_independently_predicted_improvement():
    reference = _textured_reference((150, 150), seed=45)
    target = _target_for_forward_shift(reference, 2, 1)
    config = _config(grid_rows=3, grid_cols=3, cell_halo_px=8, max_search_radius_px=4)
    evidence = compute_cell_displacements(
        {"green": reference}, {"green": target},
        np.ones(reference.shape, bool), np.ones(reference.shape, bool),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), ["green"], config,
    )

    validation = evaluate_heldout_field_improvement(
        evidence, {"green": reference}, {"green": target},
        np.ones(reference.shape, bool), np.ones(reference.shape, bool),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "regularized_bilinear_mesh", config,
    )

    assert validation.evaluated_count >= config.min_holdout_cells
    assert validation.mean_improvement is not None and validation.mean_improvement > 0.0
    assert validation.is_accepted


def test_heldout_field_validation_rejects_field_that_harms_already_aligned_images():
    reference = _textured_reference((120, 120), seed=72)
    config = _config(grid_rows=2, grid_cols=2, min_holdout_cells=3)
    samples = tuple(
        replace(_sample(row, col, (2.0, 0.0)), channel="green", phase_shift_dx_dy=(2.0, 0.0))
        for row in range(2) for col in range(2)
    )
    evidence = SparseDisplacementResult(samples, None, None, (0.0, 0.0), len(samples), 1.0)

    validation = evaluate_heldout_field_improvement(
        evidence, {"green": reference}, {"green": reference.copy()},
        np.ones(reference.shape, bool), np.ones(reference.shape, bool),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "regularized_bilinear_mesh", config,
    )

    assert validation.evaluated_count == 4
    assert validation.mean_improvement is not None and validation.mean_improvement < 0.0
    assert not validation.is_accepted
