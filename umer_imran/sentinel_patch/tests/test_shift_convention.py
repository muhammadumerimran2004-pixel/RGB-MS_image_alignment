"""The shift sign convention is the single most error-prone part of co-registration.

These tests do not trust any documentation: they build a grid that is
mis-georeferenced by a known vector and check against the analytic scene.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from sentinel_patch.config.schema import RegistrationConfig
from sentinel_patch.registration.estimators import estimate_shift_ncc
from sentinel_patch.registration.shift import apply_shift, pixel_shift_to_map, shift_magnitude_m

from . import synthetic


def ncc_estimator(reference, target):
    everywhere = np.ones(reference.shape, dtype=bool)
    # Wider than the production radius (3 px): these test errors go up to 5 px.
    cfg = RegistrationConfig(min_overlap_px=100, ncc_search_px=6)
    result = estimate_shift_ncc(reference, target, everywhere, everywhere, cfg)
    return result.col_shift_px, result.row_shift_px


# Every estimator must satisfy the same round trip. Phase 2 adds the AROSICS wrapper here.
# name -> (estimator, tolerance in metres on the recovered shift)
ESTIMATORS = {"integer_stub": (synthetic.integer_shift_stub, 1e-9), "masked_ncc": (ncc_estimator, 1.5)}

# (ex, ey): how far the Sentinel grid is mis-georeferenced, in metres.
INTEGER_ERRORS_M = [(0.0, 0.0), (30.0, -40.0), (-20.0, 10.0), (10.0, 10.0), (-50.0, -30.0)]
FRACTIONAL_ERRORS_M = [(3.7, -6.2), (-12.0, 4.0), (0.4, 0.4), (-8.3, -1.9)]


def max_error_against_truth(scene, corrected: Affine, s2_array: np.ndarray) -> float:
    """Largest |array - truth| where truth samples the scene at the corrected pixel positions."""
    x, y = synthetic.pixel_centre_xy(corrected, s2_array.shape)
    return float(np.max(np.abs(s2_array - scene(x, y))))


@pytest.mark.parametrize("estimator_name", ESTIMATORS)
@pytest.mark.parametrize("error_m", INTEGER_ERRORS_M)
def test_known_shift_roundtrip(scene, s2_transform, s2_shape, estimator_name, error_m):
    estimator, tolerance_m = ESTIMATORS[estimator_name]
    ex, ey = error_m
    reference = synthetic.sample_scene(scene, s2_transform, s2_shape)
    sentinel = synthetic.sample_scene(scene, s2_transform, s2_shape, content_offset_m=error_m)

    col_shift, row_shift = estimator(reference, sentinel)
    dx, dy = pixel_shift_to_map(s2_transform, col_shift, row_shift)

    assert (dx, dy) == pytest.approx((-ex, -ey), abs=tolerance_m)
    if tolerance_m < 1e-6:  # exact estimators must also restore the true georeferencing at every pixel
        corrected = apply_shift(s2_transform, dx, dy)
        assert max_error_against_truth(scene, corrected, sentinel) < 1e-5


@pytest.mark.parametrize("error_m", FRACTIONAL_ERRORS_M)
def test_fractional_shift_roundtrip(scene, s2_transform, s2_shape, error_m):
    """Sub-pixel accuracy target from the blueprint: within 0.15 px (1.5 m) of the truth."""
    ex, ey = error_m
    reference = synthetic.sample_scene(scene, s2_transform, s2_shape)
    sentinel = synthetic.sample_scene(scene, s2_transform, s2_shape, content_offset_m=error_m)

    dx, dy = pixel_shift_to_map(s2_transform, *ncc_estimator(reference, sentinel))

    assert (dx, dy) == pytest.approx((-ex, -ey), abs=1.5)


def test_wrong_sign_is_detected(scene, s2_transform, s2_shape):
    """Negative control: if the shift were applied with the opposite sign, the truth check must fail."""
    error_m = (30.0, -40.0)
    reference = synthetic.sample_scene(scene, s2_transform, s2_shape)
    sentinel = synthetic.sample_scene(scene, s2_transform, s2_shape, content_offset_m=error_m)

    col_shift, row_shift = synthetic.integer_shift_stub(reference, sentinel)
    dx, dy = pixel_shift_to_map(s2_transform, col_shift, row_shift)

    flipped = apply_shift(s2_transform, -dx, -dy)
    assert max_error_against_truth(scene, flipped, sentinel) > 0.01


def test_positive_dy_moves_image_north(s2_transform):
    moved = apply_shift(s2_transform, 0.0, 5.0)
    assert moved.f == s2_transform.f + 5.0
    # North-up: a larger northing is further up the page, so row 0 sits at a larger y.
    assert (moved * (0, 0))[1] > (s2_transform * (0, 0))[1]


def test_moving_content_down_is_negative_dy(s2_transform):
    dx, dy = pixel_shift_to_map(s2_transform, col_shift=0.0, row_shift=2.0)
    assert (dx, dy) == (0.0, -20.0)


def test_fractional_pixel_shift_converts_exactly(s2_transform):
    dx, dy = pixel_shift_to_map(s2_transform, col_shift=0.37, row_shift=0.62)
    assert (dx, dy) == pytest.approx((3.7, -6.2))
    assert shift_magnitude_m(dx, dy) == pytest.approx(np.hypot(3.7, 6.2))


def test_apply_shift_only_moves_origin(s2_transform):
    moved = apply_shift(s2_transform, 3.7, -6.2)
    assert (moved.a, moved.b, moved.d, moved.e) == (s2_transform.a, s2_transform.b, s2_transform.d, s2_transform.e)
    assert (moved.c, moved.f) == pytest.approx((s2_transform.c + 3.7, s2_transform.f - 6.2))


def test_rotated_grid_uses_the_full_linear_part():
    rotated = Affine.rotation(20.0) * Affine.scale(10.0, -10.0)
    rotated = Affine(rotated.a, rotated.b, 500000.0, rotated.d, rotated.e, 3500000.0)
    col_shift, row_shift = 1.5, -0.75
    dx, dy = pixel_shift_to_map(rotated, col_shift, row_shift)

    # Moving a pixel's content by (col_shift, row_shift) must move its map position by exactly (dx, dy).
    before = rotated * (4.5, 7.5)
    after = rotated * (4.5 + col_shift, 7.5 + row_shift)
    assert (after[0] - before[0], after[1] - before[1]) == pytest.approx((dx, dy))
