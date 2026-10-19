"""Behaviour of the masked-NCC estimator beyond the sign round trip."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel_patch.config.schema import RegistrationConfig
from sentinel_patch.errors import RegistrationRejected
from sentinel_patch.registration.estimators import estimate_shift_ncc, ncc_surface
from sentinel_patch.registration.shift import pixel_shift_to_map

from . import synthetic

CFG = RegistrationConfig(min_overlap_px=100, ncc_search_px=3)


def _pair(scene, transform, shape, error_m):
    return (synthetic.sample_scene(scene, transform, shape),
            synthetic.sample_scene(scene, transform, shape, content_offset_m=error_m))


def test_shift_beyond_the_search_radius_is_rejected_not_guessed(scene, s2_transform, s2_shape):
    reference, target = _pair(scene, s2_transform, s2_shape, (50.0, -40.0))  # 5 px / 4 px, radius is 3
    everywhere = np.ones(s2_shape, bool)
    with pytest.raises(RegistrationRejected) as caught:
        estimate_shift_ncc(reference, target, everywhere, everywhere, CFG)
    assert caught.value.reason_code == "PEAK_AT_SEARCH_BORDER"


def test_masked_pixels_do_not_influence_the_estimate(scene, s2_transform, s2_shape):
    """A cloud patch showing unrelated content must be ignored once it is masked."""
    error_m = (-20.0, 10.0)
    reference, target = _pair(scene, s2_transform, s2_shape, error_m)
    cloud = np.zeros(s2_shape, bool)
    cloud[:, :22] = True  # 37 % of the window
    target = target.copy()
    target[cloud] = synthetic.sample_scene(scene, s2_transform, s2_shape, content_offset_m=(-60.0, 50.0))[cloud]

    everywhere = np.ones(s2_shape, bool)
    masked = estimate_shift_ncc(reference, target, everywhere, ~cloud, CFG)

    truth_px = (2.0, 1.0)  # content must move +2 columns and +1 row (east 20 m, south 10 m)
    assert (masked.col_shift_px, masked.row_shift_px) == pytest.approx(truth_px, abs=0.15)

    # Negative control: without the mask the unrelated cloud content must visibly hurt the estimate
    # (either a refusal or a clearly worse shift), otherwise this test would prove nothing.
    try:
        unmasked = estimate_shift_ncc(reference, target, everywhere, everywhere, CFG)
    except RegistrationRejected:
        return
    assert np.hypot(unmasked.col_shift_px - truth_px[0], unmasked.row_shift_px - truth_px[1]) > 0.3


def test_noise_does_not_break_subpixel_accuracy(scene, s2_transform, s2_shape):
    rng = np.random.default_rng(1)
    reference, target = _pair(scene, s2_transform, s2_shape, (3.7, -6.2))
    target = target + rng.normal(0.0, 0.01, target.shape).astype(np.float32)  # about 3-5 % of the signal
    everywhere = np.ones(s2_shape, bool)
    result = estimate_shift_ncc(reference, target, everywhere, everywhere, CFG)
    assert pixel_shift_to_map(s2_transform, result.col_shift_px, result.row_shift_px) == pytest.approx((-3.7, 6.2), abs=1.5)
    assert result.peak > 0.9 and result.peak_ratio >= 1.15


def test_unrelated_images_have_no_trusted_peak(s2_transform, s2_shape):
    rng = np.random.default_rng(5)
    reference, target = rng.normal(size=s2_shape), rng.normal(size=s2_shape)
    everywhere = np.ones(s2_shape, bool)
    try:
        result = estimate_shift_ncc(reference, target, everywhere, everywhere, CFG)
    except RegistrationRejected:
        return  # also an acceptable outcome
    assert result.peak < 0.5


def test_periodic_rows_give_an_ambiguous_peak(s2_transform, s2_shape):
    """Crop rows repeat every few pixels: a distinct second maximum must show up as a low peak ratio."""
    cols = np.arange(s2_shape[1])
    rows = np.arange(s2_shape[0])
    stripes = np.sin(2 * np.pi * cols / 3.0)[None, :]                    # repeats every 3 columns
    envelope = np.exp(-((rows - 30.0) ** 2) / (2 * 8.0**2))[:, None]     # unique along rows
    image = stripes * envelope
    everywhere = np.ones(s2_shape, bool)
    result = estimate_shift_ncc(image, image, everywhere, everywhere, RegistrationConfig(min_overlap_px=100, ncc_search_px=4))
    assert result.second_peak is not None and result.peak_ratio < 1.15


def test_too_little_overlap_is_reported_as_small_aoi(scene, s2_transform, s2_shape):
    reference, target = _pair(scene, s2_transform, s2_shape, (0.0, 0.0))
    tiny = np.zeros(s2_shape, bool)
    tiny[:5, :5] = True
    with pytest.raises(RegistrationRejected) as caught:
        estimate_shift_ncc(reference, target, tiny, tiny, CFG)
    assert caught.value.reason_code == "AOI_TOO_SMALL"


def test_surface_matches_a_direct_correlation_at_zero_shift(scene, s2_transform, s2_shape):
    reference, target = _pair(scene, s2_transform, s2_shape, (0.0, 0.0))
    everywhere = np.ones(s2_shape, bool)
    surface, counts = ncc_surface(reference, target, everywhere, everywhere, 2, 10)
    assert surface[2, 2] == pytest.approx(1.0, abs=1e-6)
    assert counts[2, 2] == reference.size
