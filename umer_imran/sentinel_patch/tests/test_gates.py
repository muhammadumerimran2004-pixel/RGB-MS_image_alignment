"""S6: each gate in isolation, at, just below and just above its threshold."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel_patch.config.schema import RegistrationConfig
from sentinel_patch.registration import gates
from sentinel_patch.registration.estimators import NccResult

CFG = RegistrationConfig()


def _ncc(peak=0.9, ratio=2.0) -> NccResult:
    return NccResult(0.0, 0.0, (0, 0), peak, None, ratio, 1000, np.zeros((3, 3)))


def test_texture_is_scale_free():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 1.5, (50, 50))
    valid = np.ones_like(base, bool)
    assert gates.texture_score(base, valid) == pytest.approx(gates.texture_score(base * 4000.0, valid))  # sensor units do not matter


def test_uniform_scene_fails_the_texture_gate_and_textured_one_passes():
    valid = np.ones((50, 50), bool)
    yy, xx = np.mgrid[:50, :50]
    uniform = 0.3 + 0.003 * np.sin(xx / 5.0)
    textured = 0.3 + 0.2 * (xx > 25)  # a field edge
    assert not gates.gate_texture(uniform, valid, CFG).passed
    assert gates.gate_texture(textured, valid, CFG).passed


def test_texture_ignores_invalid_pixels():
    values = np.full((20, 20), 0.3)
    values[:, 10:] = np.nan
    assert gates.texture_score(values, np.ones_like(values, bool)) == pytest.approx(0.0)


@pytest.mark.parametrize("pixels, passed", [(399, False), (400, True), (5000, True)])
def test_overlap_gate_boundary(pixels, passed):
    result = gates.gate_overlap(pixels, CFG)
    assert result.passed is passed and result.reason_code == "AOI_TOO_SMALL"


@pytest.mark.parametrize("dx, dy, passed", [(15.0, 20.0, True), (15.1, 20.1, False), (0.0, 0.0, True), (-18.0, 18.0, False)])
def test_magnitude_gate_uses_the_vector_length(dx, dy, passed):
    assert gates.gate_magnitude(dx, dy, CFG).passed is passed  # (15, 20) is exactly 25 m


def test_peak_gate_needs_both_a_high_peak_and_a_clear_winner():
    assert all(g.passed for g in gates.gate_peak(_ncc(0.9, 2.0), CFG))
    weak = gates.gate_peak(_ncc(0.4, 2.0), CFG)
    ambiguous = gates.gate_peak(_ncc(0.9, 1.05), CFG)
    assert [g.passed for g in weak] == [False, True] and [g.passed for g in ambiguous] == [True, False]
    assert {g.reason_code for g in weak + ambiguous} == {"WEAK_PEAK"}


def test_holdout_must_improve_for_a_real_shift():
    assert gates.gate_holdout(0.6, 0.7, 1.5, CFG).passed
    assert not gates.gate_holdout(0.6, 0.61, 1.5, CFG).passed  # gain 0.01 < 0.02
    assert not gates.gate_holdout(0.6, 0.5, 1.5, CFG).passed


def test_holdout_for_a_tiny_shift_only_has_to_be_not_worse():
    assert gates.gate_holdout(0.9, 0.895, 0.1, CFG).passed
    assert not gates.gate_holdout(0.9, 0.85, 0.1, CFG).passed


def test_holdout_without_a_measurement_fails_closed():
    assert not gates.gate_holdout(None, 0.9, 1.5, CFG).passed


def test_first_failure_wins_in_list_order():
    results = [gates.gate_overlap(1000, CFG), gates.gate_magnitude(40, 0, CFG), gates.gate_overlap(10, CFG)]
    assert gates.first_failure(results).name == "G3_magnitude"
    assert gates.first_failure(results[:1]) is None


def test_holdout_correlation_improves_when_the_right_shift_is_applied(scene, s2_transform, s2_shape):
    from . import synthetic
    reference = synthetic.sample_scene(scene, s2_transform, s2_shape)
    target = synthetic.sample_scene(scene, s2_transform, s2_shape, content_offset_m=(15.0, -12.0))
    everywhere = np.ones(s2_shape, bool)
    # The target content sits 15 m east and 12 m south of where it belongs, so it must move 1.5 columns
    # west and 1.2 rows north (rows count downwards: -1.2).
    before, after = gates.holdout_correlation(reference, target, everywhere, everywhere, col_shift=-1.5, row_shift=-1.2)
    assert after > before and after > 0.99
    _, wrong_way = gates.holdout_correlation(reference, target, everywhere, everywhere, col_shift=1.5, row_shift=1.2)
    assert wrong_way < before
