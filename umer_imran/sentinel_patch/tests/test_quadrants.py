"""G7: the offset must be one translation, not something that varies across the field."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sentinel_patch.config.schema import RegistrationConfig
from sentinel_patch.registration.estimators import estimate_shift_ncc
from sentinel_patch.registration.quadrants import check_quadrants

from . import synthetic

CFG = RegistrationConfig(min_overlap_px=100)
SHAPE = (60, 60)
EVERYWHERE = np.ones(SHAPE, bool)


@pytest.fixture
def reference(scene, s2_transform):
    return synthetic.sample_scene(scene, s2_transform, SHAPE)


def _global_shift(reference, target, cfg=CFG, valid_ref=EVERYWHERE):
    result = estimate_shift_ncc(reference, target, valid_ref, EVERYWHERE, cfg)
    return result.col_shift_px, result.row_shift_px


def test_a_pure_translation_is_consistent_in_every_quarter(scene, s2_transform, reference):
    target = synthetic.sample_scene(scene, s2_transform, SHAPE, content_offset_m=(15.0, -12.0))
    gate, report = check_quadrants(reference, target, EVERYWHERE, EVERYWHERE, _global_shift(reference, target), CFG)
    assert gate.passed and report.evaluated == 4 and report.skipped == []
    assert gate.value < 0.2 and gate.reason_code == "NON_TRANSLATIONAL"


def test_a_rotation_is_caught_even_though_a_global_shift_exists(scene, s2_transform, reference):
    centre = s2_transform * (30.0, 30.0)
    target = synthetic.sample_scene_rotated(scene, s2_transform, SHAPE, degrees=3.0, about=centre)
    cfg = RegistrationConfig(min_overlap_px=100, ncc_search_px=5)
    gate, report = check_quadrants(reference, target, EVERYWHERE, EVERYWHERE, _global_shift(reference, target, cfg), cfg)
    assert not gate.passed and report.evaluated >= 2
    assert (gate.value is None) or gate.value > cfg.quadrant_tolerance_px


def test_a_quarter_that_wants_a_far_larger_shift_fails_the_gate(scene, s2_transform, reference):
    """One quarter shows content displaced by 6 px (beyond the search radius): inconsistent, not 'unknown'."""
    target = synthetic.sample_scene(scene, s2_transform, SHAPE, content_offset_m=(10.0, 0.0))
    other = synthetic.sample_scene(scene, s2_transform, SHAPE, content_offset_m=(70.0, 0.0))
    target = target.copy()
    target[30:, 30:] = other[30:, 30:]
    gate, report = check_quadrants(reference, target, EVERYWHERE, EVERYWHERE, (-1.0, 0.0), CFG)
    assert not gate.passed
    assert math.inf in report.deviations_px or max(report.deviations_px) > CFG.quadrant_tolerance_px


def test_a_small_overlap_is_not_evaluated_and_passes(scene, s2_transform, reference):
    small = np.zeros(SHAPE, bool)
    small[:40, :40] = True  # 40 px < the 48 px minimum
    gate, report = check_quadrants(reference, reference, small, small, (0.0, 0.0), CFG)
    assert gate.passed and gate.value is None and report.skipped == ["WINDOW_TOO_SMALL"]


def test_too_few_usable_quarters_is_not_evaluated(scene, s2_transform, reference):
    # The overlap's bounding box spans the whole window, but only the north-west quarter holds usable pixels
    # (plus a few in the opposite corner to stretch the box).
    sparse = np.zeros(SHAPE, bool)
    sparse[:30, :30] = True
    sparse[58:, 58:] = True
    gate, report = check_quadrants(reference, reference, sparse, sparse, (0.0, 0.0), CFG)
    assert gate.passed and gate.value is None
    assert report.evaluated < 2 and report.skipped


def test_no_overlap_is_not_evaluated(reference):
    nothing = np.zeros(SHAPE, bool)
    gate, report = check_quadrants(reference, reference, nothing, nothing, (0.0, 0.0), CFG)
    assert gate.passed and report.skipped == ["NO_OVERLAP"]
