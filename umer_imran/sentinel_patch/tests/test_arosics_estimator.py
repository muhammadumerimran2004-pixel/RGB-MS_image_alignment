"""AROSICS is an optional cross-check: these tests pin down when its answer may be trusted and when it is ignored."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sentinel_patch.config.schema import RegistrationConfig
from sentinel_patch.registration import arosics_estimator, gates
from sentinel_patch.registration.arosics_estimator import ArosicsResult, estimate_shift_arosics

from . import scene_files, synthetic

pytestmark = pytest.mark.skipif(not arosics_estimator.is_available(), reason="AROSICS is not installed")

TRANSFORM, SHAPE = synthetic.make_s2_transform(), (60, 60)
CFG = RegistrationConfig(min_overlap_px=100)
EVERYWHERE = np.ones(SHAPE, bool)


def _pair(seed: int, error_m: tuple[float, float]):
    scene = synthetic.make_field_scene(TRANSFORM, SHAPE, seed=seed, cells=15 + 25 * (seed % 3))
    return (synthetic.sample_scene(scene, TRANSFORM, SHAPE, supersample=5),
            synthetic.sample_scene(scene, TRANSFORM, SHAPE, error_m, supersample=5))


def _run(reference, target, cfg=CFG, valid_ref=EVERYWHERE, valid_tgt=EVERYWHERE):
    return estimate_shift_arosics(reference, target, valid_ref, valid_tgt, TRANSFORM, scene_files.CRS_UTM42, cfg)


def test_a_conclusive_answer_is_trustworthy_and_uses_our_sign_convention():
    """The premise of the agreement gate: whenever AROSICS reports enough reliability, it is close to the truth.

    The shift it reports is already the vector to ADD to the target origin, i.e. exactly our shift_m.
    """
    rng = np.random.default_rng(0)
    errors_px = []
    for seed in range(30):
        error = tuple(rng.uniform(-20.0, 20.0, 2))
        result = _run(*_pair(seed, error))
        if result.status == "ok":
            errors_px.append(math.hypot(result.shift_m[0] + error[0], result.shift_m[1] + error[1]) / 10.0)
    errors = np.array(errors_px)
    assert len(errors) >= 10, "AROSICS was almost never conclusive on sharp-edged scenes"
    assert np.median(errors) < 0.3  # a wrong sign would put every error near 3 px
    # A rare confident miss is expected (1 in 84 in a 160-trial study); a systematic one is not.
    assert (errors > CFG.agreement_tolerance_px).mean() <= 0.10


def test_no_shift_is_reported_as_no_shift():
    reference, target = _pair(seed=2, error_m=(0.0, 0.0))
    result = _run(reference, target)
    assert result.status == "ok" and math.hypot(*result.shift_m) < 0.1


def test_disabled_missing_and_tiny_windows_are_skipped_not_failed(monkeypatch):
    reference, target = _pair(seed=2, error_m=(10.0, 0.0))
    assert _run(reference, target, RegistrationConfig(arosics_mode="off")).reason == "AROSICS_DISABLED"

    small = np.zeros(SHAPE, bool)
    small[:20, :20] = True
    assert _run(reference, target, valid_ref=small, valid_tgt=small).reason == "AROSICS_WINDOW_TOO_SMALL"
    assert _run(reference, target, valid_ref=np.zeros(SHAPE, bool)).reason == "NO_OVERLAP"

    monkeypatch.setattr(arosics_estimator, "is_available", lambda: False)
    result = _run(reference, target)
    assert (result.status, result.reason) == ("skipped", "AROSICS_NOT_INSTALLED")


def test_a_crash_inside_arosics_is_contained(monkeypatch):
    import arosics

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(arosics, "COREG", explode)
    result = _run(*_pair(seed=2, error_m=(10.0, 0.0)))
    assert result.status == "inconclusive" and result.reason.startswith("AROSICS_ERROR") and "boom" in result.reason


class _FakeCoreg:
    def __init__(self, *args, success=True, reliability=90.0, shift=(3.0, -2.0), **kwargs):
        self.success, self.shift_reliability = success, reliability
        self.x_shift_map, self.y_shift_map = shift

    def calculate_spatial_shifts(self):
        pass


@pytest.mark.parametrize("fake, status, reason", [
    (dict(success=False), "inconclusive", "AROSICS_NO_SHIFT"),
    (dict(reliability=None), "inconclusive", "AROSICS_NO_RELIABILITY"),
    (dict(reliability=59.9), "inconclusive", "AROSICS_LOW_RELIABILITY"),
    (dict(reliability=60.0), "ok", None),
])
def test_reliability_decides_whether_the_answer_counts(monkeypatch, fake, status, reason):
    import arosics
    monkeypatch.setattr(arosics, "COREG", lambda *a, **k: _FakeCoreg(*a, **fake, **k))
    result = _run(*_pair(seed=2, error_m=(10.0, 0.0)))
    assert (result.status, result.reason) == (status, reason)


# -- gate G4 ------------------------------------------------------------------------------------------------------

NCC_SHIFT_M = (-15.0, 12.0)


@pytest.mark.parametrize("aro, passed, value", [
    (ArosicsResult("ok", shift_m=(-15.0, 12.0)), True, 0.0),
    (ArosicsResult("ok", shift_m=(-15.0 + 10.0, 12.0)), True, 1.0),    # exactly at the tolerance (one pixel)
    (ArosicsResult("ok", shift_m=(-15.0 + 10.1, 12.0)), False, 1.01),
    (ArosicsResult("ok", shift_m=(15.0, -12.0)), False, 3.84),         # the opposite sign: the classic bug
    (ArosicsResult("inconclusive", "AROSICS_LOW_RELIABILITY", shift_m=(40.0, 40.0)), True, None),  # no opinion
    (ArosicsResult("skipped", "AROSICS_NOT_INSTALLED"), True, None),
])
def test_agreement_gate(aro, passed, value):
    result = gates.gate_agreement(NCC_SHIFT_M, aro, 10.0, CFG)
    assert result.passed is passed and result.reason_code == "ESTIMATOR_DISAGREEMENT"
    if value is None:
        assert result.value is None
    else:
        assert result.value == pytest.approx(value, abs=0.01)


def test_disagreement_only_rejects_in_veto_mode():
    disagreeing = ArosicsResult("ok", shift_m=(15.0, -12.0))
    advisory = gates.gate_agreement(NCC_SHIFT_M, disagreeing, 10.0, RegistrationConfig())  # the default mode
    veto = gates.gate_agreement(NCC_SHIFT_M, disagreeing, 10.0, RegistrationConfig(arosics_mode="veto"))

    assert RegistrationConfig().arosics_mode == "advisory"
    assert not advisory.passed and advisory.advisory and gates.first_failure([advisory]) is None
    assert not veto.passed and not veto.advisory and gates.first_failure([veto]) is veto
    assert advisory.as_dict()["advisory"] is True
