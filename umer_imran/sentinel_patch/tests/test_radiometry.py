"""Reflectance conversion must give identical results across the processing-baseline offset change."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel_patch.errors import RadiometryUnknownError
from sentinel_patch.io.radiometry import (
    dn_to_reflectance, parse_processing_baseline, resolve_radiometry,
)

from . import synthetic


@pytest.fixture
def truth(scene, s2_transform, s2_shape):
    return synthetic.sample_scene(scene, s2_transform, s2_shape)


@pytest.mark.parametrize("baseline, offset", [("03.00", 0.0), ("02.14", 0.0), ("04.00", -0.1), ("05.11", -0.1)])
def test_radiometry_baselines(truth, baseline, offset):
    dn = synthetic.reflectance_to_dn(truth, offset=offset)
    params = resolve_radiometry(processing_baseline=baseline)
    assert (params.scale, params.offset, params.source) == (1e-4, offset, "processing_baseline")
    assert np.allclose(dn_to_reflectance(dn, params), truth, atol=1.5e-4)


def test_declared_scale_offset_matches_baseline_rule(truth):
    """Earth Search declares scale 0.0001 / offset -0.1 on baseline 05.xx items."""
    dn = synthetic.reflectance_to_dn(truth, offset=-0.1)
    declared = resolve_radiometry(scale=1e-4, offset=-0.1, processing_baseline="05.11")
    from_baseline = resolve_radiometry(processing_baseline="05.11")
    assert declared.source == "stac_raster_bands" and declared.warnings == ()
    assert np.array_equal(dn_to_reflectance(dn, declared), dn_to_reflectance(dn, from_baseline))


def test_offset_is_applied_exactly_once():
    dn = np.array([[1000, 6000]], dtype=np.uint16)  # 1000 DN is exactly zero reflectance after the offset
    refl = dn_to_reflectance(dn, resolve_radiometry(processing_baseline="05.11"))
    assert refl == pytest.approx(np.array([[0.0, 0.5]]), abs=1e-6)


def test_declared_metadata_beats_baseline_but_flags_disagreement():
    params = resolve_radiometry(scale=1e-4, offset=0.0, processing_baseline="05.11")
    assert params.offset == 0.0
    assert params.warnings == ("RADIOMETRY_OFFSET_BASELINE_MISMATCH",)


@pytest.mark.parametrize("kwargs", [
    {},
    {"processing_baseline": "garbage"},
    {"processing_baseline": ""},
    {"scale": 1e-4},
    {"offset": -0.1},
    {"scale": 0.0, "offset": -0.1},
    {"scale": -1e-4, "offset": 0.0},
])
def test_unknown_radiometry_is_refused_not_guessed(kwargs):
    with pytest.raises(RadiometryUnknownError):
        resolve_radiometry(**kwargs)


@pytest.mark.parametrize("value, expected", [
    ("05.11", (5, 11)), ("04.00", (4, 0)), ("N0511", (5, 11)), ("N0400", (4, 0)),
    (4.0, (4, 0)), (5.11, (5, 11)), (None, None), ("", None), ("5.11", None), ("v5", None),
])
def test_processing_baseline_parsing(value, expected):
    assert parse_processing_baseline(value) == expected


def test_nodata_becomes_nan_and_negative_values_are_kept():
    dn = np.array([[0, 500, 1000]], dtype=np.uint16)
    refl = dn_to_reflectance(dn, resolve_radiometry(processing_baseline="05.11"))
    assert np.isnan(refl[0, 0])
    assert refl[0, 1] == pytest.approx(-0.05, abs=1e-6)  # below the offset: legitimately negative, not clipped
    assert refl.dtype == np.float32
