import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from rasterio.transform import Affine

from drone_alignment.alignment.arosics_matcher import (
    is_arosics_available,
    match_arosics,
    ArosicsUnavailableError,
)
from drone_alignment.alignment.feature_matcher import InsufficientMatchesError
from drone_alignment.alignment.transform_estimator import TransformUnreliableError
from drone_alignment.config.schema import (
    AlignmentConfig, ArosicsConfig, ArosicsGlobalCandidateConfig, TransformValidationConfig,
)


def test_is_arosics_available():
    # Verify it returns a bool without raising
    result = is_arosics_available()
    assert isinstance(result, bool)


def test_match_arosics_unavailable_raises(monkeypatch):
    # Simulate missing arosics module
    monkeypatch.setitem(sys.modules, "arosics", None)

    rgb = np.ones((100, 100), dtype=np.float32)
    ms = np.ones((100, 100), dtype=np.float32)
    mask = np.ones((100, 100), dtype=bool)
    transform = Affine.identity()

    with pytest.raises(ArosicsUnavailableError):
        match_arosics(
            rgb_band=rgb,
            ms_band=ms,
            rgb_valid_mask=mask,
            ms_valid_mask=mask,
            registration_transform=transform,
            crs="EPSG:32632",
            registration_gsd=0.05,
            arosics_config=ArosicsGlobalCandidateConfig(),
            transform_config=TransformValidationConfig(),
        )


def _spec_coreg_mock(**attrs):
    """A COREG mock restricted to AROSICS' real attribute surface.

    Using a plain MagicMock() lets a wrong attribute name (e.g. `.reliability`
    instead of `.shift_reliability`) silently return another MagicMock instead
    of raising, which is exactly how the reliability-gate bug went undetected.
    """
    mock = MagicMock(spec=["calculate_spatial_shifts", "success", "x_shift_px", "y_shift_px", "shift_reliability"])
    for name, value in attrs.items():
        setattr(mock, name, value)
    return mock


def test_match_arosics_with_mocked_coreg_success(monkeypatch):
    mock_arosics = MagicMock()
    mock_coreg_instance = _spec_coreg_mock(
        success=True, x_shift_px=4.0, y_shift_px=-2.5, shift_reliability=88.0,
    )
    mock_arosics.COREG.return_value = mock_coreg_instance

    monkeypatch.setitem(sys.modules, "arosics", mock_arosics)

    rgb = np.ones((64, 64), dtype=np.float32)
    ms = np.ones((64, 64), dtype=np.float32)
    mask = np.ones((64, 64), dtype=bool)
    transform = Affine.translation(100.0, 200.0)
    gsd = 0.05
    arosics_cfg = ArosicsGlobalCandidateConfig(min_reliability=30.0)
    trans_cfg = TransformValidationConfig(max_translation_m=50.0)

    res = match_arosics(
        rgb_band=rgb,
        ms_band=ms,
        rgb_valid_mask=mask,
        ms_valid_mask=mask,
        registration_transform=transform,
        crs="EPSG:32632",
        registration_gsd=gsd,
        arosics_config=arosics_cfg,
        transform_config=trans_cfg,
    )

    assert res.method == "arosics"
    assert res.translation_px == (4.0, -2.5)
    assert pytest.approx(res.translation_m[0]) == 4.0 * gsd
    assert pytest.approx(res.translation_m[1]) == -2.5 * gsd
    assert res.inlier_ratio == pytest.approx(0.88)
    np.testing.assert_allclose(res.matrix, [[1.0, 0.0, 4.0], [0.0, 1.0, -2.5]])
    coreg_kwargs = mock_arosics.COREG.call_args.kwargs
    assert coreg_kwargs["ws"] == (256, 256)
    assert "grid_res" not in coreg_kwargs


def test_match_arosics_coreg_unsuccessful_raises(monkeypatch):
    mock_arosics = MagicMock()
    mock_coreg_instance = _spec_coreg_mock(success=False)
    mock_arosics.COREG.return_value = mock_coreg_instance

    monkeypatch.setitem(sys.modules, "arosics", mock_arosics)

    rgb = np.ones((64, 64), dtype=np.float32)
    ms = np.ones((64, 64), dtype=np.float32)
    mask = np.ones((64, 64), dtype=bool)

    with pytest.raises(InsufficientMatchesError, match="could not determine spatial shift"):
        match_arosics(
            rgb_band=rgb,
            ms_band=ms,
            rgb_valid_mask=mask,
            ms_valid_mask=mask,
            registration_transform=Affine.identity(),
            crs="EPSG:32632",
            registration_gsd=0.05,
            arosics_config=ArosicsGlobalCandidateConfig(),
            transform_config=TransformValidationConfig(),
        )


def test_match_arosics_missing_reliability_rejects(monkeypatch):
    # No shift_reliability set at all (e.g. AROSICS considered the shift zero) -
    # this must be treated as a failed gate, not silently pass with inlier_ratio=1.0.
    mock_arosics = MagicMock()
    mock_coreg_instance = _spec_coreg_mock(success=True, x_shift_px=0.0, y_shift_px=0.0, shift_reliability=None)
    mock_arosics.COREG.return_value = mock_coreg_instance

    monkeypatch.setitem(sys.modules, "arosics", mock_arosics)

    rgb = np.ones((64, 64), dtype=np.float32)
    ms = np.ones((64, 64), dtype=np.float32)
    mask = np.ones((64, 64), dtype=bool)

    with pytest.raises(InsufficientMatchesError, match="did not report a shift reliability"):
        match_arosics(
            rgb_band=rgb,
            ms_band=ms,
            rgb_valid_mask=mask,
            ms_valid_mask=mask,
            registration_transform=Affine.identity(),
            crs="EPSG:32632",
            registration_gsd=0.05,
            arosics_config=ArosicsGlobalCandidateConfig(),
            transform_config=TransformValidationConfig(),
        )


def test_match_arosics_low_reliability_raises(monkeypatch):
    mock_arosics = MagicMock()
    mock_coreg_instance = _spec_coreg_mock(
        success=True, shift_reliability=20.0, x_shift_px=1.0, y_shift_px=1.0,
    )
    mock_arosics.COREG.return_value = mock_coreg_instance

    monkeypatch.setitem(sys.modules, "arosics", mock_arosics)

    rgb = np.ones((64, 64), dtype=np.float32)
    ms = np.ones((64, 64), dtype=np.float32)
    mask = np.ones((64, 64), dtype=bool)

    with pytest.raises(InsufficientMatchesError, match="below minimum threshold"):
        match_arosics(
            rgb_band=rgb,
            ms_band=ms,
            rgb_valid_mask=mask,
            ms_valid_mask=mask,
            registration_transform=Affine.identity(),
            crs="EPSG:32632",
            registration_gsd=0.05,
            arosics_config=ArosicsGlobalCandidateConfig(min_reliability=50.0),
            transform_config=TransformValidationConfig(),
        )


def test_match_arosics_exceeds_max_translation_raises(monkeypatch):
    mock_arosics = MagicMock()
    mock_coreg_instance = _spec_coreg_mock(
        success=True, shift_reliability=90.0,
        x_shift_px=500.0,  # 500 px * 0.1 m/px = 50m
        y_shift_px=500.0,
    )
    mock_arosics.COREG.return_value = mock_coreg_instance

    monkeypatch.setitem(sys.modules, "arosics", mock_arosics)

    rgb = np.ones((64, 64), dtype=np.float32)
    ms = np.ones((64, 64), dtype=np.float32)
    mask = np.ones((64, 64), dtype=bool)

    with pytest.raises(TransformUnreliableError, match="exceeds max allowed threshold"):
        match_arosics(
            rgb_band=rgb,
            ms_band=ms,
            rgb_valid_mask=mask,
            ms_valid_mask=mask,
            registration_transform=Affine.identity(),
            crs="EPSG:32632",
            registration_gsd=0.1,
            arosics_config=ArosicsGlobalCandidateConfig(),
            transform_config=TransformValidationConfig(max_translation_m=10.0),
        )


@pytest.mark.skipif(not is_arosics_available(), reason="arosics package is not installed")
def test_match_arosics_real_coreg_known_shift():
    """Run the real (not mocked) AROSICS COREG on a synthetic textured pair with a
    known integer-pixel shift, to lock down the reliability attribute name and the
    sign/scale of the recovered translation against the actual library, not a mock.

    The shift and array-slicing convention below were empirically calibrated
    against the installed AROSICS version: for a target band whose content at
    array index (row, col) equals the reference band's content at
    (row + shift_y, col + shift_x), COREG reports x_shift_px == shift_x and
    y_shift_px == shift_y (both exactly, for a clean integer shift).
    """
    rng = np.random.default_rng(1)
    size = 400
    grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
    checker = (((grid_x // 20) + (grid_y // 20)) % 2).astype(np.float32) * 100.0
    noise = rng.uniform(0.0, 20.0, size=(size, size)).astype(np.float32)
    base = checker + noise

    true_shift_x_px, true_shift_y_px = 4, -3
    margin = 30
    rgb_band = base[margin:-margin, margin:-margin].copy()
    height, width = rgb_band.shape
    ms_band = base[
        margin + true_shift_y_px: margin + true_shift_y_px + height,
        margin + true_shift_x_px: margin + true_shift_x_px + width,
    ].copy()

    mask = np.ones((height, width), dtype=bool)
    transform = Affine.identity()
    gsd = 0.05
    arosics_cfg = ArosicsGlobalCandidateConfig(min_reliability=10.0, max_shift_px=20)
    trans_cfg = TransformValidationConfig(max_translation_m=50.0)

    res = match_arosics(
        rgb_band=rgb_band,
        ms_band=ms_band,
        rgb_valid_mask=mask,
        ms_valid_mask=mask,
        registration_transform=transform,
        crs="EPSG:32632",
        registration_gsd=gsd,
        arosics_config=arosics_cfg,
        transform_config=trans_cfg,
    )

    assert res.method == "arosics"
    assert res.translation_px[0] == pytest.approx(true_shift_x_px, abs=0.1)
    assert res.translation_px[1] == pytest.approx(true_shift_y_px, abs=0.1)
    assert res.inlier_ratio > 0.9  # real shift_reliability must have been read and gate passed


def test_pipeline_arosics_integration_fallback(synthetic_geo_tiff_pair, tmp_path: Path, monkeypatch):
    """The global AROSICS candidate (match_arosics, inside _estimate_global_candidate)
    is accepted when classical detectors fail.

    This deliberately calls _estimate_verified_global_context + _publish_global_context
    directly rather than the top-level align_orthomosaics(): as of this pipeline
    version, align_orthomosaics() always attempts AROSICS COREG_LOCAL first
    (_align_with_arosics_local) and, on any rejection, explicitly sets
    fallback_cfg.arosics.enabled = False before running classical/global
    estimation - so the global match_arosics candidate this test exercises is
    unreachable from align_orthomosaics() today (see AROSICS review finding
    A10 / blueprint P2.1, which removes that override). Exercising the two
    building blocks directly keeps this test meaningful without depending on
    that Phase 2 architecture change.
    """
    import drone_alignment.pipeline as pipeline_mod
    from drone_alignment.alignment.transform_estimator import TransformResult, TransformType

    # Mock detect_features to fail so classical search finds no candidate
    def fake_detect(*args, **kwargs):
        raise InsufficientMatchesError("Simulated classical detector failure.")

    monkeypatch.setattr(pipeline_mod, "detect_features", fake_detect)

    # Mock match_arosics to return the ground-truth synthetic shift (-2.0 px, -1.0 px)
    # in registration grid
    def fake_match_arosics(*args, **kwargs):
        return TransformResult(
            matrix=np.array([[1.0, 0.0, -2.0], [0.0, 1.0, -1.0]], dtype=np.float64),
            transform_type=TransformType.AFFINE,
            translation_px=(-2.0, -1.0),
            translation_m=(-0.1, -0.05),
            rotation_deg=0.0,
            scale=(1.0, 1.0),
            inlier_ratio=0.95,
            num_inliers=1,
            num_total_matches=1,
            method="arosics",
            channel_pair="unknown",
        )

    monkeypatch.setattr(pipeline_mod, "match_arosics", fake_match_arosics)

    output_dir = tmp_path / "arosics_pipeline_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AlignmentConfig(
        arosics=ArosicsConfig(enabled=True),
    )

    context = pipeline_mod._estimate_verified_global_context(
        synthetic_geo_tiff_pair["rgb_path"], synthetic_geo_tiff_pair["ms_path"], config, pipeline_mod.logger,
    )
    result = pipeline_mod._publish_global_context(context, output_dir, config, pipeline_mod.logger)

    assert result.aligned_ms_path.exists()
    assert result.report_json_path.exists()
    assert result.preview_image_path.exists()

    import json
    with open(result.report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["transform"]["method"] == "arosics"
