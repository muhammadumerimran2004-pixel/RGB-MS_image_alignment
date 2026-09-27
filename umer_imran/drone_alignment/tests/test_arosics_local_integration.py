"""End-to-end tests of run_arosics_local_refinement() against the real,
installed AROSICS library (skipped if it is not available).

These exercise the full path this pipeline actually runs: native-resolution
staging, real COREG_LOCAL tie-point detection and filtering, GCP composition,
AROSICS' own DESHIFTER warp, and the independent holdout/full-grid
verification - not a mocked substitute for any of it.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from affine import Affine

from drone_alignment.alignment.arosics_matcher import is_arosics_available
from drone_alignment.alignment.arosics_local import ArosicsLocalPublication, _affine_close, run_arosics_local_refinement
from drone_alignment.alignment.rejections import LocalRefinementRejected
from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.config.schema import AlignmentConfig, ArosicsConfig, RegistrationChannel, TransformType
from drone_alignment.io.reader import read_metadata
from drone_alignment.quality.metrics import SpatialResidualReport

pytestmark = pytest.mark.skipif(not is_arosics_available(), reason="arosics package is not installed")


def _textured_band(size: int, seed: int = 3) -> np.ndarray:
    """Smoothly-correlated synthetic texture: rich enough for reliable local
    matching, but (unlike a checkerboard) with no exact repeating period that
    could alias a phase-correlation match onto the wrong location."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1.0, size=(size, size)).astype(np.float32)
    band = cv2.GaussianBlur(base, (0, 0), sigmaX=3.0) * 200.0
    band += rng.uniform(0.0, 10.0, size=(size, size)).astype(np.float32)
    return band


def _write_pair(
    tmp_path: Path, rgb_arr: np.ndarray, ms_arr: np.ndarray, gsd: float,
) -> tuple[Path, Path, Affine]:
    transform = Affine(gsd, 0.0, 500000.0, 0.0, -gsd, 3000000.0)
    profile = dict(
        driver="GTiff", dtype="float32", width=rgb_arr.shape[1], height=rgb_arr.shape[0], count=1,
        crs="EPSG:32643", transform=transform, nodata=None,
    )
    rgb_path, ms_path = tmp_path / "rgb.tif", tmp_path / "ms.tif"
    with rasterio.open(rgb_path, "w", **profile) as dst:
        dst.write(rgb_arr, 1)
    with rasterio.open(ms_path, "w", **profile) as dst:
        dst.write(ms_arr, 1)
    return rgb_path, ms_path, transform


def _global_transform(tx: float, ty: float, gsd: float) -> TransformResult:
    return TransformResult(
        matrix=np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]]),
        transform_type=TransformType.AFFINE, translation_px=(tx, ty),
        translation_m=(tx * gsd, ty * gsd), rotation_deg=0.0, scale=(1.0, 1.0),
        inlier_ratio=1.0, num_inliers=10, num_total_matches=10, method="test", channel_pair="red-red",
    )


def _base_config() -> AlignmentConfig:
    return AlignmentConfig(
        rgb_red_band_index=1, ms_red_band_index=1,
        registration_channel_priority=[RegistrationChannel.RED],
        arosics=ArosicsConfig(),
    )


_NO_QUALITY = SpatialResidualReport(None, None, None, None, None, [], "PASS", True)


def test_smooth_local_distortion_is_accepted_and_genuinely_improves(tmp_path: Path):
    """A verified global translation leaves a smooth, slowly-varying residual
    (not expressible as a single global affine); AROSICS local refinement
    should find it, warp it out, and the *independent* holdout points
    (excluded from the GCP fit) should show real, large improvement."""
    size = 500
    gsd = 0.03
    rgb_arr = _textured_band(size)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    amplitude, wavelength = 3.0, 1000.0  # much larger than the matching window: smooth, not aliasing-prone
    dx = amplitude * np.sin(2 * np.pi * xx / wavelength)
    true_tx, true_ty = 2.0, -1.0
    src_x = (xx + dx + true_tx).astype(np.float32)
    src_y = (yy + true_ty).astype(np.float32)
    ms_arr = cv2.remap(rgb_arr, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    rgb_path, ms_path, transform = _write_pair(tmp_path, rgb_arr, ms_arr, gsd)
    rgb_meta, ms_meta = read_metadata(rgb_path), read_metadata(ms_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    publication = run_arosics_local_refinement(
        rgb_meta, ms_meta, np.ones((size, size), dtype=bool), transform, gsd,
        _global_transform(true_tx, true_ty, gsd), _NO_QUALITY, dict(
            driver="GTiff", dtype="float32", width=size, height=size, count=1,
            crs="EPSG:32643", transform=transform, nodata=None,
        ),
        _base_config(), output_dir,
    )

    assert isinstance(publication, ArosicsLocalPublication)
    assert publication.aligned_path.exists()
    holdout = publication.payload["holdout"]
    # The injected distortion has amplitude 3px; a correct warp should bring
    # the independent holdout residual down by a large margin, not just
    # barely pass the (looser) default gate.
    assert holdout["median_before_px"] > 1.5
    assert holdout["median_after_px"] < 0.5
    assert holdout["win_fraction"] == pytest.approx(1.0)

    with rasterio.open(publication.aligned_path) as aligned:
        assert aligned.width == size and aligned.height == size
        assert _affine_close(aligned.transform, transform)


def test_streaming_gdal_tps_engine_produces_a_comparable_result(tmp_path: Path):
    """The streaming GDAL TPS engine (used for rasters too large for DESHIFTER's
    in-memory warp) must apply the exact same GCP-based transform and produce
    a comparably accepted, comparably-improved result on the same scenario
    that passes with the default deshifter engine."""
    size = 500
    gsd = 0.03
    rgb_arr = _textured_band(size)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    amplitude, wavelength = 3.0, 1000.0
    dx = amplitude * np.sin(2 * np.pi * xx / wavelength)
    true_tx, true_ty = 2.0, -1.0
    src_x = (xx + dx + true_tx).astype(np.float32)
    src_y = (yy + true_ty).astype(np.float32)
    ms_arr = cv2.remap(rgb_arr, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    rgb_path, ms_path, transform = _write_pair(tmp_path, rgb_arr, ms_arr, gsd)
    rgb_meta, ms_meta = read_metadata(rgb_path), read_metadata(ms_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cfg = _base_config()
    cfg.arosics.local.warp_engine = "gdal_tps"

    publication = run_arosics_local_refinement(
        rgb_meta, ms_meta, np.ones((size, size), dtype=bool), transform, gsd,
        _global_transform(true_tx, true_ty, gsd), _NO_QUALITY, dict(
            driver="GTiff", dtype="float32", width=size, height=size, count=1,
            crs="EPSG:32643", transform=transform, nodata=None,
        ),
        cfg, output_dir,
    )

    assert publication.payload["engine"] == "gdal_tps"
    holdout = publication.payload["holdout"]
    assert holdout["median_before_px"] > 1.5
    assert holdout["median_after_px"] < 0.5
    with rasterio.open(publication.aligned_path) as aligned:
        assert aligned.width == size and aligned.height == size
        assert _affine_close(aligned.transform, transform)


def test_negligible_residual_after_global_correction_is_rejected(tmp_path: Path):
    """When the verified global result already leaves only a tiny, uniform
    residual, AROSICS local refinement must recognise there is nothing
    worthwhile left to correct (NO_LOCAL_GAIN) rather than publish a warp
    for noise-level differences."""
    size = 500
    gsd = 0.03
    rgb_arr = _textured_band(size)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    true_tx, true_ty = 2.3, -1.15
    claimed_tx, claimed_ty = 2.0, -1.0  # verified global result, ~0.3px off from the true shift
    src_x = (xx + true_tx).astype(np.float32)
    src_y = (yy + true_ty).astype(np.float32)
    ms_arr = cv2.remap(rgb_arr, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    rgb_path, ms_path, transform = _write_pair(tmp_path, rgb_arr, ms_arr, gsd)
    rgb_meta, ms_meta = read_metadata(rgb_path), read_metadata(ms_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with pytest.raises(LocalRefinementRejected) as excinfo:
        run_arosics_local_refinement(
            rgb_meta, ms_meta, np.ones((size, size), dtype=bool), transform, gsd,
            _global_transform(claimed_tx, claimed_ty, gsd), _NO_QUALITY, dict(
                driver="GTiff", dtype="float32", width=size, height=size, count=1,
                crs="EPSG:32643", transform=transform, nodata=None,
            ),
            _base_config(), output_dir,
        )

    assert excinfo.value.reason_code == "NO_LOCAL_GAIN"
    assert not (output_dir / f"{ms_path.stem}_aligned.tif").exists()


def test_mismatched_crs_is_rejected_without_touching_arosics(tmp_path: Path):
    size = 200
    gsd = 0.03
    rgb_arr = _textured_band(size)
    ms_arr = rgb_arr.copy()
    transform = Affine(gsd, 0.0, 500000.0, 0.0, -gsd, 3000000.0)
    rgb_path = tmp_path / "rgb.tif"
    ms_path = tmp_path / "ms.tif"
    rgb_profile = dict(driver="GTiff", dtype="float32", width=size, height=size, count=1,
                        crs="EPSG:32643", transform=transform, nodata=None)
    ms_profile = dict(rgb_profile, crs="EPSG:32644")  # deliberately different CRS
    with rasterio.open(rgb_path, "w", **rgb_profile) as dst:
        dst.write(rgb_arr, 1)
    with rasterio.open(ms_path, "w", **ms_profile) as dst:
        dst.write(ms_arr, 1)

    rgb_meta, ms_meta = read_metadata(rgb_path), read_metadata(ms_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with pytest.raises(LocalRefinementRejected) as excinfo:
        run_arosics_local_refinement(
            rgb_meta, ms_meta, np.ones((size, size), dtype=bool), transform, gsd,
            _global_transform(0.0, 0.0, gsd), _NO_QUALITY, dict(
                driver="GTiff", dtype="float32", width=size, height=size, count=1,
                crs="EPSG:32643", transform=transform, nodata=None,
            ),
            _base_config(), output_dir,
        )
    assert excinfo.value.reason_code == "UNSUPPORTED_CRS"


def test_homography_global_transform_is_rejected():
    from drone_alignment.io.reader import RasterMetadata

    homography_transform = TransformResult(
        matrix=np.eye(3), transform_type=TransformType.HOMOGRAPHY,
        translation_px=(0.0, 0.0), translation_m=(0.0, 0.0), rotation_deg=0.0, scale=(1.0, 1.0),
        inlier_ratio=1.0, num_inliers=10, num_total_matches=10,
    )
    fake_meta = RasterMetadata(
        path=Path("x.tif"), crs="EPSG:32643", transform=Affine.identity(), width=10, height=10,
        band_count=1, dtype="float32", nodata=None, bounds=(0, 0, 1, 1), gsd=0.03,
        band_descriptions=(None,), color_interpretations=("undefined",),
        alpha_band_index=None, spectral_band_indices=(1,),
    )
    with pytest.raises(LocalRefinementRejected) as excinfo:
        run_arosics_local_refinement(
            fake_meta, fake_meta, np.ones((10, 10), dtype=bool), Affine.identity(), 0.03,
            homography_transform, _NO_QUALITY, {"width": 10, "height": 10, "transform": Affine.identity(), "crs": "EPSG:32643"},
            _base_config(), Path("unused"),
        )
    assert excinfo.value.reason_code == "UNSUPPORTED_GLOBAL_TRANSFORM"


def test_arosics_unavailable_is_reported_as_rejection(monkeypatch, tmp_path: Path):
    """Even with the real package installed, simulate its absence and confirm
    this degrades to a clean rejection (safe global fallback), not a crash -
    matching how an optional dependency that genuinely isn't installed
    should behave for every other caller of run_arosics_local_refinement."""
    import drone_alignment.alignment.arosics_local as arosics_local_mod
    monkeypatch.setattr(arosics_local_mod, "is_arosics_available", lambda: False)

    from drone_alignment.io.reader import RasterMetadata
    fake_meta = RasterMetadata(
        path=Path("x.tif"), crs="EPSG:32643", transform=Affine.identity(), width=10, height=10,
        band_count=1, dtype="float32", nodata=None, bounds=(0, 0, 1, 1), gsd=0.03,
        band_descriptions=(None,), color_interpretations=("undefined",),
        alpha_band_index=None, spectral_band_indices=(1,),
    )
    with pytest.raises(LocalRefinementRejected) as excinfo:
        arosics_local_mod.run_arosics_local_refinement(
            fake_meta, fake_meta, np.ones((10, 10), dtype=bool), Affine.identity(), 0.03,
            _global_transform(0.0, 0.0, 0.03), _NO_QUALITY,
            {"width": 10, "height": 10, "transform": Affine.identity(), "crs": "EPSG:32643"},
            _base_config(), tmp_path,
        )
    assert excinfo.value.reason_code == "AROSICS_UNAVAILABLE"
