from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_bounds

from drone_alignment.alignment.arosics_matcher import is_arosics_available
from drone_alignment.alignment.arosics_staging import (
    STAGE_NODATA,
    ArosicsExecutionError,
    compute_global_map_correction,
    create_whitespace_safe_alias,
    stage_precorrected_target_band,
    stage_reference_band,
)
from drone_alignment.io.reader import read_metadata


def _write_ms_tif(path: Path, transform: Affine, size: int = 40, alpha: bool = False) -> None:
    rng = np.random.default_rng(0)
    band = rng.uniform(0.0, 1.0, size=(size, size)).astype(np.float32)
    count = 2 if alpha else 1
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=count, dtype="float32",
        crs="EPSG:32643", transform=transform, nodata=None,
    ) as dst:
        dst.write(band, 1)
        if alpha:
            alpha_band = np.full((size, size), 255, dtype=np.float32)
            alpha_band[0, 0] = 0  # one deliberately masked-out corner pixel
            dst.write(alpha_band, 2)
            dst.colorinterp = [rasterio.enums.ColorInterp.gray, rasterio.enums.ColorInterp.alpha]


def test_translation_correction_is_pixel_identical(tmp_path: Path):
    registration_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)  # 100x100, 1 unit/px, north-up
    global_matrix = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]])  # pure translation
    ms_path = tmp_path / "ms.tif"
    ms_transform = Affine(0.5, 0.0, 10.0, 0.0, -0.5, 90.0)
    _write_ms_tif(ms_path, ms_transform)
    ms_meta = read_metadata(ms_path)

    correction = compute_global_map_correction(registration_transform, global_matrix, ms_transform)
    assert correction.is_translation_only

    out_path = tmp_path / "staged_target.tif"
    staged = stage_precorrected_target_band(ms_meta, 1, correction, out_path)

    assert not staged.resampled
    assert staged.staged_to_source_px == Affine.identity()
    with rasterio.open(ms_path) as source, rasterio.open(out_path) as staged_ds:
        source_values = source.read(1)
        staged_values = staged_ds.read(1)
        # Pixel-for-pixel identical values; only the geotransform origin moved.
        np.testing.assert_array_equal(source_values, staged_values)
        assert staged_ds.transform != source.transform
        assert staged_ds.transform == correction.corrected_ms_transform


def test_rotation_correction_maps_back_to_source_pixels(tmp_path: Path):
    registration_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    # A small but nonzero rotation + scale, e.g. from an accepted ORB/SIFT affine.
    theta = np.radians(2.0)
    scale = 1.02
    a, b = scale * np.cos(theta), -scale * np.sin(theta)
    c, d = scale * np.sin(theta), scale * np.cos(theta)
    global_matrix = np.array([[a, b, 1.5], [c, d, -0.8]])
    ms_path = tmp_path / "ms_rot.tif"
    ms_transform = Affine(0.5, 0.0, 10.0, 0.0, -0.5, 90.0)
    _write_ms_tif(ms_path, ms_transform)
    ms_meta = read_metadata(ms_path)

    correction = compute_global_map_correction(registration_transform, global_matrix, ms_transform)
    assert not correction.is_translation_only

    out_path = tmp_path / "staged_target_rot.tif"
    staged = stage_precorrected_target_band(ms_meta, 1, correction, out_path)

    assert staged.resampled
    # For sample points on the staged grid, staged_to_source_px(p) mapped through
    # the *original* transform must land on the same map position as the staged
    # grid's own transform at p - i.e. the algebraic identity the composition
    # is built from: target_transform(p) == corrected_ms_transform(staged_to_source_px(p)).
    sample_points = [(0.0, 0.0), (10.0, 5.0), (30.0, 20.0), (5.5, 33.2)]
    for p in sample_points:
        expected_map = staged.transform * p
        source_px = staged.staged_to_source_px * p
        actual_map = correction.corrected_ms_transform * source_px
        assert expected_map[0] == pytest.approx(actual_map[0], abs=1e-6)
        assert expected_map[1] == pytest.approx(actual_map[1], abs=1e-6)


def test_staged_nodata_from_alpha(tmp_path: Path):
    ms_path = tmp_path / "ms_alpha.tif"
    ms_transform = Affine(0.5, 0.0, 10.0, 0.0, -0.5, 90.0)
    _write_ms_tif(ms_path, ms_transform, alpha=True)
    ms_meta = read_metadata(ms_path)
    assert ms_meta.alpha_band_index == 2

    registration_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    global_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    correction = compute_global_map_correction(registration_transform, global_matrix, ms_transform)

    out_path = tmp_path / "staged_alpha.tif"
    stage_precorrected_target_band(ms_meta, 1, correction, out_path)

    with rasterio.open(out_path) as staged_ds:
        values = staged_ds.read(1)
        assert values[0, 0] == STAGE_NODATA
        assert values[1, 1] != STAGE_NODATA


def test_stage_reference_band_native_resolution_no_resampling(tmp_path: Path):
    rgb_path = tmp_path / "rgb.tif"
    rgb_transform = Affine(0.3, 0.0, 0.0, 0.0, -0.3, 50.0)
    _write_ms_tif(rgb_path, rgb_transform, size=25)
    rgb_meta = read_metadata(rgb_path)

    out_path = tmp_path / "staged_ref.tif"
    staged = stage_reference_band(rgb_meta, 1, out_path)

    assert not staged.resampled
    assert staged.transform == rgb_transform
    with rasterio.open(rgb_path) as source, rasterio.open(out_path) as staged_ds:
        np.testing.assert_array_equal(source.read(1), staged_ds.read(1))
        assert staged_ds.width == source.width and staged_ds.height == source.height


def test_compute_global_map_correction_rejects_non_affine_matrix():
    registration_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    homography = np.eye(3)
    ms_transform = Affine(0.5, 0.0, 10.0, 0.0, -0.5, 90.0)
    with pytest.raises(ArosicsExecutionError, match="affine"):
        compute_global_map_correction(registration_transform, homography, ms_transform)


def test_create_whitespace_safe_alias_always_returns_a_vrt(tmp_path: Path):
    """Regression: a hard link shares the source file's own metadata verbatim,
    so a source with a band-count/band-name metadata-list length mismatch (seen
    on some real ODM outputs) survives straight through and later crashes
    GeoArray.save() with an IndexError. Only a VRT lets that per-band metadata
    be cleared, so this must never take a hard-link shortcut."""
    ms_path = tmp_path / "ms.tif"
    _write_ms_tif(ms_path, Affine(0.5, 0.0, 10.0, 0.0, -0.5, 90.0), alpha=True)
    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()

    alias_path = create_whitespace_safe_alias(ms_path, alias_dir, "target_full")

    assert alias_path.suffix == ".vrt"
    assert alias_path.exists()
    with rasterio.open(alias_path) as alias_ds, rasterio.open(ms_path) as source_ds:
        np.testing.assert_array_equal(alias_ds.read(), source_ds.read())


@pytest.mark.skipif(not is_arosics_available(), reason="geoarray (an arosics dependency) is not installed")
def test_create_whitespace_safe_alias_round_trips_through_geoarray_save(tmp_path: Path):
    """The exact operation that previously raised IndexError: AROSICS' DESHIFTER
    loads the aliased target with GeoArray and later calls .save() on it, which
    indexes into metadata.band_meta['band_names'] once per band."""
    from geoarray import GeoArray

    ms_path = tmp_path / "ms_multiband.tif"
    _write_ms_tif(ms_path, Affine(0.5, 0.0, 10.0, 0.0, -0.5, 90.0), alpha=True)
    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()

    alias_path = create_whitespace_safe_alias(ms_path, alias_dir, "target_full")

    geo_arr = GeoArray(str(alias_path))
    out_path = tmp_path / "roundtrip.tif"
    geo_arr.save(str(out_path), fmt="GTIFF")  # previously: IndexError: list index out of range
    assert out_path.exists()
    with rasterio.open(out_path) as roundtripped:
        assert roundtripped.count == geo_arr.bands
