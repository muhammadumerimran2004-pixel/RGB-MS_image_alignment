"""S4: the drone band degraded to Sentinel's resolution, on Sentinel's grid."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from sentinel_patch.config.schema import ReferenceConfig
from sentinel_patch.errors import ConfigurationError, ResourceLimitError
from sentinel_patch.registration.reference import build_reference_band

from . import scene_files, synthetic

GRID_SHAPE = (120, 120)
GRID = Affine(10.0, 0, 500200.0, 0, -10.0, 3499800.0)  # 1.2 km window; the drone covers its middle 560 m


@pytest.fixture(scope="module")
def drone(tmp_path_factory):
    scene = scene_files.full_scene()
    ms_path, _ = scene_files.write_drone_rasters(tmp_path_factory.mktemp("drone"), scene)
    return scene, ms_path


def _build(drone, band="nir", **overrides):
    scene, ms_path = drone
    values, fraction = build_reference_band(
        ms_path, scene_files.MS_BAND_MAP[band], GRID, GRID_SHAPE, scene_files.CRS_UTM42,
        ReferenceConfig(**overrides),
    )
    return scene, values, fraction


def test_reference_matches_the_scene_where_the_drone_has_data(drone):
    scene, values, _ = _build(drone)
    x, y = synthetic.pixel_centre_xy(GRID, GRID_SHAPE)
    truth = 3.0 * scene(x, y) + 0.1  # the drone camera's own gain and offset
    valid = np.isfinite(values)
    assert valid.sum() > 3000
    assert np.corrcoef(values[valid], truth[valid])[0, 1] > 0.999
    assert np.abs(values[valid] - truth[valid]).mean() < 0.03  # the PSF blur costs a little on 20-60 m blobs


def test_coverage_follows_the_drone_footprint(drone):
    _, values, fraction = _build(drone)
    assert np.isnan(values[:10]).all() and np.isnan(values[-10:]).all()  # buffer with no drone data
    assert np.isnan(values[:, :10]).all() and np.isnan(values[:, -10:]).all()
    assert fraction[60, 60] == pytest.approx(1.0) and fraction[0, 0] == 0.0
    # The MS raster is valid over 560 m x 560 m = 56 x 56 cells of 10 m.
    assert np.isfinite(values).sum() == pytest.approx(56 * 56, abs=2 * 56)


def test_data_edges_are_not_darkened(drone):
    """Normalised blurring: cells at the edge of the data must not be pulled towards zero."""
    scene, values, _ = _build(drone)
    x, y = synthetic.pixel_centre_xy(GRID, GRID_SHAPE)
    truth = 3.0 * scene(x, y) + 0.1
    edge = np.zeros(GRID_SHAPE, bool)
    inside = np.isfinite(values)
    for shift in (1, -1):
        for axis in (0, 1):
            edge |= inside & ~np.roll(inside, shift, axis=axis)
    assert edge.sum() > 100
    assert np.abs(values[edge] - truth[edge]).mean() < 0.05


def test_drone_outside_the_window_gives_an_empty_reference(drone):
    scene, ms_path = drone
    far = Affine(10.0, 0, 520000.0, 0, -10.0, 3499800.0)
    values, fraction = build_reference_band(ms_path, 4, far, (20, 20), scene_files.CRS_UTM42, ReferenceConfig())
    assert np.isnan(values).all() and fraction.max() == 0.0


def test_intermediate_grid_must_divide_the_pixel_size(drone):
    with pytest.raises(ConfigurationError):
        _build(drone, intermediate_gsd_m=3.0)


def test_oversized_windows_are_refused_not_attempted(drone):
    scene, ms_path = drone
    huge_shape = (6000, 6000)  # 60 km window at 1 m intermediate resolution
    with pytest.raises(ResourceLimitError):
        build_reference_band(ms_path, 4, GRID, huge_shape, scene_files.CRS_UTM42, ReferenceConfig())


def test_overflow_garbage_along_the_orthophoto_edge_does_not_reach_the_reference(drone, tmp_path):
    """Real aligned orthophotos carry values near 2**32 along their outermost edge (seen in all 7 files tried)."""
    import rasterio
    scene, clean_path = drone
    dirty_path = tmp_path / "dirty.tif"
    with rasterio.open(clean_path) as src:
        profile, bands = src.profile, src.read()
    rows, cols = np.nonzero(bands[2] != 0)
    edge = (rows == rows.min()) | (cols == cols.min()) | (rows == rows.max()) | (cols == cols.max())
    bands[:, rows[edge], cols[edge]] = np.float32(2**32)  # the whole outer ring of valid pixels
    with rasterio.open(dirty_path, "w", **profile) as dst:
        dst.write(bands)

    clean = build_reference_band(clean_path, 3, GRID, GRID_SHAPE, scene_files.CRS_UTM42, ReferenceConfig())[0]
    dirty = build_reference_band(dirty_path, 3, GRID, GRID_SHAPE, scene_files.CRS_UTM42, ReferenceConfig())[0]
    assert np.nanmax(dirty) < 10.0  # without the cleaning this is around 1e8
    both = np.isfinite(clean) & np.isfinite(dirty)
    assert both.sum() > 2500 and np.abs(clean[both] - dirty[both]).max() < 0.05  # the ring costs a few edge cells


def test_cleaning_leaves_a_constant_band_and_real_contrast_alone(drone):
    from sentinel_patch.registration.reference import _drop_outlier_cells
    weight = np.ones((50, 50))
    assert _drop_outlier_cells(np.full((50, 50), 0.3), weight, 50.0) == 0
    contrast = np.random.default_rng(1).normal(0.1, 0.02, (50, 50))
    contrast[10:20, 10:20] += 0.5  # a bright road: 25 sigma, real structure
    assert _drop_outlier_cells(contrast, np.ones((50, 50)), 50.0) == 0
    contrast[0, 0] = 4.29e9
    assert _drop_outlier_cells(contrast, weight, 50.0) == 1 and weight[0, 0] == 0
