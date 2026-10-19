"""The `bands/` folder must be consumable by crop_health_sentinel with no adaptation.

The easy mistakes here are silent: forgetting the reflectance scale, or leaving the -0.1 offset in
or out. Either would make every NDVI wrong by roughly 0.1 without raising an error.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest
import rasterio
from rasterio.windows import from_bounds

from crop_health_sentinel.config import load_config as load_crop_health_config
from crop_health_sentinel.models import SceneStatus
from crop_health_sentinel.pipeline import run_crop_health_report
from sentinel_patch import run_sentinel_patch
from sentinel_patch.config.schema import parse_config

from . import scene_files, synthetic

ERROR_M = (15.0, -12.0)
full_scene = scene_files.full_scene()


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    root = tmp_path_factory.mktemp("contract")
    scene = scene_files.full_scene()
    ms, rgb = scene_files.write_drone_rasters(root / "drone", scene)
    folder = scene_files.write_s2_scene(root / "s2", scene, content_offset_m=ERROR_M)
    result = run_sentinel_patch(ms, rgb, root / "out", flight_date=date(2026, 10, 10),
                                ms_band_map=scene_files.MS_BAND_MAP, config=parse_config({"local_folder": str(folder), "registration": {"arosics_mode": "off"}}))
    assert result.status == "verified"
    return root, result


def _crop_health(root, result):
    out = root / "out" / "sentinel"
    geometry = json.loads((out / "footprint.geojson").read_text(encoding="utf-8"))
    return run_crop_health_report(out / "bands", geometry, root / "health",
                                  {"farmId": "farm-1", "cropId": "crop-1", "imageryDate": "2026-10-12", "sceneId": "scene-1"},
                                  config=load_crop_health_config())


def test_crop_health_accepts_the_bands_and_reports_a_usable_scene(published):
    root, result = published
    health = _crop_health(root, result)
    assert health.scene_status is SceneStatus.ANALYTICS_READY
    assert health.spectral is not None and "ndvi" in health.spectral.indices


def test_ndvi_from_crop_health_equals_ndvi_from_our_reflectance(published):
    """Offset and scale are handled exactly once between the two packages."""
    root, result = published
    _crop_health(root, result)
    with rasterio.open(root / "health" / "phase_02" / "scene-1_ndvi.tif") as ndvi_file:
        theirs = ndvi_file.read(1)
        bounds, nodata, grid = ndvi_file.bounds, ndvi_file.nodata, ndvi_file.transform
    with rasterio.open(result.product_paths["analytic_10m"]) as ours:
        window = from_bounds(*bounds, transform=ours.transform).round_offsets().round_lengths()
        red = ours.read(3, window=window, boundless=True, fill_value=np.nan).astype(np.float64)
        nir = ours.read(4, window=window, boundless=True, fill_value=np.nan).astype(np.float64)
    direct = (nir - red) / (nir + red)

    compared = np.isfinite(direct) & (theirs != nodata) & np.isfinite(theirs)
    assert compared.sum() > 2000
    assert np.abs(theirs[compared] - direct[compared]).max() < 1e-4

    # Absolute check against the analytic scene: a mishandled -0.1 offset would be off by 0.3 or more.
    x, y = synthetic.pixel_centre_xy(grid, theirs.shape)
    f = full_scene(x, y)
    red_true, nir_true = scene_files.S2_BAND_FUNCTIONS["B04"](f), scene_files.S2_BAND_FUNCTIONS["B08"](f)
    truth = (nir_true - red_true) / (nir_true + red_true)
    assert np.abs(theirs[compared] - truth[compared]).mean() < 0.05


def test_band_files_have_the_layout_crop_health_expects(published):
    root, result = published
    bands = root / "out" / "sentinel" / "bands"
    with rasterio.open(bands / "B8.tif") as nir, rasterio.open(bands / "B5.tif") as red_edge, rasterio.open(bands / "SCL.tif") as scl:
        assert nir.count == red_edge.count == scl.count == 1 and nir.crs is not None
        assert abs(nir.transform.a) == 10.0 and abs(red_edge.transform.a) == 20.0 and abs(scl.transform.a) == 20.0
        assert nir.dtypes[0] == "float32" and scl.dtypes[0] == "uint8"
        assert np.nanmax(nir.read(1)) > 1000.0  # reflectance x 10000, not 0-1
