"""End-to-end, offline: synthetic drone stack + a mis-georeferenced Sentinel scene folder."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest
import rasterio

from sentinel_patch import SentinelPatchResult, run_sentinel_patch
from sentinel_patch.config.schema import parse_config

from . import scene_files, synthetic

FLIGHT = date(2026, 10, 10)  # the fixture scene is acquired two days later
ERROR_M = (15.0, -12.0)  # 19 m: the Sentinel grid is mis-georeferenced by this much; truth shift is (-15, +12)
TOLERANCE_M = 1.5


@pytest.fixture(scope="module")
def scene():
    return scene_files.full_scene()


@pytest.fixture
def drone(tmp_path, scene):
    return scene_files.write_drone_rasters(tmp_path / "drone", scene)


def _config(folder, **overrides):
    # These tests use the smooth, edgeless blob scene, on which frequency-domain AROSICS can be confidently wrong
    # and would add an AROSICS_DISAGREES warning; it is exercised on sharp-edged field scenes in test_phase2_pipeline.py.
    registration = {"arosics_mode": "off", **overrides.pop("registration", {})}
    return parse_config({"local_folder": str(folder), "registration": registration, **overrides})


def _run(tmp_path, drone, folder, **overrides) -> SentinelPatchResult:
    ms, rgb = drone
    return run_sentinel_patch(ms, rgb, tmp_path / "out", flight_date=FLIGHT, ms_band_map=scene_files.MS_BAND_MAP,
                              config=_config(folder, **overrides))


def _report(result):
    return json.loads(result.report_path.read_text(encoding="utf-8"))


def test_pipeline_offline(tmp_path, scene, drone):
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, folder)

    assert result.status == "verified" and result.reason_code is None
    assert result.shift_m == pytest.approx((-ERROR_M[0], -ERROR_M[1]), abs=TOLERANCE_M)

    out = tmp_path / "out" / "sentinel"
    for name in ("footprint.geojson", "candidates.json", "sentinel_report.json", "bands/B3.tif", "bands/B4.tif",
                 "bands/B5.tif", "bands/B8.tif", "bands/SCL.tif"):
        assert (out / name).is_file(), name
    assert set(result.product_paths) >= {"analytic_10m", "analytic_20m", "scl", "coverage", "crop_health_bands"}

    report = _report(result)
    assert report["schema_version"] == "sentinel_patch/1.0" and report["status"] == "verified"
    assert report["scene"]["date_gap_days"] == 2 and report["scene"]["date_tier"] == "same_period"
    assert report["scene"]["radiometry"]["B08"]["offset"] == -0.1
    assert [gate["passed"] for gate in report["registration"]["gates"]] and all(g["passed"] for g in report["registration"]["gates"])
    assert report["registration"]["holdout"]["corr_after"] > report["registration"]["holdout"]["corr_before"]
    assert report["products"]["resampled"] is False

    with rasterio.open(result.product_paths["analytic_10m"]) as product:
        tags = product.tags()
        assert tags["REGISTRATION_STATUS"] == "verified" and tags["RESAMPLED"] == "false"
        assert float(tags["SHIFT_X_M"]) == pytest.approx(result.shift_m[0], abs=1e-3)
        assert product.descriptions == ("B02", "B03", "B04", "B08")


def test_corrected_patch_lands_on_the_truth(tmp_path, scene, drone):
    """Independent check: the published pixels equal the scene sampled at their (corrected) positions."""
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, folder)

    with rasterio.open(result.product_paths["analytic_10m"]) as product:
        nir = product.read(4)
        x, y = synthetic.pixel_centre_xy(product.transform, nir.shape)
    truth = scene_files.S2_BAND_FUNCTIONS["B08"](scene(x, y))
    inside = np.isfinite(nir)
    # What the patch would look like if it were left where ESA's georeferencing put it.
    uncorrected = scene_files.S2_BAND_FUNCTIONS["B08"](scene(x + ERROR_M[0], y + ERROR_M[1]))
    corrected_error = np.abs(nir[inside] - truth[inside]).mean()
    assert corrected_error < 0.01  # residual = the estimator's sub-pixel error on blobs 20-60 m wide
    assert corrected_error < 0.1 * np.abs(uncorrected[inside] - truth[inside]).mean()


def test_no_resampling_analytic(tmp_path, scene, drone):
    """Published pixel values are exactly the source values; only the georeferencing changed."""
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, folder)
    dx, dy = result.shift_m

    with rasterio.open(folder / "B08.tif") as source, rasterio.open(result.product_paths["analytic_10m"]) as product:
        col0 = round((product.transform.c - (source.transform.c + dx)) / source.transform.a)
        row0 = round((product.transform.f - (source.transform.f + dy)) / source.transform.e)
        dn = source.read(1)[row0: row0 + product.height, col0: col0 + product.width]
        published = product.read(4)
        assert product.transform.a == source.transform.a
    expected = dn.astype(np.float32) * np.float32(1e-4) + np.float32(-0.1)
    inside = np.isfinite(published)
    assert inside.sum() > 1000
    assert np.array_equal(published[inside], expected[inside])


def test_coverage_fraction_adds_up_to_the_perimeter_area(tmp_path, scene, drone):
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, folder)
    area_m2 = _report(result)["footprint"]["area_m2"]
    with rasterio.open(result.product_paths["coverage"]) as coverage:
        fraction, touched, centre = coverage.read()
        pixel_area = abs(coverage.transform.a * coverage.transform.e)
    assert fraction.sum() * pixel_area == pytest.approx(area_m2, rel=0.01)
    assert fraction.min() >= 0.0 and fraction.max() <= 1.0
    assert centre.sum() <= touched.sum() and ((fraction > 0) <= (touched > 0)).all()


def test_low_texture_scene_is_not_shifted(tmp_path):
    flat = lambda x, y: 0.30 + 0.004 * np.sin(np.asarray(x) / 37.0) * np.cos(np.asarray(y) / 41.0)  # noqa: E731
    drone = scene_files.write_drone_rasters(tmp_path / "drone", flat)
    folder = scene_files.write_s2_scene(tmp_path / "s2", flat, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, folder)

    assert (result.status, result.reason_code, result.shift_m) == ("not_applied", "LOW_TEXTURE", (0.0, 0.0))
    with rasterio.open(result.product_paths["analytic_10m"]) as product, rasterio.open(folder / "B08.tif") as source:
        assert product.tags()["REGISTRATION_STATUS"] == "not_applied"
        assert (product.transform.c - source.transform.c) % source.transform.a == 0  # origin untouched


def test_tiny_aoi_stops_before_any_scene_is_fetched(tmp_path, scene):
    drone = scene_files.write_drone_rasters(tmp_path / "drone", scene, size_px=30, ms_collar_m=0, rgb_collar_m=0)
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, folder)
    assert (result.status, result.reason_code) == ("not_applied", "AOI_TOO_SMALL")
    assert result.product_paths == {} and not (tmp_path / "out" / "sentinel" / "candidates.json").exists()


def test_small_aoi_without_enough_overlap_is_rejected_by_the_gate(tmp_path, scene):
    """144 Sentinel pixels pass the size check, but the drone only covers them, not the buffer."""
    drone = scene_files.write_drone_rasters(tmp_path / "drone", scene, size_px=120, ms_collar_m=0, rgb_collar_m=0,
                                            origin=(500700.0, 3499300.0))
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, folder)
    assert (result.status, result.reason_code) == ("not_applied", "AOI_TOO_SMALL")
    assert "analytic_10m" in result.product_paths  # published unshifted, as configured


def test_a_small_cloud_on_the_field_is_excluded_from_matching(tmp_path, scene, drone):
    """A flagged cloud under the 10 % cut-off is accepted, and its pixels take no part in the correlation.

    (Cloud in the buffer around the field cannot matter for matching: the drone reference does not exist there.
    Only cloud over the drone-covered field does, which is why the cut-off is measured inside the AOI.)
    """
    def cloud(reflectance):
        other = {b: np.minimum(1.0, 1.6 * scene_files.S2_BAND_FUNCTIONS[b](synthetic.sample_scene(
            scene, scene_files.s2_grid_transform(), scene_files.S2_SHAPE_10M, (-60.0, 50.0))) + 0.25) for b in reflectance}
        for band in reflectance:
            reflectance[band][62:70, 62:70] = other[band][62:70, 62:70]  # bright, unrelated, 8 x 8 px on the field

    def scl(array):
        array[31:35, 31:35] = 9  # the same block at 20 m

    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M, edit_10m=cloud, edit_scl=scl)
    result = _run(tmp_path, drone, folder)
    report = _report(result)

    assert result.status == "verified"
    assert result.shift_m == pytest.approx((-ERROR_M[0], -ERROR_M[1]), abs=TOLERANCE_M)
    assert 0.04 < report["scene"]["aoi_masked_fraction"] < 0.07  # 8x8 block + 2 px dilation, about 5 % of the field
    overlap = next(g for g in report["registration"]["gates"] if g["name"] == "G2_overlap")["value"]
    assert overlap <= 3136 - 120  # 3136 valid reference pixels, minus the 144 masked ones


def test_scene_with_a_cloudy_aoi_is_rejected(tmp_path, scene, drone):
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M,
                                        edit_scl=lambda array: array.__setitem__((slice(25, 56), slice(None)), 9))
    result = _run(tmp_path, drone, folder)
    assert (result.status, result.reason_code) == ("no_scene", "NO_ACCEPTABLE_SCENE")
    candidates = json.loads((tmp_path / "out" / "sentinel" / "candidates.json").read_text())["candidates"]
    assert candidates[0]["reason_code"] == "AOI_TOO_CLOUDY" and candidates[0]["aoi_masked_fraction"] > 0.5
    assert result.product_paths == {}


def test_temporal_change_never_yields_a_wrong_accepted_shift(tmp_path, scene, drone):
    """Half the fields are 'harvested' (different content). Either the right shift or a refusal - never a wrong one."""
    def harvest(reflectance):
        other = {b: scene_files.S2_BAND_FUNCTIONS[b](synthetic.sample_scene(
            scene, scene_files.s2_grid_transform(), scene_files.S2_SHAPE_10M, (90.0, -70.0))) for b in reflectance}
        for band in reflectance:
            reflectance[band][:, 80:] = other[band][:, 80:] * 0.4

    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M, edit_10m=harvest)
    result = _run(tmp_path, drone, folder)
    if result.status == "verified":
        assert result.shift_m == pytest.approx((-ERROR_M[0], -ERROR_M[1]), abs=5.0)
    else:
        assert result.status == "not_applied" and result.shift_m == (0.0, 0.0)


def test_nearest_clear_date_wins(tmp_path, scene, drone):
    root = tmp_path / "scenes"
    scene_files.write_s2_scene(root / "far_clear", scene, content_offset_m=ERROR_M, item_id="FAR", acquired="2026-10-15T05:30:00Z")
    scene_files.write_s2_scene(root / "near_cloudy", scene, content_offset_m=ERROR_M, item_id="NEAR_CLOUDY",
                               acquired="2026-10-11T05:30:00Z",
                               edit_scl=lambda a: a.__setitem__((slice(None), slice(None)), 9))
    scene_files.write_s2_scene(root / "near_clear", scene, content_offset_m=ERROR_M, item_id="NEAR_CLEAR",
                               acquired="2026-10-12T05:30:00Z")
    result = _run(tmp_path, drone, root)

    assert result.scene_id == "NEAR_CLEAR"  # day +1 is cloudy, day +2 is the nearest clear one, day +5 never needed
    records = json.loads((tmp_path / "out" / "sentinel" / "candidates.json").read_text())["candidates"]
    assert [(r["item_id"], r["accepted"]) for r in records] == [("NEAR_CLOUDY", False), ("NEAR_CLEAR", True)]


def test_stale_date_is_flagged(tmp_path, scene, drone):
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M, acquired="2026-10-20T05:30:00Z")
    result = _run(tmp_path, drone, folder)
    assert result.status == "verified" and result.warnings == ("DATE_GAP_STALE",)
    assert _report(result)["scene"]["date_tier"] == "stale"


def test_no_scene_in_range_is_reported_not_raised(tmp_path, scene, drone):
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, acquired="2026-12-01T05:30:00Z")
    result = _run(tmp_path, drone, folder)
    assert (result.status, result.reason_code) == ("no_scene", "NO_CANDIDATES")
    assert _report(result)["scene"] is None


def test_on_reject_skip_publishes_no_rasters(tmp_path):
    flat = lambda x, y: 0.30 + 0.004 * np.sin(np.asarray(x) / 37.0) * np.cos(np.asarray(y) / 41.0)  # noqa: E731
    drone = scene_files.write_drone_rasters(tmp_path / "drone", flat)
    folder = scene_files.write_s2_scene(tmp_path / "s2", flat, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, folder, registration={"on_reject": "skip"})
    assert (result.status, result.reason_code) == ("not_applied", "LOW_TEXTURE")
    assert set(result.product_paths) == {"preview"}  # no rasters, but the picture that explains the refusal
    assert not list((tmp_path / "out" / "sentinel").glob("*.tif"))
