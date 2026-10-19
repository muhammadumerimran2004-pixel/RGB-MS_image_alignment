"""Phase 2 end to end: cross-check modes, the quadrant gate, a drone map in another UTM zone, overlay and preview."""

from __future__ import annotations

import json
from datetime import date

import matplotlib.image as mpimg
import numpy as np
import pytest
import rasterio
from pyproj import CRS, Transformer

from sentinel_patch import run_sentinel_patch
from sentinel_patch import pipeline as pipeline_module
from sentinel_patch.config.schema import parse_config
from sentinel_patch.products import MAX_OVERLAY_PIXELS, OverlayTarget, build_overlay
from sentinel_patch.registration.arosics_estimator import ArosicsResult, is_available

from . import scene_files, synthetic

FLIGHT = date(2026, 10, 10)
ERROR_M = (15.0, -12.0)
TRUTH = (-ERROR_M[0], -ERROR_M[1])
TOLERANCE_M = 1.5
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def blob():
    return scene_files.full_scene()


def _run(tmp_path, drone, folder, **registration_and_more):
    registration = registration_and_more.pop("registration", {})
    config = parse_config({"local_folder": str(folder), "registration": registration, **registration_and_more})
    ms, rgb = drone
    return run_sentinel_patch(ms, rgb, tmp_path / "out", flight_date=FLIGHT, ms_band_map=scene_files.MS_BAND_MAP, config=config)


def _report(result):
    return json.loads(result.report_path.read_text(encoding="utf-8"))


def _gate(report, prefix):
    return next(g for g in report["registration"]["gates"] if g["name"].startswith(prefix))


@pytest.fixture
def blob_case(tmp_path, blob):
    drone = scene_files.write_drone_rasters(tmp_path / "drone", blob)
    folder = scene_files.write_s2_scene(tmp_path / "s2", blob, content_offset_m=ERROR_M)
    return tmp_path, drone, folder


# -- AROSICS cross-check modes ------------------------------------------------------------------------------------

def _fake_arosics(monkeypatch, result):
    monkeypatch.setattr(pipeline_module, "estimate_shift_arosics", lambda *args, **kwargs: result)


CONFIDENT_AND_WRONG = ArosicsResult("ok", shift_m=(20.0, -15.0), reliability=90.0, window_px=32)


def test_advisory_disagreement_warns_but_still_publishes(blob_case, monkeypatch):
    tmp_path, drone, folder = blob_case
    _fake_arosics(monkeypatch, CONFIDENT_AND_WRONG)
    result = _run(tmp_path, drone, folder)

    assert result.status == "verified" and result.shift_m == pytest.approx(TRUTH, abs=TOLERANCE_M)
    assert "AROSICS_DISAGREES" in result.warnings
    gate = _gate(_report(result), "G4")
    assert gate["passed"] is False and gate["advisory"] is True and gate["value"] > 3.0


def test_veto_mode_refuses_on_a_confident_disagreement(blob_case, monkeypatch):
    tmp_path, drone, folder = blob_case
    _fake_arosics(monkeypatch, CONFIDENT_AND_WRONG)
    result = _run(tmp_path, drone, folder, registration={"arosics_mode": "veto"})

    assert (result.status, result.reason_code, result.shift_m) == ("not_applied", "ESTIMATOR_DISAGREEMENT", (0.0, 0.0))
    with rasterio.open(result.product_paths["analytic_10m"]) as product:
        assert product.tags()["REGISTRATION_STATUS"] == "not_applied" and float(product.tags()["SHIFT_X_M"]) == 0.0


def test_an_inconclusive_arosics_leaves_the_decision_to_ncc(blob_case, monkeypatch):
    tmp_path, drone, folder = blob_case
    _fake_arosics(monkeypatch, ArosicsResult("inconclusive", "AROSICS_LOW_RELIABILITY", (3.0, 3.0), 30.0, 32))
    result = _run(tmp_path, drone, folder, registration={"arosics_mode": "veto"})  # even in veto mode
    assert result.status == "verified" and result.warnings == ("AROSICS_INCONCLUSIVE",)
    assert _gate(_report(result), "G4")["value"] is None


def test_arosics_off_is_recorded_as_skipped(blob_case):
    tmp_path, drone, folder = blob_case
    result = _run(tmp_path, drone, folder, registration={"arosics_mode": "off"})
    assert result.status == "verified" and result.warnings == ()
    assert _report(result)["registration"]["estimators"]["arosics"]["reason"] == "AROSICS_DISABLED"


@pytest.mark.skipif(not is_available(), reason="AROSICS is not installed")
def test_sharp_edged_field_scene_with_default_settings(tmp_path):
    """The realistic case with the real AROSICS: the published shift is NCC's, whatever AROSICS thinks."""
    field = synthetic.make_field_scene(scene_files.s2_grid_transform(), scene_files.S2_SHAPE_10M, seed=5, cells=90)
    drone = scene_files.write_drone_rasters(tmp_path / "drone", field)
    folder = scene_files.write_s2_scene(tmp_path / "s2", field, content_offset_m=ERROR_M, supersample=5)
    result = _run(tmp_path, drone, folder)

    assert result.status == "verified" and result.shift_m == pytest.approx(TRUTH, abs=TOLERANCE_M)
    estimators = _report(result)["registration"]["estimators"]
    assert estimators["arosics"]["status"] in {"ok", "inconclusive"} and estimators["arosics"]["window_px"] == 32


# -- G7: one translation, not a distortion ------------------------------------------------------------------------

def test_a_pure_translation_passes_the_quadrant_check(blob_case):
    result = _run(*blob_case[:1], blob_case[1], blob_case[2], registration={"arosics_mode": "off"})
    report = _report(result)
    quadrants = report["registration"]["estimators"]["quadrants"]
    assert quadrants["evaluated"] == 4 and _gate(report, "G7")["passed"] and _gate(report, "G7")["value"] < 0.3


def test_a_rotated_map_is_refused_even_though_a_shift_correlates_well(tmp_path, blob):
    drone = scene_files.write_drone_rasters(tmp_path / "drone", blob)
    folder = scene_files.write_s2_scene(tmp_path / "s2", blob, content_offset_m=ERROR_M, rotation_deg=3.0)
    result = _run(tmp_path, drone, folder, registration={"arosics_mode": "off"})

    assert (result.status, result.reason_code, result.shift_m) == ("not_applied", "NON_TRANSLATIONAL", (0.0, 0.0))
    report = _report(result)
    assert _gate(report, "G5a")["passed"] and _gate(report, "G7")["value"] > 0.5  # NCC alone would have accepted it


# -- a drone map in a different UTM zone ---------------------------------------------------------------------------

def test_drone_map_in_another_utm_zone(tmp_path, blob):
    drone = scene_files.write_drone_rasters(tmp_path / "drone", blob, crs=CRS.from_epsg(32643))
    folder = scene_files.write_s2_scene(tmp_path / "s2", blob, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, folder, registration={"arosics_mode": "off"})
    report = _report(result)

    assert result.status == "verified" and result.shift_m == pytest.approx(TRUTH, abs=TOLERANCE_M)
    assert report["footprint"]["stack_crs"] == "EPSG:32643"
    assert report["registration"]["shift_crs"] == "EPSG:32642"  # the axes dE / dN belong to the Sentinel tile's CRS
    with rasterio.open(result.product_paths["analytic_10m"]) as product, rasterio.open(result.product_paths["overlay"]) as overlay:
        assert product.crs.to_epsg() == 32642 and overlay.crs.to_epsg() == 32643
        assert product.tags()["RESAMPLED"] == "false" and overlay.tags()["RESAMPLED"] == "true"
        # The overlay must show the scene at the right place once positions are converted between the two zones.
        values = overlay.read(4)
        rows, cols = np.nonzero(np.isfinite(values))
        x, y = overlay.transform * (cols[::50] + 0.5, rows[::50] + 0.5)
        east, north = Transformer.from_crs(32643, 32642, always_xy=True).transform(x, y)
        truth = scene_files.S2_BAND_FUNCTIONS["B08"](blob(east, north))
        assert np.abs(values[rows[::50], cols[::50]] - truth).mean() < 0.03
    with rasterio.open(result.product_paths["coverage"]) as coverage:
        assert coverage.read(1).sum() * 100.0 == pytest.approx(report["footprint"]["area_m2"], rel=0.02)


# -- overlay ---------------------------------------------------------------------------------------------------------

def test_overlay_is_anchored_to_the_drone_pixel_lattice_and_masked_to_the_perimeter(blob_case):
    tmp_path, drone, folder = blob_case
    result = _run(tmp_path, drone, folder, registration={"arosics_mode": "off"}, products={"overlay_gsd_m": 2.5})
    with rasterio.open(drone[0]) as ms, rasterio.open(result.product_paths["overlay"]) as overlay:
        t = overlay.transform
        assert t.a == 2.5 and -t.e == 2.5
        assert ((t.c - ms.transform.c) / 2.5) == pytest.approx(round((t.c - ms.transform.c) / 2.5), abs=1e-9)
        assert ((ms.transform.f - t.f) / 2.5) == pytest.approx(round((ms.transform.f - t.f) / 2.5), abs=1e-9)
        values = overlay.read(4)
        assert overlay.tags()["PRODUCT_ROLE"] == "overlay" and overlay.descriptions == ("B02", "B03", "B04", "B08")
        inside = np.isfinite(values)
        assert inside.sum() * 2.5**2 == pytest.approx(_report(result)["footprint"]["area_m2"], rel=0.02)
        x, y = synthetic.pixel_centre_xy(t, values.shape)
        truth = scene_files.S2_BAND_FUNCTIONS["B08"](blob_case_scene(x, y))
        assert np.abs(values[inside] - truth[inside]).mean() < 0.03  # 10 m blocks of a smooth 20-60 m scene


def blob_case_scene(x, y):
    return scene_files.full_scene()(x, y)


def test_overlay_can_be_switched_off(blob_case):
    tmp_path, drone, folder = blob_case
    result = _run(tmp_path, drone, folder, registration={"arosics_mode": "off"}, products={"write_overlay": False})
    assert "overlay" not in result.product_paths and not list((tmp_path / "out" / "sentinel").glob("*_overlay.tif"))


def test_overlay_is_coarsened_instead_of_exhausting_memory(monkeypatch):
    stack = np.random.default_rng(0).uniform(0.1, 0.5, (1, 52, 52)).astype(np.float32)
    transform = synthetic.make_s2_transform((500500.0, 3499500.0))
    from shapely.geometry import box
    target = OverlayTarget(scene_files.CRS_UTM42, transform, box(500500.0, 3498980.0, 501020.0, 3499500.0))
    monkeypatch.setattr("sentinel_patch.products.MAX_OVERLAY_PIXELS", 10_000)
    _, _, gsd = build_overlay(stack, transform, scene_files.CRS_UTM42, target, gsd_m=1.0)
    assert gsd == 8.0  # 1 -> 2 -> 4 -> 8 m; at 4 m the grid would still need 17,424 pixels
    assert MAX_OVERLAY_PIXELS == 25_000_000


# -- preview ---------------------------------------------------------------------------------------------------------

def _assert_real_picture(path):
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    image = mpimg.imread(path)
    assert image.shape[0] >= 800 and image.shape[1] >= 1000
    assert image[..., :3].std() > 0.05  # not a blank canvas


def test_preview_is_written_for_a_verified_registration(blob_case):
    tmp_path, drone, folder = blob_case
    result = _run(tmp_path, drone, folder, registration={"arosics_mode": "off"})
    _assert_real_picture(result.product_paths["preview"])
    assert _report(result)["registration"]["preview"].endswith("sentinel_preview.png")


def test_preview_explains_a_refusal_too(tmp_path, blob):
    drone = scene_files.write_drone_rasters(tmp_path / "drone", blob)
    folder = scene_files.write_s2_scene(tmp_path / "s2", blob, content_offset_m=ERROR_M, rotation_deg=3.0)
    result = _run(tmp_path, drone, folder, registration={"arosics_mode": "off"})
    assert result.status == "not_applied"
    _assert_real_picture(result.product_paths["preview"])
