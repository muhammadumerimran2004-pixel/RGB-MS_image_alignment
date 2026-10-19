"""Pipeline behaviour added with the provider chain: failover reporting and the 'partial' scene."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest
import rasterio

from sentinel_patch import run_sentinel_patch
from sentinel_patch.config.schema import parse_config
from sentinel_patch.discovery.chain import ProviderChain
from sentinel_patch.discovery.provider import ProviderUnavailableError
from sentinel_patch.discovery.providers.local_folder import LocalFolderProvider

from . import scene_files
from .test_selection_and_chain import StubProvider

FLIGHT = date(2026, 10, 10)
ERROR_M = (15.0, -12.0)


@pytest.fixture(scope="module")
def scene():
    return scene_files.full_scene()


@pytest.fixture
def drone(tmp_path, scene):
    return scene_files.write_drone_rasters(tmp_path / "drone", scene)


def _run(tmp_path, drone, provider):
    ms, rgb = drone
    config = parse_config({"registration": {"arosics_mode": "off"}})
    return run_sentinel_patch(ms, rgb, tmp_path / "out", flight_date=FLIGHT, ms_band_map=scene_files.MS_BAND_MAP,
                              config=config, provider=provider)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_provider_down_is_a_reported_no_scene_not_a_crash(tmp_path, drone):
    chain = ProviderChain([StubProvider("earth_search", error=ProviderUnavailableError("HTTP 503")),
                           StubProvider("planetary_computer", error=ProviderUnavailableError("timeout"))])
    result = _run(tmp_path, drone, chain)

    assert (result.status, result.reason_code) == ("no_scene", "PROVIDERS_UNAVAILABLE")
    failures = _json(tmp_path / "out" / "sentinel" / "candidates.json")["provider_failures"]
    assert [f["provider"] for f in failures] == ["earth_search", "planetary_computer"]


def test_scenes_flow_through_the_chain_exactly_as_through_a_bare_provider(tmp_path, scene, drone):
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, ProviderChain([LocalFolderProvider(folder)]))
    assert result.status == "verified"
    assert result.shift_m == pytest.approx((-ERROR_M[0], -ERROR_M[1]), abs=1.5)
    assert _json(tmp_path / "out" / "sentinel" / "candidates.json")["provider_failures"] == []


def test_partial_scene_is_published_flagged_with_masked_pixels_blank(tmp_path, scene, drone):
    """A cloud over ~18 % of the field is above the 10 % cut-off but below the 30 % fallback limit."""
    folder = scene_files.write_s2_scene(
        tmp_path / "s2", scene, content_offset_m=ERROR_M,
        edit_scl=lambda array: array.__setitem__((slice(30, 40), slice(30, 40)), 9))
    result = _run(tmp_path, drone, ProviderChain([LocalFolderProvider(folder)]))
    report = _json(result.report_path)

    assert result.status == "verified" and "SCENE_PARTIAL" in result.warnings
    assert report["scene"]["scene_quality"] == "partial" and 0.10 < report["scene"]["aoi_masked_fraction"] < 0.30
    candidates = _json(tmp_path / "out" / "sentinel" / "candidates.json")["candidates"]
    assert candidates[0]["reason_code"] == "PARTIAL_ACCEPTED" and candidates[0]["accepted"]

    with rasterio.open(result.product_paths["analytic_10m"]) as src, \
            rasterio.open(result.product_paths["coverage"]) as cov:
        data, inside = src.read(1), cov.read(1) > 0
    blank = np.isnan(data[inside]).mean()
    assert blank == pytest.approx(report["scene"]["aoi_masked_fraction"], abs=0.03)
    assert not np.isnan(data[inside]).all()


def test_clean_scene_has_no_blank_pixels_and_no_partial_flag(tmp_path, scene, drone):
    folder = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M)
    result = _run(tmp_path, drone, ProviderChain([LocalFolderProvider(folder)]))
    assert "SCENE_PARTIAL" not in result.warnings and _json(result.report_path)["scene"]["scene_quality"] == "clean"
    with rasterio.open(result.product_paths["analytic_10m"]) as src, rasterio.open(result.product_paths["coverage"]) as cov:
        assert not np.isnan(src.read(1)[cov.read(1) > 0]).any()
