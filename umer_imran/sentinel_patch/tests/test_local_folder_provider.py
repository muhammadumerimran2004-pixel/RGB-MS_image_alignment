"""The offline provider: scenes on disk, in the layout crop_health_sentinel already uses."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from shapely.geometry import box

from sentinel_patch.config.schema import parse_config
from sentinel_patch.discovery.providers.local_folder import LocalFolderProvider
from sentinel_patch.errors import ConfigurationError, ProviderUnavailableError

from . import scene_files

AOI = box(69.0, 31.5, 69.2, 31.7)  # lon/lat around the fixture scenes (UTM 42N, x 500000-501600)
WIDE = (datetime(2026, 10, 1, tzinfo=timezone.utc), datetime(2026, 10, 31, tzinfo=timezone.utc))


@pytest.fixture(scope="module")
def scene():
    return scene_files.full_scene()


def test_reads_a_scene_folder(tmp_path, scene):
    folder = scene_files.write_s2_scene(tmp_path / "a", scene, item_id="S2_A", acquired="2026-10-12T05:30:00Z")
    (candidate,) = LocalFolderProvider(folder).search(AOI, *WIDE)

    assert (candidate.item_id, candidate.mgrs_tile, candidate.epsg, candidate.processing_baseline) == ("S2_A", "42RXV", 32642, "05.11")
    assert candidate.datetime_utc == datetime(2026, 10, 12, 5, 30, tzinfo=timezone.utc)
    assert {"B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12", "SCL"} <= set(candidate.asset_hrefs)


def test_each_subfolder_with_a_scene_file_is_one_scene(tmp_path, scene):
    for name, when in (("one", "2026-10-05"), ("two", "2026-10-12")):
        scene_files.write_s2_scene(tmp_path / name, scene, item_id=name, acquired=f"{when}T05:30:00Z")
    (tmp_path / "notes").mkdir()
    found = LocalFolderProvider(tmp_path).search(AOI, *WIDE)
    assert sorted(c.item_id for c in found) == ["one", "two"]


def test_search_filters_by_date_and_by_location(tmp_path, scene):
    scene_files.write_s2_scene(tmp_path / "a", scene, item_id="A", acquired="2026-10-12T05:30:00Z")
    provider = LocalFolderProvider(tmp_path)
    assert provider.search(AOI, datetime(2026, 10, 13, tzinfo=timezone.utc), datetime(2026, 10, 20, tzinfo=timezone.utc)) == []
    assert provider.search(box(10.0, 10.0, 10.1, 10.1), *WIDE) == []


def test_short_band_names_are_found(tmp_path, scene):
    """crop_health_sentinel names its files B3.tif, B4.tif ...; both spellings must work."""
    folder = scene_files.write_s2_scene(tmp_path / "a", scene)
    (folder / "B04.tif").rename(folder / "B4.tif")
    (candidate,) = LocalFolderProvider(folder).search(AOI, *WIDE)
    assert candidate.asset_hrefs["B04"].endswith("B4.tif")


def test_declared_scale_and_offset_are_carried(tmp_path, scene):
    folder = scene_files.write_s2_scene(tmp_path / "a", scene)
    meta = json.loads((folder / "scene.json").read_text())
    meta["scale_offset"] = {"B04": [0.0001, -0.1]}
    (folder / "scene.json").write_text(json.dumps(meta))
    (candidate,) = LocalFolderProvider(folder).search(AOI, *WIDE)
    assert candidate.scale_offset == {"B04": (0.0001, -0.1)}


def test_naive_timestamps_are_treated_as_utc(tmp_path, scene):
    folder = scene_files.write_s2_scene(tmp_path / "a", scene, acquired="2026-10-12T05:30:00")
    (candidate,) = LocalFolderProvider(folder).search(AOI, *WIDE)
    assert candidate.datetime_utc.tzinfo is not None and candidate.datetime_utc.hour == 5


def test_broken_scene_files_raise_the_chain_friendly_error(tmp_path, scene):
    folder = scene_files.write_s2_scene(tmp_path / "a", scene)
    (folder / "scene.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ProviderUnavailableError):
        LocalFolderProvider(folder).search(AOI, *WIDE)


def test_missing_band_is_reported_by_name(tmp_path, scene):
    folder = scene_files.write_s2_scene(tmp_path / "a", scene)
    (candidate,) = LocalFolderProvider(folder).search(AOI, *WIDE)
    with pytest.raises(ProviderUnavailableError, match="B09"):
        LocalFolderProvider(folder).resolve_href(candidate, "B09")


def test_missing_folder_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError):
        LocalFolderProvider(tmp_path / "nope")


def test_without_a_local_folder_the_default_source_is_the_online_chain():
    from sentinel_patch.discovery.chain import ProviderChain
    from sentinel_patch.pipeline import _default_provider
    chain = _default_provider(parse_config({}))
    assert isinstance(chain, ProviderChain) and [p.name for p in chain.providers] == ["earth_search", "planetary_computer"]
