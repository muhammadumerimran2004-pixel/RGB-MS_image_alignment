"""S1: the perimeter is where BOTH rasters have data, not a bounding box."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from sentinel_patch.footprint import EmptyFootprintError, extract_stack_footprint, write_footprint_geojson

from . import scene_files


@pytest.fixture
def stack(tmp_path):
    return scene_files.write_drone_rasters(tmp_path / "drone", scene_files.full_scene())


def test_footprint_is_the_intersection_not_the_bounding_box(stack):
    footprint = extract_stack_footprint(*stack)
    # MS has a 20 m collar and RGB a 40 m collar on a 600 m square: the common area is 520 m x 520 m.
    assert footprint.area_m2 == pytest.approx(520 * 520, rel=0.01)
    assert footprint.area_m2 < 0.8 * 600 * 600
    assert footprint.stack_crs.to_epsg() == 32642


def test_wgs84_polygon_is_in_the_right_place(stack):
    lon, lat = extract_stack_footprint(*stack).polygon_wgs84.centroid.coords[0]
    assert 68.9 < lon < 69.4 and 31.5 < lat < 31.8  # UTM 42N, x=500800 / y=3499200


def test_small_holes_are_filled_but_large_ones_are_kept(stack):
    ms_path, rgb_path = stack
    with rasterio.open(ms_path, "r+") as ms:
        for window in (Window(250, 250, 10, 10), Window(400, 150, 80, 80)):  # 100 px and 6400 px at 1 m
            for band in range(1, ms.count + 1):
                ms.write(np.zeros((int(window.height), int(window.width)), np.float32), band, window=window)
    footprint = extract_stack_footprint(ms_path, rgb_path)
    assert footprint.area_m2 == pytest.approx(520 * 520 - 6400, rel=0.01)


def test_geojson_has_one_wgs84_feature_that_crop_health_accepts(stack, tmp_path):
    from crop_health_sentinel.io.geometry import normalize_crop_geometry

    footprint = extract_stack_footprint(*stack)
    path = write_footprint_geojson(footprint, tmp_path / "out" / "footprint.geojson")
    collection = json.loads(path.read_text(encoding="utf-8"))

    assert len(collection["features"]) == 1
    properties = collection["features"][0]["properties"]
    assert properties["stack_crs"] == "EPSG:32642" and properties["stack_crs_wkt"].startswith("POLYGON")
    assert normalize_crop_geometry(collection).equals_exact(footprint.polygon_wgs84, 1e-9)


def test_rasters_without_common_data_are_refused(stack):
    ms_path, rgb_path = stack
    with rasterio.open(ms_path, "r+") as ms:
        for band in range(1, ms.count + 1):
            ms.write(np.zeros((ms.height, ms.width), np.float32), band)
    with pytest.raises(EmptyFootprintError):
        extract_stack_footprint(ms_path, rgb_path)
