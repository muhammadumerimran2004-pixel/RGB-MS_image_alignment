from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from drone_alignment.alignment.coarse import CoarseAlignmentResult
from drone_alignment.alignment.displacement_field import RegularizedMeshField
from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.alignment.warper import (
    evaluate_native_displacement_footprint, warp_ms_with_displacement_field_tiled,
)
from drone_alignment.config.schema import TransformType, WarpConfig
from drone_alignment.io.reader import read_metadata


def _identity_transform() -> TransformResult:
    return TransformResult(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), TransformType.AFFINE,
                           (0.0, 0.0), (0.0, 0.0), 0.0, (1.0, 1.0), 1.0, 1, 1)


def test_tiled_displacement_warper_pull_samples_residual(tmp_path: Path):
    source_path = tmp_path / "source.tif"
    output_path = tmp_path / "warped.tif"
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    values = np.tile(np.arange(64, dtype=np.float32), (64, 1))
    with rasterio.open(source_path, "w", driver="GTiff", width=64, height=64, count=1,
                       dtype="float32", crs="EPSG:32642", transform=transform, nodata=0.0) as dst:
        dst.write(values, 1)
    profile = {"driver": "GTiff", "dtype": "float32", "nodata": 0.0, "width": 64, "height": 64,
               "count": 1, "crs": "EPSG:32642", "transform": transform}
    coarse = CoarseAlignmentResult(np.empty((64, 64)), np.empty((64, 64)), None, profile, 1.0, 1.0,
                                   np.ones((32, 32), bool), np.ones((32, 32), bool), {}, {}, output_profile=profile)
    # A +2 forward residual means output x=20 samples baseline x=18.
    field = RegularizedMeshField(np.array([0.0, 31.0]), np.array([0.0, 31.0]), np.full((2, 2, 2), [1.0, 0.0]))
    _, footprint = warp_ms_with_displacement_field_tiled(
        coarse, _identity_transform(), field, output_path, WarpConfig(tile_size=256), read_metadata(source_path),
        rgb_meta=read_metadata(source_path), return_footprint=True,
    )
    with rasterio.open(output_path) as warped:
        data = warped.read(1)
        assert warped.transform == transform
        assert data[30, 40] == 38.0
        assert data[30, 0] == 0.0
    assert footprint.retained_source_valid_ratio > 0.90
    assert footprint.reference_overlap_ratio > 0.90


def test_tiled_warper_applies_residual_before_inverse_affine(tmp_path: Path):
    source_path = tmp_path / "source_affine.tif"
    output_path = tmp_path / "warped_affine.tif"
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    values = np.tile(np.arange(64, dtype=np.float32), (64, 1))
    with rasterio.open(source_path, "w", driver="GTiff", width=64, height=64, count=1,
                       dtype="float32", crs="EPSG:32642", transform=transform, nodata=0.0) as dst:
        dst.write(values, 1)
    profile = {"driver": "GTiff", "dtype": "float32", "nodata": 0.0, "width": 64, "height": 64,
               "count": 1, "crs": "EPSG:32642", "transform": transform}
    coarse = CoarseAlignmentResult(
        np.empty((64, 64)), np.empty((64, 64)), None, profile, 1.0, 1.0,
        np.ones((32, 32), bool), np.ones((32, 32), bool), {}, {}, output_profile=profile,
    )
    field = RegularizedMeshField(
        np.array([0.0, 31.0]), np.array([0.0, 31.0]), np.full((2, 2, 2), [2.0, 0.0]),
    )
    affine = TransformResult(
        np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), TransformType.AFFINE,
        (0.0, 0.0), (0.0, 0.0), 0.0, (2.0, 1.0), 1.0, 1, 1,
    )

    warp_ms_with_displacement_field_tiled(
        coarse, affine, field, output_path, WarpConfig(tile_size=256), read_metadata(source_path),
    )

    with rasterio.open(output_path) as warped:
        # Registration residual +2 becomes +4 native pixels.  Source is (40-4)/2=18.
        assert warped.read(1)[30, 40] == 18.0


def test_displacement_footprint_matches_identity_geometry(tmp_path: Path):
    rgb_path, ms_path = tmp_path / "rgb.tif", tmp_path / "ms.tif"
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    values = np.ones((32, 32), dtype=np.float32)
    for path in (rgb_path, ms_path):
        with rasterio.open(path, "w", driver="GTiff", width=32, height=32, count=1, dtype="float32",
                           crs="EPSG:32642", transform=transform, nodata=0.0) as dst:
            dst.write(values, 1)
    profile = {"driver": "GTiff", "dtype": "float32", "nodata": 0.0, "width": 32, "height": 32,
               "count": 1, "crs": "EPSG:32642", "transform": transform}
    coarse = CoarseAlignmentResult(np.empty((32, 32)), np.empty((32, 32)), None, profile, 1.0, 1.0,
                                   np.ones((16, 16), bool), np.ones((16, 16), bool), {}, {}, output_profile=profile)
    field = RegularizedMeshField(np.array([0.0, 15.0]), np.array([0.0, 15.0]), np.zeros((2, 2, 2)))
    footprint = evaluate_native_displacement_footprint(
        read_metadata(rgb_path), read_metadata(ms_path), coarse, _identity_transform(), field, 256,
    )
    assert footprint.retained_source_valid_ratio == 1.0
    assert footprint.reference_overlap_ratio == 1.0
