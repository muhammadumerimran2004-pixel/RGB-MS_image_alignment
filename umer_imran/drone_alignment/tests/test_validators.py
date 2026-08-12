import pytest
from pathlib import Path
import rasterio
from rasterio.transform import from_origin

from drone_alignment.config.schema import AlignmentConfig
from drone_alignment.io.validators import (
    validate_inputs,
    AlignmentValidationError,
    compute_bounding_box_intersection,
)
from drone_alignment.io.reader import read_metadata, read_band


def test_validate_inputs_success(synthetic_geo_tiff_pair):
    config = AlignmentConfig()
    rgb_meta, ms_meta = validate_inputs(
        synthetic_geo_tiff_pair["rgb_path"],
        synthetic_geo_tiff_pair["ms_path"],
        config,
    )
    assert rgb_meta.width == 512
    assert rgb_meta.height == 512
    assert rgb_meta.band_count == 3
    assert ms_meta.width == 256
    assert ms_meta.height == 256
    assert ms_meta.band_count == 4


def test_validate_inputs_missing_file(synthetic_geo_tiff_pair):
    config = AlignmentConfig()
    non_existent = synthetic_geo_tiff_pair["rgb_path"].parent / "does_not_exist.tif"
    with pytest.raises(FileNotFoundError):
        validate_inputs(non_existent, synthetic_geo_tiff_pair["ms_path"], config)


def test_validate_inputs_invalid_band_index(synthetic_geo_tiff_pair):
    config = AlignmentConfig(ms_red_band_index=99)
    with pytest.raises(AlignmentValidationError, match="out of bounds"):
        validate_inputs(
            synthetic_geo_tiff_pair["rgb_path"],
            synthetic_geo_tiff_pair["ms_path"],
            config,
        )


def test_validate_inputs_no_overlap(tmp_path):
    crs = "EPSG:32643"
    # Image 1 at (0, 0)
    img1_path = tmp_path / "img1.tif"
    with rasterio.open(
        img1_path, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32",
        crs=crs, transform=from_origin(0, 100, 1, 1)
    ) as dst:
        dst.write_mask(True)

    # Image 2 far away at (10000, 10000)
    img2_path = tmp_path / "img2.tif"
    with rasterio.open(
        img2_path, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32",
        crs=crs, transform=from_origin(10000, 10000, 1, 1)
    ) as dst:
        dst.write_mask(True)

    config = AlignmentConfig()
    with pytest.raises(AlignmentValidationError, match="No spatial overlap"):
        validate_inputs(img1_path, img2_path, config)


def test_compute_bounding_box_intersection():
    bounds1 = (0.0, 0.0, 10.0, 10.0)
    bounds2 = (5.0, 5.0, 15.0, 15.0)
    intersection = compute_bounding_box_intersection(bounds1, bounds2)
    assert intersection == (5.0, 5.0, 10.0, 10.0)
