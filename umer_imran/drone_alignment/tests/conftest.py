import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path


@pytest.fixture
def synthetic_geo_tiff_pair(tmp_path: Path):
    """
    Creates a pair of synthetic GeoTIFF rasters with known overlap for testing.

    RGB: 512x512, 3 bands, GSD=0.03m (3cm), CRS=EPSG:32643
    MS:  256x256, 4 bands, GSD=0.06m (6cm), CRS=EPSG:32643
    Both cover overlapping spatial extents.
    """
    crs = "EPSG:32643"
    origin_x = 500000.0
    origin_y = 3000000.0

    # 1. Generate Synthetic RGB Image (512x512, 3 bands, float32)
    rgb_w, rgb_h = 512, 512
    rgb_gsd = 0.03
    rgb_transform = from_origin(origin_x, origin_y, rgb_gsd, rgb_gsd)
    
    # Create checkerboard pattern + random features
    xx, yy = np.meshgrid(np.arange(rgb_w), np.arange(rgb_h))
    pattern = ((xx // 32) + (yy // 32)) % 2
    rgb_red = (pattern * 150 + 50).astype(np.float32)
    rgb_green = (pattern * 100 + 30).astype(np.float32)
    rgb_blue = (pattern * 80 + 20).astype(np.float32)
    
    rgb_data = np.stack([rgb_red, rgb_green, rgb_blue], axis=0)

    rgb_path = tmp_path / "synthetic_rgb.tif"
    with rasterio.open(
        rgb_path,
        "w",
        driver="GTiff",
        height=rgb_h,
        width=rgb_w,
        count=3,
        dtype="float32",
        crs=crs,
        transform=rgb_transform,
    ) as dst:
        dst.write(rgb_data)

    # 2. Generate Synthetic MS Image (256x256, 4 bands, GSD=0.06m)
    # Slightly shifted origin to simulate camera offset
    ms_w, ms_h = 256, 256
    ms_gsd = 0.06
    ms_origin_x = origin_x + 0.15  # 15 cm shift
    ms_origin_y = origin_y - 0.10  # -10 cm shift
    ms_transform = from_origin(ms_origin_x, ms_origin_y, ms_gsd, ms_gsd)

    ms_xx, ms_yy = np.meshgrid(np.arange(ms_w), np.arange(ms_h))
    ms_pattern = ((ms_xx // 16) + (ms_yy // 16)) % 2
    ms_red = (ms_pattern * 140 + 55).astype(np.float32)
    ms_green = (ms_pattern * 95 + 32).astype(np.float32)
    ms_re = (ms_pattern * 110 + 40).astype(np.float32)
    ms_nir = (ms_pattern * 200 + 100).astype(np.float32)

    ms_data = np.stack([ms_red, ms_green, ms_re, ms_nir], axis=0)

    ms_path = tmp_path / "synthetic_ms.tif"
    with rasterio.open(
        ms_path,
        "w",
        driver="GTiff",
        height=ms_h,
        width=ms_w,
        count=4,
        dtype="float32",
        crs=crs,
        transform=ms_transform,
    ) as dst:
        dst.write(ms_data)

    return {
        "rgb_path": rgb_path,
        "ms_path": ms_path,
        "rgb_gsd": rgb_gsd,
        "ms_gsd": ms_gsd,
        "crs": crs,
    }
