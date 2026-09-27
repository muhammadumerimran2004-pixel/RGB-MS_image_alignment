import cv2
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path


def _world_texture(size_px: int, seed: int = 7) -> np.ndarray:
    """Smooth, non-repeating ground texture in [0, 1]. Deliberately not periodic: a
    checkerboard has no unique alignment, so it can't test whether registration works."""
    noise = np.random.default_rng(seed).normal(size=(size_px, size_px)).astype(np.float32)
    field = cv2.GaussianBlur(noise, (0, 0), sigmaX=6)
    return (field - field.min()) / (field.max() - field.min())


@pytest.fixture
def synthetic_geo_tiff_pair(tmp_path: Path):
    """
    Creates a pair of synthetic GeoTIFF rasters with known overlap for testing.

    RGB: 512x512, 3 bands, GSD=0.03m (3cm), CRS=EPSG:32643
    MS:  256x256, 4 bands, GSD=0.06m (6cm), CRS=EPSG:32643
    Both image one shared ground texture. The MS georeference is off by
    ``true_shift_m`` (the MS pixel georeferenced at P shows the ground at
    P - true_shift_m), so registration has a real, unique answer to find.
    """
    crs = "EPSG:32643"
    origin_x = 500000.0
    origin_y = 3000000.0
    true_shift_m = (0.12, -0.09)

    # Ground texture sampled at 3 cm, padded 1 m on every side of the RGB footprint.
    world_gsd, pad_m = 0.03, 1.0
    world = _world_texture(int(round((512 * 0.03 + 2 * pad_m) / world_gsd)))

    def sample_world(xs_m: np.ndarray, ys_m: np.ndarray) -> np.ndarray:
        cols = ((xs_m - (origin_x - pad_m)) / world_gsd - 0.5).astype(np.float32)
        rows = (((origin_y + pad_m) - ys_m) / world_gsd - 0.5).astype(np.float32)
        return cv2.remap(world, cols, rows, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 1. Generate Synthetic RGB Image (512x512, 3 bands, float32)
    rgb_w, rgb_h = 512, 512
    rgb_gsd = 0.03
    rgb_transform = from_origin(origin_x, origin_y, rgb_gsd, rgb_gsd)

    xx, yy = np.meshgrid(np.arange(rgb_w), np.arange(rgb_h))
    texture = sample_world(origin_x + (xx + 0.5) * rgb_gsd, origin_y - (yy + 0.5) * rgb_gsd)
    rgb_red = (texture * 180 + 40).astype(np.float32)
    rgb_green = (texture * 150 + 30).astype(np.float32)
    rgb_blue = (texture * 120 + 20).astype(np.float32)

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
    ms_texture = sample_world(
        ms_origin_x + (ms_xx + 0.5) * ms_gsd - true_shift_m[0],
        ms_origin_y - (ms_yy + 0.5) * ms_gsd - true_shift_m[1],
    )
    ms_red = (ms_texture * 170 + 45).astype(np.float32)
    ms_green = (ms_texture * 140 + 32).astype(np.float32)
    ms_re = (ms_texture * 110 + 40).astype(np.float32)
    ms_nir = (ms_texture * 200 + 100).astype(np.float32)

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
        "true_shift_m": true_shift_m,
    }
