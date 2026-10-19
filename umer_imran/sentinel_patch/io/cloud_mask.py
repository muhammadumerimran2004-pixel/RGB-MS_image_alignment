"""Reject mask (cloud / shadow / snow / nodata) on the 10 m window grid, and the AOI masked fraction.

SCL is delivered at 20 m and is known to miss cloud shadows, so cloud and
shadow classes are dilated before use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry.base import BaseGeometry

from sentinel_patch.config.schema import DiscoveryConfig

DILATED_SCL_CLASSES = (3, 8, 9, 10)  # cloud shadow, cloud medium/high probability, thin cirrus


def upsample_nearest_2x(array: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(array, 2, axis=0), 2, axis=1)


def build_reject_mask(scl_20m: np.ndarray, nodata_10m: np.ndarray, cfg: DiscoveryConfig) -> np.ndarray:
    """True where a 10 m pixel must not be used. ``nodata_10m`` is True where the 10 m bands hold no data."""
    scl = upsample_nearest_2x(scl_20m)
    if scl.shape != nodata_10m.shape:
        raise ValueError(f"SCL (x2) shape {scl.shape} does not match the 10 m window {nodata_10m.shape}.")
    reject = np.isin(scl, cfg.scl_reject_classes) | nodata_10m
    if cfg.cloud_mask_dilation_px > 0:
        cloudy = np.isin(scl, DILATED_SCL_CLASSES)
        reject |= ndimage.binary_dilation(cloudy, structure=np.ones((3, 3), bool), iterations=cfg.cloud_mask_dilation_px)
    return reject


@dataclass(frozen=True)
class AoiMaskStats:
    masked_fraction: float
    clear_pixels: int
    aoi_pixels: int


def aoi_pixel_mask(aoi: BaseGeometry, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    return rasterize([(aoi, 1)], out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint8").astype(bool)


def aoi_mask_stats(reject: np.ndarray, aoi_mask: np.ndarray) -> AoiMaskStats:
    aoi_pixels = int(aoi_mask.sum())
    if aoi_pixels == 0:
        return AoiMaskStats(1.0, 0, 0)
    masked = int((reject & aoi_mask).sum())
    return AoiMaskStats(masked / aoi_pixels, aoi_pixels - masked, aoi_pixels)
