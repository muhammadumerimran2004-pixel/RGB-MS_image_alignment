"""S3 - windowed reads of Sentinel bands. No reprojection and no resampling: pixels stay exactly as ESA wrote them."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

from sentinel_patch.errors import ProviderUnavailableError, SceneCoverageError

logger = logging.getLogger(__name__)
T = TypeVar("T")

READ_ATTEMPTS = 3
READ_RETRY_DELAY_S = 2.0

Bounds = tuple[float, float, float, float]  # left, bottom, right, top


@dataclass(frozen=True)
class BandWindow:
    band: str
    data: np.ndarray  # raw DN, exactly as stored
    transform: Affine
    crs: CRS
    nodata: float | None


def _retrying(action: Callable[[], T], what: str) -> T:
    """Run a remote raster read, retrying transient failures (a truncated HTTP transfer is common on cloud files).

    A read that keeps failing becomes ``ProviderUnavailableError`` so the scene is recorded as unreadable and the
    search moves on, instead of an unexpected crash.
    """
    last: Exception | None = None
    for attempt in range(1, READ_ATTEMPTS + 1):
        try:
            return action()
        except (rasterio.errors.RasterioIOError, OSError) as exc:  # includes CPLE_* read errors raised by GDAL
            last = exc
            logger.warning("Reading %s failed (attempt %d of %d): %s", what, attempt, READ_ATTEMPTS, exc)
            if attempt < READ_ATTEMPTS:
                time.sleep(READ_RETRY_DELAY_S * attempt)
    raise ProviderUnavailableError(f"Could not read {what} after {READ_ATTEMPTS} attempts: {last}") from last


def plan_window_10m(href_10m: str, aoi_bounds: Bounds, buffer_m: float) -> tuple[Window, Bounds]:
    """AOI bounds plus buffer, snapped outward to whole 20 m cells so 10 m and 20 m windows nest exactly.

    The AOI itself must lie inside the raster; only the buffer may be clipped at the raster edge.
    Returns the window and its map bounds.
    """
    return _retrying(lambda: _plan_window_10m(href_10m, aoi_bounds, buffer_m), "the scene header")


def _plan_window_10m(href_10m: str, aoi_bounds: Bounds, buffer_m: float) -> tuple[Window, Bounds]:
    left, bottom, right, top = aoi_bounds
    with rasterio.open(href_10m) as src:
        raster = Window(0, 0, src.width, src.height)
        aoi = from_bounds(left, bottom, right, top, src.transform)
        if (aoi.col_off < 0 or aoi.row_off < 0 or aoi.col_off + aoi.width > src.width
                or aoi.row_off + aoi.height > src.height):
            raise SceneCoverageError("The area of interest is not fully inside the scene raster.")
        wide = from_bounds(left - buffer_m, bottom - buffer_m, right + buffer_m, top + buffer_m, src.transform)
        col0, row0 = math.floor(wide.col_off), math.floor(wide.row_off)
        col1, row1 = math.ceil(wide.col_off + wide.width), math.ceil(wide.row_off + wide.height)
        col0, row0 = max(0, col0 - col0 % 2), max(0, row0 - row0 % 2)
        col1, row1 = min(src.width, col1 + col1 % 2), min(src.height, row1 + row1 % 2)
        window = Window(col0, row0, col1 - col0, row1 - row0).intersection(raster)
        return window, window_bounds(window, src.transform)


def read_band_window(href: str, bounds: Bounds, band: str) -> BandWindow:
    """Read the pixels covering ``bounds`` (which must be aligned to this raster's grid, as planned above)."""
    return _retrying(lambda: _read_band_window(href, bounds, band), f"band {band}")


def _read_band_window(href: str, bounds: Bounds, band: str) -> BandWindow:
    with rasterio.open(href) as src:
        window = from_bounds(*bounds, transform=src.transform)
        rounded = Window(round(window.col_off), round(window.row_off), round(window.width), round(window.height))
        for exact, snapped in ((window.col_off, rounded.col_off), (window.row_off, rounded.row_off),
                               (window.width, rounded.width), (window.height, rounded.height)):
            if abs(exact - snapped) > 1e-3:
                raise SceneCoverageError(f"Band {band} does not share the planned window grid (off by {exact - snapped:.3f} px).")
        data = src.read(1, window=rounded)
        return BandWindow(band, data, src.window_transform(rounded), CRS.from_user_input(src.crs), src.nodata)
