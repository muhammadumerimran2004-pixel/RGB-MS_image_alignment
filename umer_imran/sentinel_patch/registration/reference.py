"""S4 - degrade one drone band to what Sentinel-2 would see, on Sentinel's own pixel grid.

Pipeline: decimated read of the drone band -> intermediate grid (about 1 m, Sentinel CRS)
-> Gaussian PSF blur (normalised, so no-data edges do not darken the data) -> block average
to the 10 m grid. Memory stays bounded because the full-resolution drone raster is never read.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from scipy import ndimage

from sentinel_patch.config.schema import ReferenceConfig
from sentinel_patch.errors import ConfigurationError, ResourceLimitError

MAX_INTERMEDIATE_PIXELS = 25_000_000
PSF_TRUNCATE_SIGMAS = 3.0


def _block_mean(array: np.ndarray, rows: int, cols: int, factor: int) -> np.ndarray:
    return array.reshape(rows, factor, cols, factor).mean(axis=(1, 3))


def _drop_outlier_cells(data: np.ndarray, weight: np.ndarray, factor: float) -> int:
    """Zero the weight of cells that are not physically plausible; returns how many were dropped.

    A cell is an outlier when it lies more than ``factor`` robust standard deviations from the median of the
    covered cells. The test is deliberately loose (50 sigma): it removes overflow garbage, never real contrast.
    """
    covered = weight > 0.5
    if covered.sum() < 10:
        return 0
    values = data[covered]
    median = np.median(values)
    sigma = 1.4826 * np.median(np.abs(values - median))
    if not sigma > 0:  # a constant band has nothing to compare against
        return 0
    outlier = (weight > 0) & (np.abs(data - median) > factor * sigma)
    weight[outlier] = 0.0
    return int(outlier.sum())


def build_reference_band(
    aligned_ms_path: Path,
    band_index: int,
    grid_transform: Affine,
    grid_shape: tuple[int, int],
    grid_crs: CRS,
    cfg: ReferenceConfig,
    grid_gsd_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values, valid_fraction)`` on the Sentinel grid.

    ``values`` is NaN where fewer than ``min_pixel_valid_fraction`` of the 10 m cell is covered by valid
    drone pixels; ``valid_fraction`` is the covered share of each cell (0..1).
    """
    rows, cols = grid_shape
    factor = round(grid_gsd_m / cfg.intermediate_gsd_m)
    if factor < 1 or abs(factor * cfg.intermediate_gsd_m - grid_gsd_m) > 1e-6:
        raise ConfigurationError(f"intermediate_gsd_m must divide the {grid_gsd_m} m grid evenly.")
    sigma_px = cfg.psf_sigma_m / cfg.intermediate_gsd_m
    pad = math.ceil(PSF_TRUNCATE_SIGMAS * sigma_px)
    padded_shape = (rows * factor + 2 * pad, cols * factor + 2 * pad)
    if padded_shape[0] * padded_shape[1] > MAX_INTERMEDIATE_PIXELS:
        raise ResourceLimitError(
            f"Reference window needs {padded_shape[0] * padded_shape[1]:,} intermediate pixels "
            f"(limit {MAX_INTERMEDIATE_PIXELS:,}); use a smaller registration buffer."
        )
    inter_transform = grid_transform * Affine.scale(1.0 / factor) * Affine.translation(-pad, -pad)

    left, top = inter_transform.c, inter_transform.f
    right = left + inter_transform.a * padded_shape[1]
    bottom = top + inter_transform.e * padded_shape[0]

    empty = (np.full(grid_shape, np.nan, np.float32), np.zeros(grid_shape, np.float32))
    with rasterio.open(aligned_ms_path) as src:
        src_bounds = transform_bounds(grid_crs, src.crs, left, bottom, right, top, densify_pts=21)
        wanted = from_bounds(*src_bounds, transform=src.transform)
        col0, row0 = max(0, math.floor(wanted.col_off)), max(0, math.floor(wanted.row_off))
        col1 = min(src.width, math.ceil(wanted.col_off + wanted.width))
        row1 = min(src.height, math.ceil(wanted.row_off + wanted.height))
        if col1 <= col0 or row1 <= row0:
            return empty
        window = Window(col0, row0, col1 - col0, row1 - row0)

        native_gsd = max(abs(src.transform.a), abs(src.transform.e))
        read_gsd = max(native_gsd, cfg.intermediate_gsd_m)  # never upsample while reading
        out_h = max(1, round(window.height * abs(src.transform.e) / read_gsd))
        out_w = max(1, round(window.width * abs(src.transform.a) / read_gsd))
        data = src.read(band_index, window=window, out_shape=(out_h, out_w), resampling=Resampling.average)
        coverage = src.read_masks(band_index, window=window, out_shape=(out_h, out_w), resampling=Resampling.average)
        read_transform = src.window_transform(window) * Affine.scale(window.width / out_w, window.height / out_h)
        src_crs = src.crs

    weight = coverage.astype(np.float64) / 255.0
    data = data.astype(np.float64)
    weight[~np.isfinite(data)] = 0.0
    _drop_outlier_cells(data, weight, cfg.outlier_mad_factor)
    weighted = np.where(weight > 0, data, 0.0) * weight

    weighted_i = np.zeros(padded_shape, np.float32)
    weight_i = np.zeros(padded_shape, np.float32)
    for source, destination in ((weighted, weighted_i), (weight, weight_i)):
        reproject(
            source.astype(np.float32), destination, src_transform=read_transform, src_crs=src_crs,
            dst_transform=inter_transform, dst_crs=grid_crs, resampling=Resampling.bilinear,
        )

    blurred_weighted = ndimage.gaussian_filter(weighted_i, sigma_px, mode="constant", cval=0.0, truncate=PSF_TRUNCATE_SIGMAS)
    blurred_weight = ndimage.gaussian_filter(weight_i, sigma_px, mode="constant", cval=0.0, truncate=PSF_TRUNCATE_SIGMAS)
    inner = (slice(pad, pad + rows * factor), slice(pad, pad + cols * factor))

    numerator = _block_mean(blurred_weighted[inner], rows, cols, factor)
    denominator = _block_mean(blurred_weight[inner], rows, cols, factor)
    fraction = _block_mean(weight_i[inner], rows, cols, factor).astype(np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(denominator > 1e-6, numerator / denominator, np.nan).astype(np.float32)
    values[fraction < cfg.min_pixel_valid_fraction] = np.nan
    return values, fraction
