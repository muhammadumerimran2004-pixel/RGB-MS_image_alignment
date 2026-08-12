from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

from drone_alignment.config.schema import AlignmentConfig, ResolutionMode, RegistrationChannel
from drone_alignment.io.reader import RasterMetadata
from drone_alignment.io.validators import compute_bounding_box_intersection, AlignmentValidationError


@dataclass
class CoarseAlignmentResult:
    # Kept for compatibility: these now belong to the compact registration grid.
    rgb_red_array: np.ndarray
    ms_red_array: np.ndarray
    ms_all_bands: np.ndarray | None
    target_profile: dict
    target_gsd: float  # Native output GSD, not registration GSD.
    scale_factor: float
    rgb_valid_mask: np.ndarray
    ms_valid_mask: np.ndarray
    registration_bands_rgb: dict[str, np.ndarray]
    registration_bands_ms: dict[str, np.ndarray]
    alpha_band_index: int | None = None
    registration_transform: object | None = None
    registration_gsd: float | None = None
    output_profile: dict | None = None


def _source_valid_mask(path: Path, meta: RasterMetadata) -> np.ndarray:
    """Read the shared source footprint once; alpha is explicit because float alpha is not always propagated by GDAL."""
    with rasterio.open(path) as src:
        valid = src.dataset_mask() > 0
        if meta.alpha_band_index is not None:
            valid &= src.read(meta.alpha_band_index) > 0
        return valid


def _reproject_registration_band(
    path: Path, band_index: int, meta: RasterMetadata, source_mask: np.ndarray,
    shape: tuple[int, int], target_transform, target_crs,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        values = src.read(band_index)
        valid = source_mask & (src.read_masks(band_index) > 0) & np.isfinite(values)
        if src.nodata is not None:
            valid &= values != src.nodata
    safe = np.where(valid, values, 0).astype(np.float32)
    out = np.zeros(shape, dtype=np.float32)
    out_mask = np.zeros(shape, dtype=np.uint8)
    reproject(safe, out, src_transform=meta.transform, src_crs=meta.crs,
              dst_transform=target_transform, dst_crs=target_crs,
              resampling=Resampling.bilinear, src_nodata=None, dst_nodata=0.0)
    reproject(valid.astype(np.uint8), out_mask, src_transform=meta.transform, src_crs=meta.crs,
              dst_transform=target_transform, dst_crs=target_crs,
              resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    valid_out = out_mask > 0
    out[~valid_out] = 0.0
    return out, valid_out


def coarse_align(rgb_meta: RasterMetadata, ms_meta: RasterMetadata, config: AlignmentConfig) -> CoarseAlignmentResult:
    """Prepare only a compact mask-aware registration grid; native spectral data stays on disk until acceptance."""
    minx, miny, maxx, maxy = compute_bounding_box_intersection(rgb_meta.bounds, ms_meta.bounds)
    if config.coarse.target_gsd_m is not None:
        native_gsd = config.coarse.target_gsd_m
    elif config.resolution_mode == ResolutionMode.RGB_NATIVE:
        native_gsd = rgb_meta.gsd
    else:
        native_gsd = ms_meta.gsd
    margin = config.coarse.overlap_margin_px * native_gsd
    minx = max(minx - margin, min(rgb_meta.bounds[0], ms_meta.bounds[0]))
    miny = max(miny - margin, min(rgb_meta.bounds[1], ms_meta.bounds[1]))
    maxx = min(maxx + margin, max(rgb_meta.bounds[2], ms_meta.bounds[2]))
    maxy = min(maxy + margin, max(rgb_meta.bounds[3], ms_meta.bounds[3]))

    native_width = int(math.ceil((maxx - minx) / native_gsd))
    native_height = int(math.ceil((maxy - miny) / native_gsd))
    native_transform = from_bounds(minx, miny, maxx, maxy, native_width, native_height)
    max_side = max(native_width, native_height)
    reg_scale = min(1.0, config.coarse.registration_max_dimension / max_side)
    reg_width = max(1, int(round(native_width * reg_scale)))
    reg_height = max(1, int(round(native_height * reg_scale)))
    reg_transform = from_bounds(minx, miny, maxx, maxy, reg_width, reg_height)
    reg_gsd = max((maxx - minx) / reg_width, (maxy - miny) / reg_height)
    output_profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": config.warp.nodata_value,
        "width": native_width, "height": native_height, "count": ms_meta.band_count,
        "crs": rgb_meta.crs, "transform": native_transform,
    }

    rgb_source_mask = _source_valid_mask(rgb_meta.path, rgb_meta)
    ms_source_mask = _source_valid_mask(ms_meta.path, ms_meta)
    band_map = {
        "red": (config.rgb_red_band_index, config.ms_red_band_index),
        "green": (config.rgb_green_band_index, config.ms_green_band_index),
    }
    rgb_bands, ms_bands, rgb_masks, ms_masks = {}, {}, [], []
    for channel in dict.fromkeys(c.value for c in config.registration_channel_priority):
        rgb_idx, ms_idx = band_map[channel]
        rgb_arr, rgb_mask = _reproject_registration_band(
            rgb_meta.path, rgb_idx, rgb_meta, rgb_source_mask, (reg_height, reg_width), reg_transform, rgb_meta.crs
        )
        ms_arr, ms_mask = _reproject_registration_band(
            ms_meta.path, ms_idx, ms_meta, ms_source_mask, (reg_height, reg_width), reg_transform, rgb_meta.crs
        )
        rgb_bands[channel], ms_bands[channel] = rgb_arr, ms_arr
        rgb_masks.append(rgb_mask)
        ms_masks.append(ms_mask)
    rgb_valid = np.logical_and.reduce(rgb_masks)
    ms_valid = np.logical_and.reduce(ms_masks)
    common_fraction = float(np.mean(rgb_valid & ms_valid))
    if common_fraction < config.min_common_valid_fraction:
        raise AlignmentValidationError(
            f"Common valid-data fraction ({common_fraction:.2%}) is below configured minimum "
            f"({config.min_common_valid_fraction:.2%})."
        )
    return CoarseAlignmentResult(
        rgb_red_array=rgb_bands.get("red", next(iter(rgb_bands.values()))),
        ms_red_array=ms_bands.get("red", next(iter(ms_bands.values()))),
        ms_all_bands=None, target_profile=output_profile, target_gsd=native_gsd,
        scale_factor=reg_scale, rgb_valid_mask=rgb_valid, ms_valid_mask=ms_valid,
        registration_bands_rgb=rgb_bands, registration_bands_ms=ms_bands,
        alpha_band_index=ms_meta.alpha_band_index, registration_transform=reg_transform,
        registration_gsd=reg_gsd, output_profile=output_profile,
    )
