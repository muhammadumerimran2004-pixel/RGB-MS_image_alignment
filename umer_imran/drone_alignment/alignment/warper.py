from pathlib import Path
import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling

from drone_alignment.config.schema import WarpConfig, InterpolationType
from drone_alignment.alignment.coarse import CoarseAlignmentResult
from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.io.reader import RasterMetadata
from drone_alignment.quality.metrics import FootprintMetrics


def _matrix_affine(matrix: np.ndarray) -> Affine:
    if matrix.shape != (2, 3):
        raise ValueError("Native streaming currently supports affine transforms only.")
    return Affine(*matrix.reshape(-1).tolist())


def _window_transform(base: Affine, inverse_residual: Affine, window: Window) -> Affine:
    # Destination pixel p samples the baseline world position of inverse_residual(p).
    return base * inverse_residual * Affine.translation(window.col_off, window.row_off)


def _windows(width: int, height: int, tile: int):
    for row in range(0, height, tile):
        for col in range(0, width, tile):
            yield Window(col, row, min(tile, width - col), min(tile, height - row))


def _reproject_valid_mask(src, meta: RasterMetadata, dst_shape, dst_transform, dst_crs) -> np.ndarray:
    """Reproject alpha when available; otherwise use a conservative finite/non-nodata proxy band."""
    index = meta.alpha_band_index or 1
    values = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        rasterio.band(src, index), values, src_transform=src.transform, src_crs=src.crs,
        dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.nearest,
        src_nodata=None, dst_nodata=0.0,
    )
    if meta.alpha_band_index is not None:
        return values > 0
    valid = np.isfinite(values)
    if src.nodata is not None:
        valid &= values != src.nodata
    return valid


def evaluate_native_footprint(
    rgb_meta: RasterMetadata, ms_meta: RasterMetadata, coarse: CoarseAlignmentResult,
    native_transform: TransformResult, tile_size: int,
) -> FootprintMetrics:
    """Confirm coverage on the final native grid without materializing a full mask array."""
    profile = coarse.output_profile or coarse.target_profile
    width, height = profile["width"], profile["height"]
    base = profile["transform"]
    inverse = ~_matrix_affine(native_transform.matrix)
    rgb_count = baseline_ms_count = warped_ms_count = intersection = union = 0
    with rasterio.open(rgb_meta.path) as rgb, rasterio.open(ms_meta.path) as ms:
        for window in _windows(width, height, tile_size):
            shape = (int(window.height), int(window.width))
            rgb_mask = _reproject_valid_mask(rgb, rgb_meta, shape, base * Affine.translation(window.col_off, window.row_off), profile["crs"])
            ms_base = _reproject_valid_mask(ms, ms_meta, shape, base * Affine.translation(window.col_off, window.row_off), profile["crs"])
            ms_warped = _reproject_valid_mask(ms, ms_meta, shape, _window_transform(base, inverse, window), profile["crs"])
            rgb_count += int(rgb_mask.sum())
            baseline_ms_count += int(ms_base.sum())
            warped_ms_count += int(ms_warped.sum())
            intersection += int((rgb_mask & ms_warped).sum())
            union += int((rgb_mask | ms_warped).sum())
    if not rgb_count or not baseline_ms_count or not warped_ms_count:
        raise ValueError("Native mask confirmation produced an empty footprint.")
    return FootprintMetrics(
        retained_source_valid_ratio=warped_ms_count / baseline_ms_count,
        reference_overlap_ratio=intersection / rgb_count,
        target_overlap_ratio=intersection / warped_ms_count,
        overlap_coefficient=intersection / min(rgb_count, warped_ms_count),
        intersection_over_union=intersection / union,
    )


def warp_ms_to_rgb_tiled(
    coarse_result: CoarseAlignmentResult, transform_result: TransformResult,
    output_path: Path, config: WarpConfig, ms_meta: RasterMetadata,
) -> Path:
    """Stream original source bands directly to native output tiles after registration is accepted."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = (coarse_result.output_profile or coarse_result.target_profile).copy()
    profile.update(driver="GTiff", count=ms_meta.band_count, dtype=ms_meta.dtype, tiled=True,
                   blockxsize=min(512, config.tile_size), blockysize=min(512, config.tile_size),
                   compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
    base = profile["transform"]
    inverse = ~_matrix_affine(transform_result.matrix)
    resampling = Resampling.bilinear if config.reflectance_interpolation == InterpolationType.BILINEAR else Resampling.nearest
    with rasterio.open(ms_meta.path) as src, rasterio.open(output, "w", **profile) as dst:
        dst.descriptions = src.descriptions
        for window in _windows(profile["width"], profile["height"], config.tile_size):
            shape = (int(window.height), int(window.width))
            target_transform = _window_transform(base, inverse, window)
            for band in range(1, ms_meta.band_count + 1):
                tile = np.zeros(shape, dtype=np.dtype(ms_meta.dtype))
                band_resampling = Resampling.nearest if band == ms_meta.alpha_band_index else resampling
                reproject(rasterio.band(src, band), tile, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=target_transform, dst_crs=profile["crs"], resampling=band_resampling,
                          src_nodata=src.nodata, dst_nodata=config.nodata_value)
                dst.write(tile, band, window=window)
            mask = _reproject_valid_mask(src, ms_meta, shape, target_transform, profile["crs"])
            dst.write_mask((mask.astype(np.uint8) * 255), window=window)
    return output
