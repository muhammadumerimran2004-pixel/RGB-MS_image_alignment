from pathlib import Path
import cv2
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from affine import Affine
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling

from drone_alignment.config.schema import WarpConfig, InterpolationType
from drone_alignment.alignment.coarse import CoarseAlignmentResult
from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.alignment.displacement_field import ResidualField
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


def _native_residual_maps(
    window: Window,
    shape: tuple[int, int],
    profile: dict,
    registration_shape: tuple[int, int],
    field: ResidualField,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate registration-grid residuals in global native output pixels."""
    height, width = shape
    reg_height, reg_width = registration_shape
    sx, sy = profile["width"] / reg_width, profile["height"] / reg_height
    yy, xx = np.indices((height, width), dtype=np.float64)
    native_x, native_y = xx + window.col_off, yy + window.row_off
    reg_xy = np.column_stack([(native_x / sx).ravel(), (native_y / sy).ravel()])
    residual = field.evaluate(reg_xy).reshape(height, width, 2)
    return (residual[..., 0] * sx).astype(np.float32), (residual[..., 1] * sy).astype(np.float32)


def _sample_window_for_maps(map_x: np.ndarray, map_y: np.ndarray, width: int, height: int, padding: int = 2) -> Window:
    col0 = max(0, int(np.floor(np.nanmin(map_x))) - padding)
    row0 = max(0, int(np.floor(np.nanmin(map_y))) - padding)
    col1 = min(width, int(np.ceil(np.nanmax(map_x))) + padding + 1)
    row1 = min(height, int(np.ceil(np.nanmax(map_y))) + padding + 1)
    if col0 >= col1 or row0 >= row1:
        raise ValueError("Displacement field mapped a tile entirely outside the baseline output grid.")
    return Window(col0, row0, col1 - col0, row1 - row0)


def warp_ms_with_displacement_field_tiled(
    coarse_result: CoarseAlignmentResult,
    global_native_transform: TransformResult,
    field: ResidualField,
    output_path: Path,
    config: WarpConfig,
    ms_meta: RasterMetadata,
    rgb_meta: RasterMetadata | None = None,
    return_footprint: bool = False,
) -> Path | tuple[Path, FootprintMetrics]:
    """Write a candidate local warp with global coordinates and bounded memory.

    ``field`` is a registration-grid residual in the forward MS->RGB direction.
    The method first maps original MS data to the output grid through a
    ``WarpedVRT``.  Each output tile then pull-samples that baseline view using
    inverse global affine coordinates minus the local residual.  It never
    allocates a native-resolution field or source raster.
    """
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = (coarse_result.output_profile or coarse_result.target_profile).copy()
    profile.update(driver="GTiff", count=ms_meta.band_count, dtype=ms_meta.dtype, tiled=True,
                   blockxsize=min(512, config.tile_size), blockysize=min(512, config.tile_size),
                   compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
    inverse_global = ~_matrix_affine(global_native_transform.matrix)
    reflectance_resampling = Resampling.bilinear if config.reflectance_interpolation == InterpolationType.BILINEAR else Resampling.nearest
    registration_shape = coarse_result.rgb_valid_mask.shape
    rgb_count = baseline_ms_count = warped_ms_count = intersection = union = 0
    with rasterio.open(ms_meta.path) as src, rasterio.open(output, "w", **profile) as dst:
        rgb = rasterio.open(rgb_meta.path) if rgb_meta is not None else None
        try:
            with WarpedVRT(src, crs=profile["crs"], transform=profile["transform"], width=profile["width"],
                           height=profile["height"], resampling=reflectance_resampling, nodata=config.nodata_value) as baseline, \
                 WarpedVRT(src, crs=profile["crs"], transform=profile["transform"], width=profile["width"],
                           height=profile["height"], resampling=Resampling.nearest, nodata=config.nodata_value) as mask_vrt:
                dst.descriptions = src.descriptions
                for window in _windows(profile["width"], profile["height"], config.tile_size):
                    tile_shape = (int(window.height), int(window.width))
                    residual_x, residual_y = _native_residual_maps(window, tile_shape, profile, registration_shape, field)
                    yy, xx = np.indices(tile_shape, dtype=np.float32)
                    destination_x, destination_y = xx + window.col_off, yy + window.row_off
                    adjusted_x, adjusted_y = destination_x - residual_x, destination_y - residual_y
                    map_x = inverse_global.a * adjusted_x + inverse_global.b * adjusted_y + inverse_global.c
                    map_y = inverse_global.d * adjusted_x + inverse_global.e * adjusted_y + inverse_global.f
                    source_window = _sample_window_for_maps(map_x, map_y, profile["width"], profile["height"])
                    local_x = (map_x - source_window.col_off).astype(np.float32)
                    local_y = (map_y - source_window.row_off).astype(np.float32)
                    for band in range(1, ms_meta.band_count + 1):
                        interpolation = cv2.INTER_NEAREST if band == ms_meta.alpha_band_index else (
                            cv2.INTER_LINEAR if config.reflectance_interpolation == InterpolationType.BILINEAR else cv2.INTER_NEAREST
                        )
                        source_tile = baseline.read(band, window=source_window, out_dtype=ms_meta.dtype)
                        warped = cv2.remap(source_tile, local_x, local_y, interpolation,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=config.nodata_value)
                        dst.write(warped.astype(np.dtype(ms_meta.dtype), copy=False), band, window=window)
                    mask_tile = mask_vrt.dataset_mask(window=source_window)
                    warped_mask = cv2.remap(mask_tile, local_x, local_y, cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    dst.write_mask(warped_mask, window=window)
                    if rgb is not None:
                        rgb_mask = _reproject_valid_mask(
                            rgb, rgb_meta, tile_shape,
                            profile["transform"] * Affine.translation(window.col_off, window.row_off), profile["crs"],
                        )
                        baseline_mask = mask_vrt.dataset_mask(window=window) > 0
                        local_mask = warped_mask > 0
                        rgb_count += int(rgb_mask.sum())
                        baseline_ms_count += int(baseline_mask.sum())
                        warped_ms_count += int(local_mask.sum())
                        intersection += int((rgb_mask & local_mask).sum())
                        union += int((rgb_mask | local_mask).sum())
        finally:
            if rgb is not None:
                rgb.close()
    if return_footprint:
        if rgb_meta is None:
            raise ValueError("rgb_meta is required when return_footprint=True.")
        if not rgb_count or not baseline_ms_count or not warped_ms_count:
            raise ValueError("Native local-mesh export produced an empty footprint.")
        return output, FootprintMetrics(
            retained_source_valid_ratio=warped_ms_count / baseline_ms_count,
            reference_overlap_ratio=intersection / rgb_count,
            target_overlap_ratio=intersection / warped_ms_count,
            overlap_coefficient=intersection / min(rgb_count, warped_ms_count),
            intersection_over_union=intersection / union,
        )
    return output


def evaluate_native_displacement_footprint(
    rgb_meta: RasterMetadata,
    ms_meta: RasterMetadata,
    coarse: CoarseAlignmentResult,
    global_native_transform: TransformResult,
    field: ResidualField,
    tile_size: int,
) -> FootprintMetrics:
    """Measure the local-warp footprint without creating a full-size mask.

    This evaluates exactly the same destination-to-baseline map as the tiled
    displacement exporter.  It is intentionally field-aware rather than using
    affine footprint logic as a proxy for a non-rigid warp.
    """
    profile = coarse.output_profile or coarse.target_profile
    width, height = profile["width"], profile["height"]
    inverse_global = ~_matrix_affine(global_native_transform.matrix)
    registration_shape = coarse.rgb_valid_mask.shape
    rgb_count = baseline_ms_count = warped_ms_count = intersection = union = 0
    with rasterio.open(rgb_meta.path) as rgb, rasterio.open(ms_meta.path) as ms:
        with WarpedVRT(ms, crs=profile["crs"], transform=profile["transform"], width=width, height=height,
                       resampling=Resampling.nearest, nodata=0.0) as baseline:
            for window in _windows(width, height, tile_size):
                shape = (int(window.height), int(window.width))
                dst_transform = profile["transform"] * Affine.translation(window.col_off, window.row_off)
                rgb_mask = _reproject_valid_mask(rgb, rgb_meta, shape, dst_transform, profile["crs"])
                baseline_mask = baseline.dataset_mask(window=window) > 0
                residual_x, residual_y = _native_residual_maps(window, shape, profile, registration_shape, field)
                yy, xx = np.indices(shape, dtype=np.float32)
                destination_x, destination_y = xx + window.col_off, yy + window.row_off
                adjusted_x, adjusted_y = destination_x - residual_x, destination_y - residual_y
                map_x = inverse_global.a * adjusted_x + inverse_global.b * adjusted_y + inverse_global.c
                map_y = inverse_global.d * adjusted_x + inverse_global.e * adjusted_y + inverse_global.f
                source_window = _sample_window_for_maps(map_x, map_y, width, height)
                source_mask = baseline.dataset_mask(window=source_window)
                warped_mask = cv2.remap(
                    source_mask, (map_x - source_window.col_off).astype(np.float32),
                    (map_y - source_window.row_off).astype(np.float32), cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ) > 0
                rgb_count += int(rgb_mask.sum())
                baseline_ms_count += int(baseline_mask.sum())
                warped_ms_count += int(warped_mask.sum())
                intersection += int((rgb_mask & warped_mask).sum())
                union += int((rgb_mask | warped_mask).sum())
    if not rgb_count or not baseline_ms_count or not warped_ms_count:
        raise ValueError("Native local-mesh footprint confirmation produced an empty footprint.")
    return FootprintMetrics(
        retained_source_valid_ratio=warped_ms_count / baseline_ms_count,
        reference_overlap_ratio=intersection / rgb_count,
        target_overlap_ratio=intersection / warped_ms_count,
        overlap_coefficient=intersection / min(rgb_count, warped_ms_count),
        intersection_over_union=intersection / union,
    )
