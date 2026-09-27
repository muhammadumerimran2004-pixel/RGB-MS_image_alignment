"""Raster staging for AROSICS COREG_LOCAL matching.

AROSICS' own documentation is explicit: *"Please do not perform any spatial
resampling of the input images before applying this algorithm."* This module
stages native-resolution, single-band, whitespace-safe matching rasters, and
(for the target) pre-applies the already-verified global correction so
COREG_LOCAL only has to find the small *residual* misregistration it is
designed for - not the original, potentially large, offset.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

from drone_alignment.io.reader import RasterMetadata

STAGE_NODATA = -9999.0


class ArosicsExecutionError(RuntimeError):
    """Raised for an operational AROSICS staging/setup failure, not weak image evidence."""


def _valid_block_size(dimension: int) -> int:
    """Return a GTiff tile block size (a multiple of 16) that fits within dimension."""
    return max(16, min(256, dimension) // 16 * 16)


def create_whitespace_safe_alias(source_path: Path, temp_path: Path, stem: str) -> Path:
    """Return a no-space alias without copying a multi-gigabyte raster.

    Always builds a GDAL VRT (a small XML reference, not a data copy - no
    cheaper than a hard link) rather than hard-linking the source file
    directly. A hard link shares the source file's own bytes and therefore
    its own metadata verbatim; some ODM outputs carry a band-count/band-name
    metadata-list length mismatch that GeoArray tolerates on read but crashes
    on with an IndexError when later saving the array back out (as AROSICS'
    DESHIFTER does). The VRT lets every per-band metadata list be cleared
    and rebuilt with exactly one entry per band, which a hard link cannot do.
    """
    vrt_path = temp_path / f"{stem}.vrt"
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        dataset = gdal.Translate(str(vrt_path), str(source_path), format="VRT")
        if dataset is None:
            raise RuntimeError("GDAL did not create the VRT alias.")
        dataset.SetMetadata({})
        for band_number in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(band_number)
            band.SetMetadata({})
            band.SetDescription("")
        dataset.FlushCache()
        dataset = None
        return vrt_path
    except Exception as exc:
        raise ArosicsExecutionError(
            f"Could not stage a whitespace-safe alias for {source_path}: {exc}"
        ) from exc


@dataclass(frozen=True)
class StagedBand:
    """One staged single-band matching raster."""

    path: Path
    transform: Affine
    # Maps a pixel coordinate of *this staged raster* back to the original
    # source raster's pixel coordinates. Identity when no resampling was
    # needed (the normal, translation-only case).
    staged_to_source_px: Affine
    resampled: bool


def _read_band_masked(path: Path, meta: RasterMetadata, band_index: int) -> tuple[np.ndarray, Affine]:
    """Read one band as float32 with every invalid pixel set to STAGE_NODATA."""
    with rasterio.open(path) as source:
        if not 1 <= band_index <= source.count:
            raise ArosicsExecutionError(
                f"Band {band_index} is outside the available range 1..{source.count} for {path}."
            )
        values = source.read(band_index).astype(np.float32)
        valid = source.read_masks(band_index) > 0
        if meta.alpha_band_index is not None:
            valid &= source.read(meta.alpha_band_index) > 0
        valid &= np.isfinite(values)
        if source.nodata is not None:
            valid &= values != source.nodata
        values[~valid] = STAGE_NODATA
        return values, source.transform


def _write_staged_band(values: np.ndarray, transform: Affine, crs, out_path: Path) -> None:
    height, width = values.shape
    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": STAGE_NODATA,
        "width": width, "height": height, "count": 1, "crs": crs, "transform": transform,
        "tiled": True, "compress": "deflate",
        "blockxsize": _valid_block_size(width), "blockysize": _valid_block_size(height),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as destination:
        destination.write(values, 1)


def stage_reference_band(rgb_meta: RasterMetadata, band_index: int, out_path: Path) -> StagedBand:
    """Stage the reference matching band at native resolution, no resampling."""
    try:
        values, transform = _read_band_masked(rgb_meta.path, rgb_meta, band_index)
        _write_staged_band(values, transform, rgb_meta.crs, out_path)
    except ArosicsExecutionError:
        raise
    except Exception as exc:
        raise ArosicsExecutionError(f"Could not stage reference band {band_index}: {exc}") from exc
    return StagedBand(out_path, transform, Affine.identity(), resampled=False)


@dataclass(frozen=True)
class GlobalMapCorrection:
    """The verified global result, expressed as a map-space correction.

    ``g_transform`` maps an *original* MS map position to the map position
    it should occupy once the verified global result is applied (both
    expressed in the shared registration-grid coordinate system's map CRS).
    """

    g_transform: Affine
    corrected_ms_transform: Affine
    is_translation_only: bool


def compute_global_map_correction(
    registration_transform: Affine, global_matrix: np.ndarray, ms_native_transform: Affine,
) -> GlobalMapCorrection:
    """Express the verified registration-grid affine as a map-space correction of the MS raster.

    ``global_matrix`` maps MS registration-grid pixels to RGB
    (reference/destination) registration-grid pixels - the same convention
    used throughout this pipeline (see ``TransformResult.matrix`` and
    ``warper._window_transform``). Both grids share ``registration_transform``
    (``coarse_align`` reprojects RGB and MS onto the identical grid), so for
    a registration-grid point ``q``, the MS content originally at map
    position ``R(q)`` belongs at map position ``R(M(q))``. The correction
    function taking one map position to the other is therefore
    ``G = R . M . R^-1``.
    """
    if global_matrix.shape != (2, 3):
        raise ArosicsExecutionError(
            "AROSICS local refinement currently supports affine (2x3) global transforms only."
        )
    m_affine = Affine(*global_matrix.reshape(-1).tolist())
    g_transform = registration_transform * m_affine * (~registration_transform)
    is_translation_only = (
        abs(g_transform.b) < 1e-9 and abs(g_transform.d) < 1e-9
        and abs(g_transform.a - 1.0) < 1e-6 and abs(g_transform.e - 1.0) < 1e-6
    )
    corrected_ms_transform = g_transform * ms_native_transform
    return GlobalMapCorrection(g_transform, corrected_ms_transform, is_translation_only)


def _corrected_footprint_grid(corrected_ms_transform: Affine, width: int, height: int, gsd: float) -> Affine:
    """A north-up grid at ``gsd`` covering the corrected raster's map footprint."""
    corners_px = [(0.0, 0.0), (width, 0.0), (0.0, height), (width, height)]
    corners_map = [corrected_ms_transform * corner for corner in corners_px]
    xs = [corner[0] for corner in corners_map]
    ys = [corner[1] for corner in corners_map]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return from_origin(minx, maxy, gsd, gsd)


def stage_precorrected_target_band(
    ms_meta: RasterMetadata, band_index: int, correction: GlobalMapCorrection, out_path: Path,
) -> StagedBand:
    """Stage the target matching band with the verified global correction pre-applied.

    Translation-only global results (by far the common case: ORB/SIFT/phase
    correlation/AROSICS-global all normally converge on a near-identity
    affine within the configured rotation/scale bounds) need no resampling:
    the band is written pixel-for-pixel with its geotransform's origin
    updated, exactly how AROSICS itself applies a global shift. A global
    result carrying real rotation/scale is resampled once onto a north-up
    grid at the MS GSD; ``staged_to_source_px`` records how to map back so
    the eventual GCP warp still reads the *original* MS pixels, not this
    resampled intermediate.
    """
    try:
        values, _native_transform = _read_band_masked(ms_meta.path, ms_meta, band_index)
    except ArosicsExecutionError:
        raise
    except Exception as exc:
        raise ArosicsExecutionError(f"Could not stage target band {band_index}: {exc}") from exc

    if correction.is_translation_only:
        _write_staged_band(values, correction.corrected_ms_transform, ms_meta.crs, out_path)
        return StagedBand(out_path, correction.corrected_ms_transform, Affine.identity(), resampled=False)

    height, width = values.shape
    target_transform = _corrected_footprint_grid(correction.corrected_ms_transform, width, height, ms_meta.gsd)
    # Compute output dimensions from the same corner bounds used to build target_transform.
    corners_map = [correction.corrected_ms_transform * corner for corner in
                   [(0.0, 0.0), (width, 0.0), (0.0, height), (width, height)]]
    xs = [c[0] for c in corners_map]
    ys = [c[1] for c in corners_map]
    out_width = max(1, math.ceil((max(xs) - min(xs)) / ms_meta.gsd))
    out_height = max(1, math.ceil((max(ys) - min(ys)) / ms_meta.gsd))

    resampled = np.full((out_height, out_width), STAGE_NODATA, dtype=np.float32)
    try:
        reproject(
            values, resampled, src_transform=correction.corrected_ms_transform, src_crs=ms_meta.crs,
            dst_transform=target_transform, dst_crs=ms_meta.crs, resampling=Resampling.bilinear,
            src_nodata=STAGE_NODATA, dst_nodata=STAGE_NODATA,
        )
        _write_staged_band(resampled, target_transform, ms_meta.crs, out_path)
    except Exception as exc:
        raise ArosicsExecutionError(f"Could not resample rotated global correction for staging: {exc}") from exc

    staged_to_source_px = (~correction.corrected_ms_transform) * target_transform
    return StagedBand(out_path, target_transform, staged_to_source_px, resampled=True)
