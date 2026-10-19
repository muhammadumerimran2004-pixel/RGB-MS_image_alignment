"""S7 - publish the analytic Sentinel patch.

The correction is a geotransform update only: pixel arrays are never resampled. The patch is then
cropped to the perimeter, and a coverage layer says how much of each 10 m pixel lies inside it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from shapely.geometry.base import BaseGeometry

from sentinel_patch.config.schema import ProductsConfig
from sentinel_patch.discovery.provider import SceneCandidate
from sentinel_patch.io.radiometry import RadiometryParams, dn_to_reflectance
from sentinel_patch.io.window_reader import BandWindow
from sentinel_patch.registration.shift import apply_shift

COVERAGE_SUPERSAMPLE = 10
MAX_OVERLAY_PIXELS = 25_000_000
CROP_HEALTH_DN_SCALE = 10000.0
# crop_health_sentinel file names for each band it reads, and the Sentinel band that feeds them
CROP_HEALTH_FILES = {"B03": "B3.tif", "B04": "B4.tif", "B05": "B5.tif", "B08": "B8.tif", "SCL": "SCL.tif"}


@dataclass(frozen=True)
class OverlayTarget:
    """Where the visualisation overlay is drawn: the drone map's own CRS and pixel lattice."""

    crs: CRS
    lattice: Affine  # native transform of the aligned MS raster; only its origin anchors the overlay grid
    polygon: BaseGeometry  # perimeter in ``crs``


@dataclass(frozen=True)
class ProductInputs:
    candidate: SceneCandidate
    crs: CRS
    aoi: BaseGeometry  # in the scene CRS
    bands_10m: Mapping[str, BandWindow]
    bands_20m: Mapping[str, BandWindow]
    scl: BandWindow  # 20 m
    params: Mapping[str, RadiometryParams]
    status: str  # "verified" | "not_applied"
    reason_code: str | None
    shift_m: tuple[float, float]
    overlay: OverlayTarget | None = None
    mask_reject: np.ndarray | None = None  # 10 m window grid; True pixels are written as NaN ('partial' scenes)


def _compression(name: str) -> dict:
    return {} if name == "none" else {"compress": name}


def write_geotiff(
    path: Path, array: np.ndarray, transform: Affine, crs: CRS, *, nodata: float | None,
    tags: Mapping[str, str], descriptions: list[str] | None = None, compression: str = "deflate",
) -> Path:
    """Write ``array`` (bands, rows, cols) atomically: a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        with rasterio.open(
            partial, "w", driver="GTiff", height=array.shape[1], width=array.shape[2], count=array.shape[0],
            dtype=array.dtype, crs=crs, transform=transform, nodata=nodata, **_compression(compression),
        ) as dst:
            dst.write(array)
            dst.update_tags(**tags)
            for index, description in enumerate(descriptions or [], start=1):
                dst.set_band_description(index, description)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return path


def clip_window_10m(aoi: BaseGeometry, transform: Affine, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Pixel box ``(col0, row0, col1, row1)`` around the AOI plus one pixel, even-aligned so 20 m bands nest."""
    left, bottom, right, top = aoi.bounds
    inverse = ~transform
    corners = [inverse * (x, y) for x in (left, right) for y in (bottom, top)]
    cols, rows = [c[0] for c in corners], [c[1] for c in corners]
    col0, row0 = math.floor(min(cols)) - 1, math.floor(min(rows)) - 1
    col1, row1 = math.ceil(max(cols)) + 1, math.ceil(max(rows)) + 1
    col0, row0 = max(0, col0 - col0 % 2), max(0, row0 - row0 % 2)
    col1, row1 = min(shape[1], col1 + col1 % 2), min(shape[0], row1 + row1 % 2)
    return col0, row0, col1, row1


def coverage_layers(aoi: BaseGeometry, transform: Affine, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Share of each pixel inside the AOI (supersampled), plus the all-touched and centre-inside masks."""
    fine_transform = transform * Affine.scale(1.0 / COVERAGE_SUPERSAMPLE)
    fine_shape = (shape[0] * COVERAGE_SUPERSAMPLE, shape[1] * COVERAGE_SUPERSAMPLE)
    fine = rasterize([(aoi, 1)], out_shape=fine_shape, transform=fine_transform, fill=0, dtype="uint8")
    fraction = fine.reshape(shape[0], COVERAGE_SUPERSAMPLE, shape[1], COVERAGE_SUPERSAMPLE).mean(axis=(1, 3)).astype(np.float32)
    touched = rasterize([(aoi, 1)], out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint8").astype(bool)
    centre = rasterize([(aoi, 1)], out_shape=shape, transform=transform, fill=0, all_touched=False, dtype="uint8").astype(bool)
    return fraction, touched, centre


def _tags(inputs: ProductInputs, role: str = "analytic", resampled: bool = False) -> dict[str, str]:
    return {
        "S2_ITEM_ID": inputs.candidate.item_id, "S2_DATETIME": inputs.candidate.datetime_utc.isoformat(),
        "REGISTRATION_STATUS": inputs.status, "SHIFT_X_M": f"{inputs.shift_m[0]:.4f}", "SHIFT_Y_M": f"{inputs.shift_m[1]:.4f}",
        "REFLECTANCE_SCALE": "1", "RESAMPLED": str(resampled).lower(), "PRODUCT_ROLE": role,
    }


def build_overlay(stack: np.ndarray, stack_transform: Affine, stack_crs: CRS, target: OverlayTarget,
                  gsd_m: float) -> tuple[np.ndarray, Affine, float]:
    """Nearest-neighbour copy of the shifted patch on a fine grid in the drone CRS, masked to the perimeter.

    The grid is anchored to the drone map's pixel lattice. If the perimeter would need more than
    ``MAX_OVERLAY_PIXELS`` pixels the grid is coarsened and the caller is told the pixel size actually used.
    """
    min_x, min_y, max_x, max_y = target.polygon.bounds
    gsd = gsd_m
    while math.ceil((max_x - min_x) / gsd + 2) * math.ceil((max_y - min_y) / gsd + 2) > MAX_OVERLAY_PIXELS:
        gsd *= 2.0
    origin_x = target.lattice.c + math.floor((min_x - target.lattice.c) / gsd) * gsd
    origin_y = target.lattice.f - math.floor((target.lattice.f - max_y) / gsd) * gsd
    transform = Affine(gsd, 0.0, origin_x, 0.0, -gsd, origin_y)
    shape = (math.ceil((origin_y - min_y) / gsd), math.ceil((max_x - origin_x) / gsd))

    overlay = np.full((stack.shape[0], *shape), np.nan, dtype=np.float32)
    reproject(
        stack, overlay, src_transform=stack_transform, src_crs=stack_crs, dst_transform=transform, dst_crs=target.crs,
        src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest,
    )
    inside = rasterize([(target.polygon, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    overlay[:, ~inside] = np.nan
    return overlay, transform, gsd


def _reflectance_stack(windows: Mapping[str, BandWindow], params: Mapping[str, RadiometryParams],
                       box: tuple[int, int, int, int], keep: np.ndarray) -> tuple[np.ndarray, list[str]]:
    col0, row0, col1, row1 = box
    names = list(windows)
    stack = np.stack([dn_to_reflectance(windows[name].data[row0:row1, col0:col1], params[name]) for name in names])
    stack[:, ~keep] = np.nan
    return stack, names


def write_products(inputs: ProductInputs, out_dir: Path, cfg: ProductsConfig) -> dict:
    """Write every analytic product and return paths plus facts for the report."""
    dx, dy = inputs.shift_m
    first_10m = next(iter(inputs.bands_10m.values()))
    transform_10m = apply_shift(first_10m.transform, dx, dy)
    shape_10m = first_10m.data.shape
    col0, row0, col1, row1 = clip_window_10m(inputs.aoi, transform_10m, shape_10m)
    clip_transform_10m = transform_10m * Affine.translation(col0, row0)
    clip_shape_10m = (row1 - row0, col1 - col0)

    fraction, touched, centre = coverage_layers(inputs.aoi, clip_transform_10m, clip_shape_10m)
    keep_10m = fraction > 0
    if inputs.mask_reject is not None:
        keep_10m &= ~inputs.mask_reject[row0:row1, col0:col1]

    stem = f"S2_{inputs.candidate.datetime_utc:%Y%m%d}_{inputs.candidate.mgrs_tile or 'tile'}"
    paths: dict[str, Path] = {}
    comp = cfg.compression

    stack_10m, names_10m = _reflectance_stack(inputs.bands_10m, inputs.params, (col0, row0, col1, row1), keep_10m)
    paths["analytic_10m"] = write_geotiff(
        out_dir / f"{stem}_10m.tif", stack_10m, clip_transform_10m, inputs.crs, nodata=float("nan"),
        tags=_tags(inputs), descriptions=names_10m, compression=comp,
    )
    overlay_gsd = None
    if cfg.write_overlay and inputs.overlay is not None:
        overlay, overlay_transform, overlay_gsd = build_overlay(
            stack_10m, clip_transform_10m, inputs.crs, inputs.overlay, cfg.overlay_gsd_m)
        paths["overlay"] = write_geotiff(
            out_dir / f"{stem}_overlay.tif", overlay, overlay_transform, inputs.overlay.crs, nodata=float("nan"),
            tags=_tags(inputs, role="overlay", resampled=True), descriptions=names_10m, compression=comp,
        )
    paths["coverage"] = write_geotiff(
        out_dir / f"{stem}_coverage.tif", np.stack([fraction, touched.astype(np.float32), centre.astype(np.float32)]),
        clip_transform_10m, inputs.crs, nodata=None, tags=_tags(inputs),
        descriptions=["coverage_fraction", "all_touched", "centre_inside"], compression=comp,
    )

    box_20m = (col0 // 2, row0 // 2, col1 // 2, row1 // 2)
    transform_20m = apply_shift(inputs.scl.transform, dx, dy) * Affine.translation(box_20m[0], box_20m[1])
    shape_20m = (box_20m[3] - box_20m[1], box_20m[2] - box_20m[0])
    _, touched_20m, _ = coverage_layers(inputs.aoi, transform_20m, shape_20m)
    keep_20m = touched_20m
    if inputs.mask_reject is not None:
        height, width = inputs.mask_reject.shape
        reject_20m = inputs.mask_reject.reshape(height // 2, 2, width // 2, 2).any(axis=(1, 3))
        keep_20m = touched_20m & ~reject_20m[box_20m[1]:box_20m[3], box_20m[0]:box_20m[2]]

    stack_20m = names_20m = None
    if cfg.write_20m and inputs.bands_20m:
        stack_20m, names_20m = _reflectance_stack(inputs.bands_20m, inputs.params, box_20m, keep_20m)
        paths["analytic_20m"] = write_geotiff(
            out_dir / f"{stem}_20m.tif", stack_20m, transform_20m, inputs.crs, nodata=float("nan"),
            tags=_tags(inputs), descriptions=names_20m, compression=comp,
        )
    scl = inputs.scl.data[box_20m[1]:box_20m[3], box_20m[0]:box_20m[2]].copy()
    scl[~touched_20m] = 0
    paths["scl"] = write_geotiff(
        out_dir / f"{stem}_scl.tif", scl[None], transform_20m, inputs.crs, nodata=0,
        tags=_tags(inputs), descriptions=["SCL"], compression=comp,
    )

    if cfg.write_crop_health_bands:
        paths["crop_health_bands"] = _write_crop_health_bands(
            inputs, out_dir / "bands", stack_10m, names_10m, stack_20m, names_20m, scl,
            clip_transform_10m, transform_20m, comp,
        )
    return {"paths": paths, "clip_shape_10m": list(clip_shape_10m), "aoi_pixels_10m": int(touched.sum()),
            "resampled": False, "grid_snapped": False, "overlay_gsd_m": overlay_gsd}


def _write_crop_health_bands(inputs, bands_dir, stack_10m, names_10m, stack_20m, names_20m, scl,
                             transform_10m, transform_20m, compression) -> Path | None:
    """Single-band files in the layout crop_health_sentinel reads.

    Values are reflectance x 10000 with the processing-baseline offset already removed, because that
    package divides by ``io.reflectance_scale`` (10000) and knows nothing about the offset.
    """
    sources = {name: (stack_10m, names_10m, transform_10m) for name in names_10m}
    if stack_20m is not None:
        sources.update({name: (stack_20m, names_20m, transform_20m) for name in names_20m})
    missing = [band for band in CROP_HEALTH_FILES if band != "SCL" and band not in sources]
    if missing:
        return None
    for band, filename in CROP_HEALTH_FILES.items():
        if band == "SCL":
            write_geotiff(bands_dir / filename, scl[None], transform_20m, inputs.crs, nodata=0,
                          tags=_tags(inputs), compression=compression)
            continue
        stack, names, transform = sources[band]
        scaled = (stack[names.index(band)] * np.float32(CROP_HEALTH_DN_SCALE))[None].astype(np.float32)
        write_geotiff(bands_dir / filename, scaled, transform, inputs.crs, nodata=float("nan"),
                      tags=_tags(inputs), compression=compression)
    return bands_dir
