"""S1 - the perimeter of the aligned stack: pixels valid in BOTH the RGB and the aligned MS raster."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union

from sentinel_patch.errors import SentinelPatchError

logger = logging.getLogger("sentinel_patch")

MAX_DECIMATED_DIM = 2048
MAX_FILLED_HOLE_FRACTION = 0.01
SECONDARY_POLYGON_WARN_FRACTION = 0.05
SMALL_AREA_WARN_M2 = 2500.0  # 0.25 ha


class EmptyFootprintError(SentinelPatchError):
    code = "FOOTPRINT_EMPTY"


@dataclass(frozen=True)
class StackFootprint:
    polygon_stack_crs: BaseGeometry
    stack_crs: CRS
    polygon_wgs84: BaseGeometry
    area_m2: float
    grid_transform: Affine  # native transform of the aligned MS raster; anchors the overlay's pixel lattice


def _decimated_shape(height: int, width: int, max_dim: int) -> tuple[int, int]:
    scale = min(1.0, max_dim / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _read_valid_mask(path: Path, out_shape: tuple[int, int], *, use_dataset_mask: bool) -> tuple[np.ndarray, Affine, CRS]:
    """Validity at reduced size. Pixel magnitude never defines validity: GDAL masks and alpha do."""
    with rasterio.open(path) as src:
        if use_dataset_mask:  # RGB orthophotos usually carry an alpha band that dataset_mask honours
            mask = src.dataset_mask(out_shape=out_shape, resampling=Resampling.nearest)
        else:
            mask = src.read_masks(1, out_shape=out_shape, resampling=Resampling.nearest)
        transform = src.transform * Affine.scale(src.width / out_shape[1], src.height / out_shape[0])
        return mask > 0, transform, CRS.from_user_input(src.crs)


def _fill_small_holes(mask: np.ndarray) -> np.ndarray:
    holes = ndimage.binary_fill_holes(mask) & ~mask
    labels, count = ndimage.label(holes)
    if count == 0:
        return mask
    sizes = ndimage.sum(holes, labels, index=np.arange(1, count + 1))
    small = np.isin(labels, 1 + np.flatnonzero(sizes < MAX_FILLED_HOLE_FRACTION * mask.sum()))
    return mask | small


def extract_stack_footprint(aligned_ms_path: Path, rgb_path: Path, max_dim: int = MAX_DECIMATED_DIM) -> StackFootprint:
    """Valid-data polygon of RGB AND aligned MS, in the aligned MS CRS (never the raster bounding box)."""
    with rasterio.open(aligned_ms_path) as ms:
        out_shape = _decimated_shape(ms.height, ms.width, max_dim)
        native_transform = ms.transform
    ms_valid, grid_transform, crs = _read_valid_mask(aligned_ms_path, out_shape, use_dataset_mask=False)

    with rasterio.open(rgb_path) as rgb:
        rgb_shape = _decimated_shape(rgb.height, rgb.width, max_dim)
    rgb_valid, rgb_transform, rgb_crs = _read_valid_mask(rgb_path, rgb_shape, use_dataset_mask=True)
    rgb_on_grid = np.zeros(ms_valid.shape, dtype=np.uint8)
    reproject(
        rgb_valid.astype(np.uint8), rgb_on_grid, src_transform=rgb_transform, src_crs=rgb_crs,
        dst_transform=grid_transform, dst_crs=crs, resampling=Resampling.nearest,
    )

    common = ndimage.binary_opening(ms_valid & (rgb_on_grid > 0), structure=np.ones((3, 3)))
    common = _fill_small_holes(common)
    if not common.any():
        raise EmptyFootprintError("The aligned MS and RGB rasters share no valid pixels.")

    pieces = [shape(geom) for geom, value in shapes(common.astype(np.uint8), mask=common, transform=grid_transform) if value == 1]
    union = unary_union(pieces)
    polygons = list(union.geoms) if hasattr(union, "geoms") else [union]
    polygons.sort(key=lambda polygon: polygon.area, reverse=True)
    main = polygons[0]
    for extra in polygons[1:]:
        if extra.area > SECONDARY_POLYGON_WARN_FRACTION * main.area:
            logger.warning("Secondary valid-data polygon with %.0f m2 ignored; using the largest one.", extra.area)
    polygon = main.simplify(abs(grid_transform.a), preserve_topology=True).buffer(0)
    if polygon.is_empty or not polygon.is_valid:
        raise EmptyFootprintError("Stack footprint polygon is empty or invalid after cleanup.")
    if hasattr(polygon, "geoms"):  # simplify() can split a thin neck into a multi-part result
        polygon = max(polygon.geoms, key=lambda part: part.area)
    if polygon.area < SMALL_AREA_WARN_M2:
        logger.warning("Stack footprint is only %.0f m2; registration will likely be rejected as too small.", polygon.area)

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return StackFootprint(
        polygon_stack_crs=polygon, stack_crs=crs,
        polygon_wgs84=transform_geometry(to_wgs84.transform, polygon), area_m2=float(polygon.area),
        grid_transform=native_transform,
    )


def write_footprint_geojson(footprint: StackFootprint, path: Path) -> Path:
    """One WGS84 feature (RFC 7946). crop_health_sentinel requires exactly one feature, so the
    stack-CRS polygon is carried as WKT in the properties instead of as a second feature."""
    collection = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": mapping(footprint.polygon_wgs84),
            "properties": {
                "role": "stack_footprint", "area_m2": footprint.area_m2,
                "stack_crs": footprint.stack_crs.to_string(), "stack_crs_wkt": footprint.polygon_stack_crs.wkt,
            },
        }],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(collection), encoding="utf-8")
    os.replace(partial, path)
    return path
