"""Strict crop geometry normalization and raster mask construction."""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import geometry_window, rasterize
from rasterio.windows import Window, intersection, transform as window_transform
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.validation import explain_validity

from crop_health_sentinel.errors import GeometryError, ResourceLimitError
from crop_health_sentinel.models.types import ReferenceGrid


def normalize_crop_geometry(value: str | Mapping[str, object] | BaseGeometry) -> BaseGeometry:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise GeometryError(f"crop_geometry is not valid JSON: {exc}") from exc
    if isinstance(value, BaseGeometry):
        geometry = value
    elif isinstance(value, Mapping):
        object_type = value.get("type")
        if object_type == "Feature":
            value = value.get("geometry")  # type: ignore[assignment]
        elif object_type == "FeatureCollection":
            features = value.get("features")
            if not isinstance(features, list) or len(features) != 1:
                raise GeometryError("FeatureCollection must contain exactly one feature.")
            value = features[0].get("geometry") if isinstance(features[0], Mapping) else None  # type: ignore[assignment]
        if not isinstance(value, Mapping):
            raise GeometryError("Crop geometry is missing a GeoJSON geometry object.")
        if value.get("type") not in {"Polygon", "MultiPolygon"}:
            raise GeometryError("Crop geometry must be a Polygon or MultiPolygon.")
        geometry = shape(value)
    else:
        raise GeometryError("crop_geometry must be GeoJSON, a JSON string, or a Shapely geometry.")
    if geometry.is_empty or not geometry.is_valid:
        raise GeometryError(f"Crop geometry is invalid: {explain_validity(geometry)}")
    return geometry


def project_geometry(geometry: BaseGeometry, destination_crs: CRS) -> BaseGeometry:
    transformer = Transformer.from_crs("EPSG:4326", destination_crs, always_xy=True)
    projected = transform_geometry(transformer.transform, geometry)
    if projected.is_empty or not projected.is_valid:
        raise GeometryError("Crop geometry became invalid during CRS transformation.")
    return projected


def crop_window(dataset, geometry_grid: BaseGeometry, margin: int, maximum_pixels: int) -> ReferenceGrid:
    try:
        window = geometry_window(dataset, [geometry_grid.__geo_interface__], pad_x=margin, pad_y=margin)
        window = intersection(window.round_offsets().round_lengths(), Window(0, 0, dataset.width, dataset.height))
    except Exception as exc:
        raise GeometryError("Crop geometry does not overlap the B8 raster extent.") from exc
    if window.width <= 0 or window.height <= 0:
        raise GeometryError("Crop geometry does not overlap the B8 raster extent.")
    if int(window.width * window.height) > maximum_pixels:
        raise ResourceLimitError("Crop window exceeds io.max_crop_window_pixels.")
    return ReferenceGrid(
        crs=CRS.from_user_input(dataset.crs), transform=window_transform(window, dataset.transform),
        width=int(window.width), height=int(window.height), profile=dataset.profile.copy(), source_window=window,
    )


def rasterize_field_masks(geometry_grid: BaseGeometry, grid: ReferenceGrid, config) -> tuple[np.ndarray, np.ndarray]:
    field = rasterize([(geometry_grid.__geo_interface__, 1)], out_shape=(grid.height, grid.width), transform=grid.transform,
        fill=0, all_touched=config.io.rasterization == "all_touched", dtype="uint8").astype(bool)
    if not field.any():
        raise GeometryError("Crop geometry rasterizes to zero pixels.")
    inner = field.copy()
    for _ in range(config.io.edge_erosion_pixels):
        padded = np.pad(inner, 1, mode="constant", constant_values=False)
        inner = np.logical_and.reduce([
            padded[row:row + field.shape[0], column:column + field.shape[1]]
            for row in range(3) for column in range(3)
        ])
    return field, inner
