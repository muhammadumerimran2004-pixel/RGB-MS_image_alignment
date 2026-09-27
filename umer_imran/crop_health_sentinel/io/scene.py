"""Bounded Sentinel band validation, alignment, and context construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from crop_health_sentinel.errors import GeometryError, InputValidationError, RasterMetadataError, ResourceLimitError
from crop_health_sentinel.models.types import AlignedBands, SceneContext, normalize_scene_metadata
from crop_health_sentinel.version import ALGORITHM_VERSION

from .band_registry import BANDS, BandKind
from .geometry import crop_window, normalize_crop_geometry, project_geometry, rasterize_field_masks


def _resolve_paths(raw_data_dir: Path, config) -> dict[str, Path | None]:
    if not raw_data_dir.is_dir():
        raise InputValidationError(f"raw_data_dir does not exist: {raw_data_dir}")
    paths: dict[str, Path | None] = {}
    for band in BANDS:
        path = raw_data_dir / getattr(config.paths, band.config_key)
        if band.required and not path.is_file():
            raise InputValidationError(f"Missing required Sentinel band: {path.name}")
        paths[band.logical_name] = path if path.is_file() else None
    return paths


def _read_band(path: Path, grid, kind: BandKind) -> tuple[np.ndarray, np.ndarray]:
    resampling = Resampling.bilinear if kind is BandKind.CONTINUOUS else Resampling.nearest
    try:
        with rasterio.open(path) as source:
            if source.count != 1 or source.crs is None or source.transform is None:
                raise RasterMetadataError(f"Invalid raster metadata for {path.name}.")
            with WarpedVRT(source, crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height,
                           resampling=resampling) as vrt:
                data = vrt.read(1, masked=True)
    except (RasterMetadataError, GeometryError, ResourceLimitError):
        raise
    except Exception as exc:
        raise RasterMetadataError(f"Unable to read/alignment-check {path.name}: {exc}") from exc
    coverage = np.ascontiguousarray(~np.ma.getmaskarray(data), dtype=bool)
    if kind is BandKind.CONTINUOUS:
        result = np.ascontiguousarray(np.ma.filled(data, np.nan), dtype=np.float32)
    else:
        result = np.ascontiguousarray(np.ma.filled(data, 0))
    return result, coverage


def build_scene_context(raw_data_dir, crop_geometry, output_dir, scene_meta, config) -> SceneContext:
    raw_path, out_path = Path(raw_data_dir).resolve(), Path(output_dir).resolve()
    paths = _resolve_paths(raw_path, config)
    geometry_wgs84 = normalize_crop_geometry(crop_geometry)
    nir_path = paths["nir"]
    assert nir_path is not None
    try:
        with rasterio.open(nir_path) as nir:
            if nir.crs is None or nir.transform is None or nir.count != 1:
                raise RasterMetadataError("B8.tif must have one band, CRS, and affine transform.")
            geometry_grid = project_geometry(geometry_wgs84, nir.crs)
            grid = crop_window(nir, geometry_grid, config.io.crop_window_margin_pixels, config.io.max_crop_window_pixels)
    except RasterMetadataError:
        raise
    except Exception as exc:
        raise RasterMetadataError(f"Unable to inspect B8.tif: {exc}") from exc
    field, inner = rasterize_field_masks(geometry_grid, grid, config)
    values, coverages, missing = {}, {}, []
    for band in BANDS:
        path = paths[band.logical_name]
        if path is None:
            missing.append(band.logical_name)
            values[band.logical_name] = None
            continue
        value, coverage = _read_band(path, grid, band.kind)
        values[band.logical_name], coverages[band.logical_name] = value, coverage
    required_coverage = np.logical_and.reduce([coverages[band.logical_name] for band in BANDS if band.required])
    bands = AlignedBands(
        **{name: values[name] for name in ("green", "red", "red_edge", "nir", "scl", "cloud_probability", "opaque_cloud", "cirrus", "snow_ice", "qa60", "aot")},
        required_coverage_mask=np.ascontiguousarray(required_coverage),
        optional_coverage_masks={name: coverages[name] for name in coverages if name not in {"green", "red", "red_edge", "nir", "scl"}},
        missing_optional_bands=tuple(missing),
    )
    return SceneContext(metadata=normalize_scene_metadata(scene_meta), raw_data_dir=raw_path, output_dir=out_path,
        config=config, algorithm_version=ALGORITHM_VERSION, grid=grid, crop_geometry_wgs84=geometry_wgs84,
        crop_geometry_grid_crs=geometry_grid, field_mask=field, field_inner_mask=inner, bands=bands)
