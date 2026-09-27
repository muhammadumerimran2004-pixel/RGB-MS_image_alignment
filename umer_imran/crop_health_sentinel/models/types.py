"""Small, dependency-light runtime contracts available from Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from affine import Affine
from pyproj import CRS
from rasterio.windows import Window
from shapely.geometry.base import BaseGeometry

from crop_health_sentinel.errors import InputValidationError


@dataclass(frozen=True, slots=True)
class SceneMetadata:
    farm: str
    field: str
    date: str
    scene_id: str
    run_id: str


class SceneStatus(StrEnum):
    REJECT = "reject"
    DISPLAY_ONLY = "display_only"
    ANALYTICS_READY = "analytics_ready"


class QualityProvenance(StrEnum):
    COMPLETE = "complete"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class ReferenceGrid:
    crs: CRS
    transform: Affine
    width: int
    height: int
    profile: Mapping[str, Any]
    source_window: Window


@dataclass(frozen=True, slots=True)
class AlignedBands:
    green: np.ndarray
    red: np.ndarray
    red_edge: np.ndarray
    nir: np.ndarray
    scl: np.ndarray
    cloud_probability: np.ndarray | None
    opaque_cloud: np.ndarray | None
    cirrus: np.ndarray | None
    snow_ice: np.ndarray | None
    qa60: np.ndarray | None
    aot: np.ndarray | None
    required_coverage_mask: np.ndarray
    optional_coverage_masks: Mapping[str, np.ndarray]
    missing_optional_bands: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SceneContext:
    metadata: SceneMetadata
    raw_data_dir: Path
    output_dir: Path
    config: Any
    algorithm_version: str
    grid: ReferenceGrid
    crop_geometry_wgs84: BaseGeometry
    crop_geometry_grid_crs: BaseGeometry
    field_mask: np.ndarray
    field_inner_mask: np.ndarray
    bands: AlignedBands


@dataclass(frozen=True, slots=True)
class LayerAOutput:
    display_valid_mask: np.ndarray
    analytic_valid_mask: np.ndarray
    quality_weight: np.ndarray
    reason_codes: np.ndarray
    scene_status: SceneStatus
    quality_provenance: QualityProvenance
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SpectralOutput:
    reflectance: Mapping[str, np.ndarray]
    indices: Mapping[str, np.ndarray]
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class HealthOutput:
    field_health_score: float
    health_class: str
    weak_area_fraction: float | None
    pixel_score_source: str
    pixel_health_map: np.ndarray
    pixel_health_class_map: np.ndarray
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class H3CellResult:
    h3_index: str
    mean_health: float
    std_health: float
    min_health: float
    max_health: float
    actual_pixel_count: int
    analytic_pixel_count: int
    effective_analytic_pixel_count: float
    low_support: bool


@dataclass(frozen=True, slots=True)
class LayerEOutput:
    cells: tuple[H3CellResult, ...]
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FinalReportOutput:
    report: Mapping[str, Any]
    summary: Mapping[str, Any]
    report_path: Path


@dataclass(frozen=True, slots=True)
class PipelineResult:
    scene_status: SceneStatus
    layer_a: LayerAOutput
    spectral: SpectralOutput | None
    health: HealthOutput | None
    h3: LayerEOutput | None
    final_report: FinalReportOutput | None
    output_dir: Path


_METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "farm": ("farm", "farm_no", "farmId"),
    "field": ("field", "crop_unique_no", "cropUniqueNo", "cropId"),
    "date": ("date", "imageryDate"),
    "scene_id": ("scene_id", "sceneId"),
    "run_id": ("run_id", "runId"),
}
_METADATA_DEFAULTS = {
    "farm": "farm",
    "field": "crop",
    "date": "unknown-date",
    "scene_id": "sentinel-scene",
}


def _first_present(metadata: Mapping[str, Any], field: str) -> str | None:
    for key in _METADATA_ALIASES[field]:
        if key in metadata and metadata[key] is not None:
            value = str(metadata[key]).strip()
            if not value:
                raise InputValidationError(f"Scene metadata field '{key}' must not be empty.")
            return value
    return None


def normalize_scene_metadata(metadata: Mapping[str, Any]) -> SceneMetadata:
    """Normalize documented metadata aliases into the immutable runtime contract."""
    if not isinstance(metadata, Mapping):
        raise InputValidationError("scene_meta must be a mapping.")

    values = {
        field: _first_present(metadata, field) or default
        for field, default in _METADATA_DEFAULTS.items()
    }
    run_id = _first_present(metadata, "run_id") or values["scene_id"]
    return SceneMetadata(run_id=run_id, **values)
