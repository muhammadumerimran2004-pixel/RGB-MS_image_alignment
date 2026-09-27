"""Strict executable configuration for Sentinel crop-health version 1."""

from __future__ import annotations

from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crop_health_sentinel.errors import ConfigurationError
from crop_health_sentinel.version import ALGORITHM_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(StrictModel):
    green: str
    red: str
    red_edge: str
    nir: str
    scl: str
    cloud_probability: str
    opaque_cloud: str
    cirrus: str
    snow_ice: str
    qa60: str
    aot: str

    @model_validator(mode="after")
    def validate_relative_basenames(self) -> "PathsConfig":
        for name, value in self.__dict__.items():
            path = PurePath(value)
            if not value or path.name != value or ".." in path.parts:
                raise ValueError(f"paths.{name} must be a non-empty file basename.")
        return self


class IoConfig(StrictModel):
    reflectance_scale: float = Field(gt=0)
    crop_window_margin_pixels: int = Field(ge=0)
    max_crop_window_pixels: int = Field(ge=1)
    rasterization: Literal["all_touched", "center"]
    edge_erosion_pixels: int = Field(ge=0)
    use_field_inner_mask: bool
    output_compression: Literal["deflate"]
    output_tile_size: Literal[256, 512, 1024]


class QualityConfig(StrictModel):
    scl_cloud_classes: tuple[int, ...]
    scl_shadow_classes: tuple[int, ...]
    scl_snow_classes: tuple[int, ...]
    scl_unusable_classes: tuple[int, ...]
    cloud_probability_threshold: float = Field(ge=0, le=100)
    qa60_cloud_bits: tuple[int, ...]
    aot_scale: float = Field(gt=0)
    aot_haze_threshold: float = Field(ge=0)
    display_exclusions: tuple[str, ...]
    analytic_exclusions: tuple[str, ...]
    vegetation_gate_enabled: bool
    vegetation_gate_ndvi_threshold: float
    minimum_analytic_pixels: int = Field(ge=1)
    display_only_minimum_fraction: float = Field(ge=0, le=1)
    analytics_ready_minimum_fraction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "QualityConfig":
        if self.display_only_minimum_fraction >= self.analytics_ready_minimum_fraction:
            raise ValueError(
                "quality.display_only_minimum_fraction must be less than "
                "quality.analytics_ready_minimum_fraction."
            )
        return self


class SpectralConfig(StrictModel):
    denominator_epsilon: float = Field(gt=0)
    enabled_indices: tuple[Literal["ndvi", "ndre", "gndvi", "evi2"], ...]
    summary_statistic: Literal["mean"]
    percentiles: tuple[float, ...]

    @model_validator(mode="after")
    def validate_percentiles(self) -> "SpectralConfig":
        if not self.enabled_indices:
            raise ValueError("spectral.enabled_indices must contain at least one index.")
        if any(value < 0 or value > 100 for value in self.percentiles):
            raise ValueError("spectral.percentiles must be in [0, 100].")
        if tuple(sorted(self.percentiles)) != self.percentiles:
            raise ValueError("spectral.percentiles must be sorted ascending.")
        return self


class IndexRange(StrictModel):
    low: float
    high: float

    @model_validator(mode="after")
    def validate_order(self) -> "IndexRange":
        if self.low >= self.high:
            raise ValueError("index range low must be less than high.")
        return self


class ClassThresholds(StrictModel):
    very_good: float = Field(ge=0, le=100)
    good: float = Field(ge=0, le=100)
    moderate: float = Field(ge=0, le=100)
    weak: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_order(self) -> "ClassThresholds":
        if not self.very_good > self.good > self.moderate > self.weak:
            raise ValueError("health.class_thresholds must be strictly descending.")
        return self


class HealthConfig(StrictModel):
    index_ranges: dict[Literal["ndvi", "ndre", "gndvi", "evi2"], IndexRange]
    field_weights: dict[Literal["ndvi", "ndre", "gndvi", "evi2"], float]
    class_thresholds: ClassThresholds
    weak_area_index: Literal["ndvi", "ndre", "gndvi", "evi2"]
    weak_area_threshold: float
    pixel_primary_index: Literal["ndvi", "ndre", "gndvi", "evi2"]
    pixel_fallback_index: Literal["ndvi", "ndre", "gndvi", "evi2"]

    @model_validator(mode="after")
    def validate_health_configuration(self) -> "HealthConfig":
        if any(weight < 0 for weight in self.field_weights.values()):
            raise ValueError("health.field_weights must be non-negative.")
        return self


class H3Config(StrictModel):
    resolution: int = Field(ge=0, le=15)
    metric_crs: str
    minimum_effective_support: float = Field(ge=0)
    max_analytic_pixels: int = Field(ge=1)
    max_candidate_cells: int = Field(ge=1)
    batch_size: int = Field(ge=1, le=50_000)
    candidate_estimate_safety_factor: float = Field(ge=1.0)
    capacity_policy: Literal["fail"]


class ArtifactsConfig(StrictModel):
    write_previews: bool
    preview_dpi: int = Field(ge=1)
    retain_development_intermediates: bool


class SentinelCropHealthConfig(StrictModel):
    algorithm_version: str
    paths: PathsConfig
    io: IoConfig
    quality: QualityConfig
    spectral: SpectralConfig
    health: HealthConfig
    h3: H3Config
    artifacts: ArtifactsConfig

    @model_validator(mode="after")
    def validate_cross_field_contract(self) -> "SentinelCropHealthConfig":
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError(f"algorithm_version must equal {ALGORITHM_VERSION!r}.")
        enabled = set(self.spectral.enabled_indices)
        if not any(self.health.field_weights.get(index, 0) > 0 for index in enabled):
            raise ValueError("At least one enabled spectral index must have a positive field weight.")
        missing_ranges = enabled.difference(self.health.index_ranges)
        if missing_ranges:
            raise ValueError(f"Missing health.index_ranges for: {sorted(missing_ranges)}.")
        for source in (self.health.weak_area_index, self.health.pixel_primary_index, self.health.pixel_fallback_index):
            if source not in self.health.index_ranges:
                raise ValueError(f"health source {source!r} must have an index range.")
        if not self.h3.metric_crs.upper().startswith("EPSG:"):
            raise ValueError("h3.metric_crs must be an EPSG CRS identifier.")
        return self


def parse_config(value: object) -> SentinelCropHealthConfig:
    """Convert Pydantic validation details into the package's stable error type."""
    try:
        return SentinelCropHealthConfig.model_validate(value)
    except Exception as exc:  # Pydantic's public validation hierarchy is intentionally wrapped.
        raise ConfigurationError(f"Invalid Sentinel crop-health configuration: {exc}") from exc

