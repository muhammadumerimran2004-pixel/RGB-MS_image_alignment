"""Layer C: configured field/pixel health scoring."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from crop_health_sentinel.errors import InsufficientAnalyticsError
from crop_health_sentinel.io.raster_artifacts import write_single_band_raster
from crop_health_sentinel.io.structured_artifacts import write_csv, write_json
from crop_health_sentinel.models.types import HealthOutput
from crop_health_sentinel.utils.math_utils import normalize_score
from crop_health_sentinel.utils.preview import write_preview
from crop_health_sentinel.utils.stats import health_class


def process_health(context, quality, spectral, logger: logging.Logger | None = None) -> HealthOutput:
    logger = logger or logging.getLogger(__name__); cfg = context.config
    components, weighted_sum, weight_sum = {}, 0.0, 0.0
    for name in cfg.spectral.enabled_indices:
        mean = spectral.summary["indices"][name]["mean"]; weight = cfg.health.field_weights.get(name, 0)
        if mean is None or weight <= 0: continue
        bounds = cfg.health.index_ranges[name]; score = float(normalize_score(np.array([mean]), bounds.low, bounds.high)[0])
        components[name] = {"mean": mean, "score": score, "configuredWeight": weight}; weighted_sum += score * weight; weight_sum += weight
    if weight_sum == 0: raise InsufficientAnalyticsError("No enabled finite index with positive field weight.")
    field_score = weighted_sum / weight_sum; field_class = health_class(field_score, cfg.health.class_thresholds)
    weak = spectral.indices.get(cfg.health.weak_area_index); weak_fraction = None
    if weak is not None:
        valid = quality.analytic_valid_mask & np.isfinite(weak)
        if valid.any(): weak_fraction = float(np.count_nonzero(weak[valid] < cfg.health.weak_area_threshold) / valid.sum())
    source = cfg.health.pixel_primary_index if cfg.health.pixel_primary_index in spectral.indices else cfg.health.pixel_fallback_index
    if source not in spectral.indices: raise InsufficientAnalyticsError("Neither configured pixel-health index is available.")
    bounds = cfg.health.index_ranges[source]; pixel_map = normalize_score(spectral.indices[source], bounds.low, bounds.high)
    pixel_map = np.where(quality.analytic_valid_mask, pixel_map, np.nan).astype(np.float32)
    classes = np.zeros(pixel_map.shape, dtype=np.uint8); valid = np.isfinite(pixel_map)
    t = cfg.health.class_thresholds
    classes[valid] = 1; classes[valid & (pixel_map >= t.weak)] = 2; classes[valid & (pixel_map >= t.moderate)] = 3
    classes[valid & (pixel_map >= t.good)] = 4; classes[valid & (pixel_map >= t.very_good)] = 5
    summary = {"fieldHealthScore": field_score, "healthClass": field_class, "weakAreaFraction": weak_fraction,
        "weakAreaIndex": cfg.health.weak_area_index, "pixelScoreSource": source, "scoreComponents": components}
    output = HealthOutput(field_score, field_class, weak_fraction, source, pixel_map, classes, summary)
    _write_artifacts(context, output); logger.info("Layer C score=%.3f class=%s", field_score, field_class); return output


def _write_artifacts(context, output: HealthOutput) -> None:
    phase = Path(context.output_dir) / "phase_03"; prefix = context.metadata.scene_id
    write_single_band_raster(phase / f"{prefix}_pixel_health_map.tif", output.pixel_health_map, context.grid, dtype="float32", nodata=-9999.0)
    write_single_band_raster(phase / f"{prefix}_pixel_health_class_map.tif", output.pixel_health_class_map, context.grid, dtype="uint8", nodata=0)
    if context.config.artifacts.write_previews:
        write_preview(phase / f"{prefix}_pixel_health_map.png", output.pixel_health_map, "pixel health", context.config.artifacts.preview_dpi)
        write_preview(phase / f"{prefix}_pixel_health_class_map.png", output.pixel_health_class_map, "pixel health class", context.config.artifacts.preview_dpi)
    write_json(phase / f"{prefix}_layer_c_health_summary.json", output.summary)
    write_csv(phase / f"{prefix}_layer_c_health_summary.csv", list(output.summary), [output.summary])
