"""Analytical pipeline entry point.

The processing layers are deliberately introduced in later implementation phases.
This module already owns the stable public signature so callers do not depend on
internal modules while the package is being constructed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from .config.loader import load_config
from .config.schema import SentinelCropHealthConfig, parse_config
from .errors import ConfigurationError
from .io.scene import build_scene_context
from .io.structured_artifacts import write_json
from .layer_a_quality import process_quality
from .layer_b_spectral import process_spectral
from .layer_c_health import process_health
from .layer_d_report import process_final_report
from .layer_e_h3 import preflight_h3_capacity, process_h3
from .models.types import PipelineResult


def run_crop_health_report(
    raw_data_dir: str | Path,
    crop_geometry: str | Mapping[str, Any] | Any,
    output_dir: str | Path,
    scene_meta: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    config: SentinelCropHealthConfig | Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> PipelineResult:
    """Run one complete, inspectable per-crop analytical report."""
    if config is not None and config_path is not None:
        raise ConfigurationError("Pass either config or config_path, not both.")
    resolved_config = load_config(config_path) if config is None else (config if isinstance(config, SentinelCropHealthConfig) else parse_config(config))
    log = logger or logging.getLogger(__name__)
    context = build_scene_context(raw_data_dir, crop_geometry, output_dir, scene_meta, resolved_config)
    Path(context.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(context.output_dir) / "crop_polygon.json", context.crop_geometry_wgs84.__geo_interface__)
    quality = process_quality(context, log)
    if quality.scene_status.value == "reject":
        return PipelineResult(quality.scene_status, quality, None, None, None, None, Path(context.output_dir))
    preflight_h3_capacity(context, quality)
    spectral = process_spectral(context, quality, log)
    health = process_health(context, quality, spectral, log)
    h3 = process_h3(context, quality, health, log)
    report = process_final_report(context, quality, spectral, health, h3, log)
    return PipelineResult(quality.scene_status, quality, spectral, health, h3, report, Path(context.output_dir))
