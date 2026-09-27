"""Layer A: Sentinel pixel trust masks, reason codes, and scene eligibility."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from crop_health_sentinel.io.raster_artifacts import write_single_band_raster
from crop_health_sentinel.io.structured_artifacts import write_csv, write_json
from crop_health_sentinel.models.types import LayerAOutput, QualityProvenance, SceneStatus
from crop_health_sentinel.utils.math_utils import safe_divide
from crop_health_sentinel.utils.preview import write_preview


_REASONS = ("unusable", "cloud", "shadow", "haze", "snow", "invalid_numeric")


def _optional_flag(values, coverage) -> np.ndarray:
    return np.zeros_like(coverage, dtype=bool) if values is None else (coverage & (values != 0))


def process_quality(context, logger: logging.Logger | None = None) -> LayerAOutput:
    logger = logger or logging.getLogger(__name__); bands, cfg = context.bands, context.config
    selected = context.field_inner_mask if cfg.io.use_field_inner_mask else context.field_mask
    red, nir = bands.red / cfg.io.reflectance_scale, bands.nir / cfg.io.reflectance_scale
    ndvi = safe_divide(nir - red, nir + red, bands.required_coverage_mask, cfg.spectral.denominator_epsilon)
    optional = bands.optional_coverage_masks
    cloud_probability_coverage = optional.get("cloud_probability", np.zeros_like(selected, dtype=bool))
    cloud_probability = bands.cloud_probability
    cloud = np.isin(bands.scl, cfg.quality.scl_cloud_classes)
    cloud |= _optional_flag(bands.opaque_cloud, optional.get("opaque_cloud", np.zeros_like(selected, dtype=bool)))
    cloud |= _optional_flag(bands.cirrus, optional.get("cirrus", np.zeros_like(selected, dtype=bool)))
    if cloud_probability is not None: cloud |= cloud_probability_coverage & (cloud_probability >= cfg.quality.cloud_probability_threshold)
    if bands.qa60 is not None:
        qa_coverage = optional.get("qa60", np.zeros_like(selected, dtype=bool)); qa = np.zeros_like(selected, dtype=bool)
        for bit in cfg.quality.qa60_cloud_bits: qa |= (bands.qa60 & (1 << bit)) != 0
        cloud |= qa_coverage & qa
    shadow = np.isin(bands.scl, cfg.quality.scl_shadow_classes)
    snow = np.isin(bands.scl, cfg.quality.scl_snow_classes)
    snow |= _optional_flag(bands.snow_ice, optional.get("snow_ice", np.zeros_like(selected, dtype=bool)))
    haze = np.zeros_like(selected, dtype=bool)
    if bands.aot is not None:
        aot_coverage = optional.get("aot", np.zeros_like(selected, dtype=bool)); haze = aot_coverage & (bands.aot / cfg.quality.aot_scale >= cfg.quality.aot_haze_threshold)
    unusable = np.isin(bands.scl, cfg.quality.scl_unusable_classes) | ~bands.required_coverage_mask
    invalid = ~np.isfinite(red) | ~np.isfinite(nir) | ~np.isfinite(ndvi)
    conditions = {"unusable": unusable, "cloud": cloud, "shadow": shadow, "haze": haze, "snow": snow, "invalid_numeric": invalid}
    display = selected.copy(); analytic = selected.copy()
    for name in cfg.quality.display_exclusions: display &= ~conditions[name]
    for name in cfg.quality.analytic_exclusions: analytic &= ~conditions[name]
    low_vegetation = np.zeros_like(selected, dtype=bool)
    if cfg.quality.vegetation_gate_enabled:
        low_vegetation = np.isfinite(ndvi) & (ndvi < cfg.quality.vegetation_gate_ndvi_threshold); analytic &= ~low_vegetation
    reason = np.zeros(selected.shape, dtype=np.uint8); reason[~selected] = 1
    for code, name in enumerate(_REASONS, start=2): reason[conditions[name] & selected] = code
    reason[low_vegetation & selected] = 8
    clear = ~(unusable | cloud | shadow | haze | snow | invalid)
    confidence = clear.astype(np.float32)
    if cloud_probability is not None:
        confidence[cloud_probability_coverage] = np.clip(1 - cloud_probability[cloud_probability_coverage] / 100, 0, 1)
    weight = np.where(analytic, confidence, 0).astype(np.float32)
    selected_count, analytic_count, display_count = int(selected.sum()), int(analytic.sum()), int(display.sum())
    analytic_fraction, display_fraction = analytic_count / selected_count, display_count / selected_count
    if analytic_count < cfg.quality.minimum_analytic_pixels or analytic_fraction < cfg.quality.display_only_minimum_fraction: status = SceneStatus.REJECT
    elif analytic_fraction >= cfg.quality.analytics_ready_minimum_fraction: status = SceneStatus.ANALYTICS_READY
    else: status = SceneStatus.DISPLAY_ONLY
    provenance = QualityProvenance.DEGRADED if bands.missing_optional_bands else QualityProvenance.COMPLETE
    summary = {"selectedFieldPixels": selected_count, "displayValidPixels": display_count, "analyticValidPixels": analytic_count,
        "displayFraction": display_fraction, "analyticFraction": analytic_fraction, "sceneStatus": status.value,
        "qualityProvenance": provenance.value, "missingOptionalBands": list(bands.missing_optional_bands),
        "reasonCodePrecedence": ["outside_field", *_REASONS, "low_vegetation"]}
    output = LayerAOutput(display, analytic, weight, reason, status, provenance, summary)
    _write_artifacts(context, output); logger.info("Layer A status=%s analytic=%s/%s", status.value, analytic_count, selected_count)
    return output


def _write_artifacts(context, output: LayerAOutput) -> None:
    phase = Path(context.output_dir) / "phase_01"; prefix = context.metadata.scene_id
    artifacts = {"field_mask": context.field_mask, "field_inner_mask": context.field_inner_mask, "display_valid_mask": output.display_valid_mask,
        "analytic_valid_mask": output.analytic_valid_mask, "quality_weight": output.quality_weight, "reason_code_mask": output.reason_codes}
    for name, value in artifacts.items():
        floating = np.issubdtype(value.dtype, np.floating); write_single_band_raster(phase / f"{prefix}_{name}.tif", value, context.grid, dtype="float32" if floating else "uint8", nodata=-9999.0 if floating else 0)
        if context.config.artifacts.write_previews: write_preview(phase / f"{prefix}_{name}.png", value, name, context.config.artifacts.preview_dpi)
    write_json(phase / f"{prefix}_layer_a_quality_summary.json", output.summary)
    write_csv(phase / f"{prefix}_layer_a_quality_summary.csv", list(output.summary), [output.summary])
