"""Layer B: scaled reflectance, Sentinel spectral indices, and summaries."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from crop_health_sentinel.io.raster_artifacts import write_single_band_raster
from crop_health_sentinel.io.structured_artifacts import write_csv, write_json
from crop_health_sentinel.models.types import SpectralOutput
from crop_health_sentinel.utils.math_utils import safe_divide
from crop_health_sentinel.utils.preview import write_preview
from crop_health_sentinel.utils.stats import finite_summary


def process_spectral(context, quality, logger: logging.Logger | None = None) -> SpectralOutput:
    logger = logger or logging.getLogger(__name__); bands, cfg = context.bands, context.config
    mask = quality.analytic_valid_mask
    raw = {"green": bands.green, "red": bands.red, "red_edge": bands.red_edge, "nir": bands.nir}
    reflectance = {name: np.where(mask, array / cfg.io.reflectance_scale, np.nan).astype(np.float32) for name, array in raw.items()}
    nir, red, green, red_edge = reflectance["nir"], reflectance["red"], reflectance["green"], reflectance["red_edge"]
    formulas = {
        "ndvi": (nir - red, nir + red), "ndre": (nir - red_edge, nir + red_edge),
        "gndvi": (nir - green, nir + green), "evi2": (2.5 * (nir - red), nir + 2.4 * red + 1.0),
    }
    indices = {name: safe_divide(num, den, mask, cfg.spectral.denominator_epsilon) for name, (num, den) in formulas.items() if name in cfg.spectral.enabled_indices}
    summary = {"reflectance": {name: finite_summary(array[mask], cfg.spectral.percentiles) for name, array in reflectance.items()},
        "indices": {name: finite_summary(array[mask], cfg.spectral.percentiles) for name, array in indices.items()},
        "enabledIndices": list(cfg.spectral.enabled_indices)}
    output = SpectralOutput(reflectance=reflectance, indices=indices, summary=summary)
    _write_artifacts(context, output); logger.info("Layer B completed indices=%s", ",".join(indices))
    return output


def _write_artifacts(context, output: SpectralOutput) -> None:
    phase = Path(context.output_dir) / "phase_02"; prefix = context.metadata.scene_id
    for name, array in {**{f"reflectance_{key}": value for key, value in output.reflectance.items()}, **output.indices}.items():
        write_single_band_raster(phase / f"{prefix}_{name}.tif", array, context.grid, dtype="float32", nodata=-9999.0)
        if context.config.artifacts.write_previews: write_preview(phase / f"{prefix}_{name}.png", array, name, context.config.artifacts.preview_dpi)
    write_json(phase / f"{prefix}_layer_b_spectral_summary.json", output.summary)
    rows = [{"signal": section, "name": name, **stats} for section, values in (("reflectance", output.summary["reflectance"]), ("index", output.summary["indices"])) for name, stats in values.items()]
    write_csv(phase / f"{prefix}_layer_b_spectral_summary.csv", ["signal", "name", "count", "mean", "std", "min", "max", "cv", "percentiles"], rows)
