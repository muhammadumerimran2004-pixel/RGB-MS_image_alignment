"""Optional, fail-closed AROSICS global candidate matcher for geospatial co-registration.

This is the feature-free candidate tried inside global estimation
(:func:`drone_alignment.pipeline._estimate_global_candidate`) after every
classical/learned detector has been rejected. AROSICS' local refinement of
the verified global result lives in :mod:`drone_alignment.alignment.arosics_local`.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine

from drone_alignment.alignment.arosics_staging import ArosicsExecutionError
from drone_alignment.alignment.feature_matcher import InsufficientMatchesError
from drone_alignment.alignment.transform_estimator import TransformResult, TransformUnreliableError
from drone_alignment.config.schema import ArosicsGlobalCandidateConfig, TransformType, TransformValidationConfig

logger = logging.getLogger("drone_alignment")

__all__ = [
    "ArosicsUnavailableError", "ArosicsExecutionError", "is_arosics_available", "match_arosics",
]


class ArosicsUnavailableError(InsufficientMatchesError):
    """Raised when the optional AROSICS package or its geospatial dependencies are unavailable."""
    pass


def is_arosics_available() -> bool:
    """Check if AROSICS can be imported."""
    try:
        import arosics
        return True
    except ImportError:
        return False


def match_arosics(
    rgb_band: np.ndarray,
    ms_band: np.ndarray,
    rgb_valid_mask: np.ndarray,
    ms_valid_mask: np.ndarray,
    registration_transform: Affine,
    crs: Any,
    registration_gsd: float,
    arosics_config: ArosicsGlobalCandidateConfig,
    transform_config: TransformValidationConfig,
) -> TransformResult:
    """
    Execute AROSICS COREG on downsampled registration-grid bands to compute global subpixel shift.

    Writes temporary single-band GeoTIFFs, passes them to AROSICS COREG, and
    constructs a validated TransformResult.

    Args:
        rgb_band: 2D array of reference RGB registration band.
        ms_band: 2D array of target MS registration band.
        rgb_valid_mask: 2D boolean or uint8 mask of valid RGB pixels.
        ms_valid_mask: 2D boolean or uint8 mask of valid MS pixels.
        registration_transform: Affine transform for the registration grid.
        crs: Coordinate reference system for the registration grid.
        registration_gsd: Meters per pixel of the registration grid.
        arosics_config: AROSICS global-candidate matching configuration.
        transform_config: Geometric sanity thresholds.

    Returns:
        TransformResult containing the estimated 2x3 affine matrix.

    Raises:
        ArosicsUnavailableError: If AROSICS is not installed.
        InsufficientMatchesError: If AROSICS fails to find a shift or falls below reliability threshold.
        TransformUnreliableError: If the estimated shift exceeds configured limits.
    """
    try:
        from arosics import COREG
    except ImportError as exc:
        raise ArosicsUnavailableError(f"AROSICS dependencies are unavailable: {exc}") from exc

    if rgb_band.shape != ms_band.shape:
        raise ValueError("RGB and MS registration bands must share identical shape.")

    height, width = rgb_band.shape
    nodata_val = -9999.0

    with tempfile.TemporaryDirectory(prefix="arosics_coreg_") as tmpdir:
        tmp_path = Path(tmpdir)
        ref_path = tmp_path / "ref_band.tif"
        tgt_path = tmp_path / "tgt_band.tif"

        if crs is None:
            raise ArosicsExecutionError(
                "AROSICS matching requires a valid CRS; validate_inputs() should guarantee one."
            )

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": nodata_val,
            "width": width,
            "height": height,
            "count": 1,
            "crs": crs,
            "transform": registration_transform if registration_transform is not None else Affine.identity(),
        }

        ref_data = rgb_band.astype(np.float32).copy()
        ref_mask = rgb_valid_mask.astype(bool)
        ref_data[~ref_mask] = nodata_val

        tgt_data = ms_band.astype(np.float32).copy()
        tgt_mask = ms_valid_mask.astype(bool)
        tgt_data[~tgt_mask] = nodata_val

        with rasterio.open(ref_path, "w", **profile) as dst:
            dst.write(ref_data, 1)

        with rasterio.open(tgt_path, "w", **profile) as dst:
            dst.write(tgt_data, 1)

        try:
            # COREG is AROSICS' global mode.  It estimates a single sub-pixel
            # X/Y translation; it is not the grid-based COREG_LOCAL API.
            coreg = COREG(
                im_ref=str(ref_path),
                im_tgt=str(tgt_path),
                ws=arosics_config.window_size,
                max_shift=arosics_config.max_shift_px,
                nodata=(nodata_val, nodata_val),
                ignore_errors=True,
                v=False,
            )
            coreg.calculate_spatial_shifts()
        except (RuntimeError, ValueError, AssertionError) as exc:
            raise InsufficientMatchesError(f"AROSICS COREG execution error: {exc}") from exc

    if not getattr(coreg, "success", False):
        raise InsufficientMatchesError("AROSICS COREG could not determine spatial shift.")

    # AROSICS' CoReg.py exposes this shift-quality metric as `shift_reliability`,
    # not `reliability`. It is only set once a non-zero shift has been found
    # (see CoReg.py:1622-1629), so a missing/non-finite value is treated as a
    # failed gate rather than silently accepted with inlier_ratio=1.0.
    reliability = getattr(coreg, "shift_reliability", None)
    if reliability is None or not np.isfinite(reliability):
        raise InsufficientMatchesError("AROSICS COREG did not report a shift reliability.")
    if reliability < arosics_config.min_reliability:
        raise InsufficientMatchesError(
            f"AROSICS reliability ({reliability:.1f}%) is below minimum threshold ({arosics_config.min_reliability:.1f}%)."
        )

    shift_x_px = float(coreg.x_shift_px)
    shift_y_px = float(coreg.y_shift_px)

    tx_m = shift_x_px * registration_gsd
    ty_m = shift_y_px * registration_gsd
    magnitude_m = math.hypot(tx_m, ty_m)

    if magnitude_m > transform_config.max_translation_m:
        raise TransformUnreliableError(
            f"AROSICS translation ({magnitude_m:.2f}m) exceeds max allowed threshold ({transform_config.max_translation_m:.2f}m)."
        )

    matrix = np.array([
        [1.0, 0.0, shift_x_px],
        [0.0, 1.0, shift_y_px],
    ], dtype=np.float64)

    inlier_ratio = float(reliability) / 100.0

    return TransformResult(
        matrix=matrix,
        transform_type=TransformType.AFFINE,
        translation_px=(shift_x_px, shift_y_px),
        translation_m=(tx_m, ty_m),
        rotation_deg=0.0,
        scale=(1.0, 1.0),
        inlier_ratio=inlier_ratio,
        num_inliers=1,
        num_total_matches=1,
        method="arosics",
        channel_pair="unknown",
    )
