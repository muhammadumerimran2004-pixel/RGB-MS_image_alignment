"""
Road-Grid Aligner — Layer 2 alternative alignment strategy.

Exploits the natural grid structure of agricultural drone imagery (bright roads,
dark tree rows) to compute a global translation offset.  The algorithm computes
1D projection profiles of binarised road/tree masks and cross-correlates them
to recover the (dx, dy) shift between RGB and MS registration grids.

**Assumption**: farm flight grids produce imagery where roads and crop rows
run approximately parallel to image axes.  Diagonal roads dilute projection
energy; increase ``blur_kernel_size`` or use the standard automated pipeline
for scenes with strongly rotated road orientations.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from drone_alignment.config.schema import (
    RoadGridConfig,
    TransformValidationConfig,
    TransformType,
)
from drone_alignment.alignment.feature_detector import (
    normalize_to_uint8, InsufficientContrastError, InsufficientValidDataError,
)
from drone_alignment.alignment.transform_estimator import (
    TransformResult,
    TransformUnreliableError,
)

logger = logging.getLogger("drone_alignment")


class InsufficientRoadFeatureError(ValueError):
    """Raised when the image does not contain enough road-like bright features."""
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binarize_mask(
    gray: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float,
    blur_kernel: int,
    mode: str = "bright",
) -> np.ndarray:
    """
    Produce a binary mask isolating the brightest or darkest pixels.

    Parameters
    ----------
    gray : uint8 image already normalized to [0, 255].
    valid_mask : boolean array — only these pixels contribute.
    percentile : threshold percentile (top for bright, bottom for dark).
    blur_kernel : Gaussian blur kernel size (must be odd).
    mode : ``"bright"`` keeps pixels *above* the percentile;
           ``"dark"`` keeps pixels *below* the percentile.

    Returns
    -------
    Binary uint8 mask (0 or 255).
    """
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    values = blurred[valid_mask]
    if values.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    threshold = float(np.percentile(values, percentile))
    if mode == "bright":
        mask = (blurred >= threshold).astype(np.uint8) * 255
    else:
        mask = (blurred <= threshold).astype(np.uint8) * 255
    # Zero out invalid pixels
    mask[~valid_mask] = 0
    return mask


def _projection_profiles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D projection profiles by summing a binary mask along each axis.

    Returns
    -------
    (horizontal_profile, vertical_profile) : 1D float64 arrays.
        horizontal_profile[i] = sum of row i  (length H) — encodes vertical position of horizontal features.
        vertical_profile[j]   = sum of col j  (length W) — encodes horizontal position of vertical features.
    """
    binary = (mask > 0).astype(np.float64)
    horizontal = binary.sum(axis=1)  # shape (H,)
    vertical = binary.sum(axis=0)    # shape (W,)
    return horizontal, vertical


def _cross_correlate_offset(signal_a: np.ndarray, signal_b: np.ndarray) -> int:
    """
    Find the integer shift that maximises the cross-correlation between two
    1D signals: ``signal_b`` shifted by the returned offset best aligns with ``signal_a``.

    Returns the offset in pixels (positive = signal_b needs to shift right/down
    to align with signal_a).
    """
    if signal_a.size == 0 or signal_b.size == 0:
        return 0
    # Normalise to zero-mean to make the correlation meaningful
    a = signal_a - signal_a.mean()
    b = signal_b - signal_b.mean()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0
    corr = np.correlate(a, b, mode="full")
    peak = int(np.argmax(corr))
    offset = peak - (len(b) - 1)
    return offset


def _cross_correlate_offset_constrained(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    center: int,
    half_window: int,
) -> int | None:
    """
    Like ``_cross_correlate_offset`` but restricts the search to
    ``[center - half_window, center + half_window]``.

    Returns ``None`` if the constrained window is entirely outside the valid
    correlation range (i.e. the strip is unusable).
    """
    if signal_a.size == 0 or signal_b.size == 0:
        return None
    a = signal_a - signal_a.mean()
    b = signal_b - signal_b.mean()
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    corr = np.correlate(a, b, mode="full")
    # Full correlation indices map: index i → offset = i - (len(b) - 1)
    base = len(b) - 1
    lo = max(0, center + base - half_window)
    hi = min(len(corr), center + base + half_window + 1)
    if lo >= hi:
        return None
    peak = int(lo + np.argmax(corr[lo:hi]))
    return peak - base


def _strip_offsets(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    num_strips: int,
    primary_offset: int,
    guardrail_px: int,
    min_pixels: int,
    axis: str = "horizontal",
) -> list[int]:
    """
    Divide the image into ``num_strips`` strips and compute constrained local
    offsets along one axis.

    Parameters
    ----------
    mask_a, mask_b : binary uint8 masks (RGB, MS).
    num_strips : how many strips.
    primary_offset : road-derived coarse offset used as centre of search window.
    guardrail_px : half-width of search window.
    min_pixels : minimum road pixels for a strip to be valid.
    axis : ``"horizontal"`` divides rows into strips, correlates vertical
           profiles → dx offsets.  ``"vertical"`` divides columns → dy offsets.

    Returns
    -------
    List of valid local offsets (may be shorter than ``num_strips``).
    """
    h, w = mask_a.shape
    offsets: list[int] = []

    if axis == "horizontal":
        strip_size = max(1, h // num_strips)
        for i in range(num_strips):
            r0 = i * strip_size
            r1 = min(r0 + strip_size, h)
            sub_a = mask_a[r0:r1, :]
            sub_b = mask_b[r0:r1, :]
            if sub_a.sum() < min_pixels * 255 or sub_b.sum() < min_pixels * 255:
                continue
            _, prof_a = _projection_profiles(sub_a)
            _, prof_b = _projection_profiles(sub_b)
            off = _cross_correlate_offset_constrained(prof_a, prof_b, primary_offset, guardrail_px)
            if off is not None:
                offsets.append(off)
    else:  # vertical strips → dy
        strip_size = max(1, w // num_strips)
        for i in range(num_strips):
            c0 = i * strip_size
            c1 = min(c0 + strip_size, w)
            sub_a = mask_a[:, c0:c1]
            sub_b = mask_b[:, c0:c1]
            if sub_a.sum() < min_pixels * 255 or sub_b.sum() < min_pixels * 255:
                continue
            prof_a, _ = _projection_profiles(sub_a)
            prof_b, _ = _projection_profiles(sub_b)
            off = _cross_correlate_offset_constrained(prof_a, prof_b, primary_offset, guardrail_px)
            if off is not None:
                offsets.append(off)

    return offsets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_road_grid_translation(
    rgb_band: np.ndarray,
    ms_band: np.ndarray,
    rgb_mask: np.ndarray,
    ms_mask: np.ndarray,
    target_gsd: float,
    road_grid_config: RoadGridConfig,
    transform_config: TransformValidationConfig,
) -> TransformResult:
    """
    Compute a translation-only alignment from structural road/tree-row features.

    This function operates on the low-resolution registration grid arrays
    already produced by ``coarse_align()``.

    Parameters
    ----------
    rgb_band : 2-D float32 registration band (e.g. green channel).
    ms_band : 2-D float32 registration band (same channel).
    rgb_mask, ms_mask : boolean validity masks.
    target_gsd : GSD of the registration grid in metres/pixel.
    road_grid_config : tunable algorithm parameters.
    transform_config : hard translation/validation thresholds.

    Returns
    -------
    TransformResult with a 2×3 affine matrix ``[[1, 0, tx], [0, 1, ty]]``.

    Raises
    ------
    InsufficientRoadFeatureError
        If road pixels are below ``min_road_pixel_fraction``.
    TransformUnreliableError
        If the computed translation violates sanity thresholds.
    """
    cfg = road_grid_config

    # ------------------------------------------------------------------
    # Step 1: Normalize float32 reflectance → uint8 grayscale
    # ------------------------------------------------------------------
    try:
        rgb_gray = normalize_to_uint8(rgb_band, rgb_mask, apply_clahe=False)
        ms_gray = normalize_to_uint8(ms_band, ms_mask, apply_clahe=False)
    except (InsufficientContrastError, InsufficientValidDataError) as exc:
        raise InsufficientRoadFeatureError(f"Cannot extract road features: {exc}") from exc

    common_mask = rgb_mask.astype(bool) & ms_mask.astype(bool)

    # ------------------------------------------------------------------
    # Step 2: Binarize road (bright) masks
    # ------------------------------------------------------------------
    rgb_road = _binarize_mask(rgb_gray, common_mask, cfg.bright_percentile, cfg.blur_kernel_size, mode="bright")
    ms_road = _binarize_mask(ms_gray, common_mask, cfg.bright_percentile, cfg.blur_kernel_size, mode="bright")

    # Validate sufficient road coverage
    valid_count = int(common_mask.sum())
    rgb_road_count = int((rgb_road > 0).sum())
    ms_road_count = int((ms_road > 0).sum())
    min_road_pixels = int(valid_count * cfg.min_road_pixel_fraction)

    if rgb_road_count < min_road_pixels or ms_road_count < min_road_pixels:
        raise InsufficientRoadFeatureError(
            f"Insufficient road features detected: RGB road pixels={rgb_road_count}, "
            f"MS road pixels={ms_road_count}, minimum required={min_road_pixels} "
            f"({cfg.min_road_pixel_fraction:.1%} of {valid_count} valid pixels)."
        )
    logger.info(
        "Road mask coverage: RGB=%d px (%.1f%%), MS=%d px (%.1f%%)",
        rgb_road_count, 100 * rgb_road_count / max(1, valid_count),
        ms_road_count, 100 * ms_road_count / max(1, valid_count),
    )

    # ------------------------------------------------------------------
    # Step 3: 1D projection profiles → primary offset
    # ------------------------------------------------------------------
    rgb_h_prof, rgb_v_prof = _projection_profiles(rgb_road)
    ms_h_prof, ms_v_prof = _projection_profiles(ms_road)

    dy_road = _cross_correlate_offset(rgb_h_prof, ms_h_prof)
    dx_road = _cross_correlate_offset(rgb_v_prof, ms_v_prof)
    logger.info("Primary road offset: dx=%d px, dy=%d px", dx_road, dy_road)

    # ------------------------------------------------------------------
    # Step 4: Strip-based refinement (guardrailed)
    # ------------------------------------------------------------------
    dx_strips = _strip_offsets(
        rgb_road, ms_road, cfg.num_strips, dx_road, cfg.guardrail_px,
        cfg.min_strip_road_pixels, axis="horizontal",
    )
    dy_strips = _strip_offsets(
        rgb_road, ms_road, cfg.num_strips, dy_road, cfg.guardrail_px,
        cfg.min_strip_road_pixels, axis="vertical",
    )

    # Consensus: median of valid strips (falls back to primary if no strips valid)
    tx = float(np.median(dx_strips)) if dx_strips else float(dx_road)
    ty = float(np.median(dy_strips)) if dy_strips else float(dy_road)
    num_valid_strips = len(dx_strips) + len(dy_strips)
    total_strips = cfg.num_strips * 2
    logger.info(
        "Strip refinement: tx=%.1f px, ty=%.1f px (%d/%d valid strips)",
        tx, ty, num_valid_strips, total_strips,
    )

    # ------------------------------------------------------------------
    # Step 5: Dark-lane (tree-row) confirmation (optional)
    # ------------------------------------------------------------------
    if cfg.use_dark_refinement:
        rgb_dark = _binarize_mask(rgb_gray, common_mask, cfg.dark_percentile, cfg.blur_kernel_size, mode="dark")
        ms_dark = _binarize_mask(ms_gray, common_mask, cfg.dark_percentile, cfg.blur_kernel_size, mode="dark")

        dark_h_rgb, dark_v_rgb = _projection_profiles(rgb_dark)
        dark_h_ms, dark_v_ms = _projection_profiles(ms_dark)

        dy_dark = _cross_correlate_offset(dark_h_rgb, dark_h_ms)
        dx_dark = _cross_correlate_offset(dark_v_rgb, dark_v_ms)

        dx_divergence = abs(dx_dark - tx)
        dy_divergence = abs(dy_dark - ty)
        if dx_divergence > cfg.guardrail_px or dy_divergence > cfg.guardrail_px:
            logger.warning(
                "Dark-lane offset (dx=%d, dy=%d) diverges from road offset (tx=%.1f, ty=%.1f) "
                "by (%.0f, %.0f) px — possible radiometric mismatch on vegetation.",
                dx_dark, dy_dark, tx, ty, dx_divergence, dy_divergence,
            )
        else:
            logger.info(
                "Dark-lane confirmation: dx=%d, dy=%d (divergence: %.0f, %.0f px — within guardrail)",
                dx_dark, dy_dark, dx_divergence, dy_divergence,
            )

    # ------------------------------------------------------------------
    # Step 6: Validate & build TransformResult
    # ------------------------------------------------------------------
    tx_m = tx * target_gsd
    ty_m = ty * target_gsd
    magnitude_m = math.hypot(tx_m, ty_m)

    if magnitude_m > transform_config.max_translation_m:
        raise TransformUnreliableError(
            f"Road-grid translation ({magnitude_m:.2f}m) exceeds maximum allowed "
            f"threshold ({transform_config.max_translation_m:.2f}m)."
        )

    matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
    inlier_ratio = num_valid_strips / max(1, total_strips)

    return TransformResult(
        matrix=matrix,
        transform_type=TransformType.AFFINE,
        translation_px=(tx, ty),
        translation_m=(tx_m, ty_m),
        rotation_deg=0.0,
        scale=(1.0, 1.0),
        inlier_ratio=inlier_ratio,
        num_inliers=num_valid_strips,
        num_total_matches=total_strips,
        method="road_grid",
        channel_pair="unknown",
    )
