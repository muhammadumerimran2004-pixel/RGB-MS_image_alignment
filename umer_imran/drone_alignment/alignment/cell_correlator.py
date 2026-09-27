"""Feature-agnostic local residual evidence from compact registration bands.

This module never opens rasters or writes output.  It receives RGB reference
and MS target arrays already projected onto one compact registration grid,
applies the verified global affine to the target once, and estimates the small
remaining target-to-reference translation for each grid cell.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from drone_alignment.alignment.local_evidence import (
    CellMatchStatus,
    LocalMatchSample,
    SparseDisplacementResult,
)
from drone_alignment.alignment.displacement_field import fit_selected_displacement_field
from drone_alignment.config.schema import CellCorrelationConfig


@dataclass(frozen=True)
class PhaseEstimate:
    """One bounded phase-correlation peak on a target-to-reference surface."""

    shift_dx_dy: tuple[float, float]
    phase_response: float
    peak_sharpness: float
    primary_peak: float
    second_peak: float
    is_on_search_boundary: bool


@dataclass(frozen=True)
class CellBounds:
    """Core cell and halo crop bounds in x/y-exclusive coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int
    hx0: int
    hy0: int
    hx1: int
    hy1: int


@dataclass(frozen=True)
class HeldoutCellScore:
    row: int
    col: int
    channel: str
    global_score: float
    local_score: float
    improvement: float


@dataclass(frozen=True)
class HeldoutFieldValidation:
    scores: tuple[HeldoutCellScore, ...]
    evaluated_count: int
    mean_improvement: float | None
    median_improvement: float | None
    win_fraction: float | None
    is_accepted: bool
    reason: str


def _hann_window(shape: tuple[int, int]) -> np.ndarray:
    """Return a float32 2-D Hann window with ``shape == (height, width)``."""
    height, width = shape
    if height < 2 or width < 2:
        raise ValueError("Phase-correlation patches must be at least 2x2 pixels.")
    return np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)


def _cell_bounds(
    shape: tuple[int, int], row: int, col: int, config: CellCorrelationConfig,
) -> CellBounds:
    height, width = shape
    y0, y1 = row * height // config.grid_rows, (row + 1) * height // config.grid_rows
    x0, x1 = col * width // config.grid_cols, (col + 1) * width // config.grid_cols
    return CellBounds(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        hx0=max(0, x0 - config.cell_halo_px),
        hy0=max(0, y0 - config.cell_halo_px),
        hx1=min(width, x1 + config.cell_halo_px),
        hy1=min(height, y1 + config.cell_halo_px),
    )


def _prepare_patch(
    values: np.ndarray,
    valid: np.ndarray,
    config: CellCorrelationConfig,
) -> tuple[np.ndarray | None, CellMatchStatus | None]:
    """Robustly normalize one masked patch and apply a spatial Hann window."""
    mask = valid.astype(bool) & np.isfinite(values)
    if not np.any(mask):
        return None, CellMatchStatus.INSUFFICIENT_VALID_DATA
    valid_values = values[mask].astype(np.float32)
    low, high = np.percentile(valid_values, (2.0, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None, CellMatchStatus.LOW_TEXTURE
    scaled = np.zeros(values.shape, dtype=np.float32)
    scaled[mask] = np.clip((values[mask] - low) / (high - low), 0.0, 1.0)
    mean = float(np.mean(scaled[mask]))
    std = float(np.std(scaled[mask]))
    if std < config.min_texture_std:
        return None, CellMatchStatus.LOW_TEXTURE
    scaled[mask] -= mean
    scaled[~mask] = 0.0
    return scaled * _hann_window(values.shape), None


def _parabolic_offset(before: float, center: float, after: float) -> float:
    denominator = before - 2.0 * center + after
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))


def _phase_correlate_bounded(
    reference: np.ndarray,
    target: np.ndarray,
    radius: int,
    exclusion_radius: int,
) -> PhaseEstimate:
    """Estimate the target-to-reference shift within an explicit residual bound."""
    if reference.shape != target.shape or reference.ndim != 2:
        raise ValueError("Reference and target phase-correlation patches must be equally shaped 2-D arrays.")
    height, width = reference.shape
    if radius * 2 + 1 > min(height, width):
        raise ValueError("Phase-correlation patch is too small for the configured search radius.")

    ref_fft = np.fft.fft2(reference)
    target_fft = np.fft.fft2(target)
    cross_power = ref_fft * np.conj(target_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    surface = np.fft.fftshift(np.fft.ifft2(cross_power).real)

    cy, cx = height // 2, width // 2
    y0, y1 = cy - radius, cy + radius + 1
    x0, x1 = cx - radius, cx + radius + 1
    bounded = surface[y0:y1, x0:x1]
    peak_y_local, peak_x_local = np.unravel_index(int(np.argmax(bounded)), bounded.shape)
    peak_y, peak_x = y0 + peak_y_local, x0 + peak_x_local
    primary = float(surface[peak_y, peak_x])

    boundary = (
        peak_y_local in (0, bounded.shape[0] - 1)
        or peak_x_local in (0, bounded.shape[1] - 1)
    )
    sub_x = sub_y = 0.0
    if not boundary and 0 < peak_x < width - 1 and 0 < peak_y < height - 1:
        sub_x = _parabolic_offset(surface[peak_y, peak_x - 1], primary, surface[peak_y, peak_x + 1])
        sub_y = _parabolic_offset(surface[peak_y - 1, peak_x], primary, surface[peak_y + 1, peak_x])

    yy, xx = np.ogrid[: bounded.shape[0], : bounded.shape[1]]
    excluded = (xx - peak_x_local) ** 2 + (yy - peak_y_local) ** 2 <= exclusion_radius ** 2
    sidelobes = bounded[~excluded]
    if sidelobes.size == 0:
        raise ValueError("Peak exclusion radius leaves no phase-correlation sidelobes.")
    second = float(np.max(sidelobes))
    median = float(np.median(sidelobes))
    mean = float(np.mean(sidelobes))
    std = float(np.std(sidelobes))
    sharpness = (primary - median) / max(second - median, 1e-12)
    psr = (primary - mean) / max(std, 1e-12)
    response = float(np.clip(1.0 - math.exp(-max(psr, 0.0) / 8.0), 0.0, 1.0))
    return PhaseEstimate(
        shift_dx_dy=(float(peak_x - cx + sub_x), float(peak_y - cy + sub_y)),
        phase_response=response,
        peak_sharpness=float(sharpness),
        primary_peak=primary,
        second_peak=second,
        is_on_search_boundary=boundary,
    )


def _sample_source_xy(center_xy: tuple[float, float], global_matrix: np.ndarray) -> tuple[float, float]:
    inverse = cv2.invertAffineTransform(global_matrix.astype(np.float64))
    x, y = center_xy
    return (
        float(inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]),
        float(inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]),
    )


def _channel_sample(
    row: int,
    col: int,
    bounds: CellBounds,
    reference: np.ndarray,
    target: np.ndarray,
    common_valid: np.ndarray,
    global_matrix: np.ndarray,
    channel: str,
    config: CellCorrelationConfig,
) -> LocalMatchSample:
    center = ((bounds.x0 + bounds.x1) / 2.0, (bounds.y0 + bounds.y1) / 2.0)
    source_xy = _sample_source_xy(center, global_matrix)
    core_valid = common_valid[bounds.y0:bounds.y1, bounds.x0:bounds.x1]
    valid_fraction = float(np.mean(core_valid)) if core_valid.size else 0.0
    base = dict(
        row=row,
        col=col,
        center_xy=center,
        source_ms_xy=source_xy,
        predicted_rgb_xy=center,
        residual_dx_dy=(0.0, 0.0),
        displacement_dx_dy=(float(global_matrix[0, 2]), float(global_matrix[1, 2])),
        confidence=0.0,
        road_score=None,
        second_road_score=None,
        tree_residual_dx_dy=None,
        channel=channel,
        evidence=f"phase_correlation:{channel}",
        valid_fraction=valid_fraction,
    )
    if valid_fraction < config.min_valid_fraction:
        return LocalMatchSample(
            **base,
            status=CellMatchStatus.INSUFFICIENT_VALID_DATA,
            rejection_reason="core valid fraction is below configured minimum",
        )

    patch_slice = np.s_[bounds.hy0:bounds.hy1, bounds.hx0:bounds.hx1]
    reference_patch, reference_status = _prepare_patch(reference[patch_slice], common_valid[patch_slice], config)
    target_patch, target_status = _prepare_patch(target[patch_slice], common_valid[patch_slice], config)
    status = reference_status or target_status
    if status is not None:
        return LocalMatchSample(**base, status=status, rejection_reason="patch lacks usable texture")

    estimate = _phase_correlate_bounded(
        reference_patch, target_patch, config.max_search_radius_px, config.peak_exclusion_radius_px,
    )
    sharpness_score = float(np.clip(
        (estimate.peak_sharpness - 1.0) / (config.full_confidence_peak_sharpness - 1.0), 0.0, 1.0,
    ))
    confidence = estimate.phase_response * sharpness_score * valid_fraction
    shift = estimate.shift_dx_dy
    base.update(
        predicted_rgb_xy=(center[0] + shift[0], center[1] + shift[1]),
        residual_dx_dy=shift,
        displacement_dx_dy=(float(global_matrix[0, 2] + shift[0]), float(global_matrix[1, 2] + shift[1])),
        confidence=float(confidence),
        phase_response=estimate.phase_response,
        peak_sharpness=estimate.peak_sharpness,
        phase_shift_dx_dy=shift,
    )
    if estimate.is_on_search_boundary:
        status, reason = CellMatchStatus.SEARCH_BOUNDARY, "best phase peak is on the residual search boundary"
    elif estimate.phase_response < config.min_phase_response:
        status, reason = CellMatchStatus.LOW_PHASE_RESPONSE, "phase response is below configured minimum"
    elif estimate.peak_sharpness < config.min_peak_sharpness:
        status, reason = CellMatchStatus.AMBIGUOUS_PEAK, "second phase peak is too similar to the primary peak"
    elif confidence < config.min_match_confidence:
        status, reason = CellMatchStatus.LOW_CONFIDENCE, "combined phase confidence is below configured minimum"
    else:
        status, reason = CellMatchStatus.ACCEPTED, None
    return LocalMatchSample(**base, status=status, rejection_reason=reason)


def _select_channel_sample(samples: list[LocalMatchSample], config: CellCorrelationConfig) -> LocalMatchSample:
    accepted = [sample for sample in samples if sample.status == CellMatchStatus.ACCEPTED]
    ranked = sorted(samples, key=lambda sample: sample.confidence, reverse=True)
    if not accepted:
        return ranked[0]
    if len(accepted) == 1:
        return accepted[0]
    accepted.sort(key=lambda sample: sample.confidence, reverse=True)
    best, second = accepted[0], accepted[1]
    disagreement = math.dist(best.residual_dx_dy, second.residual_dx_dy)
    if disagreement > config.max_channel_disagreement_px:
        return replace(
            best,
            status=CellMatchStatus.CHANNEL_CONFLICT,
            confidence=0.0,
            rejection_reason=(
                f"accepted {best.channel} and {second.channel} shifts disagree by "
                f"{disagreement:.2f} px"
            ),
        )
    return replace(best, confidence=min(1.0, best.confidence + 0.05))


def _apply_spatial_filter(
    samples: list[LocalMatchSample], config: CellCorrelationConfig,
) -> list[LocalMatchSample]:
    accepted = [sample for sample in samples if sample.status == CellMatchStatus.ACCEPTED]
    by_cell = {(sample.row, sample.col): sample for sample in accepted}
    filtered: list[LocalMatchSample] = []
    for sample in samples:
        if sample.status != CellMatchStatus.ACCEPTED:
            filtered.append(sample)
            continue
        neighbors = [
            other for (row, col), other in by_cell.items()
            if (row, col) != (sample.row, sample.col)
            and max(abs(row - sample.row), abs(col - sample.col)) == 1
        ]
        if len(neighbors) >= 2:
            median = (
                float(np.median([other.residual_dx_dy[0] for other in neighbors])),
                float(np.median([other.residual_dx_dy[1] for other in neighbors])),
            )
            if math.dist(sample.residual_dx_dy, median) > config.max_neighbor_difference_px:
                sample = replace(
                    sample,
                    status=CellMatchStatus.SPATIAL_OUTLIER,
                    confidence=0.0,
                    rejection_reason="residual differs excessively from accepted neighboring cells",
                )
        filtered.append(sample)
    return filtered


def _spatial_coverage(samples: list[LocalMatchSample], shape: tuple[int, int]) -> float:
    if len(samples) < 3:
        return 0.0
    points = np.asarray([sample.center_xy for sample in samples], dtype=np.float32)
    hull = cv2.convexHull(points)
    height, width = shape
    return float(cv2.contourArea(hull)) / float(max(1, height * width))


def _masked_correlation(reference: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float | None:
    values_ref = reference[valid].astype(np.float64)
    values_target = target[valid].astype(np.float64)
    if values_ref.size < 64 or np.std(values_ref) == 0 or np.std(values_target) == 0:
        return None
    return float(np.corrcoef(values_ref, values_target)[0, 1])


def _holdout_samples(
    accepted: tuple[LocalMatchSample, ...], max_count: int,
) -> tuple[LocalMatchSample, ...]:
    """Return deterministic row-major samples spread across the accepted set."""
    ordered = tuple(sorted(accepted, key=lambda sample: (sample.row, sample.col)))
    if len(ordered) <= max_count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, max_count, dtype=int)
    return tuple(ordered[index] for index in indices)


def evaluate_heldout_field_improvement(
    evidence: SparseDisplacementResult,
    rgb_bands: dict[str, np.ndarray],
    ms_bands: dict[str, np.ndarray],
    rgb_mask: np.ndarray,
    ms_mask: np.ndarray,
    global_matrix: np.ndarray,
    selected_field_name: str,
    config: CellCorrelationConfig,
) -> HeldoutFieldValidation:
    """Require a field to improve cells omitted from their own fitting set.

    The target is globally warped once.  For each held-out accepted cell, the
    selected field is fit without that cell, evaluated at its center, and used
    to shift a haloed target crop.  Scores are computed only over the original
    core, so interpolation borders never count as local improvement.
    """
    if global_matrix.shape != (2, 3) or not np.isfinite(global_matrix).all():
        raise ValueError("Held-out field validation requires a finite 2x3 global affine matrix.")
    if rgb_mask.shape != ms_mask.shape or rgb_mask.ndim != 2:
        raise ValueError("Held-out field validation requires equally shaped 2-D masks.")
    shape = rgb_mask.shape
    accepted = evidence.accepted
    candidates = _holdout_samples(accepted, config.max_holdout_cells)
    if len(accepted) < 4 or not candidates:
        return HeldoutFieldValidation((), 0, None, None, None, False, "Fewer than four accepted cells are available.")

    required_channels = {sample.channel for sample in candidates}
    if None in required_channels:
        raise ValueError("Accepted cell-correlation samples must record their selected channel.")
    for channel in required_channels:
        if channel not in rgb_bands or channel not in ms_bands:
            raise ValueError(f"Missing held-out validation channel: {channel}")
        if rgb_bands[channel].shape != shape or ms_bands[channel].shape != shape:
            raise ValueError("Held-out validation bands must match mask shape.")

    height, width = shape
    warped_mask = cv2.warpAffine(
        ms_mask.astype(np.uint8), global_matrix.astype(np.float64), (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    common_valid = rgb_mask.astype(bool) & warped_mask
    warped_ms = {
        channel: cv2.warpAffine(
            ms_bands[channel].astype(np.float32), global_matrix.astype(np.float64), (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        for channel in required_channels
    }

    scores: list[HeldoutCellScore] = []
    for held_out in candidates:
        training = tuple(sample for sample in accepted if sample is not held_out)
        if len(training) < 3:
            continue
        try:
            field = fit_selected_displacement_field(
                selected_field_name, training, shape, config.grid_rows, config.grid_cols, config,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        predicted = field.evaluate(np.asarray([held_out.center_xy], dtype=np.float64))[0]
        bounds = _cell_bounds(shape, held_out.row, held_out.col, config)
        patch_slice = np.s_[bounds.hy0:bounds.hy1, bounds.hx0:bounds.hx1]
        core_y0, core_y1 = bounds.y0 - bounds.hy0, bounds.y1 - bounds.hy0
        core_x0, core_x1 = bounds.x0 - bounds.hx0, bounds.x1 - bounds.hx0
        channel = held_out.channel
        reference_patch = rgb_bands[channel][patch_slice]
        target_patch = warped_ms[channel][patch_slice]
        valid_patch = common_valid[patch_slice]
        shifted_target = cv2.warpAffine(
            target_patch, np.array([[1.0, 0.0, predicted[0]], [0.0, 1.0, predicted[1]]]),
            (target_patch.shape[1], target_patch.shape[0]), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        shifted_valid = cv2.warpAffine(
            valid_patch.astype(np.uint8), np.array([[1.0, 0.0, predicted[0]], [0.0, 1.0, predicted[1]]]),
            (valid_patch.shape[1], valid_patch.shape[0]), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)
        core = np.s_[core_y0:core_y1, core_x0:core_x1]
        global_score = _masked_correlation(reference_patch[core], target_patch[core], valid_patch[core])
        local_score = _masked_correlation(
            reference_patch[core], shifted_target[core], valid_patch[core] & shifted_valid[core],
        )
        if global_score is None or local_score is None:
            continue
        scores.append(HeldoutCellScore(
            row=held_out.row, col=held_out.col, channel=channel,
            global_score=global_score, local_score=local_score,
            improvement=local_score - global_score,
        ))

    if len(scores) < config.min_holdout_cells:
        return HeldoutFieldValidation(
            tuple(scores), len(scores), None, None, None, False,
            f"Only {len(scores)} held-out cells produced valid image-domain scores.",
        )
    improvements = np.asarray([score.improvement for score in scores], dtype=np.float64)
    mean = float(np.mean(improvements))
    median = float(np.median(improvements))
    win_fraction = float(np.mean(improvements > 0.0))
    accepted_result = (
        mean >= config.min_holdout_improvement
        and median >= 0.0
        and win_fraction >= config.min_holdout_win_fraction
    )
    reason = (
        "Held-out field verification passed."
        if accepted_result else
        "Held-out field verification did not meet improvement, median, or win-fraction gates."
    )
    return HeldoutFieldValidation(tuple(scores), len(scores), mean, median, win_fraction, accepted_result, reason)


def compute_cell_displacements(
    rgb_bands: dict[str, np.ndarray],
    ms_bands: dict[str, np.ndarray],
    rgb_mask: np.ndarray,
    ms_mask: np.ndarray,
    global_matrix: np.ndarray,
    channel_priority: list[str],
    config: CellCorrelationConfig,
) -> SparseDisplacementResult:
    """Return one guarded local MS-to-RGB residual sample for every grid cell."""
    if global_matrix.shape != (2, 3) or not np.isfinite(global_matrix).all():
        raise ValueError("Cell correlation requires a finite 2x3 global affine matrix.")
    if rgb_mask.ndim != 2 or ms_mask.ndim != 2 or rgb_mask.shape != ms_mask.shape:
        raise ValueError("RGB and MS validity masks must be equally shaped 2-D arrays.")
    shape = rgb_mask.shape
    channels = [item.value if hasattr(item, "value") else str(item) for item in channel_priority]
    if not channels:
        raise ValueError("At least one registration channel is required for cell correlation.")
    for channel in channels:
        if channel not in rgb_bands or channel not in ms_bands:
            raise ValueError(f"Missing requested registration channel: {channel}")
        if rgb_bands[channel].shape != shape or ms_bands[channel].shape != shape:
            raise ValueError("Registration bands must share the validity-mask shape.")

    height, width = shape
    warped_ms_mask = cv2.warpAffine(
        ms_mask.astype(np.uint8), global_matrix.astype(np.float64), (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    common_valid = rgb_mask.astype(bool) & warped_ms_mask
    warped_ms = {
        channel: cv2.warpAffine(
            ms_bands[channel].astype(np.float32), global_matrix.astype(np.float64), (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        for channel in channels
    }

    samples: list[LocalMatchSample] = []
    for row in range(config.grid_rows):
        for col in range(config.grid_cols):
            bounds = _cell_bounds(shape, row, col, config)
            channel_samples = [
                _channel_sample(
                    row, col, bounds, rgb_bands[channel], warped_ms[channel], common_valid,
                    global_matrix, channel, config,
                )
                for channel in channels
            ]
            samples.append(_select_channel_sample(channel_samples, config))
    samples = _apply_spatial_filter(samples, config)
    accepted = [sample for sample in samples if sample.status == CellMatchStatus.ACCEPTED]
    translation = (float(global_matrix[0, 2]), float(global_matrix[1, 2]))
    return SparseDisplacementResult(
        samples=tuple(samples),
        rgb_features=None,
        ms_features=None,
        global_translation_px=translation,
        trusted_count=len(accepted),
        coverage=_spatial_coverage(accepted, shape),
    )
