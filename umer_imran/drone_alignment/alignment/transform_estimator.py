from dataclasses import dataclass
import math
import cv2
import numpy as np

from drone_alignment.config.schema import (
    TransformValidationConfig,
    TransformType,
    FeatureDetectionConfig,
)
from drone_alignment.alignment.feature_matcher import MatchResult


class TransformUnreliableError(ValueError):
    """Raised when the estimated spatial transform violates sanity bounds."""
    pass


@dataclass(frozen=True)
class TransformResult:
    matrix: np.ndarray             # 2x3 (Affine) or 3x3 (Homography) matrix in target resolution space
    transform_type: TransformType
    translation_px: tuple[float, float]  # (tx, ty) in pixels
    translation_m: tuple[float, float]   # (tx_m, ty_m) in meters
    rotation_deg: float
    scale: tuple[float, float]           # (sx, sy)
    inlier_ratio: float
    num_inliers: int
    num_total_matches: int
    method: str = "feature"
    channel_pair: str = "unknown"
    phase_response: float | None = None
    retained_valid_ratio: float | None = None
    reference_overlap_ratio: float | None = None


def _fallback_phase_correlation(
    rgb_red: np.ndarray,
    ms_red: np.ndarray,
    valid_mask: np.ndarray | None = None,
    return_response: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Translation-only fallback using phase correlation in frequency domain."""
    # Convert to float64 for cv2.phaseCorrelate
    valid = (np.isfinite(rgb_red) & np.isfinite(ms_red) if valid_mask is None
             else valid_mask.astype(bool) & np.isfinite(rgb_red) & np.isfinite(ms_red))
    yy, xx = np.where(valid)
    if len(xx) < 16:
        raise TransformUnreliableError("Insufficient common valid data for phase correlation.")
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    crop_mask = valid[y0:y1, x0:x1]
    r64 = rgb_red[y0:y1, x0:x1].astype(np.float64)
    m64 = ms_red[y0:y1, x0:x1].astype(np.float64)
    r64 = np.where(crop_mask, r64 - np.mean(r64[crop_mask]), 0.0)
    m64 = np.where(crop_mask, m64 - np.mean(m64[crop_mask]), 0.0)

    # Apply Hann window to reduce edge boundary effects
    h, w = r64.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_64F)
    
    (dx, dy), response = cv2.phaseCorrelate(r64, m64, window=hann)
    # Note: cv2.phaseCorrelate returns shift of ms relative to rgb
    matrix = np.float64([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return (matrix, float(response)) if return_response else matrix


def validate_phase_transform(
    matrix: np.ndarray,
    response: float,
    target_gsd: float,
    config: TransformValidationConfig,
) -> TransformResult:
    """Apply the same hard translation gate to a phase candidate and preserve its evidence."""
    tx_px, ty_px = float(matrix[0, 2]), float(matrix[1, 2])
    tx_m, ty_m = tx_px * target_gsd, ty_px * target_gsd
    magnitude = math.hypot(tx_m, ty_m)
    if response < config.min_phase_correlation_response:
        raise TransformUnreliableError(
            f"Phase correlation response ({response:.3f}) is below minimum "
            f"({config.min_phase_correlation_response:.3f})."
        )
    if magnitude > config.max_translation_m:
        raise TransformUnreliableError(
            f"Estimated translation ({magnitude:.2f}m) exceeds max allowed threshold "
            f"({config.max_translation_m:.2f}m)."
        )
    return TransformResult(
        matrix=matrix, transform_type=TransformType.AFFINE,
        translation_px=(tx_px, ty_px), translation_m=(tx_m, ty_m),
        rotation_deg=0.0, scale=(1.0, 1.0), inlier_ratio=0.0,
        num_inliers=0, num_total_matches=0, method="phase_correlation",
        phase_response=response,
    )


def estimate_transform(
    match_result: MatchResult,
    scale_factor: float,
    target_gsd: float,
    config: TransformValidationConfig,
    feature_config: FeatureDetectionConfig,
) -> TransformResult:
    """
    Estimate and validate spatial transformation matrix mapping MS -> RGB.
    """
    pts_ms = match_result.pts_ms
    pts_rgb = match_result.pts_rgb

    if config.transform_type == TransformType.AFFINE:
        matrix, inlier_mask = cv2.estimateAffine2D(
            pts_ms,
            pts_rgb,
            method=cv2.RANSAC,
            ransacReprojThreshold=feature_config.ransac_reproj_threshold,
            maxIters=feature_config.ransac_max_iters,
            confidence=feature_config.ransac_confidence,
        )
    else:  # HOMOGRAPHY
        matrix, inlier_mask = cv2.findHomography(
            pts_ms,
            pts_rgb,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=feature_config.ransac_reproj_threshold,
            maxIters=feature_config.ransac_max_iters,
            confidence=feature_config.ransac_confidence,
        )

    if matrix is None:
        raise TransformUnreliableError("RANSAC transform estimation failed to converge.")

    inlier_mask = inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.ones(len(pts_ms), dtype=bool)
    num_inliers = int(np.sum(inlier_mask))
    total_matches = len(pts_ms)
    inlier_ratio = num_inliers / max(1, total_matches)

    if inlier_ratio < config.min_inlier_ratio:
        raise TransformUnreliableError(
            f"RANSAC inlier ratio ({inlier_ratio:.2%}) is below configured minimum "
            f"threshold ({config.min_inlier_ratio:.2%})."
        )

    # Convert transform matrix from downsampled pixel space to target pixel resolution space
    full_matrix = matrix.copy()
    if scale_factor != 1.0:
        inv_scale = 1.0 / scale_factor
        full_matrix[0, 2] *= inv_scale
        full_matrix[1, 2] *= inv_scale

    # Decompose Affine matrix components (for 2x3 or upper 2x3 of 3x3)
    a = full_matrix[0, 0]
    b = full_matrix[0, 1]
    c = full_matrix[1, 0]
    d = full_matrix[1, 1]
    tx_px = float(full_matrix[0, 2])
    ty_px = float(full_matrix[1, 2])

    sx = math.sqrt(a * a + c * c)
    sy = math.sqrt(b * b + d * d)
    rotation_rad = math.atan2(c, a)
    rotation_deg = math.degrees(rotation_rad)

    tx_m = tx_px * target_gsd
    ty_m = ty_px * target_gsd
    translation_m_magnitude = math.sqrt(tx_m * tx_m + ty_m * ty_m)

    # Sanity checks
    if translation_m_magnitude > config.max_translation_m:
        raise TransformUnreliableError(
            f"Estimated translation ({translation_m_magnitude:.2f}m) exceeds max allowed "
            f"threshold ({config.max_translation_m:.2f}m)."
        )

    if abs(rotation_deg) > config.max_rotation_deg:
        raise TransformUnreliableError(
            f"Estimated rotation ({rotation_deg:.2f}°) exceeds max allowed "
            f"threshold ({config.max_rotation_deg:.2f}°)."
        )

    if abs(sx - 1.0) > config.max_scale_deviation or abs(sy - 1.0) > config.max_scale_deviation:
        raise TransformUnreliableError(
            f"Estimated scale ({sx:.3f}, {sy:.3f}) deviates beyond max allowed "
            f"threshold deviation ({config.max_scale_deviation})."
        )

    return TransformResult(
        matrix=full_matrix,
        transform_type=config.transform_type,
        translation_px=(tx_px, ty_px),
        translation_m=(tx_m, ty_m),
        rotation_deg=rotation_deg,
        scale=(sx, sy),
        inlier_ratio=inlier_ratio,
        num_inliers=num_inliers,
        num_total_matches=total_matches,
    )
