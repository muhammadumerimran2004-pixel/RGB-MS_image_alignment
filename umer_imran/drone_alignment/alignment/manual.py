from dataclasses import replace
import math
import numpy as np

from drone_alignment.config.schema import TransformValidationConfig, TransformType
from drone_alignment.alignment.transform_estimator import TransformResult, TransformUnreliableError


def compute_manual_translation(
    pts_rgb: np.ndarray | list[tuple[float, float]],
    pts_ms: np.ndarray | list[tuple[float, float]],
    target_gsd: float,
    config: TransformValidationConfig,
) -> TransformResult:
    """
    Computes a translation transformation matrix mapping MS -> RGB from user-provided manual control points.

    Args:
        pts_rgb: Points (x, y) in RGB coordinate space.
        pts_ms: Corresponding points (x, y) in MS coordinate space.
        target_gsd: Resolution in meters per pixel.
        config: Transform validation thresholds.

    Returns:
        TransformResult containing the 2x3 affine matrix and translation metrics.
    """
    arr_rgb = np.asarray(pts_rgb, dtype=np.float64)
    arr_ms = np.asarray(pts_ms, dtype=np.float64)

    if len(arr_rgb) == 0 or len(arr_ms) == 0:
        raise TransformUnreliableError("At least one manual control point pair is required.")
    if len(arr_rgb) != len(arr_ms):
        raise TransformUnreliableError("Number of RGB points must match number of MS points.")

    diffs = arr_rgb - arr_ms
    tx_px = float(np.mean(diffs[:, 0]))
    ty_px = float(np.mean(diffs[:, 1]))

    tx_m = tx_px * target_gsd
    ty_m = ty_px * target_gsd
    magnitude_m = math.hypot(tx_m, ty_m)

    if magnitude_m > config.max_translation_m:
        raise TransformUnreliableError(
            f"Manual translation ({magnitude_m:.2f}m) exceeds maximum allowed threshold ({config.max_translation_m:.2f}m)."
        )

    matrix = np.array([[1.0, 0.0, tx_px], [0.0, 1.0, ty_px]], dtype=np.float64)
    num_pts = len(arr_rgb)

    return TransformResult(
        matrix=matrix,
        transform_type=TransformType.AFFINE,
        translation_px=(tx_px, ty_px),
        translation_m=(tx_m, ty_m),
        rotation_deg=0.0,
        scale=(1.0, 1.0),
        inlier_ratio=1.0,
        num_inliers=num_pts,
        num_total_matches=num_pts,
        method="manual",
        channel_pair="manual",
    )
