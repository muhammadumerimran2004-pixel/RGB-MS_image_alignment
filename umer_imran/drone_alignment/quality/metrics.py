from dataclasses import dataclass, asdict
import numpy as np

from drone_alignment.config.schema import QualityConfig, TransformValidationConfig


@dataclass
class SpatialResidualReport:
    global_rmse_px: float | None
    center_rmse_px: float | None
    edge_corner_rmse_px: float | None
    max_residual_px: float | None
    residual_drift_ratio: float | None  # edge_rmse / (center_rmse + 1e-6)
    grid_residuals: list[dict]
    status: str  # "PASS", "WARNING", "FAIL"
    is_spatial_drift_acceptable: bool
    holdout_agreement_ratio: float | None = None
    gross_mismatch_count: int = 0


@dataclass
class FootprintMetrics:
    retained_source_valid_ratio: float
    reference_overlap_ratio: float
    target_overlap_ratio: float
    overlap_coefficient: float
    intersection_over_union: float


def footprint_gate_failures(footprint: FootprintMetrics, transform_cfg: TransformValidationConfig) -> list[str]:
    """Reasons a footprint fails: only whether the correction pushed the MS off the grid.

    Mutual coverage (reference/target overlap ratios) is reported but never gated. The
    RGB and MS flights routinely cover different ground (the MS often only the centre of
    the RGB), so coverage says nothing about whether the transform is right; residual QA
    and AROSICS verification judge accuracy on whatever ground the two do share.
    """
    failures = []
    if footprint.retained_source_valid_ratio < transform_cfg.min_retained_valid_ratio:
        failures.append(
            f"retained valid footprint {footprint.retained_source_valid_ratio:.2%} is below "
            f"{transform_cfg.min_retained_valid_ratio:.2%}"
        )
    return failures


def evaluate_footprint(
    rgb_valid_mask: np.ndarray,
    ms_valid_mask: np.ndarray,
    matrix: np.ndarray,
    transform_type,
) -> FootprintMetrics:
    """Measure clipping and mutual coverage after applying a candidate transform to the MS mask."""
    import cv2
    h, w = rgb_valid_mask.shape
    source_count = int(np.count_nonzero(ms_valid_mask))
    reference_count = int(np.count_nonzero(rgb_valid_mask))
    if source_count == 0 or reference_count == 0:
        raise ValueError("Cannot calculate footprint metrics for an empty valid mask.")
    source_u8 = ms_valid_mask.astype(np.uint8)
    if matrix.shape == (2, 3):
        warped = cv2.warpAffine(source_u8, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    else:
        warped = cv2.warpPerspective(source_u8, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    after_count = int(np.count_nonzero(warped))
    intersection = int(np.count_nonzero(warped & rgb_valid_mask))
    union = int(np.count_nonzero(warped | rgb_valid_mask))
    return FootprintMetrics(
        retained_source_valid_ratio=after_count / source_count,
        reference_overlap_ratio=intersection / reference_count,
        target_overlap_ratio=intersection / max(1, after_count),
        overlap_coefficient=intersection / max(1, min(reference_count, after_count)),
        intersection_over_union=intersection / max(1, union),
    )


def evaluate_spatial_residuals(
    pts_rgb: np.ndarray,
    pts_ms_transformed: np.ndarray,
    image_shape: tuple[int, int],
    config: QualityConfig,
    screen_false_matches: bool = False,
) -> SpatialResidualReport:
    """
    Evaluates spatially-partitioned residual errors across a grid mesh.
    Distinguishes center region errors from edge/corner region errors to catch edge drift.

    ``screen_false_matches`` is for held-out *automated* feature matches only. Those
    are raw descriptor matches that RANSAC never filtered, and between RGB and MS
    bands a few of them are wrong by hundreds of pixels; one such match can inflate
    RMSE by two orders of magnitude and reject a transform that fits every other
    point. They are excluded from the RMSE gates but still count against
    ``min_holdout_agreement``. Never enable it for manual control points, where a
    large residual is a real error that must fail.
    """
    h, w = image_shape
    center_min_x = 0.25 * w
    center_max_x = 0.75 * w
    center_min_y = 0.25 * h
    center_max_y = 0.75 * h

    center_errors = []
    edge_errors = []
    grid_residuals = []
    gross_mismatch_count = 0

    for (x_ref, y_ref), (x_warped, y_warped) in zip(pts_rgb, pts_ms_transformed):
        err = float(np.sqrt((x_ref - x_warped) ** 2 + (y_ref - y_warped) ** 2))
        is_center = (center_min_x <= x_ref <= center_max_x) and (center_min_y <= y_ref <= center_max_y)
        region = "center" if is_center else "edge"
        is_gross = screen_false_matches and err > config.holdout_gross_mismatch_px

        if is_gross:
            gross_mismatch_count += 1
        elif is_center:
            center_errors.append(err)
        else:
            edge_errors.append(err)

        grid_residuals.append({
            "x": float(x_ref),
            "y": float(y_ref),
            "region": region,
            "error_px": float(err),
            "gross_mismatch": is_gross,
        })

    total_points = len(grid_residuals)
    all_errors = center_errors + edge_errors
    agreement = (len(all_errors) / total_points) if screen_false_matches and total_points else None
    if screen_false_matches and total_points and agreement < config.min_holdout_agreement:
        return SpatialResidualReport(
            global_rmse_px=float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else None,
            center_rmse_px=None,
            edge_corner_rmse_px=None,
            max_residual_px=float(np.max(all_errors)) if all_errors else None,
            residual_drift_ratio=None,
            grid_residuals=grid_residuals,
            status="FAIL",
            is_spatial_drift_acceptable=False,
            holdout_agreement_ratio=agreement,
            gross_mismatch_count=gross_mismatch_count,
        )
    if len(all_errors) == 0:
        return SpatialResidualReport(
            global_rmse_px=0.0,
            center_rmse_px=0.0,
            edge_corner_rmse_px=0.0,
            max_residual_px=0.0,
            residual_drift_ratio=1.0,
            grid_residuals=[],
            status="UNVERIFIED",
            is_spatial_drift_acceptable=False,
        )

    center_rmse = float(np.sqrt(np.mean(np.square(center_errors)))) if center_errors else 0.0
    edge_rmse = float(np.sqrt(np.mean(np.square(edge_errors)))) if edge_errors else 0.0
    global_rmse = float(np.sqrt(np.mean(np.square(all_errors))))
    max_residual = float(np.max(all_errors))
    drift_ratio = edge_rmse / (center_rmse + 1e-6)

    acceptable = (edge_rmse <= config.max_acceptable_edge_rmse_px) and (global_rmse <= config.max_acceptable_rmse_px)

    if global_rmse <= config.max_acceptable_rmse_px and edge_rmse <= config.max_acceptable_edge_rmse_px:
        status = "PASS"
    elif global_rmse <= config.max_acceptable_rmse_px * 1.5:
        status = "WARNING"
    else:
        status = "FAIL"

    return SpatialResidualReport(
        global_rmse_px=global_rmse,
        center_rmse_px=center_rmse,
        edge_corner_rmse_px=edge_rmse,
        max_residual_px=max_residual,
        residual_drift_ratio=drift_ratio,
        grid_residuals=grid_residuals,
        status=status,
        is_spatial_drift_acceptable=acceptable,
        holdout_agreement_ratio=agreement,
        gross_mismatch_count=gross_mismatch_count,
    )
