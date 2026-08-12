from dataclasses import dataclass, asdict
import numpy as np

from drone_alignment.config.schema import QualityConfig


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


@dataclass
class FootprintMetrics:
    retained_source_valid_ratio: float
    reference_overlap_ratio: float
    target_overlap_ratio: float
    overlap_coefficient: float
    intersection_over_union: float


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
) -> SpatialResidualReport:
    """
    Evaluates spatially-partitioned residual errors across a grid mesh.
    Distinguishes center region errors from edge/corner region errors to catch edge drift.
    """
    h, w = image_shape
    center_min_x = 0.25 * w
    center_max_x = 0.75 * w
    center_min_y = 0.25 * h
    center_max_y = 0.75 * h

    center_errors = []
    edge_errors = []
    grid_residuals = []

    for (x_ref, y_ref), (x_warped, y_warped) in zip(pts_rgb, pts_ms_transformed):
        err = float(np.sqrt((x_ref - x_warped) ** 2 + (y_ref - y_warped) ** 2))
        is_center = (center_min_x <= x_ref <= center_max_x) and (center_min_y <= y_ref <= center_max_y)

        region = "center" if is_center else "edge"
        if is_center:
            center_errors.append(err)
        else:
            edge_errors.append(err)

        grid_residuals.append({
            "x": float(x_ref),
            "y": float(y_ref),
            "region": region,
            "error_px": float(err),
        })

    all_errors = center_errors + edge_errors
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
    )
