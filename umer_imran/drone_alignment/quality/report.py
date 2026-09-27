from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from drone_alignment.version import __version__
from drone_alignment.config.schema import AlignmentConfig
from drone_alignment.io.reader import RasterMetadata
from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.quality.metrics import SpatialResidualReport, FootprintMetrics


def write_alignment_report(
    report_path: Path,
    rgb_meta: RasterMetadata,
    ms_meta: RasterMetadata,
    transform_result: TransformResult,
    quality_report: SpatialResidualReport,
    config: AlignmentConfig,
    aligned_ms_path: Path,
    preview_image_path: Path,
    footprint_metrics: FootprintMetrics | None = None,
    rejected_candidates: list[str] | None = None,
    *,
    requested_alignment_mode: str | None = None,
    applied_alignment_mode: str | None = None,
    fallback: dict | None = None,
    local_correlation: dict | None = None,
) -> Path:
    """
    Serializes a complete, machine-readable alignment report to JSON.
    """
    report_path_obj = Path(report_path).resolve()
    report_path_obj.parent.mkdir(parents=True, exist_ok=True)

    requested_mode = requested_alignment_mode or str(config.alignment_mode.value)
    applied_mode = applied_alignment_mode or str(config.alignment_mode.value)
    report_data = {
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alignment_mode": applied_mode,
        "requested_alignment_mode": requested_mode,
        "applied_alignment_mode": applied_mode,
        "fallback": fallback,
        "local_correlation": local_correlation,
        "inputs": {
            "rgb_path": str(rgb_meta.path),
            "ms_path": str(ms_meta.path),
            "rgb_crs": str(rgb_meta.crs),
            "ms_crs": str(ms_meta.crs),
            "rgb_gsd_m": float(rgb_meta.gsd),
            "ms_gsd_m": float(ms_meta.gsd),
            "rgb_dimensions": [rgb_meta.width, rgb_meta.height],
            "ms_dimensions": [ms_meta.width, ms_meta.height],
        },
        "transform": {
            "transform_type": str(transform_result.transform_type),
            "matrix": transform_result.matrix.tolist(),
            "translation_px": list(transform_result.translation_px),
            "translation_m": list(transform_result.translation_m),
            "rotation_deg": float(transform_result.rotation_deg),
            "scale": list(transform_result.scale),
            "inlier_ratio": float(transform_result.inlier_ratio),
            "num_inliers": int(transform_result.num_inliers),
            "num_total_matches": int(transform_result.num_total_matches),
            "method": transform_result.method,
            "channel_pair": transform_result.channel_pair,
            "phase_response": transform_result.phase_response,
        },
        "footprint": asdict(footprint_metrics) if footprint_metrics is not None else None,
        "rejected_candidates": rejected_candidates or [],
        "quality": {
            "status": quality_report.status,
            "global_rmse_px": quality_report.global_rmse_px,
            "center_rmse_px": quality_report.center_rmse_px,
            "edge_corner_rmse_px": quality_report.edge_corner_rmse_px,
            "max_residual_px": quality_report.max_residual_px,
            "residual_drift_ratio": quality_report.residual_drift_ratio,
            "is_spatial_drift_acceptable": quality_report.is_spatial_drift_acceptable,
            "grid_residuals_count": len(quality_report.grid_residuals),
        },
        "outputs": {
            "aligned_ms_path": str(aligned_ms_path),
            "preview_image_path": str(preview_image_path),
            "report_json_path": str(report_path_obj),
        },
        "config": config.model_dump(),
    }

    with open(report_path_obj, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_path_obj
