from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import rasterio

from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, DetectorType, TransformType
from drone_alignment.io.validators import validate_inputs
from drone_alignment.io.reader import read_metadata
from drone_alignment.alignment.coarse import coarse_align
from drone_alignment.alignment.manual import compute_manual_translation
from drone_alignment.alignment.feature_detector import (
    detect_features, normalize_to_uint8, InsufficientValidDataError, InsufficientContrastError,
)
from drone_alignment.alignment.feature_matcher import match_features, InsufficientMatchesError, MatchResult
from drone_alignment.alignment.transform_estimator import (
    estimate_transform, _fallback_phase_correlation, validate_phase_transform,
    TransformResult, TransformUnreliableError,
)
from drone_alignment.alignment.warper import warp_ms_to_rgb_tiled, evaluate_native_footprint
from drone_alignment.quality.metrics import (
    evaluate_spatial_residuals, evaluate_footprint, SpatialResidualReport, FootprintMetrics,
)
from drone_alignment.quality.visualization import generate_alignment_preview
from drone_alignment.quality.report import write_alignment_report

logger = logging.getLogger("drone_alignment")


@dataclass
class AlignmentResult:
    aligned_ms_path: Path
    report_json_path: Path
    preview_image_path: Path
    spatial_residual_report: SpatialResidualReport
    transform_result: TransformResult


def _validate_candidate_footprint(transform, coarse, cfg) -> tuple[TransformResult, FootprintMetrics]:
    footprint = evaluate_footprint(
        coarse.rgb_valid_mask, coarse.ms_valid_mask, transform.matrix, transform.transform_type
    )
    failures = []
    if footprint.retained_source_valid_ratio < cfg.transform.min_retained_valid_ratio:
        failures.append(
            f"retained valid footprint {footprint.retained_source_valid_ratio:.2%} is below "
            f"{cfg.transform.min_retained_valid_ratio:.2%}"
        )
    if footprint.reference_overlap_ratio < cfg.transform.min_reference_overlap_ratio:
        failures.append(
            f"reference overlap {footprint.reference_overlap_ratio:.2%} is below "
            f"{cfg.transform.min_reference_overlap_ratio:.2%}"
        )
    if failures:
        raise TransformUnreliableError("; ".join(failures))
    return replace(
        transform,
        retained_valid_ratio=footprint.retained_source_valid_ratio,
        reference_overlap_ratio=footprint.reference_overlap_ratio,
    ), footprint


def _inlier_coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull)) / float(shape[0] * shape[1])


def _split_estimation_and_verification(matches: MatchResult) -> tuple[MatchResult, np.ndarray, np.ndarray]:
    """Reserve deterministic correspondences that RANSAC never sees for post-fit verification."""
    count = len(matches.pts_rgb)
    verify_count = max(4, count // 5)
    if count - verify_count < 4:
        raise InsufficientMatchesError("Not enough matches to reserve independent verification points.")
    indices = np.arange(count)
    verify_idx = indices[::max(1, count // verify_count)][:verify_count]
    train_mask = np.ones(count, dtype=bool)
    train_mask[verify_idx] = False
    train_rgb, train_ms = matches.pts_rgb[train_mask], matches.pts_ms[train_mask]
    training = MatchResult(
        pts_rgb=train_rgb, pts_ms=train_ms, num_raw_matches=matches.num_raw_matches,
        num_good_matches=len(train_rgb), match_quality_score=matches.match_quality_score,
    )
    return training, matches.pts_rgb[verify_idx], matches.pts_ms[verify_idx]


def _masked_correlation(reference: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    values_a = reference[valid].astype(np.float64)
    values_b = target[valid].astype(np.float64)
    if values_a.size < 16 or np.std(values_a) == 0 or np.std(values_b) == 0:
        return -1.0
    return float(np.corrcoef(values_a, values_b)[0, 1])


def _to_native_transform(transform: TransformResult, coarse) -> TransformResult:
    """Conjugate a registration-grid affine into native output pixel coordinates."""
    profile = coarse.output_profile or coarse.target_profile
    reg_h, reg_w = coarse.rgb_valid_mask.shape
    sx, sy = profile["width"] / reg_w, profile["height"] / reg_h
    low = np.vstack([transform.matrix, [0.0, 0.0, 1.0]]) if transform.matrix.shape == (2, 3) else transform.matrix
    native_to_reg = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float64)
    native = np.linalg.inv(native_to_reg) @ low @ native_to_reg
    matrix = native[:2, :] if transform.matrix.shape == (2, 3) else native
    tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
    return replace(
        transform, matrix=matrix, translation_px=(tx, ty),
        translation_m=(tx * coarse.target_gsd, ty * coarse.target_gsd),
    )


def align_orthomosaics(
    rgb_path: Path,
    ms_path: Path,
    output_dir: Path,
    config: Optional[AlignmentConfig] = None,
    log: Optional[logging.Logger] = None,
) -> AlignmentResult:
    active_log = log or logger
    cfg = config or AlignmentConfig()
    output_dir_obj = Path(output_dir).resolve()
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    rgb_meta, ms_meta = validate_inputs(rgb_path, ms_path, cfg)
    coarse = coarse_align(rgb_meta, ms_meta, cfg)
    common_mask = coarse.rgb_valid_mask & coarse.ms_valid_mask
    active_log.info(
        "Valid footprints: RGB=%.2f%%, MS=%.2f%%, common=%.2f%%",
        100 * coarse.rgb_valid_mask.mean(), 100 * coarse.ms_valid_mask.mean(), 100 * common_mask.mean(),
    )

    accepted = []
    rejection_reasons: list[str] = []
    expected_errors = (
        InsufficientMatchesError, TransformUnreliableError,
        InsufficientValidDataError, InsufficientContrastError,
    )
    detectors = [cfg.features.detector]
    if cfg.features.detector != DetectorType.SIFT:
        detectors.append(DetectorType.SIFT)

    for channel_enum in cfg.registration_channel_priority:
        channel = channel_enum.value
        for detector in detectors:
            attempt = f"{channel}-{channel}/{detector.value}"
            try:
                feature_cfg = cfg.features.model_copy(update={"detector": detector})
                rgb_det, ms_det = detect_features(
                    coarse.registration_bands_rgb[channel], coarse.registration_bands_ms[channel],
                    feature_cfg, coarse.rgb_valid_mask, coarse.ms_valid_mask,
                )
                matches = match_features(rgb_det, ms_det, feature_cfg)
                training_matches, verify_rgb_ds, verify_ms_ds = _split_estimation_and_verification(matches)
                transform = estimate_transform(
                    training_matches, rgb_det.scale_factor, coarse.registration_gsd, cfg.transform, feature_cfg
                )
                train_rgb = training_matches.pts_rgb / rgb_det.scale_factor
                coverage = _inlier_coverage(train_rgb, coarse.rgb_valid_mask.shape)
                if coverage < cfg.transform.min_inlier_coverage_ratio:
                    raise TransformUnreliableError(
                        f"inlier coverage ({coverage:.2%}) is below minimum "
                        f"({cfg.transform.min_inlier_coverage_ratio:.2%})"
                    )
                transform = replace(transform, method=detector.value, channel_pair=f"{channel}-{channel}")
                transform, footprint = _validate_candidate_footprint(transform, coarse, cfg)
                pts_rgb = verify_rgb_ds / rgb_det.scale_factor
                pts_ms = verify_ms_ds / ms_det.scale_factor
                if transform.matrix.shape == (2, 3):
                    transformed = cv2.transform(pts_ms.reshape(-1, 1, 2), transform.matrix).reshape(-1, 2)
                else:
                    transformed = cv2.perspectiveTransform(pts_ms.reshape(-1, 1, 2), transform.matrix).reshape(-1, 2)
                quality = evaluate_spatial_residuals(pts_rgb, transformed, coarse.rgb_valid_mask.shape, cfg.quality)
                if quality.status != "PASS":
                    raise TransformUnreliableError(f"independent/residual QA status is {quality.status}")
                accepted.append((transform, footprint, quality, pts_rgb, transformed))
                active_log.info("Accepted candidate %s with %d/%d inliers", attempt, transform.num_inliers, transform.num_total_matches)
            except expected_errors as exc:
                reason = f"{attempt}: {exc}"
                rejection_reasons.append(reason)
                active_log.warning("Rejected candidate %s", reason)

    if accepted:
        accepted.sort(key=lambda item: (item[0].inlier_ratio, item[0].num_inliers), reverse=True)
        transform_res, footprint, quality_rep, _, _ = accepted[0]
    else:
        # Translation-only fallback is fail-closed and still passes all hard transform/footprint gates.
        channel = cfg.registration_channel_priority[0].value
        try:
            rgb_pc = normalize_to_uint8(
                coarse.registration_bands_rgb[channel], coarse.rgb_valid_mask,
                cfg.features.percentile_low, cfg.features.percentile_high,
                cfg.features.apply_clahe, cfg.features.clahe_clip_limit,
                cfg.features.clahe_grid_size, cfg.features.min_valid_pixels,
            )
            ms_pc = normalize_to_uint8(
                coarse.registration_bands_ms[channel], coarse.ms_valid_mask,
                cfg.features.percentile_low, cfg.features.percentile_high,
                cfg.features.apply_clahe, cfg.features.clahe_clip_limit,
                cfg.features.clahe_grid_size, cfg.features.min_valid_pixels,
            )
            matrix, response = _fallback_phase_correlation(rgb_pc, ms_pc, common_mask, return_response=True)
            transform_res = validate_phase_transform(matrix, response, coarse.registration_gsd, cfg.transform)
            transform_res = replace(transform_res, channel_pair=f"{channel}-{channel}")
            transform_res, footprint = _validate_candidate_footprint(transform_res, coarse, cfg)
            h, w = rgb_pc.shape
            warped_pc = cv2.warpAffine(
                ms_pc, transform_res.matrix, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            warped_mask = cv2.warpAffine(
                coarse.ms_valid_mask.astype(np.uint8), transform_res.matrix, (w, h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ) > 0
            verification_score = _masked_correlation(
                rgb_pc, warped_pc, coarse.rgb_valid_mask & warped_mask
            )
            if verification_score < cfg.transform.min_phase_verification_correlation:
                raise TransformUnreliableError(
                    f"phase image-domain verification correlation ({verification_score:.3f}) is below "
                    f"minimum ({cfg.transform.min_phase_verification_correlation:.3f})"
                )
            quality_rep = SpatialResidualReport(
                global_rmse_px=None, center_rmse_px=None, edge_corner_rmse_px=None,
                max_residual_px=None, residual_drift_ratio=None, grid_residuals=[],
                status="PASS", is_spatial_drift_acceptable=True,
            )
            active_log.info(
                "Accepted phase candidate with response %.3f and verification correlation %.3f",
                response, verification_score,
            )
        except (TransformUnreliableError, InsufficientValidDataError, InsufficientContrastError) as exc:
            rejection_reasons.append(f"{channel}-{channel}/phase_correlation: {exc}")
            raise TransformUnreliableError("No verified alignment candidate. " + " | ".join(rejection_reasons))

    native_transform = _to_native_transform(transform_res, coarse)
    native_footprint = evaluate_native_footprint(
        rgb_meta, ms_meta, coarse, native_transform, cfg.warp.tile_size
    )
    if (native_footprint.retained_source_valid_ratio < cfg.transform.min_retained_valid_ratio or
            native_footprint.reference_overlap_ratio < cfg.transform.min_reference_overlap_ratio):
        raise TransformUnreliableError(
            "Native footprint confirmation failed: "
            f"retained={native_footprint.retained_source_valid_ratio:.2%}, "
            f"reference_overlap={native_footprint.reference_overlap_ratio:.2%}"
        )
    native_transform = replace(
        native_transform, retained_valid_ratio=native_footprint.retained_source_valid_ratio,
        reference_overlap_ratio=native_footprint.reference_overlap_ratio,
    )
    ms_stem = Path(ms_path).stem
    final_path = output_dir_obj / f"{ms_stem}_aligned.tif"
    staged_path = output_dir_obj / f".{ms_stem}_aligned.partial.tif"
    try:
        warp_ms_to_rgb_tiled(coarse, native_transform, staged_path, cfg.warp, ms_meta)
        os.replace(staged_path, final_path)
    finally:
        if staged_path.exists():
            staged_path.unlink()

    preview_path = output_dir_obj / f"{ms_stem}_alignment_preview.png"
    reference_band = coarse.registration_bands_rgb[native_transform.channel_pair.split("-")[0]]
    preview_band = cfg.ms_green_band_index if native_transform.channel_pair.startswith("green") else cfg.ms_red_band_index
    with rasterio.open(final_path) as aligned:
        warped_band = aligned.read(preview_band, out_shape=reference_band.shape, resampling=rasterio.enums.Resampling.average)
    generate_alignment_preview(
        reference_band, warped_band, preview_path,
        rgb_mask=coarse.rgb_valid_mask, max_dimension=cfg.quality.preview_max_dimension,
    )
    report_path = output_dir_obj / f"{ms_stem}_alignment_report.json"
    write_alignment_report(
        report_path, rgb_meta, ms_meta, transform_res, quality_rep, cfg,
        final_path, preview_path, footprint_metrics=native_footprint,
        rejected_candidates=rejection_reasons,
    )
    return AlignmentResult(final_path, report_path, preview_path, quality_rep, native_transform)


def manual_align_orthomosaics(
    rgb_path: Path,
    ms_path: Path,
    output_dir: Path,
    pts_rgb: np.ndarray | list[tuple[float, float]],
    pts_ms: np.ndarray | list[tuple[float, float]],
    config: Optional[AlignmentConfig] = None,
    log: Optional[logging.Logger] = None,
) -> AlignmentResult:
    """
    Manually aligns a Multispectral (MS) GeoTIFF to an RGB reference using user-supplied control point pairs.

    Args:
        rgb_path: Path to RGB reference GeoTIFF.
        ms_path: Path to MS target GeoTIFF.
        output_dir: Output directory.
        pts_rgb: RGB pixel coordinates (x, y) on the native grid.
        pts_ms: MS pixel coordinates (x, y) on the native grid.
        config: Alignment configuration.
        log: Optional logger instance.

    Returns:
        AlignmentResult with paths to aligned GeoTIFF, QA preview, and JSON report.
    """
    active_log = log or logger
    cfg = config or AlignmentConfig()
    cfg = cfg.model_copy(update={"alignment_mode": AlignmentMode.MANUAL})
    output_dir_obj = Path(output_dir).resolve()
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    rgb_meta, ms_meta = validate_inputs(rgb_path, ms_path, cfg)
    coarse = coarse_align(rgb_meta, ms_meta, cfg)

    arr_rgb = np.asarray(pts_rgb, dtype=np.float64)
    arr_ms = np.asarray(pts_ms, dtype=np.float64)

    reg_inv = ~coarse.registration_transform
    reg_pts_rgb_list = []
    reg_pts_ms_list = []

    for pt_rgb, pt_ms in zip(arr_rgb, arr_ms):
        # If X > 10,000, coordinates are already projected map coordinates (e.g. QGIS UTM Easting/Northing)
        if pt_rgb[0] > 10000.0:
            map_rgb = (float(pt_rgb[0]), float(pt_rgb[1]))
        else:
            map_rgb = rgb_meta.transform * (float(pt_rgb[0]), float(pt_rgb[1]))

        if pt_ms[0] > 10000.0:
            map_ms = (float(pt_ms[0]), float(pt_ms[1]))
        else:
            map_ms = ms_meta.transform * (float(pt_ms[0]), float(pt_ms[1]))

        reg_pts_rgb_list.append(reg_inv * map_rgb)
        reg_pts_ms_list.append(reg_inv * map_ms)

    reg_pts_rgb = np.array(reg_pts_rgb_list, dtype=np.float64)
    reg_pts_ms = np.array(reg_pts_ms_list, dtype=np.float64)

    transform_res = compute_manual_translation(
        reg_pts_rgb, reg_pts_ms, coarse.registration_gsd, cfg.transform
    )
    active_log.info(
        "Manual translation (registration grid): tx=%.2f px, ty=%.2f px",
        transform_res.translation_px[0], transform_res.translation_px[1],
    )
    active_log.info(
        "Manual translation (metres): tx=%.4f m, ty=%.4f m",
        transform_res.translation_m[0], transform_res.translation_m[1],
    )
    transform_res, footprint = _validate_candidate_footprint(transform_res, coarse, cfg)

    if len(arr_rgb) >= 3:
        transformed_ms = cv2.transform(reg_pts_ms.reshape(-1, 1, 2), transform_res.matrix).reshape(-1, 2)
        quality_rep = evaluate_spatial_residuals(reg_pts_rgb, transformed_ms, coarse.rgb_valid_mask.shape, cfg.quality)
    else:
        quality_rep = SpatialResidualReport(
            global_rmse_px=0.0, center_rmse_px=0.0, edge_corner_rmse_px=0.0,
            max_residual_px=0.0, residual_drift_ratio=0.0, grid_residuals=[],
            status="PASS", is_spatial_drift_acceptable=True,
        )

    native_transform = _to_native_transform(transform_res, coarse)
    native_footprint = evaluate_native_footprint(
        rgb_meta, ms_meta, coarse, native_transform, cfg.warp.tile_size
    )
    if (native_footprint.retained_source_valid_ratio < cfg.transform.min_retained_valid_ratio or
            native_footprint.reference_overlap_ratio < cfg.transform.min_reference_overlap_ratio):
        raise TransformUnreliableError(
            "Native footprint confirmation failed: "
            f"retained={native_footprint.retained_source_valid_ratio:.2%}, "
            f"reference_overlap={native_footprint.reference_overlap_ratio:.2%}"
        )
    native_transform = replace(
        native_transform, retained_valid_ratio=native_footprint.retained_source_valid_ratio,
        reference_overlap_ratio=native_footprint.reference_overlap_ratio,
    )

    ms_stem = Path(ms_path).stem
    final_path = output_dir_obj / f"{ms_stem}_aligned.tif"
    staged_path = output_dir_obj / f".{ms_stem}_aligned.partial.tif"
    try:
        warp_ms_to_rgb_tiled(coarse, native_transform, staged_path, cfg.warp, ms_meta)
        os.replace(staged_path, final_path)
    finally:
        if staged_path.exists():
            staged_path.unlink()

    preview_path = output_dir_obj / f"{ms_stem}_alignment_preview.png"
    channel = cfg.registration_channel_priority[0].value
    reference_band = coarse.registration_bands_rgb[channel]
    preview_band = cfg.ms_red_band_index if channel == "red" else cfg.ms_green_band_index
    with rasterio.open(final_path) as aligned:
        warped_band = aligned.read(preview_band, out_shape=reference_band.shape, resampling=rasterio.enums.Resampling.average)
    generate_alignment_preview(
        reference_band, warped_band, preview_path,
        rgb_mask=coarse.rgb_valid_mask, max_dimension=cfg.quality.preview_max_dimension,
    )
    report_path = output_dir_obj / f"{ms_stem}_alignment_report.json"
    write_alignment_report(
        report_path, rgb_meta, ms_meta, transform_res, quality_rep, cfg,
        final_path, preview_path, footprint_metrics=native_footprint,
        rejected_candidates=[],
    )
    active_log.info("Manual alignment successful with %d control point pair(s).", len(arr_rgb))
    return AlignmentResult(final_path, report_path, preview_path, quality_rep, native_transform)
