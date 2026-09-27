from collections import Counter
from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import rasterio

from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, DetectorType, ManualCoordinateMode, TransformType
from drone_alignment.io.validators import validate_inputs
from drone_alignment.io.reader import RasterMetadata
from drone_alignment.alignment.coarse import coarse_align, CoarseAlignmentResult
from drone_alignment.alignment.manual import compute_manual_translation
from drone_alignment.alignment.road_grid_aligner import (
    compute_road_grid_translation, InsufficientRoadFeatureError,
)
from drone_alignment.alignment.feature_detector import (
    detect_features, normalize_to_uint8, InsufficientValidDataError, InsufficientContrastError,
)
from drone_alignment.alignment.feature_matcher import match_features, InsufficientMatchesError, MatchResult
from drone_alignment.alignment.loftr_matcher import match_loftr
from drone_alignment.alignment.representations import build_representation
from drone_alignment.alignment.transform_estimator import (
    estimate_transform, _fallback_phase_correlation, validate_phase_transform,
    TransformResult, TransformUnreliableError,
)
from drone_alignment.alignment.warper import (
    evaluate_native_displacement_footprint, evaluate_native_footprint,
    warp_ms_to_rgb_tiled, warp_ms_with_displacement_field_tiled,
)
from drone_alignment.alignment.cell_correlator import (
    compute_cell_displacements, evaluate_heldout_field_improvement,
)
from drone_alignment.alignment.displacement_field import (
    compare_displacement_fields, fit_selected_displacement_field,
)
from drone_alignment.local_mesh_export import export_local_mesh_candidate
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


@dataclass(frozen=True)
class VerifiedGlobalContext:
    """One verified global candidate and all compact state needed to publish it.

    Local alignment modes must consume this object instead of estimating their
    own baseline.  This keeps the accepted global candidate, its QA result,
    and its rejection history identical whether the final publication is
    global or local.
    """

    rgb_meta: RasterMetadata
    ms_meta: RasterMetadata
    coarse: CoarseAlignmentResult
    registration_transform: TransformResult
    registration_footprint: FootprintMetrics
    quality_report: SpatialResidualReport
    rejected_candidates: tuple[str, ...]


class LocalCorrelationRejected(RuntimeError):
    """Expected local safety-gate rejection; global publication remains safe."""

    def __init__(self, reason_code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.reason_code = reason_code
        self.details = details or {}


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



def _estimate_global_candidate(
    coarse, cfg: AlignmentConfig, active_log: logging.Logger
) -> tuple[TransformResult, FootprintMetrics, SpatialResidualReport, list[str]]:
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

    # Learned matching is deliberately an optional second line.  It is useful
    # when visible-band descriptors are not repeatable across sensors, but it
    # never bypasses the exact same geometric and independent QA above.
    if not accepted and cfg.loftr.enabled:
        learned_candidate_accepted = False
        for channel_enum in cfg.registration_channel_priority:
            channel = channel_enum.value
            for representation in cfg.loftr.representations:
                attempt = f"{channel}-{channel}/loftr:{representation.value}"
                try:
                    rgb_representation = build_representation(
                        coarse.registration_bands_rgb[channel], coarse.rgb_valid_mask,
                        representation, cfg.features,
                    )
                    ms_representation = build_representation(
                        coarse.registration_bands_ms[channel], coarse.ms_valid_mask,
                        representation, cfg.features,
                    )
                    matches = match_loftr(
                        rgb_representation, ms_representation,
                        coarse.rgb_valid_mask, coarse.ms_valid_mask, cfg.loftr,
                    )
                    if matches.num_good_matches < cfg.features.min_good_matches:
                        raise InsufficientMatchesError(
                            f"LoFTR produced {matches.num_good_matches} matches; at least "
                            f"{cfg.features.min_good_matches} are required."
                        )
                    training_matches, verify_rgb, verify_ms = _split_estimation_and_verification(matches)
                    transform = estimate_transform(
                        training_matches, 1.0, coarse.registration_gsd, cfg.transform, cfg.features
                    )
                    coverage = _inlier_coverage(training_matches.pts_rgb, coarse.rgb_valid_mask.shape)
                    if coverage < cfg.transform.min_inlier_coverage_ratio:
                        raise TransformUnreliableError(
                            f"inlier coverage ({coverage:.2%}) is below minimum "
                            f"({cfg.transform.min_inlier_coverage_ratio:.2%})"
                        )
                    transform = replace(transform, method="loftr", channel_pair=f"{channel}-{channel}")
                    transform, footprint = _validate_candidate_footprint(transform, coarse, cfg)
                    if transform.matrix.shape == (2, 3):
                        transformed = cv2.transform(verify_ms.reshape(-1, 1, 2), transform.matrix).reshape(-1, 2)
                    else:
                        transformed = cv2.perspectiveTransform(verify_ms.reshape(-1, 1, 2), transform.matrix).reshape(-1, 2)
                    quality = evaluate_spatial_residuals(verify_rgb, transformed, coarse.rgb_valid_mask.shape, cfg.quality)
                    if quality.status != "PASS":
                        raise TransformUnreliableError(f"independent/residual QA status is {quality.status}")
                    accepted.append((transform, footprint, quality, verify_rgb, transformed))
                    active_log.info("Accepted candidate %s with %d/%d inliers", attempt, transform.num_inliers, transform.num_total_matches)
                    learned_candidate_accepted = True
                    break
                except expected_errors as exc:
                    reason = f"{attempt}: {exc}"
                    rejection_reasons.append(reason)
                    active_log.warning("Rejected candidate %s", reason)
            if learned_candidate_accepted:
                break

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

    return transform_res, footprint, quality_rep, rejection_reasons


def _estimate_verified_global_context(
    rgb_path: Path,
    ms_path: Path,
    cfg: AlignmentConfig,
    active_log: logging.Logger,
) -> VerifiedGlobalContext:
    """Validate inputs and produce one reusable, verified global baseline."""
    rgb_meta, ms_meta = validate_inputs(rgb_path, ms_path, cfg)
    coarse = coarse_align(rgb_meta, ms_meta, cfg)
    transform_res, footprint, quality_rep, rejection_reasons = _estimate_global_candidate(
        coarse, cfg, active_log
    )
    return VerifiedGlobalContext(
        rgb_meta=rgb_meta,
        ms_meta=ms_meta,
        coarse=coarse,
        registration_transform=transform_res,
        registration_footprint=footprint,
        quality_report=quality_rep,
        rejected_candidates=tuple(rejection_reasons),
    )


def _publish_global_context(
    context: VerifiedGlobalContext,
    output_dir_obj: Path,
    cfg: AlignmentConfig,
    active_log: logging.Logger,
    *,
    requested_alignment_mode: str | None = None,
    applied_alignment_mode: str | None = None,
    fallback: dict | None = None,
) -> AlignmentResult:
    rgb_meta, ms_meta = context.rgb_meta, context.ms_meta
    coarse = context.coarse
    transform_res = context.registration_transform
    quality_rep = context.quality_report
    rejection_reasons = list(context.rejected_candidates)
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
    ms_stem = ms_meta.path.stem
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
        requested_alignment_mode=requested_alignment_mode,
        applied_alignment_mode=applied_alignment_mode,
        fallback=fallback,
    )
    return AlignmentResult(final_path, report_path, preview_path, quality_rep, native_transform)


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
    context = _estimate_verified_global_context(rgb_path, ms_path, cfg, active_log)
    return _publish_global_context(context, output_dir_obj, cfg, active_log)


def local_correlation_align_orthomosaics(
    rgb_path: Path,
    ms_path: Path,
    output_dir: Path,
    config: Optional[AlignmentConfig] = None,
    log: Optional[logging.Logger] = None,
) -> AlignmentResult:
    """Publish a verified local displacement field, or the verified global baseline.

    Only :class:`LocalCorrelationRejected` triggers fallback.  Operational and
    programming errors intentionally propagate so a broken local path cannot
    be reported as a successful global run.
    """
    active_log = log or logger
    base_cfg = config or AlignmentConfig()
    cfg = base_cfg.model_copy(update={"alignment_mode": AlignmentMode.LOCAL_CORRELATION})
    output_dir_obj = Path(output_dir).resolve()
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    context = _estimate_verified_global_context(rgb_path, ms_path, cfg, active_log)
    requested_mode = AlignmentMode.LOCAL_CORRELATION.value

    try:
        transform = context.registration_transform
        if transform.matrix.shape != (2, 3):
            raise LocalCorrelationRejected(
                "UNSUPPORTED_GLOBAL_TRANSFORM",
                "Local correlation currently requires an affine 2x3 global transform.",
                {"matrix_shape": list(transform.matrix.shape)},
            )
        correlation_cfg = cfg.cell_correlation
        if not correlation_cfg.enabled:
            raise LocalCorrelationRejected("LOCAL_CORRELATION_DISABLED", "Cell correlation is disabled in configuration.")

        evidence = compute_cell_displacements(
            context.coarse.registration_bands_rgb,
            context.coarse.registration_bands_ms,
            context.coarse.rgb_valid_mask,
            context.coarse.ms_valid_mask,
            transform.matrix,
            list(cfg.registration_channel_priority),
            correlation_cfg,
        )
        accepted = evidence.accepted
        accepted_rows = {sample.row for sample in accepted}
        accepted_cols = {sample.col for sample in accepted}
        if evidence.trusted_count < correlation_cfg.min_trusted_cells:
            raise LocalCorrelationRejected(
                "INSUFFICIENT_TRUSTED_CELLS",
                "Cell correlation produced too few trusted local vectors.",
                {"trusted_count": evidence.trusted_count, "minimum": correlation_cfg.min_trusted_cells},
            )
        if evidence.coverage < correlation_cfg.min_spatial_coverage or len(accepted_rows) < 2 or len(accepted_cols) < 2:
            raise LocalCorrelationRejected(
                "INSUFFICIENT_SPATIAL_COVERAGE",
                "Trusted local vectors do not cover enough of the registration canvas.",
                {
                    "spatial_coverage": evidence.coverage,
                    "minimum": correlation_cfg.min_spatial_coverage,
                    "rows": len(accepted_rows),
                    "cols": len(accepted_cols),
                },
            )

        try:
            comparison = compare_displacement_fields(
                evidence, context.coarse.rgb_valid_mask.shape,
                correlation_cfg.grid_rows, correlation_cfg.grid_cols, correlation_cfg,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise LocalCorrelationRejected("NO_SAFE_FIELD", "Unable to fit a safe local displacement field.", {"error": str(exc)}) from exc
        if comparison.selected_name is None:
            raise LocalCorrelationRejected(
                "FIELD_LOO_FAILED", comparison.reason,
                {"metrics": [metric.__dict__ for metric in comparison.metrics]},
            )
        field = fit_selected_displacement_field(
            comparison.selected_name, accepted, context.coarse.rgb_valid_mask.shape,
            correlation_cfg.grid_rows, correlation_cfg.grid_cols, correlation_cfg,
        )
        holdout = evaluate_heldout_field_improvement(
            evidence,
            context.coarse.registration_bands_rgb,
            context.coarse.registration_bands_ms,
            context.coarse.rgb_valid_mask,
            context.coarse.ms_valid_mask,
            transform.matrix,
            comparison.selected_name,
            correlation_cfg,
        )
        if not holdout.is_accepted:
            raise LocalCorrelationRejected(
                "HOLDOUT_IMPROVEMENT_FAILED", holdout.reason,
                {
                    "evaluated_count": holdout.evaluated_count,
                    "mean_improvement": holdout.mean_improvement,
                    "median_improvement": holdout.median_improvement,
                    "win_fraction": holdout.win_fraction,
                },
            )

        native_transform = _to_native_transform(transform, context.coarse)
        try:
            native_footprint = evaluate_native_displacement_footprint(
                context.rgb_meta, context.ms_meta, context.coarse, native_transform, field, cfg.warp.tile_size,
            )
        except ValueError as exc:
            raise LocalCorrelationRejected("LOCAL_FOOTPRINT_FAILED", str(exc)) from exc
        if (native_footprint.retained_source_valid_ratio < cfg.transform.min_retained_valid_ratio or
                native_footprint.reference_overlap_ratio < cfg.transform.min_reference_overlap_ratio):
            raise LocalCorrelationRejected(
                "LOCAL_FOOTPRINT_FAILED", "Local displacement field failed native footprint confirmation.",
                {
                    "retained_source_valid_ratio": native_footprint.retained_source_valid_ratio,
                    "reference_overlap_ratio": native_footprint.reference_overlap_ratio,
                },
            )

        stem = context.ms_meta.path.stem
        final_path = output_dir_obj / f"{stem}_aligned.tif"
        staged_path = output_dir_obj / f".{stem}_aligned.local_correlation.partial.tif"
        try:
            warp_ms_with_displacement_field_tiled(
                context.coarse, native_transform, field, staged_path, cfg.warp, context.ms_meta,
            )
            os.replace(staged_path, final_path)
        finally:
            if staged_path.exists():
                staged_path.unlink()

        preview_path = output_dir_obj / f"{stem}_alignment_preview.png"
        channel = transform.channel_pair.split("-")[0]
        preview_band = cfg.ms_green_band_index if channel == "green" else cfg.ms_red_band_index
        with rasterio.open(final_path) as aligned:
            warped_band = aligned.read(
                preview_band, out_shape=context.coarse.registration_bands_rgb[channel].shape,
                resampling=rasterio.enums.Resampling.average,
            )
        generate_alignment_preview(
            context.coarse.registration_bands_rgb[channel], warped_band, preview_path,
            rgb_mask=context.coarse.rgb_valid_mask, max_dimension=cfg.quality.preview_max_dimension,
        )
        status_counts = Counter(sample.status.value for sample in evidence.samples)
        local_payload = {
            "registration_shape": list(context.coarse.rgb_valid_mask.shape),
            "grid": [correlation_cfg.grid_rows, correlation_cfg.grid_cols],
            "trusted_cells": evidence.trusted_count,
            "spatial_coverage": evidence.coverage,
            "status_counts": dict(status_counts),
            "selected_field": comparison.selected_name,
            "field_metrics": [metric.__dict__ for metric in comparison.metrics],
            "holdout": {
                "evaluated_count": holdout.evaluated_count,
                "mean_improvement": holdout.mean_improvement,
                "median_improvement": holdout.median_improvement,
                "win_fraction": holdout.win_fraction,
                "accepted": holdout.is_accepted,
                "scores": [score.__dict__ for score in holdout.scores],
            },
            "cells": [
                {
                    "row": sample.row, "col": sample.col, "center_xy": list(sample.center_xy),
                    "channel": sample.channel, "residual_dx_dy": list(sample.residual_dx_dy),
                    "confidence": sample.confidence, "status": sample.status.value,
                    "phase_response": sample.phase_response, "peak_sharpness": sample.peak_sharpness,
                    "valid_fraction": sample.valid_fraction, "rejection_reason": sample.rejection_reason,
                }
                for sample in evidence.samples
            ],
        }
        report_path = output_dir_obj / f"{stem}_alignment_report.json"
        write_alignment_report(
            report_path, context.rgb_meta, context.ms_meta, transform, context.quality_report, cfg,
            final_path, preview_path, footprint_metrics=native_footprint,
            rejected_candidates=list(context.rejected_candidates),
            requested_alignment_mode=requested_mode,
            applied_alignment_mode=requested_mode,
            local_correlation=local_payload,
        )
        active_log.info(
            "Local correlation accepted: field=%s, trusted_cells=%d, held-out improvement=%.4f.",
            comparison.selected_name, evidence.trusted_count, holdout.mean_improvement,
        )
        return AlignmentResult(final_path, report_path, preview_path, context.quality_report, native_transform)
    except LocalCorrelationRejected as rejection:
        active_log.warning("Local correlation rejected (%s): %s", rejection.reason_code, rejection)
        return _publish_global_context(
            context, output_dir_obj, cfg, active_log,
            requested_alignment_mode=requested_mode,
            applied_alignment_mode=AlignmentMode.AUTOMATED.value,
            fallback={
                "reason_code": rejection.reason_code,
                "message": str(rejection),
                "details": rejection.details,
            },
        )


def local_mesh_align_orthomosaics(
    rgb_path: Path,
    ms_path: Path,
    output_dir: Path,
    config: Optional[AlignmentConfig] = None,
    log: Optional[logging.Logger] = None,
) -> AlignmentResult:
    """Export a verified local-mesh candidate, or safely fall back to global alignment.

    The local candidate is accepted only after sparse coverage, independent
    held-out structural improvement, field leave-one-out checks, and gradient
    checks inside :func:`export_local_mesh_candidate`.  Any expected local
    rejection leaves the established automated alignment pipeline in charge.
    """
    active_log = log or logger
    base_cfg = config or AlignmentConfig()
    tuned_coarse = base_cfg.coarse.model_copy(update={"registration_max_dimension": min(base_cfg.coarse.registration_max_dimension, 1024)})
    tuned_local = base_cfg.local_mesh.model_copy(update={"enabled": True, "grid_rows": 5, "grid_cols": 5})
    cfg = base_cfg.model_copy(update={
        "alignment_mode": AlignmentMode.LOCAL_MESH,
        "coarse": tuned_coarse,
        "local_mesh": tuned_local,
    })
    # Validate before attempting the relatively expensive experimental export.
    validate_inputs(rgb_path, ms_path, cfg)
    try:
        exported = export_local_mesh_candidate(rgb_path, ms_path, output_dir, cfg)
        if exported.held_out_improvement is None or exported.held_out_improvement <= 0:
            raise TransformUnreliableError(
                f"Local mesh did not improve held-out structure ({exported.held_out_improvement})."
            )
        quality = SpatialResidualReport(
            global_rmse_px=None, center_rmse_px=None, edge_corner_rmse_px=None,
            max_residual_px=None, residual_drift_ratio=None, grid_residuals=[],
            status="PASS", is_spatial_drift_acceptable=True,
        )
        active_log.info(
            "Rail-bounded local alignment accepted: field=%s, corrected_regions=%d, coverage=%.1f%%, "
            "mean held-out correlation improvement=%.3f.",
            exported.field_name, exported.trusted_cells, 100 * exported.sparse_coverage,
            exported.held_out_improvement,
        )
        return AlignmentResult(
            exported.aligned_path, exported.review_path, exported.preview_path,
            quality, exported.native_global_transform,
        )
    except (RuntimeError, ValueError, TransformUnreliableError, np.linalg.LinAlgError) as exc:
        active_log.warning("Local mesh rejected; falling back to automated global alignment: %s", exc)
        fallback_cfg = cfg.model_copy(update={"alignment_mode": AlignmentMode.AUTOMATED})
        return align_orthomosaics(rgb_path, ms_path, output_dir, fallback_cfg, active_log)


def manual_align_orthomosaics(
    rgb_path: Path,
    ms_path: Path,
    output_dir: Path,
    pts_rgb: np.ndarray | list[tuple[float, float]],
    pts_ms: np.ndarray | list[tuple[float, float]],
    coordinate_mode: ManualCoordinateMode | str = ManualCoordinateMode.PIXEL,
    config: Optional[AlignmentConfig] = None,
    log: Optional[logging.Logger] = None,
) -> AlignmentResult:
    """
    Manually aligns a Multispectral (MS) GeoTIFF to an RGB reference using user-supplied control point pairs.

    Args:
        rgb_path: Path to RGB reference GeoTIFF.
        ms_path: Path to MS target GeoTIFF.
        output_dir: Output directory.
        pts_rgb: RGB control points in the declared coordinate mode.
        pts_ms: Corresponding MS control points in the declared coordinate mode.
        coordinate_mode: ``map`` for CRS coordinates, or ``pixel`` for native
            raster pixel coordinates.  The mode is explicit; no magnitude
            heuristic is used.
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
    mode = ManualCoordinateMode(coordinate_mode)
    if arr_rgb.ndim != 2 or arr_rgb.shape[1] != 2 or arr_ms.ndim != 2 or arr_ms.shape[1] != 2:
        raise TransformUnreliableError("Manual control points must be Nx2 X/Y coordinate pairs.")
    if len(arr_rgb) != len(arr_ms) or len(arr_rgb) == 0:
        raise TransformUnreliableError("Provide the same non-zero number of RGB and MS control points.")

    reg_inv = ~coarse.registration_transform
    reg_pts_rgb_list = []
    reg_pts_ms_list = []

    for pt_rgb, pt_ms in zip(arr_rgb, arr_ms):
        if mode == ManualCoordinateMode.MAP:
            map_rgb = (float(pt_rgb[0]), float(pt_rgb[1]))
            map_ms = (float(pt_ms[0]), float(pt_ms[1]))
            rgb_left, rgb_bottom, rgb_right, rgb_top = rgb_meta.bounds
            ms_left, ms_bottom, ms_right, ms_top = ms_meta.bounds
            if not (rgb_left <= map_rgb[0] <= rgb_right and rgb_bottom <= map_rgb[1] <= rgb_top):
                raise TransformUnreliableError(
                    "An RGB map control point lies outside the RGB raster footprint. "
                    "Use the raster CRS coordinates shown by QGIS, not latitude/longitude."
                )
            if not (ms_left <= map_ms[0] <= ms_right and ms_bottom <= map_ms[1] <= ms_top):
                raise TransformUnreliableError(
                    "An MS map control point lies outside the MS raster footprint. "
                    "Use the raster CRS coordinates shown by QGIS, not latitude/longitude."
                )
        else:
            if not (0 <= pt_rgb[0] < rgb_meta.width and 0 <= pt_rgb[1] < rgb_meta.height):
                raise TransformUnreliableError("An RGB pixel control point is outside the RGB raster dimensions.")
            if not (0 <= pt_ms[0] < ms_meta.width and 0 <= pt_ms[1] < ms_meta.height):
                raise TransformUnreliableError("An MS pixel control point is outside the MS raster dimensions.")
            map_rgb = rgb_meta.transform * (float(pt_rgb[0]), float(pt_rgb[1]))
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


def road_grid_align_orthomosaics(
    rgb_path: Path,
    ms_path: Path,
    output_dir: Path,
    config: Optional[AlignmentConfig] = None,
    log: Optional[logging.Logger] = None,
) -> AlignmentResult:
    """
    Automatically aligns an MS orthomosaic to an RGB reference using the
    Road-Grid strategy: structural road/tree-row feature matching via 1D
    projection profile cross-correlation.

    This mode is fully automatic — no user input required.

    Args:
        rgb_path: Path to RGB reference GeoTIFF.
        ms_path: Path to MS target GeoTIFF.
        output_dir: Output directory.
        config: Alignment configuration.
        log: Optional logger instance.

    Returns:
        AlignmentResult with paths to aligned GeoTIFF, QA preview, and JSON report.
    """
    active_log = log or logger
    cfg = config or AlignmentConfig()
    cfg = cfg.model_copy(update={"alignment_mode": AlignmentMode.ROAD_GRID})
    output_dir_obj = Path(output_dir).resolve()
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    rgb_meta, ms_meta = validate_inputs(rgb_path, ms_path, cfg)
    coarse = coarse_align(rgb_meta, ms_meta, cfg)

    # Pick the first available registration channel
    channel = cfg.registration_channel_priority[0].value
    active_log.info("Road-Grid alignment using '%s' registration channel.", channel)

    transform_res = compute_road_grid_translation(
        rgb_band=coarse.registration_bands_rgb[channel],
        ms_band=coarse.registration_bands_ms[channel],
        rgb_mask=coarse.rgb_valid_mask,
        ms_mask=coarse.ms_valid_mask,
        target_gsd=coarse.registration_gsd,
        road_grid_config=cfg.road_grid,
        transform_config=cfg.transform,
    )
    transform_res = replace(transform_res, channel_pair=f"{channel}-{channel}")
    active_log.info(
        "Road-Grid translation (registration grid): tx=%.2f px, ty=%.2f px",
        transform_res.translation_px[0], transform_res.translation_px[1],
    )
    active_log.info(
        "Road-Grid translation (metres): tx=%.4f m, ty=%.4f m",
        transform_res.translation_m[0], transform_res.translation_m[1],
    )

    transform_res, footprint = _validate_candidate_footprint(transform_res, coarse, cfg)

    # Quality report — road-grid produces a global translation, no point-based residuals
    quality_rep = SpatialResidualReport(
        global_rmse_px=None, center_rmse_px=None, edge_corner_rmse_px=None,
        max_residual_px=None, residual_drift_ratio=None, grid_residuals=[],
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
    reference_band = coarse.registration_bands_rgb[channel]
    preview_band = cfg.ms_green_band_index if channel == "green" else cfg.ms_red_band_index
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
    active_log.info("Road-Grid alignment successful.")
    return AlignmentResult(final_path, report_path, preview_path, quality_rep, native_transform)
