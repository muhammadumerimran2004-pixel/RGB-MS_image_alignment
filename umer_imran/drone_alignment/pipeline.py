from collections import Counter
from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import rasterio

from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, DetectorType, ManualCoordinateMode
from drone_alignment.io.validators import validate_inputs
from drone_alignment.io.reader import RasterMetadata
from drone_alignment.alignment.coarse import coarse_align, CoarseAlignmentResult
from drone_alignment.alignment.manual import run_manual_alignment
from drone_alignment.alignment.control_points import build_control_point_set
from drone_alignment.alignment.field_eval import sample_field_on_lattice
from drone_alignment.alignment.arosics_matcher import match_arosics
from drone_alignment.alignment.arosics_local import run_arosics_local_refinement
from drone_alignment.alignment.rejections import LocalRefinementRejected
from drone_alignment.alignment.road_grid_aligner import (
    compute_road_grid_translation, InsufficientRoadFeatureError,
)
from drone_alignment.alignment.feature_detector import (
    detect_features, normalize_to_uint8, InsufficientValidDataError, InsufficientContrastError,
)
from drone_alignment.alignment.feature_matcher import match_features, InsufficientMatchesError, MatchResult
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
    evaluate_spatial_residuals, evaluate_footprint, footprint_gate_failures, SpatialResidualReport, FootprintMetrics,
)
from drone_alignment.quality.visualization import generate_alignment_preview
from drone_alignment.quality.report import write_alignment_report
from drone_alignment.progress import ProgressReporter

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
    is_verified: bool = True
    """False when every classical/AROSICS-global/phase-correlation candidate
    failed independent QA and ``registration_transform`` is instead the best
    computed-but-unverified candidate, kept only as an AROSICS COREG_LOCAL
    starting point. Never publish such a context directly - AROSICS local
    refinement's own gates (which do not depend on this flag) are the only
    thing allowed to turn it into a result."""


# LocalCorrelationRejected now lives in alignment.rejections as
# LocalRefinementRejected (shared across every local-refinement engine, not
# just cell correlation). Kept as an alias so existing imports/isinstance
# checks against pipeline.LocalCorrelationRejected keep working.
LocalCorrelationRejected = LocalRefinementRejected


def _validate_candidate_footprint(transform, coarse, cfg) -> tuple[TransformResult, FootprintMetrics]:
    footprint = evaluate_footprint(
        coarse.rgb_valid_mask, coarse.ms_valid_mask, transform.matrix, transform.transform_type
    )
    failures = footprint_gate_failures(footprint, cfg.transform)
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



def _unverified_quality_report(status: str = "FAIL") -> SpatialResidualReport:
    return SpatialResidualReport(
        global_rmse_px=None, center_rmse_px=None, edge_corner_rmse_px=None,
        max_residual_px=None, residual_drift_ratio=None, grid_residuals=[],
        status=status, is_spatial_drift_acceptable=False,
    )


def _rank_unverified_candidate(item: tuple[TransformResult, FootprintMetrics, SpatialResidualReport]) -> tuple:
    """Sort key: candidates with measured held-out residuals first, then higher held-out
    agreement (in 10% steps, so a tiny RMSE over a handful of agreeing points can't win),
    then lower RMSE, then higher inlier ratio."""
    transform, _footprint, quality = item
    rmse = quality.global_rmse_px
    has_rmse = rmse is not None
    agreement = quality.holdout_agreement_ratio if quality.holdout_agreement_ratio is not None else 0.0
    return (0 if has_rmse else 1, -round(agreement, 1), rmse if has_rmse else 0.0, -transform.inlier_ratio)


def _estimate_global_candidate(
    coarse, cfg: AlignmentConfig, active_log: logging.Logger
) -> tuple[TransformResult, FootprintMetrics, SpatialResidualReport, list[str], bool]:
    """Estimate the best global candidate.

    Returns ``is_verified=True`` when a candidate passed independent QA
    (classical feature matching, AROSICS' own COREG, or masked-
    correlation phase fallback) exactly as before. When every candidate is
    rejected, this no longer raises outright: it instead returns the single
    best computed-but-unverified candidate with ``is_verified=False``, so the
    caller can still hand it to AROSICS COREG_LOCAL as a starting point -
    AROSICS' own tie-point/holdout/full-grid gates do not depend on our
    classical residual QA having passed first, and are the actual safety net
    for that path. Only raises when literally no candidate could be computed
    at all (every method failed structurally, not just its QA check).
    """
    common_mask = coarse.rgb_valid_mask & coarse.ms_valid_mask
    active_log.info(
        "Valid footprints: RGB=%.2f%%, MS=%.2f%%, common=%.2f%%",
        100 * coarse.rgb_valid_mask.mean(), 100 * coarse.ms_valid_mask.mean(), 100 * common_mask.mean(),
    )

    accepted = []
    rejected_pool: list[tuple[TransformResult, FootprintMetrics, SpatialResidualReport]] = []
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
                quality = evaluate_spatial_residuals(
                    pts_rgb, transformed, coarse.rgb_valid_mask.shape, cfg.quality, screen_false_matches=True,
                )
                if quality.status != "PASS":
                    rejected_pool.append((transform, footprint, quality))
                    raise TransformUnreliableError(f"independent/residual QA status is {quality.status}")
                accepted.append((transform, footprint, quality, pts_rgb, transformed))
                active_log.info("Accepted candidate %s with %d/%d inliers", attempt, transform.num_inliers, transform.num_total_matches)
            except expected_errors as exc:
                reason = f"{attempt}: {exc}"
                rejection_reasons.append(reason)
                active_log.warning("Rejected candidate %s", reason)

    if not accepted and cfg.arosics.enabled and cfg.arosics.global_candidate.enabled:
        if cfg.arosics.band_pairs:
            arosics_candidates = [
                (pair.name, f"arosics:{pair.name}") for pair in cfg.arosics.band_pairs
            ]
        else:
            arosics_candidates = [
                (f"{channel.value}-{channel.value}", channel.value)
                for channel in cfg.registration_channel_priority
            ]

        for pair_label, band_key in arosics_candidates:
            attempt = f"{pair_label}/arosics"
            try:
                transform = match_arosics(
                    coarse.registration_bands_rgb[band_key],
                    coarse.registration_bands_ms[band_key],
                    coarse.rgb_valid_mask,
                    coarse.ms_valid_mask,
                    coarse.registration_transform,
                    coarse.target_profile.get("crs"),
                    coarse.registration_gsd,
                    cfg.arosics.global_candidate,
                    cfg.transform,
                )
                transform = replace(transform, channel_pair=pair_label)
                transform, footprint = _validate_candidate_footprint(transform, coarse, cfg)

                # Verify spatial alignment in the image domain
                h, w = coarse.rgb_valid_mask.shape
                warped_ms = cv2.warpAffine(
                    coarse.registration_bands_ms[band_key], transform.matrix, (w, h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                warped_mask = cv2.warpAffine(
                    coarse.ms_valid_mask.astype(np.uint8), transform.matrix, (w, h),
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ) > 0
                verification_score = _masked_correlation(
                    coarse.registration_bands_rgb[band_key], warped_ms, coarse.rgb_valid_mask & warped_mask
                )
                if verification_score < cfg.transform.min_phase_verification_correlation:
                    rejected_pool.append((transform, footprint, _unverified_quality_report()))
                    raise TransformUnreliableError(
                        f"AROSICS image verification correlation ({verification_score:.3f}) below minimum "
                        f"({cfg.transform.min_phase_verification_correlation:.3f})"
                    )

                quality = SpatialResidualReport(
                    global_rmse_px=None, center_rmse_px=None, edge_corner_rmse_px=None,
                    max_residual_px=None, residual_drift_ratio=None, grid_residuals=[],
                    status="PASS", is_spatial_drift_acceptable=True,
                )
                accepted.append((transform, footprint, quality, None, None))
                active_log.info(
                    "Accepted AROSICS candidate %s with verification correlation %.3f",
                    attempt, verification_score,
                )
                break
            except expected_errors as exc:
                reason = f"{attempt}: {exc}"
                rejection_reasons.append(reason)
                active_log.warning("Rejected candidate %s", reason)


    if accepted:
        accepted.sort(key=lambda item: (item[0].inlier_ratio, item[0].num_inliers), reverse=True)
        transform_res, footprint, quality_rep, _, _ = accepted[0]
        return transform_res, footprint, quality_rep, rejection_reasons, True

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
            rejected_pool.append((transform_res, footprint, _unverified_quality_report()))
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
        return transform_res, footprint, quality_rep, rejection_reasons, True
    except (TransformUnreliableError, InsufficientValidDataError, InsufficientContrastError) as exc:
        rejection_reasons.append(f"{channel}-{channel}/phase_correlation: {exc}")

    if rejected_pool:
        rejected_pool.sort(key=_rank_unverified_candidate)
        transform_res, footprint, quality_rep = rejected_pool[0]
        active_log.warning(
            "No candidate passed independent QA; using the best computed-but-unverified candidate "
            "(method=%s, channel_pair=%s) as an AROSICS COREG_LOCAL starting point only. "
            "%s",
            transform_res.method, transform_res.channel_pair, " | ".join(rejection_reasons),
        )
        return transform_res, footprint, quality_rep, rejection_reasons, False

    raise TransformUnreliableError("No verified alignment candidate. " + " | ".join(rejection_reasons))


def _estimate_verified_global_context(
    rgb_path: Path,
    ms_path: Path,
    cfg: AlignmentConfig,
    active_log: logging.Logger,
) -> VerifiedGlobalContext:
    """Validate inputs and produce one reusable, verified global baseline."""
    rgb_meta, ms_meta = validate_inputs(rgb_path, ms_path, cfg)
    coarse = coarse_align(rgb_meta, ms_meta, cfg)
    transform_res, footprint, quality_rep, rejection_reasons, is_verified = _estimate_global_candidate(
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
        is_verified=is_verified,
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
    footprint_failures = footprint_gate_failures(native_footprint, cfg.transform)
    if footprint_failures:
        raise TransformUnreliableError("Native footprint confirmation failed: " + "; ".join(footprint_failures))
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
    """Estimate and publish the verified global alignment, optionally refined by AROSICS.

    The global candidate (ORB/SIFT/phase/AROSICS-global) is always
    estimated first. When one of them passes independent QA, AROSICS'
    COREG_LOCAL then refines *that* verified result with a small local search
    and performs the warp itself; a rejection at any of its safety gates
    falls back to publishing the verified global result untouched, with the
    rejection recorded in the report.

    When *none* of them pass QA, this pipeline does not simply give up: the
    single best computed-but-unverified candidate is still handed to AROSICS
    COREG_LOCAL as a starting point, exactly as AROSICS' own coarse(COREG)-
    to-fine(COREG_LOCAL) design intends - COREG_LOCAL's own tie-point,
    coverage, neighbour-consistency, and independent holdout/full-grid
    verification gates (not our classical residual QA) are what decide
    whether to publish in that case. An unverified candidate is never
    published on its own; only AROSICS' own verification can turn it into a
    result. If AROSICS is disabled/unavailable, or its own gates also reject
    the unverified starting point, the run fails with every rejection reason
    recorded - there is genuinely no safe result to publish.
    """
    active_log = log or logger
    progress = ProgressReporter(active_log)
    progress.update(0, "starting alignment")
    cfg = config or AlignmentConfig()
    output_dir_obj = Path(output_dir).resolve()
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    progress.update(3, "validated run configuration")
    context = _estimate_verified_global_context(rgb_path, ms_path, cfg, active_log)
    progress.update(45, "selected global alignment candidate")
    requested_mode = AlignmentMode.AUTOMATED.value

    if cfg.arosics.enabled and cfg.arosics.local.enabled:
        try:
            return _refine_with_arosics_local(
                context, output_dir_obj, cfg, active_log, requested_mode,
            )
        except LocalRefinementRejected as rejection:
            if not context.is_verified:
                raise TransformUnreliableError(
                    "No verified alignment candidate: classical/AROSICS-global estimation did not pass "
                    f"independent QA, and AROSICS local refinement also rejected the best-effort starting "
                    f"point ({rejection.reason_code}): {rejection}. " + " | ".join(context.rejected_candidates)
                ) from rejection
            active_log.warning(
                "AROSICS local refinement rejected (%s): %s", rejection.reason_code, rejection,
            )
            progress.update(75, "local refinement rejected; publishing verified global result")
            result = _publish_global_context(
                context, output_dir_obj, cfg, active_log,
                requested_alignment_mode=requested_mode, applied_alignment_mode="automated_global",
                fallback={
                    "stage": "arosics_local", "reason_code": rejection.reason_code,
                    "message": str(rejection), "details": rejection.details,
                },
            )
            progress.update(100, "alignment complete")
            return result

    if not context.is_verified:
        raise TransformUnreliableError(
            "No verified alignment candidate, and AROSICS local refinement is disabled or unavailable to "
            "attempt a best-effort recovery. " + " | ".join(context.rejected_candidates)
        )
    progress.update(55, "publishing verified global result")
    result = _publish_global_context(
        context, output_dir_obj, cfg, active_log,
        requested_alignment_mode=requested_mode, applied_alignment_mode="automated_global",
    )
    progress.update(100, "alignment complete")
    return result


def _preview_reference_band(context: VerifiedGlobalContext, cfg: AlignmentConfig, band_pair_name: str) -> np.ndarray:
    """Best-effort lookup of the registration-grid RGB band matching the published band pair."""
    channel_key = f"arosics:{band_pair_name}" if cfg.arosics.band_pairs else band_pair_name.split("-")[0]
    reference_band = context.coarse.registration_bands_rgb.get(channel_key)
    if reference_band is not None:
        return reference_band
    return next(iter(context.coarse.registration_bands_rgb.values()))


def _refine_with_arosics_local(
    context: VerifiedGlobalContext,
    output_dir_obj: Path,
    cfg: AlignmentConfig,
    active_log: logging.Logger,
    requested_mode: str,
) -> AlignmentResult:
    """Delegate the refinement to alignment.arosics_local, then publish exactly like every other mode."""
    progress = ProgressReporter(active_log)
    progress.update(45, "selected global alignment candidate")
    common_mask = context.coarse.rgb_valid_mask & context.coarse.ms_valid_mask
    output_profile = context.coarse.output_profile or context.coarse.target_profile
    publication = run_arosics_local_refinement(
        context.rgb_meta, context.ms_meta, common_mask,
        context.coarse.registration_transform, context.coarse.registration_gsd,
        context.registration_transform, context.quality_report, output_profile,
        cfg, output_dir_obj, active_log, progress.update,
    )
    stem = context.ms_meta.path.stem
    preview_path = output_dir_obj / f"{stem}_alignment_preview.png"
    reference_band = _preview_reference_band(context, cfg, publication.band_pair_name)
    with rasterio.open(publication.aligned_path) as aligned:
        warped_band = aligned.read(
            publication.target_band_index, out_shape=reference_band.shape,
            resampling=rasterio.enums.Resampling.average,
        )
    generate_alignment_preview(
        reference_band, warped_band, preview_path,
        rgb_mask=context.coarse.rgb_valid_mask, max_dimension=cfg.quality.preview_max_dimension,
    )
    progress.update(99, "generated QA preview")
    report_path = output_dir_obj / f"{stem}_alignment_report.json"
    fallback = None
    if not context.is_verified:
        fallback = {
            "stage": "global_candidate", "reason_code": "UNVERIFIED_GLOBAL_USED_AS_STARTING_POINT",
            "message": (
                "No global candidate passed independent QA; the best computed-but-unverified candidate "
                "was used only as an AROSICS COREG_LOCAL starting point. This result is trusted because "
                "AROSICS' own tie-point/coverage/holdout/full-grid gates passed, not because the starting "
                "point itself was independently verified."
            ),
            "details": {"rejected_candidates": list(context.rejected_candidates)},
        }
    starting_qa = context.quality_report
    payload = dict(publication.payload)
    payload["global_starting_point_quality"] = {
        "status": starting_qa.status,
        "verified": context.is_verified,
        "global_rmse_px": starting_qa.global_rmse_px,
        "edge_corner_rmse_px": starting_qa.edge_corner_rmse_px,
        "holdout_agreement_ratio": starting_qa.holdout_agreement_ratio,
        "gross_mismatch_count": starting_qa.gross_mismatch_count,
    }
    # The published raster was approved by the local-refinement gates (tie points,
    # coverage, holdout, full-grid verification), not by the starting point's
    # classical QA, so the headline quality must be theirs.
    published_quality = _unverified_quality_report(status="PASS")
    published_quality.is_spatial_drift_acceptable = True
    write_alignment_report(
        report_path, context.rgb_meta, context.ms_meta, context.registration_transform, published_quality, cfg,
        publication.aligned_path, preview_path, footprint_metrics=publication.footprint,
        rejected_candidates=list(context.rejected_candidates),
        requested_alignment_mode=requested_mode, applied_alignment_mode="arosics_local",
        local_refinement=payload, fallback=fallback,
    )
    active_log.info("AROSICS local refinement published using band pair %s.", publication.band_pair_name)
    progress.update(100, "alignment complete")
    return AlignmentResult(
        publication.aligned_path, report_path, preview_path, published_quality, context.registration_transform,
    )


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
        footprint_failures = footprint_gate_failures(native_footprint, cfg.transform)
        if footprint_failures:
            raise LocalCorrelationRejected(
                "LOCAL_FOOTPRINT_FAILED",
                "Local displacement field failed native footprint confirmation: " + "; ".join(footprint_failures),
                {
                    "retained_source_valid_ratio": native_footprint.retained_source_valid_ratio,
                    "target_overlap_ratio": native_footprint.target_overlap_ratio,
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
    ids: Optional[list[str]] = None,
    roles: Optional[list[str]] = None,
    pixel_convention: Optional[str] = None,
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
        ids: Optional per-point identifiers (default ``P1``..``Pn``), used in
            rejection messages and the report's ``manual_control_points.points``.
        roles: Optional per-point ``"control"``/``"check"`` roles (default all
            ``"control"``). Check points are excluded from fitting and instead
            verify the published mapping's accuracy.
        pixel_convention: ``"center"`` or ``"corner"`` for pixel-mode input;
            defaults to ``cfg.manual.pixel_convention``. ``"center"`` adds
            +0.5 to both axes before applying the raster transform, since an
            integer pixel index denotes the pixel's centre.

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
    convention = pixel_convention or cfg.manual.pixel_convention
    if arr_rgb.ndim != 2 or arr_rgb.shape[1] != 2 or arr_ms.ndim != 2 or arr_ms.shape[1] != 2:
        raise TransformUnreliableError("Manual control points must be Nx2 X/Y coordinate pairs.")
    if len(arr_rgb) != len(arr_ms) or len(arr_rgb) == 0:
        raise TransformUnreliableError("Provide the same non-zero number of RGB and MS control points.")
    point_ids = list(ids) if ids is not None else [f"P{i + 1}" for i in range(len(arr_rgb))]
    point_roles = list(roles) if roles is not None else ["control"] * len(arr_rgb)
    if len(point_ids) != len(arr_rgb) or len(point_roles) != len(arr_rgb):
        raise TransformUnreliableError("ids and roles, when given, must have one entry per control point pair.")
    is_check = np.array([role == "check" for role in point_roles], dtype=bool)

    offset = 0.5 if (mode == ManualCoordinateMode.PIXEL and convention == "center") else 0.0
    reg_inv = ~coarse.registration_transform

    if mode == ManualCoordinateMode.MAP:
        map_rgb_pts = arr_rgb.copy()
        map_ms_pts = arr_ms.copy()
        rgb_left, rgb_bottom, rgb_right, rgb_top = rgb_meta.bounds
        if np.any(map_rgb_pts[:, 0] < rgb_left) or np.any(map_rgb_pts[:, 0] > rgb_right) or \
                np.any(map_rgb_pts[:, 1] < rgb_bottom) or np.any(map_rgb_pts[:, 1] > rgb_top):
            raise TransformUnreliableError(
                "An RGB map control point lies outside the RGB raster footprint. "
                "Use the raster CRS coordinates shown by QGIS, not latitude/longitude."
            )
        ms_left, ms_bottom, ms_right, ms_top = ms_meta.bounds
        if np.any(map_ms_pts[:, 0] < ms_left) or np.any(map_ms_pts[:, 0] > ms_right) or \
                np.any(map_ms_pts[:, 1] < ms_bottom) or np.any(map_ms_pts[:, 1] > ms_top):
            raise TransformUnreliableError(
                "An MS map control point lies outside the MS raster footprint. "
                "Use the raster CRS coordinates shown by QGIS, not latitude/longitude."
            )
        if str(rgb_meta.crs) != str(ms_meta.crs):
            from rasterio.warp import transform as warp_transform_coords
            xs, ys = warp_transform_coords(ms_meta.crs, rgb_meta.crs, map_ms_pts[:, 0].tolist(), map_ms_pts[:, 1].tolist())
            map_ms_pts = np.column_stack([xs, ys])
    else:
        if np.any(arr_rgb[:, 0] < 0) or np.any(arr_rgb[:, 0] >= rgb_meta.width) or \
                np.any(arr_rgb[:, 1] < 0) or np.any(arr_rgb[:, 1] >= rgb_meta.height):
            raise TransformUnreliableError("An RGB pixel control point is outside the RGB raster dimensions.")
        if np.any(arr_ms[:, 0] < 0) or np.any(arr_ms[:, 0] >= ms_meta.width) or \
                np.any(arr_ms[:, 1] < 0) or np.any(arr_ms[:, 1] >= ms_meta.height):
            raise TransformUnreliableError("An MS pixel control point is outside the MS raster dimensions.")
        rgb_px = arr_rgb + offset
        ms_px = arr_ms + offset
        map_rgb_pts = np.array([rgb_meta.transform * (float(x), float(y)) for x, y in rgb_px])
        map_ms_pts = np.array([ms_meta.transform * (float(x), float(y)) for x, y in ms_px])

    reg_pts_rgb = np.array([reg_inv * (float(x), float(y)) for x, y in map_rgb_pts], dtype=np.float64)
    reg_pts_ms = np.array([reg_inv * (float(x), float(y)) for x, y in map_ms_pts], dtype=np.float64)

    cps = build_control_point_set(point_ids, reg_pts_rgb, reg_pts_ms, is_check)
    registration_shape = coarse.rgb_valid_mask.shape
    manual_result = run_manual_alignment(
        cps, cfg.manual, cfg.transform, cfg.quality, registration_shape, coarse.registration_gsd,
    )
    transform_res = manual_result.transform
    field = manual_result.field
    quality_rep = manual_result.quality_report
    payload = manual_result.payload
    payload["coordinate_mode"] = mode.value
    active_log.info(
        "Manual %s alignment with %d control point(s) (%d check): tx=%.2f px, ty=%.2f px",
        payload["model"], len(cps.control), len(cps.check),
        transform_res.translation_px[0], transform_res.translation_px[1],
    )

    if field is not None:
        field = sample_field_on_lattice(field, registration_shape)
        payload["field_lattice_max_error_px"] = field.max_interp_error_px

    transform_res, footprint = _validate_candidate_footprint(transform_res, coarse, cfg)

    native_transform = _to_native_transform(transform_res, coarse)
    # The footprint check must be field-aware whenever a TPS field is being
    # published: checking only the affine baseline's footprint (as before)
    # can pass a candidate whose actual (baseline + field) mapping pushes
    # data outside the valid overlap.
    if field is not None:
        native_footprint = evaluate_native_displacement_footprint(
            rgb_meta, ms_meta, coarse, native_transform, field, cfg.warp.tile_size
        )
    else:
        native_footprint = evaluate_native_footprint(
            rgb_meta, ms_meta, coarse, native_transform, cfg.warp.tile_size
        )
    footprint_failures = footprint_gate_failures(native_footprint, cfg.transform)
    if footprint_failures:
        raise TransformUnreliableError("Native footprint confirmation failed: " + "; ".join(footprint_failures))
    native_transform = replace(
        native_transform, retained_valid_ratio=native_footprint.retained_source_valid_ratio,
        reference_overlap_ratio=native_footprint.reference_overlap_ratio,
    )

    ms_stem = Path(ms_path).stem
    final_path = output_dir_obj / f"{ms_stem}_aligned.tif"
    staged_path = output_dir_obj / f".{ms_stem}_aligned.partial.tif"
    try:
        if field is not None:
            warp_ms_with_displacement_field_tiled(
                coarse, native_transform, field, staged_path, cfg.warp, ms_meta
            )
        else:
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
        rejected_candidates=[], requested_alignment_mode="manual",
        applied_alignment_mode=f"manual_{payload['model']}",
        manual_control_points=payload,
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
    footprint_failures = footprint_gate_failures(native_footprint, cfg.transform)
    if footprint_failures:
        raise TransformUnreliableError("Native footprint confirmation failed: " + "; ".join(footprint_failures))
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
