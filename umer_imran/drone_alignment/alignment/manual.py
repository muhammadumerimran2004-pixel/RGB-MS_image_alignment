from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from drone_alignment.config.schema import ManualAlignmentConfig, TransformType, TransformValidationConfig, QualityConfig
from drone_alignment.alignment.transform_estimator import TransformResult, TransformUnreliableError
from drone_alignment.alignment.displacement_field import nearest_control_support, renormalized_taper
from drone_alignment.alignment.control_points import (
    ControlPointSet, GeometricModel, JacobianReport,
    _apply_affine, apply_outlier_policy, check_bounds, evaluate_jacobian,
    fit_linear, linear_parameters_at_centroid, loo_residual_vectors, robust_residuals,
    select_model, validate_geometry,
)
from drone_alignment.quality.metrics import SpatialResidualReport, evaluate_spatial_residuals


def is_scipy_available() -> bool:
    """Check if SciPy RBFInterpolator is available for Thin Plate Spline fitting."""
    try:
        from scipy.interpolate import RBFInterpolator  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class ManualThinPlateSplineField:
    """Thin-plate spline residual field for manual GCP alignment.

    Evaluates registration-grid forward residual vectors (dx, dy).
    When fallback_weight > 0, residual tapers toward zero outside the support radius,
    gracefully returning to the baseline global affine transform.
    """
    interpolator_x: object
    interpolator_y: object
    control_xy: np.ndarray
    support_radius_px: float = 0.0
    fallback_weight: float = 0.0
    name: str = "manual_thin_plate_spline"

    def __post_init__(self) -> None:
        if self.fallback_weight > 0.0:
            object.__setattr__(self, "_control_tree", cKDTree(self.control_xy))

    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        values = np.column_stack([self.interpolator_x(points), self.interpolator_y(points)])
        if self.fallback_weight <= 0.0:
            return values
        support = nearest_control_support(points, self.control_xy, self.support_radius_px, tree=self._control_tree)
        taper = renormalized_taper(support, self.fallback_weight)
        return values * taper[:, None]


@dataclass(frozen=True)
class ManualAlignmentResult:
    transform: TransformResult
    field: ManualThinPlateSplineField | None
    quality_report: SpatialResidualReport
    payload: dict


def _tps_smoothing_px(cfg: ManualAlignmentConfig, target_gsd: float) -> float:
    """SciPy's smoothing penalizes squared residual units, so a metre smoothing
    parameter is converted to registration-pixel units as (m / gsd) ** 2."""
    if cfg.tps_smoothing_m <= 0.0:
        return 0.0
    return (cfg.tps_smoothing_m / target_gsd) ** 2


def _fit_tps_field(control: ControlPointSet, matrix: np.ndarray, smoothing_px: float, cfg: ManualAlignmentConfig) -> ManualThinPlateSplineField:
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError as exc:
        raise TransformUnreliableError("Thin Plate Spline alignment requires SciPy.") from exc
    predicted = _apply_affine(matrix, control.tgt_xy)
    residual = control.ref_xy - predicted
    try:
        interp_x = RBFInterpolator(control.ref_xy, residual[:, 0], kernel="thin_plate_spline", smoothing=smoothing_px)
        interp_y = RBFInterpolator(control.ref_xy, residual[:, 1], kernel="thin_plate_spline", smoothing=smoothing_px)
    except np.linalg.LinAlgError as exc:
        raise TransformUnreliableError(
            "Thin Plate Spline fit produced a singular system; check for duplicate or near-collinear "
            "control points."
        ) from exc
    return ManualThinPlateSplineField(
        interpolator_x=interp_x, interpolator_y=interp_y, control_xy=control.ref_xy,
        support_radius_px=cfg.taper_support_radius_px, fallback_weight=cfg.taper_fallback_weight,
    )


def _checkpoint_residual_vectors(check: ControlPointSet, matrix: np.ndarray, field: ManualThinPlateSplineField | None) -> np.ndarray:
    predicted = _apply_affine(matrix, check.tgt_xy)
    if field is not None:
        predicted = predicted + field.evaluate(check.ref_xy)
    return check.ref_xy - predicted


def _points_payload(
    cps: ControlPointSet,
    residual_set: ControlPointSet, residuals,
    loo_set: ControlPointSet, loo_vectors: np.ndarray | None,
) -> list[dict]:
    n = len(cps)
    loo_by_id = {}
    if loo_vectors is not None:
        for gcp_id, vec in zip(loo_set.ids, loo_vectors):
            loo_by_id[gcp_id] = float(np.linalg.norm(vec))
    residual_by_id = {}
    z_by_id = {}
    outlier_by_id = {}
    if residuals is not None:
        for i, gcp_id in enumerate(residual_set.ids):
            residual_by_id[gcp_id] = float(residuals.residual_px[i])
            z_by_id[gcp_id] = float(residuals.robust_z[i])
            outlier_by_id[gcp_id] = bool(residuals.outlier[i])
    points = []
    for i in range(n):
        gcp_id = cps.ids[i]
        points.append({
            "id": gcp_id,
            "role": "check" if cps.is_check[i] else "control",
            "rgb": [float(cps.ref_xy[i, 0]), float(cps.ref_xy[i, 1])],
            "ms": [float(cps.tgt_xy[i, 0]), float(cps.tgt_xy[i, 1])],
            "fit_residual_px": residual_by_id.get(gcp_id),
            "loo_residual_px": loo_by_id.get(gcp_id),
            "robust_z": z_by_id.get(gcp_id),
            "outlier": outlier_by_id.get(gcp_id, False),
        })
    return points


def run_manual_alignment(
    cps: ControlPointSet,
    cfg: ManualAlignmentConfig,
    bounds: TransformValidationConfig,
    quality_cfg: QualityConfig,
    registration_shape: tuple[int, int],
    target_gsd: float,
) -> ManualAlignmentResult:
    """Fit, validate, and QA-gate a manual control-point alignment.

    ``cps`` holds every supplied point pair in registration-grid pixel
    coordinates. Raises :class:`TransformUnreliableError` for every rejection
    (geometry, bounds, outliers, LOO/checkpoint/Jacobian gates) - manual GCPs
    are few and deliberate, so a failure is reported for the user to fix
    rather than silently substituted.
    """
    scipy_available = is_scipy_available()
    valid_area_px = float(registration_shape[0]) * float(registration_shape[1])

    control_all = cps.control
    check = cps.check
    if len(control_all) == 0:
        raise TransformUnreliableError("At least one control-role point pair is required.")

    geometry = validate_geometry(control_all, min_separation_px=1.0, min_conditioning=1e-3, valid_area_px=valid_area_px)
    if not geometry.ok:
        raise TransformUnreliableError("; ".join(geometry.messages))

    tps_smoothing_px = _tps_smoothing_px(cfg, target_gsd)
    decision = select_model(cps, cfg, scipy_available, hull_coverage=geometry.ref_hull_coverage, tps_smoothing=tps_smoothing_px)
    model = decision.model
    fit_model = GeometricModel.AFFINE if model == GeometricModel.TPS else model

    matrix = fit_linear(control_all, fit_model)
    residuals = robust_residuals(control_all, matrix, threshold_mad=cfg.outlier_threshold_mad)
    control = apply_outlier_policy(control_all, residuals, cfg.outlier_policy)
    if len(control) != len(control_all):
        matrix = fit_linear(control, fit_model)
        # Re-evaluate every original control point (dropped ones included)
        # against the final, outlier-free fit, so the report explains why
        # each point was flagged rather than only showing the survivors.
        residuals = robust_residuals(control_all, matrix, threshold_mad=cfg.outlier_threshold_mad)

    centroid = control.tgt_xy.mean(axis=0)
    linear_params = linear_parameters_at_centroid(matrix, centroid)
    translation_m = (linear_params.translation_at_centroid[0] * target_gsd, linear_params.translation_at_centroid[1] * target_gsd)
    violations = check_bounds(linear_params.rotation_deg, linear_params.scale_x, linear_params.scale_y, translation_m, bounds)
    if violations:
        raise TransformUnreliableError("Manual control-point fit exceeds sanity bounds: " + "; ".join(violations))

    field: ManualThinPlateSplineField | None = None
    jacobian_report: JacobianReport | None = None
    if model == GeometricModel.TPS:
        field = _fit_tps_field(control, matrix, tps_smoothing_px, cfg)
        jacobian_report = evaluate_jacobian(field, registration_shape, cfg.min_jacobian_det, cfg.max_field_gradient)
        if not jacobian_report.ok:
            raise TransformUnreliableError(
                f"TPS residual field fails the Jacobian safety gate: min det(I+grad)={jacobian_report.min_det:.3f} "
                f"(required >= {cfg.min_jacobian_det}), max ||grad||={jacobian_report.max_gradient:.3f} "
                f"(required <= {cfg.max_field_gradient})."
            )

    loo_vectors = loo_residual_vectors(control, model, tps_smoothing_px)
    loo_rmse_value = None
    if loo_vectors is not None:
        loo_rmse_value = float(np.sqrt(np.mean(np.sum(loo_vectors ** 2, axis=1))))
        if loo_rmse_value > cfg.max_loo_rmse_px:
            raise TransformUnreliableError(
                f"Leave-one-out RMSE ({loo_rmse_value:.2f}px) exceeds max_loo_rmse_px ({cfg.max_loo_rmse_px:.2f}px)."
            )

    checkpoint_rmse_value = None
    if len(check) > 0:
        checkpoint_vectors = _checkpoint_residual_vectors(check, matrix, field)
        checkpoint_rmse_value = float(np.sqrt(np.mean(np.sum(checkpoint_vectors ** 2, axis=1))))
        if checkpoint_rmse_value > cfg.max_checkpoint_rmse_px:
            raise TransformUnreliableError(
                f"Check-point RMSE ({checkpoint_rmse_value:.2f}px) exceeds max_checkpoint_rmse_px "
                f"({cfg.max_checkpoint_rmse_px:.2f}px)."
            )

    if loo_vectors is not None:
        predicted_ref = control.ref_xy - loo_vectors
        quality_report = evaluate_spatial_residuals(control.ref_xy, predicted_ref, registration_shape, quality_cfg)
    elif len(check) > 0:
        checkpoint_vectors = _checkpoint_residual_vectors(check, matrix, field)
        predicted_ref = check.ref_xy - checkpoint_vectors
        quality_report = evaluate_spatial_residuals(check.ref_xy, predicted_ref, registration_shape, quality_cfg)
    else:
        quality_report = SpatialResidualReport(
            global_rmse_px=None, center_rmse_px=None, edge_corner_rmse_px=None, max_residual_px=None,
            residual_drift_ratio=None, grid_residuals=[], status="UNVERIFIED", is_spatial_drift_acceptable=False,
        )

    method = "manual_translation" if model == GeometricModel.TRANSLATION else (
        "manual_similarity" if model == GeometricModel.SIMILARITY else (
            "manual_affine" if model == GeometricModel.AFFINE else "manual_tps"
        )
    )
    num_pts = len(cps)
    transform_res = TransformResult(
        matrix=matrix, transform_type=TransformType.AFFINE,
        translation_px=(float(matrix[0, 2]), float(matrix[1, 2])),
        translation_m=(float(matrix[0, 2]) * target_gsd, float(matrix[1, 2]) * target_gsd),
        rotation_deg=linear_params.rotation_deg, scale=(linear_params.scale_x, linear_params.scale_y),
        inlier_ratio=len(control) / num_pts if num_pts else 1.0,
        num_inliers=len(control), num_total_matches=num_pts,
        method=method, channel_pair="manual",
    )

    points_payload = _points_payload(cps, control_all, residuals, control, loo_vectors)
    payload = {
        "model": model.value,
        "model_reason": decision.reason,
        "coordinate_mode": None,
        "pixel_convention": cfg.pixel_convention,
        "points": points_payload,
        "loo_rmse_px": {"affine": decision.loo_rmse_affine, "tps": decision.loo_rmse_tps},
        # The model-selection comparison above only ever fits affine/TPS. When
        # the selected (or forced) model is translation/similarity - or is
        # affine/TPS but decided by a forced override rather than the
        # comparison - this is the LOO RMSE of the model actually published,
        # independent of what select_model happened to compare.
        "loo_rmse_selected_px": loo_rmse_value,
        "checkpoint_rmse_px": checkpoint_rmse_value,
        "baseline": {
            "rotation_deg": linear_params.rotation_deg,
            "scale": [linear_params.scale_x, linear_params.scale_y],
            "translation_at_centroid_m": [translation_m[0], translation_m[1]],
        },
        "jacobian": None if jacobian_report is None else {
            "min_det": jacobian_report.min_det, "max_gradient": jacobian_report.max_gradient,
        },
        "field_lattice_max_error_px": None,
    }

    return ManualAlignmentResult(transform=transform_res, field=field, quality_report=quality_report, payload=payload)
