"""Manual ground-control-point geometry, model fitting, and QA gates.

Operates purely on point coordinates (registration-grid pixels): no raster
I/O, no residual-field objects. :mod:`drone_alignment.alignment.manual`
builds on this to construct the actual warp-facing field and orchestrate the
pipeline-level flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import cv2
import numpy as np

from drone_alignment.alignment.transform_estimator import TransformUnreliableError


class GeometricModel(str, Enum):
    TRANSLATION = "translation"
    SIMILARITY = "similarity"
    AFFINE = "affine"
    TPS = "tps"


@dataclass(frozen=True)
class ControlPointSet:
    ref_xy: np.ndarray      # Nx2 registration px (RGB, destination)
    tgt_xy: np.ndarray      # Nx2 registration px (MS)
    ids: tuple[str, ...]
    is_check: np.ndarray    # N bool

    def __len__(self) -> int:
        return len(self.ref_xy)

    @property
    def control(self) -> "ControlPointSet":
        mask = ~self.is_check
        return _subset(self, mask)

    @property
    def check(self) -> "ControlPointSet":
        mask = self.is_check
        return _subset(self, mask)


def _subset(cps: ControlPointSet, mask: np.ndarray) -> ControlPointSet:
    ids = tuple(np.asarray(cps.ids, dtype=object)[mask].tolist())
    return ControlPointSet(cps.ref_xy[mask], cps.tgt_xy[mask], ids, cps.is_check[mask])


def build_control_point_set(
    ids: list[str], ref_xy: np.ndarray, tgt_xy: np.ndarray, is_check: np.ndarray | list[bool] | None = None,
) -> ControlPointSet:
    ref_arr = np.asarray(ref_xy, dtype=np.float64).reshape(-1, 2)
    tgt_arr = np.asarray(tgt_xy, dtype=np.float64).reshape(-1, 2)
    if len(ref_arr) != len(tgt_arr) or len(ref_arr) != len(ids):
        raise TransformUnreliableError("Control-point ids, RGB points, and MS points must have matching lengths.")
    if len(ref_arr) == 0:
        raise TransformUnreliableError("At least one manual control point pair is required.")
    check_mask = np.zeros(len(ref_arr), dtype=bool) if is_check is None else np.asarray(is_check, dtype=bool)
    return ControlPointSet(ref_arr, tgt_arr, tuple(ids), check_mask)


def _apply_affine(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return cv2.transform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)


# --- geometry validation (P3.2) ---------------------------------------------

@dataclass(frozen=True)
class GeometryReport:
    duplicate_pairs: tuple[tuple[str, str, float], ...]
    ref_conditioning_ratio: float | None
    tgt_conditioning_ratio: float | None
    ref_hull_coverage: float | None
    ok: bool
    messages: tuple[str, ...]


def _min_pairwise_distance(xy: np.ndarray) -> tuple[float, tuple[int, int]]:
    if len(xy) < 2:
        return math.inf, (0, 0)
    diffs = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return float(dist[i, j]), (int(i), int(j))


def _conditioning_ratio(xy: np.ndarray) -> float | None:
    if len(xy) < 3:
        return None
    centered = xy - xy.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if len(singular_values) < 2 or singular_values[0] <= 0:
        return 0.0
    return float(singular_values[1] / singular_values[0])


def _hull_coverage(xy: np.ndarray, valid_area_px: float | None) -> float | None:
    if len(xy) < 3 or not valid_area_px:
        return None
    hull = cv2.convexHull(xy.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    return hull_area / valid_area_px


def validate_geometry(
    cps: ControlPointSet, min_separation_px: float = 1.0, min_conditioning: float = 1e-3,
    valid_area_px: float | None = None,
) -> GeometryReport:
    """Reject near-duplicate or collinear/degenerate control-point clouds.

    Checked independently in both the reference (RGB) and target (MS) point
    clouds: a configuration can be well-conditioned in one space and
    degenerate in the other.
    """
    messages: list[str] = []
    ok = True
    dup_pairs: list[tuple[str, str, float]] = []
    for label, xy in (("reference", cps.ref_xy), ("target", cps.tgt_xy)):
        dist, (i, j) = _min_pairwise_distance(xy)
        if dist < min_separation_px:
            dup_pairs.append((cps.ids[i], cps.ids[j], float(dist)))
            messages.append(
                f"Control points {cps.ids[i]!r} and {cps.ids[j]!r} are {dist:.2f} px apart in {label} "
                f"coordinates (< {min_separation_px:.1f} px); likely a duplicate or click error."
            )
            ok = False
    ref_ratio = _conditioning_ratio(cps.ref_xy)
    tgt_ratio = _conditioning_ratio(cps.tgt_xy)
    for label, ratio in (("Reference", ref_ratio), ("Target", tgt_ratio)):
        if ratio is not None and ratio < min_conditioning:
            messages.append(f"{label} control points are collinear or degenerate; cannot fit a 2D transform.")
            ok = False
    hull_coverage = _hull_coverage(cps.ref_xy, valid_area_px)
    return GeometryReport(
        duplicate_pairs=tuple(dup_pairs), ref_conditioning_ratio=ref_ratio,
        tgt_conditioning_ratio=tgt_ratio, ref_hull_coverage=hull_coverage, ok=ok, messages=tuple(messages),
    )


# --- baselines (P3.3) --------------------------------------------------------

def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Closed-form similarity (rotation + uniform scale + translation), src -> dst."""
    n = len(src)
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    sigma_src = float((src_c ** 2).sum()) / n
    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(cov) < 0 or np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0
    rotation = u @ s @ vt
    scale = float(np.trace(np.diag(d) @ s)) / sigma_src if sigma_src > 1e-12 else 1.0
    translation = mu_dst - scale * (rotation @ mu_src)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:2, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def fit_linear(cps: ControlPointSet, model: GeometricModel) -> np.ndarray:
    """Ordinary-least-squares fit of the target -> reference mapping (2x3 affine). No RANSAC/LMEDS."""
    tgt, ref = cps.tgt_xy, cps.ref_xy
    n = len(tgt)
    if model == GeometricModel.TRANSLATION:
        if n < 1:
            raise TransformUnreliableError("At least 1 control point is required for a translation fit.")
        t = (ref - tgt).mean(axis=0)
        return np.array([[1.0, 0.0, t[0]], [0.0, 1.0, t[1]]], dtype=np.float64)
    if model == GeometricModel.SIMILARITY:
        if n < 2:
            raise TransformUnreliableError("At least 2 control points are required for a similarity fit.")
        return _umeyama(tgt, ref)
    if model in (GeometricModel.AFFINE, GeometricModel.TPS):
        if n < 3:
            raise TransformUnreliableError("At least 3 control points are required for an affine fit.")
        design = np.hstack([tgt, np.ones((n, 1))])
        params, *_ = np.linalg.lstsq(design, ref, rcond=None)
        return params.T  # rows: [a, b, tx], [c, d, ty]
    raise ValueError(f"Unknown geometric model: {model}")


@dataclass(frozen=True)
class PointResiduals:
    residual_vectors: np.ndarray  # Nx2 (ref - predicted)
    residual_px: np.ndarray       # N euclidean residual
    robust_z: np.ndarray          # N
    outlier: np.ndarray           # N bool


def robust_residuals(cps: ControlPointSet, matrix: np.ndarray, threshold_mad: float = 3.5) -> PointResiduals:
    """r_i = |ref_i - A(tgt_i)|; robust z_i = 0.6745 (r_i - median r) / MAD, MAD floored at 0.25px."""
    predicted = _apply_affine(matrix, cps.tgt_xy)
    vectors = cps.ref_xy - predicted
    residual = np.linalg.norm(vectors, axis=1)
    if len(residual) == 0:
        return PointResiduals(vectors, residual, residual.copy(), np.zeros(0, dtype=bool))
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    mad_floor = max(mad, 0.25)
    z = 0.6745 * (residual - median) / mad_floor
    outlier = z > threshold_mad
    return PointResiduals(vectors, residual, z, outlier)


def apply_outlier_policy(cps: ControlPointSet, residuals: PointResiduals, policy: str) -> ControlPointSet:
    if not residuals.outlier.any():
        return cps
    flagged = [
        f"{cps.ids[i]} (residual={residuals.residual_px[i]:.2f}px, z={residuals.robust_z[i]:.2f})"
        for i in np.nonzero(residuals.outlier)[0]
    ]
    if policy == "reject":
        raise TransformUnreliableError(
            "Control points flagged as outliers by the robust residual test: " + "; ".join(flagged)
        )
    if policy != "warn":
        raise ValueError(f"Unknown outlier_policy: {policy!r}")
    keep = ~residuals.outlier
    return _subset(cps, keep)


@dataclass(frozen=True)
class LinearParams:
    rotation_deg: float
    scale_x: float
    scale_y: float
    translation_at_centroid: tuple[float, float]  # A(centroid) - centroid


def linear_parameters_at_centroid(matrix: np.ndarray, centroid_xy: np.ndarray) -> LinearParams:
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    scale_x = math.hypot(a, c)
    scale_y = math.hypot(b, d)
    rotation_deg = math.degrees(math.atan2(c, a))
    centroid = np.asarray(centroid_xy, dtype=np.float64).reshape(1, 2)
    mapped = _apply_affine(matrix, centroid)[0]
    translation = mapped - centroid[0]
    return LinearParams(rotation_deg, float(scale_x), float(scale_y), (float(translation[0]), float(translation[1])))


def check_bounds(
    rotation_deg: float, scale_x: float, scale_y: float, translation_m: tuple[float, float], bounds,
) -> list[str]:
    """``bounds`` is a TransformValidationConfig (max_translation_m/max_rotation_deg/max_scale_deviation)."""
    violations = []
    magnitude_m = math.hypot(*translation_m)
    if magnitude_m > bounds.max_translation_m:
        violations.append(f"translation {magnitude_m:.2f}m exceeds max {bounds.max_translation_m:.2f}m")
    if abs(rotation_deg) > bounds.max_rotation_deg:
        violations.append(f"rotation {rotation_deg:.2f}deg exceeds max {bounds.max_rotation_deg:.2f}deg")
    if abs(scale_x - 1.0) > bounds.max_scale_deviation or abs(scale_y - 1.0) > bounds.max_scale_deviation:
        violations.append(
            f"scale ({scale_x:.3f}, {scale_y:.3f}) deviates from 1.0 by more than {bounds.max_scale_deviation:.2f}"
        )
    return violations


# --- TPS residual prediction (shared by LOO and model selection) -----------

def _fit_tps_residual(cps: ControlPointSet, smoothing: float):
    from scipy.interpolate import RBFInterpolator
    baseline = fit_linear(cps, GeometricModel.AFFINE)
    predicted = _apply_affine(baseline, cps.tgt_xy)
    residual = cps.ref_xy - predicted
    interp_x = RBFInterpolator(cps.ref_xy, residual[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
    interp_y = RBFInterpolator(cps.ref_xy, residual[:, 1], kernel="thin_plate_spline", smoothing=smoothing)
    return baseline, interp_x, interp_y


def predict_tps(train_cps: ControlPointSet, query_tgt_xy: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """Predict ref-space location of query target points from a TPS fit to train_cps.

    The field is defined as a function of ref-space coordinates (needed for
    inverse warping, where the query point *is* the destination coordinate).
    For a genuine tgt -> ref prediction the true ref location is exactly the
    unknown being predicted, so it cannot be used to query the field
    directly; one fixed-point iteration - querying at the affine baseline's
    own prediction - is used instead. The residual is small relative to the
    grid scale, so a single iteration is an accurate approximation.
    """
    baseline, interp_x, interp_y = _fit_tps_residual(train_cps, smoothing)
    guess = _apply_affine(baseline, query_tgt_xy)
    residual = np.column_stack([interp_x(guess), interp_y(guess)])
    return guess + residual


# --- leave-one-out (P3.4, P3.5) ---------------------------------------------

_LOO_MIN_POINTS = 4


def loo_residual_vectors(control: ControlPointSet, model: GeometricModel, tps_smoothing: float = 0.0) -> np.ndarray | None:
    n = len(control)
    if n < _LOO_MIN_POINTS:
        return None
    vectors = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        subset = _subset(control, keep)
        query_tgt = control.tgt_xy[i:i + 1]
        try:
            if model == GeometricModel.TPS:
                predicted = predict_tps(subset, query_tgt, tps_smoothing)[0]
            else:
                matrix = fit_linear(subset, model)
                predicted = _apply_affine(matrix, query_tgt)[0]
        except (TransformUnreliableError, np.linalg.LinAlgError, ValueError):
            return None
        vectors[i] = control.ref_xy[i] - predicted
    return vectors


def loo_rmse(control: ControlPointSet, model: GeometricModel, tps_smoothing: float = 0.0) -> float | None:
    vectors = loo_residual_vectors(control, model, tps_smoothing)
    if vectors is None:
        return None
    return float(np.sqrt(np.mean(np.sum(vectors ** 2, axis=1))))


# --- model selection (P3.4) --------------------------------------------------

@dataclass(frozen=True)
class ModelDecision:
    model: GeometricModel
    reason: str
    loo_rmse_affine: float | None = None
    loo_rmse_tps: float | None = None


def select_model(
    cps: ControlPointSet, cfg, scipy_available: bool,
    hull_coverage: float | None = None, tps_smoothing: float = 0.0,
) -> ModelDecision:
    """``cfg`` is a ManualAlignmentConfig."""
    control = cps.control
    n = len(control)
    if cfg.model != "auto":
        return ModelDecision(GeometricModel(cfg.model), f"Model forced to {cfg.model!r} by configuration.")
    if n == 1:
        return ModelDecision(GeometricModel.TRANSLATION, "Only one control point: translation is the only fittable model.")
    if n == 2:
        return ModelDecision(GeometricModel.SIMILARITY, "Two control points: similarity is the richest determinable model.")
    if n < cfg.min_tps_points:
        return ModelDecision(
            GeometricModel.AFFINE,
            f"{n} control point(s) is below min_tps_points={cfg.min_tps_points}; using affine.",
        )
    if not scipy_available:
        return ModelDecision(GeometricModel.AFFINE, "SciPy is unavailable and TPS requires SciPy; using affine.")
    if hull_coverage is not None and hull_coverage < cfg.min_hull_coverage:
        return ModelDecision(
            GeometricModel.AFFINE,
            f"Control-point hull covers only {hull_coverage:.1%} of the valid area "
            f"(< {cfg.min_hull_coverage:.1%}); TPS would extrapolate. Using affine.",
        )
    loo_affine = loo_rmse(control, GeometricModel.AFFINE)
    loo_tps = loo_rmse(control, GeometricModel.TPS, tps_smoothing)
    if loo_affine is None or loo_tps is None:
        return ModelDecision(GeometricModel.AFFINE, "Leave-one-out could not be computed; using affine.", loo_affine, loo_tps)
    if loo_tps <= loo_affine * (1.0 - cfg.min_tps_gain):
        return ModelDecision(
            GeometricModel.TPS,
            f"TPS leave-one-out RMSE {loo_tps:.2f}px beats affine's {loo_affine:.2f}px by more than "
            f"{cfg.min_tps_gain:.0%}.",
            loo_affine, loo_tps,
        )
    return ModelDecision(
        GeometricModel.AFFINE,
        f"TPS did not beat affine on leave-one-out (TPS {loo_tps:.2f}px vs affine {loo_affine:.2f}px).",
        loo_affine, loo_tps,
    )


# --- Jacobian gate (P3.5) ----------------------------------------------------

@dataclass(frozen=True)
class JacobianReport:
    min_det: float
    max_gradient: float
    ok: bool


def evaluate_jacobian(field, shape: tuple[int, int], min_det: float, max_gradient: float, lattice: int = 64) -> JacobianReport:
    """Jacobian of xy -> xy + field(xy) on a lattice x lattice grid over the registration valid area."""
    height, width = shape
    ys = np.linspace(0.0, float(max(height - 1, 1)), lattice)
    xs = np.linspace(0.0, float(max(width - 1, 1)), lattice)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    values = field.evaluate(np.column_stack([xx.ravel(), yy.ravel()])).reshape(lattice, lattice, 2)
    dy_step = ys[1] - ys[0] if lattice > 1 else 1.0
    dx_step = xs[1] - xs[0] if lattice > 1 else 1.0
    d_dy = np.gradient(values, dy_step, axis=0)
    d_dx = np.gradient(values, dx_step, axis=1)
    ddx_x, ddx_y = d_dx[..., 0], d_dx[..., 1]
    ddy_x, ddy_y = d_dy[..., 0], d_dy[..., 1]
    det = (1.0 + ddx_x) * (1.0 + ddy_y) - ddx_y * ddy_x
    gradient_norm = np.sqrt(ddx_x ** 2 + ddx_y ** 2 + ddy_x ** 2 + ddy_y ** 2)
    min_det_val = float(np.min(det))
    max_gradient_val = float(np.max(gradient_norm))
    ok = min_det_val >= min_det and max_gradient_val <= max_gradient
    return JacobianReport(min_det_val, max_gradient_val, ok)
