"""Continuous residual-field fitting for sparse local registration evidence.

These fields operate on registration-grid coordinates and represent only the
residual beyond the already verified global translation.  They are deliberately
separate from raster warping so their behaviour can be tested and reviewed
before any source pixels are resampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import cv2
from scipy.spatial import cKDTree

from drone_alignment.alignment.local_evidence import LocalMatchSample, SparseDisplacementResult


def nearest_control_support(
    points: np.ndarray, control_xy: np.ndarray, sigma_px: float,
    tree: cKDTree | None = None, chunk: int = 262_144,
) -> np.ndarray:
    """Return exp(-d_min^2 / (2*sigma^2)), using each point's *nearest* control.

    A density-summed support (sum of exp(-d_i^2/2*sigma^2) over every
    control) is not 1 at a control point whenever other controls are nearby
    (it can exceed 1), and is smaller wherever controls happen to be sparse -
    i.e. it depends on point *density*, not just distance. Nearest-control
    support is exactly 1 when evaluated at any control point, independent of
    how densely the others are packed nearby.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    active_tree = tree if tree is not None else cKDTree(control_xy)
    sigma = max(1.0, sigma_px)
    support = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        end = start + chunk
        distance, _ = active_tree.query(points[start:end], k=1)
        support[start:end] = np.exp(-(distance ** 2) / (2.0 * sigma ** 2))
    return support


def renormalized_taper(support: np.ndarray, fallback_weight: float) -> np.ndarray:
    """Taper that is exactly 1 where support == 1 (at a control point) and
    decays toward 0 as support -> 0, regardless of fallback_weight.

    The naive ``support / (support + fallback_weight)`` used previously is
    not 1 at support == 1 unless fallback_weight == 0, so the field would
    quietly stop passing through its own control points as soon as any
    fallback blending was enabled.
    """
    if fallback_weight <= 0.0:
        return np.ones_like(support)
    return support * (1.0 + fallback_weight) / (support + fallback_weight)


class ResidualField(Protocol):
    name: str

    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        """Return Nx2 residual vectors for Nx2 registration-grid coordinates."""


class FieldFittingConfig(Protocol):
    """Configuration shared by every sparse residual-evidence collector."""

    field_support_radius_px: float
    field_fallback_weight: float
    rbf_smoothing: float
    max_field_gradient: float
    max_field_loo_rmse_px: float


@dataclass(frozen=True)
class FieldFitMetrics:
    name: str
    leave_one_out_rmse_px: float | None
    max_gradient: float
    mean_gradient: float
    is_gradient_safe: bool


@dataclass(frozen=True)
class FieldComparison:
    metrics: tuple[FieldFitMetrics, ...]
    selected_name: str | None
    reason: str


def _controls(samples: tuple[LocalMatchSample, ...] | list[LocalMatchSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accepted = [sample for sample in samples if sample.status.value == "accepted"]
    if len(accepted) < 3:
        raise ValueError("At least three accepted local vectors are required to fit a displacement field.")
    xy = np.asarray([sample.center_xy for sample in accepted], dtype=np.float64)
    residual = np.asarray([sample.residual_dx_dy for sample in accepted], dtype=np.float64)
    confidence = np.asarray([max(sample.confidence, 0.05) for sample in accepted], dtype=np.float64)
    return xy, residual, confidence


@dataclass(frozen=True)
class RegularizedMeshField:
    """Bilinear field with confidence-weighted IDW node estimates and global taper."""

    x_nodes: np.ndarray
    y_nodes: np.ndarray
    residual_nodes: np.ndarray  # [rows, cols, 2]
    name: str = "regularized_bilinear_mesh"

    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        x, y = points[:, 0], points[:, 1]
        col = np.clip(np.searchsorted(self.x_nodes, x, side="right") - 1, 0, len(self.x_nodes) - 2)
        row = np.clip(np.searchsorted(self.y_nodes, y, side="right") - 1, 0, len(self.y_nodes) - 2)
        x0, x1 = self.x_nodes[col], self.x_nodes[col + 1]
        y0, y1 = self.y_nodes[row], self.y_nodes[row + 1]
        tx = np.clip((x - x0) / np.maximum(x1 - x0, 1e-9), 0.0, 1.0)
        ty = np.clip((y - y0) / np.maximum(y1 - y0, 1e-9), 0.0, 1.0)
        q00 = self.residual_nodes[row, col]
        q10 = self.residual_nodes[row, col + 1]
        q01 = self.residual_nodes[row + 1, col]
        q11 = self.residual_nodes[row + 1, col + 1]
        return ((1 - tx)[:, None] * (1 - ty)[:, None] * q00 + tx[:, None] * (1 - ty)[:, None] * q10 +
                (1 - tx)[:, None] * ty[:, None] * q01 + tx[:, None] * ty[:, None] * q11)


def fit_regularized_mesh_field(
    samples: tuple[LocalMatchSample, ...] | list[LocalMatchSample],
    shape: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
    config: FieldFittingConfig,
) -> RegularizedMeshField:
    xy, residual, confidence = _controls(samples)
    height, width = shape
    # Add nodes at the image boundary.  This makes evaluation well-defined over
    # the whole canvas and allows unsupported edges to decay toward zero.
    x_nodes = np.linspace(0.0, float(width - 1), grid_cols + 1)
    y_nodes = np.linspace(0.0, float(height - 1), grid_rows + 1)
    yy, xx = np.meshgrid(y_nodes, x_nodes, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    distance_sq = ((nodes[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
    local_weight = confidence[None, :] / np.maximum(distance_sq, 1.0)
    support = (confidence[None, :] * np.exp(-distance_sq / (2.0 * config.field_support_radius_px ** 2))).sum(axis=1)
    weighted_residual = local_weight @ residual / np.maximum(local_weight.sum(axis=1, keepdims=True), 1e-12)
    taper = support / (support + config.field_fallback_weight)
    nodes_residual = weighted_residual * taper[:, None]
    return RegularizedMeshField(x_nodes, y_nodes, nodes_residual.reshape(grid_rows + 1, grid_cols + 1, 2))


@dataclass(frozen=True)
class TaperedThinPlateSplineField:
    """Optional SciPy TPS residual field, blended back to global outside support.

    ``confidence`` is retained for reporting/introspection but no longer
    enters the taper (see :func:`nearest_control_support` /
    :func:`renormalized_taper`): confidence weighting already shapes the
    fitted residual through ``_controls()``, and folding it into the taper
    as well made the taper density-dependent rather than purely
    distance-dependent, which broke the "== 1 at a control point" invariant.
    """

    interpolator_x: object
    interpolator_y: object
    control_xy: np.ndarray
    confidence: np.ndarray
    support_radius_px: float
    fallback_weight: float
    name: str = "tapered_thin_plate_spline"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_control_tree", cKDTree(self.control_xy))

    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        values = np.column_stack([self.interpolator_x(points), self.interpolator_y(points)])
        support = nearest_control_support(points, self.control_xy, self.support_radius_px, tree=self._control_tree)
        taper = renormalized_taper(support, self.fallback_weight)
        return values * taper[:, None]


def fit_tapered_thin_plate_spline_field(
    samples: tuple[LocalMatchSample, ...] | list[LocalMatchSample],
    config: FieldFittingConfig,
) -> TaperedThinPlateSplineField:
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise RuntimeError("Thin-plate spline fitting requires SciPy.") from exc
    xy, residual, confidence = _controls(samples)
    # Confidence has no direct RBFInterpolator weighting API.  A smoothing term
    # prevents an individual sparse vector from becoming a hard global bend;
    # evaluate() additionally tapers unsupported areas toward zero residual.
    kwargs = {"kernel": "thin_plate_spline", "smoothing": config.rbf_smoothing}
    return TaperedThinPlateSplineField(
        RBFInterpolator(xy, residual[:, 0], **kwargs),
        RBFInterpolator(xy, residual[:, 1], **kwargs),
        xy, confidence, config.field_support_radius_px, config.field_fallback_weight,
    )


def _gradient_metrics(field: ResidualField, shape: tuple[int, int], max_gradient: float) -> tuple[float, float, bool]:
    height, width = shape
    ys = np.linspace(0.0, float(height - 1), 33)
    xs = np.linspace(0.0, float(width - 1), 33)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    values = field.evaluate(np.column_stack([xx.ravel(), yy.ravel()])).reshape(33, 33, 2)
    dy = np.diff(values, axis=0) / max(1e-9, ys[1] - ys[0])
    dx = np.diff(values, axis=1) / max(1e-9, xs[1] - xs[0])
    gradients = np.concatenate([np.linalg.norm(dy, axis=2).ravel(), np.linalg.norm(dx, axis=2).ravel()])
    return float(np.max(gradients)), float(np.mean(gradients)), bool(np.max(gradients) <= max_gradient)


def _leave_one_out_rmse(
    fitter,
    samples: tuple[LocalMatchSample, ...] | list[LocalMatchSample],
    *args,
) -> float | None:
    accepted = [sample for sample in samples if sample.status.value == "accepted"]
    if len(accepted) < 4:
        return None
    errors: list[float] = []
    for index, sample in enumerate(accepted):
        remaining = accepted[:index] + accepted[index + 1:]
        try:
            field = fitter(remaining, *args)
            predicted = field.evaluate(np.asarray([sample.center_xy]))[0]
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None
        errors.append(float(np.linalg.norm(predicted - np.asarray(sample.residual_dx_dy))))
    return float(np.sqrt(np.mean(np.square(errors))))


def compare_displacement_fields(
    result: SparseDisplacementResult,
    shape: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
    config: FieldFittingConfig,
) -> FieldComparison:
    """Compare candidate fields by LOO vector prediction and dense gradients."""
    samples = result.accepted
    candidates: list[tuple[str, ResidualField, float | None]] = []
    mesh = fit_regularized_mesh_field(samples, shape, grid_rows, grid_cols, config)
    mesh_loo = _leave_one_out_rmse(
        lambda subset, rows, cols, cfg: fit_regularized_mesh_field(subset, shape, rows, cols, cfg),
        samples, grid_rows, grid_cols, config,
    )
    candidates.append((mesh.name, mesh, mesh_loo))
    try:
        tps = fit_tapered_thin_plate_spline_field(samples, config)
        tps_loo = _leave_one_out_rmse(lambda subset, cfg: fit_tapered_thin_plate_spline_field(subset, cfg), samples, config)
        candidates.append((tps.name, tps, tps_loo))
    except RuntimeError:
        pass
    metrics = []
    for name, field, loo in candidates:
        max_grad, mean_grad, safe = _gradient_metrics(field, shape, config.max_field_gradient)
        metrics.append(FieldFitMetrics(name, loo, max_grad, mean_grad, safe))
    viable = [
        item for item in metrics
        if item.is_gradient_safe
        and item.leave_one_out_rmse_px is not None
        and item.leave_one_out_rmse_px <= config.max_field_loo_rmse_px
    ]
    if not viable:
        return FieldComparison(
            tuple(metrics), None,
            "No field passed gradient and absolute leave-one-out requirements.",
        )
    selected = min(viable, key=lambda item: item.leave_one_out_rmse_px)
    return FieldComparison(tuple(metrics), selected.name, "Lowest leave-one-out error among gradient-safe fields.")


def fit_selected_displacement_field(
    selected_name: str,
    samples: tuple[LocalMatchSample, ...] | list[LocalMatchSample],
    shape: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
    config: FieldFittingConfig,
) -> ResidualField:
    """Construct the field named by :func:`compare_displacement_fields`."""
    if selected_name == "regularized_bilinear_mesh":
        return fit_regularized_mesh_field(samples, shape, grid_rows, grid_cols, config)
    if selected_name == "tapered_thin_plate_spline":
        return fit_tapered_thin_plate_spline_field(samples, config)
    raise ValueError(f"Unknown displacement-field selection: {selected_name}")


def warp_registration_band_with_field(
    source_band: np.ndarray,
    source_valid_mask: np.ndarray,
    global_matrix: np.ndarray,
    field: ResidualField,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull-warp one registration-grid band for fast local-field QA only.

    This shares the forward MS->RGB / inverse sampling direction used by the
    tiled native exporter, but materializes only the compact registration grid.
    """
    if global_matrix.shape != (2, 3):
        raise ValueError("Registration-grid field QA currently requires a 2x3 affine global matrix.")
    height, width = source_band.shape
    inverse = cv2.invertAffineTransform(global_matrix.astype(np.float64))
    yy, xx = np.indices((height, width), dtype=np.float32)
    residual = field.evaluate(np.column_stack([xx.ravel(), yy.ravel()])).reshape(height, width, 2)
    adjusted_x = xx - residual[..., 0]
    adjusted_y = yy - residual[..., 1]
    map_x = (inverse[0, 0] * adjusted_x + inverse[0, 1] * adjusted_y + inverse[0, 2]).astype(np.float32)
    map_y = (inverse[1, 0] * adjusted_x + inverse[1, 1] * adjusted_y + inverse[1, 2]).astype(np.float32)
    warped = cv2.remap(source_band, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_valid = cv2.remap(source_valid_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    warped[~warped_valid] = 0
    return warped, warped_valid
