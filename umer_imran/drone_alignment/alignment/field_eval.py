"""Lattice-sampled residual field evaluation.

Evaluating an RBF-based :class:`~drone_alignment.alignment.displacement_field.ResidualField`
densely (once per output pixel, potentially several million times per warp) is
expensive, and a taper-weighted field additionally allocates an
``(n_points, n_controls)`` distance array per call - hundreds of megabytes to
several gigabytes for a native-resolution tile with a few dozen control
points.

:class:`GridSampledField` instead evaluates the exact field once on a coarse
lattice over the registration grid and answers every further query with a
cheap bilinear lookup (`cv2.remap`). This mirrors GDAL's approximate GCP
transformer, which accepts a bounded interpolation error in exchange for not
evaluating the exact (expensive) transform everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from drone_alignment.alignment.displacement_field import ResidualField

# cv2.remap's map arrays must stay well under GDAL/OpenCV's ~32767-per-axis
# limit; evaluate() reshapes each chunk into a single row, so the count is
# what matters, not any one axis.
_MAX_REMAP_POINTS_PER_CALL = 30_000


@dataclass(frozen=True)
class GridSampledField:
    """Bilinear-interpolated lookup over a lattice of an exact field's values.

    The lattice covers ``[0, width] x [0, height]`` in registration-grid
    pixels, with nodes uniformly spaced by ``step_x_px`` / ``step_y_px``
    (which can differ when width != height). ``evaluate()`` never calls the
    wrapped exact field; it only interpolates the cached lattice values.
    """

    lattice_dx: np.ndarray  # [ny, nx] float32
    lattice_dy: np.ndarray  # [ny, nx] float32
    step_x_px: float
    step_y_px: float
    max_interp_error_px: float
    name: str

    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        ny, nx = self.lattice_dx.shape
        # Clamp into the lattice's valid fractional-index range. cv2.remap's
        # BORDER_REPLICATE only extends the *source* border during
        # interpolation; it does not clamp an out-of-range fractional map
        # coordinate on its own, so points slightly outside [0, width] x
        # [0, height] (rounding, a query exactly at the far edge) must be
        # clamped explicitly.
        fx = np.clip((points[:, 0] / self.step_x_px).astype(np.float32), 0.0, nx - 1.0)
        fy = np.clip((points[:, 1] / self.step_y_px).astype(np.float32), 0.0, ny - 1.0)
        dx = np.empty(len(points), dtype=np.float64)
        dy = np.empty(len(points), dtype=np.float64)
        for start in range(0, len(points), _MAX_REMAP_POINTS_PER_CALL):
            end = start + _MAX_REMAP_POINTS_PER_CALL
            map_x = fx[start:end].reshape(1, -1)
            map_y = fy[start:end].reshape(1, -1)
            dx[start:end] = cv2.remap(
                self.lattice_dx, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            ).ravel()
            dy[start:end] = cv2.remap(
                self.lattice_dy, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            ).ravel()
        return np.column_stack([dx, dy])


def _evaluate_on_lattice(
    field: ResidualField, x_nodes: np.ndarray, y_nodes: np.ndarray, chunk: int = 262_144,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(y_nodes, x_nodes, indexing="ij")
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    values = np.empty((len(flat), 2), dtype=np.float64)
    for start in range(0, len(flat), chunk):
        end = start + chunk
        values[start:end] = field.evaluate(flat[start:end])
    shape = xx.shape
    return values[:, 0].reshape(shape).astype(np.float32), values[:, 1].reshape(shape).astype(np.float32)


def _measure_max_error(
    field: ResidualField, lattice: GridSampledField, width: int, height: int, probe: int = 64,
) -> float:
    """Worst-case interpolation error at a probe x probe grid of query points.

    Deliberately off-lattice (linspace over the same domain, generally not
    landing exactly on lattice nodes) so this measures genuine interpolation
    error rather than trivially re-reading the cached values.
    """
    if width <= 0 or height <= 0:
        return 0.0
    xs = np.linspace(0.0, float(width), min(probe, max(width, 1)))
    ys = np.linspace(0.0, float(height), min(probe, max(height, 1)))
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    exact = field.evaluate(points)
    approx = lattice.evaluate(points)
    return float(np.max(np.linalg.norm(exact - approx, axis=1)))


def sample_field_on_lattice(
    field: ResidualField,
    registration_shape: tuple[int, int],
    step_px: float = 4.0,
    max_error_px: float = 0.05,
    max_refinements: int = 2,
) -> GridSampledField:
    """Evaluate ``field`` once on a lattice, refining the spacing until the
    interpolation error at a probe grid is within ``max_error_px`` (or
    ``max_refinements`` is exhausted, in which case the finest lattice tried
    is returned with its measured error recorded honestly rather than
    silently accepted).
    """
    height, width = registration_shape
    current_step = float(step_px)
    lattice: GridSampledField | None = None
    for _ in range(max_refinements + 1):
        nx = max(2, math.ceil(width / current_step) + 1)
        ny = max(2, math.ceil(height / current_step) + 1)
        x_nodes = np.linspace(0.0, float(width), nx)
        y_nodes = np.linspace(0.0, float(height), ny)
        dx, dy = _evaluate_on_lattice(field, x_nodes, y_nodes)
        step_x_actual = width / (nx - 1)
        step_y_actual = height / (ny - 1)
        candidate = GridSampledField(
            dx, dy, step_x_actual, step_y_actual, 0.0, getattr(field, "name", "field"),
        )
        error = _measure_max_error(field, candidate, width, height)
        lattice = GridSampledField(
            candidate.lattice_dx, candidate.lattice_dy, step_x_actual, step_y_actual,
            error, candidate.name,
        )
        if error <= max_error_px:
            return lattice
        current_step /= 2.0
    return lattice


def ensure_grid_sampled(
    field: ResidualField,
    registration_shape: tuple[int, int],
    step_px: float = 4.0,
    max_error_px: float = 0.05,
) -> ResidualField:
    """Wrap ``field`` in a :class:`GridSampledField` unless it already is one."""
    if isinstance(field, GridSampledField):
        return field
    return sample_field_on_lattice(field, registration_shape, step_px, max_error_px)
