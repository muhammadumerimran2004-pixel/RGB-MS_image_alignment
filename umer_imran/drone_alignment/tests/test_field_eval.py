import numpy as np
import pytest

from drone_alignment.alignment.field_eval import GridSampledField, sample_field_on_lattice, ensure_grid_sampled
from drone_alignment.alignment.manual import ManualThinPlateSplineField


class _SinusoidalField:
    """A synthetic high-curvature field, only for exercising the refinement loop."""

    name = "sine_probe"

    def __init__(self, amplitude: float, wavelength: float):
        self.amplitude = amplitude
        self.wavelength = wavelength

    def evaluate(self, xy):
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        dx = self.amplitude * np.sin(2 * np.pi * points[:, 0] / self.wavelength)
        dy = np.zeros(len(points))
        return np.column_stack([dx, dy])


def _smooth_tps_field() -> ManualThinPlateSplineField:
    from scipy.interpolate import RBFInterpolator

    rng = np.random.default_rng(0)
    control_xy = np.array([[10.0, 10.0], [100.0, 20.0], [50.0, 100.0], [120.0, 120.0], [20.0, 90.0], [90.0, 60.0]])
    residual = rng.uniform(-3.0, 3.0, size=(len(control_xy), 2))
    interp_x = RBFInterpolator(control_xy, residual[:, 0], kernel="thin_plate_spline", smoothing=0.5)
    interp_y = RBFInterpolator(control_xy, residual[:, 1], kernel="thin_plate_spline", smoothing=0.5)
    return ManualThinPlateSplineField(interp_x, interp_y, control_xy)


def test_grid_sampled_matches_exact_tps_within_tolerance():
    field = _smooth_tps_field()
    lattice = sample_field_on_lattice(field, (128, 128), step_px=4.0, max_error_px=0.05, max_refinements=2)

    assert lattice.step_x_px == pytest.approx(4.0)
    assert lattice.max_interp_error_px <= 0.05

    probe = np.array([[13.0, 27.0], [64.0, 64.0], [110.0, 5.0], [3.0, 119.0]])
    exact = field.evaluate(probe)
    approx = lattice.evaluate(probe)
    np.testing.assert_allclose(approx, exact, atol=0.05)


def test_grid_sampled_refines_step_when_error_high():
    # wavelength=16 with an initial step of 4 (4 samples/period) is too
    # coarse for linear interpolation to stay under 0.05 px; refining to
    # step=1 (16 samples/period) brings it back within tolerance.
    field = _SinusoidalField(amplitude=2.0, wavelength=16.0)
    lattice = sample_field_on_lattice(field, (128, 128), step_px=4.0, max_error_px=0.05, max_refinements=2)

    assert lattice.step_x_px < 4.0
    assert lattice.max_interp_error_px <= 0.05


def test_grid_sampled_evaluate_large_input_chunked():
    field = _smooth_tps_field()
    lattice = sample_field_on_lattice(field, (128, 128), step_px=4.0, max_error_px=0.05, max_refinements=2)

    # Larger than cv2.remap's single-call map-size comfort zone and larger
    # than the internal chunk size, forcing evaluate() through multiple
    # remap calls whose results must still be assembled in order.
    n = 75_000
    rng = np.random.default_rng(1)
    points = rng.uniform(0.0, 127.0, size=(n, 2))

    values = lattice.evaluate(points)

    assert values.shape == (n, 2)
    assert np.isfinite(values).all()
    # Spot-check a handful of individual points against a fresh single-point
    # call, to confirm chunking doesn't corrupt ordering.
    for index in (0, 1, n // 2, n - 1):
        single = lattice.evaluate(points[index:index + 1])
        # float32-level tolerance: cv2.remap's internal buffering differs
        # between a full-chunk call and a single-point call.
        np.testing.assert_allclose(values[index], single[0], atol=1e-5)


def test_ensure_grid_sampled_is_idempotent():
    field = _smooth_tps_field()
    wrapped_once = ensure_grid_sampled(field, (128, 128))
    assert isinstance(wrapped_once, GridSampledField)

    wrapped_twice = ensure_grid_sampled(wrapped_once, (128, 128))
    assert wrapped_twice is wrapped_once
