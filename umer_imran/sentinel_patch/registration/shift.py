"""The single place that defines and applies the Sentinel shift sign convention.

Convention (blueprint section 6.6):

    ``shift_m = (dx, dy)`` is the vector to ADD to the Sentinel geotransform
    origin so that Sentinel content lands on the matching drone content.
    ``new_transform = Affine(a, b, c + dx, d, e, f + dy)``.
    ``dy`` is in map units (northing): a positive ``dy`` moves the image north.

Estimators report a *pixel* shift ``(col_shift, row_shift)``: how far the
Sentinel content has to move, in pixels on the Sentinel grid, to land on the
reference. ``pixel_shift_to_map`` converts that into ``shift_m``. No other
module may build or flip a shift sign.
"""

from __future__ import annotations

import math

from affine import Affine


def pixel_shift_to_map(transform: Affine, col_shift: float, row_shift: float) -> tuple[float, float]:
    """Convert a pixel shift of the content into a map-unit ``shift_m``.

    Uses the linear part of ``transform`` so it is also correct for grids that
    are not north-up. For a north-up grid (``b = d = 0``, ``e < 0``) this is
    ``dx = a * col_shift`` and ``dy = e * row_shift``: moving content down
    (positive ``row_shift``) moves it south (negative ``dy``).
    """
    dx = transform.a * col_shift + transform.b * row_shift
    dy = transform.d * col_shift + transform.e * row_shift
    return float(dx), float(dy)


def apply_shift(transform: Affine, dx: float, dy: float) -> Affine:
    """Return ``transform`` with its origin moved by ``(dx, dy)`` map units."""
    return Affine(transform.a, transform.b, transform.c + dx, transform.d, transform.e, transform.f + dy)


def shift_magnitude_m(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)
