"""S5 - shift estimation by masked normalised cross-correlation (estimator B).

Convention (see ``shift.py``): the result is the pixel shift ``(col_shift, row_shift)`` that the
*target* (Sentinel) content must undergo to land on the reference. Both images are compared only
where both are valid, so clouds and no-data never contribute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel_patch.config.schema import RegistrationConfig
from sentinel_patch.errors import RegistrationRejected

PEAK_RATIO_CAP = 1000.0
MIN_STD = 1e-12


@dataclass(frozen=True)
class NccResult:
    col_shift_px: float
    row_shift_px: float
    integer_shift_px: tuple[int, int]  # (col, row) of the surface maximum
    peak: float
    second_peak: float | None  # best distinct local maximum, None when the peak is unambiguous
    peak_ratio: float
    overlap_px: int
    surface: np.ndarray  # [row_shift + N, col_shift + N]


def _slices(shape: tuple[int, int], row_shift: int, col_shift: int) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Overlap of reference and shifted target: ref[i, j] is compared with target[i - row_shift, j - col_shift]."""
    height, width = shape
    ref = (slice(max(0, row_shift), height + min(0, row_shift)), slice(max(0, col_shift), width + min(0, col_shift)))
    tgt = (slice(max(0, -row_shift), height + min(0, -row_shift)), slice(max(0, -col_shift), width + min(0, -col_shift)))
    return ref, tgt


def ncc_surface(
    reference: np.ndarray, target: np.ndarray, valid_ref: np.ndarray, valid_tgt: np.ndarray,
    search_px: int, min_overlap_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation for every integer shift in ``[-search_px, search_px]^2`` plus the overlap counts."""
    size = 2 * search_px + 1
    surface = np.full((size, size), np.nan)
    counts = np.zeros((size, size), dtype=int)
    for row_shift in range(-search_px, search_px + 1):
        for col_shift in range(-search_px, search_px + 1):
            ref_slice, tgt_slice = _slices(reference.shape, row_shift, col_shift)
            both = valid_ref[ref_slice] & valid_tgt[tgt_slice]
            count = int(both.sum())
            counts[row_shift + search_px, col_shift + search_px] = count
            if count < min_overlap_px:
                continue
            a = reference[ref_slice][both].astype(np.float64)
            b = target[tgt_slice][both].astype(np.float64)
            std_a, std_b = a.std(), b.std()
            if std_a < MIN_STD or std_b < MIN_STD:
                continue
            surface[row_shift + search_px, col_shift + search_px] = ((a - a.mean()) * (b - b.mean())).mean() / (std_a * std_b)
    return surface, counts


def _strict_local_maxima(surface: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(surface), surface, -np.inf)
    padded = np.pad(filled, 1, constant_values=-np.inf)
    is_max = np.isfinite(filled)
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue
            neighbour = padded[1 + d_row: 1 + d_row + filled.shape[0], 1 + d_col: 1 + d_col + filled.shape[1]]
            is_max &= filled > neighbour
    return is_max


def _subpixel_offset(surface: np.ndarray, row: int, col: int) -> tuple[float, float]:
    """Least-squares 2-D quadratic over the 3x3 neighbourhood; returns (col_offset, row_offset) in pixels."""
    if row < 1 or col < 1 or row > surface.shape[0] - 2 or col > surface.shape[1] - 2:
        return 0.0, 0.0
    patch = surface[row - 1: row + 2, col - 1: col + 2]
    if not np.isfinite(patch).all():
        return 0.0, 0.0
    y, x = np.mgrid[-1:2, -1:2]
    design = np.column_stack([np.ones(9), x.ravel(), y.ravel(), x.ravel() ** 2, y.ravel() ** 2, x.ravel() * y.ravel()])
    _, b, c, d, e, g = np.linalg.lstsq(design, patch.ravel(), rcond=None)[0]
    hessian = np.array([[2 * d, g], [g, 2 * e]])
    if not (hessian[0, 0] < 0 and np.linalg.det(hessian) > 0):
        return 0.0, 0.0
    offset = -np.linalg.solve(hessian, np.array([b, c]))
    if np.any(np.abs(offset) > 1.0):
        return 0.0, 0.0
    return float(offset[0]), float(offset[1])


def estimate_shift_ncc(
    reference: np.ndarray, target: np.ndarray, valid_ref: np.ndarray, valid_tgt: np.ndarray, cfg: RegistrationConfig,
) -> NccResult:
    """Best shift of ``target`` onto ``reference``; raises ``RegistrationRejected`` when none can be trusted."""
    valid_ref = valid_ref & np.isfinite(reference)
    valid_tgt = valid_tgt & np.isfinite(target)
    surface, counts = ncc_surface(reference, target, valid_ref, valid_tgt, cfg.ncc_search_px, cfg.min_overlap_px)
    if not np.isfinite(surface).any():
        raise RegistrationRejected(
            "AOI_TOO_SMALL", "No shift had enough overlapping valid pixels to correlate.",
            {"max_overlap_px": int(counts.max()), "min_overlap_px": cfg.min_overlap_px},
        )
    n = cfg.ncc_search_px
    row, col = np.unravel_index(int(np.nanargmax(surface)), surface.shape)
    if row in (0, 2 * n) or col in (0, 2 * n):
        raise RegistrationRejected(
            "PEAK_AT_SEARCH_BORDER", "The correlation maximum lies on the edge of the search area.",
            {"integer_shift_px": [int(col) - n, int(row) - n], "search_px": n},
        )
    peak = float(surface[row, col])
    maxima = _strict_local_maxima(surface)
    maxima[row, col] = False
    second = float(surface[maxima].max()) if maxima.any() else None
    ratio = PEAK_RATIO_CAP if second is None or second <= 0 else min(PEAK_RATIO_CAP, peak / second)
    col_off, row_off = _subpixel_offset(surface, row, col)
    return NccResult(
        col_shift_px=col - n + col_off, row_shift_px=row - n + row_off,
        integer_shift_px=(col - n, row - n), peak=peak, second_peak=second, peak_ratio=ratio,
        overlap_px=int(counts[row, col]), surface=surface,
    )
