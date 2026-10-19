"""G7 - is the offset one translation? Estimate the shift separately in each quarter of the overlap.

Sentinel-2 products are orthorectified, so the residual error against a drone map should be a single
translation. Quarters that disagree with the global shift point to something else (a rotated or scaled
drone map, a bad tie-in) that one shift cannot fix, and the patch is left unshifted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sentinel_patch.config.schema import RegistrationConfig
from sentinel_patch.errors import RegistrationRejected
from sentinel_patch.registration import gates
from sentinel_patch.registration.estimators import estimate_shift_ncc

NAME = "G7_quadrants"
REASON = "NON_TRANSLATIONAL"
MIN_QUADRANTS = 2


@dataclass(frozen=True)
class QuadrantReport:
    evaluated: int
    deviations_px: list[float] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def check_quadrants(
    reference: np.ndarray, target: np.ndarray, valid_ref: np.ndarray, valid_tgt: np.ndarray,
    global_shift_px: tuple[float, float], cfg: RegistrationConfig,
) -> tuple[gates.GateResult, QuadrantReport]:
    """Return the gate and the per-quadrant evidence. Not enough usable quarters means 'not evaluated', a pass."""
    def not_evaluated(reason: str, report: QuadrantReport | None = None):
        return gates.GateResult(NAME, None, cfg.quadrant_tolerance_px, True, REASON), report or QuadrantReport(0, [], [reason])

    overlap = valid_ref & valid_tgt & np.isfinite(reference) & np.isfinite(target)
    rows, cols = np.nonzero(overlap)
    if rows.size == 0:
        return not_evaluated("NO_OVERLAP")
    top, bottom, left, right = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    if min(bottom - top, right - left) < cfg.quadrant_check_min_px:
        return not_evaluated("WINDOW_TOO_SMALL")

    mid_row, mid_col = (top + bottom) // 2, (left + right) // 2
    deviations: list[float] = []
    skipped: list[str] = []
    for name, (r0, r1, c0, c1) in {"NW": (top, mid_row, left, mid_col), "NE": (top, mid_row, mid_col, right),
                                  "SW": (mid_row, bottom, left, mid_col), "SE": (mid_row, bottom, mid_col, right)}.items():
        quarter = np.zeros_like(valid_ref)
        quarter[r0:r1, c0:c1] = True
        quarter &= valid_ref
        if int((quarter & valid_tgt).sum()) < cfg.min_overlap_px:
            skipped.append(f"{name}:too_little_overlap")
            continue
        if gates.texture_score(reference, quarter) < cfg.min_texture_score:
            skipped.append(f"{name}:low_texture")
            continue
        try:
            local = estimate_shift_ncc(reference, target, quarter, valid_tgt, cfg)
        except RegistrationRejected as exc:
            if exc.reason_code == "PEAK_AT_SEARCH_BORDER":  # this quarter wants a far larger shift than the rest
                deviations.append(math.inf)
            else:
                skipped.append(f"{name}:{exc.reason_code}")
            continue
        deviations.append(math.hypot(local.col_shift_px - global_shift_px[0], local.row_shift_px - global_shift_px[1]))

    report = QuadrantReport(len(deviations), deviations, skipped)
    if len(deviations) < MIN_QUADRANTS:
        return not_evaluated("TOO_FEW_USABLE_QUADRANTS", report)
    worst = max(deviations)
    value = None if math.isinf(worst) else worst
    return gates.GateResult(NAME, value, cfg.quadrant_tolerance_px, worst <= cfg.quadrant_tolerance_px, REASON), report
