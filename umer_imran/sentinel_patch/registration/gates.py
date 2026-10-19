"""S6 - independent safety gates. Each gate is a pure function returning a ``GateResult``.

A failed gate never raises by itself: the orchestrator records every result in the report and the
first failure supplies the reason code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from sentinel_patch.config.schema import RegistrationConfig
from sentinel_patch.registration.estimators import NccResult

SMALL_SHIFT_PX = 0.2  # below this the holdout only has to be "not worse"
HOLDOUT_NOT_WORSE_TOLERANCE = 0.01
MIN_HOLDOUT_PIXELS = 50


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float | None
    threshold: float
    passed: bool
    reason_code: str
    advisory: bool = False  # recorded and reported, but a failure never rejects the registration

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "threshold": self.threshold, "passed": self.passed,
                "advisory": self.advisory}


def first_failure(results: list[GateResult]) -> GateResult | None:
    return next((result for result in results if not result.passed and not result.advisory), None)


def texture_score(values: np.ndarray, valid: np.ndarray) -> float:
    """Scale-free texture: coefficient of variation of the reference over valid pixels.

    Drone bands are not radiometrically calibrated, so an absolute gradient threshold would mean
    nothing; the spread relative to the mean does not depend on the sensor's units.
    """
    sample = values[valid & np.isfinite(values)].astype(np.float64)
    if sample.size < 2 or abs(sample.mean()) < 1e-12:
        return 0.0
    return float(sample.std() / abs(sample.mean()))


def gate_texture(values: np.ndarray, valid: np.ndarray, cfg: RegistrationConfig) -> GateResult:
    score = texture_score(values, valid)
    return GateResult("G1_texture", score, cfg.min_texture_score, score >= cfg.min_texture_score, "LOW_TEXTURE")


def gate_overlap(valid_pixels: int, cfg: RegistrationConfig) -> GateResult:
    return GateResult("G2_overlap", float(valid_pixels), float(cfg.min_overlap_px), valid_pixels >= cfg.min_overlap_px, "AOI_TOO_SMALL")


def gate_magnitude(dx: float, dy: float, cfg: RegistrationConfig) -> GateResult:
    magnitude = math.hypot(dx, dy)
    return GateResult("G3_magnitude", magnitude, cfg.max_shift_m, magnitude <= cfg.max_shift_m, "SHIFT_IMPLAUSIBLE")


def gate_agreement(ncc_shift_m: tuple[float, float], aro, pixel_size_m: float, cfg: RegistrationConfig) -> GateResult:
    """G4 - does an independent estimator agree? It may only object when it is itself confident.

    ``aro`` is an ``ArosicsResult``. 'skipped' and 'inconclusive' mean it has no opinion, which is a pass:
    the NCC estimate (with its own gates) then decides alone. Unless ``arosics_mode`` is 'veto' the gate is
    advisory: a disagreement is recorded and warned about but does not reject.
    """
    name, reason, advisory = "G4_agreement", "ESTIMATOR_DISAGREEMENT", cfg.arosics_mode != "veto"
    if aro.status != "ok" or aro.shift_m is None:
        return GateResult(name, None, cfg.agreement_tolerance_px, True, reason, advisory)
    difference_px = math.hypot(aro.shift_m[0] - ncc_shift_m[0], aro.shift_m[1] - ncc_shift_m[1]) / pixel_size_m
    return GateResult(name, difference_px, cfg.agreement_tolerance_px, difference_px <= cfg.agreement_tolerance_px,
                      reason, advisory)


def gate_peak(ncc: NccResult, cfg: RegistrationConfig) -> list[GateResult]:
    return [
        GateResult("G5a_ncc_peak", ncc.peak, cfg.min_ncc_peak, ncc.peak >= cfg.min_ncc_peak, "WEAK_PEAK"),
        GateResult("G5b_peak_ratio", ncc.peak_ratio, cfg.min_peak_ratio, ncc.peak_ratio >= cfg.min_peak_ratio, "WEAK_PEAK"),
    ]


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < MIN_HOLDOUT_PIXELS or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def holdout_correlation(
    reference: np.ndarray, target: np.ndarray, valid_ref: np.ndarray, valid_tgt: np.ndarray,
    col_shift: float, row_shift: float,
) -> tuple[float | None, float | None]:
    """Correlation of an independent band pair before and after applying the shift (same pixels for both)."""
    shift = (row_shift, col_shift)
    filled = np.where(valid_tgt & np.isfinite(target), target, 0.0).astype(np.float64)
    shifted = ndimage.shift(filled, shift, order=1, mode="constant", cval=0.0)
    shifted_valid = ndimage.shift((valid_tgt & np.isfinite(target)).astype(np.float64), shift, order=1, mode="constant", cval=0.0) > 0.999
    common = valid_ref & np.isfinite(reference) & valid_tgt & np.isfinite(target) & shifted_valid
    ref = reference[common].astype(np.float64)
    return _pearson(ref, target[common].astype(np.float64)), _pearson(ref, shifted[common])


def gate_holdout(before: float | None, after: float | None, shift_px_magnitude: float, cfg: RegistrationConfig) -> GateResult:
    name = "G6_holdout_band"
    if before is None or after is None:
        return GateResult(name, None, cfg.min_holdout_gain, False, "HOLDOUT_NO_GAIN")
    gain = after - before
    if shift_px_magnitude < SMALL_SHIFT_PX:
        return GateResult(name, gain, -HOLDOUT_NOT_WORSE_TOLERANCE, gain >= -HOLDOUT_NOT_WORSE_TOLERANCE, "HOLDOUT_NO_GAIN")
    return GateResult(name, gain, cfg.min_holdout_gain, gain >= cfg.min_holdout_gain, "HOLDOUT_NO_GAIN")
