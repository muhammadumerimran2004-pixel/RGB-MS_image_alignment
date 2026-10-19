"""S5 - AROSICS as an independent cross-check (estimator A).

AROSICS matches in the frequency domain and is designed for large scenes. On the small windows used here,
against a blurred drone reference, it is far noisier than the masked NCC (see ``RegistrationConfig.arosics_mode``),
so it never produces the published number and, by default, does not veto either: a disagreement is reported
and warned about. Everything that can go wrong with it - not installed, window too small, no shift found, low
reliability - is reported as 'skipped' or 'inconclusive' and leaves the NCC estimate to decide alone.

Sign convention verified experimentally: ``x_shift_map`` / ``y_shift_map`` are the vector to add to the
target's origin, which is exactly our ``shift_m``.
"""

from __future__ import annotations

import logging
import math
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS

from sentinel_patch.config.schema import RegistrationConfig

logger = logging.getLogger("sentinel_patch")

NODATA = -9999.0
MIN_WINDOW_PX = 32
MAX_WINDOW_PX = 256


@dataclass(frozen=True)
class ArosicsResult:
    status: str  # "ok" | "skipped" | "inconclusive"
    reason: str | None = None
    shift_m: tuple[float, float] | None = None
    reliability: float | None = None
    window_px: int | None = None

    def as_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "shift_m": self.shift_m,
                "reliability": self.reliability, "window_px": self.window_px}


def is_available() -> bool:
    try:
        import arosics  # noqa: F401
        return True
    except ImportError:
        return False


def _auto_window(rows: np.ndarray, cols: np.ndarray) -> int:
    extent = min(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    return 0 if extent < MIN_WINDOW_PX else min(MAX_WINDOW_PX, 2 ** int(math.floor(math.log2(extent))))


def estimate_shift_arosics(
    reference: np.ndarray, target: np.ndarray, valid_ref: np.ndarray, valid_tgt: np.ndarray,
    transform: Affine, crs: CRS, cfg: RegistrationConfig,
) -> ArosicsResult:
    if cfg.arosics_mode == "off":
        return ArosicsResult("skipped", "AROSICS_DISABLED")
    if not is_available():
        return ArosicsResult("skipped", "AROSICS_NOT_INSTALLED")

    overlap = valid_ref & valid_tgt & np.isfinite(reference) & np.isfinite(target)
    rows, cols = np.nonzero(overlap)
    if rows.size == 0:
        return ArosicsResult("skipped", "NO_OVERLAP")
    window = _auto_window(rows, cols) if cfg.arosics_window_px == "auto" else int(cfg.arosics_window_px)
    if window < MIN_WINDOW_PX:
        return ArosicsResult("skipped", "AROSICS_WINDOW_TOO_SMALL")

    gsd = abs(transform.a)
    centre_col, centre_row = (cols.min() + cols.max() + 1) / 2.0, (rows.min() + rows.max() + 1) / 2.0
    window_point = transform * (centre_col, centre_row)
    max_shift_px = math.ceil(cfg.max_shift_m / gsd) + 1

    ref = np.where(valid_ref & np.isfinite(reference), reference, NODATA).astype(np.float32)
    tgt = np.where(valid_tgt & np.isfinite(target), target, NODATA).astype(np.float32)
    profile = dict(driver="GTiff", dtype="float32", nodata=NODATA, width=ref.shape[1], height=ref.shape[0],
                   count=1, crs=crs, transform=transform)
    try:
        from arosics import COREG
        with tempfile.TemporaryDirectory(prefix="sentinel_patch_arosics_", ignore_cleanup_errors=True) as tmp:
            for name, array in (("ref", ref), ("tgt", tgt)):
                with rasterio.open(Path(tmp) / f"{name}.tif", "w", **profile) as dst:
                    dst.write(array, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coreg = COREG(str(Path(tmp) / "ref.tif"), str(Path(tmp) / "tgt.tif"), ws=(window, window),
                              wp=window_point, max_shift=max_shift_px, nodata=(NODATA, NODATA),
                              ignore_errors=True, v=False, q=True)
                coreg.calculate_spatial_shifts()
            success = bool(getattr(coreg, "success", False))
            shift = (float(coreg.x_shift_map), float(coreg.y_shift_map)) if success else None
            reliability = getattr(coreg, "shift_reliability", None)
    except Exception as exc:  # optional third-party cross-check: any failure means "no opinion", never a crash
        logger.warning("AROSICS cross-check failed and is ignored: %s", exc)
        return ArosicsResult("inconclusive", f"AROSICS_ERROR: {type(exc).__name__}: {exc}", window_px=window)

    if shift is None or not all(math.isfinite(v) for v in shift):
        return ArosicsResult("inconclusive", "AROSICS_NO_SHIFT", window_px=window)
    if reliability is None or not math.isfinite(reliability):
        return ArosicsResult("inconclusive", "AROSICS_NO_RELIABILITY", shift, None, window)
    if reliability < cfg.arosics_min_reliability:
        return ArosicsResult("inconclusive", "AROSICS_LOW_RELIABILITY", shift, float(reliability), window)
    return ArosicsResult("ok", None, shift, float(reliability), window)
