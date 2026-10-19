"""The only place that converts Sentinel-2 L2A digital numbers to reflectance.

Processing baseline 04.00 and later (since 25 January 2022) adds a
``BOA_ADD_OFFSET`` of -1000 DN, i.e. ``reflectance = DN * 0.0001 - 0.1``.
Applying it twice, or not at all, silently shifts NDVI by about 0.1, so every
caller goes through ``resolve_radiometry`` + ``dn_to_reflectance``.

Declared per-band scale/offset from the catalogue item (STAC ``raster:bands``)
is authoritative. The processing baseline is the fallback. If neither is
available the conversion is refused rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from sentinel_patch.errors import RadiometryUnknownError

DEFAULT_SCALE = 1.0e-4
BASELINE_OFFSET_FROM_MAJOR = 4
BASELINE_OFFSET = -0.1

_BASELINE_PATTERNS = (
    re.compile(r"^(\d{2})\.(\d{2})$"),   # "05.11" (STAC s2:processing_baseline)
    re.compile(r"^N(\d{2})(\d{2})$"),    # "N0511"  (ESA product names)
)


@dataclass(frozen=True)
class RadiometryParams:
    scale: float
    offset: float
    source: str  # "stac_raster_bands" | "processing_baseline"
    warnings: tuple[str, ...] = ()


def parse_processing_baseline(value: str | float | None) -> tuple[int, int] | None:
    """Parse "05.11", "N0511" or 4.0 into ``(major, minor)``; ``None`` if absent or unreadable."""
    if value is None:
        return None
    text = f"{value:05.2f}" if isinstance(value, float) else str(value).strip()
    for pattern in _BASELINE_PATTERNS:
        match = pattern.match(text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def baseline_offset(baseline: tuple[int, int]) -> float:
    return BASELINE_OFFSET if baseline[0] >= BASELINE_OFFSET_FROM_MAJOR else 0.0


def resolve_radiometry(
    *,
    scale: float | None = None,
    offset: float | None = None,
    processing_baseline: str | float | None = None,
) -> RadiometryParams:
    """Decide scale/offset for one band. Raises ``RadiometryUnknownError`` rather than guess."""
    baseline = parse_processing_baseline(processing_baseline)
    if (scale is None) != (offset is None):
        raise RadiometryUnknownError("Catalogue metadata must declare scale and offset together.")
    if scale is not None and offset is not None:
        if scale <= 0:
            raise RadiometryUnknownError(f"Declared reflectance scale must be positive, got {scale}.")
        warnings: tuple[str, ...] = ()
        if baseline is not None and not np.isclose(offset, baseline_offset(baseline)):
            warnings = ("RADIOMETRY_OFFSET_BASELINE_MISMATCH",)
        return RadiometryParams(float(scale), float(offset), "stac_raster_bands", warnings)
    if baseline is None:
        raise RadiometryUnknownError(
            "No declared scale/offset and no readable processing baseline "
            f"({processing_baseline!r}); refusing to guess the reflectance offset."
        )
    return RadiometryParams(DEFAULT_SCALE, baseline_offset(baseline), "processing_baseline")


def dn_to_reflectance(dn: np.ndarray, params: RadiometryParams, nodata: int = 0) -> np.ndarray:
    """Convert DN to float32 reflectance, ``NaN`` where ``dn == nodata``.

    Values are not clipped: DN below the offset legitimately gives small
    negative reflectance, which downstream index code must handle.
    """
    out = dn.astype(np.float32) * np.float32(params.scale) + np.float32(params.offset)
    out[dn == nodata] = np.nan
    return out
