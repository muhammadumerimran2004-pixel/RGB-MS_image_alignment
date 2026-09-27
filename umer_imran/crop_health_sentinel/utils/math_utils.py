from __future__ import annotations

import numpy as np


def safe_divide(numerator, denominator, valid_mask, epsilon: float) -> np.ndarray:
    """Return float32 division, using NaN outside stable valid calculations."""
    result = np.full(np.shape(numerator), np.nan, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(numerator) & np.isfinite(denominator)
    valid &= np.abs(denominator) > epsilon
    np.divide(numerator, denominator, out=result, where=valid, casting="unsafe")
    return result


def normalize_score(values, low: float, high: float) -> np.ndarray:
    if low >= high:
        raise ValueError("normalization low must be less than high")
    values = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(values), np.clip((values - low) / (high - low), 0, 1) * 100, np.nan).astype(np.float32)

