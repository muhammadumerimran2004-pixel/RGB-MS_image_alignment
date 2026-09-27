from __future__ import annotations

from typing import Any

import numpy as np


def finite_summary(values, percentiles: tuple[float, ...]) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "cv": None, "percentiles": {str(p): None for p in percentiles}}
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=0))
    return {"count": int(finite.size), "mean": mean, "std": std, "min": float(np.min(finite)), "max": float(np.max(finite)),
            "cv": float(std / (abs(mean) + 1e-6)), "percentiles": {str(p): float(v) for p, v in zip(percentiles, np.percentile(finite, percentiles))}}


def health_class(score: float, thresholds) -> str:
    if score >= thresholds.very_good: return "Very Good"
    if score >= thresholds.good: return "Good"
    if score >= thresholds.moderate: return "Moderate"
    if score >= thresholds.weak: return "Weak"
    return "Poor"
