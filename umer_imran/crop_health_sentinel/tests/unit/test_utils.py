from __future__ import annotations

import json

import numpy as np

from crop_health_sentinel.io.structured_artifacts import write_json
from crop_health_sentinel.utils.math_utils import normalize_score, safe_divide
from crop_health_sentinel.utils.stats import finite_summary


def test_safe_divide_marks_unstable_values_nan() -> None:
    result = safe_divide(np.array([2, 1], dtype=np.float32), np.array([1, 0], dtype=np.float32), np.array([True, True]), 1e-6)
    assert result[0] == 2 and np.isnan(result[1])


def test_normalize_score_and_empty_summary() -> None:
    assert normalize_score(np.array([0.0, 1.0]), 0, 1).tolist() == [0.0, 100.0]
    assert finite_summary(np.array([np.nan]), (5,))["mean"] is None


def test_atomic_json_serializes_numpy(tmp_path) -> None:
    path = write_json(tmp_path / "report.json", {"score": np.float32(1.0)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"score": 1.0}
