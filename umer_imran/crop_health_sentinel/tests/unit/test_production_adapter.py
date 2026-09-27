from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from crop_health_sentinel.sentinel_crop_health_report import generate_sentinel_crop_health_report, get_sentinel_crop_health_report_metadata

from .test_quality import _geometry
from .test_scene_context import _write_required_scene


def test_adapter_publishes_only_public_json_and_one_time_metadata(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path / "raw", reference_grid)
    final = Path(generate_sentinel_crop_health_report(tmp_path / "raw", _geometry(reference_grid), tmp_path / "report", sample_scene_metadata))
    payload = json.loads(final.read_text(encoding="utf-8"))
    assert final.is_absolute() and payload and set(payload[0]) == {"h3Index", "healthScore"}
    assert get_sentinel_crop_health_report_metadata(final)["sceneStatus"] == "analytics_ready"
    assert get_sentinel_crop_health_report_metadata(final) is None
    assert not (tmp_path / "report" / ".crop-health-staging").exists() or not any((tmp_path / "report" / ".crop-health-staging").iterdir())


def test_adapter_publishes_empty_list_for_rejected_scene(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path / "raw", reference_grid)
    with rasterio.open(tmp_path / "raw" / "SCL.tif", "r+") as dataset:
        dataset.write(np.full((1, 5, 5), 9, dtype=np.uint8))
    final = Path(generate_sentinel_crop_health_report(tmp_path / "raw", _geometry(reference_grid), tmp_path / "report", sample_scene_metadata))
    assert json.loads(final.read_text(encoding="utf-8")) == []
    assert get_sentinel_crop_health_report_metadata(final)["sceneStatus"] == "reject"
