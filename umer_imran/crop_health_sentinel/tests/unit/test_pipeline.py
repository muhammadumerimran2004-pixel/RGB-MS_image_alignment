from __future__ import annotations

from crop_health_sentinel.config import load_config
from crop_health_sentinel.pipeline import run_crop_health_report
from crop_health_sentinel.models import SceneStatus
import numpy as np
import rasterio

from .test_quality import _geometry
from .test_scene_context import _write_required_scene


def test_pipeline_runs_complete_non_rejected_scene(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path / "raw", reference_grid)
    result = run_crop_health_report(tmp_path / "raw", _geometry(reference_grid), tmp_path / "out", sample_scene_metadata, config=load_config())
    assert result.scene_status is SceneStatus.ANALYTICS_READY
    assert result.final_report is not None
    assert (tmp_path / "out" / "crop_polygon.json").is_file()


def test_pipeline_rejects_cloudy_scene_before_later_phases(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path / "raw", reference_grid)
    scl = tmp_path / "raw" / "SCL.tif"
    with rasterio.open(scl, "r+") as dataset:
        dataset.write(np.full((1, 5, 5), 9, dtype=np.uint8))
    result = run_crop_health_report(tmp_path / "raw", _geometry(reference_grid), tmp_path / "out", sample_scene_metadata, config=load_config())
    assert result.scene_status is SceneStatus.REJECT
    assert result.spectral is None and result.final_report is None
    assert not (tmp_path / "out" / "phase_02").exists()
