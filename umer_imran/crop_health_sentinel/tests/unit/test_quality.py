from __future__ import annotations

from crop_health_sentinel.config import load_config
from crop_health_sentinel.layer_a_quality import process_quality
from crop_health_sentinel.models import QualityProvenance, SceneStatus
from crop_health_sentinel.io.scene import build_scene_context

from .test_scene_context import _write_required_scene


def _geometry(reference_grid):
    from pyproj import Transformer
    transform = Transformer.from_crs(reference_grid["crs"], "EPSG:4326", always_xy=True)
    x0, y0 = transform.transform(500000, 3500000); x1, y1 = transform.transform(500100, 3499900)
    return {"type": "Polygon", "coordinates": [[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]}


def test_quality_marks_clear_required_scene_ready(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path, reference_grid)
    context = build_scene_context(tmp_path, _geometry(reference_grid), tmp_path / "out", sample_scene_metadata, load_config())
    result = process_quality(context)
    assert result.scene_status is SceneStatus.ANALYTICS_READY
    assert result.quality_provenance is QualityProvenance.DEGRADED
    assert result.analytic_valid_mask.any()
    assert (tmp_path / "out" / "phase_01" / "scene-1_layer_a_quality_summary.json").is_file()

