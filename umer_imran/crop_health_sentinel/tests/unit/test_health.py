from __future__ import annotations

from crop_health_sentinel.config import load_config
from crop_health_sentinel.io.scene import build_scene_context
from crop_health_sentinel.layer_a_quality import process_quality
from crop_health_sentinel.layer_b_spectral import process_spectral
from crop_health_sentinel.layer_c_health import process_health

from .test_quality import _geometry
from .test_scene_context import _write_required_scene


def test_health_output_uses_configured_index_scores(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path, reference_grid)
    context = build_scene_context(tmp_path, _geometry(reference_grid), tmp_path / "out", sample_scene_metadata, load_config())
    quality = process_quality(context); health = process_health(context, quality, process_spectral(context, quality))
    assert 0 <= health.field_health_score <= 100
    assert health.pixel_score_source == "ndre"
    assert health.pixel_health_class_map.max() in {1, 2, 3, 4, 5}
    assert (tmp_path / "out" / "phase_03" / "scene-1_pixel_health_map.tif").is_file()
