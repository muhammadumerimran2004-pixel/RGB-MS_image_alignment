from __future__ import annotations

from crop_health_sentinel.config import load_config
from crop_health_sentinel.io.scene import build_scene_context
from crop_health_sentinel.layer_a_quality import process_quality
from crop_health_sentinel.layer_b_spectral import process_spectral
from crop_health_sentinel.layer_c_health import process_health
from crop_health_sentinel.layer_d_report import process_final_report
from crop_health_sentinel.layer_e_h3 import process_h3

from .test_quality import _geometry
from .test_scene_context import _write_required_scene


def test_final_report_uses_typed_stage_outputs(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path, reference_grid)
    context = build_scene_context(tmp_path, _geometry(reference_grid), tmp_path / "out", sample_scene_metadata, load_config())
    quality = process_quality(context); spectral = process_spectral(context, quality); health = process_health(context, quality, spectral); h3 = process_h3(context, quality, health)
    report = process_final_report(context, quality, spectral, health, h3)
    assert report.report["algorithmVersion"] == "sentinel-crop-health/1.0.0"
    assert report.summary["emittedH3Cells"] == len(h3.cells)
    assert report.report_path.is_file()
