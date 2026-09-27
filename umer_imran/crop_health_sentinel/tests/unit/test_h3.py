from __future__ import annotations

from crop_health_sentinel.config import load_config
from crop_health_sentinel.io.scene import build_scene_context
from crop_health_sentinel.layer_a_quality import process_quality
from crop_health_sentinel.layer_b_spectral import process_spectral
from crop_health_sentinel.layer_c_health import process_health
from crop_health_sentinel.layer_e_h3 import process_h3
from crop_health_sentinel.layer_e_h3 import preflight_h3_capacity
from crop_health_sentinel.errors import H3CapacityError

from .test_quality import _geometry
from .test_scene_context import _write_required_scene


def test_h3_emits_sorted_exact_area_cells(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path, reference_grid)
    context = build_scene_context(tmp_path, _geometry(reference_grid), tmp_path / "out", sample_scene_metadata, load_config())
    quality = process_quality(context); spectral = process_spectral(context, quality); health = process_health(context, quality, spectral)
    output = process_h3(context, quality, health)
    assert output.cells
    assert list(cell.h3_index for cell in output.cells) == sorted(cell.h3_index for cell in output.cells)
    assert all(0 <= cell.mean_health <= 100 for cell in output.cells)


def test_h3_capacity_limit_fails_closed(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path, reference_grid)
    config = load_config(); config = config.model_copy(update={"h3": config.h3.model_copy(update={"max_analytic_pixels": 1})})
    context = build_scene_context(tmp_path, _geometry(reference_grid), tmp_path / "out", sample_scene_metadata, config)
    with __import__("pytest").raises(H3CapacityError):
        preflight_h3_capacity(context, process_quality(context))
