from __future__ import annotations

import numpy as np

from crop_health_sentinel.config import load_config
from crop_health_sentinel.io.scene import build_scene_context
from crop_health_sentinel.layer_a_quality import process_quality
from crop_health_sentinel.layer_b_spectral import process_spectral

from .test_quality import _geometry
from .test_scene_context import _write_required_scene


def test_spectral_indices_use_analytic_mask(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path, reference_grid)
    context = build_scene_context(tmp_path, _geometry(reference_grid), tmp_path / "out", sample_scene_metadata, load_config())
    result = process_spectral(context, process_quality(context))
    assert set(result.indices) == {"ndvi", "ndre", "gndvi", "evi2"}
    assert np.nanmax(np.abs(result.indices["ndvi"])) == 0
    assert result.summary["indices"]["ndvi"]["count"] > 0
    assert (tmp_path / "out" / "phase_02" / "scene-1_ndvi.tif").is_file()
