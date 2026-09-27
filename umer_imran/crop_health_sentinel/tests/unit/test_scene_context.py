from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer
from crop_health_sentinel.config import load_config
from crop_health_sentinel.errors import GeometryError
from crop_health_sentinel.io.scene import build_scene_context

from ..conftest import write_synthetic_raster


def _write_required_scene(tmp_path, reference_grid) -> None:
    transform, crs = reference_grid["transform"], reference_grid["crs"]
    for name in ("B3.tif", "B4.tif", "B8.tif"):
        write_synthetic_raster(tmp_path / name, np.full((10, 10), 1000, dtype=np.uint16), transform=transform, crs=crs)
    write_synthetic_raster(tmp_path / "B5.tif", np.full((5, 5), 1000, dtype=np.uint16),
        transform=transform * Affine.scale(2, 2), crs=crs)
    write_synthetic_raster(tmp_path / "SCL.tif", np.full((5, 5), 4, dtype=np.uint8),
        transform=transform * Affine.scale(2, 2), crs=crs)


def test_scene_context_uses_bounded_b8_grid(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path, reference_grid)
    to_wgs84 = Transformer.from_crs(reference_grid["crs"], "EPSG:4326", always_xy=True)
    x0, y0 = to_wgs84.transform(500010, 3499990)
    x1, y1 = to_wgs84.transform(500070, 3499930)
    geometry = {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
    context = build_scene_context(tmp_path, geometry,
        tmp_path / "out", sample_scene_metadata, load_config())
    assert context.grid.width <= 10 and context.grid.height <= 10
    assert context.bands.red.shape == (context.grid.height, context.grid.width)
    assert context.bands.scl.dtype == np.uint8
    assert context.bands.cloud_probability is None
    assert "cloud_probability" in context.bands.missing_optional_bands


def test_multiple_feature_collection_is_rejected(tmp_path, reference_grid, sample_scene_metadata) -> None:
    _write_required_scene(tmp_path, reference_grid)
    geometry = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}}, {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}}]}
    with pytest.raises(GeometryError, match="exactly one"):
        build_scene_context(tmp_path, geometry, tmp_path / "out", sample_scene_metadata, load_config())
