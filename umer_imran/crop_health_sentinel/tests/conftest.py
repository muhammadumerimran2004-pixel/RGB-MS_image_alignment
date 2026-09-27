"""Shared deterministic fixtures for Sentinel crop-health tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS


@pytest.fixture
def sample_scene_metadata() -> dict[str, str]:
    return {"farmId": "farm-1", "cropId": "crop-1", "imageryDate": "2026-08-19", "sceneId": "scene-1"}


@pytest.fixture
def reference_grid() -> dict[str, object]:
    return {
        "shape": (10, 10),
        "transform": Affine(10, 0, 500000, 0, -10, 3500000),
        "crs": CRS.from_epsg(32642),
        "data": np.arange(100, dtype=np.uint16).reshape(10, 10),
    }


def write_synthetic_raster(path: Path, data: np.ndarray, *, transform: Affine, crs: CRS) -> Path:
    """Write a one-band GeoTIFF used by later I/O phases."""
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dataset:
        dataset.write(data, 1)
    return path

