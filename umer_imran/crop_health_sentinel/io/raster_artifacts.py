from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio

from crop_health_sentinel.errors import ArtifactError


def write_single_band_raster(path, values, grid, *, dtype: str, nodata) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values)
    if array.shape != (grid.height, grid.width): raise ArtifactError("Raster artifact shape does not match reference grid.")
    partial = path.with_suffix(path.suffix + ".partial")
    profile = dict(grid.profile)
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    profile.update(driver="GTiff", height=grid.height, width=grid.width, count=1, dtype=dtype,
                   crs=grid.crs, transform=grid.transform, nodata=nodata, compress="deflate")
    if grid.width >= 16 and grid.height >= 16:
        profile["tiled"] = True
    try:
        output = np.where(np.isfinite(array), array, nodata) if np.issubdtype(array.dtype, np.floating) else array
        with rasterio.open(partial, "w", **profile) as dataset: dataset.write(output.astype(dtype), 1)
        with rasterio.open(partial) as check:
            if (check.width, check.height, check.count, check.crs) != (grid.width, grid.height, 1, grid.crs):
                raise ArtifactError("Published raster metadata failed validation.")
        os.replace(partial, path); return path
    except Exception as exc:
        partial.unlink(missing_ok=True)
        if isinstance(exc, ArtifactError): raise
        raise ArtifactError(f"Unable to write raster {path}: {exc}") from exc
