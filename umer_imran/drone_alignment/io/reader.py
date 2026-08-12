from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import rasterio
from rasterio.windows import Window
import affine


@dataclass(frozen=True)
class RasterMetadata:
    path: Path
    crs: rasterio.crs.CRS
    transform: affine.Affine
    width: int
    height: int
    band_count: int
    dtype: str
    nodata: Optional[float]
    bounds: tuple[float, float, float, float]  # (left, bottom, right, top)
    gsd: float
    band_descriptions: tuple[Optional[str], ...]
    color_interpretations: tuple[str, ...]
    alpha_band_index: Optional[int]
    spectral_band_indices: tuple[int, ...]


def read_metadata(path: Path) -> RasterMetadata:
    """Read raster spatial metadata without loading pixel data into memory."""
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")

    with rasterio.open(path_obj) as src:
        bounds_tuple = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        # GSD is calculated from the affine transform cell size
        gsd_x = abs(src.transform.a)
        gsd_y = abs(src.transform.e)
        gsd = float(min(gsd_x, gsd_y))

        return RasterMetadata(
            path=path_obj,
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            band_count=src.count,
            dtype=str(src.dtypes[0]),
            nodata=src.nodata,
            bounds=bounds_tuple,
            gsd=gsd,
            band_descriptions=tuple(src.descriptions),
            color_interpretations=tuple(ci.name for ci in src.colorinterp),
            alpha_band_index=next(
                (i for i, ci in enumerate(src.colorinterp, start=1) if ci.name == "alpha"),
                None,
            ),
            spectral_band_indices=tuple(
                i for i, ci in enumerate(src.colorinterp, start=1) if ci.name != "alpha"
            ),
        )


def read_band(path: Path, band_index: int) -> np.ndarray:
    """Read a single 1-indexed band as a 2D NumPy array."""
    path_obj = Path(path).resolve()
    with rasterio.open(path_obj) as src:
        if band_index < 1 or band_index > src.count:
            raise IndexError(f"Band index {band_index} out of range for file with {src.count} bands.")
        return src.read(band_index)


def read_band_with_mask(path: Path, band_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a band and its GDAL validity mask; pixel magnitude never defines validity."""
    path_obj = Path(path).resolve()
    with rasterio.open(path_obj) as src:
        if band_index < 1 or band_index > src.count:
            raise IndexError(f"Band index {band_index} out of range for file with {src.count} bands.")
        values = src.read(band_index)
        # Intersect every advertised validity source. Some GDAL builds do not propagate a
        # float32 alpha band through dataset_mask(), so read an explicit alpha band as well.
        valid = (src.dataset_mask() > 0) & (src.read_masks(band_index) > 0)
        alpha_index = next(
            (i for i, ci in enumerate(src.colorinterp, start=1) if ci.name == "alpha"), None
        )
        if alpha_index is not None:
            valid &= src.read(alpha_index) > 0
        valid &= np.isfinite(values)
        if src.nodata is not None:
            valid &= values != src.nodata
        return values, valid


def read_band_windowed(path: Path, band_index: int, window: Window) -> np.ndarray:
    """Read a single 1-indexed band within a specific Window."""
    path_obj = Path(path).resolve()
    with rasterio.open(path_obj) as src:
        if band_index < 1 or band_index > src.count:
            raise IndexError(f"Band index {band_index} out of range for file with {src.count} bands.")
        return src.read(band_index, window=window)
