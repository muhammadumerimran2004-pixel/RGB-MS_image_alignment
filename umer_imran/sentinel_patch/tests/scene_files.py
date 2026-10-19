"""Write synthetic drone rasters and a Sentinel scene folder to disk for file-level tests.

Truth model: the drone rasters are sampled from the analytic scene at their true positions. The
Sentinel files declare the standard grid but their content is displaced by ``content_offset_m``, so
the registration is expected to recover ``shift_m == -content_offset_m``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import ColorInterp

from sentinel_patch.io.radiometry import baseline_offset, parse_processing_baseline

from . import synthetic

CRS_UTM42 = CRS.from_epsg(32642)
S2_SHAPE_10M = (160, 160)  # 1.6 km
S2_SHAPE_20M = (80, 80)
S2_20M_BANDS = ("B05", "B06", "B07", "B8A", "B11", "B12")

# Band response as a function of the underlying scene value f; every sensor sees the same structure.
S2_BAND_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "B02": lambda f: 0.05 + 0.04 * f, "B03": lambda f: 0.12 - 0.10 * f, "B04": lambda f: 0.20 - 0.25 * f,
    "B05": lambda f: 0.18 - 0.12 * f, "B06": lambda f: 0.10 + 0.50 * f, "B07": lambda f: 0.12 + 0.60 * f,
    "B08": lambda f: f, "B8A": lambda f: 0.02 + 0.95 * f, "B11": lambda f: 0.25 - 0.20 * f,
    "B12": lambda f: 0.15 - 0.12 * f,
}
# The drone camera has its own gain and offset per band (uncalibrated digital values).
DRONE_BANDS = ("green", "red", "red_edge", "nir")
DRONE_FUNCTIONS = {
    "green": lambda f: 3.0 * S2_BAND_FUNCTIONS["B03"](f) + 0.1, "red": lambda f: 3.0 * S2_BAND_FUNCTIONS["B04"](f) + 0.1,
    "red_edge": lambda f: 3.0 * S2_BAND_FUNCTIONS["B05"](f) + 0.1, "nir": lambda f: 3.0 * f + 0.1,
}
MS_BAND_MAP = {name: index for index, name in enumerate(DRONE_BANDS, start=1)}


def s2_grid_transform() -> Affine:
    return synthetic.make_s2_transform()


def full_scene(blobs: int = 150, seed: int = 7):
    return synthetic.make_scene(s2_grid_transform(), S2_SHAPE_10M, seed=seed, blobs=blobs)


def write_drone_rasters(
    folder: Path, scene, *, size_px: int = 600, gsd: float = 1.0, origin: tuple[float, float] = (500500.0, 3499500.0),
    ms_collar_m: float = 20.0, rgb_collar_m: float = 40.0, crs: CRS = CRS_UTM42,
) -> tuple[Path, Path]:
    """Aligned MS (4 float bands, nodata 0) and RGBA orthophoto on the same 1 m grid with different collars.

    ``origin`` is the top-left corner in the SCENE's CRS (EPSG:32642). With a different ``crs`` the same ground is
    covered by a north-up grid in that CRS (rotated relative to the scene grid by the zones' convergence).
    """
    folder.mkdir(parents=True, exist_ok=True)
    shape = (size_px, size_px)
    if crs == CRS_UTM42:
        transform = Affine(gsd, 0.0, origin[0], 0.0, -gsd, origin[1])
        base = synthetic.sample_scene(scene, transform, shape).astype(np.float64)
    else:
        half = size_px * gsd / 2.0
        centre = Transformer.from_crs(CRS_UTM42, crs, always_xy=True).transform(origin[0] + half, origin[1] - half)
        transform = Affine(gsd, 0.0, centre[0] - half, 0.0, -gsd, centre[1] + half)
        x, y = synthetic.pixel_centre_xy(transform, shape)
        scene_x, scene_y = Transformer.from_crs(crs, CRS_UTM42, always_xy=True).transform(x, y)
        base = np.asarray(scene(scene_x, scene_y), dtype=np.float32).astype(np.float64)

    def inside(collar_m: float) -> np.ndarray:
        edge = round(collar_m / gsd)
        mask = np.zeros(shape, bool)
        mask[edge: size_px - edge, edge: size_px - edge] = True
        return mask

    ms_valid, rgb_valid = inside(ms_collar_m), inside(rgb_collar_m)
    ms_path, rgb_path = folder / "ms_aligned.tif", folder / "rgb.tif"
    with rasterio.open(ms_path, "w", driver="GTiff", height=size_px, width=size_px, count=len(DRONE_BANDS),
                       dtype="float32", crs=crs, transform=transform, nodata=0.0) as dst:
        for index, name in enumerate(DRONE_BANDS, start=1):
            band = np.where(ms_valid, DRONE_FUNCTIONS[name](base), 0.0).astype(np.float32)
            dst.write(band, index)
    with rasterio.open(rgb_path, "w", driver="GTiff", height=size_px, width=size_px, count=4, dtype="uint8",
                       crs=crs, transform=transform) as dst:
        level = np.clip((base - 0.03) / 0.5, 0.0, 1.0)  # a believable orthophoto: dark where the scene is low
        for index, (low, span) in enumerate(((70, 110), (95, 120), (55, 60)), start=1):
            dst.write((low + span * level).astype(np.uint8), index)  # never 0, so no pixel looks like no-data
        dst.write(np.where(rgb_valid, 255, 0).astype(np.uint8), 4)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    return ms_path, rgb_path


def write_s2_scene(
    folder: Path, scene, *, content_offset_m: tuple[float, float] = (0.0, 0.0), item_id: str = "S2_TEST_A",
    acquired: str = "2026-10-12T05:30:00Z", baseline: str = "05.11", scene_cloud: float = 2.0,
    edit_10m: Callable[[dict[str, np.ndarray]], None] | None = None,
    edit_scl: Callable[[np.ndarray], None] | None = None,
    supersample: int = 1,
    rotation_deg: float = 0.0,
    datatake_id: str | None = None,
) -> Path:
    """One scene folder in the LocalFolderProvider layout. ``edit_*`` mutate reflectance / SCL before encoding.

    ``supersample`` area-averages each pixel (use 5 for sharp-edged scenes); ``rotation_deg`` additionally rotates the
    content about the centre of the drone area, a distortion no single shift can undo.
    """
    folder.mkdir(parents=True, exist_ok=True)
    transform_10m, transform_20m = s2_grid_transform(), s2_grid_transform() * Affine.scale(2, 2)
    offset = baseline_offset(parse_processing_baseline(baseline))
    pivot = (500800.0, 3499200.0)

    def sample(transform, shape):
        if rotation_deg:
            moved = lambda x, y: scene(x - content_offset_m[0], y - content_offset_m[1])  # noqa: E731
            return synthetic.sample_scene_rotated(moved, transform, shape, rotation_deg, pivot)
        return synthetic.sample_scene(scene, transform, shape, content_offset_m, supersample)

    reflectance = {
        band: S2_BAND_FUNCTIONS[band](sample(transform_10m, S2_SHAPE_10M)).astype(np.float32)
        for band in ("B02", "B03", "B04", "B08")
    }
    if edit_10m:
        edit_10m(reflectance)
    reflectance_20m = {
        band: S2_BAND_FUNCTIONS[band](sample(transform_20m, S2_SHAPE_20M)).astype(np.float32)
        for band in S2_20M_BANDS
    }
    scl = np.full(S2_SHAPE_20M, 4, np.uint8)
    if edit_scl:
        edit_scl(scl)

    def write(name: str, array: np.ndarray, transform: Affine) -> None:
        with rasterio.open(folder / f"{name}.tif", "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                           count=1, dtype=array.dtype, crs=CRS_UTM42, transform=transform) as dst:
            dst.write(array, 1)

    for band, data in reflectance.items():
        write(band, synthetic.reflectance_to_dn(data, offset=offset), transform_10m)
    for band, data in reflectance_20m.items():
        write(band, synthetic.reflectance_to_dn(data, offset=offset), transform_20m)
    write("SCL", scl, transform_20m)
    (folder / "scene.json").write_text(json.dumps({
        "item_id": item_id, "datetime": acquired, "processing_baseline": baseline, "mgrs_tile": "42RXV",
        "scene_cloud_cover": scene_cloud, "relative_orbit": 120, "datatake_id": datatake_id,
    }), encoding="utf-8")
    return folder
