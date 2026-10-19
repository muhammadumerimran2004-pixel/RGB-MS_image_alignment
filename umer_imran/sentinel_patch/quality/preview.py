"""QA preview: one PNG that shows whether the shift is believable.

Panels: (1) the drone RGB with the perimeter and the shift arrow, (2) the synthetic reference, (3) Sentinel
before the shift, (4) Sentinel after the shift as a checkerboard against the reference - a correct shift makes
the field edges continue across the squares.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry.base import BaseGeometry

RGB_MAX_DIM = 900
CHECKER_TILE_PX = 6
ARROW_SCALE = 10.0  # the shift arrow is drawn this many times longer than it is


def _rings(geometry: BaseGeometry):
    parts = getattr(geometry, "geoms", [geometry])
    return [np.asarray(part.exterior.coords) for part in parts]


def _stretch(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low, high = np.percentile(finite, (2, 98))
    if high <= low:  # a constant image has nothing to stretch: show it as mid-grey, not black
        return np.where(np.isfinite(values), 0.5, np.nan)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _match_levels(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Linear brightness match of ``other`` to ``reference`` (different sensors, different gain), so a checkerboard
    shows structural mismatch only."""
    both = np.isfinite(reference) & np.isfinite(other)
    if both.sum() < 10 or np.std(other[both]) < 1e-9:
        return other
    gain, offset = np.polyfit(other[both], reference[both], 1)
    return np.clip(other * gain + offset, 0.0, 1.0)


def _extent(transform: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    left, top = transform.c, transform.f
    return left, left + transform.a * shape[1], top + transform.e * shape[0], top


def _read_rgb(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(path) as src:
        scale = min(1.0, RGB_MAX_DIM / max(src.height, src.width))
        shape = (max(1, round(src.height * scale)), max(1, round(src.width * scale)))
        bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        rgb = src.read(bands, out_shape=shape, resampling=Resampling.average).astype(np.float64)
        valid = src.dataset_mask(out_shape=shape, resampling=Resampling.nearest) > 0
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    image = np.dstack([_stretch(np.where(valid, rgb[i], np.nan)) for i in range(3)])
    return np.dstack([image, valid.astype(np.float64)]), extent


def _checkerboard(reference: np.ndarray, shifted: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(reference.shape)
    use_reference = ((rows // CHECKER_TILE_PX) + (cols // CHECKER_TILE_PX)) % 2 == 0
    blend = np.where(use_reference, reference, shifted)
    return np.where(np.isfinite(reference) & np.isfinite(shifted), blend, np.nan)


def write_preview(
    path: Path, *, rgb_path: Path, drone_polygon: BaseGeometry, scene_polygon: BaseGeometry,
    window_transform: Affine, reference: np.ndarray, sentinel: np.ndarray, shift_m: tuple[float, float],
    shift_px: tuple[float, float] | None, status: str, reason_code: str | None, title_lines: list[str],
) -> Path:
    figure = Figure(figsize=(12.5, 10.0), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    extent = _extent(window_transform, reference.shape)

    rgb, rgb_extent = _read_rgb(rgb_path)
    axes[0, 0].imshow(rgb, extent=rgb_extent)
    for ring in _rings(drone_polygon):
        axes[0, 0].plot(ring[:, 0], ring[:, 1], color="yellow", linewidth=1.2)
    centroid = drone_polygon.centroid
    if status == "verified" and any(shift_m):
        axes[0, 0].annotate("", xy=(centroid.x + ARROW_SCALE * shift_m[0], centroid.y + ARROW_SCALE * shift_m[1]),
                            xytext=(centroid.x, centroid.y), arrowprops={"arrowstyle": "-|>", "color": "red", "lw": 2.5})
        axes[0, 0].text(0.02, 0.97, f"shift x{ARROW_SCALE:.0f}: dE {shift_m[0]:+.1f} m, dN {shift_m[1]:+.1f} m",
                        transform=axes[0, 0].transAxes, va="top", color="white", backgroundcolor="black", fontsize=9)
    axes[0, 0].set_title("Drone RGB, perimeter and shift")

    before = _stretch(sentinel)
    ref_scaled = _stretch(reference)
    if status == "verified" and shift_px:
        # Move the content by the measured pixel shift; track validity separately so edges do not darken.
        moved = ndimage.shift(np.nan_to_num(sentinel, nan=0.0), (shift_px[1], shift_px[0]), order=1, mode="constant", cval=0.0)
        moved_valid = ndimage.shift(np.isfinite(sentinel).astype(np.float64), (shift_px[1], shift_px[0]),
                                    order=1, mode="constant", cval=0.0) > 0.999
        after = _stretch(np.where(moved_valid, moved, np.nan))
    else:
        after = before

    panels = [(axes[0, 1], ref_scaled, "Drone NIR degraded to Sentinel resolution"),
              (axes[1, 0], before, "Sentinel NIR, original georeferencing"),
              (axes[1, 1], _checkerboard(ref_scaled, _match_levels(ref_scaled, after)),
               "Sentinel after shift vs reference (checkerboard)" if status == "verified" else "Not applied: no shift")]
    for axis, data, title in panels:
        axis.imshow(np.ma.masked_invalid(data), extent=extent, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        if axis is not axes[0, 1]:
            for ring in _rings(scene_polygon):
                axis.plot(ring[:, 0], ring[:, 1], color="yellow", linewidth=1.0)
    for axis in axes.ravel():
        axis.tick_params(labelsize=7)
        axis.set_aspect("equal")
    head = "  |  ".join(title_lines)
    status_line = status.upper() + (f" ({reason_code})" if reason_code else "")
    figure.suptitle(f"{head}\n{status_line}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    figure.savefig(partial, format="png")
    os.replace(partial, path)
    return path
