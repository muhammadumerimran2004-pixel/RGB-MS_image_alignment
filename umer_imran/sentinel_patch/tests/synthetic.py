"""Deterministic synthetic scenes with an analytic ground truth.

The scene is a continuous function of map coordinates, ``f(x, y)``, so a test
never depends on interpolation to know where content truly is. A Sentinel grid
whose *declared* geotransform is wrong by ``(ex, ey)`` is built by sampling
``f`` at ``declared position - (ex, ey)``: its content then appears displaced
by ``+(ex, ey)`` relative to a reference sampled at the declared positions, and
the correct fix is a ``shift_m`` of ``(-ex, -ey)``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from affine import Affine

SceneFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]

S2_GSD_M = 10.0
S2_ORIGIN = (500000.0, 3500000.0)


def make_s2_transform(origin: tuple[float, float] = S2_ORIGIN, gsd: float = S2_GSD_M) -> Affine:
    """North-up grid: ``e < 0`` so a larger row index is further south."""
    return Affine(gsd, 0.0, origin[0], 0.0, -gsd, origin[1])


def make_scene(transform: Affine, shape: tuple[int, int], *, seed: int = 7, blobs: int = 28) -> SceneFunction:
    """Sum of Gaussian blobs spread over the grid footprint; returns reflectance in about [0.03, 0.6]."""
    rng = np.random.default_rng(seed)
    height, width = shape
    x0, y0 = transform.c, transform.f
    x1 = x0 + transform.a * width
    y1 = y0 + transform.e * height
    cx = rng.uniform(x0, x1, blobs)
    cy = rng.uniform(y1, y0, blobs)
    sigma = rng.uniform(20.0, 60.0, blobs)
    amplitude = rng.uniform(0.05, 0.25, blobs)

    def scene(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)[..., None]
        y = np.asarray(y, dtype=np.float64)[..., None]
        blob_sum = (amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))).sum(axis=-1)
        return 0.03 + blob_sum

    return scene


def pixel_centre_xy(transform: Affine, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates of every pixel centre, as ``(x, y)`` arrays of ``shape``."""
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    col, row = cols + 0.5, rows + 0.5
    x = transform.a * col + transform.b * row + transform.c
    y = transform.d * col + transform.e * row + transform.f
    return x, y


def make_field_scene(transform: Affine, shape: tuple[int, int], *, seed: int = 3, cells: int = 60) -> SceneFunction:
    """Sharp-edged patchwork of uniform 'fields' (nearest-seed cells): the realistic case for phase correlation.

    The smooth Gaussian scene above has no edges, which frequency-domain matchers such as AROSICS need.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    height, width = shape
    x0, y0 = transform.c, transform.f
    x1, y1 = x0 + transform.a * width, y0 + transform.e * height
    margin = 0.15 * max(abs(x1 - x0), abs(y1 - y0))
    seeds = np.column_stack([rng.uniform(min(x0, x1) - margin, max(x0, x1) + margin, cells),
                             rng.uniform(min(y0, y1) - margin, max(y0, y1) + margin, cells)])
    values = rng.uniform(0.05, 0.6, cells)
    tree = cKDTree(seeds)

    def scene(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        points = np.column_stack([x.ravel(), np.asarray(y, dtype=np.float64).ravel()])
        return values[tree.query(points)[1]].reshape(x.shape)

    return scene


def sample_scene(scene: SceneFunction, transform: Affine, shape: tuple[int, int],
                 content_offset_m: tuple[float, float] = (0.0, 0.0), supersample: int = 1) -> np.ndarray:
    """Sample ``scene`` on a grid whose content is displaced by ``content_offset_m``.

    ``content_offset_m = (ex, ey)`` means the grid is mis-georeferenced by that
    vector: its content truly at ``X`` is declared at ``X + (ex, ey)``.
    ``supersample > 1`` averages that many points per axis inside each pixel, as a real sensor integrates
    over its footprint (needed for sharp-edged scenes, where point sampling aliases).
    """
    x, y = pixel_centre_xy(transform, shape)
    pixel = abs(transform.a)
    total = np.zeros(shape, dtype=np.float64)
    for i in range(supersample):
        for j in range(supersample):
            dx = ((i + 0.5) / supersample - 0.5) * pixel
            dy = ((j + 0.5) / supersample - 0.5) * pixel
            total += scene(x + dx - content_offset_m[0], y + dy - content_offset_m[1])
    return (total / supersample**2).astype(np.float32)


def sample_scene_rotated(scene: SceneFunction, transform: Affine, shape: tuple[int, int], degrees: float,
                         about: tuple[float, float]) -> np.ndarray:
    """Sample with the content rotated about a map point: a distortion no single translation can undo."""
    x, y = pixel_centre_xy(transform, shape)
    angle = np.radians(degrees)
    rx, ry = x - about[0], y - about[1]
    return scene(about[0] + rx * np.cos(angle) - ry * np.sin(angle),
                 about[1] + rx * np.sin(angle) + ry * np.cos(angle)).astype(np.float32)


def reflectance_to_dn(reflectance: np.ndarray, *, offset: float, scale: float = 1.0e-4) -> np.ndarray:
    """Encode reflectance as L2A DN for a given processing-baseline offset (0.0 or -0.1)."""
    return np.rint((reflectance.astype(np.float64) - offset) / scale).astype(np.uint16)


def integer_shift_stub(reference: np.ndarray, target: np.ndarray, max_px: int = 6) -> tuple[int, int]:
    """Brute-force integer-pixel estimator used only to test the sign convention in Phase 0.

    Returns ``(col_shift, row_shift)``: the pixels by which the *content* of
    ``target`` must move to land on ``reference`` (the input ``pixel_shift_to_map`` expects).
    The real masked-NCC and AROSICS estimators replace it in Phase 1/2.
    """
    height, width = reference.shape
    best_score, best_shift = -np.inf, (0, 0)
    for row_shift in range(-max_px, max_px + 1):
        for col_shift in range(-max_px, max_px + 1):
            ref = reference[max(0, row_shift): height + min(0, row_shift),
                            max(0, col_shift): width + min(0, col_shift)]
            tgt = target[max(0, -row_shift): height + min(0, -row_shift),
                         max(0, -col_shift): width + min(0, -col_shift)]
            if ref.size < 100:
                continue
            score = float(np.corrcoef(ref.ravel(), tgt.ravel())[0, 1])
            if score > best_score:
                best_score, best_shift = score, (col_shift, row_shift)
    return best_shift
