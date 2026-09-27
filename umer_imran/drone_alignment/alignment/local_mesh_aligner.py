"""Sparse, nearest-road constrained local registration on a common grid.

This module intentionally does *not* warp source rasters.  It is the evidence
stage for a future displacement-field warper: given a verified global MS->RGB
translation, it finds trusted local residual vectors using feature maps built
once on the complete registration canvas.  Cells query those global maps; they
never create isolated connected components or distance transforms at cell edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import cv2
import numpy as np

from drone_alignment.alignment.feature_detector import (
    InsufficientContrastError,
    InsufficientValidDataError,
    normalize_to_uint8,
)
from drone_alignment.alignment.road_grid_aligner import _binarize_mask
from drone_alignment.config.schema import LocalMeshConfig, RoadGridConfig


from drone_alignment.alignment.local_evidence import (
    CellMatchStatus,
    LocalMatchSample,
    SparseDisplacementResult,
)


@dataclass(frozen=True)
class StructuralFeatureMap:
    """Full-canvas structural maps for one input image."""

    road_mask: np.ndarray
    tree_mask: np.ndarray
    road_skeleton: np.ndarray
    tree_skeleton: np.ndarray
    road_distance: np.ndarray
    tree_distance: np.ndarray
    road_components: np.ndarray
    orientation_deg: np.ndarray
    valid_mask: np.ndarray


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton on the complete canvas (no optional dependencies)."""
    image = ((mask > 0).astype(np.uint8) * 255).copy()
    skeleton = np.zeros_like(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        skeleton |= cv2.subtract(image, opened)
        image = cv2.erode(image, kernel)
    return skeleton > 0


def _orientation(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    # Road centreline orientation is perpendicular to the intensity gradient.
    return (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0


def build_structural_feature_map(
    band: np.ndarray,
    valid_mask: np.ndarray,
    road_config: RoadGridConfig,
) -> StructuralFeatureMap:
    """Build all structural evidence once for a complete registration image."""
    try:
        gray = normalize_to_uint8(band, valid_mask, apply_clahe=False)
    except (InsufficientContrastError, InsufficientValidDataError) as exc:
        raise ValueError(f"Cannot build local structural map: {exc}") from exc
    valid = valid_mask.astype(bool)
    road = _binarize_mask(gray, valid, road_config.bright_percentile, road_config.blur_kernel_size, "bright")
    tree = _binarize_mask(gray, valid, road_config.dark_percentile, road_config.blur_kernel_size, "dark")
    road_skeleton = _skeletonize(road)
    tree_skeleton = _skeletonize(tree)
    # OpenCV's distance transform measures distance to zero, so invert features.
    road_distance = cv2.distanceTransform((~road_skeleton).astype(np.uint8), cv2.DIST_L2, 3)
    tree_distance = cv2.distanceTransform((~tree_skeleton).astype(np.uint8), cv2.DIST_L2, 3)
    _, components = cv2.connectedComponents(road_skeleton.astype(np.uint8), connectivity=8)
    components[~valid] = 0
    return StructuralFeatureMap(
        road_mask=road > 0,
        tree_mask=tree > 0,
        road_skeleton=road_skeleton,
        tree_skeleton=tree_skeleton,
        road_distance=road_distance,
        tree_distance=tree_distance,
        road_components=components,
        orientation_deg=_orientation(gray),
        valid_mask=valid,
    )


def _cell_bounds(shape: tuple[int, int], row: int, col: int, cfg: LocalMeshConfig) -> tuple[int, int, int, int, int, int, int, int]:
    h, w = shape
    y0, y1 = row * h // cfg.grid_rows, (row + 1) * h // cfg.grid_rows
    x0, x1 = col * w // cfg.grid_cols, (col + 1) * w // cfg.grid_cols
    overlap = int(round(min(y1 - y0, x1 - x0) * cfg.cell_overlap_fraction / 2)) + cfg.cell_halo_px
    return x0, y0, x1, y1, max(0, x0 - overlap), max(0, y0 - overlap), min(w, x1 + overlap), min(h, y1 + overlap)


def _sample_distance(distance: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = distance.shape
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    values = np.full(xs.shape, np.inf, dtype=np.float32)
    values[inside] = distance[yi[inside], xi[inside]]
    return values


def _score_residual(
    source_xy: np.ndarray,
    source_orientation: np.ndarray,
    reference: StructuralFeatureMap,
    global_translation: tuple[float, float],
    residual: tuple[int, int],
) -> float:
    dx = global_translation[0] + residual[0]
    dy = global_translation[1] + residual[1]
    xs, ys = source_xy[:, 0] + dx, source_xy[:, 1] + dy
    distances = _sample_distance(reference.road_distance, xs, ys)
    finite = np.isfinite(distances)
    if finite.sum() < max(5, len(source_xy) // 2):
        return math.inf
    # Trim a minority of endpoint/out-of-window distances.
    distance_score = float(np.mean(np.sort(distances[finite])[: max(1, int(finite.sum() * 0.8))]))
    xi, yi = np.rint(xs[finite]).astype(int), np.rint(ys[finite]).astype(int)
    angle = np.abs(source_orientation[finite] - reference.orientation_deg[yi, xi])
    angle = np.minimum(angle, 180.0 - angle)
    # Break otherwise-flat road-axis ties in favour of the verified global
    # prior.  This is deliberately tiny: a genuine road-distance improvement
    # must always outweigh it, while unconstrained along-road drift is not
    # allowed to wander to a search boundary.
    prior_penalty = 0.001 * math.hypot(residual[0], residual[1])
    return distance_score + 0.03 * float(np.median(angle)) + prior_penalty


def _best_match(
    source_xy: np.ndarray,
    source_orientation: np.ndarray,
    reference: StructuralFeatureMap,
    global_translation: tuple[float, float],
    radius: int,
    separation: int,
) -> tuple[tuple[int, int], float, float, bool]:
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            candidates.append((_score_residual(source_xy, source_orientation, reference, global_translation, (dx, dy)), dx, dy))
    candidates.sort(key=lambda value: value[0])
    best_score, best_dx, best_dy = candidates[0]
    second = math.inf
    for score, dx, dy in candidates[1:]:
        if max(abs(dx - best_dx), abs(dy - best_dy)) >= separation:
            second = score
            break
    on_boundary = abs(best_dx) == radius or abs(best_dy) == radius
    return (best_dx, best_dy), best_score, second, on_boundary


def _score_tree_residual(
    source_xy: np.ndarray,
    reference: StructuralFeatureMap,
    global_translation: tuple[float, float],
    residual: tuple[int, int],
) -> float:
    xs = source_xy[:, 0] + global_translation[0] + residual[0]
    ys = source_xy[:, 1] + global_translation[1] + residual[1]
    values = _sample_distance(reference.tree_distance, xs, ys)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.inf
    return float(np.mean(np.sort(finite)[: max(1, int(finite.size * 0.8))])) + 0.001 * math.hypot(*residual)


def _tree_match(
    source_xy: np.ndarray,
    reference: StructuralFeatureMap,
    global_translation: tuple[float, float],
    radius: int,
    separation: int,
) -> tuple[tuple[int, int], float, float]:
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            score = _score_tree_residual(source_xy, reference, global_translation, (dx, dy))
            candidates.append((score, dx, dy))
    candidates.sort(key=lambda value: value[0])
    best_score, best_dx, best_dy = candidates[0]
    second = math.inf
    for score, dx, dy in candidates[1:]:
        if max(abs(dx - best_dx), abs(dy - best_dy)) >= separation:
            second = score
            break
    return (best_dx, best_dy), best_score, second


def _hold_out_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 80/20 split; tiny cells keep all evidence for matching."""
    indices = np.arange(count)
    if count < 10:
        return indices, np.empty(0, dtype=np.int64)
    validation = indices[::5]
    training = np.ones(count, dtype=bool)
    training[validation] = False
    return indices[training], validation


def _make_sample(
    row: int,
    col: int,
    rgb: StructuralFeatureMap,
    ms: StructuralFeatureMap,
    global_translation: tuple[float, float],
    cfg: LocalMeshConfig,
) -> LocalMatchSample:
    x0, y0, x1, y1, hx0, hy0, hx1, hy1 = _cell_bounds(ms.road_skeleton.shape, row, col, cfg)
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    source_y, source_x = np.where(ms.road_skeleton[y0:y1, x0:x1])
    source_xy = np.column_stack([source_x + x0, source_y + y0]).astype(np.float32)
    tree_y, tree_x = np.where(ms.tree_skeleton[y0:y1, x0:x1])
    tree_xy = np.column_stack([tree_x + x0, tree_y + y0]).astype(np.float32)
    predicted = (center[0] + global_translation[0], center[1] + global_translation[1])
    base = dict(row=row, col=col, center_xy=center, source_ms_xy=center, predicted_rgb_xy=predicted)
    if len(source_xy) < cfg.min_road_points:
        if cfg.allow_tree_only and len(tree_xy) >= cfg.min_tree_points:
            separation = max(
                cfg.ambiguity_separation_px,
                int(round(cfg.max_search_radius_px * cfg.competitor_spacing_fraction)),
            )
            tree_train_idx, tree_validation_idx = _hold_out_indices(len(tree_xy))
            tree_residual, tree_best, tree_second = _tree_match(
                tree_xy[tree_train_idx], rgb, global_translation, cfg.max_search_radius_px, separation,
            )
            tree_margin = (tree_second - tree_best) / max(tree_second, 1.0) if np.isfinite(tree_second) else 0.0
            support = min(1.0, len(tree_xy) / (2.0 * cfg.min_tree_points))
            confidence = tree_margin * support
            on_boundary = abs(tree_residual[0]) == cfg.max_search_radius_px or abs(tree_residual[1]) == cfg.max_search_radius_px
            if on_boundary:
                status = CellMatchStatus.SEARCH_BOUNDARY
            elif not np.isfinite(tree_best) or not np.isfinite(tree_second) or tree_margin < cfg.tree_ambiguity_margin:
                status = CellMatchStatus.AMBIGUOUS_ROADS
            elif confidence < cfg.tree_only_min_confidence:
                status = CellMatchStatus.LOW_CONFIDENCE
            else:
                status = CellMatchStatus.ACCEPTED
            displacement = (global_translation[0] + tree_residual[0], global_translation[1] + tree_residual[1])
            validation_global = validation_local = None
            if len(tree_validation_idx):
                validation_global = _score_tree_residual(tree_xy[tree_validation_idx], rgb, global_translation, (0, 0))
                validation_local = _score_tree_residual(tree_xy[tree_validation_idx], rgb, global_translation, tree_residual)
            return LocalMatchSample(
                **base, residual_dx_dy=tuple(map(float, tree_residual)), displacement_dx_dy=displacement,
                confidence=confidence, road_score=None,
                second_road_score=float(tree_second) if np.isfinite(tree_second) else None,
                tree_residual_dx_dy=tuple(map(float, tree_residual)), status=status, evidence="tree",
                validation_global_score=validation_global, validation_local_score=validation_local,
            )
        return LocalMatchSample(**base, residual_dx_dy=(0.0, 0.0), displacement_dx_dy=global_translation,
                                confidence=0.0, road_score=None, second_road_score=None,
                                tree_residual_dx_dy=None, status=CellMatchStatus.NO_FEATURES, evidence="none")
    source_angles = ms.orientation_deg[source_xy[:, 1].astype(int), source_xy[:, 0].astype(int)]
    road_train_idx, road_validation_idx = _hold_out_indices(len(source_xy))
    hypothesis_separation = max(
        cfg.ambiguity_separation_px,
        int(round(cfg.max_search_radius_px * cfg.competitor_spacing_fraction)),
    )
    residual, best, second, on_boundary = _best_match(
        source_xy[road_train_idx], source_angles[road_train_idx], rgb, global_translation,
        cfg.max_search_radius_px, hypothesis_separation,
    )
    if on_boundary:
        status = CellMatchStatus.SEARCH_BOUNDARY
    elif not np.isfinite(best) or not np.isfinite(second) or (second - best) / max(second, 1.0) < cfg.ambiguity_margin:
        status = CellMatchStatus.AMBIGUOUS_ROADS
    else:
        status = CellMatchStatus.ACCEPTED
    margin = 0.0 if not np.isfinite(second) else max(0.0, min(1.0, (second - best) / max(second, 1.0)))
    support = min(1.0, len(source_xy) / (2.0 * cfg.min_road_points))
    confidence = margin * support
    tree_residual = None
    if len(tree_x) >= cfg.min_tree_points:
        tree_train_idx, _ = _hold_out_indices(len(tree_xy))
        tree_residual, tree_best, tree_second = _tree_match(
            tree_xy[tree_train_idx], rgb, global_translation, cfg.max_search_radius_px, hypothesis_separation,
        )
        tree_margin = (tree_second - tree_best) / max(tree_second, 1.0) if np.isfinite(tree_second) else 0.0
        # Repetitive tree rows are common.  They may corroborate a road result
        # only when they independently identify one local displacement.
        if tree_margin >= cfg.ambiguity_margin:
            if math.hypot(tree_residual[0] - residual[0], tree_residual[1] - residual[1]) > cfg.road_tree_agreement_px:
                status = CellMatchStatus.ROAD_TREE_CONFLICT
            else:
                confidence = min(1.0, confidence + 0.15)
    if confidence < cfg.min_match_confidence and status == CellMatchStatus.ACCEPTED:
        status = CellMatchStatus.LOW_CONFIDENCE
    displacement = (global_translation[0] + residual[0], global_translation[1] + residual[1])
    validation_global = validation_local = None
    if len(road_validation_idx):
        validation_xy, validation_angles = source_xy[road_validation_idx], source_angles[road_validation_idx]
        validation_global = _score_residual(validation_xy, validation_angles, rgb, global_translation, (0, 0))
        validation_local = _score_residual(validation_xy, validation_angles, rgb, global_translation, residual)
    return LocalMatchSample(**base, residual_dx_dy=(float(residual[0]), float(residual[1])),
                            displacement_dx_dy=displacement, confidence=confidence,
                            road_score=float(best), second_road_score=float(second) if np.isfinite(second) else None,
                            tree_residual_dx_dy=tuple(map(float, tree_residual)) if tree_residual else None, status=status,
                            evidence="road+tree" if tree_residual else "road",
                            validation_global_score=validation_global, validation_local_score=validation_local)


def _apply_spatial_filter(samples: list[LocalMatchSample], cfg: LocalMeshConfig) -> list[LocalMatchSample]:
    accepted = [item for item in samples if item.status == CellMatchStatus.ACCEPTED]
    by_cell = {(item.row, item.col): item for item in accepted}
    result: list[LocalMatchSample] = []
    for item in samples:
        if item.status != CellMatchStatus.ACCEPTED:
            result.append(item)
            continue
        neighbors = [other for (r, c), other in by_cell.items()
                     if (r, c) != (item.row, item.col) and max(abs(r - item.row), abs(c - item.col)) == 1]
        if len(neighbors) >= 2:
            dx = float(np.median([other.residual_dx_dy[0] for other in neighbors]))
            dy = float(np.median([other.residual_dx_dy[1] for other in neighbors]))
            if math.hypot(item.residual_dx_dy[0] - dx, item.residual_dx_dy[1] - dy) > cfg.max_neighbor_difference_px:
                item = LocalMatchSample(**{**item.__dict__, "status": CellMatchStatus.SPATIAL_OUTLIER, "confidence": 0.0})
        result.append(item)
    return result


def compute_sparse_local_displacements(
    rgb_band: np.ndarray,
    ms_band: np.ndarray,
    rgb_mask: np.ndarray,
    ms_mask: np.ndarray,
    global_translation_px: tuple[float, float],
    road_config: RoadGridConfig,
    local_config: LocalMeshConfig,
) -> SparseDisplacementResult:
    """Return guarded local MS->RGB displacement samples; never modify imagery."""
    if rgb_band.shape != ms_band.shape or rgb_band.shape != rgb_mask.shape or rgb_band.shape != ms_mask.shape:
        raise ValueError("Local mesh inputs must share the registration-grid shape.")
    common = rgb_mask.astype(bool) & ms_mask.astype(bool)
    rgb = build_structural_feature_map(rgb_band, common, road_config)
    ms = build_structural_feature_map(ms_band, common, road_config)
    samples = [_make_sample(r, c, rgb, ms, global_translation_px, local_config)
               for r in range(local_config.grid_rows) for c in range(local_config.grid_cols)]
    samples = _apply_spatial_filter(samples, local_config)
    trusted = sum(item.status == CellMatchStatus.ACCEPTED for item in samples)
    coverage = trusted / len(samples) if samples else 0.0
    return SparseDisplacementResult(tuple(samples), rgb, ms, global_translation_px, trusted, coverage)
