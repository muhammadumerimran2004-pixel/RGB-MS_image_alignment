"""Road-frame registration using elongated bright/dark structural rails.

Unlike fixed-grid local matching, this module derives its regions from actual
roads/dry lines.  Matching begins from the shared geospatial canvas (zero local
residual), never from an unconstrained periodic phase-correlation peak.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from drone_alignment.alignment.feature_detector import normalize_to_uint8
from drone_alignment.alignment.road_grid_aligner import _binarize_mask
from drone_alignment.config.schema import LocalMeshConfig, RoadGridConfig


class RailOrientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass(frozen=True)
class StructuralRail:
    rail_id: int
    orientation: RailOrientation
    center: float
    span_start: float
    span_end: float
    width: float
    strength: float
    is_dark: bool = False
    is_exterior: bool = False


@dataclass(frozen=True)
class RailMatch:
    ms_rail: StructuralRail
    rgb_rail: StructuralRail
    deviation_px: float
    confidence: float


@dataclass(frozen=True)
class RailBoundedRegion:
    x0: int
    y0: int
    x1: int
    y1: int
    left_rail: int | None
    right_rail: int | None
    top_rail: int | None
    bottom_rail: int | None


@dataclass(frozen=True)
class RailRegionCorrection:
    region: RailBoundedRegion
    dx: float
    dy: float
    confidence: float
    x_constraints: int
    y_constraints: int
    local_improvement: float = 0.0
    accepted: bool = True


@dataclass(frozen=True)
class RailFrameResult:
    rgb_bright_rails: tuple[StructuralRail, ...]
    ms_bright_rails: tuple[StructuralRail, ...]
    bright_matches: tuple[RailMatch, ...]
    rgb_dark_rails: tuple[StructuralRail, ...]
    ms_dark_rails: tuple[StructuralRail, ...]
    dark_matches: tuple[RailMatch, ...]
    regions: tuple[RailBoundedRegion, ...]
    corrections: tuple[RailRegionCorrection, ...]


@dataclass(frozen=True)
class RailConstrainedField:
    """Separable field whose control nodes are matched structural rails."""

    x_positions: np.ndarray
    x_deviations: np.ndarray
    y_positions: np.ndarray
    y_deviations: np.ndarray
    name: str = "rail_constrained_variable_regions"

    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        dx = (np.interp(points[:, 0], self.x_positions, self.x_deviations)
              if len(self.x_positions) else np.zeros(len(points)))
        dy = (np.interp(points[:, 1], self.y_positions, self.y_deviations)
              if len(self.y_positions) else np.zeros(len(points)))
        return np.column_stack([dx, dy])


@dataclass(frozen=True)
class RailRegionField:
    """Interior corrections feathered to zero at every structural boundary."""

    corrections: tuple[RailRegionCorrection, ...]
    feather_px: float
    name: str = "rail_bounded_local_regions"

    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        output = np.zeros((len(points), 2), dtype=np.float64)
        for correction in self.corrections:
            if not correction.accepted:
                continue
            region = correction.region
            inside = ((points[:, 0] >= region.x0) & (points[:, 0] <= region.x1) &
                      (points[:, 1] >= region.y0) & (points[:, 1] <= region.y1))
            if not np.any(inside):
                continue
            selected = points[inside]
            edge_distance = np.minimum.reduce([
                selected[:, 0] - region.x0, region.x1 - selected[:, 0],
                selected[:, 1] - region.y0, region.y1 - selected[:, 1],
            ])
            # Smoothstep reaches zero exactly on rails and one in the supported
            # interior, preventing seams while keeping roads immovable anchors.
            t = np.clip(edge_distance / max(self.feather_px, 1.0), 0.0, 1.0)
            weight = t * t * (3.0 - 2.0 * t)
            output[inside, 0] = correction.dx * weight
            output[inside, 1] = correction.dy * weight
        return output


def build_rail_region_field(
    corrections: tuple[RailRegionCorrection, ...],
    config: LocalMeshConfig,
) -> RailRegionField:
    feather = max(float(config.region_boundary_margin_px), float(config.region_local_max_deviation_px * 2))
    return RailRegionField(corrections, feather)


def _merge_axis_runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value - previous > max_gap + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def _extract_oriented_rails(
    feature_mask: np.ndarray,
    orientation: RailOrientation,
    config: LocalMeshConfig,
    is_dark: bool,
) -> list[StructuralRail]:
    height, width = feature_mask.shape
    minimum_width = 1 if is_dark else config.rail_min_width_px
    if orientation == RailOrientation.HORIZONTAL:
        long_side = max(3, int(round(width * (config.dark_boundary_min_length_fraction if is_dark else config.rail_min_length_fraction))))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (long_side, minimum_width))
        opened = cv2.morphologyEx(feature_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        support = opened.mean(axis=1)
        active = support >= (config.dark_boundary_min_length_fraction if is_dark else config.rail_min_length_fraction)
        runs = _merge_axis_runs(active, config.rail_merge_gap_px)
    else:
        long_side = max(3, int(round(height * (config.dark_boundary_min_length_fraction if is_dark else config.rail_min_length_fraction))))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (minimum_width, long_side))
        opened = cv2.morphologyEx(feature_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        support = opened.mean(axis=0)
        active = support >= (config.dark_boundary_min_length_fraction if is_dark else config.rail_min_length_fraction)
        runs = _merge_axis_runs(active, config.rail_merge_gap_px)
    rails: list[StructuralRail] = []
    for start, end in runs:
        if end - start < minimum_width:
            continue
        if orientation == RailOrientation.HORIZONTAL:
            sub = opened[start:end]
            ys, xs = np.where(sub > 0)
            span_start, span_end = (float(xs.min()), float(xs.max() + 1)) if len(xs) else (0.0, 0.0)
        else:
            sub = opened[:, start:end]
            ys, xs = np.where(sub > 0)
            span_start, span_end = (float(ys.min()), float(ys.max() + 1)) if len(ys) else (0.0, 0.0)
        rails.append(StructuralRail(
            rail_id=len(rails), orientation=orientation, center=(start + end - 1) / 2.0,
            span_start=span_start, span_end=span_end, width=float(end - start),
            strength=float(np.mean(support[start:end])), is_dark=is_dark,
        ))
    return rails


def detect_structural_rails(
    band: np.ndarray,
    valid_mask: np.ndarray,
    road_config: RoadGridConfig,
    local_config: LocalMeshConfig,
) -> tuple[tuple[StructuralRail, ...], tuple[StructuralRail, ...]]:
    gray = normalize_to_uint8(band, valid_mask, apply_clahe=False)
    valid = valid_mask.astype(bool)
    bright = _binarize_mask(gray, valid, road_config.bright_percentile, road_config.blur_kernel_size, "bright") > 0
    dark = _binarize_mask(gray, valid, road_config.dark_percentile, road_config.blur_kernel_size, "dark") > 0
    bright_rails = (_extract_oriented_rails(bright, RailOrientation.HORIZONTAL, local_config, False) +
                    _extract_oriented_rails(bright, RailOrientation.VERTICAL, local_config, False))
    dark_rails = (_extract_oriented_rails(dark, RailOrientation.HORIZONTAL, local_config, True) +
                  _extract_oriented_rails(dark, RailOrientation.VERTICAL, local_config, True))
    height, width = band.shape
    dark_rails = [rail for rail in dark_rails if (
        local_config.min_region_size_px <= rail.center <=
        (height - local_config.min_region_size_px if rail.orientation == RailOrientation.HORIZONTAL
         else width - local_config.min_region_size_px)
    )]
    # A detected road is an exterior frame only when it is actually close to a
    # valid-footprint boundary. Merely being the first/last detected *internal*
    # road must not grant it exterior-anchor authority.
    yy, xx = np.where(valid)
    if len(xx):
        min_x, max_x = float(xx.min()), float(xx.max())
        min_y, max_y = float(yy.min()), float(yy.max())
        margin = float(local_config.exterior_rail_max_deviation_px * 2)
        bright_rails = [StructuralRail(**{
            **rail.__dict__,
            "is_exterior": (
                min(abs(rail.center - min_x), abs(rail.center - max_x)) <= margin
                if rail.orientation == RailOrientation.VERTICAL else
                min(abs(rail.center - min_y), abs(rail.center - max_y)) <= margin
            ),
        }) for rail in bright_rails]
    # Reassign stable IDs after combining orientations.
    bright_rails = [StructuralRail(**{**rail.__dict__, "rail_id": index}) for index, rail in enumerate(bright_rails)]
    dark_rails = [StructuralRail(**{**rail.__dict__, "rail_id": index}) for index, rail in enumerate(dark_rails)]
    return tuple(bright_rails), tuple(dark_rails)


def _span_overlap(a: StructuralRail, b: StructuralRail) -> float:
    overlap = max(0.0, min(a.span_end, b.span_end) - max(a.span_start, b.span_start))
    return overlap / max(1.0, min(a.span_end - a.span_start, b.span_end - b.span_start))


def match_structural_rails(
    rgb_rails: tuple[StructuralRail, ...],
    ms_rails: tuple[StructuralRail, ...],
    config: LocalMeshConfig,
    dark: bool = False,
) -> tuple[RailMatch, ...]:
    """Mutual-nearest rail matching around identity with hard deviation bounds."""
    candidates: list[tuple[float, StructuralRail, StructuralRail, float]] = []
    for ms in ms_rails:
        for rgb in rgb_rails:
            if ms.orientation != rgb.orientation:
                continue
            limit = (config.exterior_rail_max_deviation_px
                     if (ms.is_exterior or rgb.is_exterior) and not dark
                     else config.internal_rail_max_deviation_px)
            deviation = rgb.center - ms.center
            overlap = _span_overlap(ms, rgb)
            if abs(deviation) > limit or overlap < 0.25:
                continue
            width_penalty = abs(rgb.width - ms.width) / max(rgb.width, ms.width, 1.0)
            cost = abs(deviation) / limit + 0.5 * (1.0 - overlap) + 0.25 * width_penalty
            confidence = max(0.0, 1.0 - cost)
            candidates.append((cost, ms, rgb, confidence))
    # Greedy one-to-one assignment in ascending structural cost prevents two MS
    # roads from locking onto the same bright RGB road.
    matches: list[RailMatch] = []
    used_ms: set[int] = set()
    used_rgb: set[int] = set()
    for _, ms, rgb, confidence in sorted(candidates, key=lambda item: item[0]):
        if ms.rail_id in used_ms or rgb.rail_id in used_rgb:
            continue
        used_ms.add(ms.rail_id)
        used_rgb.add(rgb.rail_id)
        matches.append(RailMatch(ms, rgb, rgb.center - ms.center, confidence))
    return tuple(matches)


def build_rail_bounded_regions(
    matches: tuple[RailMatch, ...], shape: tuple[int, int], config: LocalMeshConfig,
    dark_matches: tuple[RailMatch, ...] = (),
) -> tuple[RailBoundedRegion, ...]:
    height, width = shape
    vertical = sorted((match for match in matches if match.rgb_rail.orientation == RailOrientation.VERTICAL),
                      key=lambda match: match.rgb_rail.center)
    horizontal = sorted((match for match in matches if match.rgb_rail.orientation == RailOrientation.HORIZONTAL),
                        key=lambda match: match.rgb_rail.center)
    # Dark rails may subdivide an already-established white-road frame, but are
    # never permitted to replace its exterior anchors.
    dark_vertical = sorted((match for match in dark_matches if match.rgb_rail.orientation == RailOrientation.VERTICAL),
                           key=lambda match: match.rgb_rail.center)
    dark_horizontal = sorted((match for match in dark_matches if match.rgb_rail.orientation == RailOrientation.HORIZONTAL),
                             key=lambda match: match.rgb_rail.center)

    def merged_edges(primary, secondary, limit):
        edges = [(0, None)] + [(round(match.rgb_rail.center), match.rgb_rail.rail_id) for match in primary] + [(limit, None)]
        for match in secondary:
            position = round(match.rgb_rail.center)
            if min(abs(position - existing[0]) for existing in edges) < config.min_region_size_px:
                continue
            edges.append((position, -(match.rgb_rail.rail_id + 1)))  # negative IDs identify secondary rails
        return sorted(edges, key=lambda edge: edge[0])

    x_edges = merged_edges(vertical, dark_vertical, width)
    y_edges = merged_edges(horizontal, dark_horizontal, height)
    regions: list[RailBoundedRegion] = []
    for (x0, left), (x1, right) in zip(x_edges, x_edges[1:]):
        for (y0, top), (y1, bottom) in zip(y_edges, y_edges[1:]):
            if x1 - x0 < config.min_region_size_px or y1 - y0 < config.min_region_size_px:
                continue
            regions.append(RailBoundedRegion(x0, y0, x1, y1, left, right, top, bottom))
    return tuple(regions)


def derive_region_corrections(
    regions: tuple[RailBoundedRegion, ...],
    bright_matches: tuple[RailMatch, ...],
    dark_matches: tuple[RailMatch, ...],
) -> tuple[RailRegionCorrection, ...]:
    """Derive normal-axis correction only from rails enclosing each region.

    Vertical rail deviations constrain X and horizontal rail deviations constrain
    Y.  No free 2-D correlation is allowed here, so a region cannot jump to a
    different road or RGB patch.
    """
    by_id = {match.rgb_rail.rail_id: match for match in bright_matches}
    dark_by_encoded_id = {-(match.rgb_rail.rail_id + 1): match for match in dark_matches}
    corrections: list[RailRegionCorrection] = []
    for region in regions:
        x_matches: list[RailMatch] = []
        y_matches: list[RailMatch] = []
        for rail_id in (region.left_rail, region.right_rail):
            match = by_id.get(rail_id) or dark_by_encoded_id.get(rail_id)
            if match is not None and match.rgb_rail.orientation == RailOrientation.VERTICAL:
                x_matches.append(match)
        for rail_id in (region.top_rail, region.bottom_rail):
            match = by_id.get(rail_id) or dark_by_encoded_id.get(rail_id)
            if match is not None and match.rgb_rail.orientation == RailOrientation.HORIZONTAL:
                y_matches.append(match)
        dx = float(np.average([match.deviation_px for match in x_matches],
                              weights=[max(match.confidence, 0.05) for match in x_matches])) if x_matches else 0.0
        dy = float(np.average([match.deviation_px for match in y_matches],
                              weights=[max(match.confidence, 0.05) for match in y_matches])) if y_matches else 0.0
        constraints = x_matches + y_matches
        confidence = float(np.mean([match.confidence for match in constraints])) if constraints else 0.0
        corrections.append(RailRegionCorrection(region, dx, dy, confidence, len(x_matches), len(y_matches)))
    return tuple(corrections)


def build_rail_constrained_field(
    result: RailFrameResult,
    shape: tuple[int, int],
    config: LocalMeshConfig | None = None,
) -> RailConstrainedField:
    """Build continuous normal-axis interpolation through every matched rail."""
    height, width = shape
    # Only primary white-road rails control displacement. Dark rails remain in
    # ``result.regions`` as subdivision boundaries and never bend the field.
    matches = result.bright_matches

    def controls(orientation: RailOrientation, limit: int) -> tuple[np.ndarray, np.ndarray]:
        oriented = [match for match in matches if match.rgb_rail.orientation == orientation]
        if not oriented:
            return np.asarray([0.0, float(limit - 1)]), np.zeros(2)
        # Multiple rail fragments at the same axis coordinate are collapsed by
        # confidence-weighted averaging. Canvas edges inherit the nearest rail,
        # avoiding unconstrained extrapolation beyond the established frame.
        grouped: dict[float, list[RailMatch]] = {}
        for match in oriented:
            grouped.setdefault(float(match.rgb_rail.center), []).append(match)
        positions = sorted(grouped)
        deviations = []
        for position in positions:
            group = grouped[position]
            deviation = float(np.average(
                [match.deviation_px for match in group],
                weights=[max(match.confidence, 0.05) for match in group],
            ))
            if config is not None and abs(deviation) < config.rail_deviation_deadband_px:
                deviation = 0.0
            deviations.append(deviation)
        if positions[0] > 0:
            positions.insert(0, 0.0)
            deviations.insert(0, deviations[0])
        if positions[-1] < limit - 1:
            positions.append(float(limit - 1))
            deviations.append(deviations[-1])
        return np.asarray(positions), np.asarray(deviations)

    x_positions, x_deviations = controls(RailOrientation.VERTICAL, width)
    y_positions, y_deviations = controls(RailOrientation.HORIZONTAL, height)
    return RailConstrainedField(x_positions, x_deviations, y_positions, y_deviations)


def _masked_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av, bv = a[mask].astype(np.float64), b[mask].astype(np.float64)
    if av.size < 64 or np.std(av) == 0 or np.std(bv) == 0:
        return -1.0
    return float(np.corrcoef(av, bv)[0, 1])


def refine_regions_with_bounded_grayscale(
    rgb_gray: np.ndarray,
    ms_gray: np.ndarray,
    valid_mask: np.ndarray,
    regions: tuple[RailBoundedRegion, ...],
    config: LocalMeshConfig,
) -> tuple[RailRegionCorrection, ...]:
    """Find at most a tiny interior correction inside each irregular region.

    The rail boundaries define the crop and are excluded by a margin. Candidate
    shifts are scored on a deterministic training subset and must improve a
    disjoint held-out subset. This stage never sees pixels from another region.
    """
    corrections: list[RailRegionCorrection] = []
    radius = config.region_local_max_deviation_px
    margin = config.region_boundary_margin_px + radius
    for region in regions:
        x0, x1 = region.x0 + margin, region.x1 - margin
        y0, y1 = region.y0 + margin, region.y1 - margin
        if x1 - x0 < 16 or y1 - y0 < 16:
            corrections.append(RailRegionCorrection(region, 0.0, 0.0, 0.0, 0, 0, accepted=False))
            continue
        reference = rgb_gray[y0:y1, x0:x1]
        base_mask = valid_mask[y0:y1, x0:x1]
        yy, xx = np.indices(reference.shape)
        train_selector = ((xx + yy) % 5) != 0
        holdout_selector = ~train_selector
        candidates: list[tuple[float, int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                target = ms_gray[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
                target_mask = valid_mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
                mask = base_mask & target_mask & train_selector
                candidates.append((_masked_corr(reference, target, mask), dx, dy))
        best_score, best_dx, best_dy = max(candidates, key=lambda item: item[0])
        base_target = ms_gray[y0:y1, x0:x1]
        base_holdout = _masked_corr(reference, base_target, base_mask & holdout_selector)
        best_target = ms_gray[y0 - best_dy:y1 - best_dy, x0 - best_dx:x1 - best_dx]
        best_target_mask = valid_mask[y0 - best_dy:y1 - best_dy, x0 - best_dx:x1 - best_dx]
        best_holdout = _masked_corr(reference, best_target, base_mask & best_target_mask & holdout_selector)
        improvement = best_holdout - base_holdout
        boundary_hit = abs(best_dx) == radius or abs(best_dy) == radius
        accepted = (not boundary_hit and improvement >= config.region_min_holdout_improvement and
                    (best_dx != 0 or best_dy != 0))
        confidence = max(0.0, min(1.0, improvement / max(config.region_min_holdout_improvement * 4, 1e-6)))
        corrections.append(RailRegionCorrection(
            region, float(best_dx if accepted else 0), float(best_dy if accepted else 0), confidence,
            int(region.left_rail is not None) + int(region.right_rail is not None),
            int(region.top_rail is not None) + int(region.bottom_rail is not None),
            improvement, accepted,
        ))
    return tuple(corrections)


def compute_rail_frame(
    rgb_band: np.ndarray, ms_band: np.ndarray, rgb_mask: np.ndarray, ms_mask: np.ndarray,
    road_config: RoadGridConfig, local_config: LocalMeshConfig,
) -> RailFrameResult:
    common = rgb_mask.astype(bool) & ms_mask.astype(bool)
    rgb_bright, rgb_dark = detect_structural_rails(rgb_band, common, road_config, local_config)
    ms_bright, ms_dark = detect_structural_rails(ms_band, common, road_config, local_config)
    bright_matches = match_structural_rails(rgb_bright, ms_bright, local_config, dark=False)
    dark_matches = match_structural_rails(rgb_dark, ms_dark, local_config, dark=True)
    # Primary white roads establish the regions. Dark rails are retained as
    # secondary boundaries/evidence and will only subdivide a primary region in
    # the next solver stage after this graph passes real-data review.
    regions = build_rail_bounded_regions(bright_matches, rgb_band.shape, local_config, dark_matches)
    corrections = derive_region_corrections(regions, bright_matches, dark_matches)
    return RailFrameResult(
        rgb_bright, ms_bright, bright_matches, rgb_dark, ms_dark, dark_matches, regions, corrections,
    )
