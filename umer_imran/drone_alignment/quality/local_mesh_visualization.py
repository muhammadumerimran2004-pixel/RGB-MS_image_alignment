"""Review-gate visualizations for sparse local registration evidence."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drone_alignment.alignment.feature_detector import normalize_to_uint8
from drone_alignment.alignment.local_mesh_aligner import CellMatchStatus, SparseDisplacementResult
from drone_alignment.alignment.displacement_field import ResidualField
from drone_alignment.alignment.rail_frame_aligner import RailFrameResult, RailOrientation


_STATUS_COLORS: dict[CellMatchStatus, tuple[int, int, int]] = {
    CellMatchStatus.ACCEPTED: (40, 210, 40),
    CellMatchStatus.NO_FEATURES: (120, 120, 120),
    CellMatchStatus.AMBIGUOUS_ROADS: (0, 180, 255),
    CellMatchStatus.SEARCH_BOUNDARY: (0, 0, 255),
    CellMatchStatus.ROAD_TREE_CONFLICT: (190, 0, 190),
    CellMatchStatus.LOW_CONFIDENCE: (0, 220, 220),
    CellMatchStatus.SPATIAL_OUTLIER: (0, 80, 255),
}


def generate_sparse_vector_diagnostic(
    reference_band: np.ndarray,
    valid_mask: np.ndarray,
    result: SparseDisplacementResult,
    output_path: Path,
    max_dimension: int = 1600,
) -> Path:
    """Render a review image; accepted vectors show residuals beyond global shift.

    The image is intentionally diagnostic rather than a proof of accuracy.  It
    lets a reviewer compare accepted/rejected cells to roads and to the existing
    checkerboard before allowing a non-rigid warp to be implemented.
    """
    if reference_band.shape != valid_mask.shape:
        raise ValueError("Reference band and valid mask must have matching shapes.")
    if not result.samples:
        raise ValueError("Cannot visualize an empty sparse-vector result.")
    gray = normalize_to_uint8(reference_band, valid_mask, apply_clahe=False)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        canvas = cv2.resize(canvas, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)

    rows = max(sample.row for sample in result.samples) + 1
    cols = max(sample.col for sample in result.samples) + 1
    draw_h, draw_w = canvas.shape[:2]
    font_scale = max(0.32, min(0.6, draw_w / 2500))
    thickness = 1 if draw_w < 1200 else 2

    for row in range(1, rows):
        y = round(row * draw_h / rows)
        cv2.line(canvas, (0, y), (draw_w - 1, y), (180, 180, 180), 1, cv2.LINE_AA)
    for col in range(1, cols):
        x = round(col * draw_w / cols)
        cv2.line(canvas, (x, 0), (x, draw_h - 1), (180, 180, 180), 1, cv2.LINE_AA)

    # Exaggerate residual arrows just enough to be visible at review scale. The
    # label preserves their true pixel values.
    arrow_scale = max(4.0, min(14.0, draw_w / 140))
    for sample in result.samples:
        color = _STATUS_COLORS[sample.status]
        x, y = round(sample.center_xy[0] * scale), round(sample.center_xy[1] * scale)
        cell_x0, cell_y0 = round(sample.col * draw_w / cols), round(sample.row * draw_h / rows)
        cell_x1, cell_y1 = round((sample.col + 1) * draw_w / cols), round((sample.row + 1) * draw_h / rows)
        cv2.rectangle(canvas, (cell_x0 + 2, cell_y0 + 2), (cell_x1 - 2, cell_y1 - 2), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), max(2, thickness + 1), color, -1, cv2.LINE_AA)
        if sample.status == CellMatchStatus.ACCEPTED:
            rx, ry = sample.residual_dx_dy
            endpoint = (round(x + rx * arrow_scale), round(y + ry * arrow_scale))
            cv2.arrowedLine(canvas, (x, y), endpoint, color, thickness, cv2.LINE_AA, tipLength=0.25)
        label = f"{sample.status.value[:3]}/{sample.evidence[:1]} {sample.confidence:.2f}"
        cv2.putText(canvas, label, (cell_x0 + 5, cell_y0 + max(14, round(18 * font_scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    legend = [
        (CellMatchStatus.ACCEPTED, "accepted: local residual arrow"),
        (CellMatchStatus.NO_FEATURES, "no features"),
        (CellMatchStatus.AMBIGUOUS_ROADS, "ambiguous road"),
        (CellMatchStatus.ROAD_TREE_CONFLICT, "road/tree conflict"),
        (CellMatchStatus.LOW_CONFIDENCE, "low confidence"),
    ]
    legend_y = 20
    for status, label in legend:
        cv2.putText(canvas, label, (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    _STATUS_COLORS[status], thickness, cv2.LINE_AA)
        legend_y += max(15, round(23 * font_scale))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not write sparse-vector diagnostic: {destination}")
    return destination


def generate_displacement_field_diagnostic(
    reference_band: np.ndarray,
    valid_mask: np.ndarray,
    field: ResidualField,
    output_path: Path,
    max_dimension: int = 1600,
    vector_grid: int = 17,
) -> Path:
    """Render residual magnitude and regularly sampled field arrows for review."""
    gray = normalize_to_uint8(reference_band, valid_mask, apply_clahe=False)
    height, width = gray.shape
    ys = np.linspace(0.0, float(height - 1), vector_grid)
    xs = np.linspace(0.0, float(width - 1), vector_grid)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    vectors = field.evaluate(np.column_stack([xx.ravel(), yy.ravel()])).reshape(vector_grid, vector_grid, 2)
    magnitude = np.linalg.norm(vectors, axis=2)
    dense = cv2.resize(magnitude.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    scale_max = max(float(np.percentile(dense, 99)), 0.05)
    heat = cv2.applyColorMap(np.clip(dense / scale_max * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = cv2.addWeighted(base, 0.45, heat, 0.55, 0.0)
    arrow_scale = max(12.0, min(48.0, width / vector_grid * 0.35))
    for y_index, y in enumerate(ys):
        for x_index, x in enumerate(xs):
            vector = vectors[y_index, x_index]
            start = (round(x), round(y))
            end = (round(x + vector[0] * arrow_scale), round(y + vector[1] * arrow_scale))
            cv2.arrowedLine(canvas, start, end, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.25)
    if max(height, width) > max_dimension:
        factor = max_dimension / max(height, width)
        canvas = cv2.resize(canvas, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, f"{field.name}: residual magnitude (0 to {scale_max:.2f} px)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{field.name}: residual magnitude (0 to {scale_max:.2f} px)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not write displacement-field diagnostic: {destination}")
    return destination


def generate_rail_frame_diagnostic(
    reference_band: np.ndarray,
    valid_mask: np.ndarray,
    result: RailFrameResult,
    output_path: Path,
    max_dimension: int = 1600,
) -> Path:
    """Draw matched bright frame rails, secondary dark rails, and variable regions."""
    gray = normalize_to_uint8(reference_band, valid_mask, apply_clahe=False)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    height, width = gray.shape
    for region in result.regions:
        cv2.rectangle(canvas, (region.x0, region.y0), (region.x1, region.y1), (120, 120, 120), 1)
    for match in result.dark_matches:
        rail = match.rgb_rail
        if rail.orientation == RailOrientation.HORIZONTAL:
            cv2.line(canvas, (round(rail.span_start), round(rail.center)),
                     (round(rail.span_end), round(rail.center)), (200, 80, 200), 1, cv2.LINE_AA)
        else:
            cv2.line(canvas, (round(rail.center), round(rail.span_start)),
                     (round(rail.center), round(rail.span_end)), (200, 80, 200), 1, cv2.LINE_AA)
    for match in result.bright_matches:
        rail = match.rgb_rail
        color = (0, 255, 0) if rail.is_exterior else (0, 220, 255)
        if rail.orientation == RailOrientation.HORIZONTAL:
            start, end = (round(rail.span_start), round(rail.center)), (round(rail.span_end), round(rail.center))
            label_at = (max(2, start[0]), max(14, start[1] - 4))
        else:
            start, end = (round(rail.center), round(rail.span_start)), (round(rail.center), round(rail.span_end))
            label_at = (max(2, start[0] + 3), max(14, start[1] + 14))
        cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"d={match.deviation_px:+.1f} c={match.confidence:.2f}", label_at,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "green=exterior frame; yellow=internal white road; magenta=dark secondary", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    if max(height, width) > max_dimension:
        factor = max_dimension / max(height, width)
        canvas = cv2.resize(canvas, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not write rail-frame diagnostic: {destination}")
    return destination
