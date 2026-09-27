"""Run an experimental local-mesh export as an atomic, standalone job.

This is deliberately separate from the normal CLI until native output QA has
been reviewed on multiple real pairs.  It is suitable for a background process:
the final TIFF appears only after all tiles are written successfully.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass, replace

import rasterio
import numpy as np

from drone_alignment.alignment.coarse import coarse_align
from drone_alignment.alignment.rail_frame_aligner import (
    build_rail_region_field, compute_rail_frame, refine_regions_with_bounded_grayscale,
)
from drone_alignment.alignment.feature_detector import normalize_to_uint8
from drone_alignment.alignment.transform_estimator import TransformResult, _fallback_phase_correlation
from drone_alignment.alignment.warper import warp_ms_with_displacement_field_tiled
from drone_alignment.config.schema import AlignmentConfig, CoarseAlignmentConfig, LocalMeshConfig
from drone_alignment.io.reader import read_metadata
from drone_alignment.quality.metrics import footprint_gate_failures
from drone_alignment.quality.visualization import generate_alignment_preview


def _config() -> AlignmentConfig:
    return AlignmentConfig(
        coarse=CoarseAlignmentConfig(registration_max_dimension=1024),
        local_mesh=LocalMeshConfig(
            enabled=True, grid_rows=5, grid_cols=5, cell_halo_px=16,
            max_search_radius_px=12, min_road_points=20, min_tree_points=20,
            min_match_confidence=0.45, ambiguity_margin=0.08,
            tree_only_min_confidence=0.55, tree_ambiguity_margin=0.12,
            max_neighbor_difference_px=8, field_support_radius_px=256,
            field_fallback_weight=0.20, rbf_smoothing=0.50, max_field_gradient=0.15,
        ),
    )


def _to_native_transform(transform: TransformResult, coarse) -> TransformResult:
    """Conjugate the compact registration affine into native output pixels."""
    profile = coarse.output_profile or coarse.target_profile
    reg_h, reg_w = coarse.rgb_valid_mask.shape
    sx, sy = profile["width"] / reg_w, profile["height"] / reg_h
    low = np.vstack([transform.matrix, [0.0, 0.0, 1.0]])
    native_to_registration = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float64)
    native = np.linalg.inv(native_to_registration) @ low @ native_to_registration
    matrix = native[:2, :]
    tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
    return replace(transform, matrix=matrix, translation_px=(tx, ty),
                   translation_m=(tx * coarse.target_gsd, ty * coarse.target_gsd))


@dataclass(frozen=True)
class LocalMeshExportResult:
    aligned_path: Path
    preview_path: Path
    review_path: Path
    native_global_transform: TransformResult
    sparse_coverage: float
    trusted_cells: int
    held_out_improvement: float | None
    field_name: str


def export_local_mesh_candidate(
    rgb_path: Path, ms_path: Path, output_dir: Path, config: AlignmentConfig | None = None,
) -> LocalMeshExportResult:
    """Export one staged local-warp candidate plus preview and review JSON."""
    cfg = config or _config()
    rgb_meta, ms_meta = read_metadata(rgb_path), read_metadata(ms_path)
    coarse = coarse_align(rgb_meta, ms_meta, cfg)
    channel = cfg.registration_channel_priority[0].value
    rgb = coarse.registration_bands_rgb[channel]
    ms = coarse.registration_bands_ms[channel]
    common = coarse.rgb_valid_mask & coarse.ms_valid_mask
    matrix, response = _fallback_phase_correlation(rgb, ms, common, return_response=True)
    global_registration = TransformResult(
        matrix=matrix, transform_type=cfg.transform.transform_type,
        translation_px=(float(matrix[0, 2]), float(matrix[1, 2])),
        translation_m=(float(matrix[0, 2]) * coarse.registration_gsd, float(matrix[1, 2]) * coarse.registration_gsd),
        rotation_deg=0.0, scale=(1.0, 1.0), inlier_ratio=0.0, num_inliers=0, num_total_matches=0,
        method="phase_correlation", channel_pair=f"{channel}-{channel}", phase_response=response,
    )
    frame = compute_rail_frame(
        rgb, ms, coarse.rgb_valid_mask, coarse.ms_valid_mask, cfg.road_grid, cfg.local_mesh,
    )
    if len(frame.bright_matches) < 2 or not frame.regions:
        raise RuntimeError(
            f"Rail frame rejected before export: bright_matches={len(frame.bright_matches)}, regions={len(frame.regions)}"
        )
    rgb_gray = normalize_to_uint8(rgb, coarse.rgb_valid_mask, apply_clahe=False)
    ms_gray = normalize_to_uint8(ms, coarse.ms_valid_mask, apply_clahe=False)
    corrections = refine_regions_with_bounded_grayscale(
        rgb_gray, ms_gray, common, frame.regions, cfg.local_mesh,
    )
    accepted_corrections = [correction for correction in corrections if correction.accepted]
    if not accepted_corrections:
        raise RuntimeError("Rail-bounded local matching found no independently improving region.")
    field = build_rail_region_field(corrections, cfg.local_mesh)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ms_path).stem
    final_path = output_dir / f"{stem}_local_mesh_candidate.tif"
    staged_path = output_dir / f".{stem}_local_mesh_candidate.run.partial.tif"
    preview_path = output_dir / f"{stem}_local_mesh_candidate_preview.png"
    review_path = output_dir / f"{stem}_local_mesh_candidate_review.json"
    native_global = _to_native_transform(global_registration, coarse)
    try:
        _, native_footprint = warp_ms_with_displacement_field_tiled(
            coarse, native_global, field, staged_path, cfg.warp, ms_meta,
            rgb_meta=rgb_meta, return_footprint=True,
        )
        footprint_failures = footprint_gate_failures(native_footprint, cfg.transform)
        if footprint_failures:
            raise RuntimeError("Local mesh native footprint rejected: " + "; ".join(footprint_failures))
        os.replace(staged_path, final_path)
    finally:
        if staged_path.exists():
            staged_path.unlink()
    with rasterio.open(final_path) as aligned:
        warped = aligned.read(cfg.ms_green_band_index, out_shape=rgb.shape, resampling=rasterio.enums.Resampling.average)
    generate_alignment_preview(rgb, warped, preview_path, rgb_mask=coarse.rgb_valid_mask,
                               max_dimension=cfg.quality.preview_max_dimension)
    review_path.write_text(json.dumps({
        "global_phase_response": response,
        "global_translation_registration_px": global_registration.translation_px,
        "rail_frame": {
            "bright_matches": len(frame.bright_matches), "dark_secondary_matches": len(frame.dark_matches),
            "regions": len(frame.regions), "accepted_region_corrections": len(accepted_corrections),
            "matches": [{"orientation": match.rgb_rail.orientation.value,
                         "rgb_center": match.rgb_rail.center, "ms_center": match.ms_rail.center,
                         "deviation_px": match.deviation_px, "confidence": match.confidence,
                         "exterior": match.rgb_rail.is_exterior} for match in frame.bright_matches],
            "corrections": [{"bounds": [item.region.x0, item.region.y0, item.region.x1, item.region.y1],
                             "dx": item.dx, "dy": item.dy, "holdout_improvement": item.local_improvement,
                             "accepted": item.accepted} for item in corrections],
        },
        "field_selection": field.name,
        "native_footprint": native_footprint.__dict__,
        "output": str(final_path), "preview": str(preview_path),
    }, indent=2), encoding="utf-8")
    return LocalMeshExportResult(
        aligned_path=final_path, preview_path=preview_path, review_path=review_path,
        native_global_transform=native_global,
        sparse_coverage=len(accepted_corrections) / len(frame.regions), trusted_cells=len(accepted_corrections),
        held_out_improvement=float(np.mean([item.local_improvement for item in accepted_corrections])),
        field_name=field.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an experimental staged local-mesh GeoTIFF candidate.")
    parser.add_argument("rgb_path", type=Path)
    parser.add_argument("ms_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export_local_mesh_candidate(args.rgb_path, args.ms_path, args.output_dir)
    print(result.aligned_path)
    print(result.preview_path)
    print(result.review_path)


if __name__ == "__main__":
    main()
