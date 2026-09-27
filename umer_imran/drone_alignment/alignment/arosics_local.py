"""AROSICS COREG_LOCAL refinement of an already-verified global result.

Design (see ALIGNMENT_V4_AROSICS_TPS_BLUEPRINT.md Phase 2): the verified
global candidate (ORB/SIFT/phase/AROSICS-global) is estimated first;
this module only ever *refines* it with a small local search, and AROSICS
performs the warp itself (its own DESHIFTER, or an equivalent streaming GDAL
thin-plate-spline warp for large rasters). Every acceptance gate here is
additional verification layered on top of AROSICS' own tie-point filtering,
not a restriction on what AROSICS is allowed to do; a rejection here always
means the caller safely publishes the global result instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from scipy.spatial import cKDTree

from drone_alignment.alignment.arosics_staging import (
    STAGE_NODATA,
    ArosicsExecutionError,
    compute_global_map_correction,
    create_whitespace_safe_alias,
    stage_precorrected_target_band,
    stage_reference_band,
)
from drone_alignment.alignment.arosics_matcher import is_arosics_available
from drone_alignment.alignment.rejections import LocalRefinementRejected
from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.alignment.warper import evaluate_written_footprint
from drone_alignment.config.schema import AlignmentConfig, ArosicsLocalConfig
from drone_alignment.io.reader import RasterMetadata
from drone_alignment.quality.metrics import FootprintMetrics, SpatialResidualReport, footprint_gate_failures

logger = logging.getLogger("drone_alignment")


# ---------------------------------------------------------------------------
# Tie-point extraction and filtering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TiePoint:
    point_id: int
    staged_px: tuple[float, float]
    source_px: tuple[float, float]
    map_xy: tuple[float, float]
    shift_m: tuple[float, float]
    shift_px: float  # in target (MS) pixels; computed by us, not trusted from AROSICS' ABS_SHIFT (map units)
    reliability: float
    ssim_before: float | None
    ssim_after: float | None
    ssim_improved: bool | None
    flags: dict[str, bool] = field(default_factory=dict)
    role: str = "fit"  # "fit" | "holdout"

    @property
    def is_outlier(self) -> bool:
        return bool(self.flags.get("OUTLIER", False)) or bool(self.flags.get("L4_OUTLIER", False))


_AROSICS_OUTFILL_VAL = -9999.0


def _extract_tie_points(table, staged_to_source_px: Affine, ms_gsd: float) -> list[TiePoint]:
    """Convert AROSICS' CoRegPoints_table (a GeoDataFrame) into TiePoint records.

    Rows with no match (``ABS_SHIFT == outFillVal``) are dropped entirely.
    ``ABS_SHIFT`` is AROSICS' vector length in *map units* (metres), not
    pixels (``Tie_Point_Grid._get_spatial_shifts`` sets it from
    ``CR.vec_length_map``); ``shift_px`` here is computed independently from
    ``X_SHIFT_M``/``Y_SHIFT_M`` so units are never ambiguous downstream.
    """
    columns = set(table.columns)
    points: list[TiePoint] = []
    for _, row in table.iterrows():
        abs_shift = row.get("ABS_SHIFT")
        if abs_shift is None or not np.isfinite(abs_shift) or float(abs_shift) == _AROSICS_OUTFILL_VAL:
            continue
        flags = {}
        for level_col in ("OUTLIER", "L1_OUTLIER", "L2_OUTLIER", "L3_OUTLIER"):
            if level_col in columns:
                flags[level_col] = bool(row[level_col])
        staged_px = (float(row["X_IM"]), float(row["Y_IM"]))
        source_px_raw = staged_to_source_px * staged_px
        shift_m = (float(row["X_SHIFT_M"]), float(row["Y_SHIFT_M"]))
        reliability_val = row.get("RELIABILITY")
        reliability = float(reliability_val) if reliability_val is not None and np.isfinite(reliability_val) else 0.0
        points.append(TiePoint(
            point_id=int(row["POINT_ID"]),
            staged_px=staged_px,
            source_px=(float(source_px_raw[0]), float(source_px_raw[1])),
            map_xy=(float(row["X_MAP"]), float(row["Y_MAP"])),
            shift_m=shift_m,
            shift_px=float(math.hypot(*shift_m) / ms_gsd),
            reliability=reliability,
            ssim_before=float(row["SSIM_BEFORE"]) if "SSIM_BEFORE" in columns and np.isfinite(row["SSIM_BEFORE"]) else None,
            ssim_after=float(row["SSIM_AFTER"]) if "SSIM_AFTER" in columns and np.isfinite(row["SSIM_AFTER"]) else None,
            ssim_improved=bool(row["SSIM_IMPROVED"]) if "SSIM_IMPROVED" in columns and row["SSIM_IMPROVED"] is not None
                          and not (isinstance(row["SSIM_IMPROVED"], float) and math.isnan(row["SSIM_IMPROVED"])) else None,
            flags=flags,
        ))
    return points


def _neighbour_filter(points: list[TiePoint], local_cfg: ArosicsLocalConfig, ms_gsd: float) -> list[TiePoint]:
    """Return points with an added L4_OUTLIER flag where the shift disagrees with nearby points."""
    valid = [p for p in points if not p.is_outlier]
    if not local_cfg.neighbour_filter or len(valid) < local_cfg.neighbour_k + 1:
        return points
    map_xy = np.array([p.map_xy for p in valid])
    shifts_m = np.array([p.shift_m for p in valid])
    tree = cKDTree(map_xy)
    k = min(local_cfg.neighbour_k + 1, len(valid))
    _, neighbour_idx = tree.query(map_xy, k=k)
    limit_m = local_cfg.max_neighbour_deviation_px * ms_gsd
    flagged_point_ids: set[int] = set()
    for row_index, point in enumerate(valid):
        neighbours = [int(idx) for idx in np.atleast_1d(neighbour_idx[row_index]) if idx != row_index]
        if not neighbours:
            continue
        median_shift_m = np.median(shifts_m[neighbours], axis=0)
        deviation_m = float(np.linalg.norm(shifts_m[row_index] - median_shift_m))
        if deviation_m > limit_m:
            flagged_point_ids.add(point.point_id)
    if not flagged_point_ids:
        return points
    updated = []
    for point in points:
        if point.point_id in flagged_point_ids:
            new_flags = dict(point.flags)
            new_flags["L4_OUTLIER"] = True
            updated.append(TiePoint(
                point.point_id, point.staged_px, point.source_px, point.map_xy, point.shift_m, point.shift_px,
                point.reliability, point.ssim_before, point.ssim_after, point.ssim_improved, new_flags, point.role,
            ))
        else:
            updated.append(point)
    return updated


# ---------------------------------------------------------------------------
# Coverage and holdout split
# ---------------------------------------------------------------------------

def _cell_indices(
    map_xy: np.ndarray, registration_transform: Affine, registration_shape: tuple[int, int], grid_n: int,
) -> np.ndarray:
    """Return (row, col) coverage-grid cell index for each map_xy point."""
    height, width = registration_shape
    inverse = ~registration_transform
    reg_xy = np.array([inverse * tuple(point) for point in map_xy])
    col = np.clip((reg_xy[:, 0] / max(width, 1) * grid_n).astype(int), 0, grid_n - 1)
    row = np.clip((reg_xy[:, 1] / max(height, 1) * grid_n).astype(int), 0, grid_n - 1)
    return np.column_stack([row, col])


def _coverage_metrics(
    valid_points: list[TiePoint], common_mask: np.ndarray, registration_transform: Affine, local_cfg: ArosicsLocalConfig,
) -> tuple[float, float]:
    """Return (cell_occupancy, hull_coverage) for the valid tie points against the common valid mask."""
    grid_n = local_cfg.coverage_grid
    height, width = common_mask.shape
    cell_h, cell_w = max(1, height // grid_n), max(1, width // grid_n)
    eligible_cells = 0
    for row in range(grid_n):
        for col in range(grid_n):
            cell = common_mask[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]
            if cell.size and cell.mean() >= 0.25:
                eligible_cells += 1
    if not valid_points:
        return 0.0, 0.0
    map_xy = np.array([p.map_xy for p in valid_points])
    cell_idx = _cell_indices(map_xy, registration_transform, common_mask.shape, grid_n)
    occupied_cells = {(int(r), int(c)) for r, c in cell_idx}
    occupancy = len(occupied_cells) / max(1, eligible_cells)

    import cv2
    inverse = ~registration_transform
    reg_xy = np.array([inverse * tuple(point) for point in map_xy]).astype(np.float32)
    hull_area = 0.0
    if len(reg_xy) >= 3:
        hull = cv2.convexHull(reg_xy)
        hull_area = float(cv2.contourArea(hull))
    common_area = float(common_mask.sum())
    hull_coverage = hull_area / common_area if common_area > 0 else 0.0
    return occupancy, hull_coverage


def _split_fit_holdout(
    valid_points: list[TiePoint], common_mask: np.ndarray, registration_transform: Affine, local_cfg: ArosicsLocalConfig,
) -> tuple[list[TiePoint], list[TiePoint]]:
    """Deterministically, spatially-stratified split into fit and holdout sets.

    One point per occupied coverage cell is preferentially chosen for holdout
    (never the only point in its cell), cycling through cells in row-major
    order until the target holdout count is reached.
    """
    n_valid = len(valid_points)
    target_holdout = max(local_cfg.min_holdout_points, round(local_cfg.holdout_fraction * n_valid))
    grid_n = local_cfg.coverage_grid
    map_xy = np.array([p.map_xy for p in valid_points])
    cell_idx = _cell_indices(map_xy, registration_transform, common_mask.shape, grid_n)

    cells: dict[tuple[int, int], list[int]] = {}
    for i, (row, col) in enumerate(cell_idx):
        cells.setdefault((int(row), int(col)), []).append(i)

    height, width = common_mask.shape
    cell_h, cell_w = max(1, height // grid_n), max(1, width // grid_n)
    inverse = ~registration_transform

    holdout_indices: list[int] = []
    for (row, col), members in sorted(cells.items()):
        if len(holdout_indices) >= target_holdout:
            break
        if len(members) < 2:
            continue  # never take the only point in a cell out of the fit set
        center_reg_x = (col + 0.5) * cell_w
        center_reg_y = (row + 0.5) * cell_h
        best_idx, best_dist = None, math.inf
        for i in members:
            reg_x, reg_y = inverse * tuple(map_xy[i])
            dist = (reg_x - center_reg_x) ** 2 + (reg_y - center_reg_y) ** 2
            if dist < best_dist:
                best_idx, best_dist = i, dist
        holdout_indices.append(best_idx)

    holdout_set = set(holdout_indices)
    fit = [p for i, p in enumerate(valid_points) if i not in holdout_set]
    holdout = [p for i, p in enumerate(valid_points) if i in holdout_set]
    return fit, holdout


# ---------------------------------------------------------------------------
# Result / publication types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArosicsLocalPublication:
    aligned_path: Path
    footprint: FootprintMetrics
    band_pair_name: str
    reference_band_index: int
    target_band_index: int
    payload: dict[str, Any]
    tie_points_geojson_path: Path | None


def _resolve_band_pairs(cfg: AlignmentConfig) -> list[tuple[str, int, int]]:
    if cfg.arosics.band_pairs:
        return [(pair.name, pair.reference_band, pair.target_band) for pair in cfg.arosics.band_pairs]
    channel = cfg.registration_channel_priority[0].value
    ref_band = cfg.rgb_green_band_index if channel == "green" else cfg.rgb_red_band_index
    tgt_band = cfg.ms_green_band_index if channel == "green" else cfg.ms_red_band_index
    return [(f"{channel}-{channel}", ref_band, tgt_band)]


def _resolve_max_shift_ref_px(
    local_cfg: ArosicsLocalConfig, global_quality: SpatialResidualReport, registration_gsd: float, rgb_gsd: float,
) -> tuple[int, dict]:
    if local_cfg.max_shift_m is not None:
        max_shift_m, source = local_cfg.max_shift_m, "configured"
    else:
        rmse_px = global_quality.global_rmse_px
        if rmse_px is not None:
            max_shift_m = float(np.clip(3.0 * rmse_px * registration_gsd, 0.10, 0.30))
            source = "auto_from_global_rmse"
        else:
            max_shift_m, source = 0.25, "auto_default"
    if local_cfg.periodic_texture_period_m is not None:
        limit = 0.4 * local_cfg.periodic_texture_period_m
        if max_shift_m > limit:
            max_shift_m, source = limit, source + "+periodic_guard"
    max_shift_ref_px = max(2, int(math.ceil(max_shift_m / rgb_gsd)))
    return max_shift_ref_px, {"max_shift_m": max_shift_m, "source": source}


def _resolve_grid_res_px(local_cfg: ArosicsLocalConfig, staged_width: int, staged_height: int) -> int:
    if local_cfg.grid_res_px is not None:
        return local_cfg.grid_res_px
    return max(16, math.ceil(max(staged_width, staged_height) / local_cfg.target_points_per_axis))


# ---------------------------------------------------------------------------
# Warp engines
# ---------------------------------------------------------------------------

def _estimate_warp_memory_gb(ms_meta: RasterMetadata, output_width: int, output_height: int) -> float:
    itemsize = np.dtype(ms_meta.dtype).itemsize
    input_gb = ms_meta.width * ms_meta.height * ms_meta.band_count * itemsize * 2 / 1e9
    output_gb = output_width * output_height * ms_meta.band_count * itemsize / 1e9
    return input_gb + output_gb


def _select_warp_engine(local_cfg: ArosicsLocalConfig, ms_meta: RasterMetadata, output_width: int, output_height: int) -> tuple[str, float]:
    estimate_gb = _estimate_warp_memory_gb(ms_meta, output_width, output_height)
    if local_cfg.warp_engine != "auto":
        return local_cfg.warp_engine, estimate_gb
    engine = "deshifter" if estimate_gb <= local_cfg.max_in_memory_warp_gb else "gdal_tps"
    return engine, estimate_gb


def _build_gcps(fit_points: list[TiePoint]):
    from osgeo import gdal
    return [
        gdal.GCP(
            point.map_xy[0] + point.shift_m[0], point.map_xy[1] + point.shift_m[1], 0.0,
            point.source_px[0], point.source_px[1],
        )
        for point in fit_points
    ]


def _warp_with_deshifter(
    target_full_alias: Path, coreg_info: dict, gcps, output_profile: dict, resampling: str, nodata_value: float,
    cpus: int | None, safe_output: Path,
) -> None:
    from arosics import DESHIFTER

    profile_transform = output_profile["transform"]
    gsd_x, gsd_y = profile_transform.a, -profile_transform.e
    target_xy_grid = [[profile_transform.c, profile_transform.c + gsd_x],
                       [profile_transform.f, profile_transform.f - gsd_y]]
    clipextent = (
        profile_transform.c, profile_transform.f - gsd_y * output_profile["height"],
        profile_transform.c + gsd_x * output_profile["width"], profile_transform.f,
    )
    info = dict(coreg_info)
    info["GCPList"] = gcps
    DESHIFTER(
        str(target_full_alias), info, path_out=str(safe_output), fmt_out="GTIFF",
        out_crea_options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
        target_xyGrid=target_xy_grid, clipextent=clipextent,
        resamp_alg=resampling, nodata=nodata_value, min_points_local_corr=len(gcps),
        CPUs=cpus, progress=False, q=True,
    ).correct_shifts()


_RESAMPLING_GDAL = {"nearest": "NEAR", "bilinear": "BILINEAR", "cubic": "CUBIC"}


def _warp_with_gdal_tps(
    target_full_alias: Path, gcps, crs_wkt: str, output_profile: dict, resampling: str, nodata_value: float,
    memory_mb: int, safe_output: Path,
) -> None:
    from osgeo import gdal
    gdal.UseExceptions()
    profile_transform = output_profile["transform"]
    gsd_x, gsd_y = profile_transform.a, -profile_transform.e
    bounds = (
        profile_transform.c, profile_transform.f - gsd_y * output_profile["height"],
        profile_transform.c + gsd_x * output_profile["width"], profile_transform.f,
    )
    vrt_path = "/vsimem/arosics_gcp_stage.vrt"
    src = gdal.Translate(vrt_path, str(target_full_alias), format="VRT", GCPs=gcps, outputSRS=crs_wkt)
    if src is None:
        raise ArosicsExecutionError("GDAL could not stage a GCP VRT for the streaming TPS warp.")
    try:
        result = gdal.Warp(
            str(safe_output), src, format="GTiff", tps=True,
            outputBounds=bounds, xRes=gsd_x, yRes=gsd_y,
            resampleAlg=_RESAMPLING_GDAL.get(resampling, "BILINEAR"), dstNodata=nodata_value,
            warpMemoryLimit=memory_mb * 1024 * 1024, multithread=True,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
        )
        if result is None:
            raise ArosicsExecutionError("GDAL streaming TPS warp did not produce an output dataset.")
        result = None
    finally:
        gdal.Unlink(vrt_path)


# ---------------------------------------------------------------------------
# Holdout / full-grid verification
# ---------------------------------------------------------------------------

def _measure_holdout_shifts(
    staged_ref_path: Path, aligned_matching_band_path: Path, holdout_points: list[TiePoint],
    window_size: tuple[int, int], max_shift_ref_px: int, ms_gsd: float,
) -> list[float | None]:
    """For each holdout point, run a windowed global COREG between the staged reference
    and the aligned output's matching band, at the point's expected (corrected) position.

    Holdout points were excluded from the GCP set, so this measures genuine,
    independent post-warp accuracy - unlike checking residuals at the fit
    points themselves, which a GCP-interpolating warp reproduces exactly by
    construction (residual == 0 there regardless of accuracy elsewhere).
    """
    from arosics import COREG

    after_px: list[float | None] = []
    for point in holdout_points:
        expected_xy = (point.map_xy[0] + point.shift_m[0], point.map_xy[1] + point.shift_m[1])
        try:
            coreg = COREG(
                im_ref=str(staged_ref_path), im_tgt=str(aligned_matching_band_path),
                wp=expected_xy, ws=window_size, max_shift=max_shift_ref_px,
                nodata=(STAGE_NODATA, STAGE_NODATA), ignore_errors=True, q=True, v=False,
            )
            coreg.calculate_spatial_shifts()
            if not getattr(coreg, "success", False):
                after_px.append(None)
                continue
            residual_m = math.hypot(coreg.x_shift_map, coreg.y_shift_map)
            after_px.append(residual_m / ms_gsd)
        except Exception:  # noqa: BLE001 - a single holdout window failing is not fatal
            after_px.append(None)
    return after_px


def _run_holdout_gates(before_px: list[float], after_px: list[float | None], local_cfg: ArosicsLocalConfig) -> dict:
    verifiable = [(b, a) for b, a in zip(before_px, after_px) if a is not None]
    if len(verifiable) < local_cfg.min_holdout_points:
        raise LocalRefinementRejected(
            "HOLDOUT_FAILED", f"Only {len(verifiable)} of {len(before_px)} holdout points could be verified; "
            f"{local_cfg.min_holdout_points} required.",
            {"verifiable": len(verifiable), "total": len(before_px)},
        )
    befores = np.array([b for b, _ in verifiable])
    afters = np.array([a for _, a in verifiable])
    median_before, median_after = float(np.median(befores)), float(np.median(afters))
    win_fraction = float(np.mean(afters < befores))
    max_after = float(np.max(afters))
    stats = {
        "median_before_px": median_before, "median_after_px": median_after,
        "win_fraction": win_fraction, "max_after_px": max_after,
        "unverifiable": len(before_px) - len(verifiable),
    }
    if median_after > local_cfg.max_holdout_ratio * median_before:
        raise LocalRefinementRejected(
            "HOLDOUT_FAILED",
            f"Holdout median residual after refinement ({median_after:.2f}px) did not improve enough "
            f"over before ({median_before:.2f}px, ratio limit {local_cfg.max_holdout_ratio}).",
            stats,
        )
    if win_fraction < local_cfg.min_holdout_win_fraction:
        raise LocalRefinementRejected(
            "HOLDOUT_FAILED",
            f"Only {win_fraction:.0%} of holdout points improved; "
            f"{local_cfg.min_holdout_win_fraction:.0%} required.",
            stats,
        )
    if max_after > local_cfg.max_holdout_after_px:
        raise LocalRefinementRejected(
            "HOLDOUT_FAILED",
            f"Worst holdout residual after refinement ({max_after:.2f}px) exceeds the limit "
            f"({local_cfg.max_holdout_after_px}px).",
            stats,
        )
    return stats


# ---------------------------------------------------------------------------
# Per-band-pair attempt
# ---------------------------------------------------------------------------

def _attempt_band_pair(
    rgb_meta: RasterMetadata,
    ms_meta: RasterMetadata,
    common_mask: np.ndarray,
    registration_transform: Affine,
    registration_gsd: float,
    global_transform: TransformResult,
    global_quality: SpatialResidualReport,
    output_profile: dict,
    pair_name: str,
    ref_band: int,
    tgt_band: int,
    cfg: AlignmentConfig,
    output_dir: Path,
    active_log: logging.Logger,
) -> ArosicsLocalPublication:
    local_cfg = cfg.arosics.local

    try:
        from arosics import COREG_LOCAL
    except ImportError as exc:
        raise LocalRefinementRejected("AROSICS_UNAVAILABLE", f"AROSICS is not importable: {exc}")

    with tempfile.TemporaryDirectory(prefix="arosics_local_") as temp_dir:
        temp_path = Path(temp_dir)
        staged_ref_path = temp_path / "reference_match.tif"
        staged_tgt_path = temp_path / "target_match.tif"

        try:
            stage_reference_band(rgb_meta, ref_band, staged_ref_path)
            correction = compute_global_map_correction(registration_transform, global_transform.matrix, ms_meta.transform)
            staged_tgt = stage_precorrected_target_band(ms_meta, tgt_band, correction, staged_tgt_path)
        except ArosicsExecutionError:
            raise  # operational error - propagate, do not silently fall back

        max_shift_ref_px, max_shift_info = _resolve_max_shift_ref_px(
            local_cfg, global_quality, registration_gsd, rgb_meta.gsd,
        )
        with rasterio.open(staged_tgt_path) as staged_tgt_ds:
            staged_width, staged_height = staged_tgt_ds.width, staged_tgt_ds.height
        grid_res_px = _resolve_grid_res_px(local_cfg, staged_width, staged_height)

        try:
            coreg_local = COREG_LOCAL(
                im_ref=str(staged_ref_path), im_tgt=str(staged_tgt_path),
                grid_res=grid_res_px, window_size=local_cfg.window_size,
                r_b4match=1, s_b4match=1, max_iter=local_cfg.max_iter, max_shift=max_shift_ref_px,
                tieP_filter_level=local_cfg.tie_point_filter_level, tieP_random_state=0,
                min_reliability=local_cfg.min_reliability,
                rs_max_outlier=local_cfg.rs_max_outlier, rs_tolerance=local_cfg.rs_tolerance, rs_random_state=0,
                align_grids=False, resamp_alg_calc="cubic",
                outFillVal=int(_AROSICS_OUTFILL_VAL), nodata=(STAGE_NODATA, STAGE_NODATA),
                projectDir=str(temp_path), CPUs=local_cfg.cpus, progress=False, q=True, ignore_errors=True,
            )
        except RuntimeError as exc:
            # COREG_LOCAL's constructor test-matches one window at the overlap centre and
            # raises when that window can't fit: a small field, not a malfunction.
            if "larger than overlap area" in str(exc):
                raise LocalRefinementRejected(
                    "WINDOW_EXCEEDS_OVERLAP",
                    f"The RGB/MS overlap is smaller than the {local_cfg.window_size} px AROSICS matching window.",
                    {"window_size": list(local_cfg.window_size)},
                ) from exc
            raise ArosicsExecutionError(f"AROSICS COREG_LOCAL setup error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - AROSICS may raise a range of internal errors
            raise ArosicsExecutionError(f"AROSICS COREG_LOCAL setup error: {exc}") from exc
        try:
            coreg_local.calculate_spatial_shifts()
        except Exception as exc:  # noqa: BLE001 - AROSICS may raise a range of internal errors
            raise ArosicsExecutionError(f"AROSICS COREG_LOCAL execution error: {exc}") from exc

        table = coreg_local.CoRegPoints_table
        if table is None or table.empty:
            raise LocalRefinementRejected(
                "INSUFFICIENT_TIE_POINTS", "AROSICS COREG_LOCAL produced no tie points.",
                {"grid_res_px": grid_res_px},
            )

        all_points = _extract_tie_points(table, staged_tgt.staged_to_source_px, ms_meta.gsd)
        all_points = _neighbour_filter(all_points, local_cfg, ms_meta.gsd)
        valid_points = [p for p in all_points if not p.is_outlier]

        tie_point_summary = {
            "grid": int(len(table)), "no_match": int(len(table)) - len(all_points),
            "valid": len(valid_points),
        }
        for level in ("L1_OUTLIER", "L2_OUTLIER", "L3_OUTLIER", "L4_OUTLIER"):
            tie_point_summary[level] = sum(1 for p in all_points if p.flags.get(level))

        if len(valid_points) < local_cfg.min_valid_tie_points:
            raise LocalRefinementRejected(
                "INSUFFICIENT_TIE_POINTS",
                f"AROSICS produced {len(valid_points)} valid tie points; "
                f"{local_cfg.min_valid_tie_points} required.",
                {"tie_points": tie_point_summary},
            )

        occupancy, hull_coverage = _coverage_metrics(valid_points, common_mask, registration_transform, local_cfg)
        coverage_info = {"cell_occupancy": occupancy, "hull_coverage": hull_coverage}
        if occupancy < local_cfg.min_cell_occupancy or hull_coverage < local_cfg.min_hull_coverage:
            raise LocalRefinementRejected(
                "INSUFFICIENT_COVERAGE",
                f"Tie-point coverage (occupancy={occupancy:.2f}, hull={hull_coverage:.2f}) is below "
                f"the minimum (occupancy={local_cfg.min_cell_occupancy}, hull={local_cfg.min_hull_coverage}).",
                {"tie_points": tie_point_summary, "coverage": coverage_info},
            )

        fit_points, holdout_points = _split_fit_holdout(valid_points, common_mask, registration_transform, local_cfg)
        if len(fit_points) < 6 or len(holdout_points) < local_cfg.min_holdout_points:
            raise LocalRefinementRejected(
                "INSUFFICIENT_TIE_POINTS",
                f"Fit/holdout split left {len(fit_points)} fit and {len(holdout_points)} holdout points.",
                {"tie_points": tie_point_summary},
            )
        tie_point_summary["fit"] = len(fit_points)
        tie_point_summary["holdout"] = len(holdout_points)

        before_px = [p.shift_px for p in holdout_points]
        if float(np.median(before_px)) < local_cfg.no_gain_floor_px:
            raise LocalRefinementRejected(
                "NO_LOCAL_GAIN",
                f"Verified global alignment already leaves a median holdout residual of "
                f"{float(np.median(before_px)):.2f}px, at or below the {local_cfg.no_gain_floor_px}px floor "
                "for attempting a local refinement.",
                {"median_before_px": float(np.median(before_px)), "tie_points": tie_point_summary},
            )

        # --- warp ---
        gcps = _build_gcps(fit_points)
        output_width, output_height = output_profile["width"], output_profile["height"]
        engine, estimated_gb = _select_warp_engine(local_cfg, ms_meta, output_width, output_height)
        if local_cfg.warp_engine != "auto" and engine == "deshifter" and estimated_gb > local_cfg.max_in_memory_warp_gb:
            raise LocalRefinementRejected(
                "MEMORY_BUDGET",
                f"deshifter engine was pinned but the estimated warp memory ({estimated_gb:.1f} GB) "
                f"exceeds max_in_memory_warp_gb ({local_cfg.max_in_memory_warp_gb}).",
                {"estimated_gb": estimated_gb},
            )

        target_full_alias = create_whitespace_safe_alias(ms_meta.path, temp_path, "target_full")
        safe_output = temp_path / "aligned.tif"
        try:
            if engine == "deshifter":
                coreg_info = dict(coreg_local.coreg_info)
                _warp_with_deshifter(
                    target_full_alias, coreg_info, gcps, output_profile, local_cfg.resampling,
                    cfg.warp.nodata_value, local_cfg.cpus, safe_output,
                )
            else:
                _warp_with_gdal_tps(
                    target_full_alias, gcps, rgb_meta.crs.to_wkt(), output_profile, local_cfg.resampling,
                    cfg.warp.nodata_value, local_cfg.gdal_warp_memory_mb, safe_output,
                )
        except ArosicsExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ArosicsExecutionError(f"AROSICS local warp ({engine}) execution error: {exc}") from exc

        if not safe_output.exists():
            raise ArosicsExecutionError(f"AROSICS local warp ({engine}) did not publish an output GeoTIFF.")
        with rasterio.open(safe_output) as written:
            grid_ok = (
                written.width == output_width and written.height == output_height
                and _affine_close(written.transform, output_profile["transform"])
            )
        if not grid_ok:
            raise ArosicsExecutionError(
                f"AROSICS local warp ({engine}) output grid does not match the pipeline output grid."
            )

        # --- holdout verification (independent of the GCPs used to fit the warp) ---
        aligned_matching_band_path = temp_path / "aligned_matching_band.tif"
        with rasterio.open(safe_output) as aligned_ds:
            band_values = aligned_ds.read(tgt_band).astype(np.float32)
            band_mask = aligned_ds.read_masks(tgt_band) > 0
            band_values[~band_mask] = STAGE_NODATA
            profile = {
                "driver": "GTiff", "dtype": "float32", "nodata": STAGE_NODATA,
                "width": aligned_ds.width, "height": aligned_ds.height, "count": 1,
                "crs": aligned_ds.crs, "transform": aligned_ds.transform,
            }
            with rasterio.open(aligned_matching_band_path, "w", **profile) as dst:
                dst.write(band_values, 1)

        after_px = _measure_holdout_shifts(
            staged_ref_path, aligned_matching_band_path, holdout_points,
            local_cfg.window_size, max_shift_ref_px, ms_meta.gsd,
        )
        holdout_stats = _run_holdout_gates(before_px, after_px, local_cfg)

        verification_info = None
        if local_cfg.full_grid_verification:
            try:
                verify_coreg = COREG_LOCAL(
                    im_ref=str(staged_ref_path), im_tgt=str(aligned_matching_band_path),
                    grid_res=grid_res_px, window_size=local_cfg.window_size,
                    r_b4match=1, s_b4match=1, max_iter=local_cfg.max_iter, max_shift=max_shift_ref_px,
                    tieP_filter_level=local_cfg.tie_point_filter_level, tieP_random_state=0,
                    min_reliability=local_cfg.min_reliability, rs_random_state=0,
                    align_grids=False, resamp_alg_calc="cubic",
                    outFillVal=int(_AROSICS_OUTFILL_VAL), nodata=(STAGE_NODATA, STAGE_NODATA),
                    projectDir=str(temp_path), CPUs=local_cfg.cpus, progress=False, q=True, ignore_errors=True,
                )
                verify_coreg.calculate_spatial_shifts()
                verify_table = verify_coreg.CoRegPoints_table
                verify_points = _extract_tie_points(verify_table, Affine.identity(), ms_meta.gsd) if verify_table is not None else []
                verify_valid = [p for p in verify_points if not p.is_outlier]
                if len(verify_valid) < max(1, local_cfg.min_valid_tie_points // 2):
                    raise LocalRefinementRejected(
                        "VERIFICATION_FAILED",
                        f"Full-grid post-warp verification found only {len(verify_valid)} valid points; "
                        f"{local_cfg.min_valid_tie_points // 2} required.",
                        {"valid": len(verify_valid)},
                    )
                p90 = float(np.percentile([p.shift_px for p in verify_valid], 90))
                verification_info = {"valid": len(verify_valid), "p90_px": p90}
                if p90 > local_cfg.max_verification_p90_px:
                    raise LocalRefinementRejected(
                        "VERIFICATION_FAILED",
                        f"Full-grid post-warp verification p90 residual ({p90:.2f}px) exceeds the limit "
                        f"({local_cfg.max_verification_p90_px}px).",
                        {"verification": verification_info},
                    )
            except LocalRefinementRejected:
                raise
            except Exception as exc:  # noqa: BLE001 - verification itself failing is not fatal to the primary result
                active_log.warning("AROSICS full-grid verification could not be completed: %s", exc)
                verification_info = {"error": str(exc)}

        # --- footprint ---
        try:
            footprint = evaluate_written_footprint(rgb_meta, ms_meta, safe_output, output_profile, cfg.warp.tile_size)
        except ValueError as exc:
            raise LocalRefinementRejected("FOOTPRINT_FAILED", str(exc))
        footprint_failures = footprint_gate_failures(footprint, cfg.transform)
        if footprint_failures:
            raise LocalRefinementRejected(
                "FOOTPRINT_FAILED", "Written footprint rejected: " + "; ".join(footprint_failures),
                {"footprint": footprint.__dict__},
            )

        # --- publish ---
        stem = ms_meta.path.stem
        final_path = output_dir / f"{stem}_aligned.tif"
        staged_partial = output_dir / f".{stem}_aligned.arosics_local.partial.tif"
        shutil.copy2(safe_output, staged_partial)
        os.replace(staged_partial, final_path)

        tie_points_geojson_path = None
        if local_cfg.export_tie_points:
            try:
                tie_points_geojson_path = output_dir / f"{stem}_arosics_tiepoints.geojson"
                _export_tie_point_table(table, fit_points, holdout_points, tie_points_geojson_path)
            except Exception as exc:  # noqa: BLE001 - export failure must not invalidate an accepted result
                active_log.warning("Could not export AROSICS tie points: %s", exc)
                tie_points_geojson_path = None

        payload = {
            "engine": engine,
            "estimated_memory_gb": estimated_gb,
            "band_pair": {"name": pair_name, "reference_band": ref_band, "target_band": tgt_band},
            "global_precorrection": {
                "translation_only": correction.is_translation_only, "resampled_matching_band": staged_tgt.resampled,
            },
            "parameters": {
                "grid_res_px": grid_res_px, "window_size": list(local_cfg.window_size),
                "max_shift_ref_px": max_shift_ref_px, "max_shift": max_shift_info,
                "min_reliability": local_cfg.min_reliability, "tie_point_filter_level": local_cfg.tie_point_filter_level,
            },
            "tie_points": tie_point_summary,
            "coverage": coverage_info,
            "holdout": holdout_stats,
            "full_grid_verification": verification_info,
            "warp": {"resampling": local_cfg.resampling},
            "tie_points_geojson": str(tie_points_geojson_path) if tie_points_geojson_path else None,
        }
        active_log.info(
            "AROSICS local refinement accepted with %s: fit=%d holdout=%d median_before=%.2fpx median_after=%.2fpx.",
            pair_name, len(fit_points), len(holdout_points),
            holdout_stats["median_before_px"], holdout_stats["median_after_px"],
        )
        return ArosicsLocalPublication(
            final_path, footprint, pair_name, ref_band, tgt_band, payload, tie_points_geojson_path,
        )


def _affine_close(a: Affine, b: Affine, rel_tol: float = 1e-6) -> bool:
    return all(math.isclose(x, y, rel_tol=rel_tol, abs_tol=1e-9) for x, y in zip(a, b))


def _export_tie_point_table(table, fit_points: list[TiePoint], holdout_points: list[TiePoint], out_path: Path) -> None:
    fit_ids = {p.point_id for p in fit_points}
    holdout_ids = {p.point_id for p in holdout_points}
    exportable = table.copy()
    exportable["role"] = exportable["POINT_ID"].apply(
        lambda pid: "fit" if pid in fit_ids else ("holdout" if pid in holdout_ids else "rejected")
    )
    exportable.to_file(str(out_path), driver="GeoJSON")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_arosics_local_refinement(
    rgb_meta: RasterMetadata,
    ms_meta: RasterMetadata,
    common_mask: np.ndarray,
    registration_transform: Affine,
    registration_gsd: float,
    global_transform: TransformResult,
    global_quality: SpatialResidualReport,
    output_profile: dict,
    cfg: AlignmentConfig,
    output_dir: Path,
    active_log: logging.Logger | None = None,
) -> ArosicsLocalPublication:
    """Refine the verified global result with AROSICS COREG_LOCAL, or reject.

    Raises :class:`LocalRefinementRejected` for every expected safety-gate
    failure (the caller should fall back to publishing the global result).
    Any other exception is an operational/programming error and propagates.
    """
    active_log = active_log or logger
    if not is_arosics_available():
        raise LocalRefinementRejected("AROSICS_UNAVAILABLE", "The optional arosics package is not installed.")
    if rgb_meta.crs != ms_meta.crs:
        raise LocalRefinementRejected(
            "UNSUPPORTED_CRS", "AROSICS local refinement requires the RGB and MS rasters to share a CRS.",
            {"rgb_crs": str(rgb_meta.crs), "ms_crs": str(ms_meta.crs)},
        )
    if global_transform.matrix.shape != (2, 3):
        raise LocalRefinementRejected(
            "UNSUPPORTED_GLOBAL_TRANSFORM", "AROSICS local refinement currently requires an affine 2x3 global transform.",
            {"matrix_shape": list(global_transform.matrix.shape)},
        )

    band_pairs = _resolve_band_pairs(cfg)
    attempts: list[dict] = []
    last_rejection: LocalRefinementRejected | None = None
    for pair_name, ref_band, tgt_band in band_pairs:
        try:
            publication = _attempt_band_pair(
                rgb_meta, ms_meta, common_mask, registration_transform, registration_gsd,
                global_transform, global_quality, output_profile, pair_name, ref_band, tgt_band,
                cfg, output_dir, active_log,
            )
            publication.payload["band_pair_attempts"] = attempts + [{"name": pair_name, "result": "accepted"}]
            return publication
        except LocalRefinementRejected as rejection:
            attempts.append({"name": pair_name, "result": rejection.reason_code, "message": str(rejection)})
            last_rejection = rejection
            active_log.warning("AROSICS local refinement rejected for band pair %s (%s): %s",
                               pair_name, rejection.reason_code, rejection)
            continue

    assert last_rejection is not None  # band_pairs is never empty
    last_rejection.details["band_pair_attempts"] = attempts
    raise last_rejection
