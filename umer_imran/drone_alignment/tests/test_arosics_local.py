import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import Point

from drone_alignment.alignment.arosics_local import (
    TiePoint,
    _affine_close,
    _coverage_metrics,
    _compute_safe_cpus,
    _estimate_warp_memory_gb,
    _extract_tie_points,
    _neighbour_filter,
    _resolve_grid_res_px,
    _resolve_max_shift_ref_px,
    _run_holdout_gates,
    _select_warp_engine,
    _split_fit_holdout,
)
from drone_alignment.alignment.rejections import LocalRefinementRejected
from drone_alignment.config.schema import ArosicsLocalConfig
from drone_alignment.io.reader import RasterMetadata
from drone_alignment.quality.metrics import SpatialResidualReport


def _tie_point(point_id, map_xy, shift_m, reliability=90.0, flags=None) -> TiePoint:
    ms_gsd = 0.05
    return TiePoint(
        point_id=point_id, staged_px=map_xy, source_px=map_xy, map_xy=map_xy, shift_m=shift_m,
        shift_px=math.hypot(*shift_m) / ms_gsd, reliability=reliability,
        ssim_before=0.5, ssim_after=0.7, ssim_improved=True, flags=flags or {},
    )


# ---------------------------------------------------------------------------
# _extract_tie_points
# ---------------------------------------------------------------------------

def _fake_table(rows: list[dict]) -> gpd.GeoDataFrame:
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(df, geometry=[Point(0, 0)] * len(rows))


def test_extract_tie_points_drops_no_match_rows():
    table = _fake_table([
        {"POINT_ID": 1, "X_IM": 1.0, "Y_IM": 1.0, "X_MAP": 10.0, "Y_MAP": 10.0,
         "X_SHIFT_M": 0.1, "Y_SHIFT_M": 0.0, "ABS_SHIFT": 0.1, "RELIABILITY": 80.0,
         "SSIM_BEFORE": 0.5, "SSIM_AFTER": 0.7, "SSIM_IMPROVED": True},
        {"POINT_ID": 2, "X_IM": 2.0, "Y_IM": 2.0, "X_MAP": 20.0, "Y_MAP": 20.0,
         "X_SHIFT_M": -9999.0, "Y_SHIFT_M": -9999.0, "ABS_SHIFT": -9999.0, "RELIABILITY": np.nan,
         "SSIM_BEFORE": np.nan, "SSIM_AFTER": np.nan, "SSIM_IMPROVED": None},
    ])
    points = _extract_tie_points(table, Affine.identity(), ms_gsd=0.05)
    assert len(points) == 1
    assert points[0].point_id == 1
    assert points[0].shift_px == pytest.approx(0.1 / 0.05)


def test_extract_tie_points_maps_staged_to_source_px():
    table = _fake_table([
        {"POINT_ID": 1, "X_IM": 10.0, "Y_IM": 20.0, "X_MAP": 5.0, "Y_MAP": 6.0,
         "X_SHIFT_M": 0.0, "Y_SHIFT_M": 0.0, "ABS_SHIFT": 0.0, "RELIABILITY": 95.0,
         "SSIM_BEFORE": 0.5, "SSIM_AFTER": 0.9, "SSIM_IMPROVED": True},
    ])
    # A staged->source mapping that doubles x and adds 3 to y.
    staged_to_source = Affine(2.0, 0.0, 0.0, 0.0, 1.0, 3.0)
    points = _extract_tie_points(table, staged_to_source, ms_gsd=0.05)
    assert points[0].source_px == pytest.approx((20.0, 23.0))
    assert points[0].flags == {}


def test_extract_tie_points_reads_outlier_flags():
    table = _fake_table([
        {"POINT_ID": 1, "X_IM": 1.0, "Y_IM": 1.0, "X_MAP": 10.0, "Y_MAP": 10.0,
         "X_SHIFT_M": 0.1, "Y_SHIFT_M": 0.0, "ABS_SHIFT": 0.1, "RELIABILITY": 80.0,
         "SSIM_BEFORE": 0.5, "SSIM_AFTER": 0.7, "SSIM_IMPROVED": True,
         "OUTLIER": True, "L1_OUTLIER": False, "L2_OUTLIER": False, "L3_OUTLIER": True},
    ])
    points = _extract_tie_points(table, Affine.identity(), ms_gsd=0.05)
    assert points[0].is_outlier
    assert points[0].flags["L3_OUTLIER"] is True


# ---------------------------------------------------------------------------
# _neighbour_filter
# ---------------------------------------------------------------------------

def test_neighbour_filter_flags_locally_inconsistent_vector():
    ms_gsd = 0.05
    cfg = ArosicsLocalConfig(neighbour_k=3, max_neighbour_deviation_px=1.0)
    # Five agreeing points plus one aliased outlier shifted by a whole "row period".
    points = [_tie_point(i, (float(i) * 10.0, 0.0), (0.01, 0.0)) for i in range(5)]
    points.append(_tie_point(99, (25.0, 0.0), (0.75, 0.0)))  # ~15px deviation at ms_gsd=0.05

    filtered = _neighbour_filter(points, cfg, ms_gsd)
    flagged = [p for p in filtered if p.point_id == 99][0]
    assert flagged.flags.get("L4_OUTLIER") is True
    for p in filtered:
        if p.point_id != 99:
            assert not p.flags.get("L4_OUTLIER")


def test_neighbour_filter_does_not_flag_consistent_points():
    ms_gsd = 0.05
    cfg = ArosicsLocalConfig(neighbour_k=3, max_neighbour_deviation_px=2.0)
    points = [_tie_point(i, (float(i) * 10.0, 0.0), (0.01 * i, 0.0)) for i in range(6)]
    filtered = _neighbour_filter(points, cfg, ms_gsd)
    assert all(not p.flags.get("L4_OUTLIER") for p in filtered)


def test_neighbour_filter_skips_outliers_already_flagged():
    ms_gsd = 0.05
    cfg = ArosicsLocalConfig(neighbour_k=3, max_neighbour_deviation_px=1.0)
    points = [_tie_point(i, (float(i) * 10.0, 0.0), (0.01, 0.0)) for i in range(5)]
    aliased = _tie_point(99, (25.0, 0.0), (0.75, 0.0), flags={"OUTLIER": True})
    points.append(aliased)
    filtered = _neighbour_filter(points, cfg, ms_gsd)
    # already-outlier points are excluded from the neighbour computation and are not re-flagged with L4
    aliased_out = [p for p in filtered if p.point_id == 99][0]
    assert "L4_OUTLIER" not in aliased_out.flags


def test_neighbour_filter_noop_when_disabled():
    ms_gsd = 0.05
    cfg = ArosicsLocalConfig(neighbour_filter=False)
    points = [_tie_point(i, (float(i) * 10.0, 0.0), (0.01, 0.0)) for i in range(5)]
    points.append(_tie_point(99, (25.0, 0.0), (5.0, 0.0)))
    filtered = _neighbour_filter(points, cfg, ms_gsd)
    assert filtered == points


# ---------------------------------------------------------------------------
# coverage and holdout split
# ---------------------------------------------------------------------------

def _grid_points(registration_transform: Affine, n_per_axis: int) -> list[TiePoint]:
    points = []
    pid = 0
    for row in range(n_per_axis):
        for col in range(n_per_axis):
            reg_x = (col + 0.5) * (100.0 / n_per_axis)
            reg_y = (row + 0.5) * (100.0 / n_per_axis)
            map_xy = registration_transform * (reg_x, reg_y)
            points.append(_tie_point(pid, map_xy, (0.01, 0.0)))
            pid += 1
    return points


def test_coverage_metrics_full_grid_is_fully_covered():
    registration_transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    common_mask = np.ones((100, 100), dtype=bool)
    cfg = ArosicsLocalConfig(coverage_grid=4)
    points = _grid_points(registration_transform, 4)  # one point per cell, well spread
    occupancy, hull_coverage = _coverage_metrics(points, common_mask, registration_transform, cfg)
    assert occupancy == pytest.approx(1.0)
    assert hull_coverage > 0.4  # points span from near 0 to near 100 in both axes


def test_coverage_metrics_clustered_points_have_low_occupancy():
    registration_transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    common_mask = np.ones((100, 100), dtype=bool)
    cfg = ArosicsLocalConfig(coverage_grid=4)
    # All points clustered in one corner cell.
    points = [_tie_point(i, (5.0 + i * 0.1, 5.0), (0.01, 0.0)) for i in range(10)]
    occupancy, hull_coverage = _coverage_metrics(points, common_mask, registration_transform, cfg)
    assert occupancy < 0.2
    assert hull_coverage < 0.05


def test_split_fit_holdout_respects_minimums_and_never_empties_a_singleton_cell():
    registration_transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    common_mask = np.ones((100, 100), dtype=bool)
    cfg = ArosicsLocalConfig(coverage_grid=4, holdout_fraction=0.2, min_holdout_points=3)
    # Two points per cell across a 4x4 grid = 32 points; each cell has exactly 2, so at most
    # one point per cell can be pulled into holdout.
    points = []
    pid = 0
    for row in range(4):
        for col in range(4):
            for _ in range(2):
                reg_x = (col + 0.5) * 25.0
                reg_y = (row + 0.5) * 25.0
                points.append(_tie_point(pid, (reg_x, reg_y), (0.01, 0.0)))
                pid += 1

    fit, holdout = _split_fit_holdout(points, common_mask, registration_transform, cfg)
    assert len(fit) + len(holdout) == len(points)
    assert len(holdout) >= cfg.min_holdout_points
    # No cell should have lost both of its points to holdout.
    fit_ids = {p.point_id for p in fit}
    for row in range(4):
        for col in range(4):
            cell_ids = {2 * (row * 4 + col), 2 * (row * 4 + col) + 1}
            assert cell_ids & fit_ids, "a cell was completely emptied by the holdout split"


def test_split_fit_holdout_skips_singleton_cells():
    registration_transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    common_mask = np.ones((100, 100), dtype=bool)
    cfg = ArosicsLocalConfig(coverage_grid=4, holdout_fraction=0.4, min_holdout_points=3)
    # Every cell has exactly one point: none may be moved to holdout without
    # emptying its cell, so the holdout target cannot be reached from here.
    points = [_tie_point(i, (12.5 + (i % 4) * 25.0, 12.5 + (i // 4) * 25.0), (0.01, 0.0)) for i in range(16)]
    fit, holdout = _split_fit_holdout(points, common_mask, registration_transform, cfg)
    assert len(holdout) == 0
    assert len(fit) == len(points)


# ---------------------------------------------------------------------------
# holdout gates
# ---------------------------------------------------------------------------

def test_holdout_gates_accept_clear_improvement():
    cfg = ArosicsLocalConfig(
        min_holdout_points=3, no_gain_floor_px=0.5, max_holdout_ratio=0.7,
        min_holdout_win_fraction=0.6, max_holdout_after_px=2.0,
    )
    before = [2.0, 2.2, 1.8, 2.1]
    after = [0.3, 0.4, 0.2, 0.35]
    stats = _run_holdout_gates(before, after, cfg)
    assert stats["median_before_px"] == pytest.approx(2.05)
    assert stats["win_fraction"] == 1.0


def test_holdout_gates_reject_insufficient_win_fraction():
    cfg = ArosicsLocalConfig(
        min_holdout_points=3, no_gain_floor_px=0.5, max_holdout_ratio=0.99,
        min_holdout_win_fraction=0.9, max_holdout_after_px=5.0,
    )
    before = [2.0, 2.0, 2.0, 2.0]
    after = [2.5, 1.0, 2.6, 1.0]  # only half improved
    with pytest.raises(LocalRefinementRejected) as excinfo:
        _run_holdout_gates(before, after, cfg)
    assert excinfo.value.reason_code == "HOLDOUT_FAILED"


def test_holdout_gates_reject_too_few_verifiable_points():
    cfg = ArosicsLocalConfig(min_holdout_points=3)
    before = [2.0, 2.0, 2.0, 2.0]
    after = [None, None, 0.3, None]  # only one verifiable
    with pytest.raises(LocalRefinementRejected) as excinfo:
        _run_holdout_gates(before, after, cfg)
    assert excinfo.value.reason_code == "HOLDOUT_FAILED"


def test_holdout_gates_reject_worst_case_residual():
    cfg = ArosicsLocalConfig(
        min_holdout_points=3, no_gain_floor_px=0.5, max_holdout_ratio=0.9,
        min_holdout_win_fraction=0.5, max_holdout_after_px=1.0,
    )
    before = [2.0, 2.0, 2.0, 2.0]
    after = [0.1, 0.1, 0.1, 3.0]  # one bad outlier after refinement
    with pytest.raises(LocalRefinementRejected) as excinfo:
        _run_holdout_gates(before, after, cfg)
    assert excinfo.value.reason_code == "HOLDOUT_FAILED"


# ---------------------------------------------------------------------------
# max_shift / grid_res resolution
# ---------------------------------------------------------------------------

def test_resolve_max_shift_uses_configured_value():
    cfg = ArosicsLocalConfig(max_shift_m=0.2)
    quality = SpatialResidualReport(1.0, None, None, None, None, [], "PASS", True)
    ref_px, info = _resolve_max_shift_ref_px(cfg, quality, registration_gsd=0.06, rgb_gsd=0.02)
    assert info["source"] == "configured"
    assert ref_px == math.ceil(0.2 / 0.02)


def test_resolve_max_shift_auto_from_global_rmse_is_clamped():
    cfg = ArosicsLocalConfig(max_shift_m=None)
    # A huge RMSE should clamp to the 0.30m ceiling, not scale unbounded.
    quality = SpatialResidualReport(1000.0, None, None, None, None, [], "PASS", True)
    ref_px, info = _resolve_max_shift_ref_px(cfg, quality, registration_gsd=0.06, rgb_gsd=0.02)
    assert info["source"] == "auto_from_global_rmse"
    assert info["max_shift_m"] == pytest.approx(0.30)


def test_resolve_max_shift_default_when_no_quality_available():
    cfg = ArosicsLocalConfig(max_shift_m=None)
    quality = SpatialResidualReport(None, None, None, None, None, [], "PASS", True)
    ref_px, info = _resolve_max_shift_ref_px(cfg, quality, registration_gsd=0.06, rgb_gsd=0.02)
    assert info["source"] == "auto_default"
    assert info["max_shift_m"] == pytest.approx(0.25)


def test_resolve_max_shift_periodic_guard_caps_search_radius():
    cfg = ArosicsLocalConfig(max_shift_m=0.5, periodic_texture_period_m=0.75)
    quality = SpatialResidualReport(1.0, None, None, None, None, [], "PASS", True)
    ref_px, info = _resolve_max_shift_ref_px(cfg, quality, registration_gsd=0.06, rgb_gsd=0.02)
    assert info["max_shift_m"] == pytest.approx(0.4 * 0.75)
    assert "periodic_guard" in info["source"]


def test_resolve_grid_res_px_explicit_override():
    cfg = ArosicsLocalConfig(grid_res_px=99)
    assert _resolve_grid_res_px(cfg, 2000, 1500) == 99


def test_resolve_grid_res_px_derived_from_target_points_per_axis():
    cfg = ArosicsLocalConfig(grid_res_px=None, target_points_per_axis=16)
    assert _resolve_grid_res_px(cfg, 1600, 800) == math.ceil(1600 / 16)


# ---------------------------------------------------------------------------
# warp engine selection
# ---------------------------------------------------------------------------

def _fake_ms_meta(width, height, band_count=4, dtype="float32") -> RasterMetadata:
    return RasterMetadata(
        path=Path("fake.tif"), crs=None, transform=Affine.identity(), width=width, height=height,
        band_count=band_count, dtype=dtype, nodata=None, bounds=(0, 0, 1, 1), gsd=0.05,
        band_descriptions=(None,) * band_count, color_interpretations=("undefined",) * band_count,
        alpha_band_index=None, spectral_band_indices=tuple(range(1, band_count + 1)),
    )


def test_select_warp_engine_auto_picks_deshifter_for_small_raster():
    cfg = ArosicsLocalConfig(warp_engine="auto", max_in_memory_warp_gb=4.0)
    ms_meta = _fake_ms_meta(1000, 1000)
    engine, estimate_gb = _select_warp_engine(cfg, ms_meta, 1000, 1000)
    assert engine == "deshifter"
    assert estimate_gb < 4.0


def test_select_warp_engine_auto_picks_gdal_tps_for_huge_raster():
    cfg = ArosicsLocalConfig(warp_engine="auto", max_in_memory_warp_gb=1.0)
    ms_meta = _fake_ms_meta(30000, 30000, band_count=5, dtype="float32")
    engine, estimate_gb = _select_warp_engine(cfg, ms_meta, 30000, 30000)
    assert engine == "gdal_tps"
    assert estimate_gb > 1.0


def test_select_warp_engine_pinned_choice_is_honoured():
    cfg = ArosicsLocalConfig(warp_engine="gdal_tps")
    ms_meta = _fake_ms_meta(100, 100)
    engine, _ = _select_warp_engine(cfg, ms_meta, 100, 100)
    assert engine == "gdal_tps"


def test_estimate_warp_memory_gb_scales_with_size_and_band_count():
    small = _fake_ms_meta(100, 100, band_count=1)
    large = _fake_ms_meta(1000, 1000, band_count=5)
    assert _estimate_warp_memory_gb(large, 1000, 1000) > _estimate_warp_memory_gb(small, 100, 100)


def test_safe_cpu_count_never_exceeds_configured_limit():
    assert 1 <= _compute_safe_cpus(100, 100, configured_cpus=2) <= 2


def test_safe_cpu_count_rejects_when_one_worker_cannot_fit(monkeypatch):
    import psutil

    class Memory:
        available = 600 * 1024**2

    monkeypatch.setattr(psutil, "virtual_memory", lambda: Memory())
    with pytest.raises(LocalRefinementRejected) as excinfo:
        _compute_safe_cpus(10_000, 10_000, configured_cpus=4)
    assert excinfo.value.reason_code == "MEMORY_BUDGET"


# ---------------------------------------------------------------------------
# affine closeness helper
# ---------------------------------------------------------------------------

def test_affine_close_true_for_identical_transforms():
    a = Affine(0.05, 0.0, 100.0, 0.0, -0.05, 200.0)
    assert _affine_close(a, a)


def test_affine_close_false_for_different_transforms():
    a = Affine(0.05, 0.0, 100.0, 0.0, -0.05, 200.0)
    b = Affine(0.05, 0.0, 100.5, 0.0, -0.05, 200.0)
    assert not _affine_close(a, b)
