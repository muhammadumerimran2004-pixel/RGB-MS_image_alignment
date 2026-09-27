import numpy as np
import pytest

from drone_alignment.alignment.local_mesh_aligner import (
    CellMatchStatus,
    build_structural_feature_map,
    compute_sparse_local_displacements,
)
from drone_alignment.config.schema import LocalMeshConfig, RoadGridConfig


def _shifted_structures(shape=(180, 180), dx=4, dy=3):
    """RGB roads/trees and an MS image shifted so MS->RGB is (dx, dy)."""
    h, w = shape
    rgb = np.full(shape, 0.20, dtype=np.float32)
    rgb[:, 36:43] = 0.85
    rgb[:, 127:134] = 0.72
    rgb[86:93, :] = 0.85
    for row in range(18, h, 24):
        rgb[row:row + 3, :] = 0.04
    ms = np.full(shape, 0.20, dtype=np.float32)
    # Pull each reference feature backwards by the intended MS->RGB displacement.
    yy, xx = np.indices(shape)
    src_x, src_y = xx + dx, yy + dy
    inside = (src_x < w) & (src_y < h)
    ms[inside] = rgb[src_y[inside], src_x[inside]]
    mask = np.ones(shape, dtype=bool)
    return rgb, ms, mask, mask


def _configs(**overrides):
    road = RoadGridConfig(blur_kernel_size=3, bright_percentile=90, dark_percentile=10)
    defaults = dict(
        grid_rows=3, grid_cols=3, cell_halo_px=8, max_search_radius_px=10,
        min_road_points=8, min_tree_points=8, min_match_confidence=0.25,
        ambiguity_margin=0.03, max_neighbor_difference_px=12,
    )
    defaults.update(overrides)
    local = LocalMeshConfig(**defaults)
    return road, local


def test_structural_map_is_global_across_cell_boundary():
    rgb, _, rgb_mask, _ = _shifted_structures()
    road, _ = _configs()
    features = build_structural_feature_map(rgb, rgb_mask, road)
    # The horizontal road crosses every future local-cell boundary; its skeleton
    # belongs to a full-canvas structure rather than a cell-local reconstruction.
    road_row = max(range(rgb.shape[0]), key=lambda row: int(features.road_skeleton[row].sum()))
    labels = features.road_components[road_row, features.road_skeleton[road_row]]
    assert len(labels) > 0
    assert np.all(labels == labels[0])


def test_local_solver_recovers_zero_residual_after_global_prior():
    rgb, ms, rgb_mask, ms_mask = _shifted_structures(dx=4, dy=3)
    road, local = _configs()
    result = compute_sparse_local_displacements(rgb, ms, rgb_mask, ms_mask, (4, 3), road, local)
    assert result.trusted_count >= 3
    for sample in result.accepted:
        assert sample.displacement_dx_dy[0] == pytest.approx(4, abs=1)
        assert sample.displacement_dx_dy[1] == pytest.approx(3, abs=1)


def test_held_out_structure_scores_improve_after_local_residual():
    rgb, ms, rgb_mask, ms_mask = _shifted_structures(dx=4, dy=3)
    road, local = _configs()
    # Deliberately give the solver an imperfect global prior.  It must fit on
    # one subset of road pixels and improve the residual on untouched pixels.
    result = compute_sparse_local_displacements(rgb, ms, rgb_mask, ms_mask, (2, 2), road, local)
    summary = result.validation_summary
    assert summary["count"] > 0
    assert summary["improvement"] is not None and summary["improvement"] > 1.0


def test_nearest_road_guardrail_excludes_stronger_distant_road():
    rgb, ms, rgb_mask, ms_mask = _shifted_structures(dx=3, dy=0)
    # Make the far road stronger in RGB.  Its required residual is ~+90 px and
    # must be outside the local, global-prior-centred search region.
    rgb[:, 127:134] = 1.0
    road, local = _configs(max_search_radius_px=8)
    result = compute_sparse_local_displacements(rgb, ms, rgb_mask, ms_mask, (3, 0), road, local)
    accepted = result.accepted
    assert accepted
    assert all(abs(sample.residual_dx_dy[0]) < 8 for sample in accepted)
    assert all(sample.status != CellMatchStatus.SEARCH_BOUNDARY for sample in accepted)


def test_ambiguous_parallel_roads_are_not_forced_to_a_vector():
    h, w = 120, 120
    rgb = np.full((h, w), 0.2, dtype=np.float32)
    rgb[:, 45:50] = 0.9
    rgb[:, 65:70] = 0.9
    ms = rgb.copy()
    mask = np.ones_like(rgb, dtype=bool)
    road, local = _configs(grid_rows=2, grid_cols=2, max_search_radius_px=15,
                           ambiguity_margin=0.20, ambiguity_separation_px=8)
    result = compute_sparse_local_displacements(rgb, ms, mask, mask, (10, 0), road, local)
    assert any(sample.status == CellMatchStatus.AMBIGUOUS_ROADS for sample in result.samples)


def test_spatial_outlier_is_rejected():
    rgb, ms, rgb_mask, ms_mask = _shifted_structures(dx=0, dy=0)
    # Corrupt one cell's local source road evidence so only its local match drifts.
    ms[:60, :60] = np.roll(ms[:60, :60], 7, axis=1)
    road, local = _configs(max_neighbor_difference_px=3)
    result = compute_sparse_local_displacements(rgb, ms, rgb_mask, ms_mask, (0, 0), road, local)
    assert any(sample.status in {CellMatchStatus.SPATIAL_OUTLIER, CellMatchStatus.AMBIGUOUS_ROADS,
                                 CellMatchStatus.SEARCH_BOUNDARY, CellMatchStatus.ROAD_TREE_CONFLICT}
               for sample in result.samples)


def test_tree_only_fallback_recovers_a_guarded_local_vector():
    h, w = 100, 100
    rgb = np.full((h, w), 0.30, dtype=np.float32)
    # Uneven dark rows make the y displacement identifiable without a bright road.
    for row in (12, 29, 57, 82):
        rgb[row:row + 3, :] = 0.03
    for col in (19, 68):
        rgb[:, col:col + 3] = 0.03
    ms = np.full_like(rgb, 0.30)
    ms[:-3] = rgb[3:]
    mask = np.ones_like(rgb, dtype=bool)
    road, local = _configs(grid_rows=2, grid_cols=2, max_search_radius_px=6,
                           min_road_points=10000, min_tree_points=5,
                           tree_only_min_confidence=0.10, tree_ambiguity_margin=0.01)
    result = compute_sparse_local_displacements(rgb, ms, mask, mask, (0, 3), road, local)
    accepted = [sample for sample in result.accepted if sample.evidence == "tree"]
    assert accepted
    assert all(sample.displacement_dx_dy[1] == pytest.approx(3, abs=1) for sample in accepted)
