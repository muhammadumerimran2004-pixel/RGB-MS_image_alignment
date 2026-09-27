import numpy as np

from drone_alignment.alignment.rail_frame_aligner import (
    RailOrientation, build_rail_constrained_field, build_rail_region_field, compute_rail_frame,
    match_structural_rails, StructuralRail,
)
from drone_alignment.config.schema import LocalMeshConfig, RoadGridConfig


def _rail_image(shape=(240, 300), dx=4, dy=-3):
    rgb = np.full(shape, 0.2, dtype=np.float32)
    rgb[48:56, :] = 0.95
    rgb[176:186, :] = 0.92
    rgb[:, 72:81] = 0.96
    rgb[:, 232:241] = 0.90
    rgb[115:119, :] = 0.03
    ms = np.full_like(rgb, 0.2)
    yy, xx = np.indices(shape)
    source_x, source_y = xx + dx, yy + dy
    inside = (source_x >= 0) & (source_x < shape[1]) & (source_y >= 0) & (source_y < shape[0])
    ms[inside] = rgb[source_y[inside], source_x[inside]]
    return rgb, ms, np.ones(shape, dtype=bool)


def test_rail_frame_builds_variable_road_bounded_regions():
    rgb, ms, mask = _rail_image(dx=4, dy=-3)
    config = LocalMeshConfig(rail_min_length_fraction=0.4, internal_rail_max_deviation_px=8,
                             exterior_rail_max_deviation_px=8, min_region_size_px=20)
    result = compute_rail_frame(rgb, ms, mask, mask,
                                RoadGridConfig(blur_kernel_size=3, bright_percentile=85), config)
    assert any(match.rgb_rail.orientation == RailOrientation.HORIZONTAL for match in result.bright_matches)
    assert any(match.rgb_rail.orientation == RailOrientation.VERTICAL for match in result.bright_matches)
    assert result.regions
    assert result.corrections
    assert all(abs(match.deviation_px) <= 8 for match in result.bright_matches)
    assert all(abs(correction.dx) <= 8 and abs(correction.dy) <= 8 for correction in result.corrections)
    field = build_rail_constrained_field(result, rgb.shape)
    values = field.evaluate(np.array([[20.0, 20.0], [150.0, 120.0], [280.0, 220.0]]))
    assert np.isfinite(values).all()
    assert np.max(np.abs(values)) <= 8


def test_region_field_is_zero_on_structural_boundaries():
    from drone_alignment.alignment.rail_frame_aligner import RailBoundedRegion, RailRegionCorrection
    region = RailBoundedRegion(10, 20, 100, 120, None, None, None, None)
    correction = RailRegionCorrection(region, 2.0, -1.0, 1.0, 2, 2, 0.2, True)
    field = build_rail_region_field((correction,), LocalMeshConfig(region_boundary_margin_px=8))
    values = field.evaluate(np.array([[10., 60.], [55., 20.], [55., 70.]]))
    assert np.allclose(values[:2], 0.0)
    assert np.allclose(values[2], [2.0, -1.0])


def test_exterior_guardrail_rejects_strong_far_road():
    exterior_ms = StructuralRail(0, RailOrientation.VERTICAL, 30, 0, 200, 8, 1.0, is_exterior=True)
    nearest_rgb = StructuralRail(0, RailOrientation.VERTICAL, 34, 0, 200, 8, 0.6, is_exterior=True)
    far_rgb = StructuralRail(1, RailOrientation.VERTICAL, 130, 0, 200, 12, 1.0)
    matches = match_structural_rails(
        (nearest_rgb, far_rgb), (exterior_ms,),
        LocalMeshConfig(exterior_rail_max_deviation_px=10, internal_rail_max_deviation_px=150),
    )
    assert len(matches) == 1
    assert matches[0].rgb_rail.rail_id == nearest_rgb.rail_id
    assert matches[0].deviation_px == 4


def test_one_to_one_matching_prevents_two_sources_using_same_road():
    rgb = (StructuralRail(0, RailOrientation.HORIZONTAL, 100, 0, 300, 8, 1.0),)
    ms = (
        StructuralRail(0, RailOrientation.HORIZONTAL, 97, 0, 300, 8, 1.0),
        StructuralRail(1, RailOrientation.HORIZONTAL, 104, 0, 300, 8, 1.0),
    )
    matches = match_structural_rails(rgb, ms, LocalMeshConfig(internal_rail_max_deviation_px=10))
    assert len(matches) == 1
