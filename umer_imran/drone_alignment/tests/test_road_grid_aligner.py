import pytest
import numpy as np
from drone_alignment.config.schema import RoadGridConfig, TransformValidationConfig
from drone_alignment.alignment.road_grid_aligner import (
    compute_road_grid_translation,
    InsufficientRoadFeatureError,
)
from drone_alignment.alignment.transform_estimator import TransformUnreliableError


def _make_grid_synthetic(shape=(256, 256), dx=5, dy=3):
    """
    Generate synthetic RGB and MS arrays with grid features (bright roads, dark crop rows)
    shifted by (dx, dy).
    """
    h, w = shape
    rgb = np.full((h, w), 0.05, dtype=np.float32)
    
    # Add bright roads (vertical road at col 100, horizontal road at row 120)
    rgb[115:125, :] = 0.5  # horizontal road
    rgb[:, 95:105] = 0.5   # vertical road
    
    # Add dark tree rows
    for r in range(20, h, 30):
        if not (115 <= r <= 125):
            rgb[r:r+5, :] = 0.01

    ms = np.full((h, w), 0.05, dtype=np.float32)
    # Apply shift (dx, dy): MS feature location = RGB feature location - shift
    # so MS(x + dx, y + dy) = RGB(x, y) => tx_px = dx, ty_px = dy
    # Create shifted features in MS
    r_start, r_end = 115 - dy, 125 - dy
    c_start, c_end = 95 - dx, 105 - dx
    
    if 0 <= r_start < h and 0 <= r_end <= h:
        ms[r_start:r_end, :] = 0.5
    if 0 <= c_start < w and 0 <= c_end <= w:
        ms[:, c_start:c_end] = 0.5

    for r in range(20, h, 30):
        r_shifted = r - dy
        if 0 <= r_shifted < h - 5 and not (r_start <= r_shifted <= r_end):
            ms[r_shifted:r_shifted+5, :] = 0.01

    valid_mask = np.ones((h, w), dtype=bool)
    return rgb, ms, valid_mask, valid_mask


def test_known_shift_recovery():
    dx_expected, dy_expected = 4, 3
    rgb, ms, mask_rgb, mask_ms = _make_grid_synthetic(shape=(256, 256), dx=dx_expected, dy=dy_expected)
    
    road_cfg = RoadGridConfig(blur_kernel_size=5, num_strips=8, guardrail_px=15)
    trans_cfg = TransformValidationConfig(max_translation_m=50.0)
    
    res = compute_road_grid_translation(
        rgb_band=rgb,
        ms_band=ms,
        rgb_mask=mask_rgb,
        ms_mask=mask_ms,
        target_gsd=0.05,
        road_grid_config=road_cfg,
        transform_config=trans_cfg,
    )
    
    assert res.method == "road_grid"
    assert pytest.approx(res.translation_px[0], abs=1.0) == dx_expected
    assert pytest.approx(res.translation_px[1], abs=1.0) == dy_expected


def test_no_road_features_raises():
    h, w = 128, 128
    rgb = np.full((h, w), 0.05, dtype=np.float32)
    ms = np.full((h, w), 0.05, dtype=np.float32)
    mask = np.ones((h, w), dtype=bool)
    
    road_cfg = RoadGridConfig(min_road_pixel_fraction=0.05)
    trans_cfg = TransformValidationConfig()
    
    with pytest.raises(InsufficientRoadFeatureError):
        compute_road_grid_translation(
            rgb_band=rgb,
            ms_band=ms,
            rgb_mask=mask,
            ms_mask=mask,
            target_gsd=0.05,
            road_grid_config=road_cfg,
            transform_config=trans_cfg,
        )


def test_strip_median_robustness():
    dx_expected, dy_expected = 5, 2
    rgb, ms, mask_rgb, mask_ms = _make_grid_synthetic(shape=(256, 256), dx=dx_expected, dy=dy_expected)
    
    # Introduce localized false feature corrupting 1 strip out of 10
    rgb[0:25, 150:200] = 0.5
    
    road_cfg = RoadGridConfig(blur_kernel_size=5, num_strips=10, guardrail_px=15)
    trans_cfg = TransformValidationConfig(max_translation_m=50.0)
    
    res = compute_road_grid_translation(
        rgb_band=rgb,
        ms_band=ms,
        rgb_mask=mask_rgb,
        ms_mask=mask_ms,
        target_gsd=0.05,
        road_grid_config=road_cfg,
        transform_config=trans_cfg,
    )
    
    assert pytest.approx(res.translation_px[0], abs=1.5) == dx_expected
    assert pytest.approx(res.translation_px[1], abs=1.5) == dy_expected


def test_dark_refinement_confirmation():
    dx_expected, dy_expected = 3, -2
    rgb, ms, mask_rgb, mask_ms = _make_grid_synthetic(shape=(256, 256), dx=dx_expected, dy=dy_expected)
    
    road_cfg = RoadGridConfig(blur_kernel_size=5, num_strips=8, use_dark_refinement=True)
    trans_cfg = TransformValidationConfig(max_translation_m=50.0)
    
    res = compute_road_grid_translation(
        rgb_band=rgb,
        ms_band=ms,
        rgb_mask=mask_rgb,
        ms_mask=mask_ms,
        target_gsd=0.05,
        road_grid_config=road_cfg,
        transform_config=trans_cfg,
    )
    
    assert pytest.approx(res.translation_px[0], abs=1.0) == dx_expected
    assert pytest.approx(res.translation_px[1], abs=1.0) == dy_expected


def test_guardrail_prevents_aliasing():
    # Test that guardrail prevents locking onto a secondary periodic peak far away
    dx_expected, dy_expected = 2, 2
    rgb, ms, mask_rgb, mask_ms = _make_grid_synthetic(shape=(256, 256), dx=dx_expected, dy=dy_expected)
    
    # Tight guardrail prevents false aliasing onto adjacent rows
    road_cfg = RoadGridConfig(blur_kernel_size=5, num_strips=8, guardrail_px=10)
    trans_cfg = TransformValidationConfig(max_translation_m=50.0)
    
    res = compute_road_grid_translation(
        rgb_band=rgb,
        ms_band=ms,
        rgb_mask=mask_rgb,
        ms_mask=mask_ms,
        target_gsd=0.05,
        road_grid_config=road_cfg,
        transform_config=trans_cfg,
    )
    
    assert abs(res.translation_px[0] - dx_expected) <= 1.0
    assert abs(res.translation_px[1] - dy_expected) <= 1.0
