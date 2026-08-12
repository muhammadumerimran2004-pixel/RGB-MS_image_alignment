import json
from pathlib import Path
import pytest
import numpy as np
import rasterio

from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, TransformValidationConfig
from drone_alignment.alignment.manual import compute_manual_translation
from drone_alignment.alignment.transform_estimator import TransformUnreliableError
from drone_alignment.pipeline import manual_align_orthomosaics


def test_compute_manual_translation_single_point():
    config = TransformValidationConfig(max_translation_m=50.0)
    pts_rgb = [(100.0, 200.0)]
    pts_ms = [(90.0, 185.0)]
    target_gsd = 0.05

    res = compute_manual_translation(pts_rgb, pts_ms, target_gsd, config)
    assert res.method == "manual"
    assert res.translation_px == (10.0, 15.0)
    assert pytest.approx(res.translation_m[0]) == 0.5
    assert pytest.approx(res.translation_m[1]) == 0.75
    np.testing.assert_allclose(res.matrix, [[1.0, 0.0, 10.0], [0.0, 1.0, 15.0]])


def test_compute_manual_translation_multiple_points():
    config = TransformValidationConfig(max_translation_m=50.0)
    pts_rgb = [(100.0, 200.0), (300.0, 400.0)]
    pts_ms = [(90.0, 180.0), (290.0, 380.0)]
    target_gsd = 0.05

    res = compute_manual_translation(pts_rgb, pts_ms, target_gsd, config)
    assert res.translation_px == (10.0, 20.0)
    assert res.num_inliers == 2


def test_compute_manual_translation_exceeds_max_translation():
    config = TransformValidationConfig(max_translation_m=1.0)
    pts_rgb = [(1000.0, 1000.0)]
    pts_ms = [(0.0, 0.0)]
    target_gsd = 0.05

    with pytest.raises(TransformUnreliableError, match="exceeds maximum allowed threshold"):
        compute_manual_translation(pts_rgb, pts_ms, target_gsd, config)


def test_manual_align_orthomosaics_pipeline(synthetic_geo_tiff_pair, tmp_path: Path):
    output_dir = tmp_path / "manual_out"
    config = AlignmentConfig(alignment_mode=AlignmentMode.MANUAL)

    pts_rgb = [(10.0, 10.0), (50.0, 50.0)]
    pts_ms = [(12.0, 11.0), (52.0, 51.0)]

    result = manual_align_orthomosaics(
        rgb_path=synthetic_geo_tiff_pair["rgb_path"],
        ms_path=synthetic_geo_tiff_pair["ms_path"],
        output_dir=output_dir,
        pts_rgb=pts_rgb,
        pts_ms=pts_ms,
        config=config,
    )

    assert result.aligned_ms_path.exists()
    assert result.report_json_path.exists()
    assert result.preview_image_path.exists()

    with open(result.report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["alignment_mode"] == "manual"
    assert report["transform"]["method"] == "manual"

    with rasterio.open(result.aligned_ms_path) as dst:
        assert dst.count == 4
        assert dst.width > 0
        assert dst.height > 0
