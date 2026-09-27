"""End-to-end test of pipeline._refine_with_arosics_local() - the glue this
phase adds inside pipeline.py (band-pair-to-preview-channel lookup, report
writing, AlignmentResult construction) - against the real AROSICS package.

_estimate_global_candidate() (ORB/SIFT feature matching) is exercised by its
own tests; this constructs the VerifiedGlobalContext directly from a real,
known translation so the test is not coupled to classical feature-detector
tuning on synthetic textures.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from drone_alignment.alignment.arosics_matcher import is_arosics_available
from drone_alignment.alignment.coarse import coarse_align
from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.config.schema import (
    AlignmentConfig, ArosicsConfig, RegistrationChannel, TransformType,
)
from drone_alignment.io.reader import read_metadata
from drone_alignment.io.validators import validate_inputs
from drone_alignment.pipeline import VerifiedGlobalContext, _refine_with_arosics_local
from drone_alignment.quality.metrics import SpatialResidualReport

pytestmark = pytest.mark.skipif(not is_arosics_available(), reason="arosics package is not installed")


def _write_textured_pair(tmp_path: Path, size: int, gsd: float, true_tx: float, true_ty: float, amplitude: float, wavelength: float):
    rng = np.random.default_rng(7)
    base = rng.uniform(0.0, 1.0, size=(size, size)).astype(np.float32)
    rgb_band = cv2.GaussianBlur(base, (0, 0), sigmaX=3.0) * 200.0
    rgb_band += rng.uniform(0.0, 10.0, size=(size, size)).astype(np.float32)

    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    dx = amplitude * np.sin(2 * np.pi * xx / wavelength)
    src_x = (xx + dx + true_tx).astype(np.float32)
    src_y = (yy + true_ty).astype(np.float32)
    ms_band = cv2.remap(rgb_band, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    crs = "EPSG:32643"
    transform = from_origin(500000.0, 3000000.0, gsd, gsd)
    rgb_path, ms_path = tmp_path / "rgb.tif", tmp_path / "ms.tif"
    profile = dict(driver="GTiff", dtype="float32", width=size, height=size, count=1,
                   crs=crs, transform=transform, nodata=None)
    with rasterio.open(rgb_path, "w", **profile) as dst:
        dst.write(rgb_band, 1)
    with rasterio.open(ms_path, "w", **profile) as dst:
        dst.write(ms_band, 1)
    return rgb_path, ms_path


@pytest.mark.parametrize("starting_status, is_verified", [("PASS", True), ("FAIL", False)])
def test_refine_with_arosics_local_publishes_a_report_reflecting_the_real_warp(
    tmp_path: Path, starting_status: str, is_verified: bool,
):
    size, gsd = 500, 0.03
    true_tx, true_ty = 2.0, -1.0
    rgb_path, ms_path = _write_textured_pair(tmp_path, size, gsd, true_tx, true_ty, amplitude=3.0, wavelength=1000.0)

    cfg = AlignmentConfig(
        rgb_red_band_index=1, ms_red_band_index=1,
        registration_channel_priority=[RegistrationChannel.RED],
        arosics=ArosicsConfig(),
    )
    rgb_meta, ms_meta = validate_inputs(rgb_path, ms_path, cfg)
    coarse = coarse_align(rgb_meta, ms_meta, cfg)

    global_transform = TransformResult(
        matrix=np.array([[1.0, 0.0, true_tx], [0.0, 1.0, true_ty]]),
        transform_type=TransformType.AFFINE, translation_px=(true_tx, true_ty),
        translation_m=(true_tx * gsd, true_ty * gsd), rotation_deg=0.0, scale=(1.0, 1.0),
        inlier_ratio=1.0, num_inliers=10, num_total_matches=10, method="test", channel_pair="red-red",
    )
    context = VerifiedGlobalContext(
        rgb_meta=rgb_meta, ms_meta=ms_meta, coarse=coarse,
        registration_transform=global_transform,
        registration_footprint=None,
        quality_report=SpatialResidualReport(96.4, None, None, None, None, [], starting_status, is_verified),
        rejected_candidates=(),
        is_verified=is_verified,
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    result = _refine_with_arosics_local(context, output_dir, cfg, __import__("logging").getLogger("test"), "automated")

    assert result.aligned_ms_path.exists()
    assert result.report_json_path.exists()
    assert result.preview_image_path.exists()

    with open(result.report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["applied_alignment_mode"] == "arosics_local"
    assert report["requested_alignment_mode"] == "automated"
    local_refinement = report["local_refinement"]
    assert local_refinement is not None
    assert local_refinement["holdout"]["median_before_px"] > 1.5
    assert local_refinement["holdout"]["median_after_px"] < 0.5

    # The headline status belongs to the gates that approved the published raster,
    # not to the starting point's classical QA (which a real Farm B run reported as
    # FAIL / 96px while publishing a gate-approved local result).
    assert report["quality"]["status"] == "PASS"
    assert result.spatial_residual_report.status == "PASS"
    starting = local_refinement["global_starting_point_quality"]
    assert starting["status"] == starting_status
    assert starting["verified"] is is_verified
    assert starting["global_rmse_px"] == 96.4

    with rasterio.open(result.aligned_ms_path) as aligned:
        assert aligned.width == size and aligned.height == size
