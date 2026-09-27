import json
from pathlib import Path

import numpy as np

from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.config.schema import AlignmentConfig, TransformType
from drone_alignment.io.reader import read_metadata
from drone_alignment.quality.metrics import SpatialResidualReport
from drone_alignment.quality.report import write_alignment_report


def _transform() -> TransformResult:
    return TransformResult(
        matrix=np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]),
        transform_type=TransformType.AFFINE,
        translation_px=(2.0, -1.0),
        translation_m=(0.12, -0.06),
        rotation_deg=0.0,
        scale=(1.0, 1.0),
        inlier_ratio=0.8,
        num_inliers=8,
        num_total_matches=10,
    )


def _quality() -> SpatialResidualReport:
    return SpatialResidualReport(None, None, None, None, None, [], "PASS", True)


def test_report_serializes_local_refinement_and_manual_control_points_payloads(
    synthetic_geo_tiff_pair, tmp_path: Path,
):
    rgb_meta = read_metadata(synthetic_geo_tiff_pair["rgb_path"])
    ms_meta = read_metadata(synthetic_geo_tiff_pair["ms_path"])
    report_path = tmp_path / "report.json"

    local_refinement_payload = {"engine": "arosics_local", "tie_points": {"valid": 42}}
    manual_control_points_payload = {"model": "tps", "loo_rmse_px": {"affine": 1.4, "tps": 0.6}}

    write_alignment_report(
        report_path, rgb_meta, ms_meta, _transform(), _quality(), AlignmentConfig(),
        synthetic_geo_tiff_pair["ms_path"], synthetic_geo_tiff_pair["ms_path"],
        local_refinement=local_refinement_payload,
        manual_control_points=manual_control_points_payload,
    )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["local_refinement"] == local_refinement_payload
    assert report["manual_control_points"] == manual_control_points_payload


def test_report_defaults_new_payloads_to_none(synthetic_geo_tiff_pair, tmp_path: Path):
    rgb_meta = read_metadata(synthetic_geo_tiff_pair["rgb_path"])
    ms_meta = read_metadata(synthetic_geo_tiff_pair["ms_path"])
    report_path = tmp_path / "report_defaults.json"

    write_alignment_report(
        report_path, rgb_meta, ms_meta, _transform(), _quality(), AlignmentConfig(),
        synthetic_geo_tiff_pair["ms_path"], synthetic_geo_tiff_pair["ms_path"],
    )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["local_refinement"] is None
    assert report["manual_control_points"] is None
    # Pre-existing payload keys must be unaffected by the new parameters.
    assert report["local_correlation"] is None
    assert report["fallback"] is None
