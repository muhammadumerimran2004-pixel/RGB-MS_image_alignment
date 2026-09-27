import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, ManualAlignmentConfig, TransformValidationConfig, QualityConfig
from drone_alignment.alignment.control_points import build_control_point_set
from drone_alignment.alignment.manual import ManualThinPlateSplineField, run_manual_alignment
from drone_alignment.alignment.transform_estimator import TransformUnreliableError
from drone_alignment.pipeline import manual_align_orthomosaics


def _apply_affine(matrix, xy):
    import cv2
    return cv2.transform(np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2), matrix).reshape(-1, 2)


# --- run_manual_alignment unit tests (no raster I/O) ------------------------
# Legacy transform.manual_tps_* -> manual.* config migration is covered in
# test_schema_migration.py.

def test_run_manual_alignment_two_points_similarity_translation_only_bounds():
    ref = np.array([(10.0, 10.0), (50.0, 10.0)])
    tgt = ref - np.array([2.0, 3.0])
    cps = build_control_point_set(["P1", "P2"], ref, tgt)
    result = run_manual_alignment(
        cps, ManualAlignmentConfig(), TransformValidationConfig(max_translation_m=50.0),
        QualityConfig(), (200, 200), 0.05,
    )
    assert result.transform.method == "manual_similarity"
    assert result.field is None
    assert result.quality_report.status == "UNVERIFIED"  # n=2 control points, no LOO, no check points


def test_run_manual_alignment_n_le_3_without_check_points_is_unverified():
    ref = np.array([(10.0, 10.0), (400.0, 20.0), (60.0, 400.0)])
    tgt = ref - np.array([2.0, 1.0])
    cps = build_control_point_set(["P1", "P2", "P3"], ref, tgt)
    result = run_manual_alignment(
        cps, ManualAlignmentConfig(), TransformValidationConfig(max_translation_m=50.0),
        QualityConfig(), (500, 500), 0.05,
    )
    assert result.quality_report.status == "UNVERIFIED"
    assert result.quality_report.global_rmse_px is None


def test_run_manual_alignment_uses_checkpoint_residuals_when_available_and_no_loo():
    ref = np.array([(10.0, 10.0), (400.0, 20.0), (60.0, 400.0)])
    tgt = ref - np.array([2.0, 1.0])
    check_ref = np.array([(200.0, 200.0)])
    check_tgt = check_ref - np.array([2.0, 1.0])
    cps = build_control_point_set(
        ["P1", "P2", "P3", "C1"],
        np.vstack([ref, check_ref]), np.vstack([tgt, check_tgt]),
        is_check=[False, False, False, True],
    )
    result = run_manual_alignment(
        cps, ManualAlignmentConfig(), TransformValidationConfig(max_translation_m=50.0),
        QualityConfig(), (500, 500), 0.05,
    )
    assert result.quality_report.status != "UNVERIFIED"
    assert result.payload["checkpoint_rmse_px"] == pytest.approx(0.0, abs=1e-6)


def test_run_manual_alignment_checkpoint_rmse_gate_rejects():
    ref = np.array([(10.0, 10.0), (400.0, 20.0), (60.0, 400.0)])
    tgt = ref - np.array([2.0, 1.0])
    check_ref = np.array([(200.0, 200.0)])
    check_tgt = check_ref - np.array([20.0, 20.0])  # way off vs the fitted translation
    cps = build_control_point_set(
        ["P1", "P2", "P3", "C1"],
        np.vstack([ref, check_ref]), np.vstack([tgt, check_tgt]),
        is_check=[False, False, False, True],
    )
    with pytest.raises(TransformUnreliableError, match="Check-point RMSE"):
        run_manual_alignment(
            cps, ManualAlignmentConfig(max_checkpoint_rmse_px=1.0), TransformValidationConfig(max_translation_m=50.0),
            QualityConfig(), (500, 500), 0.05,
        )


def test_run_manual_alignment_selects_tps_and_passes_jacobian_for_smooth_bend():
    tgt = np.array([
        (10.0, 10.0), (490.0, 10.0), (10.0, 490.0), (490.0, 490.0),
        (250.0, 10.0), (10.0, 250.0), (490.0, 250.0), (250.0, 490.0),
    ])
    baseline = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    ref_linear = _apply_affine(baseline, tgt)
    bend = 1.5 * np.sin(tgt[:, 0] / 100.0)[:, None] * np.array([1.0, 0.4]) + \
        1.5 * np.cos(tgt[:, 1] / 100.0)[:, None] * np.array([0.4, 1.0])
    ref = ref_linear + bend
    cps = build_control_point_set([f"P{i}" for i in range(8)], ref, tgt)
    result = run_manual_alignment(
        cps, ManualAlignmentConfig(min_tps_points=8, min_hull_coverage=0.01),
        TransformValidationConfig(max_translation_m=50.0), QualityConfig(), (500, 500), 0.05,
    )
    assert result.transform.method == "manual_tps"
    assert isinstance(result.field, ManualThinPlateSplineField)
    assert result.payload["jacobian"]["min_det"] > 0.0


def test_run_manual_alignment_jacobian_gate_rejects_folding_field():
    tgt = np.array([
        (10.0, 10.0), (490.0, 10.0), (10.0, 490.0), (490.0, 490.0),
        (250.0, 10.0), (10.0, 250.0), (490.0, 250.0), (250.0, 490.0),
    ])
    baseline = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    ref_linear = _apply_affine(baseline, tgt)
    # A violent, high-frequency bend whose gradient folds the mapping locally.
    bend = 60.0 * np.sin(tgt[:, 0] / 25.0)[:, None] * np.array([1.0, 0.0]) + \
        60.0 * np.cos(tgt[:, 1] / 25.0)[:, None] * np.array([0.0, 1.0])
    ref = ref_linear + bend
    cps = build_control_point_set([f"P{i}" for i in range(8)], ref, tgt)
    with pytest.raises(TransformUnreliableError, match="Jacobian"):
        run_manual_alignment(
            cps, ManualAlignmentConfig(min_tps_points=8, min_hull_coverage=0.01, min_tps_gain=0.0),
            TransformValidationConfig(max_translation_m=50.0), QualityConfig(), (500, 500), 0.05,
        )


def test_run_manual_alignment_outlier_reject_policy_raises():
    ref = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0), (5.0, 5.0)])
    tgt = ref.copy()
    tgt[2] += np.array([20.0, 20.0])
    cps = build_control_point_set([f"P{i}" for i in range(5)], ref, tgt)
    # Forced translation model: with only 2 degrees of freedom, an outlier
    # cannot be silently absorbed the way a full affine fit might, so the
    # robust residual test isolates outlier detection from the bounds gate.
    with pytest.raises(TransformUnreliableError, match="outliers"):
        run_manual_alignment(
            cps, ManualAlignmentConfig(model="translation", outlier_policy="reject"),
            TransformValidationConfig(max_translation_m=50.0), QualityConfig(), (100, 100), 0.05,
        )


def test_run_manual_alignment_outlier_warn_policy_drops_and_reports():
    ref = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0), (5.0, 5.0)])
    tgt = ref.copy()
    tgt[2] += np.array([20.0, 20.0])
    cps = build_control_point_set([f"P{i}" for i in range(5)], ref, tgt)
    result = run_manual_alignment(
        cps, ManualAlignmentConfig(model="translation", outlier_policy="warn"),
        TransformValidationConfig(max_translation_m=50.0), QualityConfig(), (100, 100), 0.05,
    )
    outlier_flags = {p["id"]: p["outlier"] for p in result.payload["points"]}
    assert outlier_flags["P2"] is True
    assert not any(v for k, v in outlier_flags.items() if k != "P2")


def test_run_manual_alignment_bounds_rejection_for_wild_rotation_scale():
    tgt = np.array([(10.0, 10.0), (80.0, 15.0), (15.0, 80.0)])
    ref = np.array([
        (17.575644482542210, -5.655664833626261),
        (100.864542042443030, 31.576170244118000),
        (-7.440186524985244, 82.079494589508260),
    ])
    cps = build_control_point_set(["P1", "P2", "P3"], ref, tgt)
    bounds = TransformValidationConfig(max_translation_m=50.0, max_rotation_deg=5.0, max_scale_deviation=0.1)
    with pytest.raises(TransformUnreliableError, match="sanity bounds"):
        run_manual_alignment(cps, ManualAlignmentConfig(), bounds, QualityConfig(), (200, 200), 0.05)


def test_run_manual_alignment_duplicate_points_rejected():
    ref = np.array([(10.0, 20.0), (80.0, 25.0), (10.3, 20.2)])
    tgt = np.array([(12.0, 19.0), (81.0, 27.0), (12.2, 19.1)])
    cps = build_control_point_set(["P1", "P2", "P3"], ref, tgt)
    with pytest.raises(TransformUnreliableError, match="apart"):
        run_manual_alignment(cps, ManualAlignmentConfig(), TransformValidationConfig(), QualityConfig(), (200, 200), 0.05)


def test_run_manual_alignment_near_collinear_points_rejected():
    ref = np.array([(10.0, 10.0), (20.0, 20.001), (30.0, 30.0)])
    tgt = np.array([(12.0, 12.0), (22.0, 22.001), (32.0, 32.0)])
    cps = build_control_point_set(["P1", "P2", "P3"], ref, tgt)
    with pytest.raises(TransformUnreliableError, match="collinear"):
        run_manual_alignment(cps, ManualAlignmentConfig(), TransformValidationConfig(), QualityConfig(), (200, 200), 0.05)


# --- pipeline-level integration tests ---------------------------------------

def test_manual_align_orthomosaics_pipeline_affine(synthetic_geo_tiff_pair, tmp_path: Path):
    output_dir = tmp_path / "manual_out"
    config = AlignmentConfig(alignment_mode=AlignmentMode.MANUAL)
    pts_rgb = [(100.0, 100.0), (400.0, 120.0), (150.0, 400.0)]
    pts_ms = [(52.0, 51.0), (203.0, 60.0), (76.0, 202.0)]

    result = manual_align_orthomosaics(
        rgb_path=synthetic_geo_tiff_pair["rgb_path"], ms_path=synthetic_geo_tiff_pair["ms_path"],
        output_dir=output_dir, pts_rgb=pts_rgb, pts_ms=pts_ms, config=config,
    )

    assert result.aligned_ms_path.exists()
    assert result.report_json_path.exists()
    assert result.preview_image_path.exists()

    with open(result.report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["requested_alignment_mode"] == "manual"
    assert report["applied_alignment_mode"] == "manual_affine"
    assert report["transform"]["method"] == "manual_affine"

    mcp = report["manual_control_points"]
    assert mcp["model"] == "affine"
    assert mcp["coordinate_mode"] == "pixel"
    assert mcp["pixel_convention"] == "center"
    assert len(mcp["points"]) == 3
    assert set(mcp["points"][0].keys()) == {
        "id", "role", "rgb", "ms", "fit_residual_px", "loo_residual_px", "robust_z", "outlier",
    }
    # 3 control points: below min_tps_points and below the leave-one-out
    # floor (n >= 4), so no LOO metric is fabricated for either model.
    assert mcp["loo_rmse_px"]["affine"] is None
    assert mcp["baseline"]["rotation_deg"] is not None
    assert mcp["jacobian"] is None
    assert mcp["field_lattice_max_error_px"] is None

    with rasterio.open(result.aligned_ms_path) as dst:
        assert dst.count == 4
        assert dst.width > 0 and dst.height > 0


def test_manual_align_orthomosaics_pipeline_map_coordinates(synthetic_geo_tiff_pair, tmp_path: Path):
    rgb_path = synthetic_geo_tiff_pair["rgb_path"]
    ms_path = synthetic_geo_tiff_pair["ms_path"]
    with rasterio.open(rgb_path) as rgb_src, rasterio.open(ms_path) as ms_src:
        pts_rgb = [rgb_src.transform * (100.0, 100.0), rgb_src.transform * (400.0, 120.0)]
        pts_ms = [ms_src.transform * (49.5, 49.5), ms_src.transform * (199.5, 59.5)]

    result = manual_align_orthomosaics(
        rgb_path=rgb_path, ms_path=ms_path, output_dir=tmp_path / "manual_map_out",
        pts_rgb=pts_rgb, pts_ms=pts_ms, coordinate_mode="map",
        config=AlignmentConfig(alignment_mode=AlignmentMode.MANUAL),
    )
    assert result.aligned_ms_path.exists()
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["manual_control_points"]["coordinate_mode"] == "map"


def test_manual_align_orthomosaics_pixel_center_vs_corner_convention_differ(synthetic_geo_tiff_pair, tmp_path: Path):
    rgb_path = synthetic_geo_tiff_pair["rgb_path"]
    ms_path = synthetic_geo_tiff_pair["ms_path"]
    pts_rgb = [(100.0, 100.0), (400.0, 120.0)]
    pts_ms = [(49.5, 49.5), (199.5, 59.5)]

    center = manual_align_orthomosaics(
        rgb_path=rgb_path, ms_path=ms_path, output_dir=tmp_path / "center",
        pts_rgb=pts_rgb, pts_ms=pts_ms, pixel_convention="center",
        config=AlignmentConfig(alignment_mode=AlignmentMode.MANUAL),
    )
    corner = manual_align_orthomosaics(
        rgb_path=rgb_path, ms_path=ms_path, output_dir=tmp_path / "corner",
        pts_rgb=pts_rgb, pts_ms=pts_ms, pixel_convention="corner",
        config=AlignmentConfig(alignment_mode=AlignmentMode.MANUAL),
    )
    assert center.transform_result.translation_px != corner.transform_result.translation_px


def test_manual_align_orthomosaics_uses_ids_and_roles_in_report(synthetic_geo_tiff_pair, tmp_path: Path):
    rgb_path = synthetic_geo_tiff_pair["rgb_path"]
    ms_path = synthetic_geo_tiff_pair["ms_path"]
    pts_rgb = [(100.0, 100.0), (400.0, 120.0), (150.0, 400.0), (300.0, 300.0)]
    pts_ms = [(52.0, 51.0), (203.0, 60.0), (76.0, 202.0), (152.0, 152.0)]
    ids = ["A", "B", "C", "CHK"]
    roles = ["control", "control", "control", "check"]

    result = manual_align_orthomosaics(
        rgb_path=rgb_path, ms_path=ms_path, output_dir=tmp_path / "ids_roles",
        pts_rgb=pts_rgb, pts_ms=pts_ms, ids=ids, roles=roles,
        config=AlignmentConfig(alignment_mode=AlignmentMode.MANUAL),
    )
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    points = report["manual_control_points"]["points"]
    assert [p["id"] for p in points] == ids
    assert [p["role"] for p in points] == roles


def test_manual_align_orthomosaics_scipy_unavailable_falls_back_to_affine(synthetic_geo_tiff_pair, tmp_path: Path, monkeypatch):
    import drone_alignment.alignment.manual as manual_mod
    monkeypatch.setattr(manual_mod, "is_scipy_available", lambda: False)

    pts_rgb = [(50.0, 50.0), (450.0, 60.0), (60.0, 450.0), (450.0, 450.0), (250.0, 50.0), (50.0, 250.0), (450.0, 250.0), (250.0, 450.0)]
    pts_ms = [(24.0, 24.0), (224.0, 29.0), (29.0, 224.0), (224.0, 224.0), (124.0, 24.0), (24.0, 124.0), (224.0, 124.0), (124.0, 224.0)]

    result = manual_align_orthomosaics(
        rgb_path=synthetic_geo_tiff_pair["rgb_path"], ms_path=synthetic_geo_tiff_pair["ms_path"],
        output_dir=tmp_path / "no_scipy",
        pts_rgb=pts_rgb, pts_ms=pts_ms,
        config=AlignmentConfig(alignment_mode=AlignmentMode.MANUAL, manual=ManualAlignmentConfig(min_tps_points=4)),
    )
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["manual_control_points"]["model"] == "affine"
    assert "SciPy" in report["manual_control_points"]["model_reason"]


def test_manual_align_orthomosaics_tps_uses_field_aware_footprint(synthetic_geo_tiff_pair, tmp_path: Path, monkeypatch):
    import drone_alignment.pipeline as pipeline_mod

    calls = {"displacement": 0, "affine": 0}
    real_displacement = pipeline_mod.evaluate_native_displacement_footprint
    real_affine = pipeline_mod.evaluate_native_footprint

    def spy_displacement(*args, **kwargs):
        calls["displacement"] += 1
        return real_displacement(*args, **kwargs)

    def spy_affine(*args, **kwargs):
        calls["affine"] += 1
        return real_affine(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "evaluate_native_displacement_footprint", spy_displacement)
    monkeypatch.setattr(pipeline_mod, "evaluate_native_footprint", spy_affine)

    # MS (target) points must stay within the 256x256 MS fixture's bounds.
    ms = np.array([
        (25.0, 25.0), (225.0, 30.0), (30.0, 225.0), (225.0, 225.0),
        (125.0, 25.0), (25.0, 125.0), (225.0, 125.0), (125.0, 225.0),
    ])
    baseline = np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 1.0]])
    ref_linear = _apply_affine(baseline, ms)
    bend = 3.0 * np.sin(ms[:, 0] / 50.0)[:, None] * np.array([1.0, 0.3]) + \
        3.0 * np.cos(ms[:, 1] / 50.0)[:, None] * np.array([0.3, 1.0])
    ref = ref_linear + bend  # stays within the 512x512 RGB fixture's bounds

    result = manual_align_orthomosaics(
        rgb_path=synthetic_geo_tiff_pair["rgb_path"], ms_path=synthetic_geo_tiff_pair["ms_path"],
        output_dir=tmp_path / "tps_footprint",
        pts_rgb=ref.tolist(), pts_ms=ms.tolist(),
        config=AlignmentConfig(alignment_mode=AlignmentMode.MANUAL, manual=ManualAlignmentConfig(min_tps_points=8, min_hull_coverage=0.01)),
    )

    assert result.aligned_ms_path.exists()
    assert calls["displacement"] == 1
    assert calls["affine"] == 0
    with open(result.report_json_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["manual_control_points"]["model"] == "tps"
    assert report["manual_control_points"]["field_lattice_max_error_px"] is not None


def test_manual_align_orthomosaics_pixel_out_of_bounds_rejected(synthetic_geo_tiff_pair, tmp_path: Path):
    with pytest.raises(TransformUnreliableError, match="outside the RGB raster"):
        manual_align_orthomosaics(
            rgb_path=synthetic_geo_tiff_pair["rgb_path"], ms_path=synthetic_geo_tiff_pair["ms_path"],
            output_dir=tmp_path / "oob", pts_rgb=[(10000.0, 10000.0), (20.0, 20.0)], pts_ms=[(10.0, 10.0), (12.0, 12.0)],
            config=AlignmentConfig(alignment_mode=AlignmentMode.MANUAL),
        )


def test_manual_tps_field_taper_passes_through_control_points_with_fallback():
    """P1.2 regression: with fallback_weight > 0 (tapering enabled), the field must
    still reproduce its own control points exactly."""
    ref = np.array([(10.0, 20.0), (80.0, 25.0), (30.0, 90.0)])
    tgt = np.array([(12.0, 19.0), (81.0, 27.0), (31.0, 88.0)])
    cps = build_control_point_set(["P1", "P2", "P3"], ref, tgt)
    # 3 points is below min_tps_points, so force the model to exercise the field directly.
    cfg_forced = ManualAlignmentConfig(model="tps", taper_fallback_weight=0.3, taper_support_radius_px=25.0)
    result = run_manual_alignment(cps, cfg_forced, TransformValidationConfig(max_translation_m=50.0), QualityConfig(), (200, 200), 0.05)
    field = result.field
    assert field is not None
    predicted = _apply_affine(result.transform.matrix, tgt) + field.evaluate(ref)
    np.testing.assert_allclose(predicted, ref, atol=1e-5)


def test_manual_tps_field_taper_decays_far_from_controls():
    field = ManualThinPlateSplineField(
        interpolator_x=lambda points: np.full(len(points), 5.0),
        interpolator_y=lambda points: np.full(len(points), 5.0),
        control_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
        support_radius_px=10.0,
        fallback_weight=0.5,
    )
    far = field.evaluate(np.array([[10_000.0, 10_000.0]]))
    assert np.linalg.norm(far[0]) < 1e-4
