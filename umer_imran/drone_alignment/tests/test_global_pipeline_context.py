from pathlib import Path

import numpy as np

from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.config.schema import AlignmentConfig, TransformType
from drone_alignment.pipeline import (
    VerifiedGlobalContext,
    _estimate_verified_global_context,
    align_orthomosaics,
)
from drone_alignment.quality.metrics import FootprintMetrics, SpatialResidualReport


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


def _footprint() -> FootprintMetrics:
    return FootprintMetrics(0.95, 0.94, 0.93, 0.98, 0.90)


def test_estimated_global_context_preserves_one_verified_candidate(monkeypatch):
    import drone_alignment.pipeline as pipeline

    rgb_meta, ms_meta, coarse = object(), object(), object()
    calls = {}

    def fake_validate(rgb_path, ms_path, config):
        calls["validated"] = (rgb_path, ms_path, config)
        return rgb_meta, ms_meta

    def fake_coarse(received_rgb, received_ms, config):
        calls["coarsened"] = (received_rgb, received_ms, config)
        return coarse

    def fake_candidate(received_coarse, config, log):
        calls["candidate"] = (received_coarse, config, log)
        return _transform(), _footprint(), _quality(), ["green/orb rejected"]

    monkeypatch.setattr(pipeline, "validate_inputs", fake_validate)
    monkeypatch.setattr(pipeline, "coarse_align", fake_coarse)
    monkeypatch.setattr(pipeline, "_estimate_global_candidate", fake_candidate)

    cfg = AlignmentConfig()
    context = _estimate_verified_global_context(Path("rgb.tif"), Path("ms.tif"), cfg, pipeline.logger)

    assert context.rgb_meta is rgb_meta
    assert context.ms_meta is ms_meta
    assert context.coarse is coarse
    assert context.registration_transform.translation_px == (2.0, -1.0)
    assert context.registration_footprint == _footprint()
    assert context.quality_report.status == "PASS"
    assert context.rejected_candidates == ("green/orb rejected",)
    assert calls["coarsened"][:2] == (rgb_meta, ms_meta)
    assert calls["candidate"][0] is coarse


def test_automated_alignment_publishes_the_context_it_estimated(monkeypatch, tmp_path: Path):
    import drone_alignment.pipeline as pipeline

    context = VerifiedGlobalContext(
        rgb_meta=object(),
        ms_meta=object(),
        coarse=object(),
        registration_transform=_transform(),
        registration_footprint=_footprint(),
        quality_report=_quality(),
        rejected_candidates=(),
    )
    captured = {}
    expected = object()

    def fake_estimate(rgb_path, ms_path, cfg, log):
        captured["estimate"] = (rgb_path, ms_path, cfg, log)
        return context

    def fake_publish(received_context, output_dir, cfg, log):
        captured["publish"] = (received_context, output_dir, cfg, log)
        return expected

    monkeypatch.setattr(pipeline, "_estimate_verified_global_context", fake_estimate)
    monkeypatch.setattr(pipeline, "_publish_global_context", fake_publish)

    result = align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, AlignmentConfig())

    assert result is expected
    assert captured["publish"][0] is context
    assert captured["publish"][1] == tmp_path.resolve()
    assert captured["estimate"][2] is captured["publish"][2]
