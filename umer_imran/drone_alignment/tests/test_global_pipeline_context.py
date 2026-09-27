from pathlib import Path

import numpy as np
import pytest

from drone_alignment.alignment.rejections import LocalRefinementRejected
from drone_alignment.alignment.transform_estimator import TransformResult, TransformUnreliableError
from drone_alignment.config.schema import AlignmentConfig, ArosicsConfig, QualityConfig, TransformType
from drone_alignment.pipeline import (
    VerifiedGlobalContext,
    _estimate_global_candidate,
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
        return _transform(), _footprint(), _quality(), ["green/orb rejected"], True

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
    assert context.is_verified is True
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

    def fake_publish(received_context, output_dir, cfg, log, **kwargs):
        captured["publish"] = (received_context, output_dir, cfg, log)
        captured["publish_kwargs"] = kwargs
        return expected

    monkeypatch.setattr(pipeline, "_estimate_verified_global_context", fake_estimate)
    monkeypatch.setattr(pipeline, "_publish_global_context", fake_publish)

    # AROSICS local refinement is attempted by default after the global
    # context is estimated (and would hit real file I/O on these placeholder
    # paths before ever reaching the mocks above); disable it so this test
    # exercises the classical estimate/publish path it is actually about.
    cfg = AlignmentConfig(arosics=ArosicsConfig(enabled=False))
    result = align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, cfg)

    assert result is expected
    assert captured["publish"][0] is context
    assert captured["publish"][1] == tmp_path.resolve()
    # align_orthomosaics() now estimates and publishes against the exact same
    # config object (no defensive copy is needed: AROSICS refinement is a
    # separate, later step that never mutates the config estimation used).
    assert captured["estimate"][2] is captured["publish"][2]
    assert captured["publish_kwargs"]["requested_alignment_mode"] == "automated"
    assert captured["publish_kwargs"]["applied_alignment_mode"] == "automated_global"


# --- unverified global candidate -> AROSICS local as a best-effort recovery -
#
# AROSICS' own coarse(COREG)-to-fine(COREG_LOCAL) design does not require a
# precise global pre-alignment: COREG_LOCAL's own tie-point/coverage/holdout/
# full-grid gates are the real safety net. When every classical/AROSICS-
# global/phase-correlation candidate fails our independent QA, the pipeline
# must still hand the best computed candidate to AROSICS local refinement
# instead of aborting outright - but must never publish that unverified
# candidate on its own.

def _unverified_context() -> VerifiedGlobalContext:
    return VerifiedGlobalContext(
        rgb_meta=object(), ms_meta=object(), coarse=object(),
        registration_transform=_transform(), registration_footprint=_footprint(),
        quality_report=_quality(), rejected_candidates=("green-green/sift: independent/residual QA status is FAIL",),
        is_verified=False,
    )


def test_unverified_global_candidate_is_handed_to_arosics_local_and_published_on_success(monkeypatch, tmp_path: Path):
    import drone_alignment.pipeline as pipeline

    context = _unverified_context()
    expected = object()
    captured = {}

    monkeypatch.setattr(pipeline, "_estimate_verified_global_context", lambda *a, **k: context)

    def fake_refine(received_context, output_dir, cfg, log, requested_mode):
        captured["refine_context"] = received_context
        return expected

    monkeypatch.setattr(pipeline, "_refine_with_arosics_local", fake_refine)

    cfg = AlignmentConfig()  # arosics.enabled and arosics.local.enabled are True by default
    result = align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, cfg)

    assert result is expected
    assert captured["refine_context"] is context


def test_unverified_global_candidate_raises_when_arosics_local_also_rejects(monkeypatch, tmp_path: Path):
    import drone_alignment.pipeline as pipeline

    context = _unverified_context()
    monkeypatch.setattr(pipeline, "_estimate_verified_global_context", lambda *a, **k: context)

    def fake_refine(*a, **k):
        raise LocalRefinementRejected("INSUFFICIENT_TIE_POINTS", "not enough valid tie points")

    monkeypatch.setattr(pipeline, "_refine_with_arosics_local", fake_refine)
    publish_calls = []
    monkeypatch.setattr(pipeline, "_publish_global_context", lambda *a, **k: publish_calls.append(1))

    cfg = AlignmentConfig()
    with pytest.raises(TransformUnreliableError, match="No verified alignment candidate"):
        align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, cfg)

    # The unverified candidate must never be silently published as a fallback.
    assert publish_calls == []


def test_unverified_global_candidate_raises_immediately_when_arosics_local_disabled(monkeypatch, tmp_path: Path):
    import drone_alignment.pipeline as pipeline

    context = _unverified_context()
    monkeypatch.setattr(pipeline, "_estimate_verified_global_context", lambda *a, **k: context)
    publish_calls = []
    monkeypatch.setattr(pipeline, "_publish_global_context", lambda *a, **k: publish_calls.append(1))

    cfg = AlignmentConfig(arosics=ArosicsConfig(enabled=False))
    with pytest.raises(TransformUnreliableError, match="No verified alignment candidate"):
        align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, cfg)

    assert publish_calls == []


def test_estimate_global_candidate_falls_back_to_unverified_when_qa_impossibly_strict(synthetic_geo_tiff_pair):
    """Real (non-mocked) exercise of _estimate_global_candidate: with residual
    and phase-correlation verification thresholds no real candidate can
    satisfy, every classical/phase candidate is computed and rejected, but
    the function must still return the best one with is_verified=False
    instead of raising."""
    from drone_alignment.io.validators import validate_inputs
    from drone_alignment.alignment.coarse import coarse_align

    cfg = AlignmentConfig(
        arosics=ArosicsConfig(enabled=False),
        quality=QualityConfig(max_acceptable_rmse_px=1e-6, max_acceptable_edge_rmse_px=1e-6),
    )
    cfg.transform.min_phase_verification_correlation = 1.0
    rgb_meta, ms_meta = validate_inputs(synthetic_geo_tiff_pair["rgb_path"], synthetic_geo_tiff_pair["ms_path"], cfg)
    coarse = coarse_align(rgb_meta, ms_meta, cfg)

    transform_res, footprint, quality_rep, rejection_reasons, is_verified = _estimate_global_candidate(
        coarse, cfg, __import__("logging").getLogger("test"),
    )

    assert is_verified is False
    assert len(rejection_reasons) > 0
    assert transform_res.matrix.shape == (2, 3)
    assert quality_rep.status == "FAIL"
