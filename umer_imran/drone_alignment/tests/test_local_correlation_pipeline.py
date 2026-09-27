import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drone_alignment.alignment.local_evidence import SparseDisplacementResult
from drone_alignment.alignment.local_evidence import CellMatchStatus, LocalMatchSample
from drone_alignment.alignment.cell_correlator import HeldoutFieldValidation
from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, TransformType
from drone_alignment.io.reader import read_metadata
from drone_alignment.pipeline import (
    VerifiedGlobalContext,
    local_correlation_align_orthomosaics,
)
from drone_alignment.quality.metrics import FootprintMetrics, SpatialResidualReport
from drone_alignment.quality.report import write_alignment_report


def _transform() -> TransformResult:
    return TransformResult(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), TransformType.AFFINE,
        (0.0, 0.0), (0.0, 0.0), 0.0, (1.0, 1.0), 1.0, 8, 8,
        channel_pair="green-green",
    )


def _quality() -> SpatialResidualReport:
    return SpatialResidualReport(None, None, None, None, None, [], "PASS", True)


def _footprint() -> FootprintMetrics:
    return FootprintMetrics(1.0, 1.0, 1.0, 1.0, 1.0)


def _context() -> VerifiedGlobalContext:
    coarse = SimpleNamespace(
        registration_bands_rgb={"green": np.ones((30, 30), np.float32)},
        registration_bands_ms={"green": np.ones((30, 30), np.float32)},
        rgb_valid_mask=np.ones((30, 30), bool),
        ms_valid_mask=np.ones((30, 30), bool),
    )
    return VerifiedGlobalContext(
        rgb_meta=object(), ms_meta=object(), coarse=coarse,
        registration_transform=_transform(), registration_footprint=_footprint(),
        quality_report=_quality(), rejected_candidates=(),
    )


def test_local_correlation_rejection_reuses_the_verified_global_context(monkeypatch, tmp_path: Path):
    import drone_alignment.pipeline as pipeline

    context = _context()
    expected = object()
    captured = {}
    evidence = SparseDisplacementResult((), None, None, (0.0, 0.0), 0, 0.0)

    monkeypatch.setattr(pipeline, "_estimate_verified_global_context", lambda *args: context)
    monkeypatch.setattr(pipeline, "compute_cell_displacements", lambda *args: evidence)

    def fake_publish(received_context, output_dir, cfg, log, **kwargs):
        captured["context"] = received_context
        captured["config"] = cfg
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(pipeline, "_publish_global_context", fake_publish)

    result = local_correlation_align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, AlignmentConfig())

    assert result is expected
    assert captured["context"] is context
    assert captured["config"].alignment_mode is AlignmentMode.LOCAL_CORRELATION
    assert captured["kwargs"]["requested_alignment_mode"] == "local_correlation"
    assert captured["kwargs"]["applied_alignment_mode"] == "automated"
    assert captured["kwargs"]["fallback"]["reason_code"] == "INSUFFICIENT_TRUSTED_CELLS"


def test_unexpected_local_correlation_error_does_not_fall_back(monkeypatch, tmp_path: Path):
    import drone_alignment.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_estimate_verified_global_context", lambda *args: _context())
    monkeypatch.setattr(pipeline, "compute_cell_displacements", lambda *args: (_ for _ in ()).throw(OSError("disk fault")))
    monkeypatch.setattr(
        pipeline, "_publish_global_context",
        lambda *args, **kwargs: pytest.fail("unexpected errors must not publish a fallback"),
    )

    with pytest.raises(OSError, match="disk fault"):
        local_correlation_align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, AlignmentConfig())


def test_local_correlation_success_uses_tiled_local_export(monkeypatch, tmp_path: Path):
    import drone_alignment.pipeline as pipeline

    config = AlignmentConfig.model_validate({
        "cell_correlation": {"grid_rows": 2, "grid_cols": 2, "min_trusted_cells": 4,
                             "min_holdout_cells": 3, "max_holdout_cells": 4},
    })
    samples = tuple(
        LocalMatchSample(
            row=row, col=col, center_xy=(15.0 + col * 10.0, 15.0 + row * 10.0),
            source_ms_xy=(15.0, 15.0), predicted_rgb_xy=(16.0, 15.0),
            residual_dx_dy=(1.0, 0.0), displacement_dx_dy=(1.0, 0.0), confidence=0.9,
            road_score=None, second_road_score=None, tree_residual_dx_dy=None,
            status=CellMatchStatus.ACCEPTED, channel="green",
        )
        for row in range(2) for col in range(2)
    )
    evidence = SparseDisplacementResult(samples, None, None, (0.0, 0.0), 4, 0.6)
    context = _context()
    context = VerifiedGlobalContext(
        rgb_meta=SimpleNamespace(), ms_meta=SimpleNamespace(path=tmp_path / "ms.tif"), coarse=context.coarse,
        registration_transform=context.registration_transform, registration_footprint=context.registration_footprint,
        quality_report=context.quality_report, rejected_candidates=(),
    )
    captured = {}

    monkeypatch.setattr(pipeline, "_estimate_verified_global_context", lambda *args: context)
    monkeypatch.setattr(pipeline, "compute_cell_displacements", lambda *args: evidence)
    monkeypatch.setattr(pipeline, "compare_displacement_fields", lambda *args: SimpleNamespace(
        selected_name="regularized_bilinear_mesh", metrics=(), reason="ok",
    ))
    monkeypatch.setattr(pipeline, "fit_selected_displacement_field", lambda *args: object())
    monkeypatch.setattr(pipeline, "evaluate_heldout_field_improvement", lambda *args: HeldoutFieldValidation(
        (), 4, 0.10, 0.10, 1.0, True, "passed",
    ))
    monkeypatch.setattr(pipeline, "_to_native_transform", lambda *args: _transform())
    monkeypatch.setattr(pipeline, "evaluate_native_displacement_footprint", lambda *args: _footprint())

    def fake_warp(coarse, native_transform, field, staged_path, *args):
        Path(staged_path).touch()
        captured["warped"] = (coarse, native_transform, field)

    class FakeAligned:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, *args, **kwargs):
            return np.zeros((30, 30), dtype=np.float32)

    monkeypatch.setattr(pipeline, "warp_ms_with_displacement_field_tiled", fake_warp)
    monkeypatch.setattr(pipeline.rasterio, "open", lambda *args, **kwargs: FakeAligned())
    monkeypatch.setattr(pipeline, "generate_alignment_preview", lambda *args, **kwargs: None)

    def fake_report(*args, **kwargs):
        captured["report"] = kwargs
        return tmp_path / "report.json"

    monkeypatch.setattr(pipeline, "write_alignment_report", fake_report)
    monkeypatch.setattr(pipeline, "_publish_global_context", lambda *args, **kwargs: pytest.fail("local success must not fall back"))

    result = local_correlation_align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, config)

    assert result.aligned_ms_path.exists()
    assert "warped" in captured
    assert captured["report"]["requested_alignment_mode"] == "local_correlation"
    assert captured["report"]["applied_alignment_mode"] == "local_correlation"
    assert captured["report"]["local_correlation"]["selected_field"] == "regularized_bilinear_mesh"


def test_report_records_requested_and_applied_modes_on_fallback(synthetic_geo_tiff_pair, tmp_path: Path):
    rgb_meta = read_metadata(synthetic_geo_tiff_pair["rgb_path"])
    ms_meta = read_metadata(synthetic_geo_tiff_pair["ms_path"])
    report_path = tmp_path / "alignment_report.json"
    config = AlignmentConfig(alignment_mode=AlignmentMode.LOCAL_CORRELATION)

    write_alignment_report(
        report_path, rgb_meta, ms_meta, _transform(), _quality(), config,
        tmp_path / "aligned.tif", tmp_path / "preview.png", footprint_metrics=_footprint(),
        requested_alignment_mode="local_correlation", applied_alignment_mode="automated",
        fallback={"reason_code": "HOLDOUT_IMPROVEMENT_FAILED", "message": "no improvement", "details": {}},
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["alignment_mode"] == "automated"
    assert report["requested_alignment_mode"] == "local_correlation"
    assert report["applied_alignment_mode"] == "automated"
    assert report["fallback"]["reason_code"] == "HOLDOUT_IMPROVEMENT_FAILED"
