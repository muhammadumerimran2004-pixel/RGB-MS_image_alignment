from pathlib import Path

import numpy as np

from drone_alignment.alignment.transform_estimator import TransformResult
from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, TransformType
from drone_alignment.local_mesh_export import LocalMeshExportResult
from drone_alignment.pipeline import AlignmentResult, local_mesh_align_orthomosaics
from drone_alignment.quality.metrics import SpatialResidualReport


def _alignment_result(tmp_path: Path) -> AlignmentResult:
    output = tmp_path / "fallback.tif"
    report = tmp_path / "fallback.json"
    preview = tmp_path / "fallback.png"
    transform = TransformResult(np.array([[1., 0., 0.], [0., 1., 0.]]), TransformType.AFFINE,
                                (0., 0.), (0., 0.), 0., (1., 1.), 1., 1, 1)
    quality = SpatialResidualReport(None, None, None, None, None, [], "PASS", True)
    return AlignmentResult(output, report, preview, quality, transform)


def test_local_mesh_rejection_falls_back_to_automated(monkeypatch, tmp_path: Path):
    import drone_alignment.pipeline as pipeline

    monkeypatch.setattr(pipeline, "validate_inputs", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(pipeline, "export_local_mesh_candidate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("coverage too low")))
    expected = _alignment_result(tmp_path)
    captured = {}

    def fake_global(*args, **kwargs):
        captured["config"] = args[3]
        return expected

    monkeypatch.setattr(pipeline, "align_orthomosaics", fake_global)
    result = local_mesh_align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, AlignmentConfig())
    assert result is expected
    assert captured["config"].alignment_mode == AlignmentMode.AUTOMATED


def test_local_mesh_success_returns_local_export(monkeypatch, tmp_path: Path):
    import drone_alignment.pipeline as pipeline

    monkeypatch.setattr(pipeline, "validate_inputs", lambda *args, **kwargs: (None, None))
    native = TransformResult(np.array([[1., 0., 0.], [0., 1., 0.]]), TransformType.AFFINE,
                             (0., 0.), (0., 0.), 0., (1., 1.), 1., 1, 1)
    exported = LocalMeshExportResult(
        tmp_path / "local.tif", tmp_path / "preview.png", tmp_path / "review.json", native,
        0.40, 10, 0.25, "tapered_thin_plate_spline",
    )
    monkeypatch.setattr(pipeline, "export_local_mesh_candidate", lambda *args, **kwargs: exported)
    result = local_mesh_align_orthomosaics(Path("rgb.tif"), Path("ms.tif"), tmp_path, AlignmentConfig())
    assert result.aligned_ms_path == exported.aligned_path
    assert result.report_json_path == exported.review_path
    assert result.transform_result is native
