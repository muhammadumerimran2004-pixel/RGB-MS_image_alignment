"""Layer D: schema-versioned explainable report assembly."""
from __future__ import annotations

from pathlib import Path

from crop_health_sentinel.io.structured_artifacts import write_csv, write_json
from crop_health_sentinel.models.types import FinalReportOutput


def process_final_report(context, quality, spectral, health, h3, logger=None) -> FinalReportOutput:
    report = {
        "schemaVersion": "1.0", "algorithmVersion": context.algorithm_version,
        "scene": {"farm": context.metadata.farm, "field": context.metadata.field, "date": context.metadata.date, "sceneId": context.metadata.scene_id, "runId": context.metadata.run_id},
        "inputs": {"requiredBands": ["B3", "B4", "B5", "B8", "SCL"], "optionalBandsMissing": list(context.bands.missing_optional_bands), "cropGeometryType": context.crop_geometry_wgs84.geom_type},
        "quality": quality.summary, "spectral": spectral.summary, "health": health.summary, "spatial": h3.summary,
        "limitations": ["Current-scene spectral assessment only; not a disease diagnosis.", "No historical trend, weather, crop-stage, yield, or machine-learning model is included."],
    }
    summary = {"sceneStatus": quality.scene_status.value, "fieldHealthScore": health.field_health_score, "healthClass": health.health_class, "emittedH3Cells": len(h3.cells)}
    phase = Path(context.output_dir) / "phase_04"; prefix = context.metadata.scene_id
    report_path = write_json(phase / f"{prefix}_final_report.json", report)
    write_csv(phase / f"{prefix}_final_report.csv", list(summary), [summary]); write_json(phase / f"{prefix}_layer_d_report_summary.json", summary); write_csv(phase / f"{prefix}_layer_d_report_summary.csv", list(summary), [summary])
    return FinalReportOutput(report, summary, report_path)
