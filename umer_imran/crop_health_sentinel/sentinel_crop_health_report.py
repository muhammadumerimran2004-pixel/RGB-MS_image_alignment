"""Production-facing Sentinel crop-health adapter."""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from .errors import PublicContractError
from .pipeline import run_crop_health_report
from .io.structured_artifacts import write_json
from .utils.lock import publication_lock

_METADATA: dict[Path, dict[str, Any]] = {}
_METADATA_LOCK = threading.RLock()

def generate_sentinel_crop_health_report(
    raw_data_dir: str | Path,
    crop_geometry: str | Mapping[str, Any] | Any,
    output_dir: str | Path,
    scene_meta: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """Generate the public ``crophealthindex.json`` artifact.

    Implemented in Phase 10 after the analytical pipeline is complete.
    """
    target = Path(output_dir).resolve(); stage = target / ".crop-health-staging" / f"{uuid.uuid4().hex}"
    with publication_lock(target):
        try:
            result = run_crop_health_report(raw_data_dir, crop_geometry, stage, scene_meta, config_path=config_path, logger=logger)
            rows = [] if result.h3 is None else [{"h3Index": cell.h3_index, "healthScore": round(cell.mean_health / 100, 4)} for cell in result.h3.cells]
            if any(set(row) != {"h3Index", "healthScore"} or not isinstance(row["h3Index"], str) or not 0 <= row["healthScore"] <= 1 for row in rows):
                raise PublicContractError("Invalid crophealthindex public payload.")
            final = write_json(target / "crophealthindex.json", rows).resolve()
            metadata = {"averageNdvi": None if result.spectral is None else result.spectral.summary["indices"].get("ndvi", {}).get("mean"),
                "fieldHealthScore": None if result.health is None else result.health.field_health_score,
                "healthClass": None if result.health is None else result.health.health_class,
                "sceneStatus": result.scene_status.value, "algorithmVersion": result.layer_a.summary.get("algorithmVersion", "sentinel-crop-health/1.0.0")}
            with _METADATA_LOCK:
                if len(_METADATA) >= 1024: _METADATA.pop(next(iter(_METADATA)))
                _METADATA[final] = metadata
            return str(final)
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def get_sentinel_crop_health_report_metadata(output_path: str | Path) -> dict[str, Any] | None:
    """Return one-time worker metadata after Phase 10 implementation."""
    with _METADATA_LOCK:
        return _METADATA.pop(Path(output_path).resolve(), None)
