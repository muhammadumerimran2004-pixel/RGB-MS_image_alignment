"""JSON reports (schema ``sentinel_patch/1.0``) written atomically."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from sentinel_patch.version import REPORT_SCHEMA_VERSION, __version__


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    os.replace(partial, path)
    return path


def versions() -> dict[str, str]:
    return {
        "sentinel_patch": __version__, "numpy": np.__version__, "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
    }


def build_report(
    *, status: str, reason_code: str | None, flight_date: date, flight_date_source: str, footprint: dict,
    scene: dict | None, registration: dict | None, products: dict | None, warnings: list[str],
    drone_metadata: dict | None = None,
) -> dict:
    return {
        "schema_version": REPORT_SCHEMA_VERSION, "status": status, "reason_code": reason_code,
        "flight_date": flight_date, "drone_metadata": {"flight_date_source": flight_date_source, **(drone_metadata or {})},
        "scene": scene, "footprint": footprint, "registration": registration, "products": products,
        "warnings": warnings, "versions": versions(),
    }
