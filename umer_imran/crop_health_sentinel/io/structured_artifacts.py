from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from crop_health_sentinel.errors import ArtifactError


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_value(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); partial = path.with_suffix(path.suffix + ".partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(_json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(partial, path); return path
    except Exception as exc:
        partial.unlink(missing_ok=True); raise ArtifactError(f"Unable to write JSON {path}: {exc}") from exc


def write_csv(path: str | Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); partial = path.with_suffix(path.suffix + ".partial")
    try:
        with partial.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise"); writer.writeheader()
            for row in rows: writer.writerow({key: _json_value(row.get(key)) for key in columns})
            stream.flush(); os.fsync(stream.fileno())
        os.replace(partial, path); return path
    except Exception as exc:
        partial.unlink(missing_ok=True); raise ArtifactError(f"Unable to write CSV {path}: {exc}") from exc
