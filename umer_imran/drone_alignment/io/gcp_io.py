"""Manual ground-control-point file readers.

Produces plain :class:`ManualGcp` records; interpreting their coordinate
units (map vs. pixel, and which CRS/raster they belong to) is the caller's
job, since a QGIS Georeferencer file can legitimately mix units between its
two point sets (reference points are always map coordinates; source points
follow ``source_units``).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ManualGcp:
    gcp_id: str
    rgb: tuple[float, float]
    ms: tuple[float, float]
    role: Literal["control", "check"] = "control"


def read_gcp_csv(path) -> list[ManualGcp]:
    """Parse a GCP CSV. Header required: id,rgb_x,rgb_y,ms_x,ms_y[,role]."""
    path_obj = Path(path)
    required = {"id", "rgb_x", "rgb_y", "ms_x", "ms_y"}
    gcps: list[ManualGcp] = []
    with open(path_obj, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"GCP CSV {path_obj} is missing required column(s): {sorted(missing)}")
        for row_num, row in enumerate(reader, start=2):
            role = (row.get("role") or "control").strip().lower() or "control"
            if role not in ("control", "check"):
                raise ValueError(f"{path_obj}:{row_num}: unknown role {role!r} (expected 'control' or 'check')")
            try:
                gcps.append(ManualGcp(
                    gcp_id=str(row["id"]).strip(),
                    rgb=(float(row["rgb_x"]), float(row["rgb_y"])),
                    ms=(float(row["ms_x"]), float(row["ms_y"])),
                    role=role,
                ))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path_obj}:{row_num}: invalid GCP row: {exc}") from exc
    if not gcps:
        raise ValueError(f"GCP CSV {path_obj} contains no data rows.")
    return gcps


def read_qgis_points(path, source_units: Literal["pixel", "map"]) -> list[ManualGcp]:
    """Parse a QGIS Georeferencer ``.points`` file.

    Columns: ``mapX,mapY,sourceX,sourceY,enable[,dX,dY,residual]``. ``mapX``/
    ``mapY`` are the RGB (reference) map coordinates; ``sourceX``/``sourceY``
    are the MS (source) coordinates in ``source_units``. Rows with
    ``enable == 0`` are skipped. When ``source_units == "pixel"``, QGIS
    stores the source row as a negative Y, which is negated back here.
    """
    path_obj = Path(path)
    with open(path_obj, "r", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"QGIS points file {path_obj} is empty.")
    if lines[0].startswith("#"):
        header = [h.strip() for h in lines[0].lstrip("#").split(",")]
        data_lines = lines[1:]
    else:
        header = ["mapX", "mapY", "sourceX", "sourceY", "enable"]
        data_lines = lines
    required = {"mapX", "mapY", "sourceX", "sourceY"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"QGIS points file {path_obj} is missing required column(s): {sorted(missing)}")
    index = {name: i for i, name in enumerate(header)}
    enable_idx = index.get("enable")

    gcps: list[ManualGcp] = []
    for row_num, line in enumerate(data_lines, start=2):
        fields = [f.strip() for f in line.split(",")]
        try:
            if enable_idx is not None and enable_idx < len(fields) and float(fields[enable_idx]) == 0.0:
                continue
            map_x = float(fields[index["mapX"]])
            map_y = float(fields[index["mapY"]])
            source_x = float(fields[index["sourceX"]])
            source_y = float(fields[index["sourceY"]])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{path_obj}:{row_num}: invalid QGIS points row: {exc}") from exc
        if source_units == "pixel":
            source_y = -source_y
        gcps.append(ManualGcp(gcp_id=f"P{row_num - 1}", rgb=(map_x, map_y), ms=(source_x, source_y), role="control"))
    if not gcps:
        raise ValueError(f"QGIS points file {path_obj} has no enabled rows.")
    return gcps
