"""Offline provider: scenes already on disk (tests, air-gapped runs, manually downloaded scenes).

Layout: a folder holding ``scene.json`` and one single-band GeoTIFF per band
(``B02.tif`` or ``B2.tif`` style, ``SCL.tif``). If the folder itself has no
``scene.json``, each sub-folder that has one is treated as a separate scene.

``scene.json``::

    {"item_id": "...", "datetime": "2026-10-12T05:30:00Z", "processing_baseline": "05.11",
     "mgrs_tile": "42RXV", "scene_cloud_cover": 3.1, "relative_orbit": 120,
     "scale_offset": {"B04": [0.0001, -0.1]}}      # optional; declared metadata beats the baseline rule
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from pyproj import Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry

from sentinel_patch.discovery.provider import ProviderUnavailableError, SceneCandidate
from sentinel_patch.errors import ConfigurationError

SCENE_FILE = "scene.json"
KNOWN_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL", "MSK_CLDPRB",
)


def _filename_options(band: str) -> list[str]:
    options = [f"{band}.tif"]
    short = re.fullmatch(r"B0(\d)", band)
    if short:
        options.append(f"B{short.group(1)}.tif")  # crop_health_sentinel naming: B3.tif, B4.tif, ...
    return options


def _parse_datetime(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class LocalFolderProvider:
    name = "local_folder"

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise ConfigurationError(f"local_folder does not exist: {self.folder}")

    def _scene_dirs(self) -> list[Path]:
        if (self.folder / SCENE_FILE).is_file():
            return [self.folder]
        return sorted(path for path in self.folder.iterdir() if (path / SCENE_FILE).is_file())

    def _candidate(self, scene_dir: Path) -> SceneCandidate:
        try:
            meta = json.loads((scene_dir / SCENE_FILE).read_text(encoding="utf-8"))
            hrefs = {
                band: str(scene_dir / name)
                for band in KNOWN_BANDS for name in _filename_options(band) if (scene_dir / name).is_file()
            }
            if not hrefs:
                raise ProviderUnavailableError(f"No band files found in {scene_dir}.")
            with rasterio.open(next(iter(hrefs.values()))) as src:
                to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                footprint = transform_geometry(to_wgs84.transform, box(*src.bounds))
                epsg = src.crs.to_epsg()
            return SceneCandidate(
                provider=self.name, item_id=str(meta["item_id"]), datetime_utc=_parse_datetime(meta["datetime"]),
                processing_baseline=meta.get("processing_baseline"), asset_hrefs=hrefs,
                mgrs_tile=meta.get("mgrs_tile"), epsg=epsg, scene_cloud_cover=meta.get("scene_cloud_cover"),
                relative_orbit=meta.get("relative_orbit"), datatake_id=meta.get("datatake_id"),
                scale_offset={band: (float(v[0]), float(v[1])) for band, v in meta.get("scale_offset", {}).items()},
                footprint_wgs84=footprint,
            )
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            raise ProviderUnavailableError(f"Unreadable scene folder {scene_dir}: {exc}") from exc

    def search(self, aoi_wgs84: BaseGeometry, start: datetime, end: datetime) -> list[SceneCandidate]:
        found = []
        for scene_dir in self._scene_dirs():
            candidate = self._candidate(scene_dir)
            if start <= candidate.datetime_utc <= end and candidate.footprint_wgs84.intersects(aoi_wgs84):
                found.append(candidate)
        return found

    def resolve_href(self, candidate: SceneCandidate, band: str) -> str:
        try:
            return candidate.asset_hrefs[band]
        except KeyError as exc:
            raise ProviderUnavailableError(f"Scene {candidate.item_id} has no band {band}.") from exc

    def gdal_env(self) -> dict[str, str]:
        return {}
