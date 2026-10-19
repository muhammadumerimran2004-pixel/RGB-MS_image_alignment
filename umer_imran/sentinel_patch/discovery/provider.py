"""The contract between scene catalogues and the rest of the package.

Everything outside ``discovery/providers/`` sees only ``SceneCandidate`` and
GDAL-openable hrefs. Nothing else may import a catalogue client or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol

from shapely.geometry.base import BaseGeometry

from sentinel_patch.errors import ProviderUnavailableError

__all__ = ["ProviderUnavailableError", "SceneCandidate", "SceneProvider"]


@dataclass(frozen=True)
class SceneCandidate:
    provider: str
    item_id: str
    datetime_utc: datetime
    processing_baseline: str | None
    asset_hrefs: Mapping[str, str]  # our band names ("B02", "SCL", ...) -> href
    mgrs_tile: str | None = None
    epsg: int | None = None
    scene_cloud_cover: float | None = None
    relative_orbit: int | None = None
    scale_offset: Mapping[str, tuple[float, float]] = field(default_factory=dict)  # declared per band
    footprint_wgs84: BaseGeometry | None = None
    datatake_id: str | None = None  # identical across catalogues for one acquisition; the de-duplication key


class SceneProvider(Protocol):
    name: str

    def search(self, aoi_wgs84: BaseGeometry, start: datetime, end: datetime) -> list[SceneCandidate]: ...

    def resolve_href(self, candidate: SceneCandidate, band: str) -> str:
        """Return a GDAL-openable location for one band (signing / building /vsis3/ paths here)."""
        ...

    def gdal_env(self) -> dict[str, str]: ...
