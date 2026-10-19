"""Shared behaviour of STAC-based Sentinel-2 L2A providers: item -> ``SceneCandidate``.

A concrete provider only supplies its asset-key map, its collection defaults and how a href becomes
GDAL-openable. Nothing outside ``discovery/providers/`` imports a catalogue client.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from sentinel_patch.config.schema import IoConfig, ProviderConfig
from sentinel_patch.discovery.http import HttpClient
from sentinel_patch.discovery.provider import ProviderUnavailableError, SceneCandidate
from sentinel_patch.discovery.stac import search_items
from sentinel_patch.io.gdal_env import base_gdal_env

logger = logging.getLogger(__name__)
REQUIRED_ASSETS = ("SCL",)


def parse_datetime(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class StacSentinel2Provider:
    name = "stac"
    default_stac_url = ""
    default_collection = ""
    asset_map: Mapping[str, str] = {}  # our band name -> the catalogue's asset key

    def __init__(self, cfg: ProviderConfig, io: IoConfig, client: HttpClient | None = None):
        self.stac_url = cfg.stac_url or self.default_stac_url
        self.collection = cfg.collection or self.default_collection
        self.client = client or HttpClient(timeout_s=io.http_timeout_s, retries=io.http_retries)
        self._io = io

    @property
    def trust_declared_radiometry(self) -> bool:
        """True only where the item's raster:bands scale/offset is known to include the baseline offset."""
        return False

    @staticmethod
    def _epsg(props: Mapping[str, Any]) -> int | None:
        if props.get("proj:epsg"):
            return int(props["proj:epsg"])
        code = str(props.get("proj:code") or "")
        return int(code.split(":")[1]) if code.upper().startswith("EPSG:") else None

    @staticmethod
    def _mgrs(props: Mapping[str, Any]) -> str | None:
        tile = props.get("s2:mgrs_tile") or str(props.get("grid:code") or "").removeprefix("MGRS-")
        return tile or None

    @staticmethod
    def _relative_orbit(props: Mapping[str, Any]) -> int | None:
        if props.get("sat:relative_orbit") is not None:
            return int(props["sat:relative_orbit"])
        found = re.search(r"_R(\d{3})_", str(props.get("s2:product_uri") or ""))
        return int(found.group(1)) if found else None

    def _scale_offset(self, assets: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
        if not self.trust_declared_radiometry:
            return {}  # the processing baseline decides (radiometry.resolve_radiometry)
        declared = {}
        for band, key in self.asset_map.items():
            raster = (assets.get(key) or {}).get("raster:bands") or [{}]
            scale, offset = raster[0].get("scale"), raster[0].get("offset")
            if scale is not None and offset is not None and band != "SCL":
                declared[band] = (float(scale), float(offset))
        return declared

    def to_candidate(self, item: Mapping[str, Any]) -> SceneCandidate | None:
        props, assets = item.get("properties") or {}, item.get("assets") or {}
        try:
            when = parse_datetime(props["datetime"])
            footprint: BaseGeometry | None = shape(item["geometry"]) if item.get("geometry") else None
            hrefs = {band: assets[key]["href"] for band, key in self.asset_map.items() if key in assets}
            if any(band not in hrefs for band in REQUIRED_ASSETS):
                logger.info("%s item %s skipped: no scene-classification asset.", self.name, item.get("id"))
                return None
            return SceneCandidate(
                provider=self.name, item_id=str(item["id"]), datetime_utc=when,
                processing_baseline=props.get("s2:processing_baseline"), asset_hrefs=hrefs,
                mgrs_tile=self._mgrs(props), epsg=self._epsg(props), scene_cloud_cover=props.get("eo:cloud_cover"),
                relative_orbit=self._relative_orbit(props), scale_offset=self._scale_offset(assets),
                footprint_wgs84=footprint, datatake_id=props.get("s2:datatake_id"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("%s item %s skipped: unreadable (%s).", self.name, item.get("id"), exc)
            return None

    def search(self, aoi_wgs84: BaseGeometry, start: datetime, end: datetime) -> list[SceneCandidate]:
        items = search_items(self.client, self.stac_url, self.collection, aoi_wgs84, start, end)
        return [c for c in (self.to_candidate(item) for item in items) if c is not None]

    def resolve_href(self, candidate: SceneCandidate, band: str) -> str:
        try:
            return candidate.asset_hrefs[band]
        except KeyError as exc:
            raise ProviderUnavailableError(f"Scene {candidate.item_id} has no band {band}.") from exc

    def gdal_env(self) -> dict[str, str]:
        return base_gdal_env(self._io)
