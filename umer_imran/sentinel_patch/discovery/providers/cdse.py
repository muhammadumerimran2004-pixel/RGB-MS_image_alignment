"""Copernicus Data Space Ecosystem: a placeholder so the provider slot, config and credential loading exist.

Credentials come from the environment only (``CDSE_S3_ACCESS_KEY`` / ``CDSE_S3_SECRET_KEY``), never from
config files. Searching is not implemented yet; the chain records that and moves on.
"""

from __future__ import annotations

import os
from datetime import datetime

from shapely.geometry.base import BaseGeometry

from sentinel_patch.config.schema import IoConfig, ProviderConfig
from sentinel_patch.discovery.provider import ProviderUnavailableError, SceneCandidate

NOT_IMPLEMENTED = "The CDSE provider is not implemented in this version."


class CdseProvider:
    name = "cdse"

    def __init__(self, cfg: ProviderConfig, io: IoConfig, client=None):
        missing = [v for v in ("CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY") if not os.environ.get(v)]
        if missing:
            raise ProviderUnavailableError(f"CDSE is not configured: set {', '.join(missing)}.")

    def search(self, aoi_wgs84: BaseGeometry, start: datetime, end: datetime) -> list[SceneCandidate]:
        raise ProviderUnavailableError(NOT_IMPLEMENTED)

    def resolve_href(self, candidate: SceneCandidate, band: str) -> str:
        raise ProviderUnavailableError(NOT_IMPLEMENTED)

    def gdal_env(self) -> dict[str, str]:
        return {}
