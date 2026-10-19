"""An ordered list of providers behind the single ``SceneProvider`` interface (blueprint 6.2.2).

``search`` asks every provider (a catalogue sometimes misses a granule another one has), records the
ones that fail, and de-duplicates the same acquisition delivered by several catalogues. Only when
*all* providers fail does it raise, so a single outage never stops a run.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

from shapely.geometry.base import BaseGeometry

from sentinel_patch.discovery.provider import ProviderUnavailableError, SceneCandidate, SceneProvider

logger = logging.getLogger(__name__)


def _dedup_key(candidate: SceneCandidate) -> tuple:
    # The same granule carries different timestamps in different catalogues, but the same datatake id.
    # Different MGRS tiles of one datatake are different granules, so the tile belongs in the key.
    if candidate.datatake_id:
        return ("datatake", candidate.datatake_id, candidate.mgrs_tile)
    return ("item", candidate.item_id)


def _contains(candidate: SceneCandidate, aoi: BaseGeometry) -> bool:
    return candidate.footprint_wgs84 is not None and candidate.footprint_wgs84.contains(aoi)


class ProviderChain:
    name = "chain"

    def __init__(self, providers: Sequence[SceneProvider], startup_failures: Sequence[dict] = ()):
        self.providers = list(providers)
        self.failures: list[dict] = list(startup_failures)
        self._by_name = {provider.name: provider for provider in self.providers}

    def search(self, aoi_wgs84: BaseGeometry, start: datetime, end: datetime) -> list[SceneCandidate]:
        kept: dict[tuple, SceneCandidate] = {}
        succeeded = 0
        for provider in self.providers:
            try:
                found = provider.search(aoi_wgs84, start, end)
            except ProviderUnavailableError as exc:
                logger.warning("Provider %s unavailable: %s", provider.name, exc)
                self.failures.append({"provider": provider.name, "code": exc.code, "message": str(exc)})
                continue
            succeeded += 1
            for candidate in found:
                key = _dedup_key(candidate)
                current = kept.get(key)
                # Provider order wins, unless only the later one fully covers the area of interest.
                if current is None or (not _contains(current, aoi_wgs84) and _contains(candidate, aoi_wgs84)):
                    kept[key] = candidate
        if not succeeded:
            reasons = "; ".join(f"{f['provider']}: {f['message']}" for f in self.failures) or "no provider enabled"
            raise ProviderUnavailableError(f"No scene provider could be reached ({reasons}).")
        return list(kept.values())

    def resolve_href(self, candidate: SceneCandidate, band: str) -> str:
        try:
            return self._by_name[candidate.provider].resolve_href(candidate, band)
        except KeyError as exc:
            raise ProviderUnavailableError(f"Provider {candidate.provider} is not part of this chain.") from exc

    def gdal_env(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for provider in self.providers:
            merged.update(provider.gdal_env())
        return merged
