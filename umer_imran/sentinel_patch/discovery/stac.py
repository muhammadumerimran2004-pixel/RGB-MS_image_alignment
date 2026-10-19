"""Minimal STAC API item search (POST /search with paging); no catalogue-specific knowledge."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from shapely.geometry.base import BaseGeometry

from sentinel_patch.discovery.http import HttpClient
from sentinel_patch.errors import ProviderUnavailableError


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def search_items(
    client: HttpClient, stac_url: str, collection: str, aoi_wgs84: BaseGeometry, start: datetime, end: datetime,
    *, page_size: int = 100, max_pages: int = 10,
) -> list[dict[str, Any]]:
    """All items of ``collection`` intersecting the AOI bounding box within ``[start, end]`` (UTC)."""
    search_url = stac_url.rstrip("/") + "/search"
    body: dict[str, Any] | None = {
        "collections": [collection], "bbox": [float(v) for v in aoi_wgs84.bounds],
        "datetime": f"{_iso(start)}/{_iso(end)}", "limit": page_size,
    }
    url, method = search_url, "POST"
    items: list[dict[str, Any]] = []
    for _ in range(max_pages):
        page = client.request_json(method, url, json_body=body)
        if not isinstance(page, dict) or not isinstance(page.get("features"), list):
            raise ProviderUnavailableError(f"{search_url} returned something that is not an item collection.")
        items.extend(page["features"])
        link = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if link is None or not page["features"]:
            return items
        url, method = link.get("href", search_url), link.get("method", "GET").upper()
        next_body = link.get("body")
        body = {**(body or {}), **next_body} if link.get("merge") and next_body else next_body
    raise ProviderUnavailableError(f"{search_url} still had more pages after {max_pages}; narrow the search.")
