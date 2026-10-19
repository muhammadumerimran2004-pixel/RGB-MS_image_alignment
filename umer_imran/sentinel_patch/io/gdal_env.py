"""GDAL options for reading Cloud-Optimized GeoTIFFs over HTTP (blueprint 6.3)."""

from __future__ import annotations

from sentinel_patch.config.schema import IoConfig


def base_gdal_env(io: IoConfig) -> dict[str, str]:
    return {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": str(io.http_retries),
        "GDAL_HTTP_RETRY_DELAY": "2",
        "GDAL_HTTP_TIMEOUT": str(io.http_timeout_s),
        "VSI_CACHE": "TRUE",
    }
