"""Earth Search (Element 84, AWS): anonymous, harmonized ``sentinel-2-c1-l2a`` collection.

Asset keys and the declared scale 0.0001 / offset -0.1 were checked against a live response
(2025-10, processing baseline 05.11).
"""

from __future__ import annotations

from sentinel_patch.discovery.providers.stac_base import StacSentinel2Provider

ASSET_MAP = {
    "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1", "B06": "rededge2",
    "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B09": "nir09", "B11": "swir16", "B12": "swir22",
    "SCL": "scl", "MSK_CLDPRB": "cloud",
}


class EarthSearchProvider(StacSentinel2Provider):
    name = "earth_search"
    default_stac_url = "https://earth-search.aws.element84.com/v1"
    default_collection = "sentinel-2-c1-l2a"
    asset_map = ASSET_MAP

    @property
    def trust_declared_radiometry(self) -> bool:
        # Only the c1 collection declares the baseline offset; any other collection uses the baseline rule.
        return self.collection == "sentinel-2-c1-l2a"

    def gdal_env(self) -> dict[str, str]:
        return {**super().gdal_env(), "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff"}
