"""Manual live checks against the real free catalogues. Deselected by default; run with ``-m network``.

They prove what recorded responses cannot: that anonymous search works, that hrefs open, that
Planetary Computer signing is accepted, and that the GDAL environment reads cloud-optimised GeoTIFFs.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from sentinel_patch.config.schema import parse_config
from sentinel_patch.discovery.providers.earth_search import EarthSearchProvider
from sentinel_patch.discovery.providers.planetary_computer import PlanetaryComputerProvider
from sentinel_patch.discovery.registry import build_chain
from sentinel_patch.io.window_reader import plan_window_10m, read_band_window

import rasterio

pytestmark = pytest.mark.network

AOI = box(72.50, 33.60, 72.51, 33.61)  # about 1 km, Pakistan; Sentinel-2 tile 43SBT
START, END = datetime(2025, 10, 1, tzinfo=timezone.utc), datetime(2025, 10, 15, 23, 59, tzinfo=timezone.utc)


@pytest.mark.parametrize("provider_class", [EarthSearchProvider, PlanetaryComputerProvider])
def test_search_and_windowed_read(provider_class):
    cfg = parse_config({})
    provider_cfg = next(p for p in cfg.providers if p.name == provider_class.name)
    provider = provider_class(provider_cfg, cfg.io)
    candidates = provider.search(AOI, START, END)
    assert candidates, "the catalogue returned no scene for a week of October over Pakistan"

    candidate = min(candidates, key=lambda c: c.scene_cloud_cover if c.scene_cloud_cover is not None else 100)
    with rasterio.Env(**provider.gdal_env()):
        href = provider.resolve_href(candidate, "B08")
        epsg_aoi = transform(Transformer.from_crs(4326, candidate.epsg, always_xy=True).transform, AOI)
        _, bounds = plan_window_10m(href, epsg_aoi.bounds, 200.0)
        nir = read_band_window(href, bounds, "B08")
        scl = read_band_window(provider.resolve_href(candidate, "SCL"), bounds, "SCL")

    assert nir.data.shape[0] >= 100 and nir.data.dtype == np.uint16 and (nir.data > 0).mean() > 0.9
    assert scl.data.shape == (nir.data.shape[0] // 2, nir.data.shape[1] // 2)


def test_default_chain_deduplicates_the_same_acquisitions():
    chain = build_chain(parse_config({}))
    found = chain.search(AOI, START, END)
    assert chain.failures == [] and found
    keys = [(c.datatake_id, c.mgrs_tile) for c in found]
    assert len(keys) == len(set(keys))
    assert {c.provider for c in found} == {"earth_search"}  # provider order wins every duplicate
