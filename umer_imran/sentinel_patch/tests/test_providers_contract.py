"""Every catalogue provider must satisfy the same contract, checked on recorded live responses (no network)."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import pytest
from shapely.geometry import box

from sentinel_patch.config.schema import IoConfig, ProviderConfig, parse_config
from sentinel_patch.discovery.http import HttpClient
from sentinel_patch.discovery.providers.cdse import CdseProvider
from sentinel_patch.discovery.providers.earth_search import EarthSearchProvider
from sentinel_patch.discovery.providers.planetary_computer import PlanetaryComputerProvider
from sentinel_patch.discovery.registry import build_chain
from sentinel_patch.errors import ProviderUnavailableError
from sentinel_patch.io.radiometry import resolve_radiometry

from .fakes import FakeResponse, FakeSession, load_fixture, pages

AOI = box(72.5, 33.6, 72.52, 33.62)
START, END = datetime(2025, 10, 1, tzinfo=timezone.utc), datetime(2025, 10, 15, tzinfo=timezone.utc)
IO = IoConfig()

CASES = {
    "earth_search": (EarthSearchProvider, "earth_search_search.json", "S2A_T43SBT_20251013T055251_L2A"),
    "planetary_computer": (PlanetaryComputerProvider, "planetary_computer_search.json",
                           "S2A_MSIL2A_20251013T055251_R048_T43SBT_20251013T081813"),
}
TOKEN = {"token": "sv=2025&sig=abc", "msft:expiry": "2026-10-18T12:59:37Z"}
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"


def make(name, fixture=None):
    cls, fixture_name, _ = CASES[name]
    body = {**(fixture or load_fixture(fixture_name)), "links": []}  # a single page
    session = FakeSession(pages(body))
    return cls(ProviderConfig(name=name), IO, HttpClient(retries=0, session=session)), session


@pytest.fixture(params=sorted(CASES))
def provider_name(request):
    return request.param


def _params(candidate, band="B04"):
    scale, offset = candidate.scale_offset.get(band) or (None, None)
    return resolve_radiometry(scale=scale, offset=offset, processing_baseline=candidate.processing_baseline)


def test_search_returns_complete_candidates(provider_name):
    candidates = make(provider_name)[0].search(AOI, START, END)
    first = next(c for c in candidates if c.item_id == CASES[provider_name][2])

    assert len(candidates) == 2 and first.provider == provider_name
    assert first.datetime_utc.tzinfo is not None and first.datetime_utc.date().isoformat() == "2025-10-13"
    assert first.processing_baseline == "05.11" and first.epsg == 32643 and first.mgrs_tile == "43SBT"
    assert first.relative_orbit == 48 and first.datatake_id == "GS2A_20251013T055251_053842_N05.11"
    assert first.footprint_wgs84 is not None and first.footprint_wgs84.contains(AOI)
    for band in ("B02", "B03", "B04", "B08", "B05", "B8A", "SCL"):
        assert first.asset_hrefs[band].startswith("https://") and first.asset_hrefs[band].split("?")[0].endswith(".tif")


def test_band_map_points_at_the_right_files(provider_name):
    first = make(provider_name)[0].search(AOI, START, END)[0]
    for band in ("B03", "B04", "B08", "SCL"):
        assert band in first.asset_hrefs[band]


def test_radiometry_follows_the_processing_baseline(provider_name):
    """PB 05.11 carries the -0.1 offset; a PB 03.xx item of the same provider must not."""
    new = make(provider_name)[0].search(AOI, START, END)[0]
    assert (_params(new).scale, _params(new).offset) == (pytest.approx(0.0001), pytest.approx(-0.1))

    old_fixture = copy.deepcopy(load_fixture(CASES[provider_name][1]))
    for item in old_fixture["features"]:
        item["properties"]["s2:processing_baseline"] = "03.01"
        for asset in item["assets"].values():  # an honest catalogue declares offset 0 for PB 03.xx
            for raster in asset.get("raster:bands", []):
                if "offset" in raster:
                    raster["offset"] = 0.0
    old = make(provider_name, old_fixture)[0].search(AOI, START, END)[0]
    assert _params(old).offset == 0.0


def test_planetary_computer_never_trusts_declared_radiometry():
    assert all(not c.scale_offset for c in make("planetary_computer")[0].search(AOI, START, END))


def test_earth_search_declares_scale_and_offset_for_reflectance_bands_only():
    first = make("earth_search")[0].search(AOI, START, END)[0]
    assert first.scale_offset["B04"] == (0.0001, -0.1) and "SCL" not in first.scale_offset


def test_item_without_a_scene_classification_asset_is_skipped(provider_name):
    body = copy.deepcopy(load_fixture(CASES[provider_name][1]))
    for key in ("scl", "SCL"):
        body["features"][0]["assets"].pop(key, None)
    assert len(make(provider_name, body)[0].search(AOI, START, END)) == 1


def test_planetary_computer_signs_hrefs_and_refreshes_an_expiring_token():
    now = [datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc)]
    session = FakeSession(lambda m, u, b: FakeResponse(200, TOKEN))
    provider = PlanetaryComputerProvider(ProviderConfig(name="planetary_computer"), IO,
                                         HttpClient(retries=0, session=session), now=lambda: now[0])
    candidate = make("planetary_computer")[0].search(AOI, START, END)[0]

    assert provider.resolve_href(candidate, "B04") == candidate.asset_hrefs["B04"] + "?sv=2025&sig=abc"
    provider.resolve_href(candidate, "B08")
    assert [call[1] for call in session.calls] == [TOKEN_URL]  # one token serves both bands

    now[0] = datetime(2026, 10, 18, 12, 56, tzinfo=timezone.utc)  # inside the 5-minute margin before expiry
    provider.resolve_href(candidate, "B04")
    assert len(session.calls) == 2


def test_an_unusable_token_response_is_a_provider_failure():
    session = FakeSession(lambda m, u, b: FakeResponse(200, {"oops": 1}))
    provider = PlanetaryComputerProvider(ProviderConfig(name="planetary_computer"), IO, HttpClient(retries=0, session=session))
    candidate = make("planetary_computer")[0].search(AOI, START, END)[0]
    with pytest.raises(ProviderUnavailableError, match="SAS token"):
        provider.resolve_href(candidate, "B04")


def test_cdse_without_credentials_cannot_start(monkeypatch):
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_KEY", raising=False)
    with pytest.raises(ProviderUnavailableError, match="not configured"):
        CdseProvider(ProviderConfig(name="cdse"), IO)


def test_cdse_with_credentials_still_reports_not_implemented(monkeypatch):
    monkeypatch.setenv("CDSE_S3_ACCESS_KEY", "k")
    monkeypatch.setenv("CDSE_S3_SECRET_KEY", "s")
    with pytest.raises(ProviderUnavailableError, match="not implemented"):
        CdseProvider(ProviderConfig(name="cdse"), IO).search(AOI, START, END)


def test_default_chain_has_the_two_free_providers_in_order_and_skips_disabled_cdse():
    chain = build_chain(parse_config({}))
    assert [p.name for p in chain.providers] == ["earth_search", "planetary_computer"] and chain.failures == []


def test_enabled_cdse_without_credentials_is_recorded_not_fatal(monkeypatch):
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    chain = build_chain(parse_config({"providers": [{"name": "earth_search"}, {"name": "cdse"}]}))
    assert [p.name for p in chain.providers] == ["earth_search"]
    assert chain.failures[0]["provider"] == "cdse" and chain.failures[0]["code"] == "PROVIDER_UNAVAILABLE"
