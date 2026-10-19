"""Date ladder, cut-off, partial fallback and provider failover: pure logic on stub candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import pytest
from shapely.geometry import box

from sentinel_patch.config.schema import DiscoveryConfig
from sentinel_patch.discovery.chain import ProviderChain
from sentinel_patch.discovery.provider import ProviderUnavailableError, SceneCandidate
from sentinel_patch.discovery.selection import CLEAN, PARTIAL, select_scene
from sentinel_patch.errors import SceneCoverageError

FLIGHT = date(2026, 10, 10)
CFG = DiscoveryConfig()
AOI = box(72.5, 33.6, 72.52, 33.62)


@dataclass
class Stats:
    masked_fraction: float
    clear_pixels: int = 5000


@dataclass
class Prepared:
    stats: Stats
    name: str


def cand(name, day_offset, provider="earth_search", **kwargs):
    when = datetime(2026, 10, 10 + day_offset, 5, 30, tzinfo=timezone.utc)
    return SceneCandidate(provider=provider, item_id=name, datetime_utc=when, processing_baseline="05.11",
                          asset_hrefs={}, **kwargs)


def run(candidates, fractions, cfg=CFG):
    """``fractions``: item_id -> masked fraction, or an exception to raise when the scene is read."""
    def prepare(candidate):
        value = fractions[candidate.item_id]
        if isinstance(value, Exception):
            raise value
        stats = value if isinstance(value, Stats) else Stats(value)
        return Prepared(stats, candidate.item_id)
    return select_scene(candidates, prepare, FLIGHT, cfg)


def by_id(selection):
    return {r["item_id"]: r for r in selection.records}


def test_nearest_acceptable_date_wins_and_farther_dates_are_never_read():
    read = []
    def prepare(candidate):
        read.append(candidate.item_id)
        return Prepared(Stats({"d1": 0.25, "d4": 0.08, "d6": 0.0}[candidate.item_id]), candidate.item_id)
    selection = select_scene([cand("d6", 6), cand("d4", 4), cand("d1", 1)], prepare, FLIGHT, CFG)

    assert selection.prepared.name == "d4" and selection.quality == CLEAN
    assert read == ["d1", "d4"]  # day 1 is too cloudy; day 4 passes; day 6 is never touched
    assert by_id(selection)["d1"]["reason_code"] == "AOI_TOO_CLOUDY" and by_id(selection)["d4"]["accepted"]


def test_tie_on_the_same_distance_goes_to_the_clearer_scene():
    selection = run([cand("before", -2), cand("after", 2)], {"before": 0.01, "after": 0.06})
    assert selection.prepared.name == "before"
    assert by_id(selection)["after"]["reason_code"] == "CLEARER_SCENE_ON_SAME_DATE"


def test_full_tie_goes_to_the_scene_after_the_flight():
    selection = run([cand("before", -2), cand("after", 2)], {"before": 0.05, "after": 0.05})
    assert selection.prepared.name == "after"


def test_full_tie_on_one_date_goes_to_provider_order():
    selection = run([cand("es", 1, "earth_search"), cand("pc", 1, "planetary_computer")], {"es": 0.0, "pc": 0.0})
    assert selection.prepared.name == "es"


def test_exactly_at_the_cutoff_is_accepted():
    assert run([cand("x", 1)], {"x": 0.10}).quality == CLEAN


def test_partial_fallback_takes_the_clearest_scene_over_the_cutoff():
    selection = run([cand("a", 1), cand("b", 3), cand("c", 5)], {"a": 0.28, "b": 0.22, "c": 0.45})
    assert (selection.prepared.name, selection.quality) == ("b", PARTIAL)
    assert by_id(selection)["b"]["accepted"] and by_id(selection)["b"]["reason_code"] == "PARTIAL_ACCEPTED"
    assert not by_id(selection)["a"]["accepted"]


def test_equal_partials_keep_the_nearer_date():
    assert run([cand("near", 1), cand("far", 4)], {"near": 0.2, "far": 0.2}).prepared.name == "near"


def test_a_clean_scene_beats_any_partial_even_when_farther():
    selection = run([cand("cloudy_near", 1), cand("clean_far", 9)], {"cloudy_near": 0.15, "clean_far": 0.02})
    assert (selection.prepared.name, selection.quality) == ("clean_far", CLEAN)


def test_best_scene_over_the_fallback_limit_is_no_scene():
    selection = run([cand("a", 1), cand("b", 2)], {"a": 0.45, "b": 0.60})
    assert selection.prepared is None and selection.quality is None and len(selection.records) == 2


def test_too_few_clear_pixels_is_rejected_whatever_the_fraction():
    selection = run([cand("tiny", 1)], {"tiny": Stats(0.0, clear_pixels=60)})
    assert selection.prepared is None and by_id(selection)["tiny"]["reason_code"] == "TOO_FEW_CLEAR_PIXELS"


def test_too_few_clear_pixels_also_blocks_a_partial():
    assert run([cand("tiny", 1)], {"tiny": Stats(0.2, clear_pixels=60)}).prepared is None


def test_unreadable_scene_is_recorded_and_the_search_continues():
    selection = run([cand("broken", 1), cand("fine", 2)], {"broken": SceneCoverageError("edge of tile"), "fine": 0.0})
    assert selection.prepared.name == "fine"
    assert by_id(selection)["broken"]["reason_code"] == "AOI_OUTSIDE_SCENE"


def test_evaluation_limit_is_recorded_not_silent():
    cfg = DiscoveryConfig(max_candidates_evaluated=2)
    selection = run([cand("a", 1), cand("b", 2), cand("c", 3)], {"a": 0.5, "b": 0.5, "c": 0.0}, cfg)
    assert selection.prepared is None and by_id(selection)["c"]["reason_code"] == "EVALUATION_LIMIT"


# -- provider chain -------------------------------------------------------------------------------

class StubProvider:
    def __init__(self, name, found=(), error=None, env=None):
        self.name, self.found, self.error, self.env = name, list(found), error, env or {}

    def search(self, aoi, start, end):
        if self.error:
            raise self.error
        return self.found

    def resolve_href(self, candidate, band):
        return f"{self.name}://{candidate.item_id}/{band}"

    def gdal_env(self):
        return self.env


def window(start=datetime(2026, 9, 25, tzinfo=timezone.utc), end=datetime(2026, 10, 25, tzinfo=timezone.utc)):
    return AOI, start, end


def test_failed_provider_is_recorded_and_the_next_one_is_used():
    chain = ProviderChain([StubProvider("a", error=ProviderUnavailableError("503")), StubProvider("b", [cand("x", 1, "b")])])
    found = chain.search(*window())
    assert [c.item_id for c in found] == ["x"]
    assert chain.failures == [{"provider": "a", "code": "PROVIDER_UNAVAILABLE", "message": "503"}]


def test_all_providers_failing_raises_with_every_reason():
    chain = ProviderChain([StubProvider("a", error=ProviderUnavailableError("down")),
                           StubProvider("b", error=ProviderUnavailableError("limit"))])
    with pytest.raises(ProviderUnavailableError, match="a: down.*b: limit"):
        chain.search(*window())


def test_a_provider_that_failed_at_startup_counts_in_the_error():
    chain = ProviderChain([], [{"provider": "cdse", "code": "PROVIDER_UNAVAILABLE", "message": "not configured"}])
    with pytest.raises(ProviderUnavailableError, match="cdse: not configured"):
        chain.search(*window())


def test_empty_result_from_a_healthy_provider_is_not_an_error():
    assert ProviderChain([StubProvider("a", [])]).search(*window()) == []


def test_same_acquisition_from_two_catalogues_is_kept_once_in_provider_order():
    first = cand("es-item", 1, "a", datatake_id="GS2A_X", mgrs_tile="43SBT")
    second = cand("pc-item", 1, "b", datatake_id="GS2A_X", mgrs_tile="43SBT")
    found = ProviderChain([StubProvider("a", [first]), StubProvider("b", [second])]).search(*window())
    assert [c.item_id for c in found] == ["es-item"]


def test_different_tiles_of_one_datatake_are_both_kept():
    t1 = cand("tile1", 1, "a", datatake_id="GS2A_X", mgrs_tile="43SBT")
    t2 = cand("tile2", 1, "b", datatake_id="GS2A_X", mgrs_tile="43SBU")
    assert len(ProviderChain([StubProvider("a", [t1]), StubProvider("b", [t2])]).search(*window())) == 2


def test_duplicate_prefers_the_copy_that_fully_covers_the_area():
    partial_cover = cand("edge", 1, "a", datatake_id="D", mgrs_tile="T", footprint_wgs84=box(72.51, 33.0, 73.0, 34.0))
    full_cover = cand("full", 1, "b", datatake_id="D", mgrs_tile="T", footprint_wgs84=box(72.0, 33.0, 73.0, 34.0))
    found = ProviderChain([StubProvider("a", [partial_cover]), StubProvider("b", [full_cover])]).search(*window())
    assert [c.item_id for c in found] == ["full"]


def test_hrefs_are_resolved_by_the_provider_that_found_the_scene():
    chain = ProviderChain([StubProvider("a"), StubProvider("b")])
    assert chain.resolve_href(cand("x", 1, "b"), "B04") == "b://x/B04"
    with pytest.raises(ProviderUnavailableError, match="not part of this chain"):
        chain.resolve_href(cand("x", 1, "zzz"), "B04")


def test_gdal_env_is_the_union_of_the_providers():
    chain = ProviderChain([StubProvider("a", env={"X": "1"}), StubProvider("b", env={"Y": "2"})])
    assert chain.gdal_env() == {"X": "1", "Y": "2"}
