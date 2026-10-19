"""Flight-date estimate from folder and file names (blueprint 6.9.1)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from sentinel_patch.date_from_path import estimate_flight_date

TODAY = date(2026, 10, 18)


def estimate(*paths, **kwargs):
    return estimate_flight_date([Path(p) for p in paths], today=TODAY, **kwargs)


@pytest.mark.parametrize("path, expected", [
    ("D:/Flights/Okara/2026-10-10_Field7/odm_orthophoto/rgb.tif", date(2026, 10, 10)),
    ("D:/f/2026_10_10/rgb.tif", date(2026, 10, 10)),
    ("D:/f/2026.10.10/rgb.tif", date(2026, 10, 10)),
    ("D:/f/DJI_0042/IMG_20261010_0042.tif", date(2026, 10, 10)),
    ("D:/f/10Oct2026/rgb.tif", date(2026, 10, 10)),
    ("D:/f/10-oct-26/rgb.tif", date(2026, 10, 10)),
    ("D:/f/Oct 10 2026/rgb.tif", date(2026, 10, 10)),
    ("D:/f/13-10-2026/rgb.tif", date(2026, 10, 13)),   # 13 cannot be a month: unambiguous
    ("D:/f/10-10-2026/rgb.tif", date(2026, 10, 10)),   # both readings equal: unambiguous
])
def test_recognised_patterns(path, expected):
    found = estimate(path)
    assert found.status == "FOUND" and found.best.value == expected and not found.best.ambiguous


@pytest.mark.parametrize("path", [
    "D:/f/DJI_0042/rgb.tif", "D:/f/v2.10/rgb.tif", "D:/f/101026/rgb.tif", "D:/f/261010/rgb.tif",
    "D:/f/Oct2026/rgb.tif", "D:/f/2026-13-45/rgb.tif", "D:/f/2026-02-30/rgb.tif", "D:/f/Field7/rgb.tif",
])
def test_things_that_are_not_dates_are_ignored(path):
    assert estimate(path).status == "NONE"


def test_numeric_date_with_year_last_is_ambiguous_and_follows_the_configured_order():
    dmy = estimate("D:/f/10-03-2026/rgb.tif", date_order="dmy").best
    mdy = estimate("D:/f/10-03-2026/rgb.tif", date_order="mdy").best
    assert dmy.value == date(2026, 3, 10) and mdy.value == date(2026, 10, 3)
    assert dmy.ambiguous and set(dmy.ambiguous_readings) == {date(2026, 3, 10), date(2026, 10, 3)}
    assert not estimate("D:/f/10-03-2026/rgb.tif").usable_unattended


def test_dates_outside_the_plausible_range_are_dropped():
    assert estimate("D:/f/2012-05-05/rgb.tif").status == "NONE"        # before Sentinel-2A
    assert estimate("D:/f/2027-01-01/rgb.tif").status == "NONE"        # in the future


def test_a_date_after_the_orthophoto_was_processed_is_impossible():
    assert estimate("D:/f/2026-10-10/rgb.tif", upper_bound=date(2026, 10, 9)).status == "NONE"
    assert estimate("D:/f/2026-10-10/rgb.tif", upper_bound=date(2026, 10, 10)).status == "FOUND"


def test_same_date_in_both_paths_merges_and_is_trusted_more():
    found = estimate("D:/a/2026-10-10/rgb.tif", "D:/b/2026-10-10/ms.tif")
    assert found.status == "FOUND" and found.best.sources == (0, 1) and found.usable_unattended


def test_conflicting_dates_are_ranked_and_never_used_unattended():
    found = estimate("D:/a/2026-10-10/rgb.tif", "D:/b/2026-10-12/ms.tif")
    assert found.status == "MULTIPLE_DATES" and len(found.candidates) == 2 and not found.usable_unattended


def test_a_closer_component_ranks_higher():
    found = estimate("D:/2026-09-01/2026-10-10_run/rgb.tif")
    assert [c.value for c in found.candidates] == [date(2026, 10, 10), date(2026, 9, 1)]


def test_scan_depth_limits_how_far_up_the_path_is_read():
    deep = "D:/2026-10-10/a/b/c/d/rgb.tif"
    assert estimate(deep, depth=3).status == "NONE"
    assert estimate(deep, depth=5).status == "FOUND"


def test_one_match_per_characters_so_an_iso_date_is_not_read_twice():
    found = estimate("D:/f/2026-10-10/rgb.tif")
    assert len(found.candidates) == 1
