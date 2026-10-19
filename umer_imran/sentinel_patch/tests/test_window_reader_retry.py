"""A flaky remote read (truncated HTTP transfer seen on the real Earth Search files) is retried, then contained."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from sentinel_patch.errors import ProviderUnavailableError
from sentinel_patch.io import window_reader


@pytest.fixture
def tif(tmp_path):
    path = tmp_path / "B08.tif"
    with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=1, dtype="uint16",
                       crs=CRS.from_epsg(32642), transform=Affine(10, 0, 500000, 0, -10, 3500000)) as dst:
        dst.write(np.arange(1600, dtype="uint16").reshape(40, 40), 1)
    return str(path)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(window_reader, "READ_RETRY_DELAY_S", 0.0)


def flaky(monkeypatch, failures):
    real, state = rasterio.open, {"left": failures, "calls": 0}

    def opener(*args, **kwargs):
        state["calls"] += 1
        if state["left"] > 0:
            state["left"] -= 1
            raise rasterio.errors.RasterioIOError("TIFFFillTile: Read error, got 542803 bytes, expected 1242551")
        return real(*args, **kwargs)
    monkeypatch.setattr(window_reader.rasterio, "open", opener)
    return state


BOUNDS = (500100.0, 3499700.0, 500300.0, 3499900.0)


def test_a_transient_failure_is_retried_and_the_read_succeeds(tif, monkeypatch):
    state = flaky(monkeypatch, failures=2)
    window = window_reader.read_band_window(tif, BOUNDS, "B08")
    assert window.data.shape == (20, 20) and state["calls"] == 3


def test_a_persistent_failure_becomes_a_provider_error_not_a_crash(tif, monkeypatch):
    state = flaky(monkeypatch, failures=99)
    with pytest.raises(ProviderUnavailableError, match="after 3 attempts.*542803"):
        window_reader.read_band_window(tif, BOUNDS, "B08")
    assert state["calls"] == 3


def test_planning_the_window_is_retried_too(tif, monkeypatch):
    state = flaky(monkeypatch, failures=1)
    window, bounds = window_reader.plan_window_10m(tif, BOUNDS, 0.0)
    assert window.width == 20 and state["calls"] == 2


def test_coverage_problems_are_not_retried(tif, monkeypatch):
    state = flaky(monkeypatch, failures=0)
    from sentinel_patch.errors import SceneCoverageError
    with pytest.raises(SceneCoverageError):
        window_reader.plan_window_10m(tif, (400000.0, 3499700.0, 400200.0, 3499900.0), 0.0)
    assert state["calls"] == 1
