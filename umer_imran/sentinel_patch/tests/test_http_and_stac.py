"""Retry behaviour and STAC paging, without the network."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from shapely.geometry import box

from sentinel_patch.discovery.http import HttpClient
from sentinel_patch.discovery.stac import search_items
from sentinel_patch.errors import ProviderUnavailableError

from .fakes import FakeResponse, FakeSession, connection_error

START, END = datetime(2025, 10, 1, tzinfo=timezone.utc), datetime(2025, 10, 15, 23, 59, 59, tzinfo=timezone.utc)


def _client(script, retries=3):
    sleeps = []
    session = FakeSession(script)
    return HttpClient(retries=retries, backoff_s=1.0, session=session, sleep=sleeps.append), session, sleeps


def test_retries_a_server_error_then_succeeds_with_exponential_backoff():
    client, session, sleeps = _client([FakeResponse(503), connection_error(), FakeResponse(200, {"ok": 1})])
    assert client.request_json("GET", "http://x") == {"ok": 1}
    assert len(session.calls) == 3 and sleeps == [1.0, 2.0]


def test_rate_limit_honours_retry_after_capped():
    client, _, sleeps = _client([FakeResponse(429, headers={"Retry-After": "500"}), FakeResponse(200, {})])
    client.request_json("GET", "http://x")
    assert sleeps == [30.0]


def test_client_errors_are_not_retried():
    client, session, sleeps = _client([FakeResponse(404)])
    with pytest.raises(ProviderUnavailableError, match="404"):
        client.request_json("GET", "http://x")
    assert len(session.calls) == 1 and sleeps == []


def test_gives_up_after_the_retry_budget():
    client, session, _ = _client([FakeResponse(500)] * 4, retries=3)
    with pytest.raises(ProviderUnavailableError, match="4 attempts"):
        client.request_json("GET", "http://x")
    assert len(session.calls) == 4


def test_invalid_json_is_a_provider_failure():
    client, _, _ = _client([FakeResponse(200, raw_text="<html>")])
    with pytest.raises(ProviderUnavailableError, match="invalid JSON"):
        client.request_json("GET", "http://x")


def test_search_follows_next_links_and_sends_the_expected_query():
    page1 = {"features": [{"id": "a"}], "links": [{"rel": "next", "method": "POST", "href": "http://c/search",
                                                   "body": {"next": "token-2"}}]}
    page2 = {"features": [{"id": "b"}], "links": []}
    client, session, _ = _client([FakeResponse(200, page1), FakeResponse(200, page2)])
    items = search_items(client, "http://c/", "coll", box(72.5, 33.6, 72.6, 33.7), START, END)

    assert [i["id"] for i in items] == ["a", "b"]
    first, second = session.calls
    assert first[0:2] == ("POST", "http://c/search")
    assert first[2]["collections"] == ["coll"] and first[2]["bbox"] == [72.5, 33.6, 72.6, 33.7]
    assert first[2]["datetime"] == "2025-10-01T00:00:00Z/2025-10-15T23:59:59Z"
    assert second[2] == {"next": "token-2"}


def test_search_refuses_a_response_that_is_not_an_item_collection():
    client, _, _ = _client([FakeResponse(200, {"message": "maintenance"})])
    with pytest.raises(ProviderUnavailableError, match="not an item collection"):
        search_items(client, "http://c", "coll", box(0, 0, 1, 1), START, END)


def test_runaway_paging_is_stopped():
    endless = {"features": [{"id": "x"}], "links": [{"rel": "next", "method": "POST", "href": "http://c/search", "body": {}}]}
    client, _, _ = _client(lambda m, u, b: FakeResponse(200, endless))
    with pytest.raises(ProviderUnavailableError, match="more pages"):
        search_items(client, "http://c", "coll", box(0, 0, 1, 1), START, END, max_pages=3)
