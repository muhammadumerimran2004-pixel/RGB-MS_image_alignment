"""A scripted stand-in for ``requests.Session`` so catalogue code is tested without the network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import requests

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, status: int = 200, body=None, headers=None, raw_text: str | None = None):
        self.status_code, self._body, self.headers, self._raw = status, body, headers or {}, raw_text

    def json(self):
        if self._raw is not None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """``script``: a list of responses / exceptions consumed in order, or a callable (method, url, json) -> response."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def request(self, method, url, json=None, timeout=None):  # noqa: A002 (mirrors requests)
        self.calls.append((method, url, json, timeout))
        result = self.script(method, url, json) if callable(self.script) else self.script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def connection_error() -> requests.ConnectionError:
    return requests.ConnectionError("boom")


def pages(*bodies: dict) -> Callable:
    """Search responses served page by page; anything after the last page is an empty collection."""
    queue = list(bodies)

    def handler(method, url, body):
        return FakeResponse(200, queue.pop(0) if queue else {"features": [], "links": []})

    return handler
