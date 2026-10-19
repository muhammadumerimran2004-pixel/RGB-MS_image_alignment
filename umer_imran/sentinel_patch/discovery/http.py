"""JSON-over-HTTP for catalogue calls: timeouts, retries with backoff, and one error type.

Only ``ProviderUnavailableError`` leaves this module, so the provider chain can treat every
network, rate-limit and server failure the same way: record it and try the next provider.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import requests

from sentinel_patch.errors import ProviderUnavailableError

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_WAIT_S = 30.0


class HttpClient:
    def __init__(
        self, *, timeout_s: float = 60.0, retries: int = 4, backoff_s: float = 1.0,
        session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep,
    ):
        self.timeout_s, self.retries, self.backoff_s = timeout_s, retries, backoff_s
        self.session = session or requests.Session()
        self._sleep = sleep

    def _wait_s(self, attempt: int, response: requests.Response | None) -> float:
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), MAX_WAIT_S)
        return min(self.backoff_s * 2**attempt, MAX_WAIT_S)

    def request_json(self, method: str, url: str, *, json_body: dict | None = None) -> Any:
        last = "no attempt made"
        for attempt in range(self.retries + 1):
            response = None
            try:
                response = self.session.request(method, url, json=json_body, timeout=self.timeout_s)
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code in RETRY_STATUSES:
                    last = f"HTTP {response.status_code}"
                elif response.status_code >= 400:
                    raise ProviderUnavailableError(f"{method} {url} failed with HTTP {response.status_code}.")
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ProviderUnavailableError(f"{method} {url} returned invalid JSON.") from exc
            if attempt < self.retries:
                self._sleep(self._wait_s(attempt, response))
        raise ProviderUnavailableError(f"{method} {url} failed after {self.retries + 1} attempts ({last}).")
