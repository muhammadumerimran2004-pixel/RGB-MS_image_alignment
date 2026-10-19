"""Microsoft Planetary Computer: anonymous, hrefs need a short-lived SAS token (``?<token>``).

The collection is NOT harmonized and its items declare no scale/offset, so radiometry always comes
from ``s2:processing_baseline``. Asset keys and the token endpoint were checked live (2025-10).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sentinel_patch.discovery.provider import ProviderUnavailableError, SceneCandidate
from sentinel_patch.discovery.providers.stac_base import StacSentinel2Provider, parse_datetime

ASSET_MAP = {band: band for band in (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL")}
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)


class PlanetaryComputerProvider(StacSentinel2Provider):
    name = "planetary_computer"
    default_stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    default_collection = "sentinel-2-l2a"
    asset_map = ASSET_MAP

    def __init__(self, *args, now=lambda: datetime.now(timezone.utc), **kwargs):
        super().__init__(*args, **kwargs)
        self._now = now
        self._token: str | None = None
        self._token_expiry: datetime | None = None

    def _token_url(self) -> str:
        return self.stac_url.rstrip("/").replace("/stac/v1", "/sas/v1/token/") + self.collection

    def _sas_token(self) -> str:
        if self._token and self._token_expiry and self._now() < self._token_expiry - TOKEN_REFRESH_MARGIN:
            return self._token
        payload = self.client.request_json("GET", self._token_url())
        try:
            self._token, self._token_expiry = str(payload["token"]), parse_datetime(payload["msft:expiry"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderUnavailableError("Planetary Computer returned an unusable SAS token.") from exc
        return self._token

    def resolve_href(self, candidate: SceneCandidate, band: str) -> str:
        href = super().resolve_href(candidate, band)
        return href if "?" in href else f"{href}?{self._sas_token()}"
