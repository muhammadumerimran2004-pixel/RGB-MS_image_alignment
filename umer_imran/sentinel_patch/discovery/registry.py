"""Provider registry: config names -> classes. A new catalogue is one file plus one entry here."""

from __future__ import annotations

import logging
from typing import Callable

from sentinel_patch.config.schema import SentinelPatchConfig
from sentinel_patch.discovery.chain import ProviderChain
from sentinel_patch.discovery.provider import ProviderUnavailableError, SceneProvider
from sentinel_patch.discovery.providers.cdse import CdseProvider
from sentinel_patch.discovery.providers.earth_search import EarthSearchProvider
from sentinel_patch.discovery.providers.planetary_computer import PlanetaryComputerProvider

logger = logging.getLogger(__name__)

PROVIDERS: dict[str, Callable[..., SceneProvider]] = {
    "earth_search": EarthSearchProvider,
    "planetary_computer": PlanetaryComputerProvider,
    "cdse": CdseProvider,
}


def build_chain(cfg: SentinelPatchConfig) -> ProviderChain:
    """Instantiate the enabled providers in config order; one that cannot start is recorded and skipped."""
    providers, failures = [], []
    for provider_cfg in cfg.providers:
        if not provider_cfg.enabled:
            continue
        try:
            providers.append(PROVIDERS[provider_cfg.name](provider_cfg, cfg.io))
        except ProviderUnavailableError as exc:
            logger.warning("Provider %s not started: %s", provider_cfg.name, exc)
            failures.append({"provider": provider_cfg.name, "code": exc.code, "message": str(exc)})
    return ProviderChain(providers, failures)
