from __future__ import annotations

from pathlib import Path

import pytest

from crop_health_sentinel.config import load_config
from crop_health_sentinel.config.schema import parse_config
from crop_health_sentinel.errors import ConfigurationError
from crop_health_sentinel.version import ALGORITHM_VERSION


def test_packaged_default_config_is_complete_and_immutable() -> None:
    config = load_config()
    assert config.algorithm_version == ALGORITHM_VERSION
    assert config.h3.max_analytic_pixels == 50_000
    with pytest.raises(Exception):
        config.h3.resolution = 12  # type: ignore[misc]


def test_unknown_key_is_rejected() -> None:
    payload = load_config().model_dump(mode="python")
    payload["unknown"] = True
    with pytest.raises(ConfigurationError, match="unknown"):
        parse_config(payload)


def test_invalid_threshold_order_is_rejected() -> None:
    payload = load_config().model_dump(mode="python")
    payload["quality"]["display_only_minimum_fraction"] = 0.40
    with pytest.raises(ConfigurationError, match="display_only_minimum_fraction"):
        parse_config(payload)


def test_custom_config_is_complete_not_merged(tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("algorithm_version: sentinel-crop-health/1.0.0\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)

