"""Load complete, versioned Sentinel crop-health configuration documents."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from crop_health_sentinel.errors import ConfigurationError

from .schema import SentinelCropHealthConfig, parse_config


DEFAULT_CONFIG_NAME = "default_config_sentinel_v1.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Configuration file {path} must contain one mapping document.")
    return loaded


def load_config(config_path: str | Path | None = None) -> SentinelCropHealthConfig:
    """Load the packaged default or one complete user-supplied YAML document.

    Custom files are never deep-merged with defaults: a configuration is an
    explicit versioned algorithm contract.
    """
    if config_path is None:
        packaged = resources.files(__package__).joinpath(DEFAULT_CONFIG_NAME)
        with resources.as_file(packaged) as resolved:
            return parse_config(_read_yaml(resolved))
    return parse_config(_read_yaml(Path(config_path)))

