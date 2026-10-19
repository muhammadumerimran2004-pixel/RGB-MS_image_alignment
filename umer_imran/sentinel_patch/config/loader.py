"""Load the packaged default or a user-supplied Sentinel patch configuration."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from sentinel_patch.errors import ConfigurationError

from .schema import SentinelPatchConfig, parse_config

DEFAULT_CONFIG_NAME = "default_sentinel_patch_v1.yaml"
ROOT_KEY = "sentinel_patch"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration file {path}: {exc}") from exc
    if not isinstance(loaded, dict) or not isinstance(loaded.get(ROOT_KEY), dict):
        raise ConfigurationError(f"Configuration file {path} must contain a '{ROOT_KEY}:' mapping.")
    return loaded[ROOT_KEY]


def load_config(config_path: str | Path | None = None) -> SentinelPatchConfig:
    """Load the packaged default or one complete user-supplied YAML document.

    A custom file is validated on its own and never deep-merged with the
    defaults; omitted keys take the schema defaults, which match the packaged YAML.
    """
    if config_path is None:
        packaged = resources.files(__package__).joinpath(DEFAULT_CONFIG_NAME)
        with resources.as_file(packaged) as resolved:
            return parse_config(_read_yaml(resolved))
    return parse_config(_read_yaml(Path(config_path)))
