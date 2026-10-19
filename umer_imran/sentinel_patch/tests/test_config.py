"""The packaged YAML and the schema defaults must agree, and bad configs must fail loudly."""

from __future__ import annotations

import pytest
import yaml

from sentinel_patch import SentinelPatchConfig, load_config
from sentinel_patch.config.schema import parse_config
from sentinel_patch.errors import ConfigurationError


def test_packaged_default_loads_and_matches_schema_defaults():
    assert load_config() == SentinelPatchConfig()


def test_default_decisions_from_the_blueprint():
    cfg = load_config()
    assert cfg.enabled is False
    assert [(p.name, p.enabled) for p in cfg.providers] == [
        ("earth_search", True), ("planetary_computer", True), ("cdse", False),
    ]
    assert (cfg.discovery.preferred_gap_days, cfg.discovery.acceptable_gap_days, cfg.discovery.max_date_gap_days) == (3, 7, 15)
    assert cfg.discovery.max_aoi_masked_fraction == 0.10
    assert cfg.discovery.fallback_max_aoi_masked_fraction == 0.30
    assert cfg.drone_metadata.date_order == "dmy" and cfg.drone_metadata.accept_path_date_estimate is False


def test_custom_file_overrides_and_keeps_other_defaults(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump({"sentinel_patch": {"discovery": {"max_date_gap_days": 10}}}), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.discovery.max_date_gap_days == 10
    assert cfg.registration.max_shift_m == 25.0


@pytest.mark.parametrize("bad", [
    {"discovery": {"max_date_gap_days": 31}},                                  # beyond the hard cap
    {"discovery": {"preferred_gap_days": 9, "acceptable_gap_days": 7}},        # tiers out of order
    {"discovery": {"max_aoi_masked_fraction": 0.4, "fallback_max_aoi_masked_fraction": 0.3}},
    {"discovery": {"max_aoi_masked_fraction": 1.5}},
    {"typo_key": 1},                                                           # extra keys are rejected
    {"providers": [{"name": "cdse", "enabled": False}]},                       # no usable scene source
    {"providers": [{"name": "somewhere_else"}]},
    {"registration": {"on_reject": "explode"}},
])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ConfigurationError):
        parse_config(bad)


def test_local_folder_replaces_the_need_for_a_provider(tmp_path):
    cfg = parse_config({"providers": [{"name": "cdse", "enabled": False}], "local_folder": str(tmp_path)})
    assert cfg.local_folder == tmp_path


def test_config_is_immutable():
    with pytest.raises(Exception):
        load_config().enabled = True


def test_file_without_root_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("discovery: {}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)
