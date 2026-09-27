from __future__ import annotations

import pytest

import crop_health_sentinel as sentinel
from crop_health_sentinel.errors import ConfigurationError, H3CapacityError
from crop_health_sentinel.models import normalize_scene_metadata


def test_public_exports_are_stable() -> None:
    assert sentinel.__version__ == "1.0.0"
    assert set(sentinel.__all__) == {
        "__version__",
        "generate_sentinel_crop_health_report",
        "get_sentinel_crop_health_report_metadata",
        "run_crop_health_report",
    }


def test_error_codes_are_stable() -> None:
    assert ConfigurationError.code == "CONFIG_INVALID"
    assert H3CapacityError.code == "H3_CAPACITY_EXCEEDED"


def test_metadata_aliases_and_defaults(sample_scene_metadata: dict[str, str]) -> None:
    metadata = normalize_scene_metadata(sample_scene_metadata)
    assert metadata.farm == "farm-1"
    assert metadata.field == "crop-1"
    assert metadata.date == "2026-08-19"
    assert metadata.scene_id == "scene-1"
    assert metadata.run_id == "scene-1"


def test_empty_metadata_alias_is_rejected() -> None:
    with pytest.raises(Exception, match="must not be empty"):
        normalize_scene_metadata({"farm": " "})

