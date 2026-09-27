"""One authoritative declaration of Sentinel source-band semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BandKind(StrEnum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BITMASK = "bitmask"


@dataclass(frozen=True, slots=True)
class BandDefinition:
    logical_name: str
    config_key: str
    kind: BandKind
    required: bool


BANDS = (
    BandDefinition("green", "green", BandKind.CONTINUOUS, True),
    BandDefinition("red", "red", BandKind.CONTINUOUS, True),
    BandDefinition("red_edge", "red_edge", BandKind.CONTINUOUS, True),
    BandDefinition("nir", "nir", BandKind.CONTINUOUS, True),
    BandDefinition("scl", "scl", BandKind.CATEGORICAL, True),
    BandDefinition("cloud_probability", "cloud_probability", BandKind.CONTINUOUS, False),
    BandDefinition("opaque_cloud", "opaque_cloud", BandKind.CATEGORICAL, False),
    BandDefinition("cirrus", "cirrus", BandKind.CATEGORICAL, False),
    BandDefinition("snow_ice", "snow_ice", BandKind.CATEGORICAL, False),
    BandDefinition("qa60", "qa60", BandKind.BITMASK, False),
    BandDefinition("aot", "aot", BandKind.CONTINUOUS, False),
)
