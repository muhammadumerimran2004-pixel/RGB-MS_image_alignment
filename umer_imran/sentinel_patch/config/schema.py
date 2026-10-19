"""Validated configuration for Sentinel patch co-registration (blueprint section 5.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentinel_patch.errors import ConfigurationError

MAX_DATE_GAP_HARD_CAP_DAYS = 30


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(StrictModel):
    name: Literal["earth_search", "planetary_computer", "cdse"]
    enabled: bool = True
    stac_url: Optional[str] = None
    collection: Optional[str] = None


class DiscoveryConfig(StrictModel):
    date_search_direction: Literal["both", "after_only", "before_only"] = "both"
    preferred_gap_days: int = Field(3, ge=0, description="At most this many days apart: tier 'same_period'.")
    acceptable_gap_days: int = Field(7, ge=0, description="At most this many days apart: tier 'near'.")
    max_date_gap_days: int = Field(
        15, ge=0, le=MAX_DATE_GAP_HARD_CAP_DAYS, description="Search limit; beyond the acceptable gap: tier 'stale'."
    )
    prefilter_max_scene_cloud: float = Field(90.0, ge=0.0, le=100.0, description="Cheap scene-level prefilter only.")
    max_aoi_masked_fraction: float = Field(
        0.10, ge=0.0, le=1.0, description="Accept a date when at most this fraction of the AOI is cloud/shadow/nodata."
    )
    fallback_max_aoi_masked_fraction: float = Field(
        0.30, ge=0.0, le=1.0, description="If no date passes, accept the best scene up to this fraction as 'partial'."
    )
    min_clear_pixels: int = Field(100, ge=1, description="Absolute floor of clear 10 m pixels inside the AOI.")
    scl_reject_classes: tuple[int, ...] = (0, 1, 3, 8, 9, 10, 11)
    cloud_mask_dilation_px: int = Field(2, ge=0, le=10)
    cloud_probability_threshold: float = Field(40.0, ge=0.0, le=100.0)
    max_candidates_evaluated: int = Field(20, ge=1)

    @model_validator(mode="after")
    def _check_ordering(self) -> "DiscoveryConfig":
        if not self.preferred_gap_days <= self.acceptable_gap_days <= self.max_date_gap_days:
            raise ValueError("require preferred_gap_days <= acceptable_gap_days <= max_date_gap_days")
        if self.fallback_max_aoi_masked_fraction < self.max_aoi_masked_fraction:
            raise ValueError("fallback_max_aoi_masked_fraction must be >= max_aoi_masked_fraction")
        return self


class IoConfig(StrictModel):
    bands_10m: tuple[str, ...] = ("B02", "B03", "B04", "B08")
    bands_20m: tuple[str, ...] = ("B05", "B06", "B07", "B8A", "B11", "B12")
    aux: tuple[str, ...] = ("SCL",)
    registration_buffer_m: float = Field(300.0, ge=0.0)
    http_timeout_s: int = Field(60, ge=1)
    http_retries: int = Field(4, ge=0)
    cache_dir: Optional[Path] = Field(None, description="Default: <output>/sentinel/.cache")


class BandPair(StrictModel):
    drone: Literal["nir", "red", "green", "red_edge"]
    sentinel: str


class ReferenceConfig(StrictModel):
    match_band: BandPair = BandPair(drone="nir", sentinel="B08")
    holdout_band: BandPair = BandPair(drone="red", sentinel="B04")
    psf_sigma_m: float = Field(5.0, gt=0.0, description="Gaussian PSF sigma; MTF(Nyquist) ~ 0.25 at 10 m.")
    intermediate_gsd_m: float = Field(1.0, gt=0.0)
    min_pixel_valid_fraction: float = Field(0.95, gt=0.0, le=1.0)
    outlier_mad_factor: float = Field(
        50.0, gt=0.0,
        description="Drone cells (about 1 m) deviating from the median by more than this many robust standard "
                    "deviations (1.4826 x MAD) are ignored. Real aligned orthophotos carry values near 4e9 along "
                    "their outermost edge (0.1-0.2 % of cells), which would otherwise dominate any correlation.",
    )


class ConsensusConfig(StrictModel):
    """Multi-scene consensus, tried only when the single best scene is refused for an ambiguity-type reason."""
    enabled: bool = True
    min_scenes: int = Field(3, ge=2, description="Independent acquisitions that must agree.")
    min_agree_fraction: float = Field(
        0.5, gt=0.0, le=1.0, description="...and at least this share of all usable votes (3 of 12 is a coincidence).")
    agreement_radius_px: float = Field(1.0, gt=0.0, description="Scenes agree when within this many Sentinel pixels.")
    max_scenes: int = Field(10, ge=2, description="Most acquisitions measured (nearest dates first).")
    min_overlap_px: int = Field(100, ge=1, description="Per-scene overlap needed to vote (the single-scene gate is stricter).")
    max_aoi_masked_fraction: float = Field(0.10, ge=0.0, le=1.0, description="Only clean scenes may vote.")


class RegistrationConfig(StrictModel):
    max_shift_m: float = Field(25.0, gt=0.0)
    arosics_window_px: int | Literal["auto"] = "auto"
    ncc_search_px: int = Field(
        4, ge=1, le=10,
        description="Integer search radius in 10 m pixels. A maximum on the outer ring is rejected, so it must "
                    "exceed max_shift_m / 10 by at least one pixel (25 m -> 4).",
    )
    arosics_mode: Literal["off", "advisory", "veto"] = Field(
        "advisory",
        description="'off': never run AROSICS. 'advisory': run it when installed, report it, warn "
                    "AROSICS_DISAGREES on a gross disagreement, never reject. 'veto': a conclusive disagreement "
                    "rejects the registration. Default advisory: on synthetic data with a blurred drone reference "
                    "against Sentinel pixels AROSICS had a median error of 0.72 px against 0.03 px for NCC, and 29 % "
                    "of its conclusive (reliability >= 60) answers differed from the correct NCC answer by more than "
                    "a pixel, so a veto would refuse many good registrations. Re-check on real imagery before using 'veto'.",
    )
    arosics_min_reliability: float = Field(
        60.0, ge=0.0, le=100.0,
        description="AROSICS' own reliability (%) needed for its shift to count as an opinion at all.",
    )
    agreement_tolerance_px: float = Field(
        1.0, gt=0.0,
        description="Maximum |AROSICS - NCC| in pixels, when AROSICS is conclusive, before it is called a disagreement.",
    )
    min_overlap_px: int = Field(400, ge=1)
    min_texture_score: float = Field(
        0.05, ge=0.0,
        description="Minimum coefficient of variation (std/mean) of the reference band. A uniform crop field sits "
                    "around 0.02-0.05; a scene with field edges, roads or canals is well above. Starting value: tune on real data.",
    )
    min_ncc_peak: float = Field(0.50, ge=-1.0, le=1.0)
    min_peak_ratio: float = Field(1.15, ge=1.0)
    min_holdout_gain: float = Field(0.02, ge=0.0)
    quadrant_check_min_px: int = Field(
        48, ge=16, description="Run the quadrant check only if the valid overlap spans at least this many 10 m pixels each way."
    )
    quadrant_tolerance_px: float = Field(0.5, gt=0.0)
    large_shift_warning_m: float = Field(
        15.0, gt=0.0,
        description="A verified shift above this, with no RTK/PPK/GCP evidence for the drone frame, adds the warning "
                    "LARGE_SHIFT_DRONE_FRAME_SUSPECT: the drone's own absolute position may be what is off.",
    )
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)
    on_reject: Literal["publish_unshifted", "skip"] = "publish_unshifted"


class ProductsConfig(StrictModel):
    write_20m: bool = True
    write_overlay: bool = True
    overlay_gsd_m: float = Field(1.0, gt=0.0)
    align_grids: bool = False
    write_crop_health_bands: bool = True
    compression: Literal["deflate", "lzw", "none"] = "deflate"


class DroneMetadataConfig(StrictModel):
    path_scan_depth: int = Field(3, ge=0, le=8)
    date_order: Literal["dmy", "mdy"] = "dmy"
    accept_path_date_estimate: bool = False
    write_sidecar: bool = True


def _default_providers() -> tuple[ProviderConfig, ...]:
    return (
        ProviderConfig(
            name="earth_search", stac_url="https://earth-search.aws.element84.com/v1", collection="sentinel-2-c1-l2a",
        ),
        ProviderConfig(
            name="planetary_computer", stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
            collection="sentinel-2-l2a",
        ),
        ProviderConfig(name="cdse", enabled=False),
    )


class SentinelPatchConfig(StrictModel):
    enabled: bool = False
    providers: tuple[ProviderConfig, ...] = Field(default_factory=_default_providers)
    local_folder: Optional[Path] = Field(None, description="If set, overrides the provider chain (offline scenes).")
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    io: IoConfig = Field(default_factory=IoConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    products: ProductsConfig = Field(default_factory=ProductsConfig)
    drone_metadata: DroneMetadataConfig = Field(default_factory=DroneMetadataConfig)

    @model_validator(mode="after")
    def _check_scene_source(self) -> "SentinelPatchConfig":
        if self.local_folder is None and not any(provider.enabled for provider in self.providers):
            raise ValueError("enable at least one provider or set local_folder")
        return self


def parse_config(value: object) -> SentinelPatchConfig:
    """Validate a config mapping, converting pydantic details into the package's error type."""
    try:
        return SentinelPatchConfig.model_validate(value)
    except Exception as exc:  # pydantic's public validation hierarchy is intentionally wrapped
        raise ConfigurationError(f"Invalid Sentinel patch configuration: {exc}") from exc
