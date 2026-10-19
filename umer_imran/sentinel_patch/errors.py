"""Error taxonomy for Sentinel patch co-registration.

Two kinds of failure are kept apart on purpose:

* ``RegistrationRejected`` is an *expected* safety-gate rejection. The caller
  publishes the patch with its original georeferencing and records the reason
  code; the run is not a failure.
* Every other ``SentinelPatchError`` stops the Sentinel stage only. The aligned
  MS product of the main alignment run is never affected.
"""

from __future__ import annotations


class SentinelPatchError(Exception):
    """Base class for expected Sentinel-stage failures."""

    code = "SENTINEL_PATCH_ERROR"


class ConfigurationError(SentinelPatchError):
    code = "CONFIG_INVALID"


class FlightDateUnknownError(SentinelPatchError):
    code = "FLIGHT_DATE_UNKNOWN"


class NoSceneError(SentinelPatchError):
    code = "NO_SCENE"


class ProviderUnavailableError(SentinelPatchError):
    """A scene provider is down, rate-limited or not configured; the chain moves to the next one."""

    code = "PROVIDER_UNAVAILABLE"


class SceneCoverageError(SentinelPatchError):
    """The scene raster does not fully contain the area of interest."""

    code = "AOI_OUTSIDE_SCENE"


class ResourceLimitError(SentinelPatchError):
    code = "RESOURCE_LIMIT_EXCEEDED"


class RadiometryUnknownError(SentinelPatchError):
    """Reflectance scale/offset cannot be determined; guessing would silently skew NDVI."""

    code = "RADIOMETRY_UNKNOWN"


class RegistrationRejected(SentinelPatchError):
    """Expected registration safety-gate rejection (see module docstring)."""

    code = "REGISTRATION_REJECTED"

    def __init__(self, reason_code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.reason_code = reason_code
        self.details = details or {}
