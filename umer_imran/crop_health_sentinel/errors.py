"""Stable, caller-safe error taxonomy for Sentinel crop-health processing."""

from __future__ import annotations


class CropHealthError(Exception):
    """Base class for expected crop-health failures."""

    code = "CROP_HEALTH_ERROR"


class ConfigurationError(CropHealthError):
    code = "CONFIG_INVALID"


class InputValidationError(CropHealthError):
    code = "INPUT_INVALID"


class GeometryError(CropHealthError):
    code = "GEOMETRY_INVALID"


class RasterMetadataError(CropHealthError):
    code = "RASTER_METADATA_INVALID"


class RasterAlignmentError(CropHealthError):
    code = "RASTER_ALIGNMENT_FAILED"


class ResourceLimitError(CropHealthError):
    code = "RESOURCE_LIMIT_EXCEEDED"


class InsufficientAnalyticsError(CropHealthError):
    code = "NO_HEALTH_INDEX"


class H3AggregationError(CropHealthError):
    code = "H3_AGGREGATION_FAILED"


class H3CapacityError(H3AggregationError):
    code = "H3_CAPACITY_EXCEEDED"


class ArtifactError(CropHealthError):
    code = "ARTIFACT_WRITE_FAILED"


class PublicContractError(CropHealthError):
    code = "PUBLIC_CONTRACT_INVALID"


class ConcurrentRunError(CropHealthError):
    code = "OUTPUT_DIR_BUSY"

