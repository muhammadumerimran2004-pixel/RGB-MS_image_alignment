from pathlib import Path
from shapely.geometry import box
from drone_alignment.config.schema import AlignmentConfig
from drone_alignment.io.reader import read_metadata, RasterMetadata
from drone_alignment.config.schema import RegistrationChannel


class AlignmentValidationError(ValueError):
    """Raised when input files fail validation for spatial alignment."""
    pass


def compute_bounding_box_intersection(
    bounds_a: tuple[float, float, float, float],
    bounds_b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Compute intersection rectangle of two bounding boxes (left, bottom, right, top).
    Returns (left, bottom, right, top) or raises AlignmentValidationError if no intersection.
    """
    box_a = box(*bounds_a)
    box_b = box(*bounds_b)

    if not box_a.intersects(box_b):
        raise AlignmentValidationError("No spatial overlap between RGB reference and MS target raster.")

    intersection = box_a.intersection(box_b)
    if intersection.is_empty or intersection.area <= 0:
        raise AlignmentValidationError("Intersection area between RGB reference and MS target is zero.")

    minx, miny, maxx, maxy = intersection.bounds
    return (minx, miny, maxx, maxy)


def validate_inputs(
    rgb_path: Path,
    ms_path: Path,
    config: AlignmentConfig,
) -> tuple[RasterMetadata, RasterMetadata]:
    """
    Validate input GeoTIFF files and return their metadata.

    Args:
        rgb_path: Path to reference RGB GeoTIFF
        ms_path: Path to target MS GeoTIFF
        config: Alignment configuration

    Returns:
        tuple of (rgb_meta, ms_meta)

    Raises:
        FileNotFoundError: If input file is missing
        AlignmentValidationError: If validation fails
    """
    rgb_path_obj = Path(rgb_path).resolve()
    ms_path_obj = Path(ms_path).resolve()

    if not rgb_path_obj.exists():
        raise FileNotFoundError(f"RGB reference file not found: {rgb_path_obj}")
    if not ms_path_obj.exists():
        raise FileNotFoundError(f"MS target file not found: {ms_path_obj}")

    try:
        rgb_meta = read_metadata(rgb_path_obj)
    except Exception as e:
        raise AlignmentValidationError(f"Failed to read RGB reference metadata: {e}") from e

    try:
        ms_meta = read_metadata(ms_path_obj)
    except Exception as e:
        raise AlignmentValidationError(f"Failed to read MS target metadata: {e}") from e

    if rgb_meta.crs is None:
        raise AlignmentValidationError(f"RGB reference raster is missing a CRS: {rgb_path_obj}")
    if ms_meta.crs is None:
        raise AlignmentValidationError(f"MS target raster is missing a CRS: {ms_path_obj}")

    if rgb_meta.width <= 0 or rgb_meta.height <= 0:
        raise AlignmentValidationError("RGB reference raster has invalid zero dimensions.")
    if ms_meta.width <= 0 or ms_meta.height <= 0:
        raise AlignmentValidationError("MS target raster has invalid zero dimensions.")

    # Report fundamental spatial incompatibility before optional band-map errors.
    compute_bounding_box_intersection(rgb_meta.bounds, ms_meta.bounds)

    if config.rgb_red_band_index < 1 or config.rgb_red_band_index > rgb_meta.band_count:
        raise AlignmentValidationError(
            f"Configured RGB red band index ({config.rgb_red_band_index}) "
            f"is out of bounds for RGB file with {rgb_meta.band_count} bands."
        )

    if config.ms_red_band_index < 1 or config.ms_red_band_index > ms_meta.band_count:
        raise AlignmentValidationError(
            f"Configured MS red band index ({config.ms_red_band_index}) "
            f"is out of bounds for MS file with {ms_meta.band_count} bands."
        )

    channel_indices = []
    for channel in config.registration_channel_priority:
        if channel == RegistrationChannel.GREEN:
            channel_indices.append(("RGB green", config.rgb_green_band_index, rgb_meta))
            channel_indices.append(("MS green", config.ms_green_band_index, ms_meta))
        else:
            channel_indices.append(("RGB red", config.rgb_red_band_index, rgb_meta))
            channel_indices.append(("MS red", config.ms_red_band_index, ms_meta))
    for label, index, meta in channel_indices:
        if index < 1 or index > meta.band_count:
            raise AlignmentValidationError(
                f"Configured {label} band index ({index}) is out of bounds for raster with {meta.band_count} bands."
            )
        if index == meta.alpha_band_index:
            raise AlignmentValidationError(f"Configured {label} band index ({index}) identifies an alpha band.")

    return (rgb_meta, ms_meta)
