from dataclasses import dataclass
import cv2
import numpy as np
from drone_alignment.config.schema import FeatureDetectionConfig, DetectorType


class InsufficientValidDataError(ValueError):
    pass


class InsufficientContrastError(ValueError):
    pass


@dataclass
class DetectionResult:
    keypoints: list
    descriptors: np.ndarray
    scale_factor: float
    image_shape: tuple[int, int]


def normalize_to_uint8(
    img: np.ndarray,
    valid_mask: np.ndarray | None = None,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    apply_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
    min_valid_pixels: int = 1,
) -> np.ndarray:
    """Normalize using formally valid pixels only; invalid fill values never affect the stretch."""
    if valid_mask is None:
        valid = np.isfinite(img) & (img > 0)  # compatibility for callers without raster masks
    else:
        if valid_mask.shape != img.shape:
            raise ValueError("Image and valid mask shapes must match.")
        valid = valid_mask.astype(bool) & np.isfinite(img)
    values = img[valid]
    if values.size < min_valid_pixels:
        raise InsufficientValidDataError(
            f"Only {values.size} valid pixels available; at least {min_valid_pixels} are required."
        )
    low, high = np.percentile(values, [percentile_low, percentile_high])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise InsufficientContrastError("Valid pixels do not contain a usable intensity range.")
    out = np.zeros(img.shape, dtype=np.uint8)
    out[valid] = np.clip((img[valid] - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_grid_size, clahe_grid_size))
        out = clahe.apply(out)
        out[~valid] = 0
    return out


def detect_features(
    rgb_band: np.ndarray,
    ms_band: np.ndarray,
    config: FeatureDetectionConfig,
    rgb_mask: np.ndarray | None = None,
    ms_mask: np.ndarray | None = None,
) -> tuple[DetectionResult, DetectionResult]:
    if rgb_band.shape != ms_band.shape:
        raise ValueError("Registration images must share a target grid.")
    if rgb_mask is None:
        rgb_mask = np.isfinite(rgb_band) & (rgb_band > 0)
    if ms_mask is None:
        ms_mask = np.isfinite(ms_band) & (ms_band > 0)
    kwargs = dict(
        percentile_low=config.percentile_low, percentile_high=config.percentile_high,
        apply_clahe=config.apply_clahe, clahe_clip_limit=config.clahe_clip_limit,
        clahe_grid_size=config.clahe_grid_size, min_valid_pixels=config.min_valid_pixels,
    )
    rgb_norm = normalize_to_uint8(rgb_band, rgb_mask, **kwargs)
    ms_norm = normalize_to_uint8(ms_band, ms_mask, **kwargs)
    h, w = rgb_norm.shape
    scale = min(1.0, config.downsample_max_dim / max(h, w))
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        rgb_norm = cv2.resize(rgb_norm, size, interpolation=cv2.INTER_AREA)
        ms_norm = cv2.resize(ms_norm, size, interpolation=cv2.INTER_AREA)
        rgb_mask = cv2.resize(rgb_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        ms_mask = cv2.resize(ms_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    detector = (cv2.ORB_create(nfeatures=config.max_keypoints) if config.detector == DetectorType.ORB
                else cv2.SIFT_create(nfeatures=config.max_keypoints))
    kp_rgb, desc_rgb = detector.detectAndCompute(rgb_norm, rgb_mask.astype(np.uint8) * 255)
    kp_ms, desc_ms = detector.detectAndCompute(ms_norm, ms_mask.astype(np.uint8) * 255)
    default_dtype = np.uint8 if config.detector == DetectorType.ORB else np.float32
    descriptor_width = 32 if config.detector == DetectorType.ORB else 128
    return (
        DetectionResult(kp_rgb or [], desc_rgb if desc_rgb is not None else np.empty((0, descriptor_width), default_dtype), scale, rgb_norm.shape),
        DetectionResult(kp_ms or [], desc_ms if desc_ms is not None else np.empty((0, descriptor_width), default_dtype), scale, ms_norm.shape),
    )
