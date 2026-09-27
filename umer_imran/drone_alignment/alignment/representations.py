"""Modality-aware, single-channel registration representations.

These functions deliberately do not change raster data.  They produce temporary
registration views only, allowing matchers to compare structure rather than
sensor-specific radiometry.
"""
from __future__ import annotations

import cv2
import numpy as np

from drone_alignment.alignment.feature_detector import normalize_to_uint8
from drone_alignment.config.schema import FeatureDetectionConfig, RepresentationType


def build_representation(
    image: np.ndarray,
    valid_mask: np.ndarray,
    representation: RepresentationType,
    feature_config: FeatureDetectionConfig,
) -> np.ndarray:
    """Create one mask-aware uint8 view suitable for a correspondence matcher."""
    normalized = normalize_to_uint8(
        image, valid_mask,
        percentile_low=feature_config.percentile_low,
        percentile_high=feature_config.percentile_high,
        apply_clahe=feature_config.apply_clahe,
        clahe_clip_limit=feature_config.clahe_clip_limit,
        clahe_grid_size=feature_config.clahe_grid_size,
        min_valid_pixels=feature_config.min_valid_pixels,
    )
    if representation == RepresentationType.NORMALIZED:
        return normalized
    if representation == RepresentationType.GABOR_ENERGY:
        source = normalized.astype(np.float32) / 255.0
        energy = np.zeros(source.shape, dtype=np.float32)
        for wavelength in (4.0, 8.0):
            for direction in range(8):
                kernel = cv2.getGaborKernel(
                    (21, 21), 3.5, np.pi * direction / 8.0, wavelength, 0.6, 0,
                    ktype=cv2.CV_32F,
                )
                energy = np.maximum(energy, np.abs(cv2.filter2D(source, cv2.CV_32F, kernel)))
        result = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        result[~valid_mask.astype(bool)] = 0
        return result
    raise ValueError(f"Unsupported registration representation: {representation}")
