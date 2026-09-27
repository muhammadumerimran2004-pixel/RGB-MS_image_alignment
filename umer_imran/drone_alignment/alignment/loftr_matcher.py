"""Optional, fail-closed LoFTR candidate matcher for RGB/MS registration."""
from __future__ import annotations

from functools import lru_cache
import os

import cv2
import numpy as np

from drone_alignment.alignment.feature_matcher import InsufficientMatchesError, MatchResult
from drone_alignment.config.schema import LoFTRConfig


class LoFTRUnavailableError(InsufficientMatchesError):
    """Raised when the optional learned model or its pretrained weights cannot load."""


@lru_cache(maxsize=3)
def _load_loftr(device: str):
    """Lazily load the official pretrained model; no import/network work at CLI startup."""
    try:
        import certifi
        import torch
        from kornia.feature import LoFTR
    except Exception as exc:  # pragma: no cover - platform dependent optional dependency
        raise LoFTRUnavailableError(f"LoFTR dependencies are unavailable: {type(exc).__name__}") from exc
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise LoFTRUnavailableError("LoFTR device is configured as CUDA but CUDA is unavailable.")
    try:
        return LoFTR(pretrained="outdoor").eval().to(resolved_device), resolved_device
    except Exception as exc:  # includes an unavailable pretrained-weight download
        raise LoFTRUnavailableError(f"Pretrained LoFTR could not load: {type(exc).__name__}") from exc


def _phase_prior(reference: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return a translation mapping target (MS) registration pixels to reference (RGB)."""
    yy, xx = np.where(valid)
    if len(xx) < 100:
        raise InsufficientMatchesError("Too little common valid area for a LoFTR phase prior.")
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    mask = valid[y0:y1, x0:x1]
    ref = reference[y0:y1, x0:x1].astype(np.float64)
    ms = target[y0:y1, x0:x1].astype(np.float64)
    ref = np.where(mask, ref - ref[mask].mean(), 0.0)
    ms = np.where(mask, ms - ms[mask].mean(), 0.0)
    shift, _ = cv2.phaseCorrelate(ref, ms, cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_64F))
    # OpenCV reports target displacement relative to reference; registration needs its inverse.
    return np.array([[1.0, 0.0, -float(shift[0])], [0.0, 1.0, -float(shift[1])]], dtype=np.float32)


def _resize_for_loftr(image: np.ndarray, max_dimension: int = 1024) -> tuple[np.ndarray, float, float]:
    h, w = image.shape
    scale = min(1.0, max_dimension / max(h, w))
    run_w = max(32, int(round((w * scale) / 8.0)) * 8)
    run_h = max(32, int(round((h * scale) / 8.0)) * 8)
    resized = cv2.resize(image, (run_w, run_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return resized, w / run_w, h / run_h


def match_loftr(
    rgb: np.ndarray,
    ms: np.ndarray,
    rgb_valid_mask: np.ndarray,
    ms_valid_mask: np.ndarray,
    config: LoFTRConfig,
) -> MatchResult:
    """Return phase-prior tiled LoFTR correspondences in original registration coordinates.

    Model confidence is only a local filter.  Geometric trust is established by
    the production pipeline after this function returns.
    """
    if rgb.shape != ms.shape or rgb.shape != rgb_valid_mask.shape or ms.shape != ms_valid_mask.shape:
        raise ValueError("LoFTR inputs and masks must share one registration grid.")
    try:
        import torch
        loaded = _load_loftr(config.device)
        model, device = loaded if isinstance(loaded, tuple) else (loaded, "cpu")  # permits simple test doubles
    except LoFTRUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary around optional dependencies
        raise LoFTRUnavailableError(f"LoFTR could not initialize: {type(exc).__name__}") from exc

    common = rgb_valid_mask.astype(bool) & ms_valid_mask.astype(bool)
    prior = _phase_prior(rgb, ms, common)
    h, w = rgb.shape
    warped_ms = cv2.warpAffine(ms, prior, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    warped_valid = cv2.warpAffine(ms_valid_mask.astype(np.uint8), prior, (w, h), flags=cv2.INTER_NEAREST) > 0
    inverse_prior = cv2.invertAffineTransform(prior)

    tile_specs: list[tuple[float, int, int, int, int, int, int]] = []
    for row in range(config.tile_grid):
        for col in range(config.tile_grid):
            y0, y1 = row * h // config.tile_grid, (row + 1) * h // config.tile_grid
            x0, x1 = col * w // config.tile_grid, (col + 1) * w // config.tile_grid
            px0, py0 = max(0, x0 - config.tile_halo_px), max(0, y0 - config.tile_halo_px)
            px1, py1 = min(w, x1 + config.tile_halo_px), min(h, y1 + config.tile_halo_px)
            valid = common[py0:py1, px0:px1] & warped_valid[py0:py1, px0:px1]
            if valid.mean() < 0.60 or min(px1 - px0, py1 - py0) < 32:
                continue
            # Prefer tiles that can constrain correspondence, rather than
            # spending model calls on uniform crop canopy or no-data margins.
            texture = float(np.std(rgb[py0:py1, px0:px1][valid]))
            tile_specs.append((texture, px0, py0, px1, py1, x0, y0))

    points_rgb: list[np.ndarray] = []
    points_ms: list[np.ndarray] = []
    raw_count = 0
    for _, px0, py0, px1, py1, _, _ in sorted(tile_specs, reverse=True)[:config.max_tiles]:
        valid = common[py0:py1, px0:px1] & warped_valid[py0:py1, px0:px1]
        try:
            rgb_patch, sx, sy = _resize_for_loftr(rgb[py0:py1, px0:px1])
            ms_patch, _, _ = _resize_for_loftr(warped_ms[py0:py1, px0:px1])
            image0 = torch.from_numpy(np.ascontiguousarray(rgb_patch)).float()[None, None].to(device) / 255.0
            image1 = torch.from_numpy(np.ascontiguousarray(ms_patch)).float()[None, None].to(device) / 255.0
            with torch.inference_mode():
                output = model({"image0": image0, "image1": image1})
        except Exception:
            continue
        keypoints0, keypoints1, confidence = (output.get("keypoints0"), output.get("keypoints1"), output.get("confidence"))
        if keypoints0 is None or keypoints1 is None or confidence is None:
            continue
        raw_count += len(keypoints0)
        k0 = keypoints0.detach().cpu().numpy() * np.array([sx, sy], dtype=np.float32)
        k1 = keypoints1.detach().cpu().numpy() * np.array([sx, sy], dtype=np.float32)
        score = confidence.detach().cpu().numpy()
        candidates = []
        for reference_xy, warped_xy, confidence_value in zip(k0, k1, score):
            if confidence_value < config.min_confidence:
                continue
            reference_xy = reference_xy + np.array([px0, py0], dtype=np.float32)
            warped_xy = warped_xy + np.array([px0, py0], dtype=np.float32)
            source_xy = cv2.transform(warped_xy.reshape(1, 1, 2), inverse_prior).reshape(2)
            rx, ry = np.rint(reference_xy).astype(int)
            mx, my = np.rint(source_xy).astype(int)
            if not (0 <= rx < w and 0 <= ry < h and 0 <= mx < w and 0 <= my < h):
                continue
            if not (rgb_valid_mask[ry, rx] and ms_valid_mask[my, mx]):
                continue
            if np.linalg.norm(reference_xy - warped_xy) > config.max_residual_from_prior_px:
                continue
            candidates.append((float(confidence_value), reference_xy, source_xy))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, reference_xy, source_xy in candidates[:config.max_matches_per_tile]:
            points_rgb.append(reference_xy)
            points_ms.append(source_xy)
    if len(points_rgb) < 4:
        raise InsufficientMatchesError(f"LoFTR produced only {len(points_rgb)} confident valid matches.")
    return MatchResult(
        pts_rgb=np.asarray(points_rgb, dtype=np.float32),
        pts_ms=np.asarray(points_ms, dtype=np.float32),
        num_raw_matches=raw_count,
        num_good_matches=len(points_rgb),
        match_quality_score=len(points_rgb) / max(1, raw_count),
    )
