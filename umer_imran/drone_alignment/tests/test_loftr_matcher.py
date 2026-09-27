import cv2
import numpy as np
import pytest

from drone_alignment.alignment.feature_matcher import InsufficientMatchesError
from drone_alignment.alignment.loftr_matcher import LoFTRUnavailableError, match_loftr
from drone_alignment.alignment.representations import build_representation
from drone_alignment.config.schema import FeatureDetectionConfig, LoFTRConfig, RepresentationType


def test_gabor_energy_representation_is_mask_aware():
    yy, xx = np.mgrid[:128, :128]
    image = (((xx // 8) + (yy // 8)) % 2).astype(np.float32)
    valid = np.ones(image.shape, dtype=bool)
    valid[:12] = False

    result = build_representation(
        image, valid, RepresentationType.GABOR_ENERGY,
        FeatureDetectionConfig(min_valid_pixels=100),
    )

    assert result.dtype == np.uint8
    assert np.all(result[:12] == 0)
    assert result[20:].max() > 0


def test_loftr_unavailable_fails_closed(monkeypatch):
    import drone_alignment.alignment.loftr_matcher as matcher

    def unavailable(_device):
        raise LoFTRUnavailableError("weights unavailable")

    monkeypatch.setattr(matcher, "_load_loftr", unavailable)
    image = np.zeros((96, 96), dtype=np.uint8)
    mask = np.ones(image.shape, dtype=bool)

    with pytest.raises(LoFTRUnavailableError):
        match_loftr(image, image, mask, mask, LoFTRConfig(tile_grid=2))


def test_tiled_loftr_matches_are_returned_in_source_coordinates(monkeypatch):
    import torch
    import drone_alignment.alignment.loftr_matcher as matcher

    class MockLoFTR:
        def __call__(self, _inputs):
            points = torch.tensor(
                [[20.0, 20.0], [40.0, 20.0], [20.0, 40.0], [40.0, 40.0], [60.0, 60.0]],
                dtype=torch.float32,
            )
            return {"keypoints0": points, "keypoints1": points, "confidence": torch.ones(5)}

    monkeypatch.setattr(matcher, "_load_loftr", lambda _device: (MockLoFTR(), "cpu"))
    rng = np.random.default_rng(42)
    image = (rng.random((300, 300)) * 255).astype(np.uint8)
    for x, y in [(50, 50), (150, 150), (250, 250), (100, 200)]:
        cv2.circle(image, (x, y), 10, 255, -1)
    mask = np.ones(image.shape, dtype=bool)

    result = match_loftr(image, image, mask, mask, LoFTRConfig(tile_grid=2, min_confidence=0.5))

    assert result.num_good_matches >= 20
    assert result.pts_rgb.shape == result.pts_ms.shape
    assert np.allclose(result.pts_rgb, result.pts_ms, atol=1.0)
