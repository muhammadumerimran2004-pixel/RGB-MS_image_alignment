import pytest
from drone_alignment.config.schema import AlignmentConfig, ResolutionMode
from drone_alignment.io.validators import validate_inputs
from drone_alignment.alignment.coarse import coarse_align


def test_coarse_align_ms_native(synthetic_geo_tiff_pair):
    config = AlignmentConfig(resolution_mode=ResolutionMode.MS_NATIVE)
    rgb_meta, ms_meta = validate_inputs(
        synthetic_geo_tiff_pair["rgb_path"],
        synthetic_geo_tiff_pair["ms_path"],
        config,
    )
    result = coarse_align(rgb_meta, ms_meta, config)

    assert result.rgb_red_array.ndim == 2
    assert result.ms_red_array.ndim == 2
    assert result.ms_all_bands is None  # Native spectral bands remain on disk until a transform is accepted.
    assert max(result.rgb_red_array.shape) <= config.coarse.registration_max_dimension
    assert pytest.approx(result.target_gsd, rel=1e-3) == synthetic_geo_tiff_pair["ms_gsd"]


def test_coarse_align_rgb_native(synthetic_geo_tiff_pair):
    config = AlignmentConfig(resolution_mode=ResolutionMode.RGB_NATIVE)
    rgb_meta, ms_meta = validate_inputs(
        synthetic_geo_tiff_pair["rgb_path"],
        synthetic_geo_tiff_pair["ms_path"],
        config,
    )
    result = coarse_align(rgb_meta, ms_meta, config)

    assert pytest.approx(result.target_gsd, rel=1e-3) == synthetic_geo_tiff_pair["rgb_gsd"]
