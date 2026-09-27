import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from drone_alignment.config.schema import AlignmentConfig, CoarseAlignmentConfig, ResolutionMode
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


def _write_raster(path, width, height, bands, origin_x, origin_y, gsd=1.0):
    rng = np.random.default_rng(0)
    data = rng.uniform(10, 250, size=(bands, height, width)).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=bands, dtype="float32",
        crs="EPSG:32643", transform=from_origin(origin_x, origin_y, gsd, gsd),
    ) as dst:
        dst.write(data)


def _frame_bounds(result):
    profile = result.output_profile
    t = profile["transform"]
    return (t.c, t.f + t.e * profile["height"], t.c + t.a * profile["width"], t.f)


@pytest.fixture
def small_ms_inside_large_rgb(tmp_path):
    """Real-world shape of Farm B A007: the RGB flight covers far more ground than the MS."""
    rgb_path, ms_path = tmp_path / "rgb.tif", tmp_path / "ms.tif"
    _write_raster(rgb_path, 200, 100, 3, 500000.0, 3000100.0)   # x 500000..500200, y 3000000..3000100
    _write_raster(ms_path, 100, 50, 4, 500050.0, 3000075.0)     # x 500050..500150, y 3000025..3000075
    return rgb_path, ms_path


def test_frame_leaves_room_for_the_real_shift_beyond_the_apparent_overlap(small_ms_inside_large_rgb):
    """GPS-only ODM runs misplace the MS by 2-6 m. The frame must extend past the
    apparent overlap so the correctly aligned MS isn't clipped at its edge."""
    cfg = AlignmentConfig(resolution_mode=ResolutionMode.MS_NATIVE)
    rgb_meta, ms_meta = validate_inputs(*small_ms_inside_large_rgb, cfg)
    result = coarse_align(rgb_meta, ms_meta, cfg)

    minx, miny, maxx, maxy = _frame_bounds(result)
    assert (minx, miny, maxx, maxy) == pytest.approx((500040.0, 3000015.0, 500160.0, 3000085.0))


def test_frame_margin_never_extends_past_the_rgb_reference(small_ms_inside_large_rgb):
    cfg = AlignmentConfig(resolution_mode=ResolutionMode.MS_NATIVE, coarse=CoarseAlignmentConfig(overlap_margin_m=500.0))
    rgb_meta, ms_meta = validate_inputs(*small_ms_inside_large_rgb, cfg)
    result = coarse_align(rgb_meta, ms_meta, cfg)

    assert _frame_bounds(result) == pytest.approx((500000.0, 3000000.0, 500200.0, 3000100.0))
