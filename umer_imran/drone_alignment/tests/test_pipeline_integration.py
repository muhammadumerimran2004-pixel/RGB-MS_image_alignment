from pathlib import Path
import pytest
import rasterio

from drone_alignment.config.schema import AlignmentConfig, ResolutionMode
from drone_alignment.pipeline import align_orthomosaics


def test_pipeline_integration_synthetic(synthetic_geo_tiff_pair, tmp_path: Path):
    output_dir = tmp_path / "aligned_out"
    config = AlignmentConfig(
        resolution_mode=ResolutionMode.MS_NATIVE,
        ms_red_band_index=1,
        rgb_red_band_index=1,
    )

    result = align_orthomosaics(
        rgb_path=synthetic_geo_tiff_pair["rgb_path"],
        ms_path=synthetic_geo_tiff_pair["ms_path"],
        output_dir=output_dir,
        config=config,
    )

    assert result.aligned_ms_path.exists()
    assert result.report_json_path.exists()
    assert result.preview_image_path.exists()

    with rasterio.open(result.aligned_ms_path) as dst:
        assert dst.count == 4
        assert dst.crs == rasterio.crs.CRS.from_string(synthetic_geo_tiff_pair["crs"])
        assert dst.width > 0
        assert dst.height > 0
