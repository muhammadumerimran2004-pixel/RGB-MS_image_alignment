import pytest
from rasterio.windows import Window

from drone_alignment.io.reader import read_metadata, read_band, read_band_windowed, read_band_with_mask
import numpy as np
import rasterio
from rasterio.transform import from_origin


def test_read_metadata(synthetic_geo_tiff_pair):
    meta = read_metadata(synthetic_geo_tiff_pair["rgb_path"])
    assert meta.width == 512
    assert meta.height == 512
    assert meta.band_count == 3
    assert meta.dtype == "float32"
    assert pytest.approx(meta.gsd, rel=1e-3) == 0.03


def test_read_band(synthetic_geo_tiff_pair):
    band_arr = read_band(synthetic_geo_tiff_pair["rgb_path"], 1)
    assert band_arr.shape == (512, 512)
    assert band_arr.dtype == "float32"

    with pytest.raises(IndexError):
        read_band(synthetic_geo_tiff_pair["rgb_path"], 10)


def test_read_band_windowed(synthetic_geo_tiff_pair):
    win = Window(col_off=10, row_off=10, width=50, height=50)
    band_win = read_band_windowed(synthetic_geo_tiff_pair["ms_path"], 1, win)
    assert band_win.shape == (50, 50)


def test_read_band_with_alpha_mask_excludes_nonzero_fill(tmp_path):
    path = tmp_path / "alpha_fill.tif"
    data = np.full((5, 10, 10), 0.05, dtype=np.float32)
    data[:4, :, :2] = np.float32(2**32)
    data[4] = 255
    data[4, :, :2] = 0
    with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=5,
                       dtype="float32", crs="EPSG:32643", transform=from_origin(0, 10, 1, 1)) as dst:
        dst.write(data)
        dst.colorinterp = (
            rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.gray, rasterio.enums.ColorInterp.gray,
            rasterio.enums.ColorInterp.alpha,
        )

    values, valid = read_band_with_mask(path, 1)
    meta = read_metadata(path)

    assert np.all(~valid[:, :2])
    assert np.all(valid[:, 2:])
    assert np.all(values[:, :2] == np.float32(2**32))
    assert meta.alpha_band_index == 5
    assert 5 not in meta.spectral_band_indices
