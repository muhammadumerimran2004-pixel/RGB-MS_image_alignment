from pathlib import Path
import pytest

from drone_alignment.io.gcp_io import read_gcp_csv, read_qgis_points


def test_read_gcp_csv_parses_rows_with_default_control_role(tmp_path: Path):
    csv_path = tmp_path / "gcps.csv"
    csv_path.write_text(
        "id,rgb_x,rgb_y,ms_x,ms_y\n"
        "P1,100.0,200.0,50.0,100.0\n"
        "P2,300.0,400.0,150.0,200.0\n",
        encoding="utf-8",
    )
    gcps = read_gcp_csv(csv_path)
    assert len(gcps) == 2
    assert gcps[0].gcp_id == "P1"
    assert gcps[0].rgb == (100.0, 200.0)
    assert gcps[0].ms == (50.0, 100.0)
    assert gcps[0].role == "control"


def test_read_gcp_csv_parses_explicit_check_role(tmp_path: Path):
    csv_path = tmp_path / "gcps.csv"
    csv_path.write_text(
        "id,rgb_x,rgb_y,ms_x,ms_y,role\n"
        "P1,100.0,200.0,50.0,100.0,control\n"
        "C1,300.0,400.0,150.0,200.0,check\n",
        encoding="utf-8",
    )
    gcps = read_gcp_csv(csv_path)
    assert gcps[1].gcp_id == "C1"
    assert gcps[1].role == "check"


def test_read_gcp_csv_missing_column_raises(tmp_path: Path):
    csv_path = tmp_path / "gcps.csv"
    csv_path.write_text("id,rgb_x,rgb_y,ms_x\nP1,1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        read_gcp_csv(csv_path)


def test_read_gcp_csv_empty_raises(tmp_path: Path):
    csv_path = tmp_path / "gcps.csv"
    csv_path.write_text("id,rgb_x,rgb_y,ms_x,ms_y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        read_gcp_csv(csv_path)


def test_read_gcp_csv_unknown_role_raises(tmp_path: Path):
    csv_path = tmp_path / "gcps.csv"
    csv_path.write_text(
        "id,rgb_x,rgb_y,ms_x,ms_y,role\nP1,1,2,3,4,bogus\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown role"):
        read_gcp_csv(csv_path)


def test_read_qgis_points_map_units(tmp_path: Path):
    points_path = tmp_path / "test.points"
    points_path.write_text(
        "#mapX,mapY,sourceX,sourceY,enable\n"
        "500010.0,3000010.0,500005.0,3000005.0,1\n"
        "500020.0,3000020.0,500015.0,3000015.0,1\n",
        encoding="utf-8",
    )
    gcps = read_qgis_points(points_path, source_units="map")
    assert len(gcps) == 2
    assert gcps[0].rgb == (500010.0, 3000010.0)
    assert gcps[0].ms == (500005.0, 3000005.0)


def test_read_qgis_points_pixel_units_negates_source_y(tmp_path: Path):
    points_path = tmp_path / "test.points"
    points_path.write_text(
        "#mapX,mapY,sourceX,sourceY,enable\n"
        "500010.0,3000010.0,100.0,-50.0,1\n",
        encoding="utf-8",
    )
    gcps = read_qgis_points(points_path, source_units="pixel")
    assert gcps[0].ms == (100.0, 50.0)


def test_read_qgis_points_skips_disabled_rows(tmp_path: Path):
    points_path = tmp_path / "test.points"
    points_path.write_text(
        "#mapX,mapY,sourceX,sourceY,enable\n"
        "500010.0,3000010.0,100.0,-50.0,1\n"
        "500020.0,3000020.0,200.0,-60.0,0\n",
        encoding="utf-8",
    )
    gcps = read_qgis_points(points_path, source_units="pixel")
    assert len(gcps) == 1
    assert gcps[0].ms == (100.0, 50.0)


def test_read_qgis_points_no_enabled_rows_raises(tmp_path: Path):
    points_path = tmp_path / "test.points"
    points_path.write_text(
        "#mapX,mapY,sourceX,sourceY,enable\n500010.0,3000010.0,100.0,-50.0,0\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no enabled rows"):
        read_qgis_points(points_path, source_units="pixel")


def test_read_qgis_points_missing_column_raises(tmp_path: Path):
    points_path = tmp_path / "test.points"
    points_path.write_text("#mapX,mapY,sourceX,enable\n1,2,3,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        read_qgis_points(points_path, source_units="map")
