import json
from pathlib import Path

import numpy as np
import rasterio
from click.testing import CliRunner

from drone_alignment.cli import main


def _write_points_csv(path: Path, rows: list[tuple[str, float, float, float, float, str]]) -> None:
    lines = ["id,rgb_x,rgb_y,ms_x,ms_y,role"]
    for gcp_id, rgb_x, rgb_y, ms_x, ms_y, role in rows:
        lines.append(f"{gcp_id},{rgb_x},{rgb_y},{ms_x},{ms_y},{role}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cli_gcp_file_csv_pixel_units_runs_manual_alignment(synthetic_geo_tiff_pair, tmp_path: Path):
    gcp_csv = tmp_path / "gcps.csv"
    _write_points_csv(gcp_csv, [
        ("P1", 100.0, 100.0, 52.0, 51.0, "control"),
        ("P2", 400.0, 120.0, 203.0, 60.0, "control"),
        ("P3", 150.0, 400.0, 76.0, 202.0, "control"),
    ])

    result = CliRunner().invoke(main, [
        str(synthetic_geo_tiff_pair["rgb_path"]), str(synthetic_geo_tiff_pair["ms_path"]),
        "--mode", "manual",
        "--output-dir", str(tmp_path / "out"),
        "--gcp-file", str(gcp_csv),
        "--gcp-format", "csv",
        "--gcp-source-units", "pixel",
    ])

    assert result.exit_code == 0, result.output
    assert "Loaded 3 point(s)" in result.output
    assert "Applied Mode:     manual_affine" in result.output

    report_path = tmp_path / "out" / "ms_aligned_report.json"
    reports = list((tmp_path / "out").glob("*_alignment_report.json"))
    assert len(reports) == 1
    with open(reports[0], encoding="utf-8") as f:
        report = json.load(f)
    mcp = report["manual_control_points"]
    assert mcp["coordinate_mode"] == "pixel"
    assert [p["id"] for p in mcp["points"]] == ["P1", "P2", "P3"]
    assert all(p["role"] == "control" for p in mcp["points"])


def test_cli_gcp_file_csv_with_check_points(synthetic_geo_tiff_pair, tmp_path: Path):
    gcp_csv = tmp_path / "gcps.csv"
    _write_points_csv(gcp_csv, [
        ("P1", 100.0, 100.0, 52.0, 51.0, "control"),
        ("P2", 400.0, 120.0, 203.0, 60.0, "control"),
        ("P3", 150.0, 400.0, 76.0, 202.0, "control"),
        ("C1", 300.0, 300.0, 152.0, 152.0, "check"),
    ])

    result = CliRunner().invoke(main, [
        str(synthetic_geo_tiff_pair["rgb_path"]), str(synthetic_geo_tiff_pair["ms_path"]),
        "--mode", "manual",
        "--output-dir", str(tmp_path / "out"),
        "--gcp-file", str(gcp_csv),
        "--gcp-source-units", "pixel",
    ])

    assert result.exit_code == 0, result.output
    assert "1 check, 3 control" in result.output
    reports = list((tmp_path / "out").glob("*_alignment_report.json"))
    with open(reports[0], encoding="utf-8") as f:
        report = json.load(f)
    roles = [p["role"] for p in report["manual_control_points"]["points"]]
    assert roles == ["control", "control", "control", "check"]


def test_cli_gcp_file_qgis_map_units(synthetic_geo_tiff_pair, tmp_path: Path):
    rgb_path = synthetic_geo_tiff_pair["rgb_path"]
    ms_path = synthetic_geo_tiff_pair["ms_path"]
    with rasterio.open(rgb_path) as rgb_src, rasterio.open(ms_path) as ms_src:
        map_rgb = [rgb_src.transform * (100.0, 100.0), rgb_src.transform * (400.0, 120.0)]
        map_ms = [ms_src.transform * (49.5, 49.5), ms_src.transform * (199.5, 59.5)]

    points_file = tmp_path / "test.points"
    lines = ["#mapX,mapY,sourceX,sourceY,enable"]
    for (mx, my), (sx, sy) in zip(map_rgb, map_ms):
        lines.append(f"{mx},{my},{sx},{sy},1")
    points_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = CliRunner().invoke(main, [
        str(rgb_path), str(ms_path),
        "--mode", "manual",
        "--output-dir", str(tmp_path / "out"),
        "--gcp-file", str(points_file),
        "--gcp-format", "qgis",
        "--gcp-source-units", "map",
    ])

    assert result.exit_code == 0, result.output
    reports = list((tmp_path / "out").glob("*_alignment_report.json"))
    with open(reports[0], encoding="utf-8") as f:
        report = json.load(f)
    assert report["manual_control_points"]["coordinate_mode"] == "map"


def test_cli_gcp_file_qgis_pixel_source_units_converts_to_map(synthetic_geo_tiff_pair, tmp_path: Path):
    rgb_path = synthetic_geo_tiff_pair["rgb_path"]
    ms_path = synthetic_geo_tiff_pair["ms_path"]
    with rasterio.open(rgb_path) as rgb_src:
        map_rgb = [rgb_src.transform * (100.0, 100.0), rgb_src.transform * (400.0, 120.0)]

    points_file = tmp_path / "test.points"
    lines = ["#mapX,mapY,sourceX,sourceY,enable"]
    # QGIS stores source (pixel) rows as negative Y.
    for (mx, my), (sx, sy) in zip(map_rgb, [(49.5, 49.5), (199.5, 59.5)]):
        lines.append(f"{mx},{my},{sx},{-sy},1")
    points_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = CliRunner().invoke(main, [
        str(rgb_path), str(ms_path),
        "--mode", "manual",
        "--output-dir", str(tmp_path / "out"),
        "--gcp-file", str(points_file),
        "--gcp-format", "qgis",
        "--gcp-source-units", "pixel",
    ])

    assert result.exit_code == 0, result.output
    reports = list((tmp_path / "out").glob("*_alignment_report.json"))
    with open(reports[0], encoding="utf-8") as f:
        report = json.load(f)
    assert report["manual_control_points"]["coordinate_mode"] == "map"
    assert (tmp_path / "out").exists()


def test_cli_manual_model_flag_forces_affine_over_tps(synthetic_geo_tiff_pair, tmp_path: Path):
    rng = np.random.default_rng(5)
    ms_pts = np.array([
        (25.0, 25.0), (225.0, 30.0), (30.0, 225.0), (225.0, 225.0),
        (125.0, 25.0), (25.0, 125.0), (225.0, 125.0), (125.0, 225.0),
    ])
    baseline = np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 1.0]])
    rgb_linear = np.column_stack([
        baseline[0, 0] * ms_pts[:, 0] + baseline[0, 1] * ms_pts[:, 1] + baseline[0, 2],
        baseline[1, 0] * ms_pts[:, 0] + baseline[1, 1] * ms_pts[:, 1] + baseline[1, 2],
    ])
    bend = 3.0 * np.sin(ms_pts[:, 0] / 50.0)[:, None] * np.array([1.0, 0.3]) + \
        3.0 * np.cos(ms_pts[:, 1] / 50.0)[:, None] * np.array([0.3, 1.0])
    rgb_pts = rgb_linear + bend

    gcp_csv = tmp_path / "gcps.csv"
    rows = [(f"P{i}", rgb_pts[i, 0], rgb_pts[i, 1], ms_pts[i, 0], ms_pts[i, 1], "control") for i in range(8)]
    _write_points_csv(gcp_csv, rows)

    result = CliRunner().invoke(main, [
        str(synthetic_geo_tiff_pair["rgb_path"]), str(synthetic_geo_tiff_pair["ms_path"]),
        "--mode", "manual",
        "--output-dir", str(tmp_path / "out"),
        "--gcp-file", str(gcp_csv),
        "--gcp-source-units", "pixel",
        "--manual-model", "affine",
    ])

    assert result.exit_code == 0, result.output
    reports = list((tmp_path / "out").glob("*_alignment_report.json"))
    with open(reports[0], encoding="utf-8") as f:
        report = json.load(f)
    assert report["manual_control_points"]["model"] == "affine"
    assert "forced" in report["manual_control_points"]["model_reason"].lower()


def test_cli_arosics_local_disable_flag_changes_banner(synthetic_geo_tiff_pair, tmp_path: Path):
    result = CliRunner().invoke(main, [
        str(synthetic_geo_tiff_pair["rgb_path"]), str(synthetic_geo_tiff_pair["ms_path"]),
        "--mode", "manual",
        "--output-dir", str(tmp_path / "out"),
        "--no-arosics-local",
        "--gcp-file", str(_make_minimal_gcp_csv(tmp_path)),
        "--gcp-source-units", "pixel",
    ])
    assert "verified global only (AROSICS local refinement disabled)" in result.output


def _make_minimal_gcp_csv(tmp_path: Path) -> Path:
    gcp_csv = tmp_path / "minimal_gcps.csv"
    _write_points_csv(gcp_csv, [
        ("P1", 100.0, 100.0, 52.0, 51.0, "control"),
        ("P2", 400.0, 120.0, 203.0, 60.0, "control"),
    ])
    return gcp_csv


def test_cli_advertises_new_manual_and_arosics_local_flags():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for flag in (
        "--arosics-local", "--arosics-max-shift-m", "--arosics-warp-engine", "--crop-row-period-m",
        "--gcp-file", "--gcp-format", "--gcp-source-units", "--manual-model", "--pixel-convention",
    ):
        assert flag in result.output
