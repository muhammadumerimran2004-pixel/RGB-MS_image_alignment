"""The Sentinel stage as the command line runs it, and its isolation from the alignment run."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from sentinel_patch.cli_stage import DEFAULT_MS_BANDS, StageOptions, parse_ms_bands, run_stage
from sentinel_patch.errors import ConfigurationError

from . import scene_files

ERROR_M = (15.0, -12.0)
FLIGHT = date(2026, 10, 10)


@pytest.fixture(scope="module")
def scene():
    return scene_files.full_scene()


@pytest.fixture
def setup(tmp_path, scene):
    ms, rgb = scene_files.write_drone_rasters(tmp_path / "drone", scene)
    s2 = scene_files.write_s2_scene(tmp_path / "s2", scene, content_offset_m=ERROR_M)
    config = tmp_path / "sentinel.yaml"
    config.write_text(f"sentinel_patch:\n  local_folder: '{s2.as_posix()}'\n  registration:\n    arosics_mode: 'off'\n",
                      encoding="utf-8")
    return SimpleNamespace(ms=ms, rgb=rgb, config=config, out=tmp_path / "aligned", tmp=tmp_path)


def stage(setup, **option_overrides):
    lines = []
    options = StageOptions(**{"flight_date": FLIGHT, "config_path": setup.config, "positioning": "rtk", **option_overrides})
    outcome = run_stage(aligned_ms_path=setup.ms, rgb_path=setup.rgb, ms_path=setup.ms, out_dir=setup.out,
                        options=options, echo=lines.append)
    return outcome, "\n".join(lines)


def test_band_map_parsing():
    assert parse_ms_bands(DEFAULT_MS_BANDS) == {"red": 1, "green": 2, "nir": 3}
    assert parse_ms_bands(" nir = 4 , red=2 ") == {"nir": 4, "red": 2}
    for bad in ("nir", "nir=0", "nir=x", "swir=4", "nir=4=5"):
        with pytest.raises(ConfigurationError, match="Invalid band map"):
            parse_ms_bands(bad)


def test_verified_run_reports_the_shift_and_products(setup):
    outcome, text = stage(setup, ms_bands="green=1,red=2,red_edge=3,nir=4")
    assert outcome.ok and outcome.status == "verified" and not outcome.strict_failure
    assert outcome.report_block["shift_m"] == pytest.approx([-ERROR_M[0], -ERROR_M[1]], abs=1.5)
    assert "Shift (E, N)" in text and "analytic_10m" in text and "Positioning:  rtk  (argument)" in text


def test_large_shift_without_rtk_evidence_is_flagged_and_with_it_is_not(setup):
    flagged, _ = stage(setup, ms_bands="green=1,red=2,red_edge=3,nir=4", positioning="gnss_only")
    assert "LARGE_SHIFT_DRONE_FRAME_SUSPECT" in flagged.report_block["warnings"]  # 19 m > 15 m
    trusted, _ = stage(setup, ms_bands="green=1,red=2,red_edge=3,nir=4", positioning="rtk")
    assert "LARGE_SHIFT_DRONE_FRAME_SUSPECT" not in trusted.report_block["warnings"]


def test_report_records_the_drone_metadata(setup):
    outcome, _ = stage(setup, ms_bands="green=1,red=2,red_edge=3,nir=4", positioning="ppk")
    report = json.loads(open(outcome.report_block["report_path"], encoding="utf-8").read())
    assert report["drone_metadata"]["flight_date_source"] == "argument"
    assert report["drone_metadata"]["positioning"] == "ppk"


def test_unknown_flight_date_fails_only_the_stage(setup):
    outcome, text = stage(setup, flight_date=None, ms_bands="green=1,red=2,red_edge=3,nir=4")
    assert not outcome.ok and outcome.reason_code == "FLIGHT_DATE_UNKNOWN" and not outcome.strict_failure
    assert "unaffected" in text and outcome.report_block["status"] == "error"


def test_strict_turns_a_missing_scene_into_a_failure(setup):
    outcome, _ = stage(setup, flight_date=date(2026, 12, 25), strict=True, ms_bands="green=1,red=2,red_edge=3,nir=4")
    assert outcome.status == "no_scene" and outcome.strict_failure
    assert not stage(setup, flight_date=date(2026, 12, 25), ms_bands="green=1,red=2,red_edge=3,nir=4")[0].strict_failure


def test_a_bug_inside_the_stage_is_contained(setup, monkeypatch):
    import sentinel_patch.cli_stage as module
    monkeypatch.setattr(module, "run_sentinel_patch", lambda *a, **k: 1 / 0)
    outcome, text = stage(setup, strict=True)
    assert outcome.reason_code == "SENTINEL_UNEXPECTED_ERROR" and "ZeroDivisionError" in text and outcome.strict_failure


def test_bad_band_map_is_a_contained_configuration_error(setup):
    outcome, _ = stage(setup, ms_bands="nir=banana")
    assert outcome.reason_code == "CONFIG_INVALID"


# -- through the real click command, with the alignment itself stubbed ---------------------------------

@pytest.fixture
def cli(setup, monkeypatch):
    from drone_alignment import cli as module

    report = setup.tmp / "alignment_report.json"
    report.write_text(json.dumps({"applied_alignment_mode": "automated"}), encoding="utf-8")
    result = SimpleNamespace(
        aligned_ms_path=setup.ms, preview_image_path=setup.tmp / "p.png", report_json_path=report,
        spatial_residual_report=SimpleNamespace(status="ok", global_rmse_px=0.5, edge_corner_rmse_px=0.7))
    monkeypatch.setattr(module, "align_orthomosaics", lambda **kwargs: result)

    def invoke(*extra):
        return CliRunner().invoke(module.main, [str(setup.rgb), str(setup.ms), "-o", str(setup.out), "-m", "automated", *extra])
    invoke.report = report
    return invoke


def test_without_the_flag_nothing_changes(cli):
    result = cli()
    assert result.exit_code == 0 and "Sentinel" not in result.output
    assert "sentinel_patch" not in json.loads(cli.report.read_text())


def test_flag_runs_the_stage_and_records_it_in_the_alignment_report(cli, setup):
    result = cli("--sentinel", "--flight-date", "2026-10-10", "--sentinel-config", str(setup.config),
                 "--sentinel-ms-bands", "green=1,red=2,red_edge=3,nir=4", "--drone-positioning", "rtk")
    assert result.exit_code == 0, result.output
    block = json.loads(cli.report.read_text())["sentinel_patch"]
    assert block["status"] == "verified" and block["shift_m"] == pytest.approx([-15, 12], abs=1.5)
    assert "Aligned MS File" in result.output and "Sentinel-2 patch" in result.output


def test_sentinel_failure_never_fails_the_alignment_but_strict_does(cli, setup):
    args = ("--sentinel", "--flight-date", "2026-12-25", "--sentinel-config", str(setup.config),
            "--sentinel-ms-bands", "green=1,red=2,red_edge=3,nir=4")
    soft = cli(*args)
    assert soft.exit_code == 0 and json.loads(cli.report.read_text())["sentinel_patch"]["status"] == "no_scene"
    assert cli(*args, "--sentinel-strict").exit_code == 3


def test_options_without_the_flag_are_called_out(cli):
    result = cli("--flight-date", "2026-10-10")
    assert result.exit_code == 0 and "ignored" in result.output


def test_bad_flight_date_is_rejected_up_front(cli):
    result = cli("--sentinel", "--flight-date", "10/10/2026")
    assert result.exit_code == 2 and "YYYY-MM-DD" in result.output


def test_band_number_the_file_does_not_have_is_refused_with_its_band_count(setup):
    outcome, text = stage(setup, ms_bands="red=2,nir=9")  # the synthetic MS file has 4 bands
    assert outcome.reason_code == "CONFIG_INVALID" and "4 band(s)" in text
