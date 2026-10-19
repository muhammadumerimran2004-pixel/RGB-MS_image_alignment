"""Flight-date chain, positioning evidence and the confirm-then-remember flow."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from PIL import Image

from sentinel_patch.config.schema import DroneMetadataConfig
from sentinel_patch.date_from_path import PathEstimate
from sentinel_patch.drone_metadata import (
    make_console_confirmer, read_image_headers, resolve_drone_metadata, sidecar_path,
)
from sentinel_patch.errors import ConfigurationError, FlightDateUnknownError

TODAY = date(2026, 10, 18)


def photo(path: Path, when: str | None, rtk_flag: int | None = None) -> Path:
    exif = Image.Exif()
    if when:
        exif.get_ifd(0x8769)[36867] = when
    xmp = (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
           f'drone-dji:RtkFlag="{rtk_flag}"/></x:xmpmeta>').encode() if rtk_flag is not None else None
    kwargs = {"exif": exif}
    if xmp:
        kwargs["xmp"] = xmp
    Image.new("RGB", (8, 8)).save(path, "JPEG", **kwargs)
    return path


def photos(folder: Path, whens, flags=None):
    folder.mkdir(parents=True, exist_ok=True)
    for index, when in enumerate(whens):
        photo(folder / f"DJI_{index:04d}.JPG", when, None if flags is None else flags[index])
    return folder


@pytest.fixture
def ortho(tmp_path):
    folder = tmp_path / "Okara" / "Field7"
    folder.mkdir(parents=True)
    return folder / "rgb.tif", folder / "ms.tif"


def resolve(ortho, **kwargs):
    return resolve_drone_metadata(rgb_path=ortho[0], ms_path=ortho[1], today=TODAY, **kwargs)


def test_argument_wins_over_everything(ortho, tmp_path):
    sidecar_path(ortho[0]).write_text(json.dumps({"flight_date": "2026-09-01"}))
    images = photos(tmp_path / "photos", ["2026:10:05 09:00:00"])
    found = resolve(ortho, flight_date=date(2026, 10, 10), images_dir=images)
    assert (found.flight_date, found.flight_date_source) == (date(2026, 10, 10), "argument")


def test_sidecar_is_used_without_any_flag(ortho):
    sidecar_path(ortho[0]).write_text(json.dumps({"flight_date": "2026-10-09", "positioning": "rtk"}))
    found = resolve(ortho)
    assert (found.flight_date, found.flight_date_source) == (date(2026, 10, 9), "flight.json")
    assert (found.positioning, found.positioning_source) == ("rtk", "flight.json")


def test_photo_exif_date_is_the_median_and_a_multi_day_flight_is_flagged(ortho, tmp_path):
    images = photos(tmp_path / "photos", ["2026:10:09 23:50:00", "2026:10:10 08:00:00", "2026:10:10 08:05:00"])
    found = resolve(ortho, images_dir=images)
    assert (found.flight_date, found.flight_date_source) == (date(2026, 10, 10), "image_exif")
    assert "MULTI_DAY_FLIGHT" in found.warnings


def test_odm_project_images_folder_is_used_and_gcp_list_sets_positioning(ortho, tmp_path):
    odm = tmp_path / "odm"
    photos(odm / "images", ["2026:10:10 08:00:00"])
    (odm / "gcp_list.txt").write_text("EPSG:32643\n")
    found = resolve(ortho, odm_project=odm)
    assert found.flight_date == date(2026, 10, 10) and (found.positioning, found.positioning_source) == ("gcp", "odm_gcp_list")


def test_rtk_fixed_in_nearly_all_photos_means_rtk(ortho, tmp_path):
    images = photos(tmp_path / "p", ["2026:10:10 08:00:00"] * 10, flags=[50] * 10)
    found = resolve(ortho, images_dir=images)
    assert (found.positioning, found.positioning_source, found.rtk_fixed_fraction) == ("rtk", "image_xmp", 1.0)


def test_rtk_flag_mostly_float_is_gnss_only_with_a_warning(ortho, tmp_path):
    images = photos(tmp_path / "p", ["2026:10:10 08:00:00"] * 10, flags=[50, 50, 34, 34, 34, 16, 16, 16, 16, 16])
    found = resolve(ortho, images_dir=images)
    assert found.positioning == "gnss_only" and "RTK_NOT_FIXED" in found.warnings and found.rtk_fixed_fraction == 0.2


def test_photos_without_rtk_tags_leave_positioning_unknown(ortho, tmp_path):
    found = resolve(ortho, images_dir=photos(tmp_path / "p", ["2026:10:10 08:00:00"]))
    assert (found.positioning, found.rtk_fixed_fraction) == ("unknown", None)


def test_positioning_argument_overrides_detection(ortho, tmp_path):
    images = photos(tmp_path / "p", ["2026:10:10 08:00:00"] * 3, flags=[50] * 3)
    assert resolve(ortho, images_dir=images, positioning="ppk").positioning == "ppk"
    with pytest.raises(ConfigurationError):
        resolve(ortho, flight_date=date(2026, 10, 10), positioning="magic")


def test_missing_images_folder_is_a_configuration_error(ortho, tmp_path):
    with pytest.raises(ConfigurationError, match="does not exist"):
        resolve(ortho, images_dir=tmp_path / "nope")


def test_header_reader_skips_photos_without_exif(tmp_path):
    folder = photos(tmp_path / "p", ["2026:10:10 08:00:00", None])
    headers = read_image_headers(folder)
    assert headers.sampled == 2 and len(headers.captured) == 1


def test_no_source_fails_with_the_options_listed(ortho):
    with pytest.raises(FlightDateUnknownError, match="--flight-date.*flight.json.*--images-dir"):
        resolve(ortho)


# -- folder-name estimate -----------------------------------------------------------------------

@pytest.fixture
def dated(tmp_path):
    folder = tmp_path / "Okara" / "2026-10-10_Field7"
    folder.mkdir(parents=True)
    return folder / "rgb.tif", folder / "ms.tif"


def test_estimate_is_not_used_without_confirmation_or_permission(dated):
    with pytest.raises(FlightDateUnknownError, match="2026-10-10"):
        resolve(dated)


def test_estimate_is_used_unattended_only_when_allowed_and_unambiguous(dated, tmp_path):
    allowed = DroneMetadataConfig(accept_path_date_estimate=True)
    found = resolve(dated, cfg=allowed)
    assert found.flight_date == date(2026, 10, 10) and found.flight_date_source == "path_estimate_unattended"
    assert "FLIGHT_DATE_ESTIMATED" in found.warnings and found.path_estimate["confirmed_by_user"] is False

    ambiguous = tmp_path / "10-03-2026"
    ambiguous.mkdir()
    with pytest.raises(FlightDateUnknownError, match="ambiguous"):
        resolve((ambiguous / "rgb.tif", ambiguous / "ms.tif"), cfg=allowed)


def test_confirmed_estimate_is_remembered_in_the_sidecar(dated):
    asked = []
    found = resolve(dated, confirm=lambda estimate: (asked.append(estimate), (estimate.best.value, False))[1])
    assert found.flight_date_source == "path_estimate_confirmed" and "FLIGHT_DATE_ESTIMATED" in found.warnings
    saved = json.loads(sidecar_path(dated[0]).read_text())
    assert saved["flight_date"] == "2026-10-10" and saved["flight_date_source"] == "path_estimate_confirmed"

    again = resolve(dated, confirm=lambda estimate: pytest.fail("must not ask a second time"))
    assert (again.flight_date, again.flight_date_source) == (date(2026, 10, 10), "path_estimate_confirmed")
    assert len(asked) == 1


def test_typed_date_is_saved_as_user_entered(dated):
    found = resolve(dated, confirm=lambda estimate: (date(2026, 10, 11), True))
    assert found.flight_date_source == "user_entered"
    assert json.loads(sidecar_path(dated[0]).read_text())["flight_date_source"] == "user_entered"


def test_declined_estimate_fails_the_stage_and_saves_nothing(dated):
    with pytest.raises(FlightDateUnknownError):
        resolve(dated, confirm=lambda estimate: None)
    assert not sidecar_path(dated[0]).exists()


def test_sidecar_is_not_written_when_disabled(dated):
    resolve(dated, cfg=DroneMetadataConfig(write_sidecar=False), confirm=lambda e: (e.best.value, False))
    assert not sidecar_path(dated[0]).exists()


def test_unwritable_folder_does_not_break_the_run(dated, monkeypatch):
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    found = resolve(dated, confirm=lambda e: (e.best.value, False))
    assert found.flight_date == date(2026, 10, 10)


# -- console confirmation -------------------------------------------------------------------------

def run_confirmer(estimate, answers):
    queue, lines = list(answers), []
    confirm = make_console_confirmer(lambda question: queue.pop(0), lines.append)
    return confirm(estimate), lines


def single(tmp_path_factory=None):
    from sentinel_patch.date_from_path import estimate_flight_date
    return estimate_flight_date([Path("D:/f/2026-10-10_Field7/rgb.tif")], today=TODAY)


def test_enter_and_yes_accept_the_single_estimate():
    for answer in ("", "y", "Yes"):
        assert run_confirmer(single(), [answer])[0] == (date(2026, 10, 10), False)


def test_no_declines():
    assert run_confirmer(single(), ["n"])[0] is None


def test_typed_date_is_accepted_and_garbage_is_asked_again():
    result, lines = run_confirmer(single(), ["tomorrow", "2026-10-11"])
    assert result == (date(2026, 10, 11), True) and any("Please answer" in line for line in lines)


def test_several_candidates_are_listed_and_a_number_picks_one():
    from sentinel_patch.date_from_path import estimate_flight_date
    many = estimate_flight_date([Path("D:/a/2026-10-10/rgb.tif"), Path("D:/b/2026-10-12/ms.tif")], today=TODAY)
    result, lines = run_confirmer(many, ["2"])
    assert result == (many.candidates[1].value, False) and sum("[" in line for line in lines) == 2
    assert run_confirmer(many, [""])[0] is None  # Enter skips when there is no single obvious choice


def test_ambiguous_estimate_shows_both_readings():
    from sentinel_patch.date_from_path import estimate_flight_date
    estimate = estimate_flight_date([Path("D:/f/10-03-2026/rgb.tif")], today=TODAY)
    _, lines = run_confirmer(estimate, ["n"])
    assert any("10 March 2026" in line and "03 October 2026" in line for line in lines)
