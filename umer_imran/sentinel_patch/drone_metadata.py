"""Flight date and positioning evidence for a drone dataset (blueprint 6.9).

This never touches the alignment engine. Raw photos are an optional, metadata-only input: only their
EXIF/XMP headers are read, never their pixels.

Flight-date sources, first hit wins (the source is recorded in the report):
  1. the ``flight_date`` argument   2. ``flight.json`` next to the RGB orthophoto
  3. EXIF of the raw photos (``images_dir``, or ``<odm_project>/images``)
  4. a date estimated from folder and file names, confirmed by the user (``date_from_path``)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from sentinel_patch.config.schema import DroneMetadataConfig
from sentinel_patch.date_from_path import PathEstimate, estimate_flight_date, read_tiff_processing_date
from sentinel_patch.errors import ConfigurationError, FlightDateUnknownError

logger = logging.getLogger(__name__)

SIDECAR_NAME = "flight.json"
POSITIONING_VALUES = ("rtk", "ppk", "gcp", "gnss_only", "unknown")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff"}
MAX_IMAGES_SAMPLED = 200
XMP_READ_BYTES = 262_144  # DJI writes its XMP packet near the start of the file
RTK_FIXED_FLAG = 50
RTK_FIXED_MIN_FRACTION = 0.9
_RTK_FLAG = re.compile(rb"RtkFlag(?:\s*=\s*\"|>)\s*(\d+)")

SOURCE_ARGUMENT, SOURCE_SIDECAR, SOURCE_EXIF = "argument", "flight.json", "image_exif"
SOURCE_PATH_CONFIRMED, SOURCE_PATH_UNATTENDED, SOURCE_USER_ENTERED = (
    "path_estimate_confirmed", "path_estimate_unattended", "user_entered")
ESTIMATED_SOURCES = {SOURCE_PATH_CONFIRMED, SOURCE_PATH_UNATTENDED}


@dataclass(frozen=True)
class DroneMetadata:
    flight_date: date | None
    flight_date_source: str
    positioning: str = "unknown"
    positioning_source: str = "default"
    rtk_fixed_fraction: float | None = None
    warnings: tuple[str, ...] = ()
    path_estimate: dict | None = None

    def as_dict(self) -> dict:
        return {
            "flight_date_source": self.flight_date_source, "positioning": self.positioning,
            "positioning_source": self.positioning_source, "rtk_fixed_fraction": self.rtk_fixed_fraction,
            "path_estimate": self.path_estimate,
        }


# --- confirmation of an estimated date ---------------------------------------------------------------

# A confirmer receives the estimate and returns (date, user_typed_it) or None to decline.
Confirmer = Callable[[PathEstimate], "tuple[date, bool] | None"]


def parse_user_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text.strip())
    except ValueError:
        return None


def describe_estimate(estimate: PathEstimate) -> list[str]:
    lines = []
    for index, candidate in enumerate(estimate.candidates, start=1):
        readings = " or ".join(f"{d:%d %B %Y}" for d in candidate.ambiguous_readings)
        suffix = f" (ambiguous: {readings})" if candidate.ambiguous else ""
        lines.append(f"  [{index}] {candidate.value.isoformat()} from \"{candidate.component}\" "
                     f"(matched \"{candidate.matched_text}\"){suffix}")
    return lines


def make_console_confirmer(prompt: Callable[[str], str], echo: Callable[[str], None]) -> Confirmer:
    """Interactive confirmation; ``prompt`` returns the user's typed answer (empty = Enter)."""
    def confirm(estimate: PathEstimate) -> tuple[date, bool] | None:
        echo("Flight date not found in metadata. Estimated from the file/folder names:")
        for line in describe_estimate(estimate):
            echo(line)
        many = len(estimate.candidates) > 1
        question = ("Pick a number, type a date (YYYY-MM-DD), or press Enter to skip: " if many else
                    f"Use {estimate.best.value.isoformat()}? [Y]es / [n]o / or type a date (YYYY-MM-DD): ")
        while True:
            answer = prompt(question).strip()
            if answer.lower() in ("n", "no") or (many and answer == ""):
                return None
            if not many and answer.lower() in ("", "y", "yes"):
                return estimate.best.value, False
            if many and answer.isdigit() and 1 <= int(answer) <= len(estimate.candidates):
                return estimate.candidates[int(answer) - 1].value, False
            typed = parse_user_date(answer)
            if typed is not None:
                return typed, True
            echo("Please answer with a choice above or a date like 2026-10-10.")
    return confirm


# --- sidecar ----------------------------------------------------------------------------------------

def sidecar_path(rgb_path: Path) -> Path:
    return Path(rgb_path).parent / SIDECAR_NAME


def read_sidecar(rgb_path: Path) -> dict | None:
    path = sidecar_path(rgb_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return None


def write_sidecar(rgb_path: Path, flight_date: date, source: str, positioning: str | None = None) -> bool:
    """Remember a confirmed or typed date so the next run never asks again. False when the folder is read-only."""
    path = sidecar_path(rgb_path)
    payload = {**(read_sidecar(rgb_path) or {}), "flight_date": flight_date.isoformat(), "flight_date_source": source}
    if positioning:
        payload["positioning"] = positioning
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except OSError as exc:
        logger.warning("Could not save %s (%s); the date will be asked again next time.", path, exc)
        return False


# --- raw photo headers ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageHeaders:
    captured: tuple[datetime, ...]
    rtk_flags: tuple[int, ...]
    sampled: int


def _list_images(images_dir: Path) -> list[Path]:
    files = sorted(p for p in Path(images_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(files) <= MAX_IMAGES_SAMPLED:
        return files
    step = len(files) / MAX_IMAGES_SAMPLED
    return [files[int(i * step)] for i in range(MAX_IMAGES_SAMPLED)]


def _exif_datetime(path: Path) -> datetime | None:
    from PIL import Image
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            text = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal, else DateTime
        return datetime.strptime(str(text), "%Y:%m:%d %H:%M:%S") if text else None
    except Exception:  # noqa: BLE001  (a photo without readable EXIF is simply not evidence)
        return None


def _rtk_flag(path: Path) -> int | None:
    try:
        with open(path, "rb") as handle:
            found = _RTK_FLAG.search(handle.read(XMP_READ_BYTES))
        return int(found.group(1)) if found else None
    except OSError:
        return None


def read_image_headers(images_dir: Path) -> ImageHeaders:
    files = _list_images(images_dir)
    captured = [t for t in (_exif_datetime(p) for p in files) if t is not None]
    flags = [f for f in (_rtk_flag(p) for p in files) if f is not None]
    return ImageHeaders(tuple(captured), tuple(flags), len(files))


def flight_date_from_headers(headers: ImageHeaders) -> tuple[date, list[str]] | None:
    if not headers.captured:
        return None
    ordered = sorted(headers.captured)
    median = ordered[len(ordered) // 2]
    warnings = ["MULTI_DAY_FLIGHT"] if ordered[0].date() != ordered[-1].date() else []
    return median.date(), warnings


def positioning_from_headers(headers: ImageHeaders) -> tuple[str, float | None, list[str]]:
    """Evidence only: an RTK flag shows RTK was active, not that the camera positions are accurate."""
    if not headers.rtk_flags:
        return "unknown", None, []
    fixed = sum(1 for flag in headers.rtk_flags if flag == RTK_FIXED_FLAG) / len(headers.rtk_flags)
    if fixed >= RTK_FIXED_MIN_FRACTION:
        return "rtk", fixed, []
    return "gnss_only", fixed, ["RTK_NOT_FIXED"]


# --- the chain --------------------------------------------------------------------------------------

def _images_folder(images_dir: Path | None, odm_project: Path | None) -> Path | None:
    if images_dir is not None:
        return Path(images_dir)
    if odm_project is not None and (Path(odm_project) / "images").is_dir():
        return Path(odm_project) / "images"  # the ODM project layout keeps the raw photos here
    return None


def resolve_drone_metadata(
    *, rgb_path: Path, ms_path: Path, flight_date: date | None = None, images_dir: Path | None = None,
    odm_project: Path | None = None, positioning: str | None = None,
    cfg: DroneMetadataConfig | None = None, confirm: Confirmer | None = None,
    today: date | None = None,
) -> DroneMetadata:
    """Find the flight date (and positioning evidence). Raises ``FlightDateUnknownError`` when nothing works."""
    cfg = cfg or DroneMetadataConfig()
    if positioning is not None and positioning not in POSITIONING_VALUES:
        raise ConfigurationError(f"positioning must be one of {', '.join(POSITIONING_VALUES)}.")
    sidecar = read_sidecar(rgb_path) or {}
    warnings: list[str] = []

    headers = None
    folder = _images_folder(images_dir, odm_project)
    if folder is not None:
        if not folder.is_dir():
            raise ConfigurationError(f"images folder does not exist: {folder}")
        headers = read_image_headers(folder)

    # positioning: explicit > flight.json > ODM gcp_list.txt > raw-photo RTK flags > unknown
    rtk_fraction = None
    if positioning is not None:
        pos, pos_source = positioning, "argument"
    elif sidecar.get("positioning") in POSITIONING_VALUES:
        pos, pos_source = sidecar["positioning"], SOURCE_SIDECAR
    elif odm_project is not None and (Path(odm_project) / "gcp_list.txt").is_file():
        pos, pos_source = "gcp", "odm_gcp_list"
    elif headers is not None and headers.rtk_flags:
        pos, rtk_fraction, extra = positioning_from_headers(headers)
        pos_source = "image_xmp"
        warnings.extend(extra)
    else:
        pos, pos_source = "unknown", "default"
    if headers is not None and rtk_fraction is None and headers.rtk_flags:
        rtk_fraction = sum(1 for f in headers.rtk_flags if f == RTK_FIXED_FLAG) / len(headers.rtk_flags)

    def done(value: date, source: str, path_estimate: dict | None = None) -> DroneMetadata:
        if source in ESTIMATED_SOURCES:
            warnings.append("FLIGHT_DATE_ESTIMATED")
        return DroneMetadata(value, source, pos, pos_source, rtk_fraction, tuple(dict.fromkeys(warnings)), path_estimate)

    if flight_date is not None:
        return done(flight_date, SOURCE_ARGUMENT)
    if sidecar.get("flight_date"):
        try:
            stored = date.fromisoformat(str(sidecar["flight_date"]))
            return done(stored, SOURCE_SIDECAR if sidecar.get("flight_date_source") in (None, SOURCE_SIDECAR)
                        else str(sidecar["flight_date_source"]))
        except ValueError:
            logger.warning("Ignoring invalid flight_date in %s.", sidecar_path(rgb_path))
    if headers is not None:
        from_headers = flight_date_from_headers(headers)
        if from_headers is not None:
            warnings.extend(from_headers[1])
            return done(from_headers[0], SOURCE_EXIF)

    upper = read_tiff_processing_date(Path(rgb_path))
    ms_upper = read_tiff_processing_date(Path(ms_path))
    bounds = [b for b in (upper, ms_upper) if b is not None]
    estimate = estimate_flight_date([Path(rgb_path), Path(ms_path)], depth=cfg.path_scan_depth,
                                    date_order=cfg.date_order, upper_bound=min(bounds) if bounds else None, today=today)
    if estimate.best is not None:
        info = _estimate_info(estimate)
        if confirm is not None:
            answer = confirm(estimate)
            if answer is not None:
                chosen, typed = answer
                source = SOURCE_USER_ENTERED if typed else SOURCE_PATH_CONFIRMED
                if cfg.write_sidecar:
                    write_sidecar(rgb_path, chosen, source, pos if pos_source != "default" else None)
                return done(chosen, source, {**info, "confirmed_by_user": True})
        elif cfg.accept_path_date_estimate and estimate.usable_unattended:
            return done(estimate.best.value, SOURCE_PATH_UNATTENDED, {**info, "confirmed_by_user": False})

    raise FlightDateUnknownError(_unknown_message(estimate))


def _estimate_info(estimate: PathEstimate) -> dict:
    best = estimate.best
    return {"matched_text": best.matched_text, "path_component": best.component, "pattern": best.pattern,
            "confidence": best.confidence, "status": estimate.status,
            "ambiguous_readings": [d.isoformat() for d in best.ambiguous_readings]}


def _unknown_message(estimate: PathEstimate) -> str:
    text = ("The flight date could not be determined. Pass --flight-date YYYY-MM-DD, put a flight.json next to the "
            "RGB orthophoto, or give --images-dir / --odm-project so the photo timestamps can be read.")
    if estimate.best is not None:
        text += " Dates found in the file or folder names (not used without confirmation): " + "; ".join(
            c.value.isoformat() + (" (ambiguous)" if c.ambiguous else "") for c in estimate.candidates) + "."
    return text
