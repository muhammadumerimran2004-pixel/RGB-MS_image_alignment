"""The Sentinel stage as the command line runs it: resolve the date, run, report, never abort the alignment.

``drone_alignment.cli`` imports this lazily and only when ``--sentinel`` is given, so installations
without ``sentinel_patch`` are unaffected. Every failure here is contained: the aligned MS product
that already exists is never touched, and the exit code stays 0 unless ``strict`` is set.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from sentinel_patch.config.loader import load_config
from sentinel_patch.drone_metadata import (
    ESTIMATED_SOURCES, POSITIONING_VALUES, make_console_confirmer, resolve_drone_metadata,
)
from sentinel_patch.errors import ConfigurationError, SentinelPatchError
from sentinel_patch.pipeline import run_sentinel_patch

logger = logging.getLogger("sentinel_patch")

# The company's multispectral orthophotos have three bands: 1 = red, 2 = green, 3 = NIR (confirmed in QGIS,
# 2026-10-19). The file does not name its bands; --sentinel-ms-bands overrides this for other cameras.
DEFAULT_MS_BANDS = "red=1,green=2,nir=3"
KNOWN_DRONE_BANDS = {"blue", "green", "red", "red_edge", "nir"}


@dataclass(frozen=True)
class StageOptions:
    flight_date: date | None = None
    images_dir: Path | None = None
    odm_project: Path | None = None
    config_path: Path | None = None
    positioning: str | None = None
    ms_bands: str = DEFAULT_MS_BANDS
    strict: bool = False
    interactive: bool = False


@dataclass(frozen=True)
class StageOutcome:
    ok: bool  # the stage ran to a verified result
    status: str
    reason_code: str | None
    report_block: dict = field(default_factory=dict)  # goes into the main alignment report under "sentinel_patch"
    strict_failure: bool = False


def parse_ms_bands(text: str) -> dict[str, int]:
    """``"green=2,red=3,nir=5"`` -> ``{"green": 2, "red": 3, "nir": 5}`` (1-indexed band numbers)."""
    result: dict[str, int] = {}
    try:
        for part in filter(None, (p.strip() for p in text.split(","))):
            name, number = part.split("=")
            name = name.strip().lower()
            if name not in KNOWN_DRONE_BANDS or int(number) < 1:
                raise ValueError(name)
            result[name] = int(number)
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid band map '{text}'. Use NAME=NUMBER pairs, e.g. {DEFAULT_MS_BANDS} "
            f"(names: {', '.join(sorted(KNOWN_DRONE_BANDS))}; numbers start at 1)."
        ) from exc
    return result


def run_stage(
    *, aligned_ms_path: Path, rgb_path: Path, ms_path: Path, out_dir: Path, options: StageOptions,
    echo: Callable[[str], None], prompt: Callable[[str], str] | None = None, verbose: bool = False,
) -> StageOutcome:
    echo("\n--- Sentinel-2 patch ---")

    def failed(code: str, message: str) -> StageOutcome:
        echo(f"[WARN] Sentinel-2 stage did not run: {message}")
        echo("       The aligned MS product above is unaffected.")
        return StageOutcome(False, "error", code, {"status": "error", "reason_code": code, "message": message},
                            strict_failure=options.strict)
    try:
        cfg = load_config(options.config_path) if options.config_path else load_config()
        ms_band_map = parse_ms_bands(options.ms_bands)
        confirm = make_console_confirmer(prompt, echo) if options.interactive and prompt else None
        metadata = resolve_drone_metadata(
            rgb_path=rgb_path, ms_path=ms_path, flight_date=options.flight_date, images_dir=options.images_dir,
            odm_project=options.odm_project, positioning=options.positioning, cfg=cfg.drone_metadata, confirm=confirm,
        )
        echo(f"  Flight date:  {metadata.flight_date.isoformat()}  (from {metadata.flight_date_source})")
        if metadata.flight_date_source in ESTIMATED_SOURCES:
            echo("                estimated from the folder name; check it if the result looks off")
        echo(f"  Positioning:  {metadata.positioning}  ({metadata.positioning_source})")
        result = run_sentinel_patch(
            aligned_ms_path, rgb_path, out_dir, flight_date=metadata.flight_date, ms_band_map=ms_band_map,
            config=cfg, drone_metadata=metadata, log=logger,
        )
    except SentinelPatchError as exc:
        return failed(exc.code, str(exc))
    except Exception as exc:  # noqa: BLE001  (a bug or network surprise must not undo a finished alignment)
        if verbose:
            echo(traceback.format_exc())
        return failed("SENTINEL_UNEXPECTED_ERROR", f"{type(exc).__name__}: {exc}")

    block = {"status": result.status, "reason_code": result.reason_code, "shift_m": list(result.shift_m),
             "scene_id": result.scene_id, "report_path": str(result.report_path), "warnings": list(result.warnings)}
    echo(f"  Status:       {result.status}" + (f" ({result.reason_code})" if result.reason_code else ""))
    if result.scene_id:
        echo(f"  Scene:        {result.scene_id}")
    if result.status == "verified":
        echo(f"  Shift (E, N): {result.shift_m[0]:+.2f} m, {result.shift_m[1]:+.2f} m")
    elif result.status == "not_applied":
        echo("  The patch was published with its ORIGINAL position: the registration did not pass every check.")
    else:
        echo("  No usable Sentinel-2 scene was found for this flight date.")
    for name, path in result.product_paths.items():
        echo(f"  {name + ':':<13} {path}")
    for warning in result.warnings:
        echo(f"  [WARN] {warning}")
    echo(f"  Report:       {result.report_path}")
    return StageOutcome(result.status == "verified", result.status, result.reason_code, block,
                        strict_failure=options.strict and result.status != "verified")
