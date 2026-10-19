"""Orchestrator: aligned MS stack in, registered Sentinel-2 patch out.

Scenes come from a provider chain (online catalogues or a local folder); ``discovery.selection`` walks
the date ladder. The fail-closed behaviour is final: a patch that does not pass every gate
is published with its original georeferencing and a reason code, never as 'aligned'.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry

from sentinel_patch.config.loader import load_config
from sentinel_patch.config.schema import SentinelPatchConfig
from sentinel_patch.drone_metadata import DroneMetadata
from sentinel_patch.discovery.date_ladder import date_gap_days, date_tier
from sentinel_patch.discovery.provider import SceneCandidate, SceneProvider
from sentinel_patch.discovery.providers.local_folder import LocalFolderProvider
from sentinel_patch.discovery.registry import build_chain
from sentinel_patch.discovery.selection import PARTIAL, select_scene
from sentinel_patch.errors import (
    ConfigurationError, FlightDateUnknownError, ProviderUnavailableError,
    RegistrationRejected, SentinelPatchError,
)
from sentinel_patch.footprint import StackFootprint, extract_stack_footprint, write_footprint_geojson
from sentinel_patch.io.cloud_mask import AoiMaskStats, aoi_mask_stats, aoi_pixel_mask, build_reject_mask
from sentinel_patch.io.radiometry import RadiometryParams, dn_to_reflectance, resolve_radiometry
from sentinel_patch.io.window_reader import BandWindow, plan_window_10m, read_band_window
from sentinel_patch.products import OverlayTarget, ProductInputs, write_products
from sentinel_patch.quality.preview import write_preview
from sentinel_patch.quality.report import build_report, write_json_atomic
from sentinel_patch.registration import gates
from sentinel_patch.registration.consensus import ConsensusResult, Vote, decide
from sentinel_patch.registration.arosics_estimator import ArosicsResult, estimate_shift_arosics
from sentinel_patch.registration.estimators import NccResult, estimate_shift_ncc
from sentinel_patch.registration.quadrants import QuadrantReport, check_quadrants
from sentinel_patch.registration.reference import build_reference_band
from sentinel_patch.registration.shift import pixel_shift_to_map, shift_magnitude_m

logger = logging.getLogger("sentinel_patch")

SENTINEL_GSD_M = 10.0
STATUS_VERIFIED = "verified"
STATUS_NOT_APPLIED = "not_applied"
STATUS_NO_SCENE = "no_scene"
STATUS_VOTE = "vote"  # internal: a scene whose own shift is plausible and may take part in a consensus
# Refusals that more independent scenes can resolve (ambiguity, noise). Never texture, distortion or a failed
# independent-band check: agreement between scenes must not override those.
CONSENSUS_ELIGIBLE = frozenset({"AOI_TOO_SMALL", "WEAK_PEAK", "PEAK_AT_SEARCH_BORDER", "SHIFT_IMPLAUSIBLE"})


@dataclass(frozen=True)
class SentinelPatchResult:
    status: str
    reason_code: str | None
    shift_m: tuple[float, float]
    report_path: Path
    product_paths: Mapping[str, Path]
    scene_id: str | None
    warnings: tuple[str, ...] = ()


@dataclass
class _Prepared:
    candidate: SceneCandidate
    crs: CRS
    aoi: BaseGeometry
    bounds: tuple[float, float, float, float]
    match: BandWindow
    holdout: BandWindow
    scl: BandWindow
    reject: np.ndarray
    stats: AoiMaskStats
    params: dict[str, RadiometryParams] = field(default_factory=dict)


@dataclass
class _Registration:
    status: str
    reason_code: str | None
    shift_m: tuple[float, float]
    gate_results: list[gates.GateResult]
    ncc: NccResult | None
    holdout: tuple[float | None, float | None]
    reference: np.ndarray  # drone band on the Sentinel grid, for the preview
    sentinel: np.ndarray  # Sentinel match band with rejected pixels blanked
    arosics: ArosicsResult | None = None
    quadrants: QuadrantReport | None = None
    warnings: list[str] = field(default_factory=list)
    consensus: ConsensusResult | None = None
    votes: list[Vote] = field(default_factory=list)
    single_scene: dict | None = None  # what the single best scene said, kept when consensus overrides it
    vote_records: list[dict] = field(default_factory=list)


def _default_provider(cfg: SentinelPatchConfig) -> SceneProvider:
    if cfg.local_folder is not None:
        return LocalFolderProvider(cfg.local_folder)
    return build_chain(cfg)


def _band_params(candidate: SceneCandidate, band: str) -> RadiometryParams:
    declared = candidate.scale_offset.get(band)
    scale, offset = declared if declared else (None, None)
    return resolve_radiometry(scale=scale, offset=offset, processing_baseline=candidate.processing_baseline)


def _scene_crs(href: str) -> CRS:
    from sentinel_patch.io.window_reader import _retrying

    def read() -> CRS:
        with rasterio.open(href) as src:
            return CRS.from_user_input(src.crs)
    return _retrying(read, "the scene header")


def _prepare(candidate: SceneCandidate, provider: SceneProvider, footprint: StackFootprint,
             cfg: SentinelPatchConfig) -> _Prepared:
    """Read the matching bands and the cloud layer for one candidate and measure its AOI quality."""
    match_band, holdout_band = cfg.reference.match_band.sentinel, cfg.reference.holdout_band.sentinel
    hrefs = {band: provider.resolve_href(candidate, band) for band in (match_band, holdout_band, "SCL")}
    crs = _scene_crs(hrefs[match_band])
    aoi = transform_geometry(Transformer.from_crs(footprint.stack_crs, crs, always_xy=True).transform,
                             footprint.polygon_stack_crs)
    _, bounds = plan_window_10m(hrefs[match_band], aoi.bounds, cfg.io.registration_buffer_m)
    match = read_band_window(hrefs[match_band], bounds, match_band)
    holdout = read_band_window(hrefs[holdout_band], bounds, holdout_band)
    scl = read_band_window(hrefs["SCL"], bounds, "SCL")
    reject = build_reject_mask(scl.data, (match.data == 0) | (holdout.data == 0), cfg.discovery)
    stats = aoi_mask_stats(reject, aoi_pixel_mask(aoi, match.transform, match.data.shape))
    prepared = _Prepared(candidate, crs, aoi, bounds, match, holdout, scl, reject, stats)
    prepared.params = {band: _band_params(candidate, band) for band in (match_band, holdout_band)}
    return prepared


def _search_range(flight_date: date, cfg: SentinelPatchConfig) -> tuple[datetime, datetime]:
    gap = timedelta(days=cfg.discovery.max_date_gap_days)
    return (datetime.combine(flight_date - gap, time.min, timezone.utc),
            datetime.combine(flight_date + gap, time.max, timezone.utc))


def _reference(cache: dict | None, aligned_ms_path: Path, band_index: int, prepared: _Prepared, ref_cfg) -> np.ndarray:
    """The drone band on this scene's 10 m grid; scenes sharing a grid (same tile) share one build."""
    key = (band_index, tuple(prepared.match.transform)[:6], prepared.match.data.shape, prepared.crs.to_string())
    if cache is not None and key in cache:
        return cache[key]
    values, _ = build_reference_band(aligned_ms_path, band_index, prepared.match.transform, prepared.match.data.shape,
                                     prepared.crs, ref_cfg)
    if cache is not None:
        cache[key] = values
    return values


def _register(prepared: _Prepared, aligned_ms_path: Path, ms_band_map: Mapping[str, int],
              cfg: SentinelPatchConfig, *, vote: bool = False, ref_cache: dict | None = None) -> _Registration:
    """Register one scene. ``vote=True``: only measure the scene's own shift for a consensus.

    A vote uses the consensus overlap floor and skips the peak-ratio test (a small field's single-scene surface is
    a ridge; the agreement between scenes replaces it) and every cross-check that needs a trustworthy peak.
    """
    ref_cfg, reg_cfg = cfg.reference, cfg.registration
    if vote:
        reg_cfg = reg_cfg.model_copy(update={"min_overlap_px": reg_cfg.consensus.min_overlap_px, "min_peak_ratio": 1.0})
    for pair in (ref_cfg.match_band, ref_cfg.holdout_band):
        if pair.drone not in ms_band_map:
            raise ConfigurationError(f"ms_band_map has no entry for drone band '{pair.drone}'.")
    with rasterio.open(aligned_ms_path) as src:
        band_count = src.count
    for name in (ref_cfg.match_band.drone, ref_cfg.holdout_band.drone):
        if not 1 <= ms_band_map[name] <= band_count:
            raise ConfigurationError(
                f"The band map says {name} is band {ms_band_map[name]}, but the aligned MS file has {band_count} "
                f"band(s). Pass the right numbers with --sentinel-ms-bands (e.g. red=1,green=2,nir=3).")
    grid_shape, transform = prepared.match.data.shape, prepared.match.transform

    ref_match = _reference(ref_cache, aligned_ms_path, ms_band_map[ref_cfg.match_band.drone], prepared, ref_cfg)
    ref_holdout = None if vote else _reference(ref_cache, aligned_ms_path, ms_band_map[ref_cfg.holdout_band.drone],
                                               prepared, ref_cfg)
    s2_match = dn_to_reflectance(prepared.match.data, prepared.params[ref_cfg.match_band.sentinel])
    s2_holdout = None if vote else dn_to_reflectance(prepared.holdout.data, prepared.params[ref_cfg.holdout_band.sentinel])

    valid_ref = np.isfinite(ref_match)
    valid_s2 = ~prepared.reject & np.isfinite(s2_match)
    # Overlap first: texture measured on a handful of pixels means nothing.
    results = [gates.gate_overlap(int((valid_ref & valid_s2).sum()), reg_cfg),
               gates.gate_texture(ref_match, valid_ref, reg_cfg)]

    shown = np.where(valid_s2, s2_match, np.nan)

    def outcome(status, reason, shift, ncc=None, holdout=(None, None), **extra) -> _Registration:
        return _Registration(status, reason, shift, results, ncc, holdout, ref_match, shown, **extra)

    def rejected(reason: str | None, **kwargs) -> _Registration:
        return outcome(STATUS_NOT_APPLIED, reason, (0.0, 0.0), **kwargs)

    failure = gates.first_failure(results)
    if failure:
        return rejected(failure.reason_code)
    try:
        ncc = estimate_shift_ncc(ref_match, s2_match, valid_ref, valid_s2, reg_cfg)
    except RegistrationRejected as exc:
        results.append(gates.GateResult("estimator", None, 0.0, False, exc.reason_code))
        return rejected(exc.reason_code)

    pixel = abs(transform.a)
    dx, dy = pixel_shift_to_map(transform, ncc.col_shift_px, ncc.row_shift_px)
    results.append(gates.gate_magnitude(dx, dy, reg_cfg))
    results.extend(gates.gate_peak(ncc, reg_cfg))
    failure = gates.first_failure(results)
    if failure:  # no point spending time on the cross-checks of a shift that is already refused
        return rejected(failure.reason_code, ncc=ncc)
    if vote:
        return outcome(STATUS_VOTE, None, (dx, dy), ncc)

    aro = estimate_shift_arosics(ref_match, s2_match, valid_ref, valid_s2, transform, prepared.crs, reg_cfg)
    agreement = gates.gate_agreement((dx, dy), aro, pixel, reg_cfg)
    results.append(agreement)
    holdout = gates.holdout_correlation(
        ref_holdout, s2_holdout, np.isfinite(ref_holdout), ~prepared.reject & np.isfinite(s2_holdout),
        ncc.col_shift_px, ncc.row_shift_px,
    )
    results.append(gates.gate_holdout(*holdout, float(np.hypot(ncc.col_shift_px, ncc.row_shift_px)), reg_cfg))
    quadrant_gate, quadrants = check_quadrants(ref_match, s2_match, valid_ref, valid_s2,
                                               (ncc.col_shift_px, ncc.row_shift_px), reg_cfg)
    results.append(quadrant_gate)
    warnings = ["AROSICS_INCONCLUSIVE"] if aro.status == "inconclusive" else []
    if agreement.advisory and not agreement.passed:
        warnings.append("AROSICS_DISAGREES")
    failure = gates.first_failure(results)
    if failure:
        return rejected(failure.reason_code, ncc=ncc, holdout=holdout, arosics=aro, quadrants=quadrants, warnings=warnings)
    return outcome(STATUS_VERIFIED, None, (dx, dy), ncc, holdout, arosics=aro, quadrants=quadrants, warnings=warnings)


def _collect_votes(candidates: list[SceneCandidate], prep, flight_date: date, aligned_ms_path: Path,
                   ms_band_map: Mapping[str, int], cfg: SentinelPatchConfig, ref_cache: dict) -> tuple[list[Vote], list[dict]]:
    """Measure the shift in up to ``consensus.max_scenes`` clean scenes, nearest date first, one vote per acquisition."""
    cons = cfg.registration.consensus
    ordered = sorted(candidates, key=lambda c: (abs(date_gap_days(c.datetime_utc, flight_date)),
                                                0 if date_gap_days(c.datetime_utc, flight_date) >= 0 else 1))
    votes: list[Vote] = []
    records: list[dict] = []
    seen: set[str] = set()
    measured = 0
    for candidate in ordered:
        if measured >= cons.max_scenes:
            break
        gap = date_gap_days(candidate.datetime_utc, flight_date)
        key = candidate.datatake_id or f"{candidate.datetime_utc:%Y%m%d%H%M}"
        record = {"provider": candidate.provider, "item_id": candidate.item_id, "date_gap_days": gap}
        records.append(record)
        if key in seen:  # another tile of an acquisition already counted: not an independent observation
            record.update(vote=False, reason_code="DUPLICATE_ACQUISITION")
            continue
        try:
            prepared = prep(candidate)
        except SentinelPatchError as exc:
            record.update(vote=False, reason_code=exc.code)
            continue
        stats = prepared.stats
        record.update(aoi_masked_fraction=stats.masked_fraction)
        if stats.masked_fraction > cons.max_aoi_masked_fraction or stats.clear_pixels < cfg.discovery.min_clear_pixels:
            record.update(vote=False, reason_code="AOI_TOO_CLOUDY")
            continue
        seen.add(key)
        measured += 1
        result = _register(prepared, aligned_ms_path, ms_band_map, cfg, vote=True, ref_cache=ref_cache)
        if result.status != STATUS_VOTE:
            record.update(vote=False, reason_code=result.reason_code)
            continue
        ncc = result.ncc
        votes.append(Vote(candidate.item_id, key, gap, result.shift_m, (ncc.col_shift_px, ncc.row_shift_px),
                          ncc.peak, ncc.peak_ratio, ncc.overlap_px))
        record.update(vote=True, reason_code=None, shift_m=list(result.shift_m), peak=ncc.peak)
    return votes, records


def _read_all_bands(prepared: _Prepared, provider: SceneProvider, cfg: SentinelPatchConfig,
                    log: logging.Logger) -> tuple[dict[str, BandWindow], dict[str, BandWindow]]:
    candidate = prepared.candidate
    result: dict[str, dict[str, BandWindow]] = {"10": {}, "20": {}}
    for key, names in (("10", cfg.io.bands_10m), ("20", cfg.io.bands_20m)):
        for band in names:
            if band not in candidate.asset_hrefs:
                log.info("Band %s is not available in %s; skipped.", band, candidate.item_id)
                continue
            result[key][band] = read_band_window(provider.resolve_href(candidate, band), prepared.bounds, band)
            prepared.params.setdefault(band, _band_params(candidate, band))
    return result["10"], result["20"]


def run_sentinel_patch(
    aligned_ms_path: Path,
    rgb_path: Path,
    output_dir: Path,
    *,
    flight_date: date | None,
    ms_band_map: Mapping[str, int],
    config: SentinelPatchConfig | None = None,
    provider: SceneProvider | None = None,
    log: logging.Logger | None = None,
    drone_metadata: DroneMetadata | None = None,
) -> SentinelPatchResult:
    """Register a Sentinel-2 patch to the aligned stack and publish it under ``<output_dir>/sentinel``.

    ``flight_date`` wins; otherwise the date in ``drone_metadata`` (see ``drone_metadata.resolve_drone_metadata``) is used.
    """
    with ExitStack() as stack:  # holds the GDAL environment of the chosen provider open for the whole run
        return _run(aligned_ms_path, rgb_path, output_dir, flight_date, ms_band_map, config, provider, log, stack,
                    drone_metadata)


def _run(aligned_ms_path, rgb_path, output_dir, flight_date, ms_band_map, config, provider, log, stack, drone_metadata):
    log = log or logger
    cfg = config or load_config()
    if flight_date is None and drone_metadata is not None:
        flight_date = drone_metadata.flight_date
    if flight_date is None:
        raise FlightDateUnknownError("No flight date supplied; pass flight_date or drone_metadata with a date.")
    flight_date_source = drone_metadata.flight_date_source if drone_metadata is not None else "argument"
    inherited_warnings = list(drone_metadata.warnings) if drone_metadata is not None else []
    out_dir = Path(output_dir) / "sentinel"
    out_dir.mkdir(parents=True, exist_ok=True)

    footprint = extract_stack_footprint(Path(aligned_ms_path), Path(rgb_path))
    write_footprint_geojson(footprint, out_dir / "footprint.geojson")
    footprint_info = {"area_m2": footprint.area_m2, "stack_crs": footprint.stack_crs.to_string()}

    def finish(status, reason, registration, products, scene, warnings, shift=(0.0, 0.0), paths=None):
        warnings = list(dict.fromkeys([*inherited_warnings, *warnings]))
        report = build_report(
            status=status, reason_code=reason, flight_date=flight_date, flight_date_source=flight_date_source,
            footprint=footprint_info, scene=scene, registration=registration, products=products, warnings=warnings,
            drone_metadata=None if drone_metadata is None else drone_metadata.as_dict(),
        )
        report_path = write_json_atomic(out_dir / "sentinel_report.json", report)
        return SentinelPatchResult(status, reason, shift, report_path, paths or {},
                                   scene["item_id"] if scene else None, tuple(warnings))

    aoi_pixels_10m = footprint.area_m2 / (SENTINEL_GSD_M**2)
    if aoi_pixels_10m < cfg.discovery.min_clear_pixels:
        log.warning("Stack covers only %.0f Sentinel pixels (minimum %d); nothing sensible to register.",
                    aoi_pixels_10m, cfg.discovery.min_clear_pixels)
        return finish(STATUS_NOT_APPLIED, "AOI_TOO_SMALL", None, None, None, [])

    provider = provider or _default_provider(cfg)
    stack.enter_context(rasterio.Env(**provider.gdal_env()))
    failures = lambda: list(getattr(provider, "failures", []))  # noqa: E731  (the chain records outages)
    buffered = footprint.polygon_stack_crs.buffer(cfg.io.registration_buffer_m)
    to_wgs84 = Transformer.from_crs(footprint.stack_crs, "EPSG:4326", always_xy=True)
    start, end = _search_range(flight_date, cfg)
    try:
        candidates = provider.search(transform_geometry(to_wgs84.transform, buffered), start, end)
    except ProviderUnavailableError as exc:
        log.warning("No scene provider could be used: %s", exc)
        write_json_atomic(out_dir / "candidates.json", {"flight_date": flight_date, "candidates": [],
                                                        "provider_failures": failures()})
        return finish(STATUS_NO_SCENE, "PROVIDERS_UNAVAILABLE", None, None, None, [])
    prepared_cache: dict = {}

    def prep(candidate: SceneCandidate) -> _Prepared:  # a scene read for the selection is reused by the consensus
        key = (candidate.provider, candidate.item_id)
        if key not in prepared_cache:
            prepared_cache[key] = _prepare(candidate, provider, footprint, cfg)
        return prepared_cache[key]

    selection = select_scene(candidates, prep, flight_date, cfg.discovery)
    write_json_atomic(out_dir / "candidates.json", {
        "flight_date": flight_date, "candidates": selection.records, "provider_failures": failures()})
    prepared = selection.prepared

    if prepared is None:
        reason = "NO_CANDIDATES" if not candidates else "NO_ACCEPTABLE_SCENE"
        log.warning("No usable Sentinel-2 scene within %d days of %s (%s).", cfg.discovery.max_date_gap_days, flight_date, reason)
        return finish(STATUS_NO_SCENE, reason, None, None, None, [])

    candidate = prepared.candidate
    gap = date_gap_days(candidate.datetime_utc, flight_date)
    tier = date_tier(gap, cfg.discovery)
    warnings = ["DATE_GAP_STALE"] if tier == "stale" else []
    partial = selection.quality == PARTIAL
    if partial:
        warnings.append("SCENE_PARTIAL")
    scene_info = {
        "provider": candidate.provider, "item_id": candidate.item_id, "datetime_utc": candidate.datetime_utc,
        "date_gap_days": gap, "date_tier": tier, "mgrs_tile": candidate.mgrs_tile, "epsg": prepared.crs.to_epsg(),
        "processing_baseline": candidate.processing_baseline, "scene_quality": selection.quality,
        "aoi_masked_fraction": prepared.stats.masked_fraction, "aoi_clear_pixels": prepared.stats.clear_pixels,
        "radiometry": {band: {"scale": p.scale, "offset": p.offset, "source": p.source}
                       for band, p in prepared.params.items()},
    }

    ref_cache: dict = {}
    registration = _register(prepared, Path(aligned_ms_path), ms_band_map, cfg, ref_cache=ref_cache)
    if (registration.status == STATUS_NOT_APPLIED and cfg.registration.consensus.enabled
            and registration.reason_code in CONSENSUS_ELIGIBLE):
        log.info("Single scene refused (%s); asking the other scenes in the window.", registration.reason_code)
        votes, vote_records = _collect_votes(candidates, prep, flight_date, Path(aligned_ms_path), ms_band_map, cfg, ref_cache)
        verdict = decide(votes, cfg.registration.consensus, abs(prepared.match.transform.a), cfg.registration.max_shift_m)
        registration.votes, registration.consensus, registration.vote_records = votes, verdict, vote_records
        registration.single_scene = {"status": registration.status, "reason_code": registration.reason_code}
        log.info("Consensus: %s", verdict.reason)
        if verdict.agreed:
            registration.status, registration.reason_code, registration.shift_m = STATUS_VERIFIED, None, verdict.shift_m
            registration.warnings.append("SHIFT_FROM_CONSENSUS")
    warnings.extend(registration.warnings)
    if (drone_metadata is not None and registration.status == STATUS_VERIFIED
            and drone_metadata.positioning in ("gnss_only", "unknown")
            and shift_magnitude_m(*registration.shift_m) > cfg.registration.large_shift_warning_m):
        warnings.append("LARGE_SHIFT_DRONE_FRAME_SUSPECT")
    ncc = registration.ncc
    consensus = registration.consensus
    agreed = consensus is not None and consensus.agreed
    shift_px = consensus.shift_px if agreed else (None if ncc is None else (ncc.col_shift_px, ncc.row_shift_px))
    registration_info = {
        "method": "consensus" if agreed else "single_scene",
        "shift_m": registration.shift_m,
        "shift_crs": prepared.crs.to_string(),  # the axes dx (east) / dy (north) refer to
        "shift_px": shift_px,
        "single_scene": registration.single_scene,
        "consensus": None if consensus is None else {
            **consensus.as_dict(), "votes_detail": [v.as_dict() for v in registration.votes],
            "scenes": registration.vote_records},
        "estimators": {
            "ncc": None if ncc is None else {
                "peak": ncc.peak, "second_peak": ncc.second_peak, "peak_ratio": ncc.peak_ratio,
                "integer_shift_px": ncc.integer_shift_px, "overlap_px": ncc.overlap_px},
            "arosics": None if registration.arosics is None else registration.arosics.as_dict(),
            "quadrants": None if registration.quadrants is None else {
                "evaluated": registration.quadrants.evaluated, "deviations_px": registration.quadrants.deviations_px,
                "skipped": registration.quadrants.skipped},
        },
        "gates": [result.as_dict() for result in registration.gate_results],
        "holdout": {"band": cfg.reference.holdout_band.sentinel,
                    "corr_before": registration.holdout[0], "corr_after": registration.holdout[1]},
    }
    preview_path = write_preview(
        out_dir / "sentinel_preview.png", rgb_path=Path(rgb_path), drone_polygon=footprint.polygon_stack_crs,
        scene_polygon=prepared.aoi, window_transform=prepared.match.transform, reference=registration.reference,
        sentinel=registration.sentinel, shift_m=registration.shift_m,
        shift_px=shift_px, status=registration.status,
        reason_code=registration.reason_code,
        title_lines=[candidate.item_id, f"{candidate.datetime_utc:%Y-%m-%d}", f"gap {gap:+d} d ({tier})"],
    )
    registration_info["preview"] = str(preview_path)
    log.info("Sentinel registration %s%s: shift (%.2f, %.2f) m.", registration.status,
             f" ({registration.reason_code})" if registration.reason_code else "", *registration.shift_m)

    if registration.status == STATUS_NOT_APPLIED and cfg.registration.on_reject == "skip":
        return finish(STATUS_NOT_APPLIED, registration.reason_code, registration_info, None, scene_info, warnings,
                      paths={"preview": preview_path})

    bands_10m, bands_20m = _read_all_bands(prepared, provider, cfg, log)
    overlay = OverlayTarget(footprint.stack_crs, footprint.grid_transform, footprint.polygon_stack_crs)
    products = write_products(
        ProductInputs(candidate, prepared.crs, prepared.aoi, bands_10m, bands_20m, prepared.scl, prepared.params,
                      registration.status, registration.reason_code, registration.shift_m, overlay,
                      prepared.reject if partial else None),
        out_dir, cfg.products,
    )
    paths = {**products["paths"], "preview": preview_path}
    products_info = {**{name: str(path) for name, path in paths.items()},
                     "resampled": products["resampled"], "grid_snapped": products["grid_snapped"],
                     "clip_shape_10m": products["clip_shape_10m"], "overlay_gsd_m": products["overlay_gsd_m"]}
    return finish(registration.status, registration.reason_code, registration_info, products_info, scene_info,
                  warnings, registration.shift_m, paths)
