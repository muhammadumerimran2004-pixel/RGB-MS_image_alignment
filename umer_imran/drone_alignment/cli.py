import json
import logging
from pathlib import Path
import click
import rasterio
import yaml

from drone_alignment.config.schema import (
    AlignmentConfig, AlignmentMode, ResolutionMode, DetectorType, ManualCoordinateMode, ArosicsBandPair,
)
from drone_alignment.io.gcp_io import read_gcp_csv, read_qgis_points
from drone_alignment.pipeline import (
    align_orthomosaics, local_correlation_align_orthomosaics, local_mesh_align_orthomosaics,
    manual_align_orthomosaics, road_grid_align_orthomosaics,
)


@click.command()
@click.argument("rgb_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("ms_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["manual", "automated", "road_grid", "local_mesh", "local_correlation", "local-correlation", "prompt"]),
    default="prompt",
    help="Alignment mode: 'manual', 'automated', 'road_grid', experimental 'local_mesh', feature-agnostic 'local-correlation', or 'prompt'.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for aligned GeoTIFF and reports (defaults to directory of MS file).",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML configuration file.",
)
@click.option(
    "--resolution",
    "-r",
    type=click.Choice(["ms", "rgb"]),
    default="ms",
    help="Output GSD mode: 'ms' preserves native MS pixel scale, 'rgb' upsamples to RGB scale.",
)
@click.option(
    "--detector",
    "-d",
    type=click.Choice(["orb", "sift"]),
    default="orb",
    help="Primary feature detector algorithm.",
)
@click.option(
    "--ms-red-band",
    type=int,
    default=1,
    help="1-indexed Red band number in the MS raster file.",
)
@click.option(
    "--enable-arosics",
    is_flag=True,
    help="Try optional AROSICS subpixel co-registration candidate.",
)
@click.option(
    "--arosics-band-pair",
    "arosics_band_pairs",
    multiple=True,
    metavar="NAME:REFERENCE_BAND:TARGET_BAND",
    help=(
        "Ordered AROSICS pair using 1-indexed bands, e.g. red_edge:4:3. "
        "The first value is the primary local AROSICS benchmark; supplying one enables AROSICS."
    ),
)
@click.option(
    "--arosics-local/--no-arosics-local",
    "arosics_local",
    default=None,
    help="Enable/disable AROSICS COREG_LOCAL refinement of the verified global result (default: enabled).",
)
@click.option(
    "--arosics-max-shift-m",
    type=float,
    default=None,
    help="Max AROSICS local search radius in metres (default: derived from the global result's residual RMSE).",
)
@click.option(
    "--arosics-warp-engine",
    type=click.Choice(["auto", "deshifter", "gdal_tps"]),
    default=None,
    help="AROSICS local warp engine: AROSICS' own DESHIFTER, streaming GDAL TPS, or 'auto' by memory estimate.",
)
@click.option(
    "--crop-row-period-m",
    type=float,
    default=None,
    help="Crop row spacing in metres; caps the local search radius below half this period to avoid row aliasing.",
)
@click.option(
    "--gcp-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Manual mode: read control points from this file instead of prompting interactively.",
)
@click.option(
    "--gcp-format",
    type=click.Choice(["csv", "qgis"]),
    default="csv",
    help="Format of --gcp-file: 'csv' (id,rgb_x,rgb_y,ms_x,ms_y[,role]) or 'qgis' (Georeferencer .points).",
)
@click.option(
    "--gcp-source-units",
    type=click.Choice(["pixel", "map"]),
    default="map",
    help="Coordinate units of the MS/source points in --gcp-file ('csv' applies this to both point sets).",
)
@click.option(
    "--manual-model",
    type=click.Choice(["auto", "translation", "similarity", "affine", "tps"]),
    default=None,
    help="Manual mode geometric model (default: 'auto', picked from control-point count and leave-one-out RMSE).",
)
@click.option(
    "--pixel-convention",
    type=click.Choice(["corner", "center"]),
    default=None,
    help="Manual mode pixel convention for pixel-unit control points (default: 'center').",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose debug logging.",
)
def main(
    rgb_path: Path,
    ms_path: Path,
    mode: str,
    output_dir: Path,
    config: Path,
    resolution: str,
    detector: str,
    ms_red_band: int,
    enable_arosics: bool,
    arosics_band_pairs: tuple[str, ...],
    arosics_local: bool | None,
    arosics_max_shift_m: float | None,
    arosics_warp_engine: str | None,
    crop_row_period_m: float | None,
    gcp_file: Path | None,
    gcp_format: str,
    gcp_source_units: str,
    manual_model: str | None,
    pixel_convention: str | None,
    verbose: bool,
):
    """
    Drone Orthomosaic Alignment CLI tool.

    Co-registers a Multispectral (MS) GeoTIFF orthomosaic to a Reference RGB GeoTIFF pivot.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Input ODM TIFFs may advertise an invalidated layout optimisation.  GDAL
    # emits that advisory every time a raster is opened; it does not affect
    # pixels, georeferencing, or the alignment result, so keep it out of the
    # interactive workflow.
    logging.getLogger("rasterio").setLevel(logging.ERROR)

    out_dir = output_dir if output_dir is not None else ms_path.parent / "aligned"

    if config is not None:
        with open(config, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        cfg = AlignmentConfig.model_validate(cfg_dict)
    else:
        cfg = AlignmentConfig(
            resolution_mode=ResolutionMode(resolution),
            ms_red_band_index=ms_red_band,
        )
        cfg.features.detector = DetectorType(detector)
        # AROSICS local co-registration is enabled by the schema default.  The
        # flag is retained for backwards-compatible explicit opt-in, but its
        # absence must not disable the primary automated engine.
        if enable_arosics:
            cfg.arosics.enabled = True

    # Flags supplement YAML configuration rather than being discarded whenever
    # --config is used. This keeps CLI experiments reproducible and explicit.
    if enable_arosics:
        cfg.arosics.enabled = True
    if arosics_band_pairs:
        parsed_pairs = []
        for raw_pair in arosics_band_pairs:
            parts = raw_pair.split(":")
            if len(parts) != 3:
                raise click.BadParameter(
                    "use NAME:REFERENCE_BAND:TARGET_BAND, e.g. red_edge:4:3",
                    param_hint="--arosics-band-pair",
                )
            name, reference_band, target_band = parts
            try:
                parsed_pairs.append(ArosicsBandPair(
                    name=name, reference_band=int(reference_band), target_band=int(target_band),
                ))
            except ValueError as exc:
                raise click.BadParameter(
                    "band numbers must be positive integers", param_hint="--arosics-band-pair",
                ) from exc
        cfg.arosics.enabled = True
        cfg.arosics.band_pairs = parsed_pairs
    if arosics_local is not None:
        cfg.arosics.local.enabled = arosics_local
    if arosics_max_shift_m is not None:
        cfg.arosics.local.max_shift_m = arosics_max_shift_m
    if arosics_warp_engine is not None:
        cfg.arosics.local.warp_engine = arosics_warp_engine
    if crop_row_period_m is not None:
        cfg.arosics.local.periodic_texture_period_m = crop_row_period_m
    if manual_model is not None:
        cfg.manual.model = manual_model
    if pixel_convention is not None:
        cfg.manual.pixel_convention = pixel_convention

    click.echo("Starting Drone Alignment Engine...")
    click.echo(f"  RGB Reference: {rgb_path}")
    click.echo(f"  MS Target:    {ms_path}")
    click.echo(f"  Output Dir:   {out_dir}")
    click.echo(f"  Resolution:   {cfg.resolution_mode.value}")
    if cfg.arosics.enabled and cfg.arosics.local.enabled:
        click.echo("  Engine:       verified global -> AROSICS local refinement (global fallback)")
    else:
        click.echo("  Engine:       verified global only (AROSICS local refinement disabled)")

    # Prompt for mode if set to 'prompt'
    selected_mode = mode
    if selected_mode == "prompt":
        click.echo("\n--- Alignment Mode Selection ---")
        click.echo("  [1] Manual correction (GCP control points)")
        click.echo("  [2] Automated recognition pipeline")
        click.echo("  [3] Road-Grid structural alignment")
        click.echo("  [4] Experimental local road/tree mesh alignment")
        click.echo("  [5] Feature-agnostic local cell correlation (safe global fallback)")
        choice = click.prompt("Select alignment mode", type=click.Choice(["1", "2", "3", "4", "5"]))
        selected_mode = {"1": "manual", "2": "automated", "3": "road_grid", "4": "local_mesh", "5": "local_correlation"}[choice]

    if selected_mode == "local-correlation":
        selected_mode = "local_correlation"

    try:
        if selected_mode == "manual":
            cfg.alignment_mode = AlignmentMode.MANUAL
            click.echo("\n--- Manual Alignment Control Points ---")
            ids = None
            roles = None

            if gcp_file is not None:
                click.echo(f"  Reading control points from {gcp_file} (format={gcp_format}, source_units={gcp_source_units})")
                if gcp_format == "csv":
                    gcps = read_gcp_csv(gcp_file)
                    selected_coordinate_mode = ManualCoordinateMode(gcp_source_units)
                    pts_rgb = [g.rgb for g in gcps]
                    pts_ms = [g.ms for g in gcps]
                else:
                    gcps = read_qgis_points(gcp_file, source_units=gcp_source_units)
                    pts_rgb = [g.rgb for g in gcps]
                    selected_coordinate_mode = ManualCoordinateMode.MAP
                    if gcp_source_units == "map":
                        pts_ms = [g.ms for g in gcps]
                    else:
                        # QGIS reference (rgb) points are always map units; its
                        # source (ms) points here are native MS pixels. Convert
                        # them to map units ourselves so the whole point set can
                        # be passed through in one coordinate_mode, applying the
                        # same pixel-centre convention the pipeline's own
                        # pixel-mode path would.
                        convention = pixel_convention or cfg.manual.pixel_convention
                        px_offset = 0.5 if convention == "center" else 0.0
                        with rasterio.open(ms_path) as ms_src:
                            ms_transform = ms_src.transform
                        pts_ms = [
                            tuple(ms_transform * (g.ms[0] + px_offset, g.ms[1] + px_offset)) for g in gcps
                        ]
                ids = [g.gcp_id for g in gcps]
                roles = [g.role for g in gcps]
                click.echo(f"  Loaded {len(gcps)} point(s): {sum(1 for r in roles if r == 'check')} check, "
                           f"{sum(1 for r in roles if r == 'control')} control")
            else:
                coordinate_mode = click.prompt(
                    "Coordinate system ([1] QGIS map X/Y in raster CRS, [2] native pixel X/Y)",
                    type=click.Choice(["1", "2"]), default="1",
                )
                selected_coordinate_mode = (ManualCoordinateMode.MAP if coordinate_mode == "1"
                                            else ManualCoordinateMode.PIXEL)
                if selected_coordinate_mode == ManualCoordinateMode.MAP:
                    click.echo("  Enter QGIS map X/Y in EPSG:32642 (not latitude/longitude).")
                else:
                    click.echo("  Enter native raster Pixel X/Y; do not use QGIS map coordinates.")
                num_pts = click.prompt("Number of control point pairs", type=int, default=1)
                pts_rgb = []
                pts_ms = []
                for i in range(1, num_pts + 1):
                    click.echo(f"\nControl Point Pair #{i}:")
                    rgb_x = click.prompt(f"  RGB Reference Point #{i} X", type=float)
                    rgb_y = click.prompt(f"  RGB Reference Point #{i} Y", type=float)
                    ms_x = click.prompt(f"  MS Target Point #{i} X", type=float)
                    ms_y = click.prompt(f"  MS Target Point #{i} Y", type=float)
                    pts_rgb.append((rgb_x, rgb_y))
                    pts_ms.append((ms_x, ms_y))

            result = manual_align_orthomosaics(
                rgb_path=rgb_path,
                ms_path=ms_path,
                output_dir=out_dir,
                pts_rgb=pts_rgb,
                pts_ms=pts_ms,
                coordinate_mode=selected_coordinate_mode,
                config=cfg,
                ids=ids,
                roles=roles,
                pixel_convention=pixel_convention,
            )
        elif selected_mode == "automated":
            cfg.alignment_mode = AlignmentMode.AUTOMATED
            click.echo(f"  Detector:     {cfg.features.detector.value}")
            result = align_orthomosaics(
                rgb_path=rgb_path,
                ms_path=ms_path,
                output_dir=out_dir,
                config=cfg,
            )
        elif selected_mode == "road_grid":
            cfg.alignment_mode = AlignmentMode.ROAD_GRID
            click.echo("  Strategy:     Road-Grid structural alignment")
            result = road_grid_align_orthomosaics(
                rgb_path=rgb_path,
                ms_path=ms_path,
                output_dir=out_dir,
                config=cfg,
            )
        elif selected_mode == "local_mesh":
            cfg.alignment_mode = AlignmentMode.LOCAL_MESH
            click.echo("  Strategy:     Experimental local road/tree mesh (falls back to global if rejected)")
            result = local_mesh_align_orthomosaics(
                rgb_path=rgb_path,
                ms_path=ms_path,
                output_dir=out_dir,
                config=cfg,
            )
        elif selected_mode == "local_correlation":
            cfg.alignment_mode = AlignmentMode.LOCAL_CORRELATION
            click.echo("  Strategy:     Feature-agnostic local cell correlation (falls back to verified global alignment)")
            result = local_correlation_align_orthomosaics(
                rgb_path=rgb_path,
                ms_path=ms_path,
                output_dir=out_dir,
                config=cfg,
            )

        click.echo("\n[SUCCESS] Alignment process complete!")
        click.echo(f"  Aligned MS File:  {result.aligned_ms_path}")
        click.echo(f"  QA Preview Image: {result.preview_image_path}")
        click.echo(f"  JSON Report:      {result.report_json_path}")
        click.echo(f"  Status:           {result.spatial_residual_report.status}")
        global_rmse = result.spatial_residual_report.global_rmse_px
        edge_rmse = result.spatial_residual_report.edge_corner_rmse_px
        click.echo(f"  Global RMSE:      {global_rmse:.2f} px" if global_rmse is not None else "  Global RMSE:      N/A (verified by mode-specific checks, see below / report)")
        click.echo(f"  Edge/Corner RMSE: {edge_rmse:.2f} px" if edge_rmse is not None else "  Edge/Corner RMSE: N/A (verified by mode-specific checks, see below / report)")

        with open(result.report_json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        click.echo(f"  Applied Mode:     {report_data.get('applied_alignment_mode')}")

        local_refinement = report_data.get("local_refinement")
        if local_refinement:
            holdout = local_refinement.get("holdout") or {}
            before, after = holdout.get("median_before_px"), holdout.get("median_after_px")
            if before is not None and after is not None:
                click.echo(f"  AROSICS Holdout:  {before:.2f} px -> {after:.2f} px")
            p90 = (local_refinement.get("full_grid_verification") or {}).get("p90_px")
            if p90 is not None:
                click.echo(f"  AROSICS Grid p90: {p90:.2f} px")
            valid_tp = (local_refinement.get("tie_points") or {}).get("valid")
            if valid_tp is not None:
                click.echo(f"  AROSICS Tie Pts:  {valid_tp} valid")

        manual_cp = report_data.get("manual_control_points")
        if manual_cp:
            click.echo(f"  Manual Model:     {manual_cp.get('model')} ({manual_cp.get('model_reason')})")
            loo_selected = manual_cp.get("loo_rmse_selected_px")
            if loo_selected is not None:
                click.echo(f"  Manual LOO RMSE:  {loo_selected:.2f} px")
            checkpoint_rmse = manual_cp.get("checkpoint_rmse_px")
            if checkpoint_rmse is not None:
                click.echo(f"  Manual Check RMSE: {checkpoint_rmse:.2f} px")

    except Exception as e:
        click.echo(f"\n[ERROR] Alignment failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
