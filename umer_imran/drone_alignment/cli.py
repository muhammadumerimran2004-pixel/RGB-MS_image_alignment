import logging
from pathlib import Path
import click
import yaml

from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, ResolutionMode, DetectorType, ManualCoordinateMode
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
    "--enable-loftr",
    is_flag=True,
    help="Try optional LoFTR structural matching if ORB/SIFT candidates are rejected.",
)
@click.option(
    "--loftr-max-tiles",
    type=click.IntRange(1, 144),
    default=None,
    help="Bound LoFTR tiles per candidate for a quick trial; omit for the configured default.",
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
    enable_loftr: bool,
    loftr_max_tiles: int | None,
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
        cfg.loftr.enabled = enable_loftr
        if loftr_max_tiles is not None:
            cfg.loftr.max_tiles = loftr_max_tiles

    click.echo("Starting Drone Alignment Engine...")
    click.echo(f"  RGB Reference: {rgb_path}")
    click.echo(f"  MS Target:    {ms_path}")
    click.echo(f"  Output Dir:   {out_dir}")
    click.echo(f"  Resolution:   {cfg.resolution_mode.value}")

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
        click.echo(f"  Global RMSE:      {global_rmse:.2f} px" if global_rmse is not None else "  Global RMSE:      N/A (manual / image-domain verification)")
        click.echo(f"  Edge/Corner RMSE: {edge_rmse:.2f} px" if edge_rmse is not None else "  Edge/Corner RMSE: N/A (manual / image-domain verification)")

    except Exception as e:
        click.echo(f"\n[ERROR] Alignment failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
