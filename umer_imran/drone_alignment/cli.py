import logging
from pathlib import Path
import click
import yaml

from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, ResolutionMode, DetectorType
from drone_alignment.pipeline import align_orthomosaics, manual_align_orthomosaics


@click.command()
@click.argument("rgb_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("ms_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["manual", "automated", "prompt"]),
    default="prompt",
    help="Alignment mode: 'manual' GCP control points, 'automated' pipeline, or 'prompt' for interactive menu.",
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
        choice = click.prompt("Select alignment mode", type=click.Choice(["1", "2"]), default="1")
        selected_mode = "manual" if choice == "1" else "automated"

    try:
        if selected_mode == "manual":
            cfg.alignment_mode = AlignmentMode.MANUAL
            click.echo("\n--- Manual Alignment Control Points ---")
            click.echo("  (Supports QGIS Map Easting/Northing in meters OR Pixel X/Y)")
            num_pts = click.prompt("Number of control point pairs", type=int, default=1)
            pts_rgb = []
            pts_ms = []
            for i in range(1, num_pts + 1):
                click.echo(f"\nControl Point Pair #{i}:")
                rgb_x = click.prompt(f"  RGB Reference Point #{i} X coordinate (Map Easting / Pixel X)", type=float)
                rgb_y = click.prompt(f"  RGB Reference Point #{i} Y coordinate (Map Northing / Pixel Y)", type=float)
                ms_x = click.prompt(f"  MS Target Point #{i} X coordinate (Map Easting / Pixel X)", type=float)
                ms_y = click.prompt(f"  MS Target Point #{i} Y coordinate (Map Northing / Pixel Y)", type=float)
                pts_rgb.append((rgb_x, rgb_y))
                pts_ms.append((ms_x, ms_y))

            result = manual_align_orthomosaics(
                rgb_path=rgb_path,
                ms_path=ms_path,
                output_dir=out_dir,
                pts_rgb=pts_rgb,
                pts_ms=pts_ms,
                config=cfg,
            )
        else:
            cfg.alignment_mode = AlignmentMode.AUTOMATED
            click.echo(f"  Detector:     {cfg.features.detector.value}")
            result = align_orthomosaics(
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
