# Sentinel Crop Health Blueprint

> New developers should start with the standalone [Developer Onboarding and Extension Guide](./CROP_HEALTH_SENTINEL_DEVELOPER_ONBOARDING.md). It expands this short blueprint into a complete handoff covering formulas, contracts, edge cases, outputs, and the steps required to add a new report.

This package is the Sentinel-specific crop-health implementation. It mirrors the Planet crop-health pipeline shape, but the data acquisition and H3 aggregation logic are Sentinel-specific.

## Package Entry Points

[`crop_health_sentinel/__init__.py`](./__init__.py) exports:

- `run_crop_health_report`
- `generate_sentinel_crop_health_report`
- `get_sentinel_crop_health_report_metadata`

The main pipeline entry is [`pipeline.py`](./pipeline.py). The final wrapper is [`sentinel_crop_health_report.py`](./sentinel_crop_health_report.py).

## Package Layout

```text
crop_health_sentinel/
  io/
    phase_io.py
    raster_io.py
    scene_io_sentinel.py
    scene_io_v3.py
  layer_a_quality/
    processor.py
  layer_b_spectral/
    processor.py
  layer_c_health/
    processor.py
  layer_d_report/
    processor.py
  layer_e_h3/
    processor.py
  models/
    v3_types.py
  utils/
    filesystem.py
    math_utils.py
    preview_utils.py
    stats_v3.py
  config/
    default_config_sentinel_v1.yaml
  pipeline.py
  sentinel_crop_health_report.py
  version.py
```

## Processing Sequence

1. Read the Sentinel config YAML.
2. Resolve Sentinel band paths under `raw_data/`.
3. Load the NIR band as the raster reference profile.
4. Align red, red-edge, green, NIR, SCL, and auxiliary bands to the reference grid.
5. Build Sentinel quality masks.
6. Rasterize the crop polygon into `field_mask` and `field_inner_mask`.
7. Run Layer A quality filtering.
8. Run Layer B spectral summaries.
9. Run Layer C health scoring.
10. Run Layer E H3 aggregation.
11. Run Layer D final report creation.

## Input Expectations

The pipeline expects a `raw_data/` directory containing at least:

- `B3.tif`
- `B4.tif`
- `B5.tif`
- `B8.tif`
- `SCL.tif`

Optional bands are consumed when present and included in the quality calculations.

The crop geometry may be provided as:

- a GeoJSON Polygon
- a GeoJSON MultiPolygon
- a Feature containing one of those geometries
- a FeatureCollection whose first feature contains one of those geometries

## Layer Outputs

### Layer A

Creates trust masks and scene status, then writes:

- field mask
- inner field mask
- display-valid mask
- analytic-valid mask
- quality weight raster
- reason-code raster
- layer summary JSON and CSV

### Layer B

Creates:

- reflectance rasters for red, red-edge, green, and NIR
- NDVI, NDRE, GNDVI, and EVI2 rasters when enabled
- summary statistics for all computed bands and indices

### Layer C

Creates:

- field health score
- health class
- weak-area fraction
- pixel health map
- pixel health class map
- summary JSON and CSV

### Layer E

Creates:

- H3 health map
- H3 cell CSV
- H3 cell GeoJSON
- H3 overlay JSON
- all polygon H3 indexes JSON

### Layer D

Creates:

- final report JSON
- final report summary JSON and CSV

## Wrapper Behavior

[`sentinel_crop_health_report.py`](./sentinel_crop_health_report.py) is the outermost convenience wrapper.

It:

- validates `raw_data/`
- validates the required Sentinel files
- validates the crop geometry
- runs the pipeline
- creates `crophealthindex.json`
- validates that the output JSON is a list of `{ h3Index, healthScore }`
- prunes intermediate files after success

The wrapper also stores small metadata records keyed by the final output path so callers can inspect:

- `averageNdvi`
- `fieldHealthScore`
- `sceneStatus`

## Implementation Notes

The Sentinel implementation intentionally keeps the code path explicit:

- quality logic is in `layer_a_quality`
- spectral math is in `layer_b_spectral`
- field scoring is in `layer_c_health`
- reporting is in `layer_d_report`
- H3 aggregation is in `layer_e_h3`

That separation makes it easier to audit, test, and update each stage independently.

