# Crop Health Sentinel: Developer Onboarding and Extension Guide

## Purpose of this document

This is a standalone technical handoff for a developer who needs to understand the Sentinel crop-health report without receiving the complete repository. It explains the business goal, processing stages, data contracts, calculations, configuration, outputs, failure behavior, and the pattern to follow when adding a new report.

It intentionally contains no credentials, customer data, satellite scenes, deployment secrets, or complete source listing. A developer can use it to design a compatible report package; a repository maintainer will still need to integrate and review that package inside the worker.

## 1. What the utility does

The `crop_health_sentinel` package converts Sentinel-2 surface-reflectance bands and one crop boundary into a spatial crop-health result.

The production result is a JSON list like this:

```json
[
  {
    "h3Index": "<H3 cell identifier>",
    "healthScore": 0.8725
  }
]
```

Each H3 cell represents a small part of the crop. `healthScore` in the production file is normalized to the range `0.0..1.0`. Internally, the pipeline performs its calculations in the range `0..100` and divides by 100 only when creating the production file.

The report is a current-scene spectral assessment. It does **not** use time-series trends, weather, crop stage, yield history, field observations, or a machine-learning model. It should therefore be interpreted as “how the vegetation signal in this image compares with configured spectral ranges,” not as a diagnosis of a specific disease or nutrient deficiency.

## 2. Where this package fits in the larger worker

The surrounding `satellite_data_processing` worker owns message handling, backend calls, imagery download, cloud gating, local workspaces, S3 upload, and result persistence. The crop-health package only owns the per-crop analytical report.

The full Sentinel path is:

```text
Queue/debug message
  -> worker validates the request and fetches farm/crop context
  -> Sentinel downloader selects and exports a scene for the farm AOI
  -> per-crop cloud gate decides which crop-health tasks may run
  -> crop_health_sentinel production wrapper validates report inputs
  -> five-layer analytical pipeline runs for one crop AOI
  -> H3 overlay is converted to crophealthindex.json
  -> report intermediates are pruned
  -> worker uploads allowed outputs and saves backend metadata
```

There are two different AOI scopes:

- The **farm AOI** is used to select/download the Sentinel scene.
- Each **crop AOI** is used for cloud gating and for one crop-health report run.

The worker currently downloads Sentinel imagery fresh for the requested date. It does not restore previously downloaded Sentinel raw data from S3 in the normal production branch.

### Two quality checks that must not be confused

1. The worker-level crop cloud gate evaluates cloud percentage for each crop before the report runs. A crop over the configured threshold is skipped.
2. Layer A inside this package creates pixel-level trust masks and assigns a scene status after the report starts.

These checks happen at different levels and serve different contracts. A new report should explicitly decide whether it uses the existing worker gate, its own Layer A rules, or both.

## 3. Package structure and responsibilities

```text
crop_health_sentinel/
  __init__.py                         Public package exports
  version.py                          Package version
  pipeline.py                         Analytical orchestration
  sentinel_crop_health_report.py      Production adapter/wrapper
  config/
    default_config_sentinel_v1.yaml   Default algorithm configuration
  models/
    v3_types.py                       In-memory stage contracts
  io/
    scene_io_sentinel.py              Band reading and grid alignment
    scene_io_v3.py                    Geometry loading and rasterization
    raster_io.py                      Single-band GeoTIFF writing
    phase_io.py                       Standard phase artifact writing
  layer_a_quality/
    processor.py                      Trust masks and scene status
  layer_b_spectral/
    processor.py                      Reflectance and spectral indices
  layer_c_health/
    processor.py                      Field and pixel health scores
  layer_e_h3/
    processor.py                      Pixel-to-H3 spatial aggregation
  layer_d_report/
    processor.py                      Final explainable report assembly
  utils/
    filesystem.py                     JSON/CSV/directory helpers
    math_utils.py                      Safe division and normalization
    stats_v3.py                       Statistics and class labels
    preview_utils.py                  PNG visualization generation
```

The unusual execution order is deliberate:

```text
Layer A -> Layer B -> Layer C -> Layer E -> Layer D
```

Layer D has phase number 4 because it is the logical final report layer. It runs after Layer E so the final report can include the H3 summary. Layer E writes to `phase_05` even though it executes first.

## 4. Public entry points

The package exposes three operations.

### `run_crop_health_report(...)`

This is the analytical pipeline. It retains detailed phase outputs and returns a rich in-memory result. Use it during algorithm development, debugging, and validation.

Conceptual inputs:

- `raw_data_dir`: directory containing Sentinel bands.
- `crop_geometry`: crop boundary as GeoJSON or a JSON string.
- `output_dir`: directory for phase outputs.
- `scene_meta`: farm, field, date, and scene identifiers.
- `config_path` or `config`: optional algorithm override.
- `logger`: optional worker-compatible or standard logger.

### `generate_sentinel_crop_health_report(...)`

This is the production adapter called by the worker. It validates inputs, runs the pipeline, converts the H3 overlay to the public JSON contract, validates the final score range, and prunes intermediate files.

Its return value is the **absolute path** to `crophealthindex.json`, not the parsed report object.

### `get_sentinel_crop_health_report_metadata(output_path)`

The adapter temporarily stores three values by final output path:

- `averageNdvi`
- `fieldHealthScore`
- `sceneStatus`

Reading the metadata removes it from the in-process cache. This is a short-lived handoff mechanism for the worker, not persistent storage and not an API for later retrieval.

Current integration caveat: the production worker reads `averageNdvi` from this metadata handoff, but it does not currently rebuild the rich pipeline result from it. Consequently, field score, health class, H3 summary, and scene status are available from a direct pipeline run but are not all propagated through the Sentinel worker result payload today. Treat that as an integration limitation to address explicitly if a new report needs those fields saved by the backend.

## 5. Input contract

### 5.1 Required raw bands

The following files must exist directly under `raw_data_dir`:

| Logical signal | Sentinel band/file | Typical native resolution | Use |
|---|---|---:|---|
| Green | `B3.tif` | 10 m | GNDVI |
| Red | `B4.tif` | 10 m | NDVI and EVI2 |
| Red edge | `B5.tif` | 20 m | NDRE |
| Near infrared | `B8.tif` | 10 m | Reference grid and all indices |
| Scene classification | `SCL.tif` | 20 m | Cloud, shadow, snow, and unusable masks |

`B8.tif` is the reference raster. Every other band is aligned to its CRS, affine transform, width, and height.

- Continuous reflectance and AOT bands use bilinear resampling.
- Categorical masks and quality flags use nearest-neighbor resampling.

This distinction is essential: bilinear interpolation is suitable for continuous measurements but would invent invalid intermediate class values for SCL/QA data.

### 5.2 Optional quality bands

The pipeline consumes these files when present:

- `MSK_CLDPRB.tif`: cloud probability
- `MSK_CLASSI_OPAQUE.tif`: opaque cloud flag
- `MSK_CLASSI_CIRRUS.tif`: cirrus flag
- `MSK_CLASSI_SNOW_ICE.tif`: snow/ice flag
- `QA60.tif`: Sentinel QA cloud/cirrus bits
- `AOT.tif`: aerosol optical thickness used as a haze indicator

When an optional raster is missing, it is represented by zeros. Consequences of this fallback must be understood:

- The missing source contributes no cloud, snow, QA, or haze flags.
- Missing cloud probability yields confidence `100`, because confidence is calculated as `100 - cloud_probability`.
- SCL remains required and still provides the baseline quality classification.

### 5.3 Crop geometry

Accepted forms are:

- GeoJSON `Polygon`
- GeoJSON `MultiPolygon`
- GeoJSON `Feature` containing either geometry
- GeoJSON `FeatureCollection`; only its first feature is used
- A JSON string containing one of the above

Coordinates are assumed to be longitude/latitude in `EPSG:4326`. Polygon rings must contain at least four points and must be closed (`first point == last point`). The geometry is transformed to the raster CRS before rasterization.

The default `all_touched` rasterization mode includes every pixel touched by the polygon. If rasterization produces zero pixels—for example, the crop lies outside the scene—the run fails.

### 5.4 Scene metadata

The normalized internal metadata fields are:

| Field | Accepted aliases | Default |
|---|---|---|
| `farm` | `farm`, `farm_no`, `farmId` | `farm` |
| `field` | `field`, `crop_unique_no`, `cropUniqueNo`, `cropId` | `crop` |
| `date` | `date`, `imageryDate` | `unknown-date` |
| `scene_id` | `scene_id`, `sceneId` | `sentinel-scene` |

The worker normally supplies farm number/farm ID, crop unique number/crop ID, request date, and run ID.

## 6. Shared in-memory contract

The immutable `SceneContext` carries all data needed by every layer:

- normalized identifiers;
- source paths and output base path;
- loaded configuration;
- reference raster profiles;
- aligned reflectance arrays; and
- aligned quality arrays.

Each layer returns a typed output object rather than forcing the next layer to rediscover files. The important contracts are:

- `LayerAOutput`: field masks, valid masks, quality weights, reason codes, scene status, summary.
- `SpectralOutput`: reflectance arrays, index arrays, summary.
- `HealthOutput`: field score/class, weak fraction, pixel health/class maps, score details, summary.
- `LayerEOutput`: H3 map, per-cell rows, summary.
- `FinalReportOutput`: assembled report and compact summary.

This pattern is worth preserving in new reports: files are durable diagnostics, while typed stage outputs are the internal execution contract.

## 7. End-to-end processing logic

### 7.1 Setup and alignment

The pipeline performs the following setup:

1. Load the supplied configuration dictionary, or load the default YAML.
2. Write the supplied crop geometry to `crop_polygon.json` in the report directory.
3. Resolve required/optional band paths from the `paths` config section.
4. Open NIR (`B8`) and copy its raster profile as the reference grid.
5. Align red, red edge, green, NIR, SCL, and optional quality bands to that grid.
6. Build Sentinel quality arrays.
7. Rasterize the crop boundary into `field_mask` and `field_inner_mask`.
8. Build `SceneContext` and call the layers.

### 7.2 Sentinel quality arrays

With the default configuration, masks are built as follows:

- **Cloud**: SCL class 8, 9, or 10; or opaque flag; or cirrus flag; or cloud probability at least 70; or QA60 bit 10/11.
- **Shadow**: SCL class 3.
- **Snow**: SCL class 11 or the snow/ice flag.
- **Unusable**: SCL class 0 or 1, non-finite required reflectance, or an invalid NDVI calculation.
- **Haze**: `(AOT / 10000) >= 0.60`.
- **Clear**: none of unusable, cloud, shadow, haze, or snow.
- **Confidence**: cloud probability converted to `0..100` confidence with `clip(100 - probability, 0, 100)`.

NDVI used during this quality setup is:

```text
NDVI = (NIR - Red) / (NIR + Red)
```

Division is allowed only where the inputs are finite and the absolute denominator is greater than `1e-6`. Invalid locations become `NaN` and are considered unusable.

## 8. Layer A — quality and trust

Layer A answers two questions:

1. Which pixels may be displayed or analyzed?
2. Does the scene have enough trusted crop coverage to continue?

### 8.1 Field and inner masks

The field polygon is rasterized on the NIR grid. An optional erosion can create an inner mask to reduce boundary mixing. Defaults are:

- `edge_erosion_pixels: 0`
- `use_field_inner_mask: false`
- `field_rasterization_mode: all_touched`

Therefore, the current default uses the full touched-pixel field mask without erosion.

### 8.2 Display mask versus analytic mask

The two masks intentionally have separate configuration.

Default display exclusions are:

- unusable;
- cloud;
- shadow;
- snow; and
- invalid numeric data.

Haze is not excluded from display by default.

Default analytic exclusions are only:

- unusable; and
- invalid numeric data.

Cloud, shadow, haze, and snow are **not** excluded from the analytic mask in the current default configuration. This is a consequential policy choice, not a universal Sentinel rule. Anyone changing it must compare score coverage and downstream behavior on representative scenes.

An optional vegetation gate can also remove analytic pixels below an NDVI threshold. It is disabled by default; its configured threshold is `0.05`.

### 8.3 Quality weight

For analytic-valid pixels:

```text
quality_weight = confidence / 100
```

Other pixels receive zero. In the current version, this raster is saved as a diagnostic but Layer B, Layer C, and Layer E do not use it to weight their calculations.

### 8.4 Reason codes

The reason-code raster uses:

| Code | Meaning |
|---:|---|
| 0 | No exclusion reason |
| 1 | Outside field |
| 2 | Unusable |
| 3 | Cloud |
| 4 | Shadow |
| 5 | Haze |
| 6 | Snow |
| 7 | Invalid numeric input |
| 8 | Low vegetation |

Assignments occur in that order, so later matching reasons overwrite earlier ones. A low-vegetation pixel therefore ends with code 8 even if another earlier condition was also true.

### 8.5 Scene status

Let:

```text
analytic_fraction = analytic_valid_pixels / selected_field_pixels
display_fraction  = display_valid_pixels  / selected_field_pixels
```

The default status decision is:

- `reject` if analytic-valid pixel count is below 5 or analytic fraction is below 0.15;
- `analytics_ready` if analytic fraction is at least 0.40; or
- `display_only` otherwise (normally 0.15 through less than 0.40).

`display_fraction` is reported but does not participate in the status decision.

If status is `reject`, the pipeline stops after Layer A. If status is `display_only`, the current implementation still runs Layers B, C, E, and D. The name should not be interpreted as a hard execution stop.

## 9. Layer B — reflectance and spectral indices

Sentinel surface-reflectance digital numbers are divided by `10000` before calculations.

Layer B computes the enabled indices over `analytic_valid_mask`:

```text
NDVI  = (NIR - Red)      / (NIR + Red)
NDRE  = (NIR - RedEdge)  / (NIR + RedEdge)
GNDVI = (NIR - Green)    / (NIR + Green)
EVI2  = 2.5 * (NIR - Red) / (NIR + 2.4 * Red + 1.0)
```

Every division uses the safe-division rule described above. Defaults enable all four indices.

For each reflectance band and enabled index, the layer records:

- valid count;
- mean;
- standard deviation;
- minimum and maximum;
- coefficient of variation, `std / (abs(mean) + epsilon)`; and
- percentiles 5, 10, 25, 50, 75, 90, and 95.

It also writes a GeoTIFF and PNG preview for each reflectance band and index.

## 10. Layer C — health scoring

Layer C produces one field score, one field class, a weak-area fraction, and a health score/class per analytic pixel.

### 10.1 Field score

The configured Layer B summary statistic is `mean`. For each available index, the mean is converted to a `0..100` score:

```text
normalized_index_score =
    clip((index_mean - configured_low) /
         (configured_high - configured_low + epsilon), 0, 1) * 100
```

Default ranges and weights are:

| Index | Low | High | Weight |
|---|---:|---:|---:|
| NDVI | 0.15 | 0.80 | 0.20 |
| NDRE | 0.05 | 0.45 | 0.40 |
| GNDVI | 0.08 | 0.70 | 0.25 |
| EVI2 | 0.10 | 0.90 | 0.15 |

The field score is:

```text
field_health_score =
    sum(normalized_index_score * index_weight) / sum(weights actually used)
```

Missing/disabled indices are skipped and the remaining weights are re-normalized by their sum. The run fails if no weighted index value is available.

Default classes are:

| Score | Class |
|---:|---|
| `>= 85` | Very Good |
| `>= 70 and < 85` | Good |
| `>= 55 and < 70` | Moderate |
| `>= 40 and < 55` | Weak |
| `< 40` | Poor |

### 10.2 Weak-area fraction

The default weak source is NDRE with threshold `0.18`:

```text
weak_area_fraction =
    analytic pixels with finite NDRE < 0.18 / all analytic-valid pixels
```

If the configured source index is unavailable, the weak fraction becomes `0.0`.

### 10.3 Pixel health map

The default pixel source is NDRE, falling back to NDVI only if NDRE is unavailable. The selected pixel index is normalized using the configured range and scaled to `0..100`. Pixel classes use codes 1 through 5 for Poor through Very Good; code 0 is background/nodata.

Configuration caveat: the current implementation chooses the normalization range by the configured **primary index name**, even when the data array came from the fallback index and the primary range exists. Keep the primary index enabled, or explicitly correct/test this behavior before relying on fallback scoring.

## 11. Layer E — H3 spatial aggregation

Layer E converts pixel health into H3 cells. Default resolution is 13.

The important design goal is to handle boundary pixels spatially rather than assigning each pixel only by its center.

### 11.1 Candidate H3 cells

The crop polygon is converted to H3 geometry. When the installed H3 library supports experimental overlap containment, every H3 cell overlapping the polygon is requested. Compatibility fallbacks use standard/legacy polygon-to-cell APIs.

The candidate polygons are transformed to an equal-area metric CRS, default `EPSG:6933`, and stored in a spatial index. The raster must have a CRS; otherwise this layer fails.

### 11.2 Area-share aggregation

For each analytic pixel:

1. Build its four-corner polygon using the raster affine transform.
2. Transform that polygon to the metric-area CRS.
3. Query potentially intersecting H3 cell polygons.
4. Calculate the real intersection area with each cell.
5. Convert areas into shares of the pixel's total H3 overlap.
6. Add `pixel_health * share` to each intersected cell.

The cell mean is:

```text
cell_mean_health = sum(pixel_health * overlap_share) / sum(overlap_share)
```

Only cells receiving analytic pixel contributions are emitted, even if additional cells came from the crop polygon candidate set.

### 11.3 Support diagnostics

Each cell records:

- `actual_pixel_count`: field pixels whose centers belong to this H3 cell;
- `analytic_pixel_count`: analytic pixels that intersect the cell;
- `effective_analytic_pixel_count`: sum of overlap shares;
- mean, minimum, maximum, and standard deviation of contributing pixel scores;
- score source (`analytic`); and
- whether effective support is below the configured threshold.

The default minimum effective support is `0.01`. Low-support cells are tagged but **not filtered out**.

The minimum/maximum/standard-deviation diagnostics use the list of intersecting pixel values, while the cell mean uses overlap shares. They therefore describe related but not identically weighted populations.

### 11.4 H3 outputs

Layer E writes:

- an H3-valued GeoTIFF and preview;
- a per-cell CSV;
- a geographic GeoJSON FeatureCollection;
- a pixel-coordinate overlay JSON used by the production index builder; and
- a JSON list of emitted H3 indexes.

It also reports the cell-score mean and p10/p50/p90 across emitted cells. These are cell-level statistics, not area-weighted field statistics.

### 11.5 Configuration values currently retained but not active

Several keys remain in the default YAML but the current H3 algorithm does not apply them as behavior:

- `include_all_field_h3_cells`
- `polygon_h3_containment`
- `min_field_pixels_per_h3`
- `min_cell_polygon_overlap_fraction`
- `missing_cell_fallback`
- `neighbor_fallback_k_ring`

The output explicitly reports fallback logic as disabled. `score_range` and H3 `class_thresholds` are copied into summaries, while current H3 color interpolation uses fixed `0..100` anchors. Do not assume that editing one of these retained keys changes the algorithm; add a test that proves any new configuration is actually consumed.

## 12. Layer D — final explainable report

Layer D combines:

- scene identity and source paths;
- Layer A trust summary;
- Layer B spectral summary;
- Layer C health summary; and
- Layer E H3 summary.

It also records explicit limitations: the report is current-scene only, contains no time trend/weather/crop-stage/ML logic, and excludes pixels according to Layer A rules.

The compact Layer D summary contains scene status, field score, and health class.

## 13. Output lifecycle

### 13.1 Direct pipeline output

A non-rejected direct pipeline run produces approximately:

```text
report_dir/
  crop_polygon.json
  phase_01/
    <scene>_field_mask.tif + preview
    <scene>_field_inner_mask.tif + preview
    <scene>_display_valid_mask.tif + preview
    <scene>_analytic_valid_mask.tif + preview
    <scene>_quality_weight.tif + preview
    <scene>_reason_code_mask.tif + preview
    <scene>_layer_a_quality_summary.json/.csv
  phase_02/
    <scene>_<index>.tif + preview
    <scene>_reflectance_<band>.tif + preview
    <scene>_layer_b_spectral_summary.json/.csv
  phase_03/
    <scene>_pixel_health_map.tif + preview
    <scene>_pixel_health_class_map.tif + preview
    <scene>_layer_c_health_summary.json/.csv
  phase_04/
    <scene>_final_report.json/.csv
    <scene>_layer_d_report_summary.json/.csv
  phase_05/
    <scene>_h3_health_map.tif + preview
    <scene>_layer_e_h3_summary.json/.csv
    <scene>_h3_cells.csv
    <scene>_h3_cells.geojson
    <scene>_h3_cells_overlay.json
    <scene>_all_polygon_h3_indexes.json
```

A rejected direct run stops after `phase_01` and returns empty summaries for later layers.

### 13.2 Production wrapper output

The production adapter finds exactly one H3 overlay, maps each cell to `{h3Index, healthScore}`, divides scores by 100, rounds to four decimal places, and writes `crophealthindex.json`.

It calls the index builder with pruning enabled, so successful production output normally contains only:

```text
report_dir/
  crophealthindex.json
```

For a rejected scene, a missing H3 overlay is allowed and the final file is an empty list. For an analytics-ready scene, a missing overlay is an error. A display-only scene currently continues through H3, so it normally also has an overlay.

On adapter failure, created/stale contents under the report directory are removed and the original exception is re-raised. The output directory should therefore be treated as owned by one report run; do not place unrelated files in it.

### 13.3 Worker upload behavior

The worker excludes the following from S3 upload:

- `ndvi_raw.tif`; and
- every file under `raw_data/` for Sentinel.

The report folder and production index follow the crop report path configuration returned by the backend. That path must be the farm S3 prefix plus exactly one final report folder.

## 14. Default configuration reference

| Area | Default | Meaning |
|---|---|---|
| Reflectance scale | `10000` | Converts Sentinel SR values to reflectance-like values |
| Safe denominator epsilon | `1e-6` | Prevents unstable division |
| Rasterization | `all_touched` | Includes boundary-touched pixels |
| Edge erosion | `0` | No boundary erosion |
| Cloud probability | `70` | Pixel cloud threshold |
| AOT haze | `0.60` after scaling | Pixel haze threshold |
| Analytics-ready | analytic fraction `>= 0.40` | Full status |
| Display-only floor | analytic fraction `>= 0.15` | Below 0.40 becomes display-only |
| Minimum analytic pixels | `5` | Below this rejects |
| Spectral statistic | `mean` | Field scoring input |
| Weak area | NDRE `< 0.18` | Weak-pixel rule |
| H3 resolution | `13` | Spatial aggregation level |
| H3 metric CRS | `EPSG:6933` | Intersection-area calculations |
| H3 support threshold | `0.01` effective pixel | Diagnostic tag only |

Configuration is part of the algorithm contract. Any threshold or weight change should be versioned, tested on representative scenes, and communicated as a report behavior change.

## 15. Failure and edge-case behavior

| Condition | Result |
|---|---|
| Required raw band missing | Adapter/pipeline fails before analysis |
| Optional quality band missing | Zero-filled fallback is used |
| Invalid/non-closed geometry | Adapter rejects it |
| FeatureCollection has multiple features | Only the first feature is used |
| Crop rasterizes to zero pixels | Pipeline fails |
| Raster CRS missing | Geometry rasterization or H3 layer fails |
| Too few analytic pixels | Scene status `reject`; no later layers |
| No usable health indices | Layer C fails |
| Primary and fallback pixel index unavailable | Layer C fails |
| No analytic pixels reach Layer E | Empty H3 result is written |
| H3 candidates cannot be created | Layer E fails |
| More than one overlay file exists | Index builder fails as ambiguous |
| Final score is outside `0..1` | Production adapter fails validation |
| Bucket generation fails in worker | Report is marked incomplete even if index generation succeeded |

## 16. Building a new report using the same structure

Use the same separation of concerns, but do not copy crop-health assumptions blindly. A suggested package is:

```text
<new_report>_sentinel/
  __init__.py
  version.py
  pipeline.py
  <new_report>_report.py
  config/default_config_v1.yaml
  models/types.py
  io/
  layer_a_quality/
  layer_b_features/
  layer_c_interpretation/
  layer_d_report/
  layer_e_spatial/          # only if a spatial cell product is needed
  utils/
  tests/
```

### Step 1: define the report contract first

Write down before implementation:

- required and optional bands;
- accepted geometry and CRS;
- score units/range;
- whether output is field-level, pixel-level, H3-level, or all three;
- exact public JSON schema;
- reject/empty-result semantics;
- artifacts retained in development versus production; and
- metadata the worker/backend needs.

Avoid calling every numeric result `healthScore`. Use domain-specific names and units.

### Step 2: create a single context and typed stage outputs

Load and align inputs once. Pass a context containing identifiers, config, profiles, and arrays to explicit stages. Return typed outputs from each stage so dependencies are visible and unit-testable.

### Step 3: keep stages single-purpose

A useful adaptation of this design is:

- **Layer A — trust:** quality masks, coverage, and eligibility.
- **Layer B — features:** physical/spectral features and summaries.
- **Layer C — interpretation:** score, classes, alerts, or recommendations.
- **Layer E — spatial aggregation:** optional H3/zonal product.
- **Layer D — report:** final explainable assembly after all needed layers.

If a report does not need H3, omit Layer E instead of creating an empty imitation. If it requires history or weather, add an explicit input/enrichment stage rather than hiding those calls inside scoring.

### Step 4: make configuration executable and versioned

Every configuration key should be read by the algorithm or removed. Separate:

- paths and band mapping;
- I/O/nodata behavior;
- quality policy;
- feature switches;
- scoring ranges/weights;
- class thresholds; and
- spatial aggregation rules.

Validate configuration at startup. Check weight/range compatibility, threshold ordering, required keys, and supported enum values.

### Step 5: provide two entry points

Follow the current distinction:

- a rich pipeline function for development and diagnostics; and
- a narrow production wrapper for validation, public-schema generation, cleanup, and absolute output-path return.

Keep cleanup in the wrapper so analytical layers remain inspectable when called directly.

### Step 6: integrate with worker routing

Repository maintainers must make coordinated changes outside the new package:

1. Register the new `reportType` in the worker report configuration.
2. Declare farm-level raster requirements. Sentinel crop health currently declares none because it reads `outputs/raw_data` directly; choose an explicit contract for the new report.
3. Replace the generic placeholder branch with a real report runner/adapter dispatch.
4. Add backend crop `reportConfigs` entries (`reportType`, `isEnabled`, `isProcessed`, path/bucket data as applicable).
5. Add a matching farm `reportPathConfigs` entry.
6. Ensure the report path is exactly one child folder beneath the farm prefix.
7. Map the runner result into the worker result/save payload.
8. Review upload exclusions so required public artifacts are retained and raw/internal artifacts are not exposed.
9. Decide whether the existing crop cloud gate applies to this report; it currently targets `cropHealth` specifically.
10. Add force-retry, already-processed, partial-failure, upload, and backend-save tests.

### Step 7: define compatibility at boundaries

The stable boundary is not the folder layout; it is the set of contracts between:

- downloader and report raw inputs;
- backend context and worker worklist;
- worker and report wrapper;
- pipeline stages;
- overlay/index builder or new public serializer;
- uploaded file path and backend save payload.

Document and test each boundary independently.

## 17. Recommended test matrix for a new report

### Unit tests

- safe division, normalization, and classification boundaries;
- each quality flag and overlapping reason-code precedence;
- each feature/index formula on small arrays;
- missing index and weight re-normalization;
- reject/display-only/analytics-ready boundaries;
- pixel-to-H3 overlap at cell boundaries;
- low-support cell tagging;
- Polygon, MultiPolygon, holes, and first-feature behavior;
- grid alignment with different source resolutions; and
- active versus missing optional bands.

### Contract tests

- missing required input produces a clear error;
- wrapper returns an absolute existing output path;
- final JSON matches the documented schema and score units;
- reject/empty output is intentional and valid;
- cleanup preserves only declared production files;
- metadata handoff returns expected values once; and
- version/config identifiers are included where required.

### Worker integration tests

- report enablement and already-processed skipping;
- force retry for one crop/report and all crops;
- cloud-gate pass and skip paths;
- correct crop AOI and scene metadata passed to the runner;
- S3 path and upload inclusion/exclusion;
- backend report availability and metadata;
- one crop failure does not corrupt other crop results; and
- report output plus bucket-generation success/failure.

### Scientific regression tests

Maintain small, approved reference scenes representing:

- clear vegetation;
- high cloud/shadow;
- a small or narrow field;
- a field crossing H3 boundaries;
- low vegetation/bare soil;
- MultiPolygon geometry; and
- missing optional quality products.

Compare masks, valid fractions, index summaries, field scores, H3 cell counts, and selected per-cell scores with tolerances. Threshold and resampling changes should never be accepted based only on “the pipeline ran.”

## 18. Common mistakes to avoid

- Treating the worker cloud gate and Layer A masks as the same check.
- Resampling SCL or QA bands with bilinear interpolation.
- Assuming optional quality files are neutral in every sense; missing cloud probability creates maximum confidence.
- Assuming `quality_weight` currently influences scoring.
- Assuming `display_only` stops later layers.
- Comparing the internal `0..100` H3 score directly with the public `0..1` score.
- Editing retained H3 YAML keys and assuming behavior changed without a consuming code path/test.
- Using a fallback pixel index with the primary index's normalization range.
- Interpreting an H3 cell mean as a disease diagnosis or historical trend.
- Placing unrelated files in a production report directory that the wrapper owns and prunes.
- Returning a valid file but forgetting worker registration, backend path configuration, upload rules, and save-payload mapping.

## 19. New-developer mental model

Remember the system in five sentences:

1. Sentinel download prepares one farm scene; the report then runs once per eligible crop boundary.
2. All bands are aligned to the NIR grid, and Layer A decides which crop pixels are trusted.
3. Layer B calculates spectral signals, and Layer C converts them into configured `0..100` health scores.
4. Layer E spatially shares boundary pixels across H3 cells using intersection area, then Layer D assembles an explainable report.
5. The production wrapper reduces all diagnostics to a validated `crophealthindex.json` with `0..1` scores, while the worker handles paths, uploads, and backend state.

## 20. Handoff checklist

A developer is ready to extend this design when they can answer all of the following:

- Which inputs are required, optional, categorical, and continuous?
- What raster grid and CRS are authoritative?
- How are display-valid and analytic-valid pixels different?
- What exact condition produces each scene status?
- Which indices feed the field score, weak fraction, and pixel map?
- Where does the score change from `0..100` to `0..1`?
- How does one pixel contribute to multiple H3 cells?
- Which artifacts exist only during direct pipeline development?
- Which configuration keys are active versus retained?
- What must change in worker routing and backend configuration for a new report type?
- What tests prove scientific behavior rather than merely successful execution?

If these answers are explicit for the new report, it can follow the same architecture without becoming tightly coupled to crop-health-specific assumptions.
