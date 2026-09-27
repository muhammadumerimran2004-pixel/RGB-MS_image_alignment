# AgriLift Sentinel Crop Health

## Decision-Complete Implementation Plan

**Status:** Approved build blueprint; no production code is implemented in this repository yet.  
**Audience:** Junior engineers working under normal code review.  
**Target package:** `crop_health_sentinel`  
**Algorithm identifier:** `sentinel-crop-health/1.0.0`  
**Public artifact:** `crophealthindex.json`  
**Source documents:** `CROP_HEALTH_SENTINEL_BLUEPRINT.md` and `CROP_HEALTH_SENTINEL_DEVELOPER_ONBOARDING.md`

When this plan conflicts with either source document, this plan is authoritative for the new implementation. The older documents explain the inherited behavior; they are not permission to restore a superseded caveat during coding.

---

## 1. Mission and Definition of Success

Build a deterministic Python package that converts one Sentinel-2 surface-reflectance scene and one crop boundary into:

1. trusted-pixel masks;
2. spectral-index rasters and summaries;
3. a field health score and per-pixel health surface;
4. area-overlap-weighted H3 health cells;
5. an explainable diagnostic report; and
6. the production contract:

```json
[
  {
    "h3Index": "8d...",
    "healthScore": 0.8725
  }
]
```

The implementation is complete only when all unit, contract, integration, resource, and scientific-regression gates in this plan pass. A successful function return is not sufficient.

### 1.1 Scope

The package owns only per-crop analysis. It owns:

- input validation after the worker has downloaded a scene;
- crop-window extraction and grid alignment;
- pixel quality decisions;
- spectral formulas;
- configured health scoring;
- H3 aggregation;
- diagnostic artifact generation;
- production JSON serialization; and
- a small compatibility metadata handoff.

### 1.2 Explicit non-goals

Do not add any of the following to this package:

- satellite search or download;
- queue/message handling;
- S3 access;
- backend API calls;
- crop-stage inference;
- disease or nutrient diagnosis;
- time-series, weather, or yield logic;
- machine-learning inference;
- farm-level cloud gating; or
- drone-image alignment.

Those belong to upstream or separate systems. Layer A remains a second, pixel-level quality gate and must not be presented as a replacement for the worker's crop cloud gate.

---

## 2. Decisions Already Made

Engineers must implement these decisions as written. Changing one requires an architecture decision record, a config/algorithm version change where behavior changes, and regression-test approval.

### ADR-001 — One crop, one scene, one invocation

`run_crop_health_report` processes exactly one crop geometry against exactly one already-downloaded Sentinel scene. Batch iteration belongs to the worker.

### ADR-002 — B8 crop window is the authoritative grid

`B8.tif` defines the CRS, base transform, pixel size, and nodata convention. The pipeline must first transform the crop geometry into the B8 CRS, calculate a crop bounding window with a two-pixel margin, intersect it with the B8 extent, and use that **cropped B8 window** as the reference grid.

This is intentionally safer than loading the full farm scene into memory. Every aligned array has exactly the reference window's `(height, width)`.

### ADR-003 — Resampling is determined by data semantics

- Continuous data (`B3`, `B4`, `B5`, `B8`, `AOT`, `MSK_CLDPRB`) uses bilinear resampling.
- Categorical/bit data (`SCL`, `QA60`, class masks) uses nearest-neighbor resampling.

No caller may override this per invocation. Adding a band requires declaring its semantic type in the band registry.

### ADR-004 — Geometry arrives in EPSG:4326

Accepted geometry forms are `Polygon`, `MultiPolygon`, `Feature`, and `FeatureCollection`. A feature collection must contain exactly one feature in the new implementation; silently selecting its first feature is forbidden. Rings may be closed during normalization, but invalid/self-intersecting geometry must fail with a clear error. Geometry reprojection always uses `Transformer(..., always_xy=True)`.

### ADR-005 — External scores use 0..1; internal scores use 0..100

All computation, diagnostic rasters, summaries, and H3 rows use `float` scores in `[0, 100]`. Only `sentinel_crop_health_report.py` divides H3 mean scores by 100 and rounds them to four decimal places for the public JSON.

### ADR-006 — Configuration is typed, strict, and executable

Pydantic models are the authority. Unknown keys are errors. Cross-field rules are validated before any raster opens. No processor reads raw dictionaries, and no numeric algorithm constant is hidden in processor code.

Every supported YAML key must be consumed. Legacy inactive keys described in the onboarding document are not included in the v1 schema.

### ADR-007 — Typed objects are stage boundaries

Large NumPy arrays are passed through frozen, slotted dataclasses. Pydantic is used for configuration and JSON-facing metadata, not for array-bearing runtime objects. A layer receives a typed context/output and returns a new typed output. Layers do not search directories for one another's files.

### ADR-008 — Scientific policy for v1

The v1 defaults are:

- display excludes unusable, cloud, shadow, snow, and invalid numeric pixels; haze remains displayable;
- analytics excludes unusable, cloud, shadow, snow, haze, and invalid numeric pixels;
- the optional low-vegetation gate is disabled;
- quality weight is a diagnostic only and does not alter Layer B, C, or E calculations;
- `reject` stops after Layer A and produces an empty public list;
- `display_only` runs later analytical layers but every report and metadata object must retain that status;
- `analytics_ready` runs all layers normally.

These defaults deliberately correct the unsafe legacy default that allowed cloud/shadow/snow into analytic scoring. A `legacy_compatibility` switch must not be added. If historical reproduction becomes necessary, create a separate versioned config file and explicit algorithm version.

### ADR-009 — Missing optional quality inputs are visible, not silently equivalent

Missing optional bands are represented by typed absence (`None`), not fabricated zero arrays. Quality logic uses available sources, records `missing_optional_bands`, and sets `quality_provenance` to `degraded` when any configured optional quality input is missing.

If cloud probability is absent, confidence is `1.0` only for pixels that all available quality signals classify as clear, and the report records `confidence_source: categorical_fallback`. It must not claim that a cloud-probability measurement of zero was observed.

### ADR-010 — Deterministic results

For identical inputs, config, dependency lock, and algorithm version, numeric outputs must be stable within declared floating-point tolerances. Sort H3 identifiers lexicographically before CSV, GeoJSON, overlay, index-list, and production JSON serialization. JSON uses `allow_nan=False`.

### ADR-011 — File-system ownership and publication

The analytical pipeline writes to the exact development output directory supplied by its caller. The production wrapper instead creates a run-scoped staging directory under the requested output directory, runs the full pipeline there, writes a temporary public JSON, validates it, then atomically replaces only `output_dir/crophealthindex.json`.

The wrapper never recursively clears the caller's output directory. It deletes only its own staging directory in `finally`. Concurrent calls to the same output directory are unsupported and must fail through a cross-process filesystem lock; a `threading.RLock` alone is insufficient.

### ADR-012 — H3 aggregation uses physical overlap area

Pixel centers are insufficient at Sentinel resolution. Each analytic pixel contributes to every intersecting H3 cell according to intersection area calculated in `EPSG:6933`. Weighted mean and weighted standard deviation use the same overlap weights. Cells with no analytic contribution are not emitted.

Exact H3 work is bounded separately from raster-window memory. Version 1 retains H3 resolution 13 as the declared output contract, but it must reject a workload exceeding configured H3 analytic-pixel or candidate-cell limits. It must not silently lower resolution, use center-only assignment, or approximate overlap through rasterization.

### ADR-013 — Fail closed

Missing inputs, invalid geometry, unknown CRS, impossible configuration, no usable health index, numeric non-finiteness in public results, ambiguous output discovery, or resource-limit violations fail the run. A rejected scene is a valid analytical outcome, not an exception.

### 2.1 Intentional supersessions of inherited behavior

| Inherited behavior in onboarding guide | Required new behavior | Reason |
|---|---|---|
| Full reference-grid arrays are implied | Read only a bounded B8 crop window | Production memory safety |
| FeatureCollection silently uses first feature | Require exactly one feature | Prevent analyzing the wrong crop silently |
| Analytic defaults allow cloud/shadow/snow/haze | Exclude all four by default | Prevent contaminated health scores |
| Missing optional bands become zero arrays | Keep typed absence and coverage provenance | Zero is an observation, absence is not |
| Missing weak index yields `0.0` weak fraction | Yield JSON null | Unknown is not zero weakness |
| Fallback pixel data may use primary range | Use selected fallback index's own range | Dimensional/scoring correctness |
| H3 min/max/std use a differently weighted population | Use one overlap-weighted population for mean/std | Internally consistent cell statistics |
| H3 may silently fall back to center containment | Require overlap-containment API | Preserve boundary cells |
| Inactive H3 config keys are retained | Reject unknown/inactive keys | Configuration must be executable |
| Production cleanup can own/prune the entire report directory | Delete only run-scoped staging | Protect unrelated/caller data |

These are deliberate version-1 design choices. They are not tasks for implementers to reconsider.

---

## 3. Technology and Dependency Contract

### 3.1 Runtime

- Python: `>=3.11`
- Array math: NumPy
- Raster I/O/reprojection: Rasterio/GDAL
- Geometry: Shapely 2.x
- CRS transforms: PyProj
- H3: h3-py 4.x, isolated behind one adapter module
- Cross-process output lock: Portalocker
- Config: Pydantic 2.x and PyYAML
- Previews: Matplotlib using the non-interactive `Agg` backend
- Tests: Pytest

Do not use Pandas for small CSV outputs. Do not use OpenCV. Do not add a web framework.

### 3.2 Packaging work

Create or extend the repository's authoritative `pyproject.toml`. If this package is later moved into the worker repository, merge dependencies into the worker's existing lock rather than introducing a second environment.

Use these dependency groups:

```toml
[project]
requires-python = ">=3.11"
dependencies = [
  "numpy>=2,<3",
  "rasterio>=1.4,<2",
  "shapely>=2,<3",
  "pyproj>=3.7,<4",
  "h3>=4.1,<5",
  "portalocker>=2.10,<4",
  "pydantic>=2.10,<3",
  "PyYAML>=6,<7",
  "matplotlib>=3.9,<4",
]

[project.optional-dependencies]
test = ["pytest>=8,<10", "pytest-cov>=6,<8"]
```

CI must lock exact resolved versions. The ranges above are source compatibility limits, not a substitute for the lock file.

---

## 4. Required Package Layout

Create this structure exactly:

```text
crop_health_sentinel/
  __init__.py
  version.py
  errors.py
  pipeline.py
  sentinel_crop_health_report.py
  config/
    __init__.py
    schema.py
    loader.py
    default_config_sentinel_v1.yaml
  models/
    __init__.py
    types.py
  io/
    __init__.py
    band_registry.py
    geometry.py
    scene.py
    raster_artifacts.py
    structured_artifacts.py
  layer_a_quality/
    __init__.py
    processor.py
  layer_b_spectral/
    __init__.py
    processor.py
  layer_c_health/
    __init__.py
    processor.py
  layer_e_h3/
    __init__.py
    h3_adapter.py
    processor.py
  layer_d_report/
    __init__.py
    processor.py
  utils/
    __init__.py
    math_utils.py
    stats.py
    preview.py
    lock.py
    naming.py
  tests/
    conftest.py
    unit/
    integration/
    contract/
    regression/
```

The unusual execution order is fixed:

```text
load/configure -> crop/read/align -> A quality -> B spectral -> C health
                                              -> E H3 -> D final report
```

Layer D runs last because it summarizes Layer E, even though its output folder is `phase_04` and H3 writes to `phase_05` for compatibility.

### 4.1 Import-direction rule

Allowed dependency direction:

```text
config/models/utils <- io <- layers <- pipeline <- production wrapper
```

- `models`, `config`, and `utils` must not import processors or pipeline code.
- Layers may import shared types, utility functions, and artifact writers.
- A layer must not import another layer's processor.
- Only `pipeline.py` orchestrates layers.
- Only the production wrapper knows the public JSON schema and pruning/staging lifecycle.

---

## 5. Canonical Types and Function Signatures

Implement the following contracts before algorithm code. Field names are part of the internal API.

### 5.1 Enums

```python
class SceneStatus(str, Enum):
    REJECT = "reject"
    DISPLAY_ONLY = "display_only"
    ANALYTICS_READY = "analytics_ready"

class QualityProvenance(str, Enum):
    COMPLETE = "complete"
    DEGRADED = "degraded"

class BandKind(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BITMASK = "bitmask"
```

### 5.2 Metadata and raster contracts

```python
@dataclass(frozen=True, slots=True)
class SceneMetadata:
    farm: str
    field: str
    date: str
    scene_id: str
    run_id: str

@dataclass(frozen=True, slots=True)
class ReferenceGrid:
    crs: CRS
    transform: Affine
    width: int
    height: int
    profile: Mapping[str, Any]
    source_window: Window

@dataclass(frozen=True, slots=True)
class AlignedBands:
    green: NDArray[np.float32]
    red: NDArray[np.float32]
    red_edge: NDArray[np.float32]
    nir: NDArray[np.float32]
    scl: NDArray[np.uint8]
    cloud_probability: NDArray[np.float32] | None
    opaque_cloud: NDArray[np.uint8] | None
    cirrus: NDArray[np.uint8] | None
    snow_ice: NDArray[np.uint8] | None
    qa60: NDArray[np.uint16] | None
    aot: NDArray[np.float32] | None
    required_coverage_mask: NDArray[np.bool_]
    optional_coverage_masks: Mapping[str, NDArray[np.bool_]]
    missing_optional_bands: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class SceneContext:
    metadata: SceneMetadata
    raw_data_dir: Path
    output_dir: Path
    config: SentinelCropHealthConfig
    algorithm_version: str
    grid: ReferenceGrid
    crop_geometry_wgs84: BaseGeometry
    crop_geometry_grid_crs: BaseGeometry
    field_mask: NDArray[np.bool_]
    field_inner_mask: NDArray[np.bool_]
    bands: AlignedBands
```

Every array in a context or stage output must be C-contiguous and exactly `grid.height x grid.width`. Validate shape once in a shared `validate_array_shape` helper and in debug assertions at layer entry.

`required_coverage_mask` is the intersection of valid source coverage for B3/B4/B5/B8/SCL. `optional_coverage_masks` contains an entry only for each optional band that is present. Optional values may affect quality only where their own coverage mask is true.

### 5.3 Stage outputs

```python
@dataclass(frozen=True, slots=True)
class LayerAOutput:
    display_valid_mask: NDArray[np.bool_]
    analytic_valid_mask: NDArray[np.bool_]
    quality_weight: NDArray[np.float32]
    reason_codes: NDArray[np.uint8]
    scene_status: SceneStatus
    quality_provenance: QualityProvenance
    summary: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class SpectralOutput:
    reflectance: Mapping[str, NDArray[np.float32]]
    indices: Mapping[str, NDArray[np.float32]]
    summary: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class HealthOutput:
    field_health_score: float
    health_class: str
    weak_area_fraction: float
    pixel_score_source: str
    pixel_health_map: NDArray[np.float32]
    pixel_health_class_map: NDArray[np.uint8]
    summary: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class H3CellResult:
    h3_index: str
    mean_health: float
    std_health: float
    min_health: float
    max_health: float
    actual_pixel_count: int
    analytic_pixel_count: int
    effective_analytic_pixel_count: float
    low_support: bool

@dataclass(frozen=True, slots=True)
class LayerEOutput:
    cells: tuple[H3CellResult, ...]
    summary: Mapping[str, Any]
    overlay_path: Path

@dataclass(frozen=True, slots=True)
class FinalReportOutput:
    report: Mapping[str, Any]
    summary: Mapping[str, Any]
    report_path: Path

@dataclass(frozen=True, slots=True)
class PipelineResult:
    scene_status: SceneStatus
    layer_a: LayerAOutput
    spectral: SpectralOutput | None
    health: HealthOutput | None
    h3: LayerEOutput | None
    final_report: FinalReportOutput | None
    output_dir: Path
```

### 5.4 Public functions

```python
def run_crop_health_report(
    raw_data_dir: str | Path,
    crop_geometry: str | Mapping[str, Any] | BaseGeometry,
    output_dir: str | Path,
    scene_meta: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    config: SentinelCropHealthConfig | Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> PipelineResult: ...

def generate_sentinel_crop_health_report(
    raw_data_dir: str | Path,
    crop_geometry: str | Mapping[str, Any] | BaseGeometry,
    output_dir: str | Path,
    scene_meta: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> str: ...  # absolute crophealthindex.json path

def get_sentinel_crop_health_report_metadata(
    output_path: str | Path,
) -> dict[str, Any] | None: ...  # destructive one-time read
```

Passing both `config` and `config_path` is an error. The production wrapper intentionally accepts only `config_path`, preventing callers from injecting an unrecorded in-memory production configuration.

Scene metadata normalization is exact:

| Canonical field | Accepted keys in priority order | Required/default |
|---|---|---|
| `farm` | `farm`, `farm_no`, `farmId` | default `"farm"` |
| `field` | `field`, `crop_unique_no`, `cropUniqueNo`, `cropId` | default `"crop"` |
| `date` | `date`, `imageryDate` | default `"unknown-date"` |
| `scene_id` | `scene_id`, `sceneId` | default `"sentinel-scene"` |
| `run_id` | `run_id`, `runId` | default to normalized `scene_id` |

Use the first present, non-null alias, convert it to `str`, trim surrounding whitespace, and reject it if empty. Defaults preserve the onboarding compatibility contract while ensuring every run has stable identifiers.

---

## 6. Error Taxonomy

Create `errors.py`. Every expected failure derives from `CropHealthError` and exposes a stable `code` string.

| Exception | Code | Use |
|---|---|---|
| `ConfigurationError` | `CONFIG_INVALID` | YAML/schema/cross-field failure |
| `InputValidationError` | `INPUT_INVALID` | missing directory/file or wrong input type |
| `GeometryError` | `GEOMETRY_INVALID` | malformed, empty, invalid, or outside-scene geometry |
| `RasterMetadataError` | `RASTER_METADATA_INVALID` | missing CRS/transform/band or unreadable metadata |
| `RasterAlignmentError` | `RASTER_ALIGNMENT_FAILED` | reprojection/read failure |
| `ResourceLimitError` | `RESOURCE_LIMIT_EXCEEDED` | crop window exceeds configured pixel limit |
| `InsufficientAnalyticsError` | `NO_HEALTH_INDEX` | later layer cannot produce any configured score |
| `H3AggregationError` | `H3_AGGREGATION_FAILED` | candidate or intersection failure |
| `H3CapacityError` | `H3_CAPACITY_EXCEEDED` | exact H3 workload exceeds explicit limits |
| `ArtifactError` | `ARTIFACT_WRITE_FAILED` | durable output cannot be validated/written |
| `PublicContractError` | `PUBLIC_CONTRACT_INVALID` | final JSON invalid or ambiguous |
| `ConcurrentRunError` | `OUTPUT_DIR_BUSY` | same production output path already running |

Do not catch `Exception` inside layers. `pipeline.py` logs stage context and re-raises. The production wrapper cleans its own staging data in `finally` and preserves the original exception chain.

---

## 7. Configuration Schema

### 7.1 Pydantic rules

All models use:

```python
model_config = ConfigDict(extra="forbid", frozen=True)
```

Build nested models for `paths`, `io`, `quality`, `spectral`, `health`, `h3`, and `artifacts`. Use enums or literals instead of unconstrained strings.

### 7.2 Default YAML

Implement this shape. Comments may be added, but keys and defaults must match.

```yaml
algorithm_version: sentinel-crop-health/1.0.0

paths:
  green: B3.tif
  red: B4.tif
  red_edge: B5.tif
  nir: B8.tif
  scl: SCL.tif
  cloud_probability: MSK_CLDPRB.tif
  opaque_cloud: MSK_CLASSI_OPAQUE.tif
  cirrus: MSK_CLASSI_CIRRUS.tif
  snow_ice: MSK_CLASSI_SNOW_ICE.tif
  qa60: QA60.tif
  aot: AOT.tif

io:
  reflectance_scale: 10000.0
  crop_window_margin_pixels: 2
  max_crop_window_pixels: 25000000
  rasterization: all_touched
  edge_erosion_pixels: 0
  use_field_inner_mask: false
  output_compression: deflate
  output_tile_size: 512

quality:
  scl_cloud_classes: [8, 9, 10]
  scl_shadow_classes: [3]
  scl_snow_classes: [11]
  scl_unusable_classes: [0, 1]
  cloud_probability_threshold: 70.0
  qa60_cloud_bits: [10, 11]
  aot_scale: 10000.0
  aot_haze_threshold: 0.60
  display_exclusions: [unusable, cloud, shadow, snow, invalid_numeric]
  analytic_exclusions: [unusable, cloud, shadow, haze, snow, invalid_numeric]
  vegetation_gate_enabled: false
  vegetation_gate_ndvi_threshold: 0.05
  minimum_analytic_pixels: 5
  display_only_minimum_fraction: 0.15
  analytics_ready_minimum_fraction: 0.40

spectral:
  denominator_epsilon: 0.000001
  enabled_indices: [ndvi, ndre, gndvi, evi2]
  summary_statistic: mean
  percentiles: [5, 10, 25, 50, 75, 90, 95]

health:
  index_ranges:
    ndvi: {low: 0.15, high: 0.80}
    ndre: {low: 0.05, high: 0.45}
    gndvi: {low: 0.08, high: 0.70}
    evi2: {low: 0.10, high: 0.90}
  field_weights:
    ndvi: 0.20
    ndre: 0.40
    gndvi: 0.25
    evi2: 0.15
  class_thresholds:
    very_good: 85.0
    good: 70.0
    moderate: 55.0
    weak: 40.0
  weak_area_index: ndre
  weak_area_threshold: 0.18
  pixel_primary_index: ndre
  pixel_fallback_index: ndvi

h3:
  resolution: 13
  metric_crs: EPSG:6933
  minimum_effective_support: 0.01
  max_analytic_pixels: 50000
  max_candidate_cells: 125000
  batch_size: 5000
  candidate_estimate_safety_factor: 1.50
  capacity_policy: fail

artifacts:
  write_previews: true
  preview_dpi: 120
  retain_development_intermediates: true
```

### 7.3 Required cross-field validators

Reject configuration unless all conditions hold:

1. algorithm version exactly matches the running package version;
2. required band paths are non-empty basenames without `..`;
3. scales and epsilon are positive;
4. `max_crop_window_pixels >= 1`;
5. tile size is one of `256`, `512`, or `1024`;
6. probability/fraction thresholds are in their correct ranges;
7. `display_only_minimum_fraction < analytics_ready_minimum_fraction`;
8. every enabled index has a health range when it can feed a score;
9. each range has `low < high`;
10. weights are non-negative and at least one enabled index has positive weight;
11. class thresholds are strictly descending and in `[0, 100]`;
12. weak and pixel source names are supported indices;
13. H3 resolution is in `[0, 15]`;
14. H3 batch size is in `[1, 50000]`, and both H3 capacity limits are positive;
15. the candidate safety factor is at least `1.0`;
16. `capacity_policy` is exactly `fail` in v1;
17. metric CRS parses and is projected; and
18. artifact tile size is compatible with GeoTIFF tiling.

`load_config` deep-merges no arbitrary partial YAML. A supplied config file must be complete. This avoids implicit default drift. Programmatic tests may instantiate a complete Pydantic model with `model_copy(update=...)`.

---

## 8. Exact Analytical Contracts

### 8.1 Band values and nodata

- Read continuous source data as `float32`.
- Convert masked/nodata source samples to `NaN` after reprojection.
- Keep reflectance digital numbers unscaled in `AlignedBands`.
- Layer B divides required reflectance bands by `reflectance_scale` once.
- Read categorical arrays into their declared integer type.
- Treat pixels outside a source raster's coverage as invalid; never as class/value zero unless zero is the source's real value.

### 8.2 Geometry and field masks

Normalization sequence:

1. parse JSON string/mapping/Shapely input;
2. unwrap `Feature`;
3. require exactly one feature for `FeatureCollection`;
4. accept only Polygon/MultiPolygon;
5. close open rings;
6. construct Shapely geometry;
7. reject empty, non-finite, or invalid geometry with `explain_validity` detail;
8. normalize coordinate ordering and retain EPSG:4326;
9. transform to reference CRS;
10. verify non-zero intersection with B8 bounds;
11. create crop reference window; and
12. rasterize on the cropped reference grid.

`field_mask` follows configured `all_touched`. If edge erosion is zero, `field_inner_mask` equals a copy of `field_mask`. If erosion is enabled, use an 8-connected binary erosion for exactly N iterations. Since SciPy is not a dependency, implement erosion with padded boolean shifts in `io/geometry.py` and test it independently.

Selected field pixels are `field_inner_mask` when `use_field_inner_mask` is true, otherwise `field_mask`.

### 8.3 Quality primitives

Calculate these boolean arrays:

```text
scl_cloud   = SCL in configured cloud classes
cloud       = scl_cloud OR opaque OR cirrus OR cloud_probability>=threshold
              OR any configured QA60 bit is set
shadow      = SCL in shadow classes
snow        = SCL in snow classes OR snow_ice flag
unusable    = SCL in unusable classes OR required source coverage invalid
haze        = finite(AOT) AND (AOT / aot_scale)>=haze_threshold
invalid_num = any required reflectance is non-finite OR quality-NDVI is invalid
clear       = NOT(unusable OR cloud OR shadow OR haze OR snow OR invalid_num)
```

The quality NDVI uses scaled or unscaled Red/NIR equivalently, but use scaled reflectance for one consistent code path:

```text
NDVI = (NIR - Red) / (NIR + Red), only if abs(denominator)>epsilon
```

Optional source handling:

- absent flags contribute `False`, but absence is recorded;
- absent AOT contributes no haze evidence and is recorded;
- absent cloud probability selects categorical confidence fallback;
- source nodata never becomes a clear observation; optional flags are evaluated only under their individual coverage masks.

When cloud probability exists only for part of the crop window, calculate confidence per pixel: use probability-derived confidence where that band's coverage mask is true, and categorical clear/not-clear confidence elsewhere. Apply the cloud probability threshold only where both coverage and a finite probability value are present.

### 8.4 Layer A masks and reason precedence

Build display/analytic masks by starting with the selected field mask and subtracting configured exclusions. Apply vegetation gating to analytics only when enabled.

Reason-code precedence is fixed from least to most dominant:

| Code | Meaning |
|---:|---|
| 0 | no exclusion |
| 1 | outside selected field |
| 2 | unusable |
| 3 | cloud |
| 4 | shadow |
| 5 | haze |
| 6 | snow |
| 7 | invalid numeric input |
| 8 | low vegetation |

Implement precedence by assignments in ascending code order. Later reasons overwrite earlier reasons. Document this in the summary as `reason_code_precedence`.

Quality weight:

```text
if cloud probability present:
    confidence = clip(1 - cloud_probability/100, 0, 1)
else:
    confidence = clear.astype(float32)

quality_weight = confidence where analytic_valid else 0
```

Status:

```text
analytic_fraction = analytic_count / selected_field_count

reject          if analytic_count < minimum_analytic_pixels
                or analytic_fraction < display_only_minimum_fraction
analytics_ready if analytic_fraction >= analytics_ready_minimum_fraction
display_only    otherwise
```

Division by zero is impossible because geometry rasterization must have produced selected field pixels; still guard and raise `GeometryError`.

### 8.5 Layer B formulas

Use `safe_divide(numerator, denominator, valid_mask, epsilon)`, returning a `float32` array filled with `NaN` outside valid calculations.

```text
NDVI  = (NIR - Red) / (NIR + Red)
NDRE  = (NIR - RedEdge) / (NIR + RedEdge)
GNDVI = (NIR - Green) / (NIR + Green)
EVI2  = 2.5 * (NIR - Red) / (NIR + 2.4*Red + 1.0)
```

Do not clip spectral indices. Invalid or unusual reflectance must remain visible in diagnostics; only health normalization clips to its configured range.

For each reflectance/index array, summarize finite analytic pixels with count, mean, population standard deviation (`ddof=0`), min, max, CV, and configured percentiles. If count is zero, use `count: 0` and JSON `null` for all numeric statistics. Never emit NaN/Infinity in JSON.

### 8.6 Layer C scoring

For index `i`:

```text
score_i = clip((summary_value_i - low_i) / (high_i - low_i), 0, 1) * 100
field_score = sum(score_i * configured_weight_i) / sum(weights actually used)
```

Use only enabled, finite index summaries with positive configured weight. Store the used/skipped indices and normalized weights in `summary.score_components`. Fail if none remain.

Class boundary order is `Very Good`, `Good`, `Moderate`, `Weak`, `Poor`, with lower-bound-inclusive comparisons.

Weak fraction denominator is the number of finite values for the configured weak index inside the analytic mask, not the count of all analytic pixels. If the weak index is disabled/unavailable, emit JSON `null`; do not fabricate `0.0`.

For the pixel map, select primary if available, otherwise fallback. Normalize using the range belonging to the **selected source index**. This fixes the known fallback-range defect. Pixels outside finite analytic source values are `NaN`; class code is zero. Valid class codes are one through five in Poor-to-Very-Good order.

### 8.7 Layer E H3 aggregation

`h3_adapter.py` is the only file allowed to import `h3`. It provides:

```python
def polygon_to_overlapping_cells(geometry_wgs84, resolution) -> tuple[str, ...]: ...
def cell_boundary_wgs84(index: str) -> Polygon: ...
def latlng_to_cell(lat: float, lng: float, resolution: int) -> str: ...
def average_cell_area_m2(resolution: int) -> float: ...
```

Use the h3-py 4.x overlap-containment API. If that API is missing, raise a dependency compatibility error; do not silently fall back to center containment because it can omit boundary cells.

Capacity preflight, before candidate geometry allocation:

```text
estimated_candidate_cells = ceil(
    metric_crop_area / average_h3_cell_area_m2(resolution)
    * candidate_estimate_safety_factor
)
```

Fail with `H3_CAPACITY_EXCEEDED` if either the finite analytic-pixel count exceeds `max_analytic_pixels` or this estimate exceeds `max_candidate_cells`. After calling the overlap API, enforce the actual candidate-cell limit as well. This policy is intentionally fail-closed: changing resolution is a versioned product/configuration decision, not an automatic fallback.

Exact, bounded aggregation sequence:

1. obtain all H3 cells overlapping the crop geometry and sort their identifiers;
2. transform cell polygons and crop geometry to metric CRS, intersect every cell polygon with the crop polygon, discard empty results, and build one Shapely 2.x `STRtree` plus index-to-cell lookup;
3. collect row-major flat indexes of finite analytic pixel-health values and partition them into batches of exactly `h3.batch_size` except the last;
4. for one batch, derive all four pixel corners from the reference affine as NumPy coordinate arrays, transform the coordinates to metric CRS in one PyProj call, and construct a Shapely array of closed quadrilaterals;
5. call `STRtree.query(pixel_polygons, predicate="intersects")` once for the full batch; use returned pixel/cell index pairs as the only candidate pairs;
6. vectorize the exact metric intersections and areas for those pairs; discard zero-area results;
7. calculate every pixel's eligible-area sum with `numpy.bincount`, divide its pair areas by that sum, and assert each contributed pixel's shares sum to one within `1e-6`;
8. accumulate weighted count, first moment, second moment, minimum, and maximum into numeric arrays indexed by candidate cell; do not retain the batch's pixel geometries after accumulation;
9. calculate field/analytic center-count diagnostics separately in bounded batches; these diagnostics must not participate in the mean; and
10. finalize supported cells and sort them by H3 index.

Pixel-coordinate geometry may be used only as a broad-phase optimization in a future version. It must never replace the exact EPSG:6933 intersection and area calculation.

For weights `w` and health values `x`:

```text
effective_count = sum(w)
mean = sum(w*x) / sum(w)
variance = max(sum(w*x*x)/sum(w) - mean*mean, 0)
std = sqrt(variance)
```

`min` and `max` are extrema of pixels with positive contribution. `actual_pixel_count` counts selected-field pixel centers in a cell. `analytic_pixel_count` counts analytic pixel centers in a cell. These center counts are diagnostics only.

Write these artifacts in lexicographic H3 order:

- H3 health raster and preview;
- H3 cells CSV;
- WGS84 H3 cell GeoJSON;
- pixel-coordinate overlay JSON;
- all emitted H3 indexes JSON; and
- Layer E summary JSON/CSV.

Overlay features must include `h3Index`, `healthScore` (0..100), and pixel coordinates obtained by transforming each H3 boundary from EPSG:4326 to the raster CRS and then applying the inverse affine transform. This overlay is diagnostic; the production builder consumes its semantic fields, not the drawn coordinates.

H3 CSV columns are fixed in this order:

```text
h3_index,mean_health,std_health,min_health,max_health,
actual_pixel_count,analytic_pixel_count,effective_analytic_pixel_count,
low_support,score_source
```

`score_source` is always `analytic` in v1. GeoJSON properties use the same names in camel case and include `healthClass`. Overlay JSON is an object with `schemaVersion`, `algorithmVersion`, `rasterWidth`, `rasterHeight`, and `cells`; each cell contains `h3Index`, `healthScore`, `healthClass`, `lowSupport`, and `pixelBoundary`, where `pixelBoundary` is a closed `[[column, row], ...]` ring of floats.

The H3 health raster is defined by center lookup: for every selected-field raster pixel, find the H3 cell containing its center and write that emitted cell's mean health; write nodata when the center cell was not emitted. This raster is visualization-only and must not feed the CSV, overlay, report, or public JSON.

### 8.8 Layer D report

Assemble a schema-versioned object:

```json
{
  "schemaVersion": "1.0",
  "algorithmVersion": "sentinel-crop-health/1.0.0",
  "scene": {},
  "inputs": {
    "requiredBands": [],
    "optionalBandsPresent": [],
    "optionalBandsMissing": [],
    "cropGeometryType": "Polygon"
  },
  "quality": {},
  "spectral": {},
  "health": {},
  "spatial": {},
  "limitations": []
}
```

Limitations must explicitly state that this is current-scene configured spectral scoring and not a causal agronomic diagnosis, time trend, weather-adjusted result, crop-stage model, or ML prediction.

---

## 9. Artifact Rules

### 9.1 Naming

Sanitize scene identifiers to ASCII `[A-Za-z0-9._-]`, replacing other runs with `_`. Reject an empty result. File names follow:

```text
<safe_scene_id>_<artifact_name>.<extension>
```

No timestamp appears in deterministic analytical file names. Run identity belongs in metadata.

### 9.1.1 Required development artifact manifest

For a non-rejected run, write exactly these artifact families (a `.png` sibling is required for each listed raster when previews are enabled):

```text
crop_polygon.json
phase_01/
  <scene>_field_mask.tif
  <scene>_field_inner_mask.tif
  <scene>_display_valid_mask.tif
  <scene>_analytic_valid_mask.tif
  <scene>_quality_weight.tif
  <scene>_reason_code_mask.tif
  <scene>_layer_a_quality_summary.json
  <scene>_layer_a_quality_summary.csv
phase_02/
  <scene>_reflectance_green.tif
  <scene>_reflectance_red.tif
  <scene>_reflectance_red_edge.tif
  <scene>_reflectance_nir.tif
  <scene>_<enabled_index>.tif
  <scene>_layer_b_spectral_summary.json
  <scene>_layer_b_spectral_summary.csv
phase_03/
  <scene>_pixel_health_map.tif
  <scene>_pixel_health_class_map.tif
  <scene>_layer_c_health_summary.json
  <scene>_layer_c_health_summary.csv
phase_04/
  <scene>_final_report.json
  <scene>_final_report.csv
  <scene>_layer_d_report_summary.json
  <scene>_layer_d_report_summary.csv
phase_05/
  <scene>_h3_health_map.tif
  <scene>_h3_cells.csv
  <scene>_h3_cells.geojson
  <scene>_h3_cells_overlay.json
  <scene>_all_polygon_h3_indexes.json
  <scene>_layer_e_h3_summary.json
  <scene>_layer_e_h3_summary.csv
```

A rejected run writes only `crop_polygon.json` and phase-01 artifacts. A disabled index produces no raster/preview and no fabricated summary entry. Do not emit convenience aliases or duplicate root-level reports.

### 9.2 Raster writing

`write_single_band_raster` must:

- verify shape against the reference grid;
- write to a sibling `.partial.tif`;
- use tiled GeoTIFF, configured block size and compression;
- use float32 nodata `-9999.0` for floating outputs and replace NaN only while writing;
- use zero nodata for class/mask outputs when zero is background;
- store CRS and affine exactly from the cropped reference grid;
- reopen and validate width, height, count, CRS, transform, dtype, and nodata; and
- publish with `os.replace`.

### 9.3 JSON and CSV

- UTF-8, newline at EOF.
- JSON indent 2, sorted keys where object ordering is not semantically specified, `allow_nan=False`.
- CSV uses `newline=""` and an explicit stable column list.
- Write `.partial`, flush, `fsync`, then `os.replace`.
- Convert NumPy scalars through one `to_json_value` helper.

### 9.4 Preview generation

Previews are diagnostics and must never affect numeric results. Do not import `matplotlib.pyplot`. Construct each image with `matplotlib.figure.Figure` and `FigureCanvasAgg`, with no runtime mutation of global `rcParams`. Guard complete figure creation, rendering, and disposal with one process-local preview lock because Matplotlib artists are not thread-safe. Use a copy of the source array, mask non-finite values, select display limits from p2/p98 of finite field pixels, include a colorbar/title/nodata legend, dispose of references in `finally`, and never mutate the passed array.

---

## 10. Implementation Phases

Each phase should be one reviewable pull request unless repository policy requires smaller PRs. Do not start a later analytical phase until the prior phase's exit gate passes.

## Phase 0 — Scaffold, dependencies, and test harness

### Files

- package `__init__.py` files
- `version.py`
- `errors.py`
- `pyproject.toml`
- `tests/conftest.py`

### Work

1. Create the package tree.
2. Export only the three public functions from the package root.
3. Set `__version__ = "1.0.0"` and `ALGORITHM_VERSION`.
4. Add dependency groups and Pytest configuration.
5. Build deterministic synthetic GeoTIFF helpers for 10 m and 20 m sources.
6. Add fixtures for Polygon, polygon-with-hole, MultiPolygon, cloud/shadow/SCL, and optional-band absence.
7. Ensure test files use temporary directories only.

### Exit gate

- Package imports in a clean environment.
- Empty test harness runs.
- No GUI backend is loaded.

## Phase 1 — Configuration and shared types

### Files

- `config/schema.py`
- `config/loader.py`
- default YAML
- `models/types.py`

### Work

1. Implement every model and validator from Sections 5 and 7.
2. Load YAML with `yaml.safe_load`.
3. Resolve the packaged default via `importlib.resources`; never use current working directory.
4. Validate supplied paths without resolving band files yet.
5. Normalize scene metadata aliases once in a helper; reject empty values after string conversion.

### Tests

- default config loads;
- unknown key fails;
- incomplete custom file fails;
- each cross-field invalid state fails with a field-specific message;
- config object is immutable;
- metadata aliases normalize correctly;
- run ID defaults to a caller-provided scene ID only in direct pipeline tests, never to random state.

### Exit gate

Every algorithm number used later can be referenced from a typed config field.

## Phase 2 — Geometry, band registry, crop window, and aligned reading

### Files

- `io/band_registry.py`
- `io/geometry.py`
- `io/scene.py`

### Required functions

```python
resolve_band_paths(raw_data_dir, config) -> ResolvedBandPaths
normalize_crop_geometry(value) -> BaseGeometry
create_reference_grid(nir_path, geometry_wgs84, config) -> tuple[ReferenceGrid, BaseGeometry]
read_aligned_bands(paths, grid, config) -> AlignedBands
rasterize_field_masks(geometry_grid_crs, grid, config) -> tuple[bool_array, bool_array]
build_scene_context(...) -> SceneContext
```

### Work details

1. Resolve paths and require files for B3/B4/B5/B8/SCL.
2. Open each raster long enough to validate a single band, CRS, affine, dimensions, readable dtype, and bounds.
3. Transform geometry and derive a clipped integer B8 window with the configured margin.
4. Enforce `width*height <= max_crop_window_pixels` before allocating.
5. Use `WarpedVRT` or `rasterio.warp.reproject` to read every band directly into the crop grid.
6. Carry coverage masks so reprojection fill values cannot masquerade as valid zero data.
7. Validate aligned shapes and finite coverage.
8. Rasterize and verify at least one selected field pixel.

### Tests

- same-grid reads are unchanged;
- B5/SCL 20 m inputs align to B8 10 m grid;
- bilinear continuous output differs correctly from nearest categorical output;
- output transform equals the crop window transform;
- outside-scene geometry fails;
- geometry touching the scene edge clips safely;
- unknown CRS fails;
- missing required file fails before allocation;
- every missing optional file is represented by `None` and its name is recorded;
- FeatureCollection cardinality other than one fails;
- hole and MultiPolygon rasterization match expected masks;
- oversized crop window fails before reading arrays.

### Exit gate

All processors can receive one bounded-memory, aligned `SceneContext` without opening source rasters themselves.

## Phase 3 — Math, statistics, and durable artifact primitives

### Files

- `utils/math_utils.py`
- `utils/stats.py`
- `utils/preview.py`
- `utils/lock.py`
- `utils/naming.py`
- `io/raster_artifacts.py`
- `io/structured_artifacts.py`

### Work

1. Implement safe division, range normalization, finite-stat summaries, and class assignment.
2. Implement atomic JSON/CSV/raster writers.
3. Implement object-oriented, locked Agg preview generation.
4. Implement the Portalocker wrapper; it must expose a context manager that acquires an exclusive non-blocking lock for one normalized output directory and converts lock contention into `ConcurrentRunError`.
5. Ensure artifacts use only values passed into them; they must not re-read analytical artifacts.

### Tests

- zero/near-zero denominator behavior;
- constant-array CV and percentile behavior;
- no-finite-values summary yields JSON-safe nulls;
- exact class boundaries;
- unsafe filename characters;
- raster round trip preserves grid and nodata;
- partial file is not published on forced write failure;
- JSON rejects NaN/Infinity;
- preview does not mutate its array, imports no `pyplot`, and serializes simultaneous render requests;
- a child Python process holding the publication lock prevents the parent process from acquiring it, then permits acquisition after the child exits.

### Exit gate

Later layers contain no duplicate serialization, statistics, or numeric-safety code.

## Phase 4 — Layer A quality and trust

### File

- `layer_a_quality/processor.py`

### Required function

```python
def process_quality(context: SceneContext, logger: Logger) -> LayerAOutput: ...
```

### Work

1. Build each primitive mask separately.
2. Compute quality NDVI with the shared safe divide.
3. Apply display and analytic exclusion sets.
4. Apply optional vegetation gate.
5. Build confidence and quality weight with explicit provenance.
6. Assign reason codes in documented precedence.
7. calculate counts/fractions/status.
8. Write phase-01 rasters, previews, summary JSON and flattened summary CSV.
9. Log counts, fractions, missing quality sources, and status.

### Tests

- each SCL class and optional flag in isolation;
- QA60 bit 10 and 11 masks;
- threshold equality for cloud probability and AOT;
- overlapping reason precedence;
- outside-field behavior;
- display/analytic policy difference for haze;
- missing cloud probability confidence fallback;
- exact reject/display-only/ready boundaries;
- minimum pixel boundary;
- no selected pixels fails;
- all output arrays have correct dtype and shape.

### Exit gate

Quality decisions are independently explainable from summary counts and reason-code raster.

## Phase 5 — Layer B spectral features

### File

- `layer_b_spectral/processor.py`

### Required function

```python
def process_spectral(context: SceneContext, quality: LayerAOutput, logger: Logger) -> SpectralOutput: ...
```

### Work

1. Scale reflectance once into float32 arrays.
2. Set pixels outside analytic mask to NaN in exposed spectral outputs.
3. Compute only enabled indices.
4. Summarize finite analytic samples.
5. Write phase-02 rasters, previews, JSON, and CSV.

### Tests

- hand-calculated 3x3 formulas;
- scale correctness;
- denominator zero and non-finite inputs;
- disabled index is absent from arrays, artifacts, and summary;
- analytic mask is honored;
- formula outputs are not clipped;
- statistics match NumPy expected values.

### Exit gate

Every spectral value used by health scoring is reproducible from documented inputs/formulas.

## Phase 6 — Layer C health interpretation

### File

- `layer_c_health/processor.py`

### Required function

```python
def process_health(context, quality, spectral, logger) -> HealthOutput: ...
```

### Work

1. Extract configured summary statistic.
2. Calculate per-index component scores.
3. Re-normalize weights actually used.
4. Calculate field score/class.
5. Calculate weak fraction or null.
6. Select pixel primary/fallback and use the selected index's range.
7. Generate pixel score/class maps.
8. Write phase-03 artifacts.

### Tests

- exact low/high and clipping boundaries;
- four-index weighted field score;
- disabled/missing index weight normalization;
- no usable weighted index fails;
- exact health class thresholds;
- weak denominator uses finite weak-index pixels;
- unavailable weak index returns null;
- primary selection;
- fallback selection uses fallback range;
- class map background zero and valid codes 1..5.

### Exit gate

The known primary-range/fallback-data defect cannot recur without a failing test.

## Phase 7 — Layer E H3 spatial aggregation

### Files

- `layer_e_h3/h3_adapter.py`
- `layer_e_h3/processor.py`

### Required function

```python
def preflight_h3_capacity(context, quality) -> H3CapacityEstimate: ...
def process_h3(context, quality, health, logger) -> LayerEOutput: ...
```

### Work

1. Implement and contract-test the H3 v4 adapter.
2. Implement metric-area candidate preflight and both hard capacity limits before constructing candidate geometry.
3. Generate overlap candidates only after preflight, then enforce the actual candidate count.
4. Build projected, crop-clipped H3 geometries and STRtree.
5. Implement vectorized, bounded-batch exact overlap accumulation and weighted moments; no per-pixel Python geometry loop is permitted.
6. Add center-count diagnostics in bounded batches.
7. Emit only positively supported cells.
8. Write phase-05 artifacts and sorted overlay.
9. Log estimated/actual candidate counts, batch size, analytic count, emitted count, total effective support, low-support count, wall time, and peak RSS.

### Tests

- known lat/lng maps to known H3 index at fixed resolution;
- polygon crossing cell boundary returns all overlaps;
- capacity estimate, analytic-pixel limit, estimated-candidate limit, and actual-candidate limit each fail with `H3_CAPACITY_EXCEEDED`;
- one pixel wholly inside one cell;
- one pixel split across two or more cells;
- overlap shares sum to one within `1e-6` per contributing pixel;
- weighted mean/std hand calculation;
- crop clipping excludes outside-polygon pixel area;
- affine rotation/shear corner construction;
- MultiPolygon and polygon hole;
- low-support tag at exact threshold;
- deterministic cell and artifact order;
- empty analytic input emits empty result safely;
- unavailable overlap API fails clearly rather than using center fallback.

### Performance gate

Benchmark at 10,000, 25,000, and 50,000 analytic pixels with representative field geometry and H3 resolution 13. Record wall time, peak RSS, candidate-cell count, emitted-cell count, and effective pixel-to-cell pair count. The 50,000-pixel run must complete within the worker's approved SLA and memory budget before the default capacity is released. Optimization must preserve exact numerical contracts. No nested scan over every cell for every pixel and no per-pixel Python Shapely loop is permitted; batched STRtree querying is mandatory.

### Exit gate

Boundary pixels contribute by physical overlap, and reference fixtures prove conservation of per-pixel share.

## Phase 8 — Layer D final report

### File

- `layer_d_report/processor.py`

### Required function

```python
def process_final_report(context, quality, spectral, health, h3, logger) -> FinalReportOutput: ...
```

### Work

1. Assemble the schema from in-memory outputs.
2. Add source/config/algorithm provenance.
3. Add scientific limitations.
4. Write full JSON, flattened CSV, compact summary JSON, and compact CSV to phase 04.

### Tests

- schema keys and versions;
- missing optional band provenance;
- display-only status remains visible;
- no NaN/Infinity;
- paths are serialized as strings;
- report values match typed layer outputs, not re-read files.

### Exit gate

A reviewer can trace public cell scores back through health, spectral, and quality summaries.

## Phase 9 — Pipeline orchestration and rejected-scene state

### File

- `pipeline.py`

### Sequence

```python
cfg = load_and_validate_config(...)
meta = normalize_scene_metadata(...)
context = build_scene_context(...)
quality = process_quality(context)
if quality.scene_status is REJECT:
    return PipelineResult(quality only)
preflight_h3_capacity(context, quality)
spectral = process_spectral(context, quality)
health = process_health(context, quality, spectral)
h3 = process_h3(context, quality, health)
report = process_final_report(context, quality, spectral, health, h3)
return PipelineResult(all outputs)
```

### Rules

- Create phase directories lazily, immediately before their layer writes.
- Write normalized crop geometry to `crop_polygon.json`.
- Add stage name and scene/run identifiers to logs.
- Do not convert expected exceptions to booleans or empty results.
- Reject is the only early successful return.
- Display-only continues all phases.

### Tests

- full clear-scene integration;
- rejected scene creates only crop geometry and phase 01;
- display-only creates all phases and retains status;
- each injected stage error prevents later phase directories;
- config and source paths are independent of current working directory;
- repeated run with same inputs produces equivalent numeric summaries.

### Exit gate

Direct development runs are complete, typed, inspectable, and deterministic.

## Phase 10 — Production adapter and public contract

### File

- `sentinel_crop_health_report.py`

### Production algorithm

1. Resolve all incoming paths to absolute paths.
2. Acquire a non-blocking, cross-process exclusive lock on `output_dir/.crophealth.publish.lock` through `utils/lock.py` and Portalocker. If acquisition times out immediately, raise `ConcurrentRunError(OUTPUT_DIR_BUSY)`.
3. Create `output_dir/.crop-health-staging/<run_id>-<uuid4>`.
4. Run the analytical pipeline in that staging directory.
5. For `reject`, build `[]`.
6. Otherwise consume `result.h3.cells` directly; do not glob for overlay files.
7. Map sorted cells to `{h3Index, healthScore}` with `round(mean_health/100, 4)`.
8. Validate exact keys, non-empty string index, unique indexes, finite numeric score, and `0 <= score <= 1`.
9. Atomically write/replace `output_dir/crophealthindex.json`.
10. Register compatibility metadata keyed by the resolved final path.
11. Delete only this invocation's staging directory in `finally`.
12. Release and close the lock handle in `finally`; OS-backed advisory locks release if a process terminates unexpectedly, so no stale-directory recovery mechanism is required.
13. Return `str(final_path.resolve())`.

Compatibility metadata:

```json
{
  "averageNdvi": 0.61,
  "fieldHealthScore": 78.4,
  "healthClass": "Good",
  "sceneStatus": "analytics_ready",
  "algorithmVersion": "sentinel-crop-health/1.0.0"
}
```

For rejected scenes, unavailable fields are JSON null. Protect the registry with `threading.RLock`, key by normalized absolute path, cap it at 1024 entries with oldest-entry eviction, and remove an entry on `get`. This is compatibility plumbing only; new worker integration should consume a richer runner result in the future.

### Contract tests

- return is an absolute existing file path;
- successful directory retains the public JSON plus any pre-existing unrelated caller files;
- diagnostic staging is removed after success and failure;
- rejected scene produces exactly `[]`;
- sorted output and four-decimal rounding;
- duplicate/non-finite/out-of-range cell fails publication;
- an existing public JSON remains untouched if the new run fails before publication;
- same-directory concurrent invocation fails with `OUTPUT_DIR_BUSY`;
- metadata is returned once then removed;
- registry cap and thread-safety behavior.

### Exit gate

Production callers see one stable public artifact and cannot observe a partially written replacement.

## Phase 11 — Worker integration (worker repository)

This phase cannot be completed solely in the present repository. The worker team must perform the following exact changes in its own codebase:

1. Register `cropHealth` as a supported Sentinel report type.
2. Map the existing Sentinel raw-data directory and crop EPSG:4326 geometry into the wrapper signature.
3. Pass normalized farm, crop, imagery date, scene ID, and worker run ID.
4. Keep the existing per-crop cloud gate upstream; record cloud-gate skip separately from Layer A rejection.
5. Use a unique output directory per farm/crop/date/report run.
6. Upload `crophealthindex.json`; exclude `raw_data`, staging, phase artifacts, and source rasters.
7. Retrieve compatibility metadata immediately after wrapper success.
8. Persist `averageNdvi`, field score, health class, scene status, algorithm version, and public object path where backend schema permits.
9. Mark an empty rejected report as processed-with-rejection, not as a runtime error.
10. Ensure one crop failure does not delete or corrupt another crop's workspace/result.
11. Implement force-retry behavior with a new unique run workspace.
12. Do not mark the backend record available until upload and backend save both succeed.

### Integration tests

- enabled/disabled and already-processed paths;
- cloud-gate skip versus Layer A reject;
- crop geometry and scene metadata mapping;
- success, rejection, pipeline failure, upload failure, backend-save failure;
- force one crop and force all crops;
- exact S3 key construction;
- raw/intermediate exclusion;
- two crops from one farm scene do not share outputs or metadata.

### Exit gate

The complete worker transaction is idempotent at the business-record level and reports partial failures accurately.

## Phase 12 — Scientific regression and release

Create a small approved fixture catalog with:

- clear healthy vegetation;
- high cloud;
- shadow and snow flags;
- haze with AOT;
- bare/low vegetation;
- narrow field;
- field crossing multiple H3 boundaries;
- Polygon with hole;
- MultiPolygon;
- missing optional quality sources; and
- mixed 10 m/20 m grid alignment.

For each fixture, version an expected manifest containing:

- selected/display/analytic pixel counts and fractions;
- reason-code histogram;
- spectral means and percentiles;
- field score/class and weak fraction;
- pixel source;
- emitted H3 count;
- selected cell means/support; and
- public JSON digest after canonical serialization.

Use tolerances, not bitwise raster equality:

- mask/count/class outputs: exact;
- index/scoring scalars: absolute tolerance `1e-6` unless reprojection is involved;
- reprojection-dependent summaries: approved tolerance `1e-4`;
- H3 weighted means: absolute tolerance `1e-5`;
- public four-decimal scores: exact.

Any intentional expected-manifest update must include a short scientific change note and before/after metrics. Never update golden outputs only because a test failed.

---

## 11. Testing Matrix by Boundary

| Boundary | Producer | Consumer | Proof required |
|---|---|---|---|
| Downloader -> package | worker | scene I/O | required names, readable rasters, CRS, overlap |
| Geometry -> crop grid | geometry I/O | context | type, validity, CRS transform, non-empty mask |
| Aligned bands -> Layer A | scene I/O | quality | identical shape/grid, nodata preserved |
| Layer A -> Layer B | quality | spectral | analytic mask and status semantics |
| Layer B -> Layer C | spectral | health | enabled names, finite summaries, range selection |
| Layer C -> Layer E | health | H3 | 0..100 pixel units, NaN background |
| Layer E -> wrapper | H3 | public serializer | unique sorted IDs, finite 0..100 means |
| Wrapper -> worker | adapter | worker | absolute path, 0..1 schema, metadata one-time read |
| Worker -> backend/S3 | worker | external systems | path, upload, status, retry idempotency |

---

## 12. Resource and Operational Requirements

### 12.1 Memory

With the 25-million-pixel crop-window limit, simultaneous float32 arrays can still consume substantial memory. Processors must:

- avoid `float64` promotion by declaring scalar/array dtypes;
- reuse temporary arrays only within a function where ownership is clear;
- never copy all bands merely to apply a mask;
- write artifacts sequentially;
- close datasets and figures deterministically; and
- release local temporaries before H3 processing where practical.

The raster-window limit does not authorize a similarly sized H3 workload. Layer E has an independent maximum of 50,000 finite analytic pixels and an estimated/actual candidate-cell maximum of 125,000 at resolution 13. These limits are intentional release defaults; any increase requires the Phase 7 benchmark series and an approved capacity/configuration change.

The Phase 9 integration test must record peak RSS on the largest approved fixture. Initial release gate: peak RSS below 2.5 GB for a 25-million-pixel window or lower the configured limit based on worker memory.

### 12.2 Logging

Use standard logging, not `print`. Include these event fields in message text or structured extras:

- algorithm version;
- farm/field/scene/run IDs;
- stage;
- crop window dimensions;
- band presence/missing list;
- selected/display/analytic counts;
- scene status;
- enabled/used indices;
- H3 candidate/emitted counts;
- elapsed milliseconds; and
- stable error code on expected failure.

Never log complete crop coordinates, credentials, signed URLs, raw arrays, or customer payloads.

### 12.3 Determinism and locale

- Do not seed or use randomness.
- Do not depend on directory iteration order.
- Force sorted output order.
- Use ISO strings passed by the caller; do not infer local time.
- CSV decimal separator is `.`.
- All transforms use explicit CRS and `always_xy=True`.

---

## 13. Pull Request Order and Ownership

| PR | Scope | May be assigned to | Depends on |
|---:|---|---|---|
| 1 | scaffold, errors, fixtures | Dev A | none |
| 2 | config and types | Dev A | PR 1 |
| 3 | geometry and scene I/O | Dev B | PR 2 |
| 4 | artifact/math utilities | Dev C | PR 2 |
| 5 | Layer A | Dev A | PR 3 + PR 4 |
| 6 | Layer B | Dev B | PR 5 |
| 7 | Layer C | Dev C | PR 6 |
| 8 | H3 adapter + Layer E | strongest geospatial developer | PR 7 |
| 9 | Layer D + pipeline | Dev A | PR 8 |
| 10 | production adapter | Dev B | PR 9 |
| 11 | worker integration | worker maintainer | PR 10 |
| 12 | regression catalog/release | team + agronomy reviewer | PR 11 |

PRs 3 and 4 may run in parallel after PR 2. All others merge in dependency order. Avoid assigning Layer E as a first geospatial task to an inexperienced developer without paired review; its instructions are complete, but geometry/API mistakes can produce plausible wrong scores.

Every PR description must include:

1. plan phase and files changed;
2. contracts implemented;
3. tests added and exact command/result;
4. artifact or metric examples where applicable;
5. memory/runtime observations where applicable; and
6. any plan deviation, which requires approval before merge.

---

## 14. Code Review Checklist

Reviewers answer every item explicitly:

- [ ] No upstream worker/cloud/download concern entered the analytical package.
- [ ] No layer searches disk for another layer's output.
- [ ] Required and optional bands remain distinct.
- [ ] Continuous/categorical resampling is correct.
- [ ] Source nodata/coverage cannot become valid zero.
- [ ] Crop window bounds memory before allocation.
- [ ] Array shapes, dtypes, and score units match contracts.
- [ ] No unconfigured numeric threshold exists in processor code.
- [ ] JSON cannot contain NaN or Infinity.
- [ ] Fallback pixel index uses its own configured range.
- [ ] Quality provenance exposes missing optional sources.
- [ ] H3 uses overlap area in metric CRS, not center-only scoring.
- [ ] Output order is deterministic.
- [ ] Partial artifacts cannot be mistaken for published artifacts.
- [ ] Production cleanup targets only its own staging directory.
- [ ] Errors retain stable codes and original causes.
- [ ] Boundary, negative, and rejection tests exist.
- [ ] Scientific output changed only with an approved regression update.

---

## 15. Release Gates

Release `1.0.0` only when all of these are true:

1. unit and contract suites pass with no warnings treated as ignored failures;
2. statement coverage is at least 90% overall and 95% for math, quality, health, H3 adapter, and public serializer modules;
3. all branch conditions for scene status and class thresholds are tested;
4. mixed-resolution integration passes;
5. rejected/display-only/ready paths pass;
6. atomic publication failure injection passes;
7. the representative memory gate passes in the worker-sized environment;
8. H3 share conservation passes;
9. regression manifests are reviewed by the technical lead and domain owner;
10. public JSON schema is accepted by the worker/backend integration test;
11. dependency lock and software bill of materials are generated; and
12. `CODEBASE_CONTEXT.md` is updated to replace “documentation only” with the actual implementation map.

Recommended commands after packaging exists:

```powershell
python -m pytest crop_health_sentinel/tests/unit -q
python -m pytest crop_health_sentinel/tests/contract -q
python -m pytest crop_health_sentinel/tests/integration -q
python -m pytest crop_health_sentinel/tests/regression -q
python -m pytest crop_health_sentinel/tests --cov=crop_health_sentinel --cov-report=term-missing
```

---

## 16. Definition of Done

The work is done when a new developer can take one approved fixture and answer, from artifacts and code contracts alone:

1. which exact source pixels were trusted and why;
2. which source/resampling created each aligned band;
3. which formula and finite samples produced each spectral statistic;
4. which indices, ranges, and normalized weights produced the field score;
5. which index and range produced every pixel health value;
6. how each boundary pixel's score was shared among H3 cells;
7. where the only 0..100 to 0..1 conversion occurred;
8. why a scene was rejected, display-only, or analytics-ready;
9. which optional evidence was missing; and
10. why the published JSON is complete, deterministic, and not partial.

If any answer requires reading an undocumented magic constant, guessing coordinate order, inspecting another repository's internal implementation, or asking which fallback behavior was intended, the implementation is not complete.

---

## 17. Short Team Execution Brief

The implementation team should follow this rule of thumb:

> Build contracts first, write one layer at a time, prove every boundary with tiny hand-calculated fixtures, and do not change scientific policy while implementing plumbing.

The only phase requiring specialist pairing is H3 overlap aggregation. All other phases have their inputs, outputs, formulas, failure behavior, filenames, test conditions, and merge order fixed above. Developers are expected to write and test the code—not redesign the system during implementation.
