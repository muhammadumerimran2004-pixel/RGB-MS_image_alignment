# AgriLift Drone Alignment Remediation Blueprint

**Document type:** Architecture blueprint, implementation rulebook, and delivery roadmap  
**Audience:** Junior and mid-level software engineers, reviewers, QA engineers, and technical leads  
**System:** `drone_alignment`  
**Status:** Plan only — this document does not implement any change  
**Primary objective:** Prevent invalid raster pixels, weak cross-spectral similarity, and unvalidated fallbacks from producing corrupted alignments that are incorrectly reported as successful.

---

## 1. Executive Summary

The current alignment run failed because the engine treated invalid multispectral pixels as real image data, exhausted its feature-based registration paths, accepted an unvalidated phase-correlation result, and then generated a false-positive QA result.

The observed run provides the following evidence:

- The RGB raster has an alpha mask and an approximately 89% valid footprint.
- The MS raster has an alpha mask and an approximately 95% valid footprint.
- Invalid MS spectral pixels contain the non-zero value `4,294,967,296` (`2^32`).
- The current normalizer accepts every finite value greater than zero as valid.
- The invalid MS pixels therefore dominate percentile normalization.
- ORB and SIFT fail to obtain a trustworthy transform.
- Phase correlation returns a translation of approximately `(2090, 2766)` pixels, or `(62.7, 83.0)` metres.
- The configured maximum translation is 50 metres, but the fallback bypasses this guardrail.
- Only approximately 36.6% of band-1 pixels remain non-zero after warping.
- The QA path compares a synthetic center point with itself and reports `PASS` with `0 px` RMSE.

This is not one isolated defect. It is a failure chain. The remediation must address every layer:

1. Raster validity and masks.
2. Radiometric preprocessing.
3. Cross-spectral band selection.
4. Feature matching.
5. Phase-correlation fallback validation.
6. Transform footprint validation.
7. Independent QA.
8. Output lifecycle, memory, and disk usage.

The engine must follow a fail-closed policy: if no candidate transform can be independently verified, it must stop with a failure result and must not publish an aligned raster as successful.

---

## 2. Scope

### 2.1 In scope

- Correct use of alpha bands, raster masks, and nodata values.
- Valid-pixel-only normalization.
- CLAHE preprocessing for cross-spectral registration.
- Configurable green-to-green and red-to-red candidate registration.
- Selection of a winning registration candidate using objective scores.
- Validation of feature-based and phase-correlation transforms through one common guardrail layer.
- Valid-footprint retention and reference-overlap metrics.
- Honest QA statuses and reports.
- Bounded preview dimensions.
- Lower peak memory usage during warping.
- Safe handling of partial and failed output artifacts.
- Unit, integration, regression, and real-data acceptance tests.
- Operational guidance for disk cleanup and failed-run diagnosis.

### 2.2 Explicitly out of scope for the first remediation release

- RIFT integration.
- GPU acceleration.
- Non-rigid or deformable registration.
- Bundle adjustment across multiple flights.
- Changes to the Sentinel-2 crop-health pipeline.
- Automatic camera-specific radiometric calibration.
- Replacing GeoTIFF with another storage format.

RIFT may be evaluated later, after masking, normalization, CLAHE, and SIFT have been corrected and measured. Adding another registration engine before fixing invalid input preparation would hide the primary defect and increase maintenance cost.

---

## 3. Non-Negotiable Engineering Rules

These rules are system invariants. Pull requests that violate them must not be approved.

### Rule 1: Validity comes from raster metadata, never pixel magnitude alone

Do not define valid data as `pixel != 0` or `pixel > 0`.

Validity must be derived from, in priority order:

1. GDAL/rasterio dataset mask.
2. Alpha band.
3. Per-band mask.
4. Declared nodata value.
5. Finiteness checks.
6. Optional configured physical-range checks.

Pixel magnitude may be used only as an additional domain check after the formal mask is applied.

### Rule 2: Masks are first-class data

Every image array used for registration must travel with a boolean valid-data mask of the same height and width.

Never pass a registration image without its mask.

### Rule 3: Spectral bands and alpha bands are different data types

An alpha band must not be treated as a reflectance band.

- Spectral reflectance: bilinear interpolation.
- Alpha/mask data: nearest-neighbour interpolation.
- Output alpha/mask: written as a mask or alpha band according to the output contract.

### Rule 4: Every transform uses the same validation gateway

ORB, SIFT, phase correlation, and any future method must submit their candidate transform to one shared validator.

No fallback may construct an accepted `TransformResult` directly.

### Rule 5: QA must be independent of the estimator

Never report residual error from fabricated points.

Never assign `inlier_ratio = 1.0` to a fallback without real inliers.

If independent verification is unavailable, status must be `UNVERIFIED` or `FAIL`, never `PASS`.

### Rule 6: A failed alignment must not look successful

The CLI must return a non-zero exit code when all candidates fail.

Reports must distinguish:

- `PASS`
- `WARNING`
- `FAIL`
- `UNVERIFIED`

Only `PASS` and explicitly permitted `WARNING` results may be published as final aligned products.

### Rule 7: Outputs are written atomically

Write final products to a run-scoped staging location first. Move them to final names only after validation passes.

Failed runs must not leave a final-looking aligned TIFF.

### Rule 8: Preview images are diagnostic thumbnails

QA previews must be downsampled to a configured maximum dimension. They must not duplicate full orthomosaic resolution.

### Rule 9: Configuration owns thresholds

No threshold may be embedded as an unexplained magic number in algorithm code.

Every threshold requires:

- A typed configuration field.
- A description and valid range.
- A documented default.
- At least one boundary test.

### Rule 10: Logs and reports must explain every rejected candidate

Candidate rejection is expected behavior, not an exceptional mystery. Record the method, channel pair, metrics, threshold, and rejection reason.

---

## 4. Target Processing Architecture

```text
Input validation
    |
    v
Raster metadata + spectral-band map + valid masks
    |
    v
Common geospatial grid and coarse reprojection
    |-- RGB registration bands
    |-- MS registration bands
    |-- RGB valid mask
    `-- MS valid mask
    |
    v
Common valid registration ROI
    |
    v
Candidate generation
    |-- Green/Green: ORB
    |-- Green/Green: SIFT
    |-- Red/Red: ORB
    |-- Red/Red: SIFT
    `-- Masked phase correlation, only if feature candidates fail
    |
    v
Shared transform validation gateway
    |-- numeric/matrix checks
    |-- translation/rotation/scale checks
    |-- inlier and spatial-distribution checks
    |-- retained-footprint check
    |-- RGB/MS valid-overlap check
    `-- independent similarity/verification check
    |
    v
Candidate ranking and winner selection
    |
    +--> no acceptable candidate --> FAIL; report reasons; publish no final TIFF
    |
    v
Band-by-band warp + output mask
    |
    v
Independent post-warp QA
    |
    +--> QA failure --> FAIL; retain optional diagnostics only
    |
    v
Atomic publication of TIFF, report, and bounded preview
```

---

## 5. Core Data Contracts

The names below are recommended contracts. Engineers may adjust exact names, but not their responsibilities.

### 5.1 `RasterMetadata`

Extend metadata to include:

- `band_descriptions: tuple[str | None, ...]`
- `color_interpretations: tuple[str, ...]`
- `alpha_band_index: int | None`
- `spectral_band_indices: tuple[int, ...]`
- `has_dataset_mask: bool`
- `nodata_by_band: tuple[float | None, ...]`

Acceptance rule: an alpha band must not appear in `spectral_band_indices`.

### 5.2 `MaskedBand`

Recommended immutable data structure:

```python
@dataclass(frozen=True)
class MaskedBand:
    values: np.ndarray
    valid_mask: np.ndarray
    band_index: int
    band_name: str
```

Contract:

- `values.shape == valid_mask.shape`
- `valid_mask.dtype == bool`
- Invalid pixels must not influence any statistical operation.
- Callers must not assume invalid pixels contain zero.

### 5.3 `CoarseAlignmentResult`

The coarse result should provide:

- Registration band arrays keyed by semantic channel name.
- RGB valid mask.
- MS valid mask.
- Common valid mask.
- Spectral MS bands or a source/window contract for later warping.
- Target geospatial profile.
- Target GSD.
- Reprojection scale information.
- Source band metadata needed to preserve output semantics.

Avoid ambiguous fields such as `ms_all_bands` if the array includes alpha as though it were spectral data.

### 5.4 `RegistrationCandidate`

Each method attempt must return a candidate or a typed rejection:

```python
@dataclass(frozen=True)
class RegistrationCandidate:
    method: str
    channel_pair: str
    matrix: np.ndarray
    transform_type: TransformType
    raw_match_count: int | None
    good_match_count: int | None
    inlier_count: int | None
    inlier_ratio: float | None
    phase_response: float | None
    source_points: np.ndarray | None
    reference_points: np.ndarray | None
```

The candidate is not accepted merely because it exists.

### 5.5 `TransformValidationResult`

Recommended fields:

- `accepted: bool`
- `reasons: list[str]`
- `translation_px`
- `translation_m`
- `rotation_deg`
- `scale`
- `retained_source_valid_ratio`
- `reference_overlap_ratio`
- `overlap_coefficient`
- `inlier_coverage_ratio`
- `independent_similarity_score`

### 5.6 `AlignmentRunResult`

The run result must include:

- Run status.
- Winning candidate, if any.
- Rejected candidates and reasons.
- Output paths only for files that actually exist.
- Disk and timing statistics.
- Whether final products were published.

---

## 6. Detailed Component Plan

## 6.1 Configuration schema

**Primary file:** `drone_alignment/config/schema.py`

Add semantic channel selection instead of assuming red-only registration.

Recommended configuration concepts:

```yaml
registration:
  channel_priority:
    - green
    - red
  try_all_channel_pairs: true
  apply_clahe: true
  clahe_clip_limit: 2.0
  clahe_grid_size: 8
  percentile_low: 2.0
  percentile_high: 98.0
  min_common_valid_fraction: 0.20

bands:
  rgb:
    red: 1
    green: 2
  ms:
    red: 1
    green: 2

phase_correlation:
  enabled: true
  min_response: 0.15
  remove_mean: true
  use_hann_window: true

transform:
  max_translation_m: 50.0
  max_rotation_deg: 5.0
  max_scale_deviation: 0.10
  min_inlier_ratio: 0.30
  min_inlier_coverage_ratio: 0.20
  min_retained_valid_ratio: 0.80
  min_reference_overlap_ratio: 0.80

quality:
  preview_max_dimension: 1600
  publish_warning_results: false
```

Junior implementation checklist:

1. Add enums for semantic channels and result statuses.
2. Add nested Pydantic models rather than adding unrelated fields to one class.
3. Constrain percentages to `(0, 1]` and percentile values to `[0, 100]`.
4. Add a model-level validation that `percentile_low < percentile_high`.
5. Add a model-level validation that configured channel band indices exist during input validation.
6. Preserve compatibility with existing red-band fields for one deprecation cycle or provide an explicit migration error.

Acceptance criteria:

- Invalid threshold values fail before raster processing begins.
- Green and red indices are independently configurable.
- Default priority is green, then red.
- Existing YAML errors include the exact invalid field path.

## 6.2 Raster metadata and mask reading

**Primary files:**

- `drone_alignment/io/reader.py`
- `drone_alignment/io/validators.py`

Required behavior:

1. Read band descriptions and color interpretation.
2. Detect alpha bands using raster metadata, not the assumption that the final band is alpha.
3. Obtain valid pixels using `dataset_mask()` or per-band masks.
4. Combine mask sources deterministically.
5. Treat non-finite spectral values as invalid.
6. Optionally reject physically impossible reflectance values through configuration, but do not silently invent a physical range.
7. Return mask and values separately.

Recommended combined validity expression:

```text
valid = dataset_mask_valid
      AND per_band_mask_valid
      AND pixel_is_finite
      AND pixel_is_not_declared_nodata
      AND optional_domain_range_check
```

Important case:

The current MS file stores `2^32` in alpha-invalid spectral pixels. A non-zero test marks those pixels valid and is therefore forbidden.

Acceptance criteria:

- A fixture containing alpha-invalid `2^32` pixels reports them invalid.
- Valid negative values are not automatically discarded unless the domain configuration rejects them.
- A raster without alpha or nodata remains supported through an all-valid mask plus finiteness checks.
- Band index validation rejects attempts to use alpha as a registration channel.

## 6.3 Coarse reprojection

**Primary file:** `drone_alignment/alignment/coarse.py`

Required behavior:

1. Calculate the common geospatial target grid as currently intended.
2. Reproject RGB and MS validity masks with nearest-neighbour interpolation.
3. Reproject only configured registration bands for feature detection.
4. Prevent invalid source values from bleeding into valid pixels during bilinear interpolation.
5. Reproject reflectance using bilinear interpolation.
6. Create a common-valid mask with `rgb_valid & ms_valid`.
7. Measure the common-valid fraction before attempting registration.
8. Fail early when the common-valid region is empty or below its configured minimum.

Implementation warning:

Masking only after bilinear reprojection is insufficient. Extreme invalid values may already have contaminated neighbouring pixels. Invalid source pixels must be excluded or replaced before interpolation, and the reprojected validity mask must still be applied afterward.

Acceptance criteria:

- The `2^32` sentinel never appears in reprojected registration bands.
- Masks contain only boolean values or categorical `0/1` values.
- Mask edges are not bilinearly blurred.
- Registration arrays share one grid and identical dimensions.

## 6.4 Valid-pixel normalization and CLAHE

**Primary file:** `drone_alignment/alignment/feature_detector.py`

Required preprocessing sequence:

1. Receive image plus mask.
2. Extract valid finite values only.
3. Calculate configured low/high percentiles from those values.
4. Clip valid pixels to that interval.
5. Scale to unsigned 8-bit.
6. Set invalid output pixels to zero.
7. Optionally apply CLAHE.
8. Reapply the mask after CLAHE.
9. Resize image and mask consistently when downsampling.
10. Pass the resized mask to OpenCV feature detection.

Pseudocode:

```python
def preprocess(image, valid_mask, config):
    valid = valid_mask & np.isfinite(image)
    if valid.sum() < config.minimum_valid_pixels:
        raise InsufficientValidDataError(...)

    low, high = percentile(image[valid], configured_percentiles)
    if high <= low:
        raise InsufficientContrastError(...)

    normalized = zeros_uint8(image.shape)
    normalized[valid] = scale_and_clip(image[valid], low, high)

    if config.apply_clahe:
        normalized = clahe(normalized)
        normalized[~valid] = 0

    return normalized, valid
```

Acceptance criteria:

- Invalid extreme values do not affect percentiles.
- Constant valid imagery fails with a typed insufficient-contrast error.
- CLAHE never turns invalid canvas into usable feature texture.
- ORB/SIFT receives a valid-data mask.
- Unit tests assert actual scaled values, not only dtype and shape.

## 6.5 Channel-pair strategy

**Primary files:**

- `drone_alignment/pipeline.py`
- `drone_alignment/config/schema.py`

Recommended initial policy:

1. Green RGB to green MS.
2. Red RGB to red MS.

For each pair:

1. Apply identical mask-aware preprocessing policy.
2. Attempt configured primary detector.
3. Attempt SIFT if the primary detector fails.
4. Submit every estimated transform to the shared validator.
5. Record both accepted and rejected candidate metrics.

Do not permanently assume green is universally superior. Green is the preferred first attempt for this dataset because it provides a visually useful representation, but red may perform better for other cameras, crops, or illumination conditions.

Candidate ranking must prioritize geometric reliability over raw match count. Recommended ranking order:

1. Accepted versus rejected.
2. Independent QA score.
3. Spatial inlier coverage.
4. Inlier ratio.
5. Number of inliers.
6. Lower residual error.

Acceptance criteria:

- One bad channel pair does not terminate the run before the next configured pair is attempted.
- Reports identify the winning pair and all rejected pairs.
- Alpha, NIR, and red-edge are never selected accidentally because of band order.

## 6.6 Feature matching and spatial distribution

**Primary files:**

- `drone_alignment/alignment/feature_detector.py`
- `drone_alignment/alignment/feature_matcher.py`
- `drone_alignment/alignment/transform_estimator.py`

Raw inlier ratio is necessary but insufficient. Twenty matches concentrated along one road or one image corner may produce an unstable affine transform.

Add an inlier-coverage measure:

```text
inlier_coverage_ratio = area(convex_hull(reference_inliers))
                      / area(valid_reference_footprint_bbox)
```

Alternative for very irregular masks: use occupied cells in an `N x N` valid grid.

Required checks:

- Minimum good matches.
- Minimum inlier count.
- Minimum inlier ratio.
- Minimum spatial coverage.
- Finite matrix coefficients.
- Non-degenerate determinant.

Acceptance criteria:

- Clustered matches can be rejected even when their count is high.
- Match coordinates are always converted correctly from downsampled to target-grid pixels.
- Tests cover both well-distributed and corner-clustered match sets.

## 6.7 Phase-correlation fallback

**Primary file:** `drone_alignment/alignment/transform_estimator.py`

Phase correlation is a candidate generator, not an automatic success path.

Required input preparation:

1. Use the common valid mask.
2. Crop both arrays to the common-valid bounding box.
3. Normalize both arrays independently using valid pixels only.
4. Set invalid pixels consistently after mean removal.
5. Remove the valid-region mean when configured.
6. Apply a Hann window when configured.
7. Preserve the returned response score.
8. Convert cropped coordinates back to full-grid coordinates correctly.

Required validation:

- Response meets `min_response`.
- Translation meets `max_translation_m`.
- Matrix passes finite and determinant checks.
- Retained valid footprint meets its threshold.
- RGB/MS overlap meets its threshold.
- Independent similarity improves over the unrefined coarse alignment by a configured minimum.

Never create synthetic inliers for phase correlation.

Recommended result representation:

- `inlier_ratio = null`
- `num_inliers = null`
- `phase_response = actual response`
- `verification_method = actual independent method`

Acceptance criteria:

- The known `(2090, 2766)` pixel result is rejected by the translation and footprint gates.
- A low-response random-noise result is rejected.
- A known synthetic translation is accepted when it satisfies all thresholds.
- Shift direction is covered by an assertion, not merely matrix shape.

## 6.8 Shared transform validation gateway

**Primary file:** `drone_alignment/alignment/transform_estimator.py` or a new focused module such as `alignment/transform_validator.py`

Every method must use this gateway.

Validation sequence:

1. Check matrix shape.
2. Check all coefficients are finite.
3. Check determinant/non-degeneracy.
4. Decompose translation, rotation, and scale.
5. Check translation threshold.
6. Check rotation threshold.
7. Check scale threshold.
8. Check method-specific evidence: inliers or phase response.
9. Warp the MS mask only.
10. Calculate footprint metrics.
11. Run independent verification.
12. Return accepted/rejected result with every reason.

The gateway should accumulate useful rejection reasons where safe instead of stopping at the first one. This gives staff better diagnostics.

Acceptance criteria:

- No caller can obtain an accepted transform without the gateway.
- Feature and phase candidates share translation and overlap rules.
- Unit tests cover each rejection reason independently.

## 6.9 Footprint and overlap metrics

**Recommended module:** `drone_alignment/quality/footprint.py`

Define metrics precisely so different engineers do not implement different meanings of “80% overlap.”

Let:

- `M_before` be valid MS pixels on the coarse target grid.
- `M_after` be `M_before` warped by the candidate transform.
- `R` be valid RGB pixels on the target grid.
- `I = M_after AND R`.

Required metrics:

```text
retained_source_valid_ratio = count(M_after) / count(M_before)

reference_overlap_ratio = count(I) / count(R)

target_overlap_ratio = count(I) / count(M_after)

overlap_coefficient = count(I) / min(count(R), count(M_after))

intersection_over_union = count(I) / count(R OR M_after)
```

The primary clipping guardrail is `retained_source_valid_ratio`.

For the current same-flight survey, a default minimum of `0.80` is reasonable. It must remain configurable because legitimate input footprints may differ.

Acceptance criteria:

- Identity transform gives approximately 1.0 retained ratio.
- Large translations produce a low retained ratio.
- Metrics handle empty masks without division errors and return a typed failure.
- Mask warping uses nearest-neighbour interpolation.

## 6.10 Independent QA

**Primary files:**

- `drone_alignment/quality/metrics.py`
- `drone_alignment/quality/report.py`

The estimator’s own inliers are useful diagnostics but are not fully independent verification.

Recommended first-release QA:

1. Reserve a subset of feature matches before transform estimation, or detect verification features using a separate grid/sample after estimation.
2. Compute residuals for real verification correspondences.
3. Measure center and edge/corner residuals.
4. Require minimum verification-point count and spatial coverage.
5. Combine residual QA with footprint checks.

If independent correspondences cannot be obtained:

- Report `UNVERIFIED`.
- Do not publish as `PASS`.
- Preserve diagnostic preview/report if configured.

Status policy:

| Condition | Status |
|---|---|
| All mandatory geometric, footprint, and independent QA gates pass | `PASS` |
| Mandatory gates pass but a documented advisory threshold is marginal | `WARNING` |
| Any mandatory gate fails | `FAIL` |
| Plausible transform exists but independent QA cannot be calculated | `UNVERIFIED` |

Acceptance criteria:

- Phase fallback can never receive a fabricated zero residual.
- A report cannot show `PASS` with zero real verification points.
- `grid_residuals_count` counts actual points only.

## 6.11 Warping and memory use

**Primary file:** `drone_alignment/alignment/warper.py`

The current implementation allocates a complete `warped_bands` array in addition to the complete coarse MS cube. This causes unnecessary peak memory and may force Windows pagefile use.

First remediation step:

1. Open the output dataset.
2. Warp one spectral band at a time.
3. Write that band immediately.
4. Release the temporary warped band before processing the next band.
5. Warp the validity mask separately with nearest-neighbour interpolation.
6. Write the output dataset mask or alpha according to the output contract.

Later optimization, if measurements require it:

- Window/tile-level inverse mapping with a calculated source window and halo.
- Do not implement tiled geometric warping without tests for seams and transform direction.

Output requirements:

- Tiled GeoTIFF.
- Compression enabled.
- Predictor appropriate for float data.
- `BIGTIFF=IF_SAFER` or equivalent policy.
- Correct nodata/mask semantics.
- Band descriptions preserved.

Acceptance criteria:

- Peak memory is materially below the old implementation on the real dataset.
- Alpha is not interpolated bilinearly.
- No tile seams are visible or numerically discontinuous.

## 6.12 Preview generation

**Primary file:** `drone_alignment/quality/visualization.py`

Required changes:

1. Downsample both images and masks to `preview_max_dimension` before composing panels.
2. Use mask-aware normalization.
3. Render invalid pixels with an explicit neutral/background colour.
4. Add footprint outlines if practical.
5. Include method, channel pair, status, and key metrics in labels.
6. Generate the preview from the selected registration channel, not a hard-coded red band.

Acceptance criteria:

- The current approximately 196 MiB preview becomes a small diagnostic artifact, normally a few MiB or less.
- The preview preserves aspect ratio.
- Invalid fill values never render as false bright features.

## 6.13 Reports and logs

**Primary files:**

- `drone_alignment/quality/report.py`
- `drone_alignment/pipeline.py`

Recommended report additions:

```json
{
  "run_status": "FAIL",
  "selected_candidate": null,
  "input_validity": {
    "rgb_valid_fraction": 0.89,
    "ms_valid_fraction": 0.95,
    "common_valid_fraction": 0.84,
    "rgb_mask_source": "alpha",
    "ms_mask_source": "alpha"
  },
  "candidates": [
    {
      "method": "sift",
      "channel_pair": "green-green",
      "accepted": false,
      "rejection_reasons": ["inlier coverage below threshold"]
    }
  ],
  "footprint": {
    "retained_source_valid_ratio": 0.37,
    "reference_overlap_ratio": 0.40
  },
  "publication": {
    "final_raster_published": false
  }
}
```

Logging requirements:

- One start/end event per stage.
- Image dimensions, GSD, and mask fractions.
- Preprocessing percentile bounds per candidate channel.
- Keypoint, match, and inlier counts.
- Phase response.
- Every guardrail result.
- Peak or sampled memory if operationally available.
- Output file sizes.
- Cleanup actions for failed staging artifacts.

Do not log whole arrays, full coordinate lists, or image data.

## 6.14 Atomic outputs and cleanup

**Primary file:** `drone_alignment/pipeline.py`

Recommended run layout:

```text
output_dir/
  .staging/
    <run_id>/
      candidate_preview.png
      aligned_ms.partial.tif
      diagnostics.json
  <ms_stem>_aligned.tif
  <ms_stem>_alignment_preview.png
  <ms_stem>_alignment_report.json
```

Policy:

- Final names appear only after `PASS`, or permitted `WARNING`.
- On `FAIL`, delete partial TIFFs by default.
- Retain a compact failure report and downsampled preview when configured.
- Never delete source rasters.
- Cleanup must target the resolved run staging directory, not a broad or unresolved path.

---

## 7. Pipeline State Machine

Implement orchestration as explicit states rather than deeply nested exception blocks.

```text
CREATED
  -> INPUTS_VALIDATED
  -> COARSE_GRID_READY
  -> CANDIDATES_EVALUATED
  -> WINNER_SELECTED
  -> WARP_STAGED
  -> QA_PASSED
  -> PUBLISHED

Any state may transition to FAILED with a typed reason.
```

Required failure categories:

- `INPUT_INVALID`
- `NO_SPATIAL_OVERLAP`
- `INSUFFICIENT_VALID_DATA`
- `INSUFFICIENT_CONTRAST`
- `INSUFFICIENT_MATCHES`
- `TRANSFORM_DEGENERATE`
- `TRANSFORM_OUT_OF_BOUNDS`
- `LOW_PHASE_RESPONSE`
- `FOOTPRINT_RETENTION_FAILED`
- `INDEPENDENT_QA_FAILED`
- `OUTPUT_WRITE_FAILED`

Do not use a broad catch to convert programming defects into algorithm fallbacks. Only typed, expected registration failures should advance to another candidate.

---

## 8. Delivery Roadmap

The phases below are dependency ordered. Do not start later algorithm work before the mask foundation is accepted.

## Phase 0: Capture the failing baseline

### Tasks

1. Preserve the current JSON report and preview as regression fixtures or documented evidence.
2. Add a small synthetic raster fixture with:
   - Four reflectance bands.
   - One alpha band.
   - Valid reflectance around `0–0.15`.
   - Alpha-invalid spectral fill equal to `2^32`.
3. Add a failing test that proves the old normalizer collapses valid contrast.
4. Record current real-data output size, runtime, and peak memory.

### Exit criteria

- The defect is reproducible without requiring the large private/source orthomosaics.
- Baseline measurements are stored in test notes or a benchmark report.

## Phase 1: Mask and metadata foundation

### Tasks

1. Extend raster metadata.
2. Implement mask-aware band reads.
3. Identify alpha versus spectral bands.
4. Add input validation for configured semantic bands.
5. Reproject masks with nearest-neighbour interpolation.
6. Prevent invalid-value interpolation contamination.

### Exit criteria

- All invalid `2^32` pixels are excluded.
- Unit and coarse-alignment tests pass.
- No registration algorithm changes are required to verify this phase.

## Phase 2: Preprocessing and channel candidates

### Tasks

1. Implement valid-only percentile normalization.
2. Add CLAHE configuration and preprocessing.
3. Pass masks to ORB/SIFT.
4. Add green and red semantic channel configuration.
5. Implement candidate attempts and metric capture.

### Exit criteria

- Green-green and red-red candidates can be attempted independently.
- Invalid canvas produces no keypoints.
- Candidate failure reasons are recorded.

## Phase 3: Unified validation and safe fallback

### Tasks

1. Introduce the shared transform validation gateway.
2. Route feature transforms through it.
3. Refactor phase correlation to return response and a candidate.
4. Route phase candidates through the same gateway.
5. Add footprint metrics.
6. Add spatial inlier-coverage validation.

### Exit criteria

- No fallback bypasses configured thresholds.
- The known catastrophic translation is rejected.
- All candidate types produce comparable validation records.

## Phase 4: Honest QA and reporting

### Tasks

1. Remove fabricated fallback control points.
2. Implement independent verification points or an approved alternative.
3. Add `UNVERIFIED` status.
4. Expand report schema.
5. Add candidate rejection information.
6. Add mask and footprint metrics.

### Exit criteria

- `PASS` is impossible without real QA evidence.
- Reports explain why a run failed without requiring source-code inspection.

## Phase 5: Output safety and resource reduction

### Tasks

1. Warp/write one band at a time.
2. Preserve/write the output mask correctly.
3. Downsample previews.
4. Add staging and atomic publication.
5. Clean partial run artifacts safely.
6. Log output sizes.

### Exit criteria

- Preview size is bounded.
- Peak memory is reduced.
- Failed runs publish no final-looking TIFF.
- Re-running with the same final name is deterministic.

## Phase 6: Real-data validation and controlled rollout

### Tasks

1. Run on the known RGB/MS pair.
2. Compare green-green and red-red evidence.
3. Inspect the output in QGIS using both automatic and fixed reflectance ranges.
4. Validate mask/alpha behavior.
5. Measure retained footprint and independent residuals.
6. Run on at least two additional survey pairs if available.
7. Tune defaults only from recorded evidence.

### Exit criteria

- No catastrophic candidate is accepted.
- At least one channel/method passes all gates on valid alignable data, or the engine honestly fails if the source pair cannot be registered.
- A reviewer signs off on visual and numeric QA.

---

## 9. Test Strategy

## 9.1 Unit tests

### Reader and masks

- Alpha-invalid `2^32` pixels are invalid.
- Dataset masks are honoured.
- Declared nodata is honoured.
- NaN and infinity are invalid.
- Valid zero or negative values remain valid when the mask permits and no physical-range policy rejects them.
- Alpha is excluded from semantic band choices.

### Normalization

- Valid reflectance `0–0.15` spans a useful 8-bit range.
- Invalid extremes do not affect percentiles.
- Invalid output remains zero after CLAHE.
- Constant imagery produces a typed failure.
- Percentile configuration boundaries are validated.

### Feature detection

- No keypoints occur outside the valid mask.
- Downsampled image and mask shapes match.
- Known textured valid regions produce keypoints.

### Transform validation

- Accept valid identity and small translations.
- Reject excessive translation.
- Reject excessive rotation.
- Reject excessive scale.
- Reject NaN/Inf matrix.
- Reject degenerate matrix.
- Reject weak phase response.
- Reject low footprint retention.
- Reject clustered inliers.

### Footprint metrics

- Identity, partial shift, no overlap, and unequal-footprint cases.
- Empty-mask handling.

### QA status

- Zero verification points cannot pass.
- Valid distributed checkpoints can pass.
- Edge drift fails the edge criterion.
- `UNVERIFIED` is serialized correctly.

## 9.2 Integration tests

1. Alpha-masked RGB/MS pair with valid small translation.
2. Same pair with invalid fill values of `2^32`.
3. Pair where green succeeds and red fails.
4. Pair where red succeeds and green fails.
5. Pair where all feature methods fail but strong phase translation succeeds.
6. Pair where phase response is weak and the entire run fails.
7. Pair where transform passes numeric bounds but clips more than 20% and fails footprint validation.
8. Failed run leaves no final aligned TIFF.
9. Successful run preserves CRS, transform, band descriptions, and valid mask.
10. Preview respects its maximum dimension.

## 9.3 Real-data acceptance tests

For each real survey pair, record:

- Input paths or immutable dataset IDs.
- Band mapping.
- Valid fractions.
- Candidate attempts.
- Winning method and channel pair.
- Inlier count, ratio, and coverage.
- Phase response if used.
- Translation, rotation, and scale.
- Footprint metrics.
- Independent QA metrics.
- Output size.
- Peak memory and runtime.
- QGIS visual review result.

Do not make visual inspection the only acceptance criterion.

## 9.4 Regression tests derived from this incident

The following assertions must remain permanently:

- `2^32` alpha-invalid fill cannot affect normalization.
- A phase transform exceeding 50 metres cannot be accepted when the configured maximum is 50 metres.
- A transform retaining approximately 36.6% of valid data cannot pass an 80% retention gate.
- Phase correlation cannot report fake `1/1` inliers.
- QA cannot report `PASS` from a point copied onto itself.

---

## 10. Candidate Selection Policy

Recommended initial attempt order:

```text
1. Green/Green + configured primary detector
2. Green/Green + SIFT
3. Red/Red + configured primary detector
4. Red/Red + SIFT
5. Masked phase correlation on the best-supported channel pair
```

If the configured primary detector is already SIFT, do not run SIFT twice.

All acceptable feature candidates should be compared before selecting a winner when `try_all_channel_pairs` is enabled. Do not automatically choose the first numerically valid candidate if a substantially stronger candidate is available.

Phase correlation should remain lower priority because it estimates translation only and provides weaker geometric evidence.

---

## 11. Error Handling Rules

Expected algorithm failures should use typed exceptions or result types:

- Insufficient valid pixels.
- Insufficient contrast.
- Insufficient keypoints.
- Insufficient matches.
- Transform estimation failure.
- Candidate validation failure.

Programming errors, shape mismatches, and unexpected I/O faults must not silently trigger the next algorithm. They should stop the run and expose a clear error.

Every exception message should state:

1. What operation failed.
2. Which channel and method were being used.
3. The measured value.
4. The required threshold.
5. Whether another candidate will be attempted.

---

## 12. Observability and Operational Diagnostics

A normal run should allow an operator to answer these questions from logs/report alone:

1. Which bands were used?
2. Where did their band identities come from?
3. How much of each input was valid?
4. What ranges were used for normalization?
5. How many keypoints and matches were found?
6. Why was each candidate accepted or rejected?
7. What transform was selected?
8. How much valid data remained after transformation?
9. How was the transform independently verified?
10. Were final outputs published?
11. How much disk and memory did the run use?

Recommended per-run identifier:

```text
<UTC timestamp>-<short random suffix>
```

Include it in logs, staging paths, and reports.

---

## 13. Disk and Memory Operations Guide

### Current behavior

The observed failed run generated approximately:

- 377.6 MiB aligned TIFF.
- 196.3 MiB QA PNG.
- Approximately 573.9 MiB total generated output.

The current implementation also holds a complete coarse MS cube and a complete warped cube, contributing to pagefile activity under memory pressure.

### Safe cleanup rule

Only delete run outputs or run-scoped staging directories. Never delete source TIFFs.

Current failed-run cleanup example:

```powershell
Remove-Item -LiteralPath ".\Stuff\aligned\odm_orthophoto_aligned.tif"
Remove-Item -LiteralPath ".\Stuff\aligned\odm_orthophoto_alignment_preview.png"
Remove-Item -LiteralPath ".\Stuff\aligned\odm_orthophoto_alignment_report.json"
```

Before adding automated cleanup, verify the resolved target is inside the configured output directory and belongs to the current run ID.

Do not attempt to delete or manually resize the Windows pagefile from application code.

---

## 14. Pull Request Decomposition

Keep reviews small enough that junior engineers can receive focused feedback.

### PR 1: Metadata and masked reads

- Metadata extensions.
- Mask-source detection.
- `MaskedBand` contract.
- Reader and validator tests.

### PR 2: Mask-aware coarse reprojection

- Reprojected masks.
- Invalid-value exclusion before interpolation.
- Common-valid ROI metrics.
- Coarse tests.

### PR 3: Preprocessing and CLAHE

- Valid-only percentile normalization.
- CLAHE.
- Masked ORB/SIFT detection.
- Preprocessing tests.

### PR 4: Semantic channel attempts

- Green/red configuration.
- Candidate collection.
- Candidate logs.
- Channel-specific tests.

### PR 5: Unified transform validator

- Numeric/geometric validation.
- Phase response.
- Footprint metrics.
- Spatial match coverage.
- Validator tests.

### PR 6: Independent QA and report schema

- Real verification evidence.
- Status model.
- Candidate/rejection reporting.
- QA tests.

### PR 7: Resource and output lifecycle

- Band-at-a-time warp.
- Correct output mask.
- Bounded preview.
- Staging and atomic publication.
- Failure cleanup tests.

### PR 8: Real-data acceptance and tuning

- Recorded benchmark.
- Threshold tuning supported by evidence.
- Operational documentation updates.

Each PR must include tests and must leave the full suite passing. Avoid combining all phases into one unreviewable change.

---

## 15. Junior Engineer Implementation Checklist

Before coding:

- Read this blueprint fully.
- Identify the exact PR scope.
- Read the target module and its existing tests.
- Write or update a failing test that demonstrates the intended behavior.
- Confirm whether an input band is spectral or alpha.

While coding:

- Keep arrays and masks paired.
- Assert matching shapes at module boundaries.
- Use typed configuration.
- Use typed expected failures.
- Preserve coordinate-space documentation: source pixels, downsampled pixels, or target-grid pixels.
- Record actual metrics instead of placeholder values.
- Avoid loading extra full-resolution arrays.

Before requesting review:

- Run focused tests.
- Run the full test suite.
- Inspect report fields for honest null/unknown values.
- Confirm failed results cannot publish final output names.
- Confirm no source data was modified.
- Document memory or file-size changes when relevant.
- Include a short “how this was tested” section in the PR.

---

## 16. Reviewer Checklist

Reject the change if any answer below is “no”:

- Are raster masks used rather than inferred from non-zero pixels?
- Are alpha bands excluded from spectral processing?
- Are invalid values excluded before percentile calculation?
- Is mask interpolation nearest-neighbour?
- Does every candidate pass through the shared validator?
- Are phase response and translation both checked?
- Is footprint retention measured with a documented formula?
- Is QA based on real evidence?
- Can zero verification points ever produce `PASS`?
- Are output writes atomic or safely staged?
- Are preview dimensions bounded?
- Do tests reproduce the incident’s critical failure modes?

---

## 17. Definition of Done

The remediation is complete only when all conditions below are satisfied:

1. The MS `2^32` invalid-fill pixels are excluded through the raster mask.
2. Valid MS reflectance is normalized into a useful 8-bit range without manual QGIS styling.
3. Green-green and red-red are configurable candidates.
4. CLAHE is mask-aware and configurable.
5. ORB, SIFT, and phase candidates use one validation gateway.
6. Phase response is retained and evaluated.
7. No candidate can bypass translation, rotation, scale, or footprint limits.
8. The catastrophic 104 m translation is rejected.
9. A 36.6% retained footprint cannot satisfy the default 80% gate.
10. QA uses real verification evidence.
11. Zero verification points cannot produce `PASS`.
12. Failed runs do not publish a final-looking aligned TIFF.
13. Preview output is dimension-bounded.
14. Peak memory is reduced through band-at-a-time warping.
15. Unit and integration suites pass.
16. The real dataset produces either a verified acceptable alignment or an honest failure.
17. A technical reviewer and a QGIS visual reviewer sign off on the release evidence.

---

## 18. Future Evaluation: RIFT

RIFT should be evaluated only after the definition of done above is met.

Evaluation questions:

- Does masked CLAHE + SIFT still fail on a meaningful portion of representative datasets?
- Does RIFT improve verified success rate rather than merely return more matches?
- What are its runtime, memory, licensing, packaging, and maintenance costs?
- Can it return evidence compatible with the shared validation gateway?

RIFT must be added as another candidate generator. It must not bypass masks, shared transform validation, footprint checks, or independent QA.

---

## 19. Recommended First Assignment

The safest first junior assignment is PR 1: metadata and masked reads.

It is bounded, testable, and foundational. The engineer should deliver:

1. A synthetic alpha-masked raster containing `2^32` invalid fill.
2. A reader API that returns values plus a boolean mask.
3. Metadata that identifies the alpha band.
4. Validation that prevents alpha from being configured as red or green.
5. Tests showing that invalid fill is excluded while valid reflectance remains unchanged.

No registration algorithms should be modified in that PR.

