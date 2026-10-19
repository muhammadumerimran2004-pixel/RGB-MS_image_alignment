# Sentinel-2 Patch Co-Registration — Implementation Blueprint

**Status:** Design complete and approved for implementation (rev 3: product-owner decisions and the flight-date source chain folded in, see §12) · **Owner:** Umer Imran · **Date:** 2026-10-18
**Audience:** developers implementing the feature. Read Sections 1–4 and §12 before writing any code.

---

## 0. TL;DR

After `drone_alignment` publishes the aligned MS orthomosaic, a new optional stage:

1. extracts the **valid-data perimeter** of the aligned RGB+MS stack,
2. finds the **best cloud-free Sentinel-2 L2A scene** closest to the flight date,
3. reads **only the window** around that perimeter (cloud-optimized GeoTIFFs, no full-tile download),
4. estimates **one sub-pixel translation** between Sentinel-2 and a drone-derived "synthetic Sentinel" reference,
5. accepts the translation only if **independent safety gates** pass (same fail-closed philosophy as the alignment engine),
6. publishes the Sentinel patch **clipped to the perimeter**, with a per-pixel coverage fraction, a QA preview and a JSON report.

The run then has two products:

| # | Product | Already exists? |
|---|---------|-----------------|
| 1 | `<ms>_aligned.tif` — MS aligned to RGB | Yes (unchanged) |
| 2 | `sentinel/…` — Sentinel-2 patch aligned to the RGB/MS stack, perimeter-clipped | **New** |

A Sentinel failure **never** invalidates product 1.

---

## 1. Feasibility verdict and expectations

**It is doable**, and the open-source building blocks (STAC catalogues, COGs, AROSICS) were designed for this kind of task. However, the result is a different kind of alignment from MS→RGB. Make sure stakeholders understand these four points:

1. **Resolution gap is 100–500×.** Drone GSD is 2–10 cm; Sentinel-2 is 10 m (B2/B3/B4/B8) and 20 m (red-edge, SWIR). A 5 ha field is only ~500 Sentinel-2 pixels (e.g. 22 × 22). We are aligning a *very small, very blurry* image to a sharp one.
2. **Only a rigid translation is estimable and needed.** Sentinel-2 L1C/L2A is orthorectified, and its residual geolocation error is close to a constant offset over a few km. There are not enough pixels for affine/TPS, and fitting them would be overfitting. **Do not reuse COREG_LOCAL / TPS for this stage.**
3. **Realistic accuracy.** On textured scenes (field boundaries, roads, canals, tree lines) expect a residual of **~0.2–0.5 Sentinel pixel (2–5 m)**. On homogeneous crop-only patches the shift is **not observable**, and the stage must say so rather than guess.
4. **"Aligned" means aligned to the drone frame, not to ground truth.** The engine aligns MS to RGB. If the RGB itself was flown without RTK/PPK/GCPs, its absolute position can also be off by metres. The Sentinel patch will then inherit that frame. This is usually what we want (pixel-on-pixel overlay with our drone data), but the report must record it.

### Why bother aligning at all? (evidence)

- Sentinel-2's absolute geolocation, after ESA's GRI geometric refinement, is quoted at **< 7 m (CE95)**, and multi-temporal co-registration at **< 5 m (CE95)**. Unrefined products are around 11–12.5 m. Studies cite **4.5–8 m (2σ)**, and independent measurements show **6–12.7 m**. A 1-pixel (10 m) offset on a 20-pixel-wide field moves ~10–20 % of the pixels across the field edge, so mixed-edge pixels contaminate field statistics.
- Practitioners comparing UAV and Sentinel-2 either co-register manually with landmarks (field/road intersections; Germany study, *Remote Sensing* 14(17):4426) or skip co-registration and accept the error (Bonn study, PMC8073067, which cautions about S2 use at field scale). Our automated, gated approach is strictly better than both, provided the gates are honest.
- A UAS-guided super-resolution study reports a registration error of ~0.81 m after registration versus ~4.6 m without it, which shows the size of the effect.

---

## 2. Terminology

| Term | Meaning |
|------|---------|
| **Stack** | The published aligned MS raster plus the RGB reference it was aligned to. They share CRS and frame. |
| **Perimeter / footprint** | Polygon of pixels that are valid in **both** RGB and aligned MS (not the raster bounding box). |
| **AOI** | Footprint transformed to the Sentinel scene CRS. |
| **Registration window** | AOI bounding box + buffer, read from Sentinel-2 for matching. Larger than the AOI. |
| **Synthetic reference** | Drone MS band degraded to Sentinel-like resolution on the Sentinel pixel grid. |
| **Shift (dx, dy)** | Metres to **add** to the Sentinel geotransform origin so that it matches the drone frame. See §6.6, the sign convention. |
| **Verified / not applied** | Final registration status. "Not applied" means the patch is published with ESA's original georeferencing and a reason code. |

---

## 3. High-level architecture

```
                 drone_alignment (existing, unchanged logic)
RGB.tif ─┐
         ├─► align_orthomosaics() ─► <ms>_aligned.tif + report.json   (PRODUCT 1)
MS.tif  ─┘                                   │
                                             ▼   (only if --sentinel and alignment succeeded)
┌───────────────────────── sentinel_patch (NEW package) ─────────────────────────┐
│ S1 footprint      valid-data polygon of RGB ∩ aligned MS                        │
│ S2 discovery      STAC search → per-AOI cloud scoring → pick scene             │
│ S3 windowed read  COG range reads of bands + SCL; DN → reflectance             │
│ S4 reference      drone NIR/Red → PSF blur → area-average onto S2 grid          │
│ S5 estimation     AROSICS COREG (translation) + independent masked NCC          │
│ S6 gates          texture, overlap, magnitude, agreement, cross-band holdout   │
│ S7 products       geotransform shift (no resampling) → clip → coverage → overlay│
│ S8 QA             preview PNG + sentinel_report.json + candidates.json          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼  (optional)
                    crop_health_sentinel.run_crop_health_report(bands/ , footprint)
```

### Design principles (inherited from the alignment engine)

1. **Fail closed, never fail silent.** Every rejection produces a reason code in the report. An unregistered patch is published as `registration_status: "not_applied"`, never passed off as aligned.
2. **Isolation.** `sentinel_patch` is a separate package. `drone_alignment` imports it lazily, only when the flag is set, so the core engine never needs network access or new dependencies.
3. **Never resample Sentinel radiometry for the analytic product.** The correction is a geotransform update. Resampling is allowed only in the explicitly labelled *overlay* product.
4. **Deterministic and offline-testable.** Scene providers sit behind an interface, and tests use a local-folder provider plus recorded STAC fixtures.
5. **Reuse, don't duplicate.** The output band folder follows `crop_health_sentinel`'s `paths` config, so its pipeline consumes our output directly.

---

## 4. Risks, failure points and mitigations

This is the most important section. Every row maps to a gate, a config value or a test.

| # | Risk / point of failure | Likelihood | Impact | Mitigation (where implemented) |
|---|---|---|---|---|
| R1 | **AOI too small** to estimate a shift (few S2 pixels) | High for small plots | Wrong shift | Register on a **buffered window** (AOI + `registration_buffer_m`, default 300 m) so boundaries, roads and neighbours contribute texture. Hard minimum of valid overlapping pixels (`min_overlap_px`, default 400). Below it → `AOI_TOO_SMALL`, not applied. (S3, S6-G2) |
| R2 | **Buffer extends beyond the drone stack.** The drone ref only covers the AOI, so the buffer adds nothing on the reference side. | Certain | Lower texture | The buffer helps only where the drone stack covers it. ODM orthos usually extend beyond the field, so the effective footprint is everything the drone saw, not just the field polygon. Footprint = **full valid stack**, not a user field polygon. (S1) **Consequence found in P1:** cloud in the buffer ring cannot affect matching (there is no reference there), so the cloud cut-off is measured inside the AOI only, and masking matters only for cloud over the field. |
| R3 | **Homogeneous scene** (one crop, no edges) → flat correlation surface | Medium | Random shift | Texture gate on the reference (gradient energy / std), NCC peak sharpness, and peak-to-second-peak ratio. Fail → `LOW_TEXTURE`. (S6-G1, G5) |
| R4 | **Temporal change** between flight and satellite pass (growth, harvest, irrigation/flooding, tillage) | High if Δt is large | Content mismatch → wrong or unstable peak | `max_date_gap_days` (default 5, hard cap 10). Rank candidates by Δt. Cross-band agreement gate. Report Δt prominently. (S2, S6-G4/G6) |
| R5 | **Clouds, haze or cloud shadow** over the AOI. Scene-level `eo:cloud_cover` is meaningless for a 5 ha AOI. SCL is known to miss cloud shadows. | High (monsoon season) | Corrupt pixels in matching and analytics | Score the **AOI masked fraction** from SCL classes {0,1,3,8,9,10,11} + nodata, with the cloud mask dilated by 2 px to catch SCL shadow omissions; optionally also MSK_CLDPRB > threshold. Apply the tiered cutoff in §6.2.1 (≤ 10 % masked = accept; otherwise move to the next date). Masked pixels are excluded from matching (AROSICS `mask_baddata_tgt`). (S2, S5) |
| R6 | **Radiometric offset (processing baseline ≥ 04.00, BOA_ADD_OFFSET = −1000)** applied wrongly → reflectance off by 0.1 | High (classic bug) | NDVI and health metrics silently wrong; matching unaffected | One function owns DN→reflectance. Use the STAC item's `raster:bands` scale/offset when present (Earth Search `sentinel-2-c1-l2a` declares scale 0.0001, offset −0.1). Otherwise derive from `s2:processing_baseline`. Never apply twice; reject "harmonized" collections unless configured. Unit-test both baselines. (S3) |
| R7 | **CRS / UTM zone mismatch.** The drone is in e.g. EPSG:32642, the S2 tile in 32643 (zone boundary at 72° E crosses Punjab). The AOI may also overlap two MGRS tiles. | Medium | Silent misplacement or a resampled analytic product | Always work in the **S2 tile's native CRS** for S3–S7, and reproject the *drone* side (ref) into it. If the AOI is fully covered by several tiles of the same datatake, prefer the tile in the same UTM zone as the drone CRS, then the tile containing the AOI fully. **Never mosaic** tiles for the analytic product; if no single tile contains the window → `AOI_SPLIT_ACROSS_TILES`, not applied (v1). (S2) |
| R8 | **Swath edge / nodata** inside the window | Low–medium | Partial coverage | Nodata (`0`) counts as masked in the AOI masked fraction (§6.2.1), so a swath edge over the AOI moves the search to the next date. (S2, S3) |
| R9 | **Spectral mismatch** between the drone camera bands (MicaSense, DJI M3M, P4M) and S2 bands; drone not radiometrically calibrated | Certain | Lower correlation, not a wrong shift | Match on **structure, not values**: z-score normalise both, use NCC (scale and offset invariant), and prefer NIR↔B8 (strongest vegetation contrast). Validate on Red↔B4 (independent band). Band mapping is configurable. (S4, S5) |
| R10 | **Scale/PSF mismatch.** Averaging drone pixels into 10 m boxes produces a sharper image than S2 actually sees (S2 MTF at Nyquist is only 0.15–0.3). | Certain | Peak bias near edges | Gaussian pre-blur (σ ≈ 5 m, derived in §6.4) before area-averaging onto the S2 grid. (S4) |
| R11 | **Wrong shift sign** (the most common co-registration bug, and one the project has already hit: see `agrilift_alignment/tests/test_direction.py`) | High without tests | Doubles the error instead of fixing it | One documented convention (§6.6), one function applies it, plus a synthetic known-shift round-trip test that must pass before anything else is merged. (S5, S7, tests) |
| R12 | **Implausibly large shift** (wrong peak, periodic structures) | Low | Large error | `max_shift_m` (default 25 m ≈ S2 CE95 7 m + drone absolute error budget + margin). Above it → `SHIFT_IMPLAUSIBLE`. (S6-G3) |
| R13 | **The drone frame itself is wrong in absolute terms.** Company practice is unconfirmed; most likely no GCPs, RTK unknown. | **Medium–high** | The S2 patch "inherits" the drone error, so a large measured shift is partly drone error | Not fixable here, but **detectable**: auto-read the positioning type from the raw drone images' DJI XMP tags (`RtkFlag` 50 = RTK fixed, 34/16 = float/single; §6.9). **`unknown` is treated as `gnss_only`** (conservative default). Report it. If `gnss_only`/`unknown` and the shift exceeds 15 m, add warning `LARGE_SHIFT_DRONE_FRAME_SUSPECT`. |
| R14 | **"Exact perimeter" is impossible at 10 m.** Boundary pixels straddle the polygon. | Certain | Mixed pixels bias statistics | Publish a **coverage-fraction band** (0–1, computed by supersampled rasterisation), and keep `all_touched` and `centre` masks. Downstream decides; `crop_health_sentinel` already has `rasterization` and `edge_erosion_pixels` settings. (S7) |
| R15 | **No acceptable scene** within the date window | High in monsoon | No product 2 | The expanding date search (§6.2) walks outward day by day up to 15 days, then falls back to the best *partial* scene (≤ 30 % masked, flagged). Only then status `NO_SCENE`, with `candidates.json` listing every scene and why it was rejected. Product 1 is unaffected and the CLI exit code is still 0 (configurable). |
| R16 | **Network / service failure, rate limits, catalogue changes, a free service disappearing** | Medium | Run aborts | Retries with exponential backoff, timeouts, GDAL HTTP env settings, and an on-disk cache keyed by item id. **An ordered provider chain behind one interface** (§6.2.2): Earth Search (anonymous) → Planetary Computer (anonymous, signed URLs) → CDSE (free account, later). A provider failure moves on to the next provider, not to an abort. (S2, S3) |
| R17 | **Flight date unknown** (an ODM orthophoto carries no capture date) | Certain | Cannot pick a scene | Date-source chain (§6.9): `--flight-date` → sidecar `flight.json` → ODM project → raw photo headers → path-name estimate, confirmed by the user (§6.9.1). None → Sentinel stage fails with `FLIGHT_DATE_UNKNOWN`; product 1 unaffected. |
| R18 | **Memory blow-up** when making the drone-grid overlay at native drone GSD (a 500 m field at 3 cm is 16 k × 16 k × 10 bands float32 ≈ 10 GB) | High if naive | OOM (exit 137 in Docker) | Overlay at `overlay_gsd_m` (default 1.0 m). Build the synthetic reference via decimated reads (`out_shape` + `Resampling.average`), never full-resolution arrays. (S4, S7) |
| R19 | **20 m bands treated as 10 m** | Medium | False precision | Keep a separate 20 m product. The `bands/` folder for `crop_health_sentinel` holds native files (its reader resamples deliberately). (S7) |
| R20 | **Sub-pixel shift moves the grid off the MGRS 10 m lattice** | Certain | Surprises GIS users ("pixels not on round coordinates") | Expected behaviour. Document it in the report (`grid_snapped: false`). Optional `align_grids: true` resamples (cubic) onto the snapped grid, but the output is then labelled `resampled: true`. |

---

## 5. Inputs, configuration and outputs

### 5.1 Inputs

| Input | Source | Required |
|---|---|---|
| `AlignmentResult` / aligned MS path | `drone_alignment` | yes |
| RGB reference path | CLI arg | yes |
| `flight_date` (ISO date, local date of capture) | Date-source chain (§6.9): `--flight-date` → `flight.json` → ODM project → raw photo headers → path-name estimate | yes, from any source |
| Raw drone images folder (JPG/TIF with EXIF/XMP), headers only | `--images-dir` | no, but recommended: gives flight date **and** positioning type |
| ODM project folder | `--odm-project` | no |
| MS band mapping (`nir`, `red`, `green`, `red_edge` → 1-indexed band numbers) | config | yes (defaults from `AlignmentConfig.ms_red_band_index` etc.) |
| `drone_positioning` | auto from `--images-dir` XMP; CLI override | no (default `unknown`, treated as `gnss_only`) |
| Sentinel config YAML | `--sentinel-config` | no |

### 5.2 Configuration (`sentinel_patch/config/schema.py`, pydantic v2, same style as `drone_alignment/config/schema.py`)

```yaml
sentinel_patch:
  enabled: false                       # master switch; CLI --sentinel sets true
  # Ordered provider chain (§6.2.2). The first provider that returns an acceptable scene wins;
  # a provider that errors or finds nothing hands over to the next one. All free today.
  providers:
    - name: earth_search               # AWS, anonymous, no account
      stac_url: https://earth-search.aws.element84.com/v1
      collection: sentinel-2-c1-l2a
    - name: planetary_computer         # Azure, anonymous, URLs signed via the planetary-computer package
      stac_url: https://planetarycomputer.microsoft.com/api/stac/v1
      collection: sentinel-2-l2a
      enabled: true
    - name: cdse                       # Copernicus Data Space, free account; credentials from env vars only
      enabled: false                   # turn on when an account exists
  local_folder: null                   # if set, overrides the chain (offline / manually downloaded scenes)

  discovery:
    # Expanding date search (§6.2.1): look at day 0, then ±1, ±2 … days, nearest first.
    date_search_direction: both        # both | after_only | before_only
    preferred_gap_days: 3              # ≤ this → "same_period"
    acceptable_gap_days: 7             # ≤ this → "near"
    max_date_gap_days: 15              # ≤ this → "stale" (warning). Hard validation cap: 30
    prefilter_max_scene_cloud: 90      # eo:cloud_cover, only a cheap prefilter
    # Patch cut-off (§6.2.1): fraction of the AOI that is cloud / shadow / nodata after dilation.
    max_aoi_masked_fraction: 0.10      # ≤ 10 % masked → accept; otherwise try the next date
    fallback_max_aoi_masked_fraction: 0.30   # if NO date passes, best partial scene ≤ 30 % (flagged "partial")
    min_clear_pixels: 100              # absolute floor for tiny AOIs (≈ 1 ha of clear 10 m pixels)
    scl_reject_classes: [0, 1, 3, 8, 9, 10, 11]
    cloud_mask_dilation_px: 2
    cloud_probability_threshold: 40    # MSK_CLDPRB / 'cloud' asset, if available
    max_candidates_evaluated: 20

  io:
    bands_10m: [B02, B03, B04, B08]
    bands_20m: [B05, B06, B07, B8A, B11, B12]
    aux: [SCL]
    registration_buffer_m: 300
    http_timeout_s: 60
    http_retries: 4
    cache_dir: null                    # default: <output>/sentinel/.cache

  reference:
    match_band: {drone: nir, sentinel: B08}
    holdout_band: {drone: red, sentinel: B04}
    psf_sigma_m: 5.0                   # see §6.4
    intermediate_gsd_m: 1.0
    min_pixel_valid_fraction: 0.95     # drone coverage needed for a 10 m ref pixel

  registration:
    max_shift_m: 25.0
    arosics_window_px: auto            # largest power of 2 ≤ min(window dims); min 32
    ncc_search_px: 4                   # integer search radius in S2 pixels; must exceed max_shift_m / 10 by >= 1 (P1 fix: 3 could never reach 25 m)
    arosics_mode: advisory             # off | advisory | veto (P2: advisory, see §6.5 Estimator A)
    arosics_min_reliability: 60        # AROSICS' own reliability (%); below it its result is not an opinion
    agreement_tolerance_px: 1.0        # |AROSICS - NCC| above this counts as disagreement (P2: was 0.30)
    min_overlap_px: 400
    min_texture_score: 0.05            # coefficient of variation of the reference band, see §6.5 G1 (P1: metric and default changed)
    min_ncc_peak: 0.50
    min_peak_ratio: 1.15
    min_holdout_gain: 0.02             # Δ correlation on holdout band
    quadrant_check_min_px: 48          # run G7 only if the valid overlap spans ≥ 48 px each way (P2: was 128)
    quadrant_tolerance_px: 0.5
    on_reject: publish_unshifted       # publish_unshifted | skip

  products:
    write_20m: true
    write_overlay: true
    overlay_gsd_m: 1.0
    align_grids: false
    write_crop_health_bands: true      # writes bands/B3.tif B4.tif B5.tif B8.tif SCL.tif
    compression: deflate
```

### 5.3 Output layout

```
<output_dir>/
  <ms>_aligned.tif                       # product 1 (unchanged)
  <ms>_alignment_report.json             # gains a "sentinel_patch" summary block
  sentinel/
    footprint.geojson                    # perimeter: ONE WGS84 feature (stack-CRS polygon as WKT in properties)
    S2_<YYYYMMDD>_<tile>_10m.tif         # B02,B03,B04,B08 reflectance float32, shifted, clipped
    S2_<YYYYMMDD>_<tile>_20m.tif         # B05..B12 reflectance float32, shifted, clipped
    S2_<YYYYMMDD>_<tile>_scl.tif         # SCL uint8, shifted, clipped (10 m version from 20 m via nearest is NOT made)
    S2_<YYYYMMDD>_<tile>_coverage.tif    # band1 coverage fraction (float32), band2 all_touched mask, band3 centre mask
    S2_<YYYYMMDD>_<tile>_overlay.tif     # drone CRS, overlay_gsd_m, nearest; VISUALISATION ONLY
    bands/  B3.tif B4.tif B5.tif B8.tif SCL.tif   # crop_health_sentinel input layout
    sentinel_preview.png
    sentinel_report.json
    candidates.json
```

Each GeoTIFF gets tags: `S2_ITEM_ID`, `S2_DATETIME`, `REGISTRATION_STATUS`, `SHIFT_X_M`, `SHIFT_Y_M`, `REFLECTANCE_SCALE=1`, `RESAMPLED=false|true`, `PRODUCT_ROLE=analytic|overlay`.

---

## 6. Stage-by-stage specification

### 6.1 S1 — Footprint extraction (`sentinel_patch/footprint.py`)

**Goal:** a valid polygon of where the aligned stack has data in both rasters.

1. Open aligned MS and RGB with rasterio. Read `dataset_mask()` at a decimated `out_shape` so that the long side is ≤ 2048 px (reuse the pattern from `coarse_align`). The MS valid mask must also exclude pixels equal to `cfg.warp.nodata_value`.
2. Reproject the RGB mask onto the MS mask grid if the grids differ (nearest). Combined mask = RGB ∧ MS.
3. Morphological clean-up: binary opening (3 px) to kill speckle, then fill holes smaller than 1 % of the area.
4. Vectorise with `rasterio.features.shapes` using the decimated transform. Union, then `buffer(0)`. Keep the **largest polygon** (warn if others are > 5 % of the area). Simplify with tolerance = 1 decimated pixel.
5. Validate: non-empty, valid, area > 0. Area < 0.25 ha → still continue (S6 decides), but log a warning.
6. Write `footprint.geojson` as **one** WGS84 feature (RFC 7946). `crop_health_sentinel` rejects any file without exactly one feature, so the stack-CRS polygon travels as WKT in `properties.stack_crs_wkt` (P1 change from the original two-feature design).

**Output dataclass**
```python
@dataclass(frozen=True)
class StackFootprint:
    polygon_stack_crs: BaseGeometry
    stack_crs: CRS
    polygon_wgs84: BaseGeometry
    area_m2: float
```

### 6.2 S2 — Scene discovery and selection (`sentinel_patch/discovery/`)

**Interface**
```python
class SceneProvider(Protocol):
    def search(self, aoi_wgs84: BaseGeometry, start: datetime, end: datetime) -> list[SceneCandidate]: ...
    def open_asset(self, candidate: SceneCandidate, asset_key: str) -> str: ...  # returns GDAL-openable href
```

`SceneCandidate` holds: `item_id, datetime_utc, mgrs_tile, epsg, processing_baseline, scene_cloud_cover, asset_hrefs, reflectance_scale_offset{band: (scale, offset)}, footprint_wgs84, relative_orbit`.

Implementations (each one is a single file under `discovery/providers/`; details in §6.2.2):
- `EarthSearchProvider` uses `pystac-client` against its `stac_url`/`collection`, with the `intersects` filter set to the **buffered** AOI in WGS84 and the `datetime` filter set to `[flight_date − max_gap, flight_date + max_gap]` (one query; the date ladder is applied locally).
- `PlanetaryComputerProvider` is the same, plus `planetary_computer.sign(item)` before any asset is opened. Its `sentinel-2-l2a` is **not harmonized**, so radiometry must come from `s2:processing_baseline` (R6).
- `CdseProvider` is a stub in v1: interface, config and credential loading from `CDSE_S3_ACCESS_KEY`/`CDSE_S3_SECRET_KEY` env vars, and `NotConfiguredError` when they are missing.
- `LocalFolderProvider` reads a folder in the `crop_health_sentinel` `paths` layout plus a small `scene.json` (`datetime`, `processing_baseline`, `item_id`). Used for tests, offline runs and manually downloaded scenes.

**Selection algorithm (`selection.py`)** — provider-agnostic, works only on `SceneCandidate`s
1. Prefilter: drop items with `eo:cloud_cover > prefilter_max_scene_cloud`.
2. Drop items whose footprint does not fully contain the buffered AOI (R7, R8).
3. Deduplicate: the same ESA granule can come from several providers and several MGRS tiles. Key on `(datatake datetime, relative orbit)`, keep one: same UTM zone as the stack → fewest nodata pixels → provider order.
4. Build the **date ladder** (§6.2.1) and walk it nearest-first.
5. For each candidate: windowed-read SCL (and `cloud` if present) over the buffered AOI, build the reject mask (SCL classes + nodata + cloud prob), dilate the cloud/shadow part by `cloud_mask_dilation_px`, and compute `aoi_masked_fraction` inside the **AOI** (not the buffer).
6. Record **every** candidate with its gap, tier, masked fraction and rejection reason in `candidates.json`.

Δt is computed using the **UTC date of the S2 acquisition vs. the local flight date**. Store both.

#### 6.2.1 Date ladder and the patch cut-off

**Date ladder (your rule: "as close as possible, then move outward").** Days are visited in the order 0, −1, +1, −2, +2, … up to `max_date_gap_days`. On each rung, all candidates of that date are scored. If two candidates have the same |Δt|, the one with the lower masked fraction wins; if that also ties, the after-flight scene wins (an arbitrary but fixed convention, so runs are reproducible). `date_search_direction` can restrict the search to one side. Sentinel-2 revisits each place every few days (three satellites have been flying since March 2025, when S2A got a temporary extension after S2C took over), so most rungs are empty. That is expected.

| Tier | Gap | Meaning | Report |
|---|---|---|---|
| `same_period` | ≤ 3 days | Crop practically unchanged | no warning |
| `near` | 4–7 days | Fine for registration and most analytics | `DATE_GAP_NEAR` info |
| `stale` | 8–15 days | Registration usually still OK on stable features; analytic comparison with the drone is risky in active growth | `DATE_GAP_STALE` warning |
| — | > 15 days | Not searched | — |

**Patch cut-off (how much of the patch may be missing).** There is no published standard for field-level clear-pixel cutoffs. Platforms report a field-clipped cloud percentage and let users filter, and a parcel-monitoring rule from the literature flags a parcel as "cloud covered" only at > 50 % masked, which is far too lax for comparing against drone data. We use a **masked-fraction** cutoff with three levels, chosen for these reasons:

1. **≤ 10 % masked → accept** and stop searching. Clouds rarely fall randomly; they mask one contiguous end of the field, so a higher masked fraction biases field statistics toward the visible part. SCL also misses cloud shadows (R5), so any cloud in the AOI means probably-undetected shadow nearby; the 2 px dilation covers ~20 m of that. 10 % keeps the bias small and still lets through scenes with a stray cloud corner.
2. **> 10 % → move to the next date rung.** This is your rule.
3. **No rung passes → fallback.** Take the scene with the lowest masked fraction if it is **≤ 30 %**, publish with `scene_quality: "partial"` and masked pixels set to NaN. 30 % masked (= 70 % clear) is used because `crop_health_sentinel` already uses 15 % clear as its "display only" floor and 40 % clear as "analytics ready"; we stay well above both so that a partial scene is still clearly usable, and its own gates make the final analytic call. Above 30 % → `NO_SCENE`.

Also: the AOI must have at least `min_clear_pixels` (100 ≈ 1 ha) clear 10 m pixels, whatever the fraction. For registration, clouds **outside** the AOI but inside the buffer only reduce the matching pixels; that is handled by `min_overlap_px` in §6.5, not here.

These numbers are **starting values**. Phase P5 (§10) records the masked fraction for every real run and re-checks the cutoffs.

#### 6.2.2 Provider modularity (free today, swappable later)

The rule: **everything outside `discovery/providers/` sees only `SceneCandidate` and GDAL-openable hrefs.** Nothing else in the package may import `pystac_client`, `planetary_computer` or any credentials.

```python
class SceneProvider(Protocol):
    name: str
    def search(self, aoi_wgs84: BaseGeometry, start: datetime, end: datetime) -> list[SceneCandidate]: ...
    def resolve_href(self, candidate: SceneCandidate, band: str) -> str: ...   # signs / builds /vsis3/ path
    def gdal_env(self) -> dict[str, str]: ...                                 # per-provider GDAL options
```

Each provider owns exactly three things:
1. **Asset map**: its asset keys → our band names (`B02 … B12, SCL, CLD`). Earth Search uses `red`, `nir`, `rededge1`…; Planetary Computer uses `B04`, `B08`, `B05`… Put each map as a constant in the provider's file.
2. **Radiometry policy**: returns `(scale, offset, source)` per band from `raster:bands` or `s2:processing_baseline`, and must state whether its collection is harmonized.
3. **Access**: anonymous (Earth Search: `AWS_NO_SIGN_REQUEST=YES`), signed URLs (Planetary Computer), or credentials (CDSE: env vars only, never config files).

Providers are registered in a dict (`PROVIDERS = {"earth_search": EarthSearchProvider, ...}`) and built from the `providers` list in config. `ProviderChain.search()` calls them in order and catches `ProviderUnavailableError` (network, 5xx, rate limit, not configured) to move on, recording each failure in `candidates.json`. The chain stops at the first provider that yields an **accepted** scene, so a cloudy result from one provider does not stop the next provider from being tried for the same dates (catalogues occasionally miss granules).

**Contract test suite**: `tests/providers/test_contract.py` is parametrised over all providers with recorded STAC responses (no network). Every provider must return candidates with all required fields, a correct band map and correct radiometry for a PB 03.xx and a PB ≥ 04.00 item. Adding a paid provider later (Sentinel Hub, Planet, UP42) means one new file + one recorded fixture + passing this suite; no other code changes.

### 6.3 S3 — Windowed read and radiometry (`sentinel_patch/io/`)

**Environment (set once in `io/gdal_env.py`, used as a `rasterio.Env` context):**
```
AWS_NO_SIGN_REQUEST=YES
GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.TIF,.tiff
GDAL_HTTP_MAX_RETRY=<http_retries>  GDAL_HTTP_RETRY_DELAY=2
GDAL_HTTP_TIMEOUT=<http_timeout_s>
VSI_CACHE=TRUE
```

**Steps**
1. Transform the stack footprint to the scene EPSG (`shapely.ops.transform` + `pyproj.Transformer`, `always_xy=True`). This is the AOI.
2. Window = `AOI.bounds` expanded by `registration_buffer_m`, snapped **outward** to whole 20 m pixels, so the 10 m and 20 m windows nest exactly.
3. Read each band with `rasterio.windows.from_bounds` → `src.read(1, window=…, boundless=False)`. **No WarpedVRT, no reprojection.** Keep the window transforms.
4. DN → reflectance in **one** function:
   ```python
   def dn_to_reflectance(dn: np.ndarray, scale: float, offset: float, nodata: int = 0) -> np.ndarray:
       out = dn.astype(np.float32) * scale + offset   # e.g. 0.0001, -0.1 for PB ≥ 04.00
       out[dn == nodata] = np.nan
       return out
   ```
   Scale/offset come from the item's `raster:bands`. If they are missing, the rule is: `offset = −0.1 if processing_baseline ≥ 4.0 else 0.0`, `scale = 1e-4`. If neither the metadata nor the baseline is available, raise `RadiometryUnknownError` (do not guess).
5. Cache raw window arrays as `.npz` + `meta.json` in `cache_dir`, keyed by `item_id/band/window`.

### 6.4 S4 — Synthetic Sentinel reference (`sentinel_patch/registration/reference.py`)

**Goal:** make the drone band look like what Sentinel-2 would see, on Sentinel's own pixel grid.

1. Target grid = the **10 m registration window grid** from S3 (S2 CRS).
2. Read the drone band (`nir` for matching, `red` for holdout) from the **aligned MS** with `rasterio.warp.reproject` into an *intermediate* grid: S2 CRS, `intermediate_gsd_m` (1 m), `Resampling.average`, aligned so that 10 intermediate pixels = 1 S2 pixel exactly. Also reproject the valid mask (average → fraction).
3. Gaussian blur the intermediate image with σ = `psf_sigma_m / intermediate_gsd_m` (normalised convolution: blur `value·mask` and `mask` separately, then divide, so the nodata edges do not darken).
   - *Derivation:* a Gaussian PSF has MTF(f) = exp(−2π²σ²f²). At Nyquist f = 1/(2·10 m) = 0.05 m⁻¹ and MTF ≈ 0.25 (ESA quotes 0.15–0.3), σ = √(−ln 0.25 / (2π²·0.0025)) ≈ 5.3 m. Default 5.0 m.
4. Block-average 10×10 → the 10 m grid. Ref pixels whose drone-valid fraction is below `min_pixel_valid_fraction` become NaN.
5. Output `SyntheticReference(match: ndarray, holdout: ndarray, valid: ndarray, transform, crs)`.

Memory: steps 2–4 must run per tile of the intermediate grid (e.g. 2048² blocks with a 3σ halo) if the window exceeds `max_in_memory_px` (default 25 M). For typical fields (≤ 2 km) one pass is fine: 2000² × 4 B = 16 MB.

### 6.5 S5 + S6 — Shift estimation and gates (`registration/estimators.py`, `registration/gates.py`)

Both estimators run on z-score-normalised images, masked by `ref_valid ∧ s2_clear`.

**Estimator A: AROSICS `COREG`** (`registration/arosics_estimator.py`; implemented in P2 as an *advisory* cross-check)
- Both arrays are written to temporary GeoTIFFs (nodata −9999, cloud and no-data blanked), as `drone_alignment/alignment/arosics_matcher.py` does.
- `ws` = largest power of 2 ≤ the smaller side of the **valid-overlap bounding box**, 32 to 256 (smaller → `skipped`, `AROSICS_WINDOW_TOO_SMALL`). `wp` = centre of that box. `max_shift` = `ceil(max_shift_m / pixel) + 1` pixels.
- **Sign convention verified experimentally:** `x_shift_map` / `y_shift_map` are already the vector to add to the target origin, i.e. our `shift_m`.
- Outcomes: `ok` (success and reliability ≥ `arosics_min_reliability`, 60), `inconclusive` (any AROSICS error, no shift, no or low reliability; warning `AROSICS_INCONCLUSIVE`), `skipped` (mode off, not installed, window too small). Anything AROSICS throws is contained: it is an optional third-party cross-check.
- **Measured limits (synthetic, so re-check on real data in P5).** AROSICS needs sharp edges; on a smooth, edgeless scene it can report 80 % reliability and be confidently wrong. With identical imagery on both sides (160 trials) its error had a median of 0.17 px; on the realistic path (blurred drone reference against sharper Sentinel pixels, 40 trials) the median was **0.72 px against 0.03 px for NCC**, and 29 % of its conclusive answers differed from the correct NCC answer by more than a pixel. A veto would therefore refuse many good registrations, so **the default is `arosics_mode: advisory`**: the disagreement is reported and warned about (`AROSICS_DISAGREES`) but never rejects. `veto` restores the original blueprint behaviour; `off` skips the run.

**Estimator B: masked normalised cross-correlation (our own, ~80 lines)**
- For integer shifts (i, j) ∈ [−N, N]² (N = `ncc_search_px`): masked Pearson correlation between ref and S2 shifted by (i, j) over the overlap. This gives a (2N+1)² surface.
- Sub-pixel refinement: fit a 2-D quadratic to the 3×3 neighbourhood of the maximum. Reject if the peak is on the border of the search area (`PEAK_AT_SEARCH_BORDER`).
- Report: `peak_value`, `second_peak_value` (max outside the 3×3 neighbourhood), `peak_ratio`, sub-pixel shift.

**Gates (all must pass; evaluated in this order, first failure wins the reason code)**

| Gate | Check | Reason code |
|---|---|---|
| G1 texture | **coefficient of variation** (std / mean) of the reference band over valid pixels ≥ `min_texture_score` (0.05). *P1 change:* the original "gradient of the z-scored image" amplifies noise on a uniform field and would pass it; drone bands are uncalibrated, so only a scale-free measure makes sense. Starting value, tune in P5. | `LOW_TEXTURE` |
| G2 overlap | valid overlapping pixels ≥ `min_overlap_px`. *P1 change:* evaluated **before** G1 (texture on a handful of pixels means nothing). Separately, a stack of fewer than `min_clear_pixels` (100) Sentinel pixels, about 1 ha, stops the run **before any scene is fetched** with status `not_applied`, reason `AOI_TOO_SMALL`, no products. | `AOI_TOO_SMALL` |
| G3 magnitude | ‖shift‖ ≤ `max_shift_m` | `SHIFT_IMPLAUSIBLE` |
| G4 agreement | ‖shift_A − shift_B‖ ≤ `agreement_tolerance_px` (**1.0 px**, was 0.30) when A is conclusive; otherwise not evaluated (a pass). *P2 change:* **advisory by default** (see Estimator A): the gate is recorded with `advisory: true` and its failure only adds the warning `AROSICS_DISAGREES`; with `arosics_mode: veto` it rejects. | `ESTIMATOR_DISAGREEMENT` |
| G5 peak quality | NCC `peak_value ≥ min_ncc_peak` and `peak_ratio ≥ min_peak_ratio` (AROSICS' own success and reliability are no longer part of this gate; they decide only whether A counts as an opinion). *P1 change:* `peak_ratio` = peak ÷ the best **distinct local maximum** of the correlation surface (1000 if there is none). The original "highest value outside the 3×3 neighbourhood" is just the shoulder of one smooth peak and would reject every genuine match on smooth imagery; a distinct second maximum is what signals ambiguity (crop rows). | `WEAK_PEAK` |
| G6 holdout band | apply the shift to the S2 **B04** window; correlation with ref **red** must improve by ≥ `min_holdout_gain` over unshifted, *unless* the shift is < 0.2 px (then just "not worse") | `HOLDOUT_NO_GAIN` |
| G7 quadrant consistency | *P2:* the bounding box of the valid overlap is split into 2×2 quarters (only if it spans ≥ `quadrant_check_min_px` = **48** pixels each way, was 128: a 128 px minimum is 1.3 km and would skip almost every field). NCC is run per quarter that has ≥ `min_overlap_px` overlap and passes the texture gate; each must be within `quadrant_tolerance_px` (0.5) of the global shift. A quarter whose peak hits the search border counts as inconsistent. Fewer than 2 usable quarters → not evaluated (a pass, recorded). | `NON_TRANSLATIONAL` |

Final shift = **estimator B**, always. *P2 change:* the original "mean of A and B" is dropped because B is about 20× more accurate in the tests; A is recorded for the reviewer.

### 6.6 Sign convention (must read)

> **`shift_m = (dx, dy)` is the vector to ADD to the Sentinel geotransform origin (`c`, `f` of the affine) so that Sentinel content lands on the matching drone content.**
> `new_transform = Affine(a, b, c + dx, d, e, f + dy)`.
> `dy` is in map units (northing). Positive `dy` moves the image **north**.

The NCC estimator works in pixel rows/cols, so the conversion is `dx = +col_shift · a`, `dy = +row_shift · e` (note `e < 0`). AROSICS reports `x_shift_map`/`y_shift_map` as the shift **of the target to apply**, so verify with the synthetic test, not by reading the docs. The test `test_known_shift_roundtrip` (§9) is the single source of truth for both estimators.

### 6.7 S7 — Products (`sentinel_patch/products.py`)

1. **Status decision.** All gates pass → `registration_status = "verified"`, shift applied. Any gate fails and `on_reject = publish_unshifted` → `"not_applied"`, shift = (0, 0), reason recorded. With `on_reject = skip` → no rasters, report only.
2. **Shift.** Update the geotransform only (§6.6). Pixel arrays are untouched.
3. **Clip.** Compute the window of the shifted grid covering `AOI.bounds` with `all_touched` semantics (pad 1 px). Crop the arrays. Pixels whose coverage fraction is 0 → NaN (reflectance) / 0 (SCL). This is the "exact perimeter" patch.
4. **Coverage band.** Rasterise the AOI polygon at 10× supersampling on the shifted grid → block mean = fraction (float32 0–1). Also write the `all_touched` mask and the centre-inside mask.
5. **20 m product.** Same shift (metres) applied to the 20 m window; clip identically.
6. **`bands/` folder** for `crop_health_sentinel`: single-band GeoTIFFs named by its `paths` config: `B3.tif` (B03), `B4.tif`, `B5.tif` (20 m native), `B8.tif`, `SCL.tif` (20 m native), plus `MSK_CLDPRB.tif` if available. Values must be **DN-scale** (reflectance × 10000) because that pipeline applies `io.reflectance_scale: 10000`. Write them as float32 DN with the offset already removed, and **document this in the config comment**; otherwise its NDVI double-counts the offset. Add a contract test (§9).
7. **Overlay.** `rasterio.warp.reproject` of the shifted 10 m product into the stack CRS at `overlay_gsd_m`, grid origin snapped to the aligned-MS origin, `Resampling.nearest` (keeps the honest 10 m blocks), clipped to the footprint. Tag `PRODUCT_ROLE=overlay`.
8. Publish atomically: write to `.partial.tif`, then `os.replace` (same pattern as `_publish_global_context`).

### 6.8 S8 — QA preview and report (`sentinel_patch/quality/`)

**Preview PNG (2×2 panels, matplotlib, reuse colour conventions from `drone_alignment/quality/visualization.py`):**
1. Drone RGB (decimated) with the footprint outline and the shift vector arrow (scaled ×10, labelled in metres).
2. Synthetic reference NIR (10 m).
3. S2 B08 before shift, footprint outline.
4. S2 B08 after shift, footprint outline + checkerboard blend with the reference.
Title: scene id, date, Δt, status, shift in metres.

**`sentinel_report.json` schema (v1)**
```json
{
  "schema_version": "sentinel_patch/1.0",
  "status": "verified | not_applied | no_scene | error",
  "reason_code": null,
  "flight_date": "2026-10-10",
  "scene": {"item_id": "...", "datetime_utc": "...", "date_gap_days": 2, "mgrs_tile": "42RXV",
            "epsg": 32642, "processing_baseline": "05.11", "relative_orbit": 120,
            "scene_cloud_cover": 3.1, "aoi_clear_fraction": 0.98,
            "radiometry": {"scale": 0.0001, "offset": -0.1, "source": "raster:bands"}},
  "footprint": {"area_m2": 52341.2, "stack_crs": "EPSG:32642"},
  "registration": {
    "shift_m": [3.4, -6.1], "shift_px": [0.34, -0.61],
    "estimators": {"arosics": {...}, "ncc": {"peak": 0.81, "peak_ratio": 1.6, "shift_px": [...]}},
    "gates": [{"name": "G1_texture", "value": 0.07, "threshold": 0.02, "passed": true}],
    "holdout": {"band": "B04", "corr_before": 0.55, "corr_after": 0.71},
    "drone_positioning": "unknown",
    "warnings": []
  },
  "products": {"analytic_10m": "...", "analytic_20m": "...", "coverage": "...", "overlay": "...",
               "crop_health_bands_dir": "...", "resampled": false, "grid_snapped": false},
  "versions": {"sentinel_patch": "1.0.0", "arosics": "...", "rasterio": "...", "gdal": "..."}
}
```
Also add a short `"sentinel_patch": {status, reason_code, shift_m, report_path}` block to the main alignment report via an optional kwarg on `write_alignment_report`.

The report gains these fields from rev 2: `scene.date_tier` (`same_period|near|stale`), `scene.scene_quality` (`clean|partial`), `scene.aoi_masked_fraction`, `scene.provider`, `drone_metadata.{flight_date_source, positioning, positioning_source, rtk_fixed_fraction}`.

### 6.9 Drone metadata detection (`sentinel_patch/drone_metadata.py`)

> **Scope note: this does not change the alignment engine.** Alignment still runs on the stitched orthophotos only. The raw photos are an **optional, metadata-only** input: we read their EXIF/XMP headers (a few KB per file, ≤ 200 files sampled) and never their pixels. No raw photos → nothing breaks; the user just types the date.

**Flight-date sources, tried in order (first hit wins; the source is recorded in the report):**

| # | Source | Effort for the user | Notes |
|---|---|---|---|
| 1 | `--flight-date YYYY-MM-DD` | Type it once | Always wins. The **default, expected path** today. |
| 2 | Sidecar `flight.json` next to the RGB orthophoto (`{"flight_date": "...", "positioning": "..."}`) | Write it once per dataset, or have the delivery step write it | Lets batch runs work without flags. Same pattern as the existing `Results/` folders. |
| 3 | ODM project folder (`--odm-project PATH`): `images.json` if present (per-image capture time), plus `gcp_list.txt` for positioning | None if the ODM folder is kept | **Verify** the `images.json` field names against the company's ODM version in P4 before relying on it. |
| 4 | `--images-dir PATH`: raw photo headers (below) | None if the photos are kept | Also the only source for the RTK flag. |
| 5 | **Path-name estimate** (§6.9.1): a date parsed from the orthophoto file names and their parent folders | Confirm once (Enter) | Last resort. Never used without confirmation unless it is unambiguous **and** the config allows it. |
| — | None found | — | Fail the **Sentinel stage only** with `FLIGHT_DATE_UNKNOWN` and a message listing these options. Product 1 is unaffected. |

Each source is a small function `(inputs) -> DroneMetadata | None`, chained like the scene providers, so a new source (e.g. a company flight-log API) is one more function.

#### 6.9.1 Path-name date estimation (`sentinel_patch/date_from_path.py`)

**What is scanned:** the file names of the RGB and MS orthophotos, plus up to `path_scan_depth` (default 3) parent folder names each. Example: `D:\Flights\Okara\2026-10-10_Field7\odm_orthophoto\rgb.tif` → scans `rgb`, `odm_orthophoto`, `2026-10-10_Field7`, `Okara`.

**Patterns, from strongest to weakest** (one regex table, each with a confidence weight):

| Pattern | Example | Interpretation | Confidence |
|---|---|---|---|
| ISO date | `2026-10-10`, `2026_10_10`, `2026.10.10` | Y-M-D | high |
| Compact 8-digit, year first | `20261010` | Y-M-D (must be a valid date) | high |
| Day + month name + year | `10Oct2026`, `10-oct-26`, `Oct 10 2026` | unambiguous | high |
| Numeric with 4-digit year last | `10-03-2026` | **ambiguous** D-M vs M-D, unless one part > 12 | medium / ambiguous |
| Compact 6-digit | `101026`, `261010` | too ambiguous | **ignored** |
| Month + year only | `Oct2026` | no day | not used as a date; shown as a hint |

Numbers must sit at token boundaries (`_ - . space`), so `IMG_20261010_0042` matches but `DJI_0042` or `v2.10` do not. Dates must be valid calendar dates.

**Ambiguity rule:** `date_order` in config (`dmy` default, for the company's Pakistan context; or `mdy`). An ambiguous numeric date is interpreted with it, but stays **ambiguous** (both readings are recorded) unless one part is > 12.

**Plausibility filters (drop a candidate if any fails):**
- Not before 2015-06-23 (Sentinel-2A launch) and not in the future.
- Not after the orthophoto's own GeoTIFF `TIFFTAG_DATETIME`, which is the processing time when present. A flight cannot happen after its orthophoto was made. Use this tag **only as an upper bound, never as the flight date**. File system modification times are never used (copying files changes them).

**Picking a result:**
- One distinct plausible date (possibly found several times) → that date. Confidence = the strongest pattern that found it, raised one level if found in both the RGB and MS paths.
- Several different dates → candidates ranked by: found in both RGB and MS paths → closer to the file (file name > parent > grandparent) → pattern strength. Status `MULTIPLE_DATES`.

**Using the result (never silent):**
- **Interactive** (CLI / PowerShell launcher): prompt
  `Flight date not found in metadata. Estimated 2026-10-10 from folder "2026-10-10_Field7". Use it? [Y]es / [n]o / or type a date:`
  With several candidates, list them numbered. With an ambiguous one, show both readings (`10 March 2026 or 3 October 2026?`).
- **Non-interactive** (Docker, batch): use the estimate only if `accept_path_date_estimate: true` in config **and** the result is unambiguous with high confidence. Otherwise fail with `FLIGHT_DATE_UNKNOWN`, and include the candidates in the error message so the user can pass `--flight-date`.
- **Remember the answer:** a confirmed or typed date is written to the sidecar `flight.json` (source #2) next to the RGB orthophoto, with `"flight_date_source": "path_estimate_confirmed"` or `"user_entered"`. The next run on the same data finds it at step 2 and never asks again. If the folder is read-only, log it and continue.

**Report:** `drone_metadata.flight_date_source = "path_estimate"`, plus `path_estimate: {matched_text, path_component, pattern, confidence, ambiguous_readings, confirmed_by_user}`. A run on an estimated date also gets the warning `FLIGHT_DATE_ESTIMATED`, so reviewers know the date tier is only as good as the folder name.

Config (under `sentinel_patch.drone_metadata`):
```yaml
drone_metadata:
  path_scan_depth: 3
  date_order: dmy                    # dmy | mdy, used only for ambiguous numeric dates
  accept_path_date_estimate: false   # non-interactive runs: allow unambiguous high-confidence estimates
  write_sidecar: true                # save confirmed / entered dates to flight.json
```

**Reading raw photo headers (`--images-dir`):**
1. List the image files (`*.jpg, *.jpeg, *.tif`), case-insensitive. Sample up to 200, spread evenly.
2. **Flight date:** EXIF `DateTimeOriginal` per image → median → date. If the images span more than one day, warn `MULTI_DAY_FLIGHT` and use the median.
3. **Positioning:** read the DJI XMP block (`drone-dji:RtkFlag`, `RtkStdLon`, `RtkStdLat`). Read the raw XMP packet from the file bytes and parse it with a regex/`xml.etree`; this needs no new dependency.
   - `RtkFlag == 50` → RTK fixed. If ≥ 90 % of sampled images are fixed → `rtk`.
   - Flag present but mostly 16/34 (single/float) → `gnss_only` with warning `RTK_NOT_FIXED`.
   - No DJI RTK tags (other brands, no RTK module) → `unknown`.
   - Note: the flag shows RTK was active, which is strong evidence but not a guarantee of an accurate camera position (Pix4D community discussion). Report it as evidence, not proof.
4. GCP usage cannot be seen in the images. If the ODM project folder is given and contains `gcp_list.txt`, set `gcp`. Otherwise the CLI option `--drone-positioning` is the only source.
5. Result: `DroneMetadata(flight_date, flight_date_source, positioning, positioning_source, rtk_fixed_fraction)`.

`positioning` changes behaviour in exactly one place: the `LARGE_SHIFT_DRONE_FRAME_SUSPECT` warning threshold (R13). It never changes gates or the shift itself.

---

## 7. Integration points in the existing code

| File | Change |
|---|---|
| `drone_alignment/cli.py` | New options: `--sentinel/--no-sentinel` (default off), `--flight-date` (ISO), `--images-dir PATH` and `--odm-project PATH` (optional, metadata only; §6.9). With `--sentinel`, the flight date comes from the first source in §6.9 that has it; if none does, only the Sentinel stage fails. Also `--sentinel-config PATH`, `--drone-positioning [rtk|ppk|gcp|gnss_only|unknown]` (override). After any successful `result = …` branch: `if sentinel: from sentinel_patch import run_sentinel_patch` (lazy import) → call → echo status and shift. Wrap it in its own `try/except` so a Sentinel error prints `[WARN]` and **does not** `Abort` (exit code 0, unless `--sentinel-strict`). |
| `drone_alignment/quality/report.py` | Optional `sentinel_patch: dict \| None = None` kwarg in `write_alignment_report`. Rewrite the report after the sentinel stage via a small `update_report_json(path, key, value)` helper (read, update, atomic write). |
| `drone_alignment/pipeline.py` | **No change.** The sentinel stage consumes `AlignmentResult` + paths only. |
| `pyproject.toml` | Add package `sentinel_patch`. New extra `sentinel = ["pystac-client>=0.8,<1", "planetary-computer>=1,<2"]`. AROSICS is already in the `arosics` extra; the sentinel stage requires it for estimator A but must run with estimator B alone if it is missing (log it, mark A skipped). |
| `Dockerfile` / `docker/pyproject.toml` | Copy `sentinel_patch/`, install the `sentinel` extra. Network egress is needed at runtime (document in DOCKER_README). |
| `run_alignment.ps1`, `scripts/run-alignment.ps1` | Prompt "Also fetch & align Sentinel-2 patch? [y/N]" → flight date prompt. |
| `crop_health_sentinel` | **No change.** Our `bands/` output + `footprint.geojson` (WGS84 feature) feed `run_crop_health_report(raw_data_dir=bands_dir, crop_geometry=footprint_wgs84, scene_meta=…)`. Optional flag `--sentinel-health` chains it. |

**Public API**
```python
def run_sentinel_patch(
    aligned_ms_path: Path,
    rgb_path: Path,
    output_dir: Path,
    flight_date: date | None = None,                  # None → taken from images_dir (§6.9)
    images_dir: Path | None = None,
    config: SentinelPatchConfig | None = None,
    ms_band_map: Mapping[str, int] | None = None,   # {"nir": 5, "red": 3, ...}
    log: logging.Logger | None = None,
) -> SentinelPatchResult: ...
```
`SentinelPatchResult(status, reason_code, shift_m, report_path, product_paths: dict[str, Path], scene_id)`.

---

## 8. Package layout

```
sentinel_patch/
  __init__.py                 # exports run_sentinel_patch, SentinelPatchResult, SentinelPatchConfig
  version.py
  errors.py                   # SentinelPatchError, NoSceneError, RadiometryUnknownError, RegistrationRejected(reason_code, details)
  config/schema.py            # pydantic models from §5.2
  config/default_sentinel_patch.yaml
  footprint.py                # S1
  drone_metadata.py           # §6.9 date-source chain + positioning from raw images
  date_from_path.py           # §6.9.1 path-name date estimation (pure functions, no I/O except TIFF tag read)
  discovery/
    provider.py               # SceneProvider protocol, SceneCandidate, ProviderUnavailableError
    chain.py                  # ProviderChain + PROVIDERS registry
    providers/
      earth_search.py         # asset map + radiometry policy + access, nothing else
      planetary_computer.py
      cdse.py                 # stub in v1
      local_folder.py
    date_ladder.py            # §6.2.1 rung order + tiers
    selection.py              # dedupe, AOI masked fraction, cut-off + fallback
  io/
    gdal_env.py
    window_reader.py          # S3
    radiometry.py             # dn_to_reflectance (THE only place)
    writer.py                 # atomic GeoTIFF writing + tags
  registration/
    reference.py              # S4
    estimators.py             # AROSICS wrapper + masked NCC
    gates.py                  # G1–G7, each a pure function returning GateResult
    shift.py                  # apply_shift(transform, dx, dy)  (THE only place; §6.6)
  products.py                 # S7
  quality/preview.py
  quality/report.py
  pipeline.py                 # run_sentinel_patch orchestrator
  tests/
    conftest.py               # synthetic S2 + drone fixtures, fake provider
    test_shift_convention.py
    test_radiometry.py
    test_footprint.py
    test_drone_metadata.py
    test_date_ladder.py
    test_selection.py
    providers/test_contract.py  # parametrised over every provider, recorded fixtures
    providers/test_chain.py
    test_reference.py
    test_estimators.py
    test_gates.py
    test_products.py
    test_crop_health_contract.py
    test_pipeline_offline.py
    test_earth_search_live.py # @pytest.mark.network, skipped by default
```

Conventions: `from __future__ import annotations`, type hints, frozen dataclasses for results, module-level `logger = logging.getLogger("sentinel_patch")`, pydantic config, no global state. These match the existing packages.

---

## 9. Test plan

### 9.1 Synthetic fixtures (`conftest.py`)
- **Drone fixture:** 600 m × 600 m at 0.5 m GSD (1200²), 5 bands, with field rectangles of different NDVI, a road cross, a tree line and a nodata collar. CRS EPSG:32642.
- **S2 fixture:** derived from the same scene. Blur σ = 5 m → block-average to 10 m → **shift by a known (dx, dy)** → add noise (σ = 2 % reflectance) → convert to DN with offset +1000 → write single-band COG-like GeoTIFFs with a 300 m margin. Also SCL (all class 4/5) and a variant with a synthetic cloud.

### 9.2 Required tests (all must be green before merge)

| Test | Assertion |
|---|---|
| `test_known_shift_roundtrip` | For shifts {(0,0), (+3.7,−6.2), (−12,+4), (+0.4,+0.4)} m the recovered shift is within 0.15 px with both estimators, and the shifted S2 overlays the truth (correlation ↑). **Sign check.** |
| `test_radiometry_baselines` | PB 03.xx (offset 0) and PB ≥ 04.00 (−0.1) give identical reflectance from appropriately encoded DN. Missing metadata → `RadiometryUnknownError`. |
| `test_footprint_excludes_collar` | Polygon area is within 2 % of the true valid area, not the bbox. |
| `test_selection_prefers_close_clear` | Given 4 fake candidates, the closest-date clear one wins; a cloudy AOI is rejected even with a low scene cloud %. |
| `test_date_ladder_order` | Rungs are visited 0, −1, +1, −2, +2 …; ties broken by masked fraction then after-flight; `after_only` / `before_only` respected; tiers assigned at 3 / 7 / 15 days. |
| `test_cutoff_and_fallback` | Day 1 at 25 % masked, day 4 at 8 % → day 4 chosen (`clean`). All rungs > 10 % with best 22 % → `partial`. Best 45 % → `NO_SCENE`. AOI with 60 clear pixels → rejected by `min_clear_pixels`. |
| `test_provider_contract` | Every registered provider passes the shared contract on recorded fixtures (band map, radiometry for both baselines, required fields). |
| `test_provider_chain_failover` | Provider 1 raises `ProviderUnavailableError` → provider 2 used; failure recorded in `candidates.json`. Same granule from two providers → deduplicated. |
| `test_drone_metadata` | Fixture JPGs with XMP `RtkFlag` 50 → `rtk`; 16 → `gnss_only` + `RTK_NOT_FIXED`; no tags → `unknown`; flight date = median `DateTimeOriginal`. Source order: flag > `flight.json` > ODM > photos > path estimate. No source → `FLIGHT_DATE_UNKNOWN`, product 1 still published. |
| `test_date_from_path` | Table-driven, ~30 cases: ISO / compact / month-name formats found; `DJI_0042`, `v2.10`, 6-digit numbers ignored; `10-03-2026` ambiguous under both `dmy` and `mdy` but resolved for `25-03-2026`; future, pre-2015 and after-`TIFFTAG_DATETIME` dates dropped; two dates → ranked `MULTIPLE_DATES`. |
| `test_path_estimate_confirmation` | Non-interactive + `accept_path_date_estimate: false` → `FLIGHT_DATE_UNKNOWN` with candidates in the message; `true` + ambiguous → still rejected; interactive confirm → `flight.json` written and found at step 2 on the next run. |
| `test_utm_zone_mismatch` | Drone in 32643, S2 in 32642 → shift recovered within 0.2 px, outputs in S2 CRS (analytic) and 32643 (overlay). |
| `test_low_texture_rejected` | Uniform field → `LOW_TEXTURE`, status `not_applied`, transform unchanged. |
| `test_tiny_aoi_rejected` | 30 m × 30 m stack → `AOI_TOO_SMALL`. |
| `test_cloud_masked_matching` | Cloud over 20 % of the window → still correct shift (masked); over 60 % of the AOI → scene rejected. |
| `test_temporal_change_detected` | S2 fixture with half the fields "harvested" → either a correct shift or a rejection, **never** a wrong accepted shift (> 0.5 px). |
| `test_no_resampling_analytic` | Analytic product pixel values == source window pixel values (bit-exact after DN→refl). |
| `test_coverage_fraction` | Coverage band sums to AOI area / 100 m² ± 1 %. |
| `test_crop_health_contract` | Run `run_crop_health_report` on our `bands/` output + footprint; NDVI mean equals the NDVI computed directly from our reflectance ± 1e-4. |
| `test_cli_sentinel_failure_isolated` | Provider raises → CLI prints a warning, exit code 0, product 1 present. |
| `test_pipeline_offline` | End-to-end with `LocalFolderProvider`; all files in §5.3 exist; report validates against the schema. |

### 9.3 Real-data validation (before the feature is called production)
1. ≥ 5 datasets, different fields, dates and crop stages; at least 2 with a road/canal in view and 1 homogeneous.
2. For each: in QGIS, pick ≥ 4 landmarks visible at 10 m (road junctions, canal corners, field corners). Measure S2-vs-drone offsets **before** and **after**.
3. Acceptance: median residual after ≤ 0.4 px (4 m) on verified runs; no verified run worse than before; homogeneous datasets → `not_applied` (not a random shift).
4. Record results in `docs/SENTINEL_PATCH_VALIDATION.md` (same style as the existing development journal).

---

## 10. Implementation phases (hand-off order)

Each phase is a mergeable PR with its tests. Do not start a phase before the previous one is green.

| Phase | Scope | Definition of done |
|---|---|---|
| **P0 — Skeleton** ✅ **DONE 2026-10-18** | Package, config schema + default YAML, errors, `shift.py`, `radiometry.py`, synthetic fixtures | `test_known_shift_roundtrip` (with a trivial estimator stub), `test_radiometry_baselines` pass. **Result:** 50 new tests green; the existing `crop_health_sentinel` suite is unaffected. Implementation notes: `sentinel_patch/tests/synthetic.py` holds the analytic-scene fixtures and `integer_shift_stub`; P1/P2 estimators must be added to `INTEGER_ESTIMATORS` in `test_shift_convention.py` (fractional-shift cases such as (+3.7, −6.2) m belong with the real sub-pixel estimator, since the stub is integer-only). |
| **P1 — Offline core** ✅ **DONE 2026-10-18** | S1 footprint, `LocalFolderProvider`, S3 reader (local files), S4 reference, NCC estimator, gates G1–G6, S7 analytic products, report JSON | `test_pipeline_offline` + all tests in §9.2 except Earth Search, UTM and CLI. **Result:** 120 `sentinel_patch` tests green (70 added in P1); the existing `crop_health_sentinel` suite is unaffected (144 passed together). A deliberate sign flip in `shift.py` makes 18 tests fail, so the sign convention is guarded. **Built:** `footprint.py`, `discovery/{provider,date_ladder}.py`, `discovery/providers/local_folder.py`, `io/{window_reader,cloud_mask}.py`, `registration/{reference,estimators,gates}.py`, `products.py` (10 m, 20 m, SCL, coverage, `bands/`), `quality/report.py`, `pipeline.py`. **Also done early:** the `crop_health_sentinel` contract test (originally P4), the date tier, nearest-date choice and 10 % cut-off. **Deliberately left for later:** the date ladder's rung/tie logic and 'partial' fallback (P3), AROSICS and G4/G7 (P2), overlay and preview PNG (P2), online providers (P3), flight-date sources (P4; P1 takes `flight_date` as an argument). **Changes to the original spec**, all explained at their sections: `ncc_search_px` 3→4, texture gate now a coefficient of variation (default 0.05), peak ratio uses a distinct local maximum, overlap gate runs before texture, tiny stacks stop early with `AOI_TOO_SMALL`, `footprint.geojson` is one WGS84 feature. |
| **P2 — AROSICS + robustness** ✅ **DONE 2026-10-18** | AROSICS estimator, G4 agreement, G7 quadrant check, UTM mismatch handling, coverage band, overlay, preview PNG | UTM, coverage and preview tests; visual check of the preview on fixtures. **Result:** 154 `sentinel_patch` tests green (34 added in P2); with the existing `crop_health_sentinel` suite, 178 passed. The previews were checked by eye (verified and refused cases); breaking the zone reprojection on purpose makes the zone test fail. **Built:** `registration/{arosics_estimator,quadrants}.py`, G4 in `gates.py`, `build_overlay` in `products.py`, `quality/preview.py`; the coverage band was already done in P1; zone mismatch already worked and is now tested (drone in EPSG:32643, Sentinel in 32642: shift recovered to (−15.14, +12.07) against a true (−15, +12)). **Main finding, changes the design:** AROSICS is not a reliable veto on small windows against a blurred drone reference (median error 0.72 px against 0.03 px for NCC; 29 % of its conclusive answers differ from the correct answer by more than a pixel), so it is **advisory by default** (`arosics_mode`), the final shift is always NCC's, `agreement_tolerance_px` is 1.0 and `quadrant_check_min_px` is 48. Details in §6.5. **All of these figures are synthetic: re-check on real imagery in P5 before deciding on `veto`.** |
| **P3 — Online discovery** ✅ DONE 2026-10-18 | Provider protocol + chain + registry, `EarthSearchProvider`, `PlanetaryComputerProvider`, `CdseProvider` stub, date ladder, cut-off + fallback, GDAL env, retries, cache, `candidates.json` | Provider contract + chain + ladder + cut-off tests; live test (manual run, `-m network`) on 2 real fields with each free provider |

**P3 result and deviations (read before P4):** 50 new offline tests plus 3 live tests (`pytest -m network`, deselected by default). Live check: both catalogues searched anonymously, a windowed B08/SCL read worked from each, including Planetary Computer's signed URLs, over one AOI in Pakistan (tile 43SBT, Oct 2025). Only one location was tried, so "2 real fields" is not met yet; P5 covers more.
1. **No `pystac-client`.** STAC search is one `POST /search` plus `next` links, done with `requests` (`discovery/http.py`, `discovery/stac.py`). This avoids a dependency whose version drift would not buy us anything. `requests` is now declared in `pyproject.toml`.
2. **Duplicate key is `(s2:datatake_id, mgrs_tile)`, not `(datetime, orbit)`.** Checked live: the same granule has timestamps 7 minutes apart in Earth Search (05:59:59) and Planetary Computer (05:52:51), and Earth Search items carry no relative orbit. Different tiles of one datatake are different granules and are both kept. Among duplicates, provider order wins unless only a later copy fully contains the AOI.
3. **The chain queries every enabled provider** (two small JSON calls) instead of stopping at the first that returns a scene; selection then works on the merged, de-duplicated list. Result is the same as "try the next provider when nothing is accepted", and simpler to reason about.
4. **Selection is per date.** All scenes of one date are measured, the clearest wins (tie: after the flight, then provider order). The first date at or below 10 % masked ends the search; farther dates are never read. If none passes, the clearest scene at or below 30 % masked is published as `partial` (warning `SCENE_PARTIAL`, masked pixels written as NaN in the 10 m and 20 m stacks and the crop_health `bands/`; the coverage layers stay geometric). Evaluation beyond `max_candidates_evaluated` is recorded as `EVALUATION_LIMIT`.
5. **All providers down** gives status `no_scene`, reason `PROVIDERS_UNAVAILABLE`, and `candidates.json` lists each failure under `provider_failures`. A provider that cannot even start (CDSE without credentials) is recorded the same way.
6. **Radiometry:** Earth Search's declared scale/offset is used only for the `sentinel-2-c1-l2a` collection; Planetary Computer's declared values are never used (its items declare none), so it always goes through the processing-baseline rule. Both tested for PB 05.11 and PB 03.01 items.
7. **No on-disk search cache.** Not built: re-runs cost two small queries, and a cache could hide a newly published scene. GDAL's `VSI_CACHE` and HTTP retries are set (`io/gdal_env.py`). A cache can be added behind `io.cache_dir` if rate limits ever bite.
8. **CDSE** is a stub: it checks `CDSE_S3_ACCESS_KEY`/`CDSE_S3_SECRET_KEY` and then reports "not implemented", so enabling it today only produces a recorded failure.
9. Planetary Computer hrefs carry `?<sas token>`, so `CPL_VSIL_CURL_ALLOWED_EXTENSIONS` is set for Earth Search only.
| **P4 — Integration** ✅ DONE 2026-10-19 (Docker image not built, see 6) | CLI flags, `drone_metadata.py` (date-source chain, EXIF date + DJI RTK flag), `date_from_path.py` + confirmation prompt + `flight.json` write-back, report hook, PowerShell launchers, Docker, `bands/` contract with `crop_health_sentinel` | CLI isolation, drone-metadata and crop-health contract tests; Docker build runs end-to-end |
| **P5 — Validation** (IN PROGRESS: part 1 done 2026-10-19, see `docs/SENTINEL_PATCH_P5_REAL_DATA_FINDINGS.md`) | §9.3 real-data campaign; log masked fraction, date tier and positioning for every run; re-check the 10 % / 30 % cut-offs and the 3 / 7 / 15-day tiers; docs | Validation doc committed; thresholds frozen in the default YAML |

**P4 result and deviations (read before P5):** 61 new offline tests (path-date estimate 25, drone metadata and confirm-then-remember 23, CLI stage and real `click` command with the alignment stubbed 13).
1. **CLI:** `--sentinel/--no-sentinel`, `--flight-date`, `--images-dir`, `--odm-project`, `--drone-positioning`, `--sentinel-config`, `--sentinel-ms-bands`, `--sentinel-strict`. The logic lives in `sentinel_patch/cli_stage.py`; `drone_alignment/cli.py` only adds the options and one lazy-import block *after* the alignment's own try/except, so a Sentinel error can never turn into the alignment's "Alignment failed" abort. Exit code is 0 unless `--sentinel-strict` (exit 3 when the result is not `verified`). Sentinel options without `--sentinel` print a warning and are ignored.
2. **Report hook:** instead of a kwarg on `write_alignment_report`, a small `update_report_json()` in `drone_alignment/quality/report.py` adds a `sentinel_patch` block (status, reason, shift, scene id, report path, warnings) to the existing alignment report, atomically. `drone_alignment/pipeline.py` is untouched.
3. **MS band numbers.** The aligned MS file does not name its bands, and `drone_alignment` only knows red/green indices. The default is `red=1,green=2,nir=3`: the company's MS orthophotos have 3 bands (confirmed by the user in QGIS, 2026-10-19; no red-edge or blue). `--sentinel-ms-bands` overrides it for other cameras. The stage refuses a band number the file does not have (`CONFIG_INVALID`, message names the file's band count). (Originally assumed MicaSense order; corrected 2026-10-19.)
4. **Flight-date sources implemented:** argument → `flight.json` → raw-photo EXIF (`--images-dir`, or `<odm-project>/images`) → folder-name estimate (confirm prompt, or unattended only with `accept_path_date_estimate: true` and an unambiguous high-confidence match). The ODM `images.json` source (#3 in 6.9) was **not** built: its field names could not be verified without the company's ODM output. `--odm-project` still supplies the raw photos folder and `gcp_list.txt` (positioning `gcp`).
5. **Positioning:** `--drone-positioning` > `flight.json` > ODM `gcp_list.txt` > DJI `RtkFlag` in the photos (≥ 90 % fixed = `rtk`, otherwise `gnss_only` + `RTK_NOT_FIXED`) > `unknown`. `LARGE_SHIFT_DRONE_FRAME_SUSPECT` (new `registration.large_shift_warning_m`, default 15) is added for a verified shift above that when positioning is `gnss_only` or `unknown`. It is only evaluated when the CLI supplies drone metadata. The `RtkFlag` parser matches the attribute and element forms of the XMP and was tested on synthetic JPEGs only, not on real DJI files.
6. **Launchers and Docker:** both PowerShell launchers parse cleanly and pass the new options (`-Sentinel`, `-FlightDate`, ...); `run_alignment.ps1` and the drag-and-drop flow of `scripts/run-alignment.ps1` ask "Also fetch & align Sentinel-2 patch? [y/N]". `Dockerfile` and `docker/pyproject.toml` now copy and install `sentinel_patch` (+ `requests`) and `DOCKER_README.md` documents it. **The image was not built or run: Docker is not installed on this machine.** The launcher prompts were not run interactively either.
7. **Not done:** the optional `--sentinel-health` chain into `crop_health_sentinel`. The `bands/` output is already verified against that package's reader (P1 contract test), so it can be run on `<output>/sentinel/bands` by hand.

**P5 part 1 additions (2026-10-19), details in `docs/SENTINEL_PATCH_P5_REAL_DATA_FINDINGS.md`:**
1. **Band order corrected** to `red=1,green=2,nir=3` (user, QGIS); the stage refuses a band number the file lacks.
2. **Real-data fixes:** the reference builder ignores overflow-garbage cells (`reference.outlier_mad_factor`, 50 robust sigmas); remote reads retry 3 times and a scene that still fails is recorded as unreadable (`io/window_reader.py`).
3. **Multi-scene consensus** (`registration/consensus.py`, config `registration.consensus`, default on). Tried only when the single best scene is refused for `AOI_TOO_SMALL` (overlap), `WEAK_PEAK`, `PEAK_AT_SEARCH_BORDER` or `SHIFT_IMPLAUSIBLE`; never after `LOW_TEXTURE`, `NON_TRANSLATIONAL`, `HOLDOUT_NO_GAIN` or an AROSICS veto. Up to 10 clean scenes (<= 10 % masked) vote, nearest date first, one vote per acquisition (tiles of one datatake count once). A vote is the scene's own NCC shift with a lower overlap floor (100 px) and no peak-ratio test. The shift is accepted when >= 3 scenes and >= 50 % of all votes lie within 1 pixel of their median and no second shift has as much support; the published shift is the members' median. Result: status `verified`, warning `SHIFT_FROM_CONSENSUS`, `registration.method: "consensus"`, the single-scene refusal kept under `registration.single_scene`, every vote under `registration.consensus`. Limit: it cannot detect a static look-alike repeated in every scene (see findings doc). 19 tests.
4. `LocalFolderProvider` reads an optional `datatake_id` from `scene.json`.

---

## 11. Explicit non-goals (v1)

- No pansharpening, super-resolution or "fusion" of Sentinel to drone GSD. Upsampling S2 does not create information that is genuinely present at the finer scale (Nature Sci. Rep. 2025).
- No local/non-rigid warping of Sentinel.
- No mosaicking of multiple tiles or dates.
- No radiometric cross-calibration of drone ↔ S2 (a possible v2, e.g. the *homonim* approach of using S2 surface reflectance as the radiometric reference for drone mosaics).
- No time series. v1 handles one scene per flight; a time series would reuse the same shift per relative orbit, which is a v2 idea.

---

## 12. Product-owner decisions (2026-10-18)

| # | Question | Decision | Where it lands |
|---|---|---|---|
| 1 | Do flights use RTK/PPK or GCPs? | **Unknown.** Probably no GCPs; RTK unconfirmed. | Auto-detect from DJI XMP when `--images-dir` is given (§6.9). Default `unknown`, treated as `gnss_only`. Still worth confirming with the flight team. |
| 2 | Date window | **As close as possible, then step outward.** | Expanding date ladder 0, ±1, ±2 … up to 15 days, with tiers `same_period` ≤ 3, `near` ≤ 7, `stale` ≤ 15 (§6.2.1). |
| 3 | When is a patch too incomplete? | **Use a researched cut-off; if it is hit, move to the next date.** | ≤ 10 % of the AOI masked (cloud/shadow/nodata, dilated) → accept; otherwise next rung. If no rung passes: best scene ≤ 30 % masked as `partial`; else `NO_SCENE` (§6.2.1). The job only warns on `NO_SCENE` (exit 0); `--sentinel-strict` makes it fail. |
| 4 | Data source | **Free for now; must be swappable later.** | Provider chain: Earth Search → Planetary Computer (both anonymous). CDSE stub (free account) ready; paid providers plug in as one file + contract test (§6.2.2). |

| 5 | Raw photos needed? / no date anywhere | **No engine redesign; orthophotos stay the only alignment input.** Estimate the date from folder/file names as a last resort, with manual confirmation. | Date-source chain + path-name estimation with confirmation and `flight.json` write-back (§6.9, §6.9.1). |

Remaining open item: confirm the positioning practice with the flight team (one message). It changes only a warning threshold, so it does not block implementation.

---

## 13. References

- ESA, Sentinel-2 geometric refinement (GRI) deployment notice — https://colhub.copernicus.eu/news/News00600
- ESA LPS 2022, S2 geometric performance (absolute < 7 m, multi-temporal < 5 m CE95) — https://earth.esa.int/living-planet-symposium-2022-presentations/23.05.Monday/Santiago_del_Chile/1330-1510/05_Enache.pdf
- Sentinel-2 Data Quality Report — https://sentinel.esa.int/documents/247904/685211/Sentinel-2+Data+Quality+Report+(DQR)
- ESA forum, radiometric offset in PB 04.00 — https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-pb04-00-products/35431
- Baseline/offset applies by baseline, not date — https://forum.step.esa.int/t/get-sentinel-2-reflectance-values-from-dn-for-mosaicking/44397/6
- Planetary Computer baseline-change notebook — https://nbviewer.org/github/microsoft/PlanetaryComputerExamples/blob/main/datasets/sentinel-2-l2a/baseline-change.ipynb
- CDSE forum, STAC offset vs Browser NDVI mismatch — https://forum.dataspace.copernicus.eu/t/sentinel-2-baseline-04-00-offset-csde-stac-derived-ndvi-doesnt-match-copernicus-browser-ndvi/5417
- Earth Search `sentinel-2-c1-l2a` collection (scale 0.0001 / offset −0.1) — https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a · https://github.com/Element84/earth-search
- S2 MTF/PSF (MTF at Nyquist 0.15–0.3) — https://sentiwiki.copernicus.eu/web/s2-mission
- Scheffler et al. 2017, AROSICS, *Remote Sensing* 9(7):676 — https://github.com/GFZ/arosics · API: https://danschef.git-pages.gfz-potsdam.de/arosics/doc/arosics.html
- Bollas et al., Sentinel-2 vs UAV, Northern Greece, *Drones* 5(2):35 — https://doi.org/10.3390/drones5020035
- Crop monitoring with S2 and UAV, NE Germany (landmark co-registration, tramlines) — https://www.mdpi.com/2072-4292/14/17/4426
- "Sentinel-2 Data for Precision Agriculture? — A UAV-Based Assessment" — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8073067/
- UAS-guided super-resolution (registration 0.81 m vs 4.6 m) — https://arxiv.org/pdf/2505.21746
- UAV + S2 fusion, upsampling limits (Sci. Rep. 2025) — https://www.nature.com/articles/s41598-025-13049-5
- SCL cloud-shadow omission; AgroShadow — https://www.mdpi.com/2072-4292/13/6/1219/htm · https://www.mdpi.com/2072-4292/15/16/3955
- s2cloudless shadow logic — https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
- homonim (S2 as radiometric reference for drone mosaics) — https://homonim.readthedocs.io/en/stable/tutorials/drone_correction.html
- Residual S2 coregistration via phase correlation — https://elisecolin.medium.com/how-to-easily-compensate-for-small-coregistration-errors-in-sentinel-2-time-series-0585c00f83e5
- DJI XMP RTK flags (50 fixed, 34/16 float/single) — https://support.sitemark.com/en/articles/9610492-how-to-read-dji-rtk-ntrip-metadata · caveat: https://community.pix4d.com/t/dji-matrice-210-rtk/2947?page=6
- Parcel "cloud covered" at > 50 % masked (agri ML review) — https://arxiv.org/pdf/2211.12584
- Field-clipped cloud/data coverage filtering (Leaf) — https://docs.withleaf.io/satellite/overview
- Clear-sky scarcity in cropping seasons (Embrapa) — https://www.alice.cnptia.embrapa.br/handle/doc/1060359
- Planetary Computer SAS signing (anonymous tokens) — https://planetarycomputer.microsoft.com/docs/concepts/sas
- CDSE S3 keys via free account (eodag provider docs) — https://eodag.readthedocs.io/en/stable/providers.html
- S2A→S2C transfer of duty, 21 Jan 2025 — https://sentinels.copernicus.eu/-/precise-timing-of-transfer-of-duty-from-sentinel-2a-to-sentinel-2c-on-21-january · S2A extension — https://europeanspaceflight.com/sentinel-2a-given-one-year-exceptional-and-temporary-extension/
