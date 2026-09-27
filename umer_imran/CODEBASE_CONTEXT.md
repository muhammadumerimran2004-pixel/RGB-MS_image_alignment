# Codebase Context & Developer Handoff Guide

> **Repository:** AgriLift Spatial & Crop Health Analytics Engine  
> **Last Updated:** September 19, 2026 (Alignment v4: Phases 0–4 done, Phase 5 real-data validation in progress)  
> **Read next:** `ALIGNMENT_V4_AROSICS_TPS_BLUEPRINT.md` (design authority) → `ALIGNMENT_V4_DEVELOPMENT_JOURNAL.md` (what worked, what didn't, real-data findings) → `AROSICS_IMPLEMENTATION_STATUS.md` (phase checklist)  
> **Purpose:** This document provides AI models and incoming developers with an authoritative reference map of the codebase. It details component responsibilities, file mappings for common changes, execution boundaries, and development guidelines.

---

## 1. Project Overview & Architecture

AgriLift processes aerial and satellite imagery to provide precision crop health analytics for farms. The codebase contains three core operational/experimental subsystems and comprehensive architectural documentation:

1. **`drone_alignment`** — Decoupled spatial co-registration engine for drone imagery. **v4 (current):** V3's verified global pipeline, followed by AROSICS `COREG_LOCAL` coarse-to-fine local refinement and warp, gated by independent holdout/full-grid verification (see §6), plus a rewritten manual GCP mode. The V3 foundation it builds on: warps OpenDroneMap (ODM) Multispectral (MS) orthomosaics onto reference RGB orthomosaic grids using low-resolution registration grids (2048–4096 px), Red-Red / Green-Green spectral matching (ORB/SIFT), candidate validation with independent correspondence splitting, masked phase correlation translation fallback, native-coordinate matrix conjugation ($M_{\text{native}} = S_{\text{destination}}^{-1} \times M_{\text{low\_res}} \times S_{\text{source}}$), tiled full-resolution streaming, native footprint confirmation, and atomic staging/publication.
2. **`agrilift_alignment`** — Advanced V2 multi-representation registration evidence subsystem. Implements multi-channel preprocessing mappers (Sobel, LoG, orientation, Gabor orientation, Gabor energy), tiled local displacement / SIFT / MIM correspondence extraction, candidate transformation fitting (similarity & affine), and held-out RMSE / spatial coverage validation.
3. **`crop_health_sentinel`** — Sentinel-2 satellite spectral reporting pipeline (5-layer architecture: Quality, Spectral, Health, H3 Spatial Aggregation, Report).

---

## 2. Directory Topography

```text
d:/Internship Stuff/Agrilift/
│
├── drone_alignment/                     # V3 Production Drone RGB-to-MS spatial co-registration engine
│   ├── __init__.py                      # Public API exports (align_orthomosaics)
│   ├── __main__.py                      # Module execution wrapper (python -m drone_alignment)
│   ├── cli.py                           # Click CLI command handler (--output-dir, --resolution, --detector, etc.)
│   ├── pipeline.py                      # Master pipeline orchestrator, fallback manager & atomic publication
│   ├── version.py                       # Version identifier (v1.0.0)
│   │
│   ├── config/                          # Typed configuration schemas
│   │   ├── __init__.py
│   │   └── schema.py                    # Pydantic models (AlignmentConfig, ResolutionMode, QualityConfig, etc.)
│   │
│   ├── io/                              # Spatial I/O & input validation
│   │   ├── __init__.py
│   │   ├── reader.py                    # GeoTIFF metadata & windowed array reading
│   │   ├── validators.py                # File existence, CRS, and bounding box overlap checks
│   │   └── gcp_io.py                    # Manual GCP file readers (CSV, QGIS Georeferencer .points)
│   │
│   ├── alignment/                       # Core mathematical alignment modules
│   │   ├── __init__.py
│   │   ├── coarse.py                    # Stage 1: Geospatial reprojection & low-res registration grid normalization
│   │   ├── feature_detector.py          # Stage 2A: Keypoint detection (ORB/SIFT on Red/Green channels)
│   │   ├── feature_matcher.py           # Stage 2B: KNN descriptor matching & Lowe ratio test
│   │   ├── transform_estimator.py       # Stage 2C: RANSAC Affine/Homography, phase correlation fallback & validation
│   │   ├── representations.py           # Image representations (normalized, Gabor energy) for local modes
│   │   ├── warper.py                    # Stage 3: Windowed tiled raster streaming & native mask evaluation
│   │   ├── road_grid_aligner.py         # Road-grid mode: 1D road-profile translation (separate mode)
│   │   ├── local_mesh_aligner.py        # Experimental local mesh mode: sparse, road-constrained (off by default)
│   │   ├── rail_frame_aligner.py        # Road-frame registration from elongated bright/dark structural rails
│   │   ├── cell_correlator.py           # local-correlation mode: feature-agnostic per-cell residual evidence
│   │   ├── local_evidence.py            # Shared dataclasses (LocalMatchSample, SparseDisplacementResult, CellMatchStatus)
│   │   ├── field_eval.py                # Lattice-sampled residual field cache (GridSampledField) for TPS warps
│   │   ├── displacement_field.py        # Residual-field fitting shared by manual TPS & local-correlation modes
│   │   ├── rejections.py                # LocalRefinementRejected: shared safe-fallback rejection type
│   │   ├── arosics_matcher.py           # AROSICS COREG global candidate (feature-free, tried last)
│   │   ├── arosics_staging.py           # Native-res band staging, global-correction pre-application, whitespace-safe VRT aliases
│   │   ├── arosics_local.py             # AROSICS COREG_LOCAL refinement: tie points, gates, dual warp engines
│   │   ├── control_points.py            # Manual GCP model fitting/selection/QA (translation/similarity/affine/TPS)
│   │   └── manual.py                    # Manual GCP orchestration (run_manual_alignment) & TPS field class
│   │
│   ├── quality/                         # Residual analysis & verification artifacts
│   │   ├── __init__.py
│   │   ├── metrics.py                   # Spatial residual error analysis & footprint metrics
│   │   ├── visualization.py             # 2x2 Visual QA preview generator (alignment_preview.png)
│   │   └── report.py                    # Machine-readable JSON report builder
│   │
│   └── tests/                           # Pytest suite: 34 test files, 226 tests (all passing 2026-09-20)
│       ├── conftest.py                  # Synthetic GeoTIFF fixture generator (synthetic_geo_tiff_pair)
│       ├── test_global_pipeline_context.py      # Verified/unverified global context & two-path publication rule
│       ├── test_arosics_local*.py, test_pipeline_arosics_local_integration.py  # COREG_LOCAL gates, real AROSICS
│       ├── test_arosics_staging.py      # Staging + VRT alias GeoArray round-trip regression
│       ├── test_control_points.py, test_manual_alignment.py, test_gcp_io.py, test_cli_manual_gcp.py  # Manual mode
│       ├── test_field_eval.py, test_displacement_*.py, test_rejections.py, test_schema_migration.py
│       └── ...                          # plus coarse/feature/transform/reader/validator/local-mode tests
│
├── agrilift_alignment/                  # V2 Experimental multi-representation registration subsystem
│   ├── __init__.py
│   ├── core.py                          # Transform dataclasses, coordinate spaces & matrix conversions
│   ├── pipeline.py                      # Multi-mapper orchestrator & candidate evaluation loop
│   ├── preprocessing.py                 # Multi-spectral mappers (Sobel, LoG, orientation, Gabor energy/orientation)
│   ├── registration.py                  # Phase proposals, tiled local/SIFT/MIM matchers & candidate fitters
│   ├── validation.py                    # Spatial coverage & held-out RMSE validation metrics
│   └── tests/                           # Pytest suite for agrilift_alignment
│       └── test_direction.py
│
├── Stuff/                               # Local sample datasets & aligned output directories
│   ├── RGB_odm_orthophoto.tif           # Sample reference RGB GeoTIFF (~147 MB)
│   ├── odm_orthophoto.tif               # Sample target MS GeoTIFF (~744 MB)
│   └── aligned_v3/                      # Verified V3 pipeline benchmark output directory
│
├── scripts/                             # run-alignment.ps1 (PowerShell runner), run-tests.ps1, run-local-correlation.ps1
├── docs/MANUAL_GCP_GUIDE.md             # Manual GCP file formats (CSV / QGIS .points) and workflow
│
├── ALIGNMENT_V4_AROSICS_TPS_BLUEPRINT.md    # ★ v4 design authority (phases P0–P5, gates, acceptance criteria)
├── ALIGNMENT_V4_DEVELOPMENT_JOURNAL.md      # ★ v4 journal: approaches, yays/nays, bugs, real-data findings
├── AROSICS_IMPLEMENTATION_STATUS.md         # v4 phase delivery checklist
├── WP7_LOCAL_CORRELATION_VALIDATION.md      # local-correlation real-data validation (failed closed, not promoted)
├── ALIGNMENT_ENGINE_LAYERS.md               # Older layer/roadmap overview (pre-v4; roadmap items now superseded)
├── DRONE_ALIGNMENT_TARGET_ARCHITECTURE_FLOW.md # Architectural flow diagram & publication gates (V3)
├── AUTOMATED_ALIGNMENT_V3_IMPLEMENTATION_PLAN.md # V3 technical implementation plan & specs
├── DRONE_ALIGNMENT_V2_IMPLEMENTATION_PLAN.md # V2 multi-representation design document
├── DRONE_ALIGNMENT_REMEDIATION_BLUEPRINT.md    # Alignment remediation & failure recovery blueprint
├── CROP_HEALTH_SENTINEL_BLUEPRINT.md    # Sentinel pipeline architecture blueprint
└── CROP_HEALTH_SENTINEL_DEVELOPER_ONBOARDING.md # Developer onboarding guide for Sentinel engine
```

---

## 3. Task & Feature Routing Matrix (Where to Look for Which Change)

Use this table to immediately identify which files to inspect or modify based on the requested task:

| Task / Change Request | Primary Target File(s) | Key Symbols / Classes |
|---|---|---|
| **Add / modify alignment parameters or defaults** | [drone_alignment/config/schema.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/config/schema.py) | `AlignmentConfig`, `ResolutionMode`, `QualityConfig` |
| **Change CLI options or command-line flags** | [drone_alignment/cli.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/cli.py) | `@click.command()`, `main` |
| **Modify input validation or CRS checking** | [drone_alignment/io/validators.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/io/validators.py) | `validate_inputs()`, `compute_bounding_box_intersection()` |
| **Modify GeoTIFF metadata reading or windowed I/O** | [drone_alignment/io/reader.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/io/reader.py) | `read_metadata()`, `read_band()`, `read_band_windowed()` |
| **Adjust coarse reprojection or GSD resolution scaling** | [drone_alignment/alignment/coarse.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/coarse.py) | `coarse_align()`, `CoarseAlignmentResult` |
| **Modify keypoint extraction (ORB / SIFT / uint8 normalization)** | [drone_alignment/alignment/feature_detector.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/feature_detector.py) | `detect_features()`, `normalize_to_uint8()` |
| **Adjust descriptor matching or Lowe ratio test logic** | [drone_alignment/alignment/feature_matcher.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/feature_matcher.py) | `match_features()`, `InsufficientMatchesError` |
| **Update RANSAC, Affine/Homography bounds, or Phase Correlation fallback** | [drone_alignment/alignment/transform_estimator.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/transform_estimator.py) | `estimate_transform()`, `_fallback_phase_correlation()` |
| **Modify raster warping, block sizes, or memory streaming** | [drone_alignment/alignment/warper.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/warper.py) | `warp_ms_to_rgb_tiled()`, `evaluate_native_footprint()` |
| **Update spatial residual analysis (center vs. edge drift)** | [drone_alignment/quality/metrics.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/quality/metrics.py) | `evaluate_spatial_residuals()`, `SpatialResidualReport`, `FootprintMetrics` |
| **Modify QA visual preview graphics (`alignment_preview.png`)** | [drone_alignment/quality/visualization.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/quality/visualization.py) | `generate_alignment_preview()` |
| **Update JSON report structure or metadata fields** | [drone_alignment/quality/report.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/quality/report.py) | `write_alignment_report()` |
| **Modify execution pipeline, error recovery, or fallback sequence** | [drone_alignment/pipeline.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/pipeline.py) | `align_orthomosaics()`, `_to_native_transform()` |
| **Change when an unverified global candidate may be used / published** | [drone_alignment/pipeline.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/pipeline.py) | `_estimate_global_candidate()`, `VerifiedGlobalContext.is_verified`, `_rank_unverified_candidate()`, `align_orthomosaics()` |
| **Modify AROSICS input staging, path aliases, or global pre-correction** | [drone_alignment/alignment/arosics_staging.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/arosics_staging.py) | `create_whitespace_safe_alias()`, `stage_reference_band()`, `stage_precorrected_target_band()`, `compute_global_map_correction()` |
| **Modify AROSICS local co-registration (tie points, gates, warp engines)** | [drone_alignment/alignment/arosics_local.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/arosics_local.py) | `run_arosics_local_refinement()`, `_attempt_band_pair()` |
| **Modify manual GCP model selection or QA gates** | [drone_alignment/alignment/control_points.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/control_points.py) | `select_model()`, `evaluate_jacobian()`, `loo_rmse()` |
| **Modify manual GCP orchestration or the TPS field class** | [drone_alignment/alignment/manual.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/manual.py) | `run_manual_alignment()`, `ManualThinPlateSplineField` |
| **Add a GCP file format (CSV / QGIS Georeferencer)** | [drone_alignment/io/gcp_io.py](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/io/gcp_io.py) | `read_gcp_csv()`, `read_qgis_points()` |
| **Explore multi-representation / Gabor / MIM correspondence matching** | [agrilift_alignment/pipeline.py](file:///d:/Internship%20Stuff/Agrilift/agrilift_alignment/pipeline.py) | `align()`, `phase_proposal()`, `tiled_mim_descriptors()` |
| **Review V3 Target Architecture flow & publication gates** | [DRONE_ALIGNMENT_TARGET_ARCHITECTURE_FLOW.md](file:///d:/Internship%20Stuff/Agrilift/DRONE_ALIGNMENT_TARGET_ARCHITECTURE_FLOW.md) | Flowchart, native transform rule, publication gates |
| **Add or update unit/integration tests** | [drone_alignment/tests/](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/tests/) | `conftest.py`, `test_*.py` |

---

## 4. Key Implementation Rules & Conventions

When modifying or extending `drone_alignment`, adhere strictly to these architectural constraints:

1. **Decoupled Memory Operations**: Submodules inside `alignment/` must receive and return NumPy arrays and metadata dataclasses — **do NOT write or read files directly inside core algorithm modules**. File I/O belongs exclusively in `io/` and `pipeline.py`.
2. **Red↔Red / Green↔Green Spectral Bridge**: Feature detection must be performed between corresponding spectral channels (e.g. RGB Red to MS Red band or RGB Green to MS Green band). Never attempt direct feature matching between RGB Red and NIR bands, as gradient inversion causes descriptor failure.
3. **Low-Resolution Registration Grid**: Registration estimation occurs on a common downsampled grid (2048–4096 px). The resulting transformation matrix is conjugated into native output space via:
   $$M_{\text{native}} = S_{\text{destination}}^{-1} \times M_{\text{low\_resolution}} \times S_{\text{source}}$$
4. **Deterministic Correspondence Splitting**: Reserve 20% of detected correspondences for independent residual QA. RANSAC fits only on training points; candidates are rejected if held-out QA status fails.
5. **Strict Parameter Validation**: All configuration options must be defined in `drone_alignment/config/schema.py` using Pydantic `BaseModel` with explicit constraints (`ge`, `gt`, `le`, `lt`, descriptions). Avoid magic numbers in algorithm code.
6. **Windowed Memory Footprint & Streaming**: Always use windowed rasterio reading and tiled block writing (`tile_size=2048`) for output GeoTIFFs. Never attempt full float32 in-memory allocation of unclipped full-extent orthomosaics.
7. **Atomic Staging & Publication**: Write output to run-scoped staging paths (`.partial.tif`) and perform atomic replacement (`os.replace`) only after all bands, native mask checks, thumbnail QA previews, and JSON reports pass.
8. **Headless OpenCV**: Use `opencv-python-headless` for all image operations to ensure compatibility with server and container environments.
9. **Candidate Order & Publication Rule (v4)**: Global candidates are tried in the order `ORB/SIFT (Green, then Red) → AROSICS COREG (feature-free) → masked phase correlation`. Then AROSICS `COREG_LOCAL` refines the winner. A result is published only if it passes **our classical residual QA** or **AROSICS local refinement's own gates**, never neither. An unverified global candidate is a `COREG_LOCAL` starting point only, and is never published by itself. See §6.
11. **Held-out QA screens false matches; manual QA never does**: automated feature-match verification calls `evaluate_spatial_residuals(..., screen_false_matches=True)` (points beyond `quality.holdout_gross_mismatch_px` are excluded from RMSE, and `quality.min_holdout_agreement` of them must agree). Manual control points use the unscreened default, because a large residual there is a real human error.
12. **Footprint gating is in one place and ignores coverage differences**: every mode calls `footprint_gate_failures()` (`quality/metrics.py`), whose only gate is that the correction keeps the MS on the grid (`min_retained_valid_ratio`, MS-in-frame before vs after). RGB/MS overlap ratios are reported, never gated: the two flights routinely cover different ground (often the MS covers only the centre of the RGB), and same-field data is aligned as provided. `min_target_overlap_ratio` was retired 2026-09-20 (journal §5.7). The frame around the overlap is `coarse.overlap_margin_m` (10 m, clamped to the RGB), wide enough for real GPS-only shifts of 2–6 m.
10. **Every real-data bug gets a regression test that reproduces it** — synthetic fixtures have clean metadata and miss things (see the journal §5.1, the ODM band-metadata `IndexError`). Prefer real AROSICS/GDAL in tests over mocks: a `MagicMock` once hid a gate that never fired (blueprint finding A1).

---

## 5. Common Commands

### Running the Test Suite
```bash
# Run main drone_alignment tests
python -m pytest drone_alignment/tests/ -v --cov=drone_alignment

# Run all test suites across the repository
python -m pytest drone_alignment/tests/ agrilift_alignment/tests/ -v
```

### Running Alignment via CLI
```bash
python -m drone_alignment \
    "Stuff/RGB_odm_orthophoto.tif" \
    "Stuff/odm_orthophoto.tif" \
    --output-dir "Stuff/aligned" \
    --resolution ms \
    --ms-red-band 1 \
    --verbose
```

### Running Alignment via Python API
```python
from pathlib import Path
from drone_alignment import align_orthomosaics
from drone_alignment.config import AlignmentConfig, ResolutionMode

result = align_orthomosaics(
    rgb_path=Path("Stuff/RGB_odm_orthophoto.tif"),
    ms_path=Path("Stuff/odm_orthophoto.tif"),
    output_dir=Path("Stuff/aligned"),
    config=AlignmentConfig(
        resolution_mode=ResolutionMode.MS_NATIVE,
        ms_red_band_index=1
    )
)
```

### Launching a run by double-click
`scripts\Run-Alignment.bat` opens PowerShell, prompts for the RGB, MS and output
paths, runs automated mode with verbose logs, and keeps the window open afterwards
(`-NoExit`), so a fast failure can't close the window before you read it. Any
`run-alignment.ps1` argument can be appended. `run_alignment.ps1` at the repo root
is an older menu-driven launcher; its option [2] is the same automated engine.

### Running the tests in a sandboxed shell
If pytest reports dozens of `PermissionError: [WinError 5] ... pytest-of-DELL`
errors, the shell can't access the shared pytest temp folder. That's not a code
failure. Add `--basetemp=<any writable dir>`.

---

## 6. Automated-Mode Architecture (v4, updated September 19, 2026)

**Full design authority: `ALIGNMENT_V4_AROSICS_TPS_BLUEPRINT.md`.** This
section is a summary; if it and the blueprint ever disagree, the blueprint
wins and this section is stale. Why the design ended up this way is in
`ALIGNMENT_V4_DEVELOPMENT_JOURNAL.md`.

**Coarse to fine, the way AROSICS is designed to be used.**

1. **Global step.** `_estimate_global_candidate` tries ORB/SIFT
   (Green↔Green, then Red↔Red), AROSICS' own `COREG` as a
   feature-free candidate, then masked phase correlation. Each candidate goes
   through classical footprint/residual QA.
   - If one passes: `VerifiedGlobalContext.is_verified = True`.
   - If **none** passes: it doesn't abort. It returns the best candidate it
     computed (lowest measured RMSE first, see `_rank_unverified_candidate`) with
     `is_verified = False`. That candidate is **only a starting point for
     `COREG_LOCAL`**. It is never published by itself.
2. **Local step.** AROSICS `COREG_LOCAL` refines the global result and
   **performs the warp itself** (its own `DESHIFTER`, or a streaming GDAL
   thin-plate-spline warp for large rasters). Its output must clear gates this
   pipeline adds on top of AROSICS' own reliability/SSIM/RANSAC filtering:
   minimum valid tie points, coverage/hull checks, neighbour consistency, a
   spatially-stratified holdout, and full-grid post-warp verification (p90 ≤
   `max_verification_p90_px`, default 1.5 px).

**Publication rule. A result needs at least one of two independent approvals:**

| Global result | Local refinement | Outcome |
|---|---|---|
| verified | passes gates | publish local result (`arosics_local`) |
| verified | rejected / disabled | publish verified global result, reason in report `fallback` (`automated_global`) |
| unverified | passes gates | publish local result (`arosics_local`); report `fallback.reason_code = UNVERIFIED_GLOBAL_USED_AS_STARTING_POINT` records that AROSICS' gates, not classical QA, approved it |
| unverified | rejected / disabled | **`TransformUnreliableError`, nothing published** |

Two earlier designs are gone; don't bring them back from git history. In one,
`COREG_LOCAL` ran first with no independent verification and no fallback. In
the other (the first v4 build), AROSICS could only refine a global result that
had already passed classical QA. On real farm data every global candidate
failed, so AROSICS never got to correct the distortion it exists to fix.

Local refinement is **on by default** (`arosics.local.enabled = True`) — no
flag is needed to enable it. `--enable-arosics` / `-EnableArosics` now only
controls the feature-free global `COREG` *candidate* and enables custom
`--arosics-band-pair` selection; use `--no-arosics-local` /
`-NoArosicsLocal` to disable local refinement and get the global-only result.

The active environment has been verified with `AROSICS 1.13.2`; the
dependency is declared as the optional `arosics` project extra:

```powershell
python -m pip install -e '.[arosics,test]'
```

AROSICS can use arbitrary 1-indexed, semantically comparable source bands.
Pass one or more ordered CLI pairs in the form
`--arosics-band-pair NAME:REFERENCE_BAND:TARGET_BAND` (for example,
`red_edge:4:3`), or set `arosics.band_pairs` in YAML. Supplying custom pairs
replaces the legacy green-green/red-red AROSICS candidates; the first pair is
the primary local benchmark. The reference raster must actually contain the
requested band — an ordinary three-band RGB GeoTIFF cannot supply red-edge or
NIR.

AROSICS/GeoArray rejects literal file paths containing whitespace. The
staging module (`arosics_staging.py`) creates no-space temporary aliases and
stages matching bands and the published output in a no-space temporary
directory before copying to the user-selected output directory; RGB, MS, and
result paths may safely contain spaces. **The alias is always a GDAL VRT with
dataset and band metadata cleared, never a hard link.** Real ODM MS outputs
carry band-name metadata that doesn't match their band count. A hard link
keeps that as-is, and AROSICS' `DESHIFTER` then crashes with `IndexError` when
it saves (regression test in `test_arosics_staging.py`). A VRT costs no more
than a hard link: it's an XML reference, not a copy.

---

## 7. Manual GCP Alignment

Manual mode is independent of AROSICS: it fits a user-supplied control-point
model — translation, similarity, affine, or thin-plate spline, auto-selected
from control-point count, hull coverage, and leave-one-out RMSE (blueprint
§7) — and warps with the pipeline's own tiled warper (`warper.py`), so the
existing lattice-sampled-field/LOO/Jacobian QA machinery applies to it. The
CLI accepts control points either interactively or from a file
(`--gcp-file`, CSV or QGIS Georeferencer `.points`, with `role=check` rows
excluded from fitting and used instead to verify the published mapping's
accuracy). See `docs/MANUAL_GCP_GUIDE.md` for the file formats and
`ALIGNMENT_V4_AROSICS_TPS_BLUEPRINT.md` §7 for the model-selection and QA-gate
design.

---

## 8. Real-Data Validation Status (Phase 5, as of September 19, 2026)

Details, numbers and reasoning are in `ALIGNMENT_V4_DEVELOPMENT_JOURNAL.md` §4. Short version:

- **Dataset:** RAH-AAA0140 Amir Waraich Farm A, under
  `C:\Users\DELL\Downloads\RAH-AAA0140Ch Amir Waraich Farm A\25-07-2026\`.
  RGB and **both** MS folders are the **same flight (2026-07-25)**. The
  `0726`/`0826` prefix is the **ODM processing month**, not a flight date.
  `Multispectral_0826-A006` is a reprocessing with `pc_quality=high`,
  `dem_resolution=1.0`. `Multispectral_0726-C982` and the RGB used `low` / `5`.
  None uses GCPs (≈0.69 m GPS error).
- **Second dataset:** NAR-AAA0002 Faisal Rafiq Farm B (rice), fields A007 and
  A008, same GPS-only `low`-quality ODM settings.
- **Current results (after the four fixes in journal §5.6):** every real pair
  gets a **verified global** result and publishes `automated_global`, PASS:
  Farm A 0726 6.2 cm, Farm A 0826 8.8 cm, Farm B A008 6.2 cm, Farm B A007
  7.7 cm (global RMSE). AROSICS local refinement still misses its own bar on
  all four (p90 1.76–2.32 px against 1.5 px).
- **Withdrawn finding:** earlier notes said 0726 failed QA from "non-affine
  distortion, worst at the edges". That was false feature matches
  contaminating the held-out RMSE. Fixed; 0726 now fits best of all.
- **Earlier finding that still stands:** on 0826, `COREG_LOCAL` found only
  **5 of 197 tie points valid** (12 needed) under the old frame; channel and
  window size don't change it.
- **Done and closed, not just run:** R2 (both warp engines — DESHIFTER 2.27px
  vs gdal_tps 2.85px p90 on 0726; parity does **not** hold, documented as a
  real implementation difference, not a bug), R3 (crop-row guard — wired
  correctly, tie points identical to the unguarded run, doesn't bind on this
  data), R4 (global-only baseline — identical output grid to R1, `gdalinfo`
  check passes), and a channel/window-size sweep that confirms the 0826
  tie-point ceiling (5 of 197 valid) is a genuine content/spectral
  decorrelation, not a tunable parameter (ruled out: search radius, red vs.
  green channel, window sizes 128/256/512).
- **Canonical R1 input:** confirmed **0826** (see journal §4.1 — it is not a
  different flight, it's the same 2026-07-25 flight reprocessed at higher ODM
  quality).
- **Still needs a human at QGIS, cannot be resolved by another automated run:**
  R5 (manual GCPs + 5 check points), the independent 12-point QGIS check
  (which will also settle whether 1.5px p90 is the right bar), and a visual
  look at R1's preview to formally close it.
- **Deliberately not done:** loosening `max_verification_p90_px` to make a
  run pass. Any threshold change should be driven by the QGIS check-point
  measurements.
