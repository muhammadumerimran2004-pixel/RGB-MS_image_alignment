# Codebase Context & Developer Handoff Guide

> **Repository:** AgriLift Spatial & Crop Health Analytics Engine  
> **Last Updated:** August 2026  
> **Purpose:** This document provides AI models and incoming developers with an authoritative reference map of the codebase. It details component responsibilities, file mappings for common changes, execution boundaries, and development guidelines.

---

## 1. Project Overview & Architecture

AgriLift processes aerial and satellite imagery to provide precision crop health analytics for farms. The codebase contains three core operational/experimental subsystems and comprehensive architectural documentation:

1. **`drone_alignment`** — Decoupled spatial co-registration engine for drone imagery (V3 Production Architecture). Warps OpenDroneMap (ODM) Multispectral (MS) orthomosaics onto reference RGB orthomosaic grids using low-resolution registration grids (2048–4096 px), Red-Red / Green-Green spectral matching (ORB/SIFT), candidate validation with independent correspondence splitting, masked phase correlation translation fallback, native-coordinate matrix conjugation ($M_{\text{native}} = S_{\text{destination}}^{-1} \times M_{\text{low\_res}} \times S_{\text{source}}$), tiled full-resolution streaming, native footprint confirmation, and atomic staging/publication.
2. **`agrilift_alignment`** — Advanced V2 multi-representation registration evidence subsystem. Implements multi-channel preprocessing mappers (Sobel, LoG, orientation, Gabor orientation, Gabor energy), tiled local displacement / SIFT / MIM / LoFTR correspondence extraction, candidate transformation fitting (similarity & affine), and held-out RMSE / spatial coverage validation.
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
│   │   └── validators.py                # File existence, CRS, and bounding box overlap checks
│   │
│   ├── alignment/                       # Core mathematical alignment modules
│   │   ├── __init__.py
│   │   ├── coarse.py                    # Stage 1: Geospatial reprojection & low-res registration grid normalization
│   │   ├── feature_detector.py          # Stage 2A: Keypoint detection (ORB/SIFT on Red/Green channels)
│   │   ├── feature_matcher.py           # Stage 2B: KNN descriptor matching & Lowe ratio test
│   │   ├── transform_estimator.py       # Stage 2C: RANSAC Affine/Homography, phase correlation fallback & validation
│   │   └── warper.py                    # Stage 3: Windowed tiled raster streaming & native mask evaluation
│   │
│   ├── quality/                         # Residual analysis & verification artifacts
│   │   ├── __init__.py
│   │   ├── metrics.py                   # Spatial residual error analysis & footprint metrics
│   │   ├── visualization.py             # 2x2 Visual QA preview generator (alignment_preview.png)
│   │   └── report.py                    # Machine-readable JSON report builder
│   │
│   └── tests/                           # Pytest automated test suite for drone_alignment
│       ├── conftest.py                  # Synthetic GeoTIFF fixture generator
│       ├── test_coarse_alignment.py
│       ├── test_feature_detector.py
│       ├── test_pipeline_integration.py
│       ├── test_reader.py
│       ├── test_residual_analysis.py
│       ├── test_transform_estimator.py
│       └── test_validators.py
│
├── agrilift_alignment/                  # V2 Experimental multi-representation registration subsystem
│   ├── __init__.py
│   ├── core.py                          # Transform dataclasses, coordinate spaces & matrix conversions
│   ├── pipeline.py                      # Multi-mapper orchestrator & candidate evaluation loop
│   ├── preprocessing.py                 # Multi-spectral mappers (Sobel, LoG, orientation, Gabor energy/orientation)
│   ├── registration.py                  # Phase proposals, tiled local/SIFT/MIM/LoFTR matchers & candidate fitters
│   ├── validation.py                    # Spatial coverage & held-out RMSE validation metrics
│   └── tests/                           # Pytest suite for agrilift_alignment
│       └── test_direction.py
│
├── Stuff/                               # Local sample datasets & aligned output directories
│   ├── RGB_odm_orthophoto.tif           # Sample reference RGB GeoTIFF (~147 MB)
│   ├── odm_orthophoto.tif               # Sample target MS GeoTIFF (~744 MB)
│   └── aligned_v3/                      # Verified V3 pipeline benchmark output directory
│
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
9. **Graceful Fallback Chain**: Pipeline orchestration in `pipeline.py` maintains the fallback order: `ORB Feature Matching → SIFT Feature Matching → Masked Phase Correlation Translation`.

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

