# AgriLift Drone Alignment Engine — Layered Architecture & Roadmap Guide

> **Repository:** AgriLift Spatial & Crop Health Analytics Engine  
> **Module:** `drone_alignment` (V3 Production Subsystem)  
> **Status:** Production Ready (V3) with Dual-Layer Alignment Strategy  

---

## 1. System Overview & Layered Architecture

The `drone_alignment` engine provides spatial co-registration of drone Multispectral (MS) orthomosaics onto reference RGB orthomosaic grids. To guarantee spatial accuracy under varying flight conditions, the engine uses a **two-tier alignment architecture** supported by a common spatial I/O foundation:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            User / CLI / API                                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
   ┌─────────────────────────────┐           ┌─────────────────────────────┐
   │ Layer 1: Manual Correction  │           │ Layer 2: Automated Pipeline │
   │ (GCP Point Pair Correction) │           │  (Feature & Phase Matching) │
   └──────────────┬──────────────┘           └──────────────┬──────────────┘
                  │                                         │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Common Shared Foundation                           │
│  - Spatial I/O & Metadata Extraction (io/reader.py, io/validators.py)       │
│  - Downsampled Registration Grid Normalization (alignment/coarse.py)        │
│  - Native Matrix Conjugation & Footprint Confirmation (pipeline.py)         │
│  - Memory-Safe Windowed Raster Streaming (alignment/warper.py)               │
│  - QA Preview & JSON Metric Reporting (quality/visualization.py, report.py) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: Manual Correction Layer (Human-in-the-Loop Override)

The Manual Correction Layer allows developers and remote sensing technicians to manually align rasters using Ground Control Points (GCPs) or visual landmark coordinates when automated feature detection is hindered by heavy cloud cover, featureless crops, or extreme seasonal illumination differences.

### 2.1 Code Map (Where the Manual Code Lies)

- **Input Prompt & Menu Interface**: [`drone_alignment/cli.py`](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/cli.py)
  - Interactively prompts the user to select manual mode and collects control point pairs (QGIS Map Coordinates or Pixel coordinates).
- **Core Math & Matrix Calculation**: [`drone_alignment/alignment/manual.py`](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/alignment/manual.py)
  - Contains `compute_manual_translation()`. Calculates the mean translation vector ($tx, ty$) and constructs the $2 \times 3$ affine matrix.
- **Pipeline Integration & Orchestration**: [`drone_alignment/pipeline.py`](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/pipeline.py)
  - `manual_align_orthomosaics()` converts map coordinates, performs native matrix conjugation, evaluates native footprints, and streams tiled output.
- **Configuration Schema**: [`drone_alignment/config/schema.py`](file:///d:/Internship%20Stuff/Agrilift/drone_alignment/config/schema.py)
  - Defines `AlignmentMode.MANUAL` enum and config options.

### 2.2 Mechanics & Coordinate Handling

```text
QGIS Map Coords (UTM East/North) OR Native Pixels (col, row)
                         │
                         ▼
        [Auto-Detect Coordinate System]
  - If X > 10,000  => QGIS Map Coordinate (meters)
  - If X <= 10,000 => Native Pixel Coordinate (col, row)
                         │
                         ▼
[Map Coordinates: (map_x, map_y)]
                         │
                         ▼ apply (~coarse.registration_transform)
[Registration Grid Pixels: (reg_x, reg_y)]
                         │
                         ▼
[Translation Vector: tx = mean(rgb_x - ms_x), ty = mean(rgb_y - ms_y)]
                         │
                         ▼
[Sanity Validation: hypot(tx_m, ty_m) <= max_translation_m (50m)]
```

---

## 3. Layer 2: Automated Recognition Pipeline

The Automated Pipeline automatically detects corresponding features between multispectral and RGB orthomosaics to produce sub-pixel alignment without human intervention.

### 3.1 Developmental Phases & Progress Status

| Phase | Description | Implementation Status | Key Module |
|---|---|---|---|
| **Phase 1: Shared Grounding & Coarse Reprojection** | Computes bounding box overlap, GSD resolution scaling, and builds a compact low-resolution registration grid (max 3072 px). | **COMPLETE** (100%) | `alignment/coarse.py` |
| **Phase 2: Cross-Spectral Feature Bridge** | Pairs matching channels (Red-Red, Green-Green) to prevent descriptor failure from NIR gradient inversions. | **COMPLETE** (100%) | `alignment/feature_detector.py` |
| **Phase 3: Keypoint Extraction & Matching** | Applies CLAHE contrast enhancement, extracts ORB/SIFT keypoints, and filters matches via Lowe's ratio test ($0.75$). | **COMPLETE** (100%) | `alignment/feature_detector.py`, `alignment/feature_matcher.py` |
| **Phase 4: RANSAC Fitting & Verification Split** | Splits matches 80/20. Fits RANSAC Affine/Homography matrices on 80%, reserves 20% for independent held-out QA verification. | **COMPLETE** (100%) | `alignment/transform_estimator.py`, `pipeline.py` |
| **Phase 5: Frequency Domain Fallback** | Fallback to Hann-windowed Masked Phase Correlation translation when feature matching yields insufficient inliers. | **COMPLETE** (100%) | `alignment/transform_estimator.py` |
| **Phase 6: Footprint & Spatial QA Gate** | Verifies native valid data retention ($\ge 80\%$) and reference overlap ($\ge 80\%$) before streaming tiled output. | **COMPLETE** (100%) | `alignment/warper.py`, `quality/metrics.py` |

### 3.2 Future Developmental Roadmap (What's Remaining)

While the V3 automated pipeline is fully operational and passes all test suites, the following future enhancements are planned:

1. **V4 Sub-Pixel Multi-Tile ECC Refinement**:
   - *Goal*: Run local Enhanced Correlation Coefficient (ECC) refinement on micro-tiles across complex terrain to handle subtle parallax distortion.
2. **Dense Deep-Learning Correspondence Engine (LoFTR/SuperPoint Integration)**:
   - *Goal*: Integrate dense deep-learning feature matching for severe seasonal vegetation changes (prototyped in experimental `agrilift_alignment/`).
3. **Non-Rigid Thin-Plate Spline (TPS) Warping**:
   - *Goal*: Extend from rigid Affine transforms to non-rigid TPS warping to correct non-linear elevation distortion caused by uncalibrated drone camera gimbals.

---

## 4. Key Architectural Invariants

Every addition to `drone_alignment` must respect these system invariants:

1. **Canonical Forward Transform**: The stored matrix always represents `MS source pixel -> RGB reference pixel`:
   $$p_{\text{rgb}} = M_{\text{ms\_to\_rgb}} \times p_{\text{ms}}$$
2. **Matrix Conjugation Rule**: Low-resolution registration matrices are conjugated to native space using scale matrices:
   $$M_{\text{native}} = S_{\text{destination}}^{-1} \times M_{\text{low\_res}} \times S_{\text{source}}$$
3. **Memory Footprint Limit**: Resident memory must stay under **1 GB RAM** by streaming full-resolution outputs in $2048 \times 2048$ windowed blocks (`warp_ms_to_rgb_tiled`).
4. **Atomic Publication**: Outputs are written to hidden staging files (`.partial.tif`) and atomically renamed (`os.replace`) only after all QA metrics pass.

---

## 5. Execution Reference

### Interactive Run (Select Manual or Automated via Menu)
```bash
python -m drone_alignment "Stuff/RGB_odm_orthophoto.tif" "Stuff/odm_orthophoto.tif" --output-dir "Stuff/aligned" -v
```

### Direct Manual Mode Run
```bash
python -m drone_alignment "Stuff/RGB_odm_orthophoto.tif" "Stuff/odm_orthophoto.tif" --output-dir "Stuff/aligned" --mode manual
```

### Direct Automated Mode Run
```bash
python -m drone_alignment "Stuff/RGB_odm_orthophoto.tif" "Stuff/odm_orthophoto.tif" --output-dir "Stuff/aligned" --mode automated --detector orb
```
