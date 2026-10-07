# AgriLift – Drone Orthomosaic Alignment Engine

A high-performance, automated spatial co-registration engine for multi-modal drone imagery in precision agriculture. It aligns Multispectral (MS) orthophotos (Red, Green, RedEdge, NIR) to reference RGB orthomosaics with subpixel accuracy, compensating for GPS drift (2–6 m), flight altitude variance, lens distortion, and crop canopy parallax.

---

## Key Features

- **AROSICS COREG_LOCAL Subpixel Refinement (Recommended & Main Engine)**:
  - **Two-Tier Architecture**: Fast global macro co-registration followed by dense local subpixel phase correlation.
  - **Non-Rigid Warping**: Thin-Plate Spline (TPS) transformation via AROSICS DESHIFTER or streaming GDAL TPS for large rasters.
  - **RAM-Aware Worker Throttling**: Dynamic memory inspection via `psutil` automatically caps parallel `loky` workers based on available RAM to prevent OOM termination.
  - **Rigorous Safety Gates**: Multi-stage filtering including SSIM, RANSAC, a $k$-NN neighbor-consistency filter (preventing crop-row aliasing), convex hull coverage, holdout test, and full-grid post-warp verification.
  - **Graceful Fallback**: If local refinement is rejected at any safety gate, the engine safely rolls back to the verified global result without failing the run.
- **Fast Global Automated Baseline**:
  - Cross-spectral feature pairing (Red-Red, Green-Green) using CLAHE-enhanced ORB/SIFT with Lowe's ratio test ($0.75$) and RANSAC affine/homography fitting.
  - Frequency-domain Hann-windowed Masked Phase Correlation fallback for feature-sparse agricultural fields.
- **Manual Control Point (GCP) Mode**:
  - Interactive or CSV/QGIS-driven Ground Control Point alignment for challenging scenes.
- **Production-Ready & Containerized**:
  - Interactive PowerShell launcher (`.\run_alignment.ps1`).
  - Headless Docker container support (`agrilift-align`).
  - Cloud-Optimized GeoTIFF (COG) compatible tiled streaming output.

---

## Installation & Prerequisites

### 1. Python Environment
- Python 3.11+
- Install the package with AROSICS support:
  ```bash
  pip install -e ".[arosics]"
  ```
  Or install required packages directly:
  ```bash
  pip install "numpy>=2,<3" "rasterio>=1.4,<2" "shapely>=2,<3" "pyproj>=3.7,<4" "portalocker>=2.10,<4" "pydantic>=2.10,<3" "PyYAML>=6,<7" "matplotlib>=3.9,<4" "click>=8.1,<9" "opencv-python-headless>=4.10,<5" "scipy>=1.11,<2" "scikit-image>=0.21,<1" "psutil>=5.9,<7" "arosics>=1.13,<2"
  ```

### 2. Docker (Optional)
If running inside a container, install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) and ensure it is running in Linux container mode.

---

## How to Run AROSICS Alignment

### Method 1: Interactive PowerShell Launcher (Recommended)

Run the interactive script from the project root:

```powershell
.\run_alignment.ps1
```

1. Enter the full or relative path to your **RGB reference GeoTIFF**.
2. Enter the path to your **Multispectral (MS) target GeoTIFF**.
3. Enter the **output directory**.
4. Choose Mode **[3]** (the default recommended option):
   ```text
   Choose alignment mode:
     [1] Manual control points
     [2] Fast global automated alignment (low-RAM fallback)
     [3] Verified global + AROSICS subpixel local refinement  [RECOMMENDED - MAIN ENGINE]
     [4] Feature-agnostic local cell correlation  [EXPERIMENTAL]

   Enter 1, 2, 3, or 4 (recommended: 3): 3
   ```

*(Pressing Enter without typing a number automatically defaults to option 3).*

---

### Method 2: Command-Line Interface (CLI)

Run `drone_alignment` directly via Python module execution:

```bash
python -m drone_alignment "C:\data\rgb.tif" "C:\data\ms.tif" \
  --output-dir "C:\data\results" \
  --mode automated \
  --resolution ms \
  --arosics-local
```

#### Common CLI Options:

| Flag | Default | Description |
|---|---|---|
| `--mode, -m` | `prompt` | Alignment mode: `automated`, `manual`, `road_grid`, `local-correlation`. |
| `--output-dir, -o` | MS directory | Target folder for aligned raster and QA reports. |
| `--resolution, -r` | `ms` | Output pixel scale: `ms` (preserves native MS GSD) or `rgb` (upsamples to RGB). |
| `--arosics-local / --no-arosics-local` | `enabled` | Enable/disable AROSICS COREG_LOCAL subpixel refinement. |
| `--detector, -d` | `orb` | Primary feature detector: `orb` or `sift`. |
| `--arosics-band-pair` | `red:1:1` | Channel pair for AROSICS correlation (`NAME:REF_BAND:TGT_BAND`). |
| `--crop-row-period-m` | `None` | Crop row spacing in meters; caps search radius to avoid row aliasing. |
| `--arosics-max-shift-m` | Auto | Maximum local search radius in meters (derived from global RMSE). |
| `--arosics-warp-engine` | `auto` | Local warp engine: `auto`, `deshifter`, or `gdal_tps`. |

---

### Method 3: Docker Container

Use the provided Windows batch runner:

```cmd
run-alignment-docker.bat C:\data\rgb.tif C:\data\ms.tif C:\data\results automated
```

Or execute directly with `docker run`:

```cmd
docker run --rm ^
  -v C:\data:/data ^
  agrilift-align ^
  /data/rgb.tif /data/ms.tif ^
  --output-dir /data/results ^
  --mode automated ^
  --resolution ms ^
  --arosics-local
```

---

## Memory & RAM Requirements: Critical Heads-Up

> [!WARNING]
> **Crash / Abrupt Process Termination Notice:**  
> AROSICS `COREG_LOCAL` distributes dense subpixel cross-correlations across parallel worker processes (`loky`). Each worker process receives copies of both reference and target float32 image rasters.
> 
> **If the process terminates abruptly, python workers disappear without a traceback, or Docker exits with code 137:**  
> - **This indicates insufficient system RAM.** The operating system kernel or Docker OOM killer forcefully terminated the workers due to memory exhaustion.
> 
> **Built-in Protection:**  
> - The engine includes an automatic RAM-aware worker throttling mechanism (`_compute_safe_cpus`) that queries free RAM using `psutil` and calculates safe worker allocations before starting AROSICS.
> 
> **Recommendations if experiencing high memory pressure:**  
> 1. Close background memory-intensive applications.
> 2. For Docker Desktop, increase allocated memory to at least **8 GB – 16 GB** (*Settings → Resources → Memory*).
> 3. If memory is strictly limited, use **Mode [2]** in `.\run_alignment.ps1` or pass `--no-arosics-local` to run the lightweight global affine alignment.

---

## Output Products

Every successful alignment run produces the following artifacts in `--output-dir`:

1. **Aligned GeoTIFF** (`<ms_stem>_aligned.tif`):
   - Multi-band raster with exact CRS, origin, and pixel alignment matching the RGB reference.
   - Preserves native radiometric bit depth and nodata masks.
   - Written using tiled streaming for low peak memory footprint.
2. **Visual QA Overlay** (`<ms_stem>_alignment_preview.png`):
   - Dual-channel composite (e.g. RGB Red in Red channel, MS Red in Green channel) to instantly spot spatial alignment and edge overlap.
3. **Audit & Metrics Report** (`<ms_stem>_alignment_report.json`):
   - Complete execution trace: applied mode, affine/TPS parameters, inlier count, residual RMSE, footprint overlap, and safety gate decisions.
4. **Tie Points Table** (`<ms_stem>_tie_points.csv`):
   - Table of validated AROSICS tie points containing source/target map coordinates, subpixel shift vectors ($dx, dy$), and reliability scores.

---

## Running Tests

Verify your local installation by running the test suite:

```bash
# Test AROSICS local alignment engine and gates
python -m pytest drone_alignment/tests/test_arosics_local.py

# Test progress reporting
python -m pytest drone_alignment/tests/test_progress.py
```
