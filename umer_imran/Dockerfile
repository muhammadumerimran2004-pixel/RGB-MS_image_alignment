# ─────────────────────────────────────────────────────────────
#  AgriLift – Drone Alignment   (headless, CLI-only image)
# ─────────────────────────────────────────────────────────────
#  Build:   docker build -t agrilift-align .
#  Run:     docker run --rm -v C:\data:/data agrilift-align \
#             /data/rgb.tif /data/ms.tif -o /data/results -m automated
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Install minimal native libs required by rasterio / GDAL / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
        gdal-bin \
        libgeos-dev \
        libproj-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- dependency layer (cached unless deps change) ---------------------
RUN pip install --no-cache-dir \
        "numpy>=2,<3" \
        "rasterio>=1.4,<2" \
        "shapely>=2,<3" \
        "pyproj>=3.7,<4" \
        "h3>=4.1,<5" \
        "portalocker>=2.10,<4" \
        "pydantic>=2.10,<3" \
        "PyYAML>=6,<7" \
        "matplotlib>=3.9,<4" \
        "click>=8.1,<9" \
        "opencv-python-headless>=4.10,<5" \
        "scipy>=1.11,<2" \
        "scikit-image>=0.21,<1" \
        "psutil>=5.9,<7" \
        "arosics>=1.13,<2"

# ---------- application layer ------------------------------------------------
# drone_alignment is self-contained — no cross-module imports.
COPY drone_alignment/    drone_alignment/

# Use a Docker-specific pyproject that only discovers drone_alignment,
# avoiding errors from missing agrilift_alignment / crop_health_sentinel.
COPY docker/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir -e .

# Create mount-point directories so they exist even without -v
RUN mkdir -p /data /input /output

# ---------- default entrypoint -----------------------------------------------
# Users pass arguments directly:
#   docker run agrilift-align /data/rgb.tif /data/ms.tif -o /data/out -m automated
ENTRYPOINT ["python", "-m", "drone_alignment"]
