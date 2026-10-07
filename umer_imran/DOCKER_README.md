# AgriLift – Drone Alignment (Docker)

Run the Drone Alignment CLI in a self-contained Docker container.  
**No Python installation required** — only [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/).

---

## Quick Start (30 seconds)

### 1. Install Docker Desktop

Download from <https://www.docker.com/products/docker-desktop/> and make sure it's running (whale icon in the system tray).

### 2. Build the image (one time only)

Open **Command Prompt** or **PowerShell** in the project folder and run:

```cmd
docker build -t agrilift-align .
```

This takes ~3–5 minutes the first time (downloads Python + GDAL). Subsequent rebuilds are fast thanks to layer caching.

### 3. Run an alignment

```cmd
run-alignment-docker.bat  C:\data\rgb.tif  C:\data\ms.tif  C:\data\results  automated
```

| Argument     | Required | Description                                                                                           |
|--------------|----------|-------------------------------------------------------------------------------------------------------|
| `RGB_FILE`   | ✅       | Path to the RGB reference orthophoto (`.tif`)                                                        |
| `MS_FILE`    | ✅       | Path to the multispectral orthophoto (`.tif`)                                                        |
| `OUTPUT_DIR` | ❌       | Where to save results (defaults to the MS file's folder)                                              |
| `MODE`       | ❌       | `automated` (default), `local-correlation`, `road_grid`, `local_mesh`, or `manual`                   |

Results (aligned GeoTIFF + JSON report) appear in `OUTPUT_DIR`.

---

## Alternative: run `docker run` directly

If you prefer not to use the batch script:

```cmd
docker run --rm ^
    -v C:\data:/data ^
    agrilift-align ^
    /data/rgb.tif /data/ms.tif ^
    --output-dir /data/results ^
    --mode automated
```

> **Key idea:** `-v C:\data:/data` makes your local `C:\data` folder visible inside the container as `/data`. Use the `/data/…` paths for all file arguments.

### Extra CLI flags

All flags from the Python CLI are supported:

```
--resolution ms|rgb
--verbose
--enable-arosics
--arosics-band-pair "1:3"
--config /data/my_config.yaml
```

---

## Copying files instead of mounting

If volume mounts cause permission issues, you can copy files in and out:

```cmd
:: Create a container (don't start it)
docker create --name align-run agrilift-align /input/rgb.tif /input/ms.tif -o /output -m automated

:: Copy data in
docker cp C:\data\rgb.tif  align-run:/input/rgb.tif
docker cp C:\data\ms.tif   align-run:/input/ms.tif

:: Run
docker start -a align-run

:: Copy results out
docker cp align-run:/output  C:\data\results

:: Clean up
docker rm align-run
```

---

## Rebuilding after code changes

```cmd
docker build -t agrilift-align .
```

The dependency layer is cached — only code changes trigger a fast (~10 s) rebuild.

---

## Memory & RAM Requirements (AROSICS Subpixel Alignment)

AROSICS `COREG_LOCAL` uses parallel worker processes (`loky`) where each worker requires access to both reference and target float32 image rasters. 

> [!WARNING]
> **Container Crash / Exit Code 137:**
> If your container terminates abruptly with **exit code 137** (or silent worker process failure), it means Docker ran out of memory (**OOMKilled**).
> - Allocate at least **8 GB – 16 GB** of memory to Docker Desktop:
>   *Docker Desktop → Settings → Resources → Advanced → Memory slider (set to >= 8 GB)*
> - Or pass memory flags: `docker run --memory="12g" --memory-swap="16g" ...`
> - If memory is constrained, you can disable local refinement with `--no-arosics-local` to run fast global-only alignment.

---

## Troubleshooting

| Problem                                    | Fix                                                                                               |
|--------------------------------------------|---------------------------------------------------------------------------------------------------|
| `docker: command not found`                | Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and restart your shell. |
| Container exits with code 137 / crashes     | **Insufficient RAM**. Increase Docker Desktop memory allocation in Settings → Resources.          |
| `Error response from daemon: drive not shared` | Open Docker Desktop → Settings → Resources → File Sharing → add the drive letter.             |
| `GDAL ERROR 4: … No such file or directory` | Check that your `-v` mount path is correct and the file exists on the host.                     |
| Build fails on `libgdal-dev`               | Make sure Docker Desktop is set to **Linux containers** (right-click tray icon → Switch…).       |
