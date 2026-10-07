@echo off
setlocal EnableDelayedExpansion

:: ───────────────────────────────────────────────────────────
::  AgriLift – Drone Alignment   (Docker quick-start)
::
::  Usage:
::    run-alignment-docker.bat  RGB_FILE  MS_FILE  [OUTPUT_DIR]  [MODE]
::
::  Examples:
::    run-alignment-docker.bat  C:\data\rgb.tif  C:\data\ms.tif
::    run-alignment-docker.bat  C:\data\rgb.tif  C:\data\ms.tif  C:\data\results  local-correlation
:: ───────────────────────────────────────────────────────────

set IMAGE_NAME=agrilift-align

:: ── Validate arguments ──
if "%~1"=="" (
    echo.
    echo   AgriLift Drone Alignment ^(Docker^)
    echo   ─────────────────────────────────
    echo.
    echo   Usage:  %~nx0  RGB_FILE  MS_FILE  [OUTPUT_DIR]  [MODE]
    echo.
    echo   Modes:  automated ^(default^), local-correlation, road_grid, local_mesh, manual
    echo.
    echo   The RGB and MS GeoTIFFs must be on the same drive letter.
    echo   If OUTPUT_DIR is omitted, results are saved next to the MS file.
    echo.
    exit /b 1
)
if "%~2"=="" (
    echo ERROR: You must provide both RGB_FILE and MS_FILE.
    exit /b 1
)

:: ── Resolve full paths ──
set "RGB_FILE=%~f1"
set "MS_FILE=%~f2"

if not exist "%RGB_FILE%" (
    echo ERROR: RGB file not found: %RGB_FILE%
    exit /b 1
)
if not exist "%MS_FILE%" (
    echo ERROR: MS file not found: %MS_FILE%
    exit /b 1
)

:: Output directory defaults to the MS file's parent directory.
if "%~3"=="" (
    for %%F in ("%MS_FILE%") do set "OUTPUT_DIR=%%~dpF"
) else (
    set "OUTPUT_DIR=%~f3"
)

:: Mode defaults to automated.
if "%~4"=="" (
    set "MODE=automated"
) else (
    set "MODE=%~4"
)

:: ── Determine the common host directory to mount ──
:: Mount the deepest common parent so the container sees both inputs + output.
:: Simple heuristic: use the drive root so paths are always valid.
set "DRIVE_LETTER=%RGB_FILE:~0,1%"
set "HOST_ROOT=%DRIVE_LETTER%:\"
set "CONTAINER_ROOT=/host_%DRIVE_LETTER%"

:: Convert Windows paths to container paths (replace \ with /).
set "C_RGB=%CONTAINER_ROOT%/%RGB_FILE:~3%"
set "C_RGB=%C_RGB:\=/%"

set "C_MS=%CONTAINER_ROOT%/%MS_FILE:~3%"
set "C_MS=%C_MS:\=/%"

set "C_OUT=%CONTAINER_ROOT%/%OUTPUT_DIR:~3%"
set "C_OUT=%C_OUT:\=/%"

:: ── Ensure the image exists ──
docker image inspect %IMAGE_NAME% >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Image "%IMAGE_NAME%" not found — building now ...
    echo.
    docker build -t %IMAGE_NAME% "%~dp0"
    if errorlevel 1 (
        echo ERROR: Docker build failed.
        exit /b 1
    )
)

:: ── Run ──
echo.
echo  AgriLift Drone Alignment
echo  ────────────────────────
echo   Mode:    %MODE%
echo   RGB:     %RGB_FILE%
echo   MS:      %MS_FILE%
echo   Output:  %OUTPUT_DIR%
echo.

docker run --rm ^
    -v "%HOST_ROOT%":%CONTAINER_ROOT% ^
    %IMAGE_NAME% ^
    "%C_RGB%" "%C_MS%" ^
    --output-dir "%C_OUT%" ^
    --mode %MODE%

if errorlevel 1 (
    echo.
    echo  ERROR: Alignment failed. Check the output above.
    exit /b 1
)

echo.
echo  Done! Results are in: %OUTPUT_DIR%
echo.
