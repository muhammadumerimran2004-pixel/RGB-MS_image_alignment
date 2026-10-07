<##
Interactive launcher for local RGB/MS orthophoto alignment.

Run from this repository:
    .\run_alignment.ps1

The paths may be relative to this directory or absolute Windows paths.
##>

$rgbPath = Read-Host "RGB reference GeoTIFF path"
$msPath = Read-Host "Multispectral target GeoTIFF path"
$outputDir = Read-Host "Output directory (a new folder is recommended)"

Write-Host ""
Write-Host "Choose alignment mode:"
Write-Host "  [1] Manual control points"
Write-Host "      Use when you have matching RGB/MS control points from QGIS."
Write-Host ""
Write-Host "  [2] Fast global automated alignment (low-RAM fallback)"
Write-Host "      ORB/SIFT with phase fallback. Single affine transform across entire raster."
Write-Host "      AROSICS local refinement is explicitly disabled."
Write-Host ""
Write-Host "  [3] Verified global + AROSICS subpixel local refinement  [RECOMMENDED - MAIN ENGINE]"
Write-Host "      Automated subpixel co-registration using COREG_LOCAL and Thin-Plate Spline (TPS) warp."
Write-Host "      Corrects non-uniform flight distortions with multi-core phase correlation."
Write-Host "      (!) HEADS UP: High-memory workload. If the process terminates abruptly or crashes,"
Write-Host "          it means system RAM is insufficient for parallel matching workers."
Write-Host ""
Write-Host "  [4] Feature-agnostic local cell correlation  [EXPERIMENTAL]"
Write-Host "      Fits a smooth local field when evidence passes strict safety gates."
Write-Host "      Expected local rejection falls back to the verified global result."
Write-Host ""
$choice = Read-Host "Enter 1, 2, 3, or 4 (recommended: 3)"
if ([string]::IsNullOrWhiteSpace($choice)) {
    $choice = "3"
}

switch ($choice) {
    "1" {
        python -m drone_alignment $rgbPath $msPath `
            --output-dir $outputDir `
            --mode manual `
            --resolution ms
    }
    "2" {
        python -m drone_alignment $rgbPath $msPath `
            --output-dir $outputDir `
            --mode automated `
            --resolution ms `
            --no-arosics-local
    }
    "3" {
        python -m drone_alignment $rgbPath $msPath `
            --output-dir $outputDir `
            --mode automated `
            --resolution ms `
            --arosics-local
    }
    "4" {
        python -m drone_alignment $rgbPath $msPath `
            --output-dir $outputDir `
            --mode local-correlation `
            --resolution ms
    }
    default {
        Write-Error "Invalid choice. Run the launcher again and enter 1, 2, 3, or 4."
        exit 1
    }
}
