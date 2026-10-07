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
Write-Host "  [2] Verified global automated alignment  [RECOMMENDED]"
Write-Host "      ORB/SIFT with phase fallback. Fastest and most reliable first run."
Write-Host "      AROSICS local refinement is explicitly disabled."
Write-Host ""
Write-Host "  [3] Verified global + AROSICS local refinement  [EXPERIMENTAL]"
Write-Host "      Attempts non-uniform local correction after global alignment."
Write-Host "      Slower, high-memory, and currently susceptible to Windows worker failures."
Write-Host ""
Write-Host "  [4] Feature-agnostic local cell correlation  [EXPERIMENTAL]"
Write-Host "      Fits a smooth local field when evidence passes strict safety gates."
Write-Host "      Expected local rejection falls back to the verified global result."
$choice = Read-Host "Enter 1, 2, 3, or 4 (recommended: 2)"

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
