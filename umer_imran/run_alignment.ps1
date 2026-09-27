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
Write-Host "  [2] Standard automated alignment (ORB/SIFT, then phase fallback)"
Write-Host "  [3] Feature-agnostic local cell correlation (safe global fallback)"
$choice = Read-Host "Enter 1, 2, or 3"

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
            --verbose
    }
    "3" {
        python -m drone_alignment $rgbPath $msPath `
            --output-dir $outputDir `
            --mode local-correlation `
            --resolution ms `
            --verbose
    }
    default {
        Write-Error "Invalid choice. Run the launcher again and enter 1, 2, or 3."
        exit 1
    }
}
