<#
.SYNOPSIS
Runs one Drone Alignment mode from the repository root.

.EXAMPLE
.\scripts\run-alignment.ps1 `
  -RgbPath 'C:\data\rgb.tif' `
  -MsPath 'C:\data\ms.tif' `
  -OutputDir 'results\automated' `
  -Mode automated `
  -DetailedLogs
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$RgbPath,

    [Parameter(Mandatory)]
    [string]$MsPath,

    [Parameter(Mandatory)]
    [string]$OutputDir,

    [ValidateSet('automated', 'local-correlation', 'road_grid', 'local_mesh', 'manual')]
    [string]$Mode = 'automated',

    [ValidateSet('ms', 'rgb')]
    [string]$Resolution = 'ms',

    [string]$ConfigPath,

    [switch]$DetailedLogs,

    [switch]$EnableLoFTR
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Remove-OuterQuotes {
    param([string]$Value)

    $trimmed = $Value.Trim()
    if ($trimmed.Length -ge 2) {
        $first = $trimmed[0]
        $last = $trimmed[$trimmed.Length - 1]
        if (($first -eq [char]34 -and $last -eq [char]34) -or
            ($first -eq [char]39 -and $last -eq [char]39)) {
            return $trimmed.Substring(1, $trimmed.Length - 2)
        }
    }
    return $trimmed
}

$RgbPath = Remove-OuterQuotes $RgbPath
$MsPath = Remove-OuterQuotes $MsPath
if ($ConfigPath) {
    $ConfigPath = Remove-OuterQuotes $ConfigPath
}

if (-not (Test-Path -LiteralPath $RgbPath -PathType Leaf)) {
    throw "RGB input was not found: $RgbPath"
}
if (-not (Test-Path -LiteralPath $MsPath -PathType Leaf)) {
    throw "MS input was not found: $MsPath"
}
if ($ConfigPath -and -not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Configuration file was not found: $ConfigPath"
}

$rgbFullPath = (Resolve-Path -LiteralPath $RgbPath).Path
$msFullPath = (Resolve-Path -LiteralPath $MsPath).Path
$outputFullPath = [System.IO.Path]::GetFullPath($OutputDir)
$pythonArgs = @(
    '-m', 'drone_alignment',
    $rgbFullPath,
    $msFullPath,
    '--output-dir', $outputFullPath,
    '--mode', $Mode,
    '--resolution', $Resolution
)
if ($ConfigPath) {
    $pythonArgs += @('--config', (Resolve-Path -LiteralPath $ConfigPath).Path)
}
if ($DetailedLogs) {
    $pythonArgs += '--verbose'
}
if ($EnableLoFTR) {
    $pythonArgs += '--enable-loftr'
}

Write-Host "Running $Mode alignment"
Write-Host "  RGB:    $rgbFullPath"
Write-Host "  MS:     $msFullPath"
Write-Host "  Output: $outputFullPath"

& python @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Alignment failed with exit code $LASTEXITCODE."
}
