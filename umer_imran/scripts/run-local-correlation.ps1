<#
.SYNOPSIS
Runs feature-agnostic local correlation with safe global fallback.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$RgbPath,

    [Parameter(Mandatory)]
    [string]$MsPath,

    [Parameter(Mandatory)]
    [string]$OutputDir,

    [ValidateSet('ms', 'rgb')]
    [string]$Resolution = 'ms',

    [string]$ConfigPath,

    [switch]$DetailedLogs
)

$runner = Join-Path $PSScriptRoot 'run-alignment.ps1'
& $runner -RgbPath $RgbPath -MsPath $MsPath -OutputDir $OutputDir `
    -Mode 'local-correlation' -Resolution $Resolution -ConfigPath $ConfigPath `
    -DetailedLogs:$DetailedLogs
exit $LASTEXITCODE
