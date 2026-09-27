<#
.SYNOPSIS
Runs the Drone Alignment or complete project test suite.
#>
[CmdletBinding()]
param(
    [ValidateSet('drone', 'all')]
    [string]$Scope = 'drone',

    [switch]$Coverage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$pytestArgs = @('-m', 'pytest')
if ($Scope -eq 'drone') {
    $pytestArgs += @('drone_alignment/tests', '-v')
} else {
    $pytestArgs += '-v'
}
if ($Coverage) {
    $pytestArgs += '--cov=drone_alignment'
}

& python @pytestArgs
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed with exit code $LASTEXITCODE."
}
