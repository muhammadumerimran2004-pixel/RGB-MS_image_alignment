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

.EXAMPLE
Omit -RgbPath/-MsPath to drag and drop the two images into a window, and omit
-OutputDir to choose the destination folder in a pop-up dialog:
.\scripts\run-alignment.ps1 -Mode automated -DetailedLogs

.EXAMPLE
Manual mode from a GCP file instead of interactive prompts:
.\scripts\run-alignment.ps1 `
  -RgbPath 'C:\data\rgb.tif' `
  -MsPath 'C:\data\ms.tif' `
  -OutputDir 'results\manual' `
  -Mode manual `
  -GcpFile 'C:\data\gcps.csv' `
  -GcpSourceUnits pixel
#>
[CmdletBinding()]
param(
    [string]$RgbPath,

    [string]$MsPath,

    [string]$OutputDir,

    [ValidateSet('automated', 'local-correlation', 'road_grid', 'local_mesh', 'manual')]
    [string]$Mode = 'automated',

    [ValidateSet('ms', 'rgb')]
    [string]$Resolution = 'ms',

    [string]$ConfigPath,

    [switch]$DetailedLogs,

    [switch]$EnableArosics,

    [string[]]$ArosicsBandPair,

    [switch]$NoArosicsLocal,

    [double]$ArosicsMaxShiftM,

    [ValidateSet('auto', 'deshifter', 'gdal_tps')]
    [string]$ArosicsWarpEngine,

    [double]$CropRowPeriodM,

    [string]$GcpFile,

    [ValidateSet('csv', 'qgis')]
    [string]$GcpFormat,

    [ValidateSet('pixel', 'map')]
    [string]$GcpSourceUnits,

    [ValidateSet('auto', 'translation', 'similarity', 'affine', 'tps')]
    [string]$ManualModel,

    [ValidateSet('corner', 'center')]
    [string]$PixelConvention
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

function Select-InputImages {
    param([string]$Rgb, [string]$Ms)

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()

    $form = New-Object System.Windows.Forms.Form
    $form.Text = 'Drone Alignment - input images'
    $form.ClientSize = New-Object System.Drawing.Size(560, 324)
    $form.FormBorderStyle = 'FixedDialog'
    $form.MaximizeBox = $false
    $form.StartPosition = 'CenterScreen'
    $form.TopMost = $true

    $zones = @{}
    $nextButton = New-Object System.Windows.Forms.Button

    $refresh = {
        foreach ($zone in $zones.Values) {
            if ($zone.Tag.Path) {
                $zone.Text = "$($zone.Tag.Caption)`n`n$(Split-Path -Leaf $zone.Tag.Path)`n$(Split-Path -Parent $zone.Tag.Path)"
                $zone.BackColor = [System.Drawing.Color]::FromArgb(214, 238, 214)
            } else {
                $zone.Text = "$($zone.Tag.Caption)`n`nDrop the GeoTIFF here, or click to browse"
                $zone.BackColor = [System.Drawing.Color]::FromArgb(236, 236, 236)
            }
        }
        $nextButton.Enabled = [bool]($zones.Rgb.Tag.Path -and $zones.Ms.Tag.Path)
    }

    $onDragEnter = {
        param($source, $e)
        if ($e.Data.GetDataPresent([System.Windows.Forms.DataFormats]::FileDrop)) {
            $e.Effect = [System.Windows.Forms.DragDropEffects]::Copy
        }
    }
    # Dropping both files at once onto either zone fills RGB then MS in the
    # order Explorer reports them, which is not reliable - check, then Swap.
    $onDragDrop = {
        param($source, $e)
        $files = @($e.Data.GetData([System.Windows.Forms.DataFormats]::FileDrop))
        if ($files.Count -ge 2) {
            $zones.Rgb.Tag.Path = $files[0]
            $zones.Ms.Tag.Path = $files[1]
        } elseif ($files.Count -eq 1) {
            $source.Tag.Path = $files[0]
        }
        & $refresh
    }
    $onClick = {
        param($source, $e)
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.Title = "Select the $($source.Tag.Caption)"
        $dialog.Filter = 'GeoTIFF (*.tif;*.tiff)|*.tif;*.tiff|All files (*.*)|*.*'
        if ($dialog.ShowDialog($form) -eq [System.Windows.Forms.DialogResult]::OK) {
            $source.Tag.Path = $dialog.FileName
            & $refresh
        }
    }

    foreach ($spec in @(
        @{ Key = 'Rgb'; Caption = 'RGB reference orthophoto'; Top = 16; Path = $Rgb },
        @{ Key = 'Ms'; Caption = 'Multispectral (MS) orthophoto'; Top = 140; Path = $Ms }
    )) {
        $zone = New-Object System.Windows.Forms.Label
        $zone.Location = New-Object System.Drawing.Point(16, $spec.Top)
        $zone.Size = New-Object System.Drawing.Size(528, 110)
        $zone.BorderStyle = 'FixedSingle'
        $zone.TextAlign = 'MiddleCenter'
        $zone.Cursor = [System.Windows.Forms.Cursors]::Hand
        $zone.AllowDrop = $true
        $zone.Tag = @{ Caption = $spec.Caption; Path = $spec.Path }
        $zone.Add_DragEnter($onDragEnter)
        $zone.Add_DragDrop($onDragDrop)
        $zone.Add_Click($onClick)
        $form.Controls.Add($zone)
        $zones[$spec.Key] = $zone
    }

    $swapButton = New-Object System.Windows.Forms.Button
    $swapButton.Text = 'Swap RGB / MS'
    $swapButton.Location = New-Object System.Drawing.Point(16, 270)
    $swapButton.Size = New-Object System.Drawing.Size(130, 36)
    $swapButton.Add_Click({
        $held = $zones.Rgb.Tag.Path
        $zones.Rgb.Tag.Path = $zones.Ms.Tag.Path
        $zones.Ms.Tag.Path = $held
        & $refresh
    })
    $form.Controls.Add($swapButton)

    $cancelButton = New-Object System.Windows.Forms.Button
    $cancelButton.Text = 'Cancel'
    $cancelButton.Location = New-Object System.Drawing.Point(318, 270)
    $cancelButton.Size = New-Object System.Drawing.Size(90, 36)
    $cancelButton.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $form.Controls.Add($cancelButton)
    $form.CancelButton = $cancelButton

    $nextButton.Text = 'Next: output folder'
    $nextButton.Location = New-Object System.Drawing.Point(414, 270)
    $nextButton.Size = New-Object System.Drawing.Size(130, 36)
    $nextButton.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.Controls.Add($nextButton)
    $form.AcceptButton = $nextButton

    & $refresh
    $result = $form.ShowDialog()
    $form.Dispose()
    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        throw 'Alignment cancelled: the RGB and MS images were not selected.'
    }
    return @($zones.Rgb.Tag.Path, $zones.Ms.Tag.Path)
}

function Select-OutputFolder {
    param([string]$StartIn)

    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = 'Choose the destination folder for the aligned outputs (a new folder is recommended).'
    $dialog.ShowNewFolderButton = $true
    if ($StartIn -and (Test-Path -LiteralPath $StartIn -PathType Container)) {
        $dialog.SelectedPath = $StartIn
    }
    # A hidden topmost owner keeps the dialog in front of the console window.
    $owner = New-Object System.Windows.Forms.Form
    $owner.TopMost = $true
    $result = $dialog.ShowDialog($owner)
    $owner.Dispose()
    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        throw 'Alignment cancelled: no output folder was selected.'
    }
    return $dialog.SelectedPath
}

if ($RgbPath) {
    $RgbPath = Remove-OuterQuotes $RgbPath
}
if ($MsPath) {
    $MsPath = Remove-OuterQuotes $MsPath
}
if (-not $RgbPath -or -not $MsPath) {
    $RgbPath, $MsPath = Select-InputImages -Rgb $RgbPath -Ms $MsPath
}
if (-not $OutputDir) {
    $OutputDir = Select-OutputFolder -StartIn (Split-Path -Parent $RgbPath)
}
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
if ($EnableArosics) {
    $pythonArgs += '--enable-arosics'
}
foreach ($pair in $ArosicsBandPair) {
    $pythonArgs += @('--arosics-band-pair', $pair)
}
if ($NoArosicsLocal) {
    $pythonArgs += '--no-arosics-local'
}
if ($PSBoundParameters.ContainsKey('ArosicsMaxShiftM')) {
    $pythonArgs += @('--arosics-max-shift-m', $ArosicsMaxShiftM)
}
if ($ArosicsWarpEngine) {
    $pythonArgs += @('--arosics-warp-engine', $ArosicsWarpEngine)
}
if ($PSBoundParameters.ContainsKey('CropRowPeriodM')) {
    $pythonArgs += @('--crop-row-period-m', $CropRowPeriodM)
}
if ($GcpFile) {
    $GcpFile = Remove-OuterQuotes $GcpFile
    if (-not (Test-Path -LiteralPath $GcpFile -PathType Leaf)) {
        throw "GCP file was not found: $GcpFile"
    }
    $pythonArgs += @('--gcp-file', (Resolve-Path -LiteralPath $GcpFile).Path)
}
if ($GcpFormat) {
    $pythonArgs += @('--gcp-format', $GcpFormat)
}
if ($GcpSourceUnits) {
    $pythonArgs += @('--gcp-source-units', $GcpSourceUnits)
}
if ($ManualModel) {
    $pythonArgs += @('--manual-model', $ManualModel)
}
if ($PixelConvention) {
    $pythonArgs += @('--pixel-convention', $PixelConvention)
}

Write-Host "Running $Mode alignment"
Write-Host "  RGB:    $rgbFullPath"
Write-Host "  MS:     $msFullPath"
Write-Host "  Output: $outputFullPath"

& python @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Alignment failed with exit code $LASTEXITCODE."
}
