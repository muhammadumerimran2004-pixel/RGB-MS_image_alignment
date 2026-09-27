# Project runner scripts

Run these commands from the repository root in PowerShell. If your PowerShell
policy blocks local scripts, prefix the command with `powershell -ExecutionPolicy Bypass -File`.

## Automated alignment

```powershell
.\scripts\run-alignment.ps1 `
  -RgbPath 'C:\data\RGB_odm_orthophoto.tif' `
  -MsPath 'C:\data\MS_odm_orthophoto.tif' `
  -OutputDir '.\results\automated' `
  -Mode automated `
  -DetailedLogs
```

## Local correlation

This mode attempts feature-agnostic local correction. If evidence, field, or
footprint gates fail, it safely writes the already verified global alignment
and records the reason in the JSON report.

```powershell
.\scripts\run-local-correlation.ps1 `
  -RgbPath 'C:\data\RGB_odm_orthophoto.tif' `
  -MsPath 'C:\data\MS_odm_orthophoto.tif' `
  -OutputDir '.\results\local-correlation' `
  -DetailedLogs
```

## AROSICS-assisted automated alignment

AROSICS is installed through the optional project extra. It is a fail-closed
candidate: the standard automated candidates are tried first, and AROSICS is
used only if they do not produce an accepted result.

```powershell
python -m pip install -e '.[arosics,test]'

.\scripts\run-alignment.ps1 `
  -RgbPath 'C:\data\RGB_odm_orthophoto.tif' `
  -MsPath 'C:\data\MS_odm_orthophoto.tif' `
  -OutputDir '.\results\arosics-assisted' `
  -Mode automated `
  -EnableArosics `
  -DetailedLogs
```

To use a YAML configuration:

```powershell
.\scripts\run-local-correlation.ps1 `
  -RgbPath 'C:\data\RGB.tif' `
  -MsPath 'C:\data\MS.tif' `
  -OutputDir '.\results\local-correlation' `
  -ConfigPath '.\my_alignment_config.yaml'
```

## Tests

```powershell
.\scripts\run-tests.ps1 -Scope drone
.\scripts\run-tests.ps1 -Scope all
.\scripts\run-tests.ps1 -Scope drone -Coverage
```
