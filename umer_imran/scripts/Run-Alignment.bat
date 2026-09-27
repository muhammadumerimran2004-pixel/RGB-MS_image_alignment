@echo off
setlocal

REM Double-clickable launcher for scripts\run-alignment.ps1.
REM Prompts for RgbPath, MsPath, OutputDir (the script's mandatory params);
REM Mode defaults to 'automated' unless you edit the line below.

powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0run-alignment.ps1" -DetailedLogs %*

endlocal
