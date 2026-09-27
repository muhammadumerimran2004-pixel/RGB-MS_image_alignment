@echo off
setlocal

REM Double-clickable launcher for scripts\run-alignment.ps1.
REM Opens a window to drag and drop (or browse for) the RGB and MS GeoTIFFs,
REM then a pop-up to choose the destination folder;
REM Mode defaults to 'automated' unless you edit the line below.

powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0run-alignment.ps1" -DetailedLogs %*

endlocal
