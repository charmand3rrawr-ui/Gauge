@echo off
REM Opens Gauge Studio (the big-button window). The Desktop shortcut points here.
setlocal
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\pythonw.exe" (
  start "" "%HERE%.venv\Scripts\pythonw.exe" -m gauge.cli gui
) else (
  start "" pythonw -m gauge.cli gui
)
endlocal
