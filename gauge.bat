@echo off
REM Gauge launcher for Windows. Uses the venv created by install.ps1 if present.
setlocal
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\python.exe" (
  "%HERE%.venv\Scripts\python.exe" -m gauge.cli %*
) else (
  python -m gauge.cli %*
)
endlocal
