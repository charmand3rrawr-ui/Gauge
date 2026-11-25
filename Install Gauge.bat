@echo off
title Gauge Installer
echo.
echo   Installing Gauge. This can take 10-20 minutes the first time (it downloads the AI models).
echo   You do not need to do anything. A window called "Gauge Studio" will open when it is done.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\bootstrap.ps1" %*
if errorlevel 1 (
  echo.
  echo   Something did not work. Take a photo of this window and ask for help, or read installer\README.md
)
echo.
pause
